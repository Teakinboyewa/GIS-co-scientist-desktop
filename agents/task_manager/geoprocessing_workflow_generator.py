from calendar import c
import os
import sys
from utils.lazy_imports import lazy_function  # IPython loads on first use
display = lazy_function("IPython.display", "display")
HTML = lazy_function("IPython.display", "HTML")
Code = lazy_function("IPython.display", "Code")

from numpy import full

# Add project root to path so 'utils' and other packages can be found
# project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

from time import time
import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning
from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from langchain_core.tools import tool
from utils.toolbase.geoprocessing_workflow_generator_tools import _build_geoprocessing_workflow_design_tools
from utils.toolbase.dataset_profile import DATA_FITNESS_CHECKS, format_candidates
# from agents.task_manager.research_plan_generator import _run_agent_loop

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict

from utils.config import load_configurations
import utils.prompts as prompts


def graphfile_path(ctx):
    """The deterministic, machine-local path of a task's geoprocessing graph:
    ``<output_dir>/graphfiles/<task_id>.graphml``.

    Generation always saves the .graphml here, so deriving the path from the
    locally-rebuilt ``output_dir`` is more reliable than trusting a stored
    absolute path — which may have been written by a different worker process
    or host (e.g. a server path that doesn't exist after a redeploy or when a
    state file is opened on another machine). Returns "" if ctx lacks the
    fields needed to build the path.
    """
    output_dir = ctx.get("output_dir", "")
    task_id = ctx.get("task_id", "")
    if not output_dir or not task_id:
        return ""
    return os.path.join(output_dir, "graphfiles", f"{task_id}.graphml")


def resolve_graphfile(ctx):
    """Return a usable graph-file path for ``ctx`` and repair the stored
    artifact in place.

    Prefers the stored artifact path when it exists on this machine; otherwise
    falls back to the deterministic per-task location (:func:`graphfile_path`).
    When the fallback is used, the artifact is updated so downstream steps use
    the valid local path. Returns "" if no graph file exists anywhere.
    """
    arts = ctx.setdefault("artifacts", {})
    stored = arts.get("geoprocessing_workflow_graph_file", "")
    if stored and os.path.exists(stored):
        return stored
    candidate = graphfile_path(ctx)
    if candidate and os.path.exists(candidate):
        arts["geoprocessing_workflow_graph_file"] = candidate
        return candidate
    return ""


def _build_geoproecessing_workflow_generation_prompt(ctx):
    task = ctx.get("rq", "N/A")
    role = prompts.geoprocessing_workflow_generator_role
    graph_task_prefix = prompts.geoprocessing_workflow_generator_task_prefix
    graph_reply_example = prompts.geoprocessing_workflow_generator_reply_example
    graph_requirement = prompts.geoprocessing_workflow_generator_requirement

    # Save graph file under the conversation/task output directory. Use the
    # shared deterministic path so every read site can reconstruct it later.
    graph_file = graphfile_path(ctx)
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
    print(f"Graph file will be saved at: {graph_file}")

    # Build input paths summary for the prompt
    data_registry = ctx["artifacts"].get("data_registry", [])
    data_input_summary = [f"- {s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']}" for s in data_registry]
    data_input_summary_str = '\n'.join(data_input_summary)


    graph_requirement = graph_requirement.copy()
    graph_requirement.append(f"Save the network into GraphML format, save it at: {graph_file}")
    graph_requirement_str =  '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])


    graph_prompt = f'Your role: {role} \n\n' + \
                f'Your task: {graph_task_prefix} \n {task} \n\n' + \
                f'Your reply needs to meet these requirements: \n {graph_requirement_str} \n\n' + \
                f'Your reply example: {graph_reply_example} \n\n' + \
                f'Candidate datasets (use only those that fit the question; see the user message): \n{data_input_summary_str} \n'
    # print("=== Graph Generation Prompt ===")
    # print(graph_prompt)
    return graph_prompt, graph_file


def _run_agent_loop(llm_with_tools, messages, tools, max_iterations, label="AGENT",
                    stream_callback=None, ctx=None):
    """Shared agent loop: stream LLM, execute tool calls, repeat until done.

    When *stream_callback* is provided **every** LLM turn is streamed
    token-by-token — content tokens flow live, tool_call_chunks accumulate
    into structured tool_calls that are executed after the turn completes.
    """
    tool_map = {t.name: t for t in tools}
    response = None
    for i in range(max_iterations):
        if ctx and helper.check_cancelled(ctx):
            return "[Cancelled by user]"

        # --- stream this turn (single round-trip) ---
        if stream_callback:
            accumulated = None
            for chunk in llm_with_tools.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    stream_callback(token)
                accumulated = chunk if accumulated is None else accumulated + chunk
            response = accumulated
        else:
            response = llm_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            # No tool calls ⇒ final response. Already streamed above.
            break

        # --- execute tool calls ---
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            args_short = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if isinstance(tool_args, dict) else str(tool_args)[:80]
            if len(args_short) > 120:
                args_short = args_short[:120] + "..."
            tool_msg = f"  [{label} TOOL] {tool_name}({args_short})"
            print(tool_msg)
            if stream_callback:
                stream_callback(f"\n🔧 [TOOL CALL] **{tool_name}**({args_short})\n")
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool error: {e}"
            if stream_callback:
                stream_callback(f"\n```\n{str(result)}\n```\n")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return response.content if hasattr(response, "content") else str(response)


def _breakdown_context(ctx):
    """What the task breakdown found: intent, scale, data needed and data available.

    The breakdown runs before this step in Spatial Analysis mode; without it the
    workflow designer only had file names and row counts to go on.
    """
    rq = ctx.get("artifacts", {}).get("rq_breakdown") or {}
    if not isinstance(rq, dict) or not rq:
        return "(no task breakdown available)"
    scale = rq.get("scale") or {}

    def items(v):
        if not v:
            return "  - none listed"
        v = v if isinstance(v, list) else [v]
        return "\n".join(f"  - {x}" for x in v)

    return "\n".join([
        f"- Intent: {rq.get('intent') or 'N/A'}",
        f"- Spatial scale: {scale.get('spatial_scale') or 'N/A'}",
        f"- Temporal scale: {scale.get('temporal_scale') or 'N/A'}",
        "- Data required:", items(rq.get("data_requirement")),
        "- Relevant data the breakdown found among the inputs:", items(rq.get("data_available")),
        f"- Sufficiency verdict: {rq.get('data_available_sufficiency') or 'N/A'}",
    ])


def _candidate_datasets(ctx):
    """One profile block per dataset in the registry: extent, CRS, period, columns."""
    return format_candidates(ctx.get("artifacts", {}).get("data_registry", []))


DATA_FITNESS_INSTRUCTIONS = f"""The datasets above are CANDIDATES, not requirements. Some may be left over
from earlier requests in this conversation and have nothing to do with this question.

Before designing the graph, decide for each candidate whether it fits this question:
{DATA_FITNESS_CHECKS}
Use the task breakdown above as a guide; call profile_dataset or sample_data to check anything unclear.

Build the workflow ONLY from datasets that fit. Leave every other dataset out of the
graph entirely: no data node, no edge, no loading step. Never use a dataset just
because it was provided."""


def generate_geoprocessing_workflow(ctx):
    """Geoprocessing workflow generation — LLM explores data, derives operational steps, and produces the plan."""

    print("GENERATING GEOPROCESSING WORKFLOW")

  
    event_cb = ctx.get("_event_callback")
    if event_cb:
        event_cb({"type": "llm_update", "content": "Generating Geoprocessing workflow...", "step": "geoprocessing_workflow_generation"})
    task = ctx.get("rq", "")
    
    # Build input paths summary for the prompt
    # data_registry = ctx["artifacts"].get("data_registry", {})
    data_registry = ctx["artifacts"].get("data_registry", [])
    data_input_summary = [f"- {s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']}" for s in data_registry]
    

    # Build tools and bind to LLM
    tools = _build_geoprocessing_workflow_design_tools(ctx)
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        model_name=llm_model("spatial_analysis.geoprocessing_workflow"), **llm_reasoning("spatial_analysis.geoprocessing_workflow")
        # model=get_ui_model(),
        # reasoning_effort="medium"
        # reasoning_effort = get_ui_reasoning_effort()
        )
    llm_with_tools = llm.bind_tools(tools)

   
    system_prompt, graph_file = _build_geoproecessing_workflow_generation_prompt(ctx)


    user_prompt = f"""Generate a geoprocessing workflow graph for:

**Research Question:** {task}

**Task breakdown:**
{_breakdown_context(ctx)}

**Candidate datasets:**
{_candidate_datasets(ctx)}

{DATA_FITNESS_INSTRUCTIONS}

Your response MUST contain THREE parts:
1. A Python code block (enclosed by ```python and ```) that builds a NetworkX DiGraph with data nodes and operation nodes, then saves it as GraphML. This code block is MANDATORY.
2. After the code block, a section headed "Data used and not used" listing EVERY candidate dataset as used or not used, each with a one-line reason (area, time, unit or content).
3. A prose explanation of the workflow: what each branch does, how they converge, and what the final output is."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    print("Exploring data and generating geoprocessing workflow...\n")
    stream_callback = ctx.get("_stream_callback")
    if stream_callback:
        stream_callback("\n\n## Generating Geoprocessing Workflow\n\n")

    full_content = _run_agent_loop(
        llm_with_tools, messages, tools, max_iterations=10, label="GENERATE",
        stream_callback=stream_callback, ctx=ctx,
    )

    # Split response into code block and explanation
    code_block = helper.extract_code(response_content=full_content) if full_content else ""
    explanation = full_content
    if code_block and "```" in full_content:
        # Remove the code block portion to get the prose explanation
        import re
        explanation = re.sub(r'```python.*?```', '', full_content, flags=re.DOTALL).strip()

    # Execute the generated graph code to create the .graphml file.
    # The .graphml at ``graph_file`` is the single source of truth consumed by
    # the downstream code-generation step — which on PythonAnywhere often runs
    # in a *different* worker process where the in-memory ctx is not shared, so
    # the only thing that survives is this file on the shared disk. If it is not
    # persisted here we must fail loudly now, rather than report success and let
    # the missing file surface as an error a request later.
    import networkx as nx
    exec_error = None
    if code_block:
        try:
            exec_globals = {
                "nx": nx,
                "networkx": nx,
                "graph_file": graph_file,
                "__builtins__": __builtins__,
            }
            exec_locals = {}
            print(f"[GeoWorkflow] Executing generated graph code...")
            exec(code_block, exec_globals, exec_locals)

            # The generated code is instructed to save the graph to ``graph_file``
            # itself (see the prompt's graph_requirement), so that on-disk file is
            # the source of truth. Only when the code built the graph but did not
            # save it do we fall back to the conventional ``G`` variable and write
            # it ourselves. We deliberately do NOT scan the namespace for an
            # arbitrary networkx object: if the code created several graphs,
            # guessing risks persisting the wrong one — better to fail loudly.
            if os.path.exists(graph_file):
                print(f"[GeoWorkflow] Graph saved by generated code: {graph_file}")
            else:
                graph_obj = exec_locals.get("G") or exec_globals.get("G")
                if isinstance(graph_obj, nx.Graph):
                    # Fix None values (GraphML doesn't support None)
                    for node in graph_obj.nodes():
                        for key, value in graph_obj.nodes[node].items():
                            if value is None:
                                graph_obj.nodes[node][key] = ""
                    for source, target in graph_obj.edges():
                        for key, value in graph_obj.edges[source, target].items():
                            if value is None:
                                graph_obj.edges[source, target][key] = ""
                    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
                    nx.write_graphml(graph_obj, graph_file)
                    print(f"[GeoWorkflow] Graph saved from `G` to: {graph_file}")
        except Exception as e:
            exec_error = e
            print(f"[GeoWorkflow] ERROR executing graph code: {e}")

    # Authoritative check: the file must exist on disk before we record success.
    # This is what turns a half-finished generation (e.g. a request truncated by
    # the PythonAnywhere worker timeout, leaving an empty graphfiles/ dir) into a
    # clear failure here instead of a "missing graph" surprise downstream.
    if not os.path.exists(graph_file):
        if exec_error is not None:
            detail = f" (error while executing the graph code: {exec_error})"
        elif not code_block:
            detail = " (the model returned no python code block to build the graph)"
        else:
            detail = (" (the graph code ran but neither saved the file nor left "
                      "a networkx graph in `G`)")
        raise RuntimeError(
            f"Geoprocessing workflow generation did not persist its graph to "
            f"{graph_file}{detail}. Refusing to report success for an empty "
            f"workflow — please regenerate the geoprocessing workflow."
        )

    ctx["artifacts"]["geoprocessing_workflow"] = full_content
    ctx["artifacts"]["geoprocessing_workflow_code"] = code_block
    ctx["artifacts"]["geoprocessing_workflow_explanation"] = explanation
    ctx["artifacts"]["geoprocessing_workflow_graph_file"] = graph_file
    return full_content, graph_file
    


    # # Clean LLM response and parse to structured JSON
    # cleaned = full_content.strip() if full_content else ""
    # if cleaned:
    #     cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    #     brace = cleaned.find("{")
    #     if brace > 0:
    #         cleaned = cleaned[brace:]

    # try:
    #     structured_plan = json.loads(cleaned)
    # except json.JSONDecodeError as e:
    #     print(f"Warning: Failed to parse research plan JSON: {e}")
    #     structured_plan = {"raw_response": cleaned}

    # state['generated_research_plan'] = structured_plan

    # # Save to files
    # elapsed_time = time.time() - start_time
    # out_dir = ctx.get("output_dir", settings.output_dir)
    # os.makedirs(out_dir, exist_ok=True)

    # plan_path_json = os.path.join(out_dir, "research_plan.json")
    # plan_data = {
    #     "research_question": rq,
    #     "research_intent": research_question_intent,
    #     **structured_plan,
    #     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    #     "time_taken_seconds": round(elapsed_time, 2),
    #     "time_taken_formatted": f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
    # }
    # with open(plan_path_json, "w", encoding="utf-8") as f:
    #     json.dump(plan_data, f, indent=2, ensure_ascii=False)

    # ctx.setdefault("artifacts", {})["research_plan_json"] = plan_path_json
    # ctx["artifacts"]["research_plan_first_draft"] = plan_data

    # print(f"\n✓ Research plan saved to: {plan_path_json}")
    # print(f"  - Objectives: {len(structured_plan.get('objectives', []))}")
    # print(f"  - Time taken: {plan_data['time_taken_formatted']}\n")

    # return state


def graphml_to_plan(G, title="Geoprocessing Workflow", research_question=""):
    """Convert a NetworkX DiGraph (from generate_geoprocessing_workflow) into
    a structured plan dict for the workflow visualization.

    Detects parallel branches in the DAG and represents them as separate
    objectives that converge at shared operations (operations with inputs
    from multiple branches).

    The graph has two node types:
      - node_type="data"       (inputs/outputs)
      - node_type="operation"  (processing steps)

    Edges go: data -> operation (inputs) and operation -> data (outputs).

    Returns:
        dict with {title, research_question, dag_mode, objectives: [...]}
    """
    import networkx as nx

    op_nodes = set(n for n, d in G.nodes(data=True) if d.get("node_type") == "operation")

    # Topological order
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo = list(G.nodes())
    op_nodes_sorted = [n for n in topo if n in op_nodes]

    # --- Compute topological level for each node (longest-path from any root) ---
    node_level = {}
    for n in topo:
        preds = list(G.predecessors(n))
        if not preds:
            node_level[n] = 0
        else:
            node_level[n] = max(node_level.get(p, 0) for p in preds) + 1

    # --- Detect parallel branches ---
    # Build an operation-only predecessor graph (handles both op→data→op and direct op→op edges)
    def op_predecessors(op_node):
        """Get operation nodes that directly precede this one."""
        preds = set()
        for pred in G.predecessors(op_node):
            if pred in op_nodes:
                # Direct op → op edge
                preds.add(pred)
            elif G.nodes[pred].get("node_type") == "data":
                # op → data → op edge
                for op_pred in G.predecessors(pred):
                    if op_pred in op_nodes:
                        preds.add(op_pred)
        return list(preds)

    def op_successors(op_node):
        """Get operation nodes that directly follow this one."""
        succs = set()
        for succ in G.successors(op_node):
            if succ in op_nodes:
                succs.add(succ)
            elif G.nodes[succ].get("node_type") == "data":
                for op_succ in G.successors(succ):
                    if op_succ in op_nodes:
                        succs.add(op_succ)
        return list(succs)

    # Find root operations (no operation predecessors) — these start branches
    root_ops = [op for op in op_nodes_sorted if not op_predecessors(op)]

    # Find convergence operations (multiple operation predecessors from different branches)
    convergence_ops = set()
    for op in op_nodes_sorted:
        op_preds = op_predecessors(op)
        if len(op_preds) > 1:
            convergence_ops.add(op)

    # Trace each root forward to build branches
    # A branch is a sequence of operations from a root until it hits a convergence op
    def _trace_branch(start_op):
        """Trace operations from start_op forward, stopping before convergence ops."""
        branch = [start_op]
        current = start_op
        while True:
            next_ops = [s for s in op_successors(current) if s not in convergence_ops]
            if len(next_ops) == 1:
                branch.append(next_ops[0])
                current = next_ops[0]
            else:
                break
        return branch

    branches = []
    assigned_ops = set()
    for root in root_ops:
        if root not in assigned_ops:
            branch = _trace_branch(root)
            branches.append(branch)
            assigned_ops.update(branch)

    # Convergence ops and any remaining ops form the "shared" sequence
    shared_ops = [op for op in op_nodes_sorted if op not in assigned_ops]

    # --- Helper to build step dict ---
    def _make_step(op, step_num):
        attrs = G.nodes[op]
        input_nodes = [pred for pred in G.predecessors(op)
                       if G.nodes[pred].get("node_type") == "data"]
        output_nodes = [succ for succ in G.successors(op)
                        if G.nodes[succ].get("node_type") == "data"]
        step = {
            "step_number": step_num,
            "operation": attrs.get("description", op),
            "description": attrs.get("description", op),
            "input_nodes": input_nodes,
            "output_node": output_nodes[0] if output_nodes else "",
            "expected_output": G.nodes[output_nodes[0]].get("description", "") if output_nodes else "",
            "operation_type": attrs.get("operation_type", ""),
            "node_level": node_level.get(op, 0),
        }
        if len(output_nodes) > 1:
            step["additional_outputs"] = output_nodes[1:]
        return step

    # --- Build objectives ---
    objectives = []
    has_parallel = len(branches) > 1

    if has_parallel:
        # Each branch becomes its own objective
        for i, branch in enumerate(branches, 1):
            steps = [_make_step(op, j) for j, op in enumerate(branch, 1)]
            # Title from first operation
            branch_title = G.nodes[branch[0]].get("description", f"Branch {i}")
            objectives.append({
                "number": i,
                "title": f"Branch {i}: {branch_title}",
                "steps": steps,
                "branch_type": "parallel",
            })

        # Shared/convergence operations as final objective
        if shared_ops:
            steps = [_make_step(op, j) for j, op in enumerate(shared_ops, 1)]
            objectives.append({
                "number": len(branches) + 1,
                "title": "Convergence: Combined Analysis",
                "steps": steps,
                "branch_type": "convergence",
            })
    else:
        # Single-branch: keep flat like before
        steps = [_make_step(op, i) for i, op in enumerate(op_nodes_sorted, 1)]
        objectives.append({
            "number": 1,
            "title": title,
            "steps": steps,
        })

    return {
        "title": title,
        "research_question": research_question,
        "dag_mode": has_parallel,
        "objectives": objectives,
    }
