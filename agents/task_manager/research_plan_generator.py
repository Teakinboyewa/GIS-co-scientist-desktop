
import os
import json
import re
import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning
from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from langchain_core.tools import tool
from utils.toolbase.research_plan_generator_tools import _build_research_design_tools
from utils.toolbase.research_plan_review_tools import _build_research_plan_review_tools
from utils.toolbase.dataset_profile import DATA_FITNESS_CHECKS, format_candidates
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END
from utils.config import load_configurations, resolve_output_dir
import utils.prompts as prompts
from agents.spatial_reasoning import knowledge_base
from agents.spatial_reasoning import pipeline as spatial_pipeline
from agents.spatial_reasoning.conditions import condition_of

import time
settings = load_configurations()

# Store ctx per task_id for research plan feedback loop
_research_plan_contexts = {}

class SharedState(TypedDict):

    """
    Represents the shared state of the graph
    """
    ctx: dict  # Context with research_breakdown, data_summary
    generated_research_plan: str  # Generated plan
    review_result: str  # Review feedback
    final_research_plan: str  # Final polished plan


# Helper function to generate markdown from structured research plan
def generate_markdown_from_structure(structured_plan: dict) -> str:
    """
    Convert structured research plan JSON to formatted markdown

    Args:
        structured_plan: Dictionary with research plan structure

    Returns:
        Formatted markdown string
    """
    markdown = []

    # Title
    title = structured_plan.get('title', 'Research Plan')
    markdown.append(f"# {title}\n")

    # Research Question
    rq = structured_plan.get('research_question', '')
    if rq:
        markdown.append("## Research Question")
        markdown.append(f"{rq}\n")

    # Objectives
    objectives = structured_plan.get('objectives', [])
    if objectives:
        markdown.append("## Objectives\n")
        for obj in objectives:
            num = obj.get('number', '')
            title_text = obj.get('title', '')
            hypothesis = obj.get('hypothesis', '')
            steps = obj.get('steps', [])

            markdown.append(f"{num}. **{title_text}**")
            markdown.append(f"   - Hypothesis: {hypothesis}")
            markdown.append("   - Steps:")
            for i, step in enumerate(steps):
                letter = chr(97 + i)  # a, b, c, ...
                if isinstance(step, dict):
                    desc = step.get("description", str(step))
                    deps = step.get("depends_on", [])
                    dep_text = f" (depends on: step {', '.join(str(d) for d in deps)})" if deps else ""
                    markdown.append(f"     {letter}. {desc}{dep_text}")
                else:
                    markdown.append(f"     {letter}. {step}")
            markdown.append("")  # Empty line between objectives

    # Data Context
    data_context = structured_plan.get('data_context', {})
    if data_context:
        markdown.append("## Data Context")
        if data_context.get('geographic_scope'):
            markdown.append(f"- **Geographic Scope**: {data_context['geographic_scope']}")
        if data_context.get('temporal_scope'):
            markdown.append(f"- **Temporal Scope**: {data_context['temporal_scope']}")
        if data_context.get('study_units'):
            markdown.append(f"- **Study Units**: {data_context['study_units']}")
        if data_context.get('population'):
            markdown.append(f"- **Population**: {data_context['population']}")
        markdown.append("")

    # Key Considerations
    considerations = structured_plan.get('key_considerations', [])
    if considerations:
        markdown.append("## Key Considerations")
        for consideration in considerations:
            markdown.append(f"- {consideration}")
        markdown.append("")

    return "\n".join(markdown)


_FINAL_ANSWER_PROMPT = (
    "You have used all available tool calls. Do not call any more tools. "
    "Using what you have verified so far, return the complete plan now as a single JSON object "
    "in the required output format."
)


def _run_agent_loop(llm_with_tools, messages, tools, max_iterations, label="AGENT",
                    stream_callback=None, ctx=None, final_llm=None):
    """Shared agent loop: stream LLM, execute tool calls, repeat until done.

    When *stream_callback* is provided **every** LLM turn is streamed
    token-by-token — the user sees content tokens arrive live, and any
    tool_call_chunks are accumulated into structured tool_calls that are
    executed after the turn completes. When no tool calls are produced,
    the turn was already the final response so we break out of the loop
    without a second round-trip.

    If the iteration budget runs out while the model is still calling tools,
    the last turn has no content. *final_llm* (the same model without tools)
    is then asked once for the answer, so the caller never gets an empty string
    just because the model was still verifying.
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
                # Stream content tokens as they arrive
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    stream_callback(token)
                # Accumulate chunks — LangChain merges content + tool_call_chunks
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

    if response is not None and getattr(response, "tool_calls", None) and final_llm is not None:
        print(f"  [{label}] Tool budget of {max_iterations} turns used up; requesting the final answer without tools")
        if ctx and helper.check_cancelled(ctx):
            return "[Cancelled by user]"
        messages.append(HumanMessage(content=_FINAL_ANSWER_PROMPT))
        response = final_llm.invoke(messages)
        messages.append(response)
        if stream_callback and response.content:
            stream_callback(response.content)

    return response.content if hasattr(response, "content") else str(response)




def _save_planner_input(ctx, system_prompt, user_prompt, tools):
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "planner_system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(system_prompt)
    with open(os.path.join(out_dir, "planner_user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(user_prompt)
    with open(os.path.join(out_dir, "planner_tools.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": t.name, "description": t.description} for t in tools], f, indent=2)


RESEARCH_DATA_FITNESS_INSTRUCTIONS = f"""The datasets above are CANDIDATES, not requirements. Some may be left over
from earlier requests in this conversation and have nothing to do with this question.

Before planning, decide for each candidate whether it fits this question:
{DATA_FITNESS_CHECKS}
Use the breakdown above as a guide; call profile_dataset or sample_data to check anything unclear.

Plan ONLY with datasets that fit: never use an unfit dataset as a source_id or an
input_node, and never add one just because it was provided. If no candidate covers an
objective's data needs, flag it with "data_gap": true rather than stretching an unfit dataset.

Add a top-level "data_fitness" list to the plan JSON with one entry per candidate:
{{"source_id": "<dataset name>", "used": true or false, "reason": "<one line: area, time, unit or content>"}}"""


def generate_research_plan(state: SharedState):
    """Agentic research plan generation — LLM explores data, derives objectives and hypotheses, and produces the plan."""

    print("GENERATING RESEARCH PLAN (Agentic)")

    start_time = time.time()
    ctx = state["ctx"]
    event_cb = ctx.get("_event_callback")
    if event_cb:
        event_cb({"type": "llm_update", "content": "Generating research plan...", "step": "research_plan_generation"})
    rq = ctx.get("rq", "")
    rq_breakdown = ctx["artifacts"].get("rq_breakdown", {})
    data_registry = ctx["artifacts"].get("data_registry", {})
    # Spatial-integration condition: how the shared knowledge base reaches
    # this agent (nothing under B0). See agents/spatial_reasoning/.
    condition = condition_of(ctx)
    knowledge_access = []
    system_prompt, kb_tools = knowledge_base.compose_planner_prompt(
        prompts.generate_research_plan_system_prompt, condition["kb_delivery"], knowledge_access)

    # Build tools and bind to LLM
    tools = _build_research_design_tools(ctx) + kb_tools
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        model_name=llm_model("research_plan.generate"), **llm_reasoning("research_plan.generate")
        )
    llm_with_tools = llm.bind_tools(tools)

    # Gather RQ Breakdown context
    research_question_intent = rq_breakdown.get("intent", "")
    research_question_category = rq_breakdown.get("category", "")
    scale = rq_breakdown.get("scale",{})
    proposed_data_requirement = rq_breakdown.get("data_requirement", [])


    user_prompt = f"""Generate a research plan for:

**Research Question:** {rq}

**RQ Breakdown context:**
- Research question intent: {research_question_intent}
- Category: {research_question_category}
- Spatial scale: {scale.get('spatial_scale', 'N/A')}
- Temporal scale: {scale.get('temporal_scale', 'N/A')}
- Proposed data requirements: {json.dumps(proposed_data_requirement, default=str) if proposed_data_requirement else 'N/A'}
- Relevant data the breakdown found among the inputs: {json.dumps(rq_breakdown.get("data_available") or [], default=str)}
- Sufficiency verdict: {rq_breakdown.get("data_available_sufficiency") or 'N/A'}

**Candidate datasets** (the name after ### is the source_id):
{format_candidates(data_registry) if data_registry else 'N/A'}

{RESEARCH_DATA_FITNESS_INSTRUCTIONS}

Use your tools to explore the datasets in detail as needed.

Generate the plan."""
    # Proposed method: the locked hypotheses the plan must test (empty otherwise).
    addendum = spatial_pipeline.planner_addendum(ctx)
    if addendum:
        user_prompt = f"{user_prompt}\n\n{addendum}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Keep the planner's exact input with the run, before the call, so even
    # a failed plan leaves a record of what this condition was given.
    _save_planner_input(ctx, system_prompt, user_prompt, tools)

    print("Agent exploring data and generating plan...\n")
    stream_callback = ctx.get("_stream_callback")
    if stream_callback:
        stream_callback("\n\n## Generating Research Plan\n\n")
    full_content = _run_agent_loop(
        llm_with_tools, messages, tools, max_iterations=10, label="GENERATE",
        stream_callback=stream_callback, ctx=ctx, final_llm=llm,
    )
    print(f"\n{full_content[:200]}..." if len(full_content) > 200 else f"\n{full_content}")

    # Clean LLM response and parse to structured JSON
    cleaned = full_content.strip() if full_content else ""
    if cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        brace = cleaned.find("{")
        if brace > 0:
            cleaned = cleaned[brace:]

    try:
        structured_plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse research plan JSON: {e}")
        structured_plan = {"raw_response": cleaned}

    state['generated_research_plan'] = structured_plan

    # Save to files
    elapsed_time = time.time() - start_time
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)

    plan_path_json = os.path.join(out_dir, "research_plan.json")
    plan_data = {
        "research_question": rq,
        "research_intent": research_question_intent,
        **structured_plan,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken_seconds": round(elapsed_time, 2),
        "time_taken_formatted": f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
    }
    with open(plan_path_json, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    # Which knowledge the planner actually looked up (B2/B3), so failures can
    # be split into 'guidance never retrieved' and 'retrieved but not applied'.
    with open(os.path.join(out_dir, "spatial_knowledge_access.json"), "w", encoding="utf-8") as f:
        json.dump({"condition": condition["name"], "kb_delivery": condition["kb_delivery"],
                   "calls": knowledge_access}, f, indent=2)

    ctx.setdefault("artifacts", {})["research_plan_json"] = plan_path_json
    ctx["artifacts"]["spatial_knowledge_access"] = knowledge_access
    ctx["artifacts"]["research_plan_first_draft"] = plan_data

    print(f"\n✓ Research plan saved to: {plan_path_json}")
    print(f"  - Objectives: {len(structured_plan.get('objectives', []))}")
    print(f"  - Time taken: {plan_data['time_taken_formatted']}\n")

    return state



def review_research_plan(state: SharedState):
    """Agentic review — LLM validates the plan """

    print("REVIEWING RESEARCH PLAN (Agentic)")

    start_time = time.time()
    ctx = state["ctx"]
    event_cb = ctx.get("_event_callback")
    if event_cb:
        event_cb({"type": "llm_update", "content": "Reviewing research plan...", "step": "research_plan_generation"})
    rq = ctx.get("rq", "")
    research_plan = state.get('generated_research_plan', "")
    rq_breakdown = ctx["artifacts"].get("rq_breakdown", {})
    research_question_intent = rq_breakdown.get("intent", "")


    # Build tools and bind to LLM
    tools = _build_research_plan_review_tools(ctx)

    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(), 
        model_name=llm_model("research_plan.review"), **llm_reasoning("research_plan.review")
        )
    llm_with_tools = llm.bind_tools(tools)

    scale = rq_breakdown.get("scale", {})

    system_prompt = prompts.review_research_plan_system_prompt

    # Include the generated plan directly in the prompt so the LLM can
    # review it even if tool calls are not supported by the current model.
    plan_json_str = json.dumps(research_plan, indent=2, default=str) if research_plan else "{}"

    user_prompt = f"""Review the generated research plan.

**Research Question:** {rq}
**Research Intent:** {research_question_intent}
**Spatial Scale:** {scale.get('spatial_scale', 'N/A')}
**Temporal Scale:** {scale.get('temporal_scale', 'N/A')}

**Generated Research Plan:**
```json
{plan_json_str}
```

Use your tools to inspect the data registry and verify column names, join keys, and step connectivity, then return the corrected plan."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    print("Agent verifying plan against actual data...\n")
    stream_callback = ctx.get("_stream_callback")
    full_content = _run_agent_loop(
        llm_with_tools, messages, tools, max_iterations=8, label="REVIEW",
        stream_callback=stream_callback, ctx=ctx, final_llm=llm,
    )
    print(f"\n{full_content[:200]}..." if len(full_content) > 200 else f"\n{full_content}")

    # Clean LLM response and parse to structured JSON
    cleaned = full_content.strip() if full_content else ""
    if cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        brace = cleaned.find("{")
        if brace > 0:
            cleaned = cleaned[brace:]
    try:
        structured_plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse research plan JSON: {e}")
        structured_plan = {"raw_response": cleaned}

    # A review that returns no usable plan must not replace the draft with an
    # empty one; keep the draft and record that the review did not apply.
    review_ok = isinstance(structured_plan, dict) and bool(structured_plan.get("objectives"))
    draft_ok = isinstance(research_plan, dict) and bool(research_plan.get("objectives"))
    if not review_ok and draft_ok:
        print("Warning: Review returned no usable plan; keeping the draft plan unchanged")
        structured_plan = {
            **research_plan,
            "review_summary": {
                "issues_found": [],
                "changes_made": [],
                "review_failed": True,
                "note": "The review returned no usable plan, so the draft plan was kept unchanged.",
                "review_raw_response": structured_plan.get("raw_response", "") if isinstance(structured_plan, dict) else "",
            },
        }

    state['review_result'] = structured_plan

    # Save to files
    elapsed_time = time.time() - start_time
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)

    review_path_json = os.path.join(out_dir, "research_plan_review.json")
    review_data = {
        **structured_plan,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken_seconds": round(elapsed_time, 2),
        "time_taken_formatted": f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
    }
    with open(review_path_json, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)


    ctx.setdefault("artifacts", {})["research_plan_review_json"] = review_path_json

    ctx["artifacts"]["research_plan_review"] = review_data

    print(f"\n✓ Review saved to: {review_path_json}")

    return state

def _resolve_node_paths(plan_data, ctx):
    """
    Resolve every step's input_nodes and output_node to actual file paths.

    Called once after plan review is finalized. Adds two fields to each step:
        - input_node_paths:  dict  {node_name: file_path}
        - output_node_path:  str   deterministic path under Outputs/

    Source data is resolved by source_id from the data registry.
    Output nodes are mapped to the per-task output directory.
    Inter-step references are resolved by walking steps in order.
    """
    data_registry = ctx.get("artifacts", {}).get("data_registry", [])
    out_dir = resolve_output_dir(ctx)

    # Build source_id → path lookup from data registry.
    # Register both the raw source_id AND source_id + actual extension
    # so "census_data" and "census_data.csv" both resolve.
    source_paths = {}
    for src in data_registry:
        source_id = src.get("file", "")
        path = src.get("path", "")
        if source_id and path:
            source_paths[source_id] = path
            actual_ext = os.path.splitext(path)[1]
            if actual_ext:
                source_paths[source_id + actual_ext] = path

    for obj in plan_data.get("objectives", []):
        # Tracks all available nodes within this objective (sources + prior outputs)
        produced_nodes = dict(source_paths)

        for step in obj.get("steps", []):
            # ── Resolve output_node path ──────────────────────────────
            output_node = step.get("output_node", "")
            if output_node:
                stem, ext = os.path.splitext(output_node)
                if not ext:
                    ext = ".csv"
                safe_name = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")[:60]
                output_node_path = os.path.join(out_dir, f"{safe_name}{ext}")
                step["output_node_path"] = output_node_path
                produced_nodes[output_node] = output_node_path

            # ── Resolve input_node paths ──────────────────────────────
            input_nodes = step.get("input_nodes", [])
            input_node_paths = {}
            for node_name in input_nodes:
                resolved = produced_nodes.get(node_name)
                if not resolved:
                    # Stem match: "data.csv" ↔ "data" regardless of extension
                    node_stem = os.path.splitext(node_name)[0]
                    for key, path in produced_nodes.items():
                        if os.path.splitext(key)[0] == node_stem:
                            resolved = path
                            break
                if resolved:
                    input_node_paths[node_name] = resolved
                else:
                    input_node_paths[node_name] = None
                    print(f"  WARNING: input_node '{node_name}' in step {step.get('step_number')} "
                          f"could not be resolved. Available: {list(produced_nodes.keys())}")
            step["input_node_paths"] = input_node_paths


def finalize_research_plan(state: SharedState):
    """Extract the corrected plan from the review result — no LLM call needed."""

    print("FINALIZING RESEARCH PLAN")

    ctx = state["ctx"]
    event_cb = ctx.get("_event_callback")
    if event_cb:
        event_cb({"type": "llm_update", "content": "Finalizing research plan...", "step": "research_plan_generation"})
    rq = ctx.get("rq", "")
    rq_breakdown = ctx.get("artifacts", {}).get("rq_breakdown", {})
    research_title = rq_breakdown.get("title", "")
    research_intent = rq_breakdown.get("intent", "")
    review_result = state.get('review_result', {})

    # Extract the plan from the review, stripping review-only metadata
    final_plan = {k: v for k, v in review_result.items() if k != "review_summary"}





    objectives = final_plan.get("objectives", [])
    total_steps = sum(len(obj.get('steps', [])) for obj in objectives)

    # Save to file
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)
    final_plan_path = os.path.join(out_dir, "research_plan_final.json")

    final_plan_data = {
        "title":research_title,
        "research_question":rq,
        "research_intent":research_intent,
        "objectives":objectives,
        "review_summary": review_result.get("review_summary", {}),
        # Which candidate datasets the plan uses and why the others were left out;
        # the reviewer may correct it, otherwise the planner's draft list stands.
        "data_fitness": (review_result.get("data_fitness")
                         or (state.get("generated_research_plan") or {}).get("data_fitness")
                         or []),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken_seconds": round(total_steps, 2),
    }
    # Proposed method only: restore locked hypotheses, write the spatial
    # assumptions report. A no-op for the baselines.
    spatial_pipeline.on_finalize(ctx, final_plan_data)

    with open(final_plan_path, "w", encoding="utf-8") as f:
        json.dump(final_plan_data, f, indent=2, ensure_ascii=False)

    # ── Resolve node paths for every step ────────────────────────────
    # Programmatically map each input_node → actual file path and
    # each output_node → deterministic path under Outputs/.
    # This is done once, centrally, so code generation gets clean paths.
    _resolve_node_paths(final_plan_data, ctx)

    # Re-save the plan with resolved paths
    with open(final_plan_path, "w", encoding="utf-8") as f:
        json.dump(final_plan_data, f, indent=2, ensure_ascii=False)

    # Store in context
    ctx.setdefault("artifacts", {})["research_plan_final_json"] = final_plan_path
    ctx["artifacts"]["research_plan_final"] = final_plan_data

    state['final_research_plan'] = final_plan

    print(f"  Objectives: {len(objectives)} | Total steps: {total_steps}")
    print(f"  Saved to: {final_plan_path}")

    return state



def plan_graph_nodes(ctx):
    """The plan stage's nodes, in order: generate → review → finalize, with
    the proposed method's nodes (spatial_pipeline.graph_nodes) inserted for
    the conditions that use them. Baselines get the three published nodes."""
    extra = spatial_pipeline.graph_nodes(ctx)
    before_plan = [(n, fn) for n, fn, pos in extra if pos == "before_plan"]
    before_finalize = [(n, fn) for n, fn, pos in extra if pos == "before_finalize"]
    return (before_plan
            + [("generate_research_plan", generate_research_plan),
               ("review_research_plan", review_research_plan)]
            + before_finalize
            + [("finalize_research_plan", finalize_research_plan)])


def build_serial_graph(ctx):
    workflow = StateGraph(SharedState)
    nodes = plan_graph_nodes(ctx)
    for name, fn in nodes:
        workflow.add_node(name, fn)
    names = [START] + [name for name, _ in nodes] + [END]
    for a, b in zip(names, names[1:]):
        workflow.add_edge(a, b)
    return workflow.compile().invoke({"ctx": ctx, "generated_research_plan": "",
                                      "review_result": "", "final_research_plan": ""})


def store_context(task_id, ctx):
    """Store ctx for later use in feedback loop."""
    _research_plan_contexts[task_id] = ctx


def get_context(task_id):
    """Retrieve stored ctx."""
    return _research_plan_contexts.get(task_id)


def revise_research_plan_stream(task_id, feedback):
    """
    Revise the research plan based on user feedback. Yields streaming updates.
    Only modifies what the feedback asks for.
    """
    ctx = _research_plan_contexts.get(task_id)
    if not ctx:
        yield {"type": "error", "error": f"No context found for task {task_id}"}
        return

    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    rq = ctx.get("rq", "")

    if not structured_plan:
        yield {"type": "error", "error": "No existing research plan found to revise"}
        return

    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(), 
        model_name=llm_model("research_plan.revise"), **llm_reasoning("research_plan.revise")
        )
    system_prompt = prompts.revise_research_plan_stream_system_prompt

    user_prompt = f"""Revise the following research plan based ONLY on the user's feedback.
Do NOT change anything the user didn't ask about.

**CURRENT RESEARCH PLAN:**
{json.dumps(structured_plan, indent=2)}

**ORIGINAL RESEARCH QUESTION:**
{rq}

**USER FEEDBACK:**
{feedback}

"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    full_content = ""
    for chunk in llm.stream(messages):
        content = chunk.content
        if content:
            full_content += content
            yield {"type": "llm_stream", "content": content}

    # Parse JSON response
    json_str = full_content.strip()
    if json_str.startswith("```json"):
        json_str = json_str.replace("```json", "").replace("```", "").strip()
    elif json_str.startswith("```"):
        json_str = json_str.replace("```", "").strip()

    try:
        revised_plan = json.loads(json_str)
    except json.JSONDecodeError:
        yield {"type": "error", "error": "Failed to parse revised plan JSON"}
        return

    # Update artifacts
    ctx.setdefault("artifacts", {})["research_plan_final"] = revised_plan

    # Save updated files
    markdown_plan = generate_markdown_from_structure(revised_plan)
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)

    final_plan_data = {**revised_plan, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "research_plan_final.json"), "w", encoding="utf-8") as f:
        json.dump(final_plan_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "research_plan_final.md"), "w", encoding="utf-8") as f:
        f.write(markdown_plan)

    # Regenerate both viz HTMLs
    from utils.research_plan_viz import save_research_plan_visualization
    save_research_plan_visualization(revised_plan, out_dir, task_id)

    yield {"type": "result", "message": "Research plan revised successfully"}