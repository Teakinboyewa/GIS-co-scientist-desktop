import os
import json

from utils.lazy_imports import lazy_function  # IPython loads on first use
Image = lazy_function("IPython.display", "Image")
display = lazy_function("IPython.display", "display")

import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning
from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from typing_extensions import TypedDict
from typing import Optional, List
from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel, Field

from agents.task_manager import TaskManager, research_plan_generator

from utils.config import load_configurations

import time
import utils.prompts as prompts
settings = load_configurations()
task_manager = TaskManager()


# ── Structured Output Schema ────────────────────────────────────────────────
class StepClassification(BaseModel):
    """Structured classification result for a research step."""
    objective: str = Field(description="The parent objective title this step belongs to")
    step_index: int = Field(description="The index/number of this step within its objective")
    step_title: str = Field(description="A concise title summarizing the step")
    task_class: str = Field(description="The classified task type. Must be one of the defined task classes.")
    reasoning: str = Field(description="Brief explanation of why this class was assigned")


# ── Task Classification Knowledge Base ──────────────────────────────────────
TASK_CLASSES = {
    "data_collection": {
        "label": "Data Collection & Acquisition",
        "description": "Gathering, downloading, querying, or assembling datasets from databases, APIs, or file sources.",
        "examples": [
            "Collect data on visitation to supermarkets from SafeGraph",
            "Gather demographic data from census tract-level datasets",
            "Download ACS 5-year estimates for median household income",
            "Retrieve spatial boundary files for US census tracts"
        ],
        "keywords": ["collect", "gather", "download", "retrieve", "acquire", "obtain", "compile", "assemble", "query", "fetch", "extract data from"]
    },
    "data_processing": {
        "label": "Data Processing & Preparation",
        "description": "Cleaning, transforming, merging, normalizing, or restructuring data for analysis. Includes joins, CRS harmonization, missing value handling, and feature engineering.",
        "examples": [
            "Merge census demographic data with visitation records by GEOID",
            "Normalize income variables using z-scores",
            "Clean and handle missing values in the poverty dataset",
            "Harmonize coordinate reference systems across spatial datasets"
        ],
        "keywords": ["merge", "join", "clean", "normalize", "transform", "preprocess", "harmonize", "restructure", "encode", "impute", "filter", "aggregate", "reshape"]
    },
    "descriptive_analysis": {
        "label": "Descriptive & Exploratory Analysis",
        "description": "Computing summary statistics, distributions, frequency tables, and initial data exploration (EDA) to understand data characteristics.",
        "examples": [
            "Conduct descriptive statistical analysis to summarize visitation patterns",
            "Generate summary statistics for socioeconomic variables across tracts",
            "Explore distributions of income and poverty rates",
            "Compute mean, median, and variance of visitation counts"
        ],
        "keywords": ["descriptive", "summary statistics", "summarize", "distribution", "EDA", "exploratory", "frequency", "mean", "median", "describe", "profile"]
    },
    "statistical_modeling": {
        "label": "Statistical & Regression Modeling",
        "description": "Fitting regression models, hypothesis testing, correlation analysis, ANOVA, and other inferential statistical methods.",
        "examples": [
            "Use Pearson correlation to explore relationships between income and visitation",
            "Fit multivariate regression with demographic controls",
            "Run OLS regression to assess the influence of poverty on food access",
            "Perform hypothesis testing on group differences in visitation"
        ],
        "keywords": ["regression", "correlation", "OLS", "hypothesis test", "ANOVA", "t-test", "chi-square", "multivariate", "logistic", "Pearson", "Spearman", "inferential", "p-value", "coefficient"]
    },
    "spatial_analysis": {
        "label": "Spatial & Geospatial Analysis",
        "description": "Spatial statistics, spatial regression (GWR, spatial lag/error), hotspot analysis, autocorrelation (Moran's I), buffering, overlay, and geocoding.",
        "examples": [
            "Conduct spatial regression to assess geographic influence on visitation",
            "Perform spatial cluster analysis to identify hotspots",
            "Calculate Moran's I for spatial autocorrelation of food access",
            "Run Geographically Weighted Regression on tract-level data"
        ],
        "keywords": ["spatial", "GWR", "Moran", "hotspot", "cluster", "autocorrelation", "kriging", "buffer", "overlay", "geocode", "geographically weighted", "spatial lag", "spatial error", "LISA"]
    },
    "visualization": {
        "label": "Visualization & Mapping",
        "description": "Creating charts, plots, maps (choropleth, heatmaps), and other visual representations of data or results.",
        "examples": [
            "Create choropleth maps highlighting regional differences in visitation",
            "Visualize findings with scatter plots of income vs visitation",
            "Generate spatial distribution plots and demographic maps",
            "Plot regression residuals on a map"
        ],
        "keywords": ["visualize", "plot", "chart", "map", "choropleth", "heatmap", "scatter", "bar chart", "histogram", "graph", "figure", "display", "illustrate", "cartographic"]
    },
    "machine_learning": {
        "label": "Machine Learning & Predictive Modeling",
        "description": "Training, tuning, and evaluating ML models including classification, clustering, random forest, neural networks, and cross-validation.",
        "examples": [
            "Train a random forest model to predict visitation rates",
            "Apply k-means clustering to segment census tracts",
            "Use gradient boosting to classify food access levels",
            "Evaluate model performance using cross-validation and RMSE"
        ],
        "keywords": ["random forest", "classification", "clustering", "k-means", "neural network", "gradient boosting", "XGBoost", "SVM", "cross-validation", "train", "predict", "feature importance", "deep learning", "ensemble"]
    },
    "synthesis_reporting": {
        "label": "Synthesis & Reporting",
        "description": "Interpreting results, drawing conclusions, writing summaries, comparing findings to hypotheses, and generating reports.",
        "examples": [
            "Synthesize regression results to assess hypothesis support",
            "Summarize key findings across all spatial models",
            "Compare observed patterns with stated hypotheses",
            "Write an interpretive summary of demographic effects on visitation"
        ],
        "keywords": ["synthesize", "summarize", "interpret", "report", "conclude", "compare", "discuss", "findings", "implications", "narrative", "write-up"]
    }
}





class SharedState(TypedDict):
    """
    Represents the shared state of the graph
    """

    ctx: dict  # Context with objective objects, research_breakdown, data_summary
    current_objective_key: str  # Which objective we're processing (e.g., "objective_1")
    objective_objects: dict  # The current objective's data (title, hypothesis, steps, ...)
    step_routing: dict  # Output from route_step_to_agent (agent assignments per step)
    synthesized_result: str  # Results of each steps synthesizer
    final_obj_results: str  # Total results of obj
    debug_info: dict  # Output from step_debugger (errors and diagnostics)


#
# def build_serial_graph(ctx):
#     """
#     Build and run the objective execution graph for ALL objectives.
#
#     Extracts all objectives from the research plan, then runs the
#     step_classification → route_steps pipeline for each one sequentially.
#     Results are accumulated per objective in ctx["artifacts"]["objective_results"].
#
#     Returns:
#         Dict mapping objective keys to their graph results.
#     """
#     # Extract all objectives from the research plan
#     research_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
#     all_objectives = task_manager.extract_research_object_for_objective(ctx, research_plan)
#
#     if not all_objectives:
#         print("[ERROR] No objectives found in research plan.")
#         return {}
#
#     # Build the graph once (reusable for each objective)
#     workflow = StateGraph(SharedState)
#
#     # Add nodes
#     workflow.add_node("step_classification", step_classifier)
#     workflow.add_node("route_steps", route_step_to_agent)
#     workflow.add_node("step_execution", step_executor)
#     workflow.add_node("step_debugger", step_debugger)
#     # workflow.add_node("step_results_synthesizer", step_results_synthesizer)
#
#
#     # Define the edges
#     workflow.add_edge(START, "step_classification")
#     workflow.add_edge("step_classification", "route_steps")
#     workflow.add_edge("route_steps", "step_execution")
#     workflow.add_conditional_edges("step_execution", _route_after_execution, {"step_debugger": "step_debugger", END: END})
#     workflow.add_edge("step_debugger", END)
#     # workflow.add_edge("step_execution", "step_results_synthesizer")
#     # workflow.add_edge("step_results_synthesizer", END)
#
#     graph = workflow.compile()
#     # print(graph.get_graph().draw_mermaid())
#     display(Image(graph.get_graph().draw_mermaid_png()))
#
#     # Loop over each objective and invoke the graph
#     all_results = {}
#     stream_cb = ctx.get("_stream_callback")
#     total_obj = len(all_objectives)
#
#     for i, (obj_key, obj_data) in enumerate(all_objectives.items(), 1):
#         if helper.check_cancelled(ctx):
#             print("[OBJECTIVE EXECUTOR] Cancelled by user")
#             break
#         obj_title = obj_data.get('title', 'Untitled')
#         num_steps = len(obj_data.get('steps', []))
#         print(f"\n{'='*60}")
#         print(f"Processing {obj_key}: {obj_title}")
#         print(f"{'='*60}")
#
#         # Set stream step tag for this objective
#         obj_step_tag = f"exec_obj{i}"
#         ctx["_stream_step"] = obj_step_tag
#
#         if stream_cb:
#             stream_cb(f"[Objective {i}/{total_obj}] {obj_title}\n")
#             stream_cb(f"  Steps: {num_steps}\n")
#
#         result = graph.invoke({
#             "ctx": ctx,
#             "current_objective_key": obj_key,
#             "objective_objects": obj_data,
#             "step_classifications": {},
#             "step_routing": {},
#             "synthesized_result": "",
#             "final_obj_results": "",
#             "debug_info": {}
#         })
#
#         all_results[obj_key] = {
#             "step_classifications": result.get("step_classifications", {}),
#             "step_routing": result.get("step_routing", {}),
#             "step_execution": result.get("synthesized_result", ""),
#         }
#
#         if stream_cb:
#             stream_cb(f"  Objective {i} complete.\n")
#
#     # Save aggregated results to ctx artifacts
#     ctx.setdefault("artifacts", {})["objective_results"] = all_results
#     print(f"\nCompleted processing {len(all_results)} objectives.")
#
#     if stream_cb:
#         stream_cb(f"\nAll {total_obj} objectives processed.\n")
#
#     return all_results


def build_serial_graph(ctx):
    """
    Build and run the objective execution graph for ALL objectives.

    Extracts all objectives from the research plan, then runs the
    step_classification → route_steps pipeline for each one sequentially.
    Results are accumulated per objective in ctx["artifacts"]["objective_results"].

    Returns:
        Dict mapping objective keys to their graph results.
    """
    # Extract all objectives from the research plan
    research_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    all_objectives = research_plan.get("objectives", {})
    # all_objectives = task_manager.extract_research_object_for_objective(ctx, research_plan)

    if not all_objectives:
        print("[ERROR] No objectives found in research plan.")
        return {}

    # Build the graph once (reusable for each objective)
    workflow = StateGraph(SharedState)



    # # Add nodes
    # workflow.add_node("step_classification", step_classifier)
    workflow.add_node("route_steps", route_step_to_agent)
    workflow.add_node("step_execution", step_executor)
    workflow.add_node("step_debugger", step_debugger)
    # # workflow.add_node("step_results_synthesizer", step_results_synthesizer)

    # Define the edges (FOR TESTING)



    # # Define the edges
    workflow.add_edge(START, "route_steps")
    workflow.add_edge("route_steps", "step_execution")
    workflow.add_conditional_edges("step_execution", _route_after_execution, {"step_debugger": "step_debugger", END: END})
    workflow.add_edge("step_debugger", END)
    # workflow.add_edge("step_execution", "step_results_synthesizer")
    # workflow.add_edge("step_results_synthesizer", END)

    graph = workflow.compile()
    # print(graph.get_graph().draw_mermaid())
    display(Image(graph.get_graph().draw_mermaid_png()))





    # # Loop over each objective and invoke the graph
    all_results = {}
    stream_cb = ctx.get("_stream_callback")
    total_obj = len(all_objectives)
    print (total_obj)
    #
    for i, obj_data in enumerate(all_objectives, 1):
        if helper.check_cancelled(ctx):
            print("[OBJECTIVE EXECUTOR] Cancelled by user")
            break
        obj_key = f"objective_{obj_data.get('number', i)}"
        obj_title = obj_data.get('title', 'Untitled')
        num_steps = len(obj_data.get('steps', []))
        print(f"\n{'='*60}")
        print(f"Processing {obj_key}: {obj_title}")
        print(f"{'='*60}")
    #
    #     # Set stream step tag for this objective
        obj_step_tag = f"exec_obj{i}"
        ctx["_stream_step"] = obj_step_tag

        if stream_cb:
            stream_cb(f"[Objective {i}/{total_obj}] {obj_title}\n")
            stream_cb(f"  Steps: {num_steps}\n")

        result = graph.invoke({
            "ctx": ctx,
            "current_objective_key": obj_key,
            "objective_objects": obj_data,
            "step_routing": {},
            "synthesized_result": "",
            "final_obj_results": "",
            # "debug_info": {}
        })
        all_results[obj_key] = {
            "step_routing": result.get("step_routing", {}),
            # "step_execution": result.get("synthesized_result", ""),
        }

        if stream_cb:
            stream_cb(f"  Objective {i} complete.\n")

    # Save aggregated results to ctx artifacts
    ctx.setdefault("artifacts", {})["objective_results"] = all_results
    print(f"\nCompleted processing {len(all_results)} objectives.")

    if stream_cb:
        stream_cb(f"\nAll {total_obj} objectives processed.\n")

    return all_results


def _build_task_class_descriptions() -> str:
    """Format task classes into a concise string for the LLM prompt."""
    lines = []
    for key, cls in TASK_CLASSES.items():
        examples_str = "; ".join(cls["examples"][:2])
        lines.append(f"- **{key}** ({cls['label']}): {cls['description']}\n  Examples: {examples_str}")
    return "\n".join(lines)



class StepRouteResult(BaseModel):
    """Structured output for step-to-agent routing."""
    agent_name: str = Field(description="Name of the selected agent from the available agents list")
    task_type: str = Field(description="The type of task to assign (e.g., 'analyze', 'collect_data', 'model', 'visualize')")
    reasoning: str = Field(description="Brief explanation of why this agent was selected for this step")
    execution_notes: str = Field(description="Specific instructions or context the agent should know when executing this step")
    task_complexity: str = Field(description="One of 'simple' (DataAgent tasks), 'medium' (standard analysis), or 'complex' (advanced modeling/spatial)")


def _fallback_agent():
    """The agent to use when routing fails: one that can execute a step.

    Falling back to the first registered agent picked the Research Question
    Understanding Agent, which has no step-execution method, so every step of
    the objective failed with an AttributeError.
    """
    registry = task_manager.agent_registry
    if "ModelingAgent" in registry:
        return "ModelingAgent"
    for name, info in registry.items():
        if hasattr(info.get("instance"), "handle_generate_and_execute_code_for_simple_operation"):
            return name
    return next(iter(registry), "unknown")


def route_step_to_agent(state: SharedState):
    """
    Use LLM to determine which agent should handle each research step.

    Uses operation_type, operation, and step description directly from each step in the research plan — no upstream step_classifier needed.

    Args:
        state: SharedState containing ctx, objective_objects.

    Returns:
        Dict updating step_routing in shared state and saving to ctx["artifacts"].
    """
    ctx = state["ctx"]
    artifacts = ctx.get("artifacts", {})
    obj_key = state["current_objective_key"]

    # ── Get current objective (already extracted per-objective by build_serial_graph) ──
    objective_objects = state.get("objective_objects", {})
    if isinstance(objective_objects, str):
        objective_objects = json.loads(objective_objects) if objective_objects else {}

    objective_title = objective_objects.get("title", "")
    steps = objective_objects.get("steps", [])

    # No-planning ablation: the router is skipped and the ModelingAgent takes
    # the step, as the ablation specifies (see agents/task_manager/ablation.py).
    from agents.task_manager.ablation import NO_PLANNING, ablation_of, pinned_routing
    if ablation_of(ctx) == NO_PLANNING:
        result = pinned_routing(steps)
        ctx.setdefault("artifacts", {}).setdefault("step_routing", {})[obj_key] = result
        task_manager.log(ctx, f"Ablation {NO_PLANNING}: all steps of {obj_key} routed to ModelingAgent")
        return {"step_routing": result}

    # Research context
    research_question = artifacts.get("research_question", "")

    # ── Build available agents list ────────────────────────────────────────
    task_manager.log(ctx, f"Routing steps to agents for {obj_key}: {objective_title}...")
    stream_cb = ctx.get("_stream_callback")
    if stream_cb:
        stream_cb(f"Routing steps to agents...\n")
    available_agents = []
    for agent_name, agent_info in task_manager.agent_registry.items():
        available_agents.append({
            "name": agent_name,
            "description": agent_info.get("description", "No description provided")
        })

    if not available_agents:
        task_manager.log(ctx, "WARNING: No agents registered. Cannot route steps.")
        return {"step_routing": {"routing": [], "error": "No agents registered"}}

    # Task class descriptions for context
    task_class_block = _build_task_class_descriptions()

    system_prompt = f"""You are an intelligent task router for a multi-agent geospatial research system.

Your job is to assign each research step to the single most appropriate agent based on:
1. The step's operation_type and operation (from the research plan)
2. The step's description and what it requires
3. Each agent's registered capabilities
4. The overall research context (question, objective, hypothesis)

## Available Agents:
{json.dumps(available_agents, indent=2)}

## Task Class Reference (for understanding step classifications):
{task_class_block}

1. MATCH CAPABILITIES: Choose the agent whose description most closely aligns with the step's operation_type and operation. Do not assign an agent to a task outside its stated capabilities.
2. FALLBACK: If no agent is a strong match, choose the agent with the broadest analytical capabilities and note the limitation in execution_notes.

## Output Format:
Return a JSON array where each element corresponds to one step, with ONLY these fields:
- "step_index": (int) 1-based step number (to match the step)
- "agent_name": (str) exact name of the selected agent from the available list
- "reasoning": (str) one-sentence justification for why this agent was chosen
- "execution_notes": (str) any specific context or instructions the agent needs for this step
- "task_complexity": (str) one of "simple", "medium", or "complex"

## Task Complexity Rules:
- "simple": Steps routed to DataAgent (data collection, data processing, basic preparation) and basic visualization. These are straightforward data operations.
- "medium": Steps involving descriptive analysis, standard statistical tests (ANOVA, correlation), or single-method analysis.
- "complex": Steps involving advanced modeling (regression, ML, GWR), multi-step analysis, spatial statistics (Moran's I, clustering), or synthesis of multiple results.
"""

    # ── Build step details from research plan fields ────────────────────────
    steps_text = ""
    for i, step in enumerate(steps, start=1):
        if isinstance(step, str):
            description = step
            operation_type = "unknown"
            operation = ""
        else:
            description = step.get("description", str(step))
            operation_type = step.get("operation_type", "unknown")
            operation = step.get("operation", "")



        steps_text += (
            f"  Step {i}:\n"
            f"    - description: {description}\n"
            f"    - operation_type: {operation_type}\n"
            f"    - operation: {operation}\n"
        )

    user_prompt = f"""Route each of the following research steps to the most appropriate agent.

## Research Context:
- Research Question: {research_question}
- Current Objective: {objective_title}


## Steps to Route (with operation_type and operation from the research plan):
{steps_text}

## Registered Agents:
{json.dumps(available_agents, indent=2)}

Use the operation_type and operation to inform your routing. Return a JSON array with routing decisions for ALL steps.
"""

    try:
        llm = GIBDChat(
            # gibd_user_key=helper.load_OpenAI_key(), 
            model_name=llm_model("execution.route_steps"), **llm_reasoning("execution.route_steps")
            )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Parse response
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        llm_decisions = json.loads(content)

        # Build lookup from LLM results by step_index
        llm_by_idx = {d["step_index"]: d for d in llm_decisions}

        # Validate agent names against registry
        valid_agents = set(task_manager.agent_registry.keys())

        # Merge: plan fields (known) + LLM fields (agent_name, reasoning, execution_notes, task_complexity)
        routing_decisions = []
        for i, step in enumerate(steps, start=1):
            if isinstance(step, str):
                description = step
                operation_type = "unknown"
                operation = ""

            else:
                description = step.get("description", str(step))
                operation_type = step.get("operation_type", "unknown")
                operation = step.get("operation", "")


            llm_info = llm_by_idx.get(i, {})
            agent_name = llm_info.get("agent_name", "")

            if agent_name not in valid_agents:
                task_manager.log(ctx, f"WARNING: LLM routed step {i} to unknown agent '{agent_name}', falling back to {_fallback_agent()}")
                agent_name = _fallback_agent()
                llm_info["execution_notes"] = llm_info.get("execution_notes", "") + " [FALLBACK: original agent not found in registry]"

            routing_decisions.append({
                "step_index": i,
                "operation": operation,
                "step_description": description,
                "operation_type": operation_type,
                "agent_name": agent_name,
                "reasoning": llm_info.get("reasoning", ""),
                "execution_notes": llm_info.get("execution_notes", ""),
                "task_complexity": llm_info.get("task_complexity", "medium"),
            })

        task_manager.log(ctx, f"Successfully routed {len(routing_decisions)} steps to agents")
        if stream_cb:
            for rd in routing_decisions:
                stream_cb(f"  Step {rd['step_index']}: {rd['operation']} → {rd['agent_name']} [{rd['task_complexity']}]\n")
        result = {"routing": routing_decisions}

    except Exception as e:
        task_manager.log(ctx, f"ERROR in step routing: {e}")
        # Fallback: assign all steps to the first available agent
        fallback_agent = _fallback_agent()
        fallback_routing = []
        for i, step in enumerate(steps, start=1):
            desc = step if isinstance(step, str) else step.get("description", f"Step {i}")
            operation_type = "" if isinstance(step, str) else step.get("operation_type", "unknown")
            operation = "" if isinstance(step, str) else step.get("operation", "")
            fallback_routing.append({
                "step_index": i,
                "step_description": desc,
                "operation_type": operation_type,
                "operation": operation,
                "agent_name": fallback_agent,
                "task_type": "analyze",
                "reasoning": "Fallback routing due to LLM error",
                "execution_notes": f"Error occurred during routing: {str(e)}"
            })
        result = {"routing": fallback_routing, "error": str(e)}

    # Save to ctx artifacts per objective (persisted across the pipeline)
    ctx.setdefault("artifacts", {}).setdefault("step_routing", {})[obj_key] = result

    task_manager.log(ctx, f"Routing complete for {obj_key}")

    # Return to update shared state (accessible by downstream nodes)
    return {"step_routing": result}




def step_executor(state: SharedState):
    """
    Execute each routed step through the appropriate pipeline (simple, medium, or complex)
    based on the task_complexity assigned during routing.

    Pipelines:
        - simple:  Direct execution (data collection, processing, basic viz)
        - medium:  Execution with validation and retry (descriptive stats, ANOVA, correlation)
        - complex: Multi-phase with LLM planning, execution, validation, and retry (regression, ML, GWR, spatial)

    Args:
        state: SharedState containing ctx, step_routing (from route_step_to_agent),
               and current_objective_key.

    Returns:
        Dict updating synthesized_result in shared state and saving to ctx["artifacts"].
    """
    ctx = state["ctx"]
    obj_key = state["current_objective_key"]
    step_routing = state.get("step_routing", {})
    routing_decisions = step_routing.get("routing", [])

    objective_objects = state.get("objective_objects", {})
    steps = objective_objects.get("steps", [])

    # ========= START OF TESTING (PLEASE DELETE) ===============================
    # DEBUG: limit execution to specific objectives (e.g. ctx["debug_only_objectives"] = ["objective_1"])
    debug_only = ctx.get("debug_only_objectives")
    if debug_only and obj_key not in debug_only:
        task_manager.log(ctx, f"[DEBUG] Skipping {obj_key} (not in debug_only_objectives)")
        return {"synthesized_result": json.dumps({"skipped": obj_key})}

    # ========= END OF TESTING (PLEASE DELETE) ===============================

    if not routing_decisions:
        task_manager.log(ctx, f"No routing decisions found for {obj_key}. Skipping execution.")
        return {"synthesized_result": json.dumps({"error": "No routing decisions"})}

    task_manager.log(ctx, f"Executing {len(routing_decisions)} steps for {obj_key}...")

    # =========================BUILDING THE SOLUTION GRAPH FOR OBJECTIVES =====================

    # TODO: Display the graph before execution
    task_manager.log(ctx, f"Executing {len(routing_decisions)} steps for {obj_key}...")
    helper.show_objective_overview_popup(ctx)


    stream_cb = ctx.get("_stream_callback")
    if stream_cb:
        stream_cb(f"Executing {len(routing_decisions)} steps...\n")


    step_results = []
    # Track output_nodes from failed or empty-data steps so downstream
    # steps that depend on them can be skipped instead of cascading failures.
    failed_output_nodes = set()   # node names whose data is missing or empty
    failed_step_numbers = set()   # step numbers that failed

    for i, operation_info in enumerate(steps):
        if helper.check_cancelled(ctx):
            break
        # Get routing info for the corresponding step
        operation_routing_info = routing_decisions[i] if i < len(routing_decisions) else {}
        agent_name = operation_routing_info.get("agent_name", "ModelingAgent")
        complexity = operation_routing_info.get("task_complexity", "medium")
        step_idx = i + 1
        operation_description = operation_info.get("step_description", "") or operation_info.get("description", f"Step {step_idx}")
        print(f"[DEBUG] Step {step_idx}: agent={agent_name}, complexity={complexity}, desc={operation_description}")

        # ── Check if this step depends on a failed/empty upstream step ────
        step_input_nodes = operation_info.get("input_nodes", [])
        step_depends_on = operation_info.get("depends_on", [])
        tainted_inputs = [n for n in step_input_nodes if n in failed_output_nodes]
        tainted_deps = [d for d in step_depends_on if d in failed_step_numbers]

        if tainted_inputs or tainted_deps:
            skip_reason_parts = []
            if tainted_inputs:
                skip_reason_parts.append(f"input(s) {tainted_inputs} from failed/empty step")
            if tainted_deps:
                skip_reason_parts.append(f"depends on failed step(s) {list(tainted_deps)}")
            skip_reason = "; ".join(skip_reason_parts)

            task_manager.log(ctx, f"\n  Step {step_idx} SKIPPED — {skip_reason}")
            if stream_cb:
                stream_cb(f"Step {step_idx}: {operation_description}\n")
                stream_cb(f"  ⚠ Skipped — {skip_reason}\n")

            # Mark this step's output as tainted too (cascade)
            step_output_node = operation_info.get("output_node", "")
            if step_output_node:
                failed_output_nodes.add(step_output_node)
            failed_step_numbers.add(step_idx)

            # Record skipped result
            event_cb = ctx.get("_event_callback")
            if event_cb:
                event_cb({
                    "type": "step_result",
                    "step": ctx.get("_stream_step", "objective_execution"),
                    "step_index": step_idx,
                    "step_description": operation_description,
                    "status": "skipped",
                    "output": "",
                    "code": "",
                    "code_file": "",
                    "objective_key": obj_key,
                    "artifacts": [],
                    "error": f"Skipped: {skip_reason}",
                    "error_traceback": "",
                    "empty_data_warning": "",
                })
            step_results.append({
                "step_index": step_idx,
                "agent_name": agent_name,
                "task_complexity": complexity,
                "status": "skipped",
                "step_description": operation_description,
                "error_message": f"Skipped: {skip_reason}",
                "objective_key": obj_key,
                "result": {},
            })
            ctx.setdefault("artifacts", {}).setdefault("step_execution_results", {})[obj_key] = step_results
            continue

        # Update stream step tag for this specific step
        obj_tag = ctx.get("_stream_step", "objective_execution")
        # Derive base obj tag (e.g. exec_obj1) and set step-level tag
        base_obj_tag = obj_tag.split("_step")[0] if "_step" in obj_tag else obj_tag
        ctx["_stream_step"] = f"{base_obj_tag}_step{step_idx}"

        task_manager.log(ctx, f"\n  Step {step_idx} → {agent_name} [{complexity}]")
        if stream_cb:
            stream_cb(f"Step {step_idx}: {operation_description}\n")
            stream_cb(f"  Agent: {agent_name} | Complexity: {complexity}\n")

        # Get agent instance from registry
        agent_info = task_manager.agent_registry.get(agent_name)
        if not agent_info:
            task_manager.log(ctx, f"  ERROR: Agent '{agent_name}' not found in registry. Skipping step {step_idx}.")
            step_results.append({
                "step_index": step_idx,
                "agent_name": agent_name,
                "task_complexity": complexity,
                "status": "error",
                "objective_key": obj_key,
                "error": f"Agent '{agent_name}' not registered"
            })
            # Taint this step's output
            step_output_node = operation_info.get("output_node", "")
            if step_output_node:
                failed_output_nodes.add(step_output_node)
            failed_step_numbers.add(step_idx)
            continue

        agent_instance = agent_info["instance"]

        # Dispatch to appropriate pipeline
        if stream_cb:
            stream_cb(f"  Executing...\n")
        pipeline_fn = PIPELINE_DISPATCH.get(complexity, _operation_code_gen_medium_pipeline)
        result = pipeline_fn(ctx, operation_info, agent_instance)

        # ── Reconcile outputs: update data registry & fix subsequent input paths ──
        _reconcile_step_outputs(ctx, operation_info, result, steps, i)

        # ── Track failed/empty steps for downstream dependency skipping ────
        inner_result = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
        _step_status = result.get("status", "unknown")
        _step_empty_data = inner_result.get("empty_data_files", [])
        if _step_status == "error" or (_step_empty_data and not inner_result.get("created_files")):
            step_output_node = operation_info.get("output_node", "")
            if step_output_node:
                failed_output_nodes.add(step_output_node)
            failed_step_numbers.add(step_idx)

        # ── Surface error info from result (always, regardless of event_cb) ────
        inner = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
        step_status = result.get("status", "unknown")
        # Build error message — fall back to error_message or exception_type
        # to handle exceptions with empty str(err) (e.g. IterationError with no text)
        error_msg = (result.get("error", "")
                     or inner.get("error", "")
                     or inner.get("error_message", "")
                     or (inner.get("exception_type", "") if step_status == "error" else ""))

        # ── Emit step_result event so frontend can display results ────────────
        event_cb = ctx.get("_event_callback")
        if event_cb:
            step_code = inner.get("code", "")
            step_output = inner.get("output", "")
            created = inner.get("created_files", [])

            # Use the root Outputs dir (not the per-task ctx["output_dir"]) so
            # relative_path includes the task_id segment and resolves via
            # /api/artifacts/<task_id>/<file>.
            from utils.config import load_configurations
            _output_dir = load_configurations().output_dir

            artifacts = []
            for fpath in (created or []):
                fname = os.path.basename(fpath)
                ext = os.path.splitext(fname)[1].lstrip(".").lower()
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    fsize = 0
                try:
                    rel = os.path.relpath(fpath, _output_dir).replace("\\", "/")
                except ValueError:
                    rel = fname
                artifacts.append({
                    "path": fpath, "relative_path": rel, "filename": fname,
                    "size": fsize, "type": ext,
                    "is_shapefile_component": ext in ("shx", "dbf", "prj", "cpg"),
                })

            event_cb({
                "type": "step_result",
                "step": ctx.get("_stream_step", "objective_execution"),
                "step_index": step_idx,
                "step_description": operation_description,
                "status": step_status,
                "output": step_output or "",
                "code": step_code,
                "code_file": inner.get("code_file", ""),
                "objective_key": obj_key,
                "artifacts": artifacts,
                "error": error_msg if error_msg else None,
                "error_traceback": inner.get("error_traceback", ""),
                "empty_data_warning": inner.get("empty_data_warning", ""),
            })

        # ── Track step result for checkpoint summary ──────────────────────────
        step_results.append({
            "step_index": step_idx,
            "agent_name": agent_name,
            "task_complexity": complexity,
            "status": step_status,
            "step_description": operation_description,
            "error_message": error_msg if error_msg else None,
            "objective_key": obj_key,
            "result": inner,
        })
        ctx.setdefault("artifacts", {}).setdefault("step_execution_results", {})[obj_key] = step_results
        # Persist context incrementally so it survives interruptions
        _task_id = ctx.get("task_id")
        if _task_id:
            research_plan_generator.store_context(_task_id, ctx)
        # Call optional disk-persist callback set by the workflow module
        _persist_cb = ctx.get("_persist_callback")
        if _persist_cb:
            try:
                _persist_cb(ctx)
            except Exception:
                pass


# ── Post-step reconciliation ────────────────────────────────────────────────

def _reconcile_step_outputs(ctx, operation_info, result, all_steps, current_idx):
    """
    After a step executes, reconcile its actual outputs with the planned paths.

    1. Register every created file in the data registry so subsequent steps'
       `list_datasets()` tool can see them.
    2. If the planned output_node_path does NOT exist on disk but one of the
       created files matches (same extension / similar name), remap subsequent
       steps' input_node_paths to the actual file.
    """
    inner = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
    created_files = inner.get("created_files", [])
    if not created_files:
        return

    # ── 1. Register created files in the data registry ──────────────────
    data_registry = ctx.setdefault("artifacts", {}).setdefault("data_registry", [])
    existing_paths = {src.get("path") for src in data_registry}

    for fpath in created_files:
        if not os.path.exists(fpath):
            continue
        norm = os.path.normpath(fpath)
        if norm in existing_paths or fpath in existing_paths:
            continue

        # Derive basic metadata
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        shape_str = ""
        try:
            if ext == ".csv":
                import pandas as pd
                df = pd.read_csv(fpath, nrows=0)
                nrows = sum(1 for _ in open(fpath, encoding="utf-8", errors="ignore")) - 1
                shape_str = f"({nrows}, {len(df.columns)})"
            elif ext in (".gpkg", ".shp", ".geojson"):
                import geopandas as gpd
                gdf = gpd.read_file(fpath, rows=0)
                import fiona
                with fiona.open(fpath) as src:
                    nrows = len(src)
                shape_str = f"({nrows}, {len(gdf.columns)})"
            elif ext == ".json":
                shape_str = f"({os.path.getsize(fpath)} bytes)"
        except Exception:
            pass

        data_registry.append({
            "file": fname,
            "path": fpath,
            "shape": shape_str,
            "source": "step_output",
        })
        existing_paths.add(fpath)

    # ── 2. Verify planned output exists; remap if needed ────────────────
    planned_output = operation_info.get("output_node_path", "")
    output_node_name = operation_info.get("output_node", "")

    if not planned_output or not output_node_name:
        return

    # Check if the planned output file actually exists
    if os.path.exists(planned_output):
        actual_output = planned_output
    else:
        # Planned output doesn't exist — find the best match among created files
        actual_output = _find_best_match(planned_output, created_files)
        if actual_output:
            task_manager.log(
                ctx,
                f"  [RECONCILE] Planned output '{os.path.basename(planned_output)}' not found. "
                f"Remapping to actual file: '{os.path.basename(actual_output)}'",
            )
        else:
            return  # No match found — nothing to remap

    # ── 3. Update input_node_paths in all subsequent steps ──────────────
    if actual_output != planned_output:
        for future_step in all_steps[current_idx + 1:]:
            inp_paths = future_step.get("input_node_paths", {})
            for node_name, node_path in inp_paths.items():
                if node_path == planned_output or node_name == output_node_name:
                    inp_paths[node_name] = actual_output
                    task_manager.log(
                        ctx,
                        f"  [RECONCILE] Updated '{node_name}' in step "
                        f"{future_step.get('step_number', '?')}: {os.path.basename(actual_output)}",
                    )


def _find_best_match(planned_path, created_files):
    """
    Find the created file that best matches the planned output path.
    Matching priority:
        1. Same extension + filename contains planned stem
        2. Same extension only
        3. Any data file (gpkg/csv/shp/geojson/json)
    """
    planned_ext = os.path.splitext(planned_path)[1].lower()
    planned_stem = os.path.splitext(os.path.basename(planned_path))[0].lower()

    # Filter to files that actually exist
    existing = [f for f in created_files if os.path.exists(f)]
    if not existing:
        return None

    # Priority 1: same extension + stem overlap
    for fp in existing:
        ext = os.path.splitext(fp)[1].lower()
        stem = os.path.splitext(os.path.basename(fp))[0].lower()
        if ext == planned_ext and (planned_stem in stem or stem in planned_stem):
            return fp

    # Priority 2: same extension
    for fp in existing:
        if os.path.splitext(fp)[1].lower() == planned_ext:
            return fp

    # Priority 3: any data file
    data_exts = {".gpkg", ".csv", ".shp", ".geojson", ".json", ".parquet", ".tif"}
    for fp in existing:
        if os.path.splitext(fp)[1].lower() in data_exts:
            return fp

    return None


# ============================= NEW CODES

def _operation_code_gen_medium_pipeline(ctx, operation_info, agent_instance):
    """
    Medium pipeline: Same as simple — direct agent execution.
    Kept as separate function for future differentiation if needed.
    """
    return _operation_code_gen_simple_pipeline(ctx, operation_info, agent_instance)

def _operation_code_gen_complex_pipeline(ctx, operation_info, agent_instance):
    """
    Complex pipeline: Same as simple — direct agent execution.
    Kept as separate function for future differentiation if needed.
    """
    return _operation_code_gen_simple_pipeline(ctx, operation_info, agent_instance)

def _operation_code_gen_simple_pipeline(ctx, operation_info, agent_instance):
    """
    Simple pipeline: Direct agent execution with minimal orchestration.
    In interactive mode, pauses after code generation for user review.

    Flow (default): Agent generates code → executes → returns result
    Flow (interactive): Agent generates code → pause for review → user approves/edits → executes
    """
    task_manager.log(ctx, f"  [SIMPLE] Executing: {operation_info.get('description', '')}")


    _settings = ctx.get("settings", {})
    interactive = _settings.get("interactive_code", False) if isinstance(_settings, dict) else getattr(_settings, "interactive_code", False)
    review_queue = ctx.get("_code_review_queue")
    response_queue = ctx.get("_code_review_response_queue")

    if interactive and review_queue and response_queue:
        return generate_code_for_operation_interactive(ctx, operation_info, agent_instance,
                                             review_queue, response_queue)

    try:
        result = agent_instance.handle_generate_and_execute_code_for_simple_operation(ctx, operation_info)
        return {
            "status": "completed" if result.get("status") != "error" else "error",
            "pipeline": "simple",
            "error_source": "code_execution" if result.get("status") == "error" else None,
            "result": result
        }
    except Exception as e:
        task_manager.log(ctx, f"  [PIPELINE ERROR] {e}")
        return {
            "status": "error",
            "pipeline": "simple",
            "error_source": "pipeline",
            "error": str(e)
        }


def generate_code_for_operation_interactive(ctx, operation_info, agent_instance, review_queue, response_queue, extra_instructions=""): #equivalent to _execute_interactive_pipeline
    """
    Interactive pipeline: generate code, pause for user review, then execute.
    Support approve (with optional edits), feedback (regenerate), and skip.

    input:
        operation_info = steps info ('operation_type', 'operation', 'description', 'input_nodes', 'output_node', 'expected_output')

    output: generated code for step, to be sent to frontend.

    """
    MAX_FEEDBACK_ROUNDS = 5



    for round_num in range(MAX_FEEDBACK_ROUNDS):
        if helper.check_cancelled(ctx):
            return {"status": "error", "pipeline": "simple", "error": "Cancelled by user"}

        #Generate code only (no execution)

        try:
            gen_result = agent_instance.handle_generate_code_for_simple_operation(ctx, operation_info, extra_instructions=extra_instructions) #equivalent to handle_simple_step_generate_only

        except Exception as e:
            return {"status": "error", "pipeline": "simple", "error_source": "pipeline", "error": str(e)}

        if gen_result.get("status") == "error":
            return {"status": "error", "pipeline": "simple", "error_source": "code_generation", "result": gen_result}

        code = gen_result.get("code", "")
        code_file = gen_result.get("code_file", "")
        input_node_paths = gen_result.get("input_node_paths", "")
        output_node_path = gen_result.get("output_node_path", "")

        operation_desccription = operation_info.get("step_description", "") or operation_info.get("description", "")


        # Send code to frontend for review via the pending queue
        stream_cb = ctx.get("_stream_callback")
        if stream_cb:
            stream_cb(f"\nCode generated. Waiting for review...\n")

        if review_queue and response_queue:
            review_queue.put({
                "type": "code_review",
                "code": code,
                "code_file": code_file,
                "step_index": operation_info.get("step_number", 0),
                "step_description": operation_desccription,
                "agent_name": operation_info.get("agent_name", ""),
                "step": ctx.get("_stream_step", "objective_execution"),
                "round": round_num + 1,
            })

            # Block until user responds
            while True:
                if helper.check_cancelled(ctx):
                    return {"status": "error", "pipeline": "simple", "error": "Cancelled by user"}
                try:
                    user_response = response_queue.get(timeout=0.5)
                    break
                except Exception:
                    continue
        else:
            # No review queue — auto-approve
            user_response = {"action": "approve"}

        action = user_response.get("action", "approve")

        if action == "approve":
            # Use user's (possibly edited) code
            final_code = user_response.get("code", code) or code
            if final_code != code and code_file:
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(final_code)
            if stream_cb:
                stream_cb(f"Running code...\n")
            exec_result = agent_instance._execute_generated_code(ctx, final_code, operation_desccription)
            exec_result["code_file"] = code_file
            return {
                "status": "completed" if exec_result.get("status") != "error" else "error",
                "pipeline": "simple",
                "error_source": "code_execution" if exec_result.get("status") == "error" else None,
                "result": exec_result,
            }

        elif action == "feedback":
            # Regenerate with user feedback
            feedback = user_response.get("feedback", "")
            if stream_cb:
                stream_cb(f"Regenerating with feedback...\n")
            extra_instructions = feedback
            continue  # next round

        elif action == "skip":
            if stream_cb:
                stream_cb(f"Step skipped by user.\n")
            return {
                "status": "completed",
                "pipeline": "simple",
                "result": {"status": "skipped", "output": "Skipped by user", "code": code,
                           "code_file": code_file, "created_files": []},
            }

    # Exhausted feedback rounds
    return {"status": "error", "pipeline": "simple",
            "error": f"Exceeded {MAX_FEEDBACK_ROUNDS} feedback rounds"}



# Pipeline dispatcher
PIPELINE_DISPATCH = {
    "simple": _operation_code_gen_simple_pipeline,
    "medium": _operation_code_gen_medium_pipeline,
    "complex": _operation_code_gen_complex_pipeline,
}





def _route_after_execution(state: SharedState) -> str:
    """Route to debugger if any steps have errors, otherwise end."""
    try:
        step_results = json.loads(state.get("synthesized_result", "") or "[]")
    except Exception:
        step_results = []

    if any(r.get("error_message") for r in step_results):
        return "step_debugger"
    return END


def step_debugger(state: SharedState) -> dict:
    """
    Escalation summary node: reached only when inline repair in step_executor failed
    to fix one or more steps. Logs the remaining failures for human review.
    Inline repair (run_debugger) has already been attempted per-step in step_executor.
    """
    ctx = state["ctx"]
    obj_key = state["current_objective_key"]

    try:
        step_results = json.loads(state.get("synthesized_result", "") or "[]")
    except Exception:
        step_results = []

    still_failed = [r for r in step_results if r.get("error_message")]
    task_manager.log(ctx, f"[DEBUGGER] {len(still_failed)} step(s) could not be auto-repaired for {obj_key}")

    debug_info = {}
    for r in still_failed:
        step_idx = r.get("step_index")
        task_manager.log(ctx, f"  [ESCALATED] Step {step_idx}: {r.get('error_message', '')[:80]}")
        debug_info[step_idx] = {
            "step_description": r.get("step_description", ""),
            "error_message":    r.get("error_message", ""),
            "error_analysis":   r.get("error_analysis", {}),
            "debug_repair":     r.get("debug_repair", {}),
        }

    return {"debug_info": debug_info}













