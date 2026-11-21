"""
Ablation: the Research workflow with the Planning Agent removed.

Requested in review: "an ablation study that removes the Planning Agent and has
the Modeling Agent directly generate code". Everything else in the workflow
runs unchanged -- data audit, RQ breakdown, execution with the debugger, result
analysis, manuscript. Only two things differ:

  1. No research plan is generated, reviewed, or finalized. In its place is a
     one-objective, one-step "plan" whose single step is the research question
     itself. Downstream stages read research_plan_final, so it has to exist;
     it carries no planning decisions (no objectives, hypotheses, method
     choices, or step decomposition).
  2. The step router does not run. The single step goes straight to the
     ModelingAgent, which writes the code for the whole question.

The step description passes on the research question and the RQ breakdown.
In the full system that information reaches code generation through the plan,
so withholding it would remove more than the Planning Agent.
"""

import json
import os
import time

from utils.config import resolve_output_dir

NO_PLANNING = "no_planning"
ABLATION_AGENT = "ModelingAgent"
OUTPUT_NODE = "analysis_results.csv"


def ablation_of(ctx):
    settings = ctx.get("settings", {})
    return settings.get("ablation") if isinstance(settings, dict) else None


def _breakdown_context(rq_breakdown):
    scale = rq_breakdown.get("scale") or {}
    lines = []
    for label, value in [
        ("Intent", rq_breakdown.get("intent")),
        ("Brief description", rq_breakdown.get("brief_description")),
        ("Category", rq_breakdown.get("category")),
        ("Spatial scale", scale.get("spatial_scale")),
        ("Temporal scale", scale.get("temporal_scale")),
        ("Data requirement", rq_breakdown.get("data_requirement")),
    ]:
        if value:
            if not isinstance(value, str):
                value = json.dumps(value, default=str)
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_no_planning_plan(ctx):
    """Write research_plan_final for a run without the Planning Agent and
    store it on ctx exactly where finalize_research_plan would."""
    from agents.task_manager.research_plan_generator import _resolve_node_paths

    rq = ctx.get("rq", "")
    rq_breakdown = ctx.get("artifacts", {}).get("rq_breakdown", {}) or {}
    registry = ctx.get("artifacts", {}).get("data_registry", []) or []
    sources = [s.get("file") for s in registry if s.get("file")]

    context = _breakdown_context(rq_breakdown)
    description = (
        f"Answer the research question end to end: {rq}"
        + (f"\n\nResearch question breakdown:\n{context}" if context else "")
        + "\n\nDo all data preparation, analysis, and visualization this requires "
          f"in one script, and save the key quantitative results to {OUTPUT_NODE}."
    )

    plan = {
        "title": rq_breakdown.get("title", "") or rq,
        "research_question": rq,
        "research_intent": rq_breakdown.get("intent", ""),
        "ablation": NO_PLANNING,
        "objectives": [{
            "number": 1,
            "title": rq,
            "hypothesis": "",
            "data_requirements": [
                {"source_id": s, "description": "Uploaded dataset", "data_gap": False}
                for s in sources
            ],
            "steps": [{
                "step_number": 1,
                "operation_type": "statistical",
                "operation": "Direct code generation for the full research question (no plan)",
                "input_nodes": sources,
                "output_node": OUTPUT_NODE,
                "description": description,
                "expected_output": "The analyses, tables, and figures needed to answer "
                                   "the research question.",
            }],
        }],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _resolve_node_paths(plan, ctx)

    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "research_plan_final.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    artifacts = ctx.setdefault("artifacts", {})
    artifacts["research_plan_final_json"] = path
    artifacts["research_plan_final"] = plan
    return plan


def pinned_routing(steps):
    """Routing decisions that send every step to the ModelingAgent, in the
    shape route_step_to_agent returns, without the router's LLM call."""
    decisions = []
    for i, step in enumerate(steps, start=1):
        step = step if isinstance(step, dict) else {"description": str(step)}
        decisions.append({
            "step_index": i,
            "operation": step.get("operation", ""),
            "step_description": step.get("description", ""),
            "operation_type": step.get("operation_type", "unknown"),
            "agent_name": ABLATION_AGENT,
            "reasoning": "Ablation (no_planning): Planning Agent removed; "
                         "ModelingAgent generates code directly.",
            "execution_notes": "",
            "task_complexity": "medium",
        })
    return {"routing": decisions}
