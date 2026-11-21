import os
import json

from utils.config import load_configurations
from agents.task_manager import TaskManager

task_manager = TaskManager()
settings = load_configurations()


# ── Step 1: Collate Results ──────────────────────────────────────────────────

def collate_results(ctx):
    """
    Gather all step execution results across objectives into a structured summary.
    Returns dict keyed by objective_key with title, hypothesis, steps, and their results.
    """
    plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    step_results = ctx.get("artifacts", {}).get("step_execution_results", {})

    task_id = ctx.get("task_id", "unknown")
    rq_preview = (ctx.get("rq", "") or "")[:80]
    obj_keys = list(step_results.keys()) if isinstance(step_results, dict) else []
    total_steps = sum(len(v) for v in step_results.values()) if isinstance(step_results, dict) else 0
    print(f"[RESULT ANALYSIS] Collating for task_id={task_id}, "
          f"objectives={obj_keys}, total_steps={total_steps}, rq={rq_preview!r}")

    # Build objective metadata lookup
    obj_meta = {}
    for obj in plan.get("objectives", []):
        key = f"objective_{obj.get('number', 1)}"
        obj_meta[key] = {
            "title": obj.get("title", ""),
            "hypothesis": obj.get("hypothesis", ""),
            "planned_steps": obj.get("steps", []),
        }

    stream_cb = ctx.get("_stream_callback")
    collated = {}
    for obj_key, steps in step_results.items():
        meta = obj_meta.get(obj_key, {})
        step_summaries = []
        for s in steps:
            inner = s.get("result", {}) if isinstance(s.get("result"), dict) else {}
            step_summaries.append({
                "step_index": s.get("step_index"),
                "description": s.get("step_description", ""),
                "status": s.get("status", "unknown"),
                "agent": s.get("agent_name", ""),
                "output_preview": (inner.get("output", "") or "")[:500],
                "code_file": inner.get("code_file", ""),
                "created_files": inner.get("created_files", []),
                "error": s.get("error_message"),
            })

        completed = sum(1 for s in step_summaries if s["status"] == "completed")
        failed = sum(1 for s in step_summaries if s["status"] != "completed")
        collated[obj_key] = {
            "title": meta.get("title", obj_key),
            "hypothesis": meta.get("hypothesis", ""),
            "planned_steps": meta.get("planned_steps", []),
            "executed_steps": step_summaries,
            "total_steps": len(step_summaries),
            "completed": completed,
            "failed": failed,
        }
        if stream_cb:
            stream_cb(f"  {obj_key}: {completed}/{len(step_summaries)} steps completed\n")

    task_manager.log(ctx, f"[RESULT ANALYSIS] Collated results for {len(collated)} objectives")
    ctx.setdefault("artifacts", {})["collated_results"] = collated
    return collated


