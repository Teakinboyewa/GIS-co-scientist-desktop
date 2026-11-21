"""
LangGraph pipeline for the Result Analysis Agent.

Collates execution results and prepares the figure catalog so the
manuscript pipeline can write directly from raw data.

Nodes:
  collate_results → prepare_figures
"""

import json
import os
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END
from utils.lazy_imports import lazy_function  # IPython loads on first use
Image = lazy_function("IPython.display", "Image")
display = lazy_function("IPython.display", "display")
Markdown = lazy_function("IPython.display", "Markdown")

from agents.result_analysis_agent.result_analyzer import (
    collate_results as _collate,
)
from agents.result_analysis_agent.figure_preparation import (
    prepare_figure_catalog,
    format_figure_catalog_for_prompt,
)
# v2 — manifest-driven catalog, active path. Old import retained above as a
# fallback that can be re-enabled by flipping USE_FIGURE_CATALOG_V2 below.
from agents.result_analysis_agent.figure_catalog_v2 import (
    build_catalog as build_catalog_v2,
    format_catalog_for_prompt as format_catalog_v2,
)

USE_FIGURE_CATALOG_V2 = True
from agents.task_manager import TaskManager
task_manager = TaskManager()


# ── Shared State ──────────────────────────────────────────────────────────────

class ResultAnalysisState(TypedDict):
    ctx: dict
    collated_results: dict


# ── Node Functions ────────────────────────────────────────────────────────────

def _stream(ctx, msg, step_tag=None):
    """Helper to stream progress via callback and optionally update step tag."""
    if step_tag:
        ctx["_stream_step"] = step_tag
    cb = ctx.get("_stream_callback")
    if cb:
        cb(msg)


def collate_node(state: ResultAnalysisState) -> dict:
    ctx = state["ctx"]
    _stream(ctx, "Collating execution results...\n", "ra_collate")

    # ── Clear stale cached data from previous runs so we rebuild from
    #    the current conversation's step_execution_results, not leftovers.
    artifacts = ctx.setdefault("artifacts", {})
    artifacts.pop("collated_results", None)
    artifacts.pop("figure_catalog", None)
    artifacts.pop("manuscript_brief", None)

    task_id = ctx.get("task_id", "?")
    output_dir = ctx.get("output_dir", "?")
    step_results = artifacts.get("step_execution_results", {})
    obj_keys = list(step_results.keys()) if isinstance(step_results, dict) else []
    task_manager.log(ctx, f"[GRAPH] Collating results... task_id={task_id}, "
                     f"output_dir={output_dir}, objectives={obj_keys}")

    collated = _collate(ctx)
    obj_count = len(collated.get("objectives", {})) if isinstance(collated, dict) else 0
    _stream(ctx, f"  Collated {obj_count} objective(s)\n")
    return {"collated_results": collated}


def figures_node(state: ResultAnalysisState) -> dict:
    ctx = state["ctx"]
    _stream(ctx, "Preparing figure catalog...\n", "ra_figures")
    task_manager.log(ctx, "[GRAPH] Preparing figure catalog...")
    if USE_FIGURE_CATALOG_V2:
        build_catalog_v2(ctx)  # populates both figure_catalog and figure_catalog_v2
        catalog = ctx["artifacts"].get("figure_catalog", [])
    else:
        catalog = prepare_figure_catalog(ctx)
    _stream(ctx, f"  Catalog: {len(catalog)} figure(s)\n")

    # Emit figure_catalog event so the frontend can display thumbnails
    event_cb = ctx.get("_event_callback")
    if event_cb and catalog:
        # Use the root Outputs dir so relative_path includes the task_id
        # segment and resolves via /api/artifacts/<task_id>/<file>.
        from utils.config import load_configurations
        output_dir = load_configurations().output_dir
        figures_payload = []
        for fig in catalog:
            fp = fig.get("file_path", "")
            try:
                rel = os.path.relpath(fp, output_dir).replace("\\", "/")
            except Exception:
                rel = os.path.basename(fp)
            figures_payload.append({
                "figure_number": fig.get("figure_number"),
                "relative_path": rel,
                "description": fig.get("description", ""),
                "auto_generated": fig.get("auto_generated", False),
                "objective_key": fig.get("objective_key", ""),
                "step_index": fig.get("step_index", 0),
                # v2 fields — let the UI pick the right icon and preview type
                # instead of treating every catalog entry as a PNG figure.
                "kind": fig.get("kind", "figure"),
                "caption": fig.get("caption", fig.get("description", "")),
            })
        event_cb({
            "type": "figure_catalog",
            "figures": figures_payload,
            "step": "ra_figures",
        })

    return {}


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_result_analysis_graph(ctx, show_graph=True):
    """
    Build and run the result analysis preparation pipeline.

    Flow:
        collate → prepare_figures

    Prepares collated results and figure catalog for the manuscript pipeline.
    """
    workflow = StateGraph(ResultAnalysisState)

    workflow.add_node("collate_results", collate_node)
    workflow.add_node("prepare_figures", figures_node)

    workflow.add_edge(START, "collate_results")
    workflow.add_edge("collate_results", "prepare_figures")
    workflow.add_edge("prepare_figures", END)

    graph = workflow.compile()

    if show_graph:
        display(Image(graph.get_graph().draw_mermaid_png()))

    task_manager.log(ctx, "=" * 60)
    task_manager.log(ctx, "[RESULT ANALYSIS GRAPH] Starting pipeline...")

    result = graph.invoke({
        "ctx": ctx,
        "collated_results": {},
    })

    task_manager.log(ctx, "[RESULT ANALYSIS GRAPH] Pipeline complete.")

    return {
        "collated_results": result.get("collated_results", {}),
    }


def build_update_result_analysis_graph(ctx, show_graph=False):
    """
    Re-collate results and rebuild figure catalog after a step re-run.

    Same as build_result_analysis_graph but with show_graph=False default.
    """
    return build_result_analysis_graph(ctx, show_graph=show_graph)
