from .result_analyzer import collate_results
from .result_analysis_graph import build_result_analysis_graph, build_update_result_analysis_graph
from .execution_flowchart import generate_execution_flowchart
from .figure_preparation import prepare_figure_catalog, format_figure_catalog_for_prompt
from .figure_catalog_v2 import (
    build_catalog as build_figure_catalog,
    record_figure,
    record_table,
    record_data_to_viz,
    promote_intermediate,
    demote_intermediate,
    list_promotable,
    format_catalog_for_prompt,
)

__all__ = [
    "collate_results",
    "build_result_analysis_graph",
    "build_update_result_analysis_graph",
    "generate_execution_flowchart",
    # legacy
    "prepare_figure_catalog", "format_figure_catalog_for_prompt",
    # v2 (manifest-driven)
    "build_figure_catalog",
    "record_figure", "record_table", "record_data_to_viz",
    "promote_intermediate", "demote_intermediate", "list_promotable",
    "format_catalog_for_prompt",
]
