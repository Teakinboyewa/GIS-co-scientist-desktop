"""
Reporting Agent — drafts a full research manuscript.

Sections: Title, Abstract, Introduction, Methodology (Data + Methods),
Results, Discussion, Conclusion, References.

Uses a LangGraph pipeline where each node is an agentic LLM call
equipped with tools to inspect the pipeline's artifacts and outputs.
"""

from agents.reporting_agent.manuscript_pipeline import (
    build_manuscript_pipeline,
    build_update_manuscript_pipeline,
)

__all__ = [
    "build_manuscript_pipeline",
    "build_update_manuscript_pipeline",
]