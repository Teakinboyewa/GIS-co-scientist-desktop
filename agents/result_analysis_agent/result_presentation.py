import os
import json
import io
import contextlib

from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from utils.llm_models import llm_model, llm_reasoning
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import utils.agm_helper as helper
from utils.config import load_configurations, resolve_output_dir
from agents.task_manager import TaskManager

task_manager = TaskManager()

settings = load_configurations()



# ── Presentation Tools ────────────────────────────────────────────────────────

def _collect_session_files(ctx):
    """Collect all file paths created during this session from step execution results."""
    session_files = set()
    step_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    for obj_key, steps in step_results.items():
        for s in steps:
            inner = s.get("result", {}) if isinstance(s.get("result"), dict) else {}
            for fp in inner.get("created_files", []):
                session_files.add(fp)
            code_file = inner.get("code_file", "")
            if code_file:
                session_files.add(code_file)
    # Also include known artifact files
    artifacts = ctx.get("artifacts", {})
    for key in ("data_audit", "data_summary",
                "research_plan_json", "research_plan_md",
                "result_presentation_md", "result_discussion_md"):
        val = artifacts.get(key, "")
        if val and isinstance(val, str) and os.path.exists(val):
            session_files.add(val)
    return session_files


def _build_presentation_tools(ctx):
    """Tools for the presentation agent to inspect actual outputs."""
    session_files = _collect_session_files(ctx)

    @tool
    def read_output_file(file_path: str) -> str:
        """Read an output file (csv, json, gpkg, html, txt) and return its content/summary.
        For CSVs: returns shape, columns, descriptive statistics, and head rows.
        For spatial files: returns CRS, geometry info, and attribute summary."""
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            ext = os.path.splitext(file_path)[1].lower()
            size = os.path.getsize(file_path)

            if ext == ".csv":
                from utils.table_summary import load_table, size_fields
                df, total, sampled = load_table(file_path)
                stats = df.describe().to_string()[:2000]
                from utils.table_rows import table_rows_text
                return json.dumps({
                    "type": "csv", **size_fields(df, total, sampled),
                    "columns": list(df.columns)[:50],
                    "statistics": stats,
                    # Every row of a results table: a preview cut models off.
                    "rows": table_rows_text(df, total_rows=total),
                }, default=str)[:12000]
            elif ext in (".gpkg", ".shp", ".geojson"):
                from utils.table_summary import load_table, size_fields
                gdf, total, sampled = load_table(file_path)
                stats = gdf.drop(columns="geometry", errors="ignore").describe().to_string()[:2000]
                return json.dumps({
                    "type": "spatial", **size_fields(gdf, total, sampled),
                    "columns": list(gdf.columns)[:50], "crs": str(gdf.crs),
                    "statistics": stats,
                    "rows": table_rows_text(gdf, total_rows=total),
                }, default=str)[:4000]
            elif ext == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                return json.dumps(data, default=str)[:3000]
            elif ext == ".html":
                return f"HTML file ({size} bytes): {file_path}"
            elif ext in (".png", ".jpg", ".jpeg", ".tif"):
                return (
                    f"Image file ({size} bytes): {file_path}\n"
                    f"Embed this in your report using: ![description]({file_path})"
                )
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()[:3000]
        except Exception as e:
            return f"Error: {e}"

    @tool
    def list_output_files(directory: str = "") -> str:
        """List all output files created during this session (not historical files)."""
        try:
            files = []
            for fp in sorted(session_files):
                if os.path.exists(fp):
                    files.append({
                        "path": fp,
                        "size": os.path.getsize(fp),
                        "ext": os.path.splitext(fp)[1],
                    })
            return json.dumps(files, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_step_details(obj_key: str, step_index: int) -> str:
        """Get execution details for a specific step including code, output, and created files."""
        try:
            steps = ctx.get("artifacts", {}).get("step_execution_results", {}).get(obj_key, [])
            for s in steps:
                if s.get("step_index") == step_index:
                    inner = s.get("result", {}) if isinstance(s.get("result"), dict) else {}
                    return json.dumps({
                        "step_index": s.get("step_index"),
                        "description": s.get("step_description", ""),
                        "status": s.get("status", ""),
                        "agent": s.get("agent_name", ""),
                        "output": (inner.get("output", "") or "")[:2000],
                        "code_file": inner.get("code_file", ""),
                        "code": (inner.get("code", "") or "")[:2000],
                        "created_files": inner.get("created_files", []),
                        "error": s.get("error_message"),
                    }, indent=2, default=str)
            return f"No step {step_index} in {obj_key}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def run_analysis_snippet(code: str) -> str:
        """Execute a Python snippet (max 50 lines) to extract statistics, create summaries,
        or compute additional metrics from the pipeline outputs. Returns stdout output.
        Use this to compute statistics not already in the output files."""
        lines = code.strip().split("\n")
        if len(lines) > 50:
            return "Error: max 50 lines."
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<analysis>", "exec"))
            return buf.getvalue()[:3000] or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_research_context() -> str:
        """Get the full research context: research question, objectives, hypotheses, and data summary."""
        plan = ctx.get("artifacts", {}).get("research_plan_final", {})
        rq = plan.get("research_question", ctx.get("rq", ""))
        objectives = []
        for obj in plan.get("objectives", []):
            objectives.append({
                "number": obj.get("number"),
                "title": obj.get("title"),
                "hypothesis": obj.get("hypothesis"),
                "steps": obj.get("steps", []),
            })
        data_context = plan.get("data_context", {})
        return json.dumps({
            "research_question": rq,
            "objectives": objectives,
            "data_context": data_context,
        }, indent=2, default=str)[:5000]

    @tool
    def get_figure_catalog() -> str:
        """Get the numbered figure catalog: all available figures (existing plots + auto-generated
        visualizations from data files). Returns figure numbers, file paths, descriptions, and
        the markdown syntax to embed each one. Call this BEFORE writing any section that references
        figures, so you know exactly what's available and can use correct figure numbers."""
        from agents.result_analysis_agent.figure_preparation import format_figure_catalog_for_prompt
        catalog = ctx.get("artifacts", {}).get("figure_catalog", [])
        if not catalog:
            return "No figures in catalog. Use list_output_files to find image files manually."
        return format_figure_catalog_for_prompt(catalog)

    return [read_output_file, list_output_files, get_step_details, run_analysis_snippet, get_research_context, get_figure_catalog]


# ── System Prompt ─────────────────────────────────────────────────────────────

PRESENTATION_SYSTEM_PROMPT = """You are an expert research analyst producing a thorough academic result presentation.

Your task is to generate a comprehensive, publication-ready RESULT REPORT in Markdown format.
This report will later be used as input for discussion/conclusion drafting — so be THOROUGH and PRECISE.

## Your Process:
1. First, call `get_research_context` to understand the research question, objectives, and hypotheses
2. Call `list_output_files` to see all generated outputs
3. For EACH objective, call `get_step_details` for every step to understand what was executed
4. Use `read_output_file` to inspect actual data files (CSVs with regression results, spatial data, etc.)
5. Use `run_analysis_snippet` if you need to compute additional statistics from the data
6. Synthesize everything into a structured markdown report

## Report Structure (FOLLOW THIS EXACTLY):

```markdown
# Results

## Overview
[Brief summary of what analyses were conducted and the overall dataset used]

## Objective 1: [Title]
### Hypothesis
[State the hypothesis being tested]

### Methods Applied
[What statistical/spatial methods were used in this objective's steps]

### Descriptive Statistics
[Summary statistics of key variables — means, std dev, ranges. USE ACTUAL NUMBERS from the data]

### Analytical Results
[Detailed results — regression coefficients, p-values, R², spatial statistics, clustering results.
Present in tables where appropriate using markdown tables. USE ACTUAL NUMBERS.]

### Visualizations
[Embed any maps, charts, or plots using markdown image syntax: ![description](file_path)]

### Key Findings
[Bulleted list of specific findings with numbers]

### Hypothesis Evaluation
[Based on the evidence: supported / not supported / partially supported. Explain why with specific statistics.]

## Objective 2: [Title]
[Same structure as above]

...

## Cross-Objective Patterns
[Patterns that emerge across multiple objectives — common spatial trends, shared predictors, etc.]

## Summary of Key Results
[Numbered list of the most important findings across all objectives]

## Data and Methodological Notes
[Sample sizes, geographic coverage, any data quality issues encountered, limitations of the methods used]
```

## Critical Rules:
1. **USE REAL NUMBERS** — Always extract and report actual statistics from the output files. Never say "the coefficient was significant" without giving the actual value.
2. **Be specific** — Report exact p-values, coefficients, R² values, sample sizes, etc.
3. **Embed figures from the catalog** — Call `get_figure_catalog` first to see all available figures. Embed them using `![Figure N: description](file_path)` with the exact figure numbers and paths from the catalog. This includes auto-generated maps from spatial data files and charts from CSV results.
4. **Markdown tables** — Use tables for regression results, ANOVA results, correlation matrices, etc.
5. **Academic tone** — Write as if this will appear in a research paper's Results section.
6. **Thoroughness** — Cover every step and every output. Don't skip analyses.
7. **Spatial results** — For spatial analyses, report Moran's I values, GWR coefficient ranges, cluster characteristics, etc.
8. **Return ONLY the markdown report** — No JSON wrapping, no code blocks around the entire output.
"""


# ── Main Functions ────────────────────────────────────────────────────────────

def generate_result_presentation(ctx, collated):
    """
    Agentic result presentation: LLM inspects all outputs and produces
    a thorough markdown report covering all objectives.
    """
    # ── Prepare figure catalog (auto-generate viz from data files) ────
    from agents.result_analysis_agent.figure_preparation import (
        prepare_figure_catalog, format_figure_catalog_for_prompt,
    )
    catalog = prepare_figure_catalog(ctx)
    figure_summary = format_figure_catalog_for_prompt(catalog) if catalog else ""

    tools = _build_presentation_tools(ctx)
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        model_name=llm_model("result_analysis.presentation"), **llm_reasoning("result_analysis.presentation")
        )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    # Build context summary for the LLM
    obj_summaries = []
    for obj_key, data in collated.items():
        steps_info = []
        for s in data["executed_steps"]:
            steps_info.append(
                f"  Step {s['step_index']}: [{s['status']}] {s['description'][:120]}"
                + (f" | files: {s['created_files']}" if s['created_files'] else "")
            )
        obj_summaries.append(
            f"### {obj_key}: {data['title']}\n"
            f"Hypothesis: {data['hypothesis']}\n"
            f"Completed: {data['completed']}/{data['total_steps']}\n"
            + "\n".join(steps_info)
        )

    user_msg = (
        "Generate a thorough result presentation report for this research.\n"
        "Start by calling get_research_context and get_figure_catalog, then inspect each step's outputs.\n\n"
        + "\n\n".join(obj_summaries)
    )
    if figure_summary:
        user_msg += f"\n\n{figure_summary}"

    messages = [
        SystemMessage(content=PRESENTATION_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    task_manager.log(ctx, "[PRESENTATION] Starting agentic result presentation...")
    stream_cb = ctx.get("_stream_callback")

    # Agent loop — allow more iterations since this is a thorough report
    max_iterations = 15
    for i in range(max_iterations):
        if helper.check_cancelled(ctx):
            return "[Cancelled by user]"
        if stream_cb:
            stream_cb(f"  Iteration {i + 1}...\n")

        # Stream this turn (single round-trip). Content tokens flow live,
        # tool_call_chunks accumulate into structured tool_calls.
        if stream_cb:
            accumulated = None
            for chunk in llm_with_tools.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    stream_cb(token)
                accumulated = chunk if accumulated is None else accumulated + chunk
            response = accumulated
        else:
            response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Final content already streamed above
            break

        for tc in response.tool_calls:
            task_manager.log(ctx, f"  [PRESENT TOOL] {tc['name']}({_short_args(tc['args'])})")
            if stream_cb:
                stream_cb(f"\n🔧 [TOOL CALL] **{tc['name']}**\n")
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            if stream_cb:
                stream_cb(f"\n```\n{str(result)}\n```\n")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Extract markdown from response
    markdown_report = response.content if hasattr(response, "content") else str(response)

    # Save to file and ctx
    out_dir = resolve_output_dir(ctx)
    report_path = os.path.join(out_dir, "result_presentation.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    ctx.setdefault("artifacts", {})["result_presentation_md"] = report_path
    ctx.setdefault("artifacts", {})["result_presentation_content"] = markdown_report
    task_manager.log(ctx, f"[PRESENTATION] Report saved to {report_path}")

    # Generate DOCX and PDF versions (non-blocking — errors here must not stop the pipeline)
    try:
        from agents.result_analysis_agent.report_converter import convert_md_report
        convert_md_report(report_path, markdown_report, ctx, task_manager, "presentation")
    except Exception as e:
        task_manager.log(ctx, f"[PRESENTATION] DOCX/PDF conversion failed (non-fatal): {e}")

    return markdown_report


def _short_args(args):
    s = str(args)
    return s[:80] + "..." if len(s) > 80 else s


# ── Targeted Update ──────────────────────────────────────────────────────────

UPDATE_PRESENTATION_PROMPT = """You are an expert research analyst. You are given an existing result presentation report and the CURRENT collated step execution results.

Some steps have been re-run since the report was originally written — the collated results reflect the latest outputs.

Your task: **compare** the collated results with what the existing report says, identify sections that are now outdated, and **update ONLY those sections**. All sections whose results haven't changed must remain EXACTLY as they were — do not rewrite, rephrase, or re-order them.

## Instructions:
1. Call `get_research_context` to understand the overall research structure
2. For each objective, call `get_step_details` to see current outputs and compare with what the report states
3. Use `read_output_file` or `run_analysis_snippet` to inspect new/changed output files
4. Rewrite ONLY the sections whose underlying data has changed (e.g., updated statistics, new files, different results)
5. If changed objectives affect "Cross-Objective Patterns" or "Summary of Key Results", update those cross-references too
6. Return the COMPLETE markdown report with the updated section(s)

## Critical Rules:
- **DO NOT regenerate unchanged sections** — copy them verbatim
- **USE REAL NUMBERS** from the step outputs
- **Maintain the same structure** as the existing report
- **Return ONLY the full markdown report** — no JSON wrapping
"""


def update_result_presentation(ctx, collated):
    """
    Update only the sections of the result presentation that have changed
    due to re-run steps. Reads the existing presentation, compares with
    current collated results, and asks LLM to update outdated sections only.
    """
    tools = _build_presentation_tools(ctx)
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(), 
        model_name=llm_model("result_analysis.presentation_update"), **llm_reasoning("result_analysis.presentation_update")
        )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    # Load existing presentation
    existing_report = ctx.get("artifacts", {}).get("result_presentation_content", "")
    if not existing_report:
        path = ctx.get("artifacts", {}).get("result_presentation_md", "")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing_report = f.read()
    if not existing_report:
        task_manager.log(ctx, "[UPDATE PRESENTATION] No existing report found — falling back to full generation.")
        return generate_result_presentation(ctx, collated)

    # Build summary of current collated results
    obj_summaries = []
    for obj_key, data in collated.items():
        steps_info = []
        for s in data["executed_steps"]:
            steps_info.append(
                f"  Step {s['step_index']}: [{s['status']}] {s['description'][:120]}"
                + (f" | files: {s['created_files']}" if s['created_files'] else "")
            )
        obj_summaries.append(
            f"### {obj_key}: {data['title']}\n"
            f"Hypothesis: {data['hypothesis']}\n"
            f"Completed: {data['completed']}/{data['total_steps']}\n"
            + "\n".join(steps_info)
        )

    user_msg = (
        f"Here is the **existing** result presentation report:\n\n"
        f"---BEGIN EXISTING REPORT---\n{existing_report}\n---END EXISTING REPORT---\n\n"
        f"Here are the **current** collated step results (some may have changed):\n\n"
        + "\n\n".join(obj_summaries) + "\n\n"
        f"Compare these current results with what the existing report says. "
        f"Inspect each objective's step details using the tools. "
        f"Update ONLY the sections whose data has changed. Return the COMPLETE updated report."
    )

    messages = [
        SystemMessage(content=UPDATE_PRESENTATION_PROMPT),
        HumanMessage(content=user_msg),
    ]

    task_manager.log(ctx, "[UPDATE PRESENTATION] Comparing current results with existing report...")
    stream_cb = ctx.get("_stream_callback")

    max_iterations = 12
    for i in range(max_iterations):
        if helper.check_cancelled(ctx):
            return "[Cancelled by user]"
        if stream_cb:
            stream_cb(f"  Iteration {i + 1}...\n")

        # Stream this turn (single round-trip). Content tokens flow live,
        # tool_call_chunks accumulate into structured tool_calls.
        if stream_cb:
            accumulated = None
            for chunk in llm_with_tools.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    stream_cb(token)
                accumulated = chunk if accumulated is None else accumulated + chunk
            response = accumulated
        else:
            response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Final content already streamed above
            break

        for tc in response.tool_calls:
            task_manager.log(ctx, f"  [UPDATE TOOL] {tc['name']}({_short_args(tc['args'])})")
            if stream_cb:
                stream_cb(f"\n🔧 [TOOL CALL] **{tc['name']}**\n")
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            if stream_cb:
                stream_cb(f"\n```\n{str(result)}\n```\n")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    markdown_report = response.content if hasattr(response, "content") else str(response)

    # Save
    out_dir = resolve_output_dir(ctx)
    report_path = os.path.join(out_dir, "result_presentation.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    ctx.setdefault("artifacts", {})["result_presentation_md"] = report_path
    ctx.setdefault("artifacts", {})["result_presentation_content"] = markdown_report
    task_manager.log(ctx, f"[UPDATE PRESENTATION] Updated report saved to {report_path}")

    try:
        from agents.result_analysis_agent.report_converter import convert_md_report
        convert_md_report(report_path, markdown_report, ctx, task_manager, "presentation")
    except Exception as e:
        task_manager.log(ctx, f"[UPDATE PRESENTATION] DOCX/PDF conversion failed (non-fatal): {e}")

    return markdown_report


# ── Entry Point ───────────────────────────────────────────────────────────────

def prepare_results(ctx):
    """
    Result preparation pipeline:
    1. Collate results across objectives
    2. Generate thorough markdown presentation
    """
    from agents.result_analysis_agent.result_analyzer import collate_results

    task_manager.log(ctx, "=" * 60)
    task_manager.log(ctx, "[RESULT PREP] Starting result preparation pipeline...")

    # Step 1: Collate
    collated = collate_results(ctx)

    # Step 2: Generate presentation
    markdown_report = generate_result_presentation(ctx, collated)

    return {
        "collated_results": collated,
        "report": markdown_report,
        "report_path": ctx["artifacts"].get("result_presentation_md", ""),
    }