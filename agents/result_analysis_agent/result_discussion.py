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
# api_key = helper.load_OpenAI_key()
settings = load_configurations()


# ── Discussion Tools ──────────────────────────────────────────────────────────

def _build_discussion_tools(ctx):
    """Tools for the discussion agent to ground its writing in actual data."""

    @tool
    def get_result_presentation() -> str:
        """Get the result presentation markdown report.
        This is your primary source for discussing findings."""
        content = ctx.get("artifacts", {}).get("result_presentation_content", "")
        if not content:
            path = ctx.get("artifacts", {}).get("result_presentation_md", "")
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
        # Truncate to avoid token overflow — keep the most important parts
        if content and len(content) > 15000:
            content = content[:15000] + "\n\n... [truncated — use read_output_file for specific details]"
        return content if content else "No result presentation found."

    @tool
    def get_research_context() -> str:
        """Get research question, objectives, hypotheses, and data context."""
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
        return json.dumps({
            "research_question": rq,
            "objectives": objectives,
            "data_context": plan.get("data_context", {}),
            "key_considerations": plan.get("key_considerations", []),
        }, indent=2, default=str)[:5000]

    @tool
    def read_output_file(file_path: str) -> str:
        """Read an output file to extract specific statistics for discussion.
        Supports csv, json, gpkg, txt files."""
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".csv":
                from utils.table_summary import load_table, size_fields
                df, total, sampled = load_table(file_path)
                stats = df.describe().to_string()[:2000]
                from utils.table_rows import table_rows_text
                return json.dumps({
                    **size_fields(df, total, sampled),
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
                    **size_fields(gdf, total, sampled), "crs": str(gdf.crs),
                    "columns": list(gdf.columns)[:50],
                    "statistics": stats,
                }, default=str)[:4000]
            elif ext == ".json":
                with open(file_path, "r") as f:
                    return json.dumps(json.load(f), default=str)[:3000]
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()[:3000]
        except Exception as e:
            return f"Error: {e}"

    @tool
    def run_analysis_snippet(code: str) -> str:
        """Execute Python code (max 50 lines) to compute additional statistics
        for the discussion. E.g., effect sizes, confidence intervals, comparisons."""
        lines = code.strip().split("\n")
        if len(lines) > 50:
            return "Error: max 50 lines."
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<discussion_analysis>", "exec"))
            return buf.getvalue()[:3000] or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_data_summary() -> str:
        """Get the data summary including variable categorizations,
        data quality, and limitations. Useful for writing the limitations section."""
        summary = ctx.get("artifacts", {}).get("data_summary_data", {})
        result = json.dumps(summary, indent=2, default=str) if summary else "No data summary."
        return result[:5000]

    # ── Literature Search & Citation ────────────────────────────────────
    citation_store = ctx.setdefault("artifacts", {}).setdefault("citation_store", {})
    _search_cache = {}

    @tool
    def search_literature(query: str, year_range: str = "") -> str:
        """Search academic literature (Semantic Scholar) for real papers to cite.
        Returns papers with titles, authors, years, venues, DOIs, and citation counts.

        Args:
            query: Search terms (e.g. "spatial autocorrelation health outcomes")
            year_range: Optional year filter (e.g. "2015-2023")
        """
        from utils.citation_tools import search_papers
        results = search_papers(query, limit=5, year_range=year_range or None)
        if isinstance(results, dict) and "error" in results:
            return f"Search error: {results['error']}. Try again or use different terms."
        if not results:
            return "No results found. Try broader or different search terms."
        for p in results:
            _search_cache[p["paper_id"]] = p
        lines = [f"Found {len(results)} papers:\n"]
        for i, p in enumerate(results, 1):
            authors_str = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors_str += " et al."
            lines.append(
                f"[{i}] paper_id: {p['paper_id']}\n"
                f"    Title: {p['title']}\n"
                f"    Authors: {authors_str}\n"
                f"    Year: {p['year']}\n"
                f"    Venue: {p['venue']}\n"
                f"    DOI: {p['doi']}\n"
                f"    Citations: {p['citation_count']}\n"
                f"    Abstract: {p['abstract'][:250]}...\n"
            )
        return "\n".join(lines)

    @tool
    def add_citation(paper_id: str) -> str:
        """Add a paper from search_literature results to the citation list.
        Returns the reference key [REF_N] and inline citation format.

        Args:
            paper_id: The paper_id from search_literature results.
        """
        from utils.citation_tools import format_authors_inline
        if paper_id not in _search_cache:
            return (f"Paper ID '{paper_id}' not found in recent search results. "
                    f"Call search_literature first.")
        for ref_num, stored in citation_store.items():
            if stored.get("paper_id") == paper_id:
                inline = format_authors_inline(stored["authors"])
                return (f"Already cited as [REF_{ref_num}]. "
                        f"Use inline: ({inline}, {stored['year']}) [REF_{ref_num}]")
        paper = _search_cache[paper_id]
        ref_num = len(citation_store) + 1
        citation_store[str(ref_num)] = paper
        inline = format_authors_inline(paper["authors"])
        return (f"Added as [REF_{ref_num}]. "
                f"Use inline: ({inline}, {paper['year']}) [REF_{ref_num}]")

    @tool
    def get_citations() -> str:
        """Get all citations added so far. Use to check existing citations before adding duplicates."""
        if not citation_store:
            return "No citations added yet."
        from utils.citation_tools import format_authors_inline
        lines = [f"Current citations ({len(citation_store)} total):\n"]
        for ref_num in sorted(citation_store.keys(), key=int):
            p = citation_store[ref_num]
            inline = format_authors_inline(p["authors"])
            lines.append(f"  [REF_{ref_num}] ({inline}, {p['year']}) — {p['title']}")
        return "\n".join(lines)

    return [
        get_result_presentation, get_research_context,
        read_output_file, run_analysis_snippet, get_data_summary,
        search_literature, add_citation, get_citations,
    ]


# ── System Prompt ─────────────────────────────────────────────────────────────

DISCUSSION_SYSTEM_PROMPT = """You are an expert academic researcher writing the DISCUSSION section of a scientific research paper in GIScience / spatial analysis / quantitative social science.

Your task is to produce a thorough, scholarly Discussion section grounded in the actual results from the analysis pipeline. This section interprets what the results MEAN, not just what they ARE.

## Your Process:
1. Call `get_research_context` to understand the research question, objectives, and hypotheses
2. Call `get_result_presentation` to read the full results report
3. Use `read_output_file` or `run_analysis_snippet` if you need to verify specific numbers
4. Call `get_data_summary` for the limitations section
5. Write the discussion grounded in these actual findings

## Discussion Structure (FOLLOW THIS EXACTLY):

```markdown
# Discussion

## Summary of Key Findings
[Open with a concise paragraph restating the research question and summarizing the most important findings across all objectives. Ground every claim in specific results (coefficients, p-values, patterns). This paragraph should give the reader the "big picture" before diving into details.]

## Detailed Discussion

### [Objective 1 Title]
#### Interpretation of Findings
[Interpret what the results mean in the context of the hypothesis. Did the data support or refute the hypothesis? Discuss the direction, magnitude, and significance of key relationships. Explain WHY these patterns might exist — what theoretical or practical mechanisms could explain the observed relationships?]

#### Comparison with Existing Literature
[Compare findings with previous research. Use `search_literature` to find relevant prior work and `add_citation` to cite them. Call `get_citations` first to see what's already been cited.]
- "These findings are consistent with (Author et al., YEAR) [REF_N] who reported that..."
- "In contrast to (Author et al., YEAR) [REF_N] who found..., our results suggest..."
- "This aligns with the broader literature on... (Author et al., YEAR) [REF_N]"

#### Novel Contributions
[What new knowledge does this objective's analysis contribute? What has not been shown before or what does this add to the existing body of knowledge?]

### [Objective 2 Title]
[Same structure: Interpretation → Literature Comparison → Novel Contributions]

### [Objective N Title]
[Same structure]

## Synthesis Across Objectives
[Discuss how findings from different objectives relate to each other. Are there overarching patterns? Do the objectives together tell a coherent story? How do the different analytical approaches (descriptive, spatial, modeling) complement each other?]

## Significance and Implications
[What do these findings mean for the field? For policy? For practice? Discuss both theoretical and practical implications. Be specific — connect implications to actual findings.]

## Limitations
[Discuss limitations honestly and specifically:]
- **Data limitations**: Coverage gaps, temporal constraints, variable proxy issues, sample size concerns
- **Methodological limitations**: Model assumptions, potential confounders not controlled for, spatial scale issues (MAUP), ecological fallacy risks
- **Analytical limitations**: Steps that failed or produced partial results, sensitivity to model specification

## Future Research Directions
[Based on the limitations and findings, what should future research address?]
- Specific follow-up analyses that could strengthen the findings
- Additional data that would help (finer spatial resolution, longitudinal data, etc.)
- Methodological improvements (different models, validation approaches)
- New research questions that emerge from these findings
```

## Critical Rules:
1. **GROUND EVERY CLAIM IN DATA** — Never make a claim without referencing specific results (numbers, p-values, coefficients). Pull these from the result presentation and output files.
2. **REAL CITATIONS** — When referencing prior research, use `search_literature` to find real papers, then `add_citation` to register them. Cite inline as: `(AuthorLastName et al., YEAR) [REF_N]` using the format returned by add_citation. Call `get_citations` first to see existing citations. Do NOT invent citations.
3. **INTERPRET, DON'T REPEAT** — The Results section already presents the numbers. The Discussion explains what they MEAN, why they matter, and how they fit into the broader literature.
4. **BE SPECIFIC ABOUT SPATIAL CONTEXT** — When discussing spatial patterns, reference specific geographic areas, clustering patterns, and spatial heterogeneity observed.
5. **HONEST LIMITATIONS** — Don't downplay limitations. Acknowledge them clearly but also explain why the findings are still valuable despite them.
6. **ACADEMIC TONE** — Write formally, in third person, as in a peer-reviewed journal article.
7. **Return ONLY the markdown** — No JSON wrapping.
"""


# ── Main Function ─────────────────────────────────────────────────────────────

def generate_discussion(ctx):
    """
    Agentic discussion generation: LLM reads all results and research context,
    then produces a thorough academic Discussion section in markdown.
    """
    tools = _build_discussion_tools(ctx)
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(), 
        model_name=llm_model("result_analysis.discussion"), **llm_reasoning("result_analysis.discussion")
        )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=DISCUSSION_SYSTEM_PROMPT),
        HumanMessage(content=(
            "Write a thorough Discussion section for this research.\n"
            "Start by calling get_research_context, then get_result_presentation. "
            "Use read_output_file and run_analysis_snippet "
            "as needed to ground your discussion in specific numbers.\n"
            "Call get_data_summary for the limitations section."
        )),
    ]

    task_manager.log(ctx, "[DISCUSSION] Starting agentic discussion generation...")
    stream_cb = ctx.get("_stream_callback")

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
            task_manager.log(ctx, f"  [DISCUSS TOOL] {tc['name']}({_short_args(tc['args'])})")
            if stream_cb:
                stream_cb(f"\n🔧 [TOOL CALL] **{tc['name']}**\n")
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            if stream_cb:
                stream_cb(f"\n```\n{str(result)}\n```\n")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Extract markdown and strip internal [REF_N] markers
    import re
    discussion_md = response.content if hasattr(response, "content") else str(response)
    discussion_md = re.sub(r'\s*\[REF_\d+\]', '', discussion_md)

    # Save
    out_dir = resolve_output_dir(ctx)
    report_path = os.path.join(out_dir, "result_discussion.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(discussion_md)

    ctx.setdefault("artifacts", {})["result_discussion_md"] = report_path
    ctx.setdefault("artifacts", {})["result_discussion_content"] = discussion_md
    task_manager.log(ctx, f"[DISCUSSION] Discussion saved to {report_path}")

    # Generate DOCX and PDF versions (non-blocking — errors here must not stop the pipeline)
    try:
        from agents.result_analysis_agent.report_converter import convert_md_report
        convert_md_report(report_path, discussion_md, ctx, task_manager, "discussion")
    except Exception as e:
        task_manager.log(ctx, f"[DISCUSSION] DOCX/PDF conversion failed (non-fatal): {e}")

    return discussion_md


def _short_args(args):
    s = str(args)
    return s[:80] + "..." if len(s) > 80 else s


# ── Targeted Update ──────────────────────────────────────────────────────────

UPDATE_DISCUSSION_PROMPT = """You are an expert academic researcher. You are given an existing Discussion section and an UPDATED result presentation report.

The result presentation has been partially updated because some analysis steps were re-run. Your task is to **update ONLY the Discussion sections that correspond to changed results**. All other sections must remain EXACTLY as they were.

## Instructions:
1. Call `get_result_presentation` to read the updated presentation
2. Compare it with the existing discussion below — identify which objective sections have new/different results
3. For objectives with changed results, rewrite the Discussion subsection (Interpretation, Literature Comparison, Novel Contributions)
4. If changes affect "Synthesis Across Objectives", "Significance and Implications", or "Limitations", update those too
5. Return the COMPLETE discussion with updated section(s)

## Critical Rules:
- **DO NOT rewrite unchanged sections** — copy them verbatim
- **GROUND EVERY CLAIM IN DATA** — use specific numbers from the updated presentation
- **REAL CITATIONS** — use `search_literature` + `add_citation` for new citations; keep existing `[REF_N]` citations from unchanged sections
- **Academic tone** — formal, third person
- **Return ONLY the full markdown** — no JSON wrapping
"""


def update_discussion(ctx):
    """
    Update only the sections of the discussion that correspond to changed
    results in the presentation. Reads the existing discussion, compares
    with the updated presentation, and rewrites only outdated sections.
    """
    tools = _build_discussion_tools(ctx)
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(), 
        model_name=llm_model("result_analysis.discussion_update"), **llm_reasoning("result_analysis.discussion_update")
        )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    # Load existing discussion
    existing_discussion = ctx.get("artifacts", {}).get("result_discussion_content", "")
    if not existing_discussion:
        path = ctx.get("artifacts", {}).get("result_discussion_md", "")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing_discussion = f.read()
    if not existing_discussion:
        task_manager.log(ctx, "[UPDATE DISCUSSION] No existing discussion found — falling back to full generation.")
        return generate_discussion(ctx)

    user_msg = (
        f"Here is the **existing** Discussion section:\n\n"
        f"---BEGIN EXISTING DISCUSSION---\n{existing_discussion}\n---END EXISTING DISCUSSION---\n\n"
        f"The result presentation has been updated with new step results. "
        f"Call `get_result_presentation` to read the updated presentation, "
        f"then compare with the existing discussion. "
        f"Update ONLY the sections whose underlying results have changed. "
        f"Return the COMPLETE updated discussion."
    )

    messages = [
        SystemMessage(content=UPDATE_DISCUSSION_PROMPT),
        HumanMessage(content=user_msg),
    ]

    task_manager.log(ctx, "[UPDATE DISCUSSION] Comparing updated presentation with existing discussion...")
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
            task_manager.log(ctx, f"  [UPDATE DISCUSS TOOL] {tc['name']}({_short_args(tc['args'])})")
            if stream_cb:
                stream_cb(f"\n🔧 [TOOL CALL] **{tc['name']}**\n")
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            if stream_cb:
                stream_cb(f"\n```\n{str(result)}\n```\n")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    discussion_md = response.content if hasattr(response, "content") else str(response)

    # Save
    out_dir = resolve_output_dir(ctx)
    report_path = os.path.join(out_dir, "result_discussion.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(discussion_md)

    ctx.setdefault("artifacts", {})["result_discussion_md"] = report_path
    ctx.setdefault("artifacts", {})["result_discussion_content"] = discussion_md
    task_manager.log(ctx, f"[UPDATE DISCUSSION] Updated discussion saved to {report_path}")

    try:
        from agents.result_analysis_agent.report_converter import convert_md_report
        convert_md_report(report_path, discussion_md, ctx, task_manager, "discussion")
    except Exception as e:
        task_manager.log(ctx, f"[UPDATE DISCUSSION] DOCX/PDF conversion failed (non-fatal): {e}")

    return discussion_md