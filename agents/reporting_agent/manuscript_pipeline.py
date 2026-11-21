"""
LangGraph pipeline for the Reporting Agent.

Full manuscript pipeline:
    title_abstract → introduction → methodology → results
    → discussion → conclusion → references → assemble

**Speed optimisation (v2):** A single ``ManuscriptBrief`` is built once at
the start of the pipeline.  It pre-extracts every piece of data the LLM
needs (full step outputs, output-file stats, figure catalog, pre-registered
citations).  Most sections are now **single-shot** (one LLM call, zero
tools).  Only Introduction and Discussion have a 3-iteration escape hatch
for finding additional literature, and Results has one for ad-hoc stats.
"""

import os
import json
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END
from utils.lazy_imports import lazy_function  # IPython loads on first use
Image = lazy_function("IPython.display", "Image")
display = lazy_function("IPython.display", "display")

from utils.config import load_configurations, resolve_output_dir
from utils.llm_models import llm_model, llm_reasoning
from agents.task_manager import TaskManager

from agents.reporting_agent.section_writers import (
    write_title_abstract,
    write_introduction,
    write_methodology,
    write_results,
    write_discussion,
    write_conclusion,
    compile_references,
)
from agents.reporting_agent.manuscript_tools import build_manuscript_brief

task_manager = TaskManager()
settings = load_configurations()


# ── State ────────────────────────────────────────────────────────────────────

class ManuscriptState(TypedDict):
    ctx: dict
    drafted_sections: dict        # section_name → markdown content
    manuscript_md: str            # final assembled manuscript
    manuscript_path: str          # file path of saved manuscript


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stream(ctx, msg, step_tag=None):
    if step_tag:
        ctx["_stream_step"] = step_tag
    cb = ctx.get("_stream_callback")
    if cb:
        cb(msg)


def _emit_event(ctx, event):
    cb = ctx.get("_event_callback")
    if cb:
        cb(event)


def _batch_literature_search(ctx, max_papers: int = 15):
    """
    Run a single batch of Semantic Scholar searches BEFORE the manuscript
    pipeline starts.  Results are cached on ``ctx["artifacts"]["pre_searched_literature"]``
    and injected into the Introduction and Discussion writer prompts so the
    LLM doesn't need to spend iterations calling ``search_literature`` itself.

    Strategy: derive 2-4 search queries from the research question and the
    titles of each objective, then deduplicate by paper_id.  This typically
    eliminates 8-15 LLM round-trips across the two literature-heavy sections.
    """
    from utils.citation_tools import search_papers

    plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    rq = plan.get("research_question", ctx.get("rq", "")) or ""
    objectives = plan.get("objectives", []) or []

    queries = []
    if rq:
        queries.append(rq[:200])
    for obj in objectives[:3]:
        title = (obj.get("title") or "").strip()
        if title:
            queries.append(title[:200])

    if not queries:
        ctx.setdefault("artifacts", {})["pre_searched_literature"] = []
        return []

    task_manager.log(
        ctx, f"[MANUSCRIPT] Batch literature search ({len(queries)} queries)..."
    )
    _stream(ctx, f"\n🔍 Pre-fetching literature ({len(queries)} queries)...\n")

    seen = set()
    collected = []
    for q in queries:
        try:
            results = search_papers(q, limit=5)
        except Exception as e:
            task_manager.log(ctx, f"[MANUSCRIPT] Literature search failed for '{q}': {e}")
            continue
        if isinstance(results, dict) and "error" in results:
            task_manager.log(
                ctx, f"[MANUSCRIPT] Literature search error for '{q}': {results['error']}"
            )
            continue
        for p in results or []:
            pid = p.get("paper_id")
            if pid and pid not in seen:
                seen.add(pid)
                collected.append(p)
                if len(collected) >= max_papers:
                    break
        if len(collected) >= max_papers:
            break

    ctx.setdefault("artifacts", {})["pre_searched_literature"] = collected
    task_manager.log(
        ctx, f"[MANUSCRIPT] Pre-fetched {len(collected)} unique papers."
    )
    _stream(ctx, f"  Cached {len(collected)} unique papers for citation.\n")
    return collected


LITERATURE_ROLES = ("foundational", "methodological", "regional", "gap_evidence")


def _curate_and_expand_literature(ctx, *, max_follow_up_queries: int = 3,
                                  papers_per_query: int = 4,
                                  max_new_papers: int = 8):
    """
    Pre-draft pass that (a) tags each pre-searched paper with a role
    (foundational / methodological / regional / gap_evidence) and
    (b) identifies role-coverage gaps and runs follow-up Semantic Scholar
    searches to fill them.

    Roles:
        foundational   — seminal / widely-cited theoretical anchors for the topic
        methodological — prior work that establishes methods this study uses
        regional       — prior studies in or near the study region / population
        gap_evidence   — papers whose LIMITATIONS the Introduction's gap claim
                         rests on (specific scale/method/data limits)

    The output is written back to ``ctx["artifacts"]["pre_searched_literature"]``
    with each paper carrying a new ``role`` field, and to a sibling key
    ``literature_by_role`` grouped for the renderer.
    """
    import json as _json
    from utils.agm_helper import GIBDChat
    import utils.agm_helper as helper
    from utils.citation_tools import search_papers
    from langchain_core.messages import SystemMessage, HumanMessage

    artifacts = ctx.setdefault("artifacts", {})
    papers = artifacts.get("pre_searched_literature") or []
    if not papers:
        artifacts["literature_by_role"] = {r: [] for r in LITERATURE_ROLES}
        return papers

    plan = artifacts.get("research_plan_final", {}) or {}
    rq = plan.get("research_question", ctx.get("rq", "")) or ""
    objective_titles = [
        (o.get("title") or "") for o in (plan.get("objectives") or []) if isinstance(o, dict)
    ]

    # Compact paper list for the classifier — id, title, venue, abstract head.
    paper_digest = []
    for p in papers:
        paper_digest.append({
            "paper_id": p.get("paper_id"),
            "title": p.get("title", ""),
            "year": p.get("year"),
            "venue": p.get("venue", ""),
            "abstract": (p.get("abstract", "") or "")[:400],
        })

    system = (
        "You are a research librarian classifying a reading list for a "
        "GIScience / spatial-analysis manuscript. For EACH paper, assign "
        "exactly one role:\n"
        "- foundational: seminal / widely-cited theoretical anchor\n"
        "- methodological: prior methods this study reuses or extends\n"
        "- regional: study in or near the study region / population\n"
        "- gap_evidence: paper whose explicit LIMITATION (scale, method, "
        "data, region, vintage) is what leaves the gap open\n\n"
        "Then propose 1–3 short follow-up search queries for roles that are "
        "UNDER-COVERED in the list relative to the research question. "
        "Prefer queries that target a specific methodological or regional "
        "gap rather than general topic phrases.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "assignments": [{"paper_id": str, "role": str, "why": str}],\n'
        '  "missing_roles": [str],\n'
        '  "follow_up_queries": [{"role": str, "query": str}]\n'
        "}"
    )
    user = _json.dumps({
        "research_question": rq,
        "objectives": objective_titles,
        "papers": paper_digest,
    }, indent=2, default=str)

    try:
        llm = GIBDChat(
            # gibd_user_key=helper.load_OpenAI_key(),
            model_name=llm_model("manuscript.literature_curation"), **llm_reasoning("manuscript.literature_curation")
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
        parsed = _json.loads(raw)
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT] Literature curation skipped: {e}")
        artifacts["literature_by_role"] = {r: list(papers) if r == "foundational" else [] for r in LITERATURE_ROLES}
        return papers

    # Apply assignments
    role_by_id = {
        a.get("paper_id"): a.get("role")
        for a in (parsed.get("assignments") or [])
        if a.get("paper_id") and a.get("role") in LITERATURE_ROLES
    }
    for p in papers:
        pid = p.get("paper_id")
        p["role"] = role_by_id.get(pid, "foundational")

    # Execute follow-up queries for under-covered roles
    follow_ups = parsed.get("follow_up_queries") or []
    seen = {p.get("paper_id") for p in papers if p.get("paper_id")}
    added = 0
    for fu in follow_ups[:max_follow_up_queries]:
        if added >= max_new_papers:
            break
        role = fu.get("role")
        query = (fu.get("query") or "").strip()
        if not query or role not in LITERATURE_ROLES:
            continue
        try:
            results = search_papers(query, limit=papers_per_query)
        except Exception as e:
            task_manager.log(ctx, f"[MANUSCRIPT] Follow-up search failed: {query} ({e})")
            continue
        if isinstance(results, dict) and "error" in results:
            continue
        for p in results or []:
            pid = p.get("paper_id")
            if not pid or pid in seen:
                continue
            p["role"] = role
            papers.append(p)
            seen.add(pid)
            added += 1
            if added >= max_new_papers:
                break

    # Group for the renderer
    by_role = {r: [] for r in LITERATURE_ROLES}
    for p in papers:
        by_role.setdefault(p.get("role", "foundational"), []).append(p)

    artifacts["pre_searched_literature"] = papers
    artifacts["literature_by_role"] = by_role

    task_manager.log(
        ctx,
        "[MANUSCRIPT] Literature curated: "
        + ", ".join(f"{r}={len(by_role.get(r, []))}" for r in LITERATURE_ROLES)
        + f"  (added {added} via follow-up queries)"
    )
    _stream(
        ctx,
        "  Curated: " + ", ".join(f"{r}={len(by_role.get(r, []))}" for r in LITERATURE_ROLES)
        + f" (+{added} new)\n"
    )
    return papers


# ── Node Functions ───────────────────────────────────────────────────────────

# Heading map used to render a clean placeholder when a section fails so later
# sections can still cross-reference its title.
_SECTION_HEADINGS = {
    "title_abstract": "# [Title generation failed]\n\n## Abstract\n",
    "introduction":   "## 1. Introduction\n",
    "methodology":    "## 2. Methodology\n",
    "results":        "## 3. Results\n",
    "discussion":     "## 4. Discussion\n",
    "conclusion":     "## 5. Conclusion\n",
}


def _checkpoint_section(ctx, section_name, content):
    """
    Persist a section to disk immediately after it's written so a later
    crash never loses already-paid-for work.  Stored under
    output_dir/manuscript_sections/<section>.md.
    """
    try:
        out_dir = resolve_output_dir(ctx)
        ckpt_dir = os.path.join(out_dir, "manuscript_sections")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{section_name}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")
        task_manager.log(ctx, f"[MANUSCRIPT] Checkpointed {section_name} → {path}")
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT] Failed to checkpoint {section_name}: {e}")


def _run_section_safely(state: ManuscriptState, *, name: str, writer, step_tag: str,
                        display_name: str) -> dict:
    """
    Run a section writer with full error handling:
      - Catches any exception (LLM timeout, parse error, etc.)
      - Inserts a clean placeholder so later sections still have something to reference
      - Checkpoints the section to disk on success OR after a failure (placeholder)
      - Emits the appropriate manuscript_section event regardless of outcome
      - NEVER re-raises — the pipeline always continues to the next section
    """
    ctx = state["ctx"]
    drafted = dict(state.get("drafted_sections", {}))
    _stream(ctx, f"\n### Drafting {display_name}...\n", step_tag)

    try:
        md = writer(ctx, drafted_sections=drafted)
        if not md or not md.strip():
            raise RuntimeError("Writer returned empty content.")
        # Normalize pipe-tables so the UI renderer (marked/GFM) picks them
        # up as tables rather than merging them into the preceding paragraph.
        md = _ensure_blank_line_before_tables(md)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        task_manager.log(ctx, f"[MANUSCRIPT] {display_name} FAILED — {err_msg}")
        _stream(
            ctx,
            f"\n⚠️ **{display_name} failed:** {err_msg}\n"
            f"_Inserting placeholder so later sections can continue. "
            f"You can re-run this section later via the targeted update endpoint._\n",
            step_tag,
        )
        heading = _SECTION_HEADINGS.get(name, f"## {display_name}\n")
        md = (
            f"{heading}\n"
            f"_[Section generation failed: {err_msg}. "
            f"Re-run via Update Manuscript to regenerate.]_\n"
        )
        _emit_event(ctx, {
            "type": "manuscript_section_error",
            "section": name,
            "error": err_msg,
            "step": step_tag,
        })

    drafted[name] = md
    _checkpoint_section(ctx, name, md)
    _emit_event(ctx, {
        "type": "manuscript_section",
        "section": name,
        "content": md,
        "step": step_tag,
    })
    return {"drafted_sections": drafted}


def title_abstract_node(state: ManuscriptState) -> dict:
    return _run_section_safely(
        state, name="title_abstract", writer=write_title_abstract,
        step_tag="ms_title_abstract", display_name="Title & Abstract",
    )


def introduction_node(state: ManuscriptState) -> dict:
    return _run_section_safely(
        state, name="introduction", writer=write_introduction,
        step_tag="ms_introduction", display_name="Introduction",
    )


def methodology_node(state: ManuscriptState) -> dict:
    return _run_section_safely(
        state, name="methodology", writer=write_methodology,
        step_tag="ms_methodology", display_name="Methodology",
    )


def results_node(state: ManuscriptState) -> dict:
    return _run_section_safely(
        state, name="results", writer=write_results,
        step_tag="ms_results", display_name="Results",
    )


def discussion_node(state: ManuscriptState) -> dict:
    return _run_section_safely(
        state, name="discussion", writer=write_discussion,
        step_tag="ms_discussion", display_name="Discussion",
    )


def conclusion_node(state: ManuscriptState) -> dict:
    return _run_section_safely(
        state, name="conclusion", writer=write_conclusion,
        step_tag="ms_conclusion", display_name="Conclusion",
    )


def references_node(state: ManuscriptState) -> dict:
    ctx = state["ctx"]
    drafted = dict(state.get("drafted_sections", {}))
    _stream(ctx, "\n### Compiling References...\n", "ms_references")

    # Assemble the body (without references) so the LLM can scan for placeholders
    body = _assemble_body(drafted)
    refs_md = compile_references(ctx, body)
    drafted["references"] = refs_md

    _emit_event(ctx, {
        "type": "manuscript_section",
        "section": "references",
        "content": refs_md,
        "step": "ms_references",
    })
    return {"drafted_sections": drafted}


def assemble_node(state: ManuscriptState) -> dict:
    """Combine all drafted sections into a single manuscript and save."""
    ctx = state["ctx"]
    drafted = dict(state.get("drafted_sections", {}))
    _stream(ctx, "\n### Assembling final manuscript...\n", "ms_assemble")

    manuscript = _assemble_full(_with_spatial_section(ctx, drafted))

    # Save markdown
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "manuscript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(manuscript)

    ctx.setdefault("artifacts", {})["manuscript_md"] = md_path
    ctx.setdefault("artifacts", {})["manuscript_content"] = manuscript
    # Record input fingerprints so a future "Rerun Manuscript" click can
    # auto-detect which sections actually need rewriting.
    ctx.setdefault("artifacts", {})["manuscript_fingerprints"] = _compute_input_fingerprints(ctx)
    task_manager.log(ctx, f"[MANUSCRIPT] Saved to {md_path}")

    # Convert to DOCX and PDF
    try:
        from agents.result_analysis_agent.report_converter import convert_md_report
        convert_md_report(md_path, manuscript, ctx, task_manager, "manuscript")
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT] DOCX/PDF conversion failed (non-fatal): {e}")

    _stream(ctx, f"\nManuscript assembled and saved ({len(manuscript)} chars).\n")

    # Compute relative path for download button
    rel_path = ""
    try:
        rel_path = os.path.relpath(md_path, settings.output_dir).replace("\\", "/")
    except ValueError:
        rel_path = os.path.basename(md_path)

    _emit_event(ctx, {
        "type": "manuscript_complete",
        "content": manuscript,
        "path": md_path,
        "relative_path": rel_path,
        "step": "ms_assemble",
    })

    return {"manuscript_md": manuscript, "manuscript_path": md_path}


# ── Assembly Helpers ─────────────────────────────────────────────────────────

_SECTION_ORDER = [
    "title_abstract",
    "introduction",
    "methodology",
    "results",
    "spatial_assumptions",   # proposed method only; see _with_spatial_section
    "discussion",
    "conclusion",
    "references",
]


def _with_spatial_section(ctx, drafted: dict) -> dict:
    """Add the proposed method's assumptions section, written from the run's
    records (agents/spatial_reasoning/report.py), not by an LLM. Absent for
    the baselines, so their manuscripts are unchanged."""
    section = ctx.get("artifacts", {}).get("spatial_assumptions_section")
    return {**drafted, "spatial_assumptions": section} if section else drafted


def _assemble_body(drafted: dict) -> str:
    """Assemble sections (without references) into a single string."""
    parts = []
    for key in _SECTION_ORDER:
        if key == "references":
            continue
        content = drafted.get(key, "")
        if content:
            parts.append(content.strip())
    return "\n\n---\n\n".join(parts)


def _assemble_full(drafted: dict) -> str:
    """Assemble ALL sections into the final manuscript."""
    import re
    parts = []
    for key in _SECTION_ORDER:
        content = drafted.get(key, "")
        if content:
            parts.append(content.strip())
    manuscript = "\n\n---\n\n".join(parts)
    # Strip internal [REF_N] tracking markers — the inline (Author, Year) citation remains
    manuscript = re.sub(r'\s*\[REF_\d+\]', '', manuscript)
    # Renumber figures and tables in reading order (first-mention appearance)
    # so Figure 1 is the first one the reader sees, not whichever one happened
    # to sort first at catalog-build time.
    manuscript = _renumber_in_reading_order(manuscript)
    # Remove dangling "Figure N" / "Table N" textual references that point to
    # a number never embedded in the manuscript body.
    manuscript = _strip_dangling_figure_refs(manuscript)
    # Guarantee a blank line above every pipe-table — marked/GFM silently
    # collapses the table into the preceding paragraph otherwise, so the
    # reader sees raw "| term | coef | ..." text instead of a rendered table.
    manuscript = _ensure_blank_line_before_tables(manuscript)
    return manuscript


def _ensure_blank_line_before_tables(md: str) -> str:
    """Insert a blank line before any pipe-table whose header is on the
    line immediately following a non-blank line.

    Detection: a line starting with `|` followed by a line matching the
    GFM separator pattern (`|---|`, `|:---:|`, `|---:|`, etc.). If the
    line ABOVE the header is non-blank, prepend a blank line.
    """
    import re
    lines = md.split("\n")
    sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_header = (
            line.lstrip().startswith("|")
            and i + 1 < len(lines)
            and sep_re.match(lines[i + 1] or "")
        )
        if is_header and out and out[-1].strip() != "":
            out.append("")
        out.append(line)
        i += 1
    return "\n".join(out)


def _renumber_in_reading_order(manuscript: str) -> str:
    """Reassign figure/table numbers in the order they first appear in the
    assembled prose.

    The catalog assigns numbers at build time based on role/objective/step
    ordering, but the writer embeds figures in whatever narrative order the
    argument requires. That can produce "Figure 8" appearing before "Figure 2"
    in the reader's view. This pass rewrites:
      * Figure labels inside `![Figure N: ...](filename)` markup
      * Table captions written as `**Table N.**`
      * Textual references of the form "Figure N" / "Fig. N" / "Table N"
    using a mapping from old number → reading-order number. Filenames and
    captions are preserved; only the numbers change.
    """
    import re

    # Build reading-order mapping for figures.
    fig_pattern = re.compile(
        r'!\[(?:Figure|Fig\.?)\s*(\d+)([^\]]*)\]\(([^)]+)\)',
        flags=re.IGNORECASE,
    )
    fig_order: list = []
    fig_map: dict = {}
    for m in fig_pattern.finditer(manuscript):
        old = m.group(1)
        if old not in fig_map:
            fig_map[old] = str(len(fig_order) + 1)
            fig_order.append(old)

    # Build reading-order mapping for tables — first embed is a caption like
    # `**Table N.**` (period optional).
    tbl_caption_pattern = re.compile(
        r'\*\*Table\s*(\d+)\.?\*\*',
        flags=re.IGNORECASE,
    )
    tbl_order: list = []
    tbl_map: dict = {}
    for m in tbl_caption_pattern.finditer(manuscript):
        old = m.group(1)
        if old not in tbl_map:
            tbl_map[old] = str(len(tbl_order) + 1)
            tbl_order.append(old)

    if not fig_map and not tbl_map:
        return manuscript

    # Rewrite figure embed labels (preserve rest of caption + filename).
    def _rewrite_fig_embed(match):
        old = match.group(1)
        rest = match.group(2)
        fname = match.group(3)
        new = fig_map.get(old, old)
        return f"![Figure {new}{rest}]({fname})"
    manuscript = fig_pattern.sub(_rewrite_fig_embed, manuscript)

    # Rewrite table captions.
    def _rewrite_tbl_caption(match):
        old = match.group(1)
        new = tbl_map.get(old, old)
        return f"**Table {new}.**"
    manuscript = tbl_caption_pattern.sub(_rewrite_tbl_caption, manuscript)

    # Rewrite textual figure references — skip matches already inside an
    # image-syntax label (handled above).
    def _rewrite_fig_text(match):
        label = match.group(1)
        old = match.group(2)
        new = fig_map.get(old)
        if new is None:
            return match.group(0)  # dangling; let the stripper handle it
        return f"{label} {new}"
    manuscript = re.sub(
        r'(?<!!\[)\b(Figure|Fig\.?)\s+(\d+)\b',
        _rewrite_fig_text, manuscript, flags=re.IGNORECASE,
    )

    # Rewrite textual table references — skip matches that are the bold
    # caption form `**Table N.**` (handled above).
    def _rewrite_tbl_text(match):
        label = match.group(1)
        old = match.group(2)
        new = tbl_map.get(old)
        if new is None:
            return match.group(0)
        return f"{label} {new}"
    manuscript = re.sub(
        r'(?<!\*\*)\b(Table)\s+(\d+)\b(?!\.?\*\*)',
        _rewrite_tbl_text, manuscript, flags=re.IGNORECASE,
    )
    return manuscript


def _strip_dangling_figure_refs(manuscript: str) -> str:
    """Neutralize textual 'Figure N'/'Table N' references with no matching embed.

    Embedded figures are detected via `![...Figure N...](...)` markup and via
    table captions written as `**Table N.**`. Any textual reference to a
    Figure/Table number that is NOT in the embedded set is rewritten to the
    neutral phrase 'a figure'/'a table' rather than leaving a dangling number
    pointing at nothing. We intentionally do NOT delete sentences — only the
    specific number reference — to keep surrounding prose intact.
    """
    import re

    embedded_figs = set(re.findall(
        r'!\[(?:Figure|Fig\.?)\s*(\d+)[^\]]*\]\([^)]+\)',
        manuscript, flags=re.IGNORECASE,
    ))
    embedded_tables = set(re.findall(
        r'\*\*Table\s*(\d+)\.?\*\*',
        manuscript, flags=re.IGNORECASE,
    ))

    def _fig_sub(match):
        num = match.group(2)
        if num in embedded_figs:
            return match.group(0)
        return "a figure"

    def _tbl_sub(match):
        num = match.group(2)
        if num in embedded_tables:
            return match.group(0)
        return "a table"

    # Match 'Figure 5', 'Fig. 5', 'Figures 3 and 4' (handle simple cases).
    # Only strip standalone references — skip matches inside image syntax.
    manuscript = re.sub(
        r'(?<!!\[)\b(Figure|Fig\.?)\s+(\d+)\b',
        _fig_sub, manuscript, flags=re.IGNORECASE,
    )
    manuscript = re.sub(
        r'\b(Table)\s+(\d+)\b(?!\.\*\*)',
        _tbl_sub, manuscript, flags=re.IGNORECASE,
    )
    return manuscript


# ── Build & Run Pipeline ────────────────────────────────────────────────────

def build_manuscript_pipeline(ctx, show_graph=True):
    """
    Build and run the full manuscript generation pipeline.

    Flow:
        title_abstract → introduction → methodology → results
        → discussion → conclusion → references → assemble

    Returns dict with manuscript_md, manuscript_path, and drafted_sections.
    """
    workflow = StateGraph(ManuscriptState)

    # Nodes
    workflow.add_node("title_abstract", title_abstract_node)
    workflow.add_node("introduction", introduction_node)
    workflow.add_node("methodology", methodology_node)
    workflow.add_node("results", results_node)
    workflow.add_node("discussion", discussion_node)
    workflow.add_node("conclusion", conclusion_node)
    workflow.add_node("references", references_node)
    workflow.add_node("assemble", assemble_node)

    # Sequential edges — each section needs the previous ones for cross-referencing
    workflow.add_edge(START, "title_abstract")
    workflow.add_edge("title_abstract", "introduction")
    workflow.add_edge("introduction", "methodology")
    workflow.add_edge("methodology", "results")
    workflow.add_edge("results", "discussion")
    workflow.add_edge("discussion", "conclusion")
    workflow.add_edge("conclusion", "references")
    workflow.add_edge("references", "assemble")
    workflow.add_edge("assemble", END)

    graph = workflow.compile()

    if show_graph:
        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
        except Exception:
            pass  # Non-critical — may not be in a notebook

    task_manager.log(ctx, "=" * 60)
    task_manager.log(ctx, "[MANUSCRIPT PIPELINE] Starting full manuscript generation...")

    # ── Pre-fetch shared literature (single batch search up front) ──────
    # Skipped silently if it fails — the writers can still call
    # search_literature when include_literature=True.
    try:
        _batch_literature_search(ctx)
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT] Batch literature search skipped: {e}")
        ctx.setdefault("artifacts", {}).setdefault("pre_searched_literature", [])

    # Classify pre-searched papers by role (foundational / methodological /
    # regional / gap_evidence) and run targeted follow-up searches to fill
    # under-covered roles. Silent on failure — downstream renderer falls back
    # to ungrouped listing.
    try:
        _curate_and_expand_literature(ctx)
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT] Literature curation skipped: {e}")

    # ── Build the manuscript brief once (pre-extracts step outputs, file
    #    stats, figure catalog, and pre-registers all literature as citations).
    #    This eliminates the per-section data-fetch tool calls that were the
    #    main source of latency.
    try:
        _stream(ctx, "\nBuilding manuscript brief (pre-extracting all data)...\n")
        brief = build_manuscript_brief(ctx)
        n_files = len(brief.get("output_files", {}))
        n_cites = len(brief.get("citation_store_snapshot", {}))
        task_manager.log(
            ctx,
            f"[MANUSCRIPT] Brief built: {n_files} file(s) extracted, "
            f"{n_cites} citation(s) pre-registered.",
        )
        _stream(
            ctx,
            f"  Pre-extracted {n_files} output file(s); "
            f"pre-registered {n_cites} citation(s).\n",
        )
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT] Brief build failed (non-fatal): {e}")

    result = graph.invoke({
        "ctx": ctx,
        "drafted_sections": {},
        "manuscript_md": "",
        "manuscript_path": "",
    })

    task_manager.log(ctx, "[MANUSCRIPT PIPELINE] Complete.")

    return {
        "manuscript_md": result.get("manuscript_md", ""),
        "manuscript_path": result.get("manuscript_path", ""),
        "drafted_sections": result.get("drafted_sections", {}),
        "manuscript_docx": ctx.get("artifacts", {}).get("result_manuscript_docx", ""),
        "manuscript_pdf": ctx.get("artifacts", {}).get("result_manuscript_pdf", ""),
    }


# ── Fingerprint-based Staleness Detection ──────────────────────────────────

# Each section depends on one or more "input buckets". When the bucket's
# fingerprint (hash of its content) changes between runs, every section
# that depends on that bucket is considered stale and re-written.
_SECTION_DEPENDENCIES = {
    "title_abstract": ["rq_plan", "execution"],
    "introduction":   ["rq_plan"],
    "methodology":    ["rq_plan"],
    "results":        ["execution"],
    "discussion":     ["execution"],
    "conclusion":     ["execution"],
}


def _compute_input_fingerprints(ctx) -> dict:
    """Hash the inputs each manuscript section depends on."""
    import hashlib

    artifacts = ctx.get("artifacts", {}) or {}

    def _h(obj):
        try:
            payload = json.dumps(obj, sort_keys=True, default=str)
        except Exception:
            payload = repr(obj)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    rq_plan_payload = {
        "rq": ctx.get("rq") or ctx.get("task") or "",
        "plan": artifacts.get("research_plan_final", {}),
    }
    execution_payload = {
        "exec": artifacts.get("step_execution_results", {}),
        "figures": artifacts.get("figure_catalog", []),
    }
    return {
        "rq_plan": _h(rq_plan_payload),
        "execution": _h(execution_payload),
    }


def _detect_stale_sections(ctx) -> list:
    """
    Return the list of sections whose input fingerprints have changed since
    the last successful generation. Missing sections in the existing
    manuscript are also considered stale.
    """
    current = _compute_input_fingerprints(ctx)
    stored = (ctx.get("artifacts", {}) or {}).get("manuscript_fingerprints", {}) or {}

    existing_manuscript = ctx.get("artifacts", {}).get("manuscript_content", "")
    existing_sections = _parse_existing_sections(existing_manuscript) if existing_manuscript else {}

    stale = []
    for section, deps in _SECTION_DEPENDENCIES.items():
        if section not in existing_sections:
            stale.append(section)
            continue
        if any(current.get(d) != stored.get(d) for d in deps):
            stale.append(section)

    return stale


# ── Targeted Update Pipeline ────────────────────────────────────────────────

def build_update_manuscript_pipeline(ctx, sections_to_update=None, show_graph=False):
    """
    Regenerate specific sections of the manuscript.

    Parameters
    ----------
    ctx : dict
        Pipeline context.
    sections_to_update : list[str] | None
        Which sections to regenerate. Pass ``None`` to auto-detect stale
        sections by comparing input fingerprints to the last successful
        generation (only sections whose inputs changed will be re-written).
        Pass an empty list to force "detect-and-run" semantics as well.
        Explicit list forces regeneration of exactly those sections.

    Returns dict with updated manuscript.
    """
    auto_detect = not sections_to_update
    if auto_detect:
        sections_to_update = _detect_stale_sections(ctx)
        task_manager.log(
            ctx,
            f"[MANUSCRIPT UPDATE] Auto-detected stale sections: {sections_to_update}",
        )
        if not sections_to_update:
            task_manager.log(ctx, "[MANUSCRIPT UPDATE] No sections need rerunning.")
            _stream(ctx, "\nNo sections need rerunning — inputs unchanged since last manuscript.\n")
            existing = ctx.get("artifacts", {}).get("manuscript_content", "")
            existing_path = ctx.get("artifacts", {}).get("manuscript_md", "")
            if existing and existing_path:
                rel_path = ""
                try:
                    rel_path = os.path.relpath(existing_path, settings.output_dir).replace("\\", "/")
                except Exception:
                    rel_path = os.path.basename(existing_path)
                _emit_event(ctx, {
                    "type": "manuscript_complete",
                    "content": existing,
                    "path": existing_path,
                    "relative_path": rel_path,
                    "step": "ms_assemble",
                })
            return {
                "manuscript_md": existing,
                "manuscript_path": existing_path,
                "drafted_sections": _parse_existing_sections(existing) if existing else {},
                "sections_updated": [],
            }

    # Load existing drafted sections from the saved manuscript
    existing_manuscript = ctx.get("artifacts", {}).get("manuscript_content", "")
    existing_sections = _parse_existing_sections(existing_manuscript) if existing_manuscript else {}

    task_manager.log(ctx, f"[MANUSCRIPT UPDATE] Updating sections: {sections_to_update}")

    # Refresh batch literature only if introduction/discussion are being rewritten
    if any(s in sections_to_update for s in ("introduction", "discussion")):
        if not ctx.get("artifacts", {}).get("pre_searched_literature"):
            try:
                _batch_literature_search(ctx)
            except Exception as e:
                task_manager.log(ctx, f"[MANUSCRIPT UPDATE] Batch literature search skipped: {e}")
        if not ctx.get("artifacts", {}).get("literature_by_role"):
            try:
                _curate_and_expand_literature(ctx)
            except Exception as e:
                task_manager.log(ctx, f"[MANUSCRIPT UPDATE] Literature curation skipped: {e}")

    # Rebuild the manuscript brief so updated sections see fresh pre-extracted data.
    try:
        build_manuscript_brief(ctx)
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT UPDATE] Brief rebuild failed (non-fatal): {e}")

    # Writer lookup
    writer_map = {
        "title_abstract": write_title_abstract,
        "introduction": write_introduction,
        "methodology": write_methodology,
        "results": write_results,
        "discussion": write_discussion,
        "conclusion": write_conclusion,
    }

    drafted = dict(existing_sections)

    display_names = {
        "title_abstract": "Title & Abstract",
        "introduction": "Introduction",
        "methodology": "Methodology",
        "results": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
    }

    for section in _SECTION_ORDER:
        if section == "references":
            continue
        if section in sections_to_update and section in writer_map:
            step_tag = f"ms_{section}"
            display = display_names.get(section, section)
            # Notify frontend to scroll to / focus the section's existing card
            _emit_event(ctx, {
                "type": "manuscript_section_start",
                "section": section,
                "display_name": display,
                "step": step_tag,
            })
            # Stream into the existing ms_<section> card (not a new ms_update_* one)
            ctx["_stream_step"] = step_tag
            _stream(ctx, f"\n### Updating {display}...\n", step_tag)
            try:
                md = writer_map[section](ctx, drafted_sections=drafted)
                if not md or not md.strip():
                    raise RuntimeError("Writer returned empty content.")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                task_manager.log(ctx, f"[MANUSCRIPT UPDATE] {display} FAILED — {err_msg}")
                _stream(ctx, f"\n⚠️ **{display} update failed:** {err_msg}\n", step_tag)
                md = drafted.get(section) or (
                    f"{_SECTION_HEADINGS.get(section, f'## {display}')}"
                    f"\n_[Section update failed: {err_msg}.]_\n"
                )
                _emit_event(ctx, {
                    "type": "manuscript_section_error",
                    "section": section,
                    "error": err_msg,
                    "step": step_tag,
                })
            drafted[section] = md
            _checkpoint_section(ctx, section, md)
            _emit_event(ctx, {
                "type": "manuscript_section",
                "section": section,
                "content": md,
                "step": step_tag,
            })

    # Always recompile references if any section changed
    body = _assemble_body(drafted)
    drafted["references"] = compile_references(ctx, body)

    # Stream a header tagged ms_assemble so the frontend creates the
    # "Full Manuscript" card before the manuscript_complete event arrives,
    # and so the preceding ms_conclusion card is finalized (spinner stops).
    _stream(ctx, "\n### Assembling final manuscript...\n", "ms_assemble")

    # Assemble and save
    manuscript = _assemble_full(_with_spatial_section(ctx, drafted))
    md_path = os.path.join(resolve_output_dir(ctx), "manuscript.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(manuscript)

    ctx.setdefault("artifacts", {})["manuscript_md"] = md_path
    ctx.setdefault("artifacts", {})["manuscript_content"] = manuscript
    ctx.setdefault("artifacts", {})["manuscript_fingerprints"] = _compute_input_fingerprints(ctx)

    # Emit manuscript_complete so the frontend replaces the Full Manuscript card.
    try:
        rel_path = os.path.relpath(md_path, settings.output_dir).replace("\\", "/")
    except Exception:
        rel_path = os.path.basename(md_path)
    _emit_event(ctx, {
        "type": "manuscript_complete",
        "content": manuscript,
        "path": md_path,
        "relative_path": rel_path,
        "step": "ms_assemble",
    })

    try:
        from agents.result_analysis_agent.report_converter import convert_md_report
        convert_md_report(md_path, manuscript, ctx, task_manager, "manuscript")
    except Exception as e:
        task_manager.log(ctx, f"[MANUSCRIPT UPDATE] DOCX/PDF conversion failed (non-fatal): {e}")

    task_manager.log(ctx, "[MANUSCRIPT UPDATE] Complete.")

    return {
        "manuscript_md": manuscript,
        "manuscript_path": md_path,
        "drafted_sections": drafted,
        "sections_updated": sections_to_update,
    }


def _parse_existing_sections(manuscript: str) -> dict:
    """
    Best-effort parse of an existing manuscript into sections.
    Uses the ``---`` separator and heading patterns to split.
    """
    import re

    sections = {}
    # Split on the horizontal rule separator used by _assemble_full
    parts = re.split(r'\n---\n', manuscript)

    # Map parts to section names by detecting headings
    section_patterns = [
        ("title_abstract", r'^#\s+'),            # Title is H1
        ("introduction", r'##\s+1\.\s+Introduction'),
        ("methodology", r'##\s+2\.\s+Methodology'),
        ("results", r'##\s+3\.\s+Results'),
        ("spatial_assumptions", r'##\s+Spatial Assumptions and Diagnostics'),
        ("discussion", r'##\s+4\.\s+Discussion'),
        ("conclusion", r'##\s+5\.\s+Conclusion'),
        ("references", r'##\s+References'),
    ]

    for part in parts:
        part_stripped = part.strip()
        for name, pattern in section_patterns:
            if re.search(pattern, part_stripped, re.MULTILINE):
                sections[name] = part_stripped
                break

    return sections