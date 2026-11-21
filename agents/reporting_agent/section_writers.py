"""
Section writers for the manuscript pipeline.

Two execution modes:

* ``_run_pure_writer``  — single LLM call, NO tools bound.  Used for sections
  whose entire input is already in the manuscript brief (title_abstract,
  methodology, conclusion).  This is the fast path.

* ``_run_agentic_writer`` — small tool-use loop (max 3 iterations) used by
  sections that may genuinely need to find additional literature
  (introduction, discussion) or compute an ad-hoc stat (results).

All section writers pull data from ``ctx["artifacts"]["manuscript_brief"]``,
which is built once at the start of the pipeline by
``manuscript_tools.build_manuscript_brief``.
"""

import os

from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from utils.llm_models import llm_model, llm_reasoning
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import utils.agm_helper as helper
from utils.config import load_configurations
from agents.task_manager import TaskManager

from agents.reporting_agent.manuscript_tools import (
    build_manuscript_tools,
    build_manuscript_brief,
    render_research_context,
    render_collated_results,
    render_results_scope_only,
    render_environment_manifest,
    render_data_summary,
    render_data_registry,
    render_research_plan,
    render_figure_catalog,
    render_output_file_stats,
    render_pre_registered_citations,
    render_drafted_sections,
)
from agents.reporting_agent import section_prompts as prompts

task_manager = TaskManager()
settings = load_configurations()


# ── Brief access ────────────────────────────────────────────────────────────

def _get_brief(ctx):
    """Fetch the manuscript brief, building it lazily if missing."""
    brief = ctx.get("artifacts", {}).get("manuscript_brief")
    if not brief:
        brief = build_manuscript_brief(ctx)
    return brief


# ── LLM execution paths ─────────────────────────────────────────────────────

def _run_pure_writer(ctx, *, system_prompt: str, user_message: str, section_tag: str) -> str:
    """
    Single-shot LLM call with NO tools bound.

    Used when the brief contains everything the section needs — eliminates
    every tool round-trip.  Streams tokens directly.
    """
    if helper.check_cancelled(ctx):
        return "[Cancelled by user]"

    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        model_name=llm_model("manuscript.section_single_call"),
         **llm_reasoning("manuscript.section_single_call"),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    stream_cb = ctx.get("_stream_callback")
    task_manager.log(ctx, f"[MANUSCRIPT] Writing {section_tag} (single-shot, no tools)...")
    if stream_cb:
        stream_cb(f"  [{section_tag}] generating (no tool calls)...\n")

    if stream_cb:
        full = ""
        for chunk in llm.stream(messages):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                full += token
                stream_cb(token)
        content = full
    else:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

    task_manager.log(ctx, f"[MANUSCRIPT] {section_tag} done ({len(content)} chars)")
    return content


def _run_agentic_writer(
    ctx,
    *,
    system_prompt: str,
    user_message: str,
    section_tag: str,
    mode: str = "minimal",
    max_iterations: int = 3,
    pre_searched: list | None = None,
) -> str:
    """
    Small tool-use loop (default 3 iterations) for sections that may need
    an escape hatch — typically `run_analysis_snippet` for ad-hoc stats or
    `search_literature` for additional papers beyond the pre-registered list.

    The brief covers the bulk of the data, so most calls finish in one
    iteration with no tool calls at all.
    """
    # Rerun-manuscript path: caller can force the writer to skip tool binding
    # entirely and do a single pure LLM call. Prevents the "Writer returned
    # empty content" failure mode when tools are called on the final iteration
    # and the loop exits before the model produces prose.
    if ctx.get("_force_no_tools"):
        return _run_pure_writer(
            ctx,
            system_prompt=system_prompt,
            user_message=user_message,
            section_tag=section_tag,
        )

    tools = build_manuscript_tools(ctx, mode=mode, pre_searched=pre_searched)
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        # model_name=get_ui_model(),
        # reasoning_effort=get_ui_reasoning_effort(),
        model_name=llm_model("manuscript.section_agentic"), **llm_reasoning("manuscript.section_agentic")
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    stream_cb = ctx.get("_stream_callback")
    task_manager.log(
        ctx,
        f"[MANUSCRIPT] Writing {section_tag} (mode={mode}, max_iter={max_iterations})...",
    )

    response = None
    for i in range(max_iterations):
        if helper.check_cancelled(ctx):
            return "[Cancelled by user]"
        if stream_cb:
            stream_cb(f"  [{section_tag}] iteration {i + 1}/{max_iterations}...\n")

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
            break

        for tc in response.tool_calls:
            task_manager.log(ctx, f"  [{section_tag} TOOL] {tc['name']}({_short(tc['args'])})")
            if stream_cb:
                stream_cb(f"\n[TOOL] **{tc['name']}**\n")
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            if stream_cb:
                stream_cb(f"\n```\n{str(result)}\n```\n")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # If we exhausted the iteration budget while the model still had pending
    # tool_calls (or came back with empty content for any other reason), do one
    # final tool-free invocation to force a written section. Without this, the
    # caller sees empty content and emits a "Writer returned empty content"
    # placeholder — which is the actual failure mode we saw in Results.
    content = response.content if (response and hasattr(response, "content")) else ""
    if (not content or not content.strip()) and response is not None:
        if stream_cb:
            stream_cb(f"\n  [{section_tag}] finalizing (tool budget exhausted)...\n")
        task_manager.log(
            ctx,
            f"[MANUSCRIPT] {section_tag} content empty after {max_iterations} iter — forcing final write.",
        )
        final_messages = messages + [
            HumanMessage(content=(
                "You have gathered enough context. Write the complete "
                f"{section_tag} section now as a single markdown response. "
                "Do NOT call any tools."
            ))
        ]
        if stream_cb:
            full = ""
            for chunk in llm.stream(final_messages):
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    full += token
                    stream_cb(token)
            content = full
        else:
            final_resp = llm.invoke(final_messages)
            content = final_resp.content if hasattr(final_resp, "content") else str(final_resp)

    task_manager.log(ctx, f"[MANUSCRIPT] {section_tag} done ({len(content)} chars)")
    return content


def _short(args):
    s = str(args)
    return s[:80] + "..." if len(s) > 80 else s


# ── Shared preamble ─────────────────────────────────────────────────────────

def _preamble(*, has_literature_tools: bool, has_escape_hatch: bool) -> str:
    """Tell the LLM what's pre-loaded and what (if anything) it can call."""
    parts = [
        "**ALL CONTEXT IS PRE-LOADED BELOW.** Read the data sections that follow "
        "and write the section directly. Do NOT request data — there are no "
        "context-fetch tools, only escape hatches.\n",
    ]
    if has_escape_hatch:
        parts.append(
            "Available escape hatches (use ONLY if a number you need is missing "
            "from the pre-extracted file stats below):\n"
            "- `run_analysis_snippet(code)` — run a small Python snippet to "
            "compute an ad-hoc statistic.\n"
        )
    if has_literature_tools:
        parts.append(
            "Pre-registered citations are listed at the bottom of this prompt. "
            "Cite them directly using the format shown — **do NOT call "
            "`add_citation` for those papers**. Only call `search_literature` + "
            "`add_citation` if you need a NEW paper not already in the list.\n"
        )
    return "".join(parts)


# ── Section writers ─────────────────────────────────────────────────────────

def write_title_abstract(ctx, drafted_sections=None):
    """Generate the Title and Abstract — single-shot, no tools."""
    brief = _get_brief(ctx)

    user_message = (
        _preamble(has_literature_tools=False, has_escape_hatch=False) +
        "Write the research title and structured abstract for this manuscript "
        "using the data below.\n\n"
        f"### Research Context\n```json\n{render_research_context(brief)}\n```\n\n"
        f"### Collated Execution Results (full step outputs)\n"
        f"```json\n{render_collated_results(brief)}\n```\n\n"
        f"{render_output_file_stats(brief)}\n"
    )

    return _run_pure_writer(
        ctx,
        system_prompt=prompts.TITLE_ABSTRACT_PROMPT,
        user_message=user_message,
        section_tag="Title & Abstract",
    )


def write_introduction(ctx, drafted_sections=None):
    """Generate the Introduction — agentic (literature mode), max 3 iterations."""
    brief = _get_brief(ctx)
    pre_searched = brief.get("literature", []) or []
    drafted_block = render_drafted_sections(drafted_sections, ["title_abstract"])
    citations_block = render_pre_registered_citations(brief)

    # Signal whether the research plan actually carries hypotheses, so the
    # writer knows to include/omit the Hypotheses paragraph instead of
    # inventing one.
    objectives = brief.get("objectives", []) or []
    has_hypotheses = any(
        (o.get("hypothesis") or "").strip()
        for o in objectives if isinstance(o, dict)
    )
    hypotheses_flag = (
        "Hypotheses: PROVIDED — include the Hypotheses paragraph."
        if has_hypotheses else
        "Hypotheses: ABSENT — OMIT the Hypotheses paragraph. Do not invent hypotheses."
    )

    user_message = (
        _preamble(
            has_literature_tools=True,
            has_escape_hatch=False,
        ) +
        "Write the Introduction section using the data below. Foreshadow "
        "findings — do NOT present them in detail.\n\n"
        f"{hypotheses_flag}\n\n"
        f"### Research Context\n```json\n{render_research_context(brief)}\n```\n\n"
        f"### Data Summary\n```json\n{render_data_summary(brief)}\n```\n\n"
        f"### Study Scope (objectives + planned approach — NO findings)\n"
        f"This is what the study DOES, not what it FOUND. Foreshadow this "
        f"approach; save all numeric results for the Results section.\n"
        f"```json\n{render_results_scope_only(brief)}\n```\n\n"
        f"{drafted_block}\n\n"
        f"{citations_block}\n"
    )

    return _run_agentic_writer(
        ctx,
        system_prompt=prompts.INTRODUCTION_PROMPT,
        user_message=user_message,
        section_tag="Introduction",
        mode="literature",
        max_iterations=3,
        pre_searched=pre_searched,
    )


def write_methodology(ctx, drafted_sections=None):
    """Generate the Methodology — agentic so it can cite method sources.

    Passes:
      - Research Plan (planned objectives/steps) — study framing
      - Data Registry + Data Summary — data specs
      - Collated Results (full step outputs) — what ACTUALLY ran, so
        reproducibility details (parameters, weights, CRS, drops, joins) come
        from the execution trace rather than the plan.
      - Output File Stats — concrete file-level facts (shapes, CRS, columns)
      - Pre-registered citations — so the LLM can cite method sources
        (e.g., GWR → Brunsdon et al. 1996, Moran's I → Moran 1950).
      - Computational environment — real Python / package versions + date.
    """
    brief = _get_brief(ctx)
    pre_searched = brief.get("literature", []) or []
    drafted_block = render_drafted_sections(drafted_sections, ["introduction"])
    citations_block = render_pre_registered_citations(brief)
    env_block = render_environment_manifest(brief, ctx=ctx)

    user_message = (
        _preamble(has_literature_tools=True, has_escape_hatch=False) +
        "Write the Methodology section (Data + Methods) using the data below. "
        "Describe what WAS DONE (from the execution trace), not only what was "
        "planned. Report actual parameter values, spatial-weights specifications, "
        "CRS codes, and join/filter rules directly from the step outputs. Cite "
        "the method sources (pre-registered list below). Do NOT report numeric "
        "findings — those belong in Results.\n\n"
        f"### Research Plan (study framing)\n```json\n{render_research_plan(brief)}\n```\n\n"
        f"### Data Registry\n```json\n{render_data_registry(brief)}\n```\n\n"
        f"### Data Summary\n```json\n{render_data_summary(brief)}\n```\n\n"
        f"### Execution Trace (full step outputs — source of truth for WHAT RAN)\n"
        f"Use this to describe actual parameters, libraries, weights matrices, "
        f"CRS, preprocessing rules, and model-assessment steps. Extract only "
        f"operational facts — numeric findings stay out of Methodology.\n"
        f"```json\n{render_collated_results(brief)}\n```\n\n"
        f"{render_output_file_stats(brief)}\n\n"
        f"{env_block}\n\n"
        f"{citations_block}\n\n"
        f"{drafted_block}\n"
    )

    return _run_agentic_writer(
        ctx,
        system_prompt=prompts.METHODOLOGY_PROMPT,
        user_message=user_message,
        section_tag="Methodology",
        mode="literature",
        max_iterations=2,
        pre_searched=pre_searched,
    )


def write_results(ctx, drafted_sections=None):
    """Generate the Results — agentic (escape hatch only), max 3 iterations."""
    brief = _get_brief(ctx)
    drafted_block = render_drafted_sections(drafted_sections, ["methodology"])

    # Same conditional-hypotheses signal the Introduction uses — so Hypothesis
    # Assessment paragraphs only appear when hypotheses exist in the plan.
    objectives = brief.get("objectives", []) or []
    has_hypotheses = any(
        (o.get("hypothesis") or "").strip()
        for o in objectives if isinstance(o, dict)
    )
    hypotheses_flag = (
        "Hypotheses: PROVIDED — include a Hypothesis Assessment at the end of each objective's subsection."
        if has_hypotheses else
        "Hypotheses: ABSENT — OMIT Hypothesis Assessment paragraphs entirely. Do not invent hypotheses."
    )

    user_message = (
        _preamble(has_literature_tools=False, has_escape_hatch=True) +
        "Write the Results section using the data below. Weave figures and "
        "tables into the prose — introduce each before it appears and point "
        "the reader to what to see in it. Use the EXACT filenames from the "
        "figure catalog; do not rename. Describe findings, do not interpret "
        "(interpretation belongs in the Discussion).\n\n"
        f"{hypotheses_flag}\n\n"
        f"### Research Context\n```json\n{render_research_context(brief)}\n```\n\n"
        f"### Figure Catalog (includes kind='table' previews to embed inline)\n"
        f"{render_figure_catalog(brief)}\n\n"
        f"### Collated Results (full step outputs — extract numbers from here)\n"
        f"```json\n{render_collated_results(brief)}\n```\n\n"
        f"{render_output_file_stats(brief)}\n\n"
        f"{drafted_block}\n"
    )

    return _run_agentic_writer(
        ctx,
        system_prompt=prompts.RESULTS_PROMPT,
        user_message=user_message,
        section_tag="Results",
        mode="minimal",
        max_iterations=3,
    )


def write_discussion(ctx, drafted_sections=None):
    """Generate the Discussion — agentic (literature mode), max 3 iterations.

    Works from the drafted Introduction + Results prose (NOT the raw collated
    execution trace) so the Discussion interprets what was actually reported
    rather than re-mining numbers. Carries the same PROVIDED/ABSENT hypothesis
    signal as Introduction and Results to gate conditional hypothesis language.
    """
    brief = _get_brief(ctx)
    pre_searched = brief.get("literature", []) or []
    drafted_block = render_drafted_sections(
        drafted_sections, ["introduction", "results"]
    )
    citations_block = render_pre_registered_citations(brief)

    objectives = brief.get("objectives", []) or []
    has_hypotheses = any(
        (o.get("hypothesis") or "").strip()
        for o in objectives if isinstance(o, dict)
    )
    hypotheses_flag = (
        "Hypotheses: PROVIDED — revisit each hypothesis with conditional "
        "supported/not-supported language tied to the reported evidence."
        if has_hypotheses else
        "Hypotheses: ABSENT — OMIT hypothesis-revisit language. Do not invent "
        "hypotheses; frame findings as observations relative to the objectives."
    )

    user_message = (
        _preamble(has_literature_tools=True, has_escape_hatch=False) +
        "Write the Discussion section using the drafted Introduction and "
        "Results below. INTERPRET — do not re-mine the execution trace and do "
        "not repeat numbers already reported in Results. Engage the literature, "
        "name mechanisms, acknowledge limitations as narrative prose, and close "
        "the loop with the research gap stated in the Introduction.\n\n"
        f"{hypotheses_flag}\n\n"
        f"### Research Context\n```json\n{render_research_context(brief)}\n```\n\n"
        f"### Figure Catalog (REFERENCE ONLY — do NOT re-embed)\n"
        f"Refer to figures by number (e.g. 'Figure 2') only if they were "
        f"actually embedded in the drafted Results below. Do NOT insert any "
        f"new `![...](...)` image syntax in the Discussion, and do NOT "
        f"reference a figure number that does not appear in the Results prose.\n"
        f"{render_figure_catalog(brief)}\n\n"
        f"{drafted_block}\n\n"
        f"{citations_block}\n"
    )

    return _run_agentic_writer(
        ctx,
        system_prompt=prompts.DISCUSSION_PROMPT,
        user_message=user_message,
        section_tag="Discussion",
        mode="literature",
        max_iterations=3,
        pre_searched=pre_searched,
    )


def write_conclusion(ctx, drafted_sections=None):
    """Generate the Conclusion — single-shot, no tools.

    Carries the same PROVIDED/ABSENT hypothesis signal as other sections so the
    Conclusion does not invent hypothesis-verdict language when the plan has
    no hypotheses.
    """
    brief = _get_brief(ctx)
    drafted_block = render_drafted_sections(
        drafted_sections, ["title_abstract", "results", "discussion"]
    )

    objectives = brief.get("objectives", []) or []
    has_hypotheses = any(
        (o.get("hypothesis") or "").strip()
        for o in objectives if isinstance(o, dict)
    )
    hypotheses_flag = (
        "Hypotheses: PROVIDED — state briefly which hypotheses were supported "
        "or not (one sentence each, at most)."
        if has_hypotheses else
        "Hypotheses: ABSENT — OMIT hypothesis-verdict language. Frame the "
        "contribution in terms of the research question and objectives."
    )

    user_message = (
        _preamble(has_literature_tools=False, has_escape_hatch=False) +
        "Write the Conclusion section using the previously drafted sections "
        "below. Synthesize at a higher level than the Results; state the "
        "CONTRIBUTION (what the field now knows that it did not know before) "
        "and the SCOPE (what this study does and does not claim). Do NOT "
        "introduce new information or new citations. Do NOT recap the "
        "methodology or re-list Discussion's implications / future research.\n\n"
        f"{hypotheses_flag}\n\n"
        f"{drafted_block}\n"
    )

    return _run_pure_writer(
        ctx,
        system_prompt=prompts.CONCLUSION_PROMPT,
        user_message=user_message,
        section_tag="Conclusion",
    )


# ── References (unchanged — uses citation_store directly) ──────────────────

def compile_references(ctx, full_manuscript: str):
    """
    Build the References section from the citation store populated by
    pre-registration + any additional add_citation calls during writing.

    Falls back to an LLM pass if the citation store is empty.
    """
    import re
    from utils.citation_tools import format_apa_reference

    stream_cb = ctx.get("_stream_callback")
    task_manager.log(ctx, "[MANUSCRIPT] Compiling references...")

    citation_store = ctx.get("artifacts", {}).get("citation_store", {})

    if not citation_store:
        task_manager.log(ctx, "[MANUSCRIPT] No citation store found — using LLM to compile references.")
        return _compile_references_llm(ctx, full_manuscript)

    # Find which [REF_N] markers actually appear in the manuscript
    used_refs = set(re.findall(r'\[REF_(\d+)\]', full_manuscript))

    lines = ["## References\n"]
    for ref_num in sorted(citation_store.keys(), key=int):
        if ref_num not in used_refs:
            continue
        paper = citation_store[ref_num]
        lines.append(format_apa_reference(paper, ref_num))

    refs_md = "\n\n".join(lines)

    if stream_cb:
        stream_cb("  Compiling references from citation store...\n")
        stream_cb(refs_md)

    task_manager.log(ctx, f"[MANUSCRIPT] {len(used_refs)} references compiled from citation store.")
    return refs_md


def _compile_references_llm(ctx, full_manuscript: str):
    """Fallback: use LLM to compile references when no citation store exists."""
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        model_name=llm_model("manuscript.references"),
         **llm_reasoning("manuscript.references"),
    )
    stream_cb = ctx.get("_stream_callback")

    citation_store = ctx.get("artifacts", {}).get("citation_store", {})
    citation_list = ""
    if citation_store:
        from utils.citation_tools import format_apa_reference
        items = []
        for ref_num in sorted(citation_store.keys(), key=int):
            items.append(format_apa_reference(citation_store[ref_num], ref_num))
        citation_list = "\n\nPre-built citation list:\n" + "\n".join(items)

    messages = [
        SystemMessage(content=prompts.REFERENCES_PROMPT),
        HumanMessage(content=(
            "Here is the full manuscript. Scan it for all [REF_N] markers "
            "and compile the corresponding references list.\n\n"
            f"---BEGIN MANUSCRIPT---\n{full_manuscript[:30000]}\n---END MANUSCRIPT---"
            f"{citation_list}"
        )),
    ]

    if stream_cb:
        stream_cb("  Compiling references...\n")
        full_content = ""
        for chunk in llm.stream(messages):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                full_content += token
                stream_cb(token)
        return full_content

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
