"""
Manuscript brief builder and minimal tool kit for the Reporting Agent.

The "manuscript brief" is built ONCE at the start of the pipeline and contains
everything section writers need (research context, full step outputs, output
file stats, figure catalog, pre-registered citations).  By front-loading this
data, we eliminate dozens of per-section LLM tool calls that previously had
to fetch data piecemeal.

The tool builder (`build_manuscript_tools`) now binds only a tiny escape
hatch — the LLM should rely on the brief for almost everything.
"""

import os
import io
import json
import contextlib

from langchain_core.tools import tool

from utils.table_summary import load_table, size_fields


# ── Constants ───────────────────────────────────────────────────────────────

# Per-step output cap inside the brief.  Was 500 in collated_results, which
# forced the LLM to call read_output_file just to see the rest.
_STEP_OUTPUT_CAP = 3000

# Per-file content cap inside the brief.
_FILE_HEAD_CAP = 1500
_FILE_DESCRIBE_CAP = 2000
_FILE_PREVIEW_CAP = 2000

# Row handling for tabular output files lives in utils/table_summary.py:
# true row counts always, statistics on the whole file up to a large cap.


# ── Brief builder ───────────────────────────────────────────────────────────

def build_manuscript_brief(ctx):
    """
    Pre-extract everything section writers need into a single dict, stored on
    ``ctx["artifacts"]["manuscript_brief"]``.

    Front-loading this data eliminates per-section tool calls (the previous
    biggest source of latency).  Specifically:

    * **Full step outputs** — collated_results truncates each step's stdout
      to 500 chars, forcing the LLM to call ``get_step_details`` /
      ``read_output_file``.  The brief stores up to 3000 chars per step.
    * **Output file stats** — every CSV / GPKG / JSON / TXT created by the
      pipeline is opened once, and head / describe / shape / columns are
      cached.  Replaces ``read_output_file``.
    * **Figure catalog** — already rendered as text, ready to drop into a
      prompt.  Replaces ``get_figure_catalog``.
    * **Pre-registered citations** — every paper from the batch literature
      search is registered in ``citation_store`` with a permanent REF_N up
      front.  The LLM can cite directly, no ``add_citation`` call needed.

    Returns the brief dict (also cached on ctx).
    """
    artifacts = ctx.setdefault("artifacts", {})

    # ── Clear any stale manuscript_brief so we always rebuild from
    #    the current conversation's data.
    artifacts.pop("manuscript_brief", None)

    task_id = ctx.get("task_id", "unknown")
    output_dir = ctx.get("output_dir", "unknown")
    rq_preview = (ctx.get("rq", "") or "")[:80]
    step_keys = list((artifacts.get("step_execution_results") or {}).keys())
    print(f"[MANUSCRIPT BRIEF] Building brief for task_id={task_id}, "
          f"output_dir={output_dir}, step_result_keys={step_keys}, "
          f"rq={rq_preview!r}")

    # ── Research context ────────────────────────────────────────────────
    plan = artifacts.get("research_plan_final", {}) or {}
    rq = plan.get("research_question", ctx.get("rq", "")) or ""
    objectives = []
    for obj in plan.get("objectives", []) or []:
        objectives.append({
            "number": obj.get("number"),
            "title": obj.get("title"),
            "hypothesis": obj.get("hypothesis"),
            "steps": obj.get("steps", []),
        })

    # ── Data summary ────────────────────────────────────────────────────
    data_summary = artifacts.get("data_summary_data", {}) or {}
    if not data_summary:
        text = artifacts.get("data_summary", "")
        if isinstance(text, str) and os.path.exists(text):
            try:
                with open(text, "r", encoding="utf-8") as f:
                    data_summary = f.read()[:8000]
            except Exception:
                data_summary = ""

    data_registry = artifacts.get("data_registry", []) or []

    # ── Full collated results (untruncated step outputs) ────────────────
    step_results = artifacts.get("step_execution_results", {}) or {}
    obj_meta = {}
    for obj in plan.get("objectives", []) or []:
        key = f"objective_{obj.get('number', 1)}"
        obj_meta[key] = {
            "title": obj.get("title", ""),
            "hypothesis": obj.get("hypothesis", ""),
            "planned_steps": obj.get("steps", []),
        }

    full_collated = {}
    output_files = set()
    _expected_dir = os.path.normcase(os.path.normpath(
        ctx.get("output_dir", "") or ""
    ))
    for obj_key, steps in step_results.items():
        meta = obj_meta.get(obj_key, {})
        full_steps = []
        for s in steps:
            inner = s.get("result", {}) if isinstance(s.get("result"), dict) else {}
            output = (inner.get("output", "") or "")[:_STEP_OUTPUT_CAP]
            created = inner.get("created_files", []) or []
            for f in created:
                if isinstance(f, str) and os.path.exists(f):
                    # Warn if file is outside the current task's output directory
                    if _expected_dir and not os.path.normcase(
                        os.path.normpath(f)
                    ).startswith(_expected_dir):
                        print(f"[MANUSCRIPT BRIEF] WARNING: file {f} is outside "
                              f"task output_dir {_expected_dir} — may be from "
                              f"a different conversation!")
                    output_files.add(f)
            full_steps.append({
                "step_index": s.get("step_index"),
                "description": s.get("step_description", ""),
                "status": s.get("status", "unknown"),
                "agent": s.get("agent_name", ""),
                "output": output,
                "created_files": [os.path.basename(f) for f in created if isinstance(f, str)],
                "error": s.get("error_message"),
            })
        full_collated[obj_key] = {
            "title": meta.get("title", obj_key),
            "hypothesis": meta.get("hypothesis", ""),
            "planned_steps": meta.get("planned_steps", []),
            "executed_steps": full_steps,
        }

    # ── Pre-extracted file stats (replaces read_output_file) ────────────
    file_stats = _extract_file_stats(output_files)

    # ── Figure catalog rendered as text ─────────────────────────────────
    figure_catalog_text = ""
    catalog_v2 = artifacts.get("figure_catalog_v2")
    catalog = artifacts.get("figure_catalog", []) or []
    if catalog_v2:
        try:
            from agents.result_analysis_agent.figure_catalog_v2 import format_catalog_for_prompt
            figure_catalog_text = format_catalog_for_prompt(catalog_v2)
            # Inline preview of kind='table' entries as markdown tables so the
            # LLM can quote concrete numbers instead of describing "the table".
            figure_catalog_text += "\n\n" + _render_table_previews(catalog_v2)
        except Exception:
            figure_catalog_text = ""
    elif catalog:
        try:
            from agents.result_analysis_agent.figure_preparation import format_figure_catalog_for_prompt
            figure_catalog_text = format_figure_catalog_for_prompt(catalog)
        except Exception:
            figure_catalog_text = ""

    # ── Pre-register pre-searched literature as real citations ──────────
    pre_searched = artifacts.get("pre_searched_literature", []) or []
    citation_store = artifacts.setdefault("citation_store", {})
    for paper in pre_searched:
        pid = paper.get("paper_id")
        if not pid:
            continue
        already = any(
            stored.get("paper_id") == pid for stored in citation_store.values()
        )
        if already:
            continue
        ref_num = str(len(citation_store) + 1)
        citation_store[ref_num] = paper

    brief = {
        "research_question": rq,
        "objectives": objectives,
        "data_context": plan.get("data_context", {}),
        "key_considerations": plan.get("key_considerations", []),
        # Proposed method only: stated assumptions and what the diagnostics found.
        "spatial_assumptions": artifacts.get("spatial_assumptions_summary"),
        "data_summary": data_summary,
        "data_registry": data_registry,
        "research_plan": plan,
        "collated_results": full_collated,
        "output_files": file_stats,
        "figure_catalog_text": figure_catalog_text,
        "literature": pre_searched,
        "literature_by_role": artifacts.get("literature_by_role", {}),
        # Snapshot of citation_store keyed by ref_num — what the LLM should cite
        "citation_store_snapshot": {k: dict(v) for k, v in citation_store.items()},
    }

    artifacts["manuscript_brief"] = brief
    return brief


def _render_table_previews(catalog_v2):
    """Render kind='table' entries as full markdown tables for the brief."""
    import pandas as pd

    chunks = []
    for fig in catalog_v2 or []:
        if fig.get("kind") != "table":
            continue
        src = fig.get("path")
        if not src or not os.path.exists(src):
            continue
        ext = os.path.splitext(src)[1].lower()
        try:
            if ext in (".csv", ".tsv"):
                df = pd.read_csv(src, sep="," if ext == ".csv" else "\t")
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(src)
            elif ext == ".parquet":
                df = pd.read_parquet(src)
            else:
                continue
        except Exception:
            continue
        cols = fig.get("columns")
        if cols:
            keep = [c for c in cols if c in df.columns]
            if keep:
                df = df[keep]
        if len(df) > 50:
            df = df.head(50)
        try:
            md = df.to_markdown(index=False)
        except Exception:
            md = df.to_string(index=False)
        chunks.append(
            f"### Table {fig.get('figure_number')}: {fig.get('caption','')}\n"
            f"Source: `{os.path.basename(src)}`\n\n{md}"
        )
    if not chunks:
        return ""
    return "## Table Contents (render these inline in the manuscript)\n\n" + "\n\n".join(chunks)


from utils.table_rows import table_rows_text


def _extract_file_stats(file_paths):
    """Open every output file once and cache head/describe/shape/columns."""
    stats = {}
    for fp in sorted(file_paths):
        if not os.path.exists(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        fname = os.path.basename(fp)
        try:
            if ext == ".csv":
                df, total, sampled = load_table(fp)
                stats[fname] = {
                    "type": "csv",
                    "path": fp,
                    **size_fields(df, total, sampled),
                    "columns": list(df.columns)[:50],
                    # Every row of a results table: a preview cut models off.
                    "head": table_rows_text(df, total_rows=total),
                    "describe": df.describe(include="all").to_string()[:_FILE_DESCRIBE_CAP],
                }
            elif ext in (".gpkg", ".shp", ".geojson"):
                gdf, total, sampled = load_table(fp)
                non_geom = gdf.drop(columns="geometry", errors="ignore")
                stats[fname] = {
                    "type": "spatial",
                    "path": fp,
                    **size_fields(gdf, total, sampled),
                    "columns": list(gdf.columns)[:50],
                    "crs": str(gdf.crs),
                    "describe": non_geom.describe(include="all").to_string()[:_FILE_DESCRIBE_CAP],
                }
            elif ext == ".json":
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                stats[fname] = {
                    "type": "json",
                    "path": fp,
                    "preview": json.dumps(data, default=str, indent=2)[:_FILE_PREVIEW_CAP],
                }
            elif ext in (".txt", ".log", ".md"):
                with open(fp, "r", encoding="utf-8") as f:
                    stats[fname] = {
                        "type": "text",
                        "path": fp,
                        "preview": f.read()[:_FILE_PREVIEW_CAP],
                    }
            elif ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
                stats[fname] = {
                    "type": "image",
                    "path": fp,
                    "size": os.path.getsize(fp),
                }
            elif ext == ".html":
                stats[fname] = {
                    "type": "html",
                    "path": fp,
                    "size": os.path.getsize(fp),
                }
        except Exception as e:
            stats[fname] = {"type": "error", "path": fp, "error": str(e)}
    return stats


# ── Brief renderers (turn brief slices into prompt-ready text) ─────────────

def render_research_context(brief, *, max_chars=8000):
    """Render the research context block (RQ, objectives, hypotheses)."""
    payload = {
        "research_question": brief.get("research_question", ""),
        "objectives": brief.get("objectives", []),
        "data_context": brief.get("data_context", {}),
        "key_considerations": brief.get("key_considerations", []),
    }
    if brief.get("spatial_assumptions"):
        payload["spatial_assumptions"] = brief["spatial_assumptions"]
    return json.dumps(payload, indent=2, default=str)[:max_chars]


def render_collated_results(brief, *, max_chars=20000):
    """Render the per-objective execution summary with FULL step outputs."""
    return json.dumps(brief.get("collated_results", {}), indent=2, default=str)[:max_chars]


def render_results_scope_only(brief, *, max_chars=4000):
    """
    Intro-safe projection of collated results: keeps *what was done* (objective
    titles, hypotheses, planned-step descriptions) and drops *what was found*
    (step outputs, numeric summaries, error details).

    Use this in the Introduction so the writer can foreshadow the approach
    without leaking numeric findings that belong in Results.
    """
    collated = brief.get("collated_results", {}) or {}
    scope = {}
    for obj_key, obj in collated.items():
        if not isinstance(obj, dict):
            continue
        planned = []
        for step in (obj.get("planned_steps") or []):
            if isinstance(step, dict):
                desc = step.get("description") or step.get("step") or step.get("title")
            else:
                desc = str(step)
            if desc:
                planned.append(desc)
        scope[obj_key] = {
            "title": obj.get("title", obj_key),
            "hypothesis": obj.get("hypothesis", ""),
            "planned_steps": planned,
        }
    return json.dumps(scope, indent=2, default=str)[:max_chars]


_METHODS_RELEVANT_PACKAGES = (
    "geopandas", "pandas", "numpy", "scipy", "shapely", "pyproj", "fiona",
    "rasterio", "rtree", "statsmodels", "scikit-learn", "sklearn",
    "pysal", "libpysal", "esda", "spreg", "mgwr", "spglm", "splot",
    "pygeoda", "osmnx", "contextily", "matplotlib", "seaborn",
    "networkx", "xarray",
)


def render_environment_manifest(brief, *, ctx=None):
    """
    Render a reproducibility manifest: Python version, library versions,
    analysis date, and (if captured) random seeds used during execution.

    The LLM drops this verbatim into the 'Software and Tools' subsection of
    Methodology so reviewers can see exactly what was used — without
    hallucinating version numbers.
    """
    import sys
    import platform
    from datetime import datetime

    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
    except ImportError:
        from importlib_metadata import version as _pkg_version, PackageNotFoundError

    py_version = ".".join(map(str, sys.version_info[:3]))
    os_info = f"{platform.system()} {platform.release()}"

    pkg_versions = {}
    for pkg in _METHODS_RELEVANT_PACKAGES:
        try:
            pkg_versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            continue
        except Exception:
            continue

    seeds = []
    if ctx is not None:
        seed = ctx.get("random_seed") or ctx.get("seed")
        if seed is not None:
            seeds.append(seed)

    manifest = {
        "python_version": py_version,
        "operating_system": os_info,
        "analysis_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "package_versions": pkg_versions,
        "random_seeds": seeds,
    }

    lines = [
        "### Computational Environment (VERBATIM — use these exact values, do NOT invent)",
        f"- **Python**: {manifest['python_version']}",
        f"- **OS**: {manifest['operating_system']}",
        f"- **Analysis date (UTC)**: {manifest['analysis_date']}",
    ]
    if pkg_versions:
        pkg_str = ", ".join(f"{k}={v}" for k, v in sorted(pkg_versions.items()))
        lines.append(f"- **Key packages**: {pkg_str}")
    if seeds:
        lines.append(f"- **Random seed(s)**: {seeds}")
    else:
        lines.append("- **Random seed(s)**: not recorded (note this as a reproducibility limitation)")
    return "\n".join(lines)


def render_data_summary(brief, *, max_chars=8000):
    ds = brief.get("data_summary", "")
    if isinstance(ds, dict):
        return json.dumps(ds, indent=2, default=str)[:max_chars]
    return str(ds)[:max_chars]


def render_data_registry(brief, *, max_chars=6000):
    return json.dumps(brief.get("data_registry", []), indent=2, default=str)[:max_chars]


def render_research_plan(brief, *, max_chars=12000):
    return json.dumps(brief.get("research_plan", {}), indent=2, default=str)[:max_chars]


def render_figure_catalog(brief):
    return brief.get("figure_catalog_text", "") or "(no figures available)"


def render_output_file_stats(brief, *, max_chars=12000):
    """Render the pre-extracted output file stats so the LLM doesn't need read_output_file."""
    files = brief.get("output_files", {}) or {}
    if not files:
        return ""

    lines = [
        "### Output File Statistics (pre-extracted — DO NOT call read_output_file)\n",
        f"Stats for {len(files)} output file(s) are listed below. Use these directly.\n",
    ]
    for fname, st in files.items():
        ftype = st.get("type", "unknown")
        if ftype == "csv":
            lines.append(
                f"\n**{fname}** (csv) — shape={st['shape']}\n"
                + (f"NOTE: {st['statistics_note']}\n" if st.get('statistics_note') else "")
                + f"Columns: {', '.join(map(str, st['columns']))}\n"
                f"Rows:\n```\n{st['head']}\n```\n"
                f"Describe:\n```\n{st['describe']}\n```"
            )
        elif ftype == "spatial":
            lines.append(
                f"\n**{fname}** (spatial) — shape={st['shape']}, CRS={st['crs']}\n"
                + (f"NOTE: {st['statistics_note']}\n" if st.get('statistics_note') else "")
                + f"Columns: {', '.join(map(str, st['columns']))}\n"
                f"Describe:\n```\n{st['describe']}\n```"
            )
        elif ftype == "json":
            lines.append(
                f"\n**{fname}** (json)\nPreview:\n```\n{st['preview']}\n```"
            )
        elif ftype == "text":
            lines.append(
                f"\n**{fname}** (text)\nPreview:\n```\n{st['preview']}\n```"
            )
        elif ftype == "image":
            lines.append(f"\n**{fname}** (image, {st.get('size', '?')} bytes)")
        elif ftype == "html":
            lines.append(f"\n**{fname}** (html, {st.get('size', '?')} bytes)")
        elif ftype == "error":
            lines.append(f"\n**{fname}** — could not read: {st.get('error', '')}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [file stats truncated]"
    return text


_ROLE_HEADERS = {
    "foundational":   "Foundational (seminal / theoretical anchors)",
    "methodological": "Methodological (prior methods this study builds on)",
    "regional":       "Regional / Contextual (study region or population)",
    "gap_evidence":   "Gap Evidence (papers whose LIMITATIONS define the gap)",
}


def render_pre_registered_citations(brief):
    """
    Render the pre-registered citation list so the LLM can cite directly with
    `(Author et al., YEAR) [REF_N]` — no `add_citation` call required.

    If a role classification is available (from
    ``_curate_and_expand_literature``), citations are grouped by role so the
    writer can pick foundational anchors for Paragraph 1, methodological
    precedents for Paragraph 2, and gap-evidence citations for the Research
    Gap paragraph. Falls back to an ungrouped listing otherwise.
    """
    store = brief.get("citation_store_snapshot", {}) or {}
    if not store:
        return ""

    from utils.citation_tools import format_authors_inline

    # Build paper_id → ref_num lookup, then group refs by the role attached
    # to the paper (if any).
    id_to_ref = {
        (p.get("paper_id") or ""): ref
        for ref, p in store.items()
    }
    by_role_papers = brief.get("literature_by_role") or {}
    role_to_refs = {r: [] for r in _ROLE_HEADERS}
    classified_refs = set()
    for role, papers in by_role_papers.items():
        if role not in role_to_refs:
            continue
        for p in papers:
            ref = id_to_ref.get(p.get("paper_id") or "")
            if ref:
                role_to_refs[role].append(ref)
                classified_refs.add(ref)
    unclassified_refs = [r for r in store.keys() if r not in classified_refs]

    lines = [
        "### Pre-Registered Citations (use directly — NO `add_citation` call needed)\n",
        f"{len(store)} paper(s) from a batch literature search are already registered "
        "with a permanent REF_N below. To cite, write the inline format **exactly** "
        "as shown next to each entry. Do NOT call `add_citation` for these papers.\n",
        "Only call `search_literature` + `add_citation` if you need a NEW paper "
        "that isn't already in this list.\n",
    ]

    def _format_entry(ref_num, p):
        authors = p.get("authors", []) or []
        inline = format_authors_inline(authors)
        year = p.get("year", "n.d.")
        title = p.get("title", "Untitled")
        venue = p.get("venue", "") or ""
        abstract = (p.get("abstract", "") or "")[:250]
        return (
            f"- **CITE AS:** `({inline}, {year}) [REF_{ref_num}]`  \n"
            f"  Title: {title}  \n"
            f"  Venue: {venue}  \n"
            f"  Abstract: {abstract}..."
        )

    has_grouping = any(role_to_refs.values())
    if has_grouping:
        lines.append(
            "\n**Papers are grouped by role.** Use FOUNDATIONAL for the opening "
            "hook, METHODOLOGICAL for the approach paragraph, REGIONAL for local "
            "context, and GAP EVIDENCE to anchor the research-gap claim. Do NOT "
            "cite a paper from the wrong role bucket — the classification is "
            "deliberate.\n"
        )
        for role, header in _ROLE_HEADERS.items():
            refs = role_to_refs.get(role) or []
            if not refs:
                lines.append(f"\n#### {header}\n_(none — consider calling `search_literature` if this role is needed)_")
                continue
            lines.append(f"\n#### {header}")
            for ref_num in sorted(refs, key=lambda r: int(r) if str(r).isdigit() else r):
                lines.append(_format_entry(ref_num, store[ref_num]))
        if unclassified_refs:
            lines.append("\n#### Unclassified")
            for ref_num in sorted(unclassified_refs, key=lambda r: int(r) if str(r).isdigit() else r):
                lines.append(_format_entry(ref_num, store[ref_num]))
    else:
        for ref_num in sorted(store.keys(), key=lambda r: int(r) if str(r).isdigit() else r):
            lines.append(_format_entry(ref_num, store[ref_num]))

    return "\n".join(lines)


def render_drafted_sections(drafted_sections, section_names, *, max_chars=10000):
    """Format specific previously-drafted sections for cross-reference."""
    parts = []
    for name in section_names:
        content = (drafted_sections or {}).get(name, "")
        if not content:
            continue
        truncated = content[:max_chars]
        if len(content) > max_chars:
            truncated += "\n\n... [truncated]"
        parts.append(f"### Drafted Section: {name}\n{truncated}")
    return "\n\n".join(parts)


# ── Tools (escape hatches only) ─────────────────────────────────────────────

def build_manuscript_tools(ctx, *, mode: str = "minimal", pre_searched: list | None = None):
    """
    Build the tool list bound to a section writer LLM.

    All context-fetch tools (get_research_context, get_collated_results,
    get_data_summary, get_data_registry, get_research_plan, get_drafted_section,
    get_figure_catalog, list_output_files, read_output_file, get_step_details)
    have been removed because the manuscript brief now provides this data
    directly inside the user message.

    Modes
    -----
    "none"        — empty list (single-shot generation, no tool loop)
    "minimal"     — `run_analysis_snippet` only (escape hatch for ad-hoc stats)
    "literature"  — minimal + search_literature + add_citation + get_citations,
                    for sections that may need to find ADDITIONAL papers beyond
                    the pre-registered list
    """
    if mode == "none":
        return []

    artifacts = ctx.setdefault("artifacts", {})
    citation_store = artifacts.setdefault("citation_store", {})
    _search_cache = {}

    # Pre-populate the search cache with batch-searched papers so add_citation
    # works without the LLM having to call search_literature first.
    if pre_searched:
        for p in pre_searched:
            if p.get("paper_id"):
                _search_cache[p["paper_id"]] = p

    @tool
    def run_analysis_snippet(code: str) -> str:
        """Execute a small Python snippet (max 50 lines) to compute additional
        statistics from pipeline outputs. Returns stdout. Use ONLY when the
        pre-extracted file stats in the prompt do not contain the number you need."""
        lines = code.strip().split("\n")
        if len(lines) > 50:
            return "Error: max 50 lines."
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<manuscript_analysis>", "exec"))
            return buf.getvalue()[:3000] or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    tools = [run_analysis_snippet]

    if mode == "literature":
        @tool
        def search_literature(query: str, year_range: str = "") -> str:
            """Search Semantic Scholar for ADDITIONAL papers (beyond the pre-registered
            list in the prompt). Returns title, authors, year, venue, DOI, citation count."""
            from utils.citation_tools import search_papers
            results = search_papers(query, limit=5, year_range=year_range or None)
            if isinstance(results, dict) and "error" in results:
                return f"Search error: {results['error']}."
            if not results:
                return "No results found."
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
                    f"    Abstract: {p['abstract'][:250]}...\n"
                )
            return "\n".join(lines)

        @tool
        def add_citation(paper_id: str) -> str:
            """Register a paper from search_literature (for NEW papers not in the
            pre-registered list). Returns the [REF_N] and inline format to use."""
            from utils.citation_tools import format_authors_inline
            if paper_id not in _search_cache:
                return (f"Paper ID '{paper_id}' not in recent search results. "
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
            """List all citations registered so far (pre-registered + any newly added)."""
            if not citation_store:
                return "No citations registered."
            from utils.citation_tools import format_authors_inline
            lines = [f"Current citations ({len(citation_store)} total):\n"]
            for ref_num in sorted(citation_store.keys(), key=int):
                p = citation_store[ref_num]
                inline = format_authors_inline(p["authors"])
                lines.append(f"  [REF_{ref_num}] ({inline}, {p['year']}) — {p['title']}")
            return "\n".join(lines)

        tools.extend([search_literature, add_citation, get_citations])

    return tools