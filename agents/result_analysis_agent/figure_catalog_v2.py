"""
Figure catalog v2 — manifest-driven, promotion-based.

Replaces the keyword-heuristic approach in `figure_preparation.py` with an
explicit contract:

  1. LLM-generated code declares figures at write time via
     `record_figure(...)` — writes/updates a sidecar `manifest.json` next to
     the image. This is the primary source of truth.

  2. Anything inside `FINAL_OUTPUT_DIR` is catalog-eligible by default, even
     without a manifest entry (deliverables shouldn't need paperwork).

  3. Intermediate figures in `OUTPUT_DIR` are NOT catalogued by default, but
     can be **promoted** — either by the LLM (via `record_figure(..., role=
     "promoted")`) or by the user via `promote_intermediate(...)` (WebUI can
     wire a button to this).

The old file `figure_preparation.py` is untouched and still importable, so
nothing downstream breaks while you decide which path to adopt.

--------------------------------------------------------------------------
Manifest file format (manifest.json, stored in OUTPUT_DIR or FINAL_OUTPUT_DIR)
--------------------------------------------------------------------------
{
  "figures": [
    {
      "path": "final_choropleth_map.png",   # relative to manifest dir
      "caption": "Distance from county centroid to nearest hospital (PA)",
      "role": "final" | "promoted" | "intermediate",
      "objective_key": "objective_1",
      "step_index": 6,
      "source_files": ["pa_counties_with_nearest_hospital_distance_utm18.gpkg"]
    }
  ]
}
"""

import json
import os
from typing import Optional


MANIFEST_FILENAME = "manifest.json"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".svg")
TABLE_EXTS = (".csv", ".xlsx", ".xls", ".tsv", ".parquet")
SPATIAL_EXTS = (".gpkg", ".shp", ".geojson")

# Kinds a manifest entry can declare:
#   "figure"     — an existing image file; include as-is.
#   "table"      — a data file that should appear as a TABLE in the manuscript
#                  (e.g. regression coefficients, summary stats).
#   "data_to_viz"— a data file that should be auto-rendered INTO a figure
#                  by the presentation layer (e.g. a .gpkg to choropleth, a
#                  coefficients.csv to bar chart).
KINDS = ("figure", "table", "data_to_viz")


# ── Manifest read/write primitives ───────────────────────────────────────────

def _manifest_path(directory: str) -> str:
    return os.path.join(directory, MANIFEST_FILENAME)


def _load_manifest(directory: str) -> dict:
    path = _manifest_path(directory)
    if not os.path.exists(path):
        return {"figures": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "figures" not in data or not isinstance(data["figures"], list):
            data["figures"] = []
        return data
    except (OSError, json.JSONDecodeError):
        return {"figures": []}


def _save_manifest(directory: str, data: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(_manifest_path(directory), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _upsert_entry(manifest: dict, entry: dict) -> None:
    """Add entry, or replace if a matching path already exists."""
    figures = manifest["figures"]
    for i, existing in enumerate(figures):
        if existing.get("path") == entry.get("path"):
            figures[i] = entry
            return
    figures.append(entry)


# ── Public API: called from LLM-generated code or the pipeline ───────────────

def record_figure(
    output_dir: str,
    file_path: str,
    caption: str,
    role: str = "intermediate",
    objective_key: Optional[str] = None,
    step_index: Optional[int] = None,
    source_files: Optional[list] = None,
    kind: str = "figure",
    columns: Optional[list] = None,
    viz_hint: Optional[dict] = None,
) -> None:
    """
    Record a figure's metadata in the manifest next to where it lives.

    Call this immediately after saving an image (or data file you intend to
    visualize) from LLM-generated code. The manifest sits alongside the file
    so the catalog can be rebuilt without re-running anything.

    Args:
        output_dir:    Either OUTPUT_DIR or FINAL_OUTPUT_DIR — whichever the
                       file was saved into. The manifest is written there.
        file_path:     Absolute or relative path of the artifact.
        caption:       One-line human-readable caption/title (used verbatim in
                       the manuscript — do not truncate).
        role:          "final"       → always in catalog (final deliverable)
                       "promoted"    → always in catalog (intermediate the
                                       author wants shown)
                       "intermediate"→ on disk, not in catalog unless promoted
        objective_key: e.g. "objective_1" — for grouping in the manuscript.
        step_index:    Step number within the objective.
        source_files:  Data files this artifact was generated from (optional,
                       for traceability).
        kind:          "figure"      → existing image file (default).
                       "table"       → data file that should appear as a TABLE
                                       in the manuscript (.csv/.xlsx/.parquet).
                       "data_to_viz" → data file to be auto-rendered as a
                                       figure by the presentation layer
                                       (.gpkg/.csv). Pair with `viz_hint`.
        columns:       For kind="table", optional subset of columns to render.
                       If None, all columns are used.
        viz_hint:      For kind="data_to_viz", instructions for auto-rendering:
                       {"type": "choropleth"|"bar"|"heatmap"|"scatter",
                        "column": "<column to plot>",
                        "cmap": "OrRd",
                        ...any other matplotlib/geopandas kwargs}
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown kind={kind!r}. Valid: {KINDS}")
    if not os.path.isabs(file_path):
        file_path = os.path.join(output_dir, file_path)

    rel_path = os.path.relpath(file_path, output_dir).replace("\\", "/")
    manifest = _load_manifest(output_dir)

    entry = {
        "path": rel_path,
        "caption": caption,
        "role": role,
        "kind": kind,
    }
    if objective_key:
        entry["objective_key"] = objective_key
    if step_index is not None:
        entry["step_index"] = step_index
    if source_files:
        entry["source_files"] = [os.path.basename(sf) for sf in source_files]
    if columns:
        entry["columns"] = list(columns)
    if viz_hint:
        entry["viz_hint"] = dict(viz_hint)

    _upsert_entry(manifest, entry)
    _save_manifest(output_dir, manifest)


def record_table(output_dir, file_path, caption, role="promoted", **kwargs):
    """Shortcut for `record_figure(..., kind='table')`.

    Use when a data file (.csv/.xlsx/.parquet) belongs in the manuscript AS
    A TABLE — not rendered into a figure. Example: regression coefficient
    tables, model comparison summaries, descriptive statistics.
    """
    return record_figure(
        output_dir=output_dir, file_path=file_path, caption=caption,
        role=role, kind="table", **kwargs,
    )


def record_data_to_viz(output_dir, file_path, caption, viz_hint,
                       role="final", **kwargs):
    """Shortcut for `record_figure(..., kind='data_to_viz')`.

    Use when the *data* is saved but no image was rendered — the manuscript
    pipeline will auto-render it using `viz_hint`. Prefer saving the image
    directly with matplotlib when possible; this is a fallback for when the
    code only produces data and wants presentation deferred.

    Example:
        record_data_to_viz(
            FINAL_OUTPUT_DIR, "counties_with_risk.gpkg",
            caption="County-level flood risk index, 2024",
            viz_hint={"type": "choropleth", "column": "risk_idx",
                      "cmap": "OrRd", "scheme": "Quantiles"},
        )
    """
    return record_figure(
        output_dir=output_dir, file_path=file_path, caption=caption,
        role=role, kind="data_to_viz", viz_hint=viz_hint, **kwargs,
    )


def promote_intermediate(
    output_dir: str,
    file_path: str,
    caption: Optional[str] = None,
    objective_key: Optional[str] = None,
    step_index: Optional[int] = None,
) -> None:
    """
    User-facing promotion: flip an intermediate figure into the catalog.

    Wire this to a WebUI button ("Include in manuscript") so humans can
    override when the LLM's judgment misses a useful intermediate.
    """
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(output_dir, file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Cannot promote missing file: {abs_path}")

    # Default caption from filename stem if none provided.
    if not caption:
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        caption = stem.replace("_", " ").strip().capitalize()

    record_figure(
        output_dir=output_dir,
        file_path=abs_path,
        caption=caption,
        role="promoted",
        objective_key=objective_key,
        step_index=step_index,
    )


def demote_intermediate(output_dir: str, file_path: str) -> bool:
    """Reverse `promote_intermediate` — drop the manifest entry. Returns True if removed."""
    manifest = _load_manifest(output_dir)
    rel_path = os.path.relpath(
        file_path if os.path.isabs(file_path) else os.path.join(output_dir, file_path),
        output_dir,
    ).replace("\\", "/")
    before = len(manifest["figures"])
    manifest["figures"] = [f for f in manifest["figures"] if f.get("path") != rel_path]
    if len(manifest["figures"]) != before:
        _save_manifest(output_dir, manifest)
        return True
    return False


# ── Catalog construction ─────────────────────────────────────────────────────

def build_catalog(ctx) -> list:
    """
    Assemble the figure catalog from manifests and final-dir contents.

    Precedence (highest first):
      1. Entries in FINAL_OUTPUT_DIR/manifest.json     → role "final"
      2. Any image in FINAL_OUTPUT_DIR not yet listed  → role "final" (auto,
         caption derived from filename until someone rewrites it)
      3. Manifest entries in OUTPUT_DIR with role
         "promoted" or "final"                         → included
         (role "intermediate" is deliberately skipped)

    Returns a list of dicts with stable ordering: final first, then promoted,
    sorted within each group by (objective_key, step_index, path).
    """
    from utils.config import resolve_output_dir

    output_dir = resolve_output_dir(ctx)
    final_dir = os.path.join(output_dir, "final_outputs")

    catalog = []
    seen_abs_paths = set()

    # 1. FINAL_OUTPUT_DIR manifest
    final_manifest = _load_manifest(final_dir)
    for entry in final_manifest["figures"]:
        abs_path = os.path.join(final_dir, entry["path"])
        if not os.path.exists(abs_path):
            continue
        catalog.append(_normalize_entry(entry, final_dir, default_role="final"))
        seen_abs_paths.add(os.path.abspath(abs_path))

    # 2. Unmanifested files in FINAL_OUTPUT_DIR — pick up images, tables,
    #    and spatial data. Kind is inferred from extension.
    if os.path.isdir(final_dir):
        for fname in sorted(os.listdir(final_dir)):
            fpath = os.path.join(final_dir, fname)
            if os.path.abspath(fpath) in seen_abs_paths:
                continue
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                kind = "figure"
            elif ext in TABLE_EXTS:
                kind = "table"
            elif ext in SPATIAL_EXTS:
                kind = "data_to_viz"
            else:
                continue
            stem = os.path.splitext(fname)[0]
            catalog.append({
                "path": fpath.replace("\\", "/"),
                "caption": stem.replace("_", " ").strip().capitalize(),
                "role": "final",
                "kind": kind,
                "objective_key": None,
                "step_index": None,
                "source_files": [],
                "auto_captioned": True,
            })
            seen_abs_paths.add(os.path.abspath(fpath))

    # 3. OUTPUT_DIR manifest — pick up promoted (and stray final) entries only
    intermediate_manifest = _load_manifest(output_dir)
    for entry in intermediate_manifest["figures"]:
        role = entry.get("role", "intermediate")
        if role not in ("promoted", "final"):
            continue
        abs_path = os.path.join(output_dir, entry["path"])
        if not os.path.exists(abs_path):
            continue
        if os.path.abspath(abs_path) in seen_abs_paths:
            continue
        catalog.append(_normalize_entry(entry, output_dir, default_role=role))
        seen_abs_paths.add(os.path.abspath(abs_path))

    # 4. Orphan images — PNGs/etc. the LLM saved but never recorded.
    #    Scans the task root AND a small, bounded set of conventional subdirs
    #    (figures/, plots/, images/, maps/). `figures/auto/` is skipped because
    #    that's where realize_data_to_viz writes — those images are already
    #    attached to their source catalog entries in step 5.
    orphan_dirs = [output_dir]
    for sub in ("figures", "plots", "images", "maps"):
        p = os.path.join(output_dir, sub)
        if os.path.isdir(p):
            orphan_dirs.append(p)
    auto_render_dir = os.path.abspath(os.path.join(output_dir, "figures", "auto"))

    for scan_dir in orphan_dirs:
        if os.path.abspath(scan_dir) == auto_render_dir:
            continue
        for fname in sorted(os.listdir(scan_dir)):
            fpath = os.path.join(scan_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if os.path.abspath(fpath) in seen_abs_paths:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            catalog.append({
                "path": fpath.replace("\\", "/"),
                "caption": stem.replace("_", " ").strip().capitalize(),
                "role": "promoted",
                "kind": "figure",
                "objective_key": None,
                "step_index": None,
                "source_files": [],
                "columns": None,
                "viz_hint": None,
                "auto_captioned": True,
            })
            seen_abs_paths.add(os.path.abspath(fpath))

    # 4b. Orphan tables — CSV/XLSX/Parquet the LLM saved to a dedicated
    #     `tables/` subdir without calling `record_table(...)`. Narrow by
    #     design: we do NOT sweep the whole OUTPUT_DIR for CSVs because that
    #     would pollute the catalog with intermediate/merged datasets. The
    #     `tables/` convention is the opt-in signal that a file is a
    #     manuscript-ready table.
    tables_dir = os.path.join(output_dir, "tables")
    if os.path.isdir(tables_dir):
        for fname in sorted(os.listdir(tables_dir)):
            fpath = os.path.join(tables_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if os.path.abspath(fpath) in seen_abs_paths:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TABLE_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            catalog.append({
                "path": fpath.replace("\\", "/"),
                "caption": stem.replace("_", " ").strip().capitalize(),
                "role": "promoted",
                "kind": "table",
                "objective_key": None,
                "step_index": None,
                "source_files": [],
                "columns": None,
                "viz_hint": None,
                "auto_captioned": True,
            })
            seen_abs_paths.add(os.path.abspath(fpath))

    # 5. Realize `data_to_viz` entries — render gpkg/csv to PNG using viz_hint
    #    so downstream (manuscript, UI thumbnails) sees real images.
    try:
        from .figure_renderer import realize_data_to_viz
        realize_data_to_viz(catalog, output_dir)
    except Exception:
        pass

    # 6. Prune unrenderable entries and collapse near-duplicates.
    #    (a) data_to_viz entries that still carry a non-image path after
    #        rendering — they would produce broken embeds. Drop them with a
    #        note in source_files so the data file itself can still be cited.
    #    (b) Multiple catalog entries pointing at content with the same
    #        normalized stem (e.g. `lisa_asthma_cluster_map.png` from an
    #        orphan scan AND an auto-rendered `lisa_asthma_cluster_map__<hash>.png`
    #        from a data_to_viz entry). Keep ONE per stem, preferring the
    #        richer entry (has caption, objective_key, viz_hint, or is a
    #        recorded manifest entry rather than an auto-captioned orphan).
    catalog = [
        f for f in catalog
        if not (f.get("kind") == "data_to_viz"
                and os.path.splitext(f.get("path", ""))[1].lower() not in IMAGE_EXTS)
    ]

    def _dedup_key(path: str) -> str:
        """Collapse hash-suffixed auto-render names back to their source stem.
        `tracts_asthma__b1dde435.png` → `tracts_asthma`.
        """
        stem = os.path.splitext(os.path.basename(path or ""))[0]
        if "__" in stem:
            stem = stem.rsplit("__", 1)[0]
        return stem.lower()

    def _entry_score(f: dict) -> int:
        """Higher score = keep. Prefer manifest-recorded over orphan-auto,
        prefer entries with captions / objectives, prefer rendered figures
        over orphan PNGs when both exist."""
        score = 0
        if not f.get("auto_captioned"):
            score += 4
        if f.get("objective_key"):
            score += 2
        if f.get("rendered_from"):  # survived realize_data_to_viz
            score += 2
        if f.get("viz_hint") or f.get("source_files"):
            score += 1
        if f.get("step_index"):
            score += 1
        return score

    by_key: dict = {}
    for f in catalog:
        k = _dedup_key(f["path"])
        if k not in by_key or _entry_score(f) > _entry_score(by_key[k]):
            by_key[k] = f
    catalog = list(by_key.values())

    # Stable ordering: final first, then promoted; within each, by objective/step.
    role_order = {"final": 0, "promoted": 1}
    catalog.sort(key=lambda f: (
        role_order.get(f["role"], 99),
        f.get("objective_key") or "",
        f.get("step_index") or 0,
        f["path"],
    ))

    # Assign stable figure numbers (1-indexed) AFTER sorting.
    for i, entry in enumerate(catalog, start=1):
        entry["figure_number"] = i

    # Dual-write: store under v2 key (canonical) and legacy key so existing
    # consumers (manuscript_tools, result_presentation, frontend events) keep
    # working. Entries below carry both new fields (caption, role, path) and
    # legacy aliases (description, file_path, auto_generated).
    legacy_shaped = [_to_legacy_shape(e) for e in catalog]
    artifacts = ctx.setdefault("artifacts", {})
    artifacts["figure_catalog_v2"] = catalog
    artifacts["figure_catalog"] = legacy_shaped
    return catalog


def _to_legacy_shape(entry: dict) -> dict:
    """Map a v2 entry to the legacy `figure_preparation` shape."""
    return {
        "figure_number": entry.get("figure_number"),
        "file_path": entry["path"],
        "objective_key": entry.get("objective_key") or "",
        "objective_number": _objective_number(entry.get("objective_key")),
        "step_index": entry.get("step_index") or 0,
        "step_description": entry.get("caption", ""),
        "type": entry.get("role", "final"),
        "auto_generated": bool(entry.get("auto_captioned")),
        "description": entry.get("caption", ""),
        "source_file": (entry.get("source_files") or [None])[0],
        # Pass-through v2 fields for consumers that want them.
        "role": entry.get("role"),
        "kind": entry.get("kind", "figure"),
        "caption": entry.get("caption"),
        "source_files": entry.get("source_files", []),
        "columns": entry.get("columns"),
        "viz_hint": entry.get("viz_hint"),
    }


def _objective_number(obj_key):
    if not obj_key:
        return 0
    tail = obj_key.split("_")[-1]
    try:
        return int(tail)
    except (TypeError, ValueError):
        return 0


def _normalize_entry(entry: dict, base_dir: str, default_role: str) -> dict:
    abs_path = os.path.join(base_dir, entry["path"])
    return {
        "path": abs_path.replace("\\", "/"),
        "caption": entry.get("caption") or os.path.splitext(os.path.basename(abs_path))[0],
        "role": entry.get("role", default_role),
        "kind": entry.get("kind", "figure"),
        "objective_key": entry.get("objective_key"),
        "step_index": entry.get("step_index"),
        "source_files": entry.get("source_files", []),
        "columns": entry.get("columns"),
        "viz_hint": entry.get("viz_hint"),
        "auto_captioned": False,
    }


# ── Introspection helpers for the WebUI promotion flow ──────────────────────

def list_promotable(ctx) -> list:
    """
    Return images in OUTPUT_DIR that are NOT already in the catalog.

    Use this to populate a "Include in manuscript" picker: the UI shows these
    as thumbnails, user clicks one, you call `promote_intermediate(...)`.
    """
    from utils.config import resolve_output_dir

    output_dir = resolve_output_dir(ctx)
    final_dir = os.path.join(output_dir, "final_outputs")
    figures_dir = os.path.join(output_dir, "figures")

    catalog = ctx.get("artifacts", {}).get("figure_catalog_v2")
    if catalog is None:
        catalog = build_catalog(ctx)
    catalogued_abs = {os.path.abspath(f["path"]) for f in catalog}

    candidates = []
    for root, _dirs, files in os.walk(output_dir):
        # Skip final_outputs (already catalogued) and internal dirs.
        if os.path.abspath(root) == os.path.abspath(final_dir):
            continue
        if os.path.basename(root).startswith("."):
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS and ext not in TABLE_EXTS and ext not in SPATIAL_EXTS:
                continue
            fpath = os.path.join(root, fname)
            if os.path.abspath(fpath) in catalogued_abs:
                continue
            candidates.append({
                "path": fpath.replace("\\", "/"),
                "suggested_caption": (
                    os.path.splitext(fname)[0].replace("_", " ").strip().capitalize()
                ),
                "in_figures_dir": os.path.abspath(root) == os.path.abspath(figures_dir),
            })

    candidates.sort(key=lambda c: c["path"])
    return candidates


def format_catalog_for_prompt(catalog: list) -> str:
    """Format catalog v2 for inclusion in manuscript/LLM prompts."""
    if not catalog:
        return "No figures available."

    lines = ["## Available Figures\n"]
    lines.append("**IMPORTANT**: Embed figures by filename only (no absolute paths).")
    lines.append("Use: `![Figure N: caption](filename.png)`\n")

    current_obj = None
    for fig in catalog:
        obj_key = fig.get("objective_key") or "ungrouped"
        if obj_key != current_obj:
            current_obj = obj_key
            header = obj_key.replace("_", " ").title() if obj_key != "ungrouped" else "Ungrouped"
            lines.append(f"\n### {header}")

        fname = os.path.basename(fig["path"])
        role_tag = f" [{fig['role']}]" if fig["role"] != "final" else ""
        auto_tag = " [auto-captioned — review recommended]" if fig.get("auto_captioned") else ""
        step_tag = f" (Step {fig['step_index']})" if fig.get("step_index") else ""
        kind = fig.get("kind", "figure")
        label = {"figure": "Figure", "table": "Table", "data_to_viz": "Figure"}.get(kind, "Figure")

        if kind == "table":
            cols = fig.get("columns")
            cols_note = f" (columns: {', '.join(cols)})" if cols else ""
            lines.append(
                f"- **{label} {fig['figure_number']}**{step_tag}{role_tag}: "
                f"{fig['caption']}{auto_tag}\n"
                f"  Source: `{fname}`{cols_note}\n"
                f"  Render AS A TABLE in the manuscript — do not embed as an image."
            )
        elif kind == "data_to_viz":
            hint = fig.get("viz_hint") or {}
            hint_str = json.dumps(hint) if hint else "(no hint — presentation layer decides)"
            lines.append(
                f"- **{label} {fig['figure_number']}**{step_tag}{role_tag}: "
                f"{fig['caption']}{auto_tag}\n"
                f"  Source data: `{fname}` (auto-render)\n"
                f"  Viz hint: {hint_str}"
            )
        else:
            lines.append(
                f"- **{label} {fig['figure_number']}**{step_tag}{role_tag}: "
                f"{fig['caption']}{auto_tag}\n"
                f"  Filename: `{fname}`\n"
                f"  Embed with: `![{label} {fig['figure_number']}: {fig['caption']}]({fname})`"
            )

    return "\n".join(lines)
