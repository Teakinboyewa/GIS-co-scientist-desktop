"""
Figure catalog preparation: scans step outputs, auto-generates visualizations
from non-image files (.gpkg → choropleth maps, .csv → charts), and builds a
numbered figure catalog for use in result presentations and manuscripts.
"""

import os
import json

from utils.lazy_imports import lazy_pyplot
plt = lazy_pyplot()  # matplotlib loads (with the Agg backend) on first use


# ── Public API ───────────────────────────────────────────────────────────────

def prepare_figure_catalog(ctx):
    """
    Build a numbered figure catalog from all step execution outputs.

    1. Scans step_execution_results for created files per objective/step.
    2. Existing image files (.png, .jpg) → added directly.
    3. Data files (.gpkg, .csv) that have no companion image → auto-generates
       appropriate visualizations (choropleth maps, charts, heatmaps).
    4. Stores the catalog in ctx["artifacts"]["figure_catalog"].

    Returns:
        list[dict] — the figure catalog.
    """
    from utils.config import resolve_output_dir
    output_dir = resolve_output_dir(ctx)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    task_id = ctx.get("task_id", "unknown")
    rq_preview = (ctx.get("rq", "") or "")[:80]
    print(f"[FIGURE PREP] task_id={task_id}, output_dir={output_dir}, "
          f"rq={rq_preview!r}")

    stream_cb = ctx.get("_stream_callback")
    step_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    objectives = plan.get("objectives", [])

    # Build a lookup: obj_key → obj metadata
    obj_lookup = {}
    for obj in objectives:
        obj_num = obj.get("number", 0)
        obj_lookup[f"objective_{obj_num}"] = obj

    catalog = []
    fig_counter = [1]  # mutable counter for closure

    if stream_cb:
        stream_cb(f"  Scanning {len(step_results)} objective(s) for figures...\n")

    for obj_key, steps in step_results.items():
        obj_meta = obj_lookup.get(obj_key, {})
        obj_num = obj_meta.get("number", obj_key.split("_")[-1])
        planned_steps = obj_meta.get("steps", [])

        for step_info in steps:
            step_idx = step_info.get("step_index", 0)
            step_desc = step_info.get("step_description", "")
            status = step_info.get("status", "")
            if status != "completed":
                continue

            inner = step_info.get("result", {}) if isinstance(step_info.get("result"), dict) else {}
            created_files = inner.get("created_files", [])
            if not created_files:
                continue

            # Separate images from data files
            image_files = []
            data_files = []
            for fp in created_files:
                if not os.path.exists(fp):
                    continue
                ext = os.path.splitext(fp)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp", ".gif"):
                    image_files.append(fp)
                elif ext in (".gpkg", ".shp", ".geojson"):
                    data_files.append(("spatial", fp))
                elif ext == ".csv":
                    data_files.append(("csv", fp))

            # Add existing images to catalog
            for img_path in image_files:
                catalog.append(_make_entry(
                    fig_counter, img_path, obj_key, obj_num, step_idx, step_desc,
                    fig_type="image", auto_generated=False,
                    description=f"{step_desc}",
                ))
                if stream_cb:
                    stream_cb(f"  Found: {os.path.basename(img_path)}\n")

            # Check which data files already have a companion image
            # (e.g., step created both results.csv and results_chart.png)
            data_files_needing_viz = []
            for dtype, dpath in data_files:
                stem = os.path.splitext(os.path.basename(dpath))[0].lower()
                has_companion = any(
                    stem in os.path.basename(img).lower() for img in image_files
                )
                if not has_companion:
                    data_files_needing_viz.append((dtype, dpath))

            # Auto-generate visualizations ONLY for data files that represent
            # analytical results relevant to the objective (not intermediate data)
            obj_title = obj_meta.get("title", "")
            for dtype, dpath in data_files_needing_viz:
                if not _is_visualization_relevant(step_desc, obj_title, dpath):
                    print(f"[FIGURE PREP] Skipping auto-viz for {os.path.basename(dpath)} "
                          f"(not relevant to objective)")
                    continue
                if stream_cb:
                    stream_cb(f"  Generating visualization for {os.path.basename(dpath)}...\n")
                try:
                    if dtype == "spatial":
                        generated = _generate_spatial_figures(
                            dpath, fig_dir, obj_num, step_idx, step_desc
                        )
                    elif dtype == "csv":
                        generated = _generate_csv_figures(
                            dpath, fig_dir, obj_num, step_idx, step_desc
                        )
                    else:
                        generated = []

                    for gen_info in generated:
                        catalog.append(_make_entry(
                            fig_counter, gen_info["file_path"],
                            obj_key, obj_num, step_idx, step_desc,
                            fig_type=gen_info["type"],
                            auto_generated=True,
                            description=gen_info["description"],
                            source_file=dpath,
                        ))
                        if stream_cb:
                            stream_cb(f"  Auto-generated: {os.path.basename(gen_info['file_path'])}\n")
                except Exception as e:
                    print(f"[FIGURE PREP] Error generating viz for {dpath}: {e}")
                    if stream_cb:
                        stream_cb(f"  Error generating viz for {os.path.basename(dpath)}: {e}\n")

    # Store in artifacts
    ctx.setdefault("artifacts", {})["figure_catalog"] = catalog
    auto_count = sum(1 for f in catalog if f['auto_generated'])
    print(f"[FIGURE PREP] Built catalog with {len(catalog)} figures "
          f"({auto_count} auto-generated)")
    return catalog


def format_figure_catalog_for_prompt(catalog):
    """Format the figure catalog as text for inclusion in LLM prompts.

    Uses shortened filenames instead of absolute paths to keep embed syntax
    concise and avoid LLM truncation of long Windows paths.
    """
    if not catalog:
        return "No figures available."

    lines = ["## Available Figures\n"]
    lines.append("**IMPORTANT**: When embedding figures, use ONLY the filename shown below.")
    lines.append("Do NOT use absolute paths. Use: `![Figure N: description](filename.png)`\n")
    current_obj = None

    for fig in catalog:
        obj_key = fig["objective_key"]
        if obj_key != current_obj:
            current_obj = obj_key
            lines.append(f"\n### {obj_key.replace('_', ' ').title()}")

        auto_tag = " [auto-generated]" if fig["auto_generated"] else ""
        # Use just the filename — the converter will resolve full paths
        fname = os.path.basename(fig['file_path'])
        lines.append(
            f"- **Figure {fig['figure_number']}** (Step {fig['step_index']}): "
            f"{fig['description']}{auto_tag}\n"
            f"  Filename: `{fname}`\n"
            f"  Embed with: `![Figure {fig['figure_number']}: {fig['description']}]({fname})`"
        )

    return "\n".join(lines)


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _is_visualization_relevant(step_desc, obj_title, file_path):
    """
    Determine if auto-generating a visualization for this data file is relevant
    to the objective.  Returns True only when the file represents an analytical
    result (not raw/intermediate data) that would benefit from a figure.

    Decision is based on:
      - Step description keywords (analysis vs. data prep)
      - File-name heuristics (result indicators vs. raw-data indicators)
      - Objective title (tie-breaker for ambiguous cases)
    """
    step_lower = step_desc.lower()
    fname_lower = os.path.basename(file_path).lower()
    obj_lower = obj_title.lower()

    # ── Keywords signaling the step produces analytical results ──
    analysis_kw = [
        "regression", "model", "gwr", "ols", "glm", "correlation",
        "cluster", "hotspot", "spatial auto", "moran", "coefficient",
        "predict", "classif", "score", "index", "result", "estimate",
        "interpolat", "statistic", "test", "anova", "compare",
        "relationship", "association", "pattern", "risk", "vulnerab",
        "exposure", "impact", "effect", "significan", "map result",
        "bivariate", "multivariate", "principal component", "factor analysis",
        "geograph", "weight", "lisa", "getis",
    ]

    # ── Keywords signaling data preparation / intermediate work ──
    prep_kw = [
        "download", "load data", "import data", "clean", "preprocess",
        "merge", "join", "filter", "subset", "extract", "collect",
        "fetch", "acquire", "prepare data", "data preparation", "geocod",
        "convert", "transform format", "reproject", "clip", "buffer",
        "read data", "load csv", "load shapefile", "read csv",
    ]

    is_analysis = any(kw in step_lower for kw in analysis_kw)
    is_prep = any(kw in step_lower for kw in prep_kw)

    # Clear data-prep step with no analysis signal → skip
    if is_prep and not is_analysis:
        return False

    # ── File-name heuristics ──
    raw_indicators = ["raw", "input_", "source_", "original", "download"]
    result_indicators = [
        "result", "output", "coef", "predict", "score", "cluster",
        "hotspot", "index", "model", "gwr", "ols", "residual",
        "summary", "lisa", "moran",
    ]

    if any(ind in fname_lower for ind in result_indicators):
        return True
    if any(ind in fname_lower for ind in raw_indicators):
        return False

    # Clear analysis step → visualize
    if is_analysis:
        return True

    # ── Tie-breaker: does the objective title hint at analytical work? ──
    if any(kw in obj_lower for kw in analysis_kw):
        return True

    # Default: conservative — don't auto-generate
    return False


def _make_entry(fig_counter, file_path, obj_key, obj_num, step_idx, step_desc,
                fig_type, auto_generated, description, source_file=None):
    """Create a catalog entry and increment the figure counter."""
    entry = {
        "figure_number": fig_counter[0],
        "file_path": file_path.replace("\\", "/"),
        "objective_key": obj_key,
        "objective_number": obj_num,
        "step_index": step_idx,
        "step_description": step_desc,
        "type": fig_type,
        "auto_generated": auto_generated,
        "description": description,
        "source_file": source_file.replace("\\", "/") if source_file else None,
    }
    fig_counter[0] += 1
    return entry


def _generate_spatial_figures(gpkg_path, fig_dir, obj_num, step_idx, step_desc):
    """Generate choropleth map(s) from a spatial file (.gpkg/.shp/.geojson)."""
    import geopandas as gpd

    gdf = gpd.read_file(gpkg_path)
    if len(gdf) == 0:
        return []

    # Filter to meaningful numeric columns
    skip = {"fid", "index", "ogc_fid", "objectid", "id"}
    numeric_cols = [
        c for c in gdf.select_dtypes(include="number").columns
        if c.lower() not in skip
    ]

    prefix = f"obj{obj_num}_s{step_idx}"
    figures = []

    if not numeric_cols:
        # Just plot geometries
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        gdf.plot(ax=ax, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"Spatial Distribution — {step_desc[:50]}", fontsize=11)
        ax.axis("off")
        out_path = os.path.join(fig_dir, f"{prefix}_map.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        figures.append({
            "file_path": out_path,
            "type": "map",
            "description": f"Spatial distribution map — {step_desc[:60]}",
        })
    else:
        # Choropleth for up to 3 most variable numeric columns
        # Pick columns with highest coefficient of variation for visual interest
        col_scores = []
        for c in numeric_cols:
            s = gdf[c].dropna()
            if len(s) > 0 and s.std() > 0:
                col_scores.append((c, s.std() / (abs(s.mean()) + 1e-9)))
        col_scores.sort(key=lambda x: x[1], reverse=True)
        top_cols = [c for c, _ in col_scores[:3]] if col_scores else numeric_cols[:3]

        for col in top_cols:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            gdf.plot(
                column=col, ax=ax, legend=True, cmap="YlOrRd",
                edgecolor="white", linewidth=0.3,
                legend_kwds={"shrink": 0.6},
            )
            ax.set_title(col, fontsize=12)
            ax.axis("off")
            safe_col = col.replace(" ", "_").replace("%", "pct").replace("/", "_")[:30]
            out_path = os.path.join(fig_dir, f"{prefix}_{safe_col}_map.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            figures.append({
                "file_path": out_path,
                "type": "choropleth_map",
                "description": f"Choropleth map of '{col}' — {step_desc[:50]}",
            })

    return figures


def _generate_csv_figures(csv_path, fig_dir, obj_num, step_idx, step_desc):
    """Generate chart(s) from a CSV result file."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    prefix = f"obj{obj_num}_s{step_idx}"
    figures = []
    fname = os.path.splitext(os.path.basename(csv_path))[0]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Skip large raw-data CSVs — only visualize result/summary files
    if len(df) > 500 and len(numeric_cols) > 10:
        return []

    # Case 1: Correlation matrix (square, symmetric, values in [-1, 1])
    if (df.shape[0] == df.shape[1]
            and len(numeric_cols) >= 3
            and df.shape[0] <= 30):
        vals = df[numeric_cols].values
        if np.nanmin(vals) >= -1.1 and np.nanmax(vals) <= 1.1:
            try:
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    df[numeric_cols].set_index(df.columns[0]) if df.columns[0] not in numeric_cols else df[numeric_cols],
                    annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    ax=ax, linewidths=0.5,
                )
                ax.set_title(f"Correlation Matrix — {step_desc[:40]}", fontsize=11)
                out_path = os.path.join(fig_dir, f"{prefix}_{fname}_heatmap.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close()
                figures.append({
                    "file_path": out_path,
                    "type": "heatmap",
                    "description": f"Correlation matrix heatmap — {step_desc[:50]}",
                })
                return figures
            except Exception:
                pass

    # Case 2: Small results table (e.g., regression coefficients, summary stats)
    if len(df) <= 30 and len(numeric_cols) >= 1:
        # Try to find a label column for the y-axis
        label_col = None
        for c in df.columns:
            if c not in numeric_cols and df[c].nunique() == len(df):
                label_col = c
                break

        # Pick the most interesting numeric column to plot
        plot_col = None
        for candidate in numeric_cols:
            cname = candidate.lower()
            # Prefer coefficient/estimate columns
            if any(kw in cname for kw in ("coef", "estimate", "beta", "mean", "value", "score")):
                plot_col = candidate
                break
        if not plot_col:
            plot_col = numeric_cols[0]

        if label_col and len(df) >= 2:
            fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
            y_pos = range(len(df))
            ax.barh(y_pos, df[plot_col].values, color="steelblue", edgecolor="white")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df[label_col].astype(str).str[:30], fontsize=9)
            ax.set_xlabel(plot_col)
            ax.set_title(f"{plot_col} — {step_desc[:40]}", fontsize=11)
            ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
            out_path = os.path.join(fig_dir, f"{prefix}_{fname}_bar.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            figures.append({
                "file_path": out_path,
                "type": "bar_chart",
                "description": f"Bar chart of '{plot_col}' — {step_desc[:50]}",
            })
            return figures

    # Case 3: Distribution of numeric columns (moderate-sized data)
    if len(df) >= 5 and len(numeric_cols) >= 1:
        plot_cols = numeric_cols[:6]  # max 6 subplots
        ncols = min(3, len(plot_cols))
        nrows = (len(plot_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, col in enumerate(plot_cols):
            ax = axes[idx]
            df[col].dropna().hist(ax=ax, bins=20, color="steelblue", edgecolor="white")
            ax.set_title(col[:25], fontsize=10)
            ax.set_ylabel("Count")

        # Hide unused subplots
        for idx in range(len(plot_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Distributions — {step_desc[:45]}", fontsize=11)
        out_path = os.path.join(fig_dir, f"{prefix}_{fname}_dist.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        figures.append({
            "file_path": out_path,
            "type": "distribution",
            "description": f"Distribution of numeric variables — {step_desc[:50]}",
        })

    return figures