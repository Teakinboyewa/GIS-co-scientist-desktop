"""
Auto-renderer for `kind='data_to_viz'` catalog entries.

When the LLM-generated code only saves data (.gpkg, .csv) and records it as
`data_to_viz` with a `viz_hint`, the presentation layer calls `render_entry`
to turn that data into a PNG just-in-time, so the manuscript sees a real
figure instead of a dangling data reference.

Rendered PNGs land under `<output_dir>/figures/auto/` to keep them out of the
task root and final_outputs (which are author/LLM-owned spaces).
"""

from __future__ import annotations

import os
import hashlib
from typing import Optional

from utils.lazy_imports import lazy_pyplot
plt = lazy_pyplot()  # matplotlib loads (with the Agg backend) on first use


AUTO_DIR = os.path.join("figures", "auto")


def _auto_dir(output_dir: str) -> str:
    d = os.path.join(output_dir, AUTO_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def _stable_name(source_path: str, viz_hint: dict, ext: str = ".png") -> str:
    base = os.path.splitext(os.path.basename(source_path))[0]
    key = f"{source_path}|{sorted((viz_hint or {}).items())}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()[:8]
    return f"{base}__{h}{ext}"


def _render_gpkg(source: str, viz_hint: dict, out_png: str, caption: str) -> None:
    import geopandas as gpd

    gdf = gpd.read_file(source)
    vtype = (viz_hint or {}).get("type", "choropleth")
    column = (viz_hint or {}).get("column")
    cmap = (viz_hint or {}).get("cmap", "viridis")
    scheme = (viz_hint or {}).get("scheme")
    legend = (viz_hint or {}).get("legend", True)

    fig, ax = plt.subplots(figsize=(9, 7))
    if vtype in ("choropleth", "map") and column and column in gdf.columns:
        kwargs = {"column": column, "cmap": cmap, "legend": legend, "ax": ax}
        if scheme:
            kwargs["scheme"] = scheme
        gdf.plot(**kwargs)
    elif vtype in ("point_map", "points"):
        gdf.plot(ax=ax, markersize=6, color=(viz_hint or {}).get("color", "#d62728"))
    elif vtype == "layered_map":
        gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.3)
    else:
        gdf.plot(ax=ax)

    ax.set_axis_off()
    ax.set_title(caption, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_csv(source: str, viz_hint: dict, out_png: str, caption: str) -> None:
    import pandas as pd

    df = pd.read_csv(source)
    vtype = (viz_hint or {}).get("type", "bar")
    column = (viz_hint or {}).get("column")
    x = (viz_hint or {}).get("x")

    fig, ax = plt.subplots(figsize=(9, 6))
    if vtype == "bar" and column and column in df.columns:
        labels = df[x] if x and x in df.columns else df.index
        ax.bar(labels, df[column])
        ax.set_ylabel(column)
        if x:
            ax.set_xlabel(x)
            plt.xticks(rotation=45, ha="right")
    elif vtype == "scatter" and x and column:
        ax.scatter(df[x], df[column], s=10, alpha=0.7)
        ax.set_xlabel(x)
        ax.set_ylabel(column)
    elif vtype == "hist" and column and column in df.columns:
        ax.hist(df[column].dropna(), bins=30)
        ax.set_xlabel(column)
    else:
        df.plot(ax=ax)

    ax.set_title(caption, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_entry(entry: dict, output_dir: str) -> Optional[str]:
    """
    Render a `data_to_viz` entry to PNG. Returns the absolute PNG path, or
    None if the source extension isn't renderable / rendering fails.

    Idempotent: the output name is a stable hash of (source, viz_hint), so a
    rebuild of the catalog reuses prior renders instead of regenerating.
    """
    source = entry.get("path")
    if not source or not os.path.exists(source):
        return None
    viz_hint = entry.get("viz_hint") or {}
    caption = entry.get("caption") or os.path.basename(source)

    ext = os.path.splitext(source)[1].lower()
    out_png = os.path.join(_auto_dir(output_dir), _stable_name(source, viz_hint))

    if os.path.exists(out_png):
        return out_png

    try:
        if ext in (".gpkg", ".shp", ".geojson"):
            _render_gpkg(source, viz_hint, out_png, caption)
        elif ext in (".csv", ".tsv", ".parquet", ".xlsx", ".xls"):
            _render_csv(source, viz_hint, out_png, caption)
        else:
            return None
    except Exception:
        return None

    return out_png if os.path.exists(out_png) else None


def realize_data_to_viz(catalog: list, output_dir: str) -> list:
    """
    In-place: for every `kind='data_to_viz'` entry with a successful render,
    swap `path` to the PNG, set `kind='figure'`, and keep the original source
    on `source_files` so the manuscript can cite the underlying data.

    Entries that fail to render keep `kind='data_to_viz'` so downstream code
    can choose to skip or cite data-only.
    """
    for entry in catalog:
        if entry.get("kind") != "data_to_viz":
            continue
        source_path = entry["path"]
        png = render_entry(entry, output_dir)
        if not png:
            continue
        existing_sources = list(entry.get("source_files") or [])
        base = os.path.basename(source_path)
        if base not in existing_sources:
            existing_sources.append(base)
        entry["source_files"] = existing_sources
        entry["path"] = png.replace("\\", "/")
        entry["kind"] = "figure"
        entry["rendered_from"] = source_path.replace("\\", "/")
    return catalog
