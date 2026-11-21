"""
Exploration and confirmation data for the diagnostics.

Diagnostics may revise the plan, so they must not see the data the
confirmatory analysis runs on (otherwise the plan is tuned to its own test).

  replicate      an independent realization of the same process is supplied
                 (ctx["settings"]["exploration_data"], set by the benchmark
                 runner); diagnostics use it, the analysis uses the data as is.
  blocked_split  no replicate (real data, WebUI): units are split by a
                 checkerboard of spatial tiles; diagnostics use one half and
                 the analysis is pointed at the other.

Datasets without geometry cannot be split spatially and are shared by both.
"""

import json
import os

import numpy as np

from agents.spatial_reasoning.assumptions import input_datasets
from utils.config import resolve_output_dir

SPLIT_DIR = "spatial_split"
TILES_PER_AXIS = 4


def _read(path):
    import geopandas as gpd
    import pandas as pd
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return gpd.read_file(path)


def _has_geometry(frame):
    return "geometry" in getattr(frame, "columns", []) and hasattr(frame, "crs")


def _tile_parity(frame, bounds):
    minx, miny, maxx, maxy = bounds
    c = frame.geometry.representative_point()
    tx = np.clip(((c.x - minx) / ((maxx - minx) or 1) * TILES_PER_AXIS).astype(int), 0, TILES_PER_AXIS - 1)
    ty = np.clip(((c.y - miny) / ((maxy - miny) or 1) * TILES_PER_AXIS).astype(int), 0, TILES_PER_AXIS - 1)
    return ((tx + ty) % 2).to_numpy()


def prepare(ctx):
    """Decide exploration/confirmation data and write split files if needed.
    Returns the split record (also saved as spatial_split/split.json)."""
    datasets = input_datasets(ctx)
    provided = (ctx.get("settings", {}) or {}).get("exploration_data") or {}
    out_dir = os.path.join(resolve_output_dir(ctx), SPLIT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    if provided and all(sid in provided for sid in datasets):
        record = {"mode": "replicate",
                  "exploration": {sid: provided[sid] for sid in datasets},
                  "confirmation": {sid: d["path"] for sid, d in datasets.items()},
                  "notes": ["Diagnostics use an independent realization of the same process."]}
    else:
        frames = {sid: _read(d["path"]) for sid, d in datasets.items()}
        spatial = {sid: f for sid, f in frames.items() if _has_geometry(f)}
        if spatial:
            crs = next(iter(spatial.values())).crs
            bounds = np.array([f.to_crs(crs).total_bounds for f in spatial.values()])
            extent = (bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max())
        record = {"mode": "blocked_split", "tiles_per_axis": TILES_PER_AXIS,
                  "exploration": {}, "confirmation": {}, "units": {}, "notes": []}
        for sid, frame in frames.items():
            if sid not in spatial:
                record["exploration"][sid] = record["confirmation"][sid] = datasets[sid]["path"]
                record["notes"].append(f"{sid} has no geometry and is shared by both halves.")
                continue
            parity = _tile_parity(frame.to_crs(crs), extent)
            for half, keep in (("exploration", parity == 0), ("confirmation", parity == 1)):
                path = os.path.join(out_dir, f"{sid}__{half}.gpkg")
                frame[keep].to_file(path, driver="GPKG")
                record[half][sid] = path
                record["units"].setdefault(sid, {})[half] = int(keep.sum())
        record["notes"].append("Diagnostics use one half of a checkerboard of spatial tiles; "
                               "the confirmatory analysis uses the other half.")

    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    ctx.setdefault("artifacts", {})["spatial_split"] = record
    return record


def load_exploration(record):
    """{source_id: frame} for the exploration side."""
    return {sid: _read(path) for sid, path in record["exploration"].items()}


def point_analysis_at_confirmation(ctx, record):
    """For a blocked split, make the plan's data paths resolve to the
    confirmation half. Runs before the plan's node paths are resolved."""
    if record["mode"] != "blocked_split":
        return
    for src in ctx.get("artifacts", {}).get("data_registry", []) or []:
        sid = src.get("file")
        if src.get("source") == "data_audit_record" and sid in record["confirmation"]:
            src.setdefault("original_path", src.get("path"))
            src["path"] = record["confirmation"][sid]
