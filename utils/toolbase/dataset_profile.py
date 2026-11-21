"""What a planner needs to judge whether a dataset fits a question.

profile_dataset(path) describes where the data is (CRS, geometry type, extent
in longitude/latitude), what it covers (row count, time span, the categories
in text columns such as state or county names) and how complete it is
(missing values per column). It never raises: a file it cannot read gets a
profile saying so.

format_profile() turns a profile into a few compact lines for an LLM prompt.
Profiles are cached by path and modification time.
"""
import os
import re
from functools import lru_cache

VECTOR_EXTS = {".geojson", ".json", ".gpkg", ".shp", ".kml", ".gml", ".fgb"}
TABLE_EXTS = {".csv", ".xlsx", ".xls", ".parquet"}
RASTER_EXTS = {".tif", ".tiff", ".img", ".vrt", ".nc"}

MAX_ATTRIBUTE_BYTES = 300 * 1024 * 1024   # skip attribute stats above this
MAX_COLUMNS = 40
TOP_VALUES = 5
_TIME_NAME = re.compile(r"(date|time|year|yr|month|day|period|timestamp)", re.I)


def profile_dataset(path):
    """A dict describing *path*; see the module docstring."""
    try:
        stat = os.stat(path)
    except OSError as e:
        return {"path": path, "error": f"cannot read file: {e}"}
    return dict(_profile(path, stat.st_mtime, stat.st_size))


@lru_cache(maxsize=128)
def _profile(path, mtime, size):
    ext = os.path.splitext(path)[1].lower()
    profile = {"path": path, "size_mb": round(size / 1e6, 2)}
    try:
        if ext in VECTOR_EXTS:
            profile.update(_vector(path, size))
        elif ext in TABLE_EXTS:
            profile.update(_table(path, ext))
        elif ext in RASTER_EXTS:
            profile.update(_raster(path))
        else:
            profile["kind"] = "unknown"
            profile["note"] = f"no profiler for {ext or 'files without an extension'}"
    except Exception as e:                      # a profile must never break planning
        profile["error"] = f"{type(e).__name__}: {e}"
    return tuple(profile.items())


def _wgs84_bounds(bounds, crs):
    """(min_lon, min_lat, max_lon, max_lat), or None when it cannot be worked out."""
    if bounds is None or crs is None:
        return None
    from pyproj import CRS, Transformer
    src = CRS.from_user_input(crs)
    if src.is_geographic and src.to_epsg() == 4326:
        out = bounds
    else:
        out = Transformer.from_crs(src, "EPSG:4326", always_xy=True).transform_bounds(*bounds)
    return [round(float(v), 4) for v in out]


def _crs_label(crs):
    if not crs:
        return None
    from pyproj import CRS
    c = CRS.from_user_input(crs)
    epsg = c.to_epsg()
    return f"EPSG:{epsg} ({c.name})" if epsg else c.name


def _vector(path, size):
    import pyogrio
    info = pyogrio.read_info(path, force_total_bounds=True)
    crs = info.get("crs")
    out = {
        "kind": "vector",
        "crs": _crs_label(crs),
        "geometry_type": info.get("geometry_type"),
        "rows": int(info.get("features") or 0),
    }
    bounds = info.get("total_bounds")
    if bounds is not None:
        try:
            out["extent_lonlat"] = _wgs84_bounds(tuple(bounds), crs)
        except Exception as e:
            out["extent_note"] = f"extent not converted to lon/lat: {e}"
    if size <= MAX_ATTRIBUTE_BYTES:
        df = pyogrio.read_dataframe(path, read_geometry=False)
        out["columns"] = _column_stats(df)
    else:
        out["columns_note"] = f"attributes not profiled (file over {MAX_ATTRIBUTE_BYTES // 1_000_000} MB)"
    return out


def _table(path, ext):
    import pandas as pd
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)
    out = {"kind": "table", "rows": len(df), "columns": _column_stats(df)}
    lon = next((c for c in df.columns if str(c).lower() in ("lon", "lng", "long", "longitude", "x")), None)
    lat = next((c for c in df.columns if str(c).lower() in ("lat", "latitude", "y")), None)
    if lon is not None and lat is not None:
        x = pd.to_numeric(df[lon], errors="coerce")
        y = pd.to_numeric(df[lat], errors="coerce")
        if x.notna().any() and y.notna().any():
            out["point_columns"] = [str(lon), str(lat)]
            out["extent_lonlat"] = [round(float(v), 4) for v in (x.min(), y.min(), x.max(), y.max())]
    return out


def _raster(path):
    try:
        import rasterio
    except Exception as e:
        return {"kind": "raster", "note": f"raster not profiled here (rasterio unavailable: {e})"}
    with rasterio.open(path) as r:
        crs = r.crs.to_wkt() if r.crs else None
        out = {
            "kind": "raster",
            "crs": _crs_label(crs),
            "size_px": [r.width, r.height],
            "bands": r.count,
            "resolution": [round(abs(v), 6) for v in r.res],
            "dtype": r.dtypes[0] if r.dtypes else None,
            "nodata": r.nodata,
        }
        out["extent_lonlat"] = _wgs84_bounds(tuple(r.bounds), crs)
    return out


def _column_stats(df):
    import pandas as pd
    stats = {}
    n = len(df)
    for col in list(df.columns)[:MAX_COLUMNS]:
        s = df[col]
        entry = {"dtype": str(s.dtype)}
        missing = int(s.isna().sum())
        if missing:
            entry["missing"] = f"{missing}/{n}"
        if pd.api.types.is_bool_dtype(s):
            pass
        elif pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                lo, hi = s.min(), s.max()
                entry["range"] = [_num(lo), _num(hi)]
                if _TIME_NAME.search(str(col)):
                    # epoch seconds / milliseconds -> dates, so the time span is readable
                    for unit, (a, b) in (("ms", (1e11, 1e13)), ("s", (1e8, 1e10))):
                        if a <= lo and hi <= b:
                            dates = pd.to_datetime([lo, hi], unit=unit, utc=True)
                            entry["range"] = [str(dates[0].date()), str(dates[1].date())]
                            entry["dtype"] += f" (epoch {unit})"
                            break
        elif pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                entry["range"] = [str(s.min()), str(s.max())]
        else:
            text = s.dropna().astype(str)
            if _TIME_NAME.search(str(col)):
                parsed = pd.to_datetime(text, errors="coerce", utc=True)
                if parsed.notna().mean() > 0.8:
                    entry["range"] = [str(parsed.min().date()), str(parsed.max().date())]
            if "range" not in entry:
                distinct = text.nunique()
                entry["distinct"] = int(distinct)
                if distinct and distinct == len(text):
                    entry["unique"] = True
                    entry["examples"] = [str(v) for v in text.head(3)]
                elif distinct <= 60:
                    entry["top"] = [f"{v} ({c})" for v, c in text.value_counts().head(TOP_VALUES).items()]
        stats[str(col)] = entry
    if len(df.columns) > MAX_COLUMNS:
        stats["..."] = {"note": f"{len(df.columns) - MAX_COLUMNS} more columns not profiled"}
    return stats


def _num(v):
    v = float(v)
    return int(v) if v.is_integer() else round(v, 4)


DATA_FITNESS_CHECKS = """- Area: does its extent (or its place names) cover the study area the question asks about?
- Time: does it cover the period the question asks about?
- Unit: is it at the unit of analysis the question needs (point, tract, county, state...), or can it be brought to it?
- Content: does it hold the variables the question needs?"""


def format_candidates(registry):
    """One profile block per data-registry entry (extent, CRS, period, columns), with its path."""
    blocks = []
    for s in registry or []:
        name, path = s.get("file", ""), s.get("path", "")
        block = format_profile(name, profile_dataset(path)) if path else f"### {name}\n- no path"
        blocks.append(f"{block}\n- path: {path} | source: {s.get('source', '')}")
    return "\n\n".join(blocks) or "(no datasets)"


def profile_tool(registry):
    """A LangChain tool, profile_dataset(dataset_name), over the given data registry."""
    from langchain_core.tools import tool

    @tool("profile_dataset")
    def _profile_tool(dataset_name: str) -> str:
        """Check whether a dataset fits the question: its CRS, geometry type, extent in
        longitude/latitude, row count, time span, missing values, and the categories in
        text columns (e.g. which states or counties it covers). Use the exact dataset
        name from explore_datasets."""
        for src in registry or []:
            if src.get("file") == dataset_name:
                return format_profile(dataset_name, profile_dataset(src.get("path", "")))
        available = [s.get("file") for s in registry or []]
        return f"Dataset '{dataset_name}' not found. Available: {available}"

    return _profile_tool


def format_profile(name, profile):
    """A few compact lines describing one dataset for a prompt."""
    head = [f"### {name}"]
    if profile.get("error"):
        head.append(f"- could not profile: {profile['error']}")
        return "\n".join(head)
    kind = profile.get("kind", "?")
    facts = [kind]
    for key in ("geometry_type", "crs"):
        if profile.get(key):
            facts.append(str(profile[key]))
    if "rows" in profile:
        facts.append(f"{profile['rows']} rows")
    head.append("- " + " | ".join(facts))
    if profile.get("extent_lonlat"):
        b = profile["extent_lonlat"]
        head.append(f"- extent (lon/lat): {b[0]}, {b[1]} to {b[2]}, {b[3]}")
    for key in ("note", "extent_note", "columns_note"):
        if profile.get(key):
            head.append(f"- {profile[key]}")
    cols = profile.get("columns") or {}
    if cols:
        parts = []
        for col, s in cols.items():
            bits = [s.get("dtype", "")] if "dtype" in s else []
            if "range" in s:
                bits.append(f"{s['range'][0]}..{s['range'][1]}")
            if "top" in s:
                bits.append("e.g. " + ", ".join(s["top"][:3]))
            elif s.get("unique"):
                bits.append(f"unique, e.g. {', '.join(s['examples'][:2])}")
            elif "distinct" in s:
                bits.append(f"{s['distinct']} distinct")
            if "missing" in s:
                bits.append(f"missing {s['missing']}")
            if "note" in s:
                bits = [s["note"]]
            parts.append(f"{col} [{'; '.join(b for b in bits if b)}]")
        head.append("- columns: " + "; ".join(parts))
    return "\n".join(head)
