# ── Exploration Tools for Agentic Plan Generation ────────────────────────────
import json
import re
from langchain_core.tools import tool

_WKT_PREFIX = re.compile(
    r"^\s*(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\b",
    re.IGNORECASE,
)


def _summarize_wkt(wkt: str) -> str:
    """Replace a WKT geometry string with a compact summary (type, vertex count, bbox)."""
    m = _WKT_PREFIX.match(wkt)
    geom_type = m.group(1).upper() if m else "GEOMETRY"
    coords = re.findall(r"-?\d+\.?\d*\s+-?\d+\.?\d*", wkt)
    n_vertices = len(coords)
    bbox = None
    if coords:
        xs, ys = [], []
        for pair in coords:
            try:
                x, y = pair.split()
                xs.append(float(x))
                ys.append(float(y))
            except ValueError:
                continue
        if xs and ys:
            bbox = [min(xs), min(ys), max(xs), max(ys)]
    return f"<{geom_type}, {n_vertices} vertices, bbox={bbox}, {len(wkt)} chars>"


def _summarize_value(v):
    """Summarize a single cell value that might be a geometry in WKT/Shapely/GeoJSON form."""
    # Shapely geometry — has .wkt attribute (checked duck-typed to avoid a hard shapely import)
    wkt_attr = getattr(v, "wkt", None)
    if isinstance(wkt_attr, str) and _WKT_PREFIX.match(wkt_attr):
        return _summarize_wkt(wkt_attr)
    # GeoJSON-like dict
    if isinstance(v, dict) and "type" in v and "coordinates" in v:
        try:
            gtype = str(v.get("type", "GEOMETRY")).upper()
            # Approx vertex count by walking coordinates
            def _count(c):
                if not isinstance(c, (list, tuple)):
                    return 0
                if c and isinstance(c[0], (int, float)):
                    return 1
                return sum(_count(x) for x in c)
            n_vertices = _count(v.get("coordinates"))
            return f"<{gtype}, {n_vertices} vertices, geojson>"
        except Exception:
            return "<GEOMETRY, geojson>"
    # WKT string
    if isinstance(v, str) and len(v) > 200 and _WKT_PREFIX.match(v):
        return _summarize_wkt(v)
    return v


def _summarize_geometries(head: dict) -> dict:
    """Walk a {column: [values]} dict and summarize any geometry values (WKT, Shapely, GeoJSON)."""
    out = {}
    for col, values in head.items():
        if not isinstance(values, list):
            out[col] = _summarize_value(values)
            continue
        out[col] = [_summarize_value(v) for v in values]
    return out


_SAMPLE_MAX_CHARS = 20_000


def _cap_sample_output(text: str) -> str:
    """Hard cap on sample_data output so no single call can blow up the LLM context."""
    if len(text) <= _SAMPLE_MAX_CHARS:
        return text
    return text[:_SAMPLE_MAX_CHARS] + f"\n... <truncated, output exceeded {_SAMPLE_MAX_CHARS} chars>"


def _build_research_design_tools(ctx):
    """Build tools that let the LLM explore datasets using the data registry (no disk I/O)."""

    registry = ctx.get("artifacts", {}).get("data_registry", [])

    @tool
    def explore_datasets() -> str:
        """List all available datasets with their column names, data types, and shape.
        Call this first to understand what data is available."""
        if not registry:
            return "No datasets in data registry."
        info = []
        for src in registry:
            info.append({
                "name": src.get("file", ""),
                "source": src.get("source", ""),
                "path": src.get("path", ""),
                "shape": src.get("shape", []),
                "columns": src.get("columns", {}),
            })
        return json.dumps(info, indent=2, default=str)

    @tool
    def sample_data(dataset_name: str) -> str:
        """Get sample rows (head) from a dataset. Use the exact dataset name from explore_datasets."""
        for src in registry:
            if src.get("file") == dataset_name:
                head = src.get("head", {})
                if not head:
                    return f"No sample rows available for '{dataset_name}'."
                return _cap_sample_output(json.dumps(_summarize_geometries(head), indent=2, default=str))
        available = [s.get("file") for s in registry]
        return f"Dataset '{dataset_name}' not found. Available: {available}"

    @tool
    def check_field_exists(field_name: str) -> str:
        """Check if a field name exists in any dataset. Returns which datasets contain it."""
        found = []
        all_cols = set()
        for src in registry:
            cols = src.get("columns", {})
            all_cols.update(cols.keys())
            if field_name in cols:
                found.append({"dataset": src.get("file"), "dtype": cols[field_name]})
        if found:
            return json.dumps({"field": field_name, "found_in": found}, indent=2)
        # Suggest close matches
        similar = [c for c in all_cols if field_name.lower() in c.lower() or c.lower() in field_name.lower()]
        return json.dumps({"field": field_name, "found_in": [], "similar_fields": similar[:10]}, indent=2)

    from utils.toolbase.dataset_profile import profile_tool
    return [explore_datasets, sample_data, check_field_exists, profile_tool(registry)]

