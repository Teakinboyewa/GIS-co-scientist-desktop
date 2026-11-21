"""
Load an output table for summarizing to an LLM, without misreporting its size.

The result-analysis and manuscript stages used to read the first 100 or 200
rows of each output file and pass that frame's shape and describe() on as if
they were the whole file. A 900-block analysis was written up as "200
blocks", with descriptive statistics from those first 200 rows.

Here the row count is always the file's true count, statistics come from the
whole file up to STATS_MAX_ROWS, and anything above that cap is labelled as a
sample in the summary itself.
"""

import os

# Statistics are computed on at most this many rows. Far above any areal
# study's unit count; exists only to bound memory on very large outputs.
STATS_MAX_ROWS = 200_000

SPATIAL_EXTS = (".gpkg", ".shp", ".geojson")


def _csv_total_rows(path, loaded, cap):
    if loaded < cap:
        return loaded
    import pandas as pd
    return sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=100_000))


def _spatial_total_rows(path, loaded, cap):
    if loaded < cap:
        return loaded
    try:
        import pyogrio
        n = pyogrio.read_info(path).get("features", -1)
        if n is not None and n >= 0:
            return int(n)
    except Exception:
        pass
    return None      # unknown: the summary says so rather than guessing


def load_table(path, max_rows=STATS_MAX_ROWS):
    """Return (frame, total_rows, sampled). frame is a DataFrame or
    GeoDataFrame of at most max_rows rows; total_rows is the file's true row
    count (None if it cannot be determined); sampled is True when frame holds
    fewer rows than the file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd
        frame = pd.read_csv(path, nrows=max_rows)
        total = _csv_total_rows(path, len(frame), max_rows)
    elif ext in SPATIAL_EXTS:
        import geopandas as gpd
        frame = gpd.read_file(path, rows=max_rows)
        total = _spatial_total_rows(path, len(frame), max_rows)
    else:
        raise ValueError(f"Not a table: {path}")
    sampled = total is None or total > len(frame)
    return frame, total, sampled


def size_fields(frame, total, sampled):
    """Shape and provenance fields for a summary dict."""
    fields = {"shape": [total if total is not None else f">={len(frame)}", frame.shape[1]],
              "rows_in_file": total}
    if sampled:
        fields["statistics_note"] = (
            f"Statistics below are computed on the first {len(frame)} rows only; "
            f"the file has {total if total is not None else 'more'} rows. "
            "Do not report them as whole-dataset statistics.")
    return fields
