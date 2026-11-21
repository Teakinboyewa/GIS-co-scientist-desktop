---
id: data_merging
agent: data
keywords:
  - merge
  - join
  - left join
  - inner join
  - spatial join
  - sjoin
  - concat
  - combine
  - link
  - fips
  - geoid
  - countyfp
  - census tract
  - overlay
  - point in polygon
  - enrich
summary: >
  Merging datasets by key (pandas merge) or geometry (geopandas sjoin).
  Always cast join keys to string + zfill, audit match rate after merge,
  and treat Census sentinels (-666666666) as NaN.
---

# Data Merging (Join, Merge, Spatial Join)

## Description

Combine two tables by a shared key (`pd.merge`) or by geometry (`gpd.sjoin`).
The correctness of every downstream step depends on the **match rate** — a
silent join that produces NaN for 50% of rows will poison later modeling.

## When to use

- Linking attribute data (CSV) to polygons (GeoPackage/SHP) on FIPS/GEOID.
- Combining variables from two CSV sources on a shared key.
- Enriching points/polygons via spatial containment.

## When NOT to use

- Reshape only (pivot/melt) — not a merge.
- Concatenating rows of identical schema — use `pd.concat`.

## Critical constraints

1. **Always cast join keys to string and `zfill` to a consistent width**
   before merging. Int-vs-string and missing-leading-zero mismatches silently
   drop every match.
   - State FIPS: `zfill(2)` (e.g. `"06"`).
   - County FIPS: `zfill(3)` (e.g. `"001"`).
   - County GEOID (state+county): `zfill(5)` (e.g. `"06001"`).
   - Tract GEOID: `zfill(11)`.
2. **Strip whitespace** on both keys: `.astype(str).str.strip()`.
3. **Replace Census sentinels with NaN** on any ACS/Decennial column:
   `df.replace(-666666666, np.nan, inplace=True)`. Values like `-999999999`,
   `-888888888`, `-666666666` are Census "suppressed / not available" codes
   — not real numbers.
4. **Audit and print match rate** after every merge. A match rate below 90%
   is a red flag — either keys are mis-formatted or the right-side dataset
   doesn't cover all left-side units (report both hypotheses).
5. **Left join keeps every left row** — NaN on the right side does NOT raise
   an error. You must inspect the output for NaN, not just row count.
6. **For spatial joins, both GeoDataFrames must be in the same projected
   CRS.** Reproject before `sjoin`.

## Paths

- `INPUT_PATH_LEFT`, `INPUT_PATH_RIGHT` — from the step's Input Node Paths.
- `OUTPUT_NODE_PATH` — merged primary output.
- `OUTPUT_DIR` — directory for join-audit log.

Never hardcode `"data.csv"`, `"dataset_a.csv"`, etc.

## Canonical example (CSV → GeoPackage left join on GEOID, with audit)

```python
import os
import numpy as np
import pandas as pd
import geopandas as gpd

# ---- 1. Load ------------------------------------------------------------
left  = gpd.read_file(INPUT_PATH_LEFT)   # polygons (or a GeoDataFrame)
right = pd.read_csv(INPUT_PATH_RIGHT)    # attribute CSV

LEFT_KEY  = "<GEOID_COLUMN_ON_LEFT>"     # e.g. "GEOID"
RIGHT_KEY = "<GEOID_COLUMN_ON_RIGHT>"    # e.g. "GEOID"
KEY_WIDTH = 5                            # 2=state, 3=county, 5=county-GEOID, 11=tract

# ---- 2. Normalize join keys on BOTH sides -------------------------------
left[LEFT_KEY]   = left[LEFT_KEY].astype(str).str.strip().str.zfill(KEY_WIDTH)
right[RIGHT_KEY] = right[RIGHT_KEY].astype(str).str.strip().str.zfill(KEY_WIDTH)

# ---- 3. Clean Census sentinels on the right side ------------------------
for col in right.select_dtypes(include=["number", "object"]).columns:
    right[col] = pd.to_numeric(right[col], errors="ignore")
right = right.replace([-666666666, -888888888, -999999999], np.nan)

# ---- 4. Left join -------------------------------------------------------
merged = left.merge(right, left_on=LEFT_KEY, right_on=RIGHT_KEY, how="left")

# ---- 5. AUDIT the merge -------------------------------------------------
right_added = [c for c in right.columns if c != RIGHT_KEY and c in merged.columns]
audit_col   = right_added[0] if right_added else None

if audit_col is not None:
    matched      = int(merged[audit_col].notna().sum())
    total        = len(merged)
    match_rate   = matched / total if total else 0.0
    unmatched_n  = total - matched

    audit_lines = [
        f"[JOIN AUDIT] merged {total} left rows",
        f"  matched on '{audit_col}': {matched}/{total} ({match_rate:.1%})",
        f"  unmatched (NaN in right side): {unmatched_n}",
    ]
    if match_rate < 0.9:
        audit_lines.append(f"  WARNING: match rate below 90% — inspect keys:")
        audit_lines.append(f"    left  {LEFT_KEY} sample: {left[LEFT_KEY].head(5).tolist()}")
        audit_lines.append(f"    right {RIGHT_KEY} sample: {right[RIGHT_KEY].head(5).tolist()}")
        missing = merged.loc[merged[audit_col].isna(), LEFT_KEY].head(10).tolist()
        audit_lines.append(f"    example unmatched left keys: {missing}")
    audit = "\n".join(audit_lines)
    print(audit)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "join_audit.txt"), "w") as f:
        f.write(audit + "\n")

# ---- 6. Save primary output --------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
if isinstance(merged, gpd.GeoDataFrame):
    merged.to_file(OUTPUT_NODE_PATH, driver="GPKG")
else:
    merged.to_csv(OUTPUT_NODE_PATH, index=False)
```

## Spatial join variant (points in polygons)

```python
import geopandas as gpd

points   = gpd.read_file(INPUT_PATH_LEFT)
polygons = gpd.read_file(INPUT_PATH_RIGHT)

# Reproject to same CRS (use polygons' CRS by convention)
if points.crs != polygons.crs:
    points = points.to_crs(polygons.crs)

joined = gpd.sjoin(points, polygons, how="left", predicate="within")

# Audit — how many points landed in a polygon?
unmatched = joined["index_right"].isna().sum()
print(f"[SJOIN AUDIT] {len(joined)-unmatched}/{len(joined)} points matched a polygon "
      f"({unmatched} outside all polygons)")
```

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Int vs string FIPS silently produces 0 matches | Cast both sides to `str.strip().str.zfill(W)` with the right width. |
| Leading zeros stripped by `pd.read_csv` | Either `dtype={'GEOID': str}` at read time, or zfill after load. |
| ACS sentinel `-666666666` treated as a real value | `df.replace([-666666666, -888888888, -999999999], np.nan)`. |
| "58 records created" looks fine but half are NaN | Always print match rate, not just row count. |
| `sjoin` returns 0 rows | Reproject to a common CRS first; verify with `points.within(polygons.unary_union).sum()`. |
| `how="inner"` after a previously-failed join | Use `how="left"` for auditing; switch to inner only after you've verified the match rate. |
| Duplicated keys on the right side explode left row count | `right.drop_duplicates(subset=[RIGHT_KEY])` before merge if right is supposed to be unique. |
| Tract vs county GEOID width mismatch | Decide which level the join is at; enforce the matching width on both sides. |
