---
id: global_morans_i
agent: modeling
keywords:
  - moran
  - global moran
  - morans i
  - spatial autocorrelation
  - spatial clustering test
  - residual spatial autocorrelation
summary: >
  Global Moran's I — test whether a variable (or regression residual) is
  spatially clustered, dispersed, or random across the entire study area.
---

# Global Moran's I

## Description

Global Moran's I is a single-number test: **is there overall spatial autocorrelation in this variable?** Positive I means similar values cluster together; negative I means neighbors differ; I near E[I] means random.

## When to use

- Screening a variable for spatial clustering before further spatial analysis.
- Checking **regression residuals** — significant Moran's I on residuals justifies moving from OLS to spatial regression or GWR.
- Comparing clustering strength across multiple variables or time periods.

## When NOT to use

- Locating **where** clusters are → use Local Moran's I (LISA).
- Testing two-variable relationships → bivariate correlation skill.
- Variable with no spatial geometry attached.

## Critical constraints

1. **Build weights on the cleaned GeoDataFrame** — drop NAs first, then `Queen.from_dataframe(gdf)`.
2. **Row-standardize** — `w.transform = "r"`.
3. **Use permutations for inference** — `permutations=999` gives stable p-values; don't rely on the analytical z-score alone.
4. **Prefer KNN if the Queen graph has islands** — see the Spatial Weights skill.

## Paths

- `INPUT_PATH` — GeoPackage/SHP from Input Node Paths.
- `OUTPUT_NODE_PATH` — CSV with Moran's I statistics (primary result).
- `OUTPUT_DIR` — for any secondary artifacts if added.

Never hardcode `"data.gpkg"`.

## Canonical example (variable + residuals)

```python
import os
import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen, KNN
from esda.moran import Moran

# INPUT_PATH / OUTPUT_NODE_PATH from step I/O section.
gdf = gpd.read_file(INPUT_PATH)

col = "<TARGET_COLUMN>"          # or a residual column like "gwr_resid"
gdf = gdf.dropna(subset=[col]).reset_index(drop=True)

# ---- Guard against empty / tiny frames (usually caused by upstream join NaN) ----
if len(gdf) == 0:
    raise ValueError(f"No rows after dropping NA in '{col}'. Check upstream join match rates.")
if len(gdf) < 10:
    print(f"[WARN] Only {len(gdf)} rows — spatial statistics will be unreliable.")

# ---- Weights ------------------------------------------------------------
w = Queen.from_dataframe(gdf)
if w.islands:
    w = KNN.from_dataframe(gdf, k=5)
w.transform = "r"

# ---- Global Moran's I ---------------------------------------------------
mi = Moran(gdf[col].values, w, permutations=999)

print(f"Global Moran's I on '{col}':")
print(f"  I          = {mi.I:.4f}")
print(f"  E[I]       = {mi.EI:.4f}")
print(f"  z (sim)    = {mi.z_sim:.4f}")
print(f"  p (sim)    = {mi.p_sim:.6f}")
print("  =>", "significant spatial autocorrelation" if mi.p_sim < 0.05 else "no significant autocorrelation")

# ---- Save ---------------------------------------------------------------
results = pd.DataFrame([{
    "variable": col,
    "morans_I": mi.I,
    "expected_I": mi.EI,
    "variance_sim": mi.VI_sim,
    "z_sim": mi.z_sim,
    "p_sim": mi.p_sim,
    "n": len(gdf),
}])
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
results.to_csv(OUTPUT_NODE_PATH, index=False)
```

## Interpretation

| Condition | Meaning |
|---|---|
| `I > E[I]` and `p_sim < 0.05` | Positive spatial autocorrelation — clustering of similar values |
| `I ≈ E[I]` and `p_sim > 0.05` | Spatially random |
| `I < E[I]` and `p_sim < 0.05` | Negative spatial autocorrelation — checkerboard/dispersion |

`E[I] = -1/(n-1)` — essentially 0 for large n.

**Residual diagnostic:** if Moran's I on OLS residuals is significant, OLS is mis-specified spatially → try spatial lag/error regression or GWR.

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Weights built on original gdf, analysis on subset | Drop NAs first, reset index, then build weights. |
| Forgot row-standardization | `w.transform = "r"`. |
| Islands in Queen weights | Swap in `KNN.from_dataframe(gdf, k=5)`. |
| Reporting only `mi.p_norm` | Prefer `mi.p_sim` (permutation-based) — more robust. |
| Interpreting as "where" | Global I tells you *whether*, not *where*. Follow up with LISA. |
