---
id: gwr
agent: modeling
keywords:
  - gwr
  - mgwr
  - geographically weighted regression
  - multiscale gwr
  - spatially varying coefficients
  - local regression
  - sel_bw
summary: >
  GWR & MGWR: projected CRS, coordinate arrays, bandwidth selection (Sel_BW),
  fitting, coefficient back-transformation, mapping, diagnostics.
---

# GWR (Geographically Weighted Regression)

## Description

Local spatial regression that fits a separate weighted regression at every
observation, producing coefficients that vary across geographic space.
Answers *where* relationships are stronger/weaker, not just the global mean.

## When to use

- You suspect spatial non-stationarity in the relationship between `y` and `X`.
- A global OLS leaves significant Moran's I in the residuals.
- The research question is explicitly about spatial variation in effects.

## When NOT to use

- Relationships are global/stationary — OLS suffices.
- `n < 50` observations, or `k` predictors is large relative to `n`.
- Data has no meaningful spatial structure.

## Critical constraints

These break GWR silently or loudly — get them right:

1. **Projected CRS only.** `coords` must be in meters. Reproject if `gdf.crs.is_geographic`.
2. **`y` must be 2-D, shape `(n, 1)`.** Use `gdf[[col]].values`, not `gdf[col].values`.
3. **`X` adds no intercept.** `mgwr` adds it internally; do not prepend a ones column.
4. **Import `Sel_BW` from `mgwr.sel_bw`**, not `mgwr.gwr`.
5. **Set `spherical=False`** with projected coords.
6. **Drop NA/inf in `y`, `X`, and `coords` together** — row masks must align.
7. **Standardize when comparing coefficients across predictors** — improves numerics and interpretability.

## Paths

The canonical example uses three variables the modeling agent substitutes
from its own INPUT / OUTPUT PATHS section. **Never hardcode filenames.**

- `INPUT_PATH` — from the step's **Input Node Paths** (confirm with
  `list_datasets()` / `get_columns()` first).
- `OUTPUT_NODE_PATH` — the step's **primary** output path
  (e.g. `gwr_results.gpkg`) — per-step unique.
- `OUTPUT_DIR` — pre-injected directory for **secondary** artifacts
  (coefficient maps, text summary). Use `os.path.join(OUTPUT_DIR, "…")`.

## Canonical example (fit → extract → save → map)

```python
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# INPUT_PATH / OUTPUT_NODE_PATH substituted from the step's I/O section.
# OUTPUT_DIR is pre-injected by the framework.
OUTPUT_MAPS    = os.path.join(OUTPUT_DIR, "gwr_coefficient_maps.png")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "gwr_summary.txt")

# ---- 1. Load & project ----------------------------------------------------
gdf = gpd.read_file(INPUT_PATH)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326, allow_override=True)
if gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=3857)   # prefer a local projected CRS when available
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

# ---- 2. Variables ---------------------------------------------------------
y_col  = "<DEPENDENT_VARIABLE>"
x_cols = ["<PREDICTOR_1>", "<PREDICTOR_2>", "<PREDICTOR_3>"]

centroids = gdf.geometry.centroid
coords    = np.column_stack([centroids.x.values, centroids.y.values]).astype(float)
y         = gdf[[y_col]].values.astype(float)          # (n, 1) — MUST be 2-D
X_raw     = gdf[x_cols].values.astype(float)           # (n, k) — no intercept

# ---- 3. Clean aligned rows -----------------------------------------------
mask = np.isfinite(np.hstack([y, X_raw, coords])).all(axis=1)
gdf_m, y, X_raw, coords = gdf.loc[mask].copy(), y[mask], X_raw[mask], coords[mask]

# ---- Guard against empty / tiny frames (usually caused by upstream join NaN) ----
if len(gdf_m) == 0:
    raise ValueError(f"No rows after dropping NA in {[y_col] + x_cols}. Check upstream join match rates.")
if len(gdf_m) < 20:
    print(f"[WARN] Only {len(gdf_m)} rows — GWR bandwidth selection will be unreliable.")

# ---- 4. Standardize (recommended) ----------------------------------------
x_scaler, y_scaler = StandardScaler(), StandardScaler()
X_s = x_scaler.fit_transform(X_raw)
y_s = y_scaler.fit_transform(y)

# ---- 5. Bandwidth selection (adaptive, AICc) -----------------------------
selector = Sel_BW(coords, y_s, X_s, fixed=False, kernel="bisquare", spherical=False)
bw = selector.search(criterion="AICc")
print(f"Selected adaptive bandwidth: {bw} neighbors")

# ---- 6. Fit --------------------------------------------------------------
model   = GWR(coords, y_s, X_s, bw=bw, fixed=False, kernel="bisquare", spherical=False)
results = model.fit()

# ---- 7. Back-transform coefficients to original units --------------------
param_names = ["Intercept"] + x_cols
params, bse, tvals, localR2 = results.params, results.bse, results.tvalues, results.localR2
y_mean, y_std = float(y_scaler.mean_[0]), float(np.sqrt(y_scaler.var_[0]))
x_means, x_stds = x_scaler.mean_, np.sqrt(x_scaler.var_)

betas = np.zeros_like(params)
for j in range(1, len(param_names)):
    betas[:, j] = (y_std / x_stds[j - 1]) * params[:, j]
betas[:, 0] = y_mean + y_std * params[:, 0] - (betas[:, 1:] * x_means.reshape(1, -1)).sum(axis=1)

# ---- 8. Write GeoPackage -------------------------------------------------
out = gdf_m.copy()
for j, name in enumerate(param_names):
    out[f"b_{name}"]  = betas[:, j]
    out[f"se_{name}"] = bse[:, j]
    out[f"t_{name}"]  = tvals[:, j]
out["gwr_localR2"] = np.asarray(localR2).flatten()
out["gwr_pred"]    = (results.predy * y_std + y_mean).flatten()
out["gwr_resid"]   = (y - out["gwr_pred"].values.reshape(-1, 1)).flatten()

os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
if os.path.exists(OUTPUT_NODE_PATH):
    os.remove(OUTPUT_NODE_PATH)
out.to_file(OUTPUT_NODE_PATH, driver="GPKG")

# ---- 9. Summary ----------------------------------------------------------
with open(OUTPUT_SUMMARY, "w") as f:
    f.write("GWR Model Summary\n")
    f.write(f"Observations: {out.shape[0]}\n")
    f.write(f"Dependent: {y_col}\n")
    f.write(f"Predictors: {', '.join(x_cols)}\n")
    f.write(f"Kernel: bisquare (adaptive)\n")
    f.write(f"Bandwidth: {bw} neighbors\n")
    f.write(f"AICc: {results.aicc:.4f}\n")
    f.write(f"R²: {results.R2:.4f}\n")
    f.write(f"Adj R²: {results.adj_R2:.4f}\n")

# ---- 10. Coefficient maps ------------------------------------------------
map_vars = [f"b_{c}" for c in x_cols] + ["gwr_localR2"]
ncols = min(3, len(map_vars))
nrows = (len(map_vars) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten() if len(map_vars) > 1 else [axes]
for ax, var in zip(axes, map_vars):
    out.plot(column=var, ax=ax, legend=True, cmap="RdBu_r", legend_kwds={"shrink": 0.6})
    ax.set_title(var, fontsize=10); ax.axis("off")
for ax in axes[len(map_vars):]:
    ax.set_visible(False)
plt.suptitle("GWR Local Coefficients", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_MAPS, dpi=150)
plt.close()
```

## MGWR variant

MGWR lets each predictor operate at its own spatial scale (one bandwidth per variable). Swap steps 5–6 above for:

```python
from mgwr.gwr import MGWR
selector = Sel_BW(coords, y_s, X_s, multi=True, kernel="bisquare", fixed=False, spherical=False)
bws = selector.search(criterion="AICc")
model   = MGWR(coords, y_s, X_s, selector, kernel="bisquare", fixed=False, spherical=False)
results = model.fit()
```

## Key results attributes

| Attribute | Shape | Meaning |
|---|---|---|
| `results.params` | `(n, k+1)` | Local coefficients (intercept + predictors) |
| `results.bse` | `(n, k+1)` | Local standard errors |
| `results.tvalues` | `(n, k+1)` | Local t-statistics |
| `results.localR2` | `(n, 1)` | Local R² |
| `results.predy` | `(n, 1)` | Predictions (in scaled space if `y_s` was used) |
| `results.R2`, `.adj_R2`, `.aicc`, `.aic`, `.bic` | scalar | Global diagnostics |

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Geographic CRS (lat/lon) | Reproject to meters before `coords = ...`. |
| `y` is 1-D | Use `gdf[[col]].values`, not `gdf[col].values`. |
| `Sel_BW` import error | `from mgwr.sel_bw import Sel_BW` (not `mgwr.gwr`). |
| Collinear predictors → unstable fit | Drop perfect collinearity; standardize. |
| NaN/inf in arrays | Build a single mask from `np.hstack([y, X, coords])`. |
| Bandwidth = n (all neighbors) | Likely no non-stationarity; compare AICc to global OLS. |
| Coefficients in scaled units | Back-transform (see step 7). |
| `spherical=True` with projected data | Set `spherical=False`. |
