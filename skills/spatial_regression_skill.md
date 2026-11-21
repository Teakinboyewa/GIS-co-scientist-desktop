---
id: spatial_regression
agent: modeling
keywords:
  - spatial lag
  - spatial error
  - sar
  - sem
  - ml_lag
  - ml_error
  - spatial regression
  - lagrange multiplier
  - lm test
  - spreg
summary: >
  Global spatial regression — Spatial Lag (SAR) and Spatial Error (SEM)
  models via spreg. Includes OLS baseline, LM diagnostics for model choice,
  ML estimation, and residual Moran's I check.
---

# Spatial Regression (Lag & Error)

## Description

Global regression models that correct for spatial autocorrelation.

- **Spatial Lag (SAR)** — `y = ρWy + Xβ + ε`. Use when the outcome itself *diffuses* across neighbors (crime, prices, disease counts).
- **Spatial Error (SEM)** — `y = Xβ + u, u = λWu + ε`. Use when autocorrelation is a *nuisance* from omitted spatially-structured variables.

## When to use

- OLS residuals show significant Moran's I → global spatial regression is warranted.
- You want **one global** coefficient per predictor, not per-location like GWR.
- Sample size is too small, or relationships too stationary, for GWR.

## When NOT to use

- Coefficients vary across space → use GWR / MGWR.
- No significant residual autocorrelation → plain OLS is fine.
- Very small `n` (< ~50) — ML estimation becomes unstable.

## Choosing Lag vs Error — LM tests

Run OLS first, then Lagrange Multiplier (LM) tests on residuals:

| LM-Lag sig? | LM-Error sig? | Choose |
|---|---|---|
| Yes | No  | Spatial **Lag** |
| No  | Yes | Spatial **Error** |
| Both sig | — | Compare **robust** LM-Lag vs robust LM-Error; pick the more significant |
| Neither | — | Stick with OLS |

## Critical constraints

1. **Build weights on the cleaned frame** — drop NAs, reset index, then `Queen.from_dataframe`.
2. **Row-standardize** — `w.transform = "r"`.
3. **`y` must be 2-D `(n, 1)`** — `gdf[[y_col]].values`, not `gdf[y_col].values`.
4. **No intercept column in `X`** — `spreg` adds it.
5. **Check residual Moran's I after fitting** — if still significant, model is under-specified.
6. **Report ρ (lag) or λ (error)** — these are *the* parameters that justify using the model.

## Paths

Use the three conventional variables from the step's INPUT / OUTPUT PATHS
section — **never hardcode filenames.**

- `INPUT_PATH` — input GeoPackage / CSV from Input Node Paths.
- `OUTPUT_NODE_PATH` — primary GPKG/CSV with fitted + residuals.
- `OUTPUT_DIR` — directory for the coefficient CSV and any diagnostics.

## Canonical example (OLS → LM tests → SAR or SEM → residual check)

```python
import os
import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen, KNN
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error

# INPUT_PATH / OUTPUT_NODE_PATH from the step's I/O section; OUTPUT_DIR pre-injected.
gdf = gpd.read_file(INPUT_PATH)

y_col  = "<DEPENDENT_VARIABLE>"
x_cols = ["<PREDICTOR_1>", "<PREDICTOR_2>", "<PREDICTOR_3>"]

# ---- 1. Clean & align ---------------------------------------------------
cols = [y_col] + x_cols
for c in cols:
    gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
gdf = gdf.dropna(subset=cols).reset_index(drop=True)

# ---- Guard against empty / tiny frames (usually caused by upstream join NaN) ----
if len(gdf) == 0:
    raise ValueError(f"No rows after dropping NA in {cols}. Check upstream join match rates.")
if len(gdf) < 10:
    print(f"[WARN] Only {len(gdf)} rows — spatial regression will be unreliable.")

y = gdf[[y_col]].values.astype(float)          # (n, 1)
X = gdf[x_cols].values.astype(float)           # (n, k), no intercept

# ---- 2. Weights ---------------------------------------------------------
w = Queen.from_dataframe(gdf)
if w.islands:
    w = KNN.from_dataframe(gdf, k=5)
w.transform = "r"

# ---- 3. OLS baseline with LM diagnostics --------------------------------
ols = OLS(y, X, w=w, spat_diag=True, moran=True, name_y=y_col, name_x=x_cols)
print(ols.summary)

lm_lag        = ols.lm_lag           # (stat, p)
lm_error      = ols.lm_error
rlm_lag       = ols.rlm_lag          # robust variants
rlm_error     = ols.rlm_error
print(f"LM-Lag   : stat={lm_lag[0]:.3f}, p={lm_lag[1]:.4f}")
print(f"LM-Error : stat={lm_error[0]:.3f}, p={lm_error[1]:.4f}")
print(f"Robust LM-Lag   : stat={rlm_lag[0]:.3f}, p={rlm_lag[1]:.4f}")
print(f"Robust LM-Error : stat={rlm_error[0]:.3f}, p={rlm_error[1]:.4f}")

# ---- 4. Pick model by LM rule ------------------------------------------
def pick_model(lm_lag, lm_error, rlm_lag, rlm_error, alpha=0.05):
    lag_sig, err_sig = lm_lag[1] < alpha, lm_error[1] < alpha
    if lag_sig and not err_sig: return "lag"
    if err_sig and not lag_sig: return "error"
    if lag_sig and err_sig:
        return "lag" if rlm_lag[1] < rlm_error[1] else "error"
    return "ols"

choice = pick_model(lm_lag, lm_error, rlm_lag, rlm_error)
print(f"LM decision: {choice.upper()}")

# ---- 5. Fit the chosen model -------------------------------------------
if choice == "lag":
    model = ML_Lag(y, X, w=w, name_y=y_col, name_x=x_cols)
    spatial_param = ("rho (ρ)", float(model.rho))
elif choice == "error":
    model = ML_Error(y, X, w=w, name_y=y_col, name_x=x_cols)
    spatial_param = ("lambda (λ)", float(model.lam))
else:
    model = ols
    spatial_param = ("n/a", np.nan)

print(model.summary)
print(f"Spatial parameter {spatial_param[0]} = {spatial_param[1]:.4f}")

# ---- 6. Residual Moran's I (did the model absorb the autocorrelation?) -
resid = np.asarray(model.u).flatten()
mi_resid = Moran(resid, w, permutations=999)
print(f"Residual Moran's I = {mi_resid.I:.4f}, p = {mi_resid.p_sim:.4f}")

# ---- 7. Save fitted values, residuals, coefficients --------------------
out = gdf.copy()
out["fitted"]   = np.asarray(model.predy).flatten()
out["residual"] = resid
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
out.to_file(OUTPUT_NODE_PATH, driver="GPKG")

coef_df = pd.DataFrame({
    "variable": ["CONSTANT"] + x_cols,
    "beta":     np.asarray(model.betas).flatten()[: 1 + len(x_cols)],
    "std_err":  np.asarray(model.std_err).flatten()[: 1 + len(x_cols)],
    "z":        np.asarray([t[0] for t in model.z_stat])[: 1 + len(x_cols)],
    "p":        np.asarray([t[1] for t in model.z_stat])[: 1 + len(x_cols)],
})
os.makedirs(OUTPUT_DIR, exist_ok=True)
coef_df.to_csv(os.path.join(OUTPUT_DIR, "spatial_regression_coefficients.csv"), index=False)
```

## Key result attributes

| Model | Attribute | Meaning |
|---|---|---|
| OLS | `ols.lm_lag`, `ols.lm_error` | (statistic, p-value) of LM tests |
| OLS | `ols.rlm_lag`, `ols.rlm_error` | Robust LM variants |
| SAR | `model.rho` | Spatial autoregressive parameter ρ |
| SEM | `model.lam` | Spatial error parameter λ |
| both | `model.betas`, `model.std_err`, `model.z_stat` | Coefficients, SE, (z, p) |
| both | `model.predy`, `model.u` | Fitted values, residuals |
| both | `model.pr2`, `model.aic`, `model.schwarz` | Pseudo-R², AIC, BIC |

## Interpretation

- **ρ > 0 (SAR):** high `y` at a location raises `y` at its neighbors — positive spillover.
- **λ > 0 (SEM):** unobserved shocks are spatially correlated; coefficients are still unbiased, but OLS SEs were wrong.
- **Coefficients in SAR are *not* marginal effects.** Use `spreg.diagnostics_sp` or compute direct / indirect / total impacts for full interpretation.
- After fitting, residual Moran's I should be **non-significant**. If it isn't, try MGWR, a mixed SAR-SEM (SARAR), or revisit predictors.

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Added intercept column to `X` | `spreg` adds it — don't prepend ones. |
| `y` is 1-D | `gdf[[col]].values`, not `gdf[col].values`. |
| Weights built on raw gdf before dropna | Drop NAs, reset index, **then** build weights. |
| Interpreting SAR β as OLS β | In SAR, effects propagate through `W`; compute impacts for marginal effects. |
| Using LM results without checking robust LM when both are significant | Always fall back to the robust LM pair when the standard pair is jointly significant. |
| Residual Moran's I still significant | Model under-specified — consider SARAR, GWR, or additional spatial predictors. |
