---
id: ols_regression
agent: modeling
keywords:
  - ols
  - ordinary least squares
  - linear regression
  - multiple regression
  - statsmodels
  - vif
  - multicollinearity check
  - regression diagnostics
summary: >
  Ordinary Least Squares regression via statsmodels — fit, coefficients with
  p-values, R² / adj-R² / AIC, VIF multicollinearity check, residual
  diagnostics, and handoff to spatial regression when residuals cluster.
---

# OLS Regression

## Description

Classical multiple linear regression: `y = β₀ + β₁x₁ + … + βₖxₖ + ε`. Used as the **baseline** before any spatial or local model — always fit OLS first, then decide whether residual diagnostics force you to escalate (GWR, SAR, SEM).

## When to use

- Baseline model before spatial regression / GWR.
- Global, stationary relationships with continuous `y`.
- You need interpretable coefficients with p-values, SEs, R², AIC.

## When NOT to use

- Residuals are spatially autocorrelated → escalate to Spatial Regression (SAR/SEM) or GWR.
- `y` is binary / count → use logistic or Poisson regression.
- Predictors strongly collinear (VIF > 10) → drop, combine, or regularize (Ridge/Lasso).

## Critical constraints

1. **Cast to numeric and drop NAs jointly** across `y` and all `X` — row masks must align.
2. **Add a constant** — statsmodels does **not** add one automatically: `sm.add_constant(X)`.
3. **Check VIF** before trusting individual coefficients. VIF > 10 is a red flag.
4. **Inspect residuals spatially** if the data has geometry — OLS assumes `ε` is independent.
5. **Report n, R², adj-R², AIC, coefficient table** — not just R² in isolation.

## Paths

The canonical example uses three variables the modeling agent substitutes
from its own INPUT / OUTPUT PATHS section. **Never hardcode `"data.csv"`.**

- `INPUT_PATH` — read from the step's **Input Node Paths**. Call
  `list_datasets()` and `get_columns()` first to confirm the file and columns.
- `OUTPUT_NODE_PATH` — the step's **primary** output path (per-step unique).
- `OUTPUT_DIR` — pre-injected directory for **secondary** artifacts
  (coefficient tables, VIF, diagnostics, plots). Use
  `os.path.join(OUTPUT_DIR, "…")`.

## Canonical example (fit → diagnostics → VIF → spatial residual check)

```python
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# INPUT_PATH / OUTPUT_NODE_PATH substituted from the step's I/O section.
# OUTPUT_DIR is pre-injected by the framework.
df = pd.read_csv(INPUT_PATH)

y_col  = "<DEPENDENT_VARIABLE>"
x_cols = ["<PREDICTOR_1>", "<PREDICTOR_2>", "<PREDICTOR_3>"]

# ---- 1. Clean ------------------------------------------------------------
cols = [y_col] + x_cols
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols).reset_index(drop=True)

y = df[y_col].values
X = df[x_cols].values
X = sm.add_constant(X)          # prepends the intercept column

# ---- 2. Fit --------------------------------------------------------------
model = sm.OLS(y, X).fit()
print(model.summary())

# ---- 3. Coefficient table ------------------------------------------------
coef_df = pd.DataFrame({
    "variable": ["const"] + x_cols,
    "beta":     model.params,
    "std_err":  model.bse,
    "t":        model.tvalues,
    "p":        model.pvalues,
    "ci_low":   model.conf_int()[:, 0],
    "ci_high":  model.conf_int()[:, 1],
})
os.makedirs(OUTPUT_DIR, exist_ok=True)
coef_df.to_csv(os.path.join(OUTPUT_DIR, "ols_coefficients.csv"), index=False)

# ---- 4. Fit statistics ---------------------------------------------------
stats = pd.DataFrame([{
    "n":        int(model.nobs),
    "k":        len(x_cols),
    "R2":       model.rsquared,
    "adj_R2":   model.rsquared_adj,
    "F":        model.fvalue,
    "F_pvalue": model.f_pvalue,
    "AIC":      model.aic,
    "BIC":      model.bic,
    "RMSE":     float(np.sqrt(np.mean(model.resid ** 2))),
}])
stats.to_csv(os.path.join(OUTPUT_DIR, "ols_fit_stats.csv"), index=False)
print(stats.to_string(index=False))

# ---- 5. VIF (multicollinearity) -----------------------------------------
vif_df = pd.DataFrame({
    "variable": x_cols,
    "VIF": [variance_inflation_factor(X, i + 1) for i in range(len(x_cols))],
})
vif_df.to_csv(os.path.join(OUTPUT_DIR, "ols_vif.csv"), index=False)
print(vif_df.to_string(index=False))
if (vif_df["VIF"] > 10).any():
    print("WARNING: VIF > 10 — drop or combine collinear predictors.")

# ---- 6. Residuals for downstream spatial diagnostics --------------------
df["ols_fitted"]   = model.fittedvalues
df["ols_residual"] = model.resid
# Primary output → OUTPUT_NODE_PATH (fitted + residuals merged back to input rows)
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
df.to_csv(OUTPUT_NODE_PATH, index=False)
```

## Residual diagnostics → when to escalate

```python
# If the data is spatial, attach residuals to the GeoDataFrame and run Moran's I
# (see global_morans_i_skill). Significant residual autocorrelation means:
#   - OLS coefficients may still be unbiased, but SEs are wrong
#   - Move to SAR / SEM (see spatial_regression_skill)
#     or GWR (see gwr_skill) if effects are non-stationary
```

Non-spatial residual checks worth running:

- **Normality of residuals** — `scipy.stats.shapiro(model.resid)` or a QQ-plot.
- **Heteroscedasticity** — `statsmodels.stats.diagnostic.het_breuschpagan`.
- **Linearity** — plot residuals vs fitted; look for curvature.

## Key result attributes

| Attribute | Meaning |
|---|---|
| `model.params`, `.bse`, `.tvalues`, `.pvalues` | Coefficients, SE, t, p |
| `model.conf_int()` | 95% confidence intervals |
| `model.rsquared`, `.rsquared_adj` | R² and adjusted R² |
| `model.fvalue`, `.f_pvalue` | Overall F-test |
| `model.aic`, `.bic` | Information criteria |
| `model.fittedvalues`, `.resid` | `ŷ` and residuals |
| `model.nobs` | Effective sample size |

## Interpretation

- Report for each predictor: **β, SE, t, p, 95% CI**.
- `adj_R²` penalizes extra predictors — use it to compare models with different `k`.
- `F_pvalue < 0.05`: at least one predictor is useful. Doesn't mean all are.
- A coefficient's sign and magnitude are only trustworthy if VIF is acceptable **and** residuals are well-behaved.

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Forgot `sm.add_constant(X)` | No intercept → biased fit. Always add it for statsmodels. |
| NaNs silently drop rows differently across columns | Build one mask across `y + X`, then `dropna(subset=cols)`. |
| High R², high VIF | Model fits the training data but coefficients are unstable. Drop or combine collinear predictors. |
| Reporting R² alone | Always pair with adj-R², n, and the coefficient table. |
| Ignoring residual autocorrelation on geographic data | Run Moran's I on residuals; escalate to SAR/SEM or GWR if significant. |
| Using sklearn `LinearRegression` when you want inference | sklearn gives no SEs / p-values. Use statsmodels for inference, sklearn for prediction pipelines. |
