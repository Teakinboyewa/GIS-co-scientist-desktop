---
id: bivariate_correlation
agent: modeling
keywords:
  - pearson
  - spearman
  - kendall
  - point-biserial
  - bivariate correlation
  - scatter
  - correlate two variables
summary: >
  Bivariate correlation between two variables: Pearson / Spearman / Kendall,
  scatter plot with trend line, cross-dataset merge-and-correlate.
---

# Bivariate Correlation

## Description

Strength and direction of the association between **two** variables. Pearson for linear, Spearman/Kendall for monotonic/ordinal, point-biserial for continuous-vs-binary.

## When to use

- Testing one hypothesized pair of variables.
- Ranking a handful of predictors against a target.
- Correlating variables from two datasets after a key-based merge.

## When NOT to use

- Many pairwise pairs at once → use the Correlation Matrix skill.
- Spatial clustering of a single variable → use Moran's I.
- Non-monotonic relationships — correlation won't capture them.

## Choosing the method

| Method | Use when | Function |
|---|---|---|
| Pearson `r` | Both continuous and roughly linear | `scipy.stats.pearsonr` |
| Spearman `ρ` | Monotonic but non-linear, ordinal, or outlier-heavy | `scipy.stats.spearmanr` |
| Kendall `τ` | Small `n`, many ties | `scipy.stats.kendalltau` |
| Point-biserial | One variable is binary (0/1) | `scipy.stats.pointbiserialr` |

## Critical constraints

1. **Cast to numeric first** — `pd.to_numeric(col, errors="coerce")`.
2. **Drop NAs pairwise** — both columns must be non-null on the same rows.
3. **Always report `n`** alongside `r` and `p`.
4. **Zero-pad FIPS/ID keys as strings** before merging across datasets.

## Paths

- `INPUT_PATH` — from Input Node Paths. For cross-dataset merges, the step
  typically lists multiple inputs — read each by its listed path.
- `OUTPUT_NODE_PATH` — primary CSV with correlation statistics.
- `OUTPUT_DIR` — for the scatter PNG.

Never hardcode `"data.csv"`, `"dataset_a.csv"`, etc.

## Canonical example (Pearson + Spearman + Kendall + scatter)

```python
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

df = pd.read_csv(INPUT_PATH)

x_col = "<X_COLUMN>"
y_col = "<Y_COLUMN>"

# ---- 1. Cast + align -----------------------------------------------------
x = pd.to_numeric(df[x_col], errors="coerce")
y = pd.to_numeric(df[y_col], errors="coerce")
mask = x.notna() & y.notna()
x, y = x[mask], y[mask]
n = len(x)

# ---- 2. Three coefficients -----------------------------------------------
pr, pp = pearsonr(x, y)
sr, sp = spearmanr(x, y)
kr, kp = kendalltau(x, y)

results = pd.DataFrame([{
    "x": x_col, "y": y_col, "n": n,
    "pearson_r": round(pr, 4),  "pearson_p": round(pp, 6),
    "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
    "kendall_tau": round(kr, 4), "kendall_p": round(kp, 6),
}])
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
results.to_csv(OUTPUT_NODE_PATH, index=False)
print(results.to_string(index=False))

# ---- 3. Scatter with trend line ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.4, s=10, edgecolors="none")
z = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, np.poly1d(z)(xs), "r--", linewidth=1.5)
ax.set_xlabel(x_col); ax.set_ylabel(y_col)
ax.set_title(f"{x_col} vs {y_col}  (Pearson r = {pr:.3f}, n = {n})")
fig.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_DIR, "bivariate_scatter.png"), dpi=150)
plt.close(fig)
```

## Cross-dataset variant (merge, then correlate)

```python
import pandas as pd
from scipy.stats import pearsonr

# When the step has two inputs, they'll be listed separately in Input Node Paths.
df_a = pd.read_csv(INPUT_PATH_A)
df_b = pd.read_csv(INPUT_PATH_B)

key_a, key_b = "<KEY_A>", "<KEY_B>"
var_a = "<VAR_FROM_A>"
vars_b = ["<VAR_FROM_B_1>", "<VAR_FROM_B_2>"]

df_a[key_a] = df_a[key_a].astype(str).str.strip()   # .str.zfill(5) for FIPS
df_b[key_b] = df_b[key_b].astype(str).str.strip()
df_a[var_a] = pd.to_numeric(df_a[var_a], errors="coerce")
for v in vars_b:
    df_b[v] = pd.to_numeric(df_b[v], errors="coerce")

merged = df_a.merge(df_b, left_on=key_a, right_on=key_b).dropna(subset=[var_a] + vars_b)

rows = []
for v in vars_b:
    r, p = pearsonr(merged[var_a], merged[v])
    rows.append({"pair": f"{var_a} vs {v}", "r": round(r, 4), "p": round(p, 6), "n": len(merged)})
import os
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT_NODE_PATH, index=False)
```

## Interpretation

| \|r\| range | Strength |
|---|---|
| 0.00 – 0.19 | Very weak |
| 0.20 – 0.39 | Weak |
| 0.40 – 0.59 | Moderate |
| 0.60 – 0.79 | Strong |
| 0.80 – 1.00 | Very strong |

Always report `r`, `p`, **and** `n`. Correlation is association, not causation.

## Common pitfalls

| Pitfall | Fix |
|---|---|
| `pearsonr` fails on NaNs | Drop pairwise: `mask = x.notna() & y.notna()`. |
| Non-numeric dtype | `pd.to_numeric(col, errors="coerce")`. |
| Pearson on skewed / non-linear data | Plot first; use Spearman/Kendall. |
| FIPS-code merge returns empty | Cast both keys to string, `zfill(5)` county, `zfill(2)` state. |
| Many pairs without correction | Apply Bonferroni/FDR when testing many hypotheses. |
