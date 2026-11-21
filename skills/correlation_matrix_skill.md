---
id: correlation_matrix
agent: modeling
keywords:
  - correlation matrix
  - multicollinearity
  - heatmap
  - pairwise correlation
  - collinearity
  - corr()
summary: >
  Full pairwise correlation heatmap, high-correlation detection,
  target-variable ranking, multicollinearity check.
---

# Correlation Matrix & Multicollinearity

## Description

All pairwise correlations among a set of numeric variables at once. Primary use: diagnosing multicollinearity among predictors before regression.

## When to use

- Screening predictors for multicollinearity before fitting a regression.
- Exploratory overview of associations across many variables.
- Ranking many predictors against a single target.

## When NOT to use

- Single variable pair → use the Bivariate Correlation skill.
- Spatial clustering of one variable → use Moran's I.

## Critical constraints

1. **Cast columns to numeric first** — `pd.to_numeric(col, errors="coerce")`. Mixed-type columns silently drop from `.corr()`.
2. **Do not pass ID/FIPS/year columns.** They're numeric but meaningless here; always name the columns explicitly.
3. **Use `method="spearman"` for skewed or ordinal data** — Pearson assumes roughly linear, roughly normal.
4. **Multicollinearity threshold** is convention, not law. `|r| > 0.7` is the usual flag; `VIF > 5–10` is the rigorous test when you need it.

## Paths

- `INPUT_PATH` — from Input Node Paths.
- `OUTPUT_NODE_PATH` — primary CSV (full correlation matrix).
- `OUTPUT_DIR` — for target-correlations CSV and heatmap PNG.

Never hardcode `"data.csv"` or `"correlation_*.csv"`.

## Canonical example (heatmap + high-correlation flags + target ranking)

```python
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(INPUT_PATH)

# Replace with actual column names
target     = "<TARGET_COLUMN>"
predictors = ["<PREDICTOR_1>", "<PREDICTOR_2>", "<PREDICTOR_3>",
              "<PREDICTOR_4>", "<PREDICTOR_5>"]
cols = [target] + predictors

# ---- 1. Cast to numeric -------------------------------------------------
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- 2. Full correlation matrix -----------------------------------------
corr = df[cols].corr(method="pearson").round(4)
os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
corr.to_csv(OUTPUT_NODE_PATH)

# ---- 3. Flag highly correlated predictor pairs --------------------------
threshold = 0.7
pairs = []
for i in range(len(predictors)):
    for j in range(i + 1, len(predictors)):
        v = corr.loc[predictors[i], predictors[j]]
        if abs(v) > threshold:
            pairs.append((predictors[i], predictors[j], v))
pairs.sort(key=lambda t: abs(t[2]), reverse=True)

if pairs:
    print(f"Predictor pairs with |r| > {threshold}:")
    for a, b, v in pairs:
        print(f"  {a} <-> {b}: {v:.4f}")
else:
    print(f"No predictor pairs with |r| > {threshold}")

# ---- 4. Target-variable ranking -----------------------------------------
target_corrs = corr[target].drop(target).sort_values(key=abs, ascending=False)
os.makedirs(OUTPUT_DIR, exist_ok=True)
target_corrs.to_csv(os.path.join(OUTPUT_DIR, "target_correlations.csv"), header=["correlation"])
print(f"\nCorrelations with {target} (abs-ranked):")
print(target_corrs.round(4).to_string())

# ---- 5. Heatmap ---------------------------------------------------------
n = len(cols)
fig, ax = plt.subplots(figsize=(max(8, n), max(6, n * 0.8)))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for i in range(n):
    for j in range(n):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.5 else "black")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(cols, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Pearson Correlation Matrix")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close(fig)
```

## Interpretation

| \|r\| range | Strength |
|---|---|
| 0.00 – 0.19 | Very weak |
| 0.20 – 0.39 | Weak |
| 0.40 – 0.59 | Moderate |
| 0.60 – 0.79 | Strong |
| 0.80 – 1.00 | Very strong |

Pairs with `|r| > 0.7` may distort regression coefficients — drop one, combine them, or use PCA/VIF.

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Non-numeric columns silently ignored | `pd.to_numeric(col, errors="coerce")` before `.corr()`. |
| Matrix cluttered by IDs/FIPS/years | List the columns explicitly; don't pass `df.select_dtypes(np.number)` blindly. |
| Heatmap unreadable with >15 columns | Prefer target-ranking output; reserve heatmap for the flagged subset. |
| Pearson on skewed / ordinal data | Use `method="spearman"`. |
| Reporting correlation without `n` | Always report sample size alongside `r`. |
