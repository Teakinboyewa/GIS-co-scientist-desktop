---
id: local_morans_i
agent: modeling
keywords:
  - lisa
  - local moran
  - moran_local
  - cluster map
  - hotspot
  - coldspot
  - high-high
  - low-low
  - spatial outlier
summary: >
  Local Moran's I (LISA) — identify local hotspots (High-High), coldspots
  (Low-Low), and spatial outliers (High-Low, Low-High) with a cluster map.
---

# Local Moran's I (LISA)

## Description

LISA decomposes Global Moran's I into per-observation contributions, classifying each unit into one of four quadrants: **High-High** (hotspot), **Low-Low** (coldspot), **High-Low** / **Low-High** (spatial outliers), or non-significant.

## When to use

- You've already seen significant Global Moran's I and want to know **where** clusters are.
- Policy targeting — identifying specific hotspots for intervention.
- Mapping spatial outliers (units that differ sharply from their neighbors).

## When NOT to use

- Single yes/no question about overall clustering → use Global Moran's I.
- Small `n` (< ~30 units) — too few permutations give unstable p-values.

## Critical constraints

1. **Only report significant clusters** — filter by `lisa.p_sim < 0.05` before labeling.
2. **Row-standardize the weights** — `w.transform = "r"`.
3. **Quadrant codes:** `1=HH, 2=LH, 3=LL, 4=HL`. Easy to mislabel.
4. **High permutation count** — `permutations=999` minimum; `9999` for publication-grade p-values.
5. **Drop NAs and reset index** before building weights.

## Paths

- `INPUT_PATH` — GeoPackage from Input Node Paths.
- `OUTPUT_NODE_PATH` — GeoPackage with LISA columns (primary).
- `OUTPUT_DIR` — directory for the cluster-map PNG.

Never hardcode `"data.gpkg"` or `"lisa_*.png"`.

## Canonical example (LISA + cluster map)

```python
import os
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libpysal.weights import Queen, KNN
from esda.moran import Moran_Local

gdf = gpd.read_file(INPUT_PATH)

col = "<TARGET_COLUMN>"
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

# ---- Local Moran's I ----------------------------------------------------
lisa = Moran_Local(gdf[col].values, w, permutations=999)

gdf["LISA_I"] = lisa.Is
gdf["LISA_p"] = lisa.p_sim
gdf["LISA_q"] = lisa.q                 # 1=HH, 2=LH, 3=LL, 4=HL

quadrant = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
gdf["LISA_cluster"] = [
    quadrant[q] if p < 0.05 else "Not Significant"
    for q, p in zip(gdf["LISA_q"], gdf["LISA_p"])
]

os.makedirs(os.path.dirname(OUTPUT_NODE_PATH), exist_ok=True)
gdf.to_file(OUTPUT_NODE_PATH, driver="GPKG")

# ---- Cluster map --------------------------------------------------------
colors = {
    "High-High": "#d7191c", "Low-Low": "#2c7bb6",
    "High-Low": "#fdae61",  "Low-High": "#abd9e9",
    "Not Significant": "#dddddd",
}
fig, ax = plt.subplots(figsize=(10, 8))
for label, color in colors.items():
    sub = gdf[gdf["LISA_cluster"] == label]
    if not sub.empty:
        sub.plot(ax=ax, color=color, label=label, edgecolor="grey", linewidth=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.set_title(f"LISA Cluster Map: {col}")
ax.axis("off")
fig.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_DIR, "lisa_cluster_map.png"), dpi=150)
plt.close(fig)
```

## Interpretation

| Quadrant | Label | Meaning |
|---|---|---|
| 1 | High-High | High value, high-value neighbors — **hotspot** |
| 3 | Low-Low  | Low value, low-value neighbors — **coldspot** |
| 2 | Low-High | Low value surrounded by highs — **spatial outlier** |
| 4 | High-Low | High value surrounded by lows — **spatial outlier** |

A feature's quadrant is only meaningful when `p_sim < 0.05`. Everything else is "Not Significant".

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Mapping raw `lisa.q` without p-filter | Mask by `p_sim < 0.05` first. |
| Confusing quadrant codes | `1=HH, 2=LH, 3=LL, 4=HL` — write it down. |
| Multiple comparisons inflate significance | For strict inference, apply FDR or a stricter alpha (e.g., 0.01). |
| Very small clusters of 1–2 units | Often noise; require a minimum cluster size in post-processing. |
| Weights drift after dropna | Reset index and rebuild weights on the cleaned frame. |
