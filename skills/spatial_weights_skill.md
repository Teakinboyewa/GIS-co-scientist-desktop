---
id: spatial_weights
agent: modeling
keywords:
  - spatial weights
  - queen
  - rook
  - knn weights
  - distance band
  - contiguity
  - libpysal
  - row-standardize
  - islands
summary: >
  Building spatial weights matrices (Queen/Rook contiguity, KNN, distance-band),
  row-standardization, island handling — the prerequisite for Moran's I, LISA,
  and spatial regression.
---

# Spatial Weights

## Description

A spatial weights matrix `W` defines which observations are neighbors and how strongly they're connected. Almost every spatial statistic (Moran's I, LISA, spatial lag/error regression) takes `W` as input.

## When to use

- Before Global or Local Moran's I.
- Before spatial regression (SAR, SEM) or GWR diagnostics.
- Any time you need to express "neighbor of" relationships between geographic units.

## When NOT to use

- Non-spatial data — no geometry, no weights.
- Purely attribute-space neighborhoods (use sklearn KNN on features instead).

## Choosing the scheme

| Scheme | Use when | Class |
|---|---|---|
| Queen contiguity | Polygons (counties, tracts); neighbors share edge or vertex | `libpysal.weights.Queen` |
| Rook contiguity | Polygons where only edge-sharing counts | `libpysal.weights.Rook` |
| KNN | Points, or polygons with islands; fixed neighbor count `k` | `libpysal.weights.KNN` |
| Distance band | Points with meaningful metric distance | `libpysal.weights.DistanceBand` |

## Critical constraints

1. **Always row-standardize** — `w.transform = "r"` before any Moran's I.
2. **Check for islands** — isolated features break Moran's I. Switch to KNN if Queen produces them.
3. **Drop NAs first, then build weights** — weights must align with the rows actually used.
4. **Project before distance-band weights** — distances are meaningless in lat/lon degrees.

## Paths

This skill only reads data — it produces a `w` object, not a file.

- `INPUT_PATH` — GeoPackage/SHP from Input Node Paths. Never hardcode.

## Canonical example

```python
import geopandas as gpd
from libpysal.weights import Queen, KNN, DistanceBand

gdf = gpd.read_file(INPUT_PATH)
col = "<ANALYSIS_COLUMN>"

# ---- 1. Drop NAs on the column you'll analyze ---------------------------
gdf = gdf.dropna(subset=[col]).reset_index(drop=True)

# ---- 2. Build weights ---------------------------------------------------
# (a) Queen contiguity — default for polygons
w = Queen.from_dataframe(gdf)

# (b) KNN fallback if Queen produces islands
if w.islands:
    print(f"Queen produced {len(w.islands)} islands — switching to KNN(k=5)")
    w = KNN.from_dataframe(gdf, k=5)

# (c) Distance band — requires projected CRS in meters
# if gdf.crs.is_geographic:
#     gdf = gdf.to_crs(epsg=3857)
# w = DistanceBand.from_dataframe(gdf, threshold=10000, binary=False)

# ---- 3. Row-standardize -------------------------------------------------
w.transform = "r"

print(f"n = {w.n}, mean neighbors = {w.mean_neighbors:.2f}, islands = {len(w.islands)}")
```

## Common pitfalls

| Pitfall | Fix |
|---|---|
| Islands break Moran's I | `if w.islands: use KNN.from_dataframe(gdf, k=5)`. |
| Forgot to row-standardize | `w.transform = "r"` every time. |
| Weights built before dropping NAs | Drop first, reset index, then build — row counts must match. |
| Distance band in lat/lon | Reproject to a meter-based CRS first. |
| `k` too small in KNN | `k=5` to `k=8` is typical; too small gives noisy LISA. |
