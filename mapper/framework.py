"""Prototype of the proposed framework (Stages 1-4) for choropleth maps.

Stage 1  anatomy: code finds the legend (color bar or class patches) and lettered markers;
         the VLM reads each marker crop up close
Stage 2  specification: the VLM extracts a structured spec (variable, units, region, unit, legend labels)
Stage 3  measured reading: code aligns the map to reference boundaries and reads every unit's color
Stage 4  analysis: ranks, shares and per-group summaries are computed from the table;
         the model answers only from that table

Scope: vertical continuous color bars, or classed legends drawn as color patches; regions with
reference boundaries in REFERENCE.
"""
import base64
import io
import json
import logging
import re
import time
from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.affinity import affine_transform
from shapely.geometry import Point

log = logging.getLogger("map_interpreter")
PROJECTS = Path(__file__).resolve().parents[2]
CHD = PROJECTS / "CHD Research/EXPERIMENTS"
# Census cartographic boundaries (2019, 1:500k). Only shapes and identifiers are ever loaded: the tract file
# also stores CHD values, so every read passes `columns`.
COUNTIES = Path(__file__).resolve().parent / "reference/cb_2019_us_county_500k/cb_2019_us_county_500k.shp"
STATES = CHD / "ARCGIS_NEW/cb_2019_us_state_500k/cb_2019_us_state_500k.shp"  # contiguous 48 + DC
TRACTS = CHD / "cb_2019_us_tract_500k/cb_2019_us_tract_500k.shp"
NATIONAL = ("united states", "contiguous", "conus", "usa", "u.s.", "america")
STATES_OUTSIDE_FILE = pd.DataFrame({"STATEFP": ["02", "15"], "STUSPS": ["AK", "HI"], "NAME": ["Alaska", "Hawaii"]})
# web mercator, lon/lat, US Albers (also fits ESRI:102003 up to a shift), US Lambert; georeference() adds
# conic and transverse Mercator projections centred on the mapped area, which approximate state plane / UTM
CANDIDATE_CRS = ["EPSG:3857", "EPSG:4326", "EPSG:5070", "ESRI:102004"]

# Cartographic knowledge base: notes added to Stage 4 when the spec matches
KNOWLEDGE = {
    "value_is_count": "The variable is a raw count. Counts are not normalized by population or area, "
                      "so they largely track population and unit size.",
}


# ---------------------------------------------------------------- helpers

def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def to_data_url(img, max_side=2048):
    """PNG data URL; whole maps are scaled down for the VLM (legend_bbox is in fractions, so it still applies)."""
    pil = Image.fromarray(img)
    pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chromatic(img, threshold=8):
    """Colored pixels; excludes white background, gray/black text, borders and markers."""
    return (img.max(-1).astype(int) - img.min(-1).astype(int)) > threshold


def inked(img):
    """Anything that is not white background, including gray and near-white legend classes."""
    return (img < 248).any(-1)


def rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=float) / 255
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = rgb @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def rasterize(geom, shape):
    mask = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty:
            continue
        draw.polygon(list(poly.exterior.coords), fill=1)
        for hole in poly.interiors:
            draw.polygon(list(hole.coords), fill=0)
    return np.asarray(mask, dtype=bool)


def longest_run(flags):
    """(start, end) of the longest run of True values, end inclusive."""
    best, start = (0, -1), None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def runs(flags):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def erode(mask, size=5):
    pad = size // 2
    padded = np.pad(mask, pad, constant_values=False)
    return np.lib.stride_tricks.sliding_window_view(padded, (size, size)).min(axis=(-1, -2))


def components(mask, min_area=40):
    """Connected components of a mask (4-connectivity), as arrays of (y, x)."""
    todo = set(zip(*np.nonzero(mask)))
    out = []
    while todo:
        seed = todo.pop()
        comp, queue = [seed], deque([seed])
        while queue:
            y, x = queue.popleft()
            for nb in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if nb in todo:
                    todo.remove(nb)
                    comp.append(nb)
                    queue.append(nb)
        if len(comp) >= min_area:
            out.append(np.array(comp))
    return out


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0)) if match else {"answer": None, "explanation": text}


def vlm_json(client, model, prompt, images):
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": url} for url in images]
    response = client.responses.create(model=model, input=[{"role": "user", "content": content}],
                                       text={"format": {"type": "json_object"}})
    return json.loads(response.output_text)


# ---------------------------------------------------------------- Stage 2: specification

SPEC_PROMPT = """Extract this map's specification. Reply in JSON with these keys:
"title", "map_type", "variable", "units",
"region": the area mapped (e.g. "Ohio", "contiguous United States"),
"unit": the enumeration unit (e.g. "county", "census tract"),
"legend_type": "continuous" (a color bar) or "classed" (separate color patches),
"legend_orientation": "vertical" or "horizontal",
"legend_ticks": for a color bar, the tick label values as numbers, ordered from the bottom (or left) end to the top (or right) end; else [],
"legend_classes": for a classed legend, the class labels exactly as written, in the order shown (top to bottom or left to right); else [],
"legend_bbox": [x0, y0, x1, y1] of the color bar or legend box as fractions of image width and height,
"value_is_count": true if the variable is a raw count rather than a rate, share, density or index,
"annotations": any statistics or notes printed on the map (e.g. "Moran's I = 0.38"), as a list of strings,
"markers": letters drawn on the map as labels, or []."""


def extract_spec(client, model, img):
    return vlm_json(client, model, SPEC_PROMPT, [to_data_url(img)])


# ---------------------------------------------------------------- Stage 1: continuous color bar

def find_colorbar(img, bbox_frac=None):
    """Pixel box (x0, y0, x1, y1) of a vertical color bar, searched near the spec's approximate bbox."""
    h, w, _ = img.shape
    if bbox_frac:
        x0, y0, x1, y1 = bbox_frac
        px, py = 0.04 * w, 0.04 * h
        sx0, sy0 = int(max(0, x0 * w - px)), int(max(0, y0 * h - py))
        sx1, sy1 = int(min(w, x1 * w + px)), int(min(h, y1 * h + py))
    else:  # fall back to the right quarter of the image
        sx0, sy0, sx1, sy1 = int(0.75 * w), 0, w, h
    crop = img[sy0:sy1, sx0:sx1].astype(int)
    chrom = chromatic(img[sy0:sy1, sx0:sx1])
    # the bar is the tallest contiguous run of colored rows whose color changes smoothly; a run across the map
    # jumps in color at every unit border, so it does not count even when it is longer
    col_runs = np.zeros(chrom.shape[1], dtype=int)
    for c in range(chrom.shape[1]):
        a, b = max(runs(chrom[:, c]), key=lambda r: r[1] - r[0], default=(0, -1))
        if b - a > 10 and (np.abs(np.diff(crop[a:b + 1, c], axis=0)).max(1) <= 20).mean() >= 0.97:
            col_runs[c] = b - a
    if not col_runs.any():
        if bbox_frac:
            return find_colorbar(img, None)
        raise ValueError("Could not find the color bar")
    c0, c1 = longest_run(col_runs >= 0.9 * col_runs.max())
    r0, r1 = longest_run(chrom[:, c0:c1 + 1].mean(1) > 0.9)
    tall_enough = (r1 - r0) > 0.2 * h and (r1 - r0) > 3 * (c1 - c0)
    if bbox_frac and not tall_enough:  # the VLM's rough box missed the bar
        return find_colorbar(img, None)
    return sx0 + c0, sy0 + r0, sx0 + c1, sy0 + r1


def tick_rows(img, bar, side):
    """Pixel rows of the short tick marks beside a vertical bar, on its right (+1) or left (-1) side."""
    x0, y0, x1, y1 = bar
    dark = img.mean(-1) < 128
    # skip the bar's frame (dark for nearly all bar rows), then look for short tick marks
    x = x1 + 1 if side > 0 else x0 - 1
    while 0 < x < img.shape[1] - 1 and dark[y0:y1 + 1, x].mean() > 0.8:
        x += side
    cols = slice(x, x + 6) if side > 0 else slice(max(0, x - 5), x + 1)
    band = dark[max(0, y0 - 2):y1 + 3, cols].any(1)
    return [max(0, y0 - 2) + (a + b) / 2 for a, b in runs(band)]


def calibrate(img, bar, ticks, low_end="bottom"):
    """Linear pixel-row -> value function from the bar's tick marks and the spec's tick labels.
    low_end is the end of the bar holding the first tick label ("top" for a horizontal bar turned on its side)."""
    x0, y0, x1, y1 = bar
    centers = tick_rows(img, bar, +1)
    if len(centers) != len(ticks):  # ticks can also sit on the other side of the bar
        centers = tick_rows(img, bar, -1)
    if ticks and len(centers) == len(ticks):
        slope, intercept = np.polyfit(sorted(centers, reverse=low_end == "bottom"), ticks, 1)  # low-end tick first
        mode = "tick marks"
    else:  # assume the bar ends at the first and last tick
        low, high = (y1, y0) if low_end == "bottom" else (y0, y1)
        slope = (ticks[-1] - ticks[0]) / (high - low)
        intercept = ticks[0] - slope * low
        mode = f"bar ends (found {len(centers)} tick marks for {len(ticks)} labels)"
    return (lambda y: slope * np.asarray(y) + intercept), mode


def sample_ramp(img, bar, to_value):
    x0, y0, x1, y1 = bar
    rows = np.arange(y0 + 1, y1)
    colors = np.median(img[y0 + 1:y1, x0 + 2:x1 - 1], axis=1)
    return rgb_to_lab(colors), to_value(rows)


# ---------------------------------------------------------------- Stage 1: classed legend

def parse_classes(labels):
    """[(label, low, high)] from labels like '0.300 - 4.200', sorted from the lowest class."""
    out = []
    for label in labels:
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", label.replace(",", ""))]
        out.append((label, numbers[0], numbers[-1]))
    return out


def flat_rectangles(crop, min_side=6, step=10, tolerance=12):
    """Boxes (x0, y0, x1, y1) of solid, non-white rectangles: rows that each hold a run of one color with the
    same left and right edges. Works when patches touch a tinted legend background or carry JPEG noise,
    and finds gray or near-white classes as well as colored ones."""
    h, w, _ = crop.shape
    pixels = crop.astype(int)
    done, active = [], []
    for y in range(h):
        row = pixels[y]
        cuts = np.nonzero(np.abs(np.diff(row, axis=0)).max(1) > step)[0] + 1
        row_runs = []
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, w] - 1):
            if e - s + 1 >= min_side:
                color = row[s:e + 1].mean(0)
                if color.min() < 248:  # skip white background
                    row_runs.append((s, e, color))
        still = []
        for s, e, color in row_runs:
            match = next((r for r in active if abs(r["x0"] - s) <= 1 and abs(r["x1"] - e) <= 1
                          and np.abs(r["color"] - color).max() <= tolerance), None)
            if match:
                active.remove(match)
                match["y1"] = y
                still.append(match)
            else:
                still.append({"x0": s, "x1": e, "y0": y, "y1": y, "color": color})
        done += active
        active = still
    done += active
    return [(r["x0"], r["y0"], r["x1"], r["y1"]) for r in done if r["y1"] - r["y0"] + 1 >= min_side]


def _patches_in(crop, n):
    """n same-size solid rectangles lined up in a column or row: the legend's class patches."""
    boxes = flat_rectangles(crop)
    same = lambda a, b, i: abs(a[i] - b[i]) <= 3
    size = lambda a, b: abs((a[2] - a[0]) - (b[2] - b[0])) <= 3 and abs((a[3] - a[1]) - (b[3] - b[1])) <= 3
    color = lambda b: rgb_to_lab(np.median(crop[b[1] + 1:b[3], b[0] + 1:b[2]].reshape(-1, 3), axis=0))
    best = None
    for b in boxes:
        for axis, order in ((0, 1), (1, 0)):  # aligned left edges (vertical legend) or top edges (horizontal)
            group = sorted({o for o in boxes if same(o, b, axis) and size(o, b)}, key=lambda o: o[order])
            if len(group) != n:
                continue
            labs = np.array([color(o) for o in group])
            gaps = np.linalg.norm(labs[:, None] - labs[None], axis=2)[np.triu_indices(n, 1)]
            if gaps.min() < 8:  # a legend's classes all differ; same-colored map blocks do not
                continue
            area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
            if best is None or area > best[0]:  # the biggest such group: map pieces are usually smaller
                best = (area, group)
    return best[1] if best else None


def find_legend_patches(img, bbox_frac, n):
    """Pixel boxes and median colors of the n class patches, in legend order."""
    h, w, _ = img.shape
    regions = []
    if bbox_frac:
        x0, y0, x1, y1 = bbox_frac
        regions.append((int(max(0, (x0 - 0.04) * w)), int(max(0, (y0 - 0.04) * h)),
                        int(min(w, (x1 + 0.04) * w)), int(min(h, (y1 + 0.04) * h))))
    # fall back to the four corners, where legends usually sit
    regions += [(0, int(0.5 * h), int(0.45 * w), h), (int(0.55 * w), int(0.5 * h), w, h),
                (0, 0, int(0.45 * w), int(0.5 * h)), (int(0.55 * w), 0, w, int(0.5 * h))]
    for sx0, sy0, sx1, sy1 in regions:
        group = _patches_in(img[sy0:sy1, sx0:sx1], n)
        if group:
            out = []
            for x0, y0, x1, y1 in group:
                inner = img[sy0 + y0 + 2:sy0 + y1 - 1, sx0 + x0 + 2:sx0 + x1 - 1].reshape(-1, 3)
                out.append(((sx0 + x0, sy0 + y0, sx0 + x1, sy0 + y1), np.median(inner, axis=0)))
            return out
    raise ValueError(f"Could not find {n} legend patches")


def classify_pixels(img, class_lab, max_delta_e=12):
    """Class index of every pixel by nearest legend color (CIELAB), -1 where none is close."""
    out = np.full(img.shape[:2], -1, dtype=int)
    chrom = inked(img)  # not just colored pixels: classes can be gray or near-white
    pixels = img[chrom]
    best = np.empty(len(pixels), dtype=int)
    for start in range(0, len(pixels), 500_000):  # chunked so large maps stay within memory
        dist = np.linalg.norm(rgb_to_lab(pixels[start:start + 500_000])[:, None, :] - class_lab[None], axis=2)
        chunk = dist.argmin(1)
        chunk[dist.min(1) > max_delta_e] = -1
        best[start:start + 500_000] = chunk
    out[chrom] = best
    return out


def sample_classes(pixclass, xs, ys):
    """Class at each point by a 3x3 majority vote (center counts double), -1 if no colored pixel."""
    h, w = pixclass.shape
    k = pixclass.max() + 1
    votes = np.zeros((len(xs), max(k, 1)))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = pixclass[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
            ok = np.nonzero(c >= 0)[0]
            votes[ok, c[ok]] += 2 if dy == dx == 0 else 1
    cls = votes.argmax(1)
    cls[votes.max(1) == 0] = -1
    return cls


# ---------------------------------------------------------------- Stage 3: georeferencing and reading

def state_table():
    states = gpd.read_file(STATES, columns=["STATEFP", "STUSPS", "NAME"], ignore_geometry=True)
    return pd.concat([states, STATES_OUTSIDE_FILE], ignore_index=True)


def match_states(region):
    """US states named in the spec's region ('Pennsylvania', 'Ohio and Indiana', 'PA'), as a table."""
    states = state_table()
    text, found = f" {region.lower()} ", []
    for name in sorted(states.NAME, key=len, reverse=True):  # longest first, so 'West Virginia' is not also 'Virginia'
        pattern = rf"\b{re.escape(name.lower())}\b"
        if re.search(pattern, text):
            found.append(name)
            text = re.sub(pattern, " ", text)
    if not found:  # postal abbreviations, only when no state is spelled out
        tokens = set(re.findall(r"\b[A-Z]{2}\b", region))
        found = [n for n, a in zip(states.NAME, states.STUSPS) if a in tokens]
    return states[states.NAME.isin(found)]


def load_reference(spec):
    """Reference boundaries for the spec's region and unit: counties, census tracts or states of the US.

    Returns {"units": GeoDataFrame with a "name" column, "groups": GeoDataFrame or None, "group_unit": ...}.
    Tract maps are summarized by groups: states for a national map, counties for a state map.
    """
    region, unit = str(spec.get("region") or ""), str(spec.get("unit") or "").lower()
    states = match_states(region)
    national = states.empty and any(k in region.lower() for k in NATIONAL)
    if states.empty and not national:
        raise KeyError(f"No reference boundaries for region={spec.get('region')!r}, unit={spec.get('unit')!r}: "
                       "only US states, counties and census tracts are available")
    fips = set(states.STATEFP) if not states.empty else set(state_table().STATEFP) - {"02", "15"}
    abbr = dict(zip(state_table().STATEFP, state_table().STUSPS))
    keep = lambda df: df[df.STATEFP.isin(fips)].reset_index(drop=True)

    if "tract" in unit or "neighborhood" in unit:
        units = keep(gpd.read_file(TRACTS, columns=["GEOID", "STATEFP"])).rename(columns={"GEOID": "name"})
        if national:
            groups = keep(gpd.read_file(STATES, columns=["STATEFP", "NAME"]))
            units["group_id"] = units.STATEFP
            groups = groups.rename(columns={"STATEFP": "group_id", "NAME": "group_name"})
            return {"units": units, "groups": groups, "group_unit": "state"}
        counties = keep(gpd.read_file(COUNTIES, columns=["GEOID", "STATEFP", "NAME"]))
        units["group_id"] = units.name.str[:5]
        label = counties.NAME + (", " + counties.STATEFP.map(abbr) if len(fips) > 1 else "")
        groups = counties.assign(group_id=counties.GEOID, group_name=label)[["group_id", "group_name", "geometry"]]
        return {"units": units, "groups": groups, "group_unit": "county"}

    if "count" in unit or "parish" in unit or "borough" in unit:
        counties = keep(gpd.read_file(COUNTIES, columns=["GEOID", "STATEFP", "NAME"]))
        # county names repeat across states, so a multi-state map names them "Washington, PA"
        name = counties.NAME + (", " + counties.STATEFP.map(abbr) if national or len(fips) > 1 else "")
        return {"units": counties.assign(name=name)[["name", "geometry"]], "groups": None, "group_unit": None}

    if "state" in unit and (national or len(fips) > 1):
        units = gpd.read_file(STATES, columns=["STATEFP", "NAME"])
        units = keep(units).rename(columns={"NAME": "name"})[["name", "geometry"]]
        return {"units": units, "groups": None, "group_unit": None}

    raise KeyError(f"No reference boundaries for region={spec.get('region')!r}, unit={spec.get('unit')!r}: "
                   "the unit must be US states, counties or census tracts")


def local_projections(fit):
    """Conic and transverse Mercator projections centred on the mapped area (stand-ins for state plane / UTM)."""
    minx, miny, maxx, maxy = fit.to_crs("EPSG:4326").total_bounds
    lon0, lat0, span = (minx + maxx) / 2, (miny + maxy) / 2, maxy - miny
    lat1, lat2 = miny + span / 6, maxy - span / 6
    return [f"+proj=lcc +lat_1={lat1:.4f} +lat_2={lat2:.4f} +lat_0={lat0:.4f} +lon_0={lon0:.4f} +datum=NAD83 +units=m",
            f"+proj=tmerc +lat_0={lat0:.4f} +lon_0={lon0:.4f} +k=0.9996 +datum=NAD83 +units=m"]


def georeference(img, fit, exclude_boxes, footprint=None):
    """Fit reference boundaries to the map's footprint; try each candidate projection.
    The footprint defaults to colored pixels; classed maps pass the pixels that match a legend color."""
    fg = chromatic(img) if footprint is None else footprint.copy()
    if footprint is not None:  # close 1-pixel gaps (unit borders drawn in gray or black are not legend colors)
        fg = erode(~erode(~fg, 3), 3)
    valid = np.ones(fg.shape, dtype=bool)  # the legend area counts for neither the map nor the outline
    for x0, y0, x1, y1 in exclude_boxes:
        valid[max(0, y0 - 6):y1 + 7, max(0, x0 - 6):x1 + 7] = False
    fg &= valid
    rows, cols = np.nonzero(fg.sum(1) >= 3)[0], np.nonzero(fg.sum(0) >= 3)[0]
    py0, py1, px0, px1 = rows.min(), rows.max(), cols.min(), cols.max()

    def score(outline, params):
        silhouette = rasterize(affine_transform(outline, params), fg.shape) & valid
        return (silhouette & fg).sum() / (silhouette | fg).sum(), silhouette

    candidates = []
    for crs in CANDIDATE_CRS + local_projections(fit):
        outline = fit.to_crs(crs).union_all()
        minx, miny, maxx, maxy = outline.bounds
        sx, sy = (px1 - px0) / (maxx - minx), (py1 - py0) / (maxy - miny)
        params = [sx, 0, 0, -sy, px0 - minx * sx, py0 + maxy * sy]
        candidates.append((score(outline, params)[0], crs, params, outline))
    candidates.sort(key=lambda c: -c[0])

    best = None
    for _, crs, params, outline in candidates[:2]:  # refine the two best projections, keep the better
        iou, params = refine_fit(outline, params, lambda p: score(outline, p)[0])
        if best is None or iou > best[0]:
            best = (iou, crs, params, outline)
    iou, crs, params, outline = best
    return iou, crs, params, rasterize(affine_transform(outline, params), fg.shape)


def refine_fit(outline, params, score):
    """Coordinate search over x/y scale and shift that maximizes the overlap score.
    The starting fit matches only the outer edges, which a legend or inset over the map can hide."""
    sx, sy, tx, ty = params[0], -params[3], params[4], params[5]
    cx, cy = outline.centroid.x, outline.centroid.y  # scale about the centre so shifts stay independent
    build = lambda sx, sy, tx, ty: [sx, 0, 0, -sy, tx, ty]
    current = (sx, sy, tx, ty)
    best = score(build(*current))
    for scale_step, shift_step in [(0.02, 4), (0.01, 2), (0.005, 1)]:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for sign in (1, -1):
                    s = list(current)
                    if i < 2:  # rescale about the outline's centre: keep its pixel position fixed
                        factor = 1 + sign * scale_step
                        centre_px = s[2] + s[0] * cx if i == 0 else s[3] - s[1] * cy
                        s[i] *= factor
                        if i == 0:
                            s[2] = centre_px - s[0] * cx
                        else:
                            s[3] = centre_px + s[1] * cy
                    else:
                        s[i] += sign * shift_step
                    value = score(build(*s))
                    if value > best:
                        best, current, improved = value, tuple(s), True
    return best, build(*current)


def to_pixels(gdf, crs, params):
    g = gdf.to_crs(crs)
    return g.set_geometry(g.geometry.apply(lambda geom: affine_transform(geom, params)))


def crs_label(crs):
    return {"+proj=lcc": "local Lambert conic", "+proj=tmerc": "local transverse Mercator"}.get(crs.split(" ")[0], crs)


def unit_pixels(geom, shape, inset=2.5):
    """(rows, cols) of the pixels inside a unit, inset from its border; rasterized within its own bounding box."""
    inner = geom.buffer(-inset)
    geom = inner if not inner.is_empty else geom
    minx, miny, maxx, maxy = geom.bounds
    x0, y0 = max(0, int(np.floor(minx))), max(0, int(np.floor(miny)))
    x1, y1 = min(shape[1], int(np.ceil(maxx)) + 1), min(shape[0], int(np.ceil(maxy)) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=int), np.array([], dtype=int)
    local = affine_transform(geom, [1, 0, 0, 1, -x0, -y0])
    ys, xs = np.nonzero(rasterize(local, (y1 - y0, x1 - x0)))
    return ys + y0, xs + x0


def measure(img, pix, ramp_lab, ramp_values):
    chrom = chromatic(img)
    rows = []
    for name, geom in zip(pix["name"], pix.geometry):
        ys, xs = unit_pixels(geom, img.shape)
        keep = chrom[ys, xs]
        ys, xs = ys[keep], xs[keep]
        n = len(ys)
        if n < 5:
            rows.append({"name": name, "value": np.nan, "confidence": "low", "pixels": n, "delta_e": np.nan})
            continue
        lab = rgb_to_lab(np.median(img[ys, xs], axis=0))
        dist = np.linalg.norm(ramp_lab - lab, axis=1)
        i = int(dist.argmin())
        confidence = "high" if dist[i] < 3 and n >= 50 else "medium" if dist[i] < 8 else "low"
        rows.append({"name": name, "value": float(ramp_values[i]), "confidence": confidence,
                     "pixels": n, "delta_e": round(float(dist[i]), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Stage 1: markers

MARKER_PROMPT = """Each image is a close-up crop of one circular label on a map.
Reply in JSON: {"letters": [...]} with the single letter in each crop, in order ("" if a crop has no letter)."""


def find_markers(img, silhouette):
    """White label circles inside the mapped area: round white blobs too thick to be boundary lines."""
    white = (img > 245).all(-1) & silhouette
    # the letter splits a circle's white interior into pieces, so merge pieces with nearby centers
    groups = []
    for comp in components(erode(white, 5)):
        near = [i for i, g in enumerate(groups) if np.linalg.norm(g.mean(0) - comp.mean(0)) < 30]
        if near:
            groups[near[0]] = np.vstack([groups[near[0]], comp])
        else:
            groups.append(comp)
    found = []
    for comp in groups:
        bh, bw = np.ptp(comp[:, 0]) + 1, np.ptp(comp[:, 1]) + 1
        if not (10 <= max(bh, bw) <= 60 and 0.75 <= bh / bw <= 1.33):  # label-sized and round, not a white gap
            continue
        cy, cx = comp.mean(0)
        r = int(max(bh, bw) / 2) + 8
        crop = img[max(0, int(cy) - r):int(cy) + r, max(0, int(cx) - r):int(cx) + r]
        found.append({"center": (cx, cy), "crop": np.asarray(Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3)))})
    return found


def read_markers(client, model, markers, pix):
    if not markers:
        return {}
    letters = vlm_json(client, model, MARKER_PROMPT, [to_data_url(m["crop"]) for m in markers])["letters"]
    out = {}
    for letter, m in zip(letters, markers):
        letter = str(letter).strip().upper()
        hit = pix.loc[pix.geometry.contains(Point(m["center"])), "name"]
        if letter:
            out[letter] = hit.iloc[0] if len(hit) else None
    return out


# ---------------------------------------------------------------- Stage 4: analysis tables

def add_position(t, pix, markers):
    """x_frac / y_frac of each unit's centre within the mapped area, and the letter marker drawn on it."""
    minx, miny, maxx, maxy = pix.total_bounds
    cent = pix.geometry.centroid
    t["x_frac"] = ((cent.x - minx) / (maxx - minx)).round(2).values
    t["y_frac"] = (1 - (cent.y - miny) / (maxy - miny)).round(2).values
    letter_of = {name: letter for letter, name in markers.items()}
    t["marker"] = t["name"].map(letter_of).fillna("")
    return t


def continuous_table(measured, pix, markers):
    t = add_position(measured.copy(), pix, markers)
    t["rank"] = t["value"].rank(ascending=False, method="min")
    t["share_pct"] = 100 * t["value"] / t["value"].sum()
    return t.sort_values("value", ascending=False).round({"value": 0, "share_pct": 1})


def classed_unit_table(pixclass, pix, classes, markers):
    """Each unit's legend class: the class most of its pixels match, with the share that agree."""
    rows = []
    for name, geom in zip(pix["name"], pix.geometry):
        ys, xs = unit_pixels(geom, pixclass.shape, inset=1.5)
        vals = pixclass[ys, xs]
        vals = vals[vals >= 0]
        if len(vals) < 3:
            rows.append({"name": name, "class": "unreadable", "class_number": np.nan, "pixel_share": np.nan,
                         "pixels": int(len(vals)), "confidence": "low"})
            continue
        counts = np.bincount(vals, minlength=len(classes))
        i = int(counts.argmax())
        share = counts[i] / len(vals)
        confidence = "high" if share >= 0.9 and len(vals) >= 30 else "medium" if share >= 0.6 else "low"
        rows.append({"name": name, "class": classes[i][0], "class_number": i + 1, "pixel_share": round(float(share), 2),
                     "pixels": int(len(vals)), "confidence": confidence})
    t = add_position(pd.DataFrame(rows), pix, markers)
    return t.sort_values(["class_number", "name"], ascending=[False, True])


TOO_SMALL = -2  # class code for units smaller than a pixel: their color cannot be read at this resolution


def grouped_class_table(cls, units, groups, classes):
    """Per-group shares of units in each class, plus an ALL row for the whole map."""
    labels = [c[0] for c in classes]
    names = dict(zip(groups["group_id"], groups["group_name"]))
    df = pd.DataFrame({"name": units["group_id"].map(names).values, "cls": cls})
    df = pd.concat([df, df.assign(name="ALL")])
    read = df[df.cls >= 0]
    counts = read.groupby("name").cls.value_counts().unstack(fill_value=0).reindex(columns=range(len(labels)), fill_value=0)
    pct = 100 * counts.div(counts.sum(1), axis=0)
    t = pct.round(1)
    t.columns = [f"pct {label}" for label in labels]
    t.insert(0, "units_read", counts.sum(1))
    t.insert(1, "units_too_small", df[df.cls == TOO_SMALL].groupby("name").size().reindex(t.index, fill_value=0))
    t.insert(2, "units_no_color", df[df.cls == -1].groupby("name").size().reindex(t.index, fill_value=0))
    t["mean_class"] = (pct / 100 * np.arange(1, len(labels) + 1)).sum(1).round(2)
    t["rank_mean_class"] = t["mean_class"].drop("ALL").rank(ascending=False, method="min")
    return t.reset_index().sort_values("mean_class", ascending=False)


# ---------------------------------------------------------------- the reader

ANSWER_PROMPT = """Answer a question about a thematic map. Use ONLY the evidence below, which code measured from the map image.

Map specification: {spec}

{table_notes}
{table}
{notes}
Question: {question}
{format}
Reply in JSON: {{"answer": ..., "evidence": [names of the rows you used], "explanation": "one sentence"}}."""


class MapReader:
    def __init__(self, client, model, sink=None):
        """sink: an optional list that receives every step as it happens ("[HH:MM:SS] ..."), for a live log."""
        self.client, self.model, self.cache = client, model, {}
        self.steps, self._t0 = [], time.time()
        self.sink = sink if sink is not None else []

    def _note(self, message):
        """One line of the live activity log (and the server log), stamped with the clock time."""
        self.sink.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        log.info("Map Interpreter %s", message)

    def _step(self, message):
        """One step of a measurement: kept with the evidence (elapsed time) and sent to the live log."""
        self.steps.append(f"[{time.time() - self._t0:5.1f}s] {message}")
        self._note(message)

    def read(self, path):
        """Run Stages 1-3 once per map and cache the evidence."""
        path = Path(path)
        if path in self.cache:
            return self.cache[path]
        self.steps, self._t0 = [], time.time()
        img = load_image(path)
        self._step(f"Loaded {path.name}: {img.shape[1]} x {img.shape[0]} px")
        self._step(f"Stage 2: asking {self.model} to read the map specification")
        spec = extract_spec(self.client, self.model, img)
        self._step(f"Specification: variable={spec.get('variable')!r}, region={spec.get('region')!r}, "
                   f"unit={spec.get('unit')!r}, legend={spec.get('legend_type')} {spec.get('legend_orientation', '')}, "
                   f"ticks={spec.get('legend_ticks') or '-'}, classes={spec.get('legend_classes') or '-'}")
        ref = load_reference(spec)
        groups = f", summarized by {len(ref['groups']):,} {ref['group_unit']} groups" if ref["groups"] is not None else ""
        self._step(f"Reference boundaries: {len(ref['units']):,} units{groups} (shapes and names only)")
        legend = spec.get("legend_type")
        if legend == "continuous":
            if ref["groups"] is not None:
                raise NotImplementedError("Color bars on census-tract maps are not supported yet")
            evidence = self._read_continuous(img, spec, ref)
        elif legend == "classed":
            if not spec.get("legend_classes"):
                raise NotImplementedError("The legend's class labels could not be read")
            if ref["groups"] is not None:
                evidence = self._read_classed_grouped(img, spec, ref)
            else:
                evidence = self._read_classed_units(img, spec, ref)
        else:
            raise NotImplementedError(f"Unsupported legend type: {legend!r}")
        evidence["spec"] = spec
        self._step(f"Done: table of {len(evidence['table']):,} rows")
        evidence["log"] = list(self.steps)
        self.cache[path] = evidence
        return evidence

    def _read_continuous(self, img, spec, ref):
        # A horizontal bar is read on the image turned on its side (transposed): the bar becomes vertical with its
        # tick marks beside it, and its left (low) end becomes the top.
        horizontal = str(spec.get("legend_orientation", "vertical")).lower().startswith("h")
        view = img.transpose(1, 0, 2) if horizontal else img
        bbox = spec.get("legend_bbox")
        if horizontal and bbox:
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        self._step(f"Stage 1: searching for the {'horizontal' if horizontal else 'vertical'} color bar")
        bar = find_colorbar(view, bbox)
        to_value, calibration = calibrate(view, bar, spec["legend_ticks"], low_end="top" if horizontal else "bottom")
        ramp_lab, ramp_values = sample_ramp(view, bar, to_value)
        if horizontal:  # back to the image's own (x, y)
            bar = (bar[1], bar[0], bar[3], bar[2])
        self._step(f"Color bar at pixels {tuple(int(v) for v in bar)}, calibrated from {calibration}; "
                   f"it spans {ramp_values.min():,.1f} to {ramp_values.max():,.1f}")
        pix, silhouette, iou, crs = self._align(img, ref["units"], [bar])
        markers = self._markers(img, silhouette, pix)
        table = continuous_table(measure(img, pix, ramp_lab, ramp_values), pix, markers)
        read = table.value.notna()
        self._step(f"Read {int(read.sum()):,} of {len(table):,} units against the color bar "
                   f"({int((table.confidence == 'low').sum())} low confidence); values {table.value.min():,.0f} "
                   f"to {table.value.max():,.0f}")
        unit = spec.get("unit", "region")
        notes = (f'Measured table: one row per {unit}. "value" is {spec.get("variable", "the mapped value")} read from '
                 'the map\'s color scale (approximate).\nrank 1 = highest value. share_pct = share of the total. '
                 'x_frac: 0 = west edge, 1 = east edge; y_frac: 0 = south edge, 1 = north edge.\n'
                 f'"marker" is the letter label drawn on that {unit}, if any. Names come from reference boundaries, not from the map.')
        cols = ["name", "value", "confidence", "rank", "share_pct", "x_frac", "y_frac", "marker"]
        return {"table": table, "table_notes": notes, "table_columns": cols, "markers": markers,
                "outline": pix[["name", "geometry"]], "image_size": (img.shape[1], img.shape[0]),
                "diagnostics": {"legend_px": bar, "calibration": calibration, "georef_iou": round(float(iou), 3),
                                "georef_crs": crs_label(crs), "low_confidence_rows": int((table.confidence == "low").sum())}}

    def _align(self, img, fit, exclude, footprint=None):
        """Stage 3 alignment, logged: reference boundaries placed on the image."""
        self._step(f"Stage 3: aligning {len(fit):,} reference outlines to the map ({len(CANDIDATE_CRS) + 2} projections)")
        iou, crs, params, silhouette = georeference(img, fit, exclude, footprint=footprint)
        quality = "good" if iou >= 0.9 else "check the overlay" if iou >= 0.8 else "poor, results unreliable"
        self._step(f"Best fit: {crs_label(crs)}, overlap (IoU) {iou:.3f}, {quality}")
        self._fit = (crs, params)
        return to_pixels(fit, crs, params), silhouette, iou, crs

    def _markers(self, img, silhouette, pix):
        found = find_markers(img, silhouette)
        if not found:
            self._step("No letter markers on the map")
            return {}
        self._step(f"Found {len(found)} marker(s); asking {self.model} to read their letters")
        markers = read_markers(self.client, self.model, found, pix)
        self._step("Markers: " + ", ".join(f"{k} -> {v}" for k, v in markers.items()))
        return markers

    def _classed_legend(self, img, spec):
        """Legend classes sorted from lowest to highest value, their CIELAB colors, and the legend area to leave
        out when fitting boundaries (patches plus their labels and frame, which can sit on top of the map)."""
        labels = spec["legend_classes"]
        self._step(f"Stage 1: searching for {len(labels)} legend patches")
        patches = find_legend_patches(img, spec.get("legend_bbox"), len(labels))
        paired = sorted(zip(parse_classes(labels), patches), key=lambda p: p[0][1])  # legend order -> value order
        self._step("Legend patches: " + "; ".join(
            f"{p[0][0]} = rgb{tuple(int(v) for v in p[1][1])}" for p in paired))
        boxes = np.array([p[1][0] for p in paired])
        x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
        x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
        pw, ph = boxes[0, 2] - boxes[0, 0], boxes[0, 3] - boxes[0, 1]
        vertical = (y1 - y0) > (x1 - x0)
        if vertical:  # labels to the right, legend title above
            area = [(x0 - pw, y0 - 3 * ph, x1 + 7 * pw, y1 + ph)]
        else:  # labels below
            area = [(x0 - pw, y0 - 3 * ph, x1 + pw, y1 + 4 * ph)]
        h, w, _ = img.shape
        if spec.get("legend_bbox"):
            bx0, by0, bx1, by1 = spec["legend_bbox"]
            area.append((int(bx0 * w), int(by0 * h), int(bx1 * w), int(by1 * h)))
        area = [(max(0, int(a)), max(0, int(b)), min(w - 1, int(c)), min(h - 1, int(d))) for a, b, c, d in area]
        return [p[0] for p in paired], rgb_to_lab(np.array([p[1][1] for p in paired])), area

    def _read_classed_units(self, img, spec, ref):
        """Classed legend on a unit map (e.g. counties): each unit gets the class most of its pixels match."""
        classes, class_lab, boxes = self._classed_legend(img, spec)
        pixclass = classify_pixels(img, class_lab)
        pix, silhouette, iou, crs = self._align(img, ref["units"], boxes, footprint=pixclass >= 0)
        markers = self._markers(img, silhouette, pix)
        hidden = pixclass.copy()  # units under the legend are unreadable, not read from the legend's own colors
        for x0, y0, x1, y1 in boxes:
            hidden[y0:y1 + 1, x0:x1 + 1] = -1
        table = classed_unit_table(hidden, pix, classes, markers)
        unit = spec.get("unit", "unit")
        per_class = table["class"].value_counts().reindex([c[0] for c in classes] + ["unreadable"], fill_value=0)
        counts = "; ".join(f"{label}: {n}" for label, n in per_class.items() if label != "unreadable" or n)
        self._step(f"Classified {len(table):,} units by majority pixel color ({int((table.confidence == 'low').sum())} "
                   f"low confidence): {counts}")
        notes = (f'Measured table: one row per {unit}. "class" is the legend class most of the {unit}\'s pixels match '
                 f'(class_number 1 = lowest class). pixel_share = share of its pixels in that class; a low share means '
                 'mixed colors, labels or borders, so treat that row with care. Exact values are not shown on a '
                 'classed map.\nx_frac: 0 = west edge, 1 = east edge; y_frac: 0 = south edge, 1 = north edge. '
                 f'"marker" is the letter label drawn on that {unit}, if any. Names come from reference boundaries, '
                 f'not from the map.\n{unit.capitalize()} count per class: {counts}.')
        cols = ["name", "class", "class_number", "pixel_share", "confidence", "x_frac", "y_frac", "marker"]
        return {"table": table, "table_notes": notes, "table_columns": cols, "markers": markers,
                "outline": pix[["name", "geometry"]], "image_size": (img.shape[1], img.shape[0]),
                "diagnostics": {"legend_classes": [c[0] for c in classes], "georef_iou": round(float(iou), 3),
                                "georef_crs": crs_label(crs), "units_read": int((table["class"] != "unreadable").sum()),
                                "units_too_small": 0, "units_no_color": int((table["class"] == "unreadable").sum()),
                                "low_confidence_rows": int((table.confidence == "low").sum())}}

    def _read_classed_grouped(self, img, spec, ref):
        classes, class_lab, boxes = self._classed_legend(img, spec)
        pixclass = classify_pixels(img, class_lab)
        outline, silhouette, iou, crs = self._align(img, ref["groups"], boxes, footprint=pixclass >= 0)
        params = self._fit[1]
        self._step(f"Reading {len(ref['units']):,} {spec.get('unit', 'unit')}s at a point inside each")
        units = ref["units"].to_crs(crs)
        pts = units.geometry.representative_point().apply(lambda g: affine_transform(g, params))
        cls = sample_classes(pixclass, np.round(pts.x.values).astype(int), np.round(pts.y.values).astype(int))
        # a unit smaller than one pixel has no pixel of its own; its sampled color belongs to a neighbor
        area_px = units.geometry.area.values * abs(params[0] * params[3])
        cls[area_px < 1] = TOO_SMALL
        self._step(f"Read {int((cls >= 0).sum()):,} units; {int((cls == TOO_SMALL).sum()):,} smaller than one pixel "
                   f"(left out); {int((cls == -1).sum()):,} with no legend color")
        table = grouped_class_table(cls, units, ref["groups"], classes)
        unit, group = spec.get("unit", "unit"), ref["group_unit"]
        notes = (f"Measured table: one row per {group}, from {len(units):,} reference {unit} boundaries "
                 f"(shapes only, no values), each read at a point inside it.\n"
                 f'"pct <class>" = percent of the {group}\'s READ {unit}s in that legend class. units_read = {unit}s whose '
                 f'color was read; units_too_small = {unit}s smaller than one pixel in this image (mostly dense urban '
                 f'areas), which cannot be read and are left out of the percentages; units_no_color = no legend color '
                 '(e.g. no data).\n'
                 'mean_class = average class number of read units (1 = lowest class); exact values are not shown on a '
                 f'classed map, so use mean_class to compare average levels between {group}s. rank_mean_class 1 = highest.\n'
                 f'Row "ALL" = the whole map. Names come from reference boundaries, not from the map. If many units are '
                 'too small, say that the percentages describe only the readable units.')
        # outline: the groups (states or counties), since tens of thousands of tract outlines are too heavy to draw
        outline = outline.rename(columns={"group_name": "name"})[["name", "geometry"]]
        return {"table": table, "table_notes": notes, "table_columns": list(table.columns), "markers": {},
                "outline": outline, "image_size": (img.shape[1], img.shape[0]),
                "diagnostics": {"legend_classes": [c[0] for c in classes], "georef_iou": round(float(iou), 3),
                                "georef_crs": crs_label(crs), "units_read": int((cls >= 0).sum()),
                                "units_too_small": int((cls == TOO_SMALL).sum()),
                                "units_no_color": int((cls == -1).sum())}}

    def _evidence_parts(self, ev):
        spec = {k: v for k, v in ev["spec"].items() if k not in ("legend_bbox", "markers")}
        notes = "\n".join(f"Cartographic note: {text}" for key, text in KNOWLEDGE.items() if ev["spec"].get(key))
        return {"spec": json.dumps(spec), "table_notes": ev["table_notes"],
                "table": ev["table"][ev["table_columns"]].to_csv(index=False), "notes": notes}

    def _ask_json(self, prompt):
        response = self.client.responses.create(model=self.model, input=prompt,
                                                text={"format": {"type": "json_object"}})
        return parse_json(response.output_text)

    def answer(self, path, question, format_text):
        ev = self.read(path)
        start = time.time()
        self._note(f"Answering {question!r} with {self.model} from the {len(ev['table']):,}-row measured table")
        reply = self._ask_json(ANSWER_PROMPT.format(**self._evidence_parts(ev), question=question, format=format_text))
        reply["evidence_check"] = check_evidence(reply.get("evidence"), ev["table"])
        check = reply["evidence_check"]
        self._note(f"Answer {reply.get('answer')!r} in {time.time() - start:.1f}s; cited {reply.get('evidence') or 'no rows'}"
                   + ("" if check["verified"] else f"; not in the table: {check['missing'] or 'nothing cited'}"))
        return reply

    def interpret(self, path):
        """Stage 5 (first version): an overview whose every claim must cite rows of the measured table."""
        ev = self.read(path)
        start = time.time()
        self._note(f"Writing an interpretation with {self.model} from the {len(ev['table']):,}-row measured table")
        reply = self._ask_json(INTERPRET_PROMPT.format(**self._evidence_parts(ev)))
        findings = reply.get("findings", [])
        for finding in findings:
            finding["evidence_check"] = check_evidence(finding.get("evidence"), ev["table"])
        unverified = sum(not f["evidence_check"]["verified"] for f in findings)
        self._note(f"Interpretation in {time.time() - start:.1f}s: {len(findings)} findings"
                   + (f", {unverified} citing rows that are not in the table" if unverified else ", all citations checked"))
        return reply


def evidence_to_json(ev, outline_tolerance=0.5):
    """Measured evidence as plain JSON, so a session can be saved and answered from later without re-measuring.
    The fitted outlines are simplified to sub-pixel tolerance to keep the file small."""
    outline = ev["outline"]
    return {
        "spec": ev["spec"], "table": json.loads(ev["table"].to_json(orient="split")), "table_notes": ev["table_notes"],
        "table_columns": ev["table_columns"], "markers": ev.get("markers", {}), "log": ev.get("log", []),
        "diagnostics": json.loads(json.dumps(ev["diagnostics"], default=str)), "image_size": list(ev["image_size"]),
        "outline": [{"name": str(n), "wkt": g.simplify(outline_tolerance, preserve_topology=False).wkt}
                    for n, g in zip(outline["name"], outline.geometry)],
    }


def evidence_from_json(data):
    """Inverse of evidence_to_json."""
    from shapely import wkt

    table = pd.DataFrame(data["table"]["data"], columns=data["table"]["columns"], index=data["table"]["index"])
    outline = gpd.GeoDataFrame({"name": [o["name"] for o in data["outline"]]},
                               geometry=[wkt.loads(o["wkt"]) for o in data["outline"]])
    return {**data, "table": table, "outline": outline, "image_size": tuple(data["image_size"])}


INTERPRET_PROMPT = """Write an interpretation of a thematic map for a GIS researcher. Use ONLY the evidence below, which code measured from the map image.

Map specification: {spec}

{table_notes}
{table}
{notes}
Reply in JSON:
{{"summary": "2-3 sentences: what the map shows and its main pattern",
  "findings": [{{"claim": "one sentence", "evidence": [names of the rows it rests on]}}],
  "caveats": ["limits of what this map or the measurement can show"]}}
Give 3-6 findings: the highest and lowest units, regional patterns or clusters, and notable contrasts. Every finding must cite at least one row."""


def check_evidence(cited, table):
    """Split cited row names into ones that exist in the measured table and ones that do not."""
    names = {str(n).strip().lower() for n in table["name"]}
    cited = [str(c) for c in (cited or [])]
    found = [c for c in cited if c.strip().lower() in names]
    return {"verified": bool(cited) and len(found) == len(cited), "found": found,
            "missing": [c for c in cited if c not in found]}


# ---------------------------------------------------------------- model-only fallback for maps the framework cannot measure

MODEL_ONLY_INTERPRET = """You are an expert cartographer. Interpret this map for a GIS researcher.
Reply in JSON: {"summary": "2-3 sentences", "findings": [{"claim": "one sentence"}], "caveats": ["..."]}.
Give 3-6 findings. Only state what is visible in the map; say "unclear" where something cannot be read."""

MODEL_ONLY_ANSWER = """Answer the question about this map. Only use what is visible in the map; say if it cannot be read.
Question: {question}
Reply in JSON: {{"answer": ..., "explanation": "one sentence"}}."""


def model_only_interpret(client, model, path):
    return vlm_json(client, model, MODEL_ONLY_INTERPRET, [to_data_url(load_image(path))])


def model_only_answer(client, model, path, question):
    return vlm_json(client, model, MODEL_ONLY_ANSWER.format(question=question), [to_data_url(load_image(path))])


# ---------------------------------------------------------------- ablation: same model plus a code tool, no structure

TOOLS_PROMPT = """Answer the question about this map. You can run Python on the attached image file ({filename}) \
to inspect pixels, crop, zoom or measure.

Question: {question}
{format}
Reply in JSON: {{"answer": ..., "source": "map" or "outside knowledge", "explanation": "one sentence"}}.
Use "outside knowledge" if any part of the answer is not shown on the map."""


def ask_with_code_tool(client, model, path, question, format_text):
    path = Path(path)
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    try:
        response = client.responses.create(
            model=model,
            tools=[{"type": "code_interpreter", "container": {"type": "auto", "file_ids": [uploaded.id]}}],
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": TOOLS_PROMPT.format(filename=path.name, question=question, format=format_text)},
                {"type": "input_image", "image_url": to_data_url(load_image(path))},
            ]}],
        )
        return parse_json(response.output_text)
    finally:
        client.files.delete(uploaded.id)
