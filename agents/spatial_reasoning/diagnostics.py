"""
Component (b): the diagnostics each spatial assumption is tested with.

Fixed Python, never LLM-generated code, so the evidence the plan is revised
against is the same in every run and cannot carry a coding error. Each
diagnostic implements the procedure of its knowledge-base entry:

  independence      → residual_dependence     OLS residual Moran's I (permutation),
                                              LISA clusters, LM and robust LM
                                              lag/error, repeated under alternative
                                              weights
  scale_and_zoning  → aggregation_sensitivity re-estimate at coarser units and an
                                              alternative zoning; compare coefficients
  global_or_local   → local_variation         MGWR covariate bandwidths vs n, AICc

Decision thresholds are module constants, stated once, so the paper can
report them exactly.
"""

import contextlib
import io
import warnings

import numpy as np

ALPHA = 0.05                 # significance level for Moran's I, LISA and LM tests
ALPHA_LM_ONLY = 0.01         # LM evidence alone (Moran's I not significant) must reach this
Z_DIFF = 1.96                # a scale change also needs |b1 - b0| / sqrt(se0^2 + se1^2) above this
PERMUTATIONS = 999
LOCAL_BANDWIDTH_SHARE = 0.5  # covariate bandwidth below this share of n reads as local
AICC_MARGIN = 2.0            # MGWR must beat OLS AICc by at least this much
SCALE_RELATIVE_CHANGE = 0.5  # |Δβ| / |β| above this, or a sign flip, is scale-sensitive
MIN_AGGREGATE_UNITS = 20     # skip an aggregation that leaves fewer units than this
MIN_LOCAL_UNITS = 30         # MGWR is not fitted on fewer units than this
RANDOM_SEED = 12345          # permutation inference is reproducible run to run

WEIGHTS = ("queen", "knn4", "knn8", "knn12")
DIAGNOSTICS = {
    "residual_dependence": "independence",
    "aggregation_sensitivity": "scale_and_zoning",
    "local_variation": "global_or_local",
}


# ── data and weights ─────────────────────────────────────────────────

def model_frame(gdf, outcome, predictors):
    """Rows with complete numeric data for the model, as (frame, y, X)."""
    cols = [outcome, *predictors]
    frame = gdf[cols + ["geometry"]].copy()
    for c in cols:
        frame[c] = frame[c].astype(float)
    frame = frame.dropna(subset=cols)
    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()].reset_index(drop=True)
    y = frame[[outcome]].to_numpy()
    X = frame[predictors].to_numpy()
    return frame, y, X


def build_weights(gdf, kind="queen"):
    """Row-standardized weights. Queen islands are attached to their nearest
    neighbor so every unit has at least one neighbor; the count is reported."""
    from libpysal import weights as lw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "queen":
            w = lw.Queen.from_dataframe(gdf, use_index=False)
            islands = len(w.islands)
            if islands:
                w = lw.attach_islands(w, lw.KNN.from_dataframe(gdf, k=1))
        elif kind.startswith("knn") and kind[3:].isdigit():
            w = lw.KNN.from_dataframe(gdf, k=int(kind[3:]))
            islands = 0
        else:
            raise ValueError(f"Unknown weights {kind!r}; use one of {WEIGHTS}")
    w.transform = "r"
    return w, islands


def _ols(y, X):
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y.ravel(), rcond=None)
    resid = y.ravel() - Xc @ beta
    n, k = Xc.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(Xc.T @ Xc)))
    rss = resid @ resid
    # Full Gaussian log-likelihood, as mgwr computes it, so the AICc is
    # comparable with MGWR's. Dropping the constants put OLS ~550 units
    # lower and made every MGWR fit look worse.
    llf = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    aicc = -2 * llf + 2 * n * (k + 1) / max(n - k - 2, 1)
    return beta, se, resid, aicc


# ── independence ─────────────────────────────────────────────────────

def residual_dependence(gdf, outcome, predictors, weights=("queen", "knn8")):
    from esda.moran import Moran, Moran_Local
    import spreg

    frame, y, X = model_frame(gdf, outcome, predictors)
    out = {"diagnostic": "residual_dependence", "n": len(frame), "by_weights": []}
    _, _, resid, _ = _ols(y, X)
    for kind in weights:
        w, islands = build_weights(frame, kind)
        np.random.seed(RANDOM_SEED)
        mi = Moran(resid, w, permutations=PERMUTATIONS)
        lisa = Moran_Local(resid, w, permutations=PERMUTATIONS, seed=RANDOM_SEED)
        sig = lisa.p_sim < ALPHA
        clusters = {name: int(np.sum(sig & (lisa.q == q)))
                    for q, name in ((1, "high_high"), (2, "low_high"), (3, "low_low"), (4, "high_low"))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = spreg.OLS(y, X, w=w, spat_diag=True, name_y=outcome, name_x=list(predictors))
        lm = {name: {"stat": float(getattr(model, attr)[0]), "p": float(getattr(model, attr)[1])}
              for name, attr in (("lm_lag", "lm_lag"), ("lm_error", "lm_error"),
                                 ("robust_lm_lag", "rlm_lag"), ("robust_lm_error", "rlm_error"))}
        out["by_weights"].append({
            "weights": kind, "islands_attached": islands,
            "moran_I": float(mi.I), "moran_p": float(mi.p_sim),
            "lisa_significant_clusters": clusters, "lm_tests": lm,
        })
    out.update(dependence_decision(out["by_weights"]))
    return out


def dependence_decision(by_weights):
    """Detected if Moran's I is significant on the primary (first) weights,
    or the same LM test (lag or error) has p < ALPHA_LM_ONLY under every
    weights. Pilot T05: Moran's I p = 0.07 while LM-lag p < 0.001 under queen
    and knn8, on data built to be spatially structured, and the critic
    objected to "no dependence" twice. The stricter level keeps out T03,
    where varying coefficients alone gave LM-lag p = 0.02."""
    primary = by_weights[0]
    by_moran = primary["moran_p"] < ALPHA
    by_lm = [t for t in ("lm_lag", "lm_error")
             if all(r["lm_tests"][t]["p"] < ALPHA_LM_ONLY for r in by_weights)]
    detected = by_moran or bool(by_lm)

    def here(r):
        return r["moran_p"] < ALPHA or any(r["lm_tests"][t]["p"] < ALPHA_LM_ONLY
                                           for t in ("lm_lag", "lm_error"))
    spec = "none"
    if detected:
        lag, err = primary["lm_tests"]["robust_lm_lag"], primary["lm_tests"]["robust_lm_error"]
        sig = [(name, t) for name, t in (("lag", lag), ("error", err)) if t["p"] < ALPHA]
        spec = max(sig, key=lambda s: s[1]["stat"])[0] if sig else "unclear"
    return {"dependence_detected": bool(detected),
            "detected_by": "moran" if by_moran else (by_lm[0] if by_lm else None),
            "robust_across_weights": all(here(r) == detected for r in by_weights),
            "suggested_specification": spec}


# ── scale and zoning ─────────────────────────────────────────────────

def _grid_ids(frame, cells_per_axis, offset=0.0):
    c = frame.geometry.centroid
    minx, miny, maxx, maxy = frame.total_bounds
    sx = (maxx - minx) / cells_per_axis or 1.0
    sy = (maxy - miny) / cells_per_axis or 1.0
    gx = np.floor((c.x - minx) / sx + offset).astype(int)
    gy = np.floor((c.y - miny) / sy + offset).astype(int)
    return gx.astype(str) + "_" + gy.astype(str)


def _zone_ids_from_layer(frame, zones):
    import geopandas as gpd
    zones = zones.to_crs(frame.crs) if zones.crs is not None and frame.crs is not None else zones
    pts = gpd.GeoDataFrame(geometry=frame.geometry.representative_point(), crs=frame.crs)
    joined = gpd.sjoin(pts, zones[["geometry"]].reset_index(names="_zone"),
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["_zone"].reindex(frame.index)


def aggregation_sensitivity(gdf, outcome, predictors, other_layers=None):
    """Re-estimate OLS after aggregating units (unweighted means) to coarser
    grids, a shifted zoning of the same size, and any coarser polygon layers
    supplied with the data; compare each predictor's coefficient with the
    original."""
    frame, y, X = model_frame(gdf, outcome, predictors)
    n = len(frame)
    beta0, se0, _, _ = _ols(y, X)
    cols = [outcome, *predictors]
    base_cells = max(int(np.sqrt(n / 4)), 2)
    schemes = [("grid_coarse", _grid_ids(frame, base_cells)),
               ("grid_coarse_shifted", _grid_ids(frame, base_cells, offset=0.5)),
               ("grid_coarser", _grid_ids(frame, max(int(np.sqrt(n / 16)), 2)))]
    for name, layer in (other_layers or {}).items():
        if 0 < len(layer) < n:
            schemes.append((f"layer:{name}", _zone_ids_from_layer(frame, layer)))

    # A change counts only if it is also larger than sampling noise. Pilot
    # T05: population, a coefficient near zero, moved by 290% and flagged the
    # data as scale-sensitive; and on tasks with no scale effect (T02, T09,
    # T10) the coarsest grid, 23-36 units, moved a coefficient past 50% by
    # chance (difference z = 1.3-1.9).
    out = {"diagnostic": "aggregation_sensitivity", "n": n,
           "original": {p: {"coef": float(b), "se": float(s)}
                        for p, b, s in zip(predictors, beta0[1:], se0[1:])},
           "aggregations": []}
    any_sensitive = False
    for name, ids in schemes:
        agg = frame[cols].groupby(np.asarray(ids)).mean().dropna()
        entry = {"scheme": name, "units": int(len(agg))}
        if len(agg) < MIN_AGGREGATE_UNITS:
            entry["skipped"] = f"fewer than {MIN_AGGREGATE_UNITS} units"
            out["aggregations"].append(entry)
            continue
        b, s, _, _ = _ols(agg[[outcome]].to_numpy(), agg[list(predictors)].to_numpy())
        entry["coefficients"] = {}
        for p, b0, s0, b1, s1 in zip(predictors, beta0[1:], se0[1:], b[1:], s[1:]):
            rel = abs(b1 - b0) / abs(b0) if b0 else float("inf")
            flip = bool(np.sign(b1) != np.sign(b0))
            z = float(abs(b1 - b0) / np.sqrt(s0 ** 2 + s1 ** 2)) if (s0 or s1) else 0.0
            sensitive = (flip or rel > SCALE_RELATIVE_CHANGE) and z > Z_DIFF
            any_sensitive |= sensitive
            entry["coefficients"][p] = {"coef": float(b1), "se": float(s1),
                                        "relative_change": float(rel), "sign_flip": flip,
                                        "z_difference": z, "sensitive": bool(sensitive)}
        out["aggregations"].append(entry)
    out["scale_sensitive"] = bool(any_sensitive)
    out["tested"] = any("coefficients" in a for a in out["aggregations"])
    return out


# ── global or local ──────────────────────────────────────────────────

def local_variation(gdf, outcome, predictors):
    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW

    frame, y, X = model_frame(gdf, outcome, predictors)
    n = len(frame)
    if n < MIN_LOCAL_UNITS:
        return {"diagnostic": "local_variation", "n": n,
                "skipped": f"fewer than {MIN_LOCAL_UNITS} units"}
    zy = (y - y.mean()) / y.std()
    zX = (X - X.mean(0)) / X.std(0)
    coords = np.column_stack([frame.geometry.centroid.x, frame.geometry.centroid.y])
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")      # mgwr prints progress bars and warnings
        selector = Sel_BW(coords, zy, zX, multi=True)
        bandwidths = selector.search()
        results = MGWR(coords, zy, zX, selector).fit()
    *_, ols_aicc = _ols(zy, zX)
    params = results.params
    out = {"diagnostic": "local_variation", "n": n,
           "mgwr_aicc": float(results.aicc), "ols_aicc": float(ols_aicc),
           "intercept_bandwidth": float(bandwidths[0]), "covariates": {}}
    better_fit = results.aicc < ols_aicc - AICC_MARGIN
    any_local = False
    for j, p in enumerate(predictors, start=1):
        bw = float(bandwidths[j])
        local = bool(bw < LOCAL_BANDWIDTH_SHARE * n and better_fit)
        any_local |= local
        q = np.percentile(params[:, j], [5, 50, 95])
        out["covariates"][p] = {
            "bandwidth": bw, "bandwidth_share_of_n": bw / n, "local": local,
            "standardized_coef_p5_p50_p95": [float(v) for v in q],
            "sign_varies": bool(q[0] < 0 < q[2]),
        }
    out["local_variation_detected"] = bool(any_local)
    return out


# ── assumption verdicts ──────────────────────────────────────────────

def assumption_status(field, assumption, results):
    """'supported', 'contradicted', 'inconclusive' or 'untested' for one
    stated assumption, from the diagnostics that ran."""
    if field == "independence":
        r = results.get("residual_dependence")
        if not r:
            return "untested", "residual dependence was not tested"
        assumed_independent = assumption.get("assumes_independent_observations")
        detected = r["dependence_detected"]
        primary = r["by_weights"][0]
        why = f"residual Moran's I = {primary['moran_I']:.3f} (p = {primary['moran_p']:.3f}, {primary['weights']} weights)"
        if r.get("detected_by") in ("lm_lag", "lm_error"):
            name = "LM-lag" if r["detected_by"] == "lm_lag" else "LM-error"
            why += f"; {name} p < {ALPHA_LM_ONLY} under every weights tested"
        return ("contradicted" if detected == bool(assumed_independent) else "supported"), why
    if field == "scale_and_zoning":
        r = results.get("aggregation_sensitivity")
        if not r or not r.get("tested"):
            return "untested", "no aggregation left enough units to compare"
        sensitive = r["scale_sensitive"]
        holds = assumption.get("assumes_results_hold_across_aggregations")
        why = "coefficients changed materially under aggregation" if sensitive else \
              "coefficients were stable across the aggregations tested"
        return ("contradicted" if sensitive == bool(holds) else "supported"), why
    if field == "global_or_local":
        r = results.get("local_variation")
        if not r:
            return "untested", "local variation was not tested"
        if r.get("skipped"):
            return "untested", f"local variation was not tested ({r['n']} units, {r['skipped']})"
        local = r["local_variation_detected"]
        dep = (results.get("residual_dependence") or {}).get("dependence_detected")
        if local and dep:
            return "inconclusive", ("MGWR suggests local variation, but residual spatial dependence "
                                    "is also present and the two can be confused")
        assumed_local = assumption.get("assumes") == "local"
        locals_ = [p for p, c in r["covariates"].items() if c["local"]]
        why = (f"local bandwidths for {', '.join(locals_)}" if locals_ else
               "all covariate bandwidths near the number of units")
        return ("supported" if local == assumed_local else "contradicted"), why
    raise ValueError(f"Unknown assumption field {field!r}")
