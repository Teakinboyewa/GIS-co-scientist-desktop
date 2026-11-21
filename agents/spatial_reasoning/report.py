"""
What the proposed method reports: which assumptions were stated, tested,
supported or contradicted, what the plan changed in response, and what
remains unresolved. Written deterministically from the run's records, so the
manuscript section cannot misstate the evidence.

  spatial_assumptions.md   the manuscript section (inserted after Results)
  ctx["artifacts"]["spatial_assumptions_summary"]
                           compact version given to the section writers
"""

import os

from agents.spatial_reasoning.assumptions import CHECK_TITLES, FIELD_TITLES, UNIT_CHECK, locked
from utils.config import resolve_output_dir

REPORT_FILE = "spatial_assumptions.md"
SECTION_TITLE = "## Spatial Assumptions and Diagnostics"


def stated_text(field, a):
    if field == UNIT_CHECK:
        return f"question about {a.get('question_unit')} units; data on {a.get('data_unit')} units"
    if field == "scale_and_zoning":
        holds = "results hold" if a["assumes_results_hold_across_aggregations"] else "results may change"
        return f"{a['analysis_unit']}; {holds} under other aggregations"
    if field == "independence":
        return "independent observations" if a["assumes_independent_observations"] else "dependent observations"
    return f"{a['assumes']} relationship"


def evidence_summary(evidence, statuses):
    """Short markdown for the chat stream."""
    lines = []
    for mid, m in evidence.items():
        r = m["results"]
        dep = r.get("residual_dependence", {})
        covers = f", full model for {', '.join(m['hypotheses'])}" if m.get("hypotheses") else ""
        lines.append(f"**{mid}** ({m['outcome']} ~ {', '.join(m['predictors'])}{covers})")
        if "error" in dep:
            lines.append(f"- residual dependence: failed ({dep['error']})")
        else:
            p = dep["by_weights"][0]
            lines.append(f"- residual Moran's I = {p['moran_I']:.3f} (p = {p['moran_p']:.3f}); "
                         f"suggested specification: {dep['suggested_specification']}")
        agg = r.get("aggregation_sensitivity", {})
        lines.append("- aggregation: " + ("failed" if "error" in agg else
                     "scale-sensitive" if agg.get("scale_sensitive") else "stable"))
        lv = r.get("local_variation", {})
        if "error" in lv:
            lines.append("- local variation: failed")
        elif lv.get("skipped"):
            lines.append(f"- local variation: not tested ({lv['skipped']})")
        else:
            bws = ", ".join(f"{p} {c['bandwidth']:.0f}/{lv['n']}" for p, c in lv["covariates"].items())
            lines.append(f"- MGWR bandwidths: {bws}")
        for hid in m.get("hypotheses", []):
            for field, s in statuses.get(hid, {}).items():
                lines.append(f"- {hid} · {CHECK_TITLES[field]}: **{s['status']}** — {s['why']}")
    return "\n".join(lines) + "\n"


def _section(ctx, plan):
    hyps = locked(ctx)
    g = ctx.get("artifacts", {}).get("spatial_grounding") or {}
    st = g.get("assumption_status") or {}
    out = [SECTION_TITLE, ""]
    mode = g.get("split_mode")
    where = {"replicate": "an independent realization of the data-generating process",
             "blocked_split": "a spatially blocked half of the study units"}.get(mode)
    if hyps:
        out.append("Before any analysis was planned, each hypothesis stated its assumptions about "
                   "scale and zoning, independence of observations, and whether the relationship is "
                   "global or local. The hypotheses were then fixed; the analysis plan could change "
                   "in response to evidence, but the hypotheses could not.")
    if where:
        out.append(f"\nEach assumption was tested with a pre-specified diagnostic on {where}, kept "
                   "separate from the data used for the confirmatory analysis. Diagnostics were run "
                   "on the full model for each outcome, with every predictor named in the hypotheses.")
    if hyps:
        out += ["", "| Hypothesis | Assumption | Stated | Status | Evidence |", "|---|---|---|---|---|"]
        for h in hyps:
            for field, a in h["assumptions"].items():
                s = st.get(h["id"], {}).get(field, {"status": "untested", "why": "not tested"})
                out.append(f"| {h['id']} | {FIELD_TITLES[field]} | {stated_text(field, a)} | "
                           f"{s['status']} | {s['why']} |")
            u = st.get(h["id"], {}).get(UNIT_CHECK)
            if u:
                out.append(f"| {h['id']} | {CHECK_TITLES[UNIT_CHECK]} | "
                           f"{stated_text(UNIT_CHECK, h['assumptions']['scale_and_zoning'])} | "
                           f"{u['status']} | {u['why']} |")
    elif g.get("evidence"):
        out += ["", evidence_summary(g["evidence"], {})]

    changes = [c for rev in g.get("revisions", []) for c in rev.get("steps_marked_exploratory", [])]
    out += ["", "### Plan changes made in response (exploratory)", ""]
    out += ([f"- Objective {c['objective']}, step {c['step']} ({c['status']}): {c['operation']} "
             f"— triggered by {c['triggered_by']}" for c in changes] or ["- None."])

    rounds = g.get("critic_rounds") or []
    if rounds:
        last = rounds[-1]
        out += ["", "### Spatial review", "",
                f"The spatial reviewer's final verdict was **{last['verdict']}** after "
                f"{len(rounds)} round(s)."]
        major = [i for i in last["issues"] if i["severity"] == "major"]
        minor = [i for i in last["issues"] if i["severity"] == "minor"]
        if major:
            out += ["Unresolved major issues:"] + [f"- [{i['kind']}] {i['detail']}" for i in major]
        if minor:
            out += ["Minor notes:"] + [f"- [{i['kind']}] {i['detail']}" for i in minor]

    unresolved = [f"{h['id']} — {FIELD_TITLES[f]} ({st.get(h['id'], {}).get(f, {}).get('status', 'untested')})"
                  for h in hyps for f in h["assumptions"]
                  if st.get(h["id"], {}).get(f, {}).get("status", "untested") in ("untested", "inconclusive")]
    out += ["", "### Assumptions that remain unresolved", ""]
    out += [f"- {u}" for u in unresolved] or ["- None."]
    lock = plan.get("hypothesis_lock") or {}
    if lock.get("untested_hypotheses"):
        out.append(f"- No objective tests: {', '.join(lock['untested_hypotheses'])}.")
    return "\n".join(out) + "\n"


def write(ctx, plan):
    text = _section(ctx, plan)
    with open(os.path.join(resolve_output_dir(ctx), REPORT_FILE), "w", encoding="utf-8") as f:
        f.write(text)
    ctx.setdefault("artifacts", {})["spatial_assumptions_section"] = text
    g = ctx["artifacts"].get("spatial_grounding") or {}
    ctx["artifacts"]["spatial_assumptions_summary"] = {
        "hypotheses": locked(ctx),
        "assumption_status": g.get("assumption_status") or {},
        "exploratory_plan_changes": [c for rev in g.get("revisions", [])
                                     for c in rev.get("steps_marked_exploratory", [])],
        "critic_final_verdict": (g.get("critic_rounds") or [{}])[-1].get("verdict"),
    }
    return text
