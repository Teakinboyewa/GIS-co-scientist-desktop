"""
The proposed method inside the research-plan graph.

  [state_assumptions] → generate → review → [spatial_grounding] → finalize
         (a)                                     (b) + (c)

Which bracketed nodes run, and what spatial_grounding does inside, follows
the condition's switches (conditions.py), so P and each ablation share one
code path. Baselines (B0–B3) get neither node and are untouched.

Three hooks connect this to research_plan_generator.py:
  graph_nodes(ctx)          the extra nodes for this condition
  planner_addendum(ctx)     text added to the planner's request
  on_finalize(ctx, plan)    restore the lock, write the assumptions report
"""

import json
import os

from agents.spatial_reasoning import assumptions, critic, diagnostics, exploration, report, revision
from agents.spatial_reasoning.conditions import condition_of
from utils.config import resolve_output_dir

DIAGNOSTICS_FILE = "spatial_diagnostics.json"
REVISIONS_FILE = "plan_revisions.json"
CRITIC_FILE = "spatial_critic.json"


def _emit(ctx, text, step="spatial_grounding"):
    cb = ctx.get("_stream_callback")
    if cb:
        cb(text)
    ev = ctx.get("_event_callback")
    if ev:
        ev({"type": "llm_update", "content": text.strip().splitlines()[0].strip("# "), "step": step})


def _save(ctx, name, obj):
    with open(os.path.join(resolve_output_dir(ctx), name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ── hooks ────────────────────────────────────────────────────────────

def graph_nodes(ctx):
    """[(name, fn, position)] with position 'before_plan' or 'before_finalize'."""
    c = condition_of(ctx)
    nodes = []
    if c["structured_assumptions"]:
        nodes.append(("state_assumptions", state_assumptions_node, "before_plan"))
    if c["diagnostics"] or c["critic"]:
        nodes.append(("spatial_grounding", spatial_grounding_node, "before_finalize"))
    return nodes


def planner_addendum(ctx):
    hyps = assumptions.locked(ctx)
    return assumptions.planner_instructions(hyps) if hyps else ""


def on_finalize(ctx, plan):
    hyps = assumptions.locked(ctx)
    if hyps:
        # The published reviewer drops hypothesis_ids; the planner's draft has them.
        draft = ctx.get("artifacts", {}).get("research_plan_first_draft")
        plan["hypothesis_lock"] = assumptions.enforce(plan, hyps, draft=draft)
    if hyps or ctx.get("artifacts", {}).get("spatial_grounding"):
        report.write(ctx, plan)
    return plan


# ── (a) ──────────────────────────────────────────────────────────────

def state_assumptions_node(state):
    ctx = state["ctx"]
    _emit(ctx, "\n\n## Stating hypotheses and spatial assumptions\n\n", "state_assumptions")
    hyps = assumptions.state_hypotheses(ctx)
    assumptions.lock(ctx, hyps)
    for h in hyps:
        a = h["assumptions"]
        s = a["scale_and_zoning"]
        _emit(ctx, f"- **{h['id']}** {h['statement']} — unit: {s['analysis_unit']} "
                   f"(question about {s.get('question_unit')} units, data on {s.get('data_unit')} units); "
                   f"independent observations: {a['independence']['assumes_independent_observations']}; "
                   f"relationship: {a['global_or_local']['assumes']}\n", "state_assumptions")
    _emit(ctx, "\nHypotheses locked.\n", "state_assumptions")
    return state


# ── (b) + (c) ────────────────────────────────────────────────────────

def full_models(items):
    """One diagnostic model per (dataset, outcome), with the union of the
    predictors of every hypothesis (or extracted model) that shares it.

    Diagnosing a reduced model mistakes an omitted, spatially smooth covariate
    for residual dependence: in the pilot (T07, T08) single-predictor
    hypotheses showed clustered residuals that vanished once the other
    predictor was included, and P fitted spatial models it did not need."""
    groups = {}
    for item in items:
        key = (item["dataset"], item["outcome"])
        g = groups.setdefault(key, {"id": f"M{len(groups) + 1}", "dataset": item["dataset"],
                                    "outcome": item["outcome"], "predictors": [], "hypotheses": []})
        g["predictors"] += [p for p in item["predictors"] if p not in g["predictors"]]
        g["hypotheses"].append(item["id"])
    return list(groups.values())


def _models(ctx, plan):
    hyps = assumptions.locked(ctx)
    return full_models(hyps or assumptions.extract_analysis_spec(ctx, plan))


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:          # a failed diagnostic is evidence too
        return {"error": f"{type(e).__name__}: {e}"}


def run_diagnostics(models, frames):
    evidence = {}
    for m in models:
        frame = frames[m["dataset"]]
        others = {sid: f for sid, f in frames.items()
                  if sid != m["dataset"] and hasattr(f, "crs") and "geometry" in f.columns}
        y, X = m["outcome"], m["predictors"]
        evidence[m["id"]] = {**m, "results": {
            "residual_dependence": _run(diagnostics.residual_dependence, frame, y, X),
            "aggregation_sensitivity": _run(diagnostics.aggregation_sensitivity, frame, y, X, others),
            "local_variation": _run(diagnostics.local_variation, frame, y, X),
        }, "requested": []}
    return evidence


def _weights_run(m):
    ran = [w["weights"] for w in m["results"]["residual_dependence"].get("by_weights", [])]
    ran += [w["weights"] for q in m["requested"] for w in q["result"].get("by_weights", [])]
    return set(ran)


def run_requests(evidence, frames, requests):
    """Run what the critic asked for; anything already run is noted, not rerun."""
    for r in requests:
        m = evidence[r["model"]]
        weights = r.get("weights") or "queen"
        if r["diagnostic"] == "residual_dependence" and weights not in _weights_run(m):
            result = _run(diagnostics.residual_dependence, frames[m["dataset"]], m["outcome"],
                          m["predictors"], weights=(weights,))
        else:
            result = {"note": f"{r['diagnostic']} ({weights if r['diagnostic'] == 'residual_dependence' else 'default'}) "
                              f"already ran for {r['model']}; see results above."}
        m["requested"].append({"request": r, "result": result})


def statuses(evidence, hyps):
    """Each hypothesis's assumptions, judged on the full model it belongs to."""
    model_of = {hid: mid for mid, m in evidence.items() for hid in m.get("hypotheses", [])}
    out = {}
    for h in hyps:
        results = {k: v for k, v in evidence.get(model_of.get(h["id"]), {}).get("results", {}).items()
                   if "error" not in v}
        out[h["id"]] = {}
        for field, assumption in h["assumptions"].items():
            status, why = diagnostics.assumption_status(field, assumption, results)
            out[h["id"]][field] = {"status": status, "why": why}
        unit = assumptions.unit_of_inference(h)
        if unit:
            out[h["id"]][assumptions.UNIT_CHECK] = {"status": unit[0], "why": unit[1]}
    return out


def evidence_text(evidence, hyps):
    st = statuses(evidence, hyps) if hyps else {}
    return json.dumps({"evidence": evidence, "assumption_status": st}, indent=1, default=str)


def spatial_grounding_node(state):
    ctx = state["ctx"]
    c = condition_of(ctx)
    plan = state["review_result"]
    hyps = assumptions.locked(ctx)
    sources = list(assumptions.input_datasets(ctx))
    evidence, split, frames, models, revisions = {}, None, None, [], []

    if c["diagnostics"]:
        _emit(ctx, "\n\n## Running spatial diagnostics on exploration data\n\n")
        split = exploration.prepare(ctx)
        frames = exploration.load_exploration(split)
        models = _models(ctx, plan)
        evidence = run_diagnostics(models, frames)
        _emit(ctx, report.evidence_summary(evidence, statuses(evidence, hyps) if hyps else {}))
        _emit(ctx, "\n## Revising the plan against the evidence\n\n")
        plan, rec = revision.revise(plan, hyps, evidence_text(evidence, hyps), sources)
        revisions.append({"trigger": "diagnostics", **rec})
    model_ids = list(evidence) or [h["id"] for h in hyps]

    rounds, previous, changes = [], None, None
    if c["critic"]:
        for r in range(critic.MAX_ROUNDS):
            _emit(ctx, f"\n## Spatial critic, round {r + 1}\n\n")
            verdict = critic.review(plan, hyps, evidence_text(evidence, hyps) if evidence else "",
                                    model_ids, c["diagnostics"], previous=previous, changes=changes,
                                    final=r == critic.MAX_ROUNDS - 1)
            rounds.append(verdict)
            _emit(ctx, f"Verdict: **{verdict['verdict']}**\n"
                       + "".join(f"- [{i['severity']}, {i['kind']}] {i['detail']}\n"
                                 for i in verdict["issues"]))
            if verdict["verdict"] == "approve" or r == critic.MAX_ROUNDS - 1:
                break
            if verdict["requested_diagnostics"]:
                run_requests(evidence, frames, verdict["requested_diagnostics"])
            # Only major issues change the plan; minor ones stay on record.
            plan, rec = revision.revise(plan, hyps, evidence_text(evidence, hyps) if evidence else "",
                                        sources, issues=critic.major(verdict))
            revisions.append({"trigger": f"critic round {r + 1}", **rec})
            previous, changes = critic.major(verdict), rec.get("steps_marked_exploratory", [])
        last = rounds[-1]
        _save(ctx, CRITIC_FILE, {"rounds": rounds, "final_verdict": last["verdict"],
                                 "unresolved_issues": critic.major(last) if last["verdict"] == "revise" else [],
                                 "minor_notes": [i for i in last["issues"] if i["severity"] == "minor"]})

    if evidence:     # saved last: critic requests may have added evidence
        _save(ctx, DIAGNOSTICS_FILE, {"split": split, "models": models, "evidence": evidence,
                                      "assumption_status": statuses(evidence, hyps)})
    if revisions:
        _save(ctx, REVISIONS_FILE, revisions)
    if split:
        exploration.point_analysis_at_confirmation(ctx, split)

    ctx.setdefault("artifacts", {})["spatial_grounding"] = {
        "split_mode": split["mode"] if split else None,
        "assumption_status": statuses(evidence, hyps) if (evidence and hyps) else {},
        "evidence": evidence, "revisions": revisions, "critic_rounds": rounds,
    }
    state["review_result"] = plan
    return state
