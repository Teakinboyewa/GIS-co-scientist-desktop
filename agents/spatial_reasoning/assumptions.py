"""
Component (a): structured spatial assumptions and the hypothesis lock.

Before any planning, each hypothesis is stated with required fields for the
three assumptions the study covers: scale and zoning, independence of
observations, and a global or local relationship. A hypothesis missing a
field is sent back; planning does not start until every field is filled.

Scale and zoning also names the unit the research question is about and the
unit one row of the data describes. No diagnostic can detect an ecological
fallacy (pilot T05: tract and district slopes agree, and the sign reverses
only at the individual level, which is not in the data), so the check is a
comparison of the two stated units: unit_of_inference().
The hypotheses are then locked: later agents (plan review, revision, critic)
cannot change them. A user may still edit them at the approval checkpoint;
that edit is recorded, not blocked.

The ablation without structure still needs the model's variables for the
diagnostics; extract_analysis_spec() reads them from the plan instead.
"""

import hashlib
import json
import os
import time

from agents.spatial_reasoning import knowledge_base as kb
from agents.spatial_reasoning.llm import InvalidResponse, ask_json
from utils.config import resolve_output_dir

LOCK_FILE = "hypotheses_locked.json"
CHECKPOINT_FILE = "hypotheses_checkpoint_edits.json"
MAX_HYPOTHESES = 3
MIN_JUSTIFICATION = 15

FIELD_TITLES = {
    "scale_and_zoning": "Scale and zoning",
    "independence": "Independence of observations",
    "global_or_local": "Global or local relationship",
}
UNIT_CHECK = "unit_of_inference"
CHECK_TITLES = {**FIELD_TITLES, UNIT_CHECK: "Unit of inference"}
UNITS = ("individual", "area")


# ── data the hypotheses may use ──────────────────────────────────────

def input_datasets(ctx):
    """The uploaded datasets: {source_id: {"path", "columns", "shape"}}."""
    out = {}
    for src in ctx.get("artifacts", {}).get("data_registry", []) or []:
        if src.get("source") == "data_audit_record" and src.get("file"):
            out[src["file"]] = {"path": src.get("path"), "shape": src.get("shape"),
                                "columns": src.get("columns") or {}}
    return out


def _numeric(dtype):
    return any(t in str(dtype).lower() for t in ("float", "int"))


def _check_variables(entry, datasets, where):
    ds = entry.get("dataset")
    if ds not in datasets:
        raise InvalidResponse(f"{where}: dataset {ds!r} is not one of {sorted(datasets)}")
    cols = datasets[ds]["columns"]
    outcome, predictors = entry.get("outcome"), entry.get("predictors")
    if not isinstance(predictors, list) or not predictors:
        raise InvalidResponse(f"{where}: 'predictors' must be a non-empty list of column names")
    for c in [outcome, *predictors]:
        if c not in cols:
            raise InvalidResponse(f"{where}: column {c!r} is not in dataset {ds!r}")
        if not _numeric(cols[c]):
            raise InvalidResponse(f"{where}: column {c!r} is not numeric ({cols[c]})")
    if outcome in predictors:
        raise InvalidResponse(f"{where}: the outcome cannot also be a predictor")


# ── (a) structured hypotheses ────────────────────────────────────────

SYSTEM = """You are a geospatial research methodologist. Before any analysis is planned, state the
hypotheses for the research question, and for each one state its spatial assumptions explicitly.

Rules:
- At most {max_h} hypotheses, each falsifiable, with an expected direction.
- Each hypothesis names one dataset, one numeric outcome column and one or more numeric predictor
  columns, exactly as they appear in the data.
- List every covariate the question asks to examine or account for in each hypothesis's predictors.
  Hypotheses differ in the claim they test, not in which covariates the model leaves out: a model
  that omits a spatially patterned covariate shows clustered residuals for that reason alone.
- For each field, state the assumption the analysis would rest on if no diagnostic were run, and
  commit to it. Spatial data alone is not a reason to assume dependent observations or
  aggregation-sensitive results: claim either only when you can name the mechanism (e.g.
  spillover between neighbors, a shared unmeasured regional cause, aggregation of dissimilar
  units) and name it in the justification.
- In scale_and_zoning, question_unit is the unit the research question's claim is about, read from
  the question itself, not from the hypothesis: a question about residents, people, households,
  students or other individuals is about "individual" units even when the hypothesis is stated
  for zones. data_unit is what one row of the dataset describes: "area" for zone, tract or grid
  aggregates, "individual" for records of single people, households or parcels.

Spatial guidance:
{guidance}

Return JSON only:
{{
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "string",
      "dataset": "source_id",
      "outcome": "column",
      "predictors": ["column", ...],
      "assumptions": {{
        "scale_and_zoning": {{
          "analysis_unit": "the unit the hypothesis is about and the data are measured on",
          "question_unit": "individual" | "area",
          "data_unit": "individual" | "area",
          "assumes_results_hold_across_aggregations": true | false,
          "justification": "string"
        }},
        "independence": {{
          "assumes_independent_observations": true | false,
          "justification": "string"
        }},
        "global_or_local": {{
          "assumes": "global" | "local",
          "justification": "string"
        }}
      }}
    }}
  ]
}}"""


def field_guidance():
    lines = []
    for field, title in FIELD_TITLES.items():
        lines.append(f"{title}:")
        for e in kb.entries():
            if e["assumption_field"] == field:
                lines.append(f"- {e['title']}: when {kb._clean(e['condition'])[0].lower()}"
                             f"{kb._clean(e['condition'])[1:]} {kb._clean(e['implication'])}")
    return "\n".join(lines)


def _user_prompt(ctx, datasets):
    rq = ctx.get("rq", "")
    rb = ctx.get("artifacts", {}).get("rq_breakdown", {}) or {}
    scale = rb.get("scale") or {}
    data = [f"- {name}: shape {d['shape']}; columns: "
            + ", ".join(f"{c} ({t})" for c, t in d["columns"].items() if c != "geometry")
            for name, d in datasets.items()]
    return (f"Research question: {rq}\n\n"
            f"Intent: {rb.get('intent', '')}\nSpatial scale: {scale.get('spatial_scale', 'N/A')}\n\n"
            "Datasets:\n" + "\n".join(data) + "\n\nState the hypotheses and their assumptions.")


def validate_hypotheses(obj, datasets):
    hyps = obj.get("hypotheses")
    if not isinstance(hyps, list) or not hyps:
        raise InvalidResponse("'hypotheses' must be a non-empty list")
    if len(hyps) > MAX_HYPOTHESES:
        raise InvalidResponse(f"at most {MAX_HYPOTHESES} hypotheses")
    seen = set()
    for i, h in enumerate(hyps, start=1):
        where = f"hypothesis {h.get('id') or i}"
        hid = h.get("id")
        if not isinstance(hid, str) or not hid or hid in seen:
            raise InvalidResponse(f"{where}: needs a unique string 'id' such as 'H{i}'")
        seen.add(hid)
        if len(str(h.get("statement", "")).strip()) < MIN_JUSTIFICATION:
            raise InvalidResponse(f"{where}: 'statement' is missing or too short")
        _check_variables(h, datasets, where)
        a = h.get("assumptions") or {}
        checks = {
            "scale_and_zoning": [("analysis_unit", str), ("assumes_results_hold_across_aggregations", bool)],
            "independence": [("assumes_independent_observations", bool)],
            "global_or_local": [("assumes", str)],
        }
        for field, required in checks.items():
            f = a.get(field)
            if not isinstance(f, dict):
                raise InvalidResponse(f"{where}: assumption '{field}' is missing")
            for key, typ in required:
                if not isinstance(f.get(key), typ) or (typ is str and not f[key].strip()):
                    raise InvalidResponse(f"{where}: '{field}.{key}' must be a {typ.__name__}")
            if len(str(f.get("justification", "")).strip()) < MIN_JUSTIFICATION:
                raise InvalidResponse(f"{where}: '{field}.justification' is missing or too short")
        for key in ("question_unit", "data_unit"):
            if a["scale_and_zoning"].get(key) not in UNITS:
                raise InvalidResponse(f"{where}: 'scale_and_zoning.{key}' must be 'individual' or 'area'")
        if a["global_or_local"]["assumes"] not in ("global", "local"):
            raise InvalidResponse(f"{where}: 'global_or_local.assumes' must be 'global' or 'local'")
    return hyps


def state_hypotheses(ctx, answer=None):
    """Ask for structured hypotheses until every field is valid; return them."""
    datasets = input_datasets(ctx)
    if not datasets:
        raise InvalidResponse("no uploaded datasets to state hypotheses about")
    system = SYSTEM.format(max_h=MAX_HYPOTHESES, guidance=field_guidance())
    return ask_json(system, _user_prompt(ctx, datasets),
                    lambda obj: validate_hypotheses(obj, datasets), answer=answer)


# ── the lock ─────────────────────────────────────────────────────────

def _digest(hypotheses):
    return hashlib.sha256(json.dumps(hypotheses, sort_keys=True).encode()).hexdigest()


def lock(ctx, hypotheses):
    record = {"hypotheses": hypotheses, "sha256": _digest(hypotheses),
              "locked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, LOCK_FILE), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    ctx.setdefault("artifacts", {})["spatial_hypotheses"] = hypotheses
    return record


def locked(ctx):
    return ctx.get("artifacts", {}).get("spatial_hypotheses") or []


def unit_of_inference(hypothesis):
    """(status, why) from comparing the question's unit with the data's, or
    None for a hypothesis stated before these fields existed."""
    a = hypothesis["assumptions"]["scale_and_zoning"]
    q, d = a.get("question_unit"), a.get("data_unit")
    if q not in UNITS or d not in UNITS:
        return None
    if q == "individual" and d == "area":
        return "contradicted", ("the question is about individuals but the data are area aggregates; "
                                "associations between area averages need not hold for the individuals "
                                "in them (ecological fallacy), so conclusions must be stated at the "
                                "areal level")
    return "supported", f"the question and the data share the same unit ({d})"


def planner_instructions(hypotheses):
    """Added to the planner's request so the plan tests exactly these."""
    lines = ["**LOCKED HYPOTHESES** — stated before planning. Do not change, add or remove "
             "hypotheses. Every objective must test one or more of them: list their ids in a "
             "\"hypothesis_ids\" field on the objective and copy the statement into \"hypothesis\"."]
    for h in hypotheses:
        a = h["assumptions"]
        lines.append(
            f"- {h['id']}: {h['statement']} [dataset {h['dataset']}; outcome {h['outcome']}; "
            f"predictors {', '.join(h['predictors'])}; unit: {a['scale_and_zoning']['analysis_unit']}; "
            f"assumes independent observations: {a['independence']['assumes_independent_observations']}; "
            f"assumes a {a['global_or_local']['assumes']} relationship]")
        unit = unit_of_inference(h)
        if unit and unit[0] == "contradicted":
            lines.append(f"  {h['id']}: {unit[1]}.")
    return "\n".join(lines)


def _words(text):
    return {w for w in "".join(c.lower() if c.isalnum() or c == "_" else " " for c in str(text)).split()
            if len(w) > 2}


def _best_match(obj, hypotheses):
    """The hypothesis whose statement and variables share the most words with
    the objective's title and hypothesis text, or None if nothing overlaps."""
    have = _words(f"{obj.get('title', '')} {obj.get('hypothesis', '')}")
    scored = []
    for h in hypotheses:
        want = _words(f"{h['statement']} {h['outcome']} {' '.join(h['predictors'])}")
        overlap = len(have & want) / max(len(want), 1)
        scored.append((overlap, h["id"]))
    best = max(scored, default=(0, None))
    return best[1] if best[0] > 0 else None


def _ids_for(obj, i, by_id, hypotheses, draft_ids):
    """(ids, how they were found) for one objective."""
    ids = [x for x in (obj.get("hypothesis_ids") or []) if x in by_id]
    if ids:
        return ids, None
    ids = [x for x in draft_ids.get(obj.get("number"), []) if x in by_id]
    if ids:
        return ids, "hypothesis_ids recovered from the planner's draft"
    match = _best_match(obj, hypotheses)
    if match:
        return [match], "hypothesis_ids missing; matched by content"
    return [hypotheses[min(i, len(hypotheses) - 1)]["id"]], "hypothesis_ids missing; assigned by position"


def enforce(plan, hypotheses, draft=None):
    """Restore the locked hypotheses on every objective. Returns a report of
    what had to be restored, which is itself a finding about the agents.

    The published plan reviewer drops "hypothesis_ids", so ids are recovered
    from the planner's *draft* (same objective number) before falling back to
    matching by content; position is the last resort and is reported."""
    by_id = {h["id"]: h for h in hypotheses}
    draft_ids = {o.get("number"): o.get("hypothesis_ids") or []
                 for o in (draft or {}).get("objectives", []) or []}
    restored, used = [], set()
    for i, obj in enumerate(plan.get("objectives", []) or []):
        ids, how = _ids_for(obj, i, by_id, hypotheses, draft_ids)
        text = " ".join(by_id[x]["statement"] for x in ids)
        if how:
            restored.append({"objective": obj.get("number"), "reason": how, "ids": ids})
        if obj.get("hypothesis") != text:
            restored.append({"objective": obj.get("number"), "reason": "hypothesis text was changed",
                             "changed_to": obj.get("hypothesis")})
        obj["hypothesis_ids"], obj["hypothesis"] = ids, text
        used.update(ids)
    return {"restored": restored, "untested_hypotheses": sorted(set(by_id) - used)}


def record_checkpoint_edits(ctx):
    """At plan approval: note any objective whose hypothesis the user edited."""
    hyps = locked(ctx)
    plan = ctx.get("artifacts", {}).get("research_plan_final") or {}
    if not hyps or not plan:
        return None
    by_id = {h["id"]: h["statement"] for h in hyps}
    edits = []
    for obj in plan.get("objectives", []) or []:
        expected = " ".join(by_id[x] for x in obj.get("hypothesis_ids") or [] if x in by_id)
        if expected and obj.get("hypothesis") != expected:
            edits.append({"objective": obj.get("number"), "locked": expected,
                          "approved": obj.get("hypothesis")})
    record = {"edited_by_user": bool(edits), "edits": edits,
              "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(resolve_output_dir(ctx), CHECKPOINT_FILE), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


# ── variables for the ablation without structure ─────────────────────

SPEC_SYSTEM = """Read a research plan and list the regression model(s) it will estimate.
Return JSON only:
{"models": [{"id": "M1", "dataset": "source_id", "outcome": "column", "predictors": ["column", ...]}]}
Use dataset ids and column names exactly as given. At most 3 models."""


def validate_spec(obj, datasets):
    models = obj.get("models")
    if not isinstance(models, list) or not models:
        raise InvalidResponse("'models' must be a non-empty list")
    for i, m in enumerate(models[:MAX_HYPOTHESES], start=1):
        m.setdefault("id", f"M{i}")
        _check_variables(m, datasets, f"model {m['id']}")
    return models[:MAX_HYPOTHESES]


def extract_analysis_spec(ctx, plan, answer=None):
    datasets = input_datasets(ctx)
    cols = {n: [c for c in d["columns"] if c != "geometry"] for n, d in datasets.items()}
    user = (f"Datasets and columns: {json.dumps(cols)}\n\nPlan:\n"
            f"{json.dumps(plan.get('objectives', []), default=str)[:12000]}")
    return ask_json(SPEC_SYSTEM, user, lambda obj: validate_spec(obj, datasets), answer=answer)
