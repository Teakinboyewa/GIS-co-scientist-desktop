"""
Diagnostics-grounded plan revision.

The LLM proposes the revised plan; the code then guarantees three things the
proposal requires regardless of what the LLM did:
  - every step still connects (inputs are a dataset or an earlier output),
  - steps added or changed by a revision are marked exploratory,
  - the locked hypotheses are not touched (restored by assumptions.enforce
    when the plan is finalized).
"""

import copy
import json
import os

from agents.spatial_reasoning.llm import InvalidResponse, ask_json

STEP_KEYS = ("operation_type", "operation", "input_nodes", "output_node", "description")

SYSTEM = """You revise a geospatial research plan in light of diagnostic evidence about its spatial
assumptions. The diagnostics ran on exploration data; the plan will be executed on separate
confirmation data.

Rules:
- Change the plan only where the evidence (or a reviewer issue) requires it, with the smallest
  change that addresses it. If the evidence supports the plan, return it unchanged. Do not add
  analyses the evidence does not call for.
- The evidence is reported for the full model of each outcome (all predictors of the hypotheses
  that share it). Add a spatial remedy (spatial lag/error model, local model such as MGWR,
  analysis at another scale) only when that full model shows the problem it fixes, and do not fit
  spatial models to reduced models. Remove remedies the evidence does not support.
- When a hypothesis's unit of inference is contradicted (the question is about individuals but
  the data are area aggregates), the plan must state its conclusions at the areal level: make the
  reporting step say that the associations are between areas and may not hold for the
  individuals in them (ecological fallacy). This is a change to what is concluded, not a new
  model.
- Do not change, add or remove hypotheses, and keep each objective's "hypothesis_ids".
- Keep the plan's JSON structure. Every step keeps operation_type, operation, input_nodes,
  output_node, description and expected_output; each input_node must be a dataset id or the
  output_node of an earlier step.
- On every step you add or change, set "triggered_by" to the diagnostic or reviewer issue behind it.
- Add a top-level "revision_log": [{"change": "...", "evidence": "..."}] (empty if nothing changed).

Return the complete plan as JSON only."""


def _stem(name):
    return os.path.splitext(str(name))[0]


def check_connectivity(plan, sources):
    """Readable problems with step inputs; empty when every input resolves."""
    available = set(sources) | {_stem(s) for s in sources}
    problems = []
    for obj in plan.get("objectives", []) or []:
        for step in obj.get("steps", []) or []:
            for node in step.get("input_nodes", []) or []:
                if node not in available and _stem(node) not in available:
                    problems.append(f"objective {obj.get('number')} step {step.get('step_number')}: "
                                    f"input {node!r} is neither a dataset nor an earlier output")
            out = step.get("output_node")
            if out:
                available |= {out, _stem(out)}
    return problems


def validate_plan(obj, sources, tolerated=()):
    objectives = obj.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise InvalidResponse("the plan must contain a non-empty 'objectives' list")
    for o in objectives:
        steps = o.get("steps")
        if not isinstance(steps, list) or not steps:
            raise InvalidResponse(f"objective {o.get('number')} has no steps")
        for s in steps:
            missing = [k for k in STEP_KEYS if not s.get(k)]
            if missing:
                raise InvalidResponse(f"objective {o.get('number')} step {s.get('step_number')} "
                                      f"is missing {missing}")
    problems = [p for p in check_connectivity(obj, sources) if p not in tolerated]
    if problems:
        raise InvalidResponse("; ".join(problems[:8]))
    return obj


def _step_key(obj_number, step):
    return (obj_number, step.get("output_node"))


def mark_exploratory(before, after):
    """Mark steps that are new or changed relative to *before*; return them."""
    old = {_step_key(o.get("number"), s): s
           for o in before.get("objectives", []) or [] for s in o.get("steps", []) or []}
    changed = []
    for o in after.get("objectives", []) or []:
        for s in o.get("steps", []) or []:
            prev = old.get(_step_key(o.get("number"), s))
            same = prev is not None and all(prev.get(k) == s.get(k) for k in STEP_KEYS)
            if not same:
                s["exploratory"] = True
                s.setdefault("triggered_by", "plan revision")
                changed.append({"objective": o.get("number"), "step": s.get("step_number"),
                                "operation": s.get("operation"),
                                "status": "added" if prev is None else "changed",
                                "triggered_by": s.get("triggered_by")})
    return changed


def revise(plan, hypotheses, evidence_text, sources, issues=None, answer=None):
    """One revision. Returns (revised plan, record of what changed)."""
    parts = [f"Current plan:\n```json\n{json.dumps(plan, indent=2, default=str)}\n```"]
    if hypotheses:
        parts.append("Locked hypotheses:\n" + "\n".join(f"- {h['id']}: {h['statement']}"
                                                         for h in hypotheses))
    if evidence_text:
        parts.append(f"Diagnostic evidence:\n{evidence_text}")
    if issues:
        parts.append("Issues raised by the spatial reviewer:\n"
                     + "\n".join(f"- [{i['kind']}] {i['detail']}" for i in issues))
    parts.append(f"Dataset ids: {', '.join(sources)}")
    # Problems the plan already had are tolerated; the revision may not add any.
    tolerated = set(check_connectivity(plan, sources))
    revised = ask_json(SYSTEM, "\n\n".join(parts),
                       lambda obj: validate_plan(obj, sources, tolerated), answer=answer)
    # Keep reviewer metadata the revision may have dropped.
    for key in ("review_summary",):
        if key in plan and key not in revised:
            revised[key] = copy.deepcopy(plan[key])
    changed = mark_exploratory(plan, revised)
    return revised, {"revision_log": revised.get("revision_log", []), "steps_marked_exploratory": changed}
