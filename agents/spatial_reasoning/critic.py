"""
Component (c): the spatial critic.

A separate reviewer checks the revised plan against the diagnostic evidence
before anything runs, and flags errors in both directions: evidence the plan
ignores, and spatial remedies the evidence does not support. It can request a
missing diagnostic from a fixed catalog, but it never edits the plan or the
hypotheses; the revision step acts on its issues.

Later rounds are follow-ups: they judge only whether the earlier major issues
were resolved and what the revision itself changed. In the pilot (T05, T09)
an open-ended second round raised new major issues about parts of the plan it
had already passed, so the critic never approved; and diagnostics requested
in the last round could never run.
"""

import json

from agents.spatial_reasoning import knowledge_base as kb
from agents.spatial_reasoning.diagnostics import DIAGNOSTICS, WEIGHTS
from agents.spatial_reasoning.llm import InvalidResponse, ask_json

ISSUE_KINDS = ("ignored_evidence", "unsupported_remedy", "missing_diagnostic", "other")
SEVERITIES = ("major", "minor")
BASES = ("unresolved", "introduced_by_revision")
MAX_ROUNDS = 2

SYSTEM = """You are a spatial-analysis reviewer. Check a research plan against the evidence about its
spatial assumptions, before any analysis runs. Look for errors in both directions:
- ignored_evidence: the evidence shows a problem (residual dependence, scale sensitivity, local
  variation) that the plan does not address;
- unsupported_remedy: the plan applies a spatial model or adjustment the evidence does not support
  (e.g. a spatial lag model when residuals are independent, GWR when bandwidths are global, a
  spatial model fitted to a reduced model whose full model shows no residual dependence);
- missing_diagnostic: a decision rests on a diagnostic that has not been run.
Judge each remedy against the evidence for its own problem. Local-variation evidence (covariate
bandwidths well below the number of units and MGWR AICc better than OLS) supports a local model
such as MGWR even when residuals are independent; residual dependence supports a spatial lag or
error model even when relationships are global. A remedy is not unsupported because a different
problem is absent.
When the evidence contradicts a locked assumption, the plan is meant to depart from it (for
example, a local model although the hypothesis assumed a global relationship). That departure is
the purpose of the revision, not an evidence-plan mismatch; the hypotheses stay as stated and the
report records the contradiction.
When the unit of inference is contradicted (the question is about individuals but the data are area
aggregates), a plan that draws individual-level conclusions from area-level associations, or does
not say that they may not hold for individuals, has a major ignored_evidence issue (ecological
fallacy). Comparing zonings does not address it.
Each issue has a severity. "major": it would change which model is estimated or a substantive
conclusion. "minor": anything else (reporting, extra robustness checks, presentation). Do not ask
for analyses beyond what the evidence calls for. Approve when there is no major issue; minor
issues may accompany an approval. You cannot change the hypotheses or edit the plan.
The evidence is reported for the full model of each outcome (all predictors of the hypotheses
that share it); judge spatial remedies against the full model.

Spatial knowledge base:
{kb}

{catalog}

Return JSON only:
{{"verdict": "approve" | "revise",
  "issues": [{{"kind": "{kinds}", "severity": "major" | "minor", "detail": "...",
              "objective": int | null, "step": int | null}}],
  "requested_diagnostics": [{{"diagnostic": "...", "model": "model id", "weights": "..."}}]}}"""

CATALOG_ON = ("Diagnostics you may request (they run on the exploration data):\n"
              f"- residual_dependence, with \"weights\" one of {', '.join(WEIGHTS)}\n"
              "- aggregation_sensitivity\n- local_variation")
CATALOG_OFF = "No diagnostics can be run; leave requested_diagnostics empty."
CATALOG_FINAL = ("This is the final review round: no further diagnostics can be run; leave "
                 "requested_diagnostics empty.")

FOLLOW_UP = """This is a follow-up review of a revised plan. Your earlier major issues and the steps the
revision added or changed are listed in the request. Judge only:
- whether each earlier major issue is resolved; if it is not, raise it again with
  "basis": "unresolved";
- problems in the steps the revision added or changed, with "basis": "introduced_by_revision".
Do not raise issues about parts of the plan the revision did not touch: they were reviewed
already. Every issue needs a "basis"."""


def validate(obj, model_ids, diagnostics_available, follow_up=False):
    if obj.get("verdict") not in ("approve", "revise"):
        raise InvalidResponse("'verdict' must be 'approve' or 'revise'")
    issues = obj.get("issues") or []
    if not isinstance(issues, list):
        raise InvalidResponse("'issues' must be a list")
    for i in issues:
        if i.get("kind") not in ISSUE_KINDS:
            raise InvalidResponse(f"issue kind must be one of {ISSUE_KINDS}")
        if i.get("severity") not in SEVERITIES:
            raise InvalidResponse(f"issue severity must be one of {SEVERITIES}")
        if not str(i.get("detail", "")).strip():
            raise InvalidResponse("every issue needs a 'detail'")
        if follow_up and i.get("basis") not in BASES:
            raise InvalidResponse(f"in a follow-up review every issue needs a 'basis', one of {BASES}")
    has_major = any(i["severity"] == "major" for i in issues)
    if obj["verdict"] == "revise" and not has_major:
        raise InvalidResponse("a 'revise' verdict needs at least one major issue; "
                              "approve when only minor issues remain")
    if obj["verdict"] == "approve" and has_major:
        raise InvalidResponse("a major issue cannot accompany an 'approve' verdict")
    requests = obj.get("requested_diagnostics") or []
    if requests and not diagnostics_available:
        raise InvalidResponse("no diagnostics can be run; requested_diagnostics must be empty")
    for r in requests:
        if r.get("diagnostic") not in DIAGNOSTICS:
            raise InvalidResponse(f"unknown diagnostic {r.get('diagnostic')!r}")
        if r.get("model") not in model_ids:
            raise InvalidResponse(f"'model' must be one of {model_ids}")
        if r["diagnostic"] == "residual_dependence" and r.get("weights", "queen") not in WEIGHTS:
            raise InvalidResponse(f"'weights' must be one of {WEIGHTS}")
    return {"verdict": obj["verdict"], "issues": issues, "requested_diagnostics": requests}


def review(plan, hypotheses, evidence_text, model_ids, diagnostics_available, answer=None,
           previous=None, changes=None, final=False):
    """*previous* (the last round's major issues) and *changes* (the steps
    the revision added or changed) make this a follow-up review. *final*
    means no round follows, so no diagnostic can be requested."""
    follow_up = previous is not None
    can_request = diagnostics_available and not final
    catalog = CATALOG_ON if can_request else CATALOG_FINAL if diagnostics_available else CATALOG_OFF
    system = SYSTEM.format(kb="\n\n".join(kb.entry_text(e) for e in kb.entries()),
                           catalog=catalog, kinds=" | ".join(ISSUE_KINDS))
    if follow_up:
        system += "\n\n" + FOLLOW_UP
    parts = [f"Plan:\n```json\n{json.dumps(plan, indent=2, default=str)}\n```"]
    if hypotheses:
        parts.append("Locked hypotheses and their stated assumptions:\n```json\n"
                     f"{json.dumps(hypotheses, indent=2)}\n```")
    parts.append(f"Evidence:\n{evidence_text or 'No diagnostics were run.'}")
    if follow_up:
        parts.append("Your earlier major issues:\n```json\n"
                     f"{json.dumps(previous, indent=2)}\n```")
        parts.append("Steps the revision added or changed:\n```json\n"
                     f"{json.dumps(changes or [], indent=2, default=str)}\n```")
    parts.append(f"Model ids: {', '.join(model_ids)}")
    return ask_json(system, "\n\n".join(parts),
                    lambda obj: validate(obj, model_ids, can_request, follow_up), answer=answer)


def major(verdict):
    """The issues that may change the plan."""
    return [i for i in verdict["issues"] if i["severity"] == "major"]
