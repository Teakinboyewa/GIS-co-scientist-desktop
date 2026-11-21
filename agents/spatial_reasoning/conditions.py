"""
Experimental conditions for the spatial-integration study (proposal Table 2
and the ablations).

A condition is a set of independent switches, so the full method and each
ablation differ only in the switches they flip:

  kb_delivery            how the shared knowledge base reaches the planner:
                         none | prompt (B1) | retrieval (B2) | skill (B3)
  structured_assumptions (a) required, locked assumption fields per hypothesis
  diagnostics            (b) assumption-mapped diagnostics on the exploration split
  critic                 (c) spatial critic over plan + evidence

P's planner prompt is the published one (kb_delivery "none"): its knowledge
base reaches the agent through the components instead — the field guidance
in (a), the diagnostic each entry names in (b), and the full entries the
critic reviews against in (c).

The condition is resolved once, when the workflow starts, and stored in
ctx["settings"]["spatial_condition"], next to the no-planning ablation flag,
so it survives both pause points (Task Breakdown and plan approval restore
ctx from saved state). An unknown name raises instead of falling back to B0:
tests/test_condition_routing.py records 15 runs that silently executed the
wrong arm, and a typo here would do the same.
"""

import json
import os
import time

from utils.config import resolve_output_dir

DEFAULT = "B0"

CONDITIONS = {
    "B0": dict(label="No integration (GIS Co-Scientist as published)",
               kb_delivery="none", structured_assumptions=False, diagnostics=False, critic=False),
    "B1": dict(label="Prompting",
               kb_delivery="prompt", structured_assumptions=False, diagnostics=False, critic=False),
    "B2": dict(label="Knowledge retrieval",
               kb_delivery="retrieval", structured_assumptions=False, diagnostics=False, critic=False),
    "B3": dict(label="Skill-based rules",
               kb_delivery="skill", structured_assumptions=False, diagnostics=False, critic=False),
    "P": dict(label="Proposed method",
              kb_delivery="none", structured_assumptions=True, diagnostics=True, critic=True),
    "P-no-structure": dict(label="Ablation: P without structured assumptions",
                           kb_delivery="none", structured_assumptions=False, diagnostics=True, critic=True),
    "P-no-diagnostics": dict(label="Ablation: P without grounded diagnostics",
                             kb_delivery="none", structured_assumptions=True, diagnostics=False, critic=True),
    "P-no-critic": dict(label="Ablation: P without the spatial critic",
                        kb_delivery="none", structured_assumptions=True, diagnostics=True, critic=False),
    "B3+structure": dict(label="Hybrid: B3 skills plus structured assumptions",
                         kb_delivery="skill", structured_assumptions=True, diagnostics=False, critic=False),
}

# Components that exist in code (agents/spatial_reasoning/pipeline.py). A
# condition needing anything else is refused rather than run without it, so
# no result is ever labelled P while missing part of P.
IMPLEMENTED_COMPONENTS = {"structured_assumptions", "diagnostics", "critic"}


class UnknownCondition(ValueError):
    pass


class ConditionNotImplemented(NotImplementedError):
    pass


def _missing_components(spec):
    wanted = {k for k in ("structured_assumptions", "diagnostics", "critic") if spec[k]}
    return sorted(wanted - IMPLEMENTED_COMPONENTS)


def resolve(name):
    """Return the full condition record for *name*, or raise."""
    key = DEFAULT if name in (None, "") else str(name).strip()
    if key not in CONDITIONS:
        raise UnknownCondition(
            f"Unknown spatial condition {name!r}. Known: {', '.join(CONDITIONS)}")
    spec = CONDITIONS[key]
    missing = _missing_components(spec)
    if missing:
        raise ConditionNotImplemented(
            f"Condition {key} needs {', '.join(missing)}, which is not built yet.")
    return {"name": key, **spec}


def condition_of(ctx):
    """The condition record stored on ctx. A ctx from before this setting
    existed is B0, which is what it ran."""
    settings = ctx.get("settings", {}) if isinstance(ctx, dict) else {}
    stored = settings.get("spatial_condition")
    if isinstance(stored, dict):
        return stored
    return resolve(stored)


def apply_to_ctx(ctx, name):
    """Resolve *name*, store it on ctx and write spatial_condition.json into
    the run's output folder, so every result on disk names its arm."""
    from agents.spatial_reasoning import code_status, knowledge_base

    record = resolve(name)
    code = code_status()
    record = {**record,
              "kb_version": knowledge_base.load_kb().get("version"),
              "kb_sha256": knowledge_base.fingerprint(),
              "code": code,
              "resolved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    ctx.setdefault("settings", {})["spatial_condition"] = record
    if code["stale"]:
        warning = ("WARNING: the spatial-reasoning code changed on disk after this server started; "
                   "this run uses the old version. Restart the web UI before running the experiment. "
                   f"Changed: {', '.join(code['changed_since_load'])}")
        print(warning)
        ctx.setdefault("logs", []).append(warning)

    out_dir = resolve_output_dir(ctx)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "spatial_condition.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record
