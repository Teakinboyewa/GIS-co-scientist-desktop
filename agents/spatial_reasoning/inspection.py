"""
Read-only views of the conditions and of each run's record, for the WebUI's
Spatial Conditions page and the per-run download links.

Prompts shown here are built by knowledge_base.compose_planner_prompt, the
same function the planner calls, so the page cannot drift from what is sent.
"""

import io
import json
import os
import re
import zipfile

import utils.prompts as prompts
from agents.spatial_reasoning import conditions, knowledge_base as kb

# Files a run writes into Outputs/<task_id>/, by download key.
RUN_FILES = {
    "condition": ("spatial_condition.json", "Condition record"),
    "system_prompt": ("planner_system_prompt.txt", "Planner system prompt"),
    "user_prompt": ("planner_user_prompt.txt", "Planner user prompt"),
    "tools": ("planner_tools.json", "Planner tools"),
    "knowledge_access": ("spatial_knowledge_access.json", "Knowledge lookups"),
    # Proposed method only
    "hypotheses": ("hypotheses_locked.json", "Locked hypotheses"),
    "diagnostics": ("spatial_diagnostics.json", "Diagnostics"),
    "revisions": ("plan_revisions.json", "Plan revisions"),
    "critic": ("spatial_critic.json", "Spatial critic"),
    "assumptions_report": ("spatial_assumptions.md", "Assumptions report"),
    "checkpoint_edits": ("hypotheses_checkpoint_edits.json", "Checkpoint edits"),
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _flags(spec):
    return {k: spec[k] for k in ("kb_delivery", "structured_assumptions", "diagnostics", "critic")}


def _built(name):
    try:
        conditions.resolve(name)
        return True
    except conditions.ConditionNotImplemented:
        return False


DELIVERY_SUMMARY = {
    "none": "Nothing is added to the planner's system prompt.",
    "prompt": "The principles and checklist are appended to the planner's system prompt.",
    "retrieval": "A one-line notice is appended to the system prompt, and the planner gets "
                 "a search tool that returns knowledge-base entries on request.",
    "skill": "An index of three skill files is appended to the system prompt, and the planner "
             "gets a tool that loads a skill file's if/then rules by name.",
}


def components(spec):
    """The proposed method's components this condition switches on, each with
    the exact prompt it sends (diagnostics are code, so they list thresholds)."""
    from agents.spatial_reasoning import assumptions, critic, diagnostics as dg, revision
    out = []
    if spec["structured_assumptions"]:
        out.append({"name": "(a) Structured assumptions",
                    "summary": "Before planning, each hypothesis must fill required fields for scale "
                               "and zoning, independence, and a global or local relationship. The "
                               "hypotheses are then locked against later agents.",
                    "prompt": assumptions.SYSTEM.format(max_h=assumptions.MAX_HYPOTHESES,
                                                        guidance=assumptions.field_guidance())})
    if spec["diagnostics"]:
        out.append({"name": "(b) Grounded diagnostics",
                    "summary": "Fixed Python on exploration data: residual Moran's I, LISA and robust "
                               "LM tests (queen and k=8 weights); OLS re-estimated at coarser and "
                               "shifted aggregations; MGWR bandwidths. Thresholds: "
                               f"alpha {dg.ALPHA}; dependence if Moran's I is significant, or the same "
                               f"LM test has p < {dg.ALPHA_LM_ONLY} under every weights; local if "
                               f"bandwidth < {dg.LOCAL_BANDWIDTH_SHARE} n and AICc improves by "
                               f"{dg.AICC_MARGIN}; scale-sensitive if a sign flips or a coefficient "
                               f"changes by more than {dg.SCALE_RELATIVE_CHANGE:.0%}, and the change "
                               f"exceeds {dg.Z_DIFF} standard errors."})
    if spec["diagnostics"] or spec["critic"]:
        out.append({"name": "Plan revision",
                    "summary": "Revises the plan against the evidence or the critic's issues. Changed "
                               "steps are marked exploratory; hypotheses cannot change.",
                    "prompt": revision.SYSTEM})
    if spec["critic"]:
        out.append({"name": "(c) Spatial critic",
                    "summary": f"Reviews the revised plan against the evidence, up to {critic.MAX_ROUNDS} "
                               "rounds; may request a diagnostic but cannot edit the plan.",
                    "prompt": critic.SYSTEM.format(
                        kb="\n\n".join(kb.entry_text(e) for e in kb.entries()),
                        catalog=critic.CATALOG_ON if spec["diagnostics"] else critic.CATALOG_OFF,
                        kinds=" | ".join(critic.ISSUE_KINDS))})
    return out


def list_conditions():
    return {
        "kb_version": kb.load_kb().get("version"),
        "kb_sha256": kb.fingerprint(),
        "conditions": [{"name": name, "label": spec["label"], "built": _built(name),
                        **_flags(spec)} for name, spec in conditions.CONDITIONS.items()],
    }


def describe_condition(name):
    if name not in conditions.CONDITIONS:
        raise conditions.UnknownCondition(f"Unknown spatial condition {name!r}")
    spec = conditions.CONDITIONS[name]
    out = {"name": name, "label": spec["label"], "built": _built(name), **_flags(spec),
           "kb_version": kb.load_kb().get("version"), "kb_sha256": kb.fingerprint()}
    if not out["built"]:
        out["note"] = ("Not built yet: the components this condition needs do not exist, "
                       "so it cannot be run and has no prompt.")
        return out
    base = prompts.generate_research_plan_system_prompt
    system_prompt, tools = kb.compose_planner_prompt(base, spec["kb_delivery"], [])
    out.update({
        "delivery_summary": DELIVERY_SUMMARY[spec["kb_delivery"]],
        "base_prompt": base,
        "addition": system_prompt[len(base):].strip("\n"),
        "system_prompt": system_prompt,
        "tools": [{"name": t.name, "description": t.description} for t in tools],
        "skill_files": ({kb.SKILL_NAMES[c]: kb.render_skill(c) for c in kb.CONCEPTS}
                        if spec["kb_delivery"] == "skill" else {}),
        "retrievable_entries": ([kb.entry_text(e) for e in kb.entries()]
                                if spec["kb_delivery"] == "retrieval" else []),
        "components": components(spec),
    })
    return out


def condition_markdown(name):
    """Everything about one condition as a single Markdown document."""
    d = describe_condition(name)
    lines = [f"# {d['name']} — {d['label']}", "",
             f"- Knowledge delivery: {d['kb_delivery']}",
             f"- Structured assumptions: {d['structured_assumptions']}",
             f"- Grounded diagnostics: {d['diagnostics']}",
             f"- Spatial critic: {d['critic']}",
             f"- Knowledge base: version {d['kb_version']}, sha256 {d['kb_sha256']}", ""]
    if not d["built"]:
        return "\n".join(lines + [d["note"], ""])
    lines += [d["delivery_summary"], "", "## Added to the planner's system prompt", ""]
    lines += ([f"```text\n{d['addition']}\n```"] if d["addition"] else ["(nothing)"])
    if d["tools"]:
        lines += ["", "## Tools added to the planner", ""]
        lines += [f"- `{t['name']}`: {' '.join(t['description'].split())}" for t in d["tools"]]
    for skill, text in d["skill_files"].items():
        lines += ["", f"## Skill file: {skill}", "", f"```text\n{text.rstrip()}\n```"]
    if d["retrievable_entries"]:
        lines += ["", "## Entries the search tool can return", ""]
        lines += [f"```text\n{t}\n```\n" for t in d["retrievable_entries"]]
    for c in d["components"]:
        lines += ["", f"## {c['name']}", "", c["summary"]]
        if c.get("prompt"):
            lines += ["", f"```text\n{c['prompt']}\n```"]
    lines += ["", "## Full planner system prompt", "", f"```text\n{d['system_prompt']}\n```", ""]
    return "\n".join(lines)


def kb_yaml_text():
    with open(kb.KB_PATH, encoding="utf-8") as f:
        return f.read()


# ── per-run records ──────────────────────────────────────────────────

def run_dir(output_root, task_id):
    if not task_id or not _SAFE_ID.match(task_id) or task_id in (".", ".."):
        raise ValueError("invalid task id")
    return os.path.join(output_root, task_id)


def run_record(output_root, task_id):
    folder = run_dir(output_root, task_id)
    files = []
    for key, (filename, label) in RUN_FILES.items():
        if os.path.exists(os.path.join(folder, filename)):
            files.append({"key": key, "filename": filename, "label": label})
    def _load(key):
        path = os.path.join(folder, RUN_FILES[key][0])
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return {"task_id": task_id, "condition": _load("condition"), "files": files,
            "assumptions": _assumption_rows(_load("hypotheses"), _load("diagnostics"))}


def _assumption_rows(locked, diagnostics):
    """One row per stated assumption, for the plan checkpoint."""
    if not locked:
        return []
    from agents.spatial_reasoning.assumptions import FIELD_TITLES
    from agents.spatial_reasoning.report import stated_text
    status = (diagnostics or {}).get("assumption_status", {})
    rows = []
    for h in locked["hypotheses"]:
        for field, a in h["assumptions"].items():
            s = status.get(h["id"], {}).get(field, {"status": "untested", "why": "not tested"})
            rows.append({"hypothesis": h["id"], "statement": h["statement"],
                         "assumption": FIELD_TITLES[field], "stated": stated_text(field, a),
                         "justification": a.get("justification", ""), **s})
    return rows


def run_file(output_root, task_id, key):
    """(path, download name) for one record file, or None."""
    if key not in RUN_FILES:
        return None
    filename = RUN_FILES[key][0]
    path = os.path.join(run_dir(output_root, task_id), filename)
    return (path, f"{task_id}_{filename}") if os.path.exists(path) else None


def run_zip(output_root, task_id):
    """All of a run's record files as zip bytes, or None if there are none."""
    folder = run_dir(output_root, task_id)
    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for filename, _ in RUN_FILES.values():
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                z.write(path, arcname=f"{task_id}/{filename}")
                count += 1
    return buf.getvalue() if count else None
