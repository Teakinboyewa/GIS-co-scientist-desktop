"""Persistent storage for Handbook Generator Experiment 2 studies.

Studies are isolated by the caller's hashed user ID and stored as JSON files.
The module intentionally contains no Flask imports so it can be tested and
reused independently from the web application.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_ROOT = os.path.join(
    PROJECT_ROOT, "Data_retrieval_skill_project", "Experiment_2_Studies"
)
_SAFE_ID = re.compile(r"^[a-f0-9-]{16,64}$")

MECHANISMS = ("rest", "stac", "ogc", "sql", "bulk")
# Open-source models are served from an OpenAI-compatible endpoint (see
# Local_models_test.ipynb). "qwen3:32b" is a reliable tool-caller from that
# catalog (needed for the Experiment 2 web-search loop), so it — plus its
# server — is the default for new studies; either can be overridden
# per-study in the UI (base_url empty routes back to GIBD/OpenAI).
DEFAULT_LOCAL_BASE_URL = "http://128.118.54.16:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen3:32b"
DEFAULT_CONTROLS = {
    "model": DEFAULT_LOCAL_MODEL,
    "base_url": DEFAULT_LOCAL_BASE_URL,
    "reasoning_effort": "low",
    "repetitions": 1,
    "max_refinements": 3,
    "timeout_seconds": 300,
    "stop_on_success": True,
    "stop_on_repeated_error": True,
    "randomize_order": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dir(user_id: str) -> str:
    safe = re.sub(r"[^a-fA-F0-9_-]", "", str(user_id or ""))[:80]
    if not safe:
        raise ValueError("A valid user ID is required.")
    path = os.path.join(STORE_ROOT, safe)
    os.makedirs(path, exist_ok=True)
    return path


def _study_path(user_id: str, study_id: str) -> str:
    if not _SAFE_ID.match(str(study_id or "")):
        raise ValueError("Invalid experiment ID.")
    return os.path.join(_user_dir(user_id), f"{study_id}.json")


def new_study(name: str = "Untitled Experiment 2 Study") -> dict:
    now = _now()
    return {
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "name": str(name or "Untitled Experiment 2 Study").strip()[:160],
        "description": "",
        "research_question": (
            "How does execution-grounded refinement affect handbook quality "
            "and retrieval performance across geospatial access mechanisms?"
        ),
        "status": "draft",
        "source_name": "",
        "documentation_url": "",
        "retrieval_task": "",
        "mechanisms": list(MECHANISMS),
        "sources": [],
        "tasks": [],
        "controls": deepcopy(DEFAULT_CONTROLS),
        "trials": [],
        "handbook_versions": [],
        "refinements": [],
        "manual_interventions": [],
        "pipeline_state": {},
        "source_slug": "",
        "created_at": now,
        "updated_at": now,
    }


def _clean_record(record: dict, existing: dict | None = None) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Experiment data must be an object.")
    base = deepcopy(existing) if existing else new_study(record.get("name", ""))

    for field in (
        "name", "description", "research_question", "status", "source_name",
        "documentation_url", "retrieval_task", "source_slug",
    ):
        if field in record:
            base[field] = str(record.get(field) or "").strip()
    if not base["name"]:
        raise ValueError("Experiment name is required.")
    base["name"] = base["name"][:160]
    base["description"] = base["description"][:5000]
    base["research_question"] = base["research_question"][:5000]
    base["source_name"] = base.get("source_name", "")[:300]
    base["documentation_url"] = base.get("documentation_url", "")[:3000]
    base["retrieval_task"] = base.get("retrieval_task", "")[:10000]
    base["source_slug"] = base.get("source_slug", "")[:100]

    mechanisms = record.get("mechanisms", base.get("mechanisms", []))
    base["mechanisms"] = [m for m in MECHANISMS if m in set(mechanisms or [])]

    for field in (
        "sources", "tasks", "trials", "handbook_versions", "refinements",
        "manual_interventions",
    ):
        value = record.get(field, base.get(field, []))
        base[field] = value if isinstance(value, list) else []

    pipeline_state = record.get(
        "pipeline_state", base.get("pipeline_state", {}))
    base["pipeline_state"] = (
        pipeline_state if isinstance(pipeline_state, dict) else {})

    controls = deepcopy(DEFAULT_CONTROLS)
    if isinstance(base.get("controls"), dict):
        controls.update(base["controls"])
    if isinstance(record.get("controls"), dict):
        controls.update(record["controls"])
    controls["model"] = str(controls.get("model") or "").strip()[:200]
    base_url = str(controls.get("base_url") or "").strip()[:500]
    controls["base_url"] = base_url if re.match(r"^https?://", base_url) else ""
    controls["repetitions"] = max(1, min(100, int(controls.get("repetitions", 1))))
    controls["max_refinements"] = max(
        0, min(20, int(controls.get("max_refinements", 3)))
    )
    controls["timeout_seconds"] = max(
        10, min(7200, int(controls.get("timeout_seconds", 300)))
    )
    base["controls"] = controls
    base["updated_at"] = _now()
    return base


def save_study(user_id: str, record: dict, study_id: str | None = None) -> dict:
    existing = get_study(user_id, study_id) if study_id else None
    cleaned = _clean_record(record, existing=existing)
    if study_id:
        cleaned["id"] = study_id
        cleaned["created_at"] = existing.get("created_at", cleaned["created_at"])
    path = _study_path(user_id, cleaned["id"])
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".study_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(cleaned, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return cleaned


def get_study(user_id: str, study_id: str | None) -> dict | None:
    if not study_id:
        return None
    path = _study_path(user_id, study_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def list_studies(user_id: str) -> list[dict]:
    directory = _user_dir(user_id)
    studies = []
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, filename), encoding="utf-8") as handle:
                study = json.load(handle)
            studies.append({
                "id": study.get("id", ""),
                "name": study.get("name", "Untitled study"),
                "status": study.get("status", "draft"),
                "source_count": (
                    len(study.get("sources", []))
                    or int(bool(study.get("source_name")
                                or study.get("documentation_url")))
                ),
                "task_count": (
                    len(study.get("tasks", []))
                    or int(bool(study.get("retrieval_task")))
                ),
                "trial_count": len(study.get("trials", [])),
                "updated_at": study.get("updated_at", ""),
            })
        except (OSError, ValueError, TypeError):
            continue
    studies.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return studies


def delete_study(user_id: str, study_id: str) -> bool:
    path = _study_path(user_id, study_id)
    if not os.path.exists(path):
        return False
    os.unlink(path)
    return True
