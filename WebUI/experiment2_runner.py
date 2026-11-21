"""Controlled batch execution for Handbook Generator Experiment 2.

The runner deliberately bypasses source selection: every trial is executed
against the exact handbook linked to its source record. Progress is yielded as
plain event dictionaries so Flask can expose it over SSE without coupling this
module to a web framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from WebUI import experiment2_store as store
from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent
from agents.data_agent import user_data_sources


# This project often lives under a deeply nested OneDrive path on Windows.
# Generated programs also choose descriptive output filenames, so even a short
# relative study path under PROJECT_ROOT can still exceed MAX_PATH. Keep the
# controlled execution workspace at a stable, short local path.
OUTPUT_ROOT = (
    os.path.join(tempfile.gettempdir(), "GISCoE2")
    if os.name == "nt"
    else os.path.join("/tmp", "gis-co-scientist-e2")
)
LEGACY_OUTPUT_ROOT = os.path.join(
    store.PROJECT_ROOT, "Data_retrieval_skill_project", "Experiment_2_Outputs"
)
_SAFE_PART = re.compile(r"[^A-Za-z0-9_-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: str) -> str:
    return _SAFE_PART.sub("", str(value or ""))[:100] or "unknown"


def experiment_output_roots(user_id: str, study_id: str) -> list[str]:
    """Return the short active output root followed by the legacy root.

    The short local execution root and identifiers leave ample room below
    Windows' traditional MAX_PATH limit even when generated retrieval code
    uses descriptive names.
    """
    active = os.path.join(
        OUTPUT_ROOT, _safe(user_id)[:12], _safe(study_id)[:12])
    legacy = os.path.join(
        LEGACY_OUTPUT_ROOT, _safe(user_id), _safe(study_id))
    return [os.path.abspath(active), os.path.abspath(legacy)]


def _study_output_root(user_id: str, study_id: str) -> str:
    root = experiment_output_roots(user_id, study_id)[0]
    os.makedirs(root, exist_ok=True)
    return root


def _cancel_path(user_id: str, study_id: str) -> str:
    directory = os.path.join(store.STORE_ROOT, _safe(user_id))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f".cancel-{_safe(study_id)}.flag")


def request_cancel(user_id: str, study_id: str) -> None:
    with open(_cancel_path(user_id, study_id), "w", encoding="utf-8") as handle:
        handle.write(_now())


def _clear_cancel(user_id: str, study_id: str) -> None:
    path = _cancel_path(user_id, study_id)
    if os.path.exists(path):
        os.unlink(path)


def _is_cancelled(user_id: str, study_id: str) -> bool:
    return os.path.exists(_cancel_path(user_id, study_id))


def _find_source_id(agent: DataRetrieverAgent, source: dict) -> str | None:
    """Resolve only an exact linked slug or exact handbook display name."""
    linked = str(source.get("handbook_slug") or "").strip()
    if linked and agent._resolve_handbook_file(linked, agent.handbook_dir):
        return linked
    wanted = str(source.get("name") or "").strip().casefold()
    if not wanted:
        return None
    for path in agent.collect_handbook_files(agent.handbook_dir):
        try:
            book = agent._load_book(path)
            if str(book.get("data_source_name") or "").strip().casefold() == wanted:
                return agent._source_id(path)
        except Exception:
            continue
    return None


def _redact_text(value, secrets: list[str]):
    text = str(value or "")
    for secret in secrets:
        secret = str(secret or "")
        if not secret:
            continue
        if len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
        elif len(secret) >= 4:
            # Short test values such as "a" or "s" previously destroyed every
            # traceback and URL containing those letters. Redact short values
            # only when they appear as a complete token.
            text = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(secret)}(?![A-Za-z0-9])",
                "[REDACTED]", text)
    return text


def _safe_url(url: str, secrets: list[str]) -> str:
    try:
        parts = urlsplit(_redact_text(url, secrets))
        query = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if any(token in key.casefold() for token in (
                    "key", "token", "secret", "password", "authorization")):
                value = "[REDACTED]"
            query.append((key, value))
        return urlunsplit((
            parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    except Exception:
        return _redact_text(url, secrets)


def _expected_extensions(expected_format: str) -> set[str]:
    value = str(expected_format or "").casefold()
    mapping = {
        "geojson": {".geojson", ".json"},
        "geopackage": {".gpkg"},
        "gpkg": {".gpkg"},
        "shapefile": {".shp"},
        "csv": {".csv"},
        "parquet": {".parquet", ".geoparquet"},
        "raster": {".tif", ".tiff", ".nc"},
        "geotiff": {".tif", ".tiff"},
        "netcdf": {".nc", ".netcdf"},
        "json": {".json", ".geojson"},
        "osm xml": {".osm", ".xml"},
        "zip": {".zip"},
    }
    result = set()
    for label, extensions in mapping.items():
        if label in value:
            result.update(extensions)
    return result


def _normalize_bbox(value):
    """Normalize a raw JSON "bbox" value into the canonical
    {"west","south","east","north"} shape every other bbox producer in this
    file (rasterio/geopandas) already emits, so downstream comparisons never
    have to handle two shapes. Accepts the already-canonical dict, or the
    common GeoJSON/STAC flat-list convention [west, south, east, north] (a
    6-element 3D bbox [west,south,elev_min,east,north,elev_max] is also
    accepted, ignoring elevation). Returns None if unparseable."""
    if isinstance(value, dict) and {"west", "south", "east", "north"} <= value.keys():
        try:
            return {k: float(value[k]) for k in ("west", "south", "east", "north")}
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        try:
            if len(value) == 4:
                west, south, east, north = (float(v) for v in value)
            elif len(value) == 6:
                west, south, _, east, north, _ = (float(v) for v in value)
            else:
                return None
            return {"west": west, "south": south, "east": east, "north": north}
        except (TypeError, ValueError):
            return None
    return None


def _inspect_output_file(path: str, display_name: str, size_bytes: int) -> dict:
    """Collect bounded, non-secret evidence for semantic task validation."""
    extension = os.path.splitext(display_name)[1].casefold()
    evidence = {
        "name": display_name,
        "size_bytes": size_bytes,
        "extension": extension,
    }
    try:
        if extension in (".json", ".geojson") and size_bytes <= 10_000_000:
            with open(path, encoding="utf-8", errors="replace") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                evidence["top_level_keys"] = list(data.keys())[:30]
                features = data.get("features")
                if isinstance(features, list):
                    evidence["feature_count"] = len(features)
                    if features and isinstance(features[0], dict):
                        props = features[0].get("properties")
                        if isinstance(props, dict):
                            evidence["sample_property_keys"] = list(
                                props.keys())[:30]
                if "bbox" in data:
                    normalized = _normalize_bbox(data.get("bbox"))
                    if normalized:
                        evidence["bbox"] = normalized
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    evidence["metadata"] = {
                        key: metadata.get(key)
                        for key in list(metadata.keys())[:12]
                        if isinstance(metadata.get(key), (
                            str, int, float, bool, type(None)))
                    }
            if "bbox" not in evidence and size_bytes <= 20_000_000:
                # Fallback for the common case where the file has no
                # top-level "bbox" key: compute a real extent from the
                # actual geometries rather than leaving spatial extent
                # unverifiable.
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(path)
                    if len(gdf):
                        if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                            gdf = gdf.to_crs(epsg=4326)
                        west, south, east, north = gdf.total_bounds
                        evidence["bbox"] = {
                            "west": float(west), "south": float(south),
                            "east": float(east), "north": float(north),
                        }
                        if gdf.crs is not None:
                            evidence["crs"] = str(gdf.crs)
                except Exception:
                    pass
        elif extension in (".csv", ".tsv") and size_bytes <= 20_000_000:
            import csv
            delimiter = "\t" if extension == ".tsv" else ","
            with open(path, encoding="utf-8", errors="replace",
                      newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                header = next(reader, [])
                lat_idx = next((i for i, col in enumerate(header)
                                if re.fullmatch(r"lat(itude)?", col.strip(), re.I)), None)
                lon_idx = next((i for i, col in enumerate(header)
                                if re.fullmatch(r"lon(gitude)?|lng", col.strip(), re.I)), None)
                row_count = 0
                lat_min = lat_max = lon_min = lon_max = None
                for row in reader:
                    row_count += 1
                    if (lat_idx is not None and lon_idx is not None
                            and lat_idx < len(row) and lon_idx < len(row)):
                        try:
                            lat_val = float(row[lat_idx])
                            lon_val = float(row[lon_idx])
                        except ValueError:
                            continue
                        lat_min = lat_val if lat_min is None else min(lat_min, lat_val)
                        lat_max = lat_val if lat_max is None else max(lat_max, lat_val)
                        lon_min = lon_val if lon_min is None else min(lon_min, lon_val)
                        lon_max = lon_val if lon_max is None else max(lon_max, lon_val)
            evidence["columns"] = header[:50]
            evidence["row_count"] = row_count
            if lat_min is not None and lon_min is not None:
                evidence["bbox"] = {
                    "west": lon_min, "south": lat_min,
                    "east": lon_max, "north": lat_max,
                }
        elif extension in (".tif", ".tiff") and size_bytes <= 500_000_000:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(path) as dataset:
                evidence["crs"] = str(dataset.crs) if dataset.crs else None
                evidence["bands"] = dataset.count
                evidence["dtype"] = (
                    str(dataset.dtypes[0]) if dataset.dtypes else None)
                if dataset.crs and dataset.bounds:
                    try:
                        west, south, east, north = transform_bounds(
                            dataset.crs, "EPSG:4326", *dataset.bounds)
                        evidence["bbox"] = {
                            "west": west, "south": south,
                            "east": east, "north": north,
                        }
                    except Exception:
                        pass
        elif extension == ".shp":
            import geopandas as gpd
            gdf = gpd.read_file(path)
            evidence["feature_count"] = len(gdf)
            if len(gdf):
                if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                    gdf = gdf.to_crs(epsg=4326)
                west, south, east, north = gdf.total_bounds
                evidence["bbox"] = {
                    "west": float(west), "south": float(south),
                    "east": float(east), "north": float(north),
                }
                evidence["crs"] = str(gdf.crs) if gdf.crs is not None else None
        elif extension in (".xml", ".osm"):
            import xml.etree.ElementTree as element_tree
            if size_bytes <= 15_000_000:
                root = element_tree.parse(path).getroot()
                evidence["root_tag"] = root.tag
                if extension == ".osm":
                    evidence["osm_counts"] = {
                        kind: len(root.findall(kind))
                        for kind in ("node", "way", "relation")
                    }
            else:
                with open(path, "rb") as handle:
                    head = handle.read(4096).decode(
                        "utf-8", errors="replace")
                evidence["root_tag"] = (
                    "osm" if "<osm" in head else "xml")
        elif extension == ".gpkg":
            import sqlite3
            with sqlite3.connect(path) as connection:
                rows = connection.execute(
                    "SELECT table_name FROM gpkg_contents LIMIT 30"
                ).fetchall()
            evidence["layers"] = [row[0] for row in rows]
            try:
                import geopandas as gpd
                gdf = gpd.read_file(path)
                if len(gdf):
                    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                        gdf = gdf.to_crs(epsg=4326)
                    west, south, east, north = gdf.total_bounds
                    evidence["bbox"] = {
                        "west": float(west), "south": float(south),
                        "east": float(east), "north": float(north),
                    }
                    if gdf.crs is not None:
                        evidence["crs"] = str(gdf.crs)
            except Exception:
                pass
    except Exception as exc:
        evidence["inspection_note"] = (
            f"Structured inspection unavailable: {type(exc).__name__}")
    return evidence


def _sanitize_execution(execution: dict, secrets: list) -> dict:
    """Redact secrets out of the per-attempt code/error/fix_explanation text
    (execute_complete_program's "attempts" list) the same way generated_code
    and error already are — these carry the same kind of content (full
    generated code across every attempt, including any substituted
    credential values) and were passed through unredacted otherwise."""
    execution = dict(execution or {})
    attempts = []
    for item in execution.get("attempts") or []:
        if not isinstance(item, dict):
            continue
        attempts.append({
            "attempt": item.get("attempt"),
            "success": bool(item.get("success")),
            "timed_out": bool(item.get("timed_out")),
            "code": _redact_text(item.get("code"), secrets)[:60000],
            "error": _redact_text(item.get("error"), secrets)[:4000],
            "fix_explanation": _redact_text(
                item.get("fix_explanation"), secrets)[:4000],
        })
    execution["attempts"] = attempts
    return execution


def _sanitize_result(result: dict, user_keys: dict, expected_format: str,
                     artifact_root: str = "") -> dict:
    secrets = [str(value) for value in (user_keys or {}).values() if value]
    files = []
    allowed_root = os.path.abspath(artifact_root) if artifact_root else ""
    for item in result.get("downloaded_files") or []:
        file_record = {
            "name": os.path.basename(str(item.get("name") or "")),
            "size_bytes": max(0, int(item.get("size_bytes") or 0)),
        }
        candidate = os.path.abspath(str(item.get("path") or ""))
        if allowed_root and candidate:
            try:
                if (os.path.commonpath([allowed_root, candidate])
                        == allowed_root and os.path.isfile(candidate)):
                    file_record["artifact_ref"] = os.path.relpath(
                        candidate, allowed_root).replace(os.sep, "/")
            except (OSError, ValueError):
                pass
        if os.path.isfile(candidate):
            file_record["evidence"] = _inspect_output_file(
                candidate, file_record["name"], file_record["size_bytes"])
        files.append(file_record)
    requests = [{
        "attempt": item.get("attempt"),
        "method": str(item.get("method") or "GET").upper(),
        "url": _safe_url(str(item.get("url") or ""), secrets),
        "time": item.get("time"),
    } for item in (result.get("http_requests") or [])]

    expected = _expected_extensions(expected_format)
    output_ok = any(item["size_bytes"] > 0 for item in files)
    format_ok = not expected or any(
        os.path.splitext(item["name"])[1].casefold() in expected for item in files)
    status = result.get("status") or "failed"
    validation_message = "A non-empty output file was produced."
    if not output_ok:
        validation_message = "No non-empty output file was produced."
    elif not format_ok:
        status = "failed"
        validation_message = (
            f"Output did not match the expected format ({expected_format}).")

    return {
        "status": status,
        "error": _redact_text(result.get("error"), secrets)[:12000],
        "required_keys": list(result.get("required_keys") or []),
        "generated_code": _redact_text(
            result.get("generated_code"), secrets)[:60000],
        "execution": _sanitize_execution(result.get("execution"), secrets),
        "http_requests": requests,
        "downloaded_files": files,
        "output_evidence": [
            item["evidence"] for item in files if item.get("evidence")
        ],
        "validation": {
            "passed": status == "passed" and output_ok and format_ok,
            "output_present": output_ok,
            "format_match": format_ok,
            "expected_format": str(expected_format or ""),
            "message": validation_message,
        },
    }


def run_study(user_id: str, api_key: str, study_id: str,
              user_keys: dict | None = None):
    """Yield progress events while sequentially running resumable trials."""
    study = store.get_study(user_id, study_id)
    if study is None:
        yield {"type": "error", "error": "Experiment not found."}
        return

    keys = {
        str(name): str(value) for name, value in (user_keys or {}).items()
        if name and value
    }
    _clear_cancel(user_id, study_id)
    handbook_dir = user_data_sources.user_handbook_dir(user_id)
    agent = DataRetrieverAgent(
        api_key=api_key,
        extra_handbook_dirs=[handbook_dir] if os.path.isdir(handbook_dir) else [],
        evaluation=True,
    )
    candidates = [
        trial for trial in study.get("trials", [])
        if trial.get("status") in ("queued", "interrupted")
    ]
    if study.get("controls", {}).get("randomize_order"):
        # Stable pseudo-random ordering keeps reruns reproducible.
        candidates.sort(key=lambda trial: hashlib.sha256(
            f"{study_id}:{trial.get('id')}".encode("utf-8")).hexdigest())

    study["status"] = "running"
    store.save_study(user_id, study, study_id=study_id)
    yield {
        "type": "batch_started", "total": len(candidates),
        "message": f"Starting {len(candidates)} queued trial(s).",
    }

    for position, trial in enumerate(candidates, 1):
        if _is_cancelled(user_id, study_id):
            trial["status"] = "interrupted"
            trial["updated_at"] = _now()
            study["status"] = "ready"
            store.save_study(user_id, study, study_id=study_id)
            yield {
                "type": "batch_cancelled", "trial_id": trial.get("id"),
                "message": "Batch stopped. Remaining trials can be resumed.",
                "study": study,
            }
            return

        source = next((item for item in study.get("sources", [])
                       if item.get("id") == trial.get("source_id")), None)
        task = next((item for item in study.get("tasks", [])
                     if item.get("id") == trial.get("task_id")), None)
        if not source or not task:
            trial.update({
                "status": "failed",
                "notes": "The linked source or task no longer exists.",
                "updated_at": _now(),
            })
            store.save_study(user_id, study, study_id=study_id)
            yield {"type": "trial_finished", "trial": trial}
            continue

        source_id = _find_source_id(agent, source)
        if not source_id:
            trial.update({
                "status": "needs_handbook",
                "notes": "Link this source to a saved handbook before running it.",
                "updated_at": _now(),
            })
            store.save_study(user_id, study, study_id=study_id)
            yield {
                "type": "trial_finished", "trial": trial,
                "message": f"{source.get('name')}: saved handbook required.",
            }
            continue

        trial.update({
            "status": "running", "started_at": _now(), "updated_at": _now(),
        })
        store.save_study(user_id, study, study_id=study_id)
        yield {
            "type": "trial_started", "trial_id": trial.get("id"),
            "position": position, "total": len(candidates),
            "source": source.get("name"), "task": task.get("name"),
            "message": f"Running {source.get('name')} · {task.get('name')}",
        }

        started = time.monotonic()

        event_queue = queue.Queue()
        result_box = {}
        stream_buffer = {"text": "", "last": time.monotonic()}

        def stream_message(message):
            if _is_cancelled(user_id, study_id):
                raise InterruptedError("Experiment batch cancelled.")
            raw = str(message or "")
            stream_buffer["text"] += raw
            elapsed = time.monotonic() - stream_buffer["last"]
            if (len(stream_buffer["text"]) >= 180 or elapsed >= 0.4
                    or "\n" in raw):
                text = stream_buffer["text"].strip()
                stream_buffer["text"] = ""
                stream_buffer["last"] = time.monotonic()
            else:
                text = ""
            if text:
                event_queue.put({
                    "type": "trial_progress",
                    "trial_id": trial.get("id"),
                    "message": _redact_text(text, list(keys.values()))[-2000:],
                })

        output_dir = os.path.join(
            _study_output_root(user_id, study_id), _safe(trial.get("id"))[:16])
        def execute_trial():
            try:
                result_box["result"] = agent.run_controlled_handbook_trial(
                    data_request=task.get("description") or task.get("name"),
                    source_ID=source_id,
                    source_name=source.get("name"),
                    user_keys=keys,
                    stream_callback=stream_message,
                    output_dir=output_dir,
                    try_count=3,
                )
            except InterruptedError:
                result_box["interrupted"] = True
                result_box["result"] = {}
            except Exception as exc:
                result_box["result"] = {"status": "failed", "error": str(exc)}
            finally:
                remaining = stream_buffer["text"].strip()
                if remaining:
                    event_queue.put({
                        "type": "trial_progress",
                        "trial_id": trial.get("id"),
                        "message": _redact_text(
                            remaining, list(keys.values()))[-2000:],
                    })
                event_queue.put(None)

        worker = threading.Thread(
            target=execute_trial,
            name=f"experiment2-{_safe(trial.get('id'))}",
            daemon=True,
        )
        worker.start()
        while True:
            try:
                event = event_queue.get(timeout=1)
            except queue.Empty:
                yield {
                    "type": "heartbeat", "trial_id": trial.get("id"),
                    "message": "Trial is still working…",
                }
                continue
            if event is None:
                break
            yield event
        worker.join()
        result = result_box.get("result") or {}
        if result_box.get("interrupted"):
            trial["status"] = "interrupted"
            trial["notes"] = "Cancelled by the researcher."

        sanitized = _sanitize_result(
            result, keys, task.get("expected_format") or "",
            artifact_root=_study_output_root(user_id, study_id))
        trial.update({
            "status": sanitized["status"] if result else "interrupted",
            "notes": sanitized.get("error") or sanitized["validation"]["message"],
            "execution_result": sanitized,
            "duration_seconds": round(time.monotonic() - started, 3),
            "finished_at": _now(),
            "updated_at": _now(),
        })
        store.save_study(user_id, study, study_id=study_id)
        yield {
            "type": "trial_finished", "trial": trial,
            "message": (
                f"{source.get('name')} · {task.get('name')}: "
                f"{trial['status'].replace('_', ' ')}"
            ),
        }

        if trial["status"] == "interrupted":
            study["status"] = "ready"
            store.save_study(user_id, study, study_id=study_id)
            yield {
                "type": "batch_cancelled",
                "message": "Batch stopped. Interrupted trials can be resumed.",
                "study": study,
            }
            return

    study["status"] = "review"
    store.save_study(user_id, study, study_id=study_id)
    _clear_cancel(user_id, study_id)
    yield {
        "type": "batch_finished", "study": study,
        "message": "Batch complete. Review failures and validation evidence.",
    }


# ── Integrated Experiment 2 pipeline ────────────────────────────────────────

_GEN_STAGE_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[stage:(\w+):(\w+)\]\s*(.*)$")
_GEN_CHAT_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[chat\]\s*(.+)$")
_GEN_ARTIFACT_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[artifact:(\w+)\]\s*(\{.*\})\s*$")


def _pipeline_state(study: dict) -> dict:
    state = study.get("pipeline_state")
    if not isinstance(state, dict):
        state = {}
        study["pipeline_state"] = state
    state.setdefault("status", "not_started")
    state.setdefault("stages", [])
    state.setdefault("current_version", "")
    state.setdefault("required_keys", [])
    return state


def _record_stage(study: dict, key: str, label: str, description: str,
                  status: str, message: str = "", detail: str = "",
                  artifact=None) -> dict:
    state = _pipeline_state(study)
    stage = next(
        (item for item in state["stages"] if item.get("key") == key), None)
    if stage is None:
        stage = {
            "key": key,
            "label": label,
            "description": description,
            "status": "waiting",
            "message": "",
            "details": [],
            "created_at": _now(),
        }
        state["stages"].append(stage)
    stage["status"] = status
    if message:
        stage["message"] = str(message)[:4000]
    if detail:
        details = stage.setdefault("details", [])
        text = str(detail).strip()
        if text and (not details or details[-1] != text):
            details.append(text[:3000])
            del details[:-30]
    if artifact is not None:
        stage["artifact"] = artifact
    if status == "running" and not stage.get("started_at"):
        stage["started_at"] = _now()
    if status in ("complete", "warning", "error", "stopped"):
        stage["finished_at"] = _now()
    return dict(stage)


def _stage_event(stage: dict) -> dict:
    return {"type": "stage", "stage": stage}


def _persist_pipeline(user_id: str, study_id: str, study: dict) -> None:
    store.save_study(user_id, study, study_id=study_id)


def _run_logged_work(work):
    """Yield captured handbook-generator logs and return the worker result."""
    from agents.data_agent.handbook_generator import (
        start_log_capture, stop_log_capture)
    from agents.data_agent import usage_ledger

    events = queue.Queue()
    box = {}
    # Usage ledgers are thread-local -- two sessions generating at once must
    # never pool their tokens -- but work() runs on the thread started below,
    # so a ledger the caller opened is invisible to every LLM call the
    # generator makes, and record() silently no-ops against it. Carrying the
    # caller's ledger across is what makes generation tokens land anywhere at
    # all; without it the phase reports zero while plainly having spent money.
    # active() is read HERE, on the caller's thread, for the same reason.
    parent_ledger = usage_ledger.active()

    def worker():
        start_log_capture(callback=lambda line: events.put(("log", line)))
        try:
            with usage_ledger.adopt(parent_ledger):
                box["result"] = work()
        except Exception as exc:
            box["error"] = exc
        finally:
            box["logs"] = stop_log_capture()
            events.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            kind, payload = events.get(timeout=1)
        except queue.Empty:
            yield {"kind": "heartbeat"}
            continue
        if kind == "done":
            break
        yield {"kind": kind, "line": payload}
    if "error" in box:
        raise box["error"]
    return box.get("result"), box.get("logs", [])


def _run_quiet_work(work, message: str):
    """Run model work off-thread while yielding useful heartbeat messages."""
    from agents.data_agent import usage_ledger

    box = {}
    # Same thread-local ledger problem as _run_logged_work: mechanism
    # selection, trace analysis and the LLM judge all run through here, and
    # their tokens vanish unless the caller's ledger comes along.
    parent_ledger = usage_ledger.active()

    def worker():
        try:
            with usage_ledger.adopt(parent_ledger):
                box["result"] = work()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=1)
        if thread.is_alive():
            yield {"type": "heartbeat", "message": message}
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _version_directory(user_id: str, study_id: str) -> str:
    path = os.path.join(_study_output_root(user_id, study_id), "_handbooks")
    os.makedirs(path, exist_ok=True)
    return path


def _materialize_version(user_id: str, study_id: str, version: str,
                         source: dict) -> tuple[str, str]:
    version_id = f"exp2_{_safe(study_id)[:12]}_{version.lower()}"
    directory = _version_directory(user_id, study_id)
    path = os.path.join(directory, f"{version_id}.toml")
    user_data_sources._write_source(path, source)
    return version_id, directory


def _source_snapshot(source: dict) -> dict:
    return {
        field: str(source.get(field, "") or "")
        for field in user_data_sources.STORED_FIELDS
    }


def _upsert_version(study: dict, version: str, source: dict,
                    source_id: str, verification=None, reason="") -> dict:
    versions = study.setdefault("handbook_versions", [])
    existing = next(
        (item for item in versions if item.get("version") == version), None)
    record = existing or {
        "id": str(uuid.uuid4()),
        "version": version,
        "created_at": _now(),
    }
    record.update({
        "source_id": source_id,
        "handbook_snapshot": _source_snapshot(source),
        "verification": verification or {},
        "reason": str(reason or ""),
    })
    if existing is None:
        versions.append(record)
    return record


def _infer_expected_format(task_text: str) -> str:
    lowered = str(task_text or "").casefold()
    formats = (
        ("GeoPackage", ("geopackage", "gpkg")),
        ("GeoJSON", ("geojson",)),
        ("Shapefile", ("shapefile", ".shp")),
        ("GeoTIFF", ("geotiff", ".tif", ".tiff")),
        ("NetCDF", ("netcdf", ".nc")),
        ("Parquet", ("parquet", "geoparquet")),
        ("CSV", ("csv",)),
        ("JSON", ("json",)),
        ("OSM XML", (".osm", "osm xml")),
    )
    return next((
        label for label, terms in formats
        if any(term in lowered for term in terms)
    ), "")


def _trial_source_and_task(study: dict, mechanism: str = "rest") -> tuple[dict, dict]:
    source_name = str(study.get("source_name") or "").strip()
    docs_url = str(study.get("documentation_url") or "").strip()
    if not source_name:
        try:
            source_name = urlsplit(docs_url).hostname or docs_url
        except Exception:
            source_name = docs_url
    source = {
        "id": f"experiment_source_{_safe(mechanism)}",
        "name": source_name,
        "mechanism": mechanism,
        "website": docs_url,
        "docs_url": docs_url,
        "created_at": _now(),
    }
    task = {
        "id": "experiment_task",
        "name": "Configured retrieval task",
        "description": str(study.get("retrieval_task") or "").strip(),
        "mechanism": "all",
        "difficulty": "Experiment",
        "expected_format": _infer_expected_format(
            str(study.get("retrieval_task") or "")),
        "created_at": _now(),
    }
    sources = [
        item for item in (study.get("sources") or [])
        if item.get("id") != source["id"]
    ]
    sources.append(source)
    study["sources"] = sources
    study["tasks"] = [task]
    return source, task


def _execute_version(user_id: str, api_key: str, study_id: str, study: dict,
                     version_record: dict, task: dict, keys: dict,
                     stage_key: str, try_count: int = 1):
    source = version_record["handbook_snapshot"]
    version = version_record["version"]
    version_dir = _version_directory(user_id, study_id)
    agent = DataRetrieverAgent(
        api_key=api_key,
        extra_handbook_dirs=[version_dir],
        evaluation=True,
    )
    model = str(study.get("controls", {}).get("model") or "").strip()
    if model:
        agent.model = model
    agent.base_url = str(study.get("controls", {}).get("base_url") or "").strip() or None

    events = queue.Queue()
    box = {}
    last_progress = {"message": ""}

    def stream_message(message):
        if _is_cancelled(user_id, study_id):
            raise InterruptedError("Experiment pipeline cancelled.")
        # The retriever streams both useful lifecycle messages and raw LLM
        # code tokens. Raw tokens are intentionally not sent to the live UI:
        # they produced a noisy bullet-by-bullet JSON/code feed. The complete
        # generated program remains available as an expandable artifact after
        # execution.
        plain = re.sub(r"[*`#]", "", str(message or "")).strip()
        progress = ""
        attempt = re.search(
            r"Executing code \(trial\s+(\d+)/(\d+)\)", plain, re.I)
        repair = re.search(
            r"Debugging code \(attempt\s+(\d+)\)", plain, re.I)
        failed_attempt = re.search(
            r"Error on trial\s+(\d+)/(\d+)", plain, re.I)

        if plain == "Generating retrieval code from the pinned handbook.":
            progress = (
                f"Reading the task and preparing retrieval code from {version}.")
        elif plain == "Executing the generated retrieval program.":
            progress = "The retrieval program is ready. Starting execution."
        elif attempt:
            progress = (
                f"Executing retrieval attempt {attempt.group(1)} of "
                f"{attempt.group(2)} and monitoring the data response.")
        elif "Code executed successfully." in plain:
            progress = (
                "Execution succeeded. Checking downloaded files and validation "
                "evidence.")
        elif "Please check your API key." in plain:
            progress = (
                "The data source rejected the supplied credential. Execution "
                "has paused safely.")
        elif "Failed to execute code after" in plain:
            progress = (
                "All permitted execution attempts have finished. Preserving "
                "the failure evidence for analysis.")
        elif repair:
            progress = (
                f"Analyzing the execution error and preparing repair attempt "
                f"{repair.group(1)}.")
        elif failed_attempt:
            progress = (
                f"Execution attempt {failed_attempt.group(1)} did not pass. "
                "Collecting the error evidence for automatic repair.")

        progress = _redact_text(progress, list(keys.values())).strip()
        if progress and progress != last_progress["message"]:
            last_progress["message"] = progress
            events.put(("progress", progress))

    output_dir = os.path.join(
        _study_output_root(user_id, study_id),
        f"{_safe(version).lower()[:18]}_{uuid.uuid4().hex[:6]}")

    def worker():
        try:
            box["result"] = agent.run_controlled_handbook_trial(
                data_request=task["description"],
                source_ID=version_record["source_id"],
                source_name=source.get("data_source_name"),
                user_keys=keys,
                stream_callback=stream_message,
                output_dir=output_dir,
                try_count=max(1, int(try_count)),
            )
        except InterruptedError:
            box["interrupted"] = True
        except Exception as exc:
            box["result"] = {"status": "failed", "error": str(exc)}
        finally:
            events.put(("done", None))

    started = time.monotonic()
    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            kind, payload = events.get(timeout=1)
        except queue.Empty:
            yield {
                "type": "heartbeat", "stage_key": stage_key,
                "elapsed_seconds": round(time.monotonic() - started),
                "message": last_progress["message"]
                           or "Preparing the retrieval program…",
            }
            continue
        if kind == "done":
            break
        yield {
            "type": "stage_progress", "stage_key": stage_key,
            "message": payload,
        }
    if box.get("interrupted"):
        raise InterruptedError("Experiment pipeline cancelled.")
    return _sanitize_result(
        box.get("result") or {"status": "failed"},
        keys,
        task.get("expected_format") or "",
        artifact_root=_study_output_root(user_id, study_id),
    )


def _failure_signature(result: dict) -> str:
    text = " ".join([
        str(result.get("status") or ""),
        str(result.get("error") or ""),
        str((result.get("validation") or {}).get("message") or ""),
    ]).lower()
    text = re.sub(r"\b\d+\b", "#", text)
    return hashlib.sha256(text[-4000:].encode("utf-8")).hexdigest()


def _failure_routing(result: dict) -> dict:
    """Separate platform failures from evidence that may refine a handbook."""
    error = str(result.get("error") or "")
    lowered = error.casefold()
    if result.get("status") == "needs_credentials":
        return {
            "origin": "credential",
            "category": "Authentication or credentials",
            "refinable": False,
            "summary": "Execution requires valid source credentials.",
        }
    platform_patterns = (
        ("permissionerror", "Execution workspace permission failure"),
        ("modulenotfounderror", "Missing execution dependency"),
        ("no space left on device", "Execution workspace storage failure"),
        ("winerror 206", "Windows output path exceeded the supported length"),
        ("filename or extension is too long",
         "Output path exceeded the supported length"),
    )
    for pattern, category in platform_patterns:
        if pattern in lowered:
            return {
                "origin": "platform",
                "category": category,
                "refinable": False,
                "summary": (
                    "The retrieval was stopped by the local execution "
                "environment, not by evidence of a handbook defect."),
            }
    if "filenotfounderror" in lowered:
        quoted_paths = re.findall(r"['\"]([^'\"]{8,})['\"]", error)
        path_too_long = any(len(path) >= 248 for path in quoted_paths)
        if path_too_long:
            return {
                "origin": "platform",
                "category": "Windows output path exceeded the supported length",
                "refinable": False,
                "summary": (
                    "The retrieval was stopped by the local output path, not "
                    "by evidence of a handbook defect."),
            }
        return {
            "origin": "retrieval",
            "category": "Generated retrieval path error",
            "refinable": True,
            "summary": (
                "The generated retrieval program referenced a file or "
                "directory that did not exist."),
        }
    if any(pattern in lowered for pattern in (
            "temporary failure in name resolution", "connectionerror",
            "connecttimeout", "readtimeout", "service unavailable")):
        return {
            "origin": "provider_or_network",
            "category": "Provider or network failure",
            "refinable": False,
            "summary": (
                "The trace indicates an external service or network failure."),
        }
    return {
        "origin": "retrieval",
        "category": "",
        "refinable": True,
        "summary": "",
    }


def _pipeline_stop(user_id: str, study_id: str, study: dict, reason: str,
                   outcome: str, message: str):
    state = _pipeline_state(study)
    state.update({
        "status": "complete",
        "outcome": outcome,
        "stop_reason": reason,
        "finished_at": _now(),
        "required_keys": [],
    })
    study["status"] = "review"
    _clear_cancel(user_id, study_id)
    _persist_pipeline(user_id, study_id, study)
    return {
        "type": "pipeline_finished",
        "message": message,
        "stop_reason": reason,
        "outcome": outcome,
        "study": study,
    }


def _run_experiment_pipeline(user_id: str, api_key: str, study_id: str,
                             keys: dict, restart: bool):
    from agents.data_agent.handbook_generator import (
        HANDBOOK_GEN_MODEL, analyze_execution_trace, generate_handbook,
        refine_handbook)

    study = store.get_study(user_id, study_id)
    if study is None:
        raise ValueError("Experiment not found.")
    if not (str(study.get("source_name") or "").strip()
            or str(study.get("documentation_url") or "").strip()):
        raise ValueError("Enter a data source name or documentation URL.")
    if not str(study.get("retrieval_task") or "").strip():
        raise ValueError("Enter the retrieval task before starting the run.")

    if restart:
        study["pipeline_state"] = {}
        study["handbook_versions"] = []
        study["refinements"] = []
        study["trials"] = []

    state = _pipeline_state(study)
    if state.get("status") == "complete" and not restart:
        yield {
            "type": "pipeline_finished",
            "message": "This experimental pipeline is already complete.",
            "stop_reason": state.get("stop_reason"),
            "outcome": state.get("outcome"),
            "study": study,
        }
        return

    _clear_cancel(user_id, study_id)
    source_meta, task = _trial_source_and_task(study)
    state.update({
        "status": "running",
        "started_at": state.get("started_at") or _now(),
        "required_keys": [],
    })
    study["status"] = "running"
    if keys:
        study.setdefault("manual_interventions", []).append({
            "id": str(uuid.uuid4()),
            "type": "credential_provision",
            "note": "Required source credentials were supplied for this run; "
                    "credential values were not stored.",
            "created_at": _now(),
        })
    _persist_pipeline(user_id, study_id, study)
    yield {
        "type": "pipeline_started",
        "message": "Starting the live handbook experiment pipeline.",
        "study": study,
    }

    versions = study.setdefault("handbook_versions", [])
    if not versions:
        documentation = _record_stage(
            study, "documentation", "Documentation discovery",
            "Locate and read the official source documentation.", "running",
            "Finding the official documentation and access requirements.")
        _persist_pipeline(user_id, study_id, study)
        yield _stage_event(documentation)

        query = str(study.get("source_name") or "").strip()
        website = str(study.get("documentation_url") or "").strip()
        if not query:
            query = website
        model = str(study.get("controls", {}).get("model")
                    or HANDBOOK_GEN_MODEL)

        generation = _run_logged_work(lambda: generate_handbook(
            query,
            website=website,
            doc_urls=[website] if website else [],
            user_key=api_key,
            model=model,
            verify=True,
            data_source_keys=keys,
        ))
        generated = None
        while True:
            try:
                event = next(generation)
            except StopIteration as done:
                generated, _ = done.value
                break
            if _is_cancelled(user_id, study_id):
                raise InterruptedError("Experiment pipeline cancelled.")
            if event.get("kind") == "heartbeat":
                yield {"type": "heartbeat", "message": "Handbook generation is still working…"}
                continue
            line = event.get("line") or ""
            stage_match = _GEN_STAGE_RE.match(line)
            chat_match = _GEN_CHAT_RE.match(line)
            artifact_match = _GEN_ARTIFACT_RE.match(line)
            if stage_match:
                raw_stage, raw_status, message = stage_match.groups()
                mapping = {
                    "source": ("documentation", "Documentation discovery",
                               "Locate and read the official source documentation."),
                    "docs": ("documentation", "Documentation discovery",
                             "Locate and read the official source documentation."),
                    "draft": ("draft", "Initial handbook draft",
                              "Create the first source-specific handbook from the documentation."),
                    "verify": ("verification", "Self-verification",
                               "Run the normal sample-download verification and repair loop."),
                }
                if raw_stage not in mapping:
                    continue
                key, label, description = mapping[raw_stage]
                status = {
                    "start": "running", "progress": "running",
                    "complete": "complete", "warning": "warning",
                    "error": "error",
                }.get(raw_status, "running")
                # Source discovery is a substep of documentation; only the docs
                # completion closes the combined stage.
                if raw_stage == "source" and status == "complete":
                    status = "running"
                stage = _record_stage(
                    study, key, label, description, status, message,
                    detail=message if raw_status == "progress" else "")
                yield _stage_event(stage)
            elif chat_match:
                current = next(
                    (item for item in reversed(state["stages"])
                     if item.get("status") == "running"), None)
                if current:
                    stage = _record_stage(
                        study, current["key"], current["label"],
                        current["description"], current["status"],
                        current.get("message", ""), detail=chat_match.group(1))
                    yield _stage_event(stage)
            elif artifact_match:
                raw_stage, raw_json = artifact_match.groups()
                key = {"source": "documentation", "docs": "documentation",
                       "draft": "draft"}.get(raw_stage)
                current = next(
                    (item for item in state["stages"]
                     if item.get("key") == key), None)
                if key and current:
                    try:
                        artifact = json.loads(raw_json)
                    except ValueError:
                        artifact = {}
                    stage = _record_stage(
                        study, key, current["label"], current["description"],
                        current["status"], current.get("message", ""),
                        artifact=artifact)
                    yield _stage_event(stage)
            else:
                # Untagged generator logs (documentation fetches, verification
                # notes) are the finest-grained view of what the stage is doing,
                # so they are attached to whichever stage is currently running.
                detail = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line).strip()
                current = next(
                    (item for item in reversed(state["stages"])
                     if item.get("status") == "running"), None)
                if detail and current:
                    stage = _record_stage(
                        study, current["key"], current["label"],
                        current["description"], current["status"],
                        current.get("message", ""), detail=detail)
                    yield _stage_event(stage)

        source = dict(generated or {})
        verification = source.pop("_verification", None) or {}
        verify_status = (
            "complete" if verification.get("verified") else "warning")
        verify_stage = _record_stage(
            study, "verification", "Self-verification",
            "Run the normal sample-download verification and repair loop.",
            verify_status,
            verification.get("note") or "Self-verification finished.",
            artifact={"verification": verification})
        yield _stage_event(verify_stage)

        h0_source_id, _ = _materialize_version(
            user_id, study_id, "H0", source)
        slug = user_data_sources.write_user_source(user_id, source)
        study["source_slug"] = slug
        source_meta["handbook_slug"] = slug
        study["sources"] = [source_meta]
        h0 = _upsert_version(
            study, "H0", source, h0_source_id,
            verification=verification,
            reason="Initial handbook after the normal self-verification stage.")
        state["current_version"] = "H0"
        h0_stage = _record_stage(
            study, "h0", "Freeze baseline handbook H0",
            "Save the self-verified initial handbook as the immutable baseline.",
            "complete", "H0 has been saved and frozen for baseline evaluation.",
            artifact={
                "version": "H0",
                "source": h0["handbook_snapshot"],
                "verification": verification,
            })
        _persist_pipeline(user_id, study_id, study)
        yield _stage_event(h0_stage)
    else:
        latest = versions[-1]
        source = latest.get("handbook_snapshot") or {}
        source_id, _ = _materialize_version(
            user_id, study_id, latest.get("version", "H0"), source)
        latest["source_id"] = source_id

    max_refinements = max(
        0, int(study.get("controls", {}).get("max_refinements", 3)))
    previous_signature = ""

    while True:
        version_record = study["handbook_versions"][-1]
        version = version_record["version"]
        iteration = int(version[1:]) if version[1:].isdigit() else 0
        state["current_version"] = version
        retrieval_key = f"retrieval_{version.lower()}"
        evaluation_key = (
            "baseline" if version == "H0"
            else f"evaluation_{version.lower()}")
        retrieval_label = (
            "Retrieval task execution with H0" if version == "H0"
            else f"Retrieval task execution with {version}")
        retrieval_stage = _record_stage(
            study, retrieval_key, retrieval_label,
            f"Execute the configured retrieval task using only {version}.",
            "running", f"Passing the configured task into the {version} retrieval pipeline.")
        _persist_pipeline(user_id, study_id, study)
        yield _stage_event(retrieval_stage)

        retrieval = _execute_version(
            user_id, api_key, study_id, study, version_record, task, keys,
            retrieval_key)
        while True:
            try:
                event = next(retrieval)
            except StopIteration as done:
                result = done.value
                break
            if event.get("type") == "stage_progress" and event.get("message"):
                _record_stage(
                    study, retrieval_key, retrieval_label,
                    f"Execute the configured retrieval task using only {version}.",
                    "running", detail=event["message"])
            yield event

        if result.get("status") == "needs_credentials":
            required = list(result.get("required_keys") or [])
            state.update({
                "status": "awaiting_credentials",
                "required_keys": required,
            })
            study["status"] = "ready"
            stage = _record_stage(
                study, retrieval_key, retrieval_label,
                f"Execute the configured retrieval task using only {version}.",
                "warning",
                "Credentials are required before this retrieval can continue.",
                artifact={"required_keys": required})
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(stage)
            yield {
                "type": "credential_required",
                "required_keys": required,
                "message": "Enter the required credentials to resume. Values "
                           "will be used for the run only and never stored.",
                "study": study,
            }
            return

        retrieval_status = "complete" if result.get("status") == "passed" else "warning"
        retrieval_stage = _record_stage(
            study, retrieval_key, retrieval_label,
            f"Execute the configured retrieval task using only {version}.",
            retrieval_status,
            "The retrieval task passed." if result.get("status") == "passed"
            else "The retrieval task finished with evidence requiring review.",
            artifact={"execution_result": result})
        yield _stage_event(retrieval_stage)

        trial = {
            "id": str(uuid.uuid4()),
            "source_id": source_meta["id"],
            "task_id": task["id"],
            "mechanism": source_meta["mechanism"],
            "access_mechanisms": list(study.get("mechanisms") or []),
            "condition": "baseline" if version == "H0" else "refined",
            "handbook_version": version,
            "repetition": 1,
            "attempt": iteration + 1,
            "status": result.get("status", "failed"),
            "failure_category": "",
            "notes": result.get("error")
                     or (result.get("validation") or {}).get("message", ""),
            "execution_result": result,
            "created_at": _now(),
            "updated_at": _now(),
        }
        study.setdefault("trials", []).append(trial)
        evaluation_label = (
            "Baseline evaluation" if version == "H0"
            else f"Post-refinement evaluation · {version}")
        evaluation_stage = _record_stage(
            study, evaluation_key, evaluation_label,
            "Collect retrieval success, failed requests, validation results, "
            "and operational evidence.",
            retrieval_status,
            "Baseline evidence has been recorded." if version == "H0"
            else f"{version} evaluation evidence has been recorded.",
            artifact={
                "version": version,
                "status": trial["status"],
                "execution_result": result,
                "trial_id": trial["id"],
            })
        _persist_pipeline(user_id, study_id, study)
        yield _stage_event(evaluation_stage)

        if result.get("status") == "passed":
            yield _pipeline_stop(
                user_id, study_id, study, "validation_satisfied", "passed",
                f"{version} satisfied the retrieval validation criteria.")
            return
        if iteration >= max_refinements:
            yield _pipeline_stop(
                user_id, study_id, study, "maximum_refinements", "stopped",
                "The configured maximum number of refinements was reached.")
            return

        signature = _failure_signature(result)
        if (previous_signature and signature == previous_signature
                and study.get("controls", {}).get(
                    "stop_on_repeated_error", True)):
            yield _pipeline_stop(
                user_id, study_id, study, "no_further_improvement", "stopped",
                "The same failure persisted after refinement, so the pipeline "
                "stopped because no further improvement was observed.")
            return
        previous_signature = signature

        analysis_key = f"analysis_{version.lower()}"
        analysis_stage = _record_stage(
            study, analysis_key, f"Execution-trace analysis · {version}",
            "Identify handbook deficiencies and distinguish them from external failures.",
            "running", "Analyzing the collected execution trace.")
        yield _stage_event(analysis_stage)
        analysis_work = _run_quiet_work(
            lambda: analyze_execution_trace(
                version_record["handbook_snapshot"],
                task["description"],
                result,
                user_key=api_key,
                model=str(study.get("controls", {}).get("model")
                          or HANDBOOK_GEN_MODEL),
            ),
            "Analyzing endpoints, parameters, validation evidence, and operational issues…",
        )
        while True:
            try:
                event = next(analysis_work)
            except StopIteration as done:
                analysis = done.value
                break
            event["stage_key"] = analysis_key
            yield event
        trial["failure_category"] = str(
            analysis.get("failure_category") or "Unknown")
        trial["trace_analysis"] = analysis
        analysis_stage = _record_stage(
            study, analysis_key, f"Execution-trace analysis · {version}",
            "Identify handbook deficiencies and distinguish them from external failures.",
            "complete", analysis.get("summary") or "Trace analysis complete.",
            artifact={"analysis": analysis})
        _persist_pipeline(user_id, study_id, study)
        yield _stage_event(analysis_stage)

        if not analysis.get("should_refine"):
            yield _pipeline_stop(
                user_id, study_id, study, "no_refinable_deficiency", "stopped",
                "The trace did not identify a handbook deficiency that could "
                "be improved automatically.")
            return

        next_version = f"H{iteration + 1}"
        refine_key = f"refinement_{next_version.lower()}"
        refine_stage = _record_stage(
            study, refine_key, f"Handbook refinement · {next_version}",
            "Revise the handbook using only the recorded execution evidence.",
            "running", f"Creating {next_version} from the {version} trace analysis.")
        yield _stage_event(refine_stage)
        feedback = (
            "Refine this handbook using the controlled execution-trace "
            "analysis below. Apply only evidence-grounded corrections and "
            "preserve instructions that already work.\n\n"
            f"Retrieval task: {task['description']}\n"
            f"Trace analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
            f"Execution error: {result.get('error', '')}"
        )
        refine_work = _run_logged_work(lambda: refine_handbook(
            version_record["handbook_snapshot"],
            feedback,
            website=str(study.get("documentation_url") or ""),
            fetch_docs=True,
            user_key=api_key,
            model=str(study.get("controls", {}).get("model")
                      or HANDBOOK_GEN_MODEL),
        ))
        while True:
            try:
                event = next(refine_work)
            except StopIteration as done:
                refined, _ = done.value
                break
            if event.get("kind") == "heartbeat":
                yield {
                    "type": "heartbeat", "stage_key": refine_key,
                    "message": f"{next_version} refinement is still working…",
                }
            elif event.get("line"):
                detail = re.sub(
                    r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", event["line"]).strip()
                if detail:
                    _record_stage(
                        study, refine_key,
                        f"Handbook refinement · {next_version}",
                        "Revise the handbook using only the recorded execution evidence.",
                        "running", detail=detail)
                yield {
                    "type": "stage_progress", "stage_key": refine_key,
                    "message": detail,
                }
        next_source = refined["source"]
        before = version_record["handbook_snapshot"]
        if (next_source.get("handbook") == before.get("handbook")
                and next_source.get("code_example") == before.get("code_example")):
            yield _pipeline_stop(
                user_id, study_id, study, "no_handbook_change", "stopped",
                "The proposed refinement produced no handbook change, so the "
                "pipeline stopped.")
            return

        next_source_id, _ = _materialize_version(
            user_id, study_id, next_version, next_source)
        user_data_sources.write_user_source(user_id, next_source)
        next_record = _upsert_version(
            study, next_version, next_source, next_source_id,
            reason=analysis.get("summary") or "Execution-grounded refinement.")
        refinement_record = {
            "id": str(uuid.uuid4()),
            "trial_id": trial["id"],
            "source_id": source_meta["id"],
            "task_id": task["id"],
            "from_version": version,
            "to_version": next_version,
            "failure_category": trial["failure_category"],
            "evidence": analysis,
            "assistant_message": refined.get("assistant_message", ""),
            "status": "applied",
            "created_at": _now(),
        }
        study.setdefault("refinements", []).append(refinement_record)
        state["current_version"] = next_version
        refine_stage = _record_stage(
            study, refine_key, f"Handbook refinement · {next_version}",
            "Revise the handbook using only the recorded execution evidence.",
            "complete",
            refined.get("assistant_message") or f"{next_version} was created.",
            artifact={
                "version": next_version,
                "from_version": version,
                "analysis": analysis,
                "source": next_record["handbook_snapshot"],
                "refinement_id": refinement_record["id"],
            })
        _persist_pipeline(user_id, study_id, study)
        yield _stage_event(refine_stage)


_MECHANISM_LABELS = {
    "rest": "REST API",
    "stac": "STAC",
    "ogc": "OGC Service",
    "sql": "SQL Interface",
    "bulk": "Bulk Download",
    "http_file": "HTTP File Download",
    "arcgis_featureserver": "ArcGIS FeatureServer",
}
_MECHANISM_GUIDANCE = {
    "rest": (
        "Use documented REST/HTTP query endpoints only. Do not substitute a "
        "bulk archive when evaluating this condition."),
    "stac": (
        "Use STAC catalogs, collections, item search, and asset retrieval. "
        "Do not substitute a non-STAC download method."),
    "ogc": (
        "Use the documented OGC service or OGC API interface such as WFS, "
        "WCS, WMS, or OGC API Features/Coverages."),
    "sql": (
        "Use the documented database or SQL query interface. Do not replace "
        "it with an unrelated REST or bulk workflow."),
    "bulk": (
        "Use an official bulk file, regional extract, archive, or repository "
        "appropriate to the requested scale. Do not use a small-area editing "
        "API to simulate a state- or country-scale bulk extract."),
    "http_file": (
        "Use a single, direct HTTP(S) file download (a plain static file "
        "link), not a query API, catalog search, or a multi-file bulk "
        "archive/repository."),
    "arcgis_featureserver": (
        "Use an Esri ArcGIS FeatureServer/MapServer REST endpoint "
        "(query/queryRelatedRecords over /FeatureServer or /MapServer), not "
        "a generic REST API, OGC service, or file download."),
}


def _condition_record(state: dict, mechanism: str) -> dict:
    conditions = state.setdefault("conditions", [])
    condition = next(
        (item for item in conditions
         if item.get("mechanism") == mechanism), None)
    if condition is None:
        condition = {
            "mechanism": mechanism,
            "label": _MECHANISM_LABELS.get(mechanism, mechanism),
            "status": "queued",
            "outcome": "",
            "stop_reason": "",
            "current_version": "",
            "started_at": "",
            "finished_at": "",
        }
        conditions.append(condition)
    return condition


def _condition_key(mechanism: str, key: str) -> str:
    return f"{_safe(mechanism).lower()}_{key}"


def _condition_label(mechanism: str, label: str) -> str:
    return f"{_MECHANISM_LABELS.get(mechanism, mechanism)} · {label}"


def _version_name(mechanism: str, iteration: int) -> str:
    return f"{_safe(mechanism).upper()}-H{max(0, int(iteration))}"


def _version_iteration(version_record: dict) -> int:
    if isinstance(version_record.get("iteration"), int):
        return max(0, version_record["iteration"])
    match = re.search(r"H(\d+)$", str(version_record.get("version") or ""))
    return int(match.group(1)) if match else 0


def _condition_versions(study: dict, mechanism: str) -> list[dict]:
    return sorted([
        item for item in study.get("handbook_versions", [])
        if item.get("mechanism") == mechanism
    ], key=_version_iteration)


def _finish_condition(user_id: str, study_id: str, study: dict,
                      condition: dict, outcome: str, reason: str,
                      message: str) -> dict:
    condition.update({
        "status": "complete",
        "outcome": outcome,
        "stop_reason": reason,
        "finished_at": _now(),
    })
    _persist_pipeline(user_id, study_id, study)
    return {
        "type": "condition_finished",
        "mechanism": condition["mechanism"],
        "outcome": outcome,
        "stop_reason": reason,
        "message": message,
        "study": study,
    }


def _semantic_analysis(semantic: dict) -> dict:
    gap = str(semantic.get("handbook_gap") or "").strip()
    deficiencies = []
    if gap:
        deficiencies.append({
            "evidence": semantic.get("summary") or (
                "Semantic validation did not satisfy the retrieval task."),
            "handbook_gap": gap,
            "recommended_change": (
                "Revise the handbook to address this evidence-grounded gap."),
        })
    return {
        "summary": semantic.get("summary") or (
            "The completed program did not satisfy semantic task validation."),
        "should_refine": bool(semantic.get("should_refine_handbook") and gap),
        "failure_category": semantic.get("failure_category")
                            or "Semantic validation failure",
        "deficiencies": deficiencies,
        "operational_issues": [],
        "validation_assessment": semantic.get("summary") or "",
    }


def _run_experiment_pipeline_v2(user_id: str, api_key: str, study_id: str,
                                keys: dict, restart: bool,
                                confirm_mechanisms=None):
    """Run independent H0/refinement conditions for every selected mechanism."""
    from agents.data_agent.handbook_generator import (
        HANDBOOK_GEN_MODEL, analyze_execution_trace,
        evaluate_retrieval_result, generate_handbook, refine_handbook)

    confirm_mechanisms = set(confirm_mechanisms or [])

    study = store.get_study(user_id, study_id)
    if study is None:
        raise ValueError("Experiment not found.")
    source_name = str(study.get("source_name") or "").strip()
    website = str(study.get("documentation_url") or "").strip()
    task_text = str(study.get("retrieval_task") or "").strip()
    if not (source_name or website):
        raise ValueError("Enter a data source name or documentation URL.")
    if not task_text:
        raise ValueError("Enter the retrieval task before starting the run.")

    mechanisms = list(dict.fromkeys(
        study.get("mechanisms") or ["rest"]))
    prior_state = study.get("pipeline_state") or {}
    legacy_reset = bool(
        not restart
        and not prior_state.get("conditions")
        and (
            prior_state.get("stages")
            or study.get("handbook_versions")
            or study.get("trials")
        )
    )
    restart = bool(restart or legacy_reset)
    if restart:
        study["pipeline_state"] = {}
        study["handbook_versions"] = []
        study["refinements"] = []
        study["trials"] = []
        study["sources"] = []
        study["tasks"] = []

    state = _pipeline_state(study)
    if state.get("status") == "complete" and not restart:
        yield {
            "type": "pipeline_finished",
            "message": "This experimental pipeline is already complete.",
            "stop_reason": state.get("stop_reason"),
            "outcome": state.get("outcome"),
            "study": study,
        }
        return

    _clear_cancel(user_id, study_id)
    state.update({
        "status": "running",
        "started_at": state.get("started_at") or _now(),
        "required_keys": [],
        "mechanism_order": mechanisms,
    })
    for mechanism in mechanisms:
        _condition_record(state, mechanism)
    study["status"] = "running"
    if keys:
        study.setdefault("manual_interventions", []).append({
            "id": str(uuid.uuid4()),
            "type": "credential_provision",
            "note": "Source credentials were supplied for this run; values "
                    "were not stored.",
            "created_at": _now(),
        })
    _persist_pipeline(user_id, study_id, study)
    yield {
        "type": "pipeline_started",
        "message": (
            (
                "The earlier run used the legacy shared-H0 design, so it was "
                "reset before starting the corrected experiment. "
                if legacy_reset else ""
            )
            + f"Starting {len(mechanisms)} mechanism-specific experimental "
              f"condition{'s' if len(mechanisms) != 1 else ''}."),
        "study": study,
    }

    model = str(study.get("controls", {}).get("model")
                or HANDBOOK_GEN_MODEL)
    # Routes generation, refinement, analysis, and evaluation calls to a
    # self-hosted open-source model server. Source discovery/access analysis
    # for H0 fall back to a Tavily tool-call loop in that case, since local
    # servers don't implement OpenAI's Responses API or hosted web_search.
    base_url = str(study.get("controls", {}).get("base_url") or "").strip() or None
    max_refinements = max(
        0, int(study.get("controls", {}).get("max_refinements", 3)))

    for mechanism in mechanisms:
        if _is_cancelled(user_id, study_id):
            raise InterruptedError("Experiment pipeline cancelled.")
        condition = _condition_record(state, mechanism)
        if condition.get("status") == "complete":
            continue
        condition.update({
            "status": "running",
            "started_at": condition.get("started_at") or _now(),
        })
        state["active_mechanism"] = mechanism
        source_meta, task = _trial_source_and_task(study, mechanism)
        _persist_pipeline(user_id, study_id, study)
        yield {
            "type": "condition_started",
            "mechanism": mechanism,
            "message": (
                f"Starting the {_MECHANISM_LABELS.get(mechanism, mechanism)} "
                "condition."),
            "study": study,
        }

        versions = _condition_versions(study, mechanism)
        if not versions:
            documentation_key = _condition_key(mechanism, "documentation")
            stage = _record_stage(
                study, documentation_key,
                _condition_label(mechanism, "Documentation discovery"),
                "Locate documentation for this exact access mechanism.",
                "running",
                "Finding official documentation and access requirements for "
                f"{_MECHANISM_LABELS.get(mechanism, mechanism)}.")
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(stage)

            base_query = source_name or website
            mechanism_query = (
                f"{base_query}\n\nRequired experimental access mechanism: "
                f"{_MECHANISM_LABELS.get(mechanism, mechanism)}. "
                f"{_MECHANISM_GUIDANCE.get(mechanism, '')}\n"
                f"Representative retrieval task for capability context: "
                f"{task_text}\nGenerate and verify a handbook specifically "
                "for this mechanism; do not silently substitute another "
                "mechanism.")
            generation = _run_logged_work(lambda: generate_handbook(
                mechanism_query,
                website=website,
                doc_urls=[website] if website else [],
                user_key=api_key,
                model=model,
                verify=True,
                data_source_keys=keys,
                base_url=base_url,
            ))
            generated = None
            while True:
                try:
                    event = next(generation)
                except StopIteration as done:
                    generated, _ = done.value
                    break
                if _is_cancelled(user_id, study_id):
                    raise InterruptedError("Experiment pipeline cancelled.")
                if event.get("kind") == "heartbeat":
                    yield {
                        "type": "heartbeat",
                        "stage_key": documentation_key,
                        "message": (
                            f"{_MECHANISM_LABELS.get(mechanism, mechanism)} "
                            "handbook generation is still working…"),
                    }
                    continue
                line = event.get("line") or ""
                stage_match = _GEN_STAGE_RE.match(line)
                chat_match = _GEN_CHAT_RE.match(line)
                artifact_match = _GEN_ARTIFACT_RE.match(line)
                if stage_match:
                    raw_stage, raw_status, message = stage_match.groups()
                    mapping = {
                        "source": ("documentation", "Documentation discovery",
                                   "Locate official mechanism documentation."),
                        "docs": ("documentation", "Documentation discovery",
                                 "Read the official mechanism documentation."),
                        "draft": ("draft", "Initial handbook draft",
                                  "Create a mechanism-specific handbook."),
                        "verify": ("verification", "Self-verification",
                                   "Run the normal sample-download verification."),
                    }
                    if raw_stage not in mapping:
                        continue
                    raw_key, raw_label, description = mapping[raw_stage]
                    key = _condition_key(mechanism, raw_key)
                    status = {
                        "start": "running", "progress": "running",
                        "complete": "complete", "warning": "warning",
                        "error": "error",
                    }.get(raw_status, "running")
                    if raw_stage == "source" and status == "complete":
                        status = "running"
                    stage = _record_stage(
                        study, key,
                        _condition_label(mechanism, raw_label),
                        description, status, message,
                        detail=message if raw_status == "progress" else "")
                    yield _stage_event(stage)
                elif chat_match:
                    current = next((
                        item for item in reversed(state["stages"])
                        if item.get("status") == "running"
                        and item.get("key", "").startswith(
                            f"{_safe(mechanism).lower()}_")
                    ), None)
                    if current:
                        stage = _record_stage(
                            study, current["key"], current["label"],
                            current["description"], current["status"],
                            current.get("message", ""),
                            detail=chat_match.group(1))
                        yield _stage_event(stage)
                elif artifact_match:
                    raw_stage, raw_json = artifact_match.groups()
                    raw_key = {
                        "source": "documentation", "docs": "documentation",
                        "draft": "draft",
                    }.get(raw_stage)
                    key = (
                        _condition_key(mechanism, raw_key)
                        if raw_key else "")
                    current = next((
                        item for item in state["stages"]
                        if item.get("key") == key
                    ), None)
                    if current:
                        try:
                            artifact = json.loads(raw_json)
                        except ValueError:
                            artifact = {}
                        stage = _record_stage(
                            study, key, current["label"],
                            current["description"], current["status"],
                            current.get("message", ""), artifact=artifact)
                        yield _stage_event(stage)

            source = dict(generated or {})
            verification = source.pop("_verification", None) or {}
            documentation_stage = _record_stage(
                study, documentation_key,
                _condition_label(mechanism, "Documentation discovery"),
                "Locate documentation for this exact access mechanism.",
                "complete",
                "The mechanism-specific documentation has been collected.")
            yield _stage_event(documentation_stage)
            draft_stage = _record_stage(
                study, _condition_key(mechanism, "draft"),
                _condition_label(mechanism, "Initial handbook draft"),
                "Create a mechanism-specific handbook.",
                "complete",
                "The documentation-only handbook draft has been created.",
                artifact={"source": _source_snapshot(source)})
            yield _stage_event(draft_stage)
            verify_key = _condition_key(mechanism, "verification")
            verify_stage = _record_stage(
                study, verify_key,
                _condition_label(mechanism, "Self-verification"),
                "Run the normal sample-download verification.",
                "complete" if verification.get("verified") else "warning",
                verification.get("note") or "Self-verification finished.",
                artifact={"verification": verification})
            yield _stage_event(verify_stage)

            version = _version_name(mechanism, 0)
            source_id, _ = _materialize_version(
                user_id, study_id, version, source)
            slug = user_data_sources.write_user_source(user_id, source)
            study.setdefault("source_slugs", {})[mechanism] = slug
            study["source_slug"] = slug
            source_meta["handbook_slug"] = slug
            version_record = _upsert_version(
                study, version, source, source_id,
                verification=verification,
                reason=(
                    "Mechanism-specific initial handbook after normal "
                    "self-verification."))
            version_record.update({
                "mechanism": mechanism,
                "base_version": "H0",
                "iteration": 0,
                "display_version": (
                    f"{_MECHANISM_LABELS.get(mechanism, mechanism)} H0"),
            })
            condition["current_version"] = version
            h0_stage = _record_stage(
                study, _condition_key(mechanism, "h0"),
                _condition_label(mechanism, "Freeze baseline handbook H0"),
                "Save the self-verified mechanism-specific baseline.",
                "complete",
                "The mechanism-specific H0 has been frozen.",
                artifact={
                    "version": version,
                    "mechanism": mechanism,
                    "source": version_record["handbook_snapshot"],
                    "verification": verification,
                })
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(h0_stage)

        if not condition.get("test_confirmed") and mechanism not in confirm_mechanisms:
            versions = _condition_versions(study, mechanism)
            latest = versions[-1] if versions else None
            confirm_key = _condition_key(mechanism, "confirm")
            confirm_stage = _record_stage(
                study, confirm_key,
                _condition_label(mechanism, "Confirm retrieval test"),
                "Wait for approval before spending a live retrieval attempt "
                "on this handbook.",
                "warning",
                "The H0 handbook is ready. Confirm to run the retrieval test.")
            condition["status"] = "awaiting_test_confirmation"
            state["status"] = "awaiting_test_confirmation"
            state["pending_confirmation_mechanism"] = mechanism
            study["status"] = "ready"
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(confirm_stage)
            yield {
                "type": "retrieval_confirmation_required",
                "mechanism": mechanism,
                "version": latest["version"] if latest else "",
                "message": (
                    f"The {_MECHANISM_LABELS.get(mechanism, mechanism)} H0 "
                    "handbook is ready. Run the retrieval test now?"),
                "study": study,
            }
            return
        condition["test_confirmed"] = True

        previous_signature = str(condition.get("last_failure_signature") or "")
        while True:
            versions = _condition_versions(study, mechanism)
            version_record = versions[-1]
            version = version_record["version"]
            iteration = _version_iteration(version_record)
            base_version = f"H{iteration}"
            condition["current_version"] = version
            state["current_version"] = version
            retrieval_key = _condition_key(
                mechanism, f"retrieval_h{iteration}")
            retrieval_label = _condition_label(
                mechanism, f"Retrieval task execution with {base_version}")
            retrieval_stage = _record_stage(
                study, retrieval_key, retrieval_label,
                "Execute one controlled baseline attempt without hidden "
                "code-repair retries.",
                "running",
                f"Executing the configured task with {base_version}.")
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(retrieval_stage)

            retrieval = _execute_version(
                user_id, api_key, study_id, study, version_record, task, keys,
                retrieval_key, try_count=1)
            while True:
                try:
                    event = next(retrieval)
                except StopIteration as done:
                    result = done.value
                    break
                if event.get("type") == "stage_progress" and event.get("message"):
                    _record_stage(
                        study, retrieval_key, retrieval_label,
                        "Execute one controlled attempt.", "running",
                        detail=event["message"])
                yield event

            if result.get("status") == "needs_credentials":
                required = list(result.get("required_keys") or [])
                condition["status"] = "awaiting_credentials"
                state.update({
                    "status": "awaiting_credentials",
                    "required_keys": required,
                })
                study["status"] = "ready"
                stage = _record_stage(
                    study, retrieval_key, retrieval_label,
                    "Execute one controlled attempt.", "warning",
                    "Valid credentials are required before this condition can "
                    "continue.", artifact={"required_keys": required})
                _persist_pipeline(user_id, study_id, study)
                yield _stage_event(stage)
                yield {
                    "type": "credential_required",
                    "required_keys": required,
                    "mechanism": mechanism,
                    "message": (
                        "Enter valid credentials to resume. Values are used "
                        "only for this run and are never stored."),
                    "study": study,
                }
                return

            routing = _failure_routing(result)
            result["failure_routing"] = routing
            semantic = None
            if result.get("status") == "passed":
                semantic_key = _condition_key(
                    mechanism, f"semantic_h{iteration}")
                semantic_stage = _record_stage(
                    study, semantic_key,
                    _condition_label(
                        mechanism,
                        f"Semantic task validation · {base_version}"),
                    "Validate mechanism, parameters, spatial and temporal "
                    "logic, and output content.",
                    "running",
                    "Checking whether the output actually satisfies the task.")
                yield _stage_event(semantic_stage)
                semantic_work = _run_quiet_work(
                    lambda: evaluate_retrieval_result(
                        version_record["handbook_snapshot"], task_text, result,
                        mechanism=_MECHANISM_LABELS.get(
                            mechanism, mechanism),
                        user_key=api_key, model=model, base_url=base_url),
                    "Inspecting request parameters and output evidence…")
                while True:
                    try:
                        event = next(semantic_work)
                    except StopIteration as done:
                        semantic = done.value
                        break
                    event["stage_key"] = semantic_key
                    yield event
                result["semantic_validation"] = semantic
                result["validation"]["semantic_passed"] = bool(
                    semantic.get("task_completed"))
                result["validation"]["message"] = (
                    semantic.get("summary")
                    or result["validation"].get("message", ""))
                if not semantic.get("task_completed"):
                    result["status"] = "failed_validation"
                semantic_stage = _record_stage(
                    study, semantic_key,
                    _condition_label(
                        mechanism,
                        f"Semantic task validation · {base_version}"),
                    "Validate the completed retrieval against the task.",
                    "complete" if semantic.get("task_completed") else "warning",
                    semantic.get("summary") or "Semantic validation finished.",
                    artifact={
                        "analysis": semantic,
                        "execution_result": result,
                    })
                yield _stage_event(semantic_stage)

            task_passed = (
                result.get("status") == "passed"
                and semantic is not None
                and semantic.get("task_completed"))
            retrieval_stage = _record_stage(
                study, retrieval_key, retrieval_label,
                "Execute one controlled attempt.",
                "complete" if task_passed else "warning",
                "The retrieval and semantic validation passed."
                if task_passed else (
                    routing.get("summary")
                    or "The retrieval produced evidence requiring review."),
                artifact={"execution_result": result})
            yield _stage_event(retrieval_stage)

            trial = {
                "id": str(uuid.uuid4()),
                "source_id": source_meta["id"],
                "task_id": task["id"],
                "mechanism": mechanism,
                "access_mechanisms": [mechanism],
                "condition": "baseline" if iteration == 0 else "refined",
                "handbook_version": version,
                "base_version": base_version,
                "repetition": 1,
                "attempt": 1,
                "status": "passed" if task_passed else result.get(
                    "status", "failed"),
                "failure_category": (
                    (semantic or {}).get("failure_category")
                    or routing.get("category", "")),
                "notes": result.get("error")
                         or (result.get("validation") or {}).get("message", ""),
                "execution_result": result,
                "semantic_validation": semantic or {},
                "created_at": _now(),
                "updated_at": _now(),
            }
            study.setdefault("trials", []).append(trial)
            evaluation_key = _condition_key(
                mechanism,
                "baseline" if iteration == 0 else f"evaluation_h{iteration}")
            evaluation_stage = _record_stage(
                study, evaluation_key,
                _condition_label(
                    mechanism,
                    "Baseline evaluation" if iteration == 0
                    else f"Post-refinement evaluation · {base_version}"),
                "Record execution, semantic checks, and operational evidence.",
                "complete" if task_passed else "warning",
                "Evaluation evidence has been recorded.",
                artifact={
                    "version": version,
                    "mechanism": mechanism,
                    "status": trial["status"],
                    "execution_result": result,
                    "trial_id": trial["id"],
                })
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(evaluation_stage)

            if task_passed:
                yield _finish_condition(
                    user_id, study_id, study, condition, "passed",
                    "validation_satisfied",
                    f"{_MECHANISM_LABELS.get(mechanism, mechanism)} "
                    f"{base_version} satisfied all validation criteria.")
                break
            if not routing.get("refinable", True):
                trial["failure_category"] = routing.get(
                    "category") or trial["failure_category"]
                yield _finish_condition(
                    user_id, study_id, study, condition, "invalid",
                    f"{routing.get('origin', 'platform')}_failure",
                    routing.get("summary") or (
                        "The condition stopped because the failure was not a "
                        "handbook deficiency."))
                break
            if iteration >= max_refinements:
                yield _finish_condition(
                    user_id, study_id, study, condition, "stopped",
                    "maximum_refinements",
                    "The configured refinement limit was reached.")
                break

            signature = _failure_signature(result)
            if (previous_signature and signature == previous_signature
                    and study.get("controls", {}).get(
                        "stop_on_repeated_error", True)):
                yield _finish_condition(
                    user_id, study_id, study, condition, "stopped",
                    "no_further_improvement",
                    "The same failure persisted after refinement.")
                break
            previous_signature = signature
            condition["last_failure_signature"] = signature

            analysis_key = _condition_key(
                mechanism, f"analysis_h{iteration}")
            analysis_stage = _record_stage(
                study, analysis_key,
                _condition_label(
                    mechanism,
                    f"Execution-trace analysis · {base_version}"),
                "Identify evidence-grounded handbook deficiencies.",
                "running", "Analyzing the controlled trace.")
            yield _stage_event(analysis_stage)
            if semantic is not None:
                analysis = _semantic_analysis(semantic)
            else:
                analysis_work = _run_quiet_work(
                    lambda: analyze_execution_trace(
                        version_record["handbook_snapshot"], task_text, result,
                        user_key=api_key, model=model, base_url=base_url),
                    "Analyzing endpoints, parameters, and operational evidence…")
                while True:
                    try:
                        event = next(analysis_work)
                    except StopIteration as done:
                        analysis = done.value
                        break
                    event["stage_key"] = analysis_key
                    yield event
            trial["failure_category"] = str(
                analysis.get("failure_category") or "Unknown")
            trial["trace_analysis"] = analysis
            analysis_stage = _record_stage(
                study, analysis_key,
                _condition_label(
                    mechanism,
                    f"Execution-trace analysis · {base_version}"),
                "Identify evidence-grounded handbook deficiencies.",
                "complete", analysis.get("summary") or "Analysis complete.",
                artifact={"analysis": analysis})
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(analysis_stage)

            if not analysis.get("should_refine"):
                yield _finish_condition(
                    user_id, study_id, study, condition, "stopped",
                    "no_refinable_deficiency",
                    "No evidence-grounded handbook deficiency was identified.")
                break

            next_iteration = iteration + 1
            next_version = _version_name(mechanism, next_iteration)
            next_base = f"H{next_iteration}"
            refine_key = _condition_key(
                mechanism, f"refinement_h{next_iteration}")
            refine_stage = _record_stage(
                study, refine_key,
                _condition_label(
                    mechanism, f"Handbook refinement · {next_base}"),
                "Revise the handbook using only recorded evidence.",
                "running", f"Creating {next_base} from the controlled trace.")
            yield _stage_event(refine_stage)
            feedback = (
                "Refine this mechanism-specific handbook using only the "
                "controlled evidence below. Do not change the required access "
                f"mechanism ({_MECHANISM_LABELS.get(mechanism, mechanism)}).\n\n"
                f"Retrieval task: {task_text}\n"
                f"Trace analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
                f"Execution error: {result.get('error', '')}"
            )
            refine_work = _run_logged_work(lambda: refine_handbook(
                version_record["handbook_snapshot"], feedback,
                website=website, fetch_docs=True, user_key=api_key,
                model=model, base_url=base_url))
            while True:
                try:
                    event = next(refine_work)
                except StopIteration as done:
                    refined, _ = done.value
                    break
                if event.get("kind") == "heartbeat":
                    yield {
                        "type": "heartbeat", "stage_key": refine_key,
                        "message": f"{next_base} refinement is still working…",
                    }
                elif event.get("line"):
                    detail = re.sub(
                        r"^\[\d{2}:\d{2}:\d{2}\]\s*", "",
                        event["line"]).strip()
                    if detail:
                        _record_stage(
                            study, refine_key,
                            _condition_label(
                                mechanism,
                                f"Handbook refinement · {next_base}"),
                            "Revise the handbook using controlled evidence.",
                            "running", detail=detail)
                        yield {
                            "type": "stage_progress",
                            "stage_key": refine_key,
                            "message": detail,
                        }
            next_source = refined["source"]
            before = version_record["handbook_snapshot"]
            if (next_source.get("handbook") == before.get("handbook")
                    and next_source.get("code_example")
                    == before.get("code_example")):
                yield _finish_condition(
                    user_id, study_id, study, condition, "stopped",
                    "no_handbook_change",
                    "The proposed refinement made no handbook change.")
                break

            next_source_id, _ = _materialize_version(
                user_id, study_id, next_version, next_source)
            user_data_sources.write_user_source(user_id, next_source)
            next_record = _upsert_version(
                study, next_version, next_source, next_source_id,
                reason=analysis.get("summary")
                       or "Execution-grounded refinement.")
            next_record.update({
                "mechanism": mechanism,
                "base_version": next_base,
                "iteration": next_iteration,
                "display_version": (
                    f"{_MECHANISM_LABELS.get(mechanism, mechanism)} "
                    f"{next_base}"),
            })
            refinement_record = {
                "id": str(uuid.uuid4()),
                "trial_id": trial["id"],
                "source_id": source_meta["id"],
                "task_id": task["id"],
                "mechanism": mechanism,
                "from_version": version,
                "to_version": next_version,
                "failure_category": trial["failure_category"],
                "evidence": analysis,
                "assistant_message": refined.get("assistant_message", ""),
                "status": "applied",
                "created_at": _now(),
            }
            study.setdefault("refinements", []).append(refinement_record)
            condition["current_version"] = next_version
            refine_stage = _record_stage(
                study, refine_key,
                _condition_label(
                    mechanism, f"Handbook refinement · {next_base}"),
                "Revise the handbook using controlled evidence.",
                "complete",
                refined.get("assistant_message") or f"{next_base} was created.",
                artifact={
                    "version": next_version,
                    "mechanism": mechanism,
                    "from_version": version,
                    "analysis": analysis,
                    "source": next_record["handbook_snapshot"],
                    "refinement_id": refinement_record["id"],
                })
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(refine_stage)

    conditions = [
        _condition_record(state, mechanism) for mechanism in mechanisms]
    passed = sum(1 for item in conditions if item.get("outcome") == "passed")
    invalid = sum(1 for item in conditions if item.get("outcome") == "invalid")
    if passed == len(conditions):
        outcome, reason = "passed", "all_mechanisms_validated"
        message = "Every selected access mechanism satisfied validation."
    elif passed:
        outcome, reason = "partial", "partial_mechanism_success"
        message = (
            f"{passed} of {len(conditions)} mechanisms passed; review the "
            "remaining condition evidence.")
    elif invalid:
        outcome, reason = "invalid", "platform_or_external_failures"
        message = (
            "No mechanism produced valid experimental evidence because one or "
            "more conditions ended in platform or external failures.")
    else:
        outcome, reason = "stopped", "no_mechanism_validated"
        message = "No selected access mechanism satisfied validation."
    yield _pipeline_stop(
        user_id, study_id, study, reason, outcome, message)


def run_experiment_pipeline(user_id: str, api_key: str, study_id: str,
                            user_keys: dict | None = None,
                            restart: bool = False,
                            confirm_mechanisms=None):
    """Yield the complete live Experiment 2 handbook/refinement lifecycle."""
    keys = {
        str(name): str(value) for name, value in (user_keys or {}).items()
        if name and value
    }
    try:
        placeholder_values = {
            "", "a", "s", "x", "test", "testing", "dummy", "sample",
            "none", "null", "n/a", "na", "changeme", "placeholder",
        }
        invalid_keys = [
            name for name, value in keys.items()
            if value.strip().casefold() in placeholder_values
            or len(value.strip()) < 4
        ]
        if invalid_keys:
            labels = ", ".join(sorted(invalid_keys))
            raise ValueError(
                "Enter complete credential values before running. Placeholder "
                f"or unusually short values were found for: {labels}.")
        yield from _run_experiment_pipeline_v2(
            user_id, api_key, study_id, keys, bool(restart),
            confirm_mechanisms=confirm_mechanisms)
    except InterruptedError:
        study = store.get_study(user_id, study_id)
        if study:
            state = _pipeline_state(study)
            state.update({
                "status": "interrupted",
                "finished_at": _now(),
                "stop_reason": "cancelled",
            })
            study["status"] = "ready"
            _persist_pipeline(user_id, study_id, study)
        yield {
            "type": "pipeline_cancelled",
            "message": "The pipeline stopped at a safe checkpoint and can be resumed.",
            "study": study,
        }
    except Exception as exc:
        study = store.get_study(user_id, study_id)
        if study:
            state = _pipeline_state(study)
            state.update({
                "status": "error",
                "finished_at": _now(),
                "error": str(exc),
            })
            study["status"] = "ready"
            stage = _record_stage(
                study, "pipeline_error", "Pipeline error",
                "An unexpected error stopped the experimental pipeline.",
                "error", str(exc))
            _persist_pipeline(user_id, study_id, study)
            yield _stage_event(stage)
        yield {
            "type": "error",
            "error": str(exc),
            "study": study,
        }
