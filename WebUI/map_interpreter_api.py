"""Map Interpreter: the React pages (/home, /map-interpreter) and their API.

Uploading only stores the map. Each step runs when the page asks for it: measuring (Stages 1-3 of
mapper/framework.py: legend anatomy, specification, measured reading against reference boundaries),
an interpretation whose claims cite measured rows, and answers to questions. Interpreting or asking
before measuring measures first. Maps the framework cannot measure yet (no reference boundaries,
unsupported legend) fall back to the model alone, labeled as unmeasured.

Each map is saved under uploads/map_interpreter/ as its image, a working copy, and <id>.json holding the
session: status, measured evidence (so framework answers need no re-measuring), activity log,
interpretations per method and the question thread. Sessions are reloaded on the first request after a
server start.
"""
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_from_directory

from utils.llm_models import llm_model

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
UPLOAD_DIR = BASE_DIR / "uploads" / "map_interpreter"
sys.path.insert(0, str(BASE_DIR.parent / "mapper"))

DEFAULT_MODEL = llm_model("map_interpreter.default")   # used when the picker sends none
MODELS = list(dict.fromkeys([DEFAULT_MODEL, "gpt-6-luna", "gpt-5.4", "gpt-5.2", "gpt-4.1", "gpt-4o"]))
ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}
MAX_SIDE = 3000  # longer side used for measurement; larger uploads are scaled down (memory and legend search time)

bp = Blueprint("map_interpreter", __name__)
log = logging.getLogger("map_interpreter")
_maps = {}
_lock = threading.RLock()
_loaded_from = None  # the upload folder whose saved sessions are in _maps

# what a session file keeps (the reader and paths are rebuilt on load)
SAVED_FIELDS = ("id", "name", "title", "model", "file", "measured", "reason", "log", "activity", "evidence",
                "evidence_store", "interpretations", "thread", "created")


def _default_title(file_name):
    """A readable session title from the map's file name: 'choropleth_map_obesity.png' -> 'Choropleth map obesity'."""
    stem = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
    return (stem[:1].upper() + stem[1:]) if stem else file_name


# ---------------------------------------------------------------- pages

@bp.route("/home")
@bp.route("/map-interpreter")
@bp.route("/map-interpreter/")
@bp.route("/map-interpreter/<map_id>")
def serve_ui(map_id=None):
    """The React app reads the path and shows the home tiles or the Map Interpreter."""
    if not (UI_DIR / "index.html").exists():
        return ("The React pages are not built yet. Run `npm install` and `npm run build` in WebUI/frontend.", 503)
    response = send_from_directory(UI_DIR, "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------- helpers

def _client():
    """OpenAI client for the framework, which needs the Responses API with image input.
    Uses the key the page sends (X-API-Key) when it is an OpenAI key, else the server's own."""
    from openai import OpenAI
    import utils.agm_helper as helper

    key = (g.get("api_key") or "").strip()
    if not key.startswith("sk-"):
        key = helper._usable_openai_fallback_key()
    if not key:
        raise PermissionError("The Map Interpreter needs an OpenAI API key (sk-...). Add one in the GIS "
                              "Co-Scientist settings, or set OPENAI_API_KEY in the server's .env.")
    return OpenAI(api_key=key)


def _error(message, status=400):
    return jsonify({"error": message}), status


def _save(entry):
    """Write the map's session file (atomically, so a crash never leaves half a file)."""
    data = {k: entry.get(k) for k in SAVED_FIELDS}
    data["work"] = Path(entry["work"]).name
    path = UPLOAD_DIR / f"{entry['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_entry(path):
    import framework as fw

    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["work"] = UPLOAD_DIR / entry.pop("work")
    entry.setdefault("activity", [])
    entry.setdefault("interpretations", {})
    entry.setdefault("thread", [])
    entry["reader"] = None
    if entry.get("measured") and entry.get("evidence_store"):
        # framework answers come from the saved table: put it back in a reader's cache, no re-measuring
        reader = fw.MapReader(None, entry.get("model"), sink=entry["activity"])
        reader.cache[Path(entry["work"])] = fw.evidence_from_json(entry["evidence_store"])
        entry["reader"] = reader
    elif entry.get("measured"):
        entry["measured"] = None  # saved before evidence was stored: measure again
    return entry


def _ensure_loaded():
    """Load the saved sessions once per upload folder (on the first request after the server starts)."""
    global _loaded_from
    with _lock:
        if _loaded_from == UPLOAD_DIR:
            return
        _loaded_from = UPLOAD_DIR
        for path in sorted(UPLOAD_DIR.glob("*.json")) if UPLOAD_DIR.exists() else []:
            try:
                entry = _load_entry(path)
                _maps.setdefault(entry["id"], entry)
            except Exception as e:
                log.warning("Map Interpreter could not load session %s: %s", path.name, e)


def _get(map_id):
    _ensure_loaded()
    with _lock:
        return _maps.get(map_id)


def _evidence_json(ev):
    """Evidence as plain JSON: spec, diagnostics, markers and the measured table."""
    table = json.loads(ev["table"][ev["table_columns"]].to_json(orient="records"))
    return {"spec": ev["spec"], "diagnostics": json.loads(json.dumps(ev["diagnostics"], default=str)),
            "log": ev.get("log", []),
            "markers": ev.get("markers", {}), "table_notes": ev["table_notes"],
            "columns": ev["table_columns"], "table": table}


def _summary(entry, full=True):
    out = {"id": entry["id"], "name": entry["name"], "title": entry.get("title") or _default_title(entry["name"]),
           "model": entry["model"], "measured": entry["measured"],
           "reason": entry.get("reason"), "image_url": f"/api/map-interpreter/maps/{entry['id']}/image",
           "created": entry.get("created")}
    if full:  # the session itself, for restoring the page
        out.update(evidence=entry.get("evidence"), log=entry.get("log", []),
                   interpretations=entry.get("interpretations", {}), thread=entry.get("thread", []))
    return out


# ---------------------------------------------------------------- API

@bp.route("/api/map-interpreter/models")
def list_models():
    return jsonify({"models": MODELS, "default": DEFAULT_MODEL})


@bp.route("/api/map-interpreter/maps", methods=["GET"])
def list_maps():
    _ensure_loaded()
    with _lock:
        entries = sorted(_maps.values(), key=lambda e: e.get("created") or 0)
    return jsonify({"maps": [_summary(e, full=False) for e in entries]})


@bp.route("/api/map-interpreter/maps/<map_id>", methods=["PATCH"])
def rename_map(map_id):
    """Rename a session. An empty title goes back to the one made from the file name."""
    entry = _get(map_id)
    if not entry:
        return _error("Map not found.", 404)
    title = str((request.get_json(silent=True) or {}).get("title") or "").strip()[:120]
    entry["title"] = title or None
    _save(entry)
    return jsonify(_summary(entry, full=False))


@bp.route("/api/map-interpreter/maps/<map_id>", methods=["DELETE"])
def delete_map(map_id):
    """Remove a map and its saved session."""
    entry = _get(map_id)
    if not entry:
        return _error("Map not found.", 404)
    with _lock:
        _maps.pop(map_id, None)
    for path in [UPLOAD_DIR / entry["file"], Path(entry["work"]), UPLOAD_DIR / f"{map_id}.json"]:
        Path(path).unlink(missing_ok=True)
    log.info("Map Interpreter removed %s (%s)", entry["name"], map_id)
    return jsonify({"deleted": map_id})


@bp.route("/api/map-interpreter/maps", methods=["POST"])
def add_map():
    """Save the uploaded map. Nothing is generated until the page asks to measure, interpret or answer."""
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return _error("Choose a map image to upload.")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED:
        return _error(f"Upload a PNG, JPG or WebP image (got {suffix or 'no extension'}).")

    from PIL import Image

    map_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original = UPLOAD_DIR / f"{map_id}{suffix}"
    upload.save(original)
    try:
        img = Image.open(original).convert("RGB")
    except Exception:
        original.unlink(missing_ok=True)
        return _error("That file could not be opened as an image.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    work = UPLOAD_DIR / f"{map_id}_work.png"
    img.save(work)

    # measured: None = not measured yet, True = measured, False = cannot be measured (model only)
    entry = {"id": map_id, "name": Path(upload.filename).name, "model": None, "file": original.name,
             "work": work, "measured": None, "reader": None, "activity": [], "interpretations": {}, "thread": [],
             "created": time.time()}
    _note(entry, f"Uploaded {entry['name']} ({img.width} x {img.height} px used for measuring); nothing generated yet")
    _ensure_loaded()
    with _lock:
        _maps[map_id] = entry
    _save(entry)
    return jsonify(_summary(entry))


def _note(entry, message):
    """Add a line to the map's live activity log (the framework adds its own steps to the same list)."""
    entry.setdefault("activity", []).append(f"[{time.strftime('%H:%M:%S')}] {message}")
    log.info("Map Interpreter %s", message)


@bp.route("/api/map-interpreter/maps/<map_id>/activity")
def map_activity(map_id):
    """Live log: the lines added since `since`, so the page can poll while a step is running."""
    entry = _get(map_id)
    if not entry:
        return _error("Map not found.", 404)
    lines = entry.get("activity", [])
    since = max(0, request.args.get("since", default=0, type=int))
    return jsonify({"lines": lines[since:], "total": len(lines)})


def _method_from_request():
    """framework = answer from the measured table; model_only = the baseline: the model looks at the image."""
    method = (request.get_json(silent=True) or {}).get("method") or "framework"
    return method if method in ("framework", "model_only") else "framework"


def _model_from_request():
    body = request.get_json(silent=True) or {}
    model = body.get("model") or DEFAULT_MODEL
    return model if model in MODELS else DEFAULT_MODEL


def _measure(entry, client, model):
    """Stages 1-3. Maps the framework cannot read are marked measured=False (model only) with the reason."""
    import framework as fw

    _note(entry, f"Measuring with {model}")
    reader = fw.MapReader(client, model, sink=entry.setdefault("activity", []))  # steps stream into the live log
    entry.update(model=model, reader=reader, evidence=None, evidence_store=None, reason=None, log=[])
    try:
        ev = reader.read(entry["work"])
        entry.update(measured=True, evidence=_evidence_json(ev), evidence_store=fw.evidence_to_json(ev))
        entry["log"] = entry["evidence"]["log"]
    except (KeyError, NotImplementedError, ValueError) as e:
        # No reference boundaries for this region, or a legend type the prototype cannot read yet
        reason = str(e).strip("'\"")
        entry.update(measured=False, reason=reason, log=reader.steps + [f"Stopped: {reason}"])
        _note(entry, f"Stopped: {reason}. Interpretations and answers will come from the model alone")
    except Exception as e:
        entry["log"] = reader.steps + [f"Failed: {type(e).__name__}: {e}"]
        _note(entry, f"Failed: {type(e).__name__}: {e}")
        raise
    finally:
        _save(entry)


@bp.route("/api/map-interpreter/maps/<map_id>/measure", methods=["POST"])
def measure_map(map_id):
    """Run (or re-run) Stages 1-3 on demand."""
    entry = _get(map_id)
    if not entry:
        return _error("Map not found (maps are kept until the server restarts).", 404)
    try:
        _measure(entry, _client(), _model_from_request())
    except PermissionError as e:
        return _error(str(e), 401)
    except Exception as e:
        entry.update(measured=None, reader=None)
        _save(entry)
        return jsonify({"error": f"Measuring the map failed: {type(e).__name__}: {e}", "log": entry.get("log", [])}), 500
    return jsonify(_summary(entry))


@bp.route("/api/map-interpreter/maps/<map_id>", methods=["GET"])
def get_map(map_id):
    entry = _get(map_id)
    return jsonify(_summary(entry)) if entry else _error("Map not found (maps are kept until the server restarts).", 404)


@bp.route("/api/map-interpreter/maps/<map_id>/image")
def map_image(map_id):
    entry = _get(map_id)
    if not entry:
        return _error("Map not found.", 404)
    return send_from_directory(UPLOAD_DIR, entry["file"])


def _svg_path(geom, tolerance=0.6):
    """SVG path data for a polygon in image pixels, simplified to sub-pixel tolerance."""
    geom = geom.simplify(tolerance, preserve_topology=False)
    parts = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            if len(coords) >= 3:
                parts.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in coords) + "Z")
    return "".join(parts)


@bp.route("/api/map-interpreter/maps/<map_id>/overlay")
def map_overlay(map_id):
    """The reference boundaries as fitted to the image during measuring, for drawing over the map."""
    entry = _get(map_id)
    if not entry:
        return _error("Map not found (maps are kept until the server restarts).", 404)
    if not entry["measured"]:
        return _error("Measure the map first: the boundaries are placed on the image while measuring.", 409)
    ev = entry["reader"].read(entry["work"])  # cached by the reader, so no model call
    width, height = ev["image_size"]
    units = [{"name": str(name), "d": _svg_path(geom)} for name, geom in zip(ev["outline"]["name"], ev["outline"].geometry)]
    return jsonify({"width": width, "height": height, "units": [u for u in units if u["d"]]})


@bp.route("/api/map-interpreter/maps/<map_id>/interpret", methods=["POST"])
def interpret_map(map_id):
    entry = _get(map_id)
    if not entry:
        return _error("Map not found (maps are kept until the server restarts).", 404)
    import framework as fw
    method = _method_from_request()
    try:
        client, model = _client(), _model_from_request()
        if method == "framework" and entry["measured"] is None:  # interpreting needs the measured table
            _measure(entry, client, model)
        if method == "framework" and entry["measured"]:
            entry["reader"].client, entry["reader"].model = client, model
            reply, used = entry["reader"].interpret(entry["work"]), "framework"
        else:
            why = "baseline selected" if method == "model_only" else "map could not be measured"
            _note(entry, f"Writing a model-only interpretation with {model} ({why}; not checked against measurements)")
            reply, used = fw.model_only_interpret(client, model, entry["work"]), "model_only"
            _note(entry, f"Model-only interpretation: {len(reply.get('findings', []))} findings")
    except PermissionError as e:
        return _error(str(e), 401)
    except Exception as e:
        _note(entry, f"Failed: interpretation: {type(e).__name__}: {e}")
        _save(entry)
        return _error(f"Interpretation failed: {type(e).__name__}: {e}", 500)
    result = {"grounded": used == "framework", "method": used, "requested_method": method, **reply}
    entry.setdefault("interpretations", {})[method] = result  # kept per selected method, like the page
    _save(entry)
    return jsonify({**result, "map": _summary(entry)})


@bp.route("/api/map-interpreter/maps/<map_id>/ask", methods=["POST"])
def ask_map(map_id):
    entry = _get(map_id)
    if not entry:
        return _error("Map not found (maps are kept until the server restarts).", 404)
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return _error("Type a question about the map.")
    import framework as fw
    method = _method_from_request()
    try:
        client, model = _client(), _model_from_request()
        if method == "framework" and entry["measured"] is None:  # answers come from the measured table
            _measure(entry, client, model)
        if method == "framework" and entry["measured"]:
            entry["reader"].client, entry["reader"].model = client, model
            reply = entry["reader"].answer(entry["work"], question, "Answer in a few words or one sentence.")
            used = "framework"
        else:
            why = "baseline selected" if method == "model_only" else "map could not be measured"
            _note(entry, f"Answering {question!r} with {model} from the image alone ({why})")
            reply, used = fw.model_only_answer(client, model, entry["work"], question), "model_only"
            _note(entry, f"Model-only answer: {reply.get('answer')!r}")
    except PermissionError as e:
        return _error(str(e), 401)
    except Exception as e:
        _note(entry, f"Failed: answering: {type(e).__name__}: {e}")
        _save(entry)
        return _error(f"Answering failed: {type(e).__name__}: {e}", 500)
    result = {"grounded": used == "framework", "method": used, "requested_method": method, "question": question,
              "asked": time.time(), **reply}
    entry.setdefault("thread", []).append(result)
    _save(entry)
    return jsonify({**result, "map": _summary(entry)})
