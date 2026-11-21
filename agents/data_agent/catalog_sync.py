"""Keep the shared data-source catalog in step with GitHub.

The catalog lives in the public repo GIS-co-scientist-catalog (main branch). At
startup the desktop app downloads it as one zip (no GitHub API, no login, no
rate limit), validates every file with catalog_rules, and only then replaces the
local catalog (DataRetriever_Handbooks/Handbooks and /Keys). Merged pull requests
therefore reach every user at their next launch.

Safety:
  * only main is ever read, so open pull requests never reach users;
  * a download that fails validation is rejected as a whole (nothing changes);
  * [API_Key] values are always blanked on write;
  * offline or any error -> the existing local catalog is kept as is;
  * local files the app needs (template.toml) are never touched.

Set GIS_CATALOG_SYNC=0 to turn sync off; GIS_CATALOG_ZIP_URL overrides the source (tests).
"""
import importlib.util
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
import zipfile

# Self-contained on purpose: the desktop launcher runs this before the app is
# loaded, and importing the `agents` package would pull in every agent (and
# scikit-learn, pandas ...), bringing back the slow startup. So the rules are
# loaded straight from the sibling file, and the folders are the same ones
# user_data_sources.py uses (HANDBOOKS_ROOT / Handbooks / Keys).
_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys
rules = _sys.modules.get("agents.data_agent.catalog_rules")  # reuse if the app already loaded it
if rules is None:
    _spec = importlib.util.spec_from_file_location("gis_catalog_rules", os.path.join(_HERE, "catalog_rules.py"))
    rules = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rules)

HANDBOOKS_ROOT = os.path.join(_HERE, "DataRetriever_Handbooks")
GLOBAL_DIR = os.path.join(HANDBOOKS_ROOT, "Handbooks")
KEYS_DIR = os.path.join(HANDBOOKS_ROOT, "Keys")

CATALOG_REPO = os.environ.get("GIS_CATALOG_REPO", "Teakinboyewa/GIS-co-scientist-catalog")
CATALOG_BRANCH = "main"
STATE_FILE = os.path.join(HANDBOOKS_ROOT, ".catalog_sync.json")
KEEP_LOCAL = {"template.toml", "template.keys"}
MAX_ZIP_BYTES = 20 * 1024 * 1024

log = logging.getLogger("catalog_sync")


def zip_url():
    return os.environ.get("GIS_CATALOG_ZIP_URL") or \
        f"https://codeload.github.com/{CATALOG_REPO}/zip/refs/heads/{CATALOG_BRANCH}"


def _load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _download(etag):
    req = urllib.request.Request(zip_url(), headers={"User-Agent": "GIS-CoScientist-desktop"})
    if etag:
        req.add_header("If-None-Match", etag)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(MAX_ZIP_BYTES + 1)
            if len(data) > MAX_ZIP_BYTES:
                raise ValueError("catalog download is unexpectedly large")
            return data, resp.headers.get("ETag")
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return None, etag  # unchanged since last sync
        raise


def _read_catalog(data):
    """Extract and validate {('handbook'|'keys', slug): bytes}. Raises on any invalid file."""
    files, problems = {}, []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/", 1)  # strip "<repo>-main/"
            if len(parts) != 2:
                continue
            kind, slug = rules.classify_path(parts[1])
            if not kind:
                continue  # README, scripts, workflows ... not part of the catalog
            if info.file_size > max(rules.MAX_HANDBOOK_BYTES, rules.MAX_KEYS_BYTES):
                problems.append(f"{parts[1]}: too large")
                continue
            body = z.read(info)
            check = rules.check_handbook if kind == "handbook" else rules.check_keys
            errors, _ = check(slug, body)
            if errors:
                problems.append(f"{parts[1]}: {'; '.join(errors)}")
            files[(kind, slug)] = body
    if problems:
        raise ValueError("catalog rejected: " + " | ".join(problems[:5]))
    if not any(k == "handbook" for k, _ in files):
        raise ValueError("catalog rejected: it contains no handbooks")
    return files


def _apply(files):
    """Write the validated catalog; remove catalog files no longer in it."""
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    wanted = {GLOBAL_DIR: set(), KEYS_DIR: set()}
    changed = 0
    for (kind, slug), body in files.items():
        if kind == "keys":
            folder, name = KEYS_DIR, f"{slug}.keys"
            body = rules.blank_key_values(body.decode("utf-8")).encode("utf-8")
        else:
            folder, name = GLOBAL_DIR, f"{slug}.toml"
        wanted[folder].add(name)
        path = os.path.join(folder, name)
        try:
            with open(path, "rb") as f:
                if f.read() == body:
                    continue
        except OSError:
            pass
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
        changed += 1
    removed = 0
    for folder, names in wanted.items():
        suffix = ".toml" if folder == GLOBAL_DIR else ".keys"
        for name in os.listdir(folder):
            if name.endswith(suffix) and name not in names and name not in KEEP_LOCAL:
                os.remove(os.path.join(folder, name))  # removed from the catalog on GitHub
                removed += 1
    return changed, removed


def sync():
    """Bring the local catalog up to date. Never raises; returns a short status string."""
    if os.environ.get("GIS_CATALOG_SYNC", "1") == "0":
        return "catalog sync is off (GIS_CATALOG_SYNC=0)"
    state = _load_state()
    try:
        data, etag = _download(state.get("etag"))
        if data is None:
            state["checked"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _save_state(state)
            return "catalog unchanged"
        files = _read_catalog(data)
        changed, removed = _apply(files)
        sources = sum(1 for k, _ in files if k == "handbook")
        state.update(etag=etag, checked=time.strftime("%Y-%m-%dT%H:%M:%S"),
                     updated=time.strftime("%Y-%m-%dT%H:%M:%S"), sources=sources)
        _save_state(state)
        return f"catalog synced: {sources} sources ({changed} updated, {removed} removed)"
    except Exception as exc:  # offline, GitHub down, bad catalog: keep what we have
        log.warning("Catalog sync skipped: %s", exc)
        return f"catalog sync skipped, keeping the local catalog ({exc})"
