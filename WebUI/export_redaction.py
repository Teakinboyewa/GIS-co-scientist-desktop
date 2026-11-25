"""Remove API keys from a conversation before it is exported to a file.

Keys reach a conversation in several ways: older runs saved them in
workflow_state.json, and generated download code (kept in messages and
outputs) may hold a key or a URL containing one. Two passes cover that:

  * every key value the app knows about (saved in Settings, in .keys files, in
    the environment, or in the conversation's own saved state) is replaced
    wherever it appears;
  * anything written like a key, such as api_key = "..." or ?key=..., is
    replaced too, which catches keys no longer saved.

The user's own copy on this computer is not changed. Only the standard library
is used here.
"""

import json
import os
import re
from urllib.parse import quote, quote_plus

PLACEHOLDER = "REMOVED_API_KEY"
STATE_FILE = "workflow_state.json"
STATE_KEY = "_data_source_keys"
TEXT_EXTS = {
    ".json", ".jsonl", ".geojson", ".topojson", ".md", ".txt", ".py", ".html",
    ".htm", ".log", ".yaml", ".yml", ".toml", ".xml", ".r", ".ipynb", ".sql",
    ".qmd", ".csv", ".tsv", ".js", ".svg", ".tex", ".ini", ".cfg", ".env",
    ".keys", ".vrt", ".kml", ".css", "",
}

_NOT_SECRET = {"demo_key", "your_api_key", PLACEHOLDER.lower()}
_ENV_NAME = re.compile(r"(?i)(?:key|token|secret|password)s?$")
_VALUE = r"(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{10,}"
_BARE = r"(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{12,}"  # a plain name is likelier here
_Q = r"\\{0,3}[\"']"          # a quote, possibly escaped inside JSON
_NAME = r"(?:[A-Za-z0-9]+_)*(?:api_?key|map_?key|app_?id|access_?token|auth_?token|client_?secret|api_?token)"
_PATTERNS = [
    # api_key = "..."   "X-Api-Key": "..."   get("FIRMS_MAP_KEY", "...")
    re.compile(rf"(?i)((?:{_NAME}|x-api-key)(?:{_Q})?\s*[:=,]\s*{_Q}){_VALUE}"),
    # key = "..." on its own (not join_key, primary_key ...)
    re.compile(rf"(?i)((?:(?<![A-Za-z0-9_])|(?<=\\n)|(?<=\\t))(?:key|token)(?:{_Q})?\s*[:=]\s*{_Q}){_BARE}"),
    # ?key=...  &api_key=...
    re.compile(rf"(?i)((?:[?&]|&amp;|%26|\\u0026)(?:{_NAME}|key|token)(?:=|%3D)){_VALUE}"),
    # NASA FIRMS puts its key in the address: /api/area/csv/<key>/...
    re.compile(r"(?i)(firms[^\s\"']{0,80}/|wms-info/[^\n]{0,60}?)[0-9a-f]{32}(?![0-9a-f])"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-.]{24,}"),
    re.compile(r"((?<![A-Za-z0-9]))(?:sk-|gibd-)[A-Za-z0-9_\-]{20,}"),
]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def state_keys(obj):
    """Every value stored under _data_source_keys, at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == STATE_KEY:
                yield from _strings(v)
            else:
                yield from state_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from state_keys(v)


def without_state_keys(obj):
    if isinstance(obj, dict):
        return {k: without_state_keys(v) for k, v in obj.items() if k != STATE_KEY}
    if isinstance(obj, list):
        return [without_state_keys(v) for v in obj]
    return obj


def _keys_files(root):
    folder = os.path.join(root, "agents", "data_agent", "DataRetriever_Handbooks")
    for dirpath, _, names in os.walk(folder):
        for name in names:
            if not name.endswith(".keys"):
                continue
            try:
                with open(os.path.join(dirpath, name), encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
            except OSError:
                continue
            section = ""
            for line in lines:
                line = line.strip()
                if line.startswith("["):
                    section = line.strip("[]").strip().lower()
                elif section == "api_key" and "=" in line and not line.startswith("#"):
                    yield line.split("=", 1)[1].strip().strip("\"'")


def collect_secrets(app_root, task_dirs, saved_keys=None):
    """Key values to remove: saved in the app, plus those in the saved state."""
    found = list(_strings(saved_keys or {}))
    found += [v for n, v in os.environ.items() if _ENV_NAME.search(n)]
    found += _keys_files(app_root)
    for folder in task_dirs:
        for dirpath, _, names in os.walk(folder):
            if STATE_FILE in names:
                try:
                    with open(os.path.join(dirpath, STATE_FILE), encoding="utf-8") as f:
                        found += state_keys(json.load(f))
                except (OSError, ValueError):
                    pass
    return {v.strip() for v in found}


class Redactor:
    def __init__(self, secrets=()):
        long, short = set(), set()
        for value in secrets:
            if len(value) < 6 or value.lower() in _NOT_SECRET or value.lower().startswith(("http://", "https://")):
                continue
            forms = {value, quote(value, safe=""), quote_plus(value), json.dumps(value)[1:-1]}
            (long if len(value) >= 8 else short).update(forms)
        parts = [re.escape(v) for v in sorted(long, key=len, reverse=True)]
        parts += [rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])"
                  for v in sorted(short, key=len, reverse=True)]
        self._known = re.compile("|".join(parts)) if parts else None

    def known(self, text):
        return self._known.sub(PLACEHOLDER, text) if self._known else text

    def text(self, text):
        if not text:
            return text
        text = self.known(text)
        for pattern in _PATTERNS:
            text = pattern.sub(lambda m: m.group(1) + PLACEHOLDER, text)
        return text

    def value(self, obj):
        """Redact every string in a nested structure of rows."""
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            return {k: self.value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.value(v) for v in obj]
        return obj

    def file(self, name, content, known_only=False):
        """Redact a file's bytes. Files that are not text come back unchanged."""
        if os.path.splitext(name)[1].lower() not in TEXT_EXTS:
            return content
        # latin-1 maps every byte to one character and back, so a file in any
        # ASCII-compatible encoding is untouched apart from the keys.
        text = content.decode("latin-1")
        if os.path.basename(name) == STATE_FILE:
            try:
                state = json.loads(content.decode("utf-8"))
            except ValueError:
                pass
            else:
                text = json.dumps(without_state_keys(state), indent=2)  # ASCII output
        return (self.known(text) if known_only else self.text(text)).encode("latin-1")
