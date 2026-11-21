"""Rules for files in the shared data-source catalog (GitHub: GIS-co-scientist-catalog).

One copy of this file lives in the desktop app (agents/data_agent/catalog_rules.py)
and an identical copy in the catalog repo (scripts/catalog_rules.py), so the app's
sync, the app's "Submit to GIBD", and the repo's pull-request check all apply the
same rules. Keep the two copies identical.

Catalog layout:
    handbooks/<slug>.toml   one data source (data_source_name, brief_description, handbook, ...)
    keys/<slug>.keys        optional: key NAMES the source needs ([API_Key], values empty)
                            and sign-up links ([Links])
"""
import configparser
import re
import tomllib

SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,99}$")
MAX_HANDBOOK_BYTES = 256 * 1024
MAX_KEYS_BYTES = 16 * 1024
REQUIRED_FIELDS = ("data_source_name", "brief_description", "handbook")
KEYS_SECTIONS = {"API_Key", "Links"}
RESERVED_SLUGS = {"template"}  # the app's own blank pattern for new handbooks

# Things that must never be published. Errors block; warnings are for the reviewer.
_SECRET_PATTERNS = [
    ("an OpenAI key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}")),
    ("a GIBD key", re.compile(r"\bgibd[-_][A-Za-z0-9_\-]{16,}")),
    ("a Tavily key", re.compile(r"\btvly-[A-Za-z0-9]{16,}")),
    ("a GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")),
    ("an AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a hard-coded key value", re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|secret|password|map_key)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]")),
]
_WARN_PATTERNS = [
    ("an email address", re.compile(r"\b[A-Za-z0-9._%+\-]+@(?!example\.(?:com|org))[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("a local file path", re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|OneDrive)|/Users/[^/\s]+/|/home/[^/\s]+/|OneDrive")),
    ("a private network address", re.compile(r"\b(?:10|127)\.\d+\.\d+\.\d+\b|\b192\.168\.\d+\.\d+\b|\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+\b")),
]


def slug_ok(slug):
    return bool(SLUG_RE.match(slug or "")) and slug.lower() not in RESERVED_SLUGS


def scan_text(text):
    """Return (errors, warnings) for secrets and personal details in text."""
    errors, warnings = [], []
    for label, pat in _SECRET_PATTERNS:
        if pat.search(text):
            errors.append(f"contains what looks like {label}")
    for label, pat in _WARN_PATTERNS:
        if pat.search(text):
            warnings.append(f"contains {label}; check it is safe to publish")
    return errors, warnings


def check_handbook(slug, data):
    """data: bytes of handbooks/<slug>.toml. Returns (errors, warnings)."""
    errors, warnings = [], []
    if not slug_ok(slug):
        errors.append(f"file name '{slug}.toml' must use letters, digits, '_' or '-' (and not be 'template')")
    if len(data) > MAX_HANDBOOK_BYTES:
        errors.append(f"is {len(data) // 1024} KB; the limit is {MAX_HANDBOOK_BYTES // 1024} KB")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors + ["is not UTF-8 text"], warnings
    try:
        book = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return errors + [f"is not valid TOML: {exc}"], warnings
    for field in REQUIRED_FIELDS:
        if not str(book.get(field, "")).strip():
            errors.append(f"is missing the required field '{field}'")
    e, w = scan_text(text)
    return errors + e, warnings + w


def check_keys(slug, data):
    """data: bytes of keys/<slug>.keys. Returns (errors, warnings)."""
    errors, warnings = [], []
    if not slug_ok(slug):
        errors.append(f"file name '{slug}.keys' must use letters, digits, '_' or '-'")
    if len(data) > MAX_KEYS_BYTES:
        errors.append("is too large for a .keys file")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors + ["is not UTF-8 text"], warnings
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str
    try:
        cfg.read_string(text)
    except configparser.Error as exc:
        return errors + [f"is not a valid .keys file: {exc}"], warnings
    for section in cfg.sections():
        if section not in KEYS_SECTIONS:
            errors.append(f"has an unknown section [{section}] (allowed: [API_Key], [Links])")
    if cfg.has_section("API_Key"):
        for name, value in cfg["API_Key"].items():
            if value.strip():
                errors.append(f"[API_Key] {name} has a value; the catalog lists key names only, "
                              "users enter their own keys in the app")
    if cfg.has_section("Links"):
        for name, value in cfg["Links"].items():
            if value.strip() and not re.match(r"^https?://\S+$", value.strip()):
                errors.append(f"[Links] {name} must be an http(s) URL")
    e, w = scan_text(text)
    return errors + e, warnings + w


def blank_key_values(text):
    """Return .keys text with every [API_Key] value emptied (names, comments, [Links] kept)."""
    out, section = [], None
    for line in text.splitlines(keepends=True):
        head = re.match(r"^\s*\[([^\]]+)\]", line)
        if head:
            section = head.group(1).strip().lower()
        m = re.match(r"^(\s*[^#;\[\s][^=:]*?\s*[=:])\s*(\S.*?)\s*$", line)
        if section == "api_key" and m and m.group(2):
            out.append(m.group(1) + "\n")
        else:
            out.append(line)
    return "".join(out)


def classify_path(path):
    """('handbook'|'keys', slug) for catalog files, else (None, None)."""
    m = re.match(r"^handbooks/([^/]+)\.toml$", path)
    if m:
        return "handbook", m.group(1)
    m = re.match(r"^keys/([^/]+)\.keys$", path)
    if m:
        return "keys", m.group(1)
    return None, None
