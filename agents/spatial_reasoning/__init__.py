"""
Spatial-knowledge integration conditions (B0-B3, P and ablations).

The web UI loads this code once, at startup. Edits made while it runs do not
reach new runs until it restarts: pilot P runs on T04 and T09 used a critic
written before severity existed, a day after it was fixed on disk. So the
source is fingerprinted at import, and every run records whether it ran the
code on disk (conditions.apply_to_ctx).
"""

import hashlib
import os

_PACKAGE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_PACKAGE))

# The package, its knowledge base, and the two files that call into it.
_EXTRA = [os.path.join(_ROOT, "spatial_kb", "spatial_kb.yaml"),
          os.path.join(_ROOT, "agents", "task_manager", "research_plan_generator.py"),
          os.path.join(_ROOT, "WebUI", "agm_workflow.py")]


def _source_files():
    own = sorted(os.path.join(_PACKAGE, f) for f in os.listdir(_PACKAGE) if f.endswith(".py"))
    return own + [p for p in _EXTRA if os.path.exists(p)]


def _hashes():
    out = {}
    for path in _source_files():
        with open(path, "rb") as f:
            out[os.path.relpath(path, _ROOT).replace(os.sep, "/")] = hashlib.sha256(f.read()).hexdigest()
    return out


def _combined(hashes):
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()


_LOADED = _hashes()


def code_status():
    """Fingerprint of the code this process loaded, and whether the files on
    disk have changed since (meaning the server needs a restart)."""
    now = _hashes()
    changed = sorted(k for k in set(_LOADED) | set(now) if _LOADED.get(k) != now.get(k))
    return {"code_sha256": _combined(_LOADED), "stale": bool(changed), "changed_since_load": changed}
