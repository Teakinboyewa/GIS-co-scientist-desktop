"""macOS starter for GIS Co-Scientist.app (run by Contents/MacOS/GIS Co-Scientist).

The app writes its data next to its own code: WebUI/conversations.db,
WebUI/uploads, Outputs/, the data-source handbooks, desktop/profile and logs.
On Windows it is installed in a per-user folder, so that works. A Mac .app is
meant to be read-only (and is often in /Applications), so this starter keeps
a working copy of the app in

    ~/Library/Application Support/GIS Co-Scientist/app

and runs desktop/launcher.py from there with the bundled Python.

When the bundled version differs from the working copy's (first launch, or an
update dragged into /Applications), the code is copied over the working copy.
The user's data is never touched: the paths in USER_DATA are skipped, the same
data the Windows installer leaves alone on upgrade. Files the bundle doesn't
have are left as they are.

Bundle layout (built by desktop/build/build_mac.py):

    GIS Co-Scientist.app/Contents/
        MacOS/GIS Co-Scientist     shell script -> runtime python + this file
        Resources/runtime/          relocatable Python with the app's packages
        Resources/app/              the app (this file is app/desktop/)
        Resources/app/VERSION
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "GIS Co-Scientist"
INSTALLED_PORT = "47041"          # same fixed port as the Windows install

# Relative paths (posix style) that hold the user's data: never copied over.
USER_DATA = (
    ".env",
    "Outputs",
    "WebUI/uploads",
    "WebUI/conversations",
    "WebUI/conversation_DBs",
    "WebUI/Outputs",
    "WebUI/cache",
    "desktop/logs",
    "desktop/profile",
    "agents/data_agent/DataRetriever_Handbooks/Handbooks_user",
    "agents/data_agent/DataRetriever_Handbooks/Handbooks_pending",
)
USER_DATA_SUFFIXES = (".db", ".db-wal", ".db-shm")
USER_DATA_NAMES = (".catalog_sync.json",)


def _is_user_data(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    if any(rel == p or rel.startswith(p + "/") for p in USER_DATA):
        return True
    name = rel.rsplit("/", 1)[-1]
    return name.endswith(USER_DATA_SUFFIXES) or name in USER_DATA_NAMES


def _read_version(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def sync_app(bundle_app: Path, app_home: Path) -> bool:
    """Copy the bundled app over the working copy when the versions differ.
    Returns True when it copied. VERSION is written last, so an interrupted
    copy is simply redone on the next launch."""
    want = _read_version(bundle_app / "VERSION")
    if want and _read_version(app_home / "VERSION") == want:
        return False
    app_home.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(bundle_app):
        rel_dir = os.path.relpath(dirpath, bundle_app)
        rel_dir = "" if rel_dir == "." else rel_dir
        # Don't descend into user-data folders at all.
        dirnames[:] = [d for d in dirnames
                       if not _is_user_data(os.path.join(rel_dir, d) if rel_dir else d)]
        for name in filenames:
            rel = os.path.join(rel_dir, name) if rel_dir else name
            if name == "VERSION" and not rel_dir:
                continue
            if _is_user_data(rel):
                continue
            dst = app_home / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(dirpath, name), dst)   # keeps mtimes, so .pyc stay valid
    (app_home / "VERSION").write_text(want or "unknown", encoding="utf-8")
    return True


def main() -> None:
    here = Path(__file__).resolve()
    bundle_app = here.parents[1]                 # Resources/app
    resources = bundle_app.parent                # Contents/Resources
    python = resources / "runtime" / "bin" / "python3"
    app_home = Path.home() / "Library" / "Application Support" / APP_NAME / "app"

    sync_app(bundle_app, app_home)

    env = os.environ
    env.setdefault("GIS_DESKTOP_PORT", INSTALLED_PORT)
    env["GIS_DESKTOP_MAC_APP"] = "1"             # launcher logs to desktop/logs/desktop.log
    env["GIS_PYTHON_EXECUTABLE"] = str(python)   # generated code runs with the bundled Python
    os.chdir(app_home)
    launcher = app_home / "desktop" / "launcher.py"
    os.execv(str(python), [str(python), "-E", "-s", str(launcher)] + sys.argv[1:])


if __name__ == "__main__":
    main()
