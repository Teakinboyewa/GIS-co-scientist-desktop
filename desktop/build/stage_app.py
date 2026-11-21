"""Assemble a self-contained GIS Co-Scientist for the installer.

Output (default: GIS_Co_Scientist_Desk/build/stage):
    stage/runtime/   a private Python 3.13: base interpreter + stdlib + the
                     venv's site-packages (lean: no Claude Agent SDK, faiss,
                     pip, tests or caches) + pywebview from desktop/deps
    stage/app/       the app code: WebUI, agents, utils, skills, spatial_kb,
                     mapper, desktop/launcher.py + assets. No .env, no data.

Run with the venv's python:  python desktop/build/stage_app.py [--stage DIR]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DESK = Path(__file__).resolve().parents[2]
VENV = Path(os.environ.get("GIS_DESKTOP_VENV", Path.home() / "venvs" / "agm_venv"))

# ---- what the lean bundle leaves out -------------------------------------
# Top-level site-packages entries, matched by prefix (case-insensitive).
SITE_EXCLUDE_PREFIXES = (
    "claude_agent_sdk",            # 208 MB; Claude-backed Generate & Test only (lean build)
    "faiss",                       # never imported by the app
    "pip", "_distutils_hack", "distutils-precedence.pth",
    "__pycache__",
)
# Directory names dropped anywhere inside site-packages.
SITE_EXCLUDE_DIRS = {"__pycache__", "tests"}
# Base install pieces the app doesn't use.
BASE_KEEP_TOP = {"DLLs", "Lib", "python.exe", "pythonw.exe", "python3.dll", "python313.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"}
STDLIB_EXCLUDE = {"site-packages", "test", "idlelib", "ensurepip", "turtledemo", "__pycache__"}

APP_DIRS = ["WebUI", "agents", "utils", "skills", "spatial_kb", "mapper"]
# Old copies kept for reference in the dev folder only (includes Keys.zip).
# Handbooks_user / Handbooks_pending (under agents/data_agent/DataRetriever_
# Handbooks) are this computer's own data sources and review queue. The
# desktop app treats everything in Handbooks_user/local as the user's own, so
# shipping them would hand the builder's private sources (and their code) to
# everyone who installs the app.
APP_DIR_EXCLUDE = {"Backed up handbooks", "Handbooks_user", "Handbooks_pending"}
WEBUI_DATA = {"conversations", "conversation_DBs", "uploads", "Outputs", "cache", "__pycache__"}
APP_FILE_EXCLUDE_SUFFIX = {".db", ".pyc", ".log"}


def base_python():
    cfg = (VENV / "pyvenv.cfg").read_text(encoding="utf-8")
    for line in cfg.splitlines():
        if line.strip().lower().startswith("home"):
            return Path(line.split("=", 1)[1].strip())
    raise SystemExit("pyvenv.cfg has no 'home' entry")


def copy_tree(src, dst, skip_dir=lambda p: False, skip_file=lambda p: False):
    n = size = 0
    for root, dirs, files in os.walk(src):
        rp = Path(root)
        dirs[:] = [d for d in dirs if not skip_dir(rp / d)]
        out = dst / rp.relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = rp / f
            if skip_file(sp):
                continue
            shutil.copy2(sp, out / f)
            n += 1
            size += sp.stat().st_size
    return n, size


def stage_runtime(rt):
    base = base_python()
    print(f"Base Python: {base}")
    for item in base.iterdir():
        if item.name not in BASE_KEEP_TOP:
            continue
        if item.is_file():
            shutil.copy2(item, rt / item.name)
        elif item.name == "Lib":
            copy_tree(item, rt / "Lib",
                      skip_dir=lambda p: p.parent == item and p.name in STDLIB_EXCLUDE or p.name == "__pycache__")
        else:
            copy_tree(item, rt / item.name, skip_dir=lambda p: p.name == "__pycache__")

    site_src = VENV / "Lib" / "site-packages"
    site_dst = rt / "Lib" / "site-packages"
    site_dst.mkdir(parents=True, exist_ok=True)
    excluded = []
    for item in sorted(site_src.iterdir()):
        if item.name.lower().startswith(SITE_EXCLUDE_PREFIXES):
            excluded.append(item.name)
            continue
        if item.is_dir():
            copy_tree(item, site_dst / item.name,
                      skip_dir=lambda p: p.name in SITE_EXCLUDE_DIRS,
                      skip_file=lambda p: p.suffix == ".pyc")
        else:
            shutil.copy2(item, site_dst / item.name)

    # pywebview and friends, unless the venv already has that package
    deps = DESK / "desktop" / "deps"
    for item in sorted(deps.iterdir()):
        if item.name in ("bin", "__pycache__") or (site_dst / item.name).exists():
            continue
        if item.is_dir():
            copy_tree(item, site_dst / item.name, skip_dir=lambda p: p.name == "__pycache__")
        else:
            shutil.copy2(item, site_dst / item.name)
    print(f"Left out of site-packages: {', '.join(excluded)}")


def stage_app(app):
    for d in APP_DIRS:
        src = DESK / d
        copy_tree(src, app / d,
                  skip_dir=lambda p: p.name in {"__pycache__", "node_modules"} | APP_DIR_EXCLUDE
                  or (p.parent == DESK / "WebUI" and p.name in WEBUI_DATA | {"frontend"}),
                  skip_file=lambda p: p.suffix in APP_FILE_EXCLUDE_SUFFIX
                  or p.name == ".catalog_sync.json")  # per-install sync state (catalog_sync.py)
    (app / "desktop" / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(DESK / "desktop" / "launcher.py", app / "desktop" / "launcher.py")
    shutil.copy2(DESK / "desktop" / "staged_app.py", app / "desktop" / "staged_app.py")
    shutil.copy2(DESK / "desktop" / "mac_bootstrap.py", app / "desktop" / "mac_bootstrap.py")  # used by the Mac .app only
    shutil.copy2(DESK / "desktop" / "assets" / "app.ico", app / "desktop" / "assets" / "app.ico")
    shutil.copy2(DESK / "requirements.txt", app / "requirements.txt")


def scrub_keys(app):
    """Blank every value in the [API_Key] section of staged .keys files, keeping
    key names, comments and the [Links] section (sign-up URLs), so data-source
    credentials typed into the dev folder can never ship. Returns (files, values blanked)."""
    import re
    files = blanked = 0
    for kf in app.rglob("*.keys"):
        lines = kf.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        out = []
        section = None
        for line in lines:
            head = re.match(r"^\s*\[([^\]]+)\]", line)
            if head:
                section = head.group(1).strip().lower()
            m = re.match(r"^(\s*[^#;\[\s][^=:]*?\s*[=:])\s*(\S.*?)\s*$", line)
            if section == "api_key" and m and m.group(2):
                out.append(m.group(1) + "\n")
                blanked += 1
            else:
                out.append(line)
        kf.write_text("".join(out), encoding="utf-8")
        files += 1
    return files, blanked


def dir_size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(DESK / "build" / "stage"))
    args = ap.parse_args()
    stage = Path(args.stage)
    if stage.exists():
        # OneDrive marks folders read-only, which blocks deleting them on Windows.
        def _clear_readonly(func, path, _exc):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(stage, onexc=_clear_readonly)
    rt, app = stage / "runtime", stage / "app"
    rt.mkdir(parents=True)
    app.mkdir(parents=True)

    stage_runtime(rt)
    stage_app(app)
    assert not (app / ".env").exists(), ".env must never be staged"
    leaked = [p for p in app.rglob("*") if p.is_dir() and p.name in ("Handbooks_user", "Handbooks_pending")]
    assert not leaked, f"personal data sources must never be staged: {leaked}"
    assert not list(app.rglob("*.zip")), "no archives in the app (they can hide credentials)"
    files, blanked = scrub_keys(app)
    print(f"Scrubbed {files} .keys files ({blanked} values blanked)")

    info = {"runtime_mb": round(dir_size(rt) / 1e6), "app_mb": round(dir_size(app) / 1e6)}
    info["total_mb"] = info["runtime_mb"] + info["app_mb"]
    (stage / "stage_info.json").write_text(json.dumps(info, indent=2))
    print(f"Staged at {stage}: {info}")


if __name__ == "__main__":
    main()
