"""Build the macOS app: dist/GIS-CoScientist-<version>-<arch>.dmg

Run ON A MAC (the bundled Python and its compiled packages must be the macOS
builds, and the .app/.dmg tools are macOS-only), from the GIS_Co_Scientist_Desk
folder:

    python3 desktop/build/build_mac.py --version 0.1.1 \\
        --python-tarball cpython-3.13.<N>+<DATE>-aarch64-apple-darwin-install_only.tar.gz

--python-tarball: a relocatable CPython 3.13 "install_only" build from
https://github.com/astral-sh/python-build-standalone/releases (a path or a URL).
Pick the one for the Mac you build on: aarch64-apple-darwin (Apple Silicon) or
x86_64-apple-darwin (Intel). The .dmg runs on that architecture only.

Steps:
  1. runtime: unpack the tarball, pip install requirements.txt (minus what the
     lean Windows build also leaves out) and pywebview (Cocoa/WebKit backend).
  2. app: the same staging as Windows (desktop/build/stage_app.py): same
     folders, same exclusions (no .env, data, conversations, personal data
     sources), credentials in .keys files blanked.
  3. GIS Co-Scientist.app: Info.plist, icon, a MacOS/ starter that runs
     desktop/mac_bootstrap.py with the bundled Python.
  4. Code signing: ad-hoc by default (lets it run on this Mac). For other Macs
     without the Gatekeeper warning, pass --sign "Developer ID Application: ..."
     and notarize (the command is printed at the end).
  5. The .dmg (the app plus an Applications shortcut to drag it onto).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_app as win_stage  # noqa: E402  (shared app staging + key scrubbing)

DESK = win_stage.DESK
APP_NAME = "GIS Co-Scientist"
BUNDLE_ID = "org.gibd.giscoscientist"
PYWEBVIEW = "pywebview==6.2.1"       # same version as the Windows build (desktop/deps)
# Left out of the lean build, as on Windows (stage_app.SITE_EXCLUDE_PREFIXES).
SKIP_REQUIREMENTS = ("claude-agent-sdk", "faiss-cpu")


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch(src: str, work: Path) -> Path:
    if src.startswith(("http://", "https://")):
        dst = work / src.rsplit("/", 1)[-1]
        print(f"Downloading {src}", flush=True)
        urllib.request.urlretrieve(src, dst)
        return dst
    p = Path(src).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"Python tarball not found: {p}")
    return p


def build_runtime(tarball: Path, runtime: Path, work: Path) -> Path:
    """Unpack the relocatable Python and install the app's packages into it."""
    with tarfile.open(tarball) as tf:
        try:
            tf.extractall(work / "py", filter="data")
        except TypeError:            # Python older than 3.11.4 (e.g. Xcode's python3)
            tf.extractall(work / "py")
    unpacked = work / "py" / "python"            # install_only tarballs unpack to python/
    if not (unpacked / "bin" / "python3").exists():
        raise SystemExit("Unexpected tarball layout: expected python/bin/python3 "
                         "(use an *install_only* build from python-build-standalone)")
    shutil.move(str(unpacked), runtime)
    py = runtime / "bin" / "python3"
    out = subprocess.run([str(py), "-c", "import sys; print(sys.version_info[:2])"],
                         check=True, capture_output=True, text=True).stdout.strip()
    if out != "(3, 13)":
        raise SystemExit(f"Need CPython 3.13 (the Windows build's version), got {out}")

    reqs = work / "requirements-mac.txt"
    lines = (DESK / "requirements.txt").read_text(encoding="utf-8").splitlines()
    keep = [l for l in lines if not l.strip().lower().startswith(SKIP_REQUIREMENTS)]
    reqs.write_text("\n".join(keep + [PYWEBVIEW]) + "\n", encoding="utf-8")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "--no-cache-dir", "-r", reqs])
    # pip itself isn't needed at run time (the Windows build drops it too).
    for name in ("pip",):
        run([py, "-m", "pip", "uninstall", "-y", name])
    return py


def make_icon(resources: Path) -> str | None:
    """App icon from the logo PNG, with the macOS sips/iconutil tools."""
    logo = DESK / "WebUI" / "icon" / "GIS-Coscientist_logo.png"
    if not logo.is_file() or not shutil.which("iconutil"):
        print("No icon made (logo or iconutil missing); the app uses the default icon.")
        return None
    iconset = resources / "app.iconset"
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = size * scale
            name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
            run(["sips", "-z", px, px, logo, "--out", iconset / name],
                stdout=subprocess.DEVNULL)
    run(["iconutil", "-c", "icns", iconset, "-o", resources / "app.icns"])
    shutil.rmtree(iconset)
    return "app.icns"


STARTER = """#!/bin/bash
# GIS Co-Scientist: run the app's starter with the bundled Python.
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
exec "$RES/runtime/bin/python3" -E -s "$RES/app/desktop/mac_bootstrap.py" "$@"
"""


def build_bundle(app_dir: Path, version: str, work: Path, tarball: Path) -> Path:
    bundle = app_dir / f"{APP_NAME}.app"
    contents = bundle / "Contents"
    macos, resources = contents / "MacOS", contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()

    # 1. runtime
    py = build_runtime(tarball, resources / "runtime", work)

    # 2. app (shared with Windows)
    app = resources / "app"
    app.mkdir()
    win_stage.stage_app(app)
    assert not (app / ".env").exists(), ".env must never be staged"
    leaked = [p for p in app.rglob("*") if p.is_dir() and p.name in ("Handbooks_user", "Handbooks_pending")]
    assert not leaked, f"personal data sources must never be staged: {leaked}"
    files, blanked = win_stage.scrub_keys(app)
    print(f"Scrubbed {files} .keys files ({blanked} values blanked)")
    (app / "VERSION").write_text(version, encoding="utf-8")
    # Precompile, so the first launch doesn't (the copy keeps the .pyc valid).
    run([py, "-m", "compileall", "-q", "-j", "0", app, resources / "runtime" / "lib"])

    # 3. .app shell
    starter = macos / APP_NAME
    starter.write_text(STARTER, encoding="utf-8")
    starter.chmod(0o755)
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.education",
    }
    icon = make_icon(resources)
    if icon:
        info["CFBundleIconFile"] = icon
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    return bundle


def sign(bundle: Path, identity: str | None) -> None:
    if identity:
        run(["codesign", "--force", "--deep", "--options", "runtime", "--timestamp",
             "--sign", identity, bundle])
    else:
        # Ad-hoc: required for the bundled binaries to run on Apple Silicon.
        run(["codesign", "--force", "--deep", "--sign", "-", bundle])


def make_dmg(bundle: Path, out: Path, work: Path) -> None:
    src = work / "dmg"
    src.mkdir()
    shutil.copytree(bundle, src / bundle.name, symlinks=True)
    (src / "Applications").symlink_to("/Applications")
    if out.exists():
        out.unlink()
    run(["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", src,
         "-ov", "-format", "UDZO", out])


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("build_mac.py must run on macOS (see the docstring).")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--python-tarball", required=True,
                    help="path or URL of a python-build-standalone 3.13 *install_only* tarball")
    ap.add_argument("--sign", metavar="IDENTITY",
                    help='e.g. "Developer ID Application: Name (TEAMID)"; default: ad-hoc')
    ap.add_argument("--keep-stage", action="store_true", help="keep build/mac_stage afterwards")
    args = ap.parse_args()

    arch = {"arm64": "arm64", "x86_64": "x86_64"}.get(platform.machine(), platform.machine())
    stage = DESK / "build" / "mac_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    dist = DESK / "dist"
    dist.mkdir(exist_ok=True)
    dmg = dist / f"GIS-CoScientist-{args.version}-{arch}.dmg"

    with tempfile.TemporaryDirectory(prefix="gcs_macbuild_") as tmp:
        work = Path(tmp)
        tarball = fetch(args.python_tarball, work)
        bundle = build_bundle(stage, args.version, work, tarball)
        sign(bundle, args.sign)
        make_dmg(bundle, dmg, work)

    size = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file())  # noqa: E731
    print(json.dumps({"app_mb": round(size(bundle) / 1e6), "dmg_mb": round(dmg.stat().st_size / 1e6)}))
    print(f"Built {dmg}")
    if args.sign:
        print("\nTo notarize (needs an app-specific password or a keychain profile):\n"
              f'  xcrun notarytool submit "{dmg}" --keychain-profile <profile> --wait\n'
              f'  xcrun stapler staple "{dmg}"')
    else:
        print("\nNot signed with a Developer ID: on other Macs, open it with right-click > Open "
              "the first time (Gatekeeper).")
    if not args.keep_stage:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
