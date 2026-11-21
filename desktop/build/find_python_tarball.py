"""Print the download URL of a relocatable CPython 3.13 for macOS, from the
latest python-build-standalone release (the --python-tarball for build_mac.py).

    python3 desktop/build/find_python_tarball.py aarch64   # Apple Silicon
    python3 desktop/build/find_python_tarball.py x86_64    # Intel

Uses GITHUB_TOKEN / GH_TOKEN when set (GitHub Actions), to avoid the API's
anonymous rate limit. Picks the plain "install_only" build (not stripped,
not free-threaded, not debug).
"""
import json
import os
import re
import sys
import urllib.request

API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=5"
MINOR = "3.13"


def main():
    arch = (sys.argv[1] if len(sys.argv) > 1 else "aarch64").strip()
    if arch not in ("aarch64", "x86_64"):
        raise SystemExit("arch must be aarch64 (Apple Silicon) or x86_64 (Intel)")
    want = re.compile(rf"^cpython-{re.escape(MINOR)}\.\d+\+\d+-{arch}-apple-darwin-install_only\.tar\.gz$")
    req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "gis-coscientist-mac-build"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.load(r)
    for rel in releases:                      # newest first; a release may still be uploading
        for asset in rel.get("assets", []):
            if want.match(asset["name"]):
                print(asset["browser_download_url"])
                return
    raise SystemExit(f"No CPython {MINOR} install_only build for {arch}-apple-darwin in the latest releases")


if __name__ == "__main__":
    main()
