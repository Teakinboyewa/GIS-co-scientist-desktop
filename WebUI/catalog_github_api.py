""""Submit to GIBD": propose one of the user's data sources to the shared catalog
as a GitHub pull request, from the desktop app.

Flow (no server of our own; everything is GitHub):
  1. Sign in once with GitHub's device flow: the app shows a short code, the user
     enters it at github.com/login/device and approves. The token (scope
     public_repo) is saved for this Windows user only.
  2. Submit: the app forks the catalog repo under the user's account (the owner
     works in the repo directly), creates a branch, writes
     handbooks/<slug>.toml (+ keys/<slug>.keys with names only) and opens a PR
     against main. The owner reviews and merges on GitHub; every app picks the
     merged source up at its next launch (agents/data_agent/catalog_sync.py).

Endpoints (all JSON):
  GET  /api/catalog/status          configured? signed in as? past submissions
  POST /api/catalog/github/start    begin sign-in -> {user_code, verification_uri}
  POST /api/catalog/github/poll     finish sign-in when the user has approved
  POST /api/catalog/github/signout  forget the saved token
  POST /api/catalog/submit          {slug} -> {pr_url}

Configuration: GITHUB_CLIENT_ID below (from the catalog owner's GitHub OAuth App,
"Enable Device Flow" ticked). The client ID is public by design; there is no secret.
"""
import base64
import json
import os
import tempfile
import threading
import time

import requests
from flask import Blueprint, g, jsonify, request

from agents.data_agent import catalog_rules as rules
from agents.data_agent import user_data_sources as uds
from agents.data_agent.catalog_sync import CATALOG_REPO

# The catalog owner's GitHub OAuth App client ID (public by design, no secret).
# Built in so the installed app works without any setup; the environment
# variable overrides it (e.g. to test with another OAuth App).
_BUILT_IN_GITHUB_CLIENT_ID = "Ov23liIKi85qQ8Q2wthH"   # OAuth App "GIS Co-Scientist" (Device Flow on)
GITHUB_CLIENT_ID = os.environ.get("GIS_CATALOG_GITHUB_CLIENT_ID", "").strip() or _BUILT_IN_GITHUB_CLIENT_ID
SCOPE = "public_repo"
API = "https://api.github.com"
_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "GIS-CoScientist-desktop"}

_DATA_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "GIS Co-Scientist")
_TOKEN_FILE = os.path.join(_DATA_DIR, "github.json")
_SUBMISSIONS_FILE = os.path.join(_DATA_DIR, "catalog_submissions.json")

bp = Blueprint("catalog_github", __name__)
_lock = threading.Lock()
_device = {}  # pending sign-in: device_code, interval, expires


class GitHubError(Exception):
    pass


# ---------------------------------------------------------------- local state

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _token():
    return _read_json(_TOKEN_FILE).get("token")


# ---------------------------------------------------------------- GitHub calls

def _gh(method, path, token, **kw):
    resp = requests.request(method, API + path, headers={**_HEADERS, "Authorization": f"Bearer {token}"},
                            timeout=30, **kw)
    if resp.status_code == 401:
        try:
            os.remove(_TOKEN_FILE)  # expired or revoked: ask to sign in again
        except OSError:
            pass
        raise GitHubError("Your GitHub sign-in has expired. Sign in again and resubmit.")
    return resp


def _gh_ok(resp, what):
    if resp.status_code >= 300:
        try:
            detail = resp.json().get("message", "")
        except ValueError:
            detail = resp.text[:200]
        raise GitHubError(f"GitHub refused to {what} (HTTP {resp.status_code}): {detail}")
    return resp.json() if resp.content else {}


def _owner_repo():
    owner, repo = CATALOG_REPO.split("/", 1)
    return owner, repo


def _head_repo(token, login):
    """The repo the branch goes in: the catalog itself for its owner, else the user's fork."""
    owner, repo = _owner_repo()
    if login.lower() == owner.lower():
        return CATALOG_REPO, False
    fork = _gh_ok(_gh("POST", f"/repos/{owner}/{repo}/forks", token, json={"default_branch_only": True}),
                  "create your copy (fork) of the catalog")
    full = fork.get("full_name") or f"{login}/{repo}"
    for _ in range(30):  # a new fork takes a few seconds to appear
        if _gh("GET", f"/repos/{full}/git/ref/heads/main", token).status_code == 200:
            break
        time.sleep(2)
    else:
        raise GitHubError("Your copy (fork) of the catalog is still being created. Try again in a minute.")
    # Bring the fork up to date so the PR only contains this source.
    r = _gh("POST", f"/repos/{full}/merge-upstream", token, json={"branch": "main"})
    if r.status_code not in (200, 409):
        _gh_ok(r, "update your copy (fork) of the catalog")
    return full, True


def _put_file(token, repo, branch, path, text, message):
    existing = _gh("GET", f"/repos/{repo}/contents/{path}", token, params={"ref": branch})
    body = {"message": message, "branch": branch,
            "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
    if existing.status_code == 200:
        body["sha"] = existing.json().get("sha")
    _gh_ok(_gh("PUT", f"/repos/{repo}/contents/{path}", token, json=body), f"write {path}")


def _source_files(slug):
    """Build the catalog files for one of the caller's private sources:
    the same handbook + keys the old admin approval produced."""
    if not g.api_key:
        raise GitHubError("Enter your API key in Settings first, so the app can find your sources.")
    uid = uds.uid_for_key(g.api_key)
    src = uds.get_user_source(uid, slug)
    if not src:
        raise GitHubError("That data source was not found among your sources.")
    with tempfile.TemporaryDirectory() as tmp:
        hb_path = os.path.join(tmp, f"{slug}.toml")
        uds._write_source(hb_path, {k: src.get(k, "") for k in uds.STORED_FIELDS})
        with open(hb_path, encoding="utf-8") as f:
            handbook = f.read()
        keys = None
        names = uds._credential_names(src)
        if names:
            kpath = os.path.join(tmp, f"{slug}.keys")
            uds._write_keys_file(kpath, names, src.get("key_signup_url", "") or "")
            with open(kpath, encoding="utf-8") as f:
                keys = rules.blank_key_values(f.read())
    return src, handbook, keys


def _pr_body(src, slug, keys, warnings, is_update):
    lines = [
        f"**{'Updated' if is_update else 'New'} data source:** {src.get('data_source_name') or slug}",
        "",
        (src.get("brief_description") or "").strip() or "_No description given._",
        "",
        f"- Website: {src.get('website') or 'not given'}",
        f"- Needs an API key: {'yes (' + src.get('key_name', '') + ')' if keys else 'no'}",
        f"- Files: `handbooks/{slug}.toml`" + (f", `keys/{slug}.keys`" if keys else ""),
        "",
        "Submitted from the GIS Co-Scientist desktop app.",
        "",
        "### Reviewer checklist",
        "- [ ] The handbook describes a real, reputable source and its official endpoints",
        "- [ ] Instructions only fetch data; nothing deletes, uploads, or runs unexpected code",
        "- [ ] No personal details, keys or tokens (the automatic check also scans for these)",
    ]
    if warnings:
        lines += ["", "### Flagged for review"] + [f"- {w}" for w in warnings]
    return "\n".join(lines)


# ---------------------------------------------------------------- endpoints

@bp.route("/api/catalog/status", methods=["GET"])
def status():
    info = _read_json(_TOKEN_FILE)
    return jsonify({
        "success": True,
        "configured": bool(GITHUB_CLIENT_ID),
        "repo": CATALOG_REPO,
        "signed_in": bool(info.get("token")),
        "login": info.get("login"),
        "submissions": _read_json(_SUBMISSIONS_FILE),
    })


@bp.route("/api/catalog/github/start", methods=["POST"])
def start_sign_in():
    if not GITHUB_CLIENT_ID:
        return jsonify({"success": False, "error": "GitHub submissions are not set up in this build yet."}), 400
    try:
        resp = requests.post("https://github.com/login/device/code", timeout=20,
                             headers={"Accept": "application/json", "User-Agent": _HEADERS["User-Agent"]},
                             data={"client_id": GITHUB_CLIENT_ID, "scope": SCOPE})
        data = resp.json()
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not reach GitHub: {exc}"}), 502
    if "device_code" not in data:
        return jsonify({"success": False, "error": data.get("error_description") or "GitHub did not start sign-in."}), 502
    with _lock:
        _device.update(device_code=data["device_code"], interval=int(data.get("interval", 5)),
                       expires=time.time() + int(data.get("expires_in", 900)))
    return jsonify({"success": True, "user_code": data["user_code"],
                    "verification_uri": data.get("verification_uri", "https://github.com/login/device"),
                    "interval": int(data.get("interval", 5))})


@bp.route("/api/catalog/github/poll", methods=["POST"])
def poll_sign_in():
    with _lock:
        pending = dict(_device)
    if not pending or time.time() > pending.get("expires", 0):
        return jsonify({"success": False, "state": "expired", "error": "Sign-in timed out. Start again."})
    try:
        data = requests.post("https://github.com/login/oauth/access_token", timeout=20,
                             headers={"Accept": "application/json", "User-Agent": _HEADERS["User-Agent"]},
                             data={"client_id": GITHUB_CLIENT_ID, "device_code": pending["device_code"],
                                   "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}).json()
    except Exception as exc:
        return jsonify({"success": False, "state": "error", "error": f"Could not reach GitHub: {exc}"})
    err = data.get("error")
    if err in ("authorization_pending", "slow_down"):
        if err == "slow_down":
            with _lock:
                _device["interval"] = _device.get("interval", 5) + 5
        return jsonify({"success": True, "state": "pending", "interval": _device.get("interval", 5)})
    if err:
        with _lock:
            _device.clear()
        msg = {"access_denied": "Sign-in was cancelled on GitHub.",
               "expired_token": "Sign-in timed out. Start again."}.get(err, data.get("error_description") or err)
        return jsonify({"success": False, "state": "failed", "error": msg})
    token = data.get("access_token")
    try:
        login = _gh_ok(_gh("GET", "/user", token), "read your GitHub account").get("login")
    except GitHubError as exc:
        return jsonify({"success": False, "state": "failed", "error": str(exc)})
    _write_json(_TOKEN_FILE, {"token": token, "login": login, "scope": data.get("scope", SCOPE),
                              "saved": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with _lock:
        _device.clear()
    return jsonify({"success": True, "state": "done", "login": login})


@bp.route("/api/catalog/github/signout", methods=["POST"])
def sign_out():
    try:
        os.remove(_TOKEN_FILE)
    except OSError:
        pass
    return jsonify({"success": True})


@bp.route("/api/catalog/submit", methods=["POST"])
def submit():
    slug = ((request.get_json(silent=True) or {}).get("slug") or "").strip()
    token = _token()
    if not token:
        return jsonify({"success": False, "need_sign_in": True, "error": "Sign in to GitHub first."}), 401
    try:
        if not rules.slug_ok(slug):
            raise GitHubError(f"'{slug}' can't be used as a catalog file name.")
        src, handbook, keys = _source_files(slug)
        errors, warnings = rules.check_handbook(slug, handbook.encode("utf-8"))
        if keys:
            e, w = rules.check_keys(slug, keys.encode("utf-8"))
            errors, warnings = errors + e, warnings + w
        if errors:
            raise GitHubError("This source can't be submitted yet: " + "; ".join(errors))

        login = _gh_ok(_gh("GET", "/user", token), "read your GitHub account")["login"]
        owner, repo = _owner_repo()
        head, is_fork = _head_repo(token, login)
        base_sha = _gh_ok(_gh("GET", f"/repos/{head}/git/ref/heads/main", token), "read the catalog")["object"]["sha"]
        is_update = _gh("GET", f"/repos/{owner}/{repo}/contents/handbooks/{slug}.toml", token).status_code == 200
        branch = f"{'update' if is_update else 'add'}-{slug}-{time.strftime('%Y%m%d%H%M%S')}"
        _gh_ok(_gh("POST", f"/repos/{head}/git/refs", token, json={"ref": f"refs/heads/{branch}", "sha": base_sha}),
               "create a branch for your submission")

        name = src.get("data_source_name") or slug
        verb = "Update" if is_update else "Add"
        _put_file(token, head, branch, f"handbooks/{slug}.toml", handbook, f"{verb} data source: {name}")
        if keys:
            _put_file(token, head, branch, f"keys/{slug}.keys", keys, f"{verb} key names for {name}")

        pr = _gh_ok(_gh("POST", f"/repos/{owner}/{repo}/pulls", token, json={
            "title": f"{verb} data source: {name}",
            "head": f"{login}:{branch}" if is_fork else branch,
            "base": "main",
            "body": _pr_body(src, slug, keys, warnings, is_update),
            "maintainer_can_modify": True,
        }), "open the pull request")

        subs = _read_json(_SUBMISSIONS_FILE)
        subs[slug] = {"pr_url": pr.get("html_url"), "number": pr.get("number"),
                      "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write_json(_SUBMISSIONS_FILE, subs)
        return jsonify({"success": True, "pr_url": pr.get("html_url"), "number": pr.get("number"),
                        "warnings": warnings})
    except GitHubError as exc:
        return jsonify({"success": False, "error": str(exc), "need_sign_in": not _token()}), 400
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": f"Could not reach GitHub: {exc}"}), 502
