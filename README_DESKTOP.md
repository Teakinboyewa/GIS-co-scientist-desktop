# GIS Co-Scientist — Desktop

The WebUI app running in its own window instead of a browser tab. `desktop/launcher.py`
starts the Flask server (`WebUI/app.py`, unchanged) on `127.0.0.1:47041`, which only this
computer can reach, and shows it in a native window (pywebview + Microsoft Edge WebView2).

## Run
- **Everyday:** double-click `GIS Co-Scientist.lnk` (no console window).
- **Troubleshooting:** `run_desktop_debug.bat` (console with live server logs).

Closing the window asks for confirmation, then stops the server and any running analysis.

**Startup:** the window shows the app in about a second, while the analysis engine
(`WebUI/app.py`, about 5 s to import on a quiet machine) loads in the background.
scikit-learn/SciPy, matplotlib, networkx, pyvis, IPython and ipywidgets load only when first
used, through `utils/lazy_imports.py`, which roughly halved the import. Keep new heavy
imports inside the functions that need them, or use that module. Nothing is shown while it loads; a red message appears only if it
fails to start. Listing and opening saved conversations work at once, because they only
read the local database (`WebUI/conversation_routes.py`, also registered by `app.py`).
Everything else the page asks for early waits for the engine rather than failing.
The log records `Analysis engine ready in N s`. Code: `desktop/staged_app.py`.

## Sharing a conversation
The server only listens on this computer, so **Share** links don't work for anyone else.
Send a file instead:
- **Export:** in the sidebar, ⋯ on a conversation → **Export to file**. The `.zip` lands in
  Downloads and holds the messages, uploaded data and outputs (`Outputs/<task>/`).
- **Import:** **Import** in the sidebar toolbar → pick the `.zip`. You become the owner; the
  conversation keeps its id, and paths into the sender's app folder are rewritten to yours.
  An API key must be saved in Settings first. Importing a conversation that is already in
  the sidebar opens it instead; importing one you deleted brings it back as it was in the file.
- API keys are removed from the file (`WebUI/export_redaction.py`); your own copy is not changed.
  Code that had a key in it shows `REMOVED_API_KEY`, so the recipient needs their own key to re-run it.
- The file contains everything else in the conversation, including the absolute folder paths
  in saved workflow state, so check it before sending it outside your group.
- Code: `WebUI/conversation_bundle.py`; tests: `tests/test_conversation_bundle.py`,
  `tests/test_export_redaction.py`.

## Deleting a conversation
Deleting (as the owner) removes the conversation for good: its database rows, stored file
copies, `WebUI/uploads/<id>/` and `Outputs/<id>/`. An `Outputs` folder another conversation
still uses is kept. Conversations deleted before this change, which were only hidden, are
cleaned up at startup, and the database is compacted once 64 MB or more of it is free.
Export a conversation first if you may need it again.
Code: `WebUI/conversation_cleanup.py`; tests: `tests/test_conversation_cleanup.py`.

## Developing with live reload (VS Code)
Open **this folder** (`GIS_Co_Scientist_Desk`) in VS Code → Run and Debug (Ctrl+Shift+D):
- **Desktop app (live reload + debug)**: Python breakpoints work in WebUI, agents, utils.
  - Save a `.js` / `.css` / `.html` in `WebUI/` → the window refreshes.
  - Save a `.py` the app uses (or `utils/llm_models.toml`) → the server restarts (~10 s) and
    the window refreshes. A restart stops any analysis that is running at that moment.
- **Desktop app + JavaScript debugging**: the same, plus breakpoints in `WebUI/*.js` in VS Code.
- In the window, F12 or right-click → Inspect opens the browser dev tools.

Without VS Code: `python desktop\launcher.py --dev`. The map interpreter's React source
(`WebUI/frontend/src`) needs `npm install` then `npm run build -- --watch` in `WebUI/frontend`;
each rebuild into `WebUI/ui` refreshes the window. Dev mode is for this folder only: the
installed app never runs it.

## Where things live (all inside this folder)
| Path | What |
|---|---|
| `desktop/launcher.py` | the desktop host |
| `desktop/deps/` | pywebview and its dependencies (kept out of the shared venv) |
| `desktop/profile/` | WebView2 profile: localStorage, cookies, cache |
| `desktop/logs/desktop.log` | server log when started without a console |
| `desktop/assets/app.ico` | window and shortcut icon |
| `WebUI/`, `agents/`, `utils/`, ... | the app itself; conversations, uploads and outputs are written under `WebUI/` |

## Setup notes
- Python comes from `%USERPROFILE%\venvs\agm_venv` (the same venv the WebUI uses). Set
  `GIS_DESKTOP_VENV` to use another one with the debug launcher, or run
  `desktop/make_shortcut.ps1 -Venv <path>` to rebuild the shortcut.
- Reinstall pywebview: `<venv>\Scripts\python.exe -m pip install --target desktop\deps pywebview`
- Ports: the installed app uses 47041; this dev folder uses its own port derived from its
  path (currently 47188), so the two never share a server. `GIS_DESKTOP_PORT` overrides it; `GIS_DESKTOP_DEBUG=1` enables the browser dev tools.
- If a desktop server is already running on the port, a second launch reuses it instead of
  starting another server on the same databases.
- API keys still come from `.env` in this dev folder. The installer never includes it.

## Building the installer (lean build, about 860 MB installed)
1. Stage a self-contained copy: `<venv>\Scripts\python.exe desktop\build\stage_app.py`
   → `build\stage\runtime` (private Python 3.13 + packages) and `build\stage\app` (code).
   Left out: Claude Agent SDK (Claude-backed Generate & Test), faiss, pip, `tests` folders,
   caches, `.env`, conversations/uploads/outputs.
2. Compile → `dist\GIS-CoScientist-Setup-<version>.exe`.

**Use the build script, which runs both steps** (from this folder, in PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File desktop\build\build_installer.ps1 -Version 0.1.2
```

Add `-SkipStage` to reuse an existing `build\stage`. Don't run `ISCC.exe` directly on this
folder's path: this OneDrive folder is so deep that some staged files exceed Windows'
260-character path limit, and Inno Setup fails with "The system cannot find the path
specified". The script maps the folder to a temporary short drive letter (`subst`), compiles
through it, then removes it. To do that by hand:

```powershell
subst G: "$PWD"; & "G:\desktop\tools\innosetup-6.7.3\tools\ISCC.exe" /DAppVersion=0.1.2 /Q "G:\desktop\build\installer.iss"; subst G: /D
```

Use a new `-Version` for each build you hand out, so installing it over an older copy upgrades it.

Inno Setup is the portable NuGet package `Tools.InnoSetup` 6.7.3, extracted into
`desktop\tools` (signed by Pyrsys B.V.), so nothing is installed system-wide.

**What the installer does:** per-user install (no admin) to
`%LOCALAPPDATA%\Programs\GIS Co-Scientist`, Start-menu and optional desktop shortcuts,
precompiles Python on install, and checks for the WebView2 runtime. Upgrades replace the
runtime and code but keep conversations, uploads and outputs. Uninstall keeps those too.

Log is at %LOCALAPPDATA%\Programs\GIS Co-Scientist\app\desktop\logs\desktop.log

**Keys in the installed app:** OpenAI/Anthropic keys are entered in the app's Settings
screen. Optional extras such as `TAVILY_API_KEY` or `AGM_CITATION_MAILTO` go in
`%APPDATA%\GIS Co-Scientist\settings.env` (same `KEY=value` format as `.env`).

## Building the Mac app (.dmg) — run on a Mac
**Without a Mac:** GitHub builds it. In this repo on GitHub: **Actions → Build Mac app →
Run workflow**, enter the version. It builds on GitHub's Mac machines for Apple Silicon
and Intel (`.github/workflows/build-mac.yml`), checks that the bundled Python loads the
app, and attaches the two .dmg files to the run under **Artifacts**. Pushing a tag such as
`v0.1.2` runs it too.

**On a Mac yourself:**
Not built or tested on a Mac yet. The Python runtime and its compiled packages must
be the macOS builds, so this has to run on macOS (your own Mac, or a cloud Mac such as a
GitHub Actions macOS runner). From this folder on the Mac:

```bash
python3 desktop/build/build_mac.py --version 0.1.2 \
    --python-tarball <cpython-3.13.N+DATE-aarch64-apple-darwin-install_only.tar.gz>
```

- `--python-tarball`: a relocatable CPython **3.13** "install_only" build (path or URL)
  from https://github.com/astral-sh/python-build-standalone/releases:
  `aarch64-apple-darwin` for Apple Silicon or `x86_64-apple-darwin` for Intel. Build on
  the kind of Mac you target; the .dmg runs on that architecture only.
- Output: `dist/GIS-CoScientist-<version>-<arch>.dmg` (the app and an Applications shortcut).
- Same contents as the Windows build (`desktop/build/stage_app.py` is shared): no `.env`,
  data, conversations or personal data sources, and `.keys` values blanked.
- **Signing:** ad-hoc by default, which runs on the Mac that built it. On other Macs the
  first open needs right-click > Open (Gatekeeper). To avoid that, pass
  `--sign "Developer ID Application: Name (TEAMID)"` (Apple Developer account) and notarize
  with the `xcrun notarytool` command the script prints.
- **Where the Mac app keeps things:** the .app stays read-only. `desktop/mac_bootstrap.py`
  keeps a working copy in `~/Library/Application Support/GIS Co-Scientist/app` and runs
  from there; conversations, uploads, outputs, data sources and settings live in that
  copy, and a newer .app replaces only the code (tests: `tests/test_mac_bootstrap.py`).
  Log: `…/GIS Co-Scientist/app/desktop/logs/desktop.log`. Optional extras (`TAVILY_API_KEY`
  etc.) go in `~/Library/Application Support/GIS Co-Scientist/settings.env`.

## Shared data sources (GitHub catalog)
Shared Sources come from the public repo **Teakinboyewa/GIS-co-scientist-catalog**; no
server of our own is involved.
- **Sync:** at every launch the app downloads the catalog from `main`
  (`agents/data_agent/catalog_sync.py`), checks every file, then replaces
  `DataRetriever_Handbooks/Handbooks` and `/Keys`. A bad download is rejected whole;
  offline keeps the current catalog. `template.toml` is never touched. The log shows a
  `[catalog]` line. Set `GIS_CATALOG_SYNC=0` to turn it off.
- **Submit to GIBD:** the **Submit** button on a My Sources card (or the form's
  "Submit to the shared catalog" toggle) signs in to GitHub with a one-time code, then
  opens a pull request with `handbooks/<slug>.toml` (+ `keys/<slug>.keys`, names only)
  (`WebUI/catalog_github_api.py`). The GitHub token is saved in
  `%APPDATA%\GIS Co-Scientist\github.json` for that Windows user only.
- **Review:** every PR runs **Check catalog** (`.github/workflows/validate.yml`) using the
  rules from `main`, so a submission can't weaken its own check. It blocks invalid
  handbooks, filled-in keys and anything that looks like a secret, and it flags emails and
  local paths. Merge on GitHub, and everyone gets the source at their next launch.
- **Rules:** `agents/data_agent/catalog_rules.py` and the repo's `scripts/catalog_rules.py`
  must stay **identical**. Copy one over the other after any change.
- **Repo working copy:** `catalog_repo/` (its own git repo). To add or fix a source
  yourself, edit it there, run `python scripts/validate_catalog.py`, then commit and push.
- **Setup still needed:** the empty GitHub repo, and the client ID from a GitHub OAuth App
  with "Enable Device Flow" ticked, set as `GITHUB_CLIENT_ID` in `WebUI/catalog_github_api.py`.
  Until then, Submit shows "Shared catalog not set up yet" and sync keeps the bundled catalog.

Personal setup notes (not in git): see `README_LOCAL.md`, if present.
