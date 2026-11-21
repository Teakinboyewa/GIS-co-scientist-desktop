"""Development server for the desktop app's dev mode (started by launcher.py).

Serves WebUI/app.py in this process, with no reloader of its own. The launcher
watches the Python files and restarts this whole process when one changes.
Werkzeug's reloader is not used because it hands its listening socket to a
second process, which breaks under the VS Code debugger (the child exits with
code 1). With one plain process per run, the debugger attaches to every
restart without trouble.

Not used by the installed app.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "WebUI"
for p in (str(WEBUI), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    os.chdir(WEBUI)  # same working directory as `python WebUI/app.py`
    from werkzeug.serving import make_server
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from staged_app import StagedApp

    def load():
        import app as webapp
        return webapp.app

    # Listen at once and load the engine in the background, as the installed
    # app does (see staged_app.py). An import error shows in the window and in
    # this terminal; saving a fix restarts this process.
    staged = StagedApp(WEBUI, load, log_path="the VS Code terminal")
    httpd = make_server(args.host, args.port, staged, threaded=True)
    staged.start_loading()
    print(f"[dev] serving on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
