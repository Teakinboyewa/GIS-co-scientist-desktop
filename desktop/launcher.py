"""GIS Co-Scientist desktop launcher.

Runs the existing Flask app (WebUI/app.py) on a private local port and shows it
in a native window (pywebview + Microsoft Edge WebView2). The web app itself is
unchanged; this file only hosts it.

Run:  pythonw desktop/launcher.py      (no console)
      python  desktop/launcher.py      (console with live logs)
"""
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

DESKTOP_DIR = Path(__file__).resolve().parent
ROOT = DESKTOP_DIR.parent                      # GIS_Co_Scientist_Desk
WEBUI_DIR = ROOT / "WebUI"
DEPS_DIR = DESKTOP_DIR / "deps"                # pywebview lives here, not in the shared venv
PROFILE_DIR = DESKTOP_DIR / "profile"          # WebView2 profile: localStorage, cookies, cache
LOG_DIR = DESKTOP_DIR / "logs"
ICON_PATH = DESKTOP_DIR / "assets" / "app.ico"

APP_TITLE = "GIS Co-Scientist"
HOST = "127.0.0.1"
# A stable port keeps the page origin stable, so localStorage (saved keys,
# settings, open conversation) survives restarts. The installed app uses 47041;
# any other copy (e.g. the dev folder) gets its own port derived from its
# location, so two copies never share one server. Falls back if it's taken.
def _default_port():
    if (ROOT.parent / "runtime").is_dir():  # installed layout: <install>/runtime + <install>/app
        return 47041
    import zlib
    return 47100 + zlib.crc32(str(ROOT).lower().encode()) % 800


PREFERRED_PORT = int(os.environ.get("GIS_DESKTOP_PORT") or _default_port())

# Dev mode (GIS_DESKTOP_DEV=1 or --dev): server restarts on Python edits, the
# window refreshes on front-end edits, dev tools + VS Code JS attach on 9222.
DEV = os.environ.get("GIS_DESKTOP_DEV") == "1" or "--dev" in sys.argv
JS_DEBUG_PORT = int(os.environ.get("GIS_DESKTOP_JS_DEBUG_PORT", "9222"))

# The venv's packages must win over anything pip pulled into deps/, so append.
sys.path.append(str(DEPS_DIR))
for p in (str(WEBUI_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------- logging

def _setup_logging():
    """Under pythonw (Windows) or as a Mac .app opened from Finder there is no
    console; send output to a log file instead."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mac_app = sys.platform == "darwin" and os.environ.get("GIS_DESKTOP_MAC_APP") == "1"
    if (sys.stdout is None or sys.stderr is None or mac_app
            or "pythonw" in os.path.basename(sys.executable).lower()):
        log = open(LOG_DIR / "desktop.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    print(f"\n===== {APP_TITLE} desktop start {time.strftime('%Y-%m-%d %H:%M:%S')} =====")


# ---------------------------------------------------------------- windows quirks

def _hide_child_consoles():
    """The app runs generated code with subprocess. From a windowed process each
    of those would flash a console window, so default them to CREATE_NO_WINDOW."""
    if os.name != "nt":
        return
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = flag
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init


def _child_python():
    """Generated code must run with python.exe (real stdout/stderr), even when the
    launcher itself was started with pythonw.exe."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        candidate = exe.with_name("python.exe")
        if candidate.is_file():
            os.environ.setdefault("GIS_PYTHON_EXECUTABLE", str(candidate))


def _load_user_settings():
    """Optional per-user settings (e.g. TAVILY_API_KEY, AGM_CITATION_MAILTO) in
    %APPDATA%/GIS Co-Scientist/settings.env (Windows) or ~/Library/Application
    Support/GIS Co-Scientist/settings.env (macOS). OpenAI/Anthropic keys are
    entered in the app's own Settings screen, so the installer never ships a .env."""
    if sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("APPDATA") or str(Path.home())
    path = Path(base) / "GIS Co-Scientist" / "settings.env"
    if path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(path, override=False)
            print(f"Loaded settings from {path}")
        except Exception as exc:
            print(f"Could not read {path}: {exc}")


def _set_app_id():
    """Give the window its own taskbar identity (and icon) instead of Python's."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GIBD.GISCoScientist.Desktop")
        except Exception:
            pass


# ---------------------------------------------------------------- server

def _health(port, timeout=1.0):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _pick_port():
    if _port_free(PREFERRED_PORT):
        return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _import_engine():
    import app as webapp  # WebUI/app.py
    return webapp.app


class Server:
    def __init__(self):
        self.port = None
        self.httpd = None
        self.error = None
        self.reused = False

    def start(self):
        """Serve the app. Runs on a background thread.

        The server listens at once and shows the page while WebUI/app.py (the
        analysis engine, 10 s or more to import) loads on another thread; see
        staged_app.py."""
        # Another copy of the desktop app already serving? Share it rather than
        # opening a second server on the same databases.
        if _health(PREFERRED_PORT):
            self.port, self.reused = PREFERRED_PORT, True
            print(f"Reusing running server on port {self.port}")
            return
        try:
            os.chdir(WEBUI_DIR)  # match `python WebUI/app.py` relative-path behaviour
            from werkzeug.serving import make_server
            from staged_app import StagedApp
            staged = StagedApp(WEBUI_DIR, _import_engine, log_path=LOG_DIR / "desktop.log")
            self.port = _pick_port()
            self.httpd = make_server(HOST, self.port, staged, threaded=True)
            staged.start_loading()
            print(f"Serving on http://{HOST}:{self.port}")
            self.httpd.serve_forever()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)

    def wait_ready(self, timeout=300):
        end = time.time() + timeout
        while time.time() < end:
            if self.error:
                return False
            if self.port and _health(self.port):
                return True
            time.sleep(0.3)
        self.error = self.error or f"The server did not respond within {timeout} seconds."
        return False

    def stop(self):
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass


class DevServer(Server):
    """Dev mode: run desktop/dev_server.py (Werkzeug reloader) as a child process,
    so the server can restart on Python edits while the window stays open."""

    def __init__(self):
        super().__init__()
        self.proc = None

    def start(self):
        if self.port is None:
            self.port = _pick_port()  # keep the same port across restarts (stable origin)
        exe = Path(sys.executable)
        python = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
        cmd = [str(python), str(DESKTOP_DIR / "dev_server.py"), "--port", str(self.port)]
        print(f"[dev] starting server: {' '.join(cmd)}")
        self.error = None
        self.proc = subprocess.Popen(cmd, cwd=str(WEBUI_DIR))

    def restart(self):
        self.stop()
        if self.proc:
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
        for _ in range(50):  # wait for the old process to let go of the port
            if _port_free(self.port):
                break
            time.sleep(0.1)
        self.start()

    def wait_ready(self, timeout=300):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc and self.proc.poll() is not None:
                self.error = f"Dev server exited with code {self.proc.returncode}. See the terminal for the traceback."
                return False
            if self.port and _health(self.port):
                return True
            time.sleep(0.3)
        self.error = f"The dev server did not respond within {timeout} seconds."
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            # The reloader runs the real server as a grandchild; end the whole tree.
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.proc.terminate()


# Front-end files whose edits should refresh the window (served uncached by app.py).
_STATIC_SUFFIXES = {".js", ".css", ".html", ".svg", ".png"}


def _static_snapshot():
    snap = {}
    for base in (WEBUI_DIR, WEBUI_DIR / "icon", WEBUI_DIR / "ui"):
        if not base.is_dir():
            continue
        paths = base.rglob("*") if base.name == "ui" else base.iterdir()
        for p in paths:
            if p.suffix.lower() in _STATIC_SUFFIXES and p.is_file():
                try:
                    snap[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
    return snap


# Python sources (and config read at import) whose edits should restart the server.
_PY_DIRS = ("WebUI", "agents", "utils", "mapper", "spatial_kb")
_PY_SUFFIXES = {".py", ".toml", ".yaml"}
# Data/output folders the app writes into while running; never watched.
_PY_SKIP_DIRS = {"__pycache__", "node_modules", "frontend", "ui", "icon", "Outputs", "uploads",
                 "conversations", "conversation_DBs", "cache", "generated_code", "data"}


def _py_snapshot():
    snap = {}
    for d in _PY_DIRS:
        for root, dirs, files in os.walk(ROOT / d):
            dirs[:] = [x for x in dirs if x not in _PY_SKIP_DIRS and not x.startswith(".")]
            for f in files:
                if os.path.splitext(f)[1].lower() in _PY_SUFFIXES:
                    p = os.path.join(root, f)
                    try:
                        snap[p] = os.stat(p).st_mtime
                    except OSError:
                        pass
    return snap


def _changed(old, new):
    diff = sorted(set(old.items()) ^ set(new.items()))
    return Path(diff[0][0]).name if diff else None


def _refresh(window, server):
    """Reload the current app page; if the window is still on the loading/error
    screen (first start failed), open the app instead."""
    base = f"http://{HOST}:{server.port}"
    try:
        current = window.get_current_url() or ""
    except Exception:
        current = ""
    if current.startswith(base):
        window.evaluate_js("location.reload()")
    else:
        window.load_url(base + "/")


def _watch_and_reload(window, server):
    """Python edit -> restart the dev server, then refresh the window.
    Front-end edit -> refresh the window."""
    static, py = _static_snapshot(), _py_snapshot()
    while True:
        time.sleep(1.0)
        new_py = _py_snapshot()
        if new_py != py:
            name = _changed(py, new_py)
            time.sleep(0.3)  # let editors finish writing (some save in two steps)
            py = _py_snapshot()
            print(f"[dev] {name} changed, restarting server...")
            server.restart()
            if server.wait_ready():
                print("[dev] server back, refreshing window")
                _refresh(window, server)
            else:
                # Usually a syntax error in the file just saved. Keep watching:
                # fixing and saving it again triggers another restart.
                print(f"[dev] server failed to start: {server.error}")
                print("[dev] fix the error and save again; the window will come back")
            static = _static_snapshot()
            continue
        new_static = _static_snapshot()
        if new_static != static:
            print(f"[dev] {_changed(static, new_static)} changed, refreshing window")
            static = new_static
            _refresh(window, server)


# ---------------------------------------------------------------- window

def _loading_html():
    logo = WEBUI_DIR / "icon" / "GIS-Coscientist_logo.png"
    img = ""
    if logo.is_file():
        import base64
        img = f'<img src="data:image/png;base64,{base64.b64encode(logo.read_bytes()).decode()}" alt="">'
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      html,body{{height:100%;margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:#f7f8fa;color:#1f2937}}
      body{{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px}}
      img{{width:120px;height:auto}} h1{{font-size:20px;font-weight:600;margin:0}}
      p{{margin:0;color:#6b7280;font-size:14px}}
      .bar{{width:220px;height:4px;border-radius:2px;background:#e5e7eb;overflow:hidden}}
      .bar i{{display:block;width:40%;height:100%;background:#2563eb;animation:s 1.2s ease-in-out infinite}}
      @keyframes s{{0%{{transform:translateX(-100%)}}100%{{transform:translateX(250%)}}}}
      pre{{max-width:80vw;max-height:50vh;overflow:auto;background:#fff;border:1px solid #e5e7eb;padding:12px;font-size:12px;text-align:left}}
      @media (prefers-color-scheme:dark){{html,body{{background:#111827;color:#f3f4f6}}.bar{{background:#374151}}}}
    </style></head><body>{img}<h1>{APP_TITLE}</h1><p id="msg">Loading&hellip;</p>
    <div class="bar" id="bar"><i></i></div></body></html>"""


def _set_window_icon(window):
    """pywebview has no Windows icon option; set it on the native WinForms form."""
    if os.name != "nt" or not ICON_PATH.is_file():
        return
    try:
        import clr  # noqa: F401  (pythonnet, installed with pywebview)
        clr.AddReference("System.Drawing")
        from System import Action
        from System.Drawing import Icon
        form = None
        for _ in range(50):  # the native form appears shortly after start
            form = getattr(window, "native", None)
            if form is not None:
                break
            time.sleep(0.1)
        if form is None:
            print("Could not set window icon: native window not ready")
            return
        icon = Icon(str(ICON_PATH))
        form.Invoke(Action(lambda: setattr(form, "Icon", icon)))
    except Exception as exc:
        print(f"Could not set window icon: {exc}")


def _sync_catalog():
    """Fetch the shared data-source catalog from GitHub (merged pull requests reach
    users this way). Loaded by file path so it never imports the `agents` package
    (and with it the slow analysis engine); runs on its own thread, never blocks
    startup, and keeps the local catalog on any failure."""
    try:
        import importlib.util
        path = ROOT / "agents" / "data_agent" / "catalog_sync.py"
        spec = importlib.util.spec_from_file_location("gis_catalog_sync", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"[catalog] {mod.sync()}")
    except Exception as exc:
        print(f"[catalog] sync skipped: {exc}")


def _boot(window, server):
    _set_window_icon(window)
    threading.Thread(target=_sync_catalog, name="catalog-sync", daemon=True).start()
    threading.Thread(target=server.start, name="flask-server", daemon=True).start()
    ready = server.wait_ready()
    if DEV:
        # Watch even if the first start failed: fixing the file restarts the server.
        print(f"[dev] live reload on; JS debugger attach port {JS_DEBUG_PORT}")
        threading.Thread(target=_watch_and_reload, args=(window, server),
                         name="dev-watch", daemon=True).start()
    if ready:
        window.load_url(f"http://{HOST}:{server.port}/")
    else:
        import html, json
        detail = json.dumps(f"<pre>{html.escape(server.error or 'Unknown error')}</pre>"
                            f"<p>Log: {html.escape(str(LOG_DIR / 'desktop.log'))}</p>")
        window.evaluate_js(
            "document.getElementById('bar').remove();"
            "document.getElementById('msg').textContent='GIS Co-Scientist couldn\\'t finish loading.';"
            f"document.body.insertAdjacentHTML('beforeend', {detail});")


class DesktopApi:
    """Calls the page can make into the desktop app (window.pywebview.api.*).

    save_text: the browser's own Save As can't keep an extension Windows
    doesn't know (.toml): a name typed without it saves as a bare file. This
    shows a native Save dialog filtered to the right type, and adds the
    extension when the typed name has none. Attributes starting with "_" are
    not exposed to the page.
    """

    def __init__(self):
        self._window = None

    def save_text(self, suggested_name, text, extension="txt", description="Text file"):
        import webview
        ext = "." + str(extension or "txt").lstrip(".").lower()
        name = str(suggested_name or "download")
        if not name.lower().endswith(ext):
            name += ext
        try:
            result = self._window.create_file_dialog(
                webview.FileDialog.SAVE, save_filename=name,
                file_types=(f"{description} (*{ext})", "All files (*.*)"))
        except Exception as exc:
            return {"error": f"Could not open the Save dialog: {exc}"}
        if not result:
            return {"cancelled": True}
        path = result if isinstance(result, str) else result[0]
        if not os.path.splitext(path)[1]:
            path += ext        # typed a name without an extension
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text or "")
        except OSError as exc:
            return {"error": f"Could not save the file: {exc}"}
        return {"saved": path}


def main():
    _setup_logging()
    if not DEV:  # in dev the server's output belongs in the VS Code terminal
        _hide_child_consoles()
    _child_python()
    _load_user_settings()
    _set_app_id()

    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True              # reports, figures, exports
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    if DEV:
        webview.settings["REMOTE_DEBUGGING_PORT"] = JS_DEBUG_PORT  # VS Code "attach" to the page
        webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False          # F12 / right-click > Inspect instead

    # Open maximized; the restored size is 90% of the screen, never larger.
    width, height = 1440, 900
    try:
        screen = webview.screens[0]
        width, height = min(width, int(screen.width * 0.9)), min(height, int(screen.height * 0.9))
    except Exception:
        pass

    server = DevServer() if DEV else Server()
    api = DesktopApi()
    window = webview.create_window(
        APP_TITLE + (" (dev)" if DEV else ""), html=_loading_html(),
        width=width, height=height, min_size=(960, 600), maximized=True,
        text_select=True, zoomable=True, background_color="#f7f8fa",
        confirm_close=True, js_api=api,
    )
    api._window = window
    window.events.closed += lambda: server.stop()

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    # Windows: Edge WebView2. macOS: pywebview's default (WebKit / Cocoa).
    gui = "edgechromium" if os.name == "nt" else None
    webview.start(_boot, (window, server), gui=gui,
                  private_mode=False, storage_path=str(PROFILE_DIR),
                  debug=DEV or os.environ.get("GIS_DESKTOP_DEBUG") == "1")

    server.stop()
    # Analysis jobs run on worker threads; don't let one keep the process alive
    # after the window is gone.
    os._exit(0)


if __name__ == "__main__":
    main()
