"""Show the app while the analysis engine is still loading.

Importing WebUI/app.py takes 10 s or more (the agents pull in scikit-learn,
pandas, IPython, langsmith, ...), and the window used to show only a loading
screen until it finished. StagedApp is a WSGI app the server can listen with
right away:

  * Until the engine is loaded, it serves the page and its static files from
    WebUI/ itself, so the full UI appears in about a second. Nothing is shown
    while the engine loads; index.html gets a small script that shows a
    message only if the engine fails to start.
  * Requests that need the engine (/api/... and any other route) wait for it,
    then go to the real Flask app, so nothing the page does early fails.
  * /api/health answers at once, so the launcher can open the window and a
    second launch can find (and reuse) this server.

Once the engine is loaded, every request goes straight to the real app.
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import traceback

from werkzeug.wrappers import Request, Response

# The page and what it loads. Everything else waits for the real app.
_STATIC_EXTS = {
    ".html", ".js", ".mjs", ".css", ".map", ".svg", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".ico", ".woff", ".woff2", ".ttf", ".json",
}
# Paths the real app answers with index.html.
_INDEX_PATHS = {"/", "/app", "/agm/"}
_STATUS_PATH = "/__desktop/engine-status"

# Nothing is shown while the engine loads. This script only puts up a message
# if the engine fails to start, so the page doesn't just stop responding.
_NOTICE = """
<script>
(function () {
  function showError(log) {
    var box = document.createElement('div');
    box.id = 'desktop-engine-notice';
    box.setAttribute('role', 'alert');
    box.style.cssText = 'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);'
      + 'z-index:2147483647;padding:9px 16px;border-radius:999px;background:#991b1b;'
      + 'color:#fff;font:13px "Segoe UI",system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.25)';
    box.textContent = "GIS Co-Scientist couldn't finish loading. Details are in " + log;
    document.body.appendChild(box);
  }
  function check() {
    var x = new XMLHttpRequest();
    x.open('GET', '%(status)s', true);
    x.onload = function () {
      var s = {};
      try { s = JSON.parse(x.responseText); } catch (e) {}
      if (s.ready) return;
      if (s.error) { showError(s.log); return; }
      setTimeout(check, 500);
    };
    x.onerror = function () { setTimeout(check, 1000); };
    x.send();
  }
  check();
})();
</script>
"""


def _conversation_early_app():
    """The database-only conversation routes, or None if they can't load
    (then those requests simply wait for the engine, as before)."""
    try:
        from WebUI.conversation_routes import create_early_app
        return create_early_app()
    except Exception:
        print("Early conversation routes unavailable; they will wait for the engine:\n"
              + traceback.format_exc(), flush=True)
        return None


class StagedApp:
    def __init__(self, webui_dir, load_app, log_path="", early_app="conversations"):
        """load_app() imports and returns the real WSGI app.

        early_app is a small Flask app whose routes are answered at once while
        the engine loads. The default builds the conversation routes
        (WebUI/conversation_routes.py: list and open saved conversations,
        which need only the database); pass None to have everything wait."""
        self.webui_dir = os.path.realpath(str(webui_dir))
        self._load_app = load_app
        if early_app == "conversations":
            early_app = _conversation_early_app()
        self.early_app = early_app
        self.log_path = str(log_path)
        self.app = None
        self.error = None
        self.ready = threading.Event()
        self.started = time.time()

    # -- loading ---------------------------------------------------------

    def load(self):
        """Import the engine. Run on its own thread."""
        try:
            self.app = self._load_app()
            print(f"Analysis engine ready in {time.time() - self.started:.1f} s", flush=True)
        except BaseException:
            self.error = traceback.format_exc()
            print("Analysis engine failed to start:\n" + self.error, flush=True)
        finally:
            self.ready.set()

    def start_loading(self):
        threading.Thread(target=self.load, name="engine-import", daemon=True).start()

    # -- WSGI ------------------------------------------------------------

    def __call__(self, environ, start_response):
        app = self.app
        if environ.get("PATH_INFO") == _STATUS_PATH:
            return self._json({"ready": app is not None, "error": bool(self.error),
                               "log": self.log_path})(environ, start_response)
        if app is not None:
            return app(environ, start_response)
        request = Request(environ)
        path = request.path or "/"
        if self.error is None:
            if path == "/api/health":
                return self._json({"status": "starting"})(environ, start_response)
            if self._early_matches(environ):
                return self.early_app(environ, start_response)
            if request.method in ("GET", "HEAD") and not path.startswith("/api/"):
                resp = self._static(path)
                if resp is not None:
                    return resp(environ, start_response)
        # Needs the engine: wait for it.
        self.ready.wait()
        if self.app is not None:
            return self.app(environ, start_response)
        return self._json({"error": "GIS Co-Scientist couldn't finish loading. "
                                    f"Details are in {self.log_path}"},
                          status=503)(environ, start_response)

    # -- helpers ---------------------------------------------------------

    def _early_matches(self, environ):
        """True when the early app has a route for this path and method."""
        if self.early_app is None:
            return False
        try:
            self.early_app.url_map.bind_to_environ(environ).match()
            return True
        except Exception:  # NotFound, MethodNotAllowed, RequestRedirect
            return False

    @staticmethod
    def _json(data, status=200):
        return Response(json.dumps(data), status=status, mimetype="application/json",
                        headers={"Cache-Control": "no-store"})

    def _static(self, path):
        rel = "index.html" if path in _INDEX_PATHS else path.lstrip("/")
        if os.path.splitext(rel)[1].lower() not in _STATIC_EXTS:
            return None
        full = os.path.realpath(os.path.join(self.webui_dir, rel))
        if not full.startswith(self.webui_dir + os.sep) or not os.path.isfile(full):
            return None
        with open(full, "rb") as f:
            body = f.read()
        if rel == "index.html":
            notice = _NOTICE.replace("%(status)s", _STATUS_PATH).encode("utf-8")
            i = body.rfind(b"</body>")
            body = body[:i] + notice + body[i:] if i != -1 else body + notice
        mime = mimetypes.guess_type(full)[0] or "application/octet-stream"
        # Same no-cache rule as app.py's static route, so edits show at once
        # and the notice-carrying index.html is never reused once ready.
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        return Response(body, mimetype=mime, headers=headers)
