"""The desktop server shows the page before the analysis engine has loaded."""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "desktop"))

from flask import Flask, jsonify  # noqa: E402
from werkzeug.test import Client  # noqa: E402

from staged_app import StagedApp  # noqa: E402


def _engine():
    app = Flask("engine")

    @app.route("/")
    def home():
        return "<html><body>real</body></html>"

    @app.route("/api/conversations")
    def convos():
        return jsonify({"conversations": ["a"]})

    @app.route("/api/health")
    def health():
        return jsonify({"status": "healthy"})

    return app


class StagedAppTest(unittest.TestCase):
    def setUp(self):
        self.webui = tempfile.mkdtemp(prefix="gcs_staged_")
        with open(os.path.join(self.webui, "index.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><h1>UI</h1></body></html>")
        with open(os.path.join(self.webui, "script.js"), "w") as f:
            f.write("console.log(1)")
        with open(os.path.join(self.webui, "conversations.db"), "w") as f:
            f.write("secret")
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        shutil.rmtree(self.webui, ignore_errors=True)

    def _staged(self, loader=None, early_app=None):
        def slow():
            self.release.wait(10)
            return _engine()
        s = StagedApp(self.webui, loader or slow, log_path="desktop.log", early_app=early_app)
        s.start_loading()
        return s, Client(s)

    def test_conversations_answer_before_engine(self):
        from WebUI import conversation_db as cdb
        from WebUI.conversation_routes import create_early_app
        orig = cdb.DB_PATH
        cdb.DB_PATH = os.path.join(self.webui, "conv.db")
        try:
            self.assertIsNone(create_early_app())  # no database yet: wait for the engine
            cdb.init_db()
            key = "sk-test"
            cid = cdb.create_conversation("Saved earlier", user_id=cdb.hash_api_key(key))["id"]
            cdb.add_message(cid, "user", "hello")
            s, c = self._staged(early_app=create_early_app())
            hdr = {"X-API-Key": key}
            t = time.time()
            listed = json.loads(c.get("/api/conversations", headers=hdr).data)
            opened = json.loads(c.get(f"/api/conversations/{cid}", headers=hdr).data)
            self.assertLess(time.time() - t, 1)
            self.assertFalse(s.ready.is_set())  # engine still loading
            self.assertEqual([x["title"] for x in listed["conversations"]], ["Saved earlier"])
            self.assertEqual([m["content"] for m in opened["messages"]], ["hello"])
            self.assertTrue(opened["is_owner"])
            # No key: nothing leaks.
            self.assertEqual(json.loads(c.get("/api/conversations").data)["conversations"], [])
            # Routes the early app lacks (or other methods) still wait.
            result = {}
            th = threading.Thread(target=lambda: result.setdefault(
                "r", c.post("/api/conversations", headers=hdr, json={})))
            th.start()
            time.sleep(0.3)
            self.assertTrue(th.is_alive())
            self.release.set()
            th.join(5)
            self.assertIn("r", result)
        finally:
            cdb.DB_PATH = orig

    def test_page_and_files_before_engine(self):
        s, c = self._staged()
        t = time.time()
        r = c.get("/")
        self.assertLess(time.time() - t, 1)
        body = r.get_data(as_text=True)
        self.assertIn("<h1>UI</h1>", body)
        # No loading notice is shown; only the failure-message script is added.
        self.assertNotIn("Loading GIS Co-Scientist", body)
        self.assertNotIn('<div id="desktop-engine-notice"', body)
        self.assertIn("/__desktop/engine-status", body)
        self.assertLess(body.index("/__desktop/engine-status"), body.index("</body>"))
        self.assertEqual(c.get("/script.js").get_data(as_text=True), "console.log(1)")
        self.assertEqual(json.loads(c.get("/api/health").data), {"status": "starting"})
        self.assertFalse(json.loads(c.get("/__desktop/engine-status").data)["ready"])

    def test_api_waits_for_engine(self):
        s, c = self._staged()
        result = {}

        def call():
            result["r"] = c.get("/api/conversations")

        th = threading.Thread(target=call)
        th.start()
        time.sleep(0.3)
        self.assertTrue(th.is_alive())  # still waiting for the engine
        self.release.set()
        th.join(5)
        self.assertEqual(json.loads(result["r"].data), {"conversations": ["a"]})
        self.assertTrue(json.loads(c.get("/__desktop/engine-status").data)["ready"])
        # After loading, everything is the real app, with no notice.
        self.assertEqual(c.get("/").get_data(as_text=True), "<html><body>real</body></html>")
        self.assertEqual(json.loads(c.get("/api/health").data), {"status": "healthy"})

    def test_serves_only_page_files_inside_webui(self):
        outside = os.path.join(os.path.dirname(self.webui), "outside_gcs_test.js")
        with open(outside, "w") as f:
            f.write("nope")
        try:
            s = StagedApp(self.webui, _engine)  # not loading: stays in the early phase
            self.assertIsNone(s._static("/conversations.db"))
            self.assertIsNone(s._static("/../outside_gcs_test.js"))
            self.assertIsNone(s._static("/missing.js"))
            self.assertIsNotNone(s._static("/script.js"))
        finally:
            os.remove(outside)

    def test_engine_failure_is_reported(self):
        def broken():
            raise ImportError("scipy blocked")
        s, c = self._staged(loader=broken)
        s.ready.wait(5)
        st = json.loads(c.get("/__desktop/engine-status").data)
        self.assertEqual((st["ready"], st["error"], st["log"]), (False, True, "desktop.log"))
        r = c.get("/api/conversations")
        self.assertEqual(r.status_code, 503)
        self.assertIn("desktop.log", json.loads(r.data)["error"])
        self.assertIn("ImportError", s.error)


if __name__ == "__main__":
    unittest.main()
