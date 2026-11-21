"""Desktop: a user's data sources belong to this computer, whatever API key is
set (like conversations). Sources saved under earlier keys are merged into one
local folder."""

import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Load by file path: importing the agents package would load every agent.
_spec = importlib.util.spec_from_file_location(
    "uds_under_test", os.path.join(ROOT, "agents", "data_agent", "user_data_sources.py"))


def _fresh_module(root):
    mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mod)
    mod.HANDBOOKS_ROOT = root
    mod.GLOBAL_DIR = os.path.join(root, "Handbooks")
    mod.USER_ROOT = os.path.join(root, "Handbooks_user")
    mod.PENDING_DIR = os.path.join(root, "Handbooks_pending")
    mod.KEYS_DIR = os.path.join(root, "Keys")
    return mod


def _write(path, text, age_s=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))


def _toml(name, desc):
    return (f'data_source_name = "{name}"\nbrief_description = "{desc}"\n'
            'handbook = "h"\ncode_example = "c"\n')


class LocalUserSourcesTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="gcs_uds_")
        u = os.path.join(self.root, "Handbooks_user")
        p = os.path.join(self.root, "Handbooks_pending")
        os.makedirs(os.path.join(self.root, "Handbooks"))
        # Key A: older CDC copy, plus its own source with an attachment and keys.
        _write(os.path.join(u, "aaaa1111", "cdc_places.toml"), _toml("CDC PLACES", "old copy"), age_s=3600)
        _write(os.path.join(u, "aaaa1111", "cdc_places.meta"), "{}", age_s=3600)
        _write(os.path.join(u, "aaaa1111", "epa_aqs.toml"), _toml("EPA AQS", "air"))
        _write(os.path.join(u, "aaaa1111", "epa_aqs.keys"), "[API_Key]\nEPA_KEY = \n")
        _write(os.path.join(u, "aaaa1111", "files", "epa_aqs", "notes.txt"), "attachment")
        # Key B: newer CDC copy and another source.
        _write(os.path.join(u, "bbbb2222", "cdc_places.toml"), _toml("CDC PLACES", "new copy"))
        _write(os.path.join(u, "bbbb2222", "worldpop.toml"), _toml("WorldPop", "pop"))
        # Review queue entries under both keys.
        _write(os.path.join(p, "aaaa1111__cdc_places.json"), '{"slug": "cdc_places", "v": "old"}', age_s=3600)
        _write(os.path.join(p, "bbbb2222__cdc_places.json"), '{"slug": "cdc_places", "v": "new"}')
        _write(os.path.join(p, "aaaa1111__epa_aqs.json"), '{"slug": "epa_aqs"}')
        self.uds = _fresh_module(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_merge_keeps_newest_and_sets_older_aside(self):
        listed = self.uds.list_user_sources("any-key-hash")
        names = sorted(s["data_source_name"] for s in listed)
        self.assertEqual(names, ["CDC PLACES", "EPA AQS", "WorldPop"])
        cdc = next(s for s in listed if s["data_source_name"] == "CDC PLACES")
        self.assertEqual(cdc["brief_description"], "new copy")

        u = self.uds.USER_ROOT
        self.assertEqual(sorted(os.listdir(u)), ["_previous_keys", "local"])
        local = os.path.join(u, "local")
        self.assertTrue(os.path.isfile(os.path.join(local, "epa_aqs.keys")))
        self.assertTrue(os.path.isfile(os.path.join(local, "files", "epa_aqs", "notes.txt")))
        older = os.path.join(u, "_previous_keys", "aaaa1111", "cdc_places.toml")
        with open(older, encoding="utf-8") as f:
            self.assertIn("old copy", f.read())        # set aside, not deleted

        pending = sorted(n for n in os.listdir(self.uds.PENDING_DIR) if n.endswith(".json"))
        self.assertEqual(pending, ["local__cdc_places.json", "local__epa_aqs.json"])
        with open(os.path.join(self.uds.PENDING_DIR, "local__cdc_places.json")) as f:
            self.assertIn('"new"', f.read())
        # Desktop shares through GitHub pull requests, so a local queue entry
        # doesn't make a source "pending".
        self.assertEqual(self.uds._source_status("whatever", "epa_aqs"), "private")
        self.uds.DESKTOP_IGNORES_LOCAL_QUEUE = False
        self.assertEqual(self.uds._source_status("whatever", "epa_aqs"), "pending")

    def test_same_sources_for_every_key(self):
        a = self.uds.list_user_sources("key-one")
        b = self.uds.list_user_sources("key-two")
        self.assertEqual([s["data_source_name"] for s in a], [s["data_source_name"] for s in b])
        slug = self.uds.write_user_source("key-three", {
            "data_source_name": "OpenTopography", "brief_description": "dem",
            "handbook": "h", "code_example": "c"})
        self.assertTrue(os.path.isfile(os.path.join(self.uds.USER_ROOT, "local", f"{slug}.toml")))
        self.assertIn("OpenTopography",
                      [s["data_source_name"] for s in self.uds.list_user_sources("key-one")])

    def test_catalog_name_match_is_not_shared(self):
        # A bundled handbook with the same name (different case, as on Windows).
        _write(os.path.join(self.uds.GLOBAL_DIR, "CDC_PLACES.toml"), _toml("CDC PLACES", "bundled"))
        srcs = {s["data_source_name"]: s for s in self.uds.list_user_sources("k")}
        self.assertNotEqual(srcs["CDC PLACES"]["status"], "approved")
        self.assertEqual(srcs["WorldPop"]["status"], "private")

    def test_status_is_shared_after_a_real_approval(self):
        slug = self.uds.write_user_source("k", {
            "data_source_name": "OpenTopography DEM", "brief_description": "dem",
            "handbook": "h", "code_example": "c"}, share=True)
        self.assertEqual(self.uds._source_status("k", slug), "private")
        self.assertTrue(self.uds.approve_pending(self.uds._pending_id("k", slug)))
        self.assertEqual(self.uds._source_status("k", slug), "approved")

    def test_merge_runs_again_harmlessly(self):
        self.uds.list_user_sources("x")
        again = _fresh_module(self.root)   # a new process
        self.assertEqual(len(again.list_user_sources("y")), 3)
        self.assertEqual(sorted(os.listdir(again.USER_ROOT)), ["_previous_keys", "local"])


if __name__ == "__main__":
    unittest.main()
