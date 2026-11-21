"""The Mac starter keeps a working copy of the app and never touches user data."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "desktop"))

import mac_bootstrap as mb  # noqa: E402


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class MacBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gcs_macboot_"))
        self.bundle = self.tmp / "bundle_app"
        self.home = self.tmp / "home_app"
        _write(self.bundle / "VERSION", "0.1.1")
        _write(self.bundle / "WebUI" / "app.py", "v1")
        _write(self.bundle / "desktop" / "launcher.py", "launcher v1")
        # Things that must never reach (or overwrite) the working copy:
        _write(self.bundle / "WebUI" / "conversations.db", "bundled db")
        _write(self.bundle / "WebUI" / "uploads" / "x.csv", "bundled upload")
        h = self.bundle / "agents" / "data_agent" / "DataRetriever_Handbooks"
        _write(h / "Handbooks" / "CDC_PLACES.toml", "catalog")
        _write(h / "Handbooks_user" / "local" / "mine.toml", "bundled private")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_launch_copies_code_but_not_data(self):
        self.assertTrue(mb.sync_app(self.bundle, self.home))
        self.assertEqual((self.home / "WebUI" / "app.py").read_text(), "v1")
        self.assertEqual((self.home / "VERSION").read_text(), "0.1.1")
        h = self.home / "agents" / "data_agent" / "DataRetriever_Handbooks"
        self.assertTrue((h / "Handbooks" / "CDC_PLACES.toml").is_file())     # shared catalog: code
        self.assertFalse((h / "Handbooks_user").exists())
        self.assertFalse((self.home / "WebUI" / "conversations.db").exists())
        self.assertFalse((self.home / "WebUI" / "uploads").exists())

    def test_same_version_does_nothing(self):
        mb.sync_app(self.bundle, self.home)
        _write(self.home / "WebUI" / "app.py", "edited locally")
        self.assertFalse(mb.sync_app(self.bundle, self.home))
        self.assertEqual((self.home / "WebUI" / "app.py").read_text(), "edited locally")

    def test_update_replaces_code_and_keeps_user_data(self):
        mb.sync_app(self.bundle, self.home)
        # The user's data in the working copy:
        _write(self.home / "WebUI" / "conversations.db", "MY CONVERSATIONS")
        _write(self.home / "WebUI" / "uploads" / "mine.gpkg", "MY UPLOAD")
        _write(self.home / "Outputs" / "task1" / "map.png", "MY OUTPUT")
        _write(self.home / "desktop" / "profile" / "Local Storage" / "x", "MY SETTINGS")
        mine = self.home / "agents" / "data_agent" / "DataRetriever_Handbooks" / "Handbooks_user" / "local" / "src.toml"
        _write(mine, "MY SOURCE")
        _write(self.home / "WebUI" / ".catalog_sync.json", "MY SYNC STATE")
        # A new version arrives:
        _write(self.bundle / "VERSION", "0.1.2")
        _write(self.bundle / "WebUI" / "app.py", "v2")
        self.assertTrue(mb.sync_app(self.bundle, self.home))
        self.assertEqual((self.home / "WebUI" / "app.py").read_text(), "v2")
        self.assertEqual((self.home / "VERSION").read_text(), "0.1.2")
        self.assertEqual((self.home / "WebUI" / "conversations.db").read_text(), "MY CONVERSATIONS")
        self.assertEqual((self.home / "WebUI" / "uploads" / "mine.gpkg").read_text(), "MY UPLOAD")
        self.assertEqual((self.home / "Outputs" / "task1" / "map.png").read_text(), "MY OUTPUT")
        self.assertEqual((self.home / "desktop" / "profile" / "Local Storage" / "x").read_text(), "MY SETTINGS")
        self.assertEqual(mine.read_text(), "MY SOURCE")
        self.assertEqual((self.home / "WebUI" / ".catalog_sync.json").read_text(), "MY SYNC STATE")

    def test_user_data_rules(self):
        yes = ["WebUI/conversations.db", "WebUI/uploads/a.csv", "Outputs/x/y.png", ".env",
               "desktop/profile/x", "desktop/logs/desktop.log", "WebUI/x.db-wal",
               "agents/data_agent/DataRetriever_Handbooks/Handbooks_pending/a.json"]
        no = ["WebUI/app.py", "WebUI/uploads_helper.py", "desktop/launcher.py",
              "agents/data_agent/DataRetriever_Handbooks/Handbooks/CDC_PLACES.toml"]
        for rel in yes:
            self.assertTrue(mb._is_user_data(rel), rel)
        for rel in no:
            self.assertFalse(mb._is_user_data(rel), rel)


if __name__ == "__main__":
    unittest.main()
