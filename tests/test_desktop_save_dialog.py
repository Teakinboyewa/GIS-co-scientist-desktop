"""DesktopApi.save_text keeps the file type when the user renames the file."""

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "desktop"))

import launcher  # noqa: E402  (module import only; main() is not run)


class _FakeWindow:
    """Stands in for the pywebview window: returns what the user 'typed'."""

    def __init__(self, typed_path):
        self.typed_path = typed_path
        self.calls = []

    def create_file_dialog(self, dialog_type, save_filename="", file_types=()):
        self.calls.append((dialog_type, save_filename, tuple(file_types)))
        return self.typed_path


class SaveTextTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="gcs_save_")
        self.api = launcher.DesktopApi()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _save(self, typed, name="cdc_places.toml"):
        self.api._window = _FakeWindow(typed)
        return self.api.save_text(name, 'data_source_name = "CDC"\n', "toml", "TOML handbook")

    def test_renamed_without_extension_gets_toml(self):
        r = self._save(os.path.join(self.dir, "my renamed source"))
        self.assertEqual(r["saved"], os.path.join(self.dir, "my renamed source.toml"))
        with open(r["saved"], encoding="utf-8") as f:
            self.assertIn("CDC", f.read())
        dialog_type, suggested, types = self.api._window.calls[0]
        self.assertEqual(suggested, "cdc_places.toml")
        self.assertEqual(types[0], "TOML handbook (*.toml)")

    def test_name_with_extension_is_kept(self):
        r = self._save(os.path.join(self.dir, "kept.toml"))
        self.assertEqual(os.path.basename(r["saved"]), "kept.toml")

    def test_dialog_may_return_a_tuple(self):
        r = self._save((os.path.join(self.dir, "from_tuple"),))
        self.assertEqual(os.path.basename(r["saved"]), "from_tuple.toml")

    def test_cancel(self):
        self.assertEqual(self._save(None), {"cancelled": True})
        self.assertEqual(os.listdir(self.dir), [])

    def test_suggested_name_gets_extension(self):
        self.api._window = _FakeWindow(None)
        self.api.save_text("no_ext_name", "x", "toml", "TOML handbook")
        self.assertEqual(self.api._window.calls[0][1], "no_ext_name.toml")


if __name__ == "__main__":
    unittest.main()
