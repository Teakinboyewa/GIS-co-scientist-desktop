"""Export a conversation on one "computer" and import it on another.

Each computer is a temp app folder plus its own conversations.db.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from WebUI import conversation_db as cdb  # noqa: E402
from WebUI import conversation_bundle as cb  # noqa: E402


class _Computer:
    def __init__(self, base, name):
        self.root = os.path.join(base, name, "GIS_Co_Scientist_Desk")
        os.makedirs(os.path.join(self.root, "WebUI"), exist_ok=True)
        self.db = os.path.join(base, name, "conversations.db")

    def use(self):
        cdb.DB_PATH = self.db
        cdb.init_db()


class ConversationBundleTest(unittest.TestCase):
    def setUp(self):
        self._orig_db = cdb.DB_PATH
        self.tmp = tempfile.mkdtemp(prefix="gcs_bundle_test_")
        self.sender = _Computer(self.tmp, "sender")
        self.recipient = _Computer(self.tmp, "recipient")
        self.sender.use()

        convo = cdb.create_conversation("Flood risk in Harris County", user_id="sender01", mode="task")
        self.cid = convo["id"]
        up_dir = os.path.join(self.sender.root, "WebUI", "uploads", self.cid)
        out_dir = os.path.join(self.sender.root, "Outputs", self.cid)
        os.makedirs(up_dir)
        os.makedirs(os.path.join(out_dir, "maps"))
        with open(os.path.join(up_dir, "parcels.geojson"), "w") as f:
            f.write('{"type": "FeatureCollection", "features": []}')
        self.upload_path = os.path.join(up_dir, "parcels.geojson")
        state = {
            "output_dir": out_dir,
            "inputs": [{"path": self.upload_path}],
            "sibling": self.sender.root + "2\\other.txt",
        }
        with open(os.path.join(out_dir, "workflow_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f)
        with open(os.path.join(out_dir, "maps", "flood.png"), "wb") as f:
            f.write(b"\x89PNG fake")

        cdb.add_message(self.cid, "user", "Map flood risk")
        cdb.add_message(self.cid, "assistant", "Done", message_type="workflow",
                        metadata={"layer": self.upload_path, "task_id": self.cid})
        cdb.add_conversation_file(self.cid, "parcels.geojson", "parcels.geojson", file_size=44)
        with open(self.upload_path, "rb") as f:
            cdb.save_file_blob(self.cid, "parcels.geojson", f.read())
        cdb.save_file_blob(self.cid, "only_in_db.csv", b"a,b\n1,2\n")
        cdb.save_artifact(self.cid, "execution_flowchart", "<html>flow</html>")

        self.bundle = os.path.join(self.tmp, "export.zip")

    def tearDown(self):
        cdb.DB_PATH = self._orig_db
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _export(self):
        self.sender.use()
        return cb.export_conversation(self.cid, self.bundle, app_root=self.sender.root)

    def test_round_trip(self):
        summary = self._export()
        self.assertEqual(summary["messages"], 2)
        with zipfile.ZipFile(self.bundle) as zf:
            names = set(zf.namelist())
        self.assertIn(f"outputs/{self.cid}/workflow_state.json", names)
        self.assertIn("uploads/parcels.geojson", names)
        # The blob identical to the upload is not stored twice.
        self.assertEqual(sum(1 for n in names if n.startswith("blobs/")), 1)

        self.recipient.use()
        convo = cb.import_conversation(self.bundle, "recip001", app_root=self.recipient.root)
        self.assertEqual(convo["id"], self.cid)
        self.assertEqual(convo["user_id"], "recip001")
        self.assertEqual(convo["title"], "Flood risk in Harris County")
        self.assertEqual(convo["mode"], "task")

        msgs = cdb.get_messages(self.cid)
        self.assertEqual([m["content"] for m in msgs], ["Map flood risk", "Done"])
        meta = json.loads(msgs[1]["metadata_json"])
        new_upload = os.path.join(self.recipient.root, "WebUI", "uploads", self.cid, "parcels.geojson")
        self.assertEqual(meta["layer"], new_upload)
        self.assertTrue(os.path.isfile(new_upload))

        out_dir = os.path.join(self.recipient.root, "Outputs", self.cid)
        with open(os.path.join(out_dir, "workflow_state.json"), encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state["output_dir"], out_dir)
        self.assertEqual(state["inputs"][0]["path"], new_upload)
        # A neighbouring folder that merely starts with the same name is untouched.
        self.assertEqual(state["sibling"], self.sender.root + "2\\other.txt")
        with open(os.path.join(out_dir, "maps", "flood.png"), "rb") as f:
            self.assertEqual(f.read(), b"\x89PNG fake")

        self.assertEqual([f["filename"] for f in cdb.get_conversation_files(self.cid)],
                         ["parcels.geojson"])
        self.assertEqual(cdb.get_file_blob(self.cid, "only_in_db.csv"), b"a,b\n1,2\n")
        self.assertIsNotNone(cdb.get_file_blob(self.cid, "parcels.geojson"))
        self.assertEqual(cdb.get_artifact(self.cid, "execution_flowchart"), "<html>flow</html>")
        listed = cdb.list_conversations(user_id="recip001")
        self.assertEqual([c["id"] for c in listed], [self.cid])

    def test_second_import_is_a_conflict(self):
        self._export()
        self.recipient.use()
        cb.import_conversation(self.bundle, "recip001", app_root=self.recipient.root)
        with self.assertRaises(cb.BundleConflict) as ctx:
            cb.import_conversation(self.bundle, "recip001", app_root=self.recipient.root)
        self.assertFalse(ctx.exception.deleted)

    def test_reimport_after_delete_on_same_computer(self):
        # Export, hide the conversation the old way (soft delete), then
        # import the file back.
        self._export()
        cdb.delete_conversation(self.cid)
        cdb.add_message(self.cid, "user", "added after the export")
        leftover = os.path.join(self.sender.root, "Outputs", self.cid, "after_export.tif")
        with open(leftover, "wb") as f:
            f.write(b"x" * 1000)
        self.assertEqual(cdb.list_conversations(user_id="sender01"), [])

        convo = cb.import_conversation(self.bundle, "sender01", app_root=self.sender.root)
        self.assertIsNone(convo["deleted_at"])
        self.assertEqual([c["id"] for c in cdb.list_conversations(user_id="sender01")], [self.cid])
        # The file's contents replace the deleted copy, not add to it.
        self.assertEqual([m["content"] for m in cdb.get_messages(self.cid)], ["Map flood risk", "Done"])
        self.assertEqual(len(cdb.get_conversation_files(self.cid)), 1)
        self.assertEqual(cdb.get_file_blob(self.cid, "only_in_db.csv"), b"a,b\n1,2\n")
        self.assertEqual(cdb.get_artifact(self.cid, "execution_flowchart"), "<html>flow</html>")
        self.assertTrue(os.path.isfile(
            os.path.join(self.sender.root, "Outputs", self.cid, "workflow_state.json")))
        # Files the deleted copy had but the export didn't are gone.
        self.assertFalse(os.path.exists(leftover))

    def test_import_needs_a_user(self):
        self._export()
        self.recipient.use()
        with self.assertRaises(cb.BundleError):
            cb.import_conversation(self.bundle, None, app_root=self.recipient.root)

    def test_rejects_paths_outside_the_app(self):
        self._export()
        bad = os.path.join(self.tmp, "bad.zip")
        with zipfile.ZipFile(self.bundle) as src, zipfile.ZipFile(bad, "w") as dst:
            for item in src.infolist():
                dst.writestr(item, src.read(item))
            dst.writestr("uploads/../../escaped.txt", "x")
        self.recipient.use()
        with self.assertRaises(cb.BundleError):
            cb.import_conversation(bad, "recip001", app_root=self.recipient.root)
        self.assertIsNone(cdb.get_conversation(self.cid))
        self.assertFalse(os.path.exists(os.path.join(self.recipient.root, "WebUI", "uploads", self.cid)))
        self.assertFalse(os.path.exists(os.path.join(self.recipient.root, "escaped.txt")))

    def test_rejects_other_zip_files(self):
        other = os.path.join(self.tmp, "other.zip")
        with zipfile.ZipFile(other, "w") as zf:
            zf.writestr("readme.txt", "hello")
        self.recipient.use()
        with self.assertRaises(cb.BundleError):
            cb.import_conversation(other, "recip001", app_root=self.recipient.root)

    def test_rewriter_styles(self):
        rw = cb.make_path_rewriter(r"C:\Users\a\Desk", r"D:\Apps\Desk")
        self.assertEqual(rw(r"C:\Users\a\Desk\Outputs\x"), r"D:\Apps\Desk\Outputs\x")
        self.assertEqual(rw(r'"C:\\Users\\a\\Desk\\Outputs"'), r'"D:\\Apps\\Desk\\Outputs"')
        self.assertEqual(rw("C:/Users/a/Desk/Outputs"), "D:/Apps/Desk/Outputs")
        self.assertEqual(rw(r"c:\users\a\desk\x"), r"D:\Apps\Desk\x")
        self.assertEqual(rw(r"C:\Users\a\Desktop\x"), r"C:\Users\a\Desktop\x")
        self.assertIsNone(cb.make_path_rewriter(r"C:\A", r"c:\a"))


if __name__ == "__main__":
    unittest.main()
