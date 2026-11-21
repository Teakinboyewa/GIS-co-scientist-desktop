"""Deleting a conversation frees its disk space."""

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from WebUI import conversation_db as cdb  # noqa: E402
from WebUI import conversation_cleanup as cc  # noqa: E402


class ConversationCleanupTest(unittest.TestCase):
    def setUp(self):
        self._orig_db = cdb.DB_PATH
        self.tmp = tempfile.mkdtemp(prefix="gcs_cleanup_test_")
        self.root = os.path.join(self.tmp, "app")
        cdb.DB_PATH = os.path.join(self.tmp, "conversations.db")
        cdb.init_db()

    def tearDown(self):
        cdb.DB_PATH = self._orig_db
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _conversation(self, title, size=1000):
        cid = cdb.create_conversation(title, user_id="u1")["id"]
        cdb.add_message(cid, "user", "hi")
        cdb.add_conversation_file(cid, "data.csv", "data.csv", file_size=size)
        cdb.save_file_blob(cid, "data.csv", b"x" * size)
        cdb.save_artifact(cid, "execution_flowchart", "<html/>")
        cdb.add_collaborator(cid, "u2")
        for d in (os.path.join(self.root, "WebUI", "uploads", cid),
                  os.path.join(self.root, "Outputs", cid)):
            os.makedirs(d)
            with open(os.path.join(d, "data.csv"), "wb") as f:
                f.write(b"x" * size)
        return cid

    def _count(self, table, column, value):
        conn = cdb._get_connection()
        try:
            return conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,)).fetchone()[0]
        finally:
            conn.close()

    def test_purge_removes_rows_and_folders(self):
        cid = self._conversation("Gone", size=5000)
        keep = self._conversation("Kept")
        freed = cc.purge_conversation(cid, app_root=self.root)
        self.assertEqual(freed, 15000)  # blob + upload + output
        self.assertIsNone(cdb.get_conversation(cid))
        for table, col in (("messages", "conversation_id"),
                           ("conversation_files", "conversation_id"),
                           ("file_blobs", "conversation_id"),
                           ("conversation_collaborators", "conversation_id"),
                           ("artifacts", "task_id")):
            self.assertEqual(self._count(table, col, cid), 0, table)
        self.assertFalse(os.path.exists(os.path.join(self.root, "WebUI", "uploads", cid)))
        self.assertFalse(os.path.exists(os.path.join(self.root, "Outputs", cid)))
        # The other conversation is untouched.
        self.assertEqual(len(cdb.get_messages(keep)), 1)
        self.assertTrue(os.path.isdir(os.path.join(self.root, "Outputs", keep)))
        self.assertIsNone(cc.purge_conversation(cid, app_root=self.root))

    def test_keeps_outputs_another_conversation_uses(self):
        cid = self._conversation("Source")
        other = cdb.create_conversation("Uses the same task", user_id="u1")["id"]
        cdb.link_task_id(other, cid)
        cc.purge_conversation(cid, app_root=self.root)
        self.assertTrue(os.path.isdir(os.path.join(self.root, "Outputs", cid)))
        self.assertEqual(cdb.get_artifact(cid, "execution_flowchart"), "<html/>")
        self.assertFalse(os.path.exists(os.path.join(self.root, "WebUI", "uploads", cid)))

    def test_purges_earlier_soft_deletes(self):
        a = self._conversation("Deleted before")
        b = self._conversation("Still here")
        cdb.delete_conversation(a)
        count, freed = cc.purge_deleted_conversations(app_root=self.root)
        self.assertEqual((count, freed), (1, 3000))
        self.assertIsNone(cdb.get_conversation(a))
        self.assertIsNotNone(cdb.get_conversation(b))
        self.assertEqual(cc.purge_deleted_conversations(app_root=self.root), (0, 0))

    def test_vacuum_shrinks_the_database(self):
        cid = self._conversation("Big", size=4 * 1024 * 1024)
        size_before = os.path.getsize(cdb.DB_PATH)
        cc.purge_conversation(cid, app_root=self.root)
        self.assertGreater(cc.free_db_bytes(), 3 * 1024 * 1024)
        self.assertFalse(cc.vacuum_if_worthwhile(min_free_bytes=1 << 40))
        self.assertTrue(cc.vacuum_if_worthwhile(min_free_bytes=1024 * 1024))
        self.assertLess(cc.free_db_bytes(), 1024 * 1024)
        self.assertLess(os.path.getsize(cdb.DB_PATH), size_before)


if __name__ == "__main__":
    unittest.main()
