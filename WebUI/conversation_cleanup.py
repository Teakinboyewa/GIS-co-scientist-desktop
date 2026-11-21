"""
Permanently remove a deleted conversation and free its disk space.

Deleting used to only hide a conversation (``deleted_at``). Nothing ever
restores one, so its uploads, outputs and stored file copies stayed on disk
for good and added up. In the desktop app, an owner's delete now removes:

    database   the conversation row, its messages, file records and
               collaborators (cascade), its file_blobs, and its artifacts
    disk       WebUI/uploads/<conversation_id>/ and Outputs/<task_id>/

An Outputs folder or artifact another live conversation still points at (the
same id as its own id or task_id) is kept.
"""

from __future__ import annotations

import os
import shutil
import threading

from WebUI import conversation_db as cdb

# Compact the database once at least this much of it is free pages.
VACUUM_MIN_FREE_BYTES = 64 * 1024 * 1024
_vacuum_lock = threading.Lock()


def default_app_root():
    from utils.config import load_configurations
    return load_configurations().root_dir


def _tree_bytes(path):
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _remove_tree(path):
    """Delete a folder; returns the bytes it held (0 if it wasn't there)."""
    if not os.path.isdir(path):
        return 0
    size = _tree_bytes(path)
    shutil.rmtree(path, ignore_errors=True)
    return size


def _ids_in_use(conn, exclude_id):
    rows = conn.execute(
        "SELECT id, task_id FROM conversations WHERE id != ? AND deleted_at IS NULL",
        (exclude_id,),
    ).fetchall()
    used = set()
    for r in rows:
        used.add(r["id"])
        if r["task_id"]:
            used.add(r["task_id"])
    return used


def purge_conversation(conversation_id, app_root=None):
    """Remove a conversation and its files for good.

    Returns the bytes freed on disk and in stored blobs, or None when there
    was no such conversation.
    """
    app_root = os.path.abspath(app_root or default_app_root())
    conn = cdb._get_connection()
    try:
        row = conn.execute(
            "SELECT id, task_id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        task_ids = {conversation_id}
        if row["task_id"]:
            task_ids.add(row["task_id"])
        task_ids -= _ids_in_use(conn, conversation_id)

        blob_bytes = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM file_blobs WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        # messages, conversation_files and collaborators cascade.
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.execute("DELETE FROM file_blobs WHERE conversation_id = ?", (conversation_id,))
        if task_ids:
            ids = sorted(task_ids)
            conn.execute(
                f"DELETE FROM artifacts WHERE task_id IN ({','.join('?' * len(ids))})", ids)
        conn.commit()
    finally:
        conn.close()

    freed = blob_bytes
    freed += _remove_tree(os.path.join(app_root, "WebUI", "uploads", conversation_id))
    for tid in task_ids:
        freed += _remove_tree(os.path.join(app_root, "Outputs", tid))
    return freed


def purge_deleted_conversations(app_root=None):
    """Remove every conversation that was only hidden by an earlier delete.
    Returns (count, bytes freed)."""
    conn = cdb._get_connection()
    try:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM conversations WHERE deleted_at IS NOT NULL").fetchall()]
    finally:
        conn.close()
    freed = 0
    for cid in ids:
        freed += purge_conversation(cid, app_root=app_root) or 0
    return len(ids), freed


def free_db_bytes():
    conn = cdb._get_connection()
    try:
        pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
        size = conn.execute("PRAGMA page_size").fetchone()[0]
        return pages * size
    finally:
        conn.close()


def vacuum_if_worthwhile(min_free_bytes=VACUUM_MIN_FREE_BYTES):
    """SQLite keeps deleted rows' space inside the file. Compact it once
    enough is free. Returns True when it compacted."""
    if not _vacuum_lock.acquire(blocking=False):
        return False
    try:
        if free_db_bytes() < min_free_bytes:
            return False
        conn = cdb._get_connection()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        return True
    except Exception as e:  # a busy database just waits for the next delete
        print(f"[cleanup] database compaction skipped: {e}")
        return False
    finally:
        _vacuum_lock.release()


def vacuum_in_background():
    threading.Thread(target=vacuum_if_worthwhile, daemon=True,
                     name="conversation-db-vacuum").start()
