"""
SQLite-backed conversation persistence for AGM WebUI.

Stores conversation metadata and individual messages so that users can
resume past sessions after a browser refresh.
"""

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone

# Database file lives alongside app.py
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.db")


def _get_connection():
    """Open a new SQLite connection (one per call for thread safety)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they do not exist. Call once at app startup."""
    conn = _get_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                task_id    TEXT,
                user_id    TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT    NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                message_type    TEXT    DEFAULT 'text',
                metadata_json   TEXT,
                timestamp       TEXT    NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversation_files (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                display_name    TEXT    NOT NULL,
                file_size       INTEGER DEFAULT 0,
                file_type       TEXT,
                audit_json      TEXT,
                added_at        TEXT    NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE(conversation_id, filename)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id       TEXT    NOT NULL,
                artifact_type TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                UNIQUE(task_id, artifact_type)
            );

            CREATE TABLE IF NOT EXISTS file_blobs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                content         BLOB    NOT NULL,
                created_at      TEXT    NOT NULL,
                UNIQUE(conversation_id, filename)
            );

            CREATE TABLE IF NOT EXISTS conversation_collaborators (
                conversation_id TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                added_at        TEXT NOT NULL,
                PRIMARY KEY (conversation_id, user_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)
        # Migration: add user_id column if upgrading from an older schema
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Migration: add deleted_at for soft deletes. When a user deletes a
        # conversation it is hidden from listings but kept in the database.
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN deleted_at TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Migration: add mode (intent mode the conversation was started in:
        # 'task' (Spatial Analysis), 'data_retriever', 'research', 'chat', 'auto').
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN mode TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Migration: add pinned flag so users can pin conversations to the top.
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Migration: removed_at marks a file the user removed from Data Layers.
        # Kept (on disk and in file_blobs) rather than deleted, but no longer
        # restored as a layer when the conversation is reopened.
        try:
            conn.execute("ALTER TABLE conversation_files ADD COLUMN removed_at TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
    finally:
        conn.close()


def hash_api_key(api_key):
    """Derive a stable user_id from an API key (SHA-256, first 16 hex chars).
    The key itself is never stored."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _row_to_dict(row):
    """Convert a sqlite3.Row to a plain dict."""
    if row is None:
        return None
    return dict(row)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── Conversations ────────────────────────────────────────────────────


def create_conversation(title=None, user_id=None, mode=None):
    """Create a new conversation and return its dict.

    ``mode`` is the intent mode the conversation was started in (e.g. 'task',
    'data_retriever', 'research', 'chat'), used to label cards in the sidebar.
    """
    conv_id = str(uuid.uuid4())
    now = _now_iso()
    title = title or "New Conversation"
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, user_id, mode) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, title, now, now, user_id, mode),
        )
        conn.commit()
        return {"id": conv_id, "title": title, "created_at": now, "updated_at": now,
                "task_id": None, "user_id": user_id, "mode": mode}
    finally:
        conn.close()


def get_conversation(conversation_id):
    """Return a single conversation dict or None."""
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def caller_owns(convo, caller_user_id):
    """Whether the caller may act as owner of ``convo``.

    Desktop app: one person uses this computer's database, so anyone with an
    API key owns every conversation in it. Conversations are moved between
    computers with Export/Import, not shared links, so there are no
    collaborators.
    """
    return bool(convo) and bool(caller_user_id)


def list_conversations(limit=50, offset=0, user_id=None):
    """Return every conversation in this computer's database, most recently
    updated first. See the note in the query on why user_id doesn't filter."""
    conn = _get_connection()
    try:
        # Correlated subqueries enrich each conversation with a short preview
        # (first user message) and a message count, so the sidebar can render
        # richer cards without an extra round-trip per conversation.
        preview_sql = (
            "(SELECT m.content FROM messages m "
            " WHERE m.conversation_id = c.id AND m.role = 'user' "
            " ORDER BY m.timestamp ASC LIMIT 1) AS preview"
        )
        count_sql = (
            "(SELECT COUNT(*) FROM messages m2 "
            " WHERE m2.conversation_id = c.id) AS message_count"
        )
        # Desktop app: the database is on this computer and belongs to the
        # person using it, so every conversation in it is theirs, including
        # imported ones and ones made under an earlier API key. Nothing is
        # shared, so is_owner is always 1 and collaborator_count always 0
        # (rows left in conversation_collaborators from the web app's
        # shared links are ignored). user_id is accepted for compatibility.
        rows = conn.execute(
            f"SELECT c.*, {preview_sql}, {count_sql}, "
            "1 AS is_owner, 0 AS collaborator_count FROM conversations c "
            "WHERE c.deleted_at IS NULL "
            "ORDER BY c.updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def add_collaborator(conversation_id, user_id):
    """Add a user as a collaborator on a conversation (idempotent).

    Used when a user opens someone else's shared conversation link and
    interacts with it — the conversation appears in their sidebar without
    being forked or duplicated. Same conversation_id, same task workspace.
    No-op if the row already exists.
    """
    if not conversation_id or not user_id:
        return False
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO conversation_collaborators "
            "(conversation_id, user_id, added_at) VALUES (?, ?, ?)",
            (conversation_id, user_id, _now_iso()),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def is_collaborator(conversation_id, user_id):
    """True when ``user_id`` has joined this conversation from a shared link."""
    if not conversation_id or not user_id:
        return False
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM conversation_collaborators WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def remove_collaborator(conversation_id, user_id):
    """Remove a user as a collaborator on a conversation (idempotent).

    Used when a collaborator (non-owner) chooses to remove a shared
    conversation from their own sidebar. Only the collaborator row is
    deleted — the conversation itself and the owner's access are untouched.
    No-op if the row does not exist.
    """
    if not conversation_id or not user_id:
        return False
    conn = _get_connection()
    try:
        conn.execute(
            "DELETE FROM conversation_collaborators "
            "WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def update_conversation_title(conversation_id, title):
    """Update the title of a conversation."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now_iso(), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_conversation_mode(conversation_id, mode):
    """Set a conversation's intent mode without touching updated_at.

    Used by the mode backfill so reclassifying old conversations doesn't
    bump them to the top of the recency-ordered sidebar.
    """
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET mode = ? WHERE id = ?",
            (mode, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_conversation_pinned(conversation_id, pinned):
    """Pin or unpin a conversation. Does not touch updated_at so pinning
    doesn't reorder the recency-sorted sidebar."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def link_task_id(conversation_id, task_id):
    """Associate a workflow task_id with a conversation."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET task_id = ?, updated_at = ? WHERE id = ?",
            (task_id, _now_iso(), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation_by_task_id(task_id):
    """Look up a conversation by its linked task_id."""
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM conversations WHERE task_id = ?", (task_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def fork_conversation(source_id, new_user_id, new_title=None):
    """Create a deep copy of a conversation owned by ``new_user_id``.

    Copies: the conversation row (with a new UUID, task_id=NULL), every message,
    every conversation_files entry, and every file_blob. The source conversation
    is not modified. Returns the new conversation dict, or None if the source
    does not exist.
    """
    conn = _get_connection()
    try:
        src = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (source_id,)
        ).fetchone()
        if src is None:
            return None

        new_id = str(uuid.uuid4())
        now = _now_iso()
        title = new_title or f"Fork of {src['title']}"

        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, task_id, user_id) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (new_id, title, now, now, new_user_id),
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, message_type, metadata_json, timestamp) "
            "SELECT ?, role, content, message_type, metadata_json, timestamp "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (new_id, source_id),
        )
        conn.execute(
            "INSERT INTO conversation_files "
            "(conversation_id, filename, display_name, file_size, file_type, audit_json, added_at) "
            "SELECT ?, filename, display_name, file_size, file_type, audit_json, added_at "
            "FROM conversation_files WHERE conversation_id = ?",
            (new_id, source_id),
        )
        conn.execute(
            "INSERT INTO file_blobs (conversation_id, filename, content, created_at) "
            "SELECT ?, filename, content, created_at "
            "FROM file_blobs WHERE conversation_id = ?",
            (new_id, source_id),
        )
        conn.commit()
        return {
            "id": new_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "task_id": None,
            "user_id": new_user_id,
        }
    finally:
        conn.close()


def delete_conversation(conversation_id):
    """Soft-delete a conversation: mark it deleted so it disappears from the
    user's history, but keep the row and all its messages in the database.

    The conversation is hidden from ``list_conversations`` but remains fully
    recoverable (clear ``deleted_at`` to restore it).
    """
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET deleted_at = ? WHERE id = ?",
            (_now_iso(), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Messages ─────────────────────────────────────────────────────────


def add_message(conversation_id, role, content, message_type="text", metadata=None):
    """Insert a message and update the conversation's updated_at timestamp."""
    import json as _json

    now = _now_iso()
    metadata_str = _json.dumps(metadata) if metadata else None
    conn = _get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, message_type, metadata_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, message_type, metadata_str, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
        return {
            "id": cur.lastrowid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "message_type": message_type,
            "metadata_json": metadata_str,
            "timestamp": now,
        }
    finally:
        conn.close()


def update_message_metadata(message_id, content=None, metadata=None):
    """Update the metadata (and optionally content) of an existing message."""
    import json as _json

    conn = _get_connection()
    try:
        if content is not None and metadata is not None:
            metadata_str = _json.dumps(metadata)
            conn.execute(
                "UPDATE messages SET content = ?, metadata_json = ? WHERE id = ?",
                (content, metadata_str, message_id),
            )
        elif metadata is not None:
            metadata_str = _json.dumps(metadata)
            conn.execute(
                "UPDATE messages SET metadata_json = ? WHERE id = ?",
                (metadata_str, message_id),
            )
        elif content is not None:
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (content, message_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_messages(conversation_id, limit=None, offset=0):
    """Return messages for a conversation ordered by timestamp ASC."""
    conn = _get_connection()
    try:
        if limit is not None:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",
                (conversation_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conversation_id,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ── Conversation Files ────────────────────────────────────────────────


def add_conversation_file(conversation_id, filename, display_name,
                          file_size=0, file_type=None, audit_json=None):
    """Associate an uploaded file with a conversation (upsert by filename)."""
    now = _now_iso()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO conversation_files "
            "(conversation_id, filename, display_name, file_size, file_type, audit_json, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(conversation_id, filename) DO UPDATE SET "
            "  file_size = excluded.file_size, "
            "  file_type = excluded.file_type, "
            "  audit_json = COALESCE(excluded.audit_json, conversation_files.audit_json), "
            # Adding the file again (a new upload or download with the same
            # name) brings back a layer the user had removed.
            "  removed_at = NULL",
            (conversation_id, filename, display_name, file_size, file_type, audit_json, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation_files(conversation_id):
    """Return all file records for a conversation."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM conversation_files WHERE conversation_id = ? ORDER BY added_at ASC",
            (conversation_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def mark_conversation_file_removed(conversation_id, filename):
    """Hide a file from the conversation's Data Layers (soft remove).

    Returns True when a matching file record was found.
    """
    conn = _get_connection()
    try:
        cur = conn.execute(
            "UPDATE conversation_files SET removed_at = ? "
            "WHERE conversation_id = ? AND filename = ?",
            (_now_iso(), conversation_id, filename),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def restore_conversation_file(conversation_id, filename):
    """Undo mark_conversation_file_removed. Returns True when a removed
    record was restored."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            "UPDATE conversation_files SET removed_at = NULL "
            "WHERE conversation_id = ? AND filename = ? AND removed_at IS NOT NULL",
            (conversation_id, filename),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def update_conversation_file_audit(conversation_id, filename, audit_json):
    """Update the audit_json for a specific file in a conversation."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE conversation_files SET audit_json = ? "
            "WHERE conversation_id = ? AND filename = ?",
            (audit_json, conversation_id, filename),
        )
        conn.commit()
    finally:
        conn.close()


# ── Artifacts (HTML visualizations) ───────────────────────────────────


def save_artifact(task_id, artifact_type, content):
    """Store (or replace) an HTML artifact for a task."""
    now = _now_iso()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO artifacts (task_id, artifact_type, content, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(task_id, artifact_type) DO UPDATE SET "
            "  content = excluded.content, "
            "  created_at = excluded.created_at",
            (task_id, artifact_type, content, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_artifact(task_id, artifact_type):
    """Retrieve a stored HTML artifact. Returns the content string or None."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT content FROM artifacts WHERE task_id = ? AND artifact_type = ?",
            (task_id, artifact_type),
        ).fetchone()
        return row["content"] if row else None
    finally:
        conn.close()


# ── File Blobs (uploaded data files) ──────────────────────────────────


def save_file_blob(conversation_id, filename, content_bytes):
    """Store (or replace) the binary content of an uploaded file."""
    now = _now_iso()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO file_blobs (conversation_id, filename, content, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(conversation_id, filename) DO UPDATE SET "
            "  content = excluded.content, "
            "  created_at = excluded.created_at",
            (conversation_id, filename, content_bytes, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_file_blob(conversation_id, filename):
    """Retrieve stored file content. Returns bytes or None."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT content FROM file_blobs WHERE conversation_id = ? AND filename = ?",
            (conversation_id, filename),
        ).fetchone()
        return bytes(row["content"]) if row else None
    finally:
        conn.close()


def restore_file_to_disk(conversation_id, filename, target_dir):
    """Restore a file from the DB to disk. Returns the file path or None."""
    content = get_file_blob(conversation_id, filename)
    if content is None:
        return None
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    with open(target_path, 'wb') as f:
        f.write(content)
    return target_path
