"""
Export a conversation to a single .zip file and import it on another computer.

The desktop app's server only listens on 127.0.0.1, so a Share link cannot
reach anyone else. A bundle carries everything the conversation needs to
reopen elsewhere:

    manifest.json          format/version, ids, the sender's app folder
    conversation.json      conversation row, messages, file records,
                           artifacts, and an index of stored file blobs
    blobs/<n>.bin          stored upload copies not already under uploads/
    uploads/...            WebUI/uploads/<conversation_id>/
    outputs/<task_id>/...  Outputs/<task_id>/ (workflow state, maps, tables)

Import keeps the conversation id. Workflows use the conversation id as the
task id (Outputs/<conversation-id>/...), and message metadata and saved
workflow state refer to it, so a new id would break every output link. The
importer becomes the owner. Absolute paths that point into the sender's app
folder are rewritten to the recipient's.

API keys are removed from the bundle (see export_redaction.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone

from WebUI import conversation_db as cdb
from WebUI.export_redaction import TEXT_EXTS, Redactor, collect_secrets

BUNDLE_FORMAT = "gis-coscientist-conversation"
BUNDLE_VERSION = 1

# Already-compressed formats: storing them is faster and the zip is no larger.
_STORED_EXTS = {
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".mp4", ".kmz", ".docx", ".xlsx", ".pptx", ".parquet",
    ".pdf",
}
# Text files whose absolute paths are rewritten on import.
_REWRITE_EXTS = {
    ".json", ".md", ".txt", ".py", ".html", ".htm", ".log", ".yaml", ".yml",
    ".toml", ".xml", ".r", ".ipynb", ".sql", ".qmd",
}
_REWRITE_MAX_BYTES = 64 * 1024 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BundleError(Exception):
    """The file is not a usable conversation bundle."""


class BundleConflict(Exception):
    """The conversation already exists on this computer."""

    def __init__(self, message, conversation_id, deleted=False):
        super().__init__(message)
        self.conversation_id = conversation_id
        self.deleted = deleted


def default_app_root():
    from utils.config import load_configurations
    return load_configurations().root_dir


def _outputs_dir(app_root):
    return os.path.join(app_root, "Outputs")


def _uploads_dir(app_root, conversation_id):
    return os.path.join(app_root, "WebUI", "uploads", conversation_id)


def _task_ids(convo):
    ids = [convo["id"]]
    if convo.get("task_id") and convo["task_id"] not in ids:
        ids.append(convo["task_id"])
    return ids


def _rows(conn, sql, args):
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_tree(zf, src_dir, arc_prefix, redactor, known_only=False):
    """Add every file under src_dir to the zip at arc_prefix/<relative path>.
    Returns (file count, total bytes)."""
    count = size = 0
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, src_dir).replace(os.sep, "/")
            ext = os.path.splitext(name)[1].lower()
            compress = zipfile.ZIP_STORED if ext in _STORED_EXTS else zipfile.ZIP_DEFLATED
            if ext in TEXT_EXTS:
                with open(path, "rb") as f:
                    content = redactor.file(name, f.read(), known_only)
                zf.writestr(f"{arc_prefix}/{rel}", content, compress_type=compress)
            else:
                zf.write(path, f"{arc_prefix}/{rel}", compress_type=compress)
            count += 1
            size += os.path.getsize(path)
    return count, size


# ── Export ────────────────────────────────────────────────────────────


def export_conversation(conversation_id, dest_path, app_root=None, saved_keys=None):
    """Write the conversation's bundle to dest_path. Returns a summary dict.

    saved_keys: the API keys saved in Settings, removed wherever they appear.
    """
    app_root = os.path.abspath(app_root or default_app_root())
    conn = cdb._get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise BundleError("Conversation not found")
        convo = dict(row)
        task_ids = _task_ids(convo)
        messages = _rows(
            conn,
            "SELECT role, content, message_type, metadata_json, timestamp "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        files = _rows(
            conn,
            "SELECT * FROM conversation_files WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        marks = ",".join("?" * len(task_ids))
        artifacts = _rows(
            conn,
            f"SELECT task_id, artifact_type, content, created_at FROM artifacts "
            f"WHERE task_id IN ({marks})",
            task_ids,
        )
        blob_rows = conn.execute(
            "SELECT filename, content, created_at FROM file_blobs WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()

    for f in files:
        f.pop("id", None)
        f.pop("conversation_id", None)

    redactor = Redactor(collect_secrets(
        app_root, [os.path.join(_outputs_dir(app_root), t) for t in task_ids], saved_keys))
    uploads = _uploads_dir(app_root, conversation_id)
    blobs = []
    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # Stored blobs are usually byte-for-byte copies of the uploads folder;
        # only carry the ones the folder doesn't already hold.
        for i, b in enumerate(blob_rows):
            content = bytes(b["content"])
            on_disk = os.path.join(uploads, b["filename"])
            same = (
                os.path.isfile(on_disk)
                and os.path.getsize(on_disk) == len(content)
                and _sha256_file(on_disk) == hashlib.sha256(content).hexdigest()
            )
            entry = {"filename": b["filename"], "created_at": b["created_at"]}
            if same:
                entry["source"] = "uploads"
            else:
                entry["source"] = f"blobs/{i}.bin"
                zf.writestr(entry["source"], redactor.file(b["filename"], content, known_only=True))
            blobs.append(entry)

        upload_count, upload_bytes = (0, 0)
        if os.path.isdir(uploads):
            upload_count, upload_bytes = _add_tree(zf, uploads, "uploads", redactor, known_only=True)

        output_count = output_bytes = 0
        included_tasks = []
        for tid in task_ids:
            tdir = os.path.join(_outputs_dir(app_root), tid)
            if os.path.isdir(tdir):
                c, s = _add_tree(zf, tdir, f"outputs/{tid}", redactor)
                output_count += c
                output_bytes += s
                included_tasks.append(tid)

        conversation = {k: convo.get(k) for k in
                        ("id", "title", "created_at", "updated_at", "task_id", "mode")}
        zf.writestr("conversation.json", json.dumps({
            "conversation": redactor.value(conversation),
            "messages": redactor.value(messages),
            "files": redactor.value(files),
            "artifacts": redactor.value(artifacts),
            "blobs": blobs,
        }, ensure_ascii=False, default=str))
        manifest = {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "conversation_id": conversation_id,
            "title": redactor.text(convo.get("title")),
            "task_ids": included_tasks,
            "app_root": app_root,
            "counts": {
                "messages": len(messages),
                "files": len(files),
                "upload_files": upload_count,
                "output_files": output_count,
            },
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return {
        "conversation_id": conversation_id,
        "title": convo.get("title"),
        "messages": len(messages),
        "upload_files": upload_count,
        "output_files": output_count,
        "data_bytes": upload_bytes + output_bytes,
        "bundle_bytes": os.path.getsize(dest_path),
    }


# ── Import ────────────────────────────────────────────────────────────


def _path_variants(root):
    """The spellings of an absolute folder path that appear in saved text:
    as written, JSON-escaped, and with forward slashes."""
    root = root.rstrip("\\/")
    variants = {
        root,
        root.replace("\\", "\\\\"),
        root.replace("\\", "/"),
        root.replace("/", "\\"),
        root.replace("/", "\\\\"),
    }
    return variants


def make_path_rewriter(old_root, new_root):
    """Return f(text) -> text that points the sender's app folder at ours,
    or None when nothing needs rewriting."""
    if not old_root or not new_root:
        return None
    old_root = old_root.rstrip("\\/")
    new_root = new_root.rstrip("\\/")
    if os.path.normcase(old_root) == os.path.normcase(new_root):
        return None
    pairs = []
    for old in _path_variants(old_root):
        # Write the new root in the same style as the match it replaces.
        if "\\\\" in old:
            new = new_root.replace("/", "\\").replace("\\", "\\\\")
        elif "\\" in old:
            new = new_root.replace("/", "\\")
        else:
            new = new_root.replace("\\", "/")
        pairs.append((old, new))
    # Longest first, so a JSON-escaped root is never half-replaced.
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    windows = bool(re.match(r"^[A-Za-z]:", old_root))
    flags = re.IGNORECASE if windows else 0
    # Only whole folder names: "...\\Desk" must not match "...\\Desk2".
    pattern = re.compile(
        "|".join(f"({re.escape(o)})(?=[\\\\/\"'\\s,;)\\]}}]|$)" for o, _ in pairs),
        flags,
    )
    replacements = [n for _, n in pairs]

    def rewrite(text):
        if not text:
            return text

        def sub(m):
            return replacements[m.lastindex - 1]

        return pattern.sub(sub, text)

    return rewrite


def _check_member(name):
    """Reject zip entries that would land outside the target folders."""
    if not name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        raise BundleError(f"Unsafe path in bundle: {name}")
    parts = name.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise BundleError(f"Unsafe path in bundle: {name}")
    return parts


def _extract_to(zf, info, dest_dir, rel_parts):
    target = os.path.abspath(os.path.join(dest_dir, *rel_parts))
    base = os.path.abspath(dest_dir)
    if not target.startswith(base + os.sep):
        raise BundleError(f"Unsafe path in bundle: {info.filename}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with zf.open(info) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    return target


def _rewrite_file(path, rewrite):
    if os.path.splitext(path)[1].lower() not in _REWRITE_EXTS:
        return
    if os.path.getsize(path) > _REWRITE_MAX_BYTES:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (UnicodeDecodeError, OSError):
        return
    new = rewrite(text)
    if new != text:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)


def read_manifest(zf):
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except KeyError:
        raise BundleError("Not a GIS Co-Scientist conversation file (no manifest.json)")
    except ValueError:
        raise BundleError("The conversation file's manifest is damaged")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError("Not a GIS Co-Scientist conversation file")
    if int(manifest.get("version") or 0) > BUNDLE_VERSION:
        raise BundleError(
            "This conversation was exported by a newer version of the app. "
            "Update GIS Co-Scientist to import it."
        )
    return manifest


def import_conversation(bundle_path, user_id, app_root=None):
    """Recreate a conversation from a bundle, owned by user_id.

    Returns the conversation dict. Raises BundleError for an unusable file and
    BundleConflict when the conversation id is already in this app.
    """
    if not user_id:
        raise BundleError("An API key is required to import a conversation")
    app_root = os.path.abspath(app_root or default_app_root())
    try:
        zf = zipfile.ZipFile(bundle_path)
    except zipfile.BadZipFile:
        raise BundleError("The file is not a zip file")

    created_dirs = []
    with zf:
        manifest = read_manifest(zf)
        try:
            data = json.loads(zf.read("conversation.json"))
        except (KeyError, ValueError):
            raise BundleError("The conversation file is missing its conversation data")
        convo = data.get("conversation") or {}
        conv_id = convo.get("id")
        if not conv_id or not _SAFE_ID.match(conv_id) or conv_id != manifest.get("conversation_id"):
            raise BundleError("The conversation file has an invalid conversation id")
        task_ids = [t for t in manifest.get("task_ids") or [] if t]
        for tid in task_ids:
            if not _SAFE_ID.match(tid):
                raise BundleError("The conversation file has an invalid task id")

        # A conversation still in use is refused. A deleted one that hasn't
        # been cleaned up yet (deletes before conversation_cleanup existed)
        # is removed first, with its folders, so nothing of it is left over.
        existing = cdb.get_conversation(conv_id)
        if existing:
            if not existing.get("deleted_at"):
                raise BundleConflict("This conversation is already in this app.", conv_id)
            from WebUI.conversation_cleanup import purge_conversation
            purge_conversation(conv_id, app_root=app_root)

        rewrite = make_path_rewriter(manifest.get("app_root"), app_root)
        uploads = _uploads_dir(app_root, conv_id)
        outputs = _outputs_dir(app_root)

        # Pass 1: validate every entry before writing anything.
        plan = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = _check_member(info.filename)
            if parts[0] == "uploads" and len(parts) > 1:
                plan.append((info, uploads, parts[1:]))
            elif parts[0] == "outputs" and len(parts) > 2:
                if parts[1] not in task_ids:
                    raise BundleError(f"Unexpected output folder in bundle: {parts[1]}")
                plan.append((info, os.path.join(outputs, parts[1]), parts[2:]))

        try:
            for d in [uploads] + [os.path.join(outputs, t) for t in task_ids]:
                if not os.path.exists(d):
                    created_dirs.append(d)
            for info, dest_dir, rel in plan:
                path = _extract_to(zf, info, dest_dir, rel)
                if rewrite and dest_dir != uploads:
                    _rewrite_file(path, rewrite)

            blobs = []
            for b in data.get("blobs") or []:
                fname = b.get("filename")
                if not fname:
                    continue
                if b.get("source") == "uploads":
                    p = os.path.join(uploads, fname)
                    if not os.path.isfile(p):
                        continue
                    with open(p, "rb") as f:
                        content = f.read()
                else:
                    src = b.get("source") or ""
                    _check_member(src)
                    if not src.startswith("blobs/"):
                        continue
                    content = zf.read(src)
                blobs.append((fname, content, b.get("created_at")))

            _insert_rows(conv_id, convo, data, blobs, user_id, rewrite)
        except Exception:
            for d in created_dirs:
                shutil.rmtree(d, ignore_errors=True)
            raise

    return cdb.get_conversation(conv_id)


def _insert_rows(conv_id, convo, data, blobs, user_id, rewrite):
    rw = rewrite or (lambda t: t)
    now = datetime.now(timezone.utc).isoformat()
    conn = cdb._get_connection()
    try:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, task_id, user_id, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conv_id, convo.get("title") or "Imported conversation",
             convo.get("created_at") or now, now,
             convo.get("task_id"), user_id, convo.get("mode")),
        )
        conn.executemany(
            "INSERT INTO messages (conversation_id, role, content, message_type, metadata_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(conv_id, m.get("role") or "assistant", rw(m.get("content") or ""),
              m.get("message_type") or "text", rw(m.get("metadata_json")),
              m.get("timestamp") or now)
             for m in data.get("messages") or []],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO conversation_files "
            "(conversation_id, filename, display_name, file_size, file_type, audit_json, added_at, removed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(conv_id, f["filename"], f.get("display_name") or f["filename"],
              f.get("file_size") or 0, f.get("file_type"), rw(f.get("audit_json")),
              f.get("added_at") or now, f.get("removed_at"))
             for f in data.get("files") or [] if f.get("filename")],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO file_blobs (conversation_id, filename, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            [(conv_id, name, content, created or now) for name, content, created in blobs],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO artifacts (task_id, artifact_type, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            [(a["task_id"], a["artifact_type"], rw(a.get("content") or ""),
              a.get("created_at") or now)
             for a in data.get("artifacts") or [] if a.get("task_id") and a.get("artifact_type")],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
