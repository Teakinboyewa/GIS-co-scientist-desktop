"""
Keep the outputs of failed code attempts out of the results.

A script that crashes halfway has often already written some tables and
figures. Those files stayed in the task folder, the figure catalog picked them
up (manifest entries and the tables/ and figures/ scans), and the manuscript
reported them -- results of code that never finished, sometimes from a
different model fit than the attempt that later succeeded (run dd5922b8:
Moran's I of residuals from the crashed first attempt, Table 6).

On failure, every file the attempt newly created is moved to
<output_dir>/_failed_attempts/<attempt>/, keeping its relative path, with the
error alongside. Nothing is deleted. The catalog drops manifest entries whose
file is gone and does not scan that folder, so the files leave the manuscript
but stay on disk for inspection.

Files that existed before the attempt are never moved, even if it modified
them: they may be an earlier step's good output, and moving them would lose it.
"""

import os
import shutil
import time
import uuid

FAILED_DIR = "_failed_attempts"
# Bookkeeping that belongs to the run, not to the attempt.
_KEEP = {"manifest.json", "workflow_state.json"}
_KEEP_DIRS = {"generated_code", FAILED_DIR}


def quarantine_new_files(output_dir, before, files, error=""):
    """Move files the failed attempt created (in `files`, absent from `before`)
    out of output_dir's results. Returns the new paths."""
    if not output_dir or not files:
        return []
    root = os.path.abspath(output_dir)
    before = {os.path.abspath(p) for p in (before or ())}
    attempt = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dest_root = os.path.join(root, FAILED_DIR, attempt)
    moved = []
    for path in files:
        src = os.path.abspath(path)
        if src in before or not os.path.isfile(src):
            continue
        rel = os.path.relpath(src, root)
        if rel.startswith(".."):
            continue
        parts = rel.replace("\\", "/").split("/")
        if parts[0] in _KEEP_DIRS or os.path.basename(src) in _KEEP:
            continue
        dest = os.path.join(dest_root, rel)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)
            moved.append(dest)
        except OSError:
            continue  # a locked file stays put rather than failing the step
    if moved:
        try:
            with open(os.path.join(dest_root, "ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(str(error) or "(no error message)")
        except OSError:
            pass
    return moved
