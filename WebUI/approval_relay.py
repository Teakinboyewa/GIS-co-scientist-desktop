"""
Cross-process approval relay.

On PythonAnywhere the web app runs as a POOL of worker processes. The
workflow pauses (source-select, data-download, EDA, code-review, …) used to
hand the user's answer back through an in-memory ``queue.Queue`` stored in a
module global — which only works if the approval POST happens to land in the
SAME process as the blocked workflow thread. Across processes the answer was
silently dropped and the workflow blocked forever.

This module passes approvals through the shared filesystem instead, keyed by
(task_id, channel). Any worker can write the answer; the worker running the
paused workflow polls for it. Keyed by task_id, so concurrent users never
collide. Writes are atomic (temp file + os.replace) so the reader never sees a
half-written file.
"""
import json
import os
import tempfile
import time


def _base_dir():
    from utils.config import load_configurations
    d = os.path.join(load_configurations().output_dir, "_approvals")
    os.makedirs(d, exist_ok=True)
    return d


def _path(task_id, channel):
    safe_task = str(task_id or "default").replace(os.sep, "_").replace("/", "_")
    safe_chan = str(channel).replace(os.sep, "_").replace("/", "_")
    return os.path.join(_base_dir(), f"{safe_task}__{safe_chan}.json")


def clear_approval(task_id, channel):
    """Remove any stale answer for this (task_id, channel) before a new wait."""
    try:
        os.remove(_path(task_id, channel))
    except FileNotFoundError:
        pass
    except Exception:
        pass


def write_approval(task_id, channel, response):
    """Persist the user's answer so the waiting workflow (in any worker) sees it."""
    p = _path(task_id, channel)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(response, f)
        os.replace(tmp, p)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def wait_for_approval(task_id, channel, timeout=1800, poll=1.0, cancel_check=None):
    """Block until an answer is written for (task_id, channel), then return it.

    Reads and deletes the answer file on success. Returns ``None`` if the
    cancel_check fires or the timeout elapses (caller treats None as abort).
    """
    p = _path(task_id, channel)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cancel_check is not None:
            try:
                if cancel_check():
                    return None
            except Exception:
                pass
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
            except Exception:
                # Half-written or transient read error — retry next tick.
                time.sleep(poll)
                continue
            try:
                os.remove(p)
            except Exception:
                pass
            return data
        time.sleep(poll)
    return None
