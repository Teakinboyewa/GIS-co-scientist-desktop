"""
AGM WebUI Workflow Handler
Orchestrates the full research workflow for the WebUI:
  1. Data Auditing & Summarization
  2. Research question breakdown
  3. Research Plan Generation
  4. Objective Execution
  5. Result Analysis Pipeline
  6. Manuscript Generation Pipeline
"""

import sys
import os
import json
import queue
import re
import threading
import time
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning


def _with_flask_ctx(fn):
    """Preserve Flask request+app context and token accounting across a spawn.

    Background threads started from inside a Flask request lose access to
    ``flask.g`` (and thus ``g.api_key``). Wrapping the target with
    ``copy_current_request_context`` keeps both the request and app context
    alive for the duration of the thread, so downstream code — including
    deeply-nested agent calls like the modeling agent's inline debugger —
    can still resolve ``g.api_key`` via ``helper.load_OpenAI_key()``.

    The usage ledger is thread-local for the same reason ``g`` is — two
    sessions running concurrently must never pool their tokens — which means a
    ledger opened on the request thread is invisible here. Nearly every LLM
    call in these workflows is made on a thread spawned through this wrapper,
    so without adopting the ledger a whole run reports zero tokens while
    plainly having cost money. Binding it at spawn time (rather than inside the
    thread) is what makes the association correct: the spawning thread is the
    one that knows which run this work belongs to.

    Falls back to the bare function if no request context is active at spawn
    time (e.g. thread spawned from another thread, notebook, or CLI).
    """
    fn = _with_usage_ledger(fn)
    try:
        from flask import copy_current_request_context, has_request_context
        if not has_request_context():
            return fn
        return copy_current_request_context(fn)
    except Exception:
        return fn


def _with_usage_ledger(fn):
    """Bind the ledger active at spawn time to the thread that runs *fn*."""
    try:
        from agents.data_agent import usage_ledger
    except Exception:
        return fn
    ledger = usage_ledger.active()
    if ledger is None:
        return fn

    import functools

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with usage_ledger.adopt(ledger):
            return fn(*args, **kwargs)

    return _wrapped

from agents.data_agent import DataAgent
from agents.rq_understanding_agent import RQUnderstandingAgent
from agents.modeling_agent import ModelingAgent
from agents.task_manager import TaskManager, research_plan_generator, geoprocessing_workflow_generator
import agents.task_manager.objective_executor as objective_executor
from agents.result_analysis_agent import build_result_analysis_graph, generate_execution_flowchart
from agents.result_analysis_agent.result_analysis_graph import build_update_result_analysis_graph
from agents.reporting_agent import build_manuscript_pipeline, build_update_manuscript_pipeline
import utils.research_plan_viz as research_plan_viz
from utils.config import load_configurations, resolve_output_dir

# ===== Agent Instances =====
task_manager = TaskManager()
data_agent = DataAgent()
rq_agent = RQUnderstandingAgent()
modeling_agent = ModelingAgent()

# Live in-memory ctx store for the Task Breakdown (rq_breakdown) pause point.
# Mirrors research_plan_generator.store_context — lets the proceed/revise
# endpoints retrieve the live ctx (including unserializable objects like
# _cancel_event) instead of round-tripping through the on-disk snapshot.
_rq_breakdown_contexts = {}

def _rq_breakdown_store_context(task_id, ctx):
    _rq_breakdown_contexts[task_id] = ctx

def _rq_breakdown_get_context(task_id):
    return _rq_breakdown_contexts.get(task_id)


def _is_current_run(ctx, task_id, saved_state=None):
    """False when *ctx* is an in-memory leftover from an earlier turn.

    ctx caches are per process and keyed by conversation id. On multi-worker
    hosting (PythonAnywhere) a Proceed/Approve click can reach a worker that
    still holds the ctx of an earlier turn in the same conversation. Resuming
    from it re-ran that turn's work: a Spatial Analysis turn whose breakdown
    asked for nothing re-downloaded the data an earlier Data Retriever turn had
    requested. The saved state on disk is written by whichever worker handled
    the latest turn, so its run_id is the current one.
    """
    if saved_state is None:
        saved_state = get_workflow_state(task_id) or {}
    saved_ctx = saved_state.get("ctx") if isinstance(saved_state, dict) else None
    saved_id = saved_ctx.get("run_id") if isinstance(saved_ctx, dict) else None
    return not saved_id or ctx.get("run_id") == saved_id


def _plan_context(task_id):
    """research_plan_generator's cached ctx, unless it belongs to an earlier turn
    (then None, so the caller restores the current turn from disk)."""
    ctx = research_plan_generator.get_context(task_id)
    if ctx is not None and not _is_current_run(ctx, task_id):
        print(f"[State] Ignoring cached ctx for {task_id}: it is from an earlier turn")
        return None
    return ctx

# ===== Configuration =====
settings = load_configurations()
model = None                  # set from UI via smart_message_handler_stream
chat_model = None             # set from UI via smart_message_handler_stream
reasoning_effort_level = None # set from UI via smart_message_handler_stream

# ── Global cancellation flag ──────────────────────────────────────
# Set this event to signal all running background threads to stop.
_cancel_event = threading.Event()


def request_cancellation():
    """Signal all running workflow threads to stop."""
    _cancel_event.set()
    print("[Workflow] Cancellation requested")


def reset_cancellation():
    """Clear the cancellation flag (call before starting a new workflow)."""
    _cancel_event.clear()
    print("[Workflow] Cancellation flag reset — ready for new workflow")


def full_cleanup():
    """Comprehensive cleanup of all global workflow state.
    Call this when cancelling to ensure the next workflow starts fresh."""
    reset_cancellation()
    # Clear all pending code review queues
    _code_review_pending.clear()
    _code_review_responses.clear()
    clear_eda_response_queue()
    print("[Workflow] Full cleanup complete")


def is_cancelled():
    """Check if cancellation has been requested."""
    return _cancel_event.is_set()


# ── Per-task cancellation & rerun registry ───────────────────────
# Scoping cancel events by task_id so concurrent reruns don't poison
# each other's generators via a single shared Event. The legacy global
# `_cancel_event` above is kept for callers that haven't been migrated.
_task_cancel_events: dict = {}          # task_id → threading.Event
_active_reruns: set = set()             # task_ids with an in-flight rerun
_rerun_registry_lock = threading.Lock()


def get_task_cancel_event(task_id):
    """Return (creating if needed) a per-task cancel Event."""
    with _rerun_registry_lock:
        ev = _task_cancel_events.get(task_id)
        if ev is None:
            ev = threading.Event()
            _task_cancel_events[task_id] = ev
        return ev


def try_acquire_rerun_slot(task_id):
    """Reserve a rerun slot for this task. Returns False if one is already active."""
    with _rerun_registry_lock:
        if task_id in _active_reruns:
            return False
        _active_reruns.add(task_id)
        return True


def release_rerun_slot(task_id):
    with _rerun_registry_lock:
        _active_reruns.discard(task_id)


def request_task_cancellation(task_id):
    ev = _task_cancel_events.get(task_id)
    if ev:
        ev.set()


# ── Interactive code review queues ────────────────────────────────
# When interactive mode is ON, the executor thread puts generated code
# on _code_review_pending, the generator yields it to the frontend,
# and the frontend's response comes back via _code_review_responses.
_code_review_pending = {}   # task_id → queue.Queue (executor → generator)
_code_review_responses = {} # task_id → queue.Queue (frontend → executor)


def get_code_review_queues(task_id):
    """Get or create the pair of code review queues for a task."""
    if task_id not in _code_review_pending:
        _code_review_pending[task_id] = queue.Queue()
        _code_review_responses[task_id] = queue.Queue()
    return _code_review_pending[task_id], _code_review_responses[task_id]


def submit_code_review(task_id, response):
    """Called by the /api/workflow/code-review endpoint to relay user decision."""
    q = _code_review_responses.get(task_id)
    if q:
        q.put(response)


def cleanup_code_review_queues(task_id):
    """Remove queues after workflow completes."""
    _code_review_pending.pop(task_id, None)
    _code_review_responses.pop(task_id, None)


# ── Execution checkpoint queues ──────────────────────────────────
# After all objectives execute, the workflow pauses to show a summary
# and waits for the user to proceed or request a status update.
_checkpoint_responses = {}  # task_id → queue.Queue

def get_checkpoint_queue(task_id):
    """Get or create the checkpoint response queue for a task."""
    if task_id not in _checkpoint_responses:
        _checkpoint_responses[task_id] = queue.Queue()
    return _checkpoint_responses[task_id]

def submit_checkpoint_response(task_id, response):
    """Called by the API endpoint to relay user's checkpoint decision.
    Returns True if a live queue was found (stream is active), False otherwise."""
    q = _checkpoint_responses.get(task_id)
    if q:
        q.put(response)
        return True
    return False

def cleanup_checkpoint_queue(task_id):
    """Remove checkpoint queue after use."""
    _checkpoint_responses.pop(task_id, None)


# ── EDA approval queue ───────────────────────────────────────────
# When the EDA agent is unsure about a column choice or preprocessing
# decision, it puts a request on this queue. The streaming loop yields
# it to the frontend, and the user's response comes back via the
# /api/eda-approval endpoint.
_eda_response_queue = None  # Set by the streaming loop during EDA


def set_eda_response_queue(q):
    """Set the active EDA response queue (called by the streaming loop)."""
    global _eda_response_queue
    _eda_response_queue = q


def submit_eda_approval(response):
    """Called by /api/eda-approval endpoint to relay user's choice."""
    if _eda_response_queue:
        _eda_response_queue.put(response)


def clear_eda_response_queue():
    """Clear the EDA response queue reference."""
    global _eda_response_queue
    _eda_response_queue = None


# ── Data download approval queue ──────────────────────────────────
# When one or more data_download_req_X fail and return no files, the
# workflow pauses and surfaces the failed request(s) to the user. The
# user can upload missing files, proceed with whatever data is available,
# or abort the workflow. Responses arrive via /api/data-download-approval.
_data_download_response_queue = None


def set_data_download_response_queue(q):
    """Set the active data-download response queue (called by the streaming loop)."""
    global _data_download_response_queue
    _data_download_response_queue = q


def submit_data_download_approval(response):
    """Called by /api/data-download-approval endpoint to relay user's choice."""
    if _data_download_response_queue:
        _data_download_response_queue.put(response)


def clear_data_download_response_queue():
    """Clear the data-download response queue reference."""
    global _data_download_response_queue
    _data_download_response_queue = None


# ── Pending-approval marker (is anybody still waiting on this prompt?) ──
# A workflow parked on a human decision is invisible from outside the worker
# thread holding it. The SSE stream that drew the prompt may be long gone --
# a reload or a closed tab ends it while the thread below carries on waiting
# -- so a reopened page cannot tell a decision that was already answered from
# one that is still blocking a live run. It has to tell them apart, because
# the two want opposite UI: a frozen record of what was decided, versus a
# panel whose buttons still work.
#
# The waiting thread therefore leaves a marker on disk and refreshes its
# heartbeat while it polls. Disk rather than memory for the same reason the
# answers travel through disk (see below): on multi-worker hosting the page
# asking the question is rarely served by the process holding the prompt.
# The heartbeat is what stops a killed process from trapping a page on a
# prompt nobody is listening for any more -- once it goes quiet the marker
# is stale, and the UI says so instead of offering buttons that would write
# an answer into the void.

PENDING_PROMPT_STALE_AFTER = 30.0   # seconds without a heartbeat -> abandoned
_PENDING_HEARTBEAT_INTERVAL = 5.0   # seconds between marker rewrites
_pending_heartbeat_at = {}          # task_id -> monotonic time of last write


def _pending_prompt_path(task_id):
    return os.path.join(_get_state_dir(task_id), "pending_approval.json")


def write_pending_prompt(task_id, kind, payload=None):
    """Advertise that a workflow thread has parked on *kind* and is waiting."""
    if not task_id:
        return
    now = time.time()
    record = {
        "kind": kind,
        "task_id": task_id,
        "step": (payload or {}).get("step", ""),
        "started_at": now,
        "heartbeat": now,
    }
    try:
        with open(_pending_prompt_path(task_id), "w", encoding="utf-8") as f:
            json.dump(record, f)
        _pending_heartbeat_at[task_id] = time.monotonic()
    except Exception as e:
        print(f"[PendingPrompt] Could not write marker for {task_id}: {e}")


def touch_pending_prompt(task_id):
    """Refresh the heartbeat, at most once per ``_PENDING_HEARTBEAT_INTERVAL``.

    Called from a 0.3s poll loop, so it must not write on every tick."""
    if not task_id:
        return
    now = time.monotonic()
    if now - _pending_heartbeat_at.get(task_id, 0.0) < _PENDING_HEARTBEAT_INTERVAL:
        return
    path = _pending_prompt_path(task_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        return
    record["heartbeat"] = time.time()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f)
        _pending_heartbeat_at[task_id] = now
    except Exception:
        pass


def clear_pending_prompt(task_id):
    """The prompt was answered, cancelled or abandoned -- stop advertising it."""
    if not task_id:
        return
    _pending_heartbeat_at.pop(task_id, None)
    try:
        os.remove(_pending_prompt_path(task_id))
    except FileNotFoundError:
        pass
    except Exception:
        pass


def read_pending_prompt(task_id):
    """The prompt this task is parked on, or None.

    The record carries a ``stale`` flag: True means the heartbeat stopped, so
    the thread that asked the question is gone and no answer can reach it.
    """
    if not task_id:
        return None
    try:
        with open(_pending_prompt_path(task_id), "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        return None
    age = max(0.0, time.time() - float(record.get("heartbeat") or 0))
    record["age_seconds"] = round(age, 1)
    record["stale"] = age > PENDING_PROMPT_STALE_AFTER
    return record


# ── Data source selection response queue ─────────────────────────
# When the DataRetrieverAgent picks a data source that is not in the
# handbook (selected_data_source_ID == "Unknown"), the workflow pauses
# and asks the user to pick from the known sources, retry with LLM,
# or abort the request. Responses arrive via /api/data-source-select.
_source_select_response_queue = None


def set_source_select_response_queue(q):
    global _source_select_response_queue
    _source_select_response_queue = q


def submit_source_select_approval(response):
    """Called by /api/data-source-select endpoint to relay user's choice."""
    if _source_select_response_queue:
        _source_select_response_queue.put(response)


def clear_source_select_response_queue():
    global _source_select_response_queue
    _source_select_response_queue = None


# ── Cross-process source-select hand-off (multi-worker hosting) ────
# On multi-worker deployments (e.g. PythonAnywhere runs several worker
# PROCESSES), the in-memory queue above is invisible across processes:
# the worker serving the streaming workflow and the worker serving the
# user's Proceed POST are usually different processes, so a queue.put in
# one is never seen by the .get in the other and the workflow deadlocks.
# To bridge that, the Proceed POST also writes the answer to a small
# per-task file on disk, and the downloader thread polls for it. Disk is
# shared by all workers, so the hand-off works regardless of which worker
# handled the click.
def _source_select_answer_path(task_id):
    return os.path.join(_get_state_dir(task_id), "source_select_answer.json")


def save_source_select_answer(task_id, answer):
    """Persist the user's source-select answer so the (possibly different)
    worker process running the download stream can pick it up."""
    if not task_id:
        return False
    try:
        with open(_source_select_answer_path(task_id), "w", encoding="utf-8") as f:
            json.dump(answer, f)
        return True
    except Exception as e:
        print(f"[SourceSelect] Could not persist answer for {task_id}: {e}")
        return False


def pop_source_select_answer(task_id):
    """Read and delete the persisted source-select answer, if any."""
    if not task_id:
        return None
    path = _source_select_answer_path(task_id)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                ans = json.load(f)
            try:
                os.remove(path)
            except OSError:
                pass
            return ans
    except Exception as e:
        print(f"[SourceSelect] Could not read answer for {task_id}: {e}")
    return None


def clear_source_select_answer(task_id):
    """Discard any stale answer file before showing a fresh prompt."""
    if not task_id:
        return
    try:
        path = _source_select_answer_path(task_id)
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _drain_source_select_queue(q):
    """Discard any leftover in-memory answers before showing a fresh prompt.

    The Proceed POST delivers the answer on BOTH channels (disk + this
    in-memory queue). When a prior request's answer was consumed via the
    disk path, its queue copy is left behind; without draining it here, the
    next request's first ``.get()`` would instantly return that stale answer
    and auto-confirm a prompt the user never saw. This is the in-memory twin
    of ``clear_source_select_answer`` and must be called alongside it.
    """
    if q is None:
        return
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ── Post-download review hand-off ──────────────────────────────────
# Same dual-channel (in-memory queue + on-disk file) design as the
# source-select hand-off above, and for the same multi-worker reason. This
# one carries the answer to the review shown AFTER each data request
# completes (Proceed / Re-run / Edit code / Feedback / Skip).
_download_review_response_queue = None


def set_download_review_response_queue(q):
    global _download_review_response_queue
    _download_review_response_queue = q


def submit_download_review_response(response):
    """Called by /api/data-download-review to relay the user's choice."""
    if _download_review_response_queue:
        _download_review_response_queue.put(response)


def clear_download_review_response_queue():
    global _download_review_response_queue
    _download_review_response_queue = None


def _download_review_answer_path(task_id):
    return os.path.join(_get_state_dir(task_id), "download_review_answer.json")


def save_download_review_answer(task_id, answer):
    """Persist the review answer so the (possibly different) worker process
    running the download stream can pick it up."""
    if not task_id:
        return False
    try:
        with open(_download_review_answer_path(task_id), "w", encoding="utf-8") as f:
            json.dump(answer, f)
        return True
    except Exception as e:
        print(f"[DownloadReview] Could not persist answer for {task_id}: {e}")
        return False


def pop_download_review_answer(task_id):
    """Read and delete the persisted review answer, if any."""
    if not task_id:
        return None
    path = _download_review_answer_path(task_id)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                ans = json.load(f)
            try:
                os.remove(path)
            except OSError:
                pass
            return ans
    except Exception as e:
        print(f"[DownloadReview] Could not read answer for {task_id}: {e}")
    return None


# ── Post-gate download re-run ────────────────────────────────────────
#
# The review gate is the only moment the workflow thread is parked waiting for
# an answer, so "Re-run" / "Edit code & re-run" / "Feedback & re-run" used to
# be reachable only while that prompt was on screen. Once the user proceeded,
# the code shown on the card was inert.
#
# These entry points make the same three corrections available afterwards, by
# writing down per request what a re-run needs (the request text, the source it
# resolved to, the code it ran) and re-entering the downloader on demand. The
# source is pinned to whatever the request originally resolved to, exactly as
# in the in-gate path -- a correction must never silently swap the data source
# underneath a run, which would invalidate the provenance the run reports.


def _rerun_context_path(task_id):
    return os.path.join(_get_state_dir(task_id), "download_rerun_context.json")


def save_rerun_context(task_id, request_index, context):
    """Record what request *request_index* would need to be re-run later."""
    if not task_id:
        return False
    path = _rerun_context_path(task_id)
    try:
        store = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f) or {}
        store[str(request_index)] = context
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f)
        return True
    except Exception as e:
        print(f"[DownloadRerun] Could not persist context for {task_id}: {e}")
        return False


def get_rerun_context(task_id, request_index):
    """The recorded re-run context for one request, or None."""
    if not task_id:
        return None
    try:
        path = _rerun_context_path(task_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f) or {}
        return store.get(str(request_index))
    except Exception as e:
        print(f"[DownloadRerun] Could not read context for {task_id}: {e}")
        return None


def rerun_data_download_stream(task_id, request_index, action="rerun",
                               code=None, feedback=None):
    """Re-run one already-completed data request. Yields SSE update dicts.

    Mirrors the in-gate re-run in ``agm_data_agent``: same downloader, same
    pinned source, same three actions. What differs is only that no workflow
    thread is waiting -- so the new outcome is streamed straight back to the
    card and the request's stored context is refreshed for a further re-run.
    """
    from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent

    step = f"data_download_req_{request_index}_cont"
    context = get_rerun_context(task_id, request_index)
    if not context:
        yield {
            "type": "error",
            "step": step,
            "error": ("This request's re-run context was not recorded, so it "
                      "cannot be re-run. Only downloads made after this "
                      "feature was added carry it."),
        }
        return

    source_id = context.get("source_id") or ""
    if not source_id:
        yield {
            "type": "error",
            "step": step,
            "error": ("Cannot re-run: the source this request used was never "
                      "recorded, and re-selecting one would change the "
                      "provenance of the run."),
        }
        return

    data_request = context.get("data_request")
    yield {
        "type": "llm_update",
        "step": step,
        "content": {
            "edit_code": "Re-running your edited download code...",
            "feedback": "Regenerating the download code with your feedback...",
        }.get(action, "Re-running the download..."),
    }

    if action == "edit_code" and code:
        # rerun_download only echoes code it generated. The edited program
        # has to reach the stream too: the card shows (and the saved run
        # keeps) the LAST code block as "Generated Code", and without this
        # the block on record would still be the one the user replaced.
        yield {
            "type": "llm_stream",
            "step": step,
            "content": f"\n\n### Your edited download code\n\n```python\n{code.strip()}\n```\n",
        }

    stream_queue = queue.Queue()
    result_holder = [None]
    error_holder = [None]

    def _stream_cb(content):
        stream_queue.put({"type": "llm_stream", "content": content, "step": step})

    def _run():
        try:
            downloader = DataRetrieverAgent(
                api_key=helper.load_OpenAI_key(),
                extra_handbook_dirs=get_user_handbook_dirs(task_id),
            )
            result_holder[0] = downloader.rerun_download(
                data_request=data_request,
                source_ID=source_id,
                source_name=context.get("source_name") or "",
                user_keys=get_data_source_keys(task_id),
                edited_code=code if action == "edit_code" else None,
                feedback=feedback if action == "feedback" else None,
                stream_callback=_stream_cb,
            )
        except Exception as e:
            error_holder[0] = e
        finally:
            stream_queue.put(None)

    thread = threading.Thread(target=_with_flask_ctx(_run), daemon=True)
    thread.start()

    while True:
        if is_cancelled():
            thread.join(timeout=3)
            return
        try:
            item = stream_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item
    thread.join()

    if error_holder[0]:
        yield {"type": "error", "step": step, "error": str(error_holder[0])}
        return

    result = result_holder[0] or {}
    # Carry the source forward: rerun_download does not re-select one, and a
    # further re-run needs to pin the same source again.
    result.setdefault("source_id", source_id)
    result.setdefault("source_name", context.get("source_name") or "")

    save_rerun_context(task_id, request_index, {
        **context,
        "code": result.get("code", context.get("code", "")),
        "status": result.get("status", ""),
    })

    files_event = _rerun_request_files_event(
        task_id, request_index, data_request, result)
    if files_event:
        yield files_event

    yield {
        "type": "data_download_rerun_complete",
        "step": step,
        "task_id": task_id,
        "request_index": request_index,
        "status": result.get("status", "failed"),
        "error": result.get("error", ""),
        "code": result.get("code", ""),
    }


def _rerun_request_files_event(task_id, idx, data_request, result):
    """Tile event for a re-run's new files, matching the in-run shape."""
    geospatial = {"geojson", "json", "gpkg", "shp", "zip", "kml", "gml",
                  "tif", "tiff", "nc", "img", "vrt"}
    tabular = {"csv", "xlsx", "xls", "parquet"}
    try:
        from agents.data_agent.DataRetrieverAgent import classify_delivered_files
        _ds, _ = classify_delivered_files((result or {}).get("downloaded_files") or [])
        dataset_paths = set(_ds)
    except Exception:
        dataset_paths = set((result or {}).get("downloaded_files") or [])
    files_payload = []
    for fpath in (result or {}).get("downloaded_files") or []:
        try:
            base = os.path.basename(fpath)
            ext = os.path.splitext(base)[1].lstrip(".").lower()
            try:
                info = helper._peek_file(fpath) or {}
            except Exception:
                info = {}
            size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            # Same staging the in-run path does: copy into the conversation's
            # uploads dir and register it in the DB. The tiles, the Data
            # Layers panel and the map all read the file back through
            # /api/uploads/<conv>/<file>, so an unstaged re-run either 404s or
            # -- when the script kept the same filename -- serves the OLD copy.
            _stage_downloaded_file_for_ui(
                {"task_id": task_id}, fpath, base, ext, size,
                {**info, "is_dataset": fpath in dataset_paths})
            files_payload.append({
                "isDataset": fpath in dataset_paths,
                "file": info.get("file") or os.path.splitext(base)[0],
                "fileName": base,
                "displayName": os.path.splitext(base)[0],
                "fileType": ext,
                "fileSize": size,
                "path": fpath,
                "shape": info.get("shape", [0, 0]),
                "isGeospatial": ext in geospatial,
                "isTabular": ext in tabular,
                "conversationId": task_id,
            })
        except Exception:
            continue
    return {
        "type": "data_request_files",
        "step": f"data_download_req_{idx}_cont",
        "request": str(data_request),
        "status": "success" if files_payload else "failed",
        "files": files_payload,
        "error": str((result or {}).get("error") or ""),
        "data_unavailable": bool((result or {}).get("data_unavailable")),
    }


def clear_download_review_answer(task_id):
    """Discard any stale review answer before showing a fresh prompt."""
    if not task_id:
        return
    try:
        path = _download_review_answer_path(task_id)
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ── User-provided data-source API keys (collected in the Web UI) ───
# These let users supply their own data-source keys (Census, EPA AQS, etc.)
# instead of relying on the bundled .keys files. Sent with the request from
# localStorage; used to pre-satisfy keys (so the prompt doesn't ask) and to
# run in autonomous mode. Interactive runs can also collect missing keys via
# the source-select confirmation prompt (those arrive in its response).
_data_source_keys_by_task = {}


def set_data_source_keys(task_id, keys):
    if task_id and isinstance(keys, dict):
        _data_source_keys_by_task[task_id] = {k: v for k, v in keys.items() if v}


def get_data_source_keys(task_id):
    return dict(_data_source_keys_by_task.get(task_id) or {})


# Per-task extra handbook directories — a user's private, contributed data
# sources. Set from the Web UI request (which knows the caller's user_id) so the
# DataRetrieverAgent can offer that user's own sources alongside the curated
# global catalog, without exposing them to anyone else.
_user_handbook_dirs_by_task = {}


def set_user_handbook_dirs(task_id, dirs):
    if task_id:
        _user_handbook_dirs_by_task[task_id] = [d for d in (dirs or []) if d]


def get_user_handbook_dirs(task_id):
    return list(_user_handbook_dirs_by_task.get(task_id) or [])


# ── Latest workflow context for ad-hoc EDA explore ─────────────────
_latest_workflow_ctx = None


def set_latest_workflow_ctx(ctx):
    """Store the latest workflow context (called after EDA completes)."""
    global _latest_workflow_ctx
    _latest_workflow_ctx = ctx


def get_latest_workflow_ctx():
    """Return the latest workflow context, or None if not available."""
    return _latest_workflow_ctx


# ── EDA Explore response queue (reuses same pattern as main EDA) ───
_eda_explore_response_queue = None


def set_eda_explore_response_queue(q):
    global _eda_explore_response_queue
    _eda_explore_response_queue = q


def submit_eda_explore_approval(response):
    if _eda_explore_response_queue:
        _eda_explore_response_queue.put(response)


def clear_eda_explore_response_queue():
    global _eda_explore_response_queue
    _eda_explore_response_queue = None


def eda_explore_stream(user_task):
    """Stream an ad-hoc EDA explore task with the same thread+queue pattern as main EDA."""
    ctx = get_latest_workflow_ctx()
    if not ctx:
        yield {"type": "error", "error": "No active workflow context found"}
        return

    explore_queue = queue.Queue()
    approval_queue = queue.Queue()
    response_queue = queue.Queue()
    set_eda_explore_response_queue(response_queue)

    # Wire streaming + approval into a local ctx copy so we don't mutate the original
    explore_ctx = dict(ctx)
    explore_ctx["_stream_callback"] = lambda content: explore_queue.put(
        {"type": "llm_stream", "content": content, "step": "eda_explore"}
    )
    explore_ctx["_approval_queue"] = approval_queue
    explore_ctx["_response_queue"] = response_queue

    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            agent = DataAgent()
            result_holder[0] = agent.eda_explore(explore_ctx, user_task)
        except Exception as e:
            error_holder[0] = e
        finally:
            explore_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_run), daemon=True)
    t.start()

    while True:
        # Check for approval requests
        try:
            approval_item = approval_queue.get_nowait()
            yield approval_item
        except queue.Empty:
            pass

        try:
            item = explore_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    clear_eda_explore_response_queue()

    if error_holder[0]:
        yield {"type": "error", "error": str(error_holder[0])}
        return

    result = result_holder[0] or {}
    if "error" in result:
        yield {"type": "error", "error": result["error"]}
        return

    # Send artifacts + completion
    yield {
        "type": "eda_explore_complete",
        "result": result.get("result", ""),
        "artifacts": result.get("artifacts", []),
    }

# =====================================================================
#  AGENT REGISTRATION
# =====================================================================

def register_agent():
    """Register all agents with the TaskManager."""
    task_manager.register_agent(
        "RQUnderstandingAgent",
        RQUnderstandingAgent(),
        description="Interprets user-defined research questions and translates them into "
                    "operational objectives, variables, hypotheses, study units, spatial extent, "
                    "and time frame.",
    )
    task_manager.register_agent(
        "DataAgent",
        DataAgent(),
        description="Generates data overview by auditing datasets. Validates data inputs, "
                    "harmonizes CRS, resolves keys (GEOID, CBG, tract), generates metadata & "
                    "data dictionary. Performs exploratory data analysis (EDA).",
    )
    task_manager.register_agent(
        "ModelingAgent",
        ModelingAgent(),
        description="Selects, configures, and executes appropriate statistical and machine "
                    "learning models from the system's model base using intelligent model "
                    "selection based on task requirements.",
    )


# =====================================================================
#  WORKFLOW CONTEXT
# =====================================================================

def init_workflow_context(rq, datasets, conversation_id=None):
    """
    Create the workflow context dict, matching the structure used in case2.ipynb.

    Each conversation gets its own output directory under Outputs/<task_id>/
    so that concurrent/past workflows never collide.

    Uses conversation_id as the task_id when available so uploads and outputs
    share the same folder. Falls back to a new UUID if no conversation_id.

    Args:
        rq: Research question string
        datasets: List of dicts with 'path' and 'name' keys
        conversation_id: Optional conversation ID to use as task_id

    Returns:
        ctx dict
    """
    task_id = conversation_id or str(uuid.uuid4())
    task_output_dir = os.path.join(settings.output_dir, task_id)
    os.makedirs(task_output_dir, exist_ok=True)

    return {
        "rq": rq,
        "datasets": datasets,
        "artifacts": {},
        "logs": [],
        "task_id": task_id,
        "output_dir": task_output_dir,
        # One id per turn. The conversation id (task_id) is shared by every turn
        # of a conversation, so it cannot tell a cached ctx from an earlier turn
        # apart from the current one; see _is_current_run.
        "run_id": uuid.uuid4().hex,
    }


# =====================================================================
#  WORKFLOW STATE MANAGEMENT
# =====================================================================

workflow_states = {}


def _get_state_dir(task_id=None):
    """Per-task state directory: Outputs/<task_id>/.

    If called without task_id (e.g., by list_workflow_states), returns the
    root outputs directory so callers can iterate per-task subdirs.
    """
    if task_id:
        state_dir = os.path.join(settings.output_dir, task_id)
    else:
        state_dir = settings.output_dir
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def _get_state_file(task_id):
    """Per-task state file: Outputs/<task_id>/workflow_state.json.

    Falls back to the legacy path Outputs/workflow_states/<task_id>.json for
    reads only (so previously saved states remain accessible). All writes
    go to the new per-task location.
    """
    new_path = os.path.join(_get_state_dir(task_id), "workflow_state.json")
    if not os.path.exists(new_path):
        legacy = os.path.join(settings.output_dir, "workflow_states", f"{task_id}.json")
        if os.path.exists(legacy):
            return legacy
    return new_path


def _build_step_summaries(ctx):
    """Build a list of step summaries from execution results for the checkpoint card."""
    execution_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})

    # Build a lookup of step descriptions from the plan
    step_desc_map = {}  # (obj_key, step_idx) → description
    for obj in structured_plan.get("objectives", []):
        obj_key = f"objective_{obj.get('number', '')}"
        for step in obj.get("steps", []):
            if isinstance(step, dict):
                step_num = step.get("step_number", 0)
                step_desc_map[(obj_key, step_num)] = step.get("description", f"Step {step_num}")

    summaries = []
    for obj_key, steps_list in execution_results.items():
        obj_num = obj_key.replace("objective_", "")
        for step_result in steps_list:
            step_idx = step_result.get("step_index", "?")
            status = step_result.get("status", "unknown")
            desc = step_desc_map.get((obj_key, step_idx), "")
            if not desc:
                desc = step_result.get("step_description", "") or step_result.get("description", f"Step {step_idx}")
            error = step_result.get("error_message") or step_result.get("error", "")
            summaries.append({
                "objective": obj_num,
                "step_index": step_idx,
                "description": desc,
                "status": status,
                "error": error[:200] if error else "",
            })

    return summaries


def _serialize_state(state):
    """Serialize state for JSON storage, handling non-serializable objects."""
    def convert(obj):
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        else:
            return f"<{type(obj).__name__}>"
    return convert(state)


def save_workflow_state(task_id, state):
    """Save workflow state for continuation. Always writes to the per-task
    location Outputs/<task_id>/workflow_state.json (never the legacy path)."""
    workflow_states[task_id] = state
    try:
        state_file = os.path.join(_get_state_dir(task_id), "workflow_state.json")
        serializable_state = _serialize_state(state)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
        print(f"[State] Saved workflow state to {state_file}")
    except Exception as e:
        print(f"[State] Warning: Could not save state to disk: {e}")
    return state


PLAN_APPROVAL_FILE = "plan_approval.json"


def mark_plan_approved(task_id):
    """Record that the user approved the research plan, so a reloaded
    conversation can tell an approved plan from one still awaiting approval."""
    path = os.path.join(_get_state_dir(task_id), PLAN_APPROVAL_FILE)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"approved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def plan_approval_status(task_id):
    """{'approved': bool, 'approved_at': str | None} for a research plan.
    Runs from before the approval record existed count as approved when
    execution had started."""
    path = os.path.join(settings.output_dir, task_id, PLAN_APPROVAL_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {"approved": True, "approved_at": json.load(f).get("approved_at")}
    state = get_workflow_state(task_id) or {}
    artifacts = (state.get("ctx") or {}).get("artifacts") or {}
    started = bool(state.get("execution_results") or artifacts.get("step_execution_results")
                   or state.get("workflow_phase") == "completed")
    return {"approved": started, "approved_at": None}


def get_workflow_state(task_id):
    """Retrieve workflow state, reading the saved file first.

    The file is the only copy every worker process shares. The in-memory dict
    is per process, so on multi-worker hosting (PythonAnywhere) it can hold a
    state that another worker has since replaced; returning it first made a
    Proceed click resume an earlier turn of the conversation. It is used only
    when the file is missing or unreadable.
    """
    try:
        state_file = _get_state_file(task_id)
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            workflow_states[task_id] = state
            return state
    except Exception as e:
        print(f"[State] Warning: Could not load state from disk: {e}")
    return workflow_states.get(task_id)


def clear_workflow_state(task_id):
    """Clear workflow state from both memory and disk (new + legacy paths)."""
    if task_id in workflow_states:
        del workflow_states[task_id]
    candidates = [
        os.path.join(settings.output_dir, task_id, "workflow_state.json"),
        os.path.join(settings.output_dir, "workflow_states", f"{task_id}.json"),
    ]
    for state_file in candidates:
        try:
            if os.path.exists(state_file):
                os.remove(state_file)
        except Exception as e:
            print(f"[State] Warning: Could not delete state file {state_file}: {e}")


def list_workflow_states():
    """List all available workflow state task IDs.

    Scans per-task directories (Outputs/<task_id>/workflow_state.json) and
    also includes any legacy Outputs/workflow_states/<task_id>.json files so
    older workflows remain discoverable.
    """
    task_ids = set()
    try:
        root = settings.output_dir
        if os.path.isdir(root):
            for entry in os.listdir(root):
                entry_path = os.path.join(root, entry)
                if os.path.isdir(entry_path) and os.path.exists(
                    os.path.join(entry_path, "workflow_state.json")
                ):
                    task_ids.add(entry)
    except Exception:
        pass
    try:
        legacy_dir = os.path.join(settings.output_dir, "workflow_states")
        if os.path.isdir(legacy_dir):
            for f in os.listdir(legacy_dir):
                if f.endswith(".json"):
                    task_ids.add(f[:-5])
    except Exception:
        pass
    return sorted(task_ids)

def _wrap_explanation_as_html(explanation_text, research_question=""):
    """Wrap the LLM prose explanation in a styled HTML page for the plan view."""
    import html as html_mod
    safe_rq = html_mod.escape(research_question)
    # Convert basic markdown-style formatting to HTML
    lines = explanation_text.split("\n")
    body_parts = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            body_parts.append(f"<h3>{html_mod.escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            body_parts.append(f"<h2>{html_mod.escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            body_parts.append(f"<h1>{html_mod.escape(stripped[2:])}</h1>")
        elif stripped.startswith("- "):
            body_parts.append(f"<li>{html_mod.escape(stripped[2:])}</li>")
        elif stripped.startswith("**") and stripped.endswith("**"):
            body_parts.append(f"<p><strong>{html_mod.escape(stripped[2:-2])}</strong></p>")
        elif stripped:
            body_parts.append(f"<p>{html_mod.escape(stripped)}</p>")
        else:
            body_parts.append("<br>")
    body_html = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Geoprocessing Workflow Plan</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f9f9f9; color: #333; line-height: 1.6; }}
  .container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; margin-top: 24px; }}
  h3 {{ color: #27ae60; }}
  .rq {{ background: #eaf2f8; padding: 12px 16px; border-left: 4px solid #3498db; margin-bottom: 20px; border-radius: 4px; font-style: italic; }}
  li {{ margin-left: 20px; margin-bottom: 4px; }}
  p {{ margin: 8px 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>Geoprocessing Workflow Plan</h1>
  <div class="rq">{safe_rq}</div>
  {body_html}
</div>
</body>
</html>"""

# =====================================================================
#  CHAT HANDLER
# =====================================================================

# Chat mode used to send the bare message, so "what can I do here?" got a
# generic answer about nothing in particular. This tells the model what the
# app is and what each part does. Keep it in step with AGSI/user_manual.html.
CHAT_SYSTEM_PROMPT = """You are the Chat assistant inside GIS Co-Scientist, a web app from Penn State GIScience. It is a multi-agent system for geospatial research: it retrieves geospatial data, runs spatial analyses, and drafts research manuscripts from one conversational interface, with a human reviewing at each checkpoint.

You answer two kinds of questions:
1. Questions about how to use THIS app ("what can I do here?", "how do I...?", "which mode should I use?"). Answer from the guide below. Be concrete: name the mode or button and give one or two example requests the user could type. Do not invent features, buttons or data sources of this app that are not in the guide; if a question about this app's features is not covered, say so and point to the User Manual (header menu, https://autonomousgis-agsi.github.io/AGSI/user_manual).
2. Everything else: general GIS, GIScience, spatial statistics, data, software, tools, research projects, papers and research-design questions (e.g. "Global vs Local Moran's I?", "What is GIS Copilot?", "how should I frame a study of air quality and rurality?"). Answer these from your own knowledge as a knowledgeable GIScience colleague. The guide is not a limit on what you know: never refuse or deflect a question just because the guide does not mention it. If you are unsure about a specific fact, say what you know and flag the uncertainty.

In Chat mode no workflow runs: you cannot download data, run code or read the user's layers. If the user asks you to do one of those, explain which mode does it and suggest the request to send there. You see the conversation's earlier turns, including the start of any workflow replies (long ones are truncated), so you can refer back to them; you cannot see the data layers or result files themselves.

Keep answers short and scannable. For "what can I do here?", give a brief overview of the modes with an example each, then offer to go deeper on any of them.

=== GUIDE TO GIS CO-SCIENTIST ===

Getting started
- Settings (top right) -> paste a GIBD key (gibd-...) or an OpenAI key (sk-...) -> Save Key. The key stays in the browser.
- Pick a mode in the mode dropdown at the bottom right of the input box, type a request, press Enter (Shift+Enter for a new line).
- Optional: add your own data with the + button (chat toolbar or Data Layers header). Formats: GeoJSON, GeoPackage, shapefile (all parts, or zipped), CSV/XLSX/Parquet tables, GeoTIFF rasters.

Modes
- Data Retriever: download data only, no analysis. Describe the variable, geography level and year; the system picks a source, shows its reasoning and a confidence score, downloads, and adds each dataset as a layer. You can ask for several datasets at once. Example: "Download 2019 county-level PM2.5 data for California and the corresponding county boundary."
- Spatial Analysis: run one specific analysis. The system breaks the task down, checks whether your data is enough, builds a geoprocessing workflow (a graph of input data -> operations -> outputs) for you to approve or revise, then writes and runs the code. Every step shows its code and log and can be edited, given feedback, or re-run. Vector outputs go to Data Layers and the map. Example: "Create a thematic map showing the distribution of cancer prevalence in South Carolina."
- Research: the full pipeline from a research question to a drafted manuscript. Stages: data audit; question understanding and task breakdown (Checkpoint 1); data acquisition with source selection (Checkpoint 2, when Auto Data is on); research plan with objectives, hypotheses and steps, plus its geoprocessing workflow (Checkpoint 3, editable); code generation and execution with automatic debugging (Checkpoint 4, review, edit and re-run steps); result analysis with a figures and tables catalog; manuscript drafting with a Semantic Scholar literature search (Title & Abstract, Introduction, Methods, Results, Discussion, Conclusion, References), downloadable as Markdown, Word or PDF; and an execution flowchart. Steps, the analysis and the manuscript can be re-run later; only the affected manuscript sections are rewritten. Example: "How do spatial inequalities in PM2.5 exposure across California counties vary with income inequality (Gini) in 2019?"
- Research (no planning): an ablation of Research mode with no Planning Agent; the Modeling Agent writes the code directly.
- Single Agent: a baseline where one agent runs the whole research workflow.
- Chat: this mode. Questions and answers, no workflow.

Toggles in the mode dropdown
- Autonomous: off by default. When on, the system approves its own checkpoints and runs end to end. It still stops if a data source needs an API key it lacks.
- Auto Data: on by default, shown for Spatial Analysis and Research. When on, the system downloads data the request needs; when off, it works with the layers you already have. If your data is already sufficient nothing is downloaded.

Built-in data sources
US Census boundaries (nation, state, county, tract, block group, metro); US Census demography (population, income, education, race, etc.; free key); CDC PLACES health estimates (county, tract, city, ZCTA); EPA AQS air quality (PM2.5, ozone, NO2, etc.; email + key); USDA ERS food access, rurality, income, SNAP; USGS earthquakes (global); OpenStreetMap boundaries, streets and POIs (global); OpenTopography DEMs (global; free key); OpenWeather (global; free key); NYT COVID-19 cases and deaths (US state/county, 2020-2023); Natural Earth base-map layers; ESRI World Imagery tiles. Data-source keys go under Settings -> Data Source Keys, or are asked for inline when needed.

Adding data sources
- The Data Sources tab/panel lists My Sources and Shared Sources. "+ Add data source" takes a name, a brief description (scope, area, time period), a handbook of technical notes (endpoints, parameters, {KEY_NAME} placeholders for credentials) and an optional Python code example, or import a .toml handbook. Sources can be submitted for sharing; an admin reviews them.
- Handbook Studio (header) generates and tests a handbook for a new source from its name and/or documentation URL, researching it with web search. It works with a GIBD key or an OpenAI key.

Workspace
- Left sidebar: New chat, saved conversations (tied to your API key), and the Data Sources tab.
- Right panel: Data Layers (each layer's menu: Details, Show Attribute Table, Load on Map Viewer, Preview in Results Panel, Download, Remove Layer), the Map Viewer, and Results & Artifacts (code, tables, statistics, figures, reports).
- Layers persist with the conversation. Result layers are sent with your next message, so follow-ups can build on them (e.g. "100 m buffers around the centroids").
- Each stage card has a Tokens chip showing the tokens its model calls used.
- Sharing: copy the browser URL (?conversation=<id>). Others with their own key join as collaborators; their first change can fork a private copy.

Troubleshooting
- Nothing runs: add a valid API key in Settings.
- Download rejected for a missing key: add the source's key under Settings -> Data Source Keys.
- Wrong dataset: be explicit about year, geography level and variable, or name the source.
- Shapefile won't load: include .shp, .shx, .dbf and .prj, or upload one .zip.
- Conversations missing: they are tied to the API key; use the same key.
- Support: Temitope Akinboyewa (tea5209@psu.edu).
"""


# Earlier turns sent with a chat message. Workflow replies (plans, code, logs)
# can be very long, so each is cut to its start and the oldest turns are
# dropped once the total passes the budget.
CHAT_HISTORY_MAX_TURNS = 20
CHAT_HISTORY_MAX_CHARS_PER_TURN = 2000
CHAT_HISTORY_MAX_CHARS = 24000


def _chat_history(conversation_id, current_message):
    """Earlier turns of the conversation as chat messages, oldest first.

    app.py saves the current user message before the handler runs, so the
    trailing copy of it is dropped here; it is appended by the caller.
    """
    if not conversation_id:
        return []
    try:
        from WebUI.conversation_db import get_messages
        rows = get_messages(conversation_id)
    except Exception as exc:  # history is a nicety; never fail the reply
        print(f"[WARN] chat history unavailable for {conversation_id}: {exc}")
        return []

    if rows and rows[-1].get("role") == "user" and \
            (rows[-1].get("content") or "").strip() == (current_message or "").strip():
        rows = rows[:-1]

    history = []
    for row in rows[-CHAT_HISTORY_MAX_TURNS:]:
        text = (row.get("content") or "").strip()
        if not text:
            continue
        if len(text) > CHAT_HISTORY_MAX_CHARS_PER_TURN:
            text = text[:CHAT_HISTORY_MAX_CHARS_PER_TURN] + "\n[...truncated]"
        role = "user" if row.get("role") == "user" else "assistant"
        history.append({"role": role, "content": text})

    total = sum(len(m["content"]) for m in history)
    while history and total > CHAT_HISTORY_MAX_CHARS:
        total -= len(history.pop(0)["content"])
    return history


def just_chat_response(message, intent_result, stream=True, conversation_id=None):
    """Stream a conversational response, with the conversation's earlier turns."""
    yield from helper.yield_status_update("Generating conversational response...", "chat_response")

    messages = ([{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
                + _chat_history(conversation_id, message)
                + [{"role": "user", "content": message}])
    print(f"[DEBUG] just_chat_response using model: {chat_model}")

    if stream:
        response = helper.client_chat_completion_stream(
            messages=messages,
            model=llm_model("chat.reply"), **llm_reasoning("chat.reply")
            )
    else:
        response = helper.client_chat_completion(
            messages=messages,
            model=llm_model("chat.reply"), **llm_reasoning("chat.reply")
            )
    if stream:
        full_response = ""
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield {"type": "llm_stream", "content": content, "step": "streaming_chat"}

        yield {
            "type": "result",
            "response": full_response,
            "intent": intent_result,
            "result_type": "chat",
        }
    else:
        full_response = response.choices[0].message.content
        yield {
            "type": "result",
            "response": full_response,
            "intent": intent_result,
            "result_type": "chat",
        }



# =====================================================================
#  DATA AUDIT (standalone, non-streaming)
# =====================================================================

def audit_uploaded_data(file_paths, file_names, ctx=None):
    """
    Audit uploaded data files and return audit records.

    Args:
        file_paths: List of file paths
        file_names: List of original file names
        ctx: Optional existing context

    Returns:
        List of audit record dicts
    """
    if ctx is None:
        ctx = {}

    datasets = [{"path": p, "name": n} for p, n in zip(file_paths, file_names)]
    ctx["datasets"] = datasets

    agent = DataAgent()
    result = agent.audit_datasets(ctx, {"params": {}})
    return result.get("audit_records", [])


# =====================================================================
#  MAIN STREAMING ENTRY POINT
# =====================================================================

def smart_message_handler_stream(message, file_paths=None, file_names=None, active_solutions=None, interactive_code=False, enable_eda=False, autonomous_mode=False, data_download_mode=False, intent_mode='research', model=None, reasoning_effort=None, api_key=None, conversation_id=None, spatial_condition=None):
    """
    Main entry point called by app.py.
    Detects intent and routes to chat or workflow handler.

    Yields:
        SSE-compatible update dicts
    """
    # Always reset cancellation at the start of a new request so a previous
    # interruption doesn't block this one.
    reset_cancellation()

    from agents.spatial_reasoning.conditions import resolve
    try:
        condition = resolve(spatial_condition)
    except (ValueError, NotImplementedError) as exc:
        yield {"type": "error", "error": str(exc)}
        return
    if condition["name"] != "B0" and intent_mode != "research":
        yield {"type": "error", "error": "Spatial conditions other than B0 require Research mode."}
        return
    spatial_condition = condition["name"]
    if intent_mode == "research":
        yield {"type": "llm_update", "step": "spatial_condition",
               "content": f"Spatial baseline: {spatial_condition} — {condition['label']}"}

    # Use the model and reasoning effort selected from the UI
    global chat_model, reasoning_effort_level
    chat_model = model
    reasoning_effort_level = reasoning_effort
    # Propagate to shared helper so all agents pick up UI selections
    from utils.agm_helper import set_ui_model
    set_ui_model(model, reasoning_effort)
    print(f"[DEBUG] UI-selected model set: chat_model = {chat_model}")
    print(f"[DEBUG] UI-selected reasoning effort: {reasoning_effort_level}")

    # Propagate the user-supplied API key into os.environ so that background
    # threads (which cannot access Flask's thread-local ``g``) will pick it
    # up via ``load_OpenAI_key()``'s ``os.getenv`` fallback.
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    print(f"\n[DEBUG] smart_message_handler_stream called")
    print(f"[DEBUG] Message: {message}")
    print(f"[DEBUG] Model: {model}")
    print(f"[DEBUG] Files uploaded: {len(file_paths) if file_paths else 0}")
    print(f"[DEBUG] Intent mode: {intent_mode}")

    # --- Intent detection: auto or manual ---
    # Ablation modes are the Research workflow with one component removed.
    ablation = None
    if intent_mode == "research_no_planning":
        ablation = "no_planning"

    if intent_mode in ("chat", "task", "research", "data_retriever", "single_agent",
                       "research_no_planning"):
        # Manual mode — skip LLM intent detection
        intent_result = {"intent": "research" if ablation else intent_mode,
                         "confidence": 1.0, "reasoning": "Manually selected by user"}
        if ablation:
            intent_result["ablation"] = ablation
        yield {
            "type": "llm_update",
            "content": f"Intent mode: {intent_result['intent'].upper()} (manual)",
            "step": "intent_detected",
            "intent": intent_result,
        }
    else:
        # Auto-detect via LLM
        yield {"type": "llm_update", "content": "Analyzing your message...", "step": "intent_detection"}

        # Stream intent detection LLM response via background thread + queue
        intent_queue = queue.Queue()
        intent_error = [None]
        intent_holder = [None]

        def _stream_cb(content):
            intent_queue.put({"type": "llm_stream", "content": content, "step": "intent_detection"})

        def _run_intent():
            try:
                intent_holder[0] = detect_user_intent(message, stream_callback=_stream_cb)
            except Exception as e:
                intent_error[0] = e
            finally:
                intent_queue.put(None)  # sentinel

        t = threading.Thread(target=_with_flask_ctx(_run_intent), daemon=True)
        t.start()

        while True:
            if is_cancelled():
                t.join(timeout=3)
                return
            try:
                item = intent_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()

        if intent_error[0]:
            raise intent_error[0]

        intent_result = intent_holder[0]
        print(f"[DEBUG] Intent detected: {intent_result}")

        yield {
            "type": "llm_update",
            "content": f"Intent detected: {intent_result['intent'].upper()} "
                       f"(confidence: {intent_result['confidence']})",
            "step": "intent_detected",
            "intent": intent_result,
        }

    # --- Chat ---
    if intent_result["intent"] == "chat":
        yield from just_chat_response(message, intent_result, stream=True,
                                      conversation_id=conversation_id)
        return

    # --- Data Retriever: download only, no analysis ---
    if intent_result["intent"] == "data_retriever":
        yield from data_retriever_workflow_handler(
            message, file_paths, file_names,
            autonomous_mode=autonomous_mode,
            conversation_id=conversation_id,
        )
        return

    # --- Single agent: the Research workflow, done by one agent (baseline) ---
    if intent_result["intent"] == "single_agent":
        yield from single_agent_workflow_handler(
            message, file_paths, file_names, conversation_id=conversation_id,
        )
        return

    # --- Research with data ---
    if intent_result["intent"] == "research" and file_paths and len(file_paths) > 0:
        yield from agm_workflow_handler(
            message, intent_result, file_paths, file_names, active_solutions,
            interactive_code=interactive_code,
            enable_eda=enable_eda,
            autonomous_mode=autonomous_mode,
            data_download_mode=data_download_mode,
            conversation_id=conversation_id,
            ablation=ablation,
            spatial_condition=spatial_condition,
        )
        return

    # --- Research without data ---
    if intent_result["intent"] == "research" and (not file_paths or len(file_paths) == 0):

        if data_download_mode:
            yield from agm_workflow_handler(
                message, intent_result, file_paths, file_names, active_solutions,
                interactive_code=interactive_code,
                enable_eda=enable_eda,
                autonomous_mode=autonomous_mode,
                data_download_mode=True,
                conversation_id=conversation_id,
                ablation=ablation,
                spatial_condition=spatial_condition,
            )
            return

        response_msg = (
            f"I understand you want to perform a {intent_result['intent']} but no data files "
            "were uploaded. Please upload your data file(s) or switch to the auto data retriever mode so I can help you with the analysis."
        )
        yield {
            "type": "llm_update",
            "content": f"{intent_result['intent'].upper()} detected but no data files uploaded",
            "step": f"{intent_result['intent']}_no_data",
        }
        yield {
            "type": "result",
            "response": response_msg,
            "intent": intent_result,
            "result_type": f"{intent_result['intent']}_without_data",
        }
        return

    # --- Task with data ---
    if intent_result["intent"] == "task" and file_paths and len(file_paths) > 0:
        yield from spatial_analysis_workflow_handler(
            message, file_paths, file_names,
            interactive_code =interactive_code,
            enable_eda=enable_eda,
            autonomous_mode=autonomous_mode,
            data_download_mode=data_download_mode,
            conversation_id=conversation_id,
        )
        return

    # --- Task without data, but data_download_mode enabled ---
    # Spatial Analysis mode can bootstrap its inputs from LLM_Find when the
    # user hasn't uploaded files but has toggled auto data download.
    if intent_result["intent"] == "task" and (not file_paths or len(file_paths) == 0) and data_download_mode:
        yield from spatial_analysis_workflow_handler(
            message, file_paths, file_names,
            interactive_code=interactive_code,
            enable_eda=enable_eda,
            autonomous_mode=autonomous_mode,
            data_download_mode=True,
            conversation_id=conversation_id,
        )
        return

    # --- Task / Research without data ---
    if intent_result["intent"] in ("task", "research") and (not file_paths or len(file_paths) == 0):
        response_msg = (
            f"I understand you want to perform a {intent_result['intent']} but no data files "
            "were uploaded. Please upload your data file(s) or switch to the auto data retriever mode so I can help you with the analysis."
        )
        yield {
            "type": "llm_update",
            "content": f"{intent_result['intent'].upper()} detected but no data files uploaded",
            "step": f"{intent_result['intent']}_no_data",
        }
        yield {
            "type": "result",
            "response": response_msg,
            "intent": intent_result,
            "result_type": f"{intent_result['intent']}_without_data",
        }
        return



    # --- Fallback ---
    yield {
        "type": "result",
        "response": "I'm not sure how to handle this request. Could you please clarify?",
        "intent": intent_result,
        "result_type": "unknown",
    }


# =====================================================================
#  SINGLE-AGENT WORKFLOW HANDLER (baseline)
# =====================================================================

def single_agent_workflow_handler(message, file_paths=None, file_names=None, conversation_id=None):
    """
    One agent runs the whole Research workflow -- data understanding, question
    breakdown, plan, execution, manuscript -- with the specialist agents'
    prompts and tools but none of the orchestration (see agents/single_agent).
    It never pauses for review, so it runs straight through to the manuscript.

    Yields:
        SSE-compatible update dicts
    """
    from agents.single_agent.single_agent import run_single_agent, finalize_manuscript

    reset_cancellation()
    file_paths = file_paths or []
    file_names = file_names or []

    if not file_paths:
        yield {
            "type": "llm_update",
            "content": "SINGLE AGENT detected but no data files uploaded",
            "step": "single_agent_no_data",
        }
        yield {
            "type": "result",
            "response": ("The single-agent baseline works only with uploaded data. "
                         "Please upload your data file(s) and send the research question again."),
            "result_type": "single_agent_without_data",
        }
        return

    datasets = [{"path": p, "name": n} for p, n in zip(file_paths, file_names)]
    ctx = init_workflow_context(rq=message, datasets=datasets, conversation_id=conversation_id)
    ctx["_cancel_event"] = _cancel_event
    task_id = ctx["task_id"]

    # The same starting registry the Research workflow builds after its audit,
    # minus the LLM summary: the agent writes that itself in Stage 1.
    registry = []
    for path in file_paths:
        info = helper._peek_file(path)
        if info:
            info["source"] = "data_audit_record"
            info["from_step"] = "data_audit_record"
            registry.append(info)
    ctx["artifacts"]["data_registry"] = registry

    yield {
        "type": "status",
        "message": f"Single agent starting: {len(registry)} dataset(s), model {llm_model("single_agent.agent")}",
        "step": "single_agent_setup",
        "task_id": task_id,
    }
    yield {
        "type": "llm_stream",
        "content": "".join(f"- {s['file']} | {s['path']} | shape: {s['shape']}\n" for s in registry),
        "step": "single_agent_setup",
    }

    events = queue.Queue()
    outcome = [None]
    failure = [None]

    def _run():
        try:
            outcome[0] = run_single_agent(ctx, events.put, is_cancelled=is_cancelled)
        except Exception as e:
            import traceback
            failure[0] = (e, traceback.format_exc())
        finally:
            events.put(None)

    t = threading.Thread(target=_with_flask_ctx(_run), daemon=True)
    t.start()
    while True:
        if is_cancelled():
            # The agent checks the flag between turns; give it a moment to
            # close its transcript, then stop streaming either way.
            t.join(timeout=3)
            return
        try:
            item = events.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item
    t.join()

    if failure[0]:
        err, tb = failure[0]
        print(f"[SINGLE AGENT] failed:\n{tb}")
        yield {"type": "error", "error": f"Single agent failed: {err}", "step": "single_agent_error"}
        return

    result = outcome[0] or {}
    manuscript, md_path = finalize_manuscript(ctx, task_manager)
    if manuscript:
        try:
            rel_path = os.path.relpath(md_path, settings.output_dir).replace("\\", "/")
        except ValueError:
            rel_path = os.path.basename(md_path)
        yield {"type": "status", "message": "Manuscript", "step": "single_agent_manuscript"}
        yield {
            "type": "manuscript_complete",
            "content": manuscript,
            "path": md_path,
            "relative_path": rel_path,
            "step": "single_agent_manuscript",
        }

    status = result.get("status", "unknown")
    headline = {
        "completed": "Single agent finished the workflow.",
        "no_manuscript": "Single agent stopped without saving a manuscript.",
        "max_turns": "Single agent hit its turn limit before finishing.",
        "cancelled": "Single agent was stopped.",
        "empty_response": "Single agent received an empty model response and stopped.",
    }.get(status, f"Single agent ended ({status}).")
    summary = f"{headline} ({result.get('turns', 0)} turns)"
    if result.get("final_message"):
        summary += "\n\n" + result["final_message"]
    yield {
        "type": "result",
        "response": summary,
        "message": summary,
        "task_id": task_id,
        "result_type": f"single_agent_{status}",
    }


# =====================================================================
#  AGM RESEARCH WORKFLOW HANDLER
# =====================================================================

def agm_workflow_handler(message, intent_result, file_paths=None, file_names=None, active_solutions=None, interactive_code=False, enable_eda=False, autonomous_mode=False, data_download_mode = False, conversation_id=None, ablation=None, spatial_condition=None, exploration_data=None):
    """
    Main workflow handler.  Mirrors the case2.ipynb pipeline:
      Step 1 - Agent registration
      Step 2 - Data audit
      Step 3 - Data summarization
      Step 4 - Research question breakdown
      Step 5 - Research plan generation (LangGraph: generate -> review -> finalize)
      Step 6 - Objective execution (LangGraph: classify -> route -> execute -> debug)
      Step 7 - Result analysis preparation (collate results + figure catalog)

    Yields:
        SSE-compatible update dicts
    """
    # Reset cancellation flag at the start of a new workflow
    reset_cancellation()

    # Normalize None → [] so the no-upload (data_download_mode) path works
    file_paths = file_paths or []
    file_names = file_names or []

    # ── Initialise context ──────────────────────────────────────────
    datasets = [{"path": p, "name": n} for p, n in zip(file_paths, file_names)]
    ctx = init_workflow_context(rq=message, datasets=datasets, conversation_id=conversation_id)
    ctx["_cancel_event"] = _cancel_event
    ctx.setdefault("settings", {})["interactive_code"] = interactive_code
    ctx["settings"]["enable_eda"] = enable_eda
    ctx["settings"]["autonomous_mode"] = autonomous_mode
    ctx["settings"]["data_download_mode"] = data_download_mode
    # Kept in settings so it survives the Task Breakdown pause and plan
    # approval (both restore ctx from the saved workflow state).
    ctx["settings"]["ablation"] = ablation
    # Spatial-integration arm (B0 when unset). Resolved before any work so an
    # unknown or unbuilt condition stops the run instead of running B0 under
    # its name; also written to spatial_condition.json in the output folder.
    from agents.spatial_reasoning.conditions import apply_to_ctx
    apply_to_ctx(ctx, spatial_condition)
    # Independent exploration data for the proposed method's diagnostics,
    # {dataset id: path}; supplied by the benchmark runner (see
    # agents/spatial_reasoning/exploration.py). Without it, P splits the data.
    ctx["settings"]["exploration_data"] = exploration_data or {}
    log_count = 0


    # ── STEP 1: Register Agents ─────────────────────────────────────
    yield from helper.yield_status_update("Registering agents...", "agent_registration")
    register_agent()

    registered = list(task_manager.agent_registry.keys()) if task_manager.agent_registry else []
    task_manager.log(ctx, f"Registered agents: {registered}")
    yield from helper.yield_new_logs(ctx, log_count, "agent_registration")
    log_count = len(ctx.get("logs", []))

    if is_cancelled(): return

    # ── STEP 2: Initial Data Audit (streamed progress) ───────────────────────
    yield from helper.yield_status_update(
        f"Auditing {len(file_paths)} uploaded file(s)...", "data_audit"
    )

    # Stream audit progress via background thread + queue
    audit_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: audit_queue.put(
        {"type": "llm_stream", "content": content, "step": "data_audit"}
    )
    audit_error = [None]

    def _run_audit():
        try:
            data_agent.audit_datasets(ctx, {})
        except Exception as e:
            audit_error[0] = e
        finally:
            audit_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_run_audit), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = audit_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)

    if audit_error[0]:
        raise audit_error[0]

    yield from helper.yield_new_logs(ctx, log_count, "data_audit")
    log_count = len(ctx.get("logs", []))

    audit_records = ctx.get("artifacts", {}).get("data_audit_records", [])
    yield {
        "type": "llm_update",
        "content": f"Data audit complete: {len(audit_records)} dataset(s) audited successfully.",
        "step": "data_audit_complete",
        "data": {"audit_records": audit_records},
    }

    # ── Initialize data registry from initial audit records ──────────────────
    all_sources = []
    seen_paths = set()
    for rec in audit_records:
        path = rec.get("path", "")
        if not path or not rec.get("ok"):
            continue
        norm = os.path.normpath(path)
        if norm in seen_paths:
            continue
        info = helper._peek_file(path)
        if info:
            info["source"] = "data_audit_record"
            info["from_step"] = "data_audit_record"
            all_sources.append(info)
            seen_paths.add(norm)

    ctx["artifacts"]["data_registry"] = all_sources

    registry_lines = [f"\nData registry initialized with {len(all_sources)} source(s):"]
    for s in all_sources:
        registry_lines.append(
            f"  -{s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']}"
        )
    registry_msg = "\n".join(registry_lines) + "\n"
    print(f"[Workflow] {registry_msg}")

    yield {"type": "llm_stream", "content": registry_msg, "step": "data_audit"}
    yield {
        "type": "llm_update",
        "content": f"Data registry initialized with {len(all_sources)} source(s).",
        "step": "data_audit_complete",
    }

    if is_cancelled(): return

    # ── STEP 3: Research Question Breakdown (streamed) ───────────────
    yield from helper.yield_status_update(
        "Breaking down research question...", "rq_breakdown"
    )

    rq_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: rq_queue.put(
        {"type": "llm_stream", "content": content, "step": "rq_breakdown"}
    )
    rq_error = [None]

    def _run_rq_breakdown():
        try:
            rq_agent.rq_breakdown(ctx)
        except Exception as e:
            rq_error[0] = e
        finally:
            rq_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_run_rq_breakdown), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = rq_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)

    if rq_error[0]:
        raise rq_error[0]

    dropped = _drop_requests_when_sufficient(ctx)
    if dropped:
        yield {"type": "llm_stream", "content": _dropped_requests_note(dropped), "step": "rq_breakdown"}

    yield from helper.yield_new_logs(ctx, log_count, "rq_breakdown")
    log_count = len(ctx.get("logs", []))

    rq_breakdown_data = ctx.get("artifacts", {}).get("rq_breakdown", {})
    task_id = ctx.get("task_id", "default")
    yield {
        "type": "llm_update",
        "content": f"Research question breakdown complete.",
        "step": "rq_breakdown_complete",
        "data": {"rq_breakdown": rq_breakdown_data, "task_id": task_id,
                 "run_id": ctx.get("run_id"), "workflow_type": "research"},
    }

    if is_cancelled(): return

    # ── PAUSE for Task Breakdown review (Proceed / Provide Feedback) ──
    # The workflow stops here. The rest (optional data_download, optional EDA,
    # research plan generation) resumes when the user clicks Proceed, which
    # calls /api/rq-breakdown/proceed/<task_id> — see
    # continue_after_rq_breakdown_stream below.
    #
    # In autonomous mode the frontend auto-clicks Proceed after a short delay,
    # so this still yields a single clean resume path.
    _rq_breakdown_store_context(task_id, ctx)
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "intent_result": intent_result,
        "audit_records": _serialize_state(audit_records),
        "autonomous_mode": autonomous_mode,
        "data_download_mode": data_download_mode,
        "enable_eda": enable_eda,
        "log_count": log_count,
        "workflow_phase": "rq_breakdown_paused",
        "workflow_type": "research",
    })
    # No instruction status here: it would permanently overwrite the Task
    # Breakdown card's title with text that goes stale after Proceed. The
    # guidance is rendered inside the Data Availability card next to its
    # Proceed / Provide Feedback buttons (displayRqBreakdownSummary).
    return


def _handle_failed_downloads_interactive(
    ctx,
    data_downloads,
    is_cancelled=None,
    step_prefix="data_download",
):
    """
    Analyze `data_downloads` for failed requests. If any request returned no
    files, pause the workflow and interactively ask the user how to proceed.

    Yields frontend events (`data_download_approval`, `llm_update`). The return
    value is delivered via StopIteration (use `yield from` at the call site):

        outcome = yield from _handle_failed_downloads_interactive(ctx, data_downloads)

    Returns one of:
        "proceed"  - caller should continue with whatever data is available
        "abort"    - caller should stop the workflow
    """
    data_requests = (
        (ctx.get("artifacts", {}) or {}).get("rq_breakdown") or {}
    ).get("data_requests") or []

    # Identify failed requests (no downloaded files)
    failed_reqs = []
    total = len(data_downloads or []) or len(data_requests)
    for idx, result in enumerate(data_downloads or [], start=1):
        success = (
            result is not None
            and result.get("status") == "success"
            and result.get("downloaded_files")
        )
        if success:
            continue
        if idx - 1 < len(data_requests):
            req_text = str(data_requests[idx - 1])
        else:
            req_text = str((result or {}).get("data_request", "Unknown request"))
        reason = (result or {}).get("error") or "No files returned"
        failed_reqs.append({"idx": idx, "request": req_text, "reason": reason})

    if not failed_reqs:
        return "proceed"

    # If the user explicitly aborted and this was the ONLY data request, end the
    # whole workflow directly — there is nothing else to proceed with, so don't
    # bother showing the "proceed / upload / abort" options card.
    if (
        total == 1
        and len(failed_reqs) == 1
        and "user aborted" in (failed_reqs[0].get("reason") or "").lower()
    ):
        return "abort_single"

    options = [
        "Proceed with available data",
        "Upload missing files",
        "Abort workflow",
    ]
    failed_summary = "\n".join(
        [f"  {f['idx']}. {f['request']} ({f['reason']})" for f in failed_reqs]
    )
    base_question = (
        f"{len(failed_reqs)} of {total} data request(s) did not return any files:\n\n"
        f"{failed_summary}\n\n"
        f"How would you like to proceed?"
    )

    # Wire a fresh response queue for this pause
    response_queue = queue.Queue()
    set_data_download_response_queue(response_queue)

    try:
        # Present the approval card to the user
        yield {
            "type": "data_download_approval",
            "step": f"{step_prefix}_approval",
            "question": base_question,
            "failed_requests": failed_reqs,
            "options": options,
        }

        while True:
            # Poll for user response so we can still honour cancellation
            try:
                response = response_queue.get(timeout=0.5)
            except queue.Empty:
                if is_cancelled and is_cancelled():
                    return "abort"
                continue

            if response is None:
                return "proceed"

            # ── File upload response (user supplied a missing file) ──
            if isinstance(response, dict) and response.get("type") == "file_upload":
                fpath = response.get("file_path")
                fname = response.get("file_name") or (
                    os.path.basename(fpath) if fpath else "uploaded_file"
                )
                audit = dict(response.get("audit") or {})

                if fpath and os.path.exists(fpath):
                    # Register in ctx["datasets"]
                    ctx.setdefault("datasets", []).append({
                        "path": fpath,
                        "name": os.path.splitext(fname)[0],
                    })
                    # Register in data_registry
                    artifacts = ctx.setdefault("artifacts", {})
                    data_registry = artifacts.get("data_registry")
                    if not isinstance(data_registry, list):
                        data_registry = []
                        artifacts["data_registry"] = data_registry
                    audit["source"] = "user_uploaded"
                    audit["from_step"] = "data_download_recovery"
                    audit.setdefault("file", os.path.splitext(fname)[0])
                    audit.setdefault("path", fpath)
                    audit.setdefault("shape", [0, 0])
                    data_registry.append(audit)

                    yield {
                        "type": "llm_update",
                        "content": f"Uploaded and registered: {fname}",
                        "step": f"{step_prefix}_approval",
                    }
                    # Re-present the prompt so the user can upload more or proceed
                    yield {
                        "type": "data_download_approval",
                        "step": f"{step_prefix}_approval",
                        "question": (
                            f"Added {fname}. You can upload more files, or "
                            f"choose how to proceed.\n\n{base_question}"
                        ),
                        "failed_requests": failed_reqs,
                        "options": options,
                    }
                    continue
                else:
                    yield {
                        "type": "llm_update",
                        "content": f"Upload failed for {fname} (file not found).",
                        "step": f"{step_prefix}_approval",
                    }
                    continue

            # ── Text/button choice ──
            choice = str(response).strip() if response is not None else ""
            lower = choice.lower()

            if "proceed" in lower:
                yield {
                    "type": "llm_update",
                    "content": "Proceeding with available data.",
                    "step": f"{step_prefix}_approval",
                }
                return "proceed"

            if "abort" in lower or "stop" in lower or "cancel" in lower:
                yield {
                    "type": "llm_update",
                    "content": "Workflow aborted by user after data download failures.",
                    "step": f"{step_prefix}_approval",
                }
                return "abort"

            if "upload" in lower:
                # The frontend will open a file dialog; wait for the next response
                continue

            # Custom free-text note — record and proceed
            yield {
                "type": "llm_update",
                "content": f"User note: {choice}. Proceeding with available data.",
                "step": f"{step_prefix}_approval",
            }
            return "proceed"
    finally:
        clear_data_download_response_queue()



# A .shp is useless on its own — GDAL/Fiona require its sidecars alongside it
# (at minimum .shx + .dbf; .prj/.cpg/.qix/.sbn/.sbx/.xml are read when present).
# When only the .shp gets reported as "the downloaded file" (e.g. a derived
# file written by gdf.to_file(), as opposed to every member of an extracted
# ZIP being individually detected), staging just that one file leaves a
# non-functional shapefile behind for anything that later re-opens it from
# the uploads copy — it "loads fine" from the original output dir but fails
# wherever the staged copy is read instead.
_SHAPEFILE_SIDECAR_EXTS = (
    ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx",
    ".shp.xml", ".shp.ea.iso.xml", ".shp.iso.xml",
)


# Vector formats GeoPandas/Fiona can round-trip. Mirrors the download path's
# VECTOR_DRIVERS map in DataRetrieverAgent.LLM_Find.
_ANALYSIS_VECTOR_DRIVERS = {
    '.geojson': 'GeoJSON',
    '.gpkg': 'GPKG',
    '.shp': 'ESRI Shapefile',
    '.kml': 'KML',
    '.gml': 'GML',
}


def _ensure_wgs84_outputs(file_paths):
    """Reproject analysis output vectors to EPSG:4326, in place.

    Downloaded data already gets this treatment (LLM_Find reprojects every
    vector it fetches), but ANALYSIS outputs never passed through it. Metric
    work — buffering by metres, distance joins — forces the generated code to
    project to something like EPSG:3857, and it frequently writes the result
    without projecting back. Mapbox GL only accepts lon/lat, so such a file
    fails to render with "Invalid LngLat latitude value: must be between -90
    and 90" (a Web-Mercator northing like 4884859 is not a latitude).

    Best-effort and non-fatal: an unreadable or CRS-less file is left exactly
    as the analysis wrote it.
    """
    if not file_paths:
        return
    # Imported lazily: geopandas is expensive to load and this module is
    # imported on every request path, most of which never touch a vector.
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"[Workflow] geopandas unavailable, skipping reprojection: {e}")
        return
    for fp in file_paths or []:
        ext = os.path.splitext(fp)[1].lower()
        driver = _ANALYSIS_VECTOR_DRIVERS.get(ext)
        if not driver or not os.path.exists(fp):
            continue
        try:
            gdf = gpd.read_file(fp)
            if gdf.empty or gdf.crs is None:
                continue
            if gdf.crs.to_epsg() == 4326:
                continue
            gdf.to_crs("EPSG:4326").to_file(fp, driver=driver)
            print(f"[Workflow] Reprojected analysis output to EPSG:4326: {fp}")
        except Exception as e:
            print(f"[Workflow] Could not reproject {fp} to EPSG:4326: {e}")


def _delivered_dataset_paths(result):
    """The files a successful download actually delivered as datasets.

    downloaded_files holds every file the generated script wrote, including
    its scratch (raw API dumps, tiles, temp JSON, the zip a shapefile came
    from). Only the datasets may become analysis inputs; registering the
    scratch confuses the planner and code generator.
    """
    if not result:
        return []
    if result.get("dataset_files") is not None:
        return list(result.get("dataset_files") or [])
    try:
        from agents.data_agent.DataRetrieverAgent import classify_delivered_files
        datasets, _ = classify_delivered_files(result.get("downloaded_files") or [])
        return datasets
    except Exception:
        return list(result.get("downloaded_files") or [])


def _stage_downloaded_file_for_ui(ctx, fpath, file_basename, file_ext, file_size, info):
    """Copy a freshly downloaded file into the conversation's uploads dir and
    register it in the DB *immediately* (not only at the end-of-run summary),
    so the UI can fetch it via /api/uploads/<conv_id>/<file> as soon as the
    request finishes — this is what makes right-click Download work on a
    partially-completed batch. Best-effort; mirrors the summary's staging and
    is idempotent (add_conversation_file upserts; copy is path-guarded)."""
    try:
        import os as _os
        import shutil as _shutil
        import json as _json
        from WebUI.conversation_db import add_conversation_file
        task_id = ctx.get("task_id")
        if not task_id or not _os.path.exists(fpath):
            return
        webui_base = _os.path.dirname(_os.path.abspath(__file__))
        upload_dir = _os.path.join(webui_base, "uploads", task_id)
        _os.makedirs(upload_dir, exist_ok=True)
        dest_path = _os.path.join(upload_dir, file_basename)
        if _os.path.normpath(dest_path) != _os.path.normpath(fpath):
            _shutil.copy2(fpath, dest_path)
        add_conversation_file(
            task_id, file_basename, _os.path.splitext(file_basename)[0],
            file_size=file_size, file_type=file_ext,
            audit_json=_json.dumps(info, default=str),
        )
        if file_basename.lower().endswith(".shp"):
            src_dir = _os.path.dirname(fpath)
            stem = file_basename[:-4]  # strip ".shp"
            for sidecar_ext in _SHAPEFILE_SIDECAR_EXTS:
                sidecar_name = stem + sidecar_ext
                sidecar_src = _os.path.join(src_dir, sidecar_name)
                if not _os.path.exists(sidecar_src):
                    continue
                sidecar_dest = _os.path.join(upload_dir, sidecar_name)
                if _os.path.normpath(sidecar_dest) == _os.path.normpath(sidecar_src):
                    continue
                try:
                    _shutil.copy2(sidecar_src, sidecar_dest)
                except Exception as _sidecar_err:
                    print(f"[Workflow] Warning: failed to stage shapefile "
                          f"sidecar {sidecar_name}: {_sidecar_err}")
    except Exception as _stage_err:
        print(f"[Workflow] Warning: failed to stage {file_basename} for UI: {_stage_err}")


_SPATIAL_RESULT_LAYER_EXTS = (".gpkg", ".geojson", ".shp", ".kml", ".zip")


def _stage_spatial_results(ctx, files):
    """Register a finished Spatial Analysis run's map outputs as conversation
    files, so they come back in Data Layers / the Map Viewer when the
    conversation is reopened (and a removal of them sticks), exactly as
    downloaded data does. Other outputs (figures, tables) stay as run
    artifacts. Best-effort."""
    for fpath in files or []:
        name = os.path.basename(fpath)
        ext = os.path.splitext(name)[1].lower()
        if ext not in _SPATIAL_RESULT_LAYER_EXTS or not os.path.exists(fpath):
            continue
        try:
            size = os.path.getsize(fpath)
        except OSError:
            size = 0
        _stage_downloaded_file_for_ui(
            ctx, fpath, name, ext.lstrip("."), size,
            {"file": os.path.splitext(name)[0], "path": fpath,
             "source": "spatial_analysis_result"})


def _post_rq_breakdown_pipeline(
    ctx,
    intent_result,
    audit_records,
    autonomous_mode,
    data_download_mode,
    enable_eda,
    log_count,
):
    """
    Run the post-Task-Breakdown portion of the main workflow:
      - optional Data Download (LLM_Find)
      - optional EDA & Preprocessing
      - Research Plan Generation (pauses at the research-plan approval gate)

    This is a generator that yields SSE-compatible update dicts. It is called
    from ``continue_after_rq_breakdown_stream`` when the user clicks Proceed
    on the Task Breakdown card.
    """
    if is_cancelled(): return

    # ── STEP (optional): Data Download (LLM_Find) ────────────────────
    rq_breakdown_data = ctx.get("artifacts", {}).get("rq_breakdown", {})
    # User-provided data-source keys (from the Web UI) for this task.
    ctx["_data_source_keys"] = get_data_source_keys(ctx.get("task_id"))
    # User's private contributed data sources (handbook dirs) for this task.
    ctx["_user_handbook_dirs"] = get_user_handbook_dirs(ctx.get("task_id"))
    if data_download_mode:
        data_requests = rq_breakdown_data.get("data_requests") or []
        if data_requests:
            # Titled for what happens FIRST (source selection per request);
            # the data_download_complete event retitles this card at the end.
            yield from helper.yield_status_update(
                f"Selecting data source(s) for {len(data_requests)} data "
                f"request(s)...",
                "data_download",
            )

            _GEOSPATIAL_EXTS_REQ = {"geojson", "json", "gpkg", "shp", "zip", "kml", "gml", "tif", "tiff", "nc", "img", "vrt"}
            _TABULAR_EXTS_REQ = {"csv", "xlsx", "xls", "parquet"}

            def _build_request_files_event(idx, data_request, result):
                """Package a completed data request into a tile-ready event."""
                files_payload = []
                if result and result.get("status") == "success":
                    # FileWriteTracker captures every write, so downloaded_files
                    # mixes the requested data with the script's own scratch --
                    # one PSU run produced 2 datasets and 35 .osm.xml tiles.
                    # Flagging them lets the UI lead with the data and fold the
                    # rest away, instead of showing 37 equal-looking results.
                    try:
                        from agents.data_agent.DataRetrieverAgent import (
                            classify_delivered_files)
                        _ds, _ = classify_delivered_files(
                            result.get("downloaded_files") or [])
                        _dataset_paths = set(_ds)
                    except Exception:
                        _dataset_paths = set(result.get("downloaded_files") or [])
                    for fpath in result.get("downloaded_files") or []:
                        try:
                            info = helper._peek_file(fpath) or {}
                        except Exception:
                            info = {}
                        file_basename = os.path.basename(fpath)
                        file_ext = os.path.splitext(file_basename)[1].lstrip(".").lower()
                        try:
                            file_size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                        except Exception:
                            file_size = 0
                        # Stage now so /api/uploads can serve this file the moment
                        # the request finishes (enables right-click Download on a
                        # partially-completed batch).
                        # is_dataset rides along in the stored audit record so a
                        # reloaded conversation can keep the script's scratch
                        # files out of the Data Layers panel, as the live run does.
                        _stage_downloaded_file_for_ui(
                            ctx, fpath, file_basename, file_ext, file_size,
                            {**info, "is_dataset": fpath in _dataset_paths})
                        files_payload.append({
                            "isDataset": fpath in _dataset_paths,
                            "file": info.get("file") or os.path.splitext(file_basename)[0],
                            "fileName": file_basename,
                            "displayName": os.path.splitext(file_basename)[0],
                            "fileType": file_ext,
                            "fileSize": file_size,
                            "path": fpath,
                            "shape": info.get("shape", [0, 0]),
                            "isGeospatial": file_ext in _GEOSPATIAL_EXTS_REQ,
                            "isTabular": file_ext in _TABULAR_EXTS_REQ,
                            "conversationId": ctx.get("task_id"),
                        })
                status_str = "success" if files_payload else "failed"
                return {
                    "type": "data_request_files",
                    "step": (f"data_download_req_{idx}_cont"
                             if (ctx.get('_synth_done_req') == idx
                                 or ctx.get('_confirm_done_req') == idx)
                             else f"data_download_req_{idx}"),
                    "request": str(data_request),
                    "status": status_str,
                    "files": files_payload,
                    # WHY there are no files, when there are none. The tiles
                    # are the only part of a request that is redrawn after a
                    # reload (the review gate is not, once answered), so a
                    # request the source could not serve used to reload as
                    # a bare "No files downloaded" with the reason gone.
                    "error": str((result or {}).get("error") or ""),
                    "data_unavailable": bool((result or {}).get("data_unavailable")),
                }

            download_queue = queue.Queue()
            # Step id for this request's stream. After the user answers a
            # confirmation prompt (or a skill synthesis runs) it switches to a
            # CONTINUATION card (data_download_req_N_cont) so the resumed work
            # — code generation, execution — renders BELOW the confirmation
            # panel / synthesis card instead of in the original request card
            # above them.
            def _req_step():
                i = ctx.get('_current_download_req', 0)
                return (f"data_download_req_{i}_cont"
                        if (ctx.get('_synth_done_req') == i
                            or ctx.get('_confirm_done_req') == i)
                        else f"data_download_req_{i}")

            ctx["_stream_callback"] = lambda content: download_queue.put(
                {"type": "llm_stream", "content": content, "step": _req_step()}
            )

            # Need-driven skill synthesis streams to its OWN card (its own step
            # id); it also flags the request so everything streamed after the
            # synthesis goes to the continuation card.
            def _synth_stream_cb(content):
                ctx["_synth_done_req"] = ctx.get('_current_download_req', 0)
                download_queue.put({
                    "type": "llm_stream", "content": content,
                    "step": f"skill_synth_req_{ctx.get('_current_download_req', 0)}",
                })
            ctx["_synthesis_stream_callback"] = _synth_stream_cb
            ctx["_register_rerun_context"] = (
                lambda idx, info: save_rerun_context(ctx.get("task_id"), idx, info))
            ctx["_request_done_callback"] = lambda idx, data_request, result: download_queue.put(
                _build_request_files_event(idx, data_request, result)
            )

            # Interactive source-select callback: for every data request the
            # downloader thread asks the user to confirm the LLM's chosen
            # data source (with confidence score). The user can proceed,
            # change source, retry, abort, or click "Always Proceed" to skip
            # future prompts in this run.
            source_select_response_queue = queue.Queue()
            _skip_source_select = [False]
            # The callback is installed in BOTH interactive and autonomous mode.
            # Autonomous mode normally proceeds without confirmation, but a
            # source that still needs an API key is a hard blocker the system
            # can't resolve on its own — so we must pause to let the user enter
            # it even when running autonomously (see the missing-keys check).
            set_source_select_response_queue(source_select_response_queue)

            def _source_select_cb(payload):
                missing_keys = payload.get("required_keys") or []
                is_credential_warning = bool(payload.get("credential_error"))
                # Skip the confirmation prompt only when nothing is blocking:
                # neither autonomous mode nor a prior "Always Proceed" can supply
                # a missing API key, so whenever one is missing we still prompt.
                # A credential warning (the entered key was rejected) is ALWAYS a
                # hard blocker — prompt even in autonomous / Always-Proceed mode.
                if not missing_keys and not is_credential_warning and (
                    autonomous_mode or _skip_source_select[0]
                ):
                    # No blocker to prompt for. If the source itself is
                    # unrecognized (no handbook exists) and nobody is at the
                    # dropdown to click "Generate & Use New Skill", LLM_Find's
                    # normal "proceed" handling for an unrecognized source
                    # would silently substitute the closest EXISTING handbook
                    # instead — defeating automatic source onboarding
                    # entirely. Trigger real generation instead, the same
                    # action a human would pick, so autonomous runs actually
                    # exercise the Handbook Generator. Skipped when this
                    # payload IS the post-synthesis confirmation (a skill was
                    # just generated) — there we want to proceed with it, not
                    # generate again.
                    if not payload.get("is_recognized", True) and not payload.get("synthesized"):
                        gen_query = (
                            (payload.get("ideal_source") or "").strip()
                            or str(payload.get("llm_choice") or "").strip()
                        )
                        if gen_query:
                            return {
                                "action": "generate",
                                "generate_query": gen_query,
                                "generate_website": "",
                            }
                    return {"action": "proceed"}
                _tid = ctx.get("task_id")
                # Drop any stale answer (disk + in-memory) so we only accept
                # a response made against THIS prompt. Draining the in-memory
                # queue prevents a previous request's answer — left behind
                # when that answer was consumed via the disk path — from
                # auto-confirming this prompt before the user even sees it.
                clear_source_select_answer(_tid)
                _drain_source_select_queue(source_select_response_queue)
                # The post-synthesis confirmation anchors under the "Generating
                # New Skill" card (its own step id) so it appears where the
                # synthesis just finished; the normal prompt anchors under the
                # data-request card as before.
                _prompt_step = (
                    f"skill_synth_req_{ctx.get('_current_download_req', 0)}"
                    if payload.get("synthesized")
                    else _req_step()
                )
                download_queue.put({
                    "type": "data_source_select_approval",
                    "step": _prompt_step,
                    "task_id": _tid,
                    "data_request": payload.get("data_request"),
                    "llm_choice": payload.get("llm_choice"),
                    "llm_explanation": payload.get("llm_explanation", ""),
                    "ideal_source": payload.get("ideal_source", ""),
                    "ideal_source_reasoning": payload.get("ideal_source_reasoning", ""),
                    "data_to_download": payload.get("data_to_download", ""),
                    "confidence": payload.get("confidence", 0),
                    "is_recognized": payload.get("is_recognized", False),
                    "available_sources": payload.get("available_sources") or [],
                    "required_keys": missing_keys,
                    "caveats": payload.get("caveats", ""),
                    "credential_error": is_credential_warning,
                    "credential_message": payload.get("credential_message", ""),
                    # Post-synthesis confirmation variant: a new skill was just
                    # generated; ask the user to proceed before downloading.
                    "synthesized": bool(payload.get("synthesized")),
                    "synthesized_status": payload.get("synthesized_status", ""),
                    "synthesized_note": payload.get("synthesized_note", ""),
                })
                # Block the downloader thread until the user answers. Poll
                # BOTH the in-memory queue (same-process / local dev) and a
                # disk file (cross-process: on multi-worker hosting the
                # Proceed POST may be served by a different worker process).
                # The marker (heartbeat refreshed on every tick) is what lets
                # a page that reopens mid-wait know this prompt is still live
                # and worth answering; the finally clears it however we leave.
                write_pending_prompt(_tid, "source_select", {"step": _prompt_step})
                response = None
                try:
                    while response is None:
                        if is_cancelled():
                            return None
                        try:
                            response = source_select_response_queue.get(timeout=0.3)
                        except queue.Empty:
                            response = pop_source_select_answer(_tid)
                        touch_pending_prompt(_tid)
                finally:
                    clear_pending_prompt(_tid)
                # A queue answer leaves the disk file untouched; clear it so
                # it cannot be misread by the next request's prompt.
                clear_source_select_answer(_tid)
                # A prompt was shown and answered: route the rest of this
                # request's stream to its CONTINUATION card so the resumed work
                # renders below the (frozen) confirmation panel. Not set on the
                # autonomous early-return above — no panel was shown there.
                ctx["_confirm_done_req"] = ctx.get('_current_download_req', 0)
                # Persist any keys the user entered into ctx so later data
                # requests for the same (or another) key-gated source reuse them
                # instead of re-prompting — important in autonomous mode.
                if isinstance(response, dict):
                    _entered = response.get("data_source_keys") or {}
                    if isinstance(_entered, dict) and _entered:
                        ctx.setdefault("_data_source_keys", {}).update(
                            {k: v for k, v in _entered.items() if v}
                        )
                if isinstance(response, dict) and response.get("action") == "always_proceed":
                    _skip_source_select[0] = True
                    # Preserve any keys the user entered with Always Proceed.
                    return {"action": "proceed",
                            "data_source_keys": response.get("data_source_keys") or {}}
                return response

            ctx["_source_select_callback"] = _source_select_cb

            # ── Post-download review (interactive runs only) ───────────
            # See the spatial-analysis pipeline for the full rationale; this
            # is the same checkpoint for the research/data-retriever path.
            if not autonomous_mode:
                download_review_queue = queue.Queue()
                set_download_review_response_queue(download_review_queue)

                def _download_review_cb(payload):
                    _tid = ctx.get("task_id")
                    clear_download_review_answer(_tid)
                    while True:
                        try:
                            download_review_queue.get_nowait()
                        except queue.Empty:
                            break
                    idx = payload.get("request_index", 0)
                    event = {
                        "type": "data_download_review",
                        "step": f"data_download_req_{idx}_cont",
                        "task_id": _tid,
                    }
                    event.update(payload)
                    download_queue.put(event)
                    write_pending_prompt(_tid, "download_review",
                                         {"step": event.get("step", "")})
                    response = None
                    try:
                        while response is None:
                            if is_cancelled():
                                return None
                            try:
                                response = download_review_queue.get(timeout=0.3)
                            except queue.Empty:
                                response = pop_download_review_answer(_tid)
                            touch_pending_prompt(_tid)
                    finally:
                        clear_pending_prompt(_tid)
                    clear_download_review_answer(_tid)
                    return response

                ctx["_download_review_callback"] = _download_review_cb

            download_error = [None]
            download_result = [None]

            def _run_download():
                try:
                    download_result[0] = data_agent.data_download(ctx)
                except Exception as e:
                    download_error[0] = e
                finally:
                    download_queue.put(None)  # sentinel

            t = threading.Thread(target=_with_flask_ctx(_run_download), daemon=True)
            t.start()

            while True:
                if is_cancelled():
                    t.join(timeout=3)
                    ctx.pop("_stream_callback", None)
                    ctx.pop("_synthesis_stream_callback", None)
                    ctx.pop("_request_done_callback", None)
                    ctx.pop("_source_select_callback", None)
                    ctx.pop("_download_review_callback", None)
                    clear_source_select_response_queue()
                    clear_download_review_response_queue()
                    return
                try:
                    item = download_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                yield item

            t.join()
            ctx.pop("_stream_callback", None)
            ctx.pop("_synthesis_stream_callback", None)
            ctx.pop("_request_done_callback", None)
            ctx.pop("_source_select_callback", None)
            ctx.pop("_download_review_callback", None)
            clear_source_select_response_queue()
            clear_download_review_response_queue()

            if download_error[0]:
                raise download_error[0]

            yield from helper.yield_new_logs(ctx, log_count, "data_download")
            log_count = len(ctx.get("logs", []))

            # ── Collect newly downloaded files, add to ctx["datasets"] ──
            data_downloads = download_result[0] or []

            # ── Interactive pause if any request failed ──
            if not autonomous_mode:
                outcome = yield from _handle_failed_downloads_interactive(
                    ctx, data_downloads, is_cancelled=is_cancelled
                )
                if outcome in ("abort", "abort_single"):
                    yield {
                        "type": "error",
                        "error": (
                            "Workflow stopped — you aborted the only data request."
                            if outcome == "abort_single"
                            else "Workflow aborted by user after data download failures."
                        ),
                    }
                    return

            existing_paths = {
                os.path.normpath(d["path"])
                for d in ctx.setdefault("datasets", [])
                if d.get("path")
            }
            new_files = []
            for result in data_downloads:
                if not result or result.get("status") != "success":
                    continue
                for fpath in _delivered_dataset_paths(result):
                    norm = os.path.normpath(fpath)
                    if norm in existing_paths:
                        continue
                    ctx["datasets"].append({
                        "path": fpath,
                        "name": os.path.splitext(os.path.basename(fpath))[0],
                    })
                    existing_paths.add(norm)
                    new_files.append(fpath)

            summary_files = []
            if new_files:
                # Refresh audit records to include the new files
                try:
                    data_agent.audit_datasets(ctx, {})
                except Exception as e:
                    print(f"[Workflow] Warning: re-audit after download failed: {e}")

                # ── Copy downloaded files into the conversation's uploads dir ──
                # This lets them load via the existing /api/uploads/<conv_id>/<file>
                # endpoint and be treated as regular data layers in the UI.
                import shutil
                from WebUI.conversation_db import add_conversation_file
                webui_base = os.path.dirname(os.path.abspath(__file__))
                task_id = ctx.get("task_id")
                conv_upload_dir = None
                if task_id:
                    conv_upload_dir = os.path.join(webui_base, "uploads", task_id)
                    os.makedirs(conv_upload_dir, exist_ok=True)

                # Append new files to ctx["artifacts"]["data_registry"] (list form)
                artifacts = ctx.setdefault("artifacts", {})
                data_registry = artifacts.get("data_registry")
                if not isinstance(data_registry, list):
                    data_registry = []
                    artifacts["data_registry"] = data_registry
                registry_paths = {
                    os.path.normpath(s["path"])
                    for s in data_registry if s.get("path")
                }
                registry_lines = []

                GEOSPATIAL_EXTS = {"geojson", "json", "gpkg", "shp", "zip", "kml", "gml", "tif", "tiff", "nc", "img", "vrt"}
                TABULAR_EXTS = {"csv", "xlsx", "xls", "parquet"}

                for fpath in new_files:
                    if os.path.normpath(fpath) in registry_paths:
                        continue
                    info = helper._peek_file(fpath)
                    if not info:
                        # Unrecognised format — still register the file with minimal metadata
                        info = {
                            "file": os.path.splitext(os.path.basename(fpath))[0],
                            "path": fpath,
                            "shape": [0, 0],
                            "columns": {},
                            "head": {},
                        }
                    info["source"] = "downloaded"
                    info["from_step"] = "data_download"
                    data_registry.append(info)
                    registry_paths.add(os.path.normpath(fpath))
                    registry_lines.append(
                        f"  -{info['file']} (downloaded) | path: {info['path']} | shape: {info['shape']}"
                    )

                    # Copy file into uploads/<conv_id>/ and register in DB
                    file_basename = os.path.basename(fpath)
                    file_ext = os.path.splitext(file_basename)[1].lstrip(".").lower()
                    display_name = os.path.splitext(file_basename)[0]
                    served_name = file_basename
                    try:
                        file_size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                    except Exception:
                        file_size = 0

                    # Stage into uploads/<conv_id>/ and register in the DB. The
                    # helper also copies a shapefile's .dbf/.shx/.prj/.cpg, which
                    # are no longer in this list now that scratch is filtered out.
                    if conv_upload_dir:
                        _stage_downloaded_file_for_ui(
                            ctx, fpath, file_basename, file_ext, file_size,
                            {**info, "is_dataset": True})

                    summary_files.append({
                        "file": info["file"],
                        "fileName": served_name,
                        "displayName": display_name,
                        "fileType": file_ext,
                        "fileSize": file_size,
                        "path": info["path"],
                        "shape": info["shape"],
                        "isGeospatial": file_ext in GEOSPATIAL_EXTS,
                        "isTabular": file_ext in TABULAR_EXTS,
                        "conversationId": task_id,
                    })

                summary = (
                    f"Downloaded {len(new_files)} new file(s) and added to data registry:\n"
                    + "\n".join(registry_lines)
                )
            else:
                summary = "No new files were downloaded. All data requests failed after retries."

            # Always yield the summary so the UI panel renders,
            # whether or not any files were downloaded.
            yield {
                "type": "data_download_summary",
                "content": summary,
                "step": "data_download",
                "files": summary_files,
            }

            yield {
                "type": "llm_update",
                "content": f"Data download complete: {len(new_files)} new file(s) added.",
                "step": "data_download_complete",
                "data": {
                    "downloaded_files": new_files,
                    "data_registry": ctx.get("artifacts", {}).get("data_registry", []),
                },
            }

            if is_cancelled(): return
        else:
            yield {
                "type": "llm_update",
                "content": "Data download mode is on, but no data requests were generated by the RQ breakdown.",
                "step": "data_download_skipped",
            }

    # ── STEP (optional): EDA & Preprocessing (agentic, streamed) ─────
    if enable_eda:
        yield from helper.yield_status_update(
            "Performing exploratory data analysis and preprocessing...", "eda"
        )

        eda_queue = queue.Queue()
        eda_approval_queue = queue.Queue()    # tool → workflow (approval requests)
        eda_response_queue = queue.Queue()    # workflow ← frontend (user responses)
        set_eda_response_queue(eda_response_queue)  # register globally for API endpoint

        ctx["_stream_callback"] = lambda content: eda_queue.put(
            {"type": "llm_stream", "content": content, "step": "eda"}
        )
        ctx["_approval_queue"] = eda_approval_queue
        ctx["_response_queue"] = eda_response_queue

        eda_error = [None]

        def _run_eda():
            try:
                data_agent.eda(ctx)
            except Exception as e:
                eda_error[0] = e
            finally:
                eda_queue.put(None)  # sentinel

        t = threading.Thread(target=_with_flask_ctx(_run_eda), daemon=True)
        t.start()

        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                ctx.pop("_approval_queue", None)
                ctx.pop("_response_queue", None)
                clear_eda_response_queue()
                return

            # Check for approval requests from the tool
            try:
                approval_item = eda_approval_queue.get_nowait()
                # Yield approval request to frontend — the tool thread is now
                # blocked on eda_response_queue.get() waiting for user response.
                yield approval_item
            except queue.Empty:
                pass

            try:
                item = eda_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()
        ctx.pop("_stream_callback", None)
        ctx.pop("_approval_queue", None)
        ctx.pop("_response_queue", None)
        clear_eda_response_queue()

        if eda_error[0]:
            raise eda_error[0]

        yield from helper.yield_new_logs(ctx, log_count, "eda")
        log_count = len(ctx.get("logs", []))

        # Collect EDA artifacts (plots, cleaned data, reports) from output dir
        eda_artifacts = []
        eda_output_dir = resolve_output_dir(ctx)
        eda_artifact_exts = (".png", ".jpg", ".csv", ".gpkg", ".json", ".html", ".geojson")
        if os.path.isdir(eda_output_dir):
            for fname in os.listdir(eda_output_dir):
                fpath = os.path.join(eda_output_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in eda_artifact_exts:
                    continue
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    fsize = 0
                try:
                    rel = os.path.relpath(fpath, settings.output_dir).replace("\\", "/")
                except ValueError:
                    rel = fname
                eda_artifacts.append({
                    "path": fpath,
                    "relative_path": rel,
                    "filename": fname,
                    "size": fsize,
                    "type": ext.lstrip("."),
                })

        eda_summary = ctx.get("artifacts", {}).get("eda_report_data", {}).get("summary", "")
        if eda_artifacts:
            yield {
                "type": "step_result",
                "step": "eda",
                "step_index": "EDA",
                "step_description": "Exploratory Data Analysis & Preprocessing",
                "status": "completed",
                "output": eda_summary,
                "artifacts": eda_artifacts,
            }

        # Store ctx so /api/eda-explore can access it later
        set_latest_workflow_ctx(ctx)

        yield from helper.yield_status_update(
            "EDA and preprocessing complete.", "eda_complete"
        )
    # else: EDA silently skipped when disabled

    if is_cancelled(): return

    # ── No-planning ablation: no plan stage at all ──────────────────
    # No plan is generated, shown, or approved, so nothing on screen suggests
    # planning happened. The placeholder plan downstream stages need is built
    # silently and execution starts in this same stream.
    from agents.task_manager.ablation import NO_PLANNING, ablation_of
    if ablation_of(ctx) == NO_PLANNING:
        yield from _no_planning_handoff(ctx, intent_result, audit_records, log_count)
        return

    # ── STEP 4: Research Plan Generation (LangGraph) ────────────────
    yield from helper.yield_status_update(
        "Generating research plan...", "research_plan_generation"
    )

    # Run the LangGraph pipeline in a background thread and stream
    # LLM chunks back to the frontend via a queue.
    chunk_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": "research_plan"}
    )
    ctx["_event_callback"] = lambda event: chunk_queue.put(event)
    graph_error = [None]

    def _run_graph():
        try:
            research_plan_generator.build_serial_graph(ctx)
        except Exception as e:
            graph_error[0] = e
        finally:
            chunk_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_run_graph), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_event_callback", None)

    if graph_error[0]:
        raise graph_error[0]

    yield from helper.yield_new_logs(ctx, log_count, "research_plan_generation")
    log_count = len(ctx.get("logs", []))

    # Build visualisation HTML files (plan view + flowchart)
    structured_plan = ctx["artifacts"]["research_plan_final"]
    task_id = ctx.get("task_id", "default")
    task_output_dir = resolve_output_dir(ctx)
    viz_paths = research_plan_viz.save_research_plan_visualization(
        structured_plan, task_output_dir, task_id
    )

    # Store ctx so the research-plan feedback loop can access it
    research_plan_generator.store_context(task_id, ctx)

    yield {
        "type": "llm_update",
        "content": "Research plan generated successfully.",
        "step": "research_plan_generation",
    }

    # Send visualisation info to frontend
    yield {
        "type": "research_plan_viz",
        "viz_id": task_id,
        "viz_url": f"/api/research-plan/{task_id}",
    }

    num_objectives = len(structured_plan.get("objectives", []))
    total_steps = sum(
        len(obj.get("steps", []))
        for obj in structured_plan.get("objectives", [])
    )

    yield {
        "type": "llm_update",
        "content": (
            f"Research plan generated: {num_objectives} objectives, "
            f"{total_steps} steps. Please review and approve to continue."
        ),
        "step": "research_plan_summary",
    }

    # ── Save state and PAUSE for user approval ─────────────────────
    # The workflow stops here. Steps 6-7 run when user clicks "Approve".
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "intent_result": intent_result,
        "data_overview": _serialize_state(ctx.get("artifacts", {}).get("data_audit_records", [])),
        "data_summary": _serialize_state(ctx.get("artifacts", {}).get("data_summary_data", {})),
        "audit_records": _serialize_state(audit_records),
        "structured_plan": _serialize_state(structured_plan),
        "num_objectives": num_objectives,
        "total_steps": total_steps,
        "log_count": log_count,
    })

    if autonomous_mode:
        result_msg = (
            f"Research plan generated with {num_objectives} objectives and "
            f"{total_steps} steps. Autonomous mode — auto-approving..."
        )
    else:
        result_msg = (
            f"Research plan generated with {num_objectives} objectives and "
            f"{total_steps} steps. Please review the plan above and click "
            f"'Approve Plan' to proceed with objective execution, or "
            f"'Provide Feedback' to revise."
        )

    yield {
        "type": "result",
        "response": result_msg,
        "intent": intent_result,
        "result_type": "research_plan_pending_approval",
        "task_id": task_id,
        "structured_plan": structured_plan,
    }


def _no_planning_handoff(ctx, intent_result, audit_records, log_count):
    """No-planning ablation: skip plan generation, review, display and the
    approval pause, and go straight to execution.

    Execution, result analysis and the manuscript read research_plan_final, so
    a one-step placeholder is stored where the planner would put it -- but it
    is never shown: no plan card, no plan visualization, no Approve button.
    """
    from agents.task_manager.ablation import build_no_planning_plan
    task_id = ctx.get("task_id", "default")

    yield {
        "type": "status",
        "message": "No planning (ablation): the Modeling Agent writes code for the whole question",
        "step": "ablation_no_planning",
    }
    yield {
        "type": "llm_stream",
        "content": ("The Planning Agent is switched off for this run. No research plan is "
                    "generated or reviewed. The research question and its breakdown go "
                    "directly to the Modeling Agent, which writes the analysis code in "
                    "one step.\n"),
        "step": "ablation_no_planning",
    }

    build_no_planning_plan(ctx)
    structured_plan = ctx["artifacts"]["research_plan_final"]
    research_plan_generator.store_context(task_id, ctx)
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "intent_result": intent_result,
        "audit_records": _serialize_state(audit_records),
        "structured_plan": _serialize_state(structured_plan),
        "num_objectives": 1,
        "total_steps": 1,
        "log_count": log_count,
    })

    yield from continue_after_approval_stream(task_id)


def _restore_rq_breakdown_ctx(task_id):
    """
    Restore ctx + workflow metadata for the Task Breakdown pause point.

    Returns (ctx, saved_state_dict) or (None, None) if nothing is saved.
    Prefers the live in-memory ctx (has unserializable _cancel_event, live
    objects, etc.) and falls back to the on-disk snapshot.
    """
    ctx = _rq_breakdown_get_context(task_id)
    saved_state = get_workflow_state(task_id) or {}

    if ctx and _is_current_run(ctx, task_id, saved_state):
        ctx["_cancel_event"] = _cancel_event
        return ctx, saved_state

    if saved_state and saved_state.get("ctx"):
        ctx = saved_state["ctx"]
        ctx["_cancel_event"] = _cancel_event
        ctx.setdefault("artifacts", {})
        if not isinstance(ctx.get("settings"), dict) or not ctx["settings"]:
            from utils.config import load_configurations, resolve_output_dir
            ctx["settings"] = load_configurations()
        ctx.setdefault("logs", [])
        ctx["task_id"] = task_id
        if "output_dir" not in ctx:
            task_output_dir = os.path.join(settings.output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)
            ctx["output_dir"] = task_output_dir
        _rq_breakdown_store_context(task_id, ctx)
        return ctx, saved_state

    return None, None


STALE_BREAKDOWN = ("This Task Breakdown is no longer waiting: a newer step in this conversation "
                   "has replaced it. Send your request again to continue.")


def _paused_breakdown(task_id, run_id=None, workflow_type=None):
    """The paused Task Breakdown a Proceed/Feedback click refers to.

    Returns (ctx, saved_state, workflow_type, error). The card sends the run_id
    and pipeline of the turn it was drawn for; the saved state must still be
    paused at that turn, otherwise the click is refused rather than resuming
    something else. The pipeline comes from the saved state. The only state
    without one is a Research pause saved before pipelines were recorded.
    """
    ctx, saved = _restore_rq_breakdown_ctx(task_id)
    if not ctx:
        return None, None, None, (f"No Task Breakdown state found for task {task_id}. "
                                  "Please restart the workflow.")
    saved = saved or {}
    if saved.get("workflow_phase") != "rq_breakdown_paused":
        return None, None, None, STALE_BREAKDOWN
    if run_id and ctx.get("run_id") != run_id:
        print(f"[State] Refused Task Breakdown click for {task_id}: card run {run_id}, "
              f"saved run {ctx.get('run_id')}")
        return None, None, None, STALE_BREAKDOWN
    saved_type = saved.get("workflow_type") or "research"
    if workflow_type and workflow_type != saved_type:
        return None, None, None, STALE_BREAKDOWN
    return ctx, saved, saved_type, None


def breakdown_waiting(task_id, run_id=None):
    """Whether the Task Breakdown of *run_id* is still paused for the user.

    Read-only: lets a card redrawn from history hide Proceed / Provide
    Feedback once its turn has moved on, instead of offering clicks that
    _paused_breakdown would refuse.
    """
    saved = get_workflow_state(task_id) or {}
    if saved.get("workflow_phase") != "rq_breakdown_paused":
        return False
    return not run_id or (saved.get("ctx") or {}).get("run_id") == run_id


def continue_after_rq_breakdown_stream(task_id, skip_data_download=False, force_data_download=False,
                                       run_id=None, workflow_type=None):
    """
    Resume the main workflow after the user clicks Proceed on the Task
    Breakdown card. Picks up from optional Data Download through Research
    Plan generation (which pauses again at its own approval gate).

    Yields:
        SSE-compatible update dicts
    """
    reset_cancellation()

    ctx, saved_state, workflow_type, error = _paused_breakdown(task_id, run_id, workflow_type)
    if error:
        yield {"type": "error", "error": error}
        return

    intent_result = saved_state.get("intent_result") or {}
    audit_records = saved_state.get("audit_records") or ctx.get("artifacts", {}).get("data_audit_records", [])
    autonomous_mode = saved_state.get("autonomous_mode", ctx.get("settings", {}).get("autonomous_mode", False))
    data_download_mode = saved_state.get("data_download_mode", ctx.get("settings", {}).get("data_download_mode", False))
    if skip_data_download:
        data_download_mode = False
        ctx.setdefault("settings", {})["data_download_mode"] = False
    elif force_data_download:
        # User turned Auto Data on from the missing-data prompt after submit.
        data_download_mode = True
        ctx.setdefault("settings", {})["data_download_mode"] = True
    enable_eda = saved_state.get("enable_eda", ctx.get("settings", {}).get("enable_eda", False))
    log_count = saved_state.get("log_count", len(ctx.get("logs", [])))

    interactive_code = saved_state.get("interactive_code", ctx.get("settings", {}).get("interactive_code", False))

    _is_data_retriever_only = bool(
        ctx.get("settings", {}).get("data_retriever_only")
    )
    # In Data Retriever mode the whole point is to fetch the identified
    # datasets — force the download path on regardless of the Auto Data
    # toggle that was saved at submit time. The frontend's Proceed button
    # also sets skip_data_download=False in this mode, so they agree.
    if _is_data_retriever_only and not skip_data_download:
        data_download_mode = True
        ctx.setdefault("settings", {})["data_download_mode"] = True
    if workflow_type == "spatial":
        yield from _post_rq_breakdown_pipeline_spatial(
            ctx=ctx,
            audit_records=audit_records,
            autonomous_mode=autonomous_mode,
            data_download_mode=data_download_mode,
            enable_eda=enable_eda,
            interactive_code=interactive_code,
            log_count=log_count,
        )
    else:
        yield from _post_rq_breakdown_pipeline(
            ctx=ctx,
            intent_result=intent_result,
            audit_records=audit_records,
            autonomous_mode=autonomous_mode,
            data_download_mode=data_download_mode,
            enable_eda=enable_eda,
            log_count=log_count,
        )


def revise_rq_breakdown_stream(task_id, feedback, run_id=None):
    """
    Re-run the Task Breakdown with the user's free-text feedback. The LLM
    revises only what the feedback asks about (surgical edit).

    Yields:
        llm_stream tokens under step="rq_breakdown" as the revised JSON
        streams back, then a final llm_update with step="rq_breakdown_complete"
        carrying the parsed revised breakdown in its data.rq_breakdown field.
    """
    reset_cancellation()

    ctx, _saved, workflow_type, error = _paused_breakdown(task_id, run_id)
    if error:
        yield {"type": "error", "error": error}
        return

    is_data_retriever_only = bool(
        ctx.get("settings", {}).get("data_retriever_only")
    )
    yield from helper.yield_status_update(
        "Revising data request based on feedback..."
        if is_data_retriever_only
        else "Revising task breakdown based on feedback...",
        "rq_breakdown",
    )

    rq_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: rq_queue.put(
        {"type": "llm_stream", "content": content, "step": "rq_breakdown"}
    )
    rq_error = [None]

    def _run_revise():
        try:
            if is_data_retriever_only:
                rq_agent.revise_data_intent_breakdown(ctx, feedback)
            else:
                rq_agent.revise_rq_breakdown(ctx, feedback)
        except Exception as e:
            rq_error[0] = e
        finally:
            rq_queue.put(None)

    t = threading.Thread(target=_with_flask_ctx(_run_revise), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = rq_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)

    if rq_error[0]:
        yield {"type": "error", "error": str(rq_error[0])}
        return

    dropped = _drop_requests_when_sufficient(ctx)
    if dropped:
        yield {"type": "llm_stream", "content": _dropped_requests_note(dropped), "step": "rq_breakdown"}

    revised = ctx.get("artifacts", {}).get("rq_breakdown", {})

    # Update the on-disk state snapshot so a browser reload after revision
    # sees the latest breakdown + buttons.
    saved_state = get_workflow_state(task_id) or {}
    saved_state["ctx"] = _serialize_state(ctx)
    save_workflow_state(task_id, saved_state)

    yield {
        "type": "llm_update",
        "content": "Research question breakdown revised.",
        "step": "rq_breakdown_complete",
        "data": {"rq_breakdown": revised, "task_id": task_id,
                 "run_id": ctx.get("run_id"), "workflow_type": workflow_type},
    }

    yield {"type": "complete", "message": "Task breakdown revision complete"}


# =====================================================================
#  SPATIAL ANALYSIS WORKFLOW HANDLER
# =====================================================================

def data_retriever_workflow_handler(message, file_paths=None, file_names=None, autonomous_mode=False, conversation_id=None):
    """
    Data Retriever WebUI handler — download data only, no analysis.

    Runs a lightweight intent parser (data_intent_breakdown) so multi-dataset
    messages are split into one entry per dataset, then pauses for the user
    to review. On Proceed, the shared post-breakdown pipeline runs LLM_Find
    for each entry and stops (data_retriever_only short-circuit).
    """
    file_paths = file_paths or []
    file_names = file_names or []

    datasets = [{"path": p, "name": n} for p, n in zip(file_paths, file_names)]
    ctx = init_workflow_context(rq=message, datasets=datasets, conversation_id=conversation_id)
    ctx["_cancel_event"] = _cancel_event
    ctx.setdefault("settings", {})["interactive_code"] = False
    ctx["settings"]["enable_eda"] = False
    ctx["settings"]["autonomous_mode"] = autonomous_mode
    ctx["settings"]["data_download_mode"] = True
    ctx["settings"]["data_retriever_only"] = True

    ctx.setdefault("artifacts", {}).setdefault("data_registry", [])
    ctx.setdefault("artifacts", {}).setdefault("data_audit_records", [])
    log_count = len(ctx.get("logs", []))

    if is_cancelled(): return

    # ── Data intent parsing: split the message into per-dataset requests ──
    yield from helper.yield_status_update(
        "Parsing your request to identify the datasets to download...",
        "rq_breakdown",
    )

    intent_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: intent_queue.put(
        {"type": "llm_stream", "content": content, "step": "rq_breakdown"}
    )
    intent_error = [None]

    def _run_intent_breakdown():
        try:
            rq_agent.data_intent_breakdown(ctx)
        except Exception as e:
            intent_error[0] = e
        finally:
            intent_queue.put(None)

    t = threading.Thread(target=_with_flask_ctx(_run_intent_breakdown), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = intent_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)

    if intent_error[0]:
        raise intent_error[0]

    yield from helper.yield_new_logs(ctx, log_count, "rq_breakdown")
    log_count = len(ctx.get("logs", []))

    rq_breakdown_data = ctx.get("artifacts", {}).get("rq_breakdown", {}) or {}
    if not rq_breakdown_data.get("data_requests"):
        rq_breakdown_data["data_requests"] = [message]
        ctx.setdefault("artifacts", {})["rq_breakdown"] = rq_breakdown_data

    task_id = ctx.get("task_id", "default")
    yield {
        "type": "llm_update",
        "content": (
            f"Identified {len(rq_breakdown_data['data_requests'])} dataset "
            f"request(s) from your message."
        ),
        "step": "rq_breakdown_complete",
        "data": {"rq_breakdown": rq_breakdown_data, "task_id": task_id,
                 "run_id": ctx.get("run_id"), "workflow_type": "spatial"},
    }

    if is_cancelled(): return

    # ── PAUSE for review (Proceed / Provide Feedback) ──
    # Resume routes through _post_rq_breakdown_pipeline_spatial, which
    # respects data_retriever_only and stops after download.
    _rq_breakdown_store_context(task_id, ctx)
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "audit_records": _serialize_state(
            ctx.get("artifacts", {}).get("data_audit_records", [])
        ),
        "autonomous_mode": autonomous_mode,
        "data_download_mode": True,
        "enable_eda": False,
        "log_count": log_count,
        "workflow_phase": "rq_breakdown_paused",
        "workflow_type": "spatial",
        "interactive_code": False,
    })
    # No instruction status here: it would permanently overwrite the card
    # title ("Identified N dataset request(s)...") with text that goes stale
    # after Proceed. The guidance is rendered inside the Data Availability
    # card next to its Proceed / Provide Feedback buttons.
    return


def spatial_analysis_workflow_handler(message, file_paths=None, file_names=None, interactive_code =False, enable_eda=False, autonomous_mode=False, data_download_mode=False, conversation_id=None):
    """
    spatial_analysis WebUI handler
    The full workflow for the WebUI:
        1. Data Auditing
        2. (optional) RQ breakdown + Data Download
        3. Geoprocessing workflow generation
        4. Operation execution
        5. Result
    """
    # Normalize None → [] so the no-upload (data_download_mode) path works
    file_paths = file_paths or []
    file_names = file_names or []

    # ── Initialise context ──────────────────────────────────────────
    datasets = [{"path": p, "name": n} for p, n in zip(file_paths, file_names)]
    ctx = init_workflow_context(rq=message, datasets=datasets, conversation_id=conversation_id)
    ctx["_cancel_event"] = _cancel_event
    ctx.setdefault("settings", {})["interactive_code"] = interactive_code
    ctx["settings"]["enable_eda"] = enable_eda
    ctx["settings"]["autonomous_mode"] = autonomous_mode
    ctx["settings"]["data_download_mode"] = data_download_mode
    log_count = 0



    # ── STEP 1: Data Audit (streamed progress) ───────────────────────
    # Skip the initial audit when no files were uploaded — the download-only
    # path will populate ctx["datasets"] from LLM_Find and audit those files
    # after they land.
    if file_paths:
        yield from helper.yield_status_update(
            f"Auditing {len(file_paths)} uploaded file(s)...", "data_audit"
        )
        # Stream audit progress via background thread + queue
        audit_queue = queue.Queue()
        ctx["_stream_callback"] = lambda content: audit_queue.put(
            {"type": "llm_stream", "content": content, "step": "data_audit"}
        )
        audit_error = [None]

        def _run_audit():
            try:
                data_agent.audit_datasets(ctx, {})
            except Exception as e:
                audit_error[0] = e
            finally:
                audit_queue.put(None)  # sentinel

        t = threading.Thread(target=_with_flask_ctx(_run_audit), daemon=True)
        t.start()

        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = audit_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()
        ctx.pop("_stream_callback", None)

        if audit_error[0]:
            raise audit_error[0]

        yield from helper.yield_new_logs(ctx, log_count, "data_audit")
        log_count = len(ctx.get("logs", []))

        audit_records = ctx.get("artifacts", {}).get("data_audit_records", [])
        yield {
            "type": "llm_update",
            "content": f"Data audit complete: {len(audit_records)} dataset(s) audited successfully.",
            "step": "data_audit_complete",
            "data": {"audit_records": audit_records},
        }

        # ── Initialize data registry from audit records ──────────────────
        all_sources = []
        seen_paths = set()
        for rec in audit_records:
            path = rec.get("path", "")
            if not path or not rec.get("ok"):
                continue
            norm = os.path.normpath(path)
            if norm in seen_paths:
                continue
            info = helper._peek_file(path)
            if info:
                info["source"] = "data_audit_record"
                info["from_step"] = "data_audit_record"
                all_sources.append(info)
                seen_paths.add(norm)

        ctx["artifacts"]["data_registry"] = all_sources

        registry_lines = [f"\nData registry initialized with {len(all_sources)} source(s):"]
        for s in all_sources:
            registry_lines.append(
                f"  -{s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']}"
            )
        registry_msg = "\n".join(registry_lines) + "\n"
        print(f"[Workflow] {registry_msg}")

        yield {"type": "llm_stream", "content": registry_msg, "step": "data_audit"}
        yield {
            "type": "llm_update",
            "content": f"Data registry initialized with {len(all_sources)} source(s).",
            "step": "data_audit_complete",
        }
    else:
        # No uploads — initialize empty data structures; the download branch
        # below is expected to fill them.
        ctx.setdefault("artifacts", {})["data_registry"] = []
        ctx.setdefault("artifacts", {}).setdefault("data_audit_records", [])
        audit_records = []

    if is_cancelled(): return

    # ── STEP: RQ Breakdown (data-needs check) + optional Data Download ─────
    # Always runs so the user is told when the data required for the question
    # is missing — even with Auto Data off. data_agent.data_download() also
    # reads its data_requests from ctx["artifacts"]["rq_breakdown"].
    # With Auto Data off and the uploaded data judged sufficient, the workflow
    # continues straight to geoprocessing without pausing.
    if True:
        # ── (a) RQ breakdown just to extract data_requests ──
        yield from helper.yield_status_update(
            "Analyzing task to identify data needs...", "rq_breakdown"
        )

        rq_queue = queue.Queue()
        ctx["_stream_callback"] = lambda content: rq_queue.put(
            {"type": "llm_stream", "content": content, "step": "rq_breakdown"}
        )
        rq_error = [None]

        def _run_rq_breakdown_sa():
            try:
                rq_agent.rq_breakdown(ctx)
            except Exception as e:
                rq_error[0] = e
            finally:
                rq_queue.put(None)  # sentinel

        t = threading.Thread(target=_with_flask_ctx(_run_rq_breakdown_sa), daemon=True)
        t.start()

        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = rq_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()
        ctx.pop("_stream_callback", None)

        if rq_error[0]:
            raise rq_error[0]

        dropped = _drop_requests_when_sufficient(ctx)
        if dropped:
            yield {"type": "llm_stream", "content": _dropped_requests_note(dropped), "step": "rq_breakdown"}

        yield from helper.yield_new_logs(ctx, log_count, "rq_breakdown")
        log_count = len(ctx.get("logs", []))

        rq_breakdown_data = ctx.get("artifacts", {}).get("rq_breakdown", {})
        task_id = ctx.get("task_id", "default")

        # Auto Data off: only stop at the review card when data is missing.
        _sufficiency = str(rq_breakdown_data.get("data_available_sufficiency") or "").strip()
        _data_missing = (not file_paths) or (
            bool(_sufficiency) and not _is_data_sufficient(
                _sufficiency, rq_breakdown_data.get("data_requests"))
        )
        if not data_download_mode and not _data_missing:
            yield {
                "type": "llm_update",
                "content": "Data check complete: the provided data covers this analysis.",
                "step": "rq_breakdown",
            }
        else:
            yield {
                "type": "llm_update",
                "content": "Task breakdown complete.",
                "step": "rq_breakdown_complete",
                "data": {"rq_breakdown": rq_breakdown_data, "task_id": task_id,
                 "run_id": ctx.get("run_id"), "workflow_type": "spatial"},
            }

            if is_cancelled(): return

            # ── PAUSE for Task Breakdown review (Proceed / Provide Feedback) ──
            _rq_breakdown_store_context(task_id, ctx)
            save_workflow_state(task_id, {
                "ctx": _serialize_state(ctx),
                "audit_records": _serialize_state(
                    ctx.get("artifacts", {}).get("data_audit_records", [])
                ),
                "autonomous_mode": autonomous_mode,
                "data_download_mode": data_download_mode,
                "enable_eda": enable_eda,
                "log_count": log_count,
                "workflow_phase": "rq_breakdown_paused",
                "workflow_type": "spatial",
                "interactive_code": interactive_code,
            })
            # No instruction status here: it would permanently overwrite the Task
            # Breakdown card's title with text that goes stale after Proceed. The
            # guidance is rendered inside the Data Availability card next to its
            # Proceed / Provide Feedback buttons (displayRqBreakdownSummary).
            return

    # ── STEP 2: Geoprocessing Workflow Generation (no data download needed) ──
    # When data_download_mode is OFF, skip straight to geoprocessing workflow.
    if is_cancelled(): return

    yield from _post_rq_breakdown_pipeline_spatial(
        ctx=ctx,
        audit_records=ctx.get("artifacts", {}).get("data_audit_records", []),
        autonomous_mode=autonomous_mode,
        data_download_mode=False,
        enable_eda=enable_eda,
        interactive_code=interactive_code,
        log_count=log_count,
    )


_INSUFFICIENT_RE = re.compile(
    r"\b(insufficient|inadequate|not (fully |entirely )?(sufficient|adequate|enough|available)"
    r"|lacks?|lacking|missing|partial(ly)?|additional data|no data|needs? (additional|more|extra))\b"
)
_SUFFICIENT_RE = re.compile(r"\b(sufficient|adequate|enough|covers?|suffices)\b")


def _is_data_sufficient(sufficiency, data_requests=None):
    """Classify the free-text data_available_sufficiency verdict.

    Mirrors isDataSufficient() in WebUI/script.js.
    """
    t = re.sub(r"^[^a-z]+", "", str(sufficiency or "").strip().lower())
    if not t:
        return False
    if re.match(r"yes\b", t):
        return True
    if re.match(r"no\b", t):
        return False
    if _INSUFFICIENT_RE.search(t):
        return False
    if _SUFFICIENT_RE.search(t):
        return True
    return not (isinstance(data_requests, list) and data_requests)


def _drop_requests_when_sufficient(ctx):
    """Clear data_requests the breakdown made despite judging the data sufficient.

    The breakdown prompt says data_requests must be null when the provided
    data is sufficient, but the model sometimes returns both a "sufficient"
    verdict and requests for data the user already has. With Auto Data on,
    Proceed downloads every listed request, so the same dataset was fetched
    again: a follow-up Spatial Analysis on data just retrieved in the same
    conversation re-downloaded it. The verdict wins; the requests are kept
    under dropped_data_requests for the record.

    Data Retriever mode is left alone (downloading is what it was asked for),
    as is a run with no datasets, where a "sufficient" verdict can't be right.

    Returns the dropped requests (empty list when nothing changed).
    """
    if ctx.get("settings", {}).get("data_retriever_only") or not ctx.get("datasets"):
        return []
    rq = ctx.get("artifacts", {}).get("rq_breakdown")
    if not isinstance(rq, dict):
        return []
    requests = rq.get("data_requests")
    if not (isinstance(requests, list) and requests):
        return []
    sufficiency = rq.get("data_available_sufficiency")
    if not sufficiency or not _is_data_sufficient(sufficiency, requests):
        return []

    rq["data_requests"] = None
    rq["dropped_data_requests"] = requests
    path = ctx.get("artifacts", {}).get("rq_breakdown_path")
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rq, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"[Workflow] Could not rewrite {path}: {e}")
    task_manager.log(ctx, f"Breakdown judged the provided data sufficient; "
                          f"dropped {len(requests)} data request(s): {requests}")
    return requests


def _dropped_requests_note(dropped):
    return ("\n\nYour data already covers this question, so nothing will be downloaded. "
            f"Skipped {len(dropped)} data request(s) the breakdown listed anyway:\n"
            + "".join(f"  - {r}\n" for r in dropped))


def _post_rq_breakdown_pipeline_spatial(
    ctx, audit_records, autonomous_mode, data_download_mode,
    enable_eda, interactive_code, log_count,
):
    """
    Everything that happens after the Task Breakdown pause point in the
    spatial-analysis workflow: optional data download, optional EDA,
    geoprocessing workflow generation, and the geoprocessing approval pause.
    """
    rq_breakdown_data = ctx.get("artifacts", {}).get("rq_breakdown", {})
    # User-provided data-source keys (from the Web UI) for this task.
    ctx["_data_source_keys"] = get_data_source_keys(ctx.get("task_id"))
    # User's private contributed data sources (handbook dirs) for this task.
    ctx["_user_handbook_dirs"] = get_user_handbook_dirs(ctx.get("task_id"))

    if data_download_mode:
        # ── (b) Data Download based on identified data_requests ──
        data_requests = rq_breakdown_data.get("data_requests") or []
        if data_requests:
            # Titled for what happens FIRST (source selection per request);
            # the data_download_complete event retitles this card at the end.
            yield from helper.yield_status_update(
                f"Selecting data source(s) for {len(data_requests)} data "
                f"request(s)...",
                "data_download",
            )

            _GEOSPATIAL_EXTS_REQ_SA = {"geojson", "json", "gpkg", "shp", "zip", "kml", "gml", "tif", "tiff", "nc", "img", "vrt"}
            _TABULAR_EXTS_REQ_SA = {"csv", "xlsx", "xls", "parquet"}

            def _build_request_files_event_sa(idx, data_request, result):
                files_payload = []
                if result and result.get("status") == "success":
                    # FileWriteTracker captures every write, so downloaded_files
                    # mixes the requested data with the script's own scratch --
                    # one PSU run produced 2 datasets and 35 .osm.xml tiles.
                    # Flagging them lets the UI lead with the data and fold the
                    # rest away, instead of showing 37 equal-looking results.
                    try:
                        from agents.data_agent.DataRetrieverAgent import (
                            classify_delivered_files)
                        _ds, _ = classify_delivered_files(
                            result.get("downloaded_files") or [])
                        _dataset_paths = set(_ds)
                    except Exception:
                        _dataset_paths = set(result.get("downloaded_files") or [])
                    for fpath in result.get("downloaded_files") or []:
                        try:
                            info = helper._peek_file(fpath) or {}
                        except Exception:
                            info = {}
                        file_basename = os.path.basename(fpath)
                        file_ext = os.path.splitext(file_basename)[1].lstrip(".").lower()
                        try:
                            file_size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                        except Exception:
                            file_size = 0
                        # Stage now so /api/uploads can serve this file the moment
                        # the request finishes (enables right-click Download on a
                        # partially-completed batch).
                        # is_dataset rides along in the stored audit record so a
                        # reloaded conversation can keep the script's scratch
                        # files out of the Data Layers panel, as the live run does.
                        _stage_downloaded_file_for_ui(
                            ctx, fpath, file_basename, file_ext, file_size,
                            {**info, "is_dataset": fpath in _dataset_paths})
                        files_payload.append({
                            "isDataset": fpath in _dataset_paths,
                            "file": info.get("file") or os.path.splitext(file_basename)[0],
                            "fileName": file_basename,
                            "displayName": os.path.splitext(file_basename)[0],
                            "fileType": file_ext,
                            "fileSize": file_size,
                            "path": fpath,
                            "shape": info.get("shape", [0, 0]),
                            "isGeospatial": file_ext in _GEOSPATIAL_EXTS_REQ_SA,
                            "isTabular": file_ext in _TABULAR_EXTS_REQ_SA,
                            "conversationId": ctx.get("task_id"),
                        })
                status_str = "success" if files_payload else "failed"
                return {
                    "type": "data_request_files",
                    "step": (f"data_download_req_{idx}_cont"
                             if (ctx.get('_synth_done_req') == idx
                                 or ctx.get('_confirm_done_req') == idx)
                             else f"data_download_req_{idx}"),
                    "request": str(data_request),
                    "status": status_str,
                    "files": files_payload,
                    # WHY there are no files, when there are none. The tiles
                    # are the only part of a request that is redrawn after a
                    # reload (the review gate is not, once answered), so a
                    # request the source could not serve used to reload as
                    # a bare "No files downloaded" with the reason gone.
                    "error": str((result or {}).get("error") or ""),
                    "data_unavailable": bool((result or {}).get("data_unavailable")),
                }

            download_queue = queue.Queue()
            # Step id for this request's stream. After the user answers a
            # confirmation prompt (or a skill synthesis runs) it switches to a
            # CONTINUATION card (data_download_req_N_cont) so the resumed work
            # — code generation, execution — renders BELOW the confirmation
            # panel / synthesis card instead of in the original request card
            # above them.
            def _req_step():
                i = ctx.get('_current_download_req', 0)
                return (f"data_download_req_{i}_cont"
                        if (ctx.get('_synth_done_req') == i
                            or ctx.get('_confirm_done_req') == i)
                        else f"data_download_req_{i}")

            ctx["_stream_callback"] = lambda content: download_queue.put(
                {"type": "llm_stream", "content": content, "step": _req_step()}
            )

            # Need-driven skill synthesis streams to its OWN card (its own step
            # id); it also flags the request so everything streamed after the
            # synthesis goes to the continuation card.
            def _synth_stream_cb(content):
                ctx["_synth_done_req"] = ctx.get('_current_download_req', 0)
                download_queue.put({
                    "type": "llm_stream", "content": content,
                    "step": f"skill_synth_req_{ctx.get('_current_download_req', 0)}",
                })
            ctx["_synthesis_stream_callback"] = _synth_stream_cb
            ctx["_register_rerun_context"] = (
                lambda idx, info: save_rerun_context(ctx.get("task_id"), idx, info))
            ctx["_request_done_callback"] = lambda idx, data_request, result: download_queue.put(
                _build_request_files_event_sa(idx, data_request, result)
            )

            # Interactive source-select callback (see research pipeline above)
            source_select_response_queue = queue.Queue()
            _skip_source_select_sa = [False]
            # Installed in BOTH interactive and autonomous mode. Autonomous mode
            # proceeds without confirmation, but still pauses to collect a
            # missing API key — a blocker it can't resolve on its own (see the
            # missing-keys check below). Covers spatial-analysis and data-
            # retriever runs, which both resume through this pipeline.
            set_source_select_response_queue(source_select_response_queue)

            def _source_select_cb_sa(payload):
                missing_keys = payload.get("required_keys") or []
                is_credential_warning = bool(payload.get("credential_error"))
                # Skip the confirmation prompt only when nothing is blocking:
                # neither autonomous mode nor a prior "Always Proceed" can supply
                # a missing API key, so whenever one is missing we still prompt.
                # A credential warning (the entered key was rejected) is ALWAYS a
                # hard blocker — prompt even in autonomous / Always-Proceed mode.
                if not missing_keys and not is_credential_warning and (
                    autonomous_mode or _skip_source_select_sa[0]
                ):
                    # No blocker to prompt for. If the source itself is
                    # unrecognized (no handbook exists) and nobody is at the
                    # dropdown to click "Generate & Use New Skill", LLM_Find's
                    # normal "proceed" handling for an unrecognized source
                    # would silently substitute the closest EXISTING handbook
                    # instead — defeating automatic source onboarding
                    # entirely. Trigger real generation instead, the same
                    # action a human would pick, so autonomous runs actually
                    # exercise the Handbook Generator. Skipped when this
                    # payload IS the post-synthesis confirmation (a skill was
                    # just generated) — there we want to proceed with it, not
                    # generate again.
                    if not payload.get("is_recognized", True) and not payload.get("synthesized"):
                        gen_query = (
                            (payload.get("ideal_source") or "").strip()
                            or str(payload.get("llm_choice") or "").strip()
                        )
                        if gen_query:
                            return {
                                "action": "generate",
                                "generate_query": gen_query,
                                "generate_website": "",
                            }
                    return {"action": "proceed"}
                _tid = ctx.get("task_id")
                # Drop any stale answer (disk + in-memory) so we only accept
                # a response made against THIS prompt. Spatial-analysis runs
                # commonly issue several data requests, so a leftover queued
                # answer from a prior request would otherwise auto-confirm the
                # next prompt without the user seeing it.
                clear_source_select_answer(_tid)
                _drain_source_select_queue(source_select_response_queue)
                # The post-synthesis confirmation anchors under the "Generating
                # New Skill" card (its own step id) so it appears where the
                # synthesis just finished; the normal prompt anchors under the
                # data-request card as before.
                _prompt_step = (
                    f"skill_synth_req_{ctx.get('_current_download_req', 0)}"
                    if payload.get("synthesized")
                    else _req_step()
                )
                download_queue.put({
                    "type": "data_source_select_approval",
                    "step": _prompt_step,
                    "task_id": _tid,
                    "data_request": payload.get("data_request"),
                    "llm_choice": payload.get("llm_choice"),
                    "llm_explanation": payload.get("llm_explanation", ""),
                    "ideal_source": payload.get("ideal_source", ""),
                    "ideal_source_reasoning": payload.get("ideal_source_reasoning", ""),
                    "data_to_download": payload.get("data_to_download", ""),
                    "confidence": payload.get("confidence", 0),
                    "is_recognized": payload.get("is_recognized", False),
                    "available_sources": payload.get("available_sources") or [],
                    "required_keys": missing_keys,
                    "caveats": payload.get("caveats", ""),
                    "credential_error": is_credential_warning,
                    "credential_message": payload.get("credential_message", ""),
                    # Post-synthesis confirmation variant: a new skill was just
                    # generated; ask the user to proceed before downloading.
                    "synthesized": bool(payload.get("synthesized")),
                    "synthesized_status": payload.get("synthesized_status", ""),
                    "synthesized_note": payload.get("synthesized_note", ""),
                })
                # Poll both the in-memory queue (local) and the disk file
                # (cross-process, for multi-worker hosting), and advertise the
                # wait so a reopened page can still answer it. See the research
                # pipeline callback above for the full rationale.
                write_pending_prompt(_tid, "source_select", {"step": _prompt_step})
                response = None
                try:
                    while response is None:
                        if is_cancelled():
                            return None
                        try:
                            response = source_select_response_queue.get(timeout=0.3)
                        except queue.Empty:
                            response = pop_source_select_answer(_tid)
                        touch_pending_prompt(_tid)
                finally:
                    clear_pending_prompt(_tid)
                clear_source_select_answer(_tid)
                # A prompt was shown and answered: route the rest of this
                # request's stream to its CONTINUATION card so the resumed work
                # renders below the (frozen) confirmation panel. Not set on the
                # autonomous early-return above — no panel was shown there.
                ctx["_confirm_done_req"] = ctx.get('_current_download_req', 0)
                # Persist any keys the user entered into ctx so later data
                # requests for the same (or another) key-gated source reuse them
                # instead of re-prompting — important in autonomous mode.
                if isinstance(response, dict):
                    _entered = response.get("data_source_keys") or {}
                    if isinstance(_entered, dict) and _entered:
                        ctx.setdefault("_data_source_keys", {}).update(
                            {k: v for k, v in _entered.items() if v}
                        )
                if isinstance(response, dict) and response.get("action") == "always_proceed":
                    _skip_source_select_sa[0] = True
                    # Preserve any keys the user entered with Always Proceed.
                    return {"action": "proceed",
                            "data_source_keys": response.get("data_source_keys") or {}}
                return response

            ctx["_source_select_callback"] = _source_select_cb_sa

            # ── Post-download review (interactive runs only) ───────────
            # Installed only when NOT autonomous: data_download() skips the
            # checkpoint entirely when no callback is present, so an
            # autonomous run is unaffected.
            if not autonomous_mode:
                download_review_queue = queue.Queue()
                set_download_review_response_queue(download_review_queue)

                def _download_review_cb_sa(payload):
                    _tid = ctx.get("task_id")
                    # Drop any stale answer (disk + queue) so we only accept a
                    # response made against THIS prompt — same reasoning as the
                    # source-select callback above.
                    clear_download_review_answer(_tid)
                    while True:
                        try:
                            download_review_queue.get_nowait()
                        except queue.Empty:
                            break
                    idx = payload.get("request_index", 0)
                    event = {
                        "type": "data_download_review",
                        "step": f"data_download_req_{idx}_cont",
                        "task_id": _tid,
                    }
                    event.update(payload)
                    download_queue.put(event)
                    write_pending_prompt(_tid, "download_review",
                                         {"step": event.get("step", "")})
                    response = None
                    try:
                        while response is None:
                            if is_cancelled():
                                return None
                            try:
                                response = download_review_queue.get(timeout=0.3)
                            except queue.Empty:
                                response = pop_download_review_answer(_tid)
                            touch_pending_prompt(_tid)
                    finally:
                        clear_pending_prompt(_tid)
                    clear_download_review_answer(_tid)
                    return response

                ctx["_download_review_callback"] = _download_review_cb_sa

            download_error = [None]
            download_result = [None]

            def _run_download_sa():
                try:
                    download_result[0] = data_agent.data_download(ctx)
                except Exception as e:
                    download_error[0] = e
                finally:
                    download_queue.put(None)  # sentinel

            t = threading.Thread(target=_with_flask_ctx(_run_download_sa), daemon=True)
            t.start()

            while True:
                if is_cancelled():
                    t.join(timeout=3)
                    ctx.pop("_stream_callback", None)
                    ctx.pop("_synthesis_stream_callback", None)
                    ctx.pop("_request_done_callback", None)
                    ctx.pop("_source_select_callback", None)
                    ctx.pop("_download_review_callback", None)
                    clear_source_select_response_queue()
                    clear_download_review_response_queue()
                    return
                try:
                    item = download_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                yield item

            t.join()
            ctx.pop("_stream_callback", None)
            ctx.pop("_synthesis_stream_callback", None)
            ctx.pop("_request_done_callback", None)
            ctx.pop("_source_select_callback", None)
            ctx.pop("_download_review_callback", None)
            clear_source_select_response_queue()
            clear_download_review_response_queue()

            if download_error[0]:
                raise download_error[0]

            yield from helper.yield_new_logs(ctx, log_count, "data_download")
            log_count = len(ctx.get("logs", []))

            # ── Collect newly downloaded files, add to ctx["datasets"] ──
            data_downloads = download_result[0] or []

            # ── Interactive pause if any request failed ──
            if not autonomous_mode:
                outcome = yield from _handle_failed_downloads_interactive(
                    ctx, data_downloads, is_cancelled=is_cancelled
                )
                if outcome in ("abort", "abort_single"):
                    yield {
                        "type": "error",
                        "error": (
                            "Workflow stopped — you aborted the only data request."
                            if outcome == "abort_single"
                            else "Workflow aborted by user after data download failures."
                        ),
                    }
                    return

            existing_paths = {
                os.path.normpath(d["path"])
                for d in ctx.setdefault("datasets", [])
                if d.get("path")
            }
            new_files = []
            for result in data_downloads:
                if not result or result.get("status") != "success":
                    continue
                for fpath in _delivered_dataset_paths(result):
                    norm = os.path.normpath(fpath)
                    if norm in existing_paths:
                        continue
                    ctx["datasets"].append({
                        "path": fpath,
                        "name": os.path.splitext(os.path.basename(fpath))[0],
                    })
                    existing_paths.add(norm)
                    new_files.append(fpath)

            summary_files = []
            if new_files:
                # Refresh audit records to include the new files so the rest
                # of the pipeline (EDA, geoprocessing workflow generation)
                # sees them as regular inputs.
                try:
                    data_agent.audit_datasets(ctx, {})
                except Exception as e:
                    print(f"[Workflow] Warning: re-audit after download failed: {e}")

                # Copy downloaded files into the conversation's uploads dir so
                # they can be served via /api/uploads/<conv_id>/<file> and
                # surface in the UI as regular data layers.
                import shutil
                from WebUI.conversation_db import add_conversation_file
                webui_base = os.path.dirname(os.path.abspath(__file__))
                task_id = ctx.get("task_id")
                conv_upload_dir = None
                if task_id:
                    conv_upload_dir = os.path.join(webui_base, "uploads", task_id)
                    os.makedirs(conv_upload_dir, exist_ok=True)

                # Append new files to ctx["artifacts"]["data_registry"]
                artifacts = ctx.setdefault("artifacts", {})
                data_registry = artifacts.get("data_registry")
                if not isinstance(data_registry, list):
                    data_registry = []
                    artifacts["data_registry"] = data_registry
                registry_paths = {
                    os.path.normpath(s["path"])
                    for s in data_registry if s.get("path")
                }
                registry_lines = []

                GEOSPATIAL_EXTS = {"geojson", "json", "gpkg", "shp", "zip", "kml", "gml", "tif", "tiff", "nc", "img", "vrt"}
                TABULAR_EXTS = {"csv", "xlsx", "xls", "parquet"}

                for fpath in new_files:
                    if os.path.normpath(fpath) in registry_paths:
                        continue
                    info = helper._peek_file(fpath)
                    if not info:
                        # Unrecognised format — still register the file with minimal metadata
                        info = {
                            "file": os.path.splitext(os.path.basename(fpath))[0],
                            "path": fpath,
                            "shape": [0, 0],
                            "columns": {},
                            "head": {},
                        }
                    info["source"] = "downloaded"
                    info["from_step"] = "data_download"
                    data_registry.append(info)
                    registry_paths.add(os.path.normpath(fpath))
                    registry_lines.append(
                        f"  -{info['file']} (downloaded) | path: {info['path']} | shape: {info['shape']}"
                    )

                    # Copy file into uploads/<conv_id>/ and register in DB
                    file_basename = os.path.basename(fpath)
                    file_ext = os.path.splitext(file_basename)[1].lstrip(".").lower()
                    display_name = os.path.splitext(file_basename)[0]
                    served_name = file_basename
                    try:
                        file_size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                    except Exception:
                        file_size = 0

                    # Stage into uploads/<conv_id>/ and register in the DB. The
                    # helper also copies a shapefile's .dbf/.shx/.prj/.cpg, which
                    # are no longer in this list now that scratch is filtered out.
                    if conv_upload_dir:
                        _stage_downloaded_file_for_ui(
                            ctx, fpath, file_basename, file_ext, file_size,
                            {**info, "is_dataset": True})

                    summary_files.append({
                        "file": info["file"],
                        "fileName": served_name,
                        "displayName": display_name,
                        "fileType": file_ext,
                        "fileSize": file_size,
                        "path": info["path"],
                        "shape": info["shape"],
                        "isGeospatial": file_ext in GEOSPATIAL_EXTS,
                        "isTabular": file_ext in TABULAR_EXTS,
                        "conversationId": task_id,
                    })

                summary = (
                    f"Downloaded {len(new_files)} new file(s) and added to data registry:\n"
                    + "\n".join(registry_lines)
                )
            else:
                summary = "No new files were downloaded. All data requests failed after retries."

            # Always yield the summary so the UI panel renders,
            # whether or not any files were downloaded.
            yield {
                "type": "data_download_summary",
                "content": summary,
                "step": "data_download",
                "files": summary_files,
            }

            yield {
                "type": "llm_update",
                "content": f"Data download complete: {len(new_files)} new file(s) added.",
                "step": "data_download_complete",
                "data": {
                    "downloaded_files": new_files,
                    "data_registry": ctx.get("artifacts", {}).get("data_registry", []),
                },
            }

            if is_cancelled(): return
        else:
            yield {
                "type": "llm_update",
                "content": "Data download mode is on, but no data requests were generated from the task.",
                "step": "data_download_skipped",
            }

    # Data Retriever mode: stop after download — no EDA, no geoprocessing.
    if ctx.get("settings", {}).get("data_retriever_only"):
        yield {
            "type": "result",
            "response": "Data retrieval complete.",
            "result_type": "data_retriever_complete",
        }
        return

    # ── STEP 2(optional): EDA & Preprocessing (agentic, streamed) ─────
    if enable_eda:
        yield from helper.yield_status_update(
            "Performing exploratory data analysis and preprocessing...", "eda"
        )

        eda_queue = queue.Queue()
        eda_approval_queue = queue.Queue()  # tool → workflow (approval requests)
        eda_response_queue = queue.Queue()  # workflow ← frontend (user responses)
        set_eda_response_queue(eda_response_queue)  # register globally for API endpoint

        ctx["_stream_callback"] = lambda content: eda_queue.put(
            {"type": "llm_stream", "content": content, "step": "eda"}
        )
        ctx["_approval_queue"] = eda_approval_queue
        ctx["_response_queue"] = eda_response_queue

        eda_error = [None]

        def _run_eda():
            try:
                data_agent.eda(ctx)
            except Exception as e:
                eda_error[0] = e
            finally:
                eda_queue.put(None)  # sentinel

        t = threading.Thread(target=_with_flask_ctx(_run_eda), daemon=True)
        t.start()

        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                ctx.pop("_approval_queue", None)
                ctx.pop("_response_queue", None)
                clear_eda_response_queue()
                return

            # Check for approval requests from the tool
            try:
                approval_item = eda_approval_queue.get_nowait()
                # Yield approval request to frontend — the tool thread is now
                # blocked on eda_response_queue.get() waiting for user response.
                yield approval_item
            except queue.Empty:
                pass

            try:
                item = eda_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()
        ctx.pop("_stream_callback", None)
        ctx.pop("_approval_queue", None)
        ctx.pop("_response_queue", None)
        clear_eda_response_queue()

        if eda_error[0]:
            raise eda_error[0]

        yield from helper.yield_new_logs(ctx, log_count, "eda")
        log_count = len(ctx.get("logs", []))

        # Collect EDA artifacts (plots, cleaned data, reports) from output dir
        eda_artifacts = []
        eda_output_dir = resolve_output_dir(ctx)
        eda_artifact_exts = (".png", ".jpg", ".csv", ".gpkg", ".json", ".html", ".geojson")
        if os.path.isdir(eda_output_dir):
            for fname in os.listdir(eda_output_dir):
                fpath = os.path.join(eda_output_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in eda_artifact_exts:
                    continue
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    fsize = 0
                try:
                    rel = os.path.relpath(fpath, settings.output_dir).replace("\\", "/")
                except ValueError:
                    rel = fname
                eda_artifacts.append({
                    "path": fpath,
                    "relative_path": rel,
                    "filename": fname,
                    "size": fsize,
                    "type": ext.lstrip("."),
                })

        eda_summary = ctx.get("artifacts", {}).get("eda_report_data", {}).get("summary", "")
        if eda_artifacts:
            yield {
                "type": "step_result",
                "step": "eda",
                "step_index": "EDA",
                "step_description": "Exploratory Data Analysis & Preprocessing",
                "status": "completed",
                "output": eda_summary,
                "artifacts": eda_artifacts,
            }

        # Store ctx so /api/eda-explore can access it later
        set_latest_workflow_ctx(ctx)

        yield from helper.yield_status_update(
            "EDA and preprocessing complete.", "eda_complete"
        )
    # else: EDA silently skipped when disabled

    if is_cancelled(): return

    # ── STEP 4: Geoprocessing Workflow Generation ────────────────
    yield from helper.yield_status_update(
        "Generating geoprocessing workflow...", "geoprocessing_workflow_generation"
    )

    chunk_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": "geoprocessing_workflow"}
    )
    ctx["_event_callback"] = lambda event: chunk_queue.put(event)
    graph_error = [None]

    def _generate_geoprocessing_workflow():
        try:
            geoprocessing_workflow_generator.generate_geoprocessing_workflow(ctx)
        except Exception as e:
            graph_error[0] = e
        finally:
            chunk_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_generate_geoprocessing_workflow), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_event_callback", None)

    if graph_error[0]:
        raise graph_error[0]

    yield from helper.yield_new_logs(ctx, log_count, "geoprocessing_workflow_generation")
    log_count = len(ctx.get("logs", []))

    # ── Build visualization HTML files ──
    task_id = ctx.get("task_id", "default")
    task_output_dir = resolve_output_dir(ctx)
    viz_dir = os.path.join(task_output_dir, "research_plan_viz")
    os.makedirs(viz_dir, exist_ok=True)

    # 1) Plan view: prose explanation wrapped in HTML
    explanation = ctx["artifacts"].get("geoprocessing_workflow_explanation", "")
    plan_html = _wrap_explanation_as_html(explanation, ctx.get("rq", ""))
    plan_path = os.path.join(viz_dir, f"research_plan_{task_id}.html")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_html)

    # 2) Geoprocessing DAG view: load GraphML and render D3 force-directed DAG.
    # Resolve via the shared helper so a stale/foreign stored path (different
    # worker or host) falls back to the deterministic per-task location.
    graph_file = geoprocessing_workflow_generator.resolve_graphfile(ctx)
    geoprocessing_path = os.path.join(viz_dir, f"research_plan_geoprocessing_{task_id}.html")
    print(f"[DEBUG] graph_file = {graph_file}, exists = {os.path.exists(graph_file) if graph_file else 'N/A'}")
    print(f"[DEBUG] geoprocessing_path = {geoprocessing_path}")
    if graph_file and os.path.exists(graph_file):
        import networkx as nx
        G = nx.read_graphml(graph_file)
        helper.show_dag_geoprocessing_workflow(
            G, output_path=geoprocessing_path,
            title="Geoprocessing Workflow",
            research_question=ctx.get("rq", ""),
        )
        print(f"[DEBUG] DAG HTML written: {os.path.exists(geoprocessing_path)}")
    else:
        print(f"[DEBUG] graph_file not found, falling back to explanation HTML")
        # Fallback: save explanation as geoprocessing view too
        with open(geoprocessing_path, "w", encoding="utf-8") as f:
            f.write(plan_html)

    # Store ctx for the feedback loop
    research_plan_generator.store_context(task_id, ctx)

    yield {
        "type": "llm_update",
        "content": "Geoprocessing workflow generated successfully.",
        "step": "geoprocessing_workflow_generation",
    }

    # Send geoprocessing workflow visualization to frontend (single event; tabs handle view switching)
    yield {
        "type": "geoprocessing_workflow_viz",
        "viz_id": task_id,
    }

    yield {
        "type": "llm_update",
        "content": "Geoprocessing workflow ready. Please review and approve to continue.",
        "step": "geoprocessing_workflow_summary",
    }

    # ── Save state and PAUSE for user approval ─────────────────────
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "audit_records": _serialize_state(audit_records),
        "log_count": log_count,
    })

    if autonomous_mode:
        result_msg = (
            "Geoprocessing workflow generated. "
            "Autonomous mode — auto-approving..."
        )
    else:
        result_msg = (
            "Geoprocessing workflow generated. "
            "Please review the workflow above and click "
            "'Approve' to proceed with operation execution, or "
            "'Provide Feedback' to revise."
        )

    yield {
        "type": "result",
        "response": result_msg,
        "result_type": "geoprocessing_workflow_pending_approval",
        "task_id": task_id,
    }

    # NOTE: Do NOT auto-continue here in autonomous mode. The frontend
    # auto-clicks the "Approve Workflow" button (see
    # spatial_analysis_displayWorkflowViz in script.js), which POSTs to
    # /api/spatial-analysis/approve/<task_id> and runs
    # spatial_analysis_continue_after_approval_stream. Calling it inline here
    # as well caused the operation/assembly code to be generated and executed
    # TWICE. This mirrors the research-plan flow, which also pauses on the
    # pending-approval result and lets the frontend drive continuation.


# =====================================================================
#  SPATIAL ANALYSIS: CONTINUE AFTER APPROVAL
# =====================================================================

def spatial_analysis_continue_after_approval_stream(task_id):
    """
    Continue the spatial analysis workflow after the user approves the
    geoprocessing workflow.  Runs operation code generation and execution.

    Yields:
        SSE-compatible update dicts
    """
    from agents.task_manager.spatial_analysis_executor import SpatialAnalysisExecutor

    # Reset cancellation flag for this continuation
    reset_cancellation()

    # ── Restore context ────────────────────────────────────────────
    ctx = _plan_context(task_id)
    if ctx:
        ctx["_cancel_event"] = _cancel_event

    # Fallback: restore from disk-saved workflow state
    if not ctx:
        saved_state = get_workflow_state(task_id)
        if saved_state and saved_state.get("ctx"):
            ctx = saved_state["ctx"]
            ctx["_cancel_event"] = _cancel_event
            if "artifacts" not in ctx:
                ctx["artifacts"] = {}
            # Re-initialize settings from config if missing or serialized
            if not isinstance(ctx.get("settings"), dict) or not ctx["settings"]:
                ctx["settings"] = load_configurations()
            # Ensure logs list exists
            ctx.setdefault("logs", [])
            # Set task_id and per-task output directory
            ctx["task_id"] = task_id
            if "output_dir" not in ctx:
                task_output_dir = os.path.join(settings.output_dir, task_id)
                os.makedirs(task_output_dir, exist_ok=True)
                ctx["output_dir"] = task_output_dir
            # Store back so subsequent calls find it
            research_plan_generator.store_context(task_id, ctx)
            print(f"[State] Restored spatial analysis context for task {task_id} from disk")

    if not ctx:
        yield {
            "type": "error",
            "error": f"No workflow context found for task {task_id}. "
                     "Please restart the workflow.",
        }
        return

    # ── Cross-worker artifact reconciliation (PythonAnywhere) ──────────
    # The in-memory context store is per-worker, so an approval handled by a
    # different worker than the one that ran generation can return a STALE ctx
    # (e.g. from the rq-breakdown phase) that lacks the geoprocessing artifacts —
    # while the authoritative checkpoint written at the approval pause has them
    # on the shared disk. Read that checkpoint *directly from disk* (bypassing
    # the per-worker in-memory cache in get_workflow_state) and backfill only the
    # artifacts the live ctx is missing, so code generation can read
    # ctx["artifacts"]["geoprocessing_workflow_code"] without a KeyError. Gaps
    # only: live in-memory objects always win over the serialized disk copy.
    try:
        state_file = _get_state_file(task_id)
        if state_file and os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                disk_state = json.load(f)
            disk_artifacts = (disk_state.get("ctx") or {}).get("artifacts") or {}
            live_artifacts = ctx.setdefault("artifacts", {})
            backfilled = [k for k in disk_artifacts if k not in live_artifacts]
            for key in backfilled:
                live_artifacts[key] = disk_artifacts[key]
            if backfilled:
                print(f"[State] Backfilled {len(backfilled)} artifact(s) from disk "
                      f"checkpoint for task {task_id}: {backfilled}")
    except Exception as e:
        print(f"[State] Warning: could not reconcile artifacts from disk: {e}")

    log_count = len(ctx.get("logs", []))

    yield from helper.yield_status_update(
        "Geoprocessing workflow approved. Starting operation code generation...",
        "geoprocessing_approved",
    )

    if is_cancelled(): return

    # ── Guard: ensure the geoprocessing workflow graph exists ──────────
    # This continuation runs in a separate request from generation (after the
    # approval pause), so a context restored from disk may carry a stale graph
    # path (absolute, from another machine), or — rarely — no usable graph at
    # all. First try to resolve/repair the path to the deterministic per-task
    # location (the common case after a worker/host change); only if the
    # .graphml is genuinely absent everywhere do we re-run generation once so
    # the user recovers without restarting.
    # Resolves the stored path or the deterministic per-task location, repairing
    # the artifact in place; only returns False when the .graphml is truly gone.
    def _graph_artifact_ready(c):
        return bool(geoprocessing_workflow_generator.resolve_graphfile(c))

    if not _graph_artifact_ready(ctx):
        yield from helper.yield_status_update(
            "Geoprocessing workflow graph was missing — regenerating it before "
            "code generation...",
            "geoprocessing_workflow_regeneration",
        )

        regen_queue = queue.Queue()
        ctx["_stream_step"] = "geoprocessing_workflow"
        ctx["_stream_callback"] = lambda content: regen_queue.put(
            {"type": "llm_stream", "content": content, "step": "geoprocessing_workflow"}
        )
        ctx["_event_callback"] = lambda event: regen_queue.put(event)
        regen_error = [None]

        def _regenerate_workflow():
            try:
                geoprocessing_workflow_generator.generate_geoprocessing_workflow(ctx)
            except Exception as e:
                regen_error[0] = e
            finally:
                regen_queue.put(None)  # sentinel

        rt = threading.Thread(target=_with_flask_ctx(_regenerate_workflow), daemon=True)
        rt.start()

        while True:
            if is_cancelled():
                rt.join(timeout=3)
                ctx.pop("_stream_callback", None)
                ctx.pop("_event_callback", None)
                ctx.pop("_stream_step", None)
                return
            try:
                item = regen_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        rt.join()
        ctx.pop("_stream_callback", None)
        ctx.pop("_event_callback", None)
        ctx.pop("_stream_step", None)

        # Persist so a subsequent restore finds the freshly built artifact.
        research_plan_generator.store_context(task_id, ctx)

        if regen_error[0] or not _graph_artifact_ready(ctx):
            detail = f": {regen_error[0]}" if regen_error[0] else ""
            yield {
                "type": "llm_update",
                "content": (
                    "Could not regenerate the geoprocessing workflow graph"
                    f"{detail}. Please restart the workflow and try again."
                ),
                "step": "operation_code_generation_error",
            }
            return

        yield from helper.yield_new_logs(ctx, log_count, "geoprocessing_workflow_regeneration")
        log_count = len(ctx.get("logs", []))

        if is_cancelled(): return

    # ── Operation Code Generation + Assembly (streamed, per-operation cards) ──
    op_queue = queue.Queue()
    ctx["_stream_step"] = "operation_code_generation"
    ctx["_stream_callback"] = lambda content: op_queue.put(
        {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "operation_code_generation")}
    )
    ctx["_event_callback"] = lambda event: op_queue.put(event)

    op_error = [None]
    op_results = [None]
    executor_ref = [None]

    def _run_operations():
        try:
            executor = SpatialAnalysisExecutor(ctx)
            executor_ref[0] = executor
            op_results[0] = executor.get_LLM_responses_for_operations(ctx)
            # Now generate assembly code (lands on its own card)
            executor.get_LLM_assembly_response(ctx)

            # Execute the assembly code
            exec_step_tag = "assembly_execution"
            ctx["_stream_step"] = exec_step_tag
            event_cb = ctx.get("_event_callback")
            if event_cb:
                event_cb({
                    "type": "status",
                    "message": "Executing assembly program...",
                    "step": exec_step_tag,
                })

            exec_result = executor.execute_assembly_code(ctx)

            if event_cb:
                created_files = exec_result.get("created_files", [])
                # Only surface files inside the per-task `final_outputs/`
                # subfolder as WebUI tiles. Intermediates (audit summaries,
                # cleaned tables, working layers) live alongside in
                # OUTPUT_DIR but are kept off the tile grid. If the LLM
                # ignored the convention and wrote nothing into
                # final_outputs/, fall back to showing every created file
                # so the user is never left with zero tiles.
                task_output_dir = ctx.get("output_dir", "")
                final_marker = os.path.normcase(os.path.normpath(
                    os.path.join(task_output_dir, "final_outputs")
                )) + os.sep
                final_files = [
                    fp for fp in created_files
                    if os.path.normcase(os.path.normpath(fp)).startswith(final_marker)
                ]
                files_for_tiles = final_files if final_files else created_files
                # Analysis outputs are often written in a projected CRS
                # (metric buffering etc.); normalize to EPSG:4326 so the Map
                # Viewer can render them.
                _ensure_wgs84_outputs(files_for_tiles)
                if exec_result.get("status") == "completed":
                    _stage_spatial_results(ctx, files_for_tiles)

                # Convert raw file paths to artifact objects for the UI.
                # `relative_path` MUST be set so the WebUI's previewArtifact()
                # can build /api/artifacts/<relative_path> — without it the
                # tile click results in /api/artifacts/undefined and the
                # Results & Artifacts panel shows nothing.
                artifact_objects = []
                for fp in files_for_tiles:
                    try:
                        file_size = os.path.getsize(fp)
                    except OSError:
                        file_size = 0
                    filename = os.path.basename(fp)
                    ext = os.path.splitext(filename)[1].lstrip('.').lower()
                    try:
                        rel = os.path.relpath(fp, settings.output_dir).replace("\\", "/") if task_output_dir else filename
                    except ValueError:
                        # Different drive on Windows — fall back to basename
                        rel = filename
                    artifact_objects.append({
                        "filename": filename,
                        "path": fp,
                        "relative_path": rel,
                        "type": ext,
                        "size": file_size,
                    })
                event_cb({
                    "type": "step_result",
                    "step": exec_step_tag,
                    "step_index": "Execution",
                    "step_description": "Assembly code execution",
                    "objective_key": "assembly",
                    "status": exec_result.get("status", "error"),
                    "code": exec_result.get("code", ""),
                    "output": exec_result.get("output", ""),
                    "error": exec_result.get("error", ""),
                    "artifacts": artifact_objects,
                })
        except Exception as e:
            op_error[0] = e
        finally:
            op_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_run_operations), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            ctx.pop("_event_callback", None)
            ctx.pop("_stream_step", None)
            return
        try:
            item = op_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_event_callback", None)
    ctx.pop("_stream_step", None)

    if op_error[0]:
        yield {
            "type": "llm_update",
            "content": f"Code generation encountered an error: {op_error[0]}",
            "step": "operation_code_generation_error",
        }
        return

    yield from helper.yield_new_logs(ctx, log_count, "operation_code_generation")
    log_count = len(ctx.get("logs", []))

    # ── Persist final state ────────────────────────────────────────
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "workflow_phase": "operations_complete",
    })

    # ── Final result ──────────────────────────────────────────────
    operations = op_results[0] or []

    autonomous_mode = ctx.get("settings", {}).get("autonomous_mode", False)
    if autonomous_mode:
        result_msg = (
            f"Code generation complete for {len(operations)} operation(s) + assembly. "
            "Autonomous mode — proceeding automatically..."
        )
    else:
        result_msg = (
            f"Code generation complete for {len(operations)} operation(s) + assembly. "
            "Please review the generated code."
        )

    yield {
        "type": "result",
        "response": result_msg,
        "result_type": "spatial_analysis_operations_complete",
        "task_id": task_id,
        "data": {"operation_count": len(operations)},
    }


# =====================================================================
#  SPATIAL ANALYSIS: REVISE GEOPROCESSING WORKFLOW (feedback loop)
# =====================================================================

def revise_geoprocessing_workflow_stream(task_id, feedback):
    """Re-run the geoprocessing workflow generator incorporating the user's
    feedback, then regenerate the plan + DAG visualization HTMLs so the
    iframe reload picks up the revised workflow.

    Yields SSE-compatible dicts: llm_stream tokens, an llm_update indicating
    completion, and a final result event.
    """
    reset_cancellation()

    ctx = _plan_context(task_id)
    if not ctx:
        saved = get_workflow_state(task_id) or {}
        if saved.get("ctx"):
            ctx = saved["ctx"]
            ctx["_cancel_event"] = _cancel_event
            research_plan_generator.store_context(task_id, ctx)
    if not ctx:
        yield {"type": "error", "error": f"No geoprocessing workflow context found for task {task_id}"}
        return

    current_explanation = ctx.get("artifacts", {}).get("geoprocessing_workflow_explanation", "")
    if not current_explanation:
        yield {"type": "error", "error": "No existing geoprocessing workflow found to revise"}
        return

    # Append the feedback as an addendum to the original research question so
    # the generator biases its graph toward the user's correction.
    original_rq = ctx.get("rq", "")
    ctx["rq"] = (
        f"{original_rq}\n\n"
        f"IMPORTANT USER FEEDBACK on the previous geoprocessing workflow:\n"
        f"{feedback}\n\n"
        f"Previous workflow explanation (for reference — revise only what the feedback asks about):\n"
        f"{current_explanation}"
    )

    yield from helper.yield_status_update(
        "Revising geoprocessing workflow based on feedback...",
        "geoprocessing_workflow_generation",
    )

    chunk_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": "geoprocessing_workflow"}
    )
    ctx["_event_callback"] = lambda event: chunk_queue.put(event)
    graph_error = [None]

    def _regen():
        try:
            geoprocessing_workflow_generator.generate_geoprocessing_workflow(ctx)
        except Exception as e:
            graph_error[0] = e
        finally:
            chunk_queue.put(None)

    t = threading.Thread(target=_with_flask_ctx(_regen), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            ctx.pop("_event_callback", None)
            ctx["rq"] = original_rq
            return
        try:
            item = chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_event_callback", None)

    # Restore the original rq so downstream execution uses the clean question.
    ctx["rq"] = original_rq

    if graph_error[0]:
        yield {"type": "error", "error": str(graph_error[0])}
        return

    # Regenerate viz HTML files so the iframe reload shows the revised workflow.
    task_output_dir = resolve_output_dir(ctx)
    viz_dir = os.path.join(task_output_dir, "research_plan_viz")
    os.makedirs(viz_dir, exist_ok=True)

    explanation = ctx["artifacts"].get("geoprocessing_workflow_explanation", "")
    plan_html = _wrap_explanation_as_html(explanation, ctx.get("rq", ""))
    plan_path = os.path.join(viz_dir, f"research_plan_{task_id}.html")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_html)

    graph_file = geoprocessing_workflow_generator.resolve_graphfile(ctx)
    geoprocessing_path = os.path.join(viz_dir, f"research_plan_geoprocessing_{task_id}.html")
    if graph_file and os.path.exists(graph_file):
        import networkx as nx
        G = nx.read_graphml(graph_file)
        helper.show_dag_geoprocessing_workflow(
            G, output_path=geoprocessing_path,
            title="Geoprocessing Workflow",
            research_question=ctx.get("rq", ""),
        )
    else:
        with open(geoprocessing_path, "w", encoding="utf-8") as f:
            f.write(plan_html)

    # Persist updated ctx so approval picks up the revised workflow.
    research_plan_generator.store_context(task_id, ctx)
    saved_state = get_workflow_state(task_id) or {}
    saved_state["ctx"] = _serialize_state(ctx)
    save_workflow_state(task_id, saved_state)

    yield {
        "type": "llm_update",
        "content": "Geoprocessing workflow revised.",
        "step": "geoprocessing_workflow_revised",
    }
    yield {
        "type": "geoprocessing_workflow_viz",
        "viz_id": task_id,
    }
    yield {"type": "result", "message": "Geoprocessing workflow revised successfully"}




# =====================================================================
#  CONTINUE AFTER PLAN APPROVAL (Steps 6-7)
# =====================================================================

def continue_after_approval_stream(task_id):
    """
    Continue the workflow after the user approves the research plan.
    Runs Steps 6 (Objective Execution) and 7 (Result Analysis Pipeline).

    Yields:
        SSE-compatible update dicts
    """
    # Reset cancellation flag for this continuation
    reset_cancellation()

    # ── Restore context ────────────────────────────────────────────
    # Prefer the live ctx stored by research_plan_generator (has full objects)
    ctx = _plan_context(task_id)
    if ctx:
        ctx["_cancel_event"] = _cancel_event

    # Fallback: restore from disk-saved workflow state
    if not ctx:
        saved_state = get_workflow_state(task_id)
        if saved_state and saved_state.get("ctx"):
            ctx = saved_state["ctx"]
            ctx["_cancel_event"] = _cancel_event
            # Restore structured plan into artifacts if not present
            if "artifacts" not in ctx:
                ctx["artifacts"] = {}
            sp = saved_state.get("structured_plan", {})
            if sp:
                # Ensure both keys exist — execution uses research_plan_final
                ctx["artifacts"].setdefault("research_plan_structured", sp)
                ctx["artifacts"].setdefault("research_plan_final", sp)
            elif "research_plan_structured" in ctx["artifacts"] and "research_plan_final" not in ctx["artifacts"]:
                ctx["artifacts"]["research_plan_final"] = ctx["artifacts"]["research_plan_structured"]
            # Re-initialize settings from config if missing or serialized
            if not isinstance(ctx.get("settings"), dict) or not ctx["settings"]:
                from utils.config import load_configurations, resolve_output_dir
                ctx["settings"] = load_configurations()
            # Ensure logs list exists
            ctx.setdefault("logs", [])
            # Set task_id and per-task output directory
            ctx["task_id"] = task_id
            if "output_dir" not in ctx:
                task_output_dir = os.path.join(settings.output_dir, task_id)
                os.makedirs(task_output_dir, exist_ok=True)
                ctx["output_dir"] = task_output_dir
            # Store back so subsequent calls find it
            research_plan_generator.store_context(task_id, ctx)
            print(f"[State] Restored workflow context for task {task_id} from disk")

    if not ctx:
        yield {
            "type": "error",
            "error": f"No workflow context found for task {task_id}. "
                     "Please restart the workflow.",
        }
        return

    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    num_objectives = len(structured_plan.get("objectives", []))
    total_steps = sum(
        len(obj.get("steps", []))
        for obj in structured_plan.get("objectives", [])
    )
    log_count = len(ctx.get("logs", []))
    mark_plan_approved(task_id)
    # Proposed method: record any hypothesis the user edited at approval.
    from agents.spatial_reasoning.assumptions import record_checkpoint_edits
    record_checkpoint_edits(ctx)
    # No-planning ablation: there is no plan to approve or count, so the
    # messages say what actually happens instead.
    from agents.task_manager.ablation import NO_PLANNING, ablation_of
    no_planning = ablation_of(ctx) == NO_PLANNING

    if not no_planning:
        yield from helper.yield_status_update(
            "Research plan approved. Starting objective execution...",
            "plan_approved",
        )

    # ── STEP 6: Objective Execution ─────────────────────────────────
    if is_cancelled(): return

    # Ensure agents are registered before execution
    if not task_manager.agent_registry:
        register_agent()

    yield from helper.yield_status_update(
        "Modeling Agent generating and running code for the research question..."
        if no_planning else
        f"Executing {num_objectives} objectives ({total_steps} total steps)...",
        "objective_execution_start",
    )

    obj_chunk_queue = queue.Queue()
    # The executor updates ctx["_stream_step"] as it progresses;
    # the callback reads it so each chunk carries the right step tag.
    ctx["_stream_step"] = "objective_execution"
    ctx["_stream_callback"] = lambda content: obj_chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "objective_execution")}
    )
    # Structured event callback — allows executor to emit rich events (step results, etc.)
    ctx["_event_callback"] = lambda event: obj_chunk_queue.put(event)

    # Persist callback — called after each step so state survives interruptions
    def _persist_after_step(c):
        save_workflow_state(task_id, {
            "ctx": _serialize_state(c),
            "structured_plan": _serialize_state(c.get("artifacts", {}).get("research_plan_final", {})),
            "execution_results": _serialize_state(c.get("artifacts", {}).get("step_execution_results", {})),
            "workflow_phase": "executing",
        })
    ctx["_persist_callback"] = _persist_after_step

    # Set up code review queues if interactive mode is ON
    interactive_code = ctx.get("settings", {}).get("interactive_code", False)
    if interactive_code:
        pending_q, response_q = get_code_review_queues(task_id)
        ctx["_code_review_queue"] = pending_q
        ctx["_code_review_response_queue"] = response_q

    obj_error = [None]
    obj_results = [None]

    def _run_objectives():
        try:
            obj_results[0] = objective_executor.build_serial_graph(ctx)
        except Exception as e:
            obj_error[0] = e
        finally:
            obj_chunk_queue.put(None)  # sentinel

    t_obj = threading.Thread(target=_with_flask_ctx(_run_objectives), daemon=True)
    t_obj.start()

    while True:
        if is_cancelled():
            t_obj.join(timeout=3)
            ctx.pop("_stream_callback", None)
            cleanup_code_review_queues(task_id)
            return

        # Check for code review requests from the executor thread
        if interactive_code:
            try:
                review_item = pending_q.get_nowait()
                review_item["task_id"] = task_id
                print(f"[Workflow] Yielding code_review event for step: {review_item.get('step', '?')}")
                yield review_item
                # Executor thread is now blocked on response_q.get().
                # The /api/workflow/code-review endpoint calls submit_code_review()
                # which puts the user's response on response_q, unblocking the executor.
                # We just keep polling obj_chunk_queue for the next streaming item.
            except queue.Empty:
                pass

        try:
            item = obj_chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    if interactive_code:
        cleanup_code_review_queues(task_id)

    t_obj.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_stream_step", None)

    if is_cancelled(): return

    if obj_error[0]:
        yield {
            "type": "llm_update",
            "content": f"Objective execution encountered an error: {obj_error[0]}",
            "step": "objective_execution_error",
        }
    else:
        yield from helper.yield_status_update(
            "All objectives executed successfully.", "objective_execution_complete"
        )

    yield from helper.yield_new_logs(ctx, log_count, "objective_execution")
    log_count = len(ctx.get("logs", []))

    # ── Execution checkpoint: pause and show summary ─────────────────
    step_summaries = _build_step_summaries(ctx)

    yield {
        "type": "execution_checkpoint",
        "step": "execution_checkpoint",
        "task_id": task_id,
        "steps": step_summaries,
    }

    # In autonomous mode, skip the checkpoint pause and proceed directly
    autonomous_mode = ctx.get("settings", {}).get("autonomous_mode", False)
    if not autonomous_mode:
        # Block until user clicks Proceed or Update Status
        checkpoint_q = get_checkpoint_queue(task_id)
        # Emit an SSE heartbeat every ~10s while waiting so the connection
        # doesn't go idle. If we yield nothing for the whole wait, some
        # browsers/proxies/OS layers close the idle stream, and the
        # post-checkpoint events that follow are silently dropped even
        # though the backend itself runs fine.
        _hb_last = time.time()
        _HB_INTERVAL = 10.0
        while True:
            if is_cancelled():
                cleanup_checkpoint_queue(task_id)
                return
            try:
                cp_response = checkpoint_q.get(timeout=0.5)
            except queue.Empty:
                if time.time() - _hb_last >= _HB_INTERVAL:
                    _hb_last = time.time()
                    yield {"type": "heartbeat", "step": "execution_checkpoint"}
                continue

            action = cp_response.get("action", "proceed")
            if action == "proceed":
                break
            elif action == "update_status":
                # Re-build summaries (steps may have been re-run) and re-emit
                step_summaries = _build_step_summaries(ctx)
                yield {
                    "type": "execution_checkpoint",
                    "step": "execution_checkpoint",
                    "task_id": task_id,
                    "steps": step_summaries,
                }
                # Keep blocking for next response
                continue

        cleanup_checkpoint_queue(task_id)

    # ── STEP 7: Result Analysis Pipeline (LangGraph) ─────────────
    if is_cancelled(): return

    yield from helper.yield_status_update(
        "Collating results and preparing figures...",
        "result_analysis_start",
    )

    ra_chunk_queue = queue.Queue()
    ctx["_stream_step"] = "result_analysis"
    ctx["_stream_callback"] = lambda content: ra_chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "result_analysis")}
    )
    # Point _event_callback to the RA queue so graph nodes can emit
    # flowchart/presentation/discussion events that get yielded immediately
    ctx["_event_callback"] = lambda event: ra_chunk_queue.put(event)
    ra_error = [None]
    ra_results = [None]

    def _run_result_analysis():
        try:
            ra_results[0] = build_result_analysis_graph(ctx, show_graph=False)
        except Exception as e:
            ra_error[0] = e
        finally:
            ra_chunk_queue.put(None)  # sentinel

    t_ra = threading.Thread(target=_with_flask_ctx(_run_result_analysis), daemon=True)
    t_ra.start()

    while True:
        if is_cancelled():
            t_ra.join(timeout=1)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = ra_chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t_ra.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_stream_step", None)
    ctx.pop("_event_callback", None)

    if ra_error[0]:
        yield {
            "type": "llm_update",
            "content": f"Result analysis encountered an error: {ra_error[0]}",
            "step": "result_analysis_error",
        }
        ra_results[0] = {
            "collated_results": {},
        }
    else:
        yield from helper.yield_status_update(
            "Result analysis pipeline complete.", "result_analysis_complete"
        )

    yield from helper.yield_new_logs(ctx, log_count, "result_analysis")
    log_count = len(ctx.get("logs", []))

    ra = ra_results[0] or {}

    # ── STEP 8: Manuscript Generation Pipeline ─────────────────────
    if is_cancelled(): return

    yield from helper.yield_status_update(
        "Running manuscript generation pipeline (title/abstract → introduction → methodology → results → discussion → conclusion → references → assemble)...",
        "manuscript_start",
    )

    ms_chunk_queue = queue.Queue()
    ctx["_stream_step"] = "manuscript"
    ctx["_stream_callback"] = lambda content: ms_chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "manuscript")}
    )
    ctx["_event_callback"] = lambda event: ms_chunk_queue.put(event)
    ms_error = [None]
    ms_results = [None]

    def _run_manuscript():
        try:
            ms_results[0] = build_manuscript_pipeline(ctx, show_graph=False)
        except Exception as e:
            ms_error[0] = e
        finally:
            ms_chunk_queue.put(None)  # sentinel

    t_ms = threading.Thread(target=_with_flask_ctx(_run_manuscript), daemon=True)
    t_ms.start()

    while True:
        if is_cancelled():
            t_ms.join(timeout=1)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = ms_chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t_ms.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_stream_step", None)
    ctx.pop("_event_callback", None)

    if ms_error[0]:
        yield {
            "type": "llm_update",
            "content": f"Manuscript generation encountered an error: {ms_error[0]}",
            "step": "manuscript_error",
        }
    else:
        yield from helper.yield_status_update(
            "Manuscript generation complete.", "manuscript_complete"
        )

    yield from helper.yield_new_logs(ctx, log_count, "manuscript")
    log_count = len(ctx.get("logs", []))

    ms = ms_results[0] or {}

    # ── Generate execution flowchart (after manuscript — final stage) ──
    yield from helper.yield_status_update(
        "Generating execution flowchart...", "ra_flowchart"
    )
    flowchart_path = ""
    try:
        flowchart_path = generate_execution_flowchart(ctx)
    except Exception as e:
        task_manager.log(ctx, f"[FLOWCHART] Generation failed (non-fatal): {e}")
    ra["flowchart_path"] = flowchart_path or ""
    if flowchart_path and os.path.exists(flowchart_path):
        try:
            output_dir = load_configurations().output_dir
            rel = os.path.relpath(flowchart_path, output_dir).replace("\\", "/")
        except Exception:
            rel = os.path.basename(flowchart_path)
        yield {
            "type": "execution_flowchart",
            "relative_path": rel,
            "step": "ra_flowchart",
        }
    elif flowchart_path:
        task_manager.log(ctx, f"[FLOWCHART] File not found after generation: {flowchart_path}")

    # ── Persist state for post-completion step editing ─────────────
    _exec_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    _all_steps = [sr for obj_steps in _exec_results.values() for sr in obj_steps]
    _completed_count = sum(1 for s in _all_steps if s.get("status") == "completed")
    _error_count = sum(1 for s in _all_steps if s.get("status") == "error")
    _skipped_count = sum(1 for s in _all_steps if s.get("status") == "skipped")
    save_workflow_state(task_id, {
        "ctx": _serialize_state(ctx),
        "structured_plan": _serialize_state(structured_plan),
        "execution_results": _serialize_state(_exec_results),
        "result_analysis": _serialize_state(ra),
        "manuscript": _serialize_state(ms),
        "workflow_phase": "completed",
        "num_objectives": num_objectives,
        "total_steps": total_steps,
        "completed_steps": _completed_count,
        "error_steps": _error_count,
        "skipped_steps": _skipped_count,
    })
    # Keep the live ctx accessible for re-run features
    research_plan_generator.store_context(task_id, ctx)

    # ── Final result ────────────────────────────────────────────────
    _skipped_part = f", {_skipped_count} skipped" if _skipped_count else ""
    summary_lines = [
        "Full AGM workflow complete.",
        ("- Research plan: none (no-planning ablation; Modeling Agent coded the question directly)"
         if no_planning else
         f"- Research plan: {num_objectives} objectives, {total_steps} steps"),
        f"- Executed steps: {_completed_count} completed, {_error_count} errors{_skipped_part}",
        "- Result analysis: flowchart, presentation, and discussion generated",
        "- Manuscript: drafted and assembled",
    ]

    yield {
        "type": "result",
        "response": "\n".join(summary_lines),
        "result_type": "workflow_complete",
        "task_id": task_id,
        "structured_plan": structured_plan,
        "objective_results": _exec_results,
        "execution_summary": {
            "total_objectives": num_objectives,
            "total_steps": total_steps,
            "completed_steps": _completed_count,
            "error_steps": _error_count,
            "skipped_steps": _skipped_count,
        },
        "result_analysis": {
            "flowchart_path": ra.get("flowchart_path", ""),
        },
        "manuscript": {
            "manuscript_md": ms.get("manuscript_md", ""),
            "manuscript_path": ms.get("manuscript_path", ""),
            "manuscript_docx": ms.get("manuscript_docx", ""),
            "manuscript_pdf": ms.get("manuscript_pdf", ""),
        },
        "ctx_snapshot": {
            "artifacts_keys": list(ctx.get("artifacts", {}).keys()),
            "log_count": len(ctx.get("logs", [])),
        },
    }


# =====================================================================
#  RESEARCH PLAN INLINE EDITING
# =====================================================================

def update_research_plan(task_id, edited_objectives):
    """
    Update the structured research plan with user's inline edits (title, hypothesis,
    step descriptions, deletions, additions).  Regenerates the HTML visualization.

    Args:
        task_id: The workflow task ID
        edited_objectives: List of objective dicts from the frontend editor

    Returns:
        dict with success/error status
    """
    ctx = _plan_context(task_id)
    if not ctx:
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            research_plan_generator.store_context(task_id, ctx)
        else:
            return {"success": False, "error": f"No context found for task {task_id}"}

    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    if not structured_plan:
        return {"success": False, "error": "No structured plan found in context"}

    # Merge edits into the existing plan (preserve fields the editor doesn't touch)
    existing_objs = {obj["number"]: obj for obj in structured_plan.get("objectives", [])}

    updated_objectives = []
    for edited_obj in edited_objectives:
        obj_num = edited_obj.get("number")
        base = existing_objs.get(obj_num, {})
        # Overwrite title, hypothesis, steps from the editor; keep other fields
        base["number"] = obj_num
        base["title"] = edited_obj.get("title", base.get("title", ""))
        base["hypothesis"] = edited_obj.get("hypothesis", base.get("hypothesis", ""))
        base["steps"] = edited_obj.get("steps", base.get("steps", []))
        updated_objectives.append(base)

    structured_plan["objectives"] = updated_objectives
    ctx["artifacts"]["research_plan_final"] = structured_plan

    # Persist updated context
    research_plan_generator.store_context(task_id, ctx)

    # Also save the updated plan JSON to disk
    task_output_dir = resolve_output_dir(ctx)
    try:
        plan_json_path = os.path.join(task_output_dir, "research_plan.json")
        with open(plan_json_path, "w", encoding="utf-8") as f:
            json.dump(structured_plan, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Could not save plan JSON: {e}")

    # Regenerate the HTML visualization
    try:
        research_plan_viz.save_research_plan_visualization(
            structured_plan, task_output_dir, task_id
        )
    except Exception as e:
        print(f"[Warning] Could not regenerate plan visualization: {e}")

    return {"success": True, "objectives_count": len(updated_objectives)}


def regen_plan_after_step_delete(task_id, edited_objectives, deletion_context):
    """
    After a step is deleted, use LLM to fix the affected objective's steps.
    Focuses on repairing data node dependencies (input_nodes/output_node)
    for steps that depended on the deleted step's output.
    """
    ctx = _plan_context(task_id)
    if not ctx:
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            research_plan_generator.store_context(task_id, ctx)
        else:
            return {"success": False, "error": f"No context found for task {task_id}"}

    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    if not structured_plan:
        return {"success": False, "error": "No structured plan found in context"}

    # First, merge the edited objectives (with the step already removed)
    existing_objs = {obj["number"]: obj for obj in structured_plan.get("objectives", [])}
    affected_obj_num = deletion_context.get("deleted_from_objective")

    updated_objectives = []
    for edited_obj in edited_objectives:
        obj_num = edited_obj.get("number")
        base = existing_objs.get(obj_num, {})
        base["number"] = obj_num
        base["title"] = edited_obj.get("title", base.get("title", ""))
        base["hypothesis"] = edited_obj.get("hypothesis", base.get("hypothesis", ""))
        base["steps"] = edited_obj.get("steps", base.get("steps", []))
        updated_objectives.append(base)

    structured_plan["objectives"] = updated_objectives

    # Find the affected objective for the LLM prompt
    affected_obj = None
    for obj in updated_objectives:
        if obj.get("number") == affected_obj_num:
            affected_obj = obj
            break

    if not affected_obj:
        # Objective itself may have been deleted — just save and return
        ctx["artifacts"]["research_plan_final"] = structured_plan
        research_plan_generator.store_context(task_id, ctx)
        _save_plan_to_disk(structured_plan, ctx)
        return {"success": True, "objectives_count": len(updated_objectives)}

    # Build the LLM prompt — lightweight, focused on data node repair
    rq = ctx.get("rq", structured_plan.get("research_question", ""))
    data_registry = ctx.get("artifacts", {}).get("data_registry", [])
    available_data = [{"file": d.get("file", ""), "path": d.get("path", "")} for d in data_registry[:20]]

    # GIBDChat lives in utils.agm_helper; langchain.schema is gone in LangChain 1.x.
    from utils.agm_helper import GIBDChat
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(), 
        model_name=llm_model("research_plan.repair_after_step_delete"), **llm_reasoning("research_plan.repair_after_step_delete")
        )

    system_prompt = """You are a research plan editor. A step was deleted from an objective, and you need to repair the remaining steps' data dependencies (input_nodes, output_node).

RULES:
- Only modify the AFFECTED OBJECTIVE — return it as a single objective JSON object.
- Keep all step descriptions intact unless they directly reference the deleted step's output.
- Fix input_nodes: if a step used the deleted step's output_node as input, either:
  (a) Redirect it to another step's output that produces equivalent data, OR
  (b) Redirect it to a source dataset if appropriate, OR
  (c) If the step cannot function without the deleted step, remove it and note why.
- Fix depends_on: remove references to the deleted step number, add correct dependencies.
- Renumber steps sequentially (1, 2, 3...) with no gaps.
- Keep output_node names descriptive and unique within the objective.
- Return ONLY the repaired objective as valid JSON (no markdown, no code blocks).

JSON structure for the objective:
{
  "number": <int>,
  "title": "<string>",
  "hypothesis": "<string>",
  "steps": [
    {
      "step_number": <int>,
      "description": "<string>",
      "depends_on": [<int>, ...],
      "input_nodes": ["<node_name>", ...],
      "output_node": "<node_name>"
    }
  ]
}"""

    user_prompt = f"""**RESEARCH QUESTION:** {rq}

**AVAILABLE SOURCE DATA:**
{json.dumps(available_data, indent=2)}

**DELETED STEP:**
- Step {deletion_context.get('deleted_step_number')} from Objective {affected_obj_num}
- Description: {deletion_context.get('deleted_step_description', '')}
- Affected steps: {json.dumps(deletion_context.get('affected_steps', []))}

**CURRENT OBJECTIVE (after deletion, needs repair):**
{json.dumps(affected_obj, indent=2)}

Repair the data dependencies and return the fixed objective JSON."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        # Parse JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        repaired_obj = json.loads(content)

        # Merge repaired objective back into the plan
        for i, obj in enumerate(updated_objectives):
            if obj.get("number") == affected_obj_num:
                # Preserve fields LLM doesn't touch
                for key in obj:
                    if key not in repaired_obj:
                        repaired_obj[key] = obj[key]
                updated_objectives[i] = repaired_obj
                break

        structured_plan["objectives"] = updated_objectives
    except Exception as e:
        print(f"[Warning] LLM plan regeneration failed: {e}")
        # Fall through — save the plan as-is (step already removed)

    # Save everything
    ctx["artifacts"]["research_plan_final"] = structured_plan
    research_plan_generator.store_context(task_id, ctx)
    _save_plan_to_disk(structured_plan, ctx)

    # Re-resolve node paths
    try:
        from agents.task_manager.research_plan_generator import _resolve_node_paths
        _resolve_node_paths(structured_plan, ctx)
        ctx["artifacts"]["research_plan_final"] = structured_plan
        research_plan_generator.store_context(task_id, ctx)
    except Exception as e:
        print(f"[Warning] Could not re-resolve node paths: {e}")

    return {"success": True, "objectives_count": len(updated_objectives)}


def _save_plan_to_disk(structured_plan, ctx):
    """Save plan JSON and regenerate HTML visualization."""
    task_output_dir = resolve_output_dir(ctx)
    try:
        plan_json_path = os.path.join(task_output_dir, "research_plan.json")
        with open(plan_json_path, "w", encoding="utf-8") as f:
            json.dump(structured_plan, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Could not save plan JSON: {e}")

    try:
        research_plan_viz.save_research_plan_visualization(
            structured_plan, task_output_dir, ctx.get("task_id")
        )
    except Exception as e:
        print(f"[Warning] Could not regenerate plan visualization: {e}")


# =====================================================================
#  POST-COMPLETION STEP EDITING & RE-RUNNING
# =====================================================================

def get_completed_steps(task_id):
    """
    Retrieve all executed steps for a completed (or interrupted) workflow.
    Returns a list of step info dicts with code, status, artifacts, and dependency info.
    """
    ctx = _plan_context(task_id)
    if not ctx:
        # Fall back to disk-persisted state
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            research_plan_generator.store_context(task_id, ctx)
    if not ctx:
        return {"error": f"No context found for task {task_id}", "steps": []}

    execution_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})

    # Resolve output_dir for computing relative artifact paths.
    # Use settings.output_dir (the root outputs dir) — NOT the per-task
    # ctx["output_dir"] — so that relative_path includes the task_id segment
    # and /api/artifacts/<relative_path> resolves to the correct file.
    try:
        from utils.config import load_configurations, resolve_output_dir
        output_dir = load_configurations().output_dir
    except Exception:
        output_dir = ""

    steps = []
    for obj in structured_plan.get("objectives", []):
        obj_key = f"objective_{obj['number']}"
        obj_steps = execution_results.get(obj_key, [])
        planned_steps = obj.get("steps", [])

        if obj_steps:
            # Use actual execution results
            for i, step_result in enumerate(obj_steps):
                step_idx = step_result.get("step_index", i + 1)
                inner = step_result.get("result", {}) if isinstance(step_result.get("result"), dict) else {}

                depends_on = []
                if i < len(planned_steps):
                    ps = planned_steps[i]
                    if isinstance(ps, dict):
                        depends_on = ps.get("depends_on", [])

                dependents = []
                for j, other_ps in enumerate(planned_steps):
                    if isinstance(other_ps, dict) and step_idx in other_ps.get("depends_on", []):
                        dependents.append(other_ps.get("step_number", j + 1))

                # Build proper artifact metadata from created_files
                created_files = inner.get("created_files", [])
                artifacts = []
                for fpath in created_files:
                    fname = os.path.basename(fpath)
                    ext = os.path.splitext(fname)[1].lstrip(".").lower()
                    try:
                        fsize = os.path.getsize(fpath) if os.path.exists(fpath) else 0
                    except OSError:
                        fsize = 0
                    try:
                        rel = os.path.relpath(fpath, output_dir).replace("\\", "/") if output_dir else fname
                    except ValueError:
                        rel = fname
                    artifacts.append({
                        "path": fpath,
                        "relative_path": rel,
                        "filename": fname,
                        "size": fsize,
                        "type": ext,
                        "is_shapefile_component": ext in ("shx", "dbf", "prj", "cpg"),
                    })

                steps.append({
                    "objective_key": obj_key,
                    "objective_title": obj.get("title", ""),
                    "objective_number": obj.get("number"),
                    "step_index": step_idx,
                    "step_description": step_result.get("step_description", ""),
                    "status": step_result.get("status", "unknown"),
                    "agent_name": step_result.get("agent_name", ""),
                    "task_complexity": step_result.get("task_complexity", ""),
                    "code": inner.get("code", ""),
                    "code_file": inner.get("code_file", ""),
                    "output": inner.get("output", ""),
                    "artifacts": artifacts,
                    "error": step_result.get("error_message"),
                    "error_traceback": inner.get("error_traceback", ""),
                    "depends_on": depends_on,
                    "dependents": dependents,
                })
        else:
            # Fallback: reconstruct from plan (for old conversations without execution data)
            for i, ps in enumerate(planned_steps):
                step_idx = i + 1
                desc = ps.get("step_description", ps.get("description", f"Step {step_idx}")) if isinstance(ps, dict) else f"Step {step_idx}"
                depends_on = ps.get("depends_on", []) if isinstance(ps, dict) else []

                dependents = []
                for j, other_ps in enumerate(planned_steps):
                    if isinstance(other_ps, dict) and step_idx in other_ps.get("depends_on", []):
                        dependents.append(other_ps.get("step_number", j + 1))

                steps.append({
                    "objective_key": obj_key,
                    "objective_title": obj.get("title", ""),
                    "objective_number": obj.get("number"),
                    "step_index": step_idx,
                    "step_description": desc,
                    "status": "completed",  # assume completed for old runs
                    "agent_name": "ModelingAgent",
                    "task_complexity": "",
                    "code": "",
                    "code_file": "",
                    "output": "",
                    "artifacts": [],
                    "error": None,
                    "error_traceback": "",
                    "depends_on": depends_on,
                    "dependents": dependents,
                })

    return {"steps": steps, "task_id": task_id}


def rerun_step_stream(task_id, objective_key, step_index, mode="edit",
                      edited_code=None, extra_instructions=""):
    """
    Re-run a single step after workflow completion.

    Args:
        task_id: The workflow task ID
        objective_key: e.g., "objective_1"
        step_index: 1-based step index within the objective
        mode: "edit" (use edited_code), "regenerate" (revise existing code with feedback),
              or "re_execute" (generate fresh code from scratch, ignoring existing code)
        edited_code: The user-edited code string (required if mode="edit")
        extra_instructions: Optional user instructions for regeneration

    Yields:
        SSE-compatible update dicts
    """
    reset_cancellation()

    # Restore live ctx (in-memory first, then fall back to disk state)
    ctx = _plan_context(task_id)
    if not ctx:
        # Try to reconstruct from disk-persisted workflow state
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            # Re-store in memory so subsequent calls are fast
            research_plan_generator.store_context(task_id, ctx)
        else:
            yield {"type": "error", "error": f"No context found for task {task_id}. The workflow may not have reached execution."}
            return

    ctx["_cancel_event"] = _cancel_event

    # Resolve output_dir once so it's available across all modes/branches
    # (edit / regenerate / re_execute) and the debugger repair fallback below.
    output_dir = resolve_output_dir(ctx)

    # Ensure agents are registered
    if not task_manager.agent_registry:
        register_agent()

    # Locate the step result to update
    execution_results = ctx.get("artifacts", {}).get("step_execution_results", {})
    obj_steps = execution_results.get(objective_key, [])
    step_index = int(step_index)  # ensure int (may arrive as string from JSON)
    target_step = None
    target_idx_in_list = None
    for i, sr in enumerate(obj_steps):
        if int(sr.get("step_index", -1)) == step_index:
            target_step = sr
            target_idx_in_list = i
            break

    # Fallback: reconstruct from research plan if step_execution_results is empty
    # (happens for conversations saved before per-step persistence was enabled)
    if target_step is None:
        structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
        obj_num = objective_key.replace("objective_", "")
        for obj in structured_plan.get("objectives", []):
            if str(obj.get("number")) == str(obj_num):
                plan_steps = obj.get("steps", [])
                if 0 < step_index <= len(plan_steps):
                    ps = plan_steps[step_index - 1]
                    # Try to get agent from routing data
                    routing_data = ctx.get("artifacts", {}).get("step_routing", {}).get(objective_key, {})
                    routing_list = routing_data.get("routing", []) if isinstance(routing_data, dict) else (routing_data if isinstance(routing_data, list) else [])
                    route_agent = "ModelingAgent"
                    for r in routing_list:
                        if isinstance(r, dict) and int(r.get("step_index", -1)) == step_index:
                            route_agent = r.get("agent_name", "ModelingAgent")
                            break
                    target_step = {
                        "step_index": step_index,
                        "step_description": ps.get("step_description", ps.get("description", f"Step {step_index}")),
                        "agent_name": route_agent,
                        "status": "unknown",
                    }
                    # Persist to execution_results so subsequent lookups work
                    if objective_key not in execution_results:
                        execution_results[objective_key] = []
                        ctx.setdefault("artifacts", {}).setdefault("step_execution_results", {})[objective_key] = execution_results[objective_key]
                    execution_results[objective_key].append(target_step)
                    target_idx_in_list = len(execution_results[objective_key]) - 1
                    print(f"[State] Reconstructed step {step_index} in {objective_key} from research plan (agent: {route_agent})")
                break

    if target_step is None:
        print(f"[DEBUG] Step {step_index} not found in {objective_key}. "
              f"Available steps: {[sr.get('step_index') for sr in obj_steps]}. "
              f"execution_results keys: {list(execution_results.keys())}")
        yield {"type": "error", "error": f"Step {step_index} not found in {objective_key}"}
        return

    step_desc = target_step.get("step_description", f"Step {step_index}")

    # Look up the original agent instance from the registry (same one used during initial execution)
    agent_name = target_step.get("agent_name", "ModelingAgent")
    agent_info = task_manager.agent_registry.get(agent_name) if agent_name else None
    agent_instance = agent_info["instance"] if agent_info else None

    if not agent_instance:
        yield {"type": "error", "error": f"Agent '{agent_name}' not found in registry. Cannot re-run."}
        return

    # ── Recover the ORIGINAL operation_info for this step from the research plan ──
    # The agent's _generate_code_for_operation reads top-level keys (description,
    # operation, input_nodes, output_node, input_node_paths, output_node_path,
    # expected_output, ...) straight from operation_info. Without this lookup,
    # the re-execute path would pass an empty shell and the LLM would regenerate
    # code with no context.
    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    obj_num = objective_key.replace("objective_", "")
    plan_step = {}
    for obj in structured_plan.get("objectives", []):
        if str(obj.get("number")) == str(obj_num):
            for ps in obj.get("steps", []):
                if isinstance(ps, dict) and ps.get("step_number") == step_index:
                    plan_step = dict(ps)  # shallow copy so we don't mutate the plan
                    break
            break
    if not plan_step:
        # Fallback: synthesize a minimal operation_info so the path still runs,
        # but warn that context is limited.
        plan_step = {
            "step_number": step_index,
            "description": step_desc,
            "operation": "",
            "input_nodes": [],
            "output_node": "",
            "input_node_paths": {},
            "output_node_path": "",
            "expected_output": "",
        }
        print(f"[Re-run] Warning: could not find plan_step for {objective_key} step {step_index}; using minimal operation_info.")

    yield from helper.yield_status_update(
        f"Re-running {objective_key} step {step_index}: {step_desc} (via {agent_name})...",
        "rerun_step"
    )

    if mode == "re_execute":
        # Force fresh code generation from scratch (ignoring any existing code)
        yield from helper.yield_status_update(
            f"Re-executing step — generating fresh code via {agent_name}...", "rerun_codegen"
        )
        inner = target_step.get("result", {}) if isinstance(target_step.get("result"), dict) else {}
        code_file = inner.get("code_file", "")
        # Pass the ORIGINAL research-plan step dict as operation_info so the agent
        # has full context (description, operation, input_nodes, output_node,
        # input_node_paths, output_node_path, expected_output).
        chunk_queue = queue.Queue()
        ctx["_stream_callback"] = lambda content: chunk_queue.put(
            {"type": "llm_stream", "content": content, "step": "rerun_codegen"}
        )
        gen_error = [None]
        gen_result = [None]

        def _run_re_execute():
            try:
                gen_result[0] = agent_instance.handle_generate_code_for_simple_operation(
                    ctx, plan_step, extra_instructions=extra_instructions
                )
            except Exception as e:
                gen_error[0] = e
            finally:
                chunk_queue.put(None)

        t = threading.Thread(target=_with_flask_ctx(_run_re_execute), daemon=True)
        t.start()
        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item
        t.join()
        ctx.pop("_stream_callback", None)
        if gen_error[0]:
            yield {"type": "error", "error": str(gen_error[0])}
            return
        if gen_result[0] and gen_result[0].get("error"):
            yield {"type": "error", "error": gen_result[0].get("error", "Code generation failed")}
            return
        code_to_run = gen_result[0]["code"]
        new_code_file = gen_result[0].get("code_file", "") or code_file

        # Save regenerated code to file
        if new_code_file:
            os.makedirs(os.path.dirname(new_code_file), exist_ok=True)
            with open(new_code_file, "w", encoding="utf-8") as f:
                f.write(code_to_run)

        # Update the step result in ctx so the new code is persisted
        if isinstance(target_step.get("result"), dict):
            target_step["result"]["code"] = code_to_run
            target_step["result"]["code_file"] = new_code_file
        else:
            target_step["result"] = {"code": code_to_run, "code_file": new_code_file}
        obj_steps[target_idx_in_list] = target_step
        ctx["artifacts"]["step_execution_results"][objective_key] = obj_steps

        # Return the freshly generated code for user review — do NOT auto-execute
        yield {
            "type": "code_revised",
            "code": code_to_run,
            "code_file": new_code_file,
            "step_index": step_index,
            "objective_key": objective_key,
            "step_description": step_desc,
            "message": "Code regenerated from scratch. Review and click Re-run to execute.",
        }
        research_plan_generator.store_context(task_id, ctx)
        return

    elif mode == "regenerate":
        # Get the existing code from the step result
        inner = target_step.get("result", {}) if isinstance(target_step.get("result"), dict) else {}
        existing_code = inner.get("code", "") or ""
        code_file = inner.get("code_file", "")

        # If code_file exists but existing_code is empty, read from the file
        if not existing_code and code_file and os.path.isfile(code_file):
            with open(code_file, "r", encoding="utf-8") as f:
                existing_code = f.read()

        if not existing_code:
            # No existing code to revise — fall back to full regeneration
            yield from helper.yield_status_update(
                f"No existing code found. Generating fresh code via {agent_name}...", "rerun_codegen"
            )
            # Pass the ORIGINAL research-plan step dict as operation_info so the
            # agent has full context for fresh code generation.
            chunk_queue = queue.Queue()
            ctx["_stream_callback"] = lambda content: chunk_queue.put(
                {"type": "llm_stream", "content": content, "step": "rerun_codegen"}
            )
            gen_error = [None]
            gen_result = [None]

            def _run_gen():
                try:
                    gen_result[0] = agent_instance.handle_generate_code_for_simple_operation(
                        ctx, plan_step, extra_instructions=extra_instructions
                    )
                except Exception as e:
                    gen_error[0] = e
                finally:
                    chunk_queue.put(None)

            t = threading.Thread(target=_with_flask_ctx(_run_gen), daemon=True)
            t.start()
            while True:
                if is_cancelled():
                    t.join(timeout=3)
                    ctx.pop("_stream_callback", None)
                    return
                try:
                    item = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                yield item
            t.join()
            ctx.pop("_stream_callback", None)
            if gen_error[0]:
                yield {"type": "error", "error": str(gen_error[0])}
                return
            if gen_result[0] and gen_result[0].get("error"):
                yield {"type": "error", "error": gen_result[0].get("error", "Code generation failed")}
                return
            code_to_run = gen_result[0]["code"]
            code_file = gen_result[0].get("code_file", "")
        else:
            # Revise existing code based on user feedback using LLM
            yield from helper.yield_status_update(
                f"Revising code based on your feedback...", "rerun_codegen"
            )

            from langchain_core.messages import HumanMessage, SystemMessage

            prior_fb = (
                ctx.get("artifacts", {})
                   .get("step_feedback_log", {})
                   .get(objective_key, {})
                   .get(str(step_index), [])
            )
            prior_fb_block = ""
            if prior_fb:
                lines = [
                    f"- [{entry.get('timestamp','')}] ({entry.get('mode','')}): {entry.get('feedback','').strip()}"
                    for entry in prior_fb if (entry.get('feedback') or '').strip()
                ]
                if lines:
                    prior_fb_block = (
                        "\n## Prior Feedback on This Step (already applied in earlier reruns — "
                        "do NOT re-apply, but keep consistent):\n" + "\n".join(lines) + "\n"
                    )

            revise_prompt = f"""You are an expert Python programmer for a geospatial research pipeline.

The user has provided feedback on existing code for this step: {step_desc}

## Existing Code:
```python
{existing_code}
```
{prior_fb_block}
## Current User Feedback:
{extra_instructions}

## Rules:
- Modify the existing code to address the user's feedback
- Keep everything that is working — only change what the feedback asks for
- The code must be COMPLETE and self-contained (all imports, all logic)
- Save output files to: {output_dir}
- Do NOT use `plt.show()` — use `plt.savefig()` instead
- Do NOT use `if __name__ == "__main__"` guards

Respond with ONLY the complete modified Python code. No explanations, no markdown code blocks — just the raw Python code."""

            # chat_model can be None here if the server was restarted or this
            # endpoint was hit before the main chat flow set the global. Fall
            # back to the same default GIBDChat uses so pydantic accepts it.
            llm = helper.GIBDChat(
                # gibd_user_key=helper.load_OpenAI_key(),
                model_name=llm_model("execution.step_revision"), **llm_reasoning("execution.step_revision"),
            )

            # Stream the revision via background thread
            chunk_queue = queue.Queue()
            ctx["_stream_callback"] = lambda content: chunk_queue.put(
                {"type": "llm_stream", "content": content, "step": "rerun_codegen"}
            )
            gen_error = [None]
            gen_result = [None]

            def _run_revise():
                try:
                    # Stream tokens from the LLM so the UI updates live.
                    accumulated = []
                    stream_cb = ctx.get("_stream_callback")
                    for chunk in llm.stream([
                        SystemMessage(content="You are an expert Python code editor. Output ONLY raw Python code."),
                        HumanMessage(content=revise_prompt),
                    ]):
                        token = chunk.content if hasattr(chunk, "content") else ""
                        if isinstance(token, list):
                            token = "".join(
                                block.get("text", "") if isinstance(block, dict) else str(block)
                                for block in token
                            )
                        if token:
                            accumulated.append(token)
                            if stream_cb:
                                stream_cb(token)
                    content = "".join(accumulated).strip()
                    # Strip markdown code fences if present
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    gen_result[0] = content
                except Exception as e:
                    gen_error[0] = e
                finally:
                    chunk_queue.put(None)

            t = threading.Thread(target=_with_flask_ctx(_run_revise), daemon=True)
            t.start()
            while True:
                if is_cancelled():
                    t.join(timeout=3)
                    ctx.pop("_stream_callback", None)
                    return
                try:
                    item = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                yield item
            t.join()
            ctx.pop("_stream_callback", None)

            if gen_error[0]:
                yield {"type": "error", "error": str(gen_error[0])}
                return
            if not gen_result[0]:
                yield {"type": "error", "error": "LLM returned empty code revision"}
                return

            code_to_run = gen_result[0]

            # Save revised code to file
            if code_file:
                os.makedirs(os.path.dirname(code_file), exist_ok=True)
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(code_to_run)

            # Update the step result in ctx so the revised code is persisted
            if isinstance(target_step.get("result"), dict):
                target_step["result"]["code"] = code_to_run
                target_step["result"]["code_file"] = code_file
            else:
                target_step["result"] = {"code": code_to_run, "code_file": code_file}
            obj_steps[target_idx_in_list] = target_step
            ctx["artifacts"]["step_execution_results"][objective_key] = obj_steps

            # Return the revised code for user review — do NOT auto-execute
            yield {
                "type": "code_revised",
                "code": code_to_run,
                "code_file": code_file,
                "step_index": step_index,
                "objective_key": objective_key,
                "step_description": step_desc,
                "message": "Code revised based on your feedback. Review and click Re-run to execute.",
            }
            # Re-persist context so the revised code is available for re-run
            research_plan_generator.store_context(task_id, ctx)
            return
    else:
        # mode == "edit": use the provided edited code
        code_to_run = edited_code
        inner = target_step.get("result", {}) if isinstance(target_step.get("result"), dict) else {}
        code_file = inner.get("code_file", "")

        # Save the edited code to the code file
        if code_file:
            os.makedirs(os.path.dirname(code_file), exist_ok=True)
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code_to_run)

    # Execute the code using the original agent instance
    yield from helper.yield_status_update("Executing code...", "rerun_exec")

    exec_result = agent_instance._execute_generated_code(ctx, code_to_run, step_desc)
    exec_result["code_file"] = code_file

    # ── Inline debugger for rerun path ────────────────────────────────
    # If execution failed, run the debugger to attempt automatic repair.
    if exec_result.get("status") == "error":
        _error = exec_result.get("error", "")
        error_message = exec_result.get("error_message", "")

        yield from helper.yield_status_update(
            f"Error detected: {error_message}. Running debugger...", "rerun_debug"
        )

        from agents.task_manager.debugger import run_debugger

        # Build input paths summary from routing data or data registry
        routing_data = ctx.get("artifacts", {}).get("step_routing", {}).get(objective_key, {})
        routing_list = routing_data.get("routing", []) if isinstance(routing_data, dict) else routing_data
        route_info = {}
        for r in routing_list:
            if isinstance(r, dict) and r.get("step_index") == step_index:
                route_info = r
                break

        # Use the plan_step already resolved at the top of this function so we
        # get input/output paths straight from the research plan.
        input_nodes = plan_step.get("input_nodes", [])
        output_node = plan_step.get("output_node", "")
        input_node_paths = plan_step.get("input_node_paths", {})
        output_node_path = plan_step.get("output_node_path", "")

        # Build input paths summary
        input_paths_summary = ""
        if input_node_paths:
            for node_name, path in input_node_paths.items():
                input_paths_summary += f"  - {node_name}: {path}\n" if path else f"  - {node_name}: UNRESOLVED\n"
        else:
            # Fallback: list all data registry entries
            data_registry = ctx.get("artifacts", {}).get("data_registry", [])
            for s in data_registry:
                if isinstance(s, dict):
                    input_paths_summary += f"  - {s.get('file', '')}: {s.get('path', '')}\n"

        failed_step = {
            "status": "error",
            "failed_code": code_to_run,
            "step_idx": step_index,
            "operation_description": step_desc,
            "result": exec_result,
            "error_message": error_message,
            "input_paths_summary": input_paths_summary,
            "output_node": output_node,
            "output_node_path": output_node_path,
        }

        # Run debugger in background thread to stream progress
        chunk_queue = queue.Queue()
        ctx["_stream_callback"] = lambda content: chunk_queue.put(
            {"type": "llm_stream", "content": content, "step": "rerun_debug"}
        )
        debug_error = [None]
        debug_result = [None]

        def _run_debug():
            try:
                debug_result[0] = run_debugger(ctx, failed_step)
            except Exception as e:
                debug_error[0] = e
            finally:
                chunk_queue.put(None)

        t = threading.Thread(target=_with_flask_ctx(_run_debug), daemon=True)
        t.start()
        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item
        t.join()
        ctx.pop("_stream_callback", None)

        if not debug_error[0] and debug_result[0]:
            repair = debug_result[0].get("repair_result", {})
            exec_result["error_analysis"] = debug_result[0].get("error_analysis", {})
            exec_result["debug_repair"] = repair

            if repair.get("success"):
                fixed_code = repair.get("fixed_code", "")

                # Re-execute fixed code to capture created_files & output —
                # error_debug() doesn't propagate execution results into
                # repair_result, so we run the fixed code here.
                if fixed_code and not repair.get("created_files"):
                    from agents.task_manager.debugger import _execute_fixed_code
                    reexec = _execute_fixed_code(fixed_code, output_dir=output_dir)
                    if reexec.get("status") == "completed":
                        repair["output"] = reexec.get("output", "")
                        repair["created_files"] = reexec.get("created_files", [])
                        repair["code_file"] = reexec.get("code_file", repair.get("code_file", ""))
                    else:
                        repair["success"] = False
                        repair["notes"] = f"Re-execution of fixed code failed: {reexec.get('error', '')}"

                fixed_output = repair.get("output", "")
                fixed_files = repair.get("created_files", [])
                fixed_code_file = repair.get("code_file", code_file)

                yield from helper.yield_status_update(
                    "Debugger repaired the code successfully.", "rerun_debug_ok"
                )

                # Replace exec_result with the repaired result
                exec_result = {
                    "status": "completed",
                    "output": fixed_output,
                    "code": fixed_code,
                    "code_file": fixed_code_file,
                    "created_files": fixed_files,
                    "error": None,
                    "error_analysis": debug_result[0].get("error_analysis", {}),
                    "debug_repair": repair,
                    "empty_data_files": [],
                    "empty_data_warning": "",
                }
                code_to_run = fixed_code
                code_file = fixed_code_file
            else:
                yield from helper.yield_status_update(
                    f"Debugger could not repair the error. {repair.get('notes', '')[:100]}",
                    "rerun_debug_fail"
                )

    # ── Fallback: regenerate entire code if debugger failed ────────────
    if exec_result.get("status") == "error":
        yield from helper.yield_status_update(
            "Regenerating entire code from scratch...", "rerun_debug_regen"
        )

        error_analysis = exec_result.get("error_analysis", {})
        root_cause = error_analysis.get("root_cause", exec_result.get("error_message", ""))
        regen_instructions = (
            f"The previous code for this step failed with: {exec_result.get('error_message', '')}\n"
            f"Debugger diagnosis: {root_cause}\n"
            f"Traceback:\n{exec_result.get('error_traceback', '')[:500]}\n\n"
            f"Generate a COMPLETELY NEW implementation that avoids this error. "
            f"Do NOT reuse the same approach — try an alternative strategy."
        )

        # Regenerate code via the agent, passing the ORIGINAL research-plan
        # step dict as operation_info so the agent keeps full context.
        chunk_queue = queue.Queue()
        ctx["_stream_callback"] = lambda content: chunk_queue.put(
            {"type": "llm_stream", "content": content, "step": "rerun_debug_regen"}
        )
        gen_error = [None]
        gen_result = [None]

        def _run_regen():
            try:
                gen_result[0] = agent_instance.handle_generate_code_for_simple_operation(
                    ctx, plan_step, extra_instructions=regen_instructions
                )
            except Exception as e:
                gen_error[0] = e
            finally:
                chunk_queue.put(None)

        t = threading.Thread(target=_with_flask_ctx(_run_regen), daemon=True)
        t.start()
        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item
        t.join()
        ctx.pop("_stream_callback", None)

        if not gen_error[0] and gen_result[0] and not gen_result[0].get("error"):
            regen_code = gen_result[0]["code"]
            regen_code_file = gen_result[0].get("code_file", "") or code_file

            if regen_code_file:
                os.makedirs(os.path.dirname(regen_code_file), exist_ok=True)
                with open(regen_code_file, "w", encoding="utf-8") as f:
                    f.write(regen_code)

            yield from helper.yield_status_update("Executing regenerated code...", "rerun_debug_regen_exec")
            regen_result = agent_instance._execute_generated_code(ctx, regen_code, step_desc)
            regen_result["code_file"] = regen_code_file

            if regen_result.get("status") == "completed":
                yield from helper.yield_status_update(
                    "Regenerated code executed successfully.", "rerun_debug_regen_ok"
                )
                exec_result = regen_result
                code_to_run = regen_code
                code_file = regen_code_file
            else:
                yield from helper.yield_status_update(
                    "Regenerated code also failed. Returning error.", "rerun_debug_regen_fail"
                )
                # Use regen result so user sees the latest error
                exec_result = regen_result
                code_to_run = regen_code
                code_file = regen_code_file

    # ── Empty-result detection & auto-retry for rerun path ────────────
    # If execution succeeded but produced no output/files or all data files
    # are empty, regenerate the code once and re-execute.
    if exec_result.get("status") == "completed":
        no_files = not exec_result.get("created_files")
        no_output = not (exec_result.get("output") or "").strip()
        empty_data = exec_result.get("empty_data_files", [])
        tabular_exts = {".csv", ".gpkg", ".geojson", ".shp", ".json", ".npy"}
        tabular_files = [f for f in exec_result.get("created_files", [])
                         if os.path.splitext(f)[1].lower() in tabular_exts]
        all_data_empty = bool(empty_data) and len(empty_data) >= len(tabular_files) and len(tabular_files) > 0

        should_retry = (no_files and no_output) or all_data_empty

        if should_retry:
            if all_data_empty:
                details = "; ".join(f"{os.path.basename(f['path'])} ({f['reason']})" for f in empty_data)
                yield from helper.yield_status_update(
                    f"Output files are empty ({details}). Regenerating code...", "rerun_empty_retry"
                )
                retry_instructions = (
                    f"The previous code ran without errors but the output files contain NO data rows: {details}. "
                    f"Common causes: wrong join key, CRS mismatch in spatial join, overly strict filter, "
                    f"inner join when left join is needed. "
                    f"Fix the code to produce non-empty output and print intermediate DataFrame shapes."
                )
            else:
                yield from helper.yield_status_update(
                    "Code produced no output or files. Regenerating code...", "rerun_empty_retry"
                )
                retry_instructions = (
                    "The previous code ran without errors but generated NO output files and NO printed output. "
                    "Make sure the code saves output files and prints a summary of the result."
                )

            # Regenerate code via the agent, passing the ORIGINAL research-plan
            # step dict as operation_info so the agent keeps full context.
            chunk_queue = queue.Queue()
            ctx["_stream_callback"] = lambda content: chunk_queue.put(
                {"type": "llm_stream", "content": content, "step": "rerun_empty_retry"}
            )
            gen_error = [None]
            gen_result = [None]

            def _run_retry_gen():
                try:
                    gen_result[0] = agent_instance.handle_generate_code_for_simple_operation(
                        ctx, plan_step, extra_instructions=retry_instructions
                    )
                except Exception as e:
                    gen_error[0] = e
                finally:
                    chunk_queue.put(None)

            t = threading.Thread(target=_with_flask_ctx(_run_retry_gen), daemon=True)
            t.start()
            while True:
                if is_cancelled():
                    t.join(timeout=3)
                    ctx.pop("_stream_callback", None)
                    return
                try:
                    item = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                yield item
            t.join()
            ctx.pop("_stream_callback", None)

            if not gen_error[0] and gen_result[0] and not gen_result[0].get("error"):
                retry_code = gen_result[0]["code"]
                retry_code_file = gen_result[0].get("code_file", "") or code_file

                # Save and execute the retried code
                if retry_code_file:
                    os.makedirs(os.path.dirname(retry_code_file), exist_ok=True)
                    with open(retry_code_file, "w", encoding="utf-8") as f:
                        f.write(retry_code)

                yield from helper.yield_status_update("Executing regenerated code...", "rerun_empty_retry_exec")
                retry_result = agent_instance._execute_generated_code(ctx, retry_code, step_desc)
                retry_result["code_file"] = retry_code_file

                # Check if retry produced non-empty results
                retry_output = (retry_result.get("output") or "").strip()
                retry_files = retry_result.get("created_files", [])
                retry_empty_data = retry_result.get("empty_data_files", [])
                if retry_result.get("status") == "completed" and (retry_output or retry_files) and not retry_empty_data:
                    yield from helper.yield_status_update(
                        "Retry succeeded — code now produces output.", "rerun_empty_retry_ok"
                    )
                    exec_result = retry_result
                    code_to_run = retry_code
                    code_file = retry_code_file
                else:
                    yield from helper.yield_status_update(
                        "Retry also produced empty results. Using original result.", "rerun_empty_retry_fail"
                    )

    # Update the step result in ctx
    new_result = {
        "step_index": step_index,
        "step_description": step_desc,
        "agent_name": target_step.get("agent_name", ""),
        "task_complexity": target_step.get("task_complexity", ""),
        "status": "completed" if exec_result.get("status") != "error" else "error",
        "result": exec_result,
        "error_message": exec_result.get("error") if exec_result.get("status") == "error" else None,
        "error_analysis": {},
        "debug_repair": {},
    }
    obj_steps[target_idx_in_list] = new_result
    ctx["artifacts"]["step_execution_results"][objective_key] = obj_steps

    # Persist user feedback so it survives a reload and can be surfaced to
    # the LLM on subsequent turns (debugger prompt, manuscript pipeline, etc.)
    if (extra_instructions or "").strip():
        fb_log = ctx["artifacts"].setdefault("step_feedback_log", {})
        fb_log.setdefault(objective_key, {}).setdefault(str(step_index), []).append({
            "feedback": extra_instructions,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        })

    # Re-persist context (memory + disk) so rerun updates and feedback
    # survive a server restart / conversation reload.
    research_plan_generator.store_context(task_id, ctx)
    save_workflow_state(task_id, {"ctx": ctx})

    # Build artifact metadata for the frontend — use settings.output_dir (root)
    # so relative_path includes the task_id segment for correct file resolution.
    output_dir = settings.output_dir
    created = exec_result.get("created_files", [])
    artifacts = []
    for fpath in created:
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lstrip(".").lower()
        try:
            fsize = os.path.getsize(fpath)
        except OSError:
            fsize = 0
        try:
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
        except ValueError:
            rel = fname
        artifacts.append({
            "path": fpath, "relative_path": rel, "filename": fname,
            "size": fsize, "type": ext,
            "is_shapefile_component": ext in ("shx", "dbf", "prj", "cpg"),
        })

    yield {
        "type": "step_result",
        "step": "rerun_result",
        "step_index": step_index,
        "step_description": step_desc,
        "status": new_result["status"],
        "output": exec_result.get("output", ""),
        "code": code_to_run,
        "objective_key": objective_key,
        "artifacts": artifacts,
        "error": new_result.get("error_message"),
        "error_traceback": exec_result.get("error_traceback", ""),
        "empty_data_warning": exec_result.get("empty_data_warning", ""),
    }

    # Compute downstream dependents
    structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
    dependents = []
    for obj in structured_plan.get("objectives", []):
        if f"objective_{obj['number']}" == objective_key:
            for ps in obj.get("steps", []):
                if isinstance(ps, dict) and step_index in ps.get("depends_on", []):
                    dependents.append(ps.get("step_number", 0))
            break

    # Yield a prompt event asking user about cascading and re-analysis
    yield {
        "type": "rerun_prompt",
        "step_index": step_index,
        "objective_key": objective_key,
        "status": new_result["status"],
        "dependents": dependents,
        "has_dependents": len(dependents) > 0,
        "message": (
            f"Step {step_index} re-run {'completed successfully' if new_result['status'] == 'completed' else 'failed'}."
            + (f" Steps {dependents} depend on this step." if dependents else "")
        ),
    }


def rerun_assembly_stream(task_id, mode="edit", edited_code=None, extra_instructions=""):
    """
    Re-run the assembly program after workflow completion.

    Args:
        task_id: The workflow task ID
        mode: "edit"       — execute the provided edited_code directly
              "re_execute" — re-execute the saved assembly code from ctx
              "regenerate" — re-generate assembly via LLM with feedback, then execute
        edited_code: User-edited assembly code (required when mode='edit')
        extra_instructions: User feedback for regeneration (used when mode='regenerate')

    Yields:
        SSE-compatible update dicts
    """
    from agents.task_manager.spatial_analysis_executor import SpatialAnalysisExecutor

    reset_cancellation()

    ctx = _plan_context(task_id)
    if not ctx:
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            research_plan_generator.store_context(task_id, ctx)
        else:
            yield {"type": "error", "error": f"No context found for task {task_id}."}
            return

    ctx["_cancel_event"] = _cancel_event

    output_dir = resolve_output_dir(ctx)
    artifacts = ctx.setdefault("artifacts", {})

    if mode == "edit":
        if not edited_code:
            yield {"type": "error", "error": "No edited code provided for assembly re-run."}
            return
        code_to_run = edited_code

    elif mode == "re_execute":
        code_to_run = artifacts.get("assembly_code", "")
        if not code_to_run:
            yield {"type": "error", "error": "No saved assembly code found. Run the full workflow first."}
            return

    elif mode == "regenerate":
        yield from helper.yield_status_update("Re-generating assembly program with feedback...", "assembly_regen")
        executor = SpatialAnalysisExecutor(ctx)
        # Rebuild operations list from saved operation_codes (still needed
        # for the downstream "prepend missing operation function definitions"
        # pass after generation).
        saved_ops = artifacts.get("operation_codes", [])
        executor.operations = [
            {"node_name": op["node_name"], "description": op.get("description", ""), "operation_code": op.get("code", "")}
            for op in saved_ops
        ]
    
        

        previous_code = artifacts.get("assembly_code", "")
        if not previous_code:
            yield {"type": "error", "error": "No previous assembly code found to revise. Run the full workflow first."}
            return

        # ── Build a lean revision prompt ───────────────────────────────
        # The full base "assemble N functions from scratch" prompt drowns
        # out user feedback and re-sends every operation body that's already
        # in the previous code. Feed the LLM only what it needs to *edit*:
        # feedback (primary), previous code, and the non-negotiable invariants
        # it must preserve while editing.
        final_output_nodes = executor._get_final_output_nodes()
        final_outputs_lines = "\n".join(
            f"  - {n['name']}: {n.get('description', '')}" for n in final_output_nodes
        ) or "  (none declared)"

        data_registry = artifacts.get("data_registry", [])
        data_lines = "\n".join(
            f"  - {s.get('file','')} | path: {s.get('path','')} | columns: {s.get('columns','')}"
            for s in data_registry
        ) or "  (none)"

        fn_count = len(executor.operations)
        feedback_text = (extra_instructions or "").strip() or "(no feedback provided — make no changes)"

        assembly_prompt = (
            "## Task\n"
            "Revise an existing assembly program. Apply ONLY the changes the user asks for — "
            "preserve everything else. Return the COMPLETE revised Python script.\n\n"
            "## User Feedback  (PRIMARY DIRECTIVE)\n"
            f"{feedback_text}\n\n"
            "## Previous Assembly Code\n"
            f"```python\n{previous_code}\n```\n\n"
            "## Hard Constraints (must be preserved while editing)\n"
            f"- The script must still define and call ALL {fn_count} operation functions present in the previous code. Do NOT drop any.\n"
            "- Use `OUTPUT_DIR` for all intermediate files and `FINAL_OUTPUT_DIR` only for the user-facing final result file(s). Do NOT redefine either variable and do NOT use bare relative paths.\n"
            "- For matplotlib: always `plt.savefig(<path>, dpi=150, bbox_inches='tight')` followed by `plt.close()`. Never call `plt.show()`.\n"
            "- Do NOT add an `if __name__ == \"__main__\"` guard.\n"
            "- Final output node(s) that must be written to FINAL_OUTPUT_DIR:\n"
            f"{final_outputs_lines}\n"
            "- Input data paths (do not invent new ones):\n"
            f"{data_lines}\n\n"
            "## Output\n"
            "Return the full revised Python program inside a single ```python fenced block. "
            "No prose before or after the code block.\n"
        )

        executor.assembly_prompt = assembly_prompt
        
        # Stream tokens live to the frontend via a queue
        regen_queue = queue.Queue()
        ctx["_stream_callback"] = lambda chunk: regen_queue.put(
            {"type": "llm_stream", "content": chunk, "step": "assembly_regen"}
        )
        regen_error = [None]

        def _run_regen():
            try:
                executor.get_LLM_assembly_response(ctx)
            except Exception as e:
                regen_error[0] = e
            finally:
                regen_queue.put(None)  # sentinel

        t = threading.Thread(target=_with_flask_ctx(_run_regen), daemon=True)
        t.start()

        while True:
            if is_cancelled():
                t.join(timeout=3)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = regen_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()
        ctx.pop("_stream_callback", None)

        if regen_error[0]:
            yield {"type": "error", "error": f"Regeneration failed: {regen_error[0]}"}
            return

        code_to_run = artifacts.get("assembly_code", "")
        if not code_to_run:
            yield {"type": "error", "error": "LLM did not return valid assembly code."}
            return

    else:
        yield {"type": "error", "error": f"Unknown mode: {mode}"}
        return

    # Prepend any missing operation function definitions
    saved_ops = artifacts.get("operation_codes", [])
    if saved_ops:
        import re as _re
        defined_fns = set(_re.findall(r"^def\s+(\w+)\s*\(", code_to_run, _re.MULTILINE))
        missing_blocks = []
        for op in saved_ops:
            fn_code = op.get("code", "").strip()
            if not fn_code:
                continue
            m = _re.search(r"^def\s+(\w+)\s*\(", fn_code, _re.MULTILINE)
            fn_name = m.group(1) if m else None
            if fn_name and fn_name not in defined_fns:
                missing_blocks.append(f"# --- {op['node_name']} ---\n{fn_code}")
        if missing_blocks:
            code_to_run = "\n\n".join(missing_blocks) + "\n\n" + code_to_run

    yield from helper.yield_status_update("Executing assembly program...", "assembly_execution")

    # Update ctx with the code to run so execute_assembly_code picks it up
    artifacts["assembly_code"] = code_to_run
    yield from helper.yield_status_update(f"new assembly code generated: {code_to_run}", "assembly_regen")


    # Run execution + inline debugger in a background thread so debugging
    # streams token-by-token to the frontend.
    exec_queue = queue.Queue()
    ctx["_stream_callback"] = lambda chunk: exec_queue.put(
        {"type": "llm_stream", "content": chunk, "step": "assembly_execution"}
    )
    ctx["_event_callback"] = lambda event: exec_queue.put(event)
    exec_error = [None]
    exec_result_ref = [None]

    def _run_exec():
        try:
            _executor = SpatialAnalysisExecutor(ctx)
            exec_result_ref[0] = _executor.execute_assembly_code(ctx)
        except Exception as e:
            exec_error[0] = e
        finally:
            exec_queue.put(None)  # sentinel

    t = threading.Thread(target=_with_flask_ctx(_run_exec), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=3)
            ctx.pop("_stream_callback", None)
            ctx.pop("_event_callback", None)
            return
        try:
            item = exec_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_event_callback", None)

    if exec_error[0]:
        yield {"type": "error", "error": f"Execution failed: {exec_error[0]}"}
        return

    exec_result = exec_result_ref[0] or {"status": "error", "error": "No result returned"}

    # Build artifact objects for UI — mirror the initial-run logic so
    # relative_path is set and only final_outputs/ files become tiles.
    created_files = exec_result.get("created_files", [])
    task_output_dir = ctx.get("output_dir", "")
    final_marker = os.path.normcase(os.path.normpath(
        os.path.join(task_output_dir, "final_outputs")
    )) + os.sep
    final_files = [
        fp for fp in created_files
        if os.path.normcase(os.path.normpath(fp)).startswith(final_marker)
    ]
    files_for_tiles = final_files if final_files else created_files
    # Analysis outputs are often written in a projected CRS (metric buffering
    # etc.); normalize to EPSG:4326 so the Map Viewer can render them.
    _ensure_wgs84_outputs(files_for_tiles)
    if exec_result.get("status") == "completed":
        _stage_spatial_results(ctx, files_for_tiles)

    artifact_objects = []
    for fp in files_for_tiles:
        try:
            file_size = os.path.getsize(fp)
        except OSError:
            file_size = 0
        filename = os.path.basename(fp)
        ext = os.path.splitext(filename)[1].lstrip('.').lower()
        try:
            rel = os.path.relpath(fp, settings.output_dir).replace("\\", "/") if task_output_dir else filename
        except ValueError:
            rel = filename
        artifact_objects.append({
            "filename": filename,
            "path": fp,
            "relative_path": rel,
            "type": ext,
            "size": file_size,
        })

    yield {
        "type": "step_result",
        "step": "assembly_execution",
        "step_index": "Execution",
        "step_description": "Assembly code execution",
        "objective_key": "assembly",
        "status": exec_result.get("status", "error"),
        "code": exec_result.get("code", code_to_run),
        "output": exec_result.get("output", ""),
        "error": exec_result.get("error", ""),
        "artifacts": artifact_objects,
    }

    # Persist user feedback so it survives a reload and can be surfaced
    # to the LLM on subsequent turns.
    if (extra_instructions or "").strip():
        fb_log = ctx.setdefault("artifacts", {}).setdefault("assembly_feedback_log", [])
        fb_log.append({
            "feedback": extra_instructions,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        })

    research_plan_generator.store_context(task_id, ctx)
    save_workflow_state(task_id, {"ctx": ctx})


def rerun_cascade_stream(task_id, objective_key, step_indices):
    """
    Cascade re-run downstream dependent steps. Each step is regenerated via LLM
    since upstream outputs have changed.

    Args:
        task_id: The workflow task ID
        objective_key: e.g., "objective_1"
        step_indices: List of 1-based step indices to re-run, in execution order

    Yields:
        SSE-compatible update dicts
    """
    for step_idx in sorted(step_indices):
        yield from helper.yield_status_update(
            f"Cascading re-run: {objective_key} step {step_idx}...",
            "cascade_rerun"
        )
        yield from rerun_step_stream(
            task_id, objective_key, step_idx,
            mode="regenerate",
            extra_instructions="Previous upstream step outputs have changed. Regenerate this step accordingly."
        )


def rerun_result_analysis_stream(task_id):
    """
    Re-run the Result Analysis pipeline (Step 7) after step re-execution.

    Yields:
        SSE-compatible update dicts
    """
    # ── Overlap guard ─────────────────────────────────────────────
    # Refuse to start a second rerun for the same task while one is
    # already in flight — prevents two pipelines from interleaving
    # writes to the same Outputs/<task_id>/ directory.
    if not try_acquire_rerun_slot(task_id):
        yield {
            "type": "error",
            "error": (
                f"A result-analysis rerun is already running for task {task_id}. "
                "Wait for it to finish (or cancel) before starting another."
            ),
        }
        return

    # Per-task cancel event — NOT the module-level global. Avoids the
    # case where a concurrent rerun on another task flips is_cancelled()
    # for this one and drops the SSE generator while its background
    # thread keeps running and writing files.
    local_cancel = get_task_cancel_event(task_id)
    local_cancel.clear()
    _is_cancelled = local_cancel.is_set

    try:
        ctx = _plan_context(task_id)
        if not ctx:
            disk_state = get_workflow_state(task_id)
            if disk_state and "ctx" in disk_state:
                ctx = disk_state["ctx"]
                research_plan_generator.store_context(task_id, ctx)
            else:
                yield {"type": "error", "error": f"No context found for task {task_id}"}
                return

        ctx["_cancel_event"] = local_cancel

        yield from helper.yield_status_update(
            "Collating results and preparing figures...", "rerun_result_analysis"
        )

        ra_chunk_queue = queue.Queue()
        ctx["_stream_step"] = "result_analysis"
        ctx["_stream_callback"] = lambda content: ra_chunk_queue.put(
            {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "result_analysis")}
        )
        ctx["_event_callback"] = lambda event: ra_chunk_queue.put(event)
        ra_error = [None]
        ra_results = [None]

        def _run_ra():
            try:
                ra_results[0] = build_result_analysis_graph(ctx, show_graph=False)
            except Exception as e:
                ra_error[0] = e
            finally:
                ra_chunk_queue.put(None)

        t = threading.Thread(target=_with_flask_ctx(_run_ra), daemon=True)
        t.start()

        while True:
            if _is_cancelled():
                t.join(timeout=1)
                ctx.pop("_stream_callback", None)
                return
            try:
                item = ra_chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

        t.join()
        ctx.pop("_stream_callback", None)
        ctx.pop("_stream_step", None)
        ctx.pop("_event_callback", None)

        if ra_error[0]:
            yield {"type": "error", "error": str(ra_error[0])}
        else:
            yield from helper.yield_status_update(
                "Result analysis pipeline re-run complete.", "rerun_result_analysis_complete"
            )

        # ── Manuscript Generation Pipeline (Step 8) ──────────────────
        if not _is_cancelled():
            yield from helper.yield_status_update(
                "Running manuscript generation pipeline...", "manuscript_start"
            )

            ms_chunk_queue = queue.Queue()
            ctx["_stream_step"] = "manuscript"
            ctx["_stream_callback"] = lambda content: ms_chunk_queue.put(
                {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "manuscript")}
            )
            ctx["_event_callback"] = lambda event: ms_chunk_queue.put(event)
            ms_error = [None]
            ms_results = [None]

            def _run_ms():
                try:
                    ms_results[0] = build_manuscript_pipeline(ctx, show_graph=False)
                except Exception as e:
                    ms_error[0] = e
                finally:
                    ms_chunk_queue.put(None)

            t_ms = threading.Thread(target=_with_flask_ctx(_run_ms), daemon=True)
            t_ms.start()

            while True:
                if _is_cancelled():
                    t_ms.join(timeout=1)
                    ctx.pop("_stream_callback", None)
                    return
                try:
                    item = ms_chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                yield item

            t_ms.join()
            ctx.pop("_stream_callback", None)
            ctx.pop("_stream_step", None)
            ctx.pop("_event_callback", None)

            if ms_error[0]:
                yield {
                    "type": "llm_update",
                    "content": f"Manuscript generation encountered an error: {ms_error[0]}",
                    "step": "manuscript_error",
                }
            else:
                yield from helper.yield_status_update(
                    "Manuscript generation complete.", "manuscript_complete"
                )

        # Generate execution flowchart (after manuscript — final stage)
        yield from helper.yield_status_update(
            "Generating execution flowchart...", "ra_flowchart"
        )
        flowchart_path = ""
        try:
            flowchart_path = generate_execution_flowchart(ctx)
        except Exception:
            pass
        if flowchart_path and os.path.exists(flowchart_path):
            try:
                output_dir = load_configurations().output_dir
                rel = os.path.relpath(flowchart_path, output_dir).replace("\\", "/")
            except Exception:
                rel = os.path.basename(flowchart_path)
            yield {
                "type": "execution_flowchart",
                "relative_path": rel,
                "step": "ra_flowchart",
            }

        # Re-persist context and workflow state
        research_plan_generator.store_context(task_id, ctx)

        ra = ra_results[0] or {}
        ms = ms_results[0] if not _is_cancelled() and ms_results[0] else {}

        # Persist full workflow state so it survives server restart
        _exec_results = ctx.get("artifacts", {}).get("step_execution_results", {})
        _all_steps = [sr for obj_steps in _exec_results.values() for sr in obj_steps]
        _completed_count = sum(1 for s in _all_steps if s.get("status") == "completed")
        _error_count = sum(1 for s in _all_steps if s.get("status") == "error")
        _skipped_count = sum(1 for s in _all_steps if s.get("status") == "skipped")
        structured_plan = ctx.get("artifacts", {}).get("research_plan_final", {})
        num_objectives = len(structured_plan.get("objectives", []))
        total_steps = sum(len(o.get("steps", [])) for o in structured_plan.get("objectives", []))
        save_workflow_state(task_id, {
            "ctx": _serialize_state(ctx),
            "structured_plan": _serialize_state(structured_plan),
            "execution_results": _serialize_state(_exec_results),
            "result_analysis": _serialize_state(ra),
            "manuscript": _serialize_state(ms),
            "workflow_phase": "completed",
            "num_objectives": num_objectives,
            "total_steps": total_steps,
            "completed_steps": _completed_count,
            "error_steps": _error_count,
            "skipped_steps": _skipped_count,
        })

        yield {
            "type": "result",
            "response": "Result analysis and manuscript generation complete.",
            "result_type": "rerun_analysis_complete",
            "result_analysis": {
                "flowchart_path": flowchart_path,
            },
        }
    finally:
        # Release the rerun slot so the next click (or a retry after a
        # crash mid-pipeline) is allowed through. Runs on normal exit,
        # early `return`, exception, AND generator-close (client abort).
        release_rerun_slot(task_id)


def update_result_analysis_stream(task_id):
    """
    Lightweight update: re-collate results and rebuild figure catalog
    after a step re-run.

    Yields:
        SSE-compatible update dicts
    """
    reset_cancellation()

    ctx = _plan_context(task_id)
    if not ctx:
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            research_plan_generator.store_context(task_id, ctx)
        else:
            yield {"type": "error", "error": f"No context found for task {task_id}"}
            return

    ctx["_cancel_event"] = _cancel_event

    yield from helper.yield_status_update(
        "Updating result analysis...", "update_result_analysis"
    )

    ra_chunk_queue = queue.Queue()
    ctx["_stream_step"] = "result_analysis"
    ctx["_stream_callback"] = lambda content: ra_chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "result_analysis")}
    )
    ctx["_event_callback"] = lambda event: ra_chunk_queue.put(event)
    ra_error = [None]
    ra_results = [None]

    def _run_update():
        try:
            ra_results[0] = build_update_result_analysis_graph(ctx, show_graph=False)
        except Exception as e:
            ra_error[0] = e
        finally:
            ra_chunk_queue.put(None)

    t = threading.Thread(target=_with_flask_ctx(_run_update), daemon=True)
    t.start()

    while True:
        if is_cancelled():
            t.join(timeout=1)
            ctx.pop("_stream_callback", None)
            return
        try:
            item = ra_chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_stream_step", None)
    ctx.pop("_event_callback", None)

    if ra_error[0]:
        yield {"type": "error", "error": str(ra_error[0])}
    else:
        yield from helper.yield_status_update(
            "Result analysis updated.", "update_result_analysis_complete"
        )

    # Generate execution flowchart after update
    flowchart_path = ""
    try:
        flowchart_path = generate_execution_flowchart(ctx)
    except Exception:
        pass
    if flowchart_path and os.path.exists(flowchart_path):
        try:
            output_dir = load_configurations().output_dir
            rel = os.path.relpath(flowchart_path, output_dir).replace("\\", "/")
        except Exception:
            rel = os.path.basename(flowchart_path)
        yield {
            "type": "execution_flowchart",
            "relative_path": rel,
            "step": "ra_flowchart",
        }

    # Re-persist context
    research_plan_generator.store_context(task_id, ctx)

    yield {
        "type": "result",
        "response": "Result analysis updated.",
        "result_type": "update_analysis_complete",
        "result_analysis": {
            "flowchart_path": flowchart_path,
        },
    }


def rerun_manuscript_stream(task_id, sections_to_update=None):
    """
    Re-run the manuscript pipeline without tool calls.

    Uses build_update_manuscript_pipeline with _force_no_tools=True so the
    section writers do a single pure LLM pass (no tool loop). Can regenerate
    the full manuscript or a targeted subset of sections.

    Parameters
    ----------
    task_id : str
    sections_to_update : list[str] | None
        e.g. ["results", "discussion"]. None = regenerate all sections.

    Yields:
        SSE-compatible update dicts
    """
    reset_cancellation()

    ctx = _plan_context(task_id)
    if not ctx:
        disk_state = get_workflow_state(task_id)
        if disk_state and "ctx" in disk_state:
            ctx = disk_state["ctx"]
            research_plan_generator.store_context(task_id, ctx)
        else:
            yield {"type": "error", "error": f"No context found for task {task_id}"}
            return

    ctx["_cancel_event"] = _cancel_event
    ctx["_force_no_tools"] = True

    label = "all sections" if not sections_to_update else ", ".join(sections_to_update)
    yield from helper.yield_status_update(
        f"Rerunning manuscript ({label}) without tool calls...", "rerun_manuscript"
    )

    ms_chunk_queue = queue.Queue()
    ctx["_stream_step"] = "manuscript"
    ctx["_stream_callback"] = lambda content: ms_chunk_queue.put(
        {"type": "llm_stream", "content": content, "step": ctx.get("_stream_step", "manuscript")}
    )
    ctx["_event_callback"] = lambda event: ms_chunk_queue.put(event)
    ms_error = [None]
    ms_results = [None]

    def _run_ms():
        try:
            ms_results[0] = build_update_manuscript_pipeline(
                ctx,
                sections_to_update=sections_to_update,
                show_graph=False,
            )
        except Exception as e:
            ms_error[0] = e
        finally:
            ms_chunk_queue.put(None)

    t_ms = threading.Thread(target=_with_flask_ctx(_run_ms), daemon=True)
    t_ms.start()

    while True:
        if is_cancelled():
            t_ms.join(timeout=1)
            ctx.pop("_stream_callback", None)
            ctx.pop("_force_no_tools", None)
            return
        try:
            item = ms_chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item

    t_ms.join()
    ctx.pop("_stream_callback", None)
    ctx.pop("_stream_step", None)
    ctx.pop("_event_callback", None)
    ctx.pop("_force_no_tools", None)

    if ms_error[0]:
        yield {"type": "error", "error": str(ms_error[0])}
        return

    yield from helper.yield_status_update(
        "Manuscript rerun complete.", "rerun_manuscript_complete"
    )

    # Re-persist context (memory + disk) so rerun updates survive reload.
    research_plan_generator.store_context(task_id, ctx)
    save_workflow_state(task_id, {"ctx": ctx})
    ms = ms_results[0] or {}

    yield {
        "type": "result",
        "response": "Manuscript rerun complete.",
        "result_type": "rerun_manuscript_complete",
        "manuscript": {
            "manuscript_path": ms.get("manuscript_path", ""),
            "sections_updated": sections_to_update or [],
        },
    }


# =====================================================================
#  PLACEHOLDER EXPORTS (keep app.py compatible with existing routes)
# =====================================================================

def get_workflow_nodes_info(task_id):
    """Placeholder - will be implemented when objective execution is added."""
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import get_workflow_nodes_info as _orig
    return _orig(task_id)


def refine_workflow_graph_stream(task_id, feedback, task=None, data_overview=None, active_solutions=None):
    """Placeholder - delegates to existing implementation."""
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import refine_workflow_graph_stream as _orig
    yield from _orig(task_id, feedback, task, data_overview, active_solutions)


def approve_workflow_graph(task_id, active_solutions=None):
    """Placeholder - delegates to existing implementation."""
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import approve_workflow_graph as _orig
    return _orig(task_id, active_solutions)


def refine_workflow_code(task_id, feedback, code_type="assembly", active_solutions=None):
    """Placeholder - delegates to existing implementation."""
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import refine_workflow_code as _orig
    return _orig(task_id, feedback, code_type, active_solutions)


def refine_and_run_stream(task_id, feedback, code_type="assembly", active_solutions=None):
    """Refine the generated code with user feedback, then execute it.

    Streams SSE-shaped dicts: a `code_refined` event with the new code,
    followed by every event from rerun_assembly_stream(mode='edit', ...)
    so the UI gets live execution output and the final `step_result`
    carrying artifacts/output/error.
    """
    try:
        result = refine_workflow_code(
            task_id=task_id,
            feedback=feedback,
            code_type=code_type,
            active_solutions=active_solutions,
        )
    except Exception as e:
        yield {"type": "error", "error": f"Code refinement failed: {e}"}
        return

    refined_code = result.get('refined_code', '')
    if not refined_code:
        yield {"type": "error", "error": "Refinement returned no code."}
        return

    # Surface the refined code to the UI before execution starts.
    yield {
        "type": "code_refined",
        "task_id": task_id,
        "code": refined_code,
        "code_file": result.get('code_file', ''),
    }

    # Reuse the assembly-rerun stream in 'edit' mode — it persists the code
    # into ctx, runs it, and yields a step_result with artifacts/output/error
    # in exactly the same shape the UI already handles for first-run results.
    yield from rerun_assembly_stream(task_id, mode="edit", edited_code=refined_code)


def continue_to_next_step(task_id):
    """Placeholder - delegates to existing implementation."""
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import continue_to_next_step as _orig
    yield from _orig(task_id)


def chat_with_ai(message):
    """Simple non-streaming chat."""
    messages = [{"role": "user", "content": message}]
    response = helper.client_chat_completion(
        messages=messages,
        model=llm_model("chat.assistant"), **llm_reasoning("chat.assistant")
        )
    return response.choices[0].message.content
