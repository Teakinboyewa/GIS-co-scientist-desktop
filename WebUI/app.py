"""
Simple Flask backend for AGM WebUI
This connects the WebUI to the AGM framework functionalities
"""

from flask import Flask, request, jsonify, send_from_directory, send_file, Response, stream_with_context, g, redirect
from flask_cors import CORS
import sys
import os
import json
import mimetypes
import re
import subprocess
import threading
import time

# Load the project-root .env into os.environ at startup so configuration vars
# (e.g. AGM_ADMIN_API_KEYS for the data-source review queue) are reliably
# available at request time regardless of the worker's working directory.
# override=False keeps any vars already set by the PythonAnywhere Web tab.
try:
    from dotenv import load_dotenv
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        override=False,
    )
except Exception:
    pass


class GeoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Shapely geometry objects."""
    def default(self, obj):
        try:
            import shapely.geometry
            if isinstance(obj, shapely.geometry.base.BaseGeometry):
                import shapely
                return shapely.geometry.mapping(obj)
        except ImportError:
            pass
        return super().default(obj)

# Add the parent directory to path to import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WebUI.conversation_db import (
    init_db,
    create_conversation,
    add_message,
    get_conversation,
    get_conversation_by_task_id,
    get_messages,
    list_conversations,
    delete_conversation,
    update_conversation_title,
    set_conversation_pinned,
    link_task_id,
    fork_conversation,
    add_collaborator,
    is_collaborator,
    caller_owns,
    remove_collaborator,
    hash_api_key,
    add_conversation_file,
    get_conversation_files,
    mark_conversation_file_removed,
    restore_conversation_file,
    update_conversation_file_audit,
    update_message_metadata,
    get_artifact,
    save_file_blob,
    restore_file_to_disk,
)

from WebUI.agm_workflow import (
    audit_uploaded_data,
    smart_message_handler_stream,
    get_workflow_nodes_info,
    refine_workflow_graph_stream,
    approve_workflow_graph,
    refine_workflow_code,
    refine_and_run_stream,
    continue_after_approval_stream,
    plan_approval_status,
    continue_after_rq_breakdown_stream,
    revise_rq_breakdown_stream,
    spatial_analysis_continue_after_approval_stream,
    rerun_data_download_stream,
    get_rerun_context,
    revise_geoprocessing_workflow_stream,
    request_cancellation,
    breakdown_waiting,
    full_cleanup,
    submit_code_review,
    submit_eda_approval,
    submit_data_download_approval,
    submit_source_select_approval,
    submit_download_review_response,
    eda_explore_stream,
    submit_eda_explore_approval,
    get_completed_steps,
    rerun_step_stream,
    rerun_cascade_stream,
    rerun_assembly_stream,
    rerun_result_analysis_stream,
    rerun_manuscript_stream,
    update_result_analysis_stream,
    update_research_plan,
)

from agents.task_manager.research_plan_generator import revise_research_plan_stream

# import importlib  # Not needed since module reload is disabled
# import agents.task_manager.task_solution_testing  # Not needed since module reload is disabled

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='/static')
CORS(app, allow_headers=['Content-Type', 'X-API-Key', 'X-Anthropic-Key', 'Authorization', 'Accept'])  # Enable CORS for frontend requests

# React pages (/home agent tiles, /map-interpreter) and the Map Interpreter API
from map_interpreter_api import bp as map_interpreter_bp
app.register_blueprint(map_interpreter_bp)

# Desktop: "Submit to GIBD" — propose a data source to the GitHub catalog as a PR
from catalog_github_api import bp as catalog_github_bp
app.register_blueprint(catalog_github_bp)

# Listing and opening conversations (database only; the desktop app also
# serves these before the engine has loaded — see conversation_routes.py)
from WebUI.conversation_routes import bp as conversation_routes_bp
app.register_blueprint(conversation_routes_bp)

# Initialize conversation history database
init_db()


def _purge_previously_deleted_conversations():
    """Deleting used to only hide a conversation, leaving its uploads and
    outputs on disk. Remove those once, in the background, at startup."""
    try:
        from WebUI.conversation_cleanup import purge_deleted_conversations, vacuum_if_worthwhile
        count, freed = purge_deleted_conversations()
        if count:
            print(f"[cleanup] removed {count} deleted conversation(s), "
                  f"freed {freed / (1024 * 1024):.0f} MB")
        vacuum_if_worthwhile()
    except Exception as e:
        print(f"[cleanup] startup cleanup of deleted conversations failed: {e}")


threading.Thread(target=_purge_previously_deleted_conversations, daemon=True,
                 name="purge-deleted-conversations").start()

# Per-conversation latch: once we've linked a task_id for a conversation in this
# process, skip the DB write on every subsequent streamed event. The SQL itself
# is idempotent; the latch is just to avoid hammering the DB on chatty streams.
_task_id_linked_for: dict = {}


def _serve_html_from_db(task_id, artifact_type):
    """Try to serve an artifact from the database. Returns a Response or None."""
    content = get_artifact(task_id, artifact_type)
    if content:
        return Response(content, mimetype='text/html')
    return None


@app.before_request
def extract_api_key():
    """Extract user-provided API key from X-API-Key header (JSON) or api_key form field (FormData).
    IMPORTANT: Only access request.form for multipart/form-data requests.
    Accessing request.form on a JSON request consumes the input stream,
    causing 'read of closed file' errors in streaming endpoints."""
    key = request.headers.get('X-API-Key', '').strip()
    anthropic_key = request.headers.get('X-Anthropic-Key', '').strip()
    if not key or not anthropic_key:
        ct = request.content_type or ''
        if 'multipart/form-data' in ct or 'application/x-www-form-urlencoded' in ct:
            try:
                key = key or (request.form.get('api_key') or '').strip()
                anthropic_key = anthropic_key or (
                    request.form.get('anthropic_api_key') or '').strip()
            except Exception:
                pass
    g.api_key = key or None

    # Anthropic key for the "Claude Agent SDK" provider option (Generate &
    # Test) -- a separate header/field, since a caller may hold an OpenAI
    # key, an Anthropic key, or both at once.
    g.anthropic_api_key = anthropic_key or None

    # Also propagate into os.environ so that background threads (which
    # cannot access Flask's thread-local ``g``) pick it up via
    # ``load_OpenAI_key()``'s ``os.getenv`` fallback. Only propagate values
    # shaped like a real key — a sentinel like "no-api" (meaning "use the
    # server's own configured key/local model") must not clobber the real
    # OPENAI_API_KEY for the rest of the process.
    if key and (key.startswith('sk-') or key.startswith(('gibd-', 'gibd_'))):
        os.environ['OPENAI_API_KEY'] = key


def _no_key_event():
    """SSE error event for missing API key."""
    return f"data: {json.dumps({'type': 'error', 'error': 'No API key provided. Enter your OpenAI API key in Settings.'})}\n\n"


def sse_data(payload):
    """Encode one payload as an SSE data frame."""
    return f"data: {json.dumps(payload, cls=GeoJSONEncoder)}\n\n"


def _studio_identity_key():
    """OpenAI or Anthropic key, whichever is present -- Handbook Studio
    ("Generate & Test") sessions can run on either provider, so a caller
    with only an Anthropic key must still get a stable user_id to hash for
    session storage. Prefers the OpenAI key when both are set, matching
    every other part of the app that identifies callers by g.api_key."""
    return g.api_key or g.anthropic_api_key


@app.route('/api/clear-key', methods=['POST'])
def clear_api_key():
    """Clear the API key from the server environment."""
    os.environ.pop('OPENAI_API_KEY', None)
    return jsonify({'status': 'ok'})

# Reload the module on every request (for development only)
# DISABLED: This was causing agent registry to be cleared on every HTTP request
# @app.before_request
# def reload_modules():
#     importlib.reload(agents.task_manager.task_solution_testing)


def sse_response(generator):
    """Create a properly-configured SSE Response with headers that prevent
    caching and buffering, ensuring subsequent streams work correctly."""
    resp = Response(stream_with_context(generator), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['X-Accel-Buffering'] = 'no'
    # Note: 'Connection' is a hop-by-hop header forbidden in WSGI (PEP 3333).
    # Waitress manages connection persistence itself.
    return resp


def _drain_detached(generator, label):
    """Finish a pipeline whose client has gone away, discarding its events."""
    try:
        for _ in generator:
            pass
    except Exception:
        app.logger.exception("Detached pipeline %s failed after disconnect", label)
    else:
        app.logger.info("Detached pipeline %s finished after disconnect", label)


def detachable_stream(generator, label=""):
    """Let a pipeline run to completion even if the browser goes away.

    Reloading the page or closing the tab makes the WSGI server close this
    response, which raises GeneratorExit inside the pipeline generator at
    whichever yield it is parked on. Everything after that point -- building
    the run record, appending it to the session, persisting -- is skipped, so a
    run that had already done all its work and spent all its tokens left no
    trace. Worse, it was invisible: the retrieval thread carried on and
    finished into nothing, and the session simply had no run in it.

    On disconnect the pipeline is handed to a background thread that keeps
    pulling events and throwing them away. The pipeline never learns the client
    left, reaches its own end, and persists exactly as it normally would.
    Nothing here writes to the session -- the pipeline stays the single writer,
    so this cannot introduce a partial or duplicate record.
    """
    detached = False
    try:
        for event in generator:
            yield event
    except GeneratorExit:
        detached = True
        threading.Thread(target=_drain_detached, args=(generator, label),
                         daemon=True).start()
        raise
    finally:
        if not detached:
            generator.close()


def _update_workflow_step_result(convo_id, update_event,
                                 objective_key_override=None,
                                 step_index_override=None):
    """Persist a step_result event back into an existing workflow message.

    Re-run / debugger paths stream step_result events carrying the repaired
    code. Without this, the saved conversation only holds the original
    (pre-debug) card, so a browser refresh loses the fix.

    Match a step_result in the stored cards by objective_key + step_index
    and replace its code / output / status / error / artifacts fields.
    Returns True on success, False if nothing was updated.
    """
    if not convo_id:
        return False

    obj_key = (objective_key_override
               if objective_key_override is not None
               else update_event.get('objective_key', ''))
    step_idx = (step_index_override
                if step_index_override is not None
                else update_event.get('step_index', ''))
    if not obj_key and step_idx == '':
        return False

    try:
        msgs = get_messages(convo_id)
    except Exception:
        return False

    new_artifacts = [
        {
            'relative_path': a.get('relative_path', ''),
            'filename': a.get('filename', ''),
            'size': a.get('size', 0),
            'type': a.get('type', ''),
            'is_shapefile_component': a.get('is_shapefile_component', False),
        }
        for a in update_event.get('artifacts', [])
    ]

    patch = {
        'status': update_event.get('status', ''),
        'output': update_event.get('output', ''),
        'code': update_event.get('code', ''),
        'error': update_event.get('error', '') or '',
        'error_traceback': update_event.get('error_traceback', '') or '',
        'artifacts': new_artifacts,
    }

    # Walk every workflow message — newest first — and patch the first
    # matching step_result we find. Previously we only inspected the most
    # recent workflow message, which silently dropped the patch when a later
    # phase (result_analysis, manuscript, etc.) had appended its own workflow
    # message above the one holding the assembly card.
    updated = False
    for target_msg in reversed(msgs):
        if target_msg.get('message_type') != 'workflow':
            continue
        try:
            meta = json.loads(target_msg.get('metadata_json') or '{}')
        except Exception:
            continue
        cards = meta.get('workflow_cards', [])
        msg_updated = False
        for card in cards:
            for sr in card.get('step_results', []) or []:
                sr_obj = sr.get('objective_key', '')
                sr_idx = sr.get('step_index', '')
                if obj_key and sr_obj != obj_key:
                    continue
                if step_idx != '' and str(sr_idx) != str(step_idx):
                    continue
                for k, v in patch.items():
                    sr[k] = v
                if obj_key and not sr.get('objective_key'):
                    sr['objective_key'] = obj_key
                msg_updated = True
                break
            if msg_updated:
                break
        if msg_updated:
            try:
                update_message_metadata(target_msg['id'], metadata=meta)
                updated = True
            except Exception:
                return False
            break
    return updated


def _update_rq_breakdown_in_message(convo_id, rq_data, task_id=None):
    """Patch the saved rq_breakdown_summary rich_content after a revision.

    The revise-stream endpoint emits a fresh ``rq_breakdown_complete`` event
    with the user-modified Task Breakdown. Without this, the conversation
    still holds the pre-feedback version, so a browser reload rebuilds the
    old data request instead of the revised one.

    Finds the most recent workflow message whose cards contain an
    ``rq_breakdown_summary`` rich_content entry and replaces its
    ``rq_breakdown`` payload with *rq_data*. Returns True on success.
    """
    if not convo_id or not rq_data:
        return False
    try:
        msgs = get_messages(convo_id)
    except Exception:
        return False

    for m in reversed(msgs):
        if m.get('message_type') != 'workflow':
            continue
        try:
            meta = json.loads(m.get('metadata_json') or '{}')
        except Exception:
            continue
        cards = meta.get('workflow_cards', [])
        updated = False
        for card in cards:
            for rc in card.get('rich_content', []) or []:
                if rc.get('content_type') == 'rq_breakdown_summary':
                    rc['rq_breakdown'] = rq_data
                    if task_id:
                        rc['task_id'] = task_id
                    updated = True
        if updated:
            try:
                update_message_metadata(m['id'], metadata=meta)
                return True
            except Exception:
                return False
    return False


def _update_rerun_in_message(convo_id, request_index, streamed_text,
                             files_event, outcome):
    """Fold a post-gate re-run of one data request into the saved run.

    The re-run streams into a card the run persisted long ago, and nothing
    wrote its outcome back: a reload rebuilt the original download -- old
    code, old tiles -- and the files the re-run fetched were nowhere in the
    conversation. Appends the re-run's log to that card's body (so the last
    code block is the code that actually ran), swaps its file tiles for the
    new ones, and refreshes the review record's code/status. The request's
    tiles may live on either the ``_cont`` card (request paused at a
    confirmation) or the base card, so whichever holds them is patched.
    Returns True when a message was updated.
    """
    if not convo_id or not request_index:
        return False
    try:
        msgs = get_messages(convo_id)
    except Exception:
        return False
    base = f'data_download_req_{request_index}'
    candidates = (f'{base}_cont', base)

    for m in reversed(msgs):
        if m.get('message_type') != 'workflow':
            continue
        try:
            meta = json.loads(m.get('metadata_json') or '{}')
        except Exception:
            continue
        cards = [c for c in meta.get('workflow_cards', [])
                 if (c.get('phase') or c.get('step')) in candidates]
        if not cards:
            continue
        card = next((c for c in cards if any(
            rc.get('content_type') == 'data_request_files'
            for rc in (c.get('rich_content') or []))), cards[0])

        if streamed_text:
            card['body'] = (card.get('body') or '') + streamed_text
        rich = [rc for rc in (card.get('rich_content') or [])
                if rc.get('content_type') != 'data_request_files']
        for rc in rich:
            if rc.get('content_type') == 'data_download_review':
                rc['code'] = outcome.get('code') or rc.get('code', '')
                rc['status'] = outcome.get('status') or rc.get('status', '')
                rc['error'] = outcome.get('error', '')
        if files_event:
            rich.append({
                'content_type': 'data_request_files',
                'request': files_event.get('request', ''),
                'status': files_event.get('status', ''),
                'files': files_event.get('files', []),
                'error': files_event.get('error', ''),
                'data_unavailable': bool(files_event.get('data_unavailable', False)),
            })
        card['rich_content'] = rich
        try:
            update_message_metadata(m['id'], metadata=meta)
            return True
        except Exception:
            return False
    return False


def _as_nonnegative_int(value):
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _workflow_usage_from_update(update):
    '''Normalize provider-specific token usage fields from a workflow event.'''
    if not isinstance(update, dict):
        return {}
    candidates = [
        update.get('usage'),
        update.get('token_usage'),
        update.get('metrics'),
        (update.get('response_metadata') or {}).get('token_usage')
            if isinstance(update.get('response_metadata'), dict) else None,
        (update.get('data') or {}).get('usage')
            if isinstance(update.get('data'), dict) else None,
    ]
    usage = next((item for item in candidates if isinstance(item, dict)), {})
    input_tokens = _as_nonnegative_int(
        usage.get('input_tokens', usage.get('prompt_tokens')))
    output_tokens = _as_nonnegative_int(
        usage.get('output_tokens', usage.get('completion_tokens')))
    total_tokens = _as_nonnegative_int(usage.get('total_tokens'))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    normalized = {}
    if input_tokens is not None:
        normalized['input_tokens'] = input_tokens
    if output_tokens is not None:
        normalized['output_tokens'] = output_tokens
    if total_tokens is not None:
        normalized['total_tokens'] = total_tokens
    duration_ms = _as_nonnegative_int(
        usage.get('duration_ms', update.get('duration_ms')))
    if duration_ms is not None:
        normalized['duration_ms'] = duration_ms
    return normalized


def _touch_workflow_card_metrics(workflow_cards, update, current_phase=None):
    '''Track process time and token evidence without creating phantom cards.'''
    if not workflow_cards or not isinstance(update, dict):
        return
    # A token_usage event names its phase outright. Deriving it from a step
    # would be wrong for 'execution' and 'setup', which _get_phase does not
    # map back to themselves.
    phase = update.get('phase')
    if not phase:
        step = update.get('step', '')
        phase = _get_phase(step) if step else current_phase
    if not phase or phase not in workflow_cards:
        return
    card = workflow_cards[phase]
    runtime = card.setdefault('_metrics_runtime', {})
    now_ms = int(time.time() * 1000)
    runtime.setdefault('started_at_ms', now_ms)
    runtime['last_event_at_ms'] = now_ms
    if update.get('type') == 'llm_stream':
        runtime['streamed_output_chars'] = (
            runtime.get('streamed_output_chars', 0)
            + len(str(update.get('content', '') or ''))
        )
    usage = _workflow_usage_from_update(update)
    for key, value in usage.items():
        runtime[key] = value


def _workflow_card_for_persistence(card):
    '''Return a DB-safe card copy with finalized timing/token metrics.'''
    saved = card.copy()
    runtime = saved.pop('_metrics_runtime', {}) or {}
    metrics = dict(saved.get('metrics') or {})
    if runtime:
        started_ms = runtime.get('started_at_ms')
        ended_ms = runtime.get('last_event_at_ms', started_ms)
        duration_ms = runtime.get('duration_ms')
        if duration_ms is None and started_ms is not None and ended_ms is not None:
            duration_ms = max(0, int(ended_ms - started_ms))
        if duration_ms is not None:
            metrics['duration_ms'] = duration_ms
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            if runtime.get(key) is not None:
                metrics[key] = runtime[key]
        chars = runtime.get('streamed_output_chars', 0)
        if metrics.get('total_tokens') is None and chars:
            metrics['estimated_output_tokens'] = max(1, (int(chars) + 3) // 4)
            metrics['token_estimate'] = True
    if metrics:
        saved['metrics'] = metrics
    return saved


def _save_workflow_cards(convo_id, workflow_cards, workflow_order,
                         accumulated_response, saved_msg_id=None,
                         usage_summary=None):
    """Persist workflow cards to the database.

    If *saved_msg_id* is provided, update the existing message in-place.
    Otherwise create a new message and return its id.
    Returns the message id (new or existing).

    *usage_summary*, when given, stores the run's token accounting alongside
    the cards. It has to be persisted: providers expose no way to look up
    historical usage, so a figure not saved here is gone once the stream ends
    and a reloaded conversation would show a run that apparently cost nothing.
    """
    if not convo_id or not workflow_cards:
        return saved_msg_id
    full_response = ''.join(accumulated_response).strip()
    cards_list = []
    for phase in workflow_order:
        card = _workflow_card_for_persistence(workflow_cards[phase])
        card['phase'] = phase
        cards_list.append(card)
    metadata = {'workflow_cards': cards_list}
    if usage_summary:
        metadata['usage_summary'] = usage_summary
    if saved_msg_id is not None:
        update_message_metadata(saved_msg_id, content=full_response,
                                metadata=metadata)
        return saved_msg_id
    else:
        msg = add_message(convo_id, 'agm', full_response,
                          message_type='workflow', metadata=metadata)
        return msg['id'] if msg else None


# Event types that mean "the pipeline is about to block on a human". A stream
# can sit on one of these for as long as nobody answers, and until it does the
# run's cards exist only in this generator's local dict -- so every one of them
# forces a write before the generator parks.
_PAUSE_EVENT_TYPES = frozenset({
    'execution_checkpoint',
    'data_source_select_approval',
    'data_download_review',
})

# Longest a run may go without its cards reaching the database. Negligible next
# to the LLM calls happening between events, and it bounds what an interrupted
# run loses to the work done since the last write.
_CHECKPOINT_INTERVAL_S = 5.0


class _CardCheckpointer:
    """Persist workflow cards mid-stream so a lost client cannot lose the run.

    Every workflow route used to write its cards exactly once, from a `finally`
    that runs when the generator is closed. A browser leaving mid-run closes it
    the same way a finished run does, so a pipeline still parked on an approval
    got saved looking exactly like one that had completed -- closing token
    summary and all -- while the worker thread underneath carried on without it.

    Writing on every pause event, and otherwise on a timer, keeps the stored
    message a description of the run as it currently stands. The message id is
    held here so the first write inserts and every later one updates that same
    row: the route's closing save goes through this object too, which is what
    stops the incremental writes and the final one from leaving two copies of
    the same run in the conversation.
    """

    def __init__(self, convo_id, cards, order, accumulated,
                 interval=_CHECKPOINT_INTERVAL_S):
        self.convo_id = convo_id
        self.cards = cards
        self.order = order
        self.accumulated = accumulated
        self.interval = interval
        self.msg_id = None
        self._last_write = 0.0

    def save(self, usage_summary=None):
        """Write the cards now. Never raises: a failed checkpoint must not take
        down the run it exists only to make recoverable."""
        self._last_write = time.monotonic()
        if not self.convo_id or not self.cards:
            return self.msg_id
        try:
            self.msg_id = _save_workflow_cards(
                self.convo_id, self.cards, self.order, self.accumulated,
                self.msg_id, usage_summary=usage_summary)
        except Exception:
            app.logger.exception("Workflow card checkpoint failed")
        return self.msg_id

    def maybe_save(self, update_type):
        """Save if this event parks the run, or if the interval has elapsed."""
        if update_type in _PAUSE_EVENT_TYPES:
            self.save()
        elif time.monotonic() - self._last_write >= self.interval:
            self.save()


def _resolve_pending_approval(task_id, content_type):
    """Clear the 'pending' flag on the newest unanswered prompt of this type.

    A prompt is persisted the moment it is shown, marked pending, so a page
    that opens while the run is parked can rebuild it with working buttons.
    Once somebody answers, that record has to stop advertising itself as a
    live question -- otherwise every later reload offers to answer a decision
    that was already made.

    Walks messages newest-first and clears the first pending match, which is
    the prompt just answered: a run issues these one at a time and cannot
    reach the next one until this one returns.
    """
    convo_id = _conversation_id_for_task(task_id)
    if not convo_id:
        return False
    try:
        msgs = get_messages(convo_id)
    except Exception:
        return False
    for msg in reversed(msgs):
        if msg.get('message_type') != 'workflow':
            continue
        try:
            meta = json.loads(msg.get('metadata_json') or '{}')
        except Exception:
            continue
        for card in meta.get('workflow_cards', []):
            for rc in card.get('rich_content', []) or []:
                if rc.get('content_type') != content_type:
                    continue
                if rc.get('pending') is not True:
                    continue
                rc['pending'] = False
                try:
                    update_message_metadata(msg['id'], metadata=meta)
                    return True
                except Exception:
                    return False
    return False


def _find_conversation_output(conversation_id, filename):
    """Path of *filename* among conversation *conversation_id*'s own outputs.

    Results added to the map from an earlier step ("Add to Map") are sent back
    by name like uploaded layers, but a Research result is never copied into
    the conversation's uploads folder; it stays under Outputs/<conversation>/.
    Only that folder is searched, by exact base name, so a name can never reach
    another conversation's files. The newest match wins (a re-run overwrites).
    """
    if not conversation_id or not filename or os.path.basename(filename) != filename:
        return None
    from utils.config import load_configurations
    base = os.path.realpath(os.path.join(load_configurations().output_dir, conversation_id))
    if not os.path.isdir(base):
        return None
    matches = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if filename in filenames:
            path = os.path.realpath(os.path.join(dirpath, filename))
            if path.startswith(base + os.sep):
                matches.append(path)
    return max(matches, key=os.path.getmtime) if matches else None


def _conversation_id_for_task(task_id):
    """Resolve the conversation id a workflow task belongs to.

    The spatial-analysis/research pipelines use the conversation id AS the
    task id (Outputs/<conversation-id>/...), but the conversations table's
    task_id column is only populated on some paths — so try the direct id
    first, then the linked column."""
    if not task_id:
        return None
    try:
        if get_conversation(task_id):
            return task_id
    except Exception:
        pass
    try:
        convo = get_conversation_by_task_id(task_id)
        return convo['id'] if convo else None
    except Exception:
        return None


@app.route('/')
def home():
    """Serve the AGM WebUI main page"""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/app')
@app.route('/agm/')
def serve_app():
    """Serve the AGM WebUI application"""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/handbook-generator')
@app.route('/handbook-studio')
@app.route('/handbook-studio/')
@app.route('/handbook-studio/<session_id>')
def serve_handbook_generator_views(session_id=None):
    """Deep-link entry points for the Handbook Generator workspace, Handbook
    Studio, and (with a trailing id) one specific Studio session. The
    client-side router (view_router.js) reads the URL on load and opens/loads
    the matching panel; every path just needs the SPA shell — session_id is
    unused here on purpose."""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/handbook-generator/generate-test')
@app.route('/handbook-generator/generate-test/<session_id>')
def redirect_legacy_studio_views(session_id=None):
    """Studio used to live under /handbook-generator/generate-test; keep those
    links (workbooks, notes, bookmarks) resolving to the new /handbook-studio
    path."""
    target = '/handbook-studio' + (f'/{session_id}' if session_id else '')
    return redirect(target, code=301)


# ── Spatial conditions: what each baseline gives the planner ─────────
# Read-only. The prompts come from the same function the planner calls
# (agents/spatial_reasoning/inspection.py), so the page shows what is sent.

def _download(text, filename, mimetype='text/plain'):
    return Response(text, mimetype=f'{mimetype}; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.route('/spatial-conditions')
def serve_spatial_conditions_page():
    return send_from_directory(BASE_DIR, 'spatial_conditions.html')


@app.route('/api/spatial-conditions')
def api_spatial_conditions():
    from agents.spatial_reasoning import inspection
    return jsonify(inspection.list_conditions())


@app.route('/api/spatial-conditions/<name>')
def api_spatial_condition(name):
    from agents.spatial_reasoning import inspection, conditions
    try:
        return jsonify(inspection.describe_condition(name))
    except conditions.UnknownCondition as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/spatial-conditions/<name>/download/<part>')
def api_spatial_condition_download(name, part):
    from agents.spatial_reasoning import inspection, conditions
    try:
        d = inspection.describe_condition(name)
    except conditions.UnknownCondition as e:
        return jsonify({'error': str(e)}), 404
    safe = name.replace('+', 'plus')
    if part == 'bundle':
        return _download(inspection.condition_markdown(name), f'{safe}_condition.md', 'text/markdown')
    if not d['built']:
        return jsonify({'error': d['note']}), 409
    if part == 'system_prompt':
        return _download(d['system_prompt'], f'{safe}_planner_system_prompt.txt')
    if part == 'addition' and d['addition']:
        return _download(d['addition'], f'{safe}_prompt_addition.txt')
    if part == 'skills' and d['skill_files']:
        text = '\n\n'.join(d['skill_files'].values())
        return _download(text, f'{safe}_skill_files.md', 'text/markdown')
    return jsonify({'error': f'Nothing to download for part {part!r}'}), 404


@app.route('/api/spatial-knowledge-base/download')
def api_spatial_kb_download():
    from agents.spatial_reasoning import inspection
    return _download(inspection.kb_yaml_text(), 'spatial_kb.yaml', 'text/yaml')


@app.route('/api/spatial-run/<task_id>')
def api_spatial_run(task_id):
    from agents.spatial_reasoning import inspection
    from utils.config import load_configurations
    try:
        return jsonify(inspection.run_record(load_configurations().output_dir, task_id))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/spatial-run/<task_id>/download/<key>')
def api_spatial_run_download(task_id, key):
    from agents.spatial_reasoning import inspection
    from utils.config import load_configurations
    root = load_configurations().output_dir
    try:
        if key == 'all':
            data = inspection.run_zip(root, task_id)
            if data is None:
                return jsonify({'error': 'No condition record for this run'}), 404
            return Response(data, mimetype='application/zip', headers={
                'Content-Disposition': f'attachment; filename="{task_id}_spatial_record.zip"'})
        found = inspection.run_file(root, task_id, key)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if not found:
        return jsonify({'error': 'File not found for this run'}), 404
    path, download_name = found
    return send_file(path, as_attachment=True, download_name=download_name)


@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve static files (CSS, JS, images, etc.)"""
    response = send_from_directory(BASE_DIR, filename)
    # Prevent browser caching of JS/CSS so changes take effect immediately
    if filename.endswith(('.js', '.css', '.html')):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response



# If you want to audit data separately (without generating a solution), you can still use this endpoint.
@app.route('/api/upload-and-audit-data', methods=['POST'])
def upload_and_audit_data():
    """
    API endpoint to upload data files and generate data overview
    Expects: Multipart form data with file(s)
    Returns JSON: {"success": True, "data_overview": [...]}
    """
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')

        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Save uploaded files — grouped by conversation_id
        conv_id = request.form.get('conversation_id')
        if conv_id:
            upload_dir = os.path.join(BASE_DIR, 'uploads', conv_id)
        else:
            upload_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        file_paths = []
        file_names = []

        for file in files:
            if file and file.filename:
                # Secure the filename
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)

                file_paths.append(file_path)
                file_names.append(filename)

        # Audit the uploaded files using DataAgent. audit=0: the Data Layers
        # panel saving a file it just added, so the layer survives a reload;
        # the audit runs when the file is used in a request.
        if request.form.get('audit', '1') == '0':
            data_overview = []
        else:
            data_overview = audit_uploaded_data(file_paths, file_names)
        # Only link files to a conversation row if one actually exists —
        # otherwise the FK insert raises IntegrityError and causes a 500.
        if conv_id:
            from WebUI.conversation_db import get_conversation
            if get_conversation(conv_id) is None:
                print(f"[upload] conversation {conv_id!r} not found — skipping conversation_files link")
            else:
                for fname, fpath in zip(file_names, file_paths):
                    # As the chat upload path does: keep the bytes in the DB
                    # so another worker or machine can restore the file.
                    try:
                        with open(fpath, 'rb') as fb:
                            save_file_blob(conv_id, fname, fb.read())
                    except Exception:
                        pass
                    audit_rec = next((json.dumps(r) for r in (data_overview or [])
                                      if r.get('name') == fname), None)
                    add_conversation_file(
                        conv_id, fname, os.path.splitext(fname)[0],
                        file_size=os.path.getsize(fpath) if os.path.exists(fpath) else 0,
                        file_type=os.path.splitext(fname)[1].lstrip('.').lower(),
                        audit_json=audit_rec,
                    )

        return jsonify({
            'success': True,
            'data_overview': data_overview,
            'files_uploaded': len(file_paths),
            'message': f'{len(file_paths)} file(s) uploaded and audited successfully'
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload_and_audit_data: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



def _get_phase(step):
    """Map a workflow step tag to its phase (mirrors _getPhase in script.js)."""
    if not step:
        return 'general'
    if step.startswith('intent') or step == 'streaming_chat':
        return 'intent'
    if step == 'agent_registration':
        return 'setup'
    if step.startswith('rq_breakdown'):
        return 'rq_breakdown'
    if step.startswith('data_audit') or step == 'data_registry_init':
        return 'data_audit'
    if step == 'data_download' or step == 'data_download_complete':
        return 'data_download'
    if step.startswith('data_download_req_'):
        return step
    # Need-driven skill synthesis: its own card per data request
    if step.startswith('skill_synth_req_'):
        return step
    if step.startswith('data_summary'):
        return 'data_summary'
    if step.startswith('eda'):
        return 'eda'
    if step.startswith('research_plan'):
        return 'research_plan'
    if step == 'execution_checkpoint':
        return 'execution_checkpoint'
    if step.startswith('exec_obj'):
        return step
    if step.startswith(('extract', 'objective', 'plan_')):
        return 'execution'
    # Spatial analysis: each operation and assembly gets its own card
    if step.startswith('operation_'):
        return step
    if step.startswith('assembly_'):
        return step
    if step.startswith('geoprocessing_'):
        return step
    if step.startswith('ra_'):
        return step
    if step.startswith('result_analysis'):
        return 'result_analysis'
    if step.startswith('ms_'):
        return step
    if step.startswith('manuscript'):
        return 'manuscript'
    # Single-agent baseline: one card per script run and per stretch of
    # reasoning between them (single_agent_exec_3, single_agent_log_4, ...)
    if step.startswith('single_agent'):
        return step
    # Ablation notices (e.g. ablation_no_planning) get their own card
    if step.startswith('ablation_'):
        return step
    return 'general'


# ---------------------------------------------------------------------------
# Token accounting for the Research / Spatial Analysis / Data Retrieval modes
# ---------------------------------------------------------------------------
#
# Every LLM call these modes make goes through utils.agm_helper.GIBDChat, which
# records into a thread-local ledger (agents/data_agent/usage_ledger.py). The
# workflow handlers are generators that run on this request thread and spawn
# worker threads through agm_workflow._with_flask_ctx, which adopts the active
# ledger -- so opening one ledger around the update loop captures the whole run,
# nested agents included.
#
# Attribution is done here rather than in the agents because a phase is only
# identifiable from the events the workflow yields; the agent making the call
# has no idea which card it belongs to.

_USAGE_CALL_FIELDS = ('model', 'phase', 'stage', 'source', 'input_tokens',
                      'output_tokens', 'total_tokens', 'reasoning_tokens', 'at')


def _usage_call_row(entry):
    """One ledger record, trimmed to what the browser renders."""
    return {key: entry.get(key) for key in _USAGE_CALL_FIELDS if key in entry}


def _usage_run_total(summary):
    """The run-level figures, without the per-phase/per-model breakdowns."""
    return {key: summary.get(key) for key in (
        'calls', 'calls_missing_usage', 'input_tokens', 'output_tokens',
        'total_tokens', 'reasoning_tokens', 'elapsed_seconds')}


def track_usage(updates, ledger_holder=None, *, initial_phase=None,
                fixed_phase=None, baseline=None):
    """Yield *updates*, interleaving token_usage events as calls are recorded.

    Follow-up streams (revisions, re-runs) pass extra context:

    * ``initial_phase`` - phase for calls made before the first stepped event.
    * ``fixed_phase``   - attribute EVERY call to this phase, ignoring event
      steps. Re-runs render into an existing card but tag their events with
      steps (``rerun_codegen``, ``assembly_regen``) that map to no card.
    * ``baseline``      - the conversation's saved usage_summary. The emitted
      totals are merged with it, so a re-run adds to the card's existing
      figure instead of replacing it with just the re-run's calls.

    Wraps a workflow generator. Each emitted ``token_usage`` event carries:

    * ``usage``  - cumulative totals for the phase that is currently running.
      Cumulative rather than a delta on purpose: ``_touch_workflow_card_metrics``
      assigns rather than accumulates, so a delta would leave each card showing
      only its most recent call.
    * ``calls``  - only the records added since the last event, so a long run
      does not resend its whole ledger on every step.
    * ``run_total`` / ``by_phase`` - the section totals the header bar shows.

    ``ledger_holder`` is an optional single-element list; the ledger is placed
    in it so the caller can build the final summary after the loop ends.
    """
    try:
        from agents.data_agent import usage_ledger
    except Exception:
        # Accounting is additive; a broken import must not stop a workflow.
        yield from updates
        return

    with usage_ledger.collect() as ledger:
        if ledger_holder is not None:
            ledger_holder.append(ledger)
        offset = 0
        phase = fixed_phase or initial_phase
        if phase:
            ledger.set_phase(phase)
        base_totals = _usage_baseline_totals(baseline)
        # The wrapped generator can change threads mid-run: a stream whose
        # client went away is drained on a background thread (see
        # detachable_stream), and the ledger stack is thread-local, so every
        # call made after the move would record into nothing and the run would
        # under-report what it spent. Re-adopt the ledger for the duration of
        # each `next()` whenever the thread running us has none of its own --
        # that call is where the work, and the LLM calls, actually happen.
        stream = iter(updates)
        while True:
            with usage_ledger.adopt(
                    None if usage_ledger.active() is ledger else ledger):
                try:
                    update = next(stream)
                except StopIteration:
                    break
            if isinstance(update, dict) and not fixed_phase:
                step = update.get('step') or ''
                if step:
                    phase = _get_phase(step)
                    ledger.set_phase(phase)
            yield update

            new_records, offset = ledger.records_since(offset)
            if not new_records:
                continue
            summary = ledger.totals()
            if base_totals:
                summary = usage_ledger.merge(base_totals, summary)
            by_phase = summary.get('by_phase') or {}
            event = {
                'type': 'token_usage',
                'phase': phase,
                'calls': [_usage_call_row(entry) for entry in new_records],
                'run_total': _usage_run_total(summary),
                'by_phase': by_phase,
                'by_model': summary.get('by_model') or {},
            }
            if phase:
                # `phase` (not `step`) is what the card-metrics path reads, so
                # attribution never depends on _get_phase being reversible.
                event['usage'] = by_phase.get(phase) or {}
            yield event


def _usage_baseline_totals(baseline):
    """A saved usage_summary event, in the shape usage_ledger.merge takes."""
    if not isinstance(baseline, dict) or not baseline.get('run_total'):
        return None
    totals = dict(baseline.get('run_total') or {})
    totals['by_phase'] = baseline.get('by_phase') or {}
    totals['by_model'] = baseline.get('by_model') or {}
    return totals


def _card_phase(card):
    return card.get('phase') or _get_phase(card.get('step') or '')


def _usage_target_message(convo_id, phases=()):
    """(message_id, metadata) of the workflow message a follow-up stream's
    tokens belong to: the latest one holding a card for any of *phases*,
    else the latest one that already has a usage_summary, else the latest
    workflow message. None when the conversation has none."""
    if not convo_id:
        return None
    try:
        msgs = get_messages(convo_id)
    except Exception:
        return None
    wanted = {p for p in phases if p}
    fallback_usage = fallback_any = None
    for m in reversed(msgs):
        if m.get('message_type') != 'workflow':
            continue
        try:
            meta = json.loads(m.get('metadata_json') or '{}')
        except Exception:
            continue
        cards = meta.get('workflow_cards') or []
        if wanted and any(_card_phase(c) in wanted for c in cards):
            return m['id'], meta
        if fallback_usage is None and meta.get('usage_summary'):
            fallback_usage = (m['id'], meta)
        if fallback_any is None:
            fallback_any = (m['id'], meta)
    return fallback_usage or fallback_any


def track_followup_usage(updates, convo_id, *, initial_phase=None, fixed_phase=None):
    """track_usage for a follow-up stream on an existing conversation.

    Live token events are merged with the conversation's saved figures, and
    when the stream ends its calls are added to the saved usage_summary (so
    they survive a reload) and a merged usage_summary event is emitted.
    """
    hint = fixed_phase or initial_phase
    target = _usage_target_message(convo_id, [hint] if hint else [])
    baseline = target[1].get('usage_summary') if target else None
    holder = []
    yield from track_usage(updates, holder, initial_phase=initial_phase,
                           fixed_phase=fixed_phase, baseline=baseline)
    ledger = holder[0] if holder else None
    if ledger is None or not ledger.records:
        return
    # Re-resolve by the phases the calls actually landed in, and re-read the
    # message: the stream may have updated it (step results, revised plans).
    phases = {r.get('phase') for r in ledger.records}
    target = _usage_target_message(convo_id, sorted(p for p in phases if p)) or target
    baseline = target[1].get('usage_summary') if target else None
    event = usage_summary_event(ledger)
    if event and baseline and baseline.get('run_total'):
        from agents.data_agent import usage_ledger
        merged = usage_ledger.merge(_usage_baseline_totals(baseline), ledger.totals())
        event = {
            'type': 'usage_summary',
            'run_total': _usage_run_total(merged),
            'by_phase': merged.get('by_phase') or {},
            'by_model': merged.get('by_model') or {},
            'calls': list(baseline.get('calls') or []) + event['calls'],
        }
    if not event:
        return
    if target:
        msg_id, meta = target
        meta['usage_summary'] = event
        try:
            update_message_metadata(msg_id, metadata=meta)
        except Exception:
            pass
    yield event


def _exec_step_phase(objective_key, step_index):
    """Card phase of a Research execution step: objective_2 + 3 -> exec_obj2_step3."""
    num = str(objective_key or '').rsplit('_', 1)[-1]
    return f"exec_obj{num}_step{step_index}" if num.isdigit() else None


def _tracked_cascade(task_id, objective_key, step_indices, convo_id):
    """rerun_cascade_stream, with each re-run step's tokens on its own card."""
    for step_idx in sorted(step_indices):
        from utils.agm_helper import yield_status_update
        yield from yield_status_update(
            f"Cascading re-run: {objective_key} step {step_idx}...", "cascade_rerun")
        yield from track_followup_usage(
            rerun_step_stream(
                task_id, objective_key, step_idx, mode="regenerate",
                extra_instructions="Previous upstream step outputs have changed. "
                                   "Regenerate this step accordingly."),
            convo_id, fixed_phase=_exec_step_phase(objective_key, step_idx))

def usage_summary_event(ledger):
    """The end-of-run section total, or None when nothing was recorded."""
    if ledger is None:
        return None
    try:
        summary = ledger.totals()
    except Exception:
        return None
    if not summary.get('calls'):
        return None
    return {
        'type': 'usage_summary',
        'run_total': _usage_run_total(summary),
        'by_phase': summary.get('by_phase') or {},
        'by_model': summary.get('by_model') or {},
        'calls': [_usage_call_row(entry) for entry in ledger.records],
    }


@app.route('/api/chat-with-ai-stream', methods=['POST'])
def chat_with_ai_stream():
    """
    Streaming version of chat endpoint that sends real-time LLM updates
    Uses Server-Sent Events (SSE) to stream responses
    """
    api_key = g.api_key

    # ── Parse request data BEFORE the generator ──────────────────────────
    # Reading request.form / request.files / request.get_json() inside the
    # generator fails because the WSGI input stream is closed by the time
    # the generator executes.  Extract everything eagerly here.
    message = None
    file_paths = []
    file_names = []
    interactive_code = False
    enable_eda = False
    autonomous_mode = False
    data_download_mode = False
    intent_mode = 'research'
    spatial_condition = 'B0'
    selected_model = None
    reasoning_effort = None
    conversation_id = None
    parse_error = None

    try:
        if request.is_json:
            data = request.get_json()
            if not data or 'message' not in data:
                parse_error = 'Missing message in request body'
            else:
                message = data['message']
                interactive_code = data.get('interactive_code', False)
                enable_eda = data.get('enable_eda', False)
                autonomous_mode = data.get('autonomous_mode', False)
                data_download_mode = data.get('data_download_mode', False)
                intent_mode = data.get('intent_mode', 'research')
                spatial_condition = data.get('spatial_condition', 'B0')
                selected_model = data.get('model')
                reasoning_effort = data.get('reasoning_effort')
                conversation_id = data.get('conversation_id') or None
        else:
            message = request.form.get('message')
            interactive_code = request.form.get('interactive_code', 'false').lower() == 'true'
            enable_eda = request.form.get('enable_eda', 'false').lower() == 'true'
            autonomous_mode = request.form.get('autonomous_mode', 'false').lower() == 'true'
            data_download_mode = request.form.get('data_download_mode', 'false').lower() == 'true'
            intent_mode = request.form.get('intent_mode', 'research')
            spatial_condition = request.form.get('spatial_condition', 'B0')
            selected_model = request.form.get('model')
            reasoning_effort = request.form.get('reasoning_effort')
            conversation_id = request.form.get('conversation_id') or None
            if not message:
                parse_error = 'Missing message in request'
            else:
                # Create conversation early so uploads go into its subfolder
                if not conversation_id:
                    title = (message[:80] + '...') if len(message) > 80 else message
                    user_id = hash_api_key(api_key)
                    convo = create_conversation(title=title, user_id=user_id, mode=intent_mode)
                    conversation_id = convo['id']

                # Save uploaded files NOW, while the stream is still open
                if 'files' in request.files:
                    files = request.files.getlist('files')
                    if files and files[0].filename != '':
                        if conversation_id:
                            upload_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
                        else:
                            upload_dir = os.path.join(BASE_DIR, 'uploads')
                        os.makedirs(upload_dir, exist_ok=True)

                        # Shapefile companion extensions — save them but don't
                        # add to file_paths/file_names (the .shp entry is enough;
                        # geopandas/fiona auto-discovers sibling files).
                        shp_companion_exts = {'.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.fbn', '.fbx', '.ain', '.aih', '.ixs', '.mxs', '.atx', '.xml'}

                        for file in files:
                            if file and file.filename:
                                from werkzeug.utils import secure_filename
                                filename = secure_filename(file.filename)
                                file_path = os.path.join(upload_dir, filename)
                                file.save(file_path)

                                # Persist file content to DB for cross-machine sharing
                                if conversation_id:
                                    try:
                                        with open(file_path, 'rb') as fb:
                                            save_file_blob(conversation_id, filename, fb.read())
                                    except Exception:
                                        pass

                                ext = os.path.splitext(filename)[1].lower()
                                if ext not in shp_companion_exts:
                                    file_paths.append(file_path)
                                    file_names.append(filename)

                # Handle restored layers — files already on disk from a prior conversation
                existing_files = request.form.getlist('existing_files')
                if existing_files:
                    if conversation_id:
                        upload_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
                    else:
                        upload_dir = os.path.join(BASE_DIR, 'uploads')
                    from werkzeug.utils import secure_filename as _sf
                    for fname in existing_files:
                        safe = _sf(fname)
                        fpath = os.path.join(upload_dir, safe)
                        if not os.path.exists(fpath) and conversation_id:
                            # Try restoring from DB
                            restored = restore_file_to_disk(conversation_id, safe, upload_dir)
                            if restored:
                                fpath = restored
                        if not os.path.exists(fpath) and conversation_id:
                            # A result layer from an earlier step of this conversation
                            found = _find_conversation_output(conversation_id, safe)
                            if found:
                                fpath = found
                        if os.path.exists(fpath) and safe not in file_names:
                            file_paths.append(fpath)
                            file_names.append(safe)
    except Exception as e:
        parse_error = str(e)

    # Build file metadata for DB persistence (before generator closes request)
    _file_metadata = []
    for fname, fpath in zip(file_names, file_paths):
        _file_metadata.append({
            'filename': fname,
            'display_name': os.path.splitext(fname)[0],
            'file_size': os.path.getsize(fpath) if os.path.exists(fpath) else 0,
            'file_type': os.path.splitext(fname)[1].lstrip('.').lower(),
        })

    # ── Generator (uses only pre-extracted locals) ───────────────────────
    def generate():
        nonlocal conversation_id

        if not api_key:
            yield _no_key_event()
            return
        try:
            if parse_error:
                yield f"data: {json.dumps({'type': 'error', 'error': parse_error})}\n\n"
                return

            # ── Conversation persistence: create or reuse ────────────
            if not conversation_id:
                title = (message[:80] + '...') if message and len(message) > 80 else (message or 'New Conversation')
                user_id = hash_api_key(api_key)
                convo = create_conversation(title=title, user_id=user_id, mode=intent_mode)
                conversation_id = convo['id']

            # Save the user message
            add_message(conversation_id, 'user', message or '')

            # Persist file associations for this conversation
            for fm in _file_metadata:
                add_conversation_file(
                    conversation_id, fm['filename'], fm['display_name'],
                    fm['file_size'], fm['file_type'],
                )

            # Tell the frontend which conversation this belongs to
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"

            # Send initial status only for auto mode; manual modes skip
            # straight to the workflow, which emits its own status updates.
            if intent_mode == 'auto':
                yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...'})}\n\n"

            # Accumulate AGM response text for persistence
            accumulated_response = []

            # Track workflow card data for persistence
            workflow_cards = {}   # phase -> {status, body, step}
            workflow_order = []   # preserve insertion order of phases
            wf_current_phase = None
            # Cards reach the database as the run produces them, not only when
            # the generator closes -- see _CardCheckpointer.
            _ckpt = _CardCheckpointer(conversation_id, workflow_cards,
                                      workflow_order, accumulated_response)
            # track_usage fills this with the ledger for the run, so the final
            # section total can still be built after the loop has ended.
            _run_ledger = []

            # Use streaming handler that yields progress updates.
            # track_usage opens the ledger every LLM call in this run records
            # into, and injects token_usage events as the calls happen.
            for update in track_usage(smart_message_handler_stream(
                message=message,
                file_paths=file_paths if file_paths else None,
                file_names=file_names if file_names else None,
                active_solutions=active_solutions,
                interactive_code=interactive_code,
                enable_eda=enable_eda,
                autonomous_mode=autonomous_mode,
                data_download_mode=data_download_mode,
                intent_mode=intent_mode,
                spatial_condition=spatial_condition,
                model=selected_model,
                reasoning_effort=reasoning_effort,
                api_key=api_key,
                conversation_id=conversation_id,
            ), _run_ledger):
                # Accumulate text from streaming chunks and result messages
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    # Track workflow streams (non-chat) into card bodies
                    stream_step = update.get('step', '')
                    is_workflow_stream = stream_step and stream_step != 'streaming_chat'
                    # Only accumulate chat streams into the final text;
                    # workflow streams are already captured in card metadata.
                    if not is_workflow_stream:
                        accumulated_response.append(update.get('content', ''))
                    if is_workflow_stream:
                        phase = _get_phase(stream_step)
                        if phase not in workflow_cards:
                            # Use the first line of stream content as the card
                            # status (matches live frontend appendToWorkflowCard
                            # behaviour which shows the first line as the header).
                            first_line = (update.get('content', '') or '').split('\n')[0].strip()
                            workflow_cards[phase] = {
                                'status': first_line or phase,
                                'body': '',
                                'step': stream_step,
                            }
                            workflow_order.append(phase)
                        workflow_cards[phase]['body'] += update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'llm_update':
                    step = update.get('step', '')
                    silent_steps = ['intent_detected', 'chat_response']
                    if step not in silent_steps:
                        phase = _get_phase(step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['status'] = update.get('content', '')
                        wf_current_phase = phase
                        # Persist Task Breakdown structured summary so it can be
                        # rebuilt on browser reload / shared conversation link.
                        if step == 'rq_breakdown_complete':
                            rq_data = (update.get('data') or {}).get('rq_breakdown')
                            rq_task_id = (update.get('data') or {}).get('task_id')
                            if rq_data:
                                workflow_cards[phase].setdefault('rich_content', [])
                                workflow_cards[phase]['rich_content'].append({
                                    'content_type': 'rq_breakdown_summary',
                                    'rq_breakdown': rq_data,
                                    'task_id': rq_task_id,
                                    'run_id': (update.get('data') or {}).get('run_id'),
                                    'workflow_type': (update.get('data') or {}).get('workflow_type'),
                                    'step': step,
                                })
                elif utype == 'status':
                    step = update.get('step', 'general')
                    phase = _get_phase(step)
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                        workflow_order.append(phase)
                    workflow_cards[phase]['status'] = update.get('message', '')
                    wf_current_phase = phase
                elif utype == 'research_plan_stream':
                    if wf_current_phase and wf_current_phase in workflow_cards:
                        workflow_cards[wf_current_phase]['body'] += update.get('content', '')
                elif utype == 'step_result':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('step_results', [])
                        workflow_cards[phase]['step_results'].append({
                            'step_index': update.get('step_index', ''),
                            'step_description': update.get('step_description', ''),
                            'status': update.get('status', ''),
                            'output': update.get('output', ''),
                            'code': update.get('code', ''),
                            'objective_key': update.get('objective_key', ''),
                            'error': update.get('error', ''),
                            'error_traceback': update.get('error_traceback', ''),
                            'artifacts': [
                                {
                                    'relative_path': a.get('relative_path', ''),
                                    'filename': a.get('filename', ''),
                                    'size': a.get('size', 0),
                                    'type': a.get('type', ''),
                                    'is_shapefile_component': a.get('is_shapefile_component', False),
                                }
                                for a in update.get('artifacts', [])
                            ],
                        })
                elif utype == 'execution_checkpoint':
                    phase = 'execution_checkpoint'
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': 'Execution Summary', 'body': '', 'step': 'execution_checkpoint'}
                        workflow_order.append(phase)
                    workflow_cards[phase]['checkpoint_steps'] = update.get('steps', [])
                    # Stash task_id on the checkpoint card so the frontend can
                    # recover it on reload — otherwise "Proceed to Result
                    # Analysis" has no task_id to send when the DB row's
                    # task_id column is still NULL (e.g. for conversations
                    # whose run was interrupted before the final `result`
                    # event that historically triggered `link_task_id`).
                    _ev_tid = update.get('task_id')
                    if _ev_tid:
                        workflow_cards[phase]['task_id'] = _ev_tid
                    # Metrics belong to the checkpoint card, not to whatever
                    # phase happened to be streaming before it. The save itself
                    # is the checkpointer's below: execution_checkpoint is a
                    # pause event, so it writes before the generator parks.
                    _touch_workflow_card_metrics(workflow_cards, update, phase)
                elif utype == 'figure_catalog':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': 'Figure Catalog Completed', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        else:
                            workflow_cards[phase]['status'] = 'Figure Catalog Completed'
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'figure_catalog',
                            'figures': update.get('figures', []),
                            'step': step,
                        })
                elif utype == 'execution_flowchart':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'execution_flowchart',
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'result_presentation':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'result_presentation',
                            'content': update.get('content', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'result_discussion':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'result_discussion',
                            'content': update.get('content', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'code_review':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'code_review',
                            'code': update.get('code', ''),
                            'step_index': update.get('step_index', ''),
                            'step_description': update.get('step_description', ''),
                            'agent_name': update.get('agent_name', ''),
                            'step': step,
                        })
                elif utype == 'data_download_summary':
                    step = update.get('step', 'data_download')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_download_summary',
                            'content': update.get('content', ''),
                            'files': update.get('files', []),
                        })
                elif utype == 'data_request_files':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        # One tile panel per request: a retry's files
                        # replace the previous attempt's, as they do live
                        # (displayDataRequestTiles removes the old panel).
                        workflow_cards[phase]['rich_content'] = [
                            rc for rc in workflow_cards[phase]['rich_content']
                            if rc.get('content_type') != 'data_request_files']
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_request_files',
                            'request': update.get('request', ''),
                            'status': update.get('status', ''),
                            'files': update.get('files', []),
                            'error': update.get('error', ''),
                            'data_unavailable': bool(update.get('data_unavailable', False)),
                        })
                elif utype == 'data_source_select_approval':
                    # Persist the source-select review (ideal source, selected
                    # source, explanation, confidence) so it survives reload.
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        # Stored as the whole prompt, not a summary of it. A
                        # reloaded page has to be able to REBUILD this prompt
                        # and answer it -- the run may still be parked on it --
                        # and the fields dropped here were exactly the ones
                        # needed to do that: task_id (nothing to route an answer
                        # to), required_keys (no key inputs), and the credential
                        # / post-synthesis variants (wrong panel entirely).
                        # 'pending' is cleared by the answer endpoint, so a
                        # restored prompt can tell a live question from the
                        # record of a settled one.
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_source_select_approval',
                            'step': step,
                            'task_id': update.get('task_id', ''),
                            'pending': True,
                            'data_request': update.get('data_request', ''),
                            'llm_choice': update.get('llm_choice', ''),
                            'llm_explanation': update.get('llm_explanation', ''),
                            'ideal_source': update.get('ideal_source', ''),
                            'ideal_source_reasoning': update.get('ideal_source_reasoning', ''),
                            'data_to_download': update.get('data_to_download', ''),
                            'confidence': update.get('confidence', 0),
                            'is_recognized': update.get('is_recognized', True),
                            'available_sources': update.get('available_sources', []),
                            'required_keys': update.get('required_keys', []),
                            'caveats': update.get('caveats', ''),
                            'credential_error': update.get('credential_error', False),
                            'credential_missing': update.get('credential_missing', False),
                            'credential_message': update.get('credential_message', ''),
                            'synthesized': update.get('synthesized', False),
                            'synthesized_status': update.get('synthesized_status', ''),
                            'synthesized_note': update.get('synthesized_note', ''),
                        })
                elif utype == 'data_download_review':
                    # The post-download review gate was never persisted at all,
                    # so a reload while the run waited here left no panel and no
                    # way to answer -- the run just looked stalled. Same
                    # 'pending' contract as the source-select prompt above.
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_download_review',
                            'step': step,
                            'task_id': update.get('task_id', ''),
                            'pending': True,
                            'request_index': update.get('request_index'),
                            'total_requests': update.get('total_requests'),
                            'data_request': update.get('data_request', ''),
                            'status': update.get('status', ''),
                            'error': update.get('error', ''),
                            'data_unavailable': bool(update.get('data_unavailable', False)),
                            'code': update.get('code', ''),
                            'source_name': update.get('source_name', ''),
                            'source_id': update.get('source_id', ''),
                            'summary': update.get('summary', {}),
                        })
                elif utype == 'research_plan_viz':
                    step = update.get('step', 'research_plan')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'research_plan_viz',
                            'viz_id': update.get('viz_id', ''),
                        })
                elif utype == 'geoprocessing_workflow_viz':
                    step = update.get('step', 'geoprocessing_workflow')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'geoprocessing_workflow_viz',
                            'viz_id': update.get('viz_id', ''),
                        })
                elif utype == 'manuscript_section':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_section',
                            'section': update.get('section', ''),
                            'content': update.get('content', ''),
                            'step': step,
                        })
                elif utype == 'manuscript_complete':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_complete',
                            'content': update.get('content', ''),
                            'path': update.get('path', ''),
                            'step': step,
                        })
                elif utype == 'error':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'error',
                            'error': update.get('error', update.get('message', '')),
                            'step': step,
                        })
                elif utype == 'eda_explore_complete':
                    step = update.get('step', 'eda_explore')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'eda_explore_complete',
                            'result': update.get('result', ''),
                            'artifacts': [
                                {
                                    'relative_path': a.get('relative_path', ''),
                                    'filename': a.get('filename', ''),
                                    'size': a.get('size', 0),
                                    'type': a.get('type', ''),
                                }
                                for a in update.get('artifacts', [])
                            ],
                        })
                elif utype == 'result':
                    accumulated_response.append(update.get('message', ''))

                # Link workflow task_id as soon as ANY update carries one, not
                # only on the final `result` event. Users frequently reload the
                # browser mid-run (e.g., at the execution checkpoint); if we
                # wait for the final result, the DB row for this conversation
                # keeps task_id = NULL and the "Proceed to Result Analysis"
                # button fails with a missing task_id error after reload.
                # The link_task_id SQL is idempotent (UPDATE ... WHERE id = ?),
                # but we still guard with a per-conversation latch so we don't
                # hit the DB on every single event.
                if not _task_id_linked_for.get(conversation_id):
                    evt_task_id = update.get('task_id')
                    if evt_task_id:
                        try:
                            link_task_id(conversation_id, evt_task_id)
                            _task_id_linked_for[conversation_id] = evt_task_id
                        except Exception:
                            pass

                # Send each update as an SSE event
                _touch_workflow_card_metrics(workflow_cards, update, wf_current_phase)
                _ckpt.maybe_save(utype)
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client

            # Section total: every LLM call this run made, by phase and model.
            _usage_event = usage_summary_event(_run_ledger[0] if _run_ledger else None)
            if _usage_event:
                yield sse_data(_usage_event)

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Stream complete'})}\n\n"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in chat_with_ai_stream: {error_details}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Save accumulated AGM response — update the incrementally-saved
            # message or create a new one if checkpoint save was skipped.
            try:
                full_response = ''.join(accumulated_response).strip()
                if full_response or workflow_cards:
                    if workflow_cards:
                        # Same row the checkpoints have been updating.
                        _ckpt.convo_id = _ckpt.convo_id or conversation_id
                        _ckpt.save(usage_summary=usage_summary_event(
                            _run_ledger[0] if _run_ledger else None))
                    else:
                        add_message(conversation_id, 'agm', full_response)
            except Exception:
                pass

    # Detached rather than cancelled on disconnect: the pipeline below owns
    # worker threads that keep running regardless, and killing only the
    # generator left them finishing into nothing. See detachable_stream.
    return sse_response(detachable_stream(generate(), 'chat-with-ai'))


@app.route('/api/graph/<task_id>')
def serve_graph(task_id):
    """
    Serve the generated graph HTML file for a specific task
    """
    try:
        from utils.config import load_configurations
        settings = load_configurations()
        # Only serve from the per-task directory — no global fallback so
        # files from other conversations can never bleed through.
        graph_dir = os.path.join(settings.output_dir, task_id, "graphfiles")
        graph_file = os.path.join(graph_dir, f"{task_id}.html")
        if os.path.exists(graph_file):
            return send_from_directory(graph_dir, f"{task_id}.html")
        return jsonify({'error': 'Graph file not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research-plan/<task_id>')
def serve_research_plan(task_id):
    """
    Serve the research plan visualization HTML for a specific task
    """
    try:
        from utils.config import load_configurations
        settings = load_configurations()
        # Serve only from the per-task directory — no global fallback.
        viz_dir = os.path.join(settings.output_dir, task_id, "research_plan_viz")
        viz_file = os.path.join(viz_dir, f"research_plan_{task_id}.html")
        if os.path.exists(viz_file):
            return send_from_directory(viz_dir, f"research_plan_{task_id}.html")
        # DB fallback for shared conversations
        db_resp = _serve_html_from_db(task_id, "research_plan")
        if db_resp:
            return db_resp
        return jsonify({'error': 'Research plan visualization not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research-plan-flowchart/<task_id>')
def serve_research_plan_flowchart(task_id):
    """Serve the research plan flowchart HTML for a specific task"""
    try:
        from utils.config import load_configurations
        settings = load_configurations()
        viz_dir = os.path.join(settings.output_dir, task_id, "research_plan_viz")
        viz_file = os.path.join(viz_dir, f"research_plan_flowchart_{task_id}.html")
        if os.path.exists(viz_file):
            return send_from_directory(viz_dir, f"research_plan_flowchart_{task_id}.html")
        # DB fallback for shared conversations
        db_resp = _serve_html_from_db(task_id, "research_plan_flowchart")
        if db_resp:
            return db_resp
        return jsonify({'error': 'Flowchart not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research-plan-geoprocessing/<task_id>')
def serve_research_plan_geoprocessing(task_id):
    """Serve the ArcGIS Model Builder-style geoprocessing workflow HTML for a specific task"""
    try:
        from utils.config import load_configurations
        settings = load_configurations()
        viz_dir = os.path.join(settings.output_dir, task_id, "research_plan_viz")
        viz_file = os.path.join(viz_dir, f"research_plan_geoprocessing_{task_id}.html")
        if os.path.exists(viz_file):
            return send_from_directory(viz_dir, f"research_plan_geoprocessing_{task_id}.html")
        # DB fallback for shared conversations
        db_resp = _serve_html_from_db(task_id, "research_plan_geoprocessing")
        if db_resp:
            return db_resp
        return jsonify({'error': 'Geoprocessing workflow not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research-plan/update/<task_id>', methods=['POST'])
def update_research_plan_endpoint(task_id):
    """Save inline edits to the research plan (objective/step edits, deletions, additions)."""
    try:
        data = request.get_json()
        if not data or 'objectives' not in data:
            return jsonify({'success': False, 'error': 'Missing objectives'}), 400
        result = update_research_plan(task_id, data['objectives'])
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/research-plan/regen-after-delete/<task_id>', methods=['POST'])
def regen_plan_after_delete_endpoint(task_id):
    """Regenerate affected parts of the plan after a step deletion using LLM."""
    api_key = g.api_key
    try:
        data = request.get_json()
        if not data or 'objectives' not in data or 'deletion_context' not in data:
            return jsonify({'success': False, 'error': 'Missing objectives or deletion_context'}), 400

        if not api_key:
            return jsonify({'success': False, 'error': 'No API key'}), 400
        os.environ['OPENAI_API_KEY'] = api_key

        from WebUI.agm_workflow import regen_plan_after_step_delete
        result = regen_plan_after_step_delete(task_id, data['objectives'], data['deletion_context'])
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rq-breakdown/proceed/<task_id>', methods=['POST'])
def proceed_rq_breakdown(task_id):
    """Resume the workflow after the user clicks Proceed on the Task Breakdown card.

    Streams everything after the Task Breakdown pause point (optional
    data_download, optional EDA, research plan generation) via SSE, and
    persists workflow cards to the conversation message row so a browser
    reload / shared link rebuilds the state correctly.
    """
    api_key = g.api_key

    req_data = request.get_json(silent=True) or {}
    convo_id = req_data.get('conversation_id') or request.args.get('conversation_id')
    skip_data_download = bool(req_data.get('skip_data_download'))
    force_data_download = bool(req_data.get('force_data_download'))
    # The turn the clicked card belongs to; the workflow refuses the click if
    # the saved state has moved on to another turn (see _paused_breakdown).
    card_run_id = req_data.get('run_id') or None
    card_workflow_type = req_data.get('workflow_type') or None
    # Stash any user-provided data-source keys (from the Web UI / localStorage)
    # so the download phase can use them without editing the bundled .keys files.
    from WebUI.agm_workflow import set_data_source_keys, set_user_handbook_dirs
    set_data_source_keys(task_id, req_data.get('data_source_keys') or {})
    # Expose the caller's private, contributed data sources to this task only.
    try:
        from agents.data_agent.user_data_sources import user_handbook_dir
        _uid = hash_api_key(api_key) if api_key else None
        if _uid:
            set_user_handbook_dirs(task_id, [user_handbook_dir(_uid)])
    except Exception as _e:
        print(f"[Workflow] Could not set user handbook dir: {_e}")
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    # Link the task NOW rather than waiting for an event to carry a task_id.
    # Nothing this stream emits does, so a run interrupted anywhere in the
    # download phase left conversations.task_id NULL — and every control that
    # needs a task id (Proceed, re-run, answering a parked approval) had
    # nothing to send. The UPDATE is idempotent.
    if convo_id:
        try:
            link_task_id(convo_id, task_id)
        except Exception:
            pass

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            # No "approved / continuing" status here: it duplicated the
            # workflow-level message and rendered as a stale standalone card.
            # The workflow stream itself announces the next real step
            # ("Selecting data source(s)..." / research-plan status).

            # Accumulate events for conversation persistence (same shape as approve_research_plan).
            workflow_cards = {}
            workflow_order = []
            wf_current_phase = None
            accumulated_response = []
            # This route had no incremental save at all: its whole run lived in
            # the dicts above until the closing `finally`. A reload during the
            # (long, approval-gated) download phase therefore saved a run that
            # was still going as though it had ended there.
            _ckpt = _CardCheckpointer(convo_id, workflow_cards,
                                      workflow_order, accumulated_response)

            _run_ledger = []   # filled by track_usage; holds this run's ledger
            for update in track_usage(continue_after_rq_breakdown_stream(
                    task_id, skip_data_download=skip_data_download,
                    force_data_download=force_data_download,
                    run_id=card_run_id, workflow_type=card_workflow_type), _run_ledger):
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    stream_step = update.get('step', '')
                    is_workflow_stream = stream_step and stream_step != 'streaming_chat'
                    if not is_workflow_stream:
                        accumulated_response.append(update.get('content', ''))
                    if is_workflow_stream:
                        phase = _get_phase(stream_step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': phase, 'body': '', 'step': stream_step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['body'] += update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'llm_update':
                    step = update.get('step', '')
                    silent_steps = ['intent_detected', 'chat_response']
                    if step not in silent_steps:
                        phase = _get_phase(step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['status'] = update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'status':
                    step = update.get('step', 'general')
                    phase = _get_phase(step)
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                        workflow_order.append(phase)
                    workflow_cards[phase]['status'] = update.get('message', '')
                    wf_current_phase = phase
                elif utype == 'research_plan_viz':
                    step = update.get('step', 'research_plan')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'research_plan_viz',
                            'viz_id': update.get('viz_id', ''),
                        })
                elif utype == 'geoprocessing_workflow_viz':
                    step = update.get('step', 'geoprocessing_workflow')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'geoprocessing_workflow_viz',
                            'viz_id': update.get('viz_id', ''),
                        })
                elif utype == 'data_download_summary':
                    step = update.get('step', 'data_download')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_download_summary',
                            'content': update.get('content', ''),
                            'files': update.get('files', []),
                        })
                elif utype == 'data_request_files':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        # One tile panel per request: a retry's files
                        # replace the previous attempt's, as they do live
                        # (displayDataRequestTiles removes the old panel).
                        workflow_cards[phase]['rich_content'] = [
                            rc for rc in workflow_cards[phase]['rich_content']
                            if rc.get('content_type') != 'data_request_files']
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_request_files',
                            'request': update.get('request', ''),
                            'status': update.get('status', ''),
                            'files': update.get('files', []),
                            'error': update.get('error', ''),
                            'data_unavailable': bool(update.get('data_unavailable', False)),
                        })
                elif utype == 'data_source_select_approval':
                    # Persist the source-select review (ideal source, selected
                    # source, explanation, confidence) so it survives reload.
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        # Stored as the whole prompt, not a summary of it. A
                        # reloaded page has to be able to REBUILD this prompt
                        # and answer it -- the run may still be parked on it --
                        # and the fields dropped here were exactly the ones
                        # needed to do that: task_id (nothing to route an answer
                        # to), required_keys (no key inputs), and the credential
                        # / post-synthesis variants (wrong panel entirely).
                        # 'pending' is cleared by the answer endpoint, so a
                        # restored prompt can tell a live question from the
                        # record of a settled one.
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_source_select_approval',
                            'step': step,
                            'task_id': update.get('task_id', ''),
                            'pending': True,
                            'data_request': update.get('data_request', ''),
                            'llm_choice': update.get('llm_choice', ''),
                            'llm_explanation': update.get('llm_explanation', ''),
                            'ideal_source': update.get('ideal_source', ''),
                            'ideal_source_reasoning': update.get('ideal_source_reasoning', ''),
                            'data_to_download': update.get('data_to_download', ''),
                            'confidence': update.get('confidence', 0),
                            'is_recognized': update.get('is_recognized', True),
                            'available_sources': update.get('available_sources', []),
                            'required_keys': update.get('required_keys', []),
                            'caveats': update.get('caveats', ''),
                            'credential_error': update.get('credential_error', False),
                            'credential_missing': update.get('credential_missing', False),
                            'credential_message': update.get('credential_message', ''),
                            'synthesized': update.get('synthesized', False),
                            'synthesized_status': update.get('synthesized_status', ''),
                            'synthesized_note': update.get('synthesized_note', ''),
                        })
                elif utype == 'data_download_review':
                    # The post-download review gate was never persisted at all,
                    # so a reload while the run waited here left no panel and no
                    # way to answer -- the run just looked stalled. Same
                    # 'pending' contract as the source-select prompt above.
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'data_download_review',
                            'step': step,
                            'task_id': update.get('task_id', ''),
                            'pending': True,
                            'request_index': update.get('request_index'),
                            'total_requests': update.get('total_requests'),
                            'data_request': update.get('data_request', ''),
                            'status': update.get('status', ''),
                            'error': update.get('error', ''),
                            'data_unavailable': bool(update.get('data_unavailable', False)),
                            'code': update.get('code', ''),
                            'source_name': update.get('source_name', ''),
                            'source_id': update.get('source_id', ''),
                            'summary': update.get('summary', {}),
                        })
                elif utype == 'step_result':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('step_results', [])
                        workflow_cards[phase]['step_results'].append({
                            'step_index': update.get('step_index', ''),
                            'step_description': update.get('step_description', ''),
                            'status': update.get('status', ''),
                            'output': update.get('output', ''),
                            'code': update.get('code', ''),
                            'objective_key': update.get('objective_key', ''),
                            'error': update.get('error', ''),
                            'error_traceback': update.get('error_traceback', ''),
                            'artifacts': [
                                {
                                    'relative_path': a.get('relative_path', ''),
                                    'filename': a.get('filename', ''),
                                    'size': a.get('size', 0),
                                    'type': a.get('type', ''),
                                    'is_shapefile_component': a.get('is_shapefile_component', False),
                                }
                                for a in update.get('artifacts', [])
                            ],
                        })
                elif utype == 'result':
                    accumulated_response.append(update.get('response', ''))

                _touch_workflow_card_metrics(workflow_cards, update, wf_current_phase)
                _ckpt.maybe_save(utype)
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)

            # Section total: every LLM call this run made, by phase and model.
            _usage_event = usage_summary_event(_run_ledger[0] if _run_ledger else None)
            if _usage_event:
                yield sse_data(_usage_event)

            yield f"data: {json.dumps({'type': 'complete', 'message': 'Workflow advanced past Task Breakdown'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            _ckpt.save(usage_summary=usage_summary_event(
                _run_ledger[0] if _run_ledger else None))

    return sse_response(detachable_stream(generate(), f'rq-proceed:{task_id}'))


@app.route('/api/rq-breakdown/revise-stream/<task_id>', methods=['POST'])
def revise_rq_breakdown_endpoint(task_id):
    """Stream Task Breakdown revision based on the user's feedback.

    Mirrors revise_research_plan_endpoint: a thin wrapper around
    revise_rq_breakdown_stream that relays each update as SSE.

    Also persists the revised rq_breakdown back into the conversation
    message metadata so a browser refresh rebuilds the modified data
    request rather than the original one.
    """
    try:
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({'error': 'Missing feedback'}), 400
        feedback = data['feedback']
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    api_key = g.api_key

    req_data = request.get_json(silent=True) or {}
    card_run_id = req_data.get('run_id') or None
    convo_id = req_data.get('conversation_id') or request.args.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Revising task breakdown...'})}\n\n"
            for update in track_followup_usage(revise_rq_breakdown_stream(task_id, feedback, run_id=card_run_id),
                                               convo_id, initial_phase='rq_breakdown'):
                if (update.get('type') == 'llm_update'
                        and update.get('step') == 'rq_breakdown_complete'):
                    rq_data = (update.get('data') or {}).get('rq_breakdown')
                    rq_task_id = (update.get('data') or {}).get('task_id') or task_id
                    if rq_data and convo_id:
                        try:
                            _update_rq_breakdown_in_message(convo_id, rq_data, rq_task_id)
                        except Exception:
                            pass
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/rq-breakdown/status/<task_id>')
def rq_breakdown_status(task_id):
    """Whether a Task Breakdown card can still Proceed / take feedback."""
    if not re.fullmatch(r'[A-Za-z0-9_.\-]+', task_id) or task_id in ('.', '..'):
        return jsonify({'error': 'invalid task id'}), 400
    return jsonify({'waiting': breakdown_waiting(task_id, request.args.get('run_id') or None)})


@app.route('/api/research-plan/revise-stream/<task_id>', methods=['POST'])
def revise_research_plan_endpoint(task_id):
    """Stream research plan revision based on user feedback."""
    try:
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({'error': 'Missing feedback'}), 400
        feedback = data['feedback']
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    api_key = g.api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Revising research plan...'})}\n\n"
            for update in track_followup_usage(revise_research_plan_stream(task_id, feedback),
                                               _conversation_id_for_task(task_id), initial_phase='research_plan'):
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Research plan revision complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/research-plan/status/<task_id>')
def research_plan_status(task_id):
    """Whether a research plan was approved, for the plan card on reload."""
    if not re.fullmatch(r'[A-Za-z0-9_.\-]+', task_id) or task_id in ('.', '..'):
        return jsonify({'error': 'invalid task id'}), 400
    return jsonify(plan_approval_status(task_id))


@app.route('/api/research-plan/approve/<task_id>', methods=['POST'])
def approve_research_plan(task_id):
    """Approve the research plan and stream objective execution + result analysis."""
    api_key = g.api_key

    # Get conversation_id from request body or query param, fall back to task_id lookup
    req_data = request.get_json(silent=True) or {}
    convo_id = req_data.get('conversation_id') or request.args.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Research plan approved. Starting execution...'})}\n\n"

            # Accumulate events for conversation persistence
            workflow_cards = {}
            workflow_order = []
            wf_current_phase = None
            accumulated_response = []
            _ckpt = _CardCheckpointer(convo_id, workflow_cards,
                                      workflow_order, accumulated_response)

            _run_ledger = []   # filled by track_usage; holds this run's ledger
            for update in track_usage(continue_after_approval_stream(task_id), _run_ledger):
                # Mirror the persistence logic from chat_with_ai_stream
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    stream_step = update.get('step', '')
                    is_workflow_stream = stream_step and stream_step != 'streaming_chat'
                    if not is_workflow_stream:
                        accumulated_response.append(update.get('content', ''))
                    if is_workflow_stream:
                        phase = _get_phase(stream_step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': phase, 'body': '', 'step': stream_step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['body'] += update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'llm_update':
                    step = update.get('step', '')
                    silent_steps = ['intent_detected', 'chat_response']
                    if step not in silent_steps:
                        phase = _get_phase(step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['status'] = update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'status':
                    step = update.get('step', 'general')
                    phase = _get_phase(step)
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                        workflow_order.append(phase)
                    workflow_cards[phase]['status'] = update.get('message', '')
                    wf_current_phase = phase
                elif utype == 'step_result':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('step_results', [])
                        workflow_cards[phase]['step_results'].append({
                            'step_index': update.get('step_index', ''),
                            'step_description': update.get('step_description', ''),
                            'status': update.get('status', ''),
                            'output': update.get('output', ''),
                            'code': update.get('code', ''),
                            'objective_key': update.get('objective_key', ''),
                            'error': update.get('error', ''),
                            'error_traceback': update.get('error_traceback', ''),
                            'artifacts': [
                                {
                                    'relative_path': a.get('relative_path', ''),
                                    'filename': a.get('filename', ''),
                                    'size': a.get('size', 0),
                                    'type': a.get('type', ''),
                                    'is_shapefile_component': a.get('is_shapefile_component', False),
                                }
                                for a in update.get('artifacts', [])
                            ],
                        })
                elif utype == 'execution_checkpoint':
                    phase = 'execution_checkpoint'
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': 'Execution Summary', 'body': '', 'step': 'execution_checkpoint'}
                        workflow_order.append(phase)
                    workflow_cards[phase]['checkpoint_steps'] = update.get('steps', [])
                    # Metrics belong to the checkpoint card; the save itself is
                    # the checkpointer's, which treats this as a pause event and
                    # writes before the generator parks on the Proceed click.
                    _touch_workflow_card_metrics(workflow_cards, update, phase)
                elif utype == 'figure_catalog':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': 'Figure Catalog Completed', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        else:
                            workflow_cards[phase]['status'] = 'Figure Catalog Completed'
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'figure_catalog',
                            'figures': update.get('figures', []),
                            'step': step,
                        })
                elif utype == 'execution_flowchart':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'execution_flowchart',
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'result_presentation':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'result_presentation',
                            'content': update.get('content', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'result_discussion':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'result_discussion',
                            'content': update.get('content', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'code_review':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'code_review',
                            'code': update.get('code', ''),
                            'step_index': update.get('step_index', ''),
                            'step_description': update.get('step_description', ''),
                            'agent_name': update.get('agent_name', ''),
                            'step': step,
                        })
                elif utype == 'manuscript_section':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_section',
                            'section': update.get('section', ''),
                            'content': update.get('content', ''),
                            'step': step,
                        })
                elif utype == 'manuscript_complete':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_complete',
                            'content': update.get('content', ''),
                            'path': update.get('path', ''),
                            'step': step,
                        })
                elif utype == 'error':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'error',
                            'error': update.get('error', update.get('message', '')),
                            'step': step,
                        })
                elif utype == 'result':
                    accumulated_response.append(update.get('message', update.get('response', '')))

                _touch_workflow_card_metrics(workflow_cards, update, wf_current_phase)
                _ckpt.maybe_save(utype)
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)

            # Section total: every LLM call this run made, by phase and model.
            _usage_event = usage_summary_event(_run_ledger[0] if _run_ledger else None)
            if _usage_event:
                yield sse_data(_usage_event)

            yield f"data: {json.dumps({'type': 'complete', 'message': 'Workflow complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Same row the checkpoints have been updating, now with the
            # closing token summary.
            _ckpt.save(usage_summary=usage_summary_event(
                _run_ledger[0] if _run_ledger else None))

    return sse_response(detachable_stream(generate(), f'plan-approve:{task_id}'))


@app.route('/api/geoprocessing-workflow/revise-stream/<task_id>', methods=['POST'])
def revise_geoprocessing_workflow_endpoint(task_id):
    """Stream geoprocessing workflow revision based on the user's feedback."""
    api_key = g.api_key
    req_data = request.get_json(silent=True) or {}
    feedback = req_data.get('feedback', '')
    # track_followup_usage records this revision's token use against the
    # conversation; convo_id was never set here, so every geoprocessing
    # feedback failed with "name 'convo_id' is not defined".
    convo_id = (req_data.get('conversation_id') or request.args.get('conversation_id')
                or _conversation_id_for_task(task_id))

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            for update in track_followup_usage(revise_geoprocessing_workflow_stream(task_id, feedback),
                                               convo_id, initial_phase='geoprocessing_workflow'):
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Geoprocessing workflow revision complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/spatial-analysis/approve/<task_id>', methods=['POST'])
def spatial_analysis_approve_workflow(task_id):
    """Approve the geoprocessing workflow and stream operation code generation."""
    api_key = g.api_key

    req_data = request.get_json(silent=True) or {}
    convo_id = req_data.get('conversation_id') or request.args.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            # NOTE: do not emit an "approved / starting" status here. The
            # continuation stream (continue_after_geoprocessing_approval_stream)
            # already yields that exact message as its first status update, so
            # emitting it here too renders the line twice in the chat.

            workflow_cards = {}
            workflow_order = []
            wf_current_phase = None
            accumulated_response = []
            _ckpt = _CardCheckpointer(convo_id, workflow_cards,
                                      workflow_order, accumulated_response)

            _run_ledger = []   # filled by track_usage; holds this run's ledger
            for update in track_usage(
                    spatial_analysis_continue_after_approval_stream(task_id), _run_ledger):
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    stream_step = update.get('step', '')
                    is_workflow_stream = stream_step and stream_step != 'streaming_chat'
                    if not is_workflow_stream:
                        accumulated_response.append(update.get('content', ''))
                    if is_workflow_stream:
                        phase = _get_phase(stream_step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': phase, 'body': '', 'step': stream_step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['body'] += update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'llm_update':
                    step = update.get('step', '')
                    silent_steps = ['intent_detected', 'chat_response']
                    if step not in silent_steps:
                        phase = _get_phase(step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['status'] = update.get('content', '')
                        wf_current_phase = phase
                elif utype == 'status':
                    step = update.get('step', 'general')
                    phase = _get_phase(step)
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                        workflow_order.append(phase)
                    workflow_cards[phase]['status'] = update.get('message', '')
                    wf_current_phase = phase
                elif utype == 'step_result':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('step_results', [])
                        workflow_cards[phase]['step_results'].append({
                            'step_index': update.get('step_index', ''),
                            'step_description': update.get('step_description', ''),
                            'status': update.get('status', ''),
                            'output': update.get('output', ''),
                            'code': update.get('code', ''),
                            'objective_key': update.get('objective_key', ''),
                            'error': update.get('error', ''),
                            'error_traceback': update.get('error_traceback', ''),
                            'artifacts': [
                                {
                                    'relative_path': a.get('relative_path', ''),
                                    'filename': a.get('filename', ''),
                                    'size': a.get('size', 0),
                                    'type': a.get('type', ''),
                                    'is_shapefile_component': a.get('is_shapefile_component', False),
                                }
                                for a in update.get('artifacts', [])
                            ],
                        })
                elif utype == 'error':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'error',
                            'error': update.get('error', update.get('message', '')),
                            'step': step,
                        })
                elif utype == 'result':
                    accumulated_response.append(update.get('message', update.get('response', '')))

                _touch_workflow_card_metrics(workflow_cards, update, wf_current_phase)
                _ckpt.maybe_save(utype)
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)

            # Section total: every LLM call this run made, by phase and model.
            _usage_event = usage_summary_event(_run_ledger[0] if _run_ledger else None)
            if _usage_event:
                yield sse_data(_usage_event)

            yield f"data: {json.dumps({'type': 'complete', 'message': 'Operation code generation complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Was an unconditional add_message, which would now append a second
            # copy of a run the checkpoints have already written. Going through
            # the checkpointer updates that same row instead.
            _ckpt.save(usage_summary=usage_summary_event(
                _run_ledger[0] if _run_ledger else None))

    return sse_response(detachable_stream(generate(), f'spatial-approve:{task_id}'))


# =====================================================================
#  POST-COMPLETION STEP EDITING & RE-RUNNING
# =====================================================================

@app.route('/api/workflow/steps/<task_id>', methods=['GET'])
def get_workflow_steps(task_id):
    """Return all executed steps with code, status, and dependency info."""
    try:
        result = get_completed_steps(task_id)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/workflow/rerun-step/<task_id>', methods=['POST'])
def rerun_step_endpoint(task_id):
    """Re-run a single step with edited or regenerated code. Streaming SSE response."""
    api_key = g.api_key
    data = request.get_json() or {}
    objective_key = data.get('objective_key', '')
    step_index = int(data.get('step_index', 0))
    mode = data.get('mode', 'edit')
    edited_code = data.get('code', None)
    extra_instructions = data.get('extra_instructions', '')

    convo_id = data.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': f'Re-running step {step_index}...'})}\n\n"
            for update in track_followup_usage(
                    rerun_step_stream(task_id, objective_key, step_index,
                                      mode=mode, edited_code=edited_code,
                                      extra_instructions=extra_instructions),
                    convo_id, fixed_phase=_exec_step_phase(objective_key, step_index)):
                if update.get('type') == 'step_result':
                    try:
                        _update_workflow_step_result(
                            convo_id, update,
                            objective_key_override=objective_key,
                            step_index_override=step_index,
                        )
                    except Exception:
                        pass
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Step re-run complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/rerun-assembly/<task_id>', methods=['POST'])
def rerun_assembly_endpoint(task_id):
    """Re-run the assembly program with edited/rerun/regenerated code. Streaming SSE response."""
    api_key = g.api_key
    data = request.get_json() or {}
    mode = data.get('mode', 're_execute')
    edited_code = data.get('code', None)
    extra_instructions = data.get('extra_instructions', '')

    convo_id = data.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Re-running assembly program...'})}\n\n"
            for update in track_followup_usage(
                    rerun_assembly_stream(task_id, mode=mode, edited_code=edited_code,
                                          extra_instructions=extra_instructions),
                    convo_id, fixed_phase='assembly_execution'):
                if update.get('type') == 'step_result':
                    try:
                        _update_workflow_step_result(
                            convo_id, update,
                            objective_key_override='assembly',
                            step_index_override='Execution',
                        )
                    except Exception:
                        pass
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Assembly re-run complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/cascade-rerun/<task_id>', methods=['POST'])
def cascade_rerun_endpoint(task_id):
    """Cascade re-run to downstream dependent steps. Streaming SSE response."""
    api_key = g.api_key
    data = request.get_json() or {}
    objective_key = data.get('objective_key', '')
    step_indices = data.get('step_indices', [])

    convo_id = data.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Cascading re-run to dependent steps...'})}\n\n"
            for update in _tracked_cascade(task_id, objective_key, step_indices, convo_id):
                if update.get('type') == 'step_result':
                    try:
                        _update_workflow_step_result(
                            convo_id, update,
                            objective_key_override=objective_key,
                        )
                    except Exception:
                        pass
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Cascade re-run complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/rerun-analysis/<task_id>', methods=['POST'])
def rerun_analysis_endpoint(task_id):
    """Re-run the Result Analysis pipeline (Step 7). Streaming SSE response."""
    api_key = g.api_key

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    # Resolve conversation_id for DB persistence
    req_data = request.get_json(silent=True) or {}
    convo_id = req_data.get('conversation_id') or request.args.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    # Overlap guard — refuse a second rerun for a task already in flight.
    # Return 409 so the UI can surface a clean message instead of a
    # silent failure (prior behaviour: two pipelines writing the same
    # Outputs/<task_id>/ directory with interleaved logs).
    # Through the package, like every other import here: the bare name only
    # resolves when WebUI/ is on sys.path (python WebUI/app.py), not under
    # PythonAnywhere's WSGI, and it loads a second copy whose set the
    # re-runs never register in.
    from WebUI.agm_workflow import _active_reruns as _active_rerun_set
    if task_id in _active_rerun_set:
        return jsonify({
            'error': 'rerun_already_running',
            'message': f'A result-analysis rerun is already running for task {task_id}. '
                       f'Wait for it to finish before starting another.',
        }), 409

    def generate():
        if not api_key:
            yield _no_key_event()
            return

        # Accumulate cards for DB persistence
        workflow_cards = {}
        workflow_order = []
        wf_current_phase = None

        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Re-running result analysis pipeline...'})}\n\n"
            for update in track_followup_usage(rerun_result_analysis_stream(task_id),
                                               convo_id, initial_phase='result_analysis'):
                # Accumulate cards (same logic as approve endpoint)
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    step = update.get('step', '')
                    if step:
                        phase = _get_phase(step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': phase, 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['body'] += update.get('content', '')
                        wf_current_phase = phase
                elif utype in ('status', 'llm_update'):
                    step = update.get('step', 'general')
                    phase = _get_phase(step)
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                        workflow_order.append(phase)
                    workflow_cards[phase]['status'] = update.get('message', update.get('content', ''))
                    wf_current_phase = phase
                elif utype == 'figure_catalog':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': 'Figure Catalog Completed', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        else:
                            workflow_cards[phase]['status'] = 'Figure Catalog Completed'
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'figure_catalog',
                            'figures': update.get('figures', []),
                            'step': step,
                        })
                elif utype == 'execution_flowchart':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'execution_flowchart',
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'result_presentation':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'result_presentation',
                            'content': update.get('content', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'result_discussion':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'result_discussion',
                            'content': update.get('content', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })
                elif utype == 'manuscript_section':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_section',
                            'section': update.get('section', ''),
                            'content': update.get('content', ''),
                            'step': step,
                        })
                elif utype == 'manuscript_complete':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_complete',
                            'content': update.get('content', ''),
                            'path': update.get('path', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })

                _touch_workflow_card_metrics(workflow_cards, update, wf_current_phase)
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Result analysis re-run complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Save result analysis + manuscript cards to the conversation DB
            try:
                if convo_id and workflow_cards:
                    cards_list = []
                    for phase in workflow_order:
                        card = _workflow_card_for_persistence(workflow_cards[phase])
                        card['phase'] = phase
                        cards_list.append(card)
                    metadata = {'workflow_cards': cards_list}
                    add_message(convo_id, 'agm', '',
                                message_type='workflow', metadata=metadata)
            except Exception:
                pass

    return sse_response(generate())


@app.route('/api/workflow/update-analysis/<task_id>', methods=['POST'])
def update_analysis_endpoint(task_id):
    """Update result analysis (flowchart, presentation, discussion) without
    re-verifying or fixing steps.  Streaming SSE response."""
    api_key = g.api_key

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Updating result analysis...'})}\n\n"
            for update in track_followup_usage(update_result_analysis_stream(task_id),
                                               _conversation_id_for_task(task_id), initial_phase='result_analysis'):
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Result analysis updated'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/rerun-manuscript/<task_id>', methods=['POST'])
def rerun_manuscript_endpoint(task_id):
    """Re-run manuscript generation without tool calls. Optional `sections`
    list in the POST body restricts regeneration to specific sections.
    Streaming SSE response."""
    api_key = g.api_key

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    data = request.get_json(silent=True) or {}
    sections = data.get('sections') or None
    if sections is not None and not isinstance(sections, list):
        sections = [str(sections)]

    # Resolve conversation_id for DB persistence
    convo_id = data.get('conversation_id') or request.args.get('conversation_id')
    if not convo_id:
        # Resolve via the shared helper, which also falls back to treating the
        # task id AS the conversation id. These workflows name the output dir
        # after the conversation, but conversations.task_id stays NULL on any
        # path that never called link_task_id — so the old lookup returned
        # None and silently disabled every DB-persistence branch below.
        convo_id = _conversation_id_for_task(task_id)

    def generate():
        if not api_key:
            yield _no_key_event()
            return

        workflow_cards = {}
        workflow_order = []
        wf_current_phase = None

        try:
            label = "all sections" if not sections else ", ".join(sections)
            yield f"data: {json.dumps({'type': 'status', 'message': f'Rerunning manuscript ({label})...'})}\n\n"
            for update in track_followup_usage(
                    rerun_manuscript_stream(task_id, sections_to_update=sections),
                    convo_id, initial_phase='manuscript'):
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    step = update.get('step', '')
                    if step:
                        phase = _get_phase(step)
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': phase, 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase]['body'] += update.get('content', '')
                        wf_current_phase = phase
                elif utype in ('status', 'llm_update'):
                    step = update.get('step', 'general')
                    phase = _get_phase(step)
                    if phase not in workflow_cards:
                        workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                        workflow_order.append(phase)
                    workflow_cards[phase]['status'] = update.get('message', update.get('content', ''))
                    wf_current_phase = phase
                elif utype == 'manuscript_section_start':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {
                                'status': f"Updating {update.get('display_name', '')}...",
                                'body': '',
                                'step': step,
                            }
                            workflow_order.append(phase)
                        wf_current_phase = phase
                elif utype == 'manuscript_section':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_section',
                            'section': update.get('section', ''),
                            'content': update.get('content', ''),
                            'step': step,
                        })
                elif utype == 'manuscript_complete':
                    step = update.get('step', '')
                    phase = _get_phase(step) if step else wf_current_phase
                    if phase:
                        if phase not in workflow_cards:
                            workflow_cards[phase] = {'status': '', 'body': '', 'step': step}
                            workflow_order.append(phase)
                        workflow_cards[phase].setdefault('rich_content', [])
                        workflow_cards[phase]['rich_content'].append({
                            'content_type': 'manuscript_complete',
                            'content': update.get('content', ''),
                            'path': update.get('path', ''),
                            'relative_path': update.get('relative_path', ''),
                            'step': step,
                        })

                _touch_workflow_card_metrics(workflow_cards, update, wf_current_phase)
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Manuscript rerun complete'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            try:
                if convo_id and workflow_cards:
                    cards_list = []
                    for phase in workflow_order:
                        card = _workflow_card_for_persistence(workflow_cards[phase])
                        card['phase'] = phase
                        cards_list.append(card)
                    metadata = {'workflow_cards': cards_list}
                    add_message(convo_id, 'agm', '',
                                message_type='workflow', metadata=metadata)
            except Exception:
                pass

    return sse_response(generate())


@app.route('/api/cancel', methods=['POST'])
def cancel_workflow():
    """Signal all running backend workflow threads to stop and clean up state."""
    request_cancellation()
    # Give threads a moment to see the flag, then do full cleanup
    import threading
    from WebUI.agm_workflow import _with_flask_ctx
    def _deferred_cleanup():
        import time
        time.sleep(2)  # wait for threads to notice cancellation
        full_cleanup()
    threading.Thread(target=_with_flask_ctx(_deferred_cleanup), daemon=True).start()
    return jsonify({'success': True, 'message': 'Cancellation requested'})


@app.route('/api/workflow/code-review/<task_id>', methods=['POST'])
def code_review_endpoint(task_id):
    """Receive user's code review decision (approve/feedback/skip) and relay to executor."""
    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({'success': False, 'error': 'Missing action'}), 400

    action = data['action']  # "approve", "feedback", or "skip"
    response = {
        "action": action,
        "code": data.get("code", ""),
        "feedback": data.get("feedback", ""),
    }
    submit_code_review(task_id, response)
    return jsonify({'success': True, 'message': f'Code review action: {action}'})


@app.route('/api/workflow/checkpoint/<task_id>', methods=['POST'])
def checkpoint_endpoint(task_id):
    """Receive user's checkpoint decision (proceed or update_status)."""
    from WebUI.agm_workflow import submit_checkpoint_response
    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({'success': False, 'error': 'Missing action'}), 400
    queued = submit_checkpoint_response(task_id, data)
    return jsonify({'success': True, 'queued': queued, 'message': f'Checkpoint action: {data["action"]}'})


@app.route('/api/eda-approval', methods=['POST'])
def eda_approval_endpoint():
    """Receive user's response to an EDA approval request and relay to the EDA tool.
    Supports both JSON (text options) and multipart/form-data (file upload)."""
    if request.content_type and 'multipart/form-data' in request.content_type:
        # File upload response
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        from werkzeug.utils import secure_filename
        conv_id = request.form.get('conversation_id')
        if conv_id:
            upload_dir = os.path.join(BASE_DIR, 'uploads', conv_id)
        else:
            upload_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        # Audit the uploaded file
        audit_records = audit_uploaded_data([file_path], [filename])
        audit_record = audit_records[0] if audit_records else {}

        # Send structured response so the tool can load and register the dataset
        submit_eda_approval({
            "type": "file_upload",
            "file_path": file_path,
            "file_name": filename,
            "audit": audit_record,
        })
        return jsonify({'success': True, 'message': f'File uploaded and audited: {filename}', 'file_path': file_path})
    else:
        # Text option response
        data = request.get_json()
        if not data or 'response' not in data:
            return jsonify({'success': False, 'error': 'Missing response'}), 400

        submit_eda_approval(data['response'])
        return jsonify({'success': True, 'message': 'EDA approval submitted'})


@app.route('/api/data-download-approval', methods=['POST'])
def data_download_approval_endpoint():
    """Receive user's response when one or more data download requests failed.
    Supports JSON (text/button choice) and multipart/form-data (file upload)."""
    if request.content_type and 'multipart/form-data' in request.content_type:
        # File upload response — user is providing a missing file manually
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        from werkzeug.utils import secure_filename
        conv_id = request.form.get('conversation_id')
        if conv_id:
            upload_dir = os.path.join(BASE_DIR, 'uploads', conv_id)
        else:
            upload_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        # Audit the uploaded file so downstream steps have metadata
        audit_records = audit_uploaded_data([file_path], [filename])
        audit_record = audit_records[0] if audit_records else {}

        submit_data_download_approval({
            "type": "file_upload",
            "file_path": file_path,
            "file_name": filename,
            "audit": audit_record,
        })
        return jsonify({
            'success': True,
            'message': f'File uploaded and audited: {filename}',
            'file_path': file_path,
        })
    else:
        # Text/button choice response
        data = request.get_json()
        if not data or 'response' not in data:
            return jsonify({'success': False, 'error': 'Missing response'}), 400

        submit_data_download_approval(data['response'])
        return jsonify({'success': True, 'message': 'Data download approval submitted'})


@app.route('/api/data-source-select', methods=['POST'])
def data_source_select_endpoint():
    """Receive the user's choice when the DataRetrieverAgent could not
    recognize its LLM-selected data source. Expected JSON shape:
        {"action": "use", "source": "OpenStreetMap"}
        {"action": "retry"}
        {"action": "abort"}
    """
    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({'success': False, 'error': 'Missing action'}), 400
    task_id = data.get('task_id')
    # Cross-process hand-off: persist the answer to disk so the worker
    # PROCESS running the download stream can pick it up even when this POST
    # was served by a different worker (the case on multi-worker hosting).
    if task_id:
        from WebUI.agm_workflow import save_source_select_answer
        save_source_select_answer(task_id, data)
    # Also feed the in-memory queue for the same-process / local-dev case.
    submit_source_select_approval(data)
    # The stored prompt stops being a live question now that it has an answer.
    if task_id:
        _resolve_pending_approval(task_id, 'data_source_select_approval')
    return jsonify({'success': True, 'message': f'Source select action: {data["action"]}'})


@app.route('/api/workflow/pending-approval/<task_id>', methods=['GET'])
def pending_approval_endpoint(task_id):
    """Report whether a worker is still parked on an approval for this task.

    A restored prompt looks identical whether the run behind it is alive and
    waiting or died months ago, and the page needs to tell those apart before
    it offers buttons: answering a dead prompt writes a file nobody will ever
    read, which looks to the user exactly like the app ignoring the click.

    ``waiting`` is true only while the marker's heartbeat is fresh, so a
    server that was restarted (or killed) mid-wait reports false and the page
    can say the run ended instead of pretending it can be resumed.
    """
    from WebUI.agm_workflow import read_pending_prompt, PENDING_PROMPT_STALE_AFTER
    record = read_pending_prompt(task_id)
    if not record:
        return jsonify({'waiting': False, 'stale': False})
    return jsonify({
        'waiting': not record.get('stale', False),
        'stale': bool(record.get('stale', False)),
        'kind': record.get('kind', ''),
        'step': record.get('step', ''),
        'age_seconds': record.get('age_seconds', 0),
        'stale_after_seconds': PENDING_PROMPT_STALE_AFTER,
    })


@app.route('/api/data-download/rerun/<task_id>', methods=['POST'])
def rerun_data_download_endpoint(task_id):
    """Re-run one already-completed data request, after its review gate closed.

    The gate is the only moment the workflow thread is parked, so the card's
    "Edit code & re-run" / "Feedback & re-run" controls used to die the moment
    the user proceeded. This re-enters the downloader on demand instead,
    against the source that request already resolved to.

    Expected JSON:
        {"request_index": 1, "action": "rerun"}
        {"request_index": 1, "action": "edit_code", "code": "..."}
        {"request_index": 1, "action": "feedback", "feedback": "..."}
    """
    api_key = g.api_key
    data = request.get_json() or {}
    try:
        request_index = int(data.get('request_index'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Missing or invalid request_index'}), 400
    action = (data.get('action') or 'rerun').strip().lower()
    if action not in ('rerun', 'edit_code', 'feedback'):
        return jsonify({'error': f'Unsupported action: {action}'}), 400
    code = data.get('code')
    feedback = data.get('feedback')
    if action == 'edit_code' and not (code or '').strip():
        return jsonify({'error': 'No code supplied for edit_code'}), 400
    if action == 'feedback' and not (feedback or '').strip():
        return jsonify({'error': 'No feedback supplied'}), 400

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        _run_ledger = []   # filled by track_usage; holds this run's ledger
        # What the re-run streamed, for writing back into the saved run once
        # it finishes (see _update_rerun_in_message).
        streamed, files_event, outcome = [], None, None
        try:
            for update in track_usage(rerun_data_download_stream(
                    task_id, request_index, action=action,
                    code=code, feedback=feedback), _run_ledger):
                utype = update.get('type', '')
                if utype == 'llm_stream':
                    streamed.append(update.get('content', ''))
                elif utype == 'data_request_files':
                    files_event = update
                elif utype == 'data_download_rerun_complete':
                    outcome = update
                yield sse_data(update)
                time.sleep(0.01)

            if outcome is not None:
                # The spatial-analysis / data-retriever pipelines use the
                # conversation id as the task id, so this is the conversation.
                if not _update_rerun_in_message(task_id, request_index,
                                                ''.join(streamed), files_event,
                                                outcome):
                    app.logger.warning(
                        'Download re-run of request %s in %s was not written '
                        'back to the conversation; a reload will show the '
                        'original download.', request_index, task_id)

            # Section total: every LLM call this run made, by phase and model.
            _usage_event = usage_summary_event(_run_ledger[0] if _run_ledger else None)
            if _usage_event:
                yield sse_data(_usage_event)

            yield sse_data({'type': 'complete', 'message': 'Download re-run complete'})
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sse_data({'type': 'error', 'error': str(e)})

    return sse_response(generate())


@app.route('/api/data-download/rerun-available/<task_id>/<int:request_index>')
def rerun_data_download_available(task_id, request_index):
    """Whether this request can still be re-run, and the code it last ran.

    The card asks before offering the controls: a download made before the
    context was recorded cannot be re-run, and offering a button that always
    errors is worse than not offering it.
    """
    context = get_rerun_context(task_id, request_index)
    return jsonify({
        'available': bool(context and context.get('source_id')),
        'code': (context or {}).get('code', ''),
        'source_name': (context or {}).get('source_name', ''),
    })


@app.route('/api/data-download-review', methods=['POST'])
def data_download_review_endpoint():
    """Receive the user's choice at the post-download review checkpoint.

    Shown after each data request completes in an interactive (non-autonomous)
    run, so a person can see what actually came back — record counts, columns,
    CRS/extent — before the workflow consumes it. Expected JSON:
        {"action": "proceed"}                       accept and continue
        {"action": "always_proceed"}                accept, skip later reviews
        {"action": "finish"}                        accept, end the whole stage
        {"action": "rerun"}                         regenerate + re-execute
        {"action": "edit_code", "code": "..."}      run the user's edited code
        {"action": "feedback", "feedback": "..."}   regenerate with guidance
        {"action": "skip"}                          drop this dataset
    """
    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({'success': False, 'error': 'Missing action'}), 400
    task_id = data.get('task_id')
    # Dual-channel hand-off (disk + in-memory queue), same as source-select:
    # on multi-worker hosting the POST and the download stream are usually
    # served by different worker processes.
    if task_id:
        from WebUI.agm_workflow import save_download_review_answer
        save_download_review_answer(task_id, data)
    submit_download_review_response(data)
    if task_id:
        _resolve_pending_approval(task_id, 'data_download_review')
    return jsonify({'success': True,
                    'message': f'Download review action: {data["action"]}'})


# ── User-contributed data sources ────────────────────────────────────────────
# Lets users register a retrievable data source (a "handbook") from the UI. The
# source is saved on the server under the caller's user_id and is usable
# immediately in their own sessions (Tier 1), and auto-submitted to a review
# queue an admin can promote into the global catalog (Tier 2). See
# agents/data_agent/user_data_sources.py.

def _is_admin(api_key):
    """A caller is an admin if their key (or its hash) is listed in the
    AGM_ADMIN_API_KEYS / AGM_ADMIN_USER_IDS environment variables
    (comma-separated)."""
    if not api_key:
        return False
    raw = {k.strip() for k in os.environ.get('AGM_ADMIN_API_KEYS', '').split(',') if k.strip()}
    ids = {k.strip() for k in os.environ.get('AGM_ADMIN_USER_IDS', '').split(',') if k.strip()}
    if api_key in raw:
        return True
    return hash_api_key(api_key) in ids


@app.route('/api/whoami', methods=['GET'])
def whoami_endpoint():
    """Return the caller's user_id (hash) and whether they are an admin — used
    by the UI to decide whether to show the review queue."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    return jsonify({
        'success': True,
        'user_id': hash_api_key(api_key),
        'is_admin': _is_admin(api_key),
    })


# ── Handbook Generator Experiment 2 studies ─────────────────────────────────

@app.route('/api/handbook-experiments', methods=['GET', 'POST'])
def handbook_experiments_endpoint():
    """List studies or create a new server-persisted Experiment 2 study."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import experiment2_store as store
    user_id = hash_api_key(api_key)
    if request.method == 'GET':
        return jsonify({'success': True, 'studies': store.list_studies(user_id)})
    try:
        body = request.get_json(silent=True) or {}
        study = store.save_study(user_id, body)
        return jsonify({'success': True, 'study': study}), 201
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/handbook-experiments/<study_id>',
           methods=['GET', 'PUT', 'DELETE'])
def handbook_experiment_endpoint(study_id):
    """Read, update, or delete one caller-owned Experiment 2 study."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import experiment2_store as store
    user_id = hash_api_key(api_key)
    try:
        study = store.get_study(user_id, study_id)
        if study is None:
            return jsonify({'success': False, 'error': 'Experiment not found'}), 404
        if request.method == 'GET':
            return jsonify({'success': True, 'study': study})
        if request.method == 'DELETE':
            store.delete_study(user_id, study_id)
            return jsonify({'success': True})
        updated = store.save_study(
            user_id, request.get_json(silent=True) or {}, study_id=study_id)
        return jsonify({'success': True, 'study': updated})
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/handbook-experiments/<study_id>/run-stream',
           methods=['POST'])
def handbook_experiment_run_stream(study_id):
    """Run the live Experiment 2 handbook/refinement pipeline over SSE."""
    api_key = g.api_key
    if not api_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    data_source_keys = body.get('data_source_keys') or {}
    restart = bool(body.get('restart'))
    confirm_mechanisms = body.get('confirm_mechanisms') or []
    if not isinstance(data_source_keys, dict):
        return jsonify({
            'success': False, 'error': 'data_source_keys must be an object.'
        }), 400
    if not isinstance(confirm_mechanisms, list):
        return jsonify({
            'success': False, 'error': 'confirm_mechanisms must be an array.'
        }), 400
    user_id = hash_api_key(api_key)

    def generate():
        from WebUI.experiment2_runner import run_experiment_pipeline
        try:
            for event in run_experiment_pipeline(
                    user_id, api_key, study_id,
                    user_keys=data_source_keys, restart=restart,
                    confirm_mechanisms=confirm_mechanisms):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Experiment 2 batch execution failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(generate())


@app.route('/api/handbook-experiments/<study_id>/cancel', methods=['POST'])
def handbook_experiment_cancel(study_id):
    """Request cooperative cancellation of a running Experiment 2 batch."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import experiment2_store as store
    from WebUI.experiment2_runner import request_cancel
    user_id = hash_api_key(api_key)
    try:
        if store.get_study(user_id, study_id) is None:
            return jsonify({
                'success': False, 'error': 'Experiment not found'
            }), 404
        request_cancel(user_id, study_id)
        return jsonify({
            'success': True,
            'message': 'Cancellation requested. The active trial will stop at '
                       'the next safe checkpoint.',
        })
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/handbook-experiments/<study_id>/output-preview',
           methods=['GET'])
def handbook_experiment_output_preview(study_id):
    """Preview or download one caller-owned Experiment 2 output artifact."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401

    from WebUI import experiment2_store as store
    from WebUI.experiment2_runner import experiment_output_roots

    user_id = hash_api_key(api_key)
    try:
        if store.get_study(user_id, study_id) is None:
            return jsonify({
                'success': False, 'error': 'Experiment not found'
            }), 404

        study_roots = experiment_output_roots(user_id, study_id)
        study_root = study_roots[0]
        artifact_ref = str(request.args.get('ref') or '').strip().replace('\\', '/')
        candidate = None

        def containing_root(path):
            try:
                absolute = os.path.abspath(path)
                return next((
                    root for root in study_roots
                    if os.path.commonpath([root, absolute]) == root
                ), None)
            except (OSError, ValueError):
                return None

        if artifact_ref:
            if artifact_ref.startswith('/') or '..' in artifact_ref.split('/'):
                return jsonify({
                    'success': False, 'error': 'Invalid output reference'
                }), 400
            parts = [part for part in artifact_ref.split('/') if part]
            for root in study_roots:
                possible = os.path.abspath(os.path.join(root, *parts))
                if os.path.isfile(possible):
                    candidate = possible
                    study_root = root
                    break
            if candidate is None:
                candidate = os.path.abspath(os.path.join(study_roots[0], *parts))
            if not containing_root(candidate):
                return jsonify({
                    'success': False, 'error': 'Output access denied'
                }), 403
        else:
            # Compatibility for runs created before artifact references were
            # persisted: resolve the recorded file name only inside this
            # caller-owned study and prefer the newest exact size match.
            requested_name = os.path.basename(
                str(request.args.get('name') or '').strip())
            if not requested_name:
                return jsonify({
                    'success': False, 'error': 'Output reference is required'
                }), 400
            try:
                requested_size = int(request.args.get('size') or -1)
            except (TypeError, ValueError):
                requested_size = -1
            matches = []
            for root in study_roots:
                if not os.path.isdir(root):
                    continue
                for directory, dirnames, filenames in os.walk(root):
                    dirnames[:] = [name for name in dirnames
                                   if name != '_handbooks']
                    if requested_name in filenames:
                        path = os.path.join(directory, requested_name)
                        try:
                            if (requested_size >= 0
                                    and os.path.getsize(path) != requested_size):
                                continue
                            matches.append(path)
                        except OSError:
                            continue
            if matches:
                candidate = max(matches, key=os.path.getmtime)
                study_root = containing_root(candidate) or study_roots[0]

        if (not candidate or not containing_root(candidate)
                or not os.path.isfile(candidate)):
            return jsonify({
                'success': False, 'error': 'Output file is no longer available'
            }), 404

        study_root = containing_root(candidate) or study_root
        artifact_ref = os.path.relpath(
            candidate, study_root).replace(os.sep, '/')
        filename = os.path.basename(candidate)
        size_bytes = os.path.getsize(candidate)
        mime_type = (
            mimetypes.guess_type(filename)[0] or 'application/octet-stream')

        if request.args.get('download') == '1':
            return send_file(
                candidate, as_attachment=True, download_name=filename,
                mimetype=mime_type, conditional=True)

        if request.args.get('raw') == '1':
            if not (mime_type.startswith('image/')
                    or mime_type == 'application/pdf'):
                return jsonify({
                    'success': False,
                    'error': 'Inline binary preview is not available for this file type',
                }), 415
            response = send_file(
                candidate, as_attachment=False, download_name=filename,
                mimetype=mime_type, conditional=True, max_age=0)
            response.headers['Cache-Control'] = 'private, no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        extension = os.path.splitext(filename)[1].casefold()
        text_extensions = {
            '.txt', '.csv', '.tsv', '.json', '.geojson', '.xml', '.osm',
            '.html', '.htm', '.py', '.md', '.toml', '.yaml', '.yml', '.log',
            '.sql', '.js', '.css', '.prj', '.cpg',
        }
        image_types = {
            'image/png', 'image/jpeg', 'image/gif', 'image/webp',
        }
        preview = {
            'artifact_ref': artifact_ref,
            'name': filename,
            'size_bytes': size_bytes,
            'mime_type': mime_type,
            'kind': 'binary',
            'content': '',
            'truncated': False,
        }
        if mime_type in image_types:
            preview['kind'] = 'image'
        elif mime_type == 'application/pdf':
            preview['kind'] = 'pdf'
        elif (extension in text_extensions or mime_type.startswith('text/')
              or mime_type in (
                  'application/json', 'application/geo+json',
                  'application/xml', 'application/javascript')):
            limit = 300_000
            with open(candidate, 'rb') as handle:
                raw = handle.read(limit + 1)
            preview['truncated'] = len(raw) > limit
            text = raw[:limit].decode('utf-8', errors='replace')
            if extension in ('.json', '.geojson') and not preview['truncated']:
                try:
                    text = json.dumps(
                        json.loads(text), ensure_ascii=False, indent=2)
                except (ValueError, TypeError):
                    pass
            preview['kind'] = 'text'
            preview['content'] = text

        return jsonify({'success': True, 'preview': preview})
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        app.logger.exception('Could not preview Experiment 2 output')
        return jsonify({'success': False, 'error': str(exc)}), 500


# ── "Generate & Test" handbook sessions ─────────────────────────────────────
# A separate, simpler, OpenAI-only mode: one source, one handbook, one
# retrieval test, with a human review pause in between. Deliberately
# independent of the Experiment 2 routes/store/runner above.

@app.route('/api/handbook-studio-providers', methods=['GET'])
def handbook_studio_providers():
    """Which "Generate & Test" LLM providers are actually usable right now --
    lets the setup UI gray out Claude Agent SDK with a real reason instead of
    just failing on first use if the server-side package/CLI isn't there."""
    from agents.data_agent import claude_agent_provider
    return jsonify({
        'success': True,
        'providers': {
            'openai': {'available': True},
            'claude': {
                'available': claude_agent_provider.available(),
                'reason': claude_agent_provider.unavailable_reason(),
            },
        },
    })


@app.route('/api/handbook-studio/validation-spec-template')
def handbook_studio_validation_spec_template():
    """The commented starter validation spec, served from the one place it
    is defined so the UI template can't drift from the parser."""
    from agents.data_agent.validation import SPEC_TEMPLATE
    return jsonify({'success': True, 'template': SPEC_TEMPLATE})


@app.route('/api/handbook-studio-sessions', methods=['GET', 'POST'])
def handbook_studio_sessions_endpoint():
    """List sessions or create a new server-persisted Generate & Test session."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    if request.method == 'GET':
        return jsonify({'success': True, 'sessions': store.list_sessions(user_id)})
    try:
        body = request.get_json(silent=True) or {}
        session = store.save_session(user_id, body)
        return jsonify({'success': True, 'session': session}), 201
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def _studio_session_for_read(session_id):
    """(owner_user_id, session, is_owner) for a READ of one Studio session.

    The caller's own session wins. Otherwise -- no key, or a key that isn't
    the owner's -- a session the owner has flagged public is served read-only,
    which is how a manuscript's appendix links open for any reader. A private
    session and an unknown id both come back as (None, None, False) so the
    response can't be used to probe which ids exist.
    """
    from WebUI import handbook_studio_store as store
    identity_key = _studio_identity_key()
    if identity_key:
        user_id = hash_api_key(identity_key)
        try:
            session = store.get_session(user_id, session_id)
        except ValueError:
            session = None
        if session is not None:
            return user_id, session, True
    found = store.find_public_session(session_id)
    if found:
        owner_id, session = found
        return owner_id, session, False
    return None, None, False


@app.route('/handbook-studio/shared/<session_id>')
def serve_shared_studio_session(session_id):
    """Read-only deep link to a public Studio session. The client router
    (view_router.js) opens the Studio in shared mode; no key is required."""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/handbook-studio-sessions/<session_id>/share', methods=['POST'])
def handbook_studio_share_endpoint(session_id):
    """Owner-only: make a session public (read-only link) or private again."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    body = request.get_json(silent=True) or {}
    public = bool(body.get('public', not session.get('public')))
    updated = store.save_session(user_id, {**session, 'public': public},
                                 session_id=session_id)
    return jsonify({
        'success': True,
        'public': updated.get('public', False),
        'public_since': updated.get('public_since'),
        'url': f"{request.host_url.rstrip('/')}/handbook-studio/shared/{session_id}",
        'session': updated,
    })


@app.route('/api/handbook-studio-sessions/<session_id>',
           methods=['GET', 'PUT', 'DELETE'])
def handbook_studio_session_endpoint(session_id):
    """Read, update, or delete one caller-owned Generate & Test session.
    A public session can be READ by anyone (see _studio_session_for_read)."""
    if request.method == 'GET':
        _owner, session, is_owner = _studio_session_for_read(session_id)
        if session is None:
            if not _studio_identity_key():
                return jsonify({'success': False, 'error': 'No API key provided'}), 401
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        return jsonify({'success': True, 'session': session,
                        'is_owner': is_owner, 'shared_view': not is_owner})
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    try:
        session = store.get_session(user_id, session_id)
        if session is None:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        if request.method == 'DELETE':
            store.delete_session(user_id, session_id)
            return jsonify({'success': True})
        updated = store.save_session(
            user_id, request.get_json(silent=True) or {}, session_id=session_id)
        return jsonify({'success': True, 'session': updated})
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/handbook-studio-sessions/<session_id>/generate-stream',
           methods=['POST'])
def handbook_studio_generate_stream(session_id):
    """Generate one handbook (OpenAI or Claude, per the session's provider)
    and pause for human review, over SSE."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_generate_stream
        try:
            for event in run_generate_stream(
                    user_id, api_key_value, session_id,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Generate & Test handbook generation failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'generate:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-stream',
           methods=['POST'])
def handbook_studio_test_stream(session_id):
    """Run one controlled retrieval trial against the reviewed handbook, over SSE."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    edited_task = body.get('retrieval_task')
    edited_source = body.get('generated_source') or {}
    data_source_keys = body.get('data_source_keys') or {}
    if not isinstance(edited_source, dict):
        return jsonify({
            'success': False, 'error': 'generated_source must be an object.'
        }), 400
    if not isinstance(data_source_keys, dict):
        return jsonify({
            'success': False, 'error': 'data_source_keys must be an object.'
        }), 400
    user_id = hash_api_key(identity_key)
    # Which arm this run belongs to. The SESSION is the authority: it is what
    # the author set, what the UI shows, and what is persisted. This used to
    # default a missing body field to True, which silently overrode a control
    # session -- the runner only consults the session when it receives None, so
    # a caller that forgot the field turned every control run into a treatment
    # run while the UI still displayed "no handbook". Fifteen runs were lost
    # that way. None means "not specified, use the session"; the body can still
    # override deliberately.
    use_handbook = (None if body.get('use_handbook') is None
                    else body.get('use_handbook') is not False)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_test_stream
        try:
            for event in run_test_stream(
                    user_id, api_key_value, session_id,
                    edited_task=edited_task, edited_source=edited_source,
                    user_keys=data_source_keys,
                    anthropic_api_key=anthropic_key_value,
                    use_handbook=use_handbook):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Generate & Test retrieval test failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'test:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-refine-stream',
           methods=['POST'])
def handbook_studio_test_refine_stream(session_id):
    """Run a controlled retrieval trial, automatically revising the handbook
    (H1, H2, H3) and retrying on evidence-grounded failures, over SSE."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    edited_task = body.get('retrieval_task')
    edited_source = body.get('generated_source') or {}
    data_source_keys = body.get('data_source_keys') or {}
    max_refinements = body.get('max_refinements', 3)
    if not isinstance(edited_source, dict):
        return jsonify({
            'success': False, 'error': 'generated_source must be an object.'
        }), 400
    if not isinstance(data_source_keys, dict):
        return jsonify({
            'success': False, 'error': 'data_source_keys must be an object.'
        }), 400
    try:
        max_refinements = max(0, min(10, int(max_refinements)))
    except (TypeError, ValueError):
        max_refinements = 3
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_test_with_refinement_stream
        try:
            for event in run_test_with_refinement_stream(
                    user_id, api_key_value, session_id,
                    edited_task=edited_task, edited_source=edited_source,
                    user_keys=data_source_keys, max_refinements=max_refinements,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Generate & Test retrieval test (with refinement) failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'test-refine:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/validate-stream',
           methods=['POST'])
def handbook_studio_validate_stream(session_id, run_id):
    """Judge one completed test run's correctness on demand, over SSE --
    separate from and not triggered by Review & Test. method is one of
    "mechanical" (deterministic checks against the session's pre-registered
    validation spec, no LLM; "metadata" is accepted as its former name),
    "manual" (a per-dimension human adjudication rubric), or "llm" (the
    evidence-grounded LLM judge, kept as a comparison arm). Each call
    APPENDS to the run's validation log; nothing is overwritten."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    method = body.get('method')
    manual_verdict = body.get('manual_verdict') or {}
    if not isinstance(manual_verdict, dict):
        return jsonify({
            'success': False, 'error': 'manual_verdict must be an object.'
        }), 400
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_validation_stream
        try:
            for event in run_validation_stream(
                    user_id, api_key_value, session_id, run_id, method,
                    manual_verdict=manual_verdict,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Handbook Studio validation failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'validate:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/refine-from-validation-stream',
           methods=['POST'])
def handbook_studio_refine_from_validation_stream(session_id, run_id):
    """Send a Validation-judged-incomplete test run back for a single
    handbook revision, over SSE -- triggered deliberately from the
    Validation tab. Does not auto-retest afterward."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_validation_refine_stream
        try:
            for event in run_validation_refine_stream(
                    user_id, api_key_value, session_id, run_id,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Handbook Studio refine-from-validation failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'validate-refine:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/cancel', methods=['POST'])
def handbook_studio_cancel(session_id):
    """Request cooperative cancellation of a running Generate & Test session."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    from WebUI.handbook_studio_runner import request_cancel
    user_id = hash_api_key(identity_key)
    try:
        if store.get_session(user_id, session_id) is None:
            return jsonify({
                'success': False, 'error': 'Session not found'
            }), 404
        request_cancel(user_id, session_id)
        return jsonify({
            'success': True,
            'message': 'Cancellation requested. The active step will stop at '
                       'the next safe checkpoint.',
        })
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


def _studio_preview_geometry(path, extension, max_features=5000):
    """Vector artifact -> GeoJSON the browser can draw without its own GDAL.

    Returns None when the file cannot be read as vector data, so the caller
    can fall back rather than reporting a broken map layer.
    """
    try:
        if extension == '.geojson':
            if os.path.getsize(path) > 40 * 1024 * 1024:
                return {'kind': 'binary'}
            with open(path, encoding='utf-8', errors='replace') as handle:
                data = json.load(handle)
            features = data.get('features') if isinstance(data, dict) else None
            truncated = False
            if isinstance(features, list) and len(features) > max_features:
                data = dict(data, features=features[:max_features])
                truncated = True
            return {'kind': 'geojson', 'data': data,
                    'feature_count': len(features) if isinstance(features, list) else None,
                    'truncated': truncated}
        import geopandas as gpd
        frame = gpd.read_file(path)
        if frame.crs is not None and str(frame.crs) != 'EPSG:4326':
            frame = frame.to_crs(epsg=4326)
        total = len(frame)
        truncated = total > max_features
        if truncated:
            frame = frame.iloc[:max_features]
        return {'kind': 'geojson', 'data': json.loads(frame.to_json()),
                'feature_count': total, 'truncated': truncated}
    except Exception:
        return None


def _sidecar_crs(path):
    """The CRS from an ESRI .prj sitting beside a raster that carries none.

    A world file plus a .prj is a normal way to georeference a plain TIFF --
    it is exactly what a PIL-written mosaic gets -- and GDAL applies the .tfw
    but not the .prj. rasterio then reports a correct transform with no CRS,
    and everything downstream treats the file as unplaceable even though the
    projection is sitting in a text file next to it.
    """
    try:
        from rasterio.crs import CRS
    except Exception:
        return None
    base = os.path.splitext(path)[0]
    for candidate in (base + '.prj', base + '.PRJ'):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, encoding='utf-8', errors='replace') as handle:
                wkt = handle.read().strip()
            if wkt:
                return CRS.from_wkt(wkt)
        except Exception:
            return None
    return None


def _studio_render_raster_png(path, max_dim=1024):
    """Render a raster to a WGS84-aligned RGBA PNG for map display.

    Three things have to happen here or satellite imagery is unreadable, which
    is why this is done server-side rather than in the browser:

    * **Reprojection.** Sentinel-2 and Landsat ship in UTM. Mapbox image
      sources take four lon/lat corners, so the pixels must be warped to
      EPSG:4326 first; pasting UTM pixels onto a lon/lat rectangle puts them in
      the wrong place, and the browser-side loader in map.html simply refuses
      the file instead.
    * **A percentile stretch.** Reflectance rasters are uint16 with a long
      bright tail, so a min/max stretch maps almost everything to near-black.
      The 2nd-98th percentile of the valid pixels is what makes the scene
      visible; this is a display stretch only and is never fed back into any
      analysis.
    * **Downsampling before reading.** A full Sentinel tile is ~10980 px square
      per band; decoding that to send to a browser preview would cost hundreds
      of megabytes for an image displayed a few hundred pixels wide.

    Returns (png_bytes, info) or (None, None) when the file cannot be rendered.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import ColorInterp, Resampling
        from rasterio.vrt import WarpedVRT
        from PIL import Image
        import io

        from affine import Affine
        from rasterio.warp import calculate_default_transform

        with rasterio.open(path) as src:
            # No CRS means nothing can be placed on a map; the footprint
            # summary still reports the raw bounds.
            source_crs = src.crs or _sidecar_crs(path)
            if not source_crs:
                return None, {'reason': 'This raster has no CRS, and no .prj '
                                        'sits beside it, so it cannot be '
                                        'placed on a map.'}
            # Overviews are what make a big raster cheap to preview: GDAL reads
            # a small pre-built pyramid level instead of the full grid. A COG
            # (Sentinel-2, most STAC assets) has them, and a 10980x10980 tile
            # renders in well under a second. A plain tiled TIFF may not, and
            # then every pixel must be decompressed to produce any view at all
            # -- a 96000x96000 LZW DEM measured at 258s, which no browser
            # request survives. Refuse those with a reason rather than hanging.
            pixels = src.width * src.height
            if pixels > 50_000_000 and not src.overviews(1):
                return None, {'reason': (
                    f'This raster is {src.width}x{src.height} with no overview '
                    'pyramid, so a preview would require decompressing every '
                    'pixel. Build overviews (gdaladdo, or save as a COG) to '
                    'preview it.')}
            # Size the warp DOWN FRONT rather than warping at native
            # resolution and shrinking afterwards. A default WarpedVRT on a
            # 96000x96000 continental tile reprojects every pixel before the
            # downsample ever runs -- measured at 258s, which no browser
            # request survives. Scaling the transform first makes the warp
            # itself produce only the pixels that will be displayed.
            transform, vw, vh = calculate_default_transform(
                source_crs, 'EPSG:4326', src.width, src.height, *src.bounds)
            scale = max(vw, vh) / float(max_dim)
            if scale > 1:
                transform = transform * Affine.scale(scale, scale)
                vw = max(1, int(vw / scale))
                vh = max(1, int(vh / scale))
            with WarpedVRT(src, src_crs=source_crs, crs='EPSG:4326', transform=transform,
                           width=vw, height=vh,
                           resampling=Resampling.bilinear) as vrt:
                out_w, out_h = vrt.width, vrt.height

                # Prefer declared colour interpretation; fall back to the first
                # three bands, which is the convention for a true-colour
                # composite. A single band is rendered as grayscale.
                interp = list(vrt.colorinterp)
                rgb = [interp.index(c) + 1 for c in
                       (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                       if c in interp]
                if len(rgb) != 3:
                    rgb = [1, 2, 3][:vrt.count]
                indexes = rgb if len(rgb) == 3 else [1]

                data = vrt.read(indexes, out_shape=(len(indexes), out_h, out_w),
                                resampling=Resampling.bilinear,
                                masked=True).astype('float32')
                # The warp fills outside the source footprint; that fill and
                # any nodata become transparent rather than black bars.
                mask = vrt.read_masks(indexes[0], out_shape=(out_h, out_w),
                                      resampling=Resampling.nearest)

                bands = []
                for i in range(data.shape[0]):
                    band = data[i]
                    valid = band.compressed() if hasattr(band, 'compressed') else band.ravel()
                    valid = valid[np.isfinite(valid)]
                    if valid.size == 0:
                        bands.append(np.zeros((out_h, out_w), dtype='uint8'))
                        continue
                    lo, hi = np.percentile(valid, [2, 98])
                    if hi <= lo:
                        lo, hi = float(valid.min()), float(valid.max())
                    if hi <= lo:
                        hi = lo + 1.0
                    stretched = np.clip((np.asarray(band, dtype='float32') - lo)
                                        / (hi - lo), 0, 1) * 255.0
                    bands.append(stretched.astype('uint8'))

                if len(bands) == 1:
                    bands = bands * 3
                alpha = np.where(mask > 0, 255, 0).astype('uint8')
                # A Sentinel/Landsat scene is a square grid holding a diagonal
                # swath, and the corners outside the swath are plain zeros with
                # no nodata value declared -- so the mask calls them valid and
                # they render as an opaque black wedge sitting over the
                # basemap. For a multi-band composite, all-bands-zero means no
                # data by near-universal convention, so drop those to
                # transparent. NOT applied to single-band rasters, where zero
                # is usually a real measurement (sea level in a DEM, zero
                # population in a density grid).
                if data.shape[0] >= 3 and src.nodata is None:
                    blank = np.all(np.asarray(data) == 0, axis=0)
                    alpha[blank] = 0
                rgba = np.dstack(bands[:3] + [alpha])
                buffer = io.BytesIO()
                Image.fromarray(rgba, mode='RGBA').save(buffer, format='PNG',
                                                        optimize=True)
                west, south, east, north = vrt.bounds
                return buffer.getvalue(), {
                    'bbox': {'west': west, 'south': south,
                             'east': east, 'north': north},
                    'width': out_w, 'height': out_h,
                    'bands_used': indexes,
                    'source_size': [src.width, src.height],
                }
    except Exception:
        app.logger.exception('Raster render failed for %s', path)
        return None, {'reason': 'This raster could not be decoded.'}


def _studio_preview_raster(path):
    """Raster artifact -> footprint and band summary (no pixels shipped)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(path) as src:
            source_crs = src.crs or _sidecar_crs(path)
            summary = {
                'kind': 'raster', 'width': src.width, 'height': src.height,
                'bands': src.count, 'dtype': str(src.dtypes[0]),
                'crs': str(source_crs) if source_crs else None,
                'bbox': None,
            }
            if not source_crs:
                # The old code returned the raw bounds here as though they were
                # lon/lat. For a projected raster that is a bbox in the
                # millions of degrees: the map silently drew nothing, which
                # read as "the file is empty" rather than "we don't know where
                # this goes".
                summary['reason'] = (
                    'This raster carries no CRS and has no .prj beside it, so '
                    'it cannot be placed on a map. Its data is fine — only its '
                    'location is unknown.')
                return summary
            west, south, east, north = transform_bounds(
                source_crs, 'EPSG:4326', *src.bounds)
            summary['bbox'] = {'west': west, 'south': south,
                               'east': east, 'north': north}
            return summary
    except Exception:
        return None


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/rerun-code',
           methods=['POST'])
def handbook_studio_rerun_code(session_id, run_id):
    """Re-execute one test run's stored retrieval code to regenerate its files.

    Artifacts are written to a temp directory that the OS eventually sweeps, so
    a run's data can vanish long before anyone reviews it. The code survives in
    the session record, and re-running it puts the files back.

    This re-runs the SAME code; it does not re-generate code and does not touch
    the handbook or the recorded verdict. What comes back is today's data from
    the source, which is the right evidence for judging where a query pointed
    and which parameters it used, but NOT for judging completeness of the
    original run -- a near-real-time source has moved on since. The response is
    labelled `rerun: True` so callers never conflate the two.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    from utils.python_exe import python_executable
    from WebUI.handbook_studio_runner import _session_output_root
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404
    # A round can hold several attempts (the debugger retrying after an error).
    # Re-running a specific one is how you check whether an earlier failure
    # still reproduces; omitting the index runs the round's final code.
    result = run.get('result') or {}
    body = request.get_json(silent=True) or {}
    attempts = ((result.get('execution') or {}).get('attempts') or [])
    attempt_index = body.get('attempt')
    label = 'final'
    if attempt_index is not None:
        try:
            attempt_index = int(attempt_index)
            chosen = attempts[attempt_index]
        except (TypeError, ValueError, IndexError):
            return jsonify({'success': False,
                            'error': f'No attempt {attempt_index} on this run'}), 400
        code = chosen.get('code') or ''
        label = f"attempt {chosen.get('attempt', attempt_index + 1)}"
    else:
        code = result.get('generated_code') or ''
    if not code.strip():
        return jsonify({'success': False,
                        'error': f'This run has no stored code for {label}'}), 400

    run_dir = _inspect_run_dir(user_id, session_id, run)
    if run_dir is None:
        # Original directory is gone (or the run recorded no artifact_ref);
        # write into a fresh, clearly-named directory under the same root.
        run_dir = os.path.join(_session_output_root(user_id, session_id),
                               f'rerun_{run_id[:8]}')
    if attempt_index is not None:
        # Keep each attempt's output apart, so comparing attempt 1 against
        # attempt 3 is a matter of looking at two folders, not one clobbered.
        run_dir = os.path.join(run_dir, f'attempt_{attempt_index + 1}')
    os.makedirs(run_dir, exist_ok=True)

    rewritten, rewrites = _studio_rewrite_output_paths(code, run_dir)
    user_script = os.path.join(run_dir, '_rerun_user.py')
    with open(user_script, 'w', encoding='utf-8') as handle:
        handle.write(rewritten)
    boot = os.path.join(run_dir, '_rerun_boot.py')
    with open(boot, 'w', encoding='utf-8') as handle:
        handle.write(_STUDIO_RERUN_SHIM
                     + f"\nimport runpy\nrunpy.run_path({user_script!r}, run_name='__main__')\n")

    http_log = os.path.join(run_dir, '_rerun_http.jsonl')
    open(http_log, 'w').close()
    env = dict(os.environ)
    env['RERUN_HTTP_LOG'] = http_log
    env['PYTHONUNBUFFERED'] = '1'

    timeout = int(session.get('execution_timeout_seconds') or 600)
    started = time.time()
    try:
        proc = subprocess.run([python_executable(), boot], cwd=run_dir, env=env,
                              capture_output=True, text=True, timeout=timeout)
        returncode, stdout, stderr, timed_out = (
            proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = -1, True
        stdout = exc.stdout.decode('utf-8', 'replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = exc.stderr.decode('utf-8', 'replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')

    # Collect anything written since the run started, across the WHOLE session
    # root rather than just run_dir. Generated code composes its own output
    # paths, and a hardcoded absolute directory that only partly rewrites can
    # land the files beside the run folder instead of inside it. Scanning by
    # modification time finds them wherever they went, so a successful run
    # never reports "0 files" while its output sits one directory over.
    internal = {'_rerun_user.py', '_rerun_boot.py', '_rerun_http.jsonl'}
    session_root = _session_output_root(user_id, session_id)
    cutoff = started - 2  # small allowance for filesystem timestamp skew
    files = []
    for base, dirs, names in os.walk(session_root):
        dirs[:] = [d for d in dirs if d != '_handbook']
        for name in sorted(names):
            if name in internal:
                continue
            full = os.path.join(base, name)
            try:
                if os.path.getmtime(full) < cutoff:
                    continue
                size = os.path.getsize(full)
            except OSError:
                continue
            rel = os.path.relpath(full, session_root)
            files.append({
                'name': name,
                'artifact_ref': rel.replace(os.sep, '/'),
                'size_bytes': size,
                'extension': os.path.splitext(name)[1].lower(),
            })

    requests_seen = []
    try:
        with open(http_log, encoding='utf-8') as handle:
            for line in handle:
                try:
                    requests_seen.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass

    from datetime import datetime as _dt, timezone as _tz
    run_record = {
        'at': _dt.now(_tz.utc).isoformat(),
        'returncode': returncode, 'timed_out': timed_out,
        'elapsed_seconds': round(time.time() - started, 1),
        'stdout': stdout[-8000:], 'stderr': stderr[-8000:],
        'files': files, 'http_requests': requests_seen,
    }
    # Stored against the code that produced it. Held only in the browser
    # before, so a refresh discarded the evidence while its files stayed on
    # disk with nothing pointing at them.
    target = chosen if attempt_index is not None else result
    if attempt_index is None:
        run['result'] = result
    field = 'code' if attempt_index is not None else 'generated_code'
    is_final = attempt_index is None or attempt_index == len(attempts) - 1
    _record_version_run(target, field, run_record, run=run, is_final=is_final)
    store.save_session(user_id, session, session_id)

    return jsonify({
        'success': True, 'rerun': True, 'ran': label,
        'attempt': attempt_index,
        'returncode': returncode, 'timed_out': timed_out,
        'elapsed_seconds': run_record['elapsed_seconds'],
        'stdout': run_record['stdout'], 'stderr': run_record['stderr'],
        'files': files, 'path_rewrites': rewrites,
        'http_requests': requests_seen,
        'current_version': target.get('current_version'),
    })


def _version_label(origin):
    return {
        'agent': 'Agent original',
        'edit': 'Hand edit',
        'feedback': 'Revised from feedback',
    }.get(origin, 'Version')


def _ensure_code_versions(target, field, run=None, is_final=False):
    """Give a code slot a version list, backfilling one for older records.

    Records written before versioning kept only two snapshots: `original_code`
    (what the agent wrote, filed on first change) and the current code. Those
    become v1 and v2 here, deterministically -- the browser derives the same
    two for display before anything has been saved, so the ids it offers match
    the ones this produces.

    Nothing is invented: intermediate states that were never stored cannot be
    recovered, and the version list says only what the record actually holds.
    """
    versions = target.get('code_versions')
    if isinstance(versions, list) and versions:
        return versions

    current = str(target.get(field) or '')
    original = str(target.get('original_code') or '')
    history = target.get('code_feedback') or []
    last_feedback = history[-1] if isinstance(history, list) and history else {}

    versions = []
    if original and original != current:
        versions.append({
            'id': 'v1', 'n': 1, 'origin': 'agent',
            'label': _version_label('agent'), 'code': original,
            'at': (run or {}).get('finished_at') or (run or {}).get('started_at'),
            'feedback': None, 'explanation': None,
            'lines_added': None, 'lines_removed': None, 'run': None,
        })
        origin = 'feedback' if target.get('code_revised_from_feedback') else 'edit'
        versions.append({
            'id': 'v2', 'n': 2, 'origin': origin,
            'label': _version_label(origin), 'code': current,
            'at': target.get('code_revised_at') or target.get('code_edited_at'),
            'feedback': last_feedback.get('feedback'),
            'explanation': last_feedback.get('explanation'),
            'lines_added': last_feedback.get('lines_added'),
            'lines_removed': last_feedback.get('lines_removed'),
            'run': None,
        })
    else:
        versions.append({
            'id': 'v1', 'n': 1, 'origin': 'agent',
            'label': _version_label('agent'), 'code': current,
            'at': (run or {}).get('finished_at') or (run or {}).get('started_at'),
            'feedback': None, 'explanation': None,
            'lines_added': None, 'lines_removed': None, 'run': None,
        })

    # The files the original test run recorded belong to the code that produced
    # them -- the first version -- but only when this slot IS the code that ran
    # last. Attaching them to an earlier attempt would credit one attempt with
    # another's output.
    if is_final and run:
        result = run.get('result') or {}
        recorded = result.get('downloaded_files') or []
        if recorded:
            # Noted as a count, NOT as a run record. Copying the file list here
            # made the round's own outputs render twice at once -- inside the
            # card as this version's result, and again in the round's "Data
            # outputs" section -- which read as two different sets of files.
            versions[0]['original_output_files'] = len(recorded)

    target['code_versions'] = versions
    target['current_version'] = versions[-1]['id']
    return versions


def _append_code_version(target, field, code, origin, run=None, is_final=False,
                         **meta):
    """Record a new code version and make it the current one."""
    from datetime import datetime as _dt, timezone as _tz
    versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
    # An unchanged body is not a new version; it would clutter the picker with
    # entries a reviewer cannot tell apart.
    if versions and (versions[-1].get('code') or '') == code:
        target['current_version'] = versions[-1]['id']
        return versions[-1]
    entry = {
        'id': f'v{len(versions) + 1}',
        'n': len(versions) + 1,
        'origin': origin,
        'label': _version_label(origin),
        'code': code,
        'at': _dt.now(_tz.utc).isoformat(),
        'feedback': meta.get('feedback'),
        'explanation': meta.get('explanation'),
        'lines_added': meta.get('lines_added'),
        'lines_removed': meta.get('lines_removed'),
        'run': None,
    }
    versions.append(entry)
    target['code_versions'] = versions
    target['current_version'] = entry['id']
    return entry


def _record_version_run(target, field, run_record, run=None, is_final=False):
    """Attach a re-run's outcome to whichever version was executed.

    Re-run output used to live only in the browser's memory, so a refresh threw
    away the evidence a reviewer had just generated while the files sat on disk
    unreferenced. Stored on the version, it survives the reload and stays tied
    to the exact code that produced it.
    """
    versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
    current_id = target.get('current_version') or versions[-1]['id']
    for entry in versions:
        if entry.get('id') == current_id:
            entry['run'] = run_record
            return entry
    versions[-1]['run'] = run_record
    return versions[-1]


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/attempt-code',
           methods=['POST'])
def handbook_studio_edit_code(session_id, run_id):
    """Replace one test run's stored retrieval code with a hand-edited version.

    The re-run endpoint above executes whatever code the record holds, so
    editing here and re-running there is how a reviewer fixes a dead endpoint,
    a wrong bbox or a missing key by hand instead of asking the model for
    another round.

    What the model originally wrote is never lost: the first edit copies it to
    `original_code` and flags the attempt `code_edited`, so the experiment can
    still separate agent-authored code from human-repaired code (and `revert`
    puts it back). Recorded verdicts, files and usage are left alone -- they
    describe the ORIGINAL run, and rewriting them would make an edited attempt
    look like the agent had produced it.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from datetime import datetime as _dt, timezone as _tz
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    result = run.get('result') or {}
    attempts = ((result.get('execution') or {}).get('attempts') or [])
    attempt_index = body.get('attempt')
    revert = bool(body.get('revert'))
    # Where the code lives depends on the shape of the record: a round with a
    # debugger loop stores one entry per attempt, an older single-shot round
    # stores only `generated_code`.
    if attempt_index is not None:
        try:
            attempt_index = int(attempt_index)
            target = attempts[attempt_index]
        except (TypeError, ValueError, IndexError):
            return jsonify({'success': False,
                            'error': f'No attempt {attempt_index} on this run'}), 400
    else:
        # A record whose `result` slot was missing entirely gets one, so the
        # edit has somewhere to live rather than being written to a throwaway.
        run['result'] = result
        target = result

    if revert:
        original = target.get('original_code')
        if not original:
            return jsonify({'success': False,
                            'error': 'This code has not been edited.'}), 400
        code = original
    else:
        code = body.get('code')
        if not isinstance(code, str) or not code.strip():
            return jsonify({'success': False, 'error': 'Code cannot be empty.'}), 400
        if len(code) > 400_000:
            return jsonify({'success': False,
                            'error': 'Code is too long to store (400 KB limit).'}), 400
        code = code.replace('\r\n', '\n')

    field = 'code' if attempt_index is not None else 'generated_code'
    is_final = attempt_index is None or attempt_index == len(attempts) - 1
    # BEFORE the slot is touched. Backfilling afterwards would describe the new
    # code with the old version's label and then treat the append as a no-op,
    # losing whatever was being replaced.
    _ensure_code_versions(target, field, run=run, is_final=is_final)
    if revert:
        # Revert moves back to the agent's own version rather than filing a
        # copy of it: the picker should show the history that happened, not a
        # new entry every time someone looks at the original.
        versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
        target[field] = code
        for entry in versions:
            if (entry.get('code') or '') == code:
                target['current_version'] = entry['id']
                break
        target.pop('original_code', None)
        target['code_edited'] = False
        target['code_edited_at'] = None
    else:
        # Keyed on original_code, not code_edited: a model revision from
        # reviewer feedback also files the agent's version here, and keying on
        # the hand-edit flag would overwrite it with that revision -- leaving
        # Revert pointing at a revision rather than at what the agent wrote.
        if not target.get('original_code'):
            target['original_code'] = target.get(field) or ''
        target[field] = code
        target['code_edited'] = True
        target['code_edited_at'] = _dt.now(_tz.utc).isoformat()
        _append_code_version(target, field, code, 'edit',
                             run=run, is_final=is_final)

    store.save_session(user_id, session, session_id)
    return jsonify({
        'success': True,
        'attempt': attempt_index,
        'code': code,
        'code_edited': bool(target.get('code_edited')),
        'original_code': target.get('original_code') or '',
        'code_versions': target.get('code_versions') or [],
        'current_version': target.get('current_version'),
    })




def _studio_revision_slot(session, run, body):
    """Locate the code slot a revision targets.

    Returns ``(target, field, attempt_index, error)`` where ``error`` is
    ``(message, status)`` or None. Mirrors the hand-edit endpoint's record
    shapes: a debugger loop stores one entry per attempt, an older single-shot
    round only ``generated_code``.
    """
    result = run.get('result') or {}
    attempts = ((result.get('execution') or {}).get('attempts') or [])
    attempt_index = body.get('attempt')
    if attempt_index is not None:
        try:
            attempt_index = int(attempt_index)
            target = attempts[attempt_index]
        except (TypeError, ValueError, IndexError):
            return None, None, None, (f'No attempt {attempt_index} on this run', 400)
        return target, 'code', attempt_index, None
    run['result'] = result
    return result, 'generated_code', None, None


def _studio_store_revision(user_id, session, session_id, target, field,
                           current_code, feedback, revision, provider):
    """Persist one feedback-driven revision. Returns ``(payload, error)``.

    What the agent originally wrote is preserved in ``original_code`` (so
    Revert still restores it), but the attempt is flagged
    ``code_revised_from_feedback`` rather than ``code_edited``: for the
    experiment, code a model rewrote from an instruction is not the same
    evidence as code a human typed.
    """
    from datetime import datetime as _dt, timezone as _tz
    from WebUI import handbook_studio_store as store

    revised = (revision.get('code') or '').replace('\r\n', '\n').strip()
    if not revised:
        # A reply with no code block means the model answered in prose. Storing
        # nothing and reporting success would look like a silent no-op, so the
        # reason is handed back instead.
        return None, ('The model replied without any code. It said: '
                      + ((revision.get('raw') or '').strip()[:400] or '(nothing)'), 422)
    if len(revised) > 400_000:
        return None, ('Revised code is too long to store (400 KB limit).', 400)
    if revised == current_code.strip():
        return None, ('The model returned the same code unchanged. '
                      'Try saying more specifically what to change.', 422)

    # How big the change actually was. A feedback revision is supposed to touch
    # only what was asked for; showing the size on the card is what lets a
    # reviewer notice a model that reformatted the whole file instead.
    import difflib
    diff = list(difflib.unified_diff(current_code.splitlines(),
                                     revised.splitlines(), n=0, lineterm=''))
    lines_added = sum(1 for line in diff
                      if line.startswith('+') and not line.startswith('+++'))
    lines_removed = sum(1 for line in diff
                        if line.startswith('-') and not line.startswith('---'))

    now = _dt.now(_tz.utc).isoformat()
    # Snapshot the code being replaced before replacing it, for the same reason
    # the edit endpoint does: a backfill that runs afterwards sees only the new
    # body and the version it was meant to preserve is gone.
    _ensure_code_versions(target, field)
    # The agent's own version is filed once and never overwritten, whether the
    # first change came from a hand edit or from feedback -- Revert has to
    # reach the code the agent actually wrote, not an intermediate revision.
    if not target.get('original_code'):
        target['original_code'] = current_code
    target[field] = revised
    target['code_revised_from_feedback'] = True
    target['code_revised_at'] = now
    history = target.get('code_feedback')
    if not isinstance(history, list):
        history = []
    history.append({
        'feedback': feedback,
        'explanation': revision.get('explanation') or '',
        'lines_added': lines_added,
        'lines_removed': lines_removed,
        'at': now,
        'model': revision.get('model') or '',
        'provider': provider,
    })
    target['code_feedback'] = history
    _append_code_version(target, field, revised, 'feedback',
                         feedback=feedback,
                         explanation=revision.get('explanation') or '',
                         lines_added=lines_added, lines_removed=lines_removed)

    store.save_session(user_id, session, session_id)
    return {
        'code': revised,
        'code_versions': target.get('code_versions') or [],
        'current_version': target.get('current_version'),
        'explanation': revision.get('explanation') or '',
        'original_code': target.get('original_code') or '',
        'code_revised_from_feedback': True,
        'code_feedback': history,
    }, None


def _studio_revision_inputs(session, run, target, field):
    """The context a revision is given besides the feedback itself."""
    result = run.get('result') or {}
    source = session.get('generated_source') or {}
    return {
        'current_code': str(target.get(field) or ''),
        'task_text': str(run.get('retrieval_task')
                         or session.get('retrieval_task') or ''),
        'handbook_text': str(source.get('handbook') or ''),
        'last_error': str(target.get('error') or result.get('error') or ''),
    }


def _studio_revise_agent(session, openai_key, anthropic_key):
    """The retrieval agent configured the way this session's tests ran."""
    from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent
    provider = str(session.get('provider') or 'openai')
    if provider not in ('openai', 'claude'):
        provider = 'openai'
    agent = DataRetrieverAgent(
        api_key=openai_key, evaluation=True,
        model=session.get('data_retrieval_model'),
        reasoning_effort=session.get('data_retrieval_reasoning_effort') or None,
        provider=provider, anthropic_api_key=anthropic_key)
    return agent, provider


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/select-version',
           methods=['POST'])
def handbook_studio_select_version(session_id, run_id):
    """Load one stored version of a run's retrieval code.

    Selecting a version puts its code back in the slot the re-run endpoint
    executes, so "load v1 and run it again" needs no copy-paste. The version's
    own recorded output travels with it, which is what makes comparing two
    versions a matter of clicking between them.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    version_id = str(body.get('version') or '').strip()
    if not version_id:
        return jsonify({'success': False, 'error': 'No version given.'}), 400

    target, field, attempt_index, error = _studio_revision_slot(session, run, body)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]
    attempts = ((run.get('result') or {}).get('execution') or {}).get('attempts') or []
    is_final = attempt_index is None or attempt_index == len(attempts) - 1

    versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
    chosen = next((v for v in versions if v.get('id') == version_id), None)
    if chosen is None:
        return jsonify({'success': False,
                        'error': f'No version {version_id} on this code.'}), 404

    target[field] = chosen.get('code') or ''
    target['current_version'] = chosen['id']
    store.save_session(user_id, session, session_id)
    return jsonify({
        'success': True, 'attempt': attempt_index,
        'version': chosen['id'], 'code': target[field],
        'code_versions': versions, 'current_version': chosen['id'],
    })


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/revise-code',
           methods=['POST'])
def handbook_studio_revise_code(session_id, run_id):
    """Rewrite one test run's retrieval code from a reviewer's written feedback.

    The sibling of the hand-edit endpoint above: instead of typing the fix
    yourself, you say what is wrong ("it downloads the whole state -- filter to
    the county in the retrieval task") and the model returns a revised program,
    which is stored exactly where a hand edit would be stored and executed by
    the same re-run endpoint.

    The UI uses the streaming variant below; this one is the plain
    request/response form, kept for scripted callers.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    feedback = str(body.get('feedback') or '').strip()
    if not feedback:
        return jsonify({'success': False,
                        'error': 'Say what should change before sending.'}), 400
    if len(feedback) > 20_000:
        return jsonify({'success': False,
                        'error': 'Feedback is too long (20 KB limit).'}), 400

    target, field, attempt_index, error = _studio_revision_slot(session, run, body)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]

    inputs = _studio_revision_inputs(session, run, target, field)
    if not inputs['current_code'].strip():
        return jsonify({'success': False,
                        'error': 'This attempt has no code to revise.'}), 400
    attempts = ((run.get('result') or {}).get('execution') or {}).get('attempts') or []
    _ensure_code_versions(target, field, run=run,
                          is_final=(attempt_index is None
                                    or attempt_index == len(attempts) - 1))

    try:
        agent, provider = _studio_revise_agent(session, g.api_key, g.anthropic_api_key)
        revision = agent.revise_code_from_feedback(
            feedback, inputs['current_code'], inputs['task_text'],
            inputs['handbook_text'], last_error=inputs['last_error'])
        revision['model'] = agent.model
    except Exception as exc:
        import traceback
        print(f"[revise-code] Error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__,
        }), 502

    payload, error = _studio_store_revision(
        user_id, session, session_id, target, field,
        inputs['current_code'], feedback, revision, provider)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]
    return jsonify({'success': True, 'attempt': attempt_index, **payload})


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/revise-code-stream',
           methods=['POST'])
def handbook_studio_revise_code_stream(session_id, run_id):
    """The same revision, streamed as the model writes it.

    Rewriting a few hundred lines takes a minute or more. Without a stream the
    card sits on a disabled button for that whole time, which is
    indistinguishable from a dead button -- so the model's output is forwarded
    as it arrives and the stored result is sent as a final event.

    Pre-flight problems (no key, unknown session/run, empty feedback) are
    reported as ordinary JSON errors, since nothing has been streamed yet; once
    the stream opens, everything including failure is an SSE event.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    feedback = str(body.get('feedback') or '').strip()
    if not feedback:
        return jsonify({'success': False,
                        'error': 'Say what should change before sending.'}), 400
    if len(feedback) > 20_000:
        return jsonify({'success': False,
                        'error': 'Feedback is too long (20 KB limit).'}), 400

    target, field, attempt_index, error = _studio_revision_slot(session, run, body)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]

    inputs = _studio_revision_inputs(session, run, target, field)
    if not inputs['current_code'].strip():
        return jsonify({'success': False,
                        'error': 'This attempt has no code to revise.'}), 400
    attempts = ((run.get('result') or {}).get('execution') or {}).get('attempts') or []
    _ensure_code_versions(target, field, run=run,
                          is_final=(attempt_index is None
                                    or attempt_index == len(attempts) - 1))

    # Read out of the request context before the generator starts: by the time
    # it runs, `g` belongs to no request.
    openai_key, anthropic_key = g.api_key, g.anthropic_api_key

    def generate():
        import queue
        import threading

        yield sse_data({'type': 'revise_started', 'attempt': attempt_index,
                        'lines': len(inputs['current_code'].splitlines())})

        chunks = queue.Queue()
        box = {}

        def worker():
            # Storing happens HERE, in the worker, not after the send loop
            # below: if the browser goes away mid-stream, the generator is
            # closed at its next yield and anything after it never runs. A
            # revision the model has already been paid for would then be lost,
            # which is the one outcome a reviewer cannot recover from -- they
            # would see nothing on the card and have no way to know it ran.
            try:
                agent, provider = _studio_revise_agent(
                    session, openai_key, anthropic_key)
                box['provider'] = provider
                revision = agent.revise_code_from_feedback(
                    feedback, inputs['current_code'], inputs['task_text'],
                    inputs['handbook_text'], last_error=inputs['last_error'],
                    stream_callback=lambda text: chunks.put(text))
                revision['model'] = agent.model
                payload, store_error = _studio_store_revision(
                    user_id, session, session_id, target, field,
                    inputs['current_code'], feedback, revision, provider)
                if store_error:
                    box['error'] = store_error[0]
                else:
                    box['payload'] = payload
            except Exception as exc:          # surfaced as an SSE error below
                import traceback
                print(f"[revise-code-stream] Error: {traceback.format_exc()}")
                box['error'] = (f'{type(exc).__name__}: {exc}'
                                if str(exc) else type(exc).__name__)
            finally:
                chunks.put(None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            chunk = chunks.get()
            if chunk is None:
                break
            if chunk:
                yield sse_data({'type': 'revise_delta', 'text': chunk})

        if box.get('error'):
            yield sse_data({'type': 'error', 'error': box['error']})
            return
        yield sse_data({'type': 'revise_done', 'attempt': attempt_index,
                        **(box.get('payload') or {})})

    return sse_response(generate())


# Absolute output directories the generated code baked in -- often from a
# different machine entirely (the Windows runs wrote to %TEMP%\GISHBStudio).
# The deeper "…/<user>/<session>/test_<hex>" tail is OPTIONAL: some generated
# code stores only the GISHBStudio root and joins subdirectories itself, and
# missing those left the path pointing at a drive letter that does not exist on
# this machine, so the re-run wrote nowhere.
# ONE pattern, ONE pass. The Windows and POSIX roots are alternatives of a
# single regex rather than two successive substitutions, because the
# replacement path itself lives under the POSIX root: running a second sub over
# the output made that pattern match the text just written and substitute
# again, yielding ".../attempt_1/<user>/<session>/rerun_x/attempt_1/file".
# re.sub never re-scans what it has already emitted, so a single pass cannot
# double up.
#
# A trailing separator IS consumed, and _studio_rewrite_output_paths puts one
# back — that pairing keeps every shape valid. Consuming without restoring
# glued a filename onto the directory; restoring without consuming left a
# stray Windows backslash mid-path.
_STUDIO_OUT_DIR_RE = re.compile(
    r'(?:'
    r'[A-Za-z]:(?:\\{1,2}|/)(?:[^"\'\n]*?)GISHBStudio'
    r'(?:(?:\\{1,2}|/)[^"\'\n]*?(?:test_[0-9a-f]+|_handbook))?'
    r'|'
    r'/tmp/gis-co-scientist-hbstudio(?:/[^"\'\n]*?test_[0-9a-f]+)?'
    r'|'
    # The durable root (handbook_studio_runner._default_output_root) on any
    # OS. Code generated on Windows and re-run on Linux must be caught here:
    # left alone, the backslashes are legal filename characters there, and the
    # whole Windows path becomes one oddly named directory the UI cannot open.
    r'(?:[A-Za-z]:(?:\\{1,2}|/)|/)(?:[^"\'\n]*?)gis-co-scientist(?:\\{1,2}|/)outputs'
    r'(?:(?:\\{1,2}|/)[^"\'\n]*?(?:test_[0-9a-f]+|rerun_[0-9a-f]+|_handbook))?'
    r')'
    r'(?:\\{1,2}|/)?'
)

# Logs each request with its FINAL url (parameters merged) and POST bodies --
# what the original capture missed, which is why query parameters could not be
# checked mechanically for most runs.
_STUDIO_RERUN_SHIM = '''
import json as _json, os as _os, time as _time
_LOG = _os.environ.get("RERUN_HTTP_LOG")
def _install():
    try:
        import requests
    except Exception:
        return
    _orig = requests.sessions.Session.request
    def _wrapped(self, method, url, *a, **kw):
        rec = {"method": method, "url": url, "t": _time.time()}
        try:
            resp = _orig(self, method, url, *a, **kw)
            rec["url"] = resp.url
            rec["status"] = resp.status_code
            return resp
        except Exception as exc:
            rec["error"] = type(exc).__name__
            raise
        finally:
            body = kw.get("data") or kw.get("json")
            if body is not None:
                rec["body"] = str(body)[:4000]
            if _LOG:
                try:
                    with open(_LOG, "a", encoding="utf-8") as fh:
                        fh.write(_json.dumps(rec) + "\\n")
                except Exception:
                    pass
    requests.sessions.Session.request = _wrapped
_install()
'''


def _studio_rewrite_output_paths(code, out_dir):
    """Point every hardcoded artifact directory at out_dir.

    Whatever followed the directory in the original string has to keep working.
    Generated code stores these three ways, and all of them must survive:

        out_dir  = r"...\\GISHBStudio\\<user>\\<session>\\test_ab12"
        out_file = r"...\\GISHBStudio\\<user>\\<session>\\test_ab12\\data.geojson"
        root     = r"...\\GISHBStudio"            # joined with more parts later

    So when the match consumed a trailing separator, the replacement keeps one.
    Dropping it is what produced paths like ".../attempt_1PADUS_NPS.geojson",
    where the filename was glued straight onto the directory.
    """
    replacement = out_dir.replace('\\', '/')
    count = 0

    def _sub(match):
        nonlocal count
        count += 1
        text = match.group(0)
        return replacement + '/' if text[-1:] in ('\\', '/') else replacement

    return _STUDIO_OUT_DIR_RE.sub(_sub, code), count


# ── Inspect tab: look at the data a test run actually returned ──────────────
# The Validation tab records a verdict; these endpoints are what let a reviewer
# SEE what they are judging -- the retrieved file itself, where it sits on a
# map, and the place the task named. Everything is scoped to the caller's own
# session output root, and paths are resolved before use so a crafted
# artifact_ref cannot escape it.

_INSPECT_MAX_PREVIEW = 5 * 1024 * 1024
_INSPECT_TEXT_EXT = {'.csv', '.tsv', '.json', '.geojson', '.txt', '.xml',
                     '.toml', '.meta', '.jsonl', '.wkt', '.prj', '.cpg'}


def _inspect_run_dir(user_id, session_id, run):
    """Absolute directory holding one test run's artifacts, or None."""
    from WebUI.handbook_studio_runner import _session_output_root
    refs = [f.get('artifact_ref') for f in
            ((run.get('result') or {}).get('downloaded_files') or [])
            if f.get('artifact_ref')]
    if not refs:
        return None
    first = refs[0].replace('\\', '/').split('/')[0]
    root = _session_output_root(user_id, session_id)
    candidate = os.path.abspath(os.path.join(root, first))
    if not candidate.startswith(os.path.abspath(root) + os.sep):
        return None
    return candidate if os.path.isdir(candidate) else None


def _inspect_find_run(session, run_id):
    for run in (session.get('test_runs') or []):
        if run.get('id') == run_id:
            return run
    return None


def _evicted_message(session_roots, artifact_ref, name):
    """Why a recorded output is missing, if the retention policy removed it.

    Distinguishes "evicted to stay within the server's storage quota" from a
    file that was never there, so the UI can tell the user to re-run rather
    than leaving them to wonder whether the run produced anything.
    """
    from WebUI import output_retention
    parts = [p for p in str(artifact_ref or '').replace('\\', '/').split('/') if p]
    run_dirs = []
    for root in session_roots:
        if parts:
            run_dirs.append(os.path.join(root, parts[0]))
        elif os.path.isdir(root):
            run_dirs.extend(os.path.join(root, d) for d in os.listdir(root)
                            if d != '_handbook')
    for run_dir in run_dirs:
        reason = output_retention.evicted_reason(run_dir, os.path.basename(name or ''))
        if reason:
            return reason
    return None


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/artifacts')
def handbook_studio_run_artifacts(session_id, run_id):
    """List the files one test run left on disk, with their recorded evidence.

    A run whose artifacts have been swept from the temp directory returns an
    empty list plus `available: False`, so the UI can say the data is gone
    rather than implying the run produced nothing.
    """
    user_id, session, _is_owner = _studio_session_for_read(session_id)
    if session is None:
        if not _studio_identity_key():
            return jsonify({'success': False, 'error': 'No API key provided'}), 401
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    result = run.get('result') or {}
    evidence = {str(item.get('name')): item
                for item in (result.get('output_evidence') or [])
                if isinstance(item, dict)}
    run_dir = _inspect_run_dir(user_id, session_id, run)
    files = []
    if run_dir:
        for base, _dirs, names in os.walk(run_dir):
            for name in sorted(names):
                path = os.path.join(base, name)
                rel = os.path.relpath(path, run_dir).replace('\\', '/')
                ext = os.path.splitext(name)[1].lower()
                files.append({
                    'name': name,
                    'rel': rel,
                    'size_bytes': os.path.getsize(path),
                    'extension': ext,
                    'previewable': ext in _INSPECT_TEXT_EXT,
                    'evidence': evidence.get(name) or {},
                })
    from WebUI import output_retention
    return jsonify({
        'success': True,
        'available': bool(run_dir),
        'files': files,
        # Files the retention policy removed since the run, so the UI can say
        # "evicted, re-run to regenerate" instead of implying they never were.
        'evicted': output_retention.evicted_files(run_dir) if run_dir else {},
        'retention': run.get('retention') or {},
        'output_evidence': result.get('output_evidence') or [],
        'http_requests': result.get('http_requests') or [],
        'retrieval_task': run.get('retrieval_task') or session.get('retrieval_task') or '',
    })


class _ZipSink:
    """A write-only, non-seekable target for zipfile that we drain as we go.

    zipfile needs only write()/tell()/flush() on a non-seekable stream (it
    then emits data descriptors), so the whole archive never exists on disk
    or in memory -- only the chunk currently being sent.
    """

    def __init__(self):
        self._buf = bytearray()
        self._pos = 0

    def write(self, data):
        self._buf += data
        self._pos += len(data)
        return len(data)

    def tell(self):
        return self._pos

    def flush(self):
        pass

    def drain(self):
        out = bytes(self._buf)
        self._buf.clear()
        return out


def _stream_zip(run_dir, arc_root):
    import zipfile
    sink = _ZipSink()
    with zipfile.ZipFile(sink, 'w', zipfile.ZIP_DEFLATED) as archive:
        for base, _dirs, names in os.walk(run_dir):
            for name in sorted(names):
                if name.startswith('.') and name.endswith('.json'):
                    continue  # runner bookkeeping (.retention.json, .evicted.json)
                path = os.path.join(base, name)
                arcname = os.path.join(arc_root, os.path.relpath(path, run_dir))
                try:
                    with open(path, 'rb') as src, archive.open(arcname, 'w') as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                            yield sink.drain()
                except OSError:
                    continue
                yield sink.drain()
    yield sink.drain()


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/download-all')
def handbook_studio_run_download_all(session_id, run_id):
    """Everything one test run left on disk, as a zip streamed straight to
    the browser. Exists so a user on a quota-limited host can claim a run's
    output in one click before the retention policy evicts its bulk files."""
    user_id, session, _is_owner = _studio_session_for_read(session_id)
    if session is None:
        if not _studio_identity_key():
            return jsonify({'success': False, 'error': 'No API key provided'}), 401
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404
    run_dir = _inspect_run_dir(user_id, session_id, run)
    if not run_dir:
        return jsonify({'success': False,
                        'error': 'Artifacts for this run are no longer on disk'}), 410
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', str(session.get('name') or 'session'))[:40]
    filename = f"{stem}_run{run.get('test_number') or ''}_{run_id[:8]}.zip"
    response = Response(stream_with_context(_stream_zip(run_dir, filename[:-4])),
                        mimetype='application/zip')
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Cache-Control'] = 'private, no-store'
    return response


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>'
           '/artifacts/<path:rel>')
def handbook_studio_run_artifact_file(session_id, run_id, rel):
    """Return one artifact: parsed preview by default, raw bytes with ?raw=1."""
    user_id, session, _is_owner = _studio_session_for_read(session_id)
    if session is None:
        if not _studio_identity_key():
            return jsonify({'success': False, 'error': 'No API key provided'}), 401
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404
    run_dir = _inspect_run_dir(user_id, session_id, run)
    if not run_dir:
        return jsonify({'success': False,
                        'error': 'Artifacts for this run are no longer on disk'}), 410

    path = os.path.abspath(os.path.join(run_dir, rel))
    if not path.startswith(run_dir + os.sep) or not os.path.isfile(path):
        from WebUI import output_retention
        evicted = output_retention.evicted_reason(run_dir, os.path.basename(rel))
        if evicted:
            return jsonify({'success': False, 'evicted': True, 'error': evicted}), 410
        return jsonify({'success': False, 'error': 'File not found'}), 404

    if request.args.get('raw'):
        return send_file(path, as_attachment=True,
                         download_name=os.path.basename(path))

    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    if size > _INSPECT_MAX_PREVIEW:
        return jsonify({'success': True, 'kind': 'too_large',
                        'size_bytes': size, 'name': os.path.basename(path)})

    # GeoJSON goes back as geometry the map can draw; everything else textual
    # goes back as text for the table/code view.
    if ext in ('.geojson', '.json'):
        try:
            with open(path, encoding='utf-8', errors='replace') as handle:
                data = json.load(handle)
        except (ValueError, OSError) as exc:
            return jsonify({'success': False, 'error': f'Unreadable: {exc}'}), 400
        kind = 'geojson' if isinstance(data, dict) and data.get('type') in (
            'FeatureCollection', 'Feature', 'GeometryCollection') else 'json'
        return jsonify({'success': True, 'kind': kind, 'name': os.path.basename(path),
                        'size_bytes': size, 'data': data})

    if ext in _INSPECT_TEXT_EXT:
        with open(path, encoding='utf-8', errors='replace') as handle:
            text = handle.read(_INSPECT_MAX_PREVIEW)
        return jsonify({'success': True,
                        'kind': 'csv' if ext in ('.csv', '.tsv') else 'text',
                        'name': os.path.basename(path), 'size_bytes': size,
                        'text': text})

    # Vector formats GDAL can read (gpkg/shp) are converted to GeoJSON so the
    # map can draw them without the browser needing a GDAL of its own.
    if ext in ('.gpkg', '.shp'):
        try:
            import geopandas as gpd
            frame = gpd.read_file(path)
            if frame.crs is not None and str(frame.crs) != 'EPSG:4326':
                frame = frame.to_crs(epsg=4326)
            if len(frame) > 5000:
                frame = frame.iloc[:5000]
            return jsonify({'success': True, 'kind': 'geojson',
                            'name': os.path.basename(path), 'size_bytes': size,
                            'truncated': len(frame) >= 5000,
                            'data': json.loads(frame.to_json())})
        except Exception as exc:
            return jsonify({'success': False, 'error': f'Could not read: {exc}'}), 400

    if ext in ('.tif', '.tiff'):
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(path) as src:
                west, south, east, north = transform_bounds(
                    src.crs, 'EPSG:4326', *src.bounds) if src.crs else (
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top)
                return jsonify({
                    'success': True, 'kind': 'raster',
                    'name': os.path.basename(path), 'size_bytes': size,
                    'width': src.width, 'height': src.height,
                    'bands': src.count, 'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'bbox': {'west': west, 'south': south,
                             'east': east, 'north': north},
                })
        except Exception as exc:
            return jsonify({'success': False, 'error': f'Could not read: {exc}'}), 400

    return jsonify({'success': True, 'kind': 'binary',
                    'name': os.path.basename(path), 'size_bytes': size})


@app.route('/api/handbook-studio/reference-place')
def handbook_studio_reference_place():
    """Resolve a place name to an outline, for comparing against an extent.

    Backed by the same local-first cache the deterministic spatial check uses,
    so the map and the mechanical verdict always agree about where a place is.
    """
    place = (request.args.get('q') or '').strip()
    if not place:
        return jsonify({'success': False, 'error': 'No place given'}), 400
    from agents.data_agent.validation import reference_geometry
    allow_network = request.args.get('offline') != '1'
    geometry, provenance = reference_geometry(place, allow_network=allow_network)
    if geometry is None:
        return jsonify({'success': True, 'resolved': False,
                        'place': place, 'provenance': provenance})
    minx, miny, maxx, maxy = geometry.bounds
    rings = []
    geoms = ([geometry] if geometry.geom_type == 'Polygon'
             else list(geometry.geoms) if geometry.geom_type == 'MultiPolygon'
             else [])
    span = max(maxx - minx, maxy - miny)
    tolerance = max(span / 400.0, 1e-4)
    for poly in sorted(geoms, key=lambda p: -p.area)[:60]:
        simple = poly.simplify(tolerance, preserve_topology=True)
        ring = simple.exterior if not simple.is_empty else poly.exterior
        coords = [round(v, 5) for xy in ring.coords for v in xy]
        if len(coords) >= 8:
            rings.append(coords)
    return jsonify({
        'success': True, 'resolved': True, 'place': place,
        'provenance': provenance, 'rings': rings,
        'bbox': {'west': minx, 'south': miny, 'east': maxx, 'north': maxy},
    })


@app.route('/api/handbook-studio/basemap')
def handbook_studio_basemap():
    """Coastline rings for the Inspect map.

    No tile server is involved: the map is vector-only and self-contained, so
    it works offline and behind a proxy. Natural Earth 1:110m ships inside
    pyogrio's test fixtures, so this costs no download.
    """
    cached = getattr(app, '_hbs_basemap_cache', None)
    if cached is not None:
        return jsonify({'success': True, 'rings': cached})
    generated = os.path.join(
        BASE_DIR, '..', 'eval', 'retro', 'out', 'basemap.json')
    rings = []
    if os.path.exists(generated):
        try:
            with open(generated) as handle:
                rings = json.load(handle)
        except (ValueError, OSError):
            rings = []
    if not rings:
        try:
            import geopandas as gpd
            import pyogrio
            source = os.path.join(
                os.path.dirname(pyogrio.__file__), 'tests', 'fixtures',
                'naturalearth_lowres', 'naturalearth_lowres.shp')
            frame = gpd.read_file(source)
            frame['geometry'] = frame.geometry.simplify(0.02, preserve_topology=True)
            for geom in frame.geometry:
                if geom is None or geom.is_empty:
                    continue
                polys = ([geom] if geom.geom_type == 'Polygon'
                         else list(geom.geoms))
                for poly in polys:
                    coords = [round(v, 3) for xy in poly.exterior.coords for v in xy]
                    if len(coords) >= 8:
                        rings.append(coords)
        except Exception:
            rings = []
    app._hbs_basemap_cache = rings
    return jsonify({'success': True, 'rings': rings})


@app.route('/api/handbook-studio-sessions/<session_id>/duplicate', methods=['POST'])
def handbook_studio_duplicate(session_id):
    """Clone a session's generated handbook into a new session, skipping
    generation, so the same handbook can be tested against a different
    retrieval task (e.g. a different difficulty tier of the same source)."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    try:
        new_session = store.duplicate_session(user_id, session_id)
        return jsonify({'success': True, 'session': new_session}), 201
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/handbook-studio-sessions/<session_id>/output-preview',
           methods=['GET'])
def handbook_studio_output_preview(session_id):
    """Preview or download one caller-owned Generate & Test output artifact."""
    from WebUI.handbook_studio_runner import _session_output_roots

    user_id, session, _is_owner = _studio_session_for_read(session_id)
    try:
        if session is None:
            if not _studio_identity_key():
                return jsonify({'success': False, 'error': 'No API key provided'}), 401
            return jsonify({
                'success': False, 'error': 'Session not found'
            }), 404

        # Both roots: outputs written before the durable root existed still
        # live under the old /tmp path, and a run recorded months ago should
        # still open its files rather than 404 because a constant changed.
        session_roots = _session_output_roots(user_id, session_id)
        session_root = session_roots[0]
        artifact_ref = str(request.args.get('ref') or '').strip().replace('\\', '/')
        candidate = None

        def containing_root(path):
            try:
                absolute = os.path.abspath(path)
                return next((
                    root for root in session_roots
                    if os.path.commonpath([root, absolute]) == root
                ), None)
            except (OSError, ValueError):
                return None

        if artifact_ref:
            if artifact_ref.startswith('/') or '..' in artifact_ref.split('/'):
                return jsonify({
                    'success': False, 'error': 'Invalid output reference'
                }), 400
            parts = [part for part in artifact_ref.split('/') if part]
            for root in session_roots:
                possible = os.path.abspath(os.path.join(root, *parts))
                if os.path.isfile(possible):
                    candidate = possible
                    session_root = root
                    break
            if candidate is None:
                candidate = os.path.abspath(os.path.join(session_roots[0], *parts))
            if not containing_root(candidate):
                return jsonify({
                    'success': False, 'error': 'Output access denied'
                }), 403
        else:
            requested_name = os.path.basename(
                str(request.args.get('name') or '').strip())
            if not requested_name:
                return jsonify({
                    'success': False, 'error': 'Output reference is required'
                }), 400
            try:
                requested_size = int(request.args.get('size') or -1)
            except (TypeError, ValueError):
                requested_size = -1
            matches = []
            for root in session_roots:
                if not os.path.isdir(root):
                    continue
                for directory, dirnames, filenames in os.walk(root):
                    dirnames[:] = [name for name in dirnames
                                   if name != '_handbook']
                    if requested_name in filenames:
                        path = os.path.join(directory, requested_name)
                        try:
                            if (requested_size >= 0
                                    and os.path.getsize(path) != requested_size):
                                continue
                            matches.append(path)
                        except OSError:
                            continue
            if matches:
                candidate = max(matches, key=os.path.getmtime)
                session_root = containing_root(candidate) or session_roots[0]

        if (not candidate or not containing_root(candidate)
                or not os.path.isfile(candidate)):
            evicted = _evicted_message(
                session_roots, artifact_ref,
                os.path.basename(candidate or '') or request.args.get('name'))
            if evicted:
                return jsonify({'success': False, 'evicted': True,
                                'error': evicted}), 410
            app.logger.warning(
                'Studio output missing: ref=%r name=%r resolved=%r roots=%r',
                artifact_ref, request.args.get('name'), candidate, session_roots)
            return jsonify({
                'success': False, 'error': 'Output file is no longer available'
            }), 404

        session_root = containing_root(candidate) or session_root
        artifact_ref = os.path.relpath(
            candidate, session_root).replace(os.sep, '/')
        filename = os.path.basename(candidate)
        size_bytes = os.path.getsize(candidate)
        mime_type = (
            mimetypes.guess_type(filename)[0] or 'application/octet-stream')

        if request.args.get('download') == '1':
            return send_file(
                candidate, as_attachment=True, download_name=filename,
                mimetype=mime_type, conditional=True)

        if request.args.get('raw') == '1':
            if not (mime_type.startswith('image/')
                    or mime_type == 'application/pdf'):
                return jsonify({
                    'success': False,
                    'error': 'Inline binary preview is not available for this file type',
                }), 415
            response = send_file(
                candidate, as_attachment=False, download_name=filename,
                mimetype=mime_type, conditional=True, max_age=0)
            response.headers['Cache-Control'] = 'private, no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        extension = os.path.splitext(filename)[1].casefold()

        # ?attributes=1 asks for the data behind the preview rather than the
        # preview itself: the feature table for a vector or tabular output,
        # the properties/band/value description for a raster. Read here, on
        # the artifact path, so a large file is never shipped to the browser
        # just to count its rows.
        if request.args.get('attributes') == '1':
            table_exts = {'.geojson', '.json', '.gpkg', '.shp', '.zip',
                          '.csv', '.tsv', '.xlsx', '.xls', '.parquet'}
            raster_exts = {'.tif', '.tiff'}
            try:
                if extension in raster_exts:
                    return jsonify({
                        'success': True, 'kind': 'raster', 'name': filename,
                        **describe_raster(candidate),
                    })
                if extension in table_exts:
                    max_rows = request.args.get('max_rows', 500, type=int)
                    return jsonify({
                        'success': True, 'kind': 'table', 'name': filename,
                        **read_attribute_table(candidate, max_rows),
                    })
            except ImportError as exc:
                return jsonify({
                    'success': False,
                    'error': f'Missing library on the server: {exc}',
                }), 500
            except Exception as exc:
                return jsonify({
                    'success': False,
                    'error': f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__,
                }), 422
            return jsonify({
                'success': False,
                'error': 'This file type has no attribute table.',
            }), 415

        text_extensions = {
            '.txt', '.csv', '.tsv', '.json', '.geojson', '.xml', '.osm',
            '.html', '.htm', '.py', '.md', '.toml', '.yaml', '.yml', '.log',
            '.sql', '.js', '.css', '.prj', '.cpg',
        }
        image_types = {
            'image/png', 'image/jpeg', 'image/gif', 'image/webp',
        }
        preview = {
            'artifact_ref': artifact_ref,
            'name': filename,
            'size_bytes': size_bytes,
            'mime_type': mime_type,
            'kind': 'binary',
            'content': '',
            'truncated': False,
        }
        # Geospatial and tabular kinds are classified BEFORE the generic text
        # branch: a .geojson is perfectly readable as text, but a reviewer
        # judging whether the right area came back needs it on a map, and a
        # .csv needs columns, not a wall of commas.
        if mime_type in image_types:
            preview['kind'] = 'image'
        elif mime_type == 'application/pdf':
            preview['kind'] = 'pdf'
        elif extension in ('.geojson', '.gpkg', '.shp'):
            geo = _studio_preview_geometry(candidate, extension)
            if geo:
                preview.update(geo)
        elif extension in ('.tif', '.tiff'):
            # ?pixels=1 asks for the rendered image itself rather than the
            # metadata summary. Served from this route so it inherits the same
            # ownership and path-containment checks the summary already passed.
            if request.args.get('pixels'):
                # Cached on disk keyed by path+mtime+size: rendering is pure,
                # so the same file never pays for it twice, and panning back to
                # a raster is instant.
                import hashlib
                stat = os.stat(candidate)
                key = hashlib.sha256(
                    f'{candidate}|{stat.st_mtime_ns}|{stat.st_size}'.encode()
                ).hexdigest()[:32]
                cache_dir = os.path.join(session_roots[0], '_previews')
                cached = os.path.join(cache_dir, key + '.png')
                png = None
                if os.path.isfile(cached):
                    with open(cached, 'rb') as handle:
                        png = handle.read()
                if png is None:
                    png, info = _studio_render_raster_png(candidate)
                    if not png:
                        return jsonify({
                            'success': False,
                            'error': (info or {}).get(
                                'reason', 'This raster could not be rendered.')
                        }), 422
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cached, 'wb') as handle:
                            handle.write(png)
                    except OSError:
                        pass          # a cache miss is not a failure
                response = app.response_class(png, mimetype='image/png')
                response.headers['Cache-Control'] = 'private, max-age=3600'
                return response
            raster = _studio_preview_raster(candidate)
            if raster:
                preview.update(raster)
                # Tells the viewer a real image can be fetched, so it draws
                # pixels instead of falling back to the footprint rectangle.
                # Only claimed when the raster can be placed at all: promising
                # pixels for an unplaceable file sent the viewer to ?pixels=1
                # for a 422 it could do nothing with.
                preview['has_pixels'] = bool(raster.get('bbox'))
        elif extension in ('.csv', '.tsv'):
            limit = 300_000
            with open(candidate, 'rb') as handle:
                raw = handle.read(limit + 1)
            preview['truncated'] = len(raw) > limit
            preview['kind'] = 'csv'
            preview['delimiter'] = '\t' if extension == '.tsv' else ','
            preview['content'] = raw[:limit].decode('utf-8', errors='replace')
        elif (extension in text_extensions or mime_type.startswith('text/')
              or mime_type in (
                  'application/json', 'application/geo+json',
                  'application/xml', 'application/javascript')):
            limit = 300_000
            with open(candidate, 'rb') as handle:
                raw = handle.read(limit + 1)
            preview['truncated'] = len(raw) > limit
            text = raw[:limit].decode('utf-8', errors='replace')
            if extension in ('.json', '.geojson') and not preview['truncated']:
                try:
                    text = json.dumps(
                        json.loads(text), ensure_ascii=False, indent=2)
                except (ValueError, TypeError):
                    pass
            preview['kind'] = 'text'
            preview['content'] = text

        return jsonify({'success': True, 'preview': preview})
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        app.logger.exception('Could not preview Generate & Test output')
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/data-sources', methods=['GET', 'POST'])
def data_sources_endpoint():
    """GET: list the caller's private contributed data sources.
    POST: create/update one. Both methods are served from a single view so the
    route reliably allows POST across Flask/Werkzeug versions."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from agents.data_agent import user_data_sources as uds

    if request.method == 'GET':
        return jsonify({'success': True, 'sources': uds.list_user_sources(hash_api_key(api_key))})

    # POST: create/update. Accepts JSON, or multipart/form-data when attaching
    # files. Fields: data_source_name (required), brief_description, handbook,
    # code_example; optional file field 'attachments' (one or more).
    attachments = []
    ct = request.content_type or ''
    if 'multipart/form-data' in ct:
        fields = {k: request.form.get(k, '') for k in uds.STORED_FIELDS}
        share = (request.form.get('share') or '').lower() == 'true'
        original_slug = (request.form.get('original_slug') or '').strip()
        for fs in request.files.getlist('attachments'):
            if fs and fs.filename:
                attachments.append((fs.filename, fs.read()))
    else:
        data = request.get_json(silent=True) or {}
        fields = {k: data.get(k, '') for k in uds.STORED_FIELDS}
        share = bool(data.get('share'))
        original_slug = (data.get('original_slug') or '').strip()

    if not (fields.get('data_source_name') or '').strip():
        return jsonify({'success': False, 'error': 'Data source name is required'}), 400

    try:
        uid = hash_api_key(api_key)
        slug = uds.write_user_source(uid, fields, attachments=attachments, share=share)
        # On edit, if the name (slug) changed, remove the source's old file(s).
        if original_slug and original_slug != slug:
            uds.delete_user_source(uid, original_slug)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Could not save data source: {e}'}), 500
    return jsonify({'success': True, 'slug': slug})


def _stream_handbook_run(work, error_label='Handbook generation failed'):
    """Run ``work()`` (a handbook generate/refine call) in a worker thread and
    stream its progress log live over SSE.

    The worker registers a log-capture callback that pushes each line onto a
    queue; the SSE generator forwards them to the browser as
    {'type': 'log', 'line': ...} events while the run is still going, with
    pings during quiet stretches so proxies don't drop the connection.
    ``work`` must return the final event dict (given the full ``logs`` list),
    or raise — ValueError is reported as-is, anything else as a wrapped error.
    """
    import queue as _queue
    import re as _re
    import threading as _threading
    from agents.data_agent.handbook_generator import (
        start_log_capture, stop_log_capture)

    # Captured lines carry a "[HH:MM:SS] " prefix added by the generator's
    # logger, so the markers are matched anywhere after it.
    stage_re = _re.compile(
        r'^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[stage:(\w+):(\w+)\]\s*(.*)$')
    chat_re = _re.compile(
        r'^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[chat\]\s*(.+)$')
    artifact_re = _re.compile(
        r'^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[artifact:(\w+)\]\s*(\{.*\})\s*$')

    def generate():
        q = _queue.Queue()
        result = {}

        def progress_event(line):
            """Translate internal stage markers into stable UI events."""
            m = stage_re.match(line) if isinstance(line, str) else None
            if not m:
                return None
            stage, status, message = m.groups()
            if stage not in {'source', 'docs', 'draft', 'verify', 'review'}:
                return None
            if status not in {'start', 'progress', 'complete', 'warning', 'error'}:
                return None
            return {'type': 'progress', 'stage': stage, 'status': status,
                    'message': message.strip()}

        def worker():
            # Capture is thread-local, so it must start inside this thread.
            start_log_capture(callback=lambda line: q.put(('log', line)))
            try:
                result['final'] = work()
            except Exception as e:
                import traceback
                traceback.print_exc()
                result['error'] = e
            finally:
                result['logs'] = stop_log_capture()
                q.put(('done', None))

        _threading.Thread(target=worker, daemon=True).start()
        while True:
            try:
                kind, payload = q.get(timeout=15)
            except _queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue
            if kind == 'log':
                art = artifact_re.match(payload) if isinstance(payload, str) else None
                if art and art.group(1) in {'source', 'docs', 'draft', 'verify', 'review'}:
                    try:
                        art_data = json.loads(art.group(2))
                    except ValueError:
                        art_data = None
                    if isinstance(art_data, dict):
                        # Structured stage deliverable (links, facts, draft
                        # preview, produced files); kept out of the live log
                        # view — the raw line still lands in the final logs.
                        yield f"data: {json.dumps({'type': 'artifact', 'stage': art.group(1), 'data': art_data})}\n\n"
                        continue
                progress = progress_event(payload)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                chat = chat_re.match(payload) if isinstance(payload, str) else None
                if chat:
                    yield f"data: {json.dumps({'type': 'assistant', 'message': chat.group(1).strip()})}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'line': payload})}\n\n"
            else:
                break

        logs = result.get('logs', [])
        err = result.get('error')
        if err is not None:
            msg = str(err) if isinstance(err, ValueError) else \
                f'{error_label}: {err}'
            yield f"data: {json.dumps({'type': 'error', 'success': False, 'error': msg, 'logs': logs})}\n\n"
            return
        final = dict(result.get('final') or {})
        final.update({'type': 'complete', 'success': True, 'logs': logs})
        yield f"data: {json.dumps(final)}\n\n"

    return sse_response(generate())


@app.route('/api/data-sources/generate', methods=['POST'])
def generate_data_source_endpoint():
    """Draft a handbook for a data source with the LLM (GPT-6 Luna by default).

    Body (JSON): {name | data_source_name (required), website?, notes? |
    brief_description?}. Streams SSE: {'type':'log', 'line'} progress events
    live during the run, then one final {'type':'complete', success, source,
    verification, logs} (or {'type':'error', ...}) event. `source` has the same
    field shape the "Add data source" form consumes (_dsPrefill). Nothing is
    saved — the user reviews and edits the draft, then saves via POST
    /api/data-sources as usual."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401

    data = request.get_json(silent=True) or {}
    # Free-form description of the source to document (the dedicated AI input).
    # Falls back to legacy name/query fields for older callers.
    query = (data.get('query') or data.get('name')
             or data.get('data_source_name') or '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'Describe the data source to generate a handbook'}), 400
    website = (data.get('website') or '').strip()
    # Optional explicit documentation URL(s): a string or a list. Combined with
    # the website and any URLs found in the query, these pages are fetched and
    # used to ground the handbook in the source's real docs.
    raw_docs = data.get('doc_urls') or data.get('api_url') or []
    doc_urls = [raw_docs] if isinstance(raw_docs, str) else list(raw_docs)
    # Draft-only generation is retained for the normal Handbook Generator.
    # Experiment 2 always self-verifies the draft before freezing it as H0.
    # This is the H0 (baseline) condition in handbook experiments.
    verify = data.get('verify')
    verify = True if verify is None else str(verify).lower() not in ('false', '0', 'no', '')

    from agents.data_agent.handbook_generator import (
        generate_handbook, HANDBOOK_GEN_MODEL)

    def work():
        source = generate_handbook(query, website=website, doc_urls=doc_urls,
                                   user_key=api_key, verify=verify)
        # The self-test report rides inside source['_verification']; also lift
        # it to a top-level 'verification' field so the UI can show pass/fail
        # without digging.
        verification = source.pop('_verification', None)
        return {'source': source, 'verification': verification,
                'model': HANDBOOK_GEN_MODEL}

    return _stream_handbook_run(work)


@app.route('/api/data-sources/refine', methods=['POST'])
def refine_data_source_endpoint():
    """Refine an existing handbook draft via chat. Body (JSON):
    {source (current handbook fields, required), message (required),
    history? (prior [{role,content}] turns), website?}. Streams SSE like
    /generate: live {'type':'log'} events, then one final {'type':'complete',
    success, source, assistant_message, logs} event. Stateless — the client
    keeps the draft + history, which suits the multi-worker deploy."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401

    data = request.get_json(silent=True) or {}
    current = data.get('source') if isinstance(data.get('source'), dict) else {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'success': False, 'error': 'Enter a message describing what to change'}), 400
    history = data.get('history') if isinstance(data.get('history'), list) else []
    website = (data.get('website') or current.get('website') or '').strip()

    from agents.data_agent.handbook_generator import (
        refine_handbook, HANDBOOK_GEN_MODEL)

    def work():
        result = refine_handbook(current, message, history=history,
                                 website=website, user_key=api_key)
        return {'source': result['source'],
                'assistant_message': result['assistant_message'],
                'model': HANDBOOK_GEN_MODEL}

    return _stream_handbook_run(work, error_label='Handbook refinement failed')


@app.route('/api/data-sources/verify', methods=['POST'])
def verify_data_source_endpoint():
    """Re-run a generated handbook's sample download with user-supplied
    data-source credentials. Body (JSON): {source (handbook fields, required),
    keys? ({ENV_NAME: value})}. Streams SSE like /generate: live 'log' and
    'progress' events, then one final {'type':'complete', success, source,
    verification, logs} event. Credentials are used only for this run — they
    are never stored, logged, or echoed back."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401

    data = request.get_json(silent=True) or {}
    source = data.get('source') if isinstance(data.get('source'), dict) else {}
    if not str(source.get('code_example', '') or '').strip():
        return jsonify({'success': False,
                        'error': 'The handbook has no code example to verify'}), 400
    raw_keys = data.get('keys') if isinstance(data.get('keys'), dict) else {}
    keys = {str(k).strip(): str(v).strip() for k, v in raw_keys.items()
            if str(k).strip() and str(v).strip()}

    from agents.data_agent.handbook_generator import (
        verify_handbook, HANDBOOK_GEN_MODEL, _log)

    def work():
        _log('[stage:verify:start] Running the sample download with your credentials in an isolated process.')
        revised, report = verify_handbook(source, data_source_keys=keys,
                                          user_key=api_key)
        status = (report or {}).get('status', 'skipped')
        if status == 'passed':
            _log('[stage:verify:complete] The sample download completed successfully with your credentials.')
        elif status == 'passed_smoke':
            _log('[stage:verify:warning] A required credential was still missing, so the full test could not run.')
        else:
            _log('[stage:verify:warning] The sample download did not complete; you can refine the handbook and try again.')
        _log('[stage:review:complete] Your handbook is ready to review, edit, and save.')
        return {'source': revised, 'verification': report,
                'model': HANDBOOK_GEN_MODEL}

    return _stream_handbook_run(work, error_label='Handbook verification failed')


@app.route('/api/data-sources/catalog', methods=['GET'])
def data_sources_catalog_endpoint():
    """List the curated global Handbook catalog (available to everyone).

    No API key needed: the catalog is public (it is bundled with the app and
    synced from the public GitHub repo), and this only lists it — credential
    *names* a source needs, never values. Deleting from it stays admin-only."""
    from agents.data_agent import user_data_sources as uds
    return jsonify({'success': True, 'sources': uds.list_global_sources()})


@app.route('/api/data-sources/<slug>', methods=['DELETE'])
def delete_data_source_endpoint(slug):
    """Delete one of the caller's private data sources (and withdraw it from the
    review queue)."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from agents.data_agent import user_data_sources as uds
    removed = uds.delete_user_source(hash_api_key(api_key), slug)
    return jsonify({'success': True, 'removed': removed})


@app.route('/api/data-sources/catalog/<slug>', methods=['DELETE'])
def delete_catalog_data_source_endpoint(slug):
    """Admin: delete a source from the global (shared) catalog."""
    api_key = g.api_key
    if not _is_admin(api_key):
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    from agents.data_agent import user_data_sources as uds
    removed = uds.delete_global_source(slug)
    return jsonify({'success': True, 'removed': removed})


@app.route('/api/data-sources/<slug>/share', methods=['POST'])
def share_data_source_endpoint(slug):
    """Submit an existing private source to the review queue (Private → Pending)."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from agents.data_agent import user_data_sources as uds
    uid = hash_api_key(api_key)
    src = uds.get_user_source(uid, slug)
    if not src:
        return jsonify({'success': False, 'error': 'Source not found'}), 404
    uds.submit_to_queue(uid, slug, src)
    return jsonify({'success': True})


@app.route('/api/data-sources/admin/pending', methods=['GET'])
def list_pending_data_sources_endpoint():
    """Admin: list submissions awaiting review."""
    api_key = g.api_key
    if not _is_admin(api_key):
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    from agents.data_agent import user_data_sources as uds
    return jsonify({'success': True, 'pending': uds.list_pending()})


@app.route('/api/data-sources/admin/<pending_id>/approve', methods=['POST'])
def approve_data_source_endpoint(pending_id):
    """Admin: publish a submission into the curated global catalog."""
    api_key = g.api_key
    if not _is_admin(api_key):
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    from agents.data_agent import user_data_sources as uds
    ok = uds.approve_pending(pending_id)
    return jsonify({'success': ok})


@app.route('/api/data-sources/admin/<pending_id>/reject', methods=['POST'])
def reject_data_source_endpoint(pending_id):
    """Admin: reject a submission (with an optional reason the contributor sees)."""
    api_key = g.api_key
    if not _is_admin(api_key):
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    from agents.data_agent import user_data_sources as uds
    reason = ((request.get_json(silent=True) or {}).get('reason') or '').strip()
    ok = uds.reject_pending(pending_id, reason)
    return jsonify({'success': ok})


@app.route('/api/eda-explore', methods=['POST'])
def eda_explore_endpoint():
    """Stream an ad-hoc EDA task with real-time output and interactive support."""
    data = request.get_json()
    if not data or 'task' not in data:
        return jsonify({'success': False, 'error': 'Missing task'}), 400

    task = data['task']

    def generate():
        try:
            for update in eda_explore_stream(task):
                yield f"data: {json.dumps(update, default=str)}\n\n"
                time.sleep(0.01)
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/eda-explore-approval', methods=['POST'])
def eda_explore_approval_endpoint():
    """Relay user approval response during ad-hoc EDA explore."""
    content_type = request.content_type or ''

    if 'multipart/form-data' in content_type:
        # File upload — same logic as main eda-approval
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        conv_id = request.form.get('conversation_id')
        if conv_id:
            upload_dir = os.path.join(BASE_DIR, 'uploads', conv_id)
        else:
            upload_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, file.filename)
        file.save(save_path)

        audit_records = audit_uploaded_data([save_path], [file.filename])
        submit_eda_explore_approval({
            "type": "file_upload",
            "file_path": save_path,
            "file_name": file.filename,
            "audit_records": audit_records,
        })
        return jsonify({'success': True, 'message': 'File uploaded', 'audit_records': audit_records})
    else:
        data = request.get_json()
        if not data or 'response' not in data:
            return jsonify({'success': False, 'error': 'Missing response'}), 400
        submit_eda_explore_approval(data['response'])
        return jsonify({'success': True, 'message': 'Approval submitted'})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'AGM Backend is running'})



# Store active solution instances by task_id
active_solutions = {}

@app.route('/api/workflow/nodes/<task_id>', methods=['GET'])
def get_workflow_nodes(task_id):
    """
    Get detailed information about all nodes in a workflow graph
    """
    try:
        result = get_workflow_nodes_info(task_id)
        return jsonify({
            'success': True,
            **result
        })

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_workflow_nodes: {error_details}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/workflow/refine-stream/<task_id>', methods=['POST'])
def refine_workflow_stream(task_id):
    """
    Streaming version of workflow graph refinement
    Sends real-time LLM updates via Server-Sent Events
    Expects JSON: {"feedback": "user feedback text", "task": "...", "data_overview": "..."}
    """
    # Get request data BEFORE the generator to avoid context issues
    try:
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({'error': 'Missing feedback in request body'}), 400

        feedback = data['feedback']
        task = data.get('task')
        data_overview = data.get('data_overview')

        print(f"[refine_workflow_stream] Starting stream for task {task_id}")
        print(f"[refine_workflow_stream] Feedback: {feedback}")

    except Exception as e:
        print(f"[refine_workflow_stream] Error parsing request: {e}")
        return jsonify({'error': str(e)}), 400

    api_key = g.api_key

    def generate():
        if not api_key:
            yield _no_key_event()
            return
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting workflow refinement...'})}\n\n"

            # Use streaming handler that yields progress updates
            for update in refine_workflow_graph_stream(
                task_id=task_id,
                feedback=feedback,
                task=task,
                data_overview=data_overview,
                active_solutions=active_solutions
            ):
                # Send each update as an SSE event
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Workflow refinement complete'})}\n\n"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in refine_workflow_stream generator: {error_details}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/approve/<task_id>', methods=['POST'])
def approve_workflow(task_id):
    """
    Mark a workflow as approved by the user
    """
    try:
        result = approve_workflow_graph(task_id, active_solutions=active_solutions)

        return jsonify({
            'success': True,
            'message': 'Workflow approved',
            **result
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in approve_workflow: {error_details}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/workflow/generate-code-stream/<task_id>', methods=['POST'])
def generate_code_stream_endpoint(task_id):
    """
    Streaming version of code generation endpoint (without execution)
    Sends real-time LLM updates via Server-Sent Events
    """
    def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting code generation...'})}\n\n"

            # Import the streaming version
            from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import generate_code_stream

            # Use streaming handler that yields progress updates
            for update in generate_code_stream(
                task_id=task_id,
                active_solutions=active_solutions
            ):
                # Send each update as an SSE event
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Code generation complete'})}\n\n"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in generate_code_stream: {error_details}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/execute-code-stream/<task_id>', methods=['POST'])
def execute_code_stream_endpoint(task_id):
    """
    Streaming version of code execution endpoint
    Sends real-time LLM updates via Server-Sent Events
    """
    def generate():
        try:
            # Get code from request
            data = request.get_json()
            if not data or 'code' not in data:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No code provided'})}\n\n"
                return

            code = data['code']

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting code execution...'})}\n\n"

            # Import the streaming version
            from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import execute_code_stream

            # Use streaming handler that yields progress updates
            for update in execute_code_stream(
                task_id=task_id,
                code=code,
                active_solutions=active_solutions
            ):
                # Send each update as an SSE event
                yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Code execution complete'})}\n\n"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in execute_code_stream: {error_details}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/refine-code/<task_id>', methods=['POST'])
def refine_code(task_id):
    """
    Refine generated code based on user feedback
    Expects JSON: {"feedback": "user feedback text", "code_type": "assembly"}
    """
    try:
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({'error': 'Missing feedback in request body'}), 400

        feedback = data['feedback']
        code_type = data.get('code_type', 'assembly')

        result = refine_workflow_code(
            task_id=task_id,
            feedback=feedback,
            code_type=code_type,
            active_solutions=active_solutions
        )

        return jsonify({
            'success': True,
            'message': 'Code refined successfully',
            **result
        })

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in refine_code: {error_details}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/workflow/refine-and-run/<task_id>', methods=['POST'])
def refine_and_run(task_id):
    """Refine the generated code with user feedback, then execute it. SSE stream."""
    data = request.get_json() or {}
    feedback = data.get('feedback', '')
    code_type = data.get('code_type', 'assembly')
    if not feedback:
        return jsonify({'error': 'Missing feedback in request body'}), 400

    # Without this, a refine-and-run only ever reached the live page: the
    # stored conversation kept its PRE-refinement card, so a reload dropped
    # any artifact the refined code newly produced (e.g. a GeoJSON added to
    # the assembly output via feedback) even though the file was on disk.
    # The sibling re-run endpoints already persist this way.
    convo_id = data.get('conversation_id') or _conversation_id_for_task(task_id)

    def generate():
        try:
            for update in refine_and_run_stream(
                task_id=task_id,
                feedback=feedback,
                code_type=code_type,
                active_solutions=active_solutions,
            ):
                if update.get('type') == 'step_result':
                    try:
                        _update_workflow_step_result(
                            convo_id, update,
                            objective_key_override=(
                                'assembly' if code_type == 'assembly' else None),
                            step_index_override=(
                                'Execution' if code_type == 'assembly' else None),
                        )
                    except Exception:
                        pass
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            import traceback
            print(f"Error in refine_and_run stream: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return sse_response(generate())


@app.route('/api/workflow/download-code/<task_id>', methods=['GET'])
def download_code(task_id):
    """
    Download the generated code as a .py file
    """
    try:
        from utils.config import load_configurations
        from flask import send_file

        settings = load_configurations()
        # Only serve from the per-task directory — no global fallback.
        code_dir = os.path.join(settings.output_dir, task_id, "generated_code")
        code_file = os.path.join(code_dir, f"{task_id}.py")

        if not os.path.exists(code_file):
            return jsonify({'error': 'Code file not found'}), 404

        return send_file(
            code_file,
            as_attachment=True,
            download_name=f"workflow_{task_id}.py",
            mimetype='text/x-python'
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in download_code: {error_details}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflow/continue/<task_id>', methods=['GET'])
def continue_workflow_endpoint(task_id):
    """
    Continue workflow to next step after code execution.
    Streaming endpoint that returns SSE events.
    """
    try:
        from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_testing import continue_to_next_step

        def generate():
            try:
                print(f"[continue_workflow] Starting continuation for task {task_id}")
                for update in continue_to_next_step(task_id):
                    yield f"data: {json.dumps(update, cls=GeoJSONEncoder)}\n\n"

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in continue_workflow generator: {error_details}")
                yield f"data: {json.dumps({{'type': 'error', 'message': str(e)}})}\n\n"

        return sse_response(generate())

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in continue_workflow_endpoint: {error_details}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/uploads/<conversation_id>/<filename>')
def serve_uploaded_file(conversation_id, filename):
    """Serve an uploaded data file from a conversation (disk or DB fallback)."""
    try:
        from werkzeug.utils import secure_filename
        safe = secure_filename(filename)
        conv_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
        file_path = os.path.join(conv_dir, safe)
        if not os.path.exists(file_path):
            # Try restoring from DB
            restored = restore_file_to_disk(conversation_id, safe, conv_dir)
            if not restored:
                return jsonify({'error': 'File not found'}), 404
            file_path = restored
        return send_file(file_path, as_attachment=False)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tif_bounds/<conversation_id>/<filename>')
def get_tif_bounds(conversation_id, filename):
    """Return the WGS84 bounding box of a GeoTIFF upload using rasterio."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from werkzeug.utils import secure_filename
        safe = secure_filename(filename)
        conv_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
        file_path = os.path.join(conv_dir, safe)
        if not os.path.exists(file_path):
            restored = restore_file_to_disk(conversation_id, safe, conv_dir)
            if not restored:
                return jsonify({'error': 'File not found'}), 404
            file_path = restored
        with rasterio.open(file_path) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            # bounds = (west, south, east, north)
            return jsonify({
                'west':  bounds[0],
                'south': bounds[1],
                'east':  bounds[2],
                'north': bounds[3],
                'crs':   str(src.crs),
                'width': src.width,
                'height': src.height,
                'bands': src.count,
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/artifacts/<path:filename>')
def serve_artifact(filename):
    """
    Serve generated artifact files (maps, CSVs, etc.).

    The requested path MUST be scoped to a task: the first path segment is
    treated as the task_id and must resolve to an existing task subdirectory
    under Outputs/. Files living directly in Outputs/ (unscoped) are rejected.
    """
    try:
        from utils.config import load_configurations
        settings = load_configurations()
        outputs_dir = os.path.abspath(settings.output_dir)

        # Normalize separators and strip any leading slashes
        norm = filename.replace('\\', '/').lstrip('/')
        parts = [p for p in norm.split('/') if p not in ('', '.')]
        # if len(parts) < 2:
        #     # Must have at least <task_id>/<filename>
        #     return jsonify({'error': 'Artifact path must be scoped to a task (expected <task_id>/<path>)'}), 400

        task_id = parts[0]
        # Reject paths that attempt to escape or point to the legacy global dirs
        if task_id in ('', '.', '..', 'workflow_states'):
            return jsonify({'error': 'Invalid task_id in artifact path'}), 400

        # Construct the full path: outputs_dir/task_id/...
        task_dir = os.path.abspath(os.path.join(outputs_dir, task_id))
        if not task_dir.startswith(outputs_dir + os.sep) and task_dir != outputs_dir:
            return jsonify({'error': 'Access denied'}), 403
        if not os.path.isdir(task_dir):
            return jsonify({'error': f'Unknown task_id: {task_id}'}), 404

        # Reconstruct file path: outputs_dir/task_id/remaining_parts
        file_path = os.path.abspath(os.path.join(outputs_dir, task_id, *parts[1:]))

        # Security check: file must live within the resolved task directory
        if not file_path.startswith(task_dir + os.sep):
            return jsonify({'error': 'Access denied'}), 403

        if not os.path.exists(file_path):
            # DB fallback for execution flowcharts
            if filename.endswith('execution_flowchart.html'):
                db_resp = _serve_html_from_db(task_id, "execution_flowchart")
                if db_resp:
                    return db_resp
            return jsonify({'error': 'File not found'}), 404

        # conditional=True enables 304 revalidation via If-Modified-Since/If-None-Match.
        # no-cache forces the browser to revalidate on every request rather than
        # serving a stale copy from cache after a re-run regenerates the file at
        # the same path.
        resp = send_file(file_path, as_attachment=False, conditional=True, max_age=0)
        resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
        return resp

    except Exception as e:
        print(f"Error serving artifact: {e}")
        return jsonify({'error': str(e)}), 500


# ── Figure catalog v2 — user-promotion endpoints ─────────────────────────────

def _resolve_task_dir(task_id):
    """Return the absolute task output dir, or (None, error_response)."""
    from utils.config import load_configurations
    settings = load_configurations()
    outputs_dir = os.path.abspath(settings.output_dir)
    if not task_id or os.sep in task_id or '/' in task_id or '\\' in task_id or task_id in ('.', '..'):
        return None, (jsonify({'error': 'Invalid task_id'}), 400)
    task_dir = os.path.abspath(os.path.join(outputs_dir, task_id))
    if not task_dir.startswith(outputs_dir + os.sep) and task_dir != outputs_dir:
        return None, (jsonify({'error': 'Access denied'}), 403)
    if not os.path.isdir(task_dir):
        return None, (jsonify({'error': f'Unknown task_id: {task_id}'}), 404)
    return task_dir, None


def _safe_under(task_dir, file_path):
    """Return absolute file_path if it lives under task_dir, else None."""
    abs_path = os.path.abspath(
        file_path if os.path.isabs(file_path) else os.path.join(task_dir, file_path)
    )
    if not abs_path.startswith(task_dir + os.sep):
        return None
    return abs_path


@app.route('/api/figures/promotable')
def list_promotable_figures():
    """List images in OUTPUT_DIR that are not already in the catalog.

    Query: ?task_id=<id>
    Returns: [{path, suggested_caption, in_figures_dir}, ...]
    """
    task_id = request.args.get('task_id')
    task_dir, err = _resolve_task_dir(task_id)
    if err:
        return err
    try:
        from agents.result_analysis_agent.figure_catalog_v2 import list_promotable
        ctx = {"output_dir": task_dir, "task_id": task_id, "artifacts": {}}
        items = list_promotable(ctx)
        # Return task-relative paths so the frontend can resolve via /api/artifacts
        for it in items:
            rel = os.path.relpath(it["path"], os.path.dirname(task_dir)).replace("\\", "/")
            it["relative_path"] = rel
        return jsonify({"figures": items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/figures/promote', methods=['POST'])
def promote_figure():
    """Promote an intermediate figure into the manuscript catalog.

    Body: {task_id, file_path, caption?, objective_key?, step_index?}
    """
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id')
    file_path = data.get('file_path')
    if not task_id or not file_path:
        return jsonify({'error': 'task_id and file_path are required'}), 400

    task_dir, err = _resolve_task_dir(task_id)
    if err:
        return err
    abs_path = _safe_under(task_dir, file_path)
    if not abs_path or not os.path.exists(abs_path):
        return jsonify({'error': 'File not found under task directory'}), 404

    try:
        from agents.result_analysis_agent.figure_catalog_v2 import promote_intermediate
        promote_intermediate(
            output_dir=task_dir,
            file_path=abs_path,
            caption=data.get('caption'),
            objective_key=data.get('objective_key'),
            step_index=data.get('step_index'),
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/figures/demote', methods=['POST'])
def demote_figure():
    """Reverse a promotion."""
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id')
    file_path = data.get('file_path')
    if not task_id or not file_path:
        return jsonify({'error': 'task_id and file_path are required'}), 400

    task_dir, err = _resolve_task_dir(task_id)
    if err:
        return err
    abs_path = _safe_under(task_dir, file_path)
    if not abs_path:
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        from agents.result_analysis_agent.figure_catalog_v2 import demote_intermediate
        removed = demote_intermediate(task_dir, abs_path)
        return jsonify({'status': 'ok', 'removed': removed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report-download/<report_type>/<fmt>')
def download_report(report_type, fmt):
    """
    Download a result report in the requested format (md, docx, pdf).
    report_type: 'presentation' or 'discussion'
    fmt: 'md', 'docx', or 'pdf'
    """
    try:
        from utils.config import load_configurations
        settings = load_configurations()
        # task_id is REQUIRED so reports are always served from the per-task
        # directory — never from a global Outputs/ fallback that could
        # cross-contaminate conversations.
        tid = request.args.get('task_id')
        if not tid:
            return jsonify({'error': 'task_id query parameter is required'}), 400
        # Reject traversal / path-separators in task_id
        if os.sep in tid or '/' in tid or '\\' in tid or tid in ('', '.', '..'):
            return jsonify({'error': 'Invalid task_id'}), 400

        outputs_dir = os.path.join(settings.output_dir, tid)
        if not os.path.isdir(outputs_dir):
            return jsonify({'error': f'Unknown task_id: {tid}'}), 404

        ext_map = {'md': '.md', 'docx': '.docx', 'pdf': '.pdf'}

        if fmt not in ext_map:
            return jsonify({'error': f'Unsupported format: {fmt}'}), 400

        if report_type not in ('presentation', 'discussion', 'manuscript'):
            return jsonify({'error': f'Unknown report type: {report_type}'}), 400

        # manuscript.md vs result_presentation.md / result_discussion.md
        base_name = 'manuscript' if report_type == 'manuscript' else f"result_{report_type}"

        file_path = os.path.join(outputs_dir, base_name + ext_map[fmt])

        # If DOCX/PDF doesn't exist yet, generate on-the-fly from the .md
        if not os.path.exists(file_path) and fmt in ('docx', 'pdf'):
            md_path = os.path.join(outputs_dir, base_name + '.md')
            if not os.path.exists(md_path):
                return jsonify({'error': 'Source markdown report not found'}), 404
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            from agents.result_analysis_agent.report_converter import _md_to_docx, _md_to_pdf
            if fmt == 'docx':
                _md_to_docx(md_content, file_path)
            else:
                _md_to_pdf(md_content, file_path)

        if not os.path.exists(file_path):
            return jsonify({'error': 'Report file not found'}), 404

        # Security check
        if not os.path.abspath(file_path).startswith(os.path.abspath(outputs_dir)):
            return jsonify({'error': 'Access denied'}), 403

        mime_types = {
            'md': 'text/markdown',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'pdf': 'application/pdf',
        }

        return send_file(
            file_path,
            as_attachment=True,
            download_name=base_name + ext_map[fmt],
            mimetype=mime_types.get(fmt),
        )

    except Exception as e:
        print(f"Error downloading report: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert-geopackage', methods=['POST'])
def convert_geopackage():
    """
    Convert uploaded GeoPackage to GeoJSON using geopandas.
    """
    import tempfile
    import shutil

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if not file.filename.lower().endswith('.gpkg'):
            return jsonify({'error': 'File must be a GeoPackage (.gpkg)'}), 400

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)

            print(f"GeoPackage saved to: {file_path}")

            # Use geopandas to read the GeoPackage
            import geopandas as gpd
            import fiona

            # List layers in the GeoPackage
            layers = fiona.listlayers(file_path)
            print(f"Layers in GeoPackage: {layers}")

            if not layers:
                return jsonify({'error': 'No layers found in GeoPackage'}), 400

            # Read the first layer
            gdf = gpd.read_file(file_path, layer=layers[0])

            # Store original CRS for logging
            original_crs = str(gdf.crs) if gdf.crs else 'Unknown'
            print(f"Original CRS: {original_crs}")

            # Reproject to EPSG:4326 (WGS84) if not already in that CRS
            if gdf.crs and gdf.crs.to_string() != 'EPSG:4326':
                print(f"Reprojecting from {gdf.crs} to EPSG:4326...")
                gdf = gdf.to_crs('EPSG:4326')
                print("Reprojection complete")
            elif not gdf.crs:
                print("WARNING: No CRS found, assuming EPSG:4326")
                gdf = gdf.set_crs('EPSG:4326')

            # Convert Timestamp columns to strings to avoid JSON serialization errors
            import pandas as pd
            import numpy as np
            for col in gdf.columns:
                if pd.api.types.is_datetime64_any_dtype(gdf[col]):
                    gdf[col] = gdf[col].astype(str)
                # Replace Infinity/-Infinity with NaN, then NaN with None for JSON compatibility
                elif pd.api.types.is_numeric_dtype(gdf[col]):
                    gdf[col] = gdf[col].replace([np.inf, -np.inf], np.nan)

            # Convert to GeoJSON
            import json, re
            geojson_str = gdf.to_json()
            # Infinity/-Infinity/NaN are not valid JSON — replace with null
            geojson_str = re.sub(r'\b-?Infinity\b', 'null', geojson_str)
            geojson_str = re.sub(r'\bNaN\b', 'null', geojson_str)
            geojson_dict = json.loads(geojson_str)

            return jsonify({
                'success': True,
                'geojson': geojson_dict,
                'feature_count': len(geojson_dict.get('features', [])),
                'layers': layers,
                'loaded_layer': layers[0],
                'original_crs': original_crs,
                'output_crs': 'EPSG:4326',
                'crs': 'EPSG:4326'  # Always return EPSG:4326 now
            })

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except ImportError as e:
        return jsonify({'error': f'Missing library: {str(e)}. Run: pip install geopandas fiona pyogrio'}), 500
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error converting GeoPackage: {error_details}")
        # Return the actual exception type + message so the frontend can show it
        return jsonify({'error': f'{type(e).__name__}: {str(e)}' if str(e) else type(e).__name__}), 500




@app.route('/api/convert-shapefile', methods=['POST'])
def convert_shapefile():
    """
    Convert uploaded Shapefile components (.shp, .dbf, .shx, .prj, etc.) to GeoJSON
    using geopandas.  Accepts multipart/form-data with one or more files.
    """
    import tempfile
    import shutil

    try:
        if not request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        # Must contain at least a .shp
        shp_file = None
        for f in files:
            if f.filename.lower().endswith('.shp'):
                shp_file = f
                break
        if not shp_file:
            return jsonify({'error': 'No .shp file found in upload'}), 400

        # Save all components to a temp directory
        temp_dir = tempfile.mkdtemp()

        try:
            for f in files:
                save_path = os.path.join(temp_dir, f.filename)
                f.save(save_path)
                print(f"[convert-shapefile] Saved: {f.filename}")

            shp_path = os.path.join(temp_dir, shp_file.filename)
            print(f"[convert-shapefile] Reading shapefile: {shp_path}")

            import geopandas as gpd

            gdf = gpd.read_file(shp_path)
            print(f"[convert-shapefile] Features: {len(gdf)}, CRS: {gdf.crs}")

            original_crs = str(gdf.crs) if gdf.crs else 'Unknown'

            # Reproject to EPSG:4326 (WGS84) for the map viewer
            if gdf.crs and gdf.crs.to_string() != 'EPSG:4326':
                print(f"[convert-shapefile] Reprojecting from {gdf.crs} to EPSG:4326...")
                gdf = gdf.to_crs('EPSG:4326')
            elif not gdf.crs:
                print("[convert-shapefile] WARNING: No CRS found, assuming EPSG:4326")
                gdf = gdf.set_crs('EPSG:4326')

            # Sanitise columns for JSON
            import pandas as pd
            import numpy as np
            for col in gdf.columns:
                if pd.api.types.is_datetime64_any_dtype(gdf[col]):
                    gdf[col] = gdf[col].astype(str)
                elif pd.api.types.is_numeric_dtype(gdf[col]):
                    gdf[col] = gdf[col].replace([np.inf, -np.inf], np.nan)

            import json, re
            geojson_str = gdf.to_json()
            geojson_str = re.sub(r'\b-?Infinity\b', 'null', geojson_str)
            geojson_str = re.sub(r'\bNaN\b', 'null', geojson_str)
            geojson_dict = json.loads(geojson_str)

            return jsonify({
                'success': True,
                'geojson': geojson_dict,
                'feature_count': len(geojson_dict.get('features', [])),
                'original_crs': original_crs,
                'output_crs': 'EPSG:4326'
            })

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except ImportError as e:
        return jsonify({'error': f'Missing library: {str(e)}. Run: pip install geopandas fiona pyogrio'}), 500
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[convert-shapefile] Error: {error_details}")
        return jsonify({'error': f'{type(e).__name__}: {str(e)}' if str(e) else type(e).__name__}), 500


def _json_number(value):
    """Make a numpy / None / non-finite value JSON-safe."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number or number in (float('inf'), float('-inf')):
        return None
    return number


def read_attribute_table(path, max_rows=500):
    """Read one file's attribute table as columns + JSON-safe rows.

    Handles anything geopandas opens (.gpkg, .shp, .geojson, .zip) and the
    tabular formats pandas reads (.csv, .tsv, .xlsx, .parquet), so a caller
    can hand it whatever the user clicked without branching first.
    """
    import json
    import re

    import numpy as np
    import pandas as pd

    extension = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    if extension in ('csv', 'tsv', 'txt'):
        # sep=None lets the python engine sniff comma vs tab
        frame = pd.read_csv(path, sep=None, engine='python')
    elif extension in ('xlsx', 'xls'):
        frame = pd.read_excel(path)
    elif extension == 'parquet':
        frame = pd.read_parquet(path)
    else:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        # Drop geometry column — we only want the attribute table
        frame = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))

    total_rows = len(frame)
    if total_rows > max_rows:
        frame = frame.head(max_rows)

    # Sanitise values for JSON
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    records_str = frame.to_json(orient='records')
    records_str = re.sub(r'\b-?Infinity\b', 'null', records_str)
    records_str = re.sub(r'\bNaN\b', 'null', records_str)
    rows = json.loads(records_str)

    return {
        'columns': [str(c) for c in frame.columns],
        'rows': rows,
        'total_rows': total_rows,
        'returned_rows': len(rows),
    }


def describe_raster(path, max_categories=60):
    """Describe a raster: properties, per-band statistics, value counts.

    A raster has no feature table, so this is what stands in for one: the file
    and CRS properties, statistics per band, and — for a classified raster
    (land cover, a model's output classes) — the count of pixels per value,
    which is the closest thing it has to an attribute table.

    Statistics come from a decimated read for large rasters; the returned
    'sampled' flag says whether that happened.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    num = _json_number

    with rasterio.open(path) as src:
        # Statistics come from a decimated read — a full read of a large
        # raster is slow and the summary numbers barely move.
        MAX_DIM = 1024
        scale = max(src.width / MAX_DIM, src.height / MAX_DIM, 1.0)
        out_h = max(1, int(round(src.height / scale)))
        out_w = max(1, int(round(src.width / scale)))
        sampled = (out_w != src.width or out_h != src.height)

        source_crs = src.crs or _sidecar_crs(path)
        try:
            wgs84 = transform_bounds(source_crs, 'EPSG:4326', *src.bounds) if source_crs else None
        except Exception:
            wgs84 = None

        res_x, res_y = src.res
        properties = {
            'file_name': os.path.basename(path),
            'driver': src.driver,
            'crs': str(source_crs) if source_crs else None,
            'width': src.width,
            'height': src.height,
            'band_count': src.count,
            'dtypes': list(src.dtypes),
            'nodata': num(src.nodata),
            'pixel_size_x': num(res_x),
            'pixel_size_y': num(res_y),
            'total_pixels': int(src.width) * int(src.height),
            'bounds': {
                'left': num(src.bounds.left),
                'bottom': num(src.bounds.bottom),
                'right': num(src.bounds.right),
                'top': num(src.bounds.top),
            },
            'wgs84_bounds': ({
                'west': num(wgs84[0]), 'south': num(wgs84[1]),
                'east': num(wgs84[2]), 'north': num(wgs84[3]),
            } if wgs84 else None),
            'transform': [num(x) for x in list(src.transform)[:6]],
        }

        try:
            colorinterp = [ci.name for ci in src.colorinterp]
        except Exception:
            colorinterp = []
        descriptions = src.descriptions or ()

        bands = []
        for i in range(1, src.count + 1):
            arr = src.read(i, out_shape=(out_h, out_w), masked=True)
            valid = arr.compressed()
            entry = {
                'band': i,
                'description': descriptions[i - 1] if i <= len(descriptions) else None,
                'dtype': src.dtypes[i - 1],
                'nodata': num(src.nodatavals[i - 1]),
                'color_interp': colorinterp[i - 1] if i <= len(colorinterp) else None,
                'valid_pixels': int(valid.size),
                'nodata_pixels': int(arr.size - valid.size),
            }
            if valid.size:
                entry.update({
                    'min': num(np.min(valid)),
                    'max': num(np.max(valid)),
                    'mean': num(np.mean(valid)),
                    'std': num(np.std(valid)),
                })
            else:
                entry.update({'min': None, 'max': None, 'mean': None, 'std': None})
            bands.append(entry)

        # A categorical raster (land cover, classified output) gets a
        # value/count table. Continuous rasters get None and the caller shows
        # statistics only.
        categories = None
        if src.count:
            first = src.read(1, out_shape=(out_h, out_w), masked=True).compressed()
            if first.size:
                looks_discrete = (np.issubdtype(first.dtype, np.integer)
                                  or bool(np.all(np.mod(first, 1) == 0)))
                if looks_discrete:
                    values, counts = np.unique(first, return_counts=True)
                    if len(values) <= max_categories:
                        try:
                            cmap = src.colormap(1)
                        except Exception:
                            cmap = None
                        total = int(counts.sum())
                        categories = []
                        for idx in np.argsort(-counts):
                            value = values[idx]
                            rgba = cmap.get(int(value)) if cmap else None
                            categories.append({
                                'value': num(value),
                                'count': int(counts[idx]),
                                'percent': (round(float(counts[idx]) * 100.0 / total, 3)
                                            if total else None),
                                'color': ('#%02x%02x%02x' % tuple(rgba[:3])) if rgba else None,
                            })

        tags = {}
        try:
            for key, value in (src.tags() or {}).items():
                tags[str(key)] = str(value)
        except Exception:
            pass

    return {
        'properties': properties,
        'bands': bands,
        'categories': categories,
        'tags': tags,
        'sampled': sampled,
        'sample_shape': [out_w, out_h],
    }


@app.route('/api/read-attributes', methods=['POST'])
def read_attributes():
    """
    Read the attribute table from any geospatial file supported by geopandas
    (.gpkg, .shp, .geojson, .json, .zip containing shapefile, etc.), or from a
    tabular file pandas can read (.csv, .tsv, .xlsx, .xls, .parquet).
    Accepts multipart/form-data with one or more 'files'.
    Returns columns and rows as JSON (geometry is excluded).
    An optional 'max_rows' form field caps the returned rows (default 500).
    """
    import tempfile
    import shutil

    try:
        if not request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        max_rows = request.form.get('max_rows', 500, type=int)

        temp_dir = tempfile.mkdtemp()

        try:
            # Save all uploaded files to temp dir
            primary_path = None
            for f in files:
                save_path = os.path.join(temp_dir, f.filename)
                f.save(save_path)
                ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
                # Pick the main file to open (gpkg > shp > geojson/json > zip)
                if ext == 'gpkg':
                    primary_path = save_path
                elif ext == 'shp' and not (primary_path and primary_path.endswith('.gpkg')):
                    primary_path = save_path
                elif ext in ('geojson', 'json') and not primary_path:
                    primary_path = save_path
                elif ext == 'zip' and not primary_path:
                    primary_path = save_path

            if not primary_path:
                # Fallback: just use the first file
                primary_path = os.path.join(temp_dir, files[0].filename)

            print(f"[read-attributes] Reading: {primary_path}")
            table = read_attribute_table(primary_path, max_rows)
            print(f"[read-attributes] Rows: {table['total_rows']}, Columns: {table['columns']}")

            return jsonify({'success': True, **table})

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except ImportError as e:
        return jsonify({'error': f'Missing library: {str(e)}. Run: pip install geopandas fiona pyogrio'}), 500
    except Exception as e:
        import traceback
        print(f"[read-attributes] Error: {traceback.format_exc()}")
        return jsonify({'error': f'{type(e).__name__}: {str(e)}' if str(e) else type(e).__name__}), 500


@app.route('/api/raster-info', methods=['POST'])
def raster_info():
    """
    Describe a raster the way /api/read-attributes describes a vector layer:
    file + CRS properties, per-band statistics, and — for categorical rasters
    (land cover, classifications) — a value/count table, which is the closest
    thing a raster has to an attribute table.

    Accepts either an uploaded file ('files' or 'file' in multipart form data)
    or a server-backed upload named by 'conversation_id' + 'filename'.
    """
    import tempfile
    import shutil
    from werkzeug.utils import secure_filename

    temp_dir = None
    try:
        uploaded = request.files.getlist('files') or request.files.getlist('file')
        if uploaded:
            temp_dir = tempfile.mkdtemp()
            first = uploaded[0]
            file_path = os.path.join(temp_dir, secure_filename(first.filename or 'raster.tif'))
            first.save(file_path)
        else:
            conversation_id = request.form.get('conversation_id') or request.args.get('conversation_id')
            filename = request.form.get('filename') or request.args.get('filename')
            if not (conversation_id and filename):
                return jsonify({'error': 'No raster supplied'}), 400
            safe = secure_filename(filename)
            conv_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
            file_path = os.path.join(conv_dir, safe)
            if not os.path.exists(file_path):
                restored = restore_file_to_disk(conversation_id, safe, conv_dir)
                if not restored:
                    return jsonify({'error': 'File not found'}), 404
                file_path = restored

        max_categories = request.form.get('max_categories', 60, type=int)
        return jsonify({'success': True, **describe_raster(file_path, max_categories)})

    except ImportError as e:
        return jsonify({'error': f'Missing library: {str(e)}. Run: pip install rasterio'}), 500
    except Exception as e:
        import traceback
        print(f"[raster-info] Error: {traceback.format_exc()}")
        return jsonify({'error': f'{type(e).__name__}: {str(e)}' if str(e) else type(e).__name__}), 500
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ── Conversation History Endpoints ────────────────────────────────────


# GET /api/conversations and GET /api/conversations/<id> live in
# conversation_routes.py (registered near the top), so the desktop app can
# serve them before this file has finished loading.


@app.route('/api/conversations', methods=['POST'])
def create_conversation_endpoint():
    """Create a new conversation for the current user."""
    data = request.get_json() or {}
    title = data.get('title', 'New Conversation')
    user_id = hash_api_key(g.api_key)
    convo = create_conversation(title=title, user_id=user_id)
    return jsonify({'success': True, 'conversation': convo}), 201


@app.route('/api/conversations/<conversation_id>/files/<path:filename>', methods=['DELETE'])
def remove_conversation_file_endpoint(conversation_id, filename):
    """Remove a file from the conversation's Data Layers so it stays removed
    after a reload. Soft: the file is kept on disk and in the database.
    Owner-only; a collaborator's removal stays local to their page."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({'error': 'Conversation not found'}), 404
    caller_user_id = hash_api_key(g.api_key)
    if not caller_owns(convo, caller_user_id):
        return jsonify({'error': 'Not authorized to modify this conversation'}), 403
    found = mark_conversation_file_removed(conversation_id, filename)
    if not found:
        # Uploads are stored under werkzeug's secure_filename (spaces become
        # underscores, etc.), while the page knows the name as picked.
        from werkzeug.utils import secure_filename
        safe = secure_filename(filename)
        if safe and safe != filename:
            found = mark_conversation_file_removed(conversation_id, safe)
    return jsonify({'success': True, 'removed': found})


@app.route('/api/conversations/<conversation_id>/files/<path:filename>/restore', methods=['POST'])
def restore_conversation_file_endpoint(conversation_id, filename):
    """Bring a removed file back into Data Layers (e.g. re-added from its
    download tile), so it is restored on reload again. Owner-only."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({'error': 'Conversation not found'}), 404
    caller_user_id = hash_api_key(g.api_key)
    if not caller_owns(convo, caller_user_id):
        return jsonify({'error': 'Not authorized to modify this conversation'}), 403
    restored = restore_conversation_file(conversation_id, filename)
    if not restored:
        from werkzeug.utils import secure_filename
        safe = secure_filename(filename)
        if safe and safe != filename:
            restored = restore_conversation_file(conversation_id, safe)
    return jsonify({'success': True, 'restored': restored})


@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation_endpoint(conversation_id):
    """Remove a conversation from the caller's history.

    Owners delete the conversation for everyone, permanently: its messages,
    uploads and outputs are removed so deleted conversations don't fill the
    disk (see conversation_cleanup.py). Collaborators (non-owners who opened
    a shared link) only have their own collaborator row removed, so the
    conversation disappears from their sidebar without affecting the owner
    or other collaborators.
    """
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({'error': 'Conversation not found'}), 404
    caller_user_id = hash_api_key(g.api_key)
    if not caller_user_id:
        return jsonify({'error': 'API key required'}), 401
    if caller_owns(convo, caller_user_id):
        from WebUI.conversation_cleanup import purge_conversation, vacuum_in_background
        try:
            freed = purge_conversation(conversation_id) or 0
        except Exception as e:
            # Still hide it, so the user's delete takes effect; the startup
            # cleanup retries removing its files.
            print(f"[delete] purge failed for {conversation_id}: {e}")
            delete_conversation(conversation_id)
            freed = 0
        try:
            from WebUI.agm_workflow import workflow_states
            for tid in {conversation_id, convo.get('task_id')}:
                workflow_states.pop(tid, None)
        except Exception:
            pass
        vacuum_in_background()
        return jsonify({'success': True, 'deleted': True, 'freed_bytes': freed,
                        'message': 'Conversation deleted'})
    # Non-owner: just leave the shared conversation.
    remove_collaborator(conversation_id, caller_user_id)
    return jsonify({'success': True, 'left': True, 'message': 'Removed from your conversations'})


@app.route('/api/conversations/<conversation_id>', methods=['PATCH'])
def update_conversation_endpoint(conversation_id):
    """Update conversation title or link a task_id. Owner-only."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({'error': 'Conversation not found'}), 404
    caller_user_id = hash_api_key(g.api_key)
    if not caller_owns(convo, caller_user_id):
        return jsonify({'error': 'Not authorized to modify this conversation'}), 403
    data = request.get_json() or {}
    if 'title' in data:
        update_conversation_title(conversation_id, data['title'])
    if 'task_id' in data:
        link_task_id(conversation_id, data['task_id'])
    if 'pinned' in data:
        set_conversation_pinned(conversation_id, bool(data['pinned']))
    convo = get_conversation(conversation_id)
    return jsonify({'success': True, 'conversation': convo})


@app.route('/api/conversations/<conversation_id>/collaborate', methods=['POST'])
def collaborate_on_conversation(conversation_id):
    """Add the calling user as a collaborator on a shared conversation.

    Used when a user opens someone else's shared link and starts interacting
    with it — the conversation gets added to their sidebar without being
    forked or duplicated. Same conversation_id, same task workspace, same
    artifacts. Idempotent: calling repeatedly is a no-op.
    """
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({'error': 'Conversation not found'}), 404

    caller_user_id = hash_api_key(g.api_key)
    if not caller_user_id:
        return jsonify({'error': 'API key required'}), 401

    # No-op if the caller already owns it.
    if caller_owns(convo, caller_user_id):
        return jsonify({'success': True, 'already_owner': True}), 200

    add_collaborator(conversation_id, caller_user_id)
    return jsonify({'success': True, 'conversation_id': conversation_id}), 200


@app.route('/api/conversations/<conversation_id>/fork', methods=['POST'])
def fork_conversation_endpoint(conversation_id):
    """Create a copy of an existing conversation owned by the current user.

    Used when a user opens a shared conversation link and starts mutating it —
    we fork first so the original (owned by someone else) stays untouched.
    Copies messages, conversation_files, file_blobs, and the on-disk uploads
    folder. Rejects the request if the source still has a live workflow
    running (task_id linked and task is active), to avoid copying mid-run
    state.
    """
    src = get_conversation(conversation_id)
    if not src:
        return jsonify({'error': 'Conversation not found'}), 404

    caller_user_id = hash_api_key(g.api_key)
    if not caller_user_id:
        return jsonify({'error': 'API key required to fork a conversation'}), 401

    # Block fork while the source's workflow is still running.
    src_task_id = src.get('task_id')
    if src_task_id:
        try:
            from WebUI.agm_workflow import _is_task_running  # optional helper
            running = _is_task_running(src_task_id)
        except Exception:
            # If we can't tell, be conservative and check the per-conversation
            # latch in this process — a live run sets _task_id_linked_for.
            running = bool(_task_id_linked_for.get(conversation_id))
        if running:
            return jsonify({
                'error': 'Original conversation is still running. '
                         'Wait for it to finish before forking.'
            }), 409

    new_convo = fork_conversation(conversation_id, new_user_id=caller_user_id)
    if not new_convo:
        return jsonify({'error': 'Failed to fork conversation'}), 500

    # Copy on-disk uploads so the fork is independent from the source.
    try:
        src_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
        dst_dir = os.path.join(BASE_DIR, 'uploads', new_convo['id'])
        if os.path.isdir(src_dir):
            import shutil as _shutil
            _shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    except Exception as e:
        print(f"[fork] upload folder copy failed: {e}")

    msgs = get_messages(new_convo['id'])
    files = get_conversation_files(new_convo['id'])
    return jsonify({
        'success': True,
        'conversation': new_convo,
        'messages': msgs,
        'files': files,
    }), 201


# ── Conversation export / import (sharing from the desktop app) ──────
# The desktop server only listens on 127.0.0.1, so Share links can't reach
# anyone else. Export writes the conversation, its uploads and its outputs to
# one .zip; Import recreates it on the recipient's computer.
#
# Export is two steps: POST builds the zip on disk and returns a one-time
# token, then the page navigates to GET .../<token> to download it. A plain
# download navigation can't carry the X-API-Key header, and fetching the zip
# into the page first would hold a multi-GB file in memory.

import shutil
import tempfile
import uuid
from datetime import datetime

_conversation_exports = {}           # token -> {path, filename, created}
_conversation_exports_lock = threading.Lock()
_EXPORT_TTL_SECONDS = 3600


def _drop_stale_exports():
    cutoff = time.time() - _EXPORT_TTL_SECONDS
    with _conversation_exports_lock:
        stale = [t for t, e in _conversation_exports.items() if e['created'] < cutoff]
        entries = [_conversation_exports.pop(t) for t in stale]
    for e in entries:
        shutil.rmtree(os.path.dirname(e['path']), ignore_errors=True)


@app.route('/api/conversations/<conversation_id>/export', methods=['POST'])
def export_conversation_endpoint(conversation_id):
    """Build a conversation's .zip and return a token to download it.
    Open to the owner and to collaborators."""
    from WebUI.conversation_bundle import export_conversation, BundleError
    from werkzeug.utils import secure_filename

    convo = get_conversation(conversation_id)
    if not convo or convo.get('deleted_at'):
        return jsonify({'error': 'Conversation not found'}), 404
    caller_user_id = hash_api_key(g.api_key)
    if not caller_user_id:
        return jsonify({'error': 'API key required to export a conversation'}), 401
    if not caller_owns(convo, caller_user_id):
        return jsonify({'error': 'Not authorized to export this conversation'}), 403

    _drop_stale_exports()
    work_dir = tempfile.mkdtemp(prefix='gcs_export_')
    path = os.path.join(work_dir, 'conversation.zip')
    try:
        # The keys saved in Settings, so they can be removed from the file.
        saved_keys = {'request': g.api_key,
                      'saved': (request.get_json(silent=True) or {}).get('saved_keys')}
        summary = export_conversation(conversation_id, path, saved_keys=saved_keys)
    except BundleError as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[export] failed for {conversation_id}: {e}")
        return jsonify({'error': f'Export failed: {e}'}), 500

    title = secure_filename(convo.get('title') or '')[:60].strip('_') or 'conversation'
    filename = f"GIS-CoScientist_{title}_{datetime.now().strftime('%Y-%m-%d')}.zip"
    token = uuid.uuid4().hex
    with _conversation_exports_lock:
        _conversation_exports[token] = {'path': path, 'filename': filename, 'created': time.time()}
    return jsonify({
        'success': True,
        'download_url': f'/api/conversation-exports/{token}',
        'filename': filename,
        'summary': summary,
    })


@app.route('/api/conversation-exports/<token>', methods=['GET'])
def download_conversation_export(token):
    """Send a built export once, then delete it. The token is the credential."""
    with _conversation_exports_lock:
        entry = _conversation_exports.pop(token, None)
    if not entry or not os.path.isfile(entry['path']):
        return jsonify({'error': 'This export has expired. Export the conversation again.'}), 404
    resp = send_file(entry['path'], as_attachment=True, download_name=entry['filename'],
                     mimetype='application/zip', max_age=0)
    work_dir = os.path.dirname(entry['path'])
    resp.call_on_close(lambda: shutil.rmtree(work_dir, ignore_errors=True))
    return resp


@app.route('/api/conversations/import', methods=['POST'])
def import_conversation_endpoint():
    """Recreate a conversation from an exported .zip, owned by the caller."""
    from WebUI.conversation_bundle import import_conversation, BundleError, BundleConflict

    caller_user_id = hash_api_key(g.api_key)
    if not caller_user_id:
        return jsonify({'error': 'Add your API key in Settings before importing a conversation'}), 401
    upload = request.files.get('bundle')
    if not upload or not upload.filename:
        return jsonify({'error': 'No file was sent'}), 400

    work_dir = tempfile.mkdtemp(prefix='gcs_import_')
    try:
        path = os.path.join(work_dir, 'bundle.zip')
        upload.save(path)
        try:
            convo = import_conversation(path, caller_user_id)
        except BundleConflict as e:
            return jsonify({
                'error': str(e),
                'conversation_id': e.conversation_id,
                'deleted': e.deleted,
            }), 409
        except BundleError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            print(f"[import] failed: {e}")
            return jsonify({'error': f'Import failed: {e}'}), 500
        return jsonify({'success': True, 'conversation': convo}), 201
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    print("Starting AGM Backend Server...")
    print("\nWeb Interface:")
    print("  - GET  /     - Welcome page")
    print("  - GET  /app  - AGM WebUI Application")
    print("\nAPI Endpoints:")
    print("  - POST /api/parse-research-question")
    print("  - POST /api/upload-and-audit-data")
    print("  - POST /api/chat-with-ai")
    print("  - POST /api/generate-solution")
    print("  - POST /api/convert-geopackage")
    print("  - GET  /api/health")
    print("\nDebug Endpoints:")
    print("  - POST /api/test-intent  - Test intent detection")
    print("\nServer running at http://localhost:5000")

    # Disable auto-reload to prevent server restarts during workflow execution
    # When files are generated in Outputs/, Flask was detecting changes and restarting,
    # which kills EventSource connections and causes ERR_CONNECTION_RESET errors
    print("\n Auto-reload ENABLED")
    # print(f"\n⚠️  Auto-reload ENABLED")
    # print("   (Prevents server restarts when workflow generates files)")
    # print("   NOTE: You must manually restart Flask after editing Python code\n")

    app.run(debug=True, host='0.0.0.0', port=4041, use_reloader=False, threaded=True,
            exclude_patterns=["*/Outputs/*", "*/generated_code/*", "*/data/*"])
