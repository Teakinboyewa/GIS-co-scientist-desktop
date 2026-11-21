"""
Conversation read routes that need only the local database.

Listing and opening saved conversations reads conversations.db and nothing
else, so these routes live here, away from app.py, whose import loads the
whole analysis engine (10 s or more). app.py registers this blueprint as
usual. The desktop app also serves it on its own, through create_early_app(),
while the engine is still loading, so the sidebar fills and a conversation
opens straight away (see desktop/staged_app.py).

Keep this module's imports to Flask and conversation_db.
"""

import os

from flask import Blueprint, Flask, g, jsonify, request

from WebUI import conversation_db
from WebUI.conversation_db import (
    get_conversation,
    get_conversation_files,
    get_messages,
    hash_api_key,
    list_conversations,
    restore_file_to_disk,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

bp = Blueprint("conversation_routes", __name__)


@bp.route('/api/conversations', methods=['GET'])
def list_conversations_endpoint():
    """List saved conversations for the current user."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    user_id = hash_api_key(g.api_key)
    # Without a key there is no user scope — return nothing rather than
    # leaking every conversation in the DB (list_conversations treats a
    # falsy user_id as "no filter").
    if not user_id:
        return jsonify({'success': True, 'conversations': []})
    convos = list_conversations(limit=limit, offset=offset, user_id=user_id)
    return jsonify({'success': True, 'conversations': convos})


@bp.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation_endpoint(conversation_id):
    """Get a conversation with all its messages."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({'error': 'Conversation not found'}), 404
    msgs = get_messages(conversation_id)
    all_files = get_conversation_files(conversation_id)
    # Files the user removed from Data Layers are not restored as layers. Their
    # names are sent separately so the download cards' file tiles (which also
    # register layers on reload) can skip them too.
    files = [f for f in all_files if not f.get('removed_at')]
    removed_files = [f['filename'] for f in all_files if f.get('removed_at')]
    # Check which files still exist on disk — check conversation subfolder first, then flat uploads.
    # If missing on disk, try restoring from the database.
    conv_upload_dir = os.path.join(BASE_DIR, 'uploads', conversation_id)
    flat_upload_dir = os.path.join(BASE_DIR, 'uploads')
    for f in files:
        conv_path = os.path.join(conv_upload_dir, f['filename'])
        flat_path = os.path.join(flat_upload_dir, f['filename'])
        if os.path.exists(conv_path) or os.path.exists(flat_path):
            f['file_exists'] = True
        else:
            # Try restoring from DB
            restored = restore_file_to_disk(conversation_id, f['filename'], conv_upload_dir)
            f['file_exists'] = restored is not None
    # Desktop app: every conversation in this computer's database is the
    # user's own (imported ones included), so it never opens as "shared".
    is_owner = True
    return jsonify({
        'success': True,
        'conversation': convo,
        'messages': msgs,
        'files': files,
        'removed_files': removed_files,
        'is_owner': is_owner,
    })


def create_early_app():
    """A small Flask app with just these routes, for the desktop app to serve
    while the engine loads. Returns None before the database exists (first
    launch): the engine creates it, so those requests wait for the engine."""
    if not os.path.isfile(conversation_db.DB_PATH):
        return None
    early = Flask("conversation_routes_early")

    @early.before_request
    def _api_key():
        # Same as app.py's extract_api_key for these GET routes: the key
        # comes in the X-API-Key header.
        g.api_key = request.headers.get('X-API-Key', '').strip() or None

    early.register_blueprint(bp)
    return early
