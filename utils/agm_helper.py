import configparser
from email.mime import base
import json as _json
import logging
import os
import re
import sys
from os import mkdir

import requests as _requests
from flask import ctx





# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

def load_config():
    config = configparser.ConfigParser()

    config_path = os.path.join(current_script_dir, 'config.ini')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            config.write(f)

    # print(config_path)
    config.read(config_path)
    return config


def load_OpenAI_key():
    """Return the GIBD API key (``USER_KEY``).

    Inside a Flask request the key comes from the UI (``g.api_key``).
    Outside of a request it falls back to the ``OPENAI_API_KEY`` env var / ``.env``.
    """
    # When called inside a Flask request, use the key the user provided via the UI
    try:
        from flask import g
        key = getattr(g, 'api_key', None)
        if key:
            return key
    except RuntimeError:
        pass  # Not in a Flask request context (e.g. notebooks, standalone scripts)

    # Outside of a web request, fall back to the environment / .env
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    return os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# GIBD-only enforcement
# ---------------------------------------------------------------------------
# Easy toggle: set ALLOW_OPENAI = True to re-enable plain OpenAI keys,
# or False (default) to accept GIBD keys only. The OpenAI routing code below
# is kept intact either way — just flip this one flag when you need OpenAI.
# (Mirrors the frontend ALLOW_OPENAI_KEY flag in script.js.)
ALLOW_OPENAI = True


def _ensure_gibd_only(key):
    """When ALLOW_OPENAI is False, reject a present non-GIBD key instead of
    silently falling back to OpenAI. GIBD keys and the local 'no-api' sentinel
    are always allowed."""
    if ALLOW_OPENAI:
        return
    if key and not str(key).startswith("gibd-") and key != "no-api":
        raise RuntimeError(
            "This app is configured for GIBD keys only — the provided key is not "
            "a GIBD key (it should start with 'gibd-'). Enter your GIBD key in "
            "Settings. (To re-enable OpenAI, set ALLOW_OPENAI = True in "
            "utils/agm_helper.py.)"
        )

# ---------------------------------------------------------------------------
# UI-selected model & reasoning effort (set by agm_workflow.py on each request)
# ---------------------------------------------------------------------------

_ui_model = None
_ui_reasoning_effort = None

def set_ui_model(model, reasoning_effort=None):
    """Called by agm_workflow.py to propagate UI selections to all agents."""
    global _ui_model, _ui_reasoning_effort
    _ui_model = model
    _ui_reasoning_effort = reasoning_effort

def get_ui_model():
    """Return the model selected in the UI, or 'gpt-6-luna' as fallback."""
    return _ui_model or "gpt-6-luna"

def get_ui_reasoning_effort():
    """Return the reasoning effort selected in the UI, or None."""
    return _ui_reasoning_effort

# ---------------------------------------------------------------------------
# GIBD API helpers
# ---------------------------------------------------------------------------

GIBD_API_URL = "https://www.gibd.online"
# GIBD_SERVICE = "Autonomous Geographic Modelling Agent"
GIBD_SERVICE = "GIS Co-Scientist"
LOCAL_BASE_URL = "http://128.118.54.16:11434/v1"
LOCAL_MODEL_NAME = "deepseek-v3:latest"  # Override applied for every "no-api" call


LOCAL_EXTRA_BODY = {"enable_thinking": False}  # Disable Gemma's thinking phase


def _usable_openai_fallback_key():
    """Return an API key that is safe to pass to the public OpenAI API.

    A GIBD user key (``gibd-...``) is NOT a valid OpenAI key — sending it to
    api.openai.com only produces a confusing 401 and leaks the key into
    OpenAI's error logs. So the fallback only kicks in if the env-configured
    key clearly looks like a real OpenAI key (``sk-...`` / ``sk-proj-...``).
    Returns ``None`` if no usable key is configured.
    """
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY") or ""
    api_key = api_key.strip()
    if not api_key:
        return None
    if api_key.startswith("gibd-") or api_key.startswith("gibd_"):
        return None
    if not api_key.startswith("sk-"):
        # Not a recognizable OpenAI key shape — refuse to try it so we don't
        # mask the real GIBD error with a generic 401.
        return None
    return api_key


class _DotDict:
    """Wrap a dict so that nested keys can be accessed with dot notation.

    This lets existing code like ``response.choices[0].message.content``
    keep working when the underlying data comes from a raw JSON dict
    instead of the OpenAI Python SDK.
    """

    def __init__(self, d):
        for k, v in (d if isinstance(d, dict) else {}).items():
            if isinstance(v, dict):
                setattr(self, k, _DotDict(v))
            elif isinstance(v, list):
                setattr(self, k, [_DotDict(i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        # Return None for missing attributes, matching OpenAI SDK behaviour
        # (e.g. chunk.choices[0].delta.content is None when key absent).
        return None

    def get(self, key, default=None):
        val = getattr(self, key, default)
        return default if val is None else val


# ── HTTP resilience for GIBD calls ───────────────────────────────────────────
# The GIBD proxy occasionally times out on connect or returns transient 5xx
# responses.  Without timeouts and retries a single blip kills a long-running
# manuscript pipeline (8 sequential LLM calls).  These constants apply to
# every request to gibd.online.

# (connect_timeout, read_timeout) — read is generous for slow LLM responses
GIBD_HTTP_TIMEOUT = (15, 600)
GIBD_MAX_RETRIES = 4
GIBD_BACKOFF_BASE = 2.0  # seconds — 2, 4, 8, 16


def _gibd_post_with_retry(url, *, json=None, stream=False, label="GIBD"):
    """
    POST to a GIBD endpoint with timeouts and exponential backoff on
    transient failures (connection errors, read timeouts, 5xx, 429).

    Raises RuntimeError on permanent failure (4xx other than 429, or
    after all retries exhausted).
    """
    import time as _time

    last_err = None
    for attempt in range(GIBD_MAX_RETRIES):
        try:
            resp = _requests.post(
                url, json=json, stream=stream, timeout=GIBD_HTTP_TIMEOUT
            )
        except (_requests.ConnectionError, _requests.Timeout) as e:
            last_err = e
            if attempt < GIBD_MAX_RETRIES - 1:
                wait = GIBD_BACKOFF_BASE * (2 ** attempt)
                print(f"[{label}] {type(e).__name__} on attempt {attempt + 1}/{GIBD_MAX_RETRIES} — retrying in {wait}s...")
                _time.sleep(wait)
                continue
            raise RuntimeError(
                f"{label} unreachable after {GIBD_MAX_RETRIES} attempts: {e}"
            ) from e

        # Retry transient server errors and rate limits
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < GIBD_MAX_RETRIES - 1:
                wait = GIBD_BACKOFF_BASE * (2 ** attempt)
                print(f"[{label}] HTTP {resp.status_code} on attempt {attempt + 1}/{GIBD_MAX_RETRIES} — retrying in {wait}s...")
                _time.sleep(wait)
                continue
            raise RuntimeError(
                f"{label} failed after {GIBD_MAX_RETRIES} attempts: HTTP {resp.status_code}"
            )

        return resp

    # Defensive — loop should always either return or raise
    raise RuntimeError(f"{label} failed: {last_err}")


def _gibd_handshake(user_key):
    """Step 1 of GIBD flow – obtain a ``question_id``."""
    resp = _gibd_post_with_retry(
        f"{GIBD_API_URL}/api/request-question-id",
        json={"user_api_key": user_key, "service_name": GIBD_SERVICE},
        label="GIBD handshake",
    )
    if resp.status_code != 201:
        raise RuntimeError(f"GIBD handshake failed ({resp.status_code}): {resp.text}")
    return resp.json()["question_id"]



def gibd_chat_completion(user_key, model, messages, **kwargs):
    """Non-streaming chat completion through the GIBD proxy.

    Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
    can be passed as keyword arguments and will be forwarded in the payload.

    Returns a ``_DotDict`` that supports the same attribute access as an
    OpenAI ``ChatCompletion`` object, e.g.
    ``response.choices[0].message.content``.
    """

    q_id = _gibd_handshake(user_key)
    payload = {
        "question_id": q_id,
        "service_name": GIBD_SERVICE,
        "model": model,
        "messages": messages,
        "stream": False,
        **kwargs,
    }
    resp = _gibd_post_with_retry(
        f"{GIBD_API_URL}/api/openai/{user_key}",
        json=payload,
        label="GIBD inference",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"GIBD inference failed ({resp.status_code}): {resp.text}")
    return _DotDict(resp.json())




def gibd_chat_completion_stream(user_key, model, messages, **kwargs):
    """Streaming chat completion through the GIBD proxy.

    Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
    can be passed as keyword arguments and will be forwarded in the payload.

    Yields ``_DotDict`` chunks that support the same attribute access as
    OpenAI streaming chunks, e.g. ``chunk.choices[0].delta.content``.
    """

    q_id = _gibd_handshake(user_key)
    payload = {
        "question_id": q_id,
        "service_name": GIBD_SERVICE,
        "model": model,
        "messages": messages,
        "stream": True,
        **kwargs,
    }
    resp = _gibd_post_with_retry(
        f"{GIBD_API_URL}/api/openai/{user_key}",
        json=payload,
        stream=True,
        label="GIBD stream",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"GIBD stream failed ({resp.status_code}): {resp.text}")

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").strip()
        # Strip the SSE "data:" prefix (with or without trailing space)
        if decoded.startswith("data:"):
            decoded = decoded[5:].lstrip()
        if not decoded or decoded == "[DONE]":
            break
        try:
            chunk = _json.loads(decoded)
            if "error" in chunk:
                raise RuntimeError(f"LLM API error: {chunk['error']}")
            yield _DotDict(chunk)
        except _json.JSONDecodeError:
            continue


def openai_chat_completion(user_key, model, messages, **kwargs):
    """Non-streaming chat completion via the OpenAI API directly.

      Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
      can be passed as keyword arguments.

      Returns the native OpenAI ``ChatCompletion`` object, so callers can use
      ``response.choices[0].message.content`` as usual.
    """
    from openai import OpenAI
    client = OpenAI(api_key=user_key)
    return client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )


def local_chat_completion(model, messages, base_url=None, api_key=None, **kwargs):
    """Non Streaming chat completion from a local/self-hosted OpenAI-compatible
    server. ``base_url``/``model`` default to the shared LOCAL_* constants but
    can be overridden per-call (e.g. from an Experiment 2 study's controls)."""
    from openai import OpenAI
    client = OpenAI(
        base_url=base_url or LOCAL_BASE_URL,
        api_key=api_key or "no-api",
    )
    return client.chat.completions.create(
        model=model or LOCAL_MODEL_NAME,
        messages=messages, stream=False,
        # extra_body=LOCAL_EXTRA_BODY,
        **kwargs
    )



def openai_chat_completion_stream(user_key, model, messages, **kwargs):
    """Streaming chat completion via the OpenAI API directly.

    Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
    can be passed as keyword arguments.

    Yields native OpenAI streaming chunks, e.g.
    ``chunk.choices[0].delta.content``.
    """
    from openai import OpenAI
    client = OpenAI(api_key=user_key)
    response = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    )
    for chunk in response:
        yield chunk

def local_chat_completion_stream(model, messages, base_url=None, api_key=None, **kwargs):
    """Streaming chat completion from a local/self-hosted OpenAI-compatible
    server. ``base_url``/``model`` default to the shared LOCAL_* constants but
    can be overridden per-call (e.g. from an Experiment 2 study's controls)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url or LOCAL_BASE_URL, api_key=api_key or "no-api")

    response = client.chat.completions.create(
        model=model or LOCAL_MODEL_NAME, messages=messages, stream=True,
        # extra_body=LOCAL_EXTRA_BODY,
        **kwargs
    )
    for chunk in response:
        yield chunk



def _route_chat_completion(model, messages, user_key=None, base_url=None, **kwargs):
    """Route a non-streaming chat completion to GIBD, a custom/local server,
    or OpenAI based on ``base_url``/the key.

    A caller-supplied ``base_url`` (e.g. an Experiment 2 study's open-source
    model settings) always wins and routes to that OpenAI-compatible server.
    Otherwise, if ``user_key`` starts with ``"gibd-"`` the call is forwarded
    to :func:`gibd_chat_completion`; a ``"no-api"`` key routes to the default
    local server; anything else goes directly to :func:`openai_chat_completion`.

    Returns an object that supports ``response.choices[0].message.content``
    in every path (``_DotDict`` from GIBD, native ``ChatCompletion`` otherwise).
    """
    key = user_key or load_OpenAI_key()
    _ensure_gibd_only(key)
    if base_url:
        print(f"Using custom base_url ({base_url}) without streaming")
        return local_chat_completion(
            model, messages, base_url=base_url, api_key=key, **kwargs)
    if key and key.startswith("gibd-"):
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using GIBD-Key without streaming")
        return gibd_chat_completion(key, model, messages, **kwargs)
    elif key and key == "no-api":
        print("Using Local model without streaming")
        return local_chat_completion(model, messages, **kwargs)
    else:
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using openai API key and without streaming")
        return openai_chat_completion(key, model, messages, **kwargs)


def _route_chat_completion_stream(model, messages, user_key=None, base_url=None, **kwargs):
    """Route a streaming chat completion to GIBD, a custom/local server, or
    OpenAI based on ``base_url``/the key. See :func:`client_chat_completion`
    for the routing rules.

    Yields streaming chunks with the same ``chunk.choices[0].delta.content``
    shape in every path.
    """
    key = user_key or load_OpenAI_key()
    _ensure_gibd_only(key)
    if base_url:
        print(f"Using custom base_url ({base_url}) and streaming")
        yield from local_chat_completion_stream(
            model, messages, base_url=base_url, api_key=key, **kwargs)
    elif key and key.startswith("gibd-"):
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using GIBD-key and streaming")
        yield from gibd_chat_completion_stream(key, model, messages, **kwargs)
    elif key and key=="no-api":
        print("Using Local model and streaming")
        yield from local_chat_completion_stream(model, messages, **kwargs)
    else:
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using openai API key and streaming")
        yield from openai_chat_completion_stream(key, model, messages, **kwargs)



def _usage_from_chunk(chunk):
    """The usage block on a streamed chunk, whatever object carries it.

    OpenAI SDK chunks expose ``.usage``; the GIBD proxy's chunks arrive as
    ``_DotDict``, which returns None for absent keys. Both are handled, and a
    chunk without usage yields None so callers can keep the last one seen.
    """
    usage = getattr(chunk, "usage", None)
    if usage is None and isinstance(chunk, dict):
        usage = chunk.get("usage")
    return usage or None


def client_chat_completion(model, messages, user_key=None, base_url=None,
                           record_usage=True, **kwargs):
    """Non-streaming chat completion, recorded against the active usage ledger.

    This is the second of the two LLM families in this codebase (the other is
    :class:`GIBDChat`). The Research, Spatial Analysis and Data Retrieval modes
    all reach it -- the task manager, the spatial analysis executor and the RQ
    understanding agent call it directly -- so it is where their token cost has
    to be captured.

    Pass ``record_usage=False`` for a caller that records the same call itself;
    see ``DataRetrieverAgent._stream_with_usage``, which does its own
    per-chunk accounting and would otherwise be counted twice.
    """
    response = _route_chat_completion(
        model, messages, user_key=user_key, base_url=base_url, **kwargs)
    if record_usage:
        _record_llm_usage(model, _usage_from_chunk(response) or {}, source="chat")
    return response


def client_chat_completion_stream(model, messages, user_key=None, base_url=None,
                                  record_usage=True, **kwargs):
    """Streaming chat completion, recorded against the active usage ledger.

    OpenAI-style backends send the usage figures only on a final chunk, and
    only when ``stream_options`` asks for them. Not every backend this routes
    to accepts that argument, and because the router is a GENERATOR a rejection
    does not surface until the first chunk is pulled -- so a try/except around
    the call alone would never fire. The first chunk is therefore pulled here:
    if that fails, the whole stream is retried without ``stream_options``
    before any content has reached the caller. Losing the token count is
    acceptable; losing the stream is not.
    """
    if not record_usage or "stream_options" in kwargs:
        # The caller is doing its own accounting, or has already specified how
        # it wants usage reported. Either way, stay out of the way.
        yield from _route_chat_completion_stream(
            model, messages, user_key=user_key, base_url=base_url, **kwargs)
        return

    first = None
    iterator = None
    try:
        iterator = iter(_route_chat_completion_stream(
            model, messages, user_key=user_key, base_url=base_url,
            stream_options={"include_usage": True}, **kwargs))
        first = next(iterator)
    except StopIteration:
        iterator = None
    except Exception:
        print("[client_chat_completion_stream] backend rejected stream_options; "
              "retrying without usage capture")
        yield from _route_chat_completion_stream(
            model, messages, user_key=user_key, base_url=base_url, **kwargs)
        return

    usage = None
    if first is not None:
        usage = _usage_from_chunk(first) or usage
        yield first
        for chunk in iterator:
            usage = _usage_from_chunk(chunk) or usage
            yield chunk

    # Recorded once, after the stream is exhausted -- one call, not one per
    # chunk. A backend that accepted stream_options but sent no usage leaves
    # this empty, and the ledger counts the call and flags it as missing rather
    # than pretending it was free.
    _record_llm_usage(model, usage or {}, source="chat-stream")


# ---------------------------------------------------------------------------
# LangChain-compatible GIBD chat model for API call that include toolcall
# ---------------------------------------------------------------------------

from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


def _lc_message_to_dict(msg: BaseMessage) -> dict:
    """Convert a LangChain message to OpenAI dict format."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AIMessage):
        d: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _json.dumps(tc["args"]),
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
        }
    # Fallback for any other message type
    return {"role": getattr(msg, "type", "user"), "content": msg.content}


def _dict_to_ai_message(d: dict) -> AIMessage:
    """Convert an OpenAI response message dict to a LangChain ``AIMessage``."""
    content = d.get("content", "") or ""
    tool_calls = []
    if d.get("tool_calls"):
        for tc in d["tool_calls"]:
            func = tc.get("function", {})
            try:
                args = _json.loads(func.get("arguments", "{}"))
            except _json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {"name": func.get("name", ""), "args": args, "id": tc.get("id", "")}
            )
    return AIMessage(content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
#
# Every LLM call the Research, Spatial Analysis and Data Retrieval modes make
# goes through GIBDChat, so this is the one place where usage can be captured
# for all three. It has to happen at call time: once a run finishes, per-request
# usage is unrecoverable -- the GIBD proxy and OpenAI expose no way to look up
# historical usage keyed by anything this application knows.
#
# Recording is a no-op when no ledger is active (notebooks, CLI, tests), so
# instrumenting a call site never changes behaviour outside a collection scope.

def _usage_ledger():
    """Import lazily so a partially-installed environment still loads helper."""
    try:
        from agents.data_agent import usage_ledger
        return usage_ledger
    except Exception:
        return None


def _record_llm_usage(model, usage, source=""):
    """Record one call against the active ledger; returns the entry or None."""
    ledger = _usage_ledger()
    if ledger is None:
        return None
    try:
        return ledger.record(model, usage, source=source)
    except Exception:
        # Accounting must never take down a working LLM call.
        return None


def _usage_from_lc_message(msg):
    """Pull usage off a LangChain message, whichever shape the provider used.

    ``ChatOpenAI`` puts it on ``usage_metadata`` (input_tokens/output_tokens)
    on new versions and under ``response_metadata['token_usage']``
    (prompt_tokens/completion_tokens) on older ones. Both are handled so the
    direct-OpenAI fallback path is accounted for as well as the proxy.
    """
    if msg is None:
        return None
    meta = getattr(msg, "usage_metadata", None)
    if isinstance(meta, dict) and meta:
        return meta
    response_meta = getattr(msg, "response_metadata", None) or {}
    if isinstance(response_meta, dict):
        token_usage = response_meta.get("token_usage") or response_meta.get("usage")
        if isinstance(token_usage, dict) and token_usage:
            return token_usage
    return None


def _record_chat_result(result, model, source=""):
    """Record usage from a ChatResult produced by a delegated ChatOpenAI call."""
    try:
        generations = getattr(result, "generations", None) or []
        for generation in generations:
            usage = _usage_from_lc_message(getattr(generation, "message", None))
            if usage:
                _record_llm_usage(model, usage, source=source)
                break
        else:
            llm_output = getattr(result, "llm_output", None) or {}
            usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
            _record_llm_usage(model, usage or {}, source=source)
    except Exception:
        pass
    return result


def _record_stream_chunk(chunk, model, source=""):
    """Record usage carried on a streamed chunk. Returns True when it did.

    Streaming responses report usage only on a final chunk, and only when the
    request asked for it. Chunks without usage are ignored, so this is safe to
    call on every chunk.
    """
    usage = _usage_from_lc_message(getattr(chunk, "message", None))
    if not usage:
        return False
    _record_llm_usage(model, usage, source=source)
    return True


# Models whose Chat Completions endpoint refuses function tools together with
# any reasoning level other than "none" (400: "Function tools with
# reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions").
# Their tool-using calls are sent with reasoning_effort="none"; calls without
# tools keep the model's default or the level the caller set.
NO_REASONING_WITH_TOOLS = {"gpt-6-luna"}


def tool_call_reasoning_effort(model_name, has_tools, requested):
    """The reasoning_effort to send, or None to send none."""
    if has_tools and model_name in NO_REASONING_WITH_TOOLS:
        return "none"
    return requested


class GIBDChat(BaseChatModel):
    """LangChain chat model that routes through the GIBD proxy API.

    Drop-in replacement for ``ChatOpenAI``. Take the model from
    utils/llm_models.toml by the call's name rather than writing it here::

        from utils.llm_models import llm_model
        llm = GIBDChat(model_name=llm_model("research_plan.generate"))
    """

    model_name: str = ""
    gibd_user_key: str = ""
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" — for reasoning models

    @property
    def _llm_type(self) -> str:
        return "gibd-chat"

    def bind_tools(self, tools, **kwargs):
        """Bind tools so they are sent with every request."""
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    # -- non-streaming ---------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        key = self.gibd_user_key or load_OpenAI_key()
        _ensure_gibd_only(key)
        # Local model (Ollama) — explicit "no-api" sentinel
        if key == "no-api":
            print("Using Local model for GIBDChat without streaming")
            from langchain_openai import ChatOpenAI
            oai_llm = ChatOpenAI(
                api_key="no-api",
                base_url=LOCAL_BASE_URL,
                model=LOCAL_MODEL_NAME,
                # extra_body=LOCAL_EXTRA_BODY,
            )
            fwd = {}
            if "tools" in kwargs:
                fwd["tools"] = kwargs["tools"]
            return _record_chat_result(
                oai_llm._generate(messages, stop=stop, run_manager=run_manager, **fwd),
                LOCAL_MODEL_NAME, source="local")
        # Route to direct OpenAI when the key isn't a GIBD key
        if not (key and key.startswith("gibd-")):
            print("Using openai key for the GIBDChat without streaming")
            from langchain_openai import ChatOpenAI
            effort = tool_call_reasoning_effort(self.model_name, "tools" in kwargs, None)
            oai_llm = ChatOpenAI(api_key=key, model=self.model_name,
                                 **({"reasoning_effort": effort} if effort else {}))
            fwd = {}
            if "tools" in kwargs:
                fwd["tools"] = kwargs["tools"]
            return _record_chat_result(
                oai_llm._generate(messages, stop=stop, run_manager=run_manager, **fwd),
                self.model_name, source="openai")

        print("Using GIBD key for GIBDChat without streaming")
        oai_messages = [_lc_message_to_dict(m) for m in messages]

        q_id = _gibd_handshake(key)
        payload: dict = {
            "question_id": q_id,
            "service_name": GIBD_SERVICE,
            "model": self.model_name,
            "messages": oai_messages,
            "stream": False,
        }
        # Forward tools when bound via .bind_tools()
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if stop:
            payload["stop"] = stop
        effort = tool_call_reasoning_effort(self.model_name, "tools" in kwargs, self.reasoning_effort)
        if effort:
            payload["reasoning_effort"] = effort

        resp = _gibd_post_with_retry(
            f"{GIBD_API_URL}/api/openai/{key}",
            json=payload,
            label="GIBD inference",
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GIBD inference failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        ai_msg = _dict_to_ai_message(data["choices"][0]["message"])
        # The proxy passes OpenAI's usage block through untouched. Keep it on
        # the message too, so callers that inspect response_metadata (LangGraph
        # nodes, the debugger's retry path) see the same numbers the ledger got.
        usage = data.get("usage") or {}
        if usage:
            ai_msg.response_metadata = {
                **(getattr(ai_msg, "response_metadata", None) or {}),
                "token_usage": usage,
                "model_name": data.get("model") or self.model_name,
            }
        _record_llm_usage(data.get("model") or self.model_name, usage, source="gibd")
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # -- streaming -------------------------------------------------------------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        key = self.gibd_user_key or load_OpenAI_key()
        _ensure_gibd_only(key)
        # Local model (Ollama) — explicit "no-api" sentinel
        if key == "no-api":
            print("Using Local model for GIBDChat with streaming")
            from langchain_openai import ChatOpenAI
            oai_llm = ChatOpenAI(
                api_key="no-api",
                base_url=LOCAL_BASE_URL,
                model=LOCAL_MODEL_NAME,
                streaming=True,
                # extra_body=LOCAL_EXTRA_BODY,
            )
            fwd = {}
            if "tools" in kwargs:
                fwd["tools"] = kwargs["tools"]
            for chunk in oai_llm._stream(messages, stop=stop, run_manager=run_manager, **fwd):
                _record_stream_chunk(chunk, LOCAL_MODEL_NAME, source="local")
                yield chunk
            return
        # Route to direct OpenAI when the key isn't a GIBD key
        if not (key and key.startswith("gibd-")):
            print("Using openai key for GIBDChat with streaming")
            from langchain_openai import ChatOpenAI
            # stream_usage asks OpenAI for the final usage-only chunk. Without
            # it a streamed call reports nothing and silently reads as free.
            # Older langchain_openai has no such field and rejects it, so a
            # refusal costs the accounting, never the call.
            effort = tool_call_reasoning_effort(self.model_name, "tools" in kwargs, None)
            extra = {"reasoning_effort": effort} if effort else {}
            try:
                oai_llm = ChatOpenAI(api_key=key, model=self.model_name,
                                     streaming=True, stream_usage=True, **extra)
            except Exception:
                oai_llm = ChatOpenAI(api_key=key, model=self.model_name, streaming=True, **extra)
            fwd = {}
            if "tools" in kwargs:
                fwd["tools"] = kwargs["tools"]
            for chunk in oai_llm._stream(messages, stop=stop, run_manager=run_manager, **fwd):
                _record_stream_chunk(chunk, self.model_name, source="openai")
                yield chunk
            return
        print("Using GIBD key for GIBDChat with streaming")
        oai_messages = [_lc_message_to_dict(m) for m in messages]

        q_id = _gibd_handshake(key)
        payload: dict = {
            "question_id": q_id,
            "service_name": GIBD_SERVICE,
            "model": self.model_name,
            "messages": oai_messages,
            "stream": True,
            # OpenAI only sends usage on a streamed response when asked. The
            # proxy forwards this through; a backend that ignores it just
            # omits the trailing chunk, which is handled below.
            "stream_options": {"include_usage": True},
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
            print(f"[GIBDChat._stream] Sending {len(kwargs['tools'])} tools to GIBD proxy")
        else:
            print("[GIBDChat._stream] No tools in kwargs")
        if stop:
            payload["stop"] = stop
        effort = tool_call_reasoning_effort(self.model_name, "tools" in kwargs, self.reasoning_effort)
        if effort:
            payload["reasoning_effort"] = effort

        resp = _gibd_post_with_retry(
            f"{GIBD_API_URL}/api/openai/{key}",
            json=payload,
            stream=True,
            label="GIBD stream",
        )
        if resp.status_code == 400 and "stream_options" in payload:
            # A backend that does not know the field rejects the whole request.
            # Losing the token count is acceptable; losing the stream is not.
            print("[GIBDChat._stream] backend rejected stream_options; "
                  "retrying without usage reporting")
            payload.pop("stream_options", None)
            resp = _gibd_post_with_retry(
                f"{GIBD_API_URL}/api/openai/{key}",
                json=payload,
                stream=True,
                label="GIBD stream",
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GIBD stream failed ({resp.status_code}): {resp.text}"
            )

        chunk_id = None
        stream_usage = None
        stream_model = None

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data:"):
                decoded = decoded[5:].lstrip()
            if not decoded or decoded == "[DONE]":
                break
            try:
                chunk = _json.loads(decoded)
            except _json.JSONDecodeError:
                continue
            if "error" in chunk:
                raise RuntimeError(f"LLM API error: {chunk['error']}")
            # The final chunk of a stream_options request carries usage and an
            # empty choices list. Dropping it (as this loop used to) is what
            # made every streamed call report zero tokens.
            if chunk.get("usage"):
                stream_usage = chunk["usage"]
                stream_model = chunk.get("model") or stream_model
            choices = chunk.get("choices") or []
            if not choices:
                continue
            if chunk_id is None:
                chunk_id = chunk.get("id", "gibd-stream")
            delta = choices[0].get("delta", {})

            # Handle tool calls in delta
            tc_chunks = []
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    from langchain_core.messages import ToolCallChunk
                    tc_chunks.append(ToolCallChunk(
                        name=tc.get("function", {}).get("name"),
                        args=tc.get("function", {}).get("arguments", ""),
                        id=tc.get("id"),
                        index=tc.get("index", 0),
                    ))

            msg_chunk = AIMessageChunk(
                content=delta.get("content", "") or "",
                id=chunk_id,
                tool_call_chunks=tc_chunks if tc_chunks else [],
            )
            yield ChatGenerationChunk(message=msg_chunk)

        # Recorded after the loop so the whole streamed response counts as one
        # call, not one per chunk. A backend that ignored stream_options leaves
        # stream_usage as None; the ledger then counts the call and flags it as
        # missing usage rather than pretending it was free.
        _record_llm_usage(stream_model or self.model_name, stream_usage or {},
                          source="gibd-stream")


def extract_code(response_content):
    """
    Extract python code from the content within the LLM response
    """
    python_code = ""
    python_code_match = re.search(r"```(?:python)?(.*?)```", response_content, re.DOTALL)
    if python_code_match:
        python_code = python_code_match.group(1).strip()
        # print(python_code)
    return python_code








def check_cancelled(ctx):
    """Return True if the workflow has been cancelled by the user."""
    ev = ctx.get("_cancel_event")
    return ev.is_set() if ev else False


def yield_new_logs(ctx, last_log_count, step='general'):
    """
    Helper function to yield any new logs added to context since last check

    Args:
        ctx: Context dict with logs
        last_log_count: Number of logs before the operation
        step: Step identifier for the log type

    Yields:
        Log update dictionaries

    Returns:
        Current number of logs (for next call)
    """
    current_logs = ctx.get('logs', [])
    if len(current_logs) > last_log_count:
        new_logs = current_logs[last_log_count:]
        for log_msg in new_logs:
            yield {
                'type': 'log',
                'content': log_msg,
                'step': step
            }
    return len(current_logs)


def yield_status_update(content, step='general'):
    """
    Helper function to yield a status update message

    Args:
        content: Status message to display
        step: Step identifier for the update type

    Yields:
        Status update dictionary
    """
    yield {
        'type': 'llm_update',
        'content': content,
        'step': step
    }


# def create_research_plan_flowchart_interactive(structured_plan, output_path="research_plan_flowchart.html"):
#     """
#     Create FULLY INTERACTIVE flowchart with draggable nodes using vis.js.
#     Features: Drag nodes in any direction, zoom, pan
#
#     Args:
#         structured_plan: Dictionary with research plan structure
#         output_path: Path to save HTML file
#
#     Returns:
#         Path to the saved flowchart
#     """
#     # Build nodes and edges with manual positioning
#     nodes = []
#     edges = []
#     node_id = 0
#
#     # Title node (centered at top)
#     title = structured_plan.get('title', 'Research Plan')
#     nodes.append({
#         'id': node_id,
#         'label': title,
#         'x': 0,
#         'y': 0,
#         'color': {'background': '#ADD8E6', 'border': '#000'},
#         'font': {'size': 16, 'bold': True},
#         'shape': 'box',
#         'fixed': False  # Allow movement
#     })
#     title_id = node_id
#     node_id += 1
#
#     # Objectives and steps
#     objectives = structured_plan.get('objectives', [])
#     y_offset = 150
#
#     for obj_idx, obj in enumerate(objectives):
#         obj_title = f"Obj {obj['number']}\n{obj['title'][:40]}..."
#         obj_y = y_offset * (obj_idx + 1)
#
#         obj_node_id = node_id
#         nodes.append({
#             'id': obj_node_id,
#             'label': obj_title,
#             'x': 0,
#             'y': obj_y,
#             'color': {'background': '#90EE90', 'border': '#000'},
#             'shape': 'box',
#             'font': {'size': 12},
#             'fixed': False
#         })
#         edges.append({'from': title_id, 'to': obj_node_id, 'arrows': 'to'})
#         node_id += 1
#
#         # Steps (spread horizontally)
#         steps = obj.get('steps', [])
#         num_steps = len(steps)
#         step_spacing = 200
#         start_x = -(num_steps - 1) * step_spacing / 2
#
#         for i, step in enumerate(steps):
#             step_label = f"{chr(97+i)}. {step[:40]}..."
#             step_x = start_x + (i * step_spacing)
#             step_y = obj_y + 120
#
#             step_node_id = node_id
#             nodes.append({
#                 'id': step_node_id,
#                 'label': step_label,
#                 'x': step_x,
#                 'y': step_y,
#                 'color': {'background': '#FFFFE0', 'border': '#000'},
#                 'shape': 'box',
#                 'font': {'size': 10},
#                 'fixed': False
#             })
#             edges.append({'from': obj_node_id, 'to': step_node_id, 'arrows': 'to'})
#             node_id += 1
#
#     # Create HTML with vis.js
#     import json as json_module
#
#     nodes_json = json_module.dumps(nodes)
#     edges_json = json_module.dumps(edges)
#
#     html_content = f"""<!DOCTYPE html>
# <html>
# <head>
#     <title>Research Plan - Interactive Flowchart</title>
#     <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
#     <style>
#         #mynetwork {{
#             width: 100%;
#             height: 800px;
#             border: 1px solid lightgray;
#         }}
#         h1 {{ text-align: center; font-family: Arial; }}
#     </style>
# </head>
# <body>
#     <h1>{title}</h1>
#     <div id="mynetwork"></div>
#     <script>
#         var nodes = new vis.DataSet({nodes_json});
#         var edges = new vis.DataSet({edges_json});
#
#         var container = document.getElementById('mynetwork');
#         var data = {{ nodes: nodes, edges: edges }};
#
#         var options = {{
#             physics: {{ enabled: false }},  // No physics - manual positioning
#             interaction: {{
#                 dragNodes: true,     // Drag nodes freely
#                 dragView: true,      // Pan canvas
#                 zoomView: true       // Zoom with scroll
#             }},
#             edges: {{ smooth: {{ type: 'cubicBezier' }} }}
#         }};
#
#         var network = new vis.Network(container, data, options);
#     </script>
# </body>
# </html>"""
#
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(html_content)
#
#     print(f"✓ Interactive flowchart saved to: {output_path}")
#     print(f"  Features: Drag nodes, zoom, pan, physics simulation")
#
#     return output_path


def _peek_file(file_path):
    """Read a data file and return a dict with columns, dtypes, shape, and head.
    Returns None if the file can't be read or isn't a data file."""
    import pandas as pd
    if not os.path.exists(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df_peek = pd.read_csv(file_path, nrows=3)
            with open(file_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1
        elif ext in (".gpkg", ".shp", ".geojson"):
            import geopandas as _gpd
            df_peek = _gpd.read_file(file_path, rows=3)
            total_rows = len(_gpd.read_file(file_path, ignore_geometry=True))
        elif ext in (".xlsx", ".xls"):
            df_peek = pd.read_excel(file_path, nrows=3)
            total_rows = len(pd.read_excel(file_path))
        elif ext == ".zip":
            import geopandas as _gpd
            df_peek = _gpd.read_file(file_path, rows=3)
            total_rows = len(_gpd.read_file(file_path, ignore_geometry=True))
        elif ext in (".tif", ".tiff", ".nc", ".img", ".vrt"):
            try:
                import rasterio
                with rasterio.open(file_path) as src:
                    bands = src.count
                    height = src.height
                    width = src.width
                    crs = str(src.crs) if src.crs else "unknown"
                    band_names = [src.descriptions[i] or f"band_{i+1}" for i in range(bands)]
                return {
                    "file": os.path.splitext(os.path.basename(file_path))[0],
                    "path": file_path,
                    "shape": [height, width],
                    "columns": {b: "raster_band" for b in band_names},
                    "head": {"bands": bands, "crs": crs, "width": width, "height": height},
                }
            except Exception:
                # Fallback: return basic info without rasterio
                return {
                    "file": os.path.splitext(os.path.basename(file_path))[0],
                    "path": file_path,
                    "shape": [0, 0],
                    "columns": {},
                    "head": {},
                }
        else:
            return None
        # Drop the active geometry column from the preview — Shapely objects
        # serialize to full WKT and can balloon the LLM context to millions of
        # tokens (a single MULTIPOLYGON can be MBs). Awareness of the geometry
        # column is preserved via the `columns` dict and the dataset's geometry
        # block; only the raw coordinate values are withheld.
        try:
            preview_df = df_peek.head(2)
            geom_name = getattr(getattr(df_peek, "geometry", None), "name", None)
            if geom_name and geom_name in preview_df.columns:
                preview_df = preview_df.drop(columns=[geom_name], errors="ignore")
            head_dict = preview_df.to_dict(orient="list")
        except Exception:
            head_dict = df_peek.head(2).to_dict(orient="list")
        return {
            "file": os.path.splitext(os.path.basename(file_path))[0],
            "path": file_path,
            "shape": [total_rows, len(df_peek.columns)],
            "columns": {c: str(df_peek[c].dtype) for c in df_peek.columns},
            "head": head_dict,
        }
    except Exception:
        return None


def resolve_data_for_step(ctx):
    """Deterministically resolve the best data source for the current step.

    Priority cascade:
      1. Data registry (last step's output) — most recent data in the pipeline
      2. Data registry all (EDA cleaned > EDA outputs > loaded datasets)
      3. Original dataset (last resort)

    Returns a dict with verified info:
      {
        "path": str,           # File path to load
        "source": str,         # "data_registry" | "eda_cleaned" | "eda_output" | "loaded_dataset" | "original"
        "columns": {col: dtype},
        "shape": [rows, cols],
        "head": str,           # First 3 rows as string
        "from_step": int|None, # Step index that produced this file (if from registry)
        "other_available": list # Other available data sources (name, path, columns, source)
      }
    Or None if no data could be resolved.
    """
    artifacts = ctx.get("artifacts", {})
    all_sources = artifacts.get("data_registry_all", [])

    primary = None

    # Priority 1: Data registry — the most recent step output
    registry = artifacts.get("data_registry", {})
    if registry.get("path") and os.path.exists(registry["path"]):
        info = _peek_file(registry["path"])
        if info:
            info["source"] = "data_registry"
            info["from_step"] = registry.get("produced_by")
            primary = info

    # Priority 2: Walk through data_registry_all (EDA cleaned > EDA outputs > loaded datasets)
    if not primary and all_sources:
        for src in all_sources:
            src_path = src.get("path", "")
            if src_path and os.path.exists(src_path):
                info = _peek_file(src_path)
                if info:
                    info["source"] = src.get("source", "eda_output")
                    info["from_step"] = src.get("from_step")
                    primary = info
                    break

    # Priority 3: EDA cleaned data (fallback if registry_all not populated)
    if not primary:
        eda_path = artifacts.get("cleaned_data_path", "")
        if eda_path and os.path.exists(eda_path):
            info = _peek_file(eda_path)
            if info:
                info["source"] = "eda_cleaned"
                info["from_step"] = None
                primary = info

    # Priority 4: Original dataset (last resort)
    if not primary:
        data_summary = artifacts.get("data_summary_data", {})
        if isinstance(data_summary, dict):
            original_path = data_summary.get("dataset_overview", {}).get("data_path", "")
            if original_path and os.path.exists(original_path):
                info = _peek_file(original_path)
                if info:
                    info["source"] = "original"
                    info["from_step"] = None
                    primary = info

    if not primary:
        return None

    # Attach other available sources (excluding the primary) so the LLM knows alternatives exist
    primary_norm = os.path.normpath(primary["path"])
    other = []
    for src in all_sources:
        src_path = src.get("path", "")
        if src_path and os.path.normpath(src_path) != primary_norm:
            other.append({
                "file": src.get("file", os.path.basename(src_path)),
                "path": src_path,
                "source": src.get("source", "unknown"),
                "columns": list(src.get("columns", {}).keys()) if isinstance(src.get("columns"), dict) else src.get("columns", []),
                "shape": src.get("shape", "unknown"),
            })
    primary["other_available"] = other

    return primary


def update_data_registry(ctx, created_files, step_index=None):
    """Update the data registry after a step executes.

    Scans created_files for the most recent data file (CSV, GPKG, etc.)
    and records its path, columns, and shape in ctx["artifacts"]["data_registry"].
    Non-data files (plots, JSON, HTML) are skipped.

    Args:
        ctx: Context dictionary
        created_files: List of file paths created/modified by the step
        step_index: The step number that produced these files
    """
    DATA_EXTENSIONS = {".csv", ".gpkg", ".shp", ".geojson", ".xlsx", ".xls"}

    # Find the most recent data file (prefer last one — usually the final output)
    data_file = None
    for fpath in reversed(created_files):
        ext = os.path.splitext(fpath)[1].lower()
        if ext in DATA_EXTENSIONS and os.path.exists(fpath):
            data_file = fpath
            break

    if not data_file:
        return  # No data file produced — registry unchanged

    info = _peek_file(data_file)
    if not info:
        return


    new_entry = {
        "file": info.get("file", os.path.basename(data_file)),
        "path": data_file,
        "source": "step_output",
        "from_step": step_index,
        "columns": info["columns"],
        "shape": info["shape"],
    }

    # Append to data_registry (the main list used by tools),
    # replacing any existing entry for the same path.
    data_registry = ctx.setdefault("artifacts", {}).setdefault("data_registry", [])
    if not isinstance(data_registry, list):
        # Fix: data_registry was overwritten as a dict by a prior bug — reset to list
        data_registry = []
        ctx["artifacts"]["data_registry"] = data_registry
    norm = os.path.normpath(data_file)
    ctx["artifacts"]["data_registry"] = [
        s for s in data_registry if os.path.normpath(s.get("path", "")) != norm
    ] + [new_entry]




def build_plan_dag(plan, save_path=None):
    """Build a NetworkX DiGraph from a research plan following the data → operation → data pattern.

    Args:
        plan: Research plan dict with 'objectives' list
        save_path: Optional path to save as GraphML file

    Returns:
        nx.DiGraph
    """
    import networkx as nx

    G = nx.DiGraph()
    objectives = plan.get("objectives", [])

    for obj in objectives:
        obj_num = obj["number"]
        obj_title = obj.get("title", "")

        # Add source data nodes from data_requirements
        for dr in obj.get("data_requirements", []):
            source_id = dr.get("source_id", "")
            if source_id and source_id not in G:
                G.add_node(source_id,
                           node_type="data",
                           description=dr.get("description", ""),
                           objective=0)  # shared across objectives

        # Add operation and output data nodes for each step
        for step in obj.get("steps", []):
            step_num = step["step_number"]
            op_node = f"obj{obj_num}_step{step_num}"
            out_node = step["output_node"]

            # Operation node
            G.add_node(op_node,
                       node_type="operation",
                       operation=step.get("operation", ""),
                       operation_type=step.get("operation_type", ""),
                       description=step.get("description", ""),
                       expected_output=step.get("expected_output", ""),
                       objective=obj_num,
                       objective_title=obj_title,
                       step_number=step_num)

            # Edges: input_nodes → operation
            for inp in step.get("input_nodes", []):
                if inp not in G:
                    G.add_node(inp, node_type="data", description="", objective=obj_num)
                G.add_edge(inp, op_node)

            # Output data node
            G.add_node(out_node,
                       node_type="data",
                       description=step.get("expected_output") or "",
                       objective=obj_num)

            # Edge: operation → output
            G.add_edge(op_node, out_node)

    if save_path:
        # Replace None values with empty strings for GraphML compatibility
        for _, attrs in G.nodes(data=True):
            for k, v in attrs.items():
                if v is None:
                    attrs[k] = ""
        nx.write_graphml(G, save_path)
        print(f"DAG saved to {save_path}")

    return G


def show_plan_overview(plan):
    """Render a top-level overview DAG: Research Question → Objectives → Steps (chained).

    Objectives branch independently from the RQ node. Steps within each objective
    are chained sequentially.

    Returns:
        pyvis Network object — call nt.show('filename.html') to display
    """
    from pyvis.network import Network

    nt = Network(height="700px", width="100%", directed=True, notebook=True)
    nt.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)

    colors_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    objectives = plan.get("objectives", [])
    rq = plan.get("research_question", "Research Question")

    # Root node: Research Question
    nt.add_node("RQ", label=rq[:60], title=f"<b>Research Question</b><br>{rq}",
                color="#2C3E50", font={"color": "white"}, shape="box", size=35)

    for obj in objectives:
        obj_num = obj["number"]
        obj_id = f"obj_{obj_num}"
        obj_title = obj.get("title", "")
        hypothesis = obj.get("hypothesis", "")
        color = colors_list[(obj_num - 1) % len(colors_list)]

        # Objective node
        nt.add_node(obj_id,
                    label=f"Obj {obj_num}: {obj_title[:45]}",
                    title=f"<b>Objective {obj_num}</b><br>{obj_title}<br><br><b>Hypothesis:</b> {hypothesis}",
                    color=color, shape="ellipse", size=30)
        nt.add_edge("RQ", obj_id, width=2)

        # Step nodes chained sequentially
        steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))
        prev_id = obj_id
        for step in steps:
            step_num = step["step_number"]
            step_id = f"obj{obj_num}_s{step_num}"
            operation = step.get("operation", "")
            desc = step.get("description", "")
            expected = step.get("expected_output", "")

            tooltip = f"<b>Step {obj_num}.{step_num}</b><br><b>Op:</b> {operation}<br><br>{desc}"
            if expected:
                tooltip += f"<br><br><b>Expected:</b> {expected}"

            nt.add_node(step_id,
                        label=f"S{obj_num}.{step_num}: {operation[:30]}",
                        title=tooltip,
                        color=color, shape="box", size=20,
                        borderWidth=1)
            nt.add_edge(prev_id, step_id, color=color)
            prev_id = step_id

    return nt


def show_plan_overview_interactive(plan, output_path="Outputs/research_plan_overview.html"):
    """Render a fixed-position interactive overview matching show_plan_overview_static style.

    Same visual style as the static version (rounded boxes, same colors, same layout)
    but rendered as interactive HTML. Clicking a node displays its details in a panel.

    Args:
        plan: Research plan dict
        output_path: Path to save HTML file

    Returns:
        Path to the saved HTML file
    """
    import colorsys
    import json
    import os

    objectives = plan.get("objectives", [])
    rq = plan.get("research_question", "Research Question")
    plan_title = plan.get("title", "Research Plan Overview")
    research_intent = plan.get("research_intent", "")
    colors_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    def lighter_shade(hex_color, amount=0.25):
        r, g, b = int(hex_color[1:3], 16) / 255, int(hex_color[3:5], 16) / 255, int(hex_color[5:7], 16) / 255
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        lr, lg, lb = colorsys.hls_to_rgb(h, min(l + amount, 0.92), s)
        return f"#{int(lr*255):02x}{int(lg*255):02x}{int(lb*255):02x}"

    # Build node data for JS
    nodes = []
    edges = []
    details = {}

    # Layout parameters (in SVG coordinates)
    col_spacing = 280 #Horizontal spacing between objective columns
    n_obj = len(objectives)
    total_width = (n_obj - 1) * col_spacing
    start_x = 500 - total_width // 2  # center in viewbox
    rq_y = 50
    obj_y = 180 #objective row Y position
    step_start_y = 310 # first step row Y position
    row_spacing = 100 #vertical distance between step nodes

    max_steps = max((len(obj.get("steps", [])) for obj in objectives), default=1)
    svg_height = step_start_y + max_steps * row_spacing + 80
    svg_width = max(1000, total_width + 400)

    # RQ node
    rq_width = min(svg_width - 100, 600)
    nodes.append({
        "id": "RQ", "x": svg_width // 2, "y": rq_y,
        "w": rq_width, "h": 60,
        "label": rq[:80], "color": "#2C3E50",
        "text_color": "white", "font_size": 13
    })
    details["RQ"] = f"<h3>Research Question</h3><p>{rq}</p>"
    if research_intent:
        details["RQ"] += f"<p><b>Intent:</b> {research_intent}</p>"

    for i, obj in enumerate(objectives):
        obj_num = obj["number"]
        color = colors_list[(obj_num - 1) % len(colors_list)]
        x = start_x + i * col_spacing
        obj_id = f"obj_{obj_num}"
        obj_title = obj.get("title", "")
        hypothesis = obj.get("hypothesis", "")

        nodes.append({
            "id": obj_id, "x": x, "y": obj_y,
            "w": 240, "h": 55,
            "label": f"Obj {obj_num}: {obj_title[:38]}", "color": color,
            "text_color": "white", "font_size": 11
        })
        edges.append({"from": "RQ", "to": obj_id, "color": "#555"})

        # Build detail HTML for objective
        dr_html = ""
        for dr in obj.get("data_requirements", []):
            dr_html += f"<li><b>{dr.get('source_id', '')}</b> (suitability: {dr.get('suitability_score', '')})"
            prep = dr.get("data_preprocessing") or {}
            if prep.get("required"):
                dr_html += f" — preprocessing: {prep.get('method', '')}"
            dr_html += "</li>"

        details[obj_id] = (
            f"<h3>Objective {obj_num}</h3>"
            f"<p><b>Title:</b> {obj_title}</p>"
            f"<p><b>Hypothesis:</b> {hypothesis}</p>"
            f"<p><b>Data Requirements:</b></p><ul>{dr_html}</ul>"
        )

        # Steps
        steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))
        prev_id = obj_id
        for j, step in enumerate(steps):
            step_num = step["step_number"]
            step_id = f"obj{obj_num}_s{step_num}"
            operation = step.get("operation", "")
            desc = step.get("description", "")
            expected = step.get("expected_output") or ""
            op_type = step.get("operation_type", "")
            input_nodes_list = step.get("input_nodes", [])
            output_node = step.get("output_node", "")

            light_color = lighter_shade(color)
            sy = step_start_y + j * row_spacing
            #Step shape
            nodes.append({
                "id": step_id, "x": x, "y": sy,
                "w": 220, "h": 50, # width and height of step box
                "label": f"S{obj_num}.{step_num}: {operation[:26]}", "color": light_color,
                "text_color": "#222", "font_size": 10,
                "shape": "parallelogram"
            })
            edges.append({"from": prev_id, "to": step_id, "color": color})
            prev_id = step_id

            step_detail = (
                f"<h3>Step {obj_num}.{step_num}</h3>"
                f"<p><b>Operation:</b> {operation}</p>"
                f"<p><b>Type:</b> {op_type}</p>"
                f"<p><b>Input:</b> {', '.join(input_nodes_list)}</p>"
                f"<p><b>Output Node:</b> {output_node}</p>"
                f"<p><b>Description:</b> {desc}</p>"
            )
            if expected:
                step_detail += f"<p><b>Expected Output:</b> {expected}</p>"
            details[step_id] = step_detail

    # Build node lookup for edge routing
    node_lookup = {n["id"]: n for n in nodes}

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{plan_title}</title>
<style>
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; display: flex; flex-direction: column; height: 100vh; }}
  h1 {{ text-align: center; padding: 12px 0 4px 0; margin: 0; font-size: 18px; color: #2C3E50; }}
  .container {{ display: flex; flex: 1; overflow: hidden; }}
  .graph-panel {{ flex: 1; overflow: auto; }}
  svg {{ display: block; margin: 0 auto; }}
  .node-box {{ cursor: pointer; transition: filter 0.15s; }}
  .node-box:hover {{ filter: brightness(1.12); }}
  .detail-panel {{
    width: 320px; min-width: 280px; background: white; border-left: 2px solid #ddd;
    padding: 20px; overflow-y: auto; box-shadow: -2px 0 8px rgba(0,0,0,0.08);
  }}
  .detail-panel h3 {{ margin-top: 0; color: #2C3E50; border-bottom: 2px solid #4C72B0; padding-bottom: 8px; }}
  .detail-panel p {{ line-height: 1.6; color: #333; font-size: 13px; }}
  .detail-panel ul {{ padding-left: 18px; }}
  .detail-panel li {{ margin-bottom: 6px; font-size: 13px; }}
  .detail-placeholder {{ color: #999; font-style: italic; margin-top: 40px; text-align: center; }}
  marker {{ overflow: visible; }}
</style>
</head>
<body>
<h1>{plan_title}</h1>
<div class="container">
<div class="graph-panel">
<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
"""

    # Draw edges first (behind nodes)
    for edge in edges:
        fn = node_lookup[edge["from"]]
        tn = node_lookup[edge["to"]]
        x1 = fn["x"]
        y1 = fn["y"] + fn["h"] // 2
        x2 = tn["x"]
        y2 = tn["y"] - tn["h"] // 2
        ec = edge["color"]
        html += f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ec}" stroke-width="2" marker-end="url(#arrowhead)"/>\n'

    # Draw nodes
    for n in nodes:
        nx_ = n["x"] - n["w"] // 2
        ny = n["y"] - n["h"] // 2
        nid = n["id"]
        html += f'    <g class="node-box" onclick="showDetail(\'{nid}\')">\n'
        slant = 15
        if n.get("shape") == "parallelogram":
            pts = (f"{nx_ + slant},{ny} "
                   f"{nx_ + n['w']},{ny} "
                   f"{nx_ + n['w'] - slant},{ny + n['h']} "
                   f"{nx_},{ny + n['h']}")
            html += f'      <polygon points="{pts}" fill="{n["color"]}" stroke="#333" stroke-width="1.5"/>\n'
        else:
            html += f'      <rect x="{nx_}" y="{ny}" width="{n["w"]}" height="{n["h"]}" rx="10" ry="10" fill="{n["color"]}" stroke="#333" stroke-width="1.5"/>\n'

        # Word-wrap the label into multiple lines
        label = n["label"]
        fs = n["font_size"]
        max_chars = n["w"] // (fs * 0.55)  # approximate chars per line
        lines = []
        words = label.split()
        current = ""
        for word in words:
            if current and len(current) + len(word) + 1 > max_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)

        line_height = fs + 3
        total_text_h = len(lines) * line_height
        text_start_y = n["y"] - total_text_h / 2 + fs * 0.4

        for li, line_text in enumerate(lines):
            ty = text_start_y + li * line_height
            # Escape HTML entities in label
            safe_text = line_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f'      <text x="{n["x"]}" y="{ty}" text-anchor="middle" fill="{n["text_color"]}" font-size="{fs}" font-weight="bold" font-family="Segoe UI, Arial, sans-serif">{safe_text}</text>\n'

        html += f'    </g>\n'

    html += """  </svg>
</div>
<div class="detail-panel" id="detailPanel">
  <p class="detail-placeholder">Click on any node to see its details.</p>
</div>
</div>
<script>
const details = """ + json.dumps(details) + """;

function showDetail(nodeId) {
  const panel = document.getElementById('detailPanel');
  if (details[nodeId]) {
    panel.innerHTML = details[nodeId];
  } else {
    panel.innerHTML = '<p class="detail-placeholder">No details available.</p>';
  }
  // Highlight selected node
  document.querySelectorAll('.node-box rect').forEach(r => r.setAttribute('stroke-width', '1.5'));
  const clicked = event.currentTarget.querySelector('rect');
  if (clicked) {
    clicked.setAttribute('stroke-width', '3.5');
    clicked.setAttribute('stroke', '#FFD700');
  }
}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Interactive overview saved to: {output_path}")
    return output_path


def show_plan_overview_popup(plan, output_path="Outputs/research_plan_overview_popup.html"):
    """Render an interactive overview with popup detail boxes instead of a side panel.

    Same SVG layout as show_plan_overview_interactive (rounded boxes, same colors,
    same node positions) but clicking a node opens a floating popup near the node
    instead of showing details in a permanent side panel.

    Args:
        plan: Research plan dict
        output_path: Path to save HTML file

    Returns:
        Path to the saved HTML file
    """
    import colorsys
    import json
    import os

    objectives = plan.get("objectives", [])
    rq = plan.get("research_question", "Research Question")
    plan_title = plan.get("title", "Research Plan Overview")
    research_intent = plan.get("research_intent", "")
    colors_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    def lighter_shade(hex_color, amount=0.25):
        r, g, b = int(hex_color[1:3], 16) / 255, int(hex_color[3:5], 16) / 255, int(hex_color[5:7], 16) / 255
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        lr, lg, lb = colorsys.hls_to_rgb(h, min(l + amount, 0.92), s)
        return f"#{int(lr*255):02x}{int(lg*255):02x}{int(lb*255):02x}"

    def _step_workflow_svg(input_nodes_list, operation, output_node, obj_color):
        """Generate an inline SVG showing: input_node(s) -> [operation] -> output_node."""
        import html as _html

        inputs = input_nodes_list or ["(none)"]
        op_label = operation or "(operation)"
        out_label = output_node or "(none)"

        box_h = 30
        box_rx = 6
        row_gap = 8
        font_size = 10
        arrow_len = 30

        def _box_w(text, minimum=80, maximum=160):
            return max(minimum, min(maximum, len(text) * 7 + 20))

        in_w = max(_box_w(t) for t in inputs)
        op_w = _box_w(op_label, minimum=100, maximum=180)
        out_w = _box_w(out_label)

        n_inputs = len(inputs)
        input_block_h = n_inputs * box_h + (n_inputs - 1) * row_gap
        svg_h = max(input_block_h, box_h) + 20
        svg_w = in_w + arrow_len + op_w + arrow_len + out_w + 20

        mid_y = svg_h / 2
        op_top = mid_y - box_h / 2
        out_top = mid_y - box_h / 2
        input_start_y = mid_y - input_block_h / 2

        in_x = 10
        op_x = in_x + in_w + arrow_len
        out_x = op_x + op_w + arrow_len

        lighter_op = lighter_shade(obj_color, 0.15)
        lighter_in = lighter_shade(obj_color, 0.35)

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}"'
            f' style="display:block;margin:10px auto;">\n'
            f'  <defs><marker id="wfah" markerWidth="8" markerHeight="6" refX="8" refY="3"'
            f' orient="auto"><polygon points="0 0,8 3,0 6" fill="#666"/></marker></defs>\n'
        )

        # Input nodes
        for idx, inp in enumerate(inputs):
            iy = input_start_y + idx * (box_h + row_gap)
            safe = _html.escape(inp[:20])
            svg += (
                f'  <rect x="{in_x}" y="{iy}" width="{in_w}" height="{box_h}"'
                f' rx="{box_rx}" fill="{lighter_in}" stroke="#888" stroke-width="1"/>\n'
                f'  <text x="{in_x + in_w/2}" y="{iy + box_h/2 + 4}" text-anchor="middle"'
                f' font-size="{font_size}" font-family="Segoe UI,Arial,sans-serif"'
                f' fill="#333">{safe}</text>\n'
            )
            ax1 = in_x + in_w
            ay = iy + box_h / 2
            svg += (
                f'  <line x1="{ax1}" y1="{ay}" x2="{op_x}" y2="{mid_y}"'
                f' stroke="#666" stroke-width="1.5" marker-end="url(#wfah)"/>\n'
            )

        # Operation node
        safe_op = _html.escape(op_label[:24])
        svg += (
            f'  <rect x="{op_x}" y="{op_top}" width="{op_w}" height="{box_h}"'
            f' rx="{box_rx}" fill="{lighter_op}" stroke="{obj_color}" stroke-width="2"/>\n'
            f'  <text x="{op_x + op_w/2}" y="{op_top + box_h/2 + 4}" text-anchor="middle"'
            f' font-size="{font_size}" font-weight="bold"'
            f' font-family="Segoe UI,Arial,sans-serif" fill="#222">{safe_op}</text>\n'
        )

        # Arrow: operation -> output
        svg += (
            f'  <line x1="{op_x + op_w}" y1="{mid_y}" x2="{out_x}" y2="{mid_y}"'
            f' stroke="#666" stroke-width="1.5" marker-end="url(#wfah)"/>\n'
        )

        # Output node
        safe_out = _html.escape(out_label[:22])
        svg += (
            f'  <rect x="{out_x}" y="{out_top}" width="{out_w}" height="{box_h}"'
            f' rx="{box_rx}" fill="#d4edda" stroke="#28a745" stroke-width="1.5"/>\n'
            f'  <text x="{out_x + out_w/2}" y="{out_top + box_h/2 + 4}" text-anchor="middle"'
            f' font-size="{font_size}" font-family="Segoe UI,Arial,sans-serif"'
            f' fill="#155724">{safe_out}</text>\n'
        )

        svg += '</svg>'
        return svg

    # Build node data for JS
    nodes = []
    edges = []
    details = {}

    # Layout parameters (in SVG coordinates)
    col_spacing = 280
    n_obj = len(objectives)
    total_width = (n_obj - 1) * col_spacing
    start_x = 500 - total_width // 2
    rq_y = 50
    obj_y = 180
    step_start_y = 310
    row_spacing = 100

    max_steps = max((len(obj.get("steps", [])) for obj in objectives), default=1)
    svg_height = step_start_y + max_steps * row_spacing + 80
    svg_width = max(1000, total_width + 400)

    # RQ node
    rq_width = min(svg_width - 100, 600)
    nodes.append({
        "id": "RQ", "x": svg_width // 2, "y": rq_y,
        "w": rq_width, "h": 60,
        "label": rq[:80], "color": "#2C3E50",
        "text_color": "white", "font_size": 13
    })
    details["RQ"] = f"<h3>Research Question</h3><p>{rq}</p>"
    if research_intent:
        details["RQ"] += f"<p><b>Intent:</b> {research_intent}</p>"

    for i, obj in enumerate(objectives):
        obj_num = obj["number"]
        color = colors_list[(obj_num - 1) % len(colors_list)]
        x = start_x + i * col_spacing
        obj_id = f"obj_{obj_num}"
        obj_title = obj.get("title", "")
        hypothesis = obj.get("hypothesis", "")

        nodes.append({
            "id": obj_id, "x": x, "y": obj_y,
            "w": 240, "h": 55,
            "label": f"Obj {obj_num}: {obj_title[:38]}", "color": color,
            "text_color": "white", "font_size": 11
        })
        edges.append({"from": "RQ", "to": obj_id, "color": "#555"})

        dr_html = ""
        for dr in obj.get("data_requirements", []):
            dr_html += f"<li><b>{dr.get('source_id', '')}</b> (suitability: {dr.get('suitability_score', '')})"
            prep = dr.get("data_preprocessing") or {}
            if prep.get("required"):
                dr_html += f" — preprocessing: {prep.get('method', '')}"
            dr_html += "</li>"

        details[obj_id] = (
            f"<h3>Objective {obj_num}</h3>"
            f"<p><b>Title:</b> {obj_title}</p>"
            f"<p><b>Hypothesis:</b> {hypothesis}</p>"
            f"<p><b>Data Requirements:</b></p><ul>{dr_html}</ul>"
        )

        steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))
        prev_id = obj_id
        for j, step in enumerate(steps):
            step_num = step["step_number"]
            step_id = f"obj{obj_num}_s{step_num}"
            operation = step.get("operation", "")
            desc = step.get("description", "")
            expected = step.get("expected_output") or ""
            op_type = step.get("operation_type", "")
            input_nodes_list = step.get("input_nodes", [])
            output_node = step.get("output_node", "")

            light_color = lighter_shade(color)
            sy = step_start_y + j * row_spacing
            nodes.append({
                "id": step_id, "x": x, "y": sy,
                "w": 220, "h": 50,
                "label": f"S{obj_num}.{step_num}: {operation[:26]}", "color": light_color,
                "text_color": "#222", "font_size": 10,
                "shape": "parallelogram"
            })
            edges.append({"from": prev_id, "to": step_id, "color": color})
            prev_id = step_id

            # Build inline SVG workflow diagram for this step
            workflow_svg = _step_workflow_svg(input_nodes_list, operation, output_node, color)

            step_detail = (
                f"<h3>Step {obj_num}.{step_num}</h3>"
                f"<p style='color:#666;font-size:11px;margin-bottom:4px;'>Workflow</p>"
                f"{workflow_svg}"
                f"<hr style='border:none;border-top:1px solid #e9ecef;margin:12px 0;'>"
                f"<p><b>Operation:</b> {operation}</p>"
                f"<p><b>Type:</b> {op_type}</p>"
                f"<p><b>Input:</b> {', '.join(input_nodes_list)}</p>"
                f"<p><b>Output Node:</b> {output_node}</p>"
                f"<p><b>Description:</b> {desc}</p>"
            )
            if expected:
                step_detail += f"<p><b>Expected Output:</b> {expected}</p>"
            details[step_id] = step_detail

    node_lookup = {n["id"]: n for n in nodes}

    # Generate HTML with popup instead of side panel
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{plan_title}</title>
<style>
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; display: flex; flex-direction: column; height: 100vh; }}
  h1 {{ text-align: center; padding: 12px 0 4px 0; margin: 0; font-size: 18px; color: #2C3E50; }}
  .graph-panel {{ flex: 1; overflow: hidden; position: relative; cursor: grab; }}
  svg {{ display: block; width: 100%; height: 100%; }}
  .node-box {{ cursor: pointer; transition: filter 0.15s; }}
  .node-box:hover {{ filter: brightness(1.12); }}
  marker {{ overflow: visible; }}

  /* Zoom controls toolbar */
  .zoom-toolbar {{
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    z-index: 50;
    background: white;
    border-radius: 8px;
    padding: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    border: 1px solid #ddd;
  }}
  .zoom-btn {{
    width: 32px;
    height: 32px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: #fff;
    cursor: pointer;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #444;
    transition: all 0.15s;
    padding: 0;
  }}
  .zoom-btn:hover {{ background: #e9ecef; color: #111; }}
  .zoom-btn:active {{ background: #dee2e6; }}
  .zoom-level {{
    font-size: 10px;
    text-align: center;
    color: #888;
    padding: 2px 0;
    user-select: none;
  }}
  .zoom-divider {{
    height: 1px;
    background: #e0e0e0;
    margin: 2px 0;
  }}

  /* Popup overlay */
  .popup-overlay {{
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.35);
    z-index: 100;
    justify-content: center;
    align-items: center;
  }}
  .popup-overlay.visible {{ display: flex; }}

  /* Popup box */
  .popup-box {{
    background: white;
    border-radius: 10px;
    padding: 24px 28px;
    max-width: 580px;
    width: 92%;
    max-height: 75vh;
    overflow-y: auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    position: relative;
    animation: popIn 0.18s ease-out;
  }}
  @keyframes popIn {{
    from {{ transform: scale(0.92); opacity: 0; }}
    to {{ transform: scale(1); opacity: 1; }}
  }}
  .popup-box h3 {{ margin-top: 0; color: #2C3E50; border-bottom: 2px solid #4C72B0; padding-bottom: 8px; }}
  .popup-box p {{ line-height: 1.6; color: #333; font-size: 13px; }}
  .popup-box ul {{ padding-left: 18px; }}
  .popup-box li {{ margin-bottom: 6px; font-size: 13px; }}
  .popup-box svg {{ max-width: 100%; height: auto; }}
  .popup-close {{
    position: absolute;
    top: 10px; right: 14px;
    background: none; border: none;
    font-size: 20px; color: #999;
    cursor: pointer; line-height: 1;
    padding: 4px 8px; border-radius: 4px;
  }}
  .popup-close:hover {{ background: #f0f0f0; color: #333; }}
</style>
</head>
<body>
<h1>{plan_title}</h1>
<div class="graph-panel" id="graphPanel">
<!-- Zoom controls -->
<div class="zoom-toolbar">
  <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
  <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
  <div class="zoom-divider"></div>
  <span class="zoom-level" id="zoomLevel">100%</span>
  <div class="zoom-divider"></div>
  <button class="zoom-btn" onclick="zoomReset()" title="Reset view">&#8634;</button>
  <button class="zoom-btn" onclick="zoomFit()" title="Fit to view">&#11036;</button>
</div>
<svg id="flowchartSvg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  <g id="zoomGroup">
"""

    # Draw edges
    for edge in edges:
        fn = node_lookup[edge["from"]]
        tn = node_lookup[edge["to"]]
        x1 = fn["x"]
        y1 = fn["y"] + fn["h"] // 2
        x2 = tn["x"]
        y2 = tn["y"] - tn["h"] // 2
        ec = edge["color"]
        html += f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ec}" stroke-width="2" marker-end="url(#arrowhead)"/>\n'

    # Draw nodes
    for n in nodes:
        nx_ = n["x"] - n["w"] // 2
        ny = n["y"] - n["h"] // 2
        nid = n["id"]
        html += f'    <g class="node-box" onclick="showPopup(\'{nid}\')">\n'
        slant = 15
        if n.get("shape") == "parallelogram":
            pts = (f"{nx_ + slant},{ny} "
                   f"{nx_ + n['w']},{ny} "
                   f"{nx_ + n['w'] - slant},{ny + n['h']} "
                   f"{nx_},{ny + n['h']}")
            html += f'      <polygon points="{pts}" fill="{n["color"]}" stroke="#333" stroke-width="1.5"/>\n'
        else:
            html += f'      <rect x="{nx_}" y="{ny}" width="{n["w"]}" height="{n["h"]}" rx="10" ry="10" fill="{n["color"]}" stroke="#333" stroke-width="1.5"/>\n'

        label = n["label"]
        fs = n["font_size"]
        max_chars = n["w"] // (fs * 0.55)
        lines = []
        words = label.split()
        current = ""
        for word in words:
            if current and len(current) + len(word) + 1 > max_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)

        line_height = fs + 3
        total_text_h = len(lines) * line_height
        text_start_y = n["y"] - total_text_h / 2 + fs * 0.4

        for li, line_text in enumerate(lines):
            ty = text_start_y + li * line_height
            safe_text = line_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f'      <text x="{n["x"]}" y="{ty}" text-anchor="middle" fill="{n["text_color"]}" font-size="{fs}" font-weight="bold" font-family="Segoe UI, Arial, sans-serif">{safe_text}</text>\n'

        html += f'    </g>\n'

    html += """  </g>
</svg>
</div>

<!-- Popup overlay -->
<div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
  <div class="popup-box" id="popupBox" onclick="event.stopPropagation()">
    <button class="popup-close" onclick="closePopup()">&times;</button>
    <div id="popupContent"></div>
  </div>
</div>

<script>
const details = """ + json.dumps(details) + """;

// ─── Zoom & Pan state ───
let scale = 1;
let panX = 0, panY = 0;
let isPanning = false;
let _wasDragged = false;
let panStartX = 0, panStartY = 0;
let _mdX = 0, _mdY = 0;
const PAN_THRESHOLD = 4;
const MIN_SCALE = 0.2;
const MAX_SCALE = 5;
const ZOOM_STEP = 0.15;
const svg = document.getElementById('flowchartSvg');
const zoomGroup = document.getElementById('zoomGroup');
const graphPanel = document.getElementById('graphPanel');

function applyTransform() {
  zoomGroup.setAttribute('transform', `translate(${panX}, ${panY}) scale(${scale})`);
  document.getElementById('zoomLevel').textContent = Math.round(scale * 100) + '%';
}

function zoomIn() {
  scale = Math.min(MAX_SCALE, scale + ZOOM_STEP);
  applyTransform();
}

function zoomOut() {
  scale = Math.max(MIN_SCALE, scale - ZOOM_STEP);
  applyTransform();
}

function zoomReset() {
  scale = 1;
  panX = 0;
  panY = 0;
  applyTransform();
}

function zoomFit() {
  const bbox = zoomGroup.getBBox();
  const vb = svg.viewBox.baseVal;
  const pad = 30;
  const scaleX = (vb.width - pad * 2) / bbox.width;
  const scaleY = (vb.height - pad * 2) / bbox.height;
  scale = Math.min(scaleX, scaleY, MAX_SCALE);
  panX = (vb.width / scale - bbox.width) / 2 - bbox.x + pad / scale;
  panY = (vb.height / scale - bbox.height) / 2 - bbox.y + pad / scale;
  applyTransform();
}

// Mouse wheel zoom (centered on cursor)
graphPanel.addEventListener('wheel', function(e) {
  e.preventDefault();
  const rect = svg.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const vb = svg.viewBox.baseVal;
  const svgX = (mouseX / rect.width) * vb.width;
  const svgY = (mouseY / rect.height) * vb.height;

  const oldScale = scale;
  if (e.deltaY < 0) {
    scale = Math.min(MAX_SCALE, scale + ZOOM_STEP);
  } else {
    scale = Math.max(MIN_SCALE, scale - ZOOM_STEP);
  }

  // Adjust pan so zoom centers on cursor
  panX = svgX / scale - svgX / oldScale + panX;
  panY = svgY / scale - svgY / oldScale + panY;
  applyTransform();
}, { passive: false });

// Pan with left-click drag (or middle mouse)
graphPanel.addEventListener('mousedown', function(e) {
  if (e.button === 0 || e.button === 1) {
    e.preventDefault();
    isPanning = true;
    _wasDragged = false;
    panStartX = e.clientX;
    panStartY = e.clientY;
    _mdX = e.clientX;
    _mdY = e.clientY;
    graphPanel.style.cursor = 'grabbing';
  }
});

window.addEventListener('mousemove', function(e) {
  if (!isPanning) return;
  const dist = Math.abs(e.clientX - _mdX) + Math.abs(e.clientY - _mdY);
  if (dist > PAN_THRESHOLD) _wasDragged = true;
  const vb = svg.viewBox.baseVal;
  const rect = svg.getBoundingClientRect();
  const dx = (e.clientX - panStartX) * (vb.width / rect.width) / scale;
  const dy = (e.clientY - panStartY) * (vb.height / rect.height) / scale;
  panX += dx;
  panY += dy;
  panStartX = e.clientX;
  panStartY = e.clientY;
  applyTransform();
});

window.addEventListener('mouseup', function() {
  if (isPanning) {
    isPanning = false;
    graphPanel.style.cursor = '';
  }
});

// Suppress node clicks after a drag
graphPanel.addEventListener('click', function(e) {
  if (_wasDragged) {
    e.stopPropagation();
    _wasDragged = false;
  }
}, true);

// ─── Node popup ───
function showPopup(nodeId) {
  const content = document.getElementById('popupContent');
  const overlay = document.getElementById('popupOverlay');
  if (details[nodeId]) {
    content.innerHTML = details[nodeId];
  } else {
    content.innerHTML = '<p style="color:#999; font-style:italic;">No details available.</p>';
  }
  overlay.classList.add('visible');

  // Highlight selected node
  document.querySelectorAll('.node-box rect, .node-box polygon').forEach(el => {
    el.setAttribute('stroke-width', '1.5');
    el.setAttribute('stroke', '#333');
  });
  const clicked = event.currentTarget.querySelector('rect, polygon');
  if (clicked) {
    clicked.setAttribute('stroke-width', '3.5');
    clicked.setAttribute('stroke', '#FFD700');
  }
}

function closePopup(e) {
  // If called from overlay click, only close if clicking the overlay itself
  if (e && e.target !== document.getElementById('popupOverlay')) return;
  document.getElementById('popupOverlay').classList.remove('visible');
  // Reset highlights
  document.querySelectorAll('.node-box rect, .node-box polygon').forEach(el => {
    el.setAttribute('stroke-width', '1.5');
    el.setAttribute('stroke', '#333');
  });
}

// Close on Escape key
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closePopup();
});

// Fit to view on initial load
window.addEventListener('DOMContentLoaded', function() {
  setTimeout(zoomFit, 100);
});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Interactive popup overview saved to: {output_path}")
    return output_path

# def show_geoprocessing_workflow_plan_overview_popup(plan, output_path="Outputs/geoprocessing_workflow_plan_overview_popup.html"):
#     """Render an ArcGIS Model Builder-style SVG workflow for each objective.
#
#     Layout per objective (left-to-right chain):
#         [Input Data (blue oval)] -> [Operation (yellow rect)] -> [Output Data (green oval)] -> ...
#
#     Output nodes from earlier steps are reused when they appear as inputs to later
#     steps (arrows drawn from the existing green oval), just like Model Builder.
#
#     Clicking any node opens a popup with full details.  Supports zoom & pan.
#
#     Args:
#         plan: Research plan dict (objectives -> steps -> input_nodes / output_node)
#         output_path: Path to save the HTML file
#
#     Returns:
#         Path to the saved HTML file
#     """
#     import json
#     import os
#     import html as _html
#
#     objectives = plan.get("objectives", [])
#     plan_title = plan.get("title", "Geoprocessing Workflow")
#
#     # ── ArcGIS Model Builder palette ──────────────────────────────────
#     INPUT_FILL    = "#6BA3D6";  INPUT_STROKE  = "#2F74B5"   # blue ovals
#     TOOL_FILL     = "#F5D576";  TOOL_STROKE   = "#C4A030"   # yellow rects
#     OUTPUT_FILL   = "#6BB56A";  OUTPUT_STROKE = "#3D8B3D"   # green ovals
#     BAND_COLORS   = ["#edf2f9", "#fdf5e8", "#edf9ed", "#f9edee", "#f0edf9"]
#
#     # ── Node dimensions ──────────────────────────────────────────────
#     OVAL_RX   = 85;  OVAL_RY   = 32
#     RECT_W    = 180; RECT_H    = 62; RECT_RX = 5
#     GAP       = 60          # horizontal gap between connected shapes
#     STACK_GAP = 14          # vertical gap between stacked input ovals
#     LEFT_PAD  = 380         # room for RQ node + gap + objective label
#     TOP_PAD   = 60
#     ROW_GAP   = 50          # vertical space between objective bands
#     FONT      = "Segoe UI, Arial, sans-serif"
#     FS_NODE   = 14
#     FS_OBJ    = 17
#
#     # ── Helpers ──────────────────────────────────────────────────────
#     def wrap_svg_text(text, cx, cy, max_chars=18, fs=FS_NODE, bold=False, fill="#333"):
#         """Return SVG <text> element(s), wrapping into up to 2 lines."""
#         fw = ' font-weight="bold"' if bold else ""
#         safe = lambda t: _html.escape(t)
#         if len(text) <= max_chars:
#             return (f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="{fs}"{fw}'
#                     f' font-family="{FONT}" fill="{fill}">{safe(text)}</text>')
#         # best split near the middle at a space
#         mid = len(text) // 2
#         best = mid
#         for off in range(mid):
#             if text[mid - off] == " ":
#                 best = mid - off; break
#             if mid + off < len(text) and text[mid + off] == " ":
#                 best = mid + off; break
#         l1 = safe(text[:best].strip()[:max_chars])
#         l2 = safe(text[best:].strip()[:max_chars])
#         dy = fs * 0.6
#         return (f'<text text-anchor="middle" font-size="{fs}"{fw} font-family="{FONT}" fill="{fill}">'
#                 f'<tspan x="{cx}" y="{cy - dy + 2}">{l1}</tspan>'
#                 f'<tspan x="{cx}" y="{cy + dy + 2}">{l2}</tspan></text>')
#
#     # ── First pass: compute layout per objective row ─────────────────
#     rows = []
#     max_row_width = 0
#
#     for i_obj, obj in enumerate(objectives):
#         obj_num = obj["number"]
#         steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))
#
#         placed = {}          # data_name -> {cx, cy_off, node_id}
#         nodes_list = []
#         edges_list = []      # (from_id, to_id, edge_type)
#         x = LEFT_PAD
#         max_stack_h = RECT_H
#         last_out_cx = None   # cx of the most recent output oval in this chain
#         prev_step_out_name = None  # output_node name of immediately previous step
#
#         for j, step in enumerate(steps):
#             step_num  = step["step_number"]
#             operation = step.get("operation", "")
#             inp_names = step.get("input_nodes", [])
#             out_name  = step.get("output_node", "")
#             desc      = step.get("description", "")
#             expected  = step.get("expected_output", "") or ""
#             op_type   = step.get("operation_type", "")
#
#             new_inputs    = [n for n in inp_names if n not in placed]
#             reused_inputs = [n for n in inp_names if n in placed]
#
#             # ── new input ovals ──
#             input_ids = {}
#             if new_inputs:
#                 n = len(new_inputs)
#                 block_h = n * (OVAL_RY * 2) + (n - 1) * STACK_GAP
#                 max_stack_h = max(max_stack_h, block_h)
#                 start_y = -(block_h / 2) + OVAL_RY
#                 inp_cx = x + OVAL_RX          # center so left edge is at x
#                 for k, inp in enumerate(new_inputs):
#                     y_off = start_y + k * (OVAL_RY * 2 + STACK_GAP)
#                     nid = f"o{obj_num}_s{step_num}_i{k}"
#                     nodes_list.append({"id": nid, "type": "input", "name": inp,
#                                        "cx": inp_cx, "cy_off": y_off})
#                     placed[inp] = {"cx": inp_cx, "cy_off": y_off, "node_id": nid}
#                     input_ids[inp] = nid
#                 x = inp_cx + OVAL_RX + GAP    # advance past right edge + gap
#
#             # ── operation rectangle ──
#             op_cx = x + RECT_W / 2
#             op_id = f"o{obj_num}_s{step_num}_op"
#             nodes_list.append({"id": op_id, "type": "operation", "name": operation,
#                                "cx": op_cx, "cy_off": 0,
#                                "step_num": step_num, "desc": desc,
#                                "expected": expected, "op_type": op_type,
#                                "input_nodes": inp_names, "output_node": out_name})
#
#             for inp in new_inputs:
#                 edges_list.append((input_ids[inp], op_id, "direct"))
#
#             # ── reused inputs: chain vs distant ──
#             # Chain: the immediately previous step's output flows horizontally
#             #        into this operation (normal arrow, no reference node needed).
#             # Distant: output from an earlier (non-adjacent) step → reference
#             #          node above the previous output column + diagonal arrow.
#             chain_input = None
#             distant_reused = []
#             if not new_inputs and prev_step_out_name in reused_inputs:
#                 chain_input = prev_step_out_name
#                 distant_reused = [n for n in reused_inputs if n != chain_input]
#             else:
#                 distant_reused = reused_inputs
#
#             if chain_input:
#                 edges_list.append((placed[chain_input]["node_id"], op_id, "direct"))
#             elif prev_step_out_name and prev_step_out_name in placed:
#                 # No data-chain but keep the visual flow continuous
#                 edges_list.append((placed[prev_step_out_name]["node_id"], op_id, "direct"))
#
#             if distant_reused:
#                 ref_col_cx = last_out_cx if last_out_cx else (op_cx - RECT_W / 2 - GAP - OVAL_RX)
#                 ref_start_y = -(OVAL_RY * 2 + 16)
#                 for k, inp in enumerate(distant_reused):
#                     ref_y = ref_start_y - k * (OVAL_RY * 2 + STACK_GAP)
#                     ref_id = f"o{obj_num}_s{step_num}_ref{k}"
#                     nodes_list.append({"id": ref_id, "type": "output_ref",
#                                        "name": inp,
#                                        "cx": ref_col_cx, "cy_off": ref_y})
#                     edges_list.append((ref_id, op_id, "reference"))
#                     max_stack_h = max(max_stack_h, abs(ref_y) + OVAL_RY)
#
#             x = op_cx + RECT_W / 2 + GAP
#
#             # ── output oval ──
#             if out_name:
#                 out_id = f"o{obj_num}_s{step_num}_out"
#                 is_final = (j == len(steps) - 1)
#                 out_cx = x + OVAL_RX           # center so left edge is at x
#                 nodes_list.append({"id": out_id, "type": "output", "name": out_name,
#                                    "cx": out_cx, "cy_off": 0, "final": is_final,
#                                    "desc": expected})
#                 edges_list.append((op_id, out_id, "direct"))
#                 placed[out_name] = {"cx": out_cx, "cy_off": 0, "node_id": out_id}
#                 last_out_cx = out_cx           # remember for reference node positioning
#                 prev_step_out_name = out_name  # track for chain detection
#                 x = out_cx + OVAL_RX + GAP     # advance past right edge + gap
#
#         row_h = max(max_stack_h * 2 + 50, 90)
#         rows.append({"obj": obj, "nodes": nodes_list, "edges": edges_list,
#                      "width": x, "height": row_h})
#         max_row_width = max(max_row_width, x)
#
#     # ── Assign absolute Y positions ──────────────────────────────────
#     y_cursor   = TOP_PAD
#     for row in rows:
#         row["cy"]  = y_cursor + row["height"] / 2
#         row["top"] = y_cursor
#         y_cursor  += row["height"] + ROW_GAP
#
#     # ── Research Question node (far left, arrows diverge to each obj) ─
#     RQ_RECT_W = 130; RQ_RECT_H = 56; RQ_RX = 8
#     RQ_FILL = "#2C3E50"; RQ_STROKE = "#1A252F"
#     OBJ_LABEL_X = 220     # x position for objective labels (after RQ node)
#     rq_text = plan.get("research_question", "Research Question")
#     all_cy = [row["cy"] for row in rows]
#     rq_mid_y = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50
#     rq_cx = 10 + RQ_RECT_W / 2   # left edge at x=10
#     rq_cy = rq_mid_y
#     rq_id = "rq_node"
#
#     # Create objective title nodes and connect RQ → obj_title → first chain node
#     OBJ_NODE_W = 120; OBJ_NODE_H = 42; OBJ_NODE_RX = 6
#     OBJ_NODE_FILL = "#34495E"; OBJ_NODE_STROKE = "#2C3E50"
#     obj_title_nodes = []
#     rq_edges = []
#     for row in rows:
#         obj = row["obj"]
#         obj_num = obj["number"]
#         oid = f"obj_title_{obj_num}"
#         obj_cx = OBJ_LABEL_X + OBJ_NODE_W / 2
#         obj_title_nodes.append({
#             "id": oid, "cx": obj_cx, "abs_cy": row["cy"],
#             "type": "obj_title", "name": obj.get("title", ""),
#             "obj_num": obj_num,
#             "hypothesis": obj.get("hypothesis", ""),
#         })
#         rq_edges.append(("rq_node", oid))
#
#     # ── Pipeline tail: Result Collation → Results Analysis →
#     #    Discussion of Results → Manuscript ─────────────────────────
#     STAGE_RECT_W  = 190;  STAGE_RECT_H = 60; STAGE_RX = 5
#     STAGE_OVAL_RX = 90;   STAGE_OVAL_RY = 32
#     STAGE_GAP     = 55
#     PIPELINE_FILL   = "#8E44AD";  PIPELINE_STROKE = "#6C3483"   # purple ops
#     MANUSCRIPT_FILL = "#C0392B";  MANUSCRIPT_STROKE = "#922B21" # red final
#
#     # Collect the final output node id from each objective
#     final_out_ids = []
#     for row in rows:
#         for nd in row["nodes"]:
#             if nd.get("final"):
#                 final_out_ids.append(nd["id"])
#
#     # Vertical centre across all objective rows
#     all_cy = [row["cy"] for row in rows]
#     mid_y  = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50
#
#     # Chain:  [Result Collation] → [Results Analysis] →
#     #         [Discussion of Results] → [Manuscript]
#     stage_labels = [
#         ("Result Collation",       "collation"),   # yellow like operation nodes
#         ("Results Analysis",       "operation"),
#         ("Discussion of Results",  "operation"),
#         ("Manuscript",             "final"),
#     ]
#     stage_nodes = []   # list of {id, cx, cy, type, name}
#     stage_edges = []   # (from_id, to_id)
#     sx = max_row_width + STAGE_GAP
#     for idx, (label, stype) in enumerate(stage_labels):
#         sid = f"stage_{idx}"
#         if stype == "final":
#             cx = sx + STAGE_OVAL_RX
#             stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
#                                 "type": "stage_final", "name": label})
#             sx = cx + STAGE_OVAL_RX + STAGE_GAP
#         elif stype == "collation":
#             cx = sx + STAGE_RECT_W / 2
#             stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
#                                 "type": "stage_collation", "name": label})
#             sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
#         else:
#             cx = sx + STAGE_RECT_W / 2
#             stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
#                                 "type": "stage_op", "name": label})
#             sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
#         if idx > 0:
#             stage_edges.append((f"stage_{idx-1}", sid))
#
#     # Edges from each objective's final output → Result Collation
#     for fid in final_out_ids:
#         stage_edges.append((fid, "stage_0"))
#
#     svg_width  = max(1200, int(sx) + 60)
#     svg_height = max(400, int(y_cursor) + 30)
#
#     # ── Build node lookup & popup-details dict ───────────────────────
#     node_lookup = {}
#     details = {}
#
#     for row in rows:
#         obj     = row["obj"]
#         obj_num = obj["number"]
#         for nd in row["nodes"]:
#             nd["abs_cy"] = row["cy"] + nd["cy_off"]
#             node_lookup[nd["id"]] = nd
#
#             if nd["type"] == "input":
#                 details[nd["id"]] = (
#                     f"<h3>Input Data</h3>"
#                     f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>")
#             elif nd["type"] == "operation":
#                 d = (f"<h3>Step {obj_num}.{nd['step_num']}: {_html.escape(nd['name'])}</h3>"
#                      f"<p><b>Operation Type:</b> {_html.escape(nd.get('op_type',''))}</p>"
#                      f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in nd.get('input_nodes',[]))}</p>"
#                      f"<p><b>Output:</b> {_html.escape(nd.get('output_node',''))}</p>"
#                      f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>")
#                 if nd.get("expected"):
#                     d += f"<p><b>Expected Output:</b> {_html.escape(nd['expected'])}</p>"
#                 details[nd["id"]] = d
#             elif nd["type"] in ("output", "output_ref"):
#                 label = "Output Data" if nd["type"] == "output" else "Reused Output"
#                 details[nd["id"]] = (
#                     f"<h3>{label}</h3>"
#                     f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>"
#                     + (f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>"
#                        if nd.get("desc") else ""))
#
#     # Register RQ node in lookup + details
#     node_lookup[rq_id] = {"id": rq_id, "type": "rq", "name": rq_text,
#                           "cx": rq_cx, "abs_cy": rq_cy}
#     details[rq_id] = (f"<h3>Research Question</h3>"
#                       f"<p>{_html.escape(rq_text)}</p>")
#
#     # Register pipeline stage nodes in lookup + details
#     stage_descs = {
#         "Result Collation":      "Collate outputs and results from all objectives into a unified dataset for analysis.",
#         "Results Analysis":      "Perform statistical analysis, validation, and interpretation of the collated results.",
#         "Discussion of Results": "Synthesise findings, compare with hypotheses, and discuss implications.",
#         "Manuscript":            "Final research deliverable: paper, report, or document combining all findings.",
#     }
#     for sn in stage_nodes:
#         node_lookup[sn["id"]] = sn
#         details[sn["id"]] = (f"<h3>{_html.escape(sn['name'])}</h3>"
#                              f"<p>{_html.escape(stage_descs.get(sn['name'], ''))}</p>")
#
#     # Register objective text anchors in lookup (for arrow targeting only)
#     for otn in obj_title_nodes:
#         node_lookup[otn["id"]] = otn
#
#     # ── Generate SVG body ────────────────────────────────────────────
#     svg = []
#
#     # background bands + objective text labels
#     for i_r, row in enumerate(rows):
#         bc = BAND_COLORS[i_r % len(BAND_COLORS)]
#         svg.append(f'<rect x="0" y="{row["top"]-5}" width="{svg_width}" '
#                    f'height="{row["height"]+10}" fill="{bc}" rx="6"/>')
#         obj = row["obj"]
#         svg.append(f'<text x="{OBJ_LABEL_X}" y="{row["cy"]-10}" font-size="{FS_OBJ}" '
#                    f'font-weight="bold" font-family="{FONT}" fill="#333">'
#                    f'Obj {obj["number"]}</text>')
#         t = obj.get("title", "")
#         if t:
#             short = t[:22] + (".." if len(t) > 22 else "")
#             svg.append(f'<text x="{OBJ_LABEL_X}" y="{row["cy"]+8}" font-size="{FS_NODE}" '
#                        f'font-family="{FONT}" fill="#666">{_html.escape(short)}</text>')
#
#     # ── RQ node (rendered before edges so edges draw on top) ────────
#     svg.append(f'<g class="node-box" onclick="showPopup(\'{rq_id}\')">')
#     rq_rx_ = rq_cx - RQ_RECT_W / 2; rq_ry_ = rq_cy - RQ_RECT_H / 2
#     svg.append(f'  <rect x="{rq_rx_}" y="{rq_ry_}" width="{RQ_RECT_W}" '
#                f'height="{RQ_RECT_H}" rx="{RQ_RX}" '
#                f'fill="{RQ_FILL}" stroke="{RQ_STROKE}" '
#                f'stroke-width="2" data-os="{RQ_STROKE}"/>')
#     svg.append("  " + wrap_svg_text("Research Question", rq_cx, rq_cy, 20, fs=15, bold=True, fill="white"))
#     svg.append('</g>')
#
#     # ── RQ edges (RQ → objective text labels) ────────────────────
#     #   Diverging S-curves: exit RQ horizontally, arrive at obj text horizontally
#     for fid, tid in rq_edges:
#         fn = node_lookup[fid]; tn = node_lookup[tid]
#         x1 = fn["cx"] + RQ_RECT_W / 2       # right edge of RQ rect
#         y1 = fn["abs_cy"]
#         x2 = OBJ_LABEL_X - 5                # just before the obj text
#         y2 = tn["abs_cy"]
#         # S-curve control points: leave RQ horizontally, arrive at text horizontally
#         spread = (x2 - x1) * 0.5
#         svg.append(f'<path d="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
#                    f'fill="none" stroke="#555" stroke-width="2" '
#                    f'marker-end="url(#mb_arrow)"/>')
#
#
#     # edges (drawn first so nodes paint over them)
#     for row in rows:
#         for fid, tid, etype in row["edges"]:
#             fn = node_lookup[fid]; tn = node_lookup[tid]
#
#             if etype == "reference":
#                 # Reference node is above-left of the operation →
#                 # diagonal arrow from right edge of ref oval to left edge of op rect
#                 x1 = fn["cx"] + OVAL_RX              # right edge of reference oval
#                 y1 = fn["abs_cy"]                     # center-y of reference oval
#                 x2 = tn["cx"] - RECT_W / 2           # left edge of operation rect
#                 y2 = tn["abs_cy"]                     # center-y of operation
#                 svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                            f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#             else:
#                 # Horizontal flow: right side of source → left side of target
#                 x1 = fn["cx"] + (OVAL_RX if fn["type"] != "operation" else RECT_W / 2)
#                 y1 = fn["abs_cy"]
#                 x2 = tn["cx"] - (OVAL_RX if tn["type"] != "operation" else RECT_W / 2)
#                 y2 = tn["abs_cy"]
#
#                 if abs(y1 - y2) > 5:
#                     # Stacked inputs at different y — smooth S-curve
#                     mx = (x1 + x2) / 2
#                     svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
#                                f'fill="none" stroke="#555" stroke-width="2" '
#                                f'marker-end="url(#mb_arrow)"/>')
#                 else:
#                     svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                                f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # nodes
#     for row in rows:
#         for nd in row["nodes"]:
#             cx = nd["cx"]; cy = nd["abs_cy"]; nid = nd["id"]
#             svg.append(f'<g class="node-box" onclick="showPopup(\'{nid}\')">')
#
#             if nd["type"] == "input":
#                 svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
#                            f'fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="1.5" '
#                            f'data-os="{INPUT_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))
#
#             elif nd["type"] == "operation":
#                 rx = cx - RECT_W / 2; ry_ = cy - RECT_H / 2
#                 svg.append(f'  <rect x="{rx}" y="{ry_}" width="{RECT_W}" height="{RECT_H}" '
#                            f'rx="{RECT_RX}" fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
#                            f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 18, bold=True, fill="#333"))
#
#             elif nd["type"] == "output":
#                 orx = OVAL_RX + (5 if nd.get("final") else 0)
#                 ory = OVAL_RY + (3 if nd.get("final") else 0)
#                 sw  = "2.5" if nd.get("final") else "1.5"
#                 svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{orx}" ry="{ory}" '
#                            f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="{sw}" '
#                            f'data-os="{OUTPUT_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))
#
#             elif nd["type"] == "output_ref":
#                 # Reused output — green oval placed above the operation
#                 svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
#                            f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="1.5" '
#                            f'data-os="{OUTPUT_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))
#
#             svg.append('</g>')
#
#     # ── Pipeline stage edges ─────────────────────────────────────────
#     for fid, tid in stage_edges:
#         fn = node_lookup[fid]; tn = node_lookup[tid]
#         # Exit point
#         if fn.get("type") in ("output", "stage_final"):
#             x1 = fn["cx"] + (STAGE_OVAL_RX if fn["type"] == "stage_final" else OVAL_RX)
#         elif fn.get("type") in ("stage_op", "stage_collation"):
#             x1 = fn["cx"] + STAGE_RECT_W / 2
#         else:
#             x1 = fn["cx"] + (OVAL_RX if fn.get("type") != "operation" else RECT_W / 2)
#         y1 = fn["abs_cy"]
#         # Entry point
#         if tn.get("type") == "stage_final":
#             x2 = tn["cx"] - STAGE_OVAL_RX
#         elif tn.get("type") in ("stage_op", "stage_collation"):
#             x2 = tn["cx"] - STAGE_RECT_W / 2
#         else:
#             x2 = tn["cx"] - OVAL_RX
#         y2 = tn["abs_cy"]
#
#         if abs(y1 - y2) > 5:
#             mx = (x1 + x2) / 2
#             svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
#                        f'fill="none" stroke="#555" stroke-width="2" '
#                        f'marker-end="url(#mb_arrow)"/>')
#         else:
#             svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                        f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # ── Pipeline stage nodes ─────────────────────────────────────────
#     for sn in stage_nodes:
#         cx = sn["cx"]; cy = sn["abs_cy"]; sid = sn["id"]
#         svg.append(f'<g class="node-box" onclick="showPopup(\'{sid}\')">')
#         if sn["type"] == "stage_collation":
#             rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
#             svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
#                        f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
#                        f'fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
#                        f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="#333"))
#         elif sn["type"] == "stage_op":
#             rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
#             svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
#                        f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
#                        f'fill="{PIPELINE_FILL}" stroke="{PIPELINE_STROKE}" '
#                        f'stroke-width="2" data-os="{PIPELINE_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="white"))
#         elif sn["type"] == "stage_final":
#             svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{STAGE_OVAL_RX}" '
#                        f'ry="{STAGE_OVAL_RY}" fill="{MANUSCRIPT_FILL}" '
#                        f'stroke="{MANUSCRIPT_STROKE}" stroke-width="2.5" '
#                        f'data-os="{MANUSCRIPT_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 18, fs=14, bold=True, fill="white"))
#         svg.append('</g>')
#
#     svg_body = "\n    ".join(svg)
#
#     # ── Assemble HTML ────────────────────────────────────────────────
#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>{_html.escape(plan_title)}</title>
# <style>
#   body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
#   h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
#   .graph-panel {{ flex:1; overflow:hidden; position:relative; cursor:grab; }}
#   svg {{ display:block; width:100%; height:100%; }}
#   .node-box {{ cursor:pointer; transition:filter .15s; }}
#   .node-box:hover {{ filter:brightness(1.12) drop-shadow(0 0 4px rgba(0,0,0,.3)); }}
#   .zoom-toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
#   .zoom-btn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; color:#444; transition:all .15s; padding:0; }}
#   .zoom-btn:hover {{ background:#e9ecef; }}
#   .zoom-level {{ font-size:10px; text-align:center; color:#888; padding:2px 0; user-select:none; }}
#   .zoom-divider {{ height:1px; background:#e0e0e0; margin:2px 0; }}
#   .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
#   .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
#   .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
#   .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
#   .popup-overlay.visible {{ display:flex; }}
#   .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
#   @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
#   .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
#   .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
#   .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
#   .popup-close:hover {{ background:#f0f0f0; color:#333; }}
# </style>
# </head>
# <body>
# <h1>{_html.escape(plan_title)}</h1>
# <div class="graph-panel" id="graphPanel">
#   <div class="zoom-toolbar">
#     <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
#     <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
#     <div class="zoom-divider"></div>
#     <span class="zoom-level" id="zoomLevel">100%</span>
#     <div class="zoom-divider"></div>
#     <button class="zoom-btn" onclick="zoomReset()" title="Reset">&#8634;</button>
#     <button class="zoom-btn" onclick="zoomFit()" title="Fit">&#11036;</button>
#   </div>
#   <div class="legend">
#     <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:2px"></div> Operation / Tool</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Output Data</div>
#   </div>
#   <svg id="flowchartSvg" viewBox="0 0 {svg_width} {svg_height}">
#     <defs>
#       <marker id="mb_arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
#         <polygon points="0 0,10 3.5,0 7" fill="#555"/>
#       </marker>
#     </defs>
#     <g id="zoomGroup">
#     {svg_body}
#     </g>
#   </svg>
# </div>
#
# <div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
#   <div class="popup-box" onclick="event.stopPropagation()">
#     <button class="popup-close" onclick="closePopup()">&times;</button>
#     <div id="popupContent"></div>
#   </div>
# </div>
#
# <script>
# const details = """ + json.dumps(details) + """;
#
# let scale=1, panX=0, panY=0, isPanning=false, _wasDragged=false, psX=0, psY=0, _mdX=0, _mdY=0;
# const PT=4;
# const MN=0.2, MX=5, ZS=0.15;
# const svg=document.getElementById('flowchartSvg');
# const zg=document.getElementById('zoomGroup');
# const gp=document.getElementById('graphPanel');
#
# function applyTransform(){
#   zg.setAttribute('transform','translate('+panX+','+panY+') scale('+scale+')');
#   document.getElementById('zoomLevel').textContent=Math.round(scale*100)+'%';
# }
# function zoomIn(){ scale=Math.min(MX,scale+ZS); applyTransform(); }
# function zoomOut(){ scale=Math.max(MN,scale-ZS); applyTransform(); }
# function zoomReset(){ scale=1;panX=0;panY=0;applyTransform(); }
# function zoomFit(){
#   var b=zg.getBBox(),v=svg.viewBox.baseVal,p=30;
#   scale=Math.min((v.width-p*2)/b.width,(v.height-p*2)/b.height,MX);
#   panX=(v.width/scale-b.width)/2-b.x+p/scale;
#   panY=(v.height/scale-b.height)/2-b.y+p/scale;
#   applyTransform();
# }
#
# gp.addEventListener('wheel',function(e){
#   e.preventDefault();
#   var r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
#   var sx=((e.clientX-r.left)/r.width)*v.width, sy=((e.clientY-r.top)/r.height)*v.height;
#   var os=scale;
#   scale=e.deltaY<0?Math.min(MX,scale+ZS):Math.max(MN,scale-ZS);
#   panX=sx/scale-sx/os+panX; panY=sy/scale-sy/os+panY; applyTransform();
# },{passive:false});
#
# gp.addEventListener('mousedown',function(e){
#   if(e.button===0||e.button===1){
#     e.preventDefault();isPanning=true;_wasDragged=false;
#     psX=e.clientX;psY=e.clientY;_mdX=e.clientX;_mdY=e.clientY;
#     gp.style.cursor='grabbing';
#   }
# });
# window.addEventListener('mousemove',function(e){
#   if(!isPanning)return;
#   var dist=Math.abs(e.clientX-_mdX)+Math.abs(e.clientY-_mdY);
#   if(dist>PT)_wasDragged=true;
#   var v=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
#   panX+=(e.clientX-psX)*(v.width/r.width)/scale;
#   panY+=(e.clientY-psY)*(v.height/r.height)/scale;
#   psX=e.clientX;psY=e.clientY;applyTransform();
# });
# window.addEventListener('mouseup',function(){ if(isPanning){isPanning=false;gp.style.cursor='';} });
# gp.addEventListener('click',function(e){ if(_wasDragged){e.stopPropagation();_wasDragged=false;} },true);
#
# function showPopup(nodeId){
#   var c=document.getElementById('popupContent');
#   var o=document.getElementById('popupOverlay');
#   c.innerHTML=details[nodeId]||'<p style="color:#999">No details.</p>';
#   o.classList.add('visible');
#   // reset all highlights
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   });
#   var cl=event.currentTarget.querySelector('ellipse,rect');
#   if(cl){ cl.setAttribute('stroke-width','3.5'); cl.setAttribute('stroke','#FFD700'); }
# }
# function closePopup(e){
#   if(e&&e.target!==document.getElementById('popupOverlay'))return;
#   document.getElementById('popupOverlay').classList.remove('visible');
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   });
# }
# document.addEventListener('keydown',function(e){ if(e.key==='Escape')closePopup(); });
# window.addEventListener('DOMContentLoaded',function(){ setTimeout(zoomFit,100); });
# </script>
# </body>
# </html>"""
#
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     print(f"Model Builder workflow saved to: {output_path}")
#     return output_path


# def show_geoprocessing_workflow_plan_overview_popup(plan, output_path="Outputs/geoprocessing_workflow_plan_overview_popup.html"):
#     """Render an ArcGIS Model Builder-style SVG workflow for each objective.
#
#     Layout per objective (left-to-right chain):
#         [Input Data (blue oval)] -> [Operation (yellow rect)] -> [Output Data (green oval)] -> ...
#
#     Output nodes from earlier steps are reused when they appear as inputs to later
#     steps (arrows drawn from the existing green oval), just like Model Builder.
#
#     Clicking any node opens a popup with full details.  Supports zoom & pan.
#
#     Args:
#         plan: Research plan dict (objectives -> steps -> input_nodes / output_node)
#         output_path: Path to save the HTML file
#
#     Returns:
#         Path to the saved HTML file
#     """
#     import json
#     import os
#     import html as _html
#
#     objectives = plan.get("objectives", [])
#     plan_title = plan.get("title", "Geoprocessing Workflow")
#
#     # ── ArcGIS Model Builder palette ──────────────────────────────────
#     INPUT_FILL    = "#6BA3D6";  INPUT_STROKE  = "#2F74B5"   # blue ovals
#     TOOL_FILL     = "#F5D576";  TOOL_STROKE   = "#C4A030"   # yellow rects
#     OUTPUT_FILL   = "#6BB56A";  OUTPUT_STROKE = "#3D8B3D"   # green ovals
#     BAND_COLORS   = ["#edf2f9", "#fdf5e8", "#edf9ed", "#f9edee", "#f0edf9"]
#
#     # ── Node dimensions ──────────────────────────────────────────────
#     OVAL_RX   = 85;  OVAL_RY   = 32
#     RECT_W    = 180; RECT_H    = 62; RECT_RX = 5
#     GAP       = 60          # horizontal gap between connected shapes
#     STACK_GAP = 14          # vertical gap between stacked input ovals
#     LEFT_PAD  = 380         # room for RQ node + gap + objective label
#     TOP_PAD   = 60
#     ROW_GAP   = 50          # vertical space between objective bands
#     FONT      = "Segoe UI, Arial, sans-serif"
#     FS_NODE   = 14
#     FS_OBJ    = 17
#
#     # ── Helpers ──────────────────────────────────────────────────────
#     def wrap_svg_text(text, cx, cy, max_chars=18, fs=FS_NODE, bold=False, fill="#333"):
#         """Return SVG <text> element(s), wrapping into up to 2 lines."""
#         fw = ' font-weight="bold"' if bold else ""
#         safe = lambda t: _html.escape(t)
#         if len(text) <= max_chars:
#             return (f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="{fs}"{fw}'
#                     f' font-family="{FONT}" fill="{fill}">{safe(text)}</text>')
#         # best split near the middle at a space
#         mid = len(text) // 2
#         best = mid
#         for off in range(mid):
#             if text[mid - off] == " ":
#                 best = mid - off; break
#             if mid + off < len(text) and text[mid + off] == " ":
#                 best = mid + off; break
#         l1 = safe(text[:best].strip()[:max_chars])
#         l2 = safe(text[best:].strip()[:max_chars])
#         dy = fs * 0.6
#         return (f'<text text-anchor="middle" font-size="{fs}"{fw} font-family="{FONT}" fill="{fill}">'
#                 f'<tspan x="{cx}" y="{cy - dy + 2}">{l1}</tspan>'
#                 f'<tspan x="{cx}" y="{cy + dy + 2}">{l2}</tspan></text>')
#
#     # ── First pass: compute layout per objective row ─────────────────
#     rows = []
#     max_row_width = 0
#     global_placed = {}       # data_name -> {node_id} — tracks outputs across ALL objectives
#     cross_obj_edges = []     # (source_node_id, target_node_id) — cross-objective reuse arrows
#
#     for i_obj, obj in enumerate(objectives):
#         obj_num = obj["number"]
#         steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))
#
#         placed = {}          # data_name -> {cx, cy_off, node_id}
#         nodes_list = []
#         edges_list = []      # (from_id, to_id, edge_type)
#         x = LEFT_PAD
#         max_stack_h = RECT_H
#         last_out_cx = None   # cx of the most recent output oval in this chain
#         prev_step_out_name = None  # output_node name of immediately previous step
#
#         for j, step in enumerate(steps):
#             step_num  = step["step_number"]
#             operation = step.get("operation", "")
#             inp_names = step.get("input_nodes", [])
#             out_name  = step.get("output_node", "")
#             desc      = step.get("description", "")
#             expected  = step.get("expected_output", "") or ""
#             op_type   = step.get("operation_type", "")
#
#             new_inputs    = [n for n in inp_names if n not in placed and n not in global_placed]
#             reused_inputs = [n for n in inp_names if n in placed]
#             cross_reused  = [n for n in inp_names if n not in placed and n in global_placed]
#
#             # ── new input ovals ──
#             input_ids = {}
#             if new_inputs:
#                 n = len(new_inputs)
#                 block_h = n * (OVAL_RY * 2) + (n - 1) * STACK_GAP
#                 max_stack_h = max(max_stack_h, block_h)
#                 start_y = -(block_h / 2) + OVAL_RY
#                 inp_cx = x + OVAL_RX          # center so left edge is at x
#                 for k, inp in enumerate(new_inputs):
#                     y_off = start_y + k * (OVAL_RY * 2 + STACK_GAP)
#                     nid = f"o{obj_num}_s{step_num}_i{k}"
#                     nodes_list.append({"id": nid, "type": "input", "name": inp,
#                                        "cx": inp_cx, "cy_off": y_off})
#                     placed[inp] = {"cx": inp_cx, "cy_off": y_off, "node_id": nid}
#                     input_ids[inp] = nid
#                 x = inp_cx + OVAL_RX + GAP    # advance past right edge + gap
#
#             # ── operation rectangle ──
#             op_cx = x + RECT_W / 2
#             op_id = f"o{obj_num}_s{step_num}_op"
#             nodes_list.append({"id": op_id, "type": "operation", "name": operation,
#                                "cx": op_cx, "cy_off": 0,
#                                "step_num": step_num, "desc": desc,
#                                "expected": expected, "op_type": op_type,
#                                "input_nodes": inp_names, "output_node": out_name})
#
#             for inp in new_inputs:
#                 edges_list.append((input_ids[inp], op_id, "direct"))
#
#             # ── reused inputs: chain vs distant ──
#             # Chain: the immediately previous step's output flows horizontally
#             #        into this operation (normal arrow, no reference node needed).
#             # Distant: output from an earlier (non-adjacent) step → reference
#             #          node above the previous output column + diagonal arrow.
#             chain_input = None
#             distant_reused = []
#             if not new_inputs and prev_step_out_name in reused_inputs:
#                 chain_input = prev_step_out_name
#                 distant_reused = [n for n in reused_inputs if n != chain_input]
#             else:
#                 distant_reused = reused_inputs
#
#             if chain_input:
#                 edges_list.append((placed[chain_input]["node_id"], op_id, "direct"))
#             elif prev_step_out_name and prev_step_out_name in placed:
#                 # No data-chain but keep the visual flow continuous
#                 edges_list.append((placed[prev_step_out_name]["node_id"], op_id, "direct"))
#
#             if distant_reused:
#                 for inp in distant_reused:
#                     original = placed[inp]
#                     edges_list.append((original["node_id"], op_id, "reference"))
#                 # Extra vertical space for curved reference arrows above nodes
#                 max_stack_h = max(max_stack_h, OVAL_RY * 2 + 60)
#
#             # ── cross-objective reuse: arrow from original output in another row ──
#             if cross_reused:
#                 for inp in cross_reused:
#                     cross_obj_edges.append((global_placed[inp]["node_id"], op_id))
#
#             x = op_cx + RECT_W / 2 + GAP
#
#             # ── output oval ──
#             if out_name:
#                 out_id = f"o{obj_num}_s{step_num}_out"
#                 is_final = (j == len(steps) - 1)
#                 out_cx = x + OVAL_RX           # center so left edge is at x
#                 nodes_list.append({"id": out_id, "type": "output", "name": out_name,
#                                    "cx": out_cx, "cy_off": 0, "final": is_final,
#                                    "desc": expected})
#                 edges_list.append((op_id, out_id, "direct"))
#                 placed[out_name] = {"cx": out_cx, "cy_off": 0, "node_id": out_id}
#                 last_out_cx = out_cx           # remember for reference node positioning
#                 prev_step_out_name = out_name  # track for chain detection
#                 x = out_cx + OVAL_RX + GAP     # advance past right edge + gap
#
#         row_h = max(max_stack_h * 2 + 50, 90)
#         rows.append({"obj": obj, "nodes": nodes_list, "edges": edges_list,
#                      "width": x, "height": row_h})
#         max_row_width = max(max_row_width, x)
#
#         # Update global_placed with all outputs from this objective
#         for name, info in placed.items():
#             global_placed[name] = {"node_id": info["node_id"]}
#
#     # ── Assign absolute Y positions ──────────────────────────────────
#     y_cursor   = TOP_PAD
#     for row in rows:
#         row["cy"]  = y_cursor + row["height"] / 2
#         row["top"] = y_cursor
#         y_cursor  += row["height"] + ROW_GAP
#
#     # ── Research Question node (far left, arrows diverge to each obj) ─
#     RQ_RECT_W = 130; RQ_RECT_H = 56; RQ_RX = 8
#     RQ_FILL = "#2C3E50"; RQ_STROKE = "#1A252F"
#     OBJ_LABEL_X = 220     # x position for objective labels (after RQ node)
#     rq_text = plan.get("research_question", "Research Question")
#     all_cy = [row["cy"] for row in rows]
#     rq_mid_y = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50
#     rq_cx = 10 + RQ_RECT_W / 2   # left edge at x=10
#     rq_cy = rq_mid_y
#     rq_id = "rq_node"
#
#     # Create objective title nodes and connect RQ → obj_title → first chain node
#     OBJ_NODE_W = 120; OBJ_NODE_H = 42; OBJ_NODE_RX = 6
#     OBJ_NODE_FILL = "#34495E"; OBJ_NODE_STROKE = "#2C3E50"
#     obj_title_nodes = []
#     rq_edges = []
#     for row in rows:
#         obj = row["obj"]
#         obj_num = obj["number"]
#         oid = f"obj_title_{obj_num}"
#         obj_cx = OBJ_LABEL_X + OBJ_NODE_W / 2
#         obj_title_nodes.append({
#             "id": oid, "cx": obj_cx, "abs_cy": row["cy"],
#             "type": "obj_title", "name": obj.get("title", ""),
#             "obj_num": obj_num,
#             "hypothesis": obj.get("hypothesis", ""),
#         })
#         rq_edges.append(("rq_node", oid))
#
#     # ── Pipeline tail: Result Collation → Results Analysis →
#     #    Discussion of Results → Manuscript ─────────────────────────
#     STAGE_RECT_W  = 190;  STAGE_RECT_H = 60; STAGE_RX = 5
#     STAGE_OVAL_RX = 90;   STAGE_OVAL_RY = 32
#     STAGE_GAP     = 55
#     PIPELINE_FILL   = "#8E44AD";  PIPELINE_STROKE = "#6C3483"   # purple ops
#     MANUSCRIPT_FILL = "#C0392B";  MANUSCRIPT_STROKE = "#922B21" # red final
#
#     # Collect the final output node id from each objective
#     final_out_ids = []
#     for row in rows:
#         for nd in row["nodes"]:
#             if nd.get("final"):
#                 final_out_ids.append(nd["id"])
#
#     # Vertical centre across all objective rows
#     all_cy = [row["cy"] for row in rows]
#     mid_y  = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50
#
#     # Chain:  [Result Collation] → [Results Analysis] →
#     #         [Discussion of Results] → [Manuscript]
#     stage_labels = [
#         ("Result Collation",       "collation"),   # yellow like operation nodes
#         ("Results Analysis",       "operation"),
#         ("Discussion of Results",  "operation"),
#         ("Manuscript",             "final"),
#     ]
#     stage_nodes = []   # list of {id, cx, cy, type, name}
#     stage_edges = []   # (from_id, to_id)
#     sx = max_row_width + STAGE_GAP
#     for idx, (label, stype) in enumerate(stage_labels):
#         sid = f"stage_{idx}"
#         if stype == "final":
#             cx = sx + STAGE_OVAL_RX
#             stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
#                                 "type": "stage_final", "name": label})
#             sx = cx + STAGE_OVAL_RX + STAGE_GAP
#         elif stype == "collation":
#             cx = sx + STAGE_RECT_W / 2
#             stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
#                                 "type": "stage_collation", "name": label})
#             sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
#         else:
#             cx = sx + STAGE_RECT_W / 2
#             stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
#                                 "type": "stage_op", "name": label})
#             sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
#         if idx > 0:
#             stage_edges.append((f"stage_{idx-1}", sid))
#
#     # Edges from each objective's final output → Result Collation
#     for fid in final_out_ids:
#         stage_edges.append((fid, "stage_0"))
#
#     svg_width  = max(1200, int(sx) + 60)
#     svg_height = max(400, int(y_cursor) + 30)
#
#     # ── Build node lookup & popup-details dict ───────────────────────
#     node_lookup = {}
#     details = {}
#
#     for row in rows:
#         obj     = row["obj"]
#         obj_num = obj["number"]
#         for nd in row["nodes"]:
#             nd["abs_cy"] = row["cy"] + nd["cy_off"]
#             node_lookup[nd["id"]] = nd
#
#             if nd["type"] == "input":
#                 details[nd["id"]] = (
#                     f"<h3>Input Data</h3>"
#                     f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>")
#             elif nd["type"] == "operation":
#                 d = (f"<h3>Step {obj_num}.{nd['step_num']}: {_html.escape(nd['name'])}</h3>"
#                      f"<p><b>Operation Type:</b> {_html.escape(nd.get('op_type',''))}</p>"
#                      f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in nd.get('input_nodes',[]))}</p>"
#                      f"<p><b>Output:</b> {_html.escape(nd.get('output_node',''))}</p>"
#                      f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>")
#                 if nd.get("expected"):
#                     d += f"<p><b>Expected Output:</b> {_html.escape(nd['expected'])}</p>"
#                 details[nd["id"]] = d
#             elif nd["type"] in ("output", "output_ref"):
#                 label = "Output Data" if nd["type"] == "output" else "Reused Output"
#                 details[nd["id"]] = (
#                     f"<h3>{label}</h3>"
#                     f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>"
#                     + (f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>"
#                        if nd.get("desc") else ""))
#
#     # Register RQ node in lookup + details
#     node_lookup[rq_id] = {"id": rq_id, "type": "rq", "name": rq_text,
#                           "cx": rq_cx, "abs_cy": rq_cy}
#     details[rq_id] = (f"<h3>Research Question</h3>"
#                       f"<p>{_html.escape(rq_text)}</p>")
#
#     # Register pipeline stage nodes in lookup + details
#     stage_descs = {
#         "Result Collation":      "Collate outputs and results from all objectives into a unified dataset for analysis.",
#         "Results Analysis":      "Perform statistical analysis, validation, and interpretation of the collated results.",
#         "Discussion of Results": "Synthesise findings, compare with hypotheses, and discuss implications.",
#         "Manuscript":            "Final research deliverable: paper, report, or document combining all findings.",
#     }
#     for sn in stage_nodes:
#         node_lookup[sn["id"]] = sn
#         details[sn["id"]] = (f"<h3>{_html.escape(sn['name'])}</h3>"
#                              f"<p>{_html.escape(stage_descs.get(sn['name'], ''))}</p>")
#
#     # Register objective title nodes in lookup + details (clickable with popup)
#     for otn in obj_title_nodes:
#         node_lookup[otn["id"]] = otn
#         obj_num = otn["obj_num"]
#         obj_name = otn["name"]
#         hypothesis = otn.get("hypothesis", "")
#         d = (f"<h3>Objective {obj_num}</h3>"
#              f"<p><b>Title:</b> {_html.escape(obj_name)}</p>")
#         if hypothesis:
#             d += f"<p><b>Hypothesis:</b> {_html.escape(hypothesis)}</p>"
#         # Include step summary
#         for row in rows:
#             if row["obj"].get("number") == obj_num:
#                 steps = sorted(row["obj"].get("steps", []), key=lambda s: s.get("step_number", 0))
#                 if steps:
#                     d += "<p><b>Steps:</b></p><ol>"
#                     for s in steps:
#                         d += f"<li><b>{_html.escape(s.get('operation',''))}</b>: {_html.escape(s.get('description',''))}</li>"
#                     d += "</ol>"
#                 break
#         details[otn["id"]] = d
#
#     # ── Generate SVG body ────────────────────────────────────────────
#     svg = []
#
#     # background bands + objective rectangle nodes
#     for i_r, row in enumerate(rows):
#         bc = BAND_COLORS[i_r % len(BAND_COLORS)]
#         svg.append(f'<rect x="0" y="{row["top"]-5}" width="{svg_width}" '
#                    f'height="{row["height"]+10}" fill="{bc}" rx="6"/>')
#         obj = row["obj"]
#         obj_num = obj["number"]
#         oid = f"obj_title_{obj_num}"
#         obj_cx = OBJ_LABEL_X + OBJ_NODE_W / 2
#         obj_cy = row["cy"]
#         rx_ = obj_cx - OBJ_NODE_W / 2
#         ry_ = obj_cy - OBJ_NODE_H / 2
#         svg.append(f'<g class="node-box" onclick="showPopup(\'{oid}\')">')
#         svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{OBJ_NODE_W}" height="{OBJ_NODE_H}" '
#                    f'rx="{OBJ_NODE_RX}" fill="{OBJ_NODE_FILL}" stroke="{OBJ_NODE_STROKE}" '
#                    f'stroke-width="2" data-os="{OBJ_NODE_STROKE}"/>')
#         svg.append("  " + wrap_svg_text(f"Obj {obj_num}", obj_cx, obj_cy, 16, fs=14, bold=True, fill="white"))
#         svg.append('</g>')
#
#     # ── RQ node (rendered before edges so edges draw on top) ────────
#     svg.append(f'<g class="node-box" onclick="showPopup(\'{rq_id}\')">')
#     rq_rx_ = rq_cx - RQ_RECT_W / 2; rq_ry_ = rq_cy - RQ_RECT_H / 2
#     svg.append(f'  <rect x="{rq_rx_}" y="{rq_ry_}" width="{RQ_RECT_W}" '
#                f'height="{RQ_RECT_H}" rx="{RQ_RX}" '
#                f'fill="{RQ_FILL}" stroke="{RQ_STROKE}" '
#                f'stroke-width="2" data-os="{RQ_STROKE}"/>')
#     svg.append("  " + wrap_svg_text("Research Question", rq_cx, rq_cy, 20, fs=15, bold=True, fill="white"))
#     svg.append('</g>')
#
#     # ── RQ edges (RQ → objective rectangle nodes) ────────────────
#     #   Diverging S-curves: exit RQ horizontally, arrive at obj rect left edge
#     for fid, tid in rq_edges:
#         fn = node_lookup[fid]; tn = node_lookup[tid]
#         x1 = fn["cx"] + RQ_RECT_W / 2       # right edge of RQ rect
#         y1 = fn["abs_cy"]
#         x2 = tn["cx"] - OBJ_NODE_W / 2      # left edge of objective rect
#         y2 = tn["abs_cy"]
#         # S-curve control points: leave RQ horizontally, arrive at obj rect horizontally
#         spread = (x2 - x1) * 0.5
#         svg.append(f'<path d="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
#                    f'fill="none" stroke="#555" stroke-width="2" '
#                    f'marker-end="url(#mb_arrow)"/>')
#
#     # ── Objective rect → all first-step input nodes ────────────────
#     for row in rows:
#         obj_num = row["obj"]["number"]
#         oid = f"obj_title_{obj_num}"
#         otn = node_lookup[oid]
#         # Find all input nodes that appear before the first operation
#         first_inputs = [nd for nd in row["nodes"] if nd["type"] == "input"
#                         and nd["id"].startswith(f"o{obj_num}_s")]
#         # Only take inputs from the very first step
#         first_step_inputs = []
#         if first_inputs:
#             first_step_prefix = first_inputs[0]["id"].rsplit("_i", 1)[0]  # e.g. "o1_s1"
#             first_step_inputs = [nd for nd in first_inputs
#                                  if nd["id"].startswith(first_step_prefix + "_i")]
#         targets = first_step_inputs if first_step_inputs else row["nodes"][:1]
#         x1 = otn["cx"] + OBJ_NODE_W / 2       # right edge of obj rect
#         y1 = otn["abs_cy"]
#         for tgt in targets:
#             if tgt["type"] == "input":
#                 x2 = tgt["cx"] - OVAL_RX
#             else:
#                 x2 = tgt["cx"] - RECT_W / 2
#             y2 = tgt["abs_cy"]
#             if abs(y1 - y2) > 5:
#                 spread = (x2 - x1) * 0.5
#                 svg.append(f'<path d="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
#                            f'fill="none" stroke="#555" stroke-width="2" '
#                            f'marker-end="url(#mb_arrow)"/>')
#             else:
#                 svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                            f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # edges (drawn first so nodes paint over them)
#     for row in rows:
#         for fid, tid, etype in row["edges"]:
#             fn = node_lookup[fid]; tn = node_lookup[tid]
#
#             if etype == "reference":
#                 # Reused output → curved arrow from original node directly to operation
#                 # Arc height proportional to distance so shorter arcs nest inside longer ones
#                 x1 = fn["cx"]                          # top-center of original output oval
#                 y1 = fn["abs_cy"] - OVAL_RY
#                 x2 = tn["cx"]                          # top-center of operation rect
#                 y2 = tn["abs_cy"] - RECT_H / 2
#                 distance = abs(x2 - x1)
#                 arc_h = max(45, distance * 0.25)       # longer spans → higher arcs
#                 ctrl_y = min(y1, y2) - arc_h
#                 mx = (x1 + x2) / 2
#                 svg.append(f'<path d="M{x1},{y1} Q{mx},{ctrl_y} {x2},{y2}" '
#                            f'fill="none" stroke="#3D8B3D" stroke-width="2" '
#                            f'stroke-dasharray="6,3" marker-end="url(#mb_arrow_green)"/>')
#             else:
#                 # Horizontal flow: right side of source → left side of target
#                 x1 = fn["cx"] + (OVAL_RX if fn["type"] != "operation" else RECT_W / 2)
#                 y1 = fn["abs_cy"]
#                 x2 = tn["cx"] - (OVAL_RX if tn["type"] != "operation" else RECT_W / 2)
#                 y2 = tn["abs_cy"]
#
#                 if abs(y1 - y2) > 5:
#                     # Stacked inputs at different y — smooth S-curve
#                     mx = (x1 + x2) / 2
#                     svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
#                                f'fill="none" stroke="#555" stroke-width="2" '
#                                f'marker-end="url(#mb_arrow)"/>')
#                 else:
#                     svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                                f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # nodes
#     for row in rows:
#         for nd in row["nodes"]:
#             cx = nd["cx"]; cy = nd["abs_cy"]; nid = nd["id"]
#             svg.append(f'<g class="node-box" onclick="showPopup(\'{nid}\')">')
#
#             if nd["type"] == "input":
#                 svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
#                            f'fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="1.5" '
#                            f'data-os="{INPUT_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))
#
#             elif nd["type"] == "operation":
#                 rx = cx - RECT_W / 2; ry_ = cy - RECT_H / 2
#                 svg.append(f'  <rect x="{rx}" y="{ry_}" width="{RECT_W}" height="{RECT_H}" '
#                            f'rx="{RECT_RX}" fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
#                            f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 18, bold=True, fill="#333"))
#
#             elif nd["type"] == "output":
#                 orx = OVAL_RX + (5 if nd.get("final") else 0)
#                 ory = OVAL_RY + (3 if nd.get("final") else 0)
#                 sw  = "2.5" if nd.get("final") else "1.5"
#                 svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{orx}" ry="{ory}" '
#                            f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="{sw}" '
#                            f'data-os="{OUTPUT_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))
#
#             elif nd["type"] == "output_ref":
#                 # Reused output — green oval placed above the operation
#                 svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
#                            f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="1.5" '
#                            f'data-os="{OUTPUT_STROKE}"/>')
#                 svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))
#
#             svg.append('</g>')
#
#     # ── Cross-objective reuse edges (rendered after nodes so arrowheads stay visible) ──
#     # Group by target so multiple arrows landing on the same operation spread out
#     from collections import defaultdict
#     cross_by_target = defaultdict(list)
#     for fid, tid in cross_obj_edges:
#         cross_by_target[tid].append(fid)
#
#     for tid, fids in cross_by_target.items():
#         tn = node_lookup[tid]
#         n = len(fids)
#         spread = min(RECT_W * 0.6, n * 30)            # total spread width
#         for k, fid in enumerate(fids):
#             fn = node_lookup[fid]
#             x1 = fn["cx"]                              # bottom-center of source oval
#             y1 = fn["abs_cy"] + OVAL_RY
#             # Spread landing points evenly across the top of the operation
#             offset = (k - (n - 1) / 2) * (spread / max(n - 1, 1)) if n > 1 else 0
#             x2 = tn["cx"] + offset                     # spread across top edge
#             y2 = tn["abs_cy"] - RECT_H / 2
#             # Last control point directly above endpoint so arrow arrives vertically
#             svg.append(f'<path d="M{x1},{y1} C{x1},{(y1+y2)/2} {x2},{y2 - 40} {x2},{y2}" '
#                        f'fill="none" stroke="#E67E22" stroke-width="2" '
#                        f'stroke-dasharray="8,4" marker-end="url(#mb_arrow_orange)"/>')
#
#     # ── Pipeline stage edges ─────────────────────────────────────────
#     for fid, tid in stage_edges:
#         fn = node_lookup[fid]; tn = node_lookup[tid]
#         # Exit point
#         if fn.get("type") in ("output", "stage_final"):
#             x1 = fn["cx"] + (STAGE_OVAL_RX if fn["type"] == "stage_final" else OVAL_RX)
#         elif fn.get("type") in ("stage_op", "stage_collation"):
#             x1 = fn["cx"] + STAGE_RECT_W / 2
#         else:
#             x1 = fn["cx"] + (OVAL_RX if fn.get("type") != "operation" else RECT_W / 2)
#         y1 = fn["abs_cy"]
#         # Entry point
#         if tn.get("type") == "stage_final":
#             x2 = tn["cx"] - STAGE_OVAL_RX
#         elif tn.get("type") in ("stage_op", "stage_collation"):
#             x2 = tn["cx"] - STAGE_RECT_W / 2
#         else:
#             x2 = tn["cx"] - OVAL_RX
#         y2 = tn["abs_cy"]
#
#         if abs(y1 - y2) > 5:
#             mx = (x1 + x2) / 2
#             svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
#                        f'fill="none" stroke="#555" stroke-width="2" '
#                        f'marker-end="url(#mb_arrow)"/>')
#         else:
#             svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                        f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # ── Pipeline stage nodes ─────────────────────────────────────────
#     for sn in stage_nodes:
#         cx = sn["cx"]; cy = sn["abs_cy"]; sid = sn["id"]
#         svg.append(f'<g class="node-box" onclick="showPopup(\'{sid}\')">')
#         if sn["type"] == "stage_collation":
#             rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
#             svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
#                        f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
#                        f'fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
#                        f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="#333"))
#         elif sn["type"] == "stage_op":
#             rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
#             svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
#                        f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
#                        f'fill="{PIPELINE_FILL}" stroke="{PIPELINE_STROKE}" '
#                        f'stroke-width="2" data-os="{PIPELINE_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="white"))
#         elif sn["type"] == "stage_final":
#             svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{STAGE_OVAL_RX}" '
#                        f'ry="{STAGE_OVAL_RY}" fill="{MANUSCRIPT_FILL}" '
#                        f'stroke="{MANUSCRIPT_STROKE}" stroke-width="2.5" '
#                        f'data-os="{MANUSCRIPT_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 18, fs=14, bold=True, fill="white"))
#         svg.append('</g>')
#
#     svg_body = "\n    ".join(svg)
#
#     # ── Assemble HTML ────────────────────────────────────────────────
#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>{_html.escape(plan_title)}</title>
# <style>
#   body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
#   h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
#   .graph-panel {{ flex:1; overflow:hidden; position:relative; cursor:grab; }}
#   svg {{ display:block; width:100%; height:100%; }}
#   .node-box {{ cursor:pointer; transition:filter .15s; }}
#   .node-box:hover {{ filter:brightness(1.12) drop-shadow(0 0 4px rgba(0,0,0,.3)); }}
#   .zoom-toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
#   .zoom-btn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; color:#444; transition:all .15s; padding:0; }}
#   .zoom-btn:hover {{ background:#e9ecef; }}
#   .zoom-level {{ font-size:10px; text-align:center; color:#888; padding:2px 0; user-select:none; }}
#   .zoom-divider {{ height:1px; background:#e0e0e0; margin:2px 0; }}
#   .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
#   .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
#   .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
#   .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
#   .popup-overlay.visible {{ display:flex; }}
#   .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
#   @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
#   .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
#   .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
#   .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
#   .popup-close:hover {{ background:#f0f0f0; color:#333; }}
# </style>
# </head>
# <body>
# <h1>{_html.escape(plan_title)}</h1>
# <div class="graph-panel" id="graphPanel">
#   <div class="zoom-toolbar">
#     <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
#     <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
#     <div class="zoom-divider"></div>
#     <span class="zoom-level" id="zoomLevel">100%</span>
#     <div class="zoom-divider"></div>
#     <button class="zoom-btn" onclick="zoomReset()" title="Reset">&#8634;</button>
#     <button class="zoom-btn" onclick="zoomFit()" title="Fit">&#11036;</button>
#   </div>
#   <div class="legend">
#     <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:2px"></div> Operation / Tool</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Output Data</div>
#     <div class="legend-item"><div class="legend-swatch" style="border:2px dashed #3D8B3D;background:transparent;border-radius:0"></div> Reused (within obj)</div>
#     <div class="legend-item"><div class="legend-swatch" style="border:2px dashed #E67E22;background:transparent;border-radius:0"></div> Reused (cross obj)</div>
#   </div>
#   <svg id="flowchartSvg" viewBox="0 0 {svg_width} {svg_height}">
#     <defs>
#       <marker id="mb_arrow" markerWidth="14" markerHeight="12" refX="13" refY="6" orient="auto" markerUnits="userSpaceOnUse">
#         <polygon points="0 0,14 6,0 12" fill="#555"/>
#       </marker>
#       <marker id="mb_arrow_green" markerWidth="14" markerHeight="12" refX="13" refY="6" orient="auto" markerUnits="userSpaceOnUse">
#         <polygon points="0 0,14 6,0 12" fill="#3D8B3D"/>
#       </marker>
#       <marker id="mb_arrow_orange" markerWidth="14" markerHeight="12" refX="13" refY="6" orient="auto" markerUnits="userSpaceOnUse">
#         <polygon points="0 0,14 6,0 12" fill="#E67E22"/>
#       </marker>
#     </defs>
#     <g id="zoomGroup">
#     {svg_body}
#     </g>
#   </svg>
# </div>
#
# <div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
#   <div class="popup-box" onclick="event.stopPropagation()">
#     <button class="popup-close" onclick="closePopup()">&times;</button>
#     <div id="popupContent"></div>
#   </div>
# </div>
#
# <script>
# const details = """ + json.dumps(details) + """;
#
# let scale=1, panX=0, panY=0, isPanning=false, _wasDragged=false, psX=0, psY=0, _mdX=0, _mdY=0;
# const PT=4;
# const MN=0.2, MX=5, ZS=0.15;
# const svg=document.getElementById('flowchartSvg');
# const zg=document.getElementById('zoomGroup');
# const gp=document.getElementById('graphPanel');
#
# function applyTransform(){
#   zg.setAttribute('transform','translate('+panX+','+panY+') scale('+scale+')');
#   document.getElementById('zoomLevel').textContent=Math.round(scale*100)+'%';
# }
# function zoomIn(){ scale=Math.min(MX,scale+ZS); applyTransform(); }
# function zoomOut(){ scale=Math.max(MN,scale-ZS); applyTransform(); }
# function zoomReset(){ scale=1;panX=0;panY=0;applyTransform(); }
# function zoomFit(){
#   var b=zg.getBBox(),v=svg.viewBox.baseVal,p=30;
#   scale=Math.min((v.width-p*2)/b.width,(v.height-p*2)/b.height,MX);
#   panX=(v.width/scale-b.width)/2-b.x+p/scale;
#   panY=(v.height/scale-b.height)/2-b.y+p/scale;
#   applyTransform();
# }
#
# gp.addEventListener('wheel',function(e){
#   e.preventDefault();
#   var r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
#   var sx=((e.clientX-r.left)/r.width)*v.width, sy=((e.clientY-r.top)/r.height)*v.height;
#   var os=scale;
#   scale=e.deltaY<0?Math.min(MX,scale+ZS):Math.max(MN,scale-ZS);
#   panX=sx/scale-sx/os+panX; panY=sy/scale-sy/os+panY; applyTransform();
# },{passive:false});
#
# gp.addEventListener('mousedown',function(e){
#   if(e.button===0||e.button===1){
#     e.preventDefault();isPanning=true;_wasDragged=false;
#     psX=e.clientX;psY=e.clientY;_mdX=e.clientX;_mdY=e.clientY;
#     gp.style.cursor='grabbing';
#   }
# });
# window.addEventListener('mousemove',function(e){
#   if(!isPanning)return;
#   var dist=Math.abs(e.clientX-_mdX)+Math.abs(e.clientY-_mdY);
#   if(dist>PT)_wasDragged=true;
#   var v=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
#   panX+=(e.clientX-psX)*(v.width/r.width)/scale;
#   panY+=(e.clientY-psY)*(v.height/r.height)/scale;
#   psX=e.clientX;psY=e.clientY;applyTransform();
# });
# window.addEventListener('mouseup',function(){ if(isPanning){isPanning=false;gp.style.cursor='';} });
# gp.addEventListener('click',function(e){ if(_wasDragged){e.stopPropagation();_wasDragged=false;} },true);
#
# function showPopup(nodeId){
#   var c=document.getElementById('popupContent');
#   var o=document.getElementById('popupOverlay');
#   c.innerHTML=details[nodeId]||'<p style="color:#999">No details.</p>';
#   o.classList.add('visible');
#   // reset all highlights
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   });
#   var cl=event.currentTarget.querySelector('ellipse,rect');
#   if(cl){ cl.setAttribute('stroke-width','3.5'); cl.setAttribute('stroke','#FFD700'); }
# }
# function closePopup(e){
#   if(e&&e.target!==document.getElementById('popupOverlay'))return;
#   document.getElementById('popupOverlay').classList.remove('visible');
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   });
# }
# document.addEventListener('keydown',function(e){ if(e.key==='Escape')closePopup(); });
# window.addEventListener('DOMContentLoaded',function(){ setTimeout(zoomFit,100); });
# </script>
# </body>
# </html>"""
#
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     print(f"Model Builder workflow saved to: {output_path}")
#     return output_path

def show_geoprocessing_workflow_plan_overview_popup_new(plan, output_path="Outputs/geoprocessing_workflow_plan_overview_popup.html"):
    """Render an ArcGIS Model Builder-style SVG workflow for each objective.

    Layout per objective (left-to-right chain):
        [Input Data (blue oval)] -> [Operation (yellow rect)] -> [Output Data (green oval)] -> ...

    Output nodes from earlier steps are reused when they appear as inputs to later
    steps (arrows drawn from the existing green oval), just like Model Builder.

    Clicking any node opens a popup with full details.  Supports zoom & pan.

    Args:
        plan: Research plan dict (objectives -> steps -> input_nodes / output_node)
        output_path: Path to save the HTML file

    Returns:
        Path to the saved HTML file
    """
    import json
    import os
    import html as _html

    objectives = plan.get("objectives", [])
    plan_title = plan.get("title", "Geoprocessing Workflow")

    # ── ArcGIS Model Builder palette ──────────────────────────────────
    INPUT_FILL    = "#6BA3D6";  INPUT_STROKE  = "#2F74B5"   # blue ovals
    TOOL_FILL     = "#F5D576";  TOOL_STROKE   = "#C4A030"   # yellow rects
    OUTPUT_FILL   = "#6BB56A";  OUTPUT_STROKE = "#3D8B3D"   # green ovals
    BAND_COLORS   = ["#edf2f9", "#fdf5e8", "#edf9ed", "#f9edee", "#f0edf9"]

    # ── Node dimensions ──────────────────────────────────────────────
    OVAL_RX   = 85;  OVAL_RY   = 32
    RECT_W    = 180; RECT_H    = 62; RECT_RX = 5
    GAP       = 60          # horizontal gap between connected shapes
    STACK_GAP = 14          # vertical gap between stacked input ovals
    LEFT_PAD  = 380         # room for RQ node + gap + objective label
    TOP_PAD   = 60
    ROW_GAP   = 50          # vertical space between objective bands
    FONT      = "Segoe UI, Arial, sans-serif"
    FS_NODE   = 14
    FS_OBJ    = 17

    # ── Helpers ──────────────────────────────────────────────────────
    def wrap_svg_text(text, cx, cy, max_chars=18, fs=FS_NODE, bold=False, fill="#333"):
        """Return SVG <text> element(s), wrapping into up to 2 lines."""
        fw = ' font-weight="bold"' if bold else ""
        safe = lambda t: _html.escape(t)
        if len(text) <= max_chars:
            return (f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="{fs}"{fw}'
                    f' font-family="{FONT}" fill="{fill}">{safe(text)}</text>')
        # best split near the middle at a space
        mid = len(text) // 2
        best = mid
        for off in range(mid):
            if text[mid - off] == " ":
                best = mid - off; break
            if mid + off < len(text) and text[mid + off] == " ":
                best = mid + off; break
        l1 = safe(text[:best].strip()[:max_chars])
        l2 = safe(text[best:].strip()[:max_chars])
        dy = fs * 0.6
        return (f'<text text-anchor="middle" font-size="{fs}"{fw} font-family="{FONT}" fill="{fill}">'
                f'<tspan x="{cx}" y="{cy - dy + 2}">{l1}</tspan>'
                f'<tspan x="{cx}" y="{cy + dy + 2}">{l2}</tspan></text>')

    # ── First pass: compute layout per objective row ─────────────────
    rows = []
    max_row_width = 0
    global_placed = {}       # data_name -> {node_id} — tracks outputs across ALL objectives
    cross_obj_edges = []     # (source_node_id, target_node_id) — cross-objective reuse arrows

    for i_obj, obj in enumerate(objectives):
        obj_num = obj["number"]
        steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))

        placed = {}          # data_name -> {cx, cy_off, node_id}
        nodes_list = []
        edges_list = []      # (from_id, to_id, edge_type)
        x = LEFT_PAD
        max_stack_h = RECT_H
        last_out_cx = None   # cx of the most recent output oval in this chain
        prev_step_out_name = None  # output_node name of immediately previous step

        for j, step in enumerate(steps):
            step_num  = step["step_number"]
            operation = step.get("operation", "")
            inp_names = step.get("input_nodes", [])
            out_name  = step.get("output_node", "")
            desc      = step.get("description", "")
            expected  = step.get("expected_output", "") or ""
            op_type   = step.get("operation_type", "")

            new_inputs    = [n for n in inp_names if n not in placed and n not in global_placed]
            reused_inputs = [n for n in inp_names if n in placed]
            cross_reused  = [n for n in inp_names if n not in placed and n in global_placed]

            # ── new input ovals ──
            input_ids = {}
            if new_inputs:
                n = len(new_inputs)
                block_h = n * (OVAL_RY * 2) + (n - 1) * STACK_GAP
                max_stack_h = max(max_stack_h, block_h)
                start_y = -(block_h / 2) + OVAL_RY
                inp_cx = x + OVAL_RX          # center so left edge is at x
                for k, inp in enumerate(new_inputs):
                    y_off = start_y + k * (OVAL_RY * 2 + STACK_GAP)
                    nid = f"o{obj_num}_s{step_num}_i{k}"
                    nodes_list.append({"id": nid, "type": "input", "name": inp,
                                       "cx": inp_cx, "cy_off": y_off})
                    placed[inp] = {"cx": inp_cx, "cy_off": y_off, "node_id": nid}
                    input_ids[inp] = nid
                x = inp_cx + OVAL_RX + GAP    # advance past right edge + gap

            # ── operation rectangle ──
            op_cx = x + RECT_W / 2
            op_id = f"o{obj_num}_s{step_num}_op"
            nodes_list.append({"id": op_id, "type": "operation", "name": operation,
                               "cx": op_cx, "cy_off": 0,
                               "step_num": step_num, "desc": desc,
                               "expected": expected, "op_type": op_type,
                               "input_nodes": inp_names, "output_node": out_name})

            for inp in new_inputs:
                edges_list.append((input_ids[inp], op_id, "direct"))

            # ── reused inputs: chain vs distant ──
            # Chain: the immediately previous step's output flows horizontally
            #        into this operation (normal arrow, no reference node needed).
            # Distant: output from an earlier (non-adjacent) step → reference
            #          node above the previous output column + diagonal arrow.
            chain_input = None
            distant_reused = []
            if not new_inputs and prev_step_out_name in reused_inputs:
                chain_input = prev_step_out_name
                distant_reused = [n for n in reused_inputs if n != chain_input]
            else:
                distant_reused = reused_inputs

            if chain_input:
                edges_list.append((placed[chain_input]["node_id"], op_id, "direct"))
            elif prev_step_out_name and prev_step_out_name in placed:
                # No data-chain but keep the visual flow continuous
                edges_list.append((placed[prev_step_out_name]["node_id"], op_id, "direct"))

            if distant_reused:
                for inp in distant_reused:
                    original = placed[inp]
                    edges_list.append((original["node_id"], op_id, "reference"))
                # Extra vertical space for curved reference arrows above nodes
                max_stack_h = max(max_stack_h, OVAL_RY * 2 + 60)

            # ── cross-objective reuse: arrow from original output in another row ──
            if cross_reused:
                for inp in cross_reused:
                    cross_obj_edges.append((global_placed[inp]["node_id"], op_id))

            x = op_cx + RECT_W / 2 + GAP

            # ── output oval ──
            if out_name:
                out_id = f"o{obj_num}_s{step_num}_out"
                is_final = (j == len(steps) - 1)
                out_cx = x + OVAL_RX           # center so left edge is at x
                nodes_list.append({"id": out_id, "type": "output", "name": out_name,
                                   "cx": out_cx, "cy_off": 0, "final": is_final,
                                   "desc": expected})
                edges_list.append((op_id, out_id, "direct"))
                placed[out_name] = {"cx": out_cx, "cy_off": 0, "node_id": out_id}
                last_out_cx = out_cx           # remember for reference node positioning
                prev_step_out_name = out_name  # track for chain detection
                x = out_cx + OVAL_RX + GAP     # advance past right edge + gap

        row_h = max(max_stack_h * 2 + 50, 90)
        rows.append({"obj": obj, "nodes": nodes_list, "edges": edges_list,
                     "width": x, "height": row_h})
        max_row_width = max(max_row_width, x)

        # Update global_placed with all outputs from this objective
        for name, info in placed.items():
            global_placed[name] = {"node_id": info["node_id"]}

    # ── Assign absolute Y positions ──────────────────────────────────
    y_cursor   = TOP_PAD
    for row in rows:
        row["cy"]  = y_cursor + row["height"] / 2
        row["top"] = y_cursor
        y_cursor  += row["height"] + ROW_GAP

    # ── Research Question node (far left, arrows diverge to each obj) ─
    RQ_RECT_W = 130; RQ_RECT_H = 56; RQ_RX = 8
    RQ_FILL = "#2C3E50"; RQ_STROKE = "#1A252F"
    OBJ_LABEL_X = 220     # x position for objective labels (after RQ node)
    rq_text = plan.get("research_question", "Research Question")
    all_cy = [row["cy"] for row in rows]
    rq_mid_y = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50
    rq_cx = 10 + RQ_RECT_W / 2   # left edge at x=10
    rq_cy = rq_mid_y
    rq_id = "rq_node"

    # Create objective title nodes and connect RQ → obj_title → first chain node
    OBJ_NODE_W = 120; OBJ_NODE_H = 42; OBJ_NODE_RX = 6
    OBJ_NODE_FILL = "#34495E"; OBJ_NODE_STROKE = "#2C3E50"
    obj_title_nodes = []
    rq_edges = []
    for row in rows:
        obj = row["obj"]
        obj_num = obj["number"]
        oid = f"obj_title_{obj_num}"
        obj_cx = OBJ_LABEL_X + OBJ_NODE_W / 2
        obj_title_nodes.append({
            "id": oid, "cx": obj_cx, "abs_cy": row["cy"],
            "type": "obj_title", "name": obj.get("title", ""),
            "obj_num": obj_num,
            "hypothesis": obj.get("hypothesis", ""),
        })
        rq_edges.append(("rq_node", oid))

    # ── Pipeline tail: Result Collation → Results Analysis →
    #    Discussion of Results → Manuscript ─────────────────────────
    STAGE_RECT_W  = 190;  STAGE_RECT_H = 60; STAGE_RX = 5
    STAGE_OVAL_RX = 90;   STAGE_OVAL_RY = 32
    STAGE_GAP     = 55
    PIPELINE_FILL   = "#8E44AD";  PIPELINE_STROKE = "#6C3483"   # purple ops
    MANUSCRIPT_FILL = "#C0392B";  MANUSCRIPT_STROKE = "#922B21" # red final

    # Collect the final output node id from each objective
    final_out_ids = []
    for row in rows:
        for nd in row["nodes"]:
            if nd.get("final"):
                final_out_ids.append(nd["id"])

    # Vertical centre across all objective rows
    all_cy = [row["cy"] for row in rows]
    mid_y  = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50

    # Chain:  [Result Collation] → [Results Analysis] →
    #         [Discussion of Results] → [Manuscript]
    stage_labels = [
        ("Result Collation",       "collation"),   # yellow like operation nodes
        ("Results Analysis",       "operation"),
        ("Discussion of Results",  "operation"),
        ("Manuscript",             "final"),
    ]
    stage_nodes = []   # list of {id, cx, cy, type, name}
    stage_edges = []   # (from_id, to_id)
    sx = max_row_width + STAGE_GAP
    for idx, (label, stype) in enumerate(stage_labels):
        sid = f"stage_{idx}"
        if stype == "final":
            cx = sx + STAGE_OVAL_RX
            stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
                                "type": "stage_final", "name": label})
            sx = cx + STAGE_OVAL_RX + STAGE_GAP
        elif stype == "collation":
            cx = sx + STAGE_RECT_W / 2
            stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
                                "type": "stage_collation", "name": label})
            sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
        else:
            cx = sx + STAGE_RECT_W / 2
            stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
                                "type": "stage_op", "name": label})
            sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
        if idx > 0:
            stage_edges.append((f"stage_{idx-1}", sid))

    # Edges from each objective's final output → Result Collation
    for fid in final_out_ids:
        stage_edges.append((fid, "stage_0"))

    svg_width  = max(1200, int(sx) + 60)
    svg_height = max(400, int(y_cursor) + 30)

    # ── Build node lookup & popup-details dict ───────────────────────
    node_lookup = {}
    details = {}

    for row in rows:
        obj     = row["obj"]
        obj_num = obj["number"]
        for nd in row["nodes"]:
            nd["abs_cy"] = row["cy"] + nd["cy_off"]
            node_lookup[nd["id"]] = nd

            if nd["type"] == "input":
                details[nd["id"]] = (
                    f"<h3>Input Data</h3>"
                    f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>")
            elif nd["type"] == "operation":
                d = (f"<h3>Step {obj_num}.{nd['step_num']}: {_html.escape(nd['name'])}</h3>"
                     f"<p><b>Operation Type:</b> {_html.escape(nd.get('op_type',''))}</p>"
                     f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in nd.get('input_nodes',[]))}</p>"
                     f"<p><b>Output:</b> {_html.escape(nd.get('output_node',''))}</p>"
                     f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>")
                if nd.get("expected"):
                    d += f"<p><b>Expected Output:</b> {_html.escape(nd['expected'])}</p>"
                details[nd["id"]] = d
            elif nd["type"] in ("output", "output_ref"):
                label = "Output Data" if nd["type"] == "output" else "Reused Output"
                details[nd["id"]] = (
                    f"<h3>{label}</h3>"
                    f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>"
                    + (f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>"
                       if nd.get("desc") else ""))

    # Register RQ node in lookup + details
    node_lookup[rq_id] = {"id": rq_id, "type": "rq", "name": rq_text,
                          "cx": rq_cx, "abs_cy": rq_cy}
    details[rq_id] = (f"<h3>Research Question</h3>"
                      f"<p>{_html.escape(rq_text)}</p>")

    # Register pipeline stage nodes in lookup + details
    stage_descs = {
        "Result Collation":      "Collate outputs and results from all objectives into a unified dataset for analysis.",
        "Results Analysis":      "Perform statistical analysis, validation, and interpretation of the collated results.",
        "Discussion of Results": "Synthesise findings, compare with hypotheses, and discuss implications.",
        "Manuscript":            "Final research deliverable: paper, report, or document combining all findings.",
    }
    for sn in stage_nodes:
        node_lookup[sn["id"]] = sn
        details[sn["id"]] = (f"<h3>{_html.escape(sn['name'])}</h3>"
                             f"<p>{_html.escape(stage_descs.get(sn['name'], ''))}</p>")

    # Register objective title nodes in lookup + details (clickable with popup)
    for otn in obj_title_nodes:
        node_lookup[otn["id"]] = otn
        obj_num = otn["obj_num"]
        obj_name = otn["name"]
        hypothesis = otn.get("hypothesis", "")
        d = (f"<h3>Objective {obj_num}</h3>"
             f"<p><b>Title:</b> {_html.escape(obj_name)}</p>")
        if hypothesis:
            d += f"<p><b>Hypothesis:</b> {_html.escape(hypothesis)}</p>"
        # Include step summary
        for row in rows:
            if row["obj"].get("number") == obj_num:
                steps = sorted(row["obj"].get("steps", []), key=lambda s: s.get("step_number", 0))
                if steps:
                    d += "<p><b>Steps:</b></p><ol>"
                    for s in steps:
                        d += f"<li><b>{_html.escape(s.get('operation',''))}</b>: {_html.escape(s.get('description',''))}</li>"
                    d += "</ol>"
                break
        details[otn["id"]] = d

    # ── Generate SVG body ────────────────────────────────────────────
    svg = []

    # background bands + objective rectangle nodes
    for i_r, row in enumerate(rows):
        bc = BAND_COLORS[i_r % len(BAND_COLORS)]
        svg.append(f'<rect x="0" y="{row["top"]-5}" width="{svg_width}" '
                   f'height="{row["height"]+10}" fill="{bc}" rx="6"/>')
        obj = row["obj"]
        obj_num = obj["number"]
        oid = f"obj_title_{obj_num}"
        obj_cx = OBJ_LABEL_X + OBJ_NODE_W / 2
        obj_cy = row["cy"]
        rx_ = obj_cx - OBJ_NODE_W / 2
        ry_ = obj_cy - OBJ_NODE_H / 2
        svg.append(f'<g class="node-box" data-nid="{oid}" onclick="showPopup(\'{oid}\')" '
                   f'onmouseenter="hlNode(\'{oid}\')" onmouseleave="ulNode()">')
        svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{OBJ_NODE_W}" height="{OBJ_NODE_H}" '
                   f'rx="{OBJ_NODE_RX}" fill="{OBJ_NODE_FILL}" stroke="{OBJ_NODE_STROKE}" '
                   f'stroke-width="2" data-os="{OBJ_NODE_STROKE}"/>')
        svg.append("  " + wrap_svg_text(f"Obj {obj_num}", obj_cx, obj_cy, 16, fs=14, bold=True, fill="white"))
        svg.append('</g>')

    # ── RQ node (rendered before edges so edges draw on top) ────────
    svg.append(f'<g class="node-box" data-nid="{rq_id}" onclick="showPopup(\'{rq_id}\')" '
               f'onmouseenter="hlNode(\'{rq_id}\')" onmouseleave="ulNode()">')
    rq_rx_ = rq_cx - RQ_RECT_W / 2; rq_ry_ = rq_cy - RQ_RECT_H / 2
    svg.append(f'  <rect x="{rq_rx_}" y="{rq_ry_}" width="{RQ_RECT_W}" '
               f'height="{RQ_RECT_H}" rx="{RQ_RX}" '
               f'fill="{RQ_FILL}" stroke="{RQ_STROKE}" '
               f'stroke-width="2" data-os="{RQ_STROKE}"/>')
    svg.append("  " + wrap_svg_text("Research Question", rq_cx, rq_cy, 20, fs=15, bold=True, fill="white"))
    svg.append('</g>')

    # ── RQ edges (RQ → objective rectangle nodes) ────────────────
    #   Diverging S-curves: exit RQ horizontally, arrive at obj rect left edge
    for fid, tid in rq_edges:
        fn = node_lookup[fid]; tn = node_lookup[tid]
        x1 = fn["cx"] + RQ_RECT_W / 2       # right edge of RQ rect
        y1 = fn["abs_cy"]
        x2 = tn["cx"] - OBJ_NODE_W / 2      # left edge of objective rect
        y2 = tn["abs_cy"]
        # S-curve control points: leave RQ horizontally, arrive at obj rect horizontally
        spread = (x2 - x1) * 0.5
        svg.append(f'<path class="edge-line" data-from="{fid}" data-to="{tid}" '
                   f'd="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
                   f'fill="none" stroke="#555" stroke-width="2" '
                   f'marker-end="url(#mb_arrow)"/>')

    # ── Objective rect → all first-step input nodes ────────────────
    for row in rows:
        obj_num = row["obj"]["number"]
        oid = f"obj_title_{obj_num}"
        otn = node_lookup[oid]
        # Find all input nodes that appear before the first operation
        first_inputs = [nd for nd in row["nodes"] if nd["type"] == "input"
                        and nd["id"].startswith(f"o{obj_num}_s")]
        # Only take inputs from the very first step
        first_step_inputs = []
        if first_inputs:
            first_step_prefix = first_inputs[0]["id"].rsplit("_i", 1)[0]  # e.g. "o1_s1"
            first_step_inputs = [nd for nd in first_inputs
                                 if nd["id"].startswith(first_step_prefix + "_i")]
        targets = first_step_inputs if first_step_inputs else row["nodes"][:1]
        x1 = otn["cx"] + OBJ_NODE_W / 2       # right edge of obj rect
        y1 = otn["abs_cy"]
        for tgt in targets:
            if tgt["type"] == "input":
                x2 = tgt["cx"] - OVAL_RX
            else:
                x2 = tgt["cx"] - RECT_W / 2
            y2 = tgt["abs_cy"]
            if abs(y1 - y2) > 5:
                spread = (x2 - x1) * 0.5
                svg.append(f'<path class="edge-line" data-from="{oid}" data-to="{tgt["id"]}" '
                           f'd="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
                           f'fill="none" stroke="#555" stroke-width="2" '
                           f'marker-end="url(#mb_arrow)"/>')
            else:
                svg.append(f'<line class="edge-line" data-from="{oid}" data-to="{tgt["id"]}" '
                           f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                           f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')

    # edges (drawn first so nodes paint over them)
    for row in rows:
        for fid, tid, etype in row["edges"]:
            fn = node_lookup[fid]; tn = node_lookup[tid]

            if etype == "reference":
                x1 = fn["cx"]
                y1 = fn["abs_cy"] - OVAL_RY
                x2 = tn["cx"]
                y2 = tn["abs_cy"] - RECT_H / 2
                distance = abs(x2 - x1)
                arc_h = max(45, distance * 0.25)
                ctrl_y = min(y1, y2) - arc_h
                mx = (x1 + x2) / 2
                svg.append(f'<path class="edge-line" data-from="{fid}" data-to="{tid}" '
                           f'd="M{x1},{y1} Q{mx},{ctrl_y} {x2},{y2}" '
                           f'fill="none" stroke="#3D8B3D" stroke-width="2" '
                           f'stroke-dasharray="6,3" marker-end="url(#mb_arrow_green)"/>')
            else:
                x1 = fn["cx"] + (OVAL_RX if fn["type"] != "operation" else RECT_W / 2)
                y1 = fn["abs_cy"]
                x2 = tn["cx"] - (OVAL_RX if tn["type"] != "operation" else RECT_W / 2)
                y2 = tn["abs_cy"]

                if abs(y1 - y2) > 5:
                    mx = (x1 + x2) / 2
                    svg.append(f'<path class="edge-line" data-from="{fid}" data-to="{tid}" '
                               f'd="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
                               f'fill="none" stroke="#555" stroke-width="2" '
                               f'marker-end="url(#mb_arrow)"/>')
                else:
                    svg.append(f'<line class="edge-line" data-from="{fid}" data-to="{tid}" '
                               f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                               f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')

    # nodes
    for row in rows:
        for nd in row["nodes"]:
            cx = nd["cx"]; cy = nd["abs_cy"]; nid = nd["id"]
            svg.append(f'<g class="node-box" data-nid="{nid}" onclick="showPopup(\'{nid}\')" '
                       f'onmouseenter="hlNode(\'{nid}\')" onmouseleave="ulNode()">')

            if nd["type"] == "input":
                svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
                           f'fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="1.5" '
                           f'data-os="{INPUT_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))

            elif nd["type"] == "operation":
                rx = cx - RECT_W / 2; ry_ = cy - RECT_H / 2
                svg.append(f'  <rect x="{rx}" y="{ry_}" width="{RECT_W}" height="{RECT_H}" '
                           f'rx="{RECT_RX}" fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
                           f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 18, bold=True, fill="#333"))

            elif nd["type"] == "output":
                orx = OVAL_RX + (5 if nd.get("final") else 0)
                ory = OVAL_RY + (3 if nd.get("final") else 0)
                sw  = "2.5" if nd.get("final") else "1.5"
                svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{orx}" ry="{ory}" '
                           f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="{sw}" '
                           f'data-os="{OUTPUT_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))

            elif nd["type"] == "output_ref":
                # Reused output — green oval placed above the operation
                svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
                           f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="1.5" '
                           f'data-os="{OUTPUT_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))

            svg.append('</g>')

    # ── Cross-objective reuse edges (rendered after nodes so arrowheads stay visible) ──
    # Group by target so multiple arrows landing on the same operation spread out
    from collections import defaultdict
    cross_by_target = defaultdict(list)
    for fid, tid in cross_obj_edges:
        cross_by_target[tid].append(fid)

    for tid, fids in cross_by_target.items():
        tn = node_lookup[tid]
        n = len(fids)
        spread = min(RECT_W * 0.6, n * 30)            # total spread width
        for k, fid in enumerate(fids):
            fn = node_lookup[fid]
            x1 = fn["cx"]                              # bottom-center of source oval
            y1 = fn["abs_cy"] + OVAL_RY
            # Spread landing points evenly across the top of the operation
            offset = (k - (n - 1) / 2) * (spread / max(n - 1, 1)) if n > 1 else 0
            x2 = tn["cx"] + offset                     # spread across top edge
            y2 = tn["abs_cy"] - RECT_H / 2
            # Last control point directly above endpoint so arrow arrives vertically
            svg.append(f'<path class="edge-line" data-from="{fid}" data-to="{tid}" '
                       f'd="M{x1},{y1} C{x1},{(y1+y2)/2} {x2},{y2 - 40} {x2},{y2}" '
                       f'fill="none" stroke="#E67E22" stroke-width="2" '
                       f'stroke-dasharray="8,4" marker-end="url(#mb_arrow_orange)"/>')

    # ── Pipeline stage edges ─────────────────────────────────────────
    for fid, tid in stage_edges:
        fn = node_lookup[fid]; tn = node_lookup[tid]
        if fn.get("type") in ("output", "stage_final"):
            x1 = fn["cx"] + (STAGE_OVAL_RX if fn["type"] == "stage_final" else OVAL_RX)
        elif fn.get("type") in ("stage_op", "stage_collation"):
            x1 = fn["cx"] + STAGE_RECT_W / 2
        else:
            x1 = fn["cx"] + (OVAL_RX if fn.get("type") != "operation" else RECT_W / 2)
        y1 = fn["abs_cy"]
        if tn.get("type") == "stage_final":
            x2 = tn["cx"] - STAGE_OVAL_RX
        elif tn.get("type") in ("stage_op", "stage_collation"):
            x2 = tn["cx"] - STAGE_RECT_W / 2
        else:
            x2 = tn["cx"] - OVAL_RX
        y2 = tn["abs_cy"]

        if abs(y1 - y2) > 5:
            mx = (x1 + x2) / 2
            svg.append(f'<path class="edge-line" data-from="{fid}" data-to="{tid}" '
                       f'd="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
                       f'fill="none" stroke="#555" stroke-width="2" '
                       f'marker-end="url(#mb_arrow)"/>')
        else:
            svg.append(f'<line class="edge-line" data-from="{fid}" data-to="{tid}" '
                       f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                       f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')

    # ── Pipeline stage nodes ─────────────────────────────────────────
    for sn in stage_nodes:
        cx = sn["cx"]; cy = sn["abs_cy"]; sid = sn["id"]
        svg.append(f'<g class="node-box" data-nid="{sid}" onclick="showPopup(\'{sid}\')" '
                   f'onmouseenter="hlNode(\'{sid}\')" onmouseleave="ulNode()">')
        if sn["type"] == "stage_collation":
            rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
            svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
                       f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
                       f'fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
                       f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
            svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="#333"))
        elif sn["type"] == "stage_op":
            rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
            svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
                       f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
                       f'fill="{PIPELINE_FILL}" stroke="{PIPELINE_STROKE}" '
                       f'stroke-width="2" data-os="{PIPELINE_STROKE}"/>')
            svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="white"))
        elif sn["type"] == "stage_final":
            svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{STAGE_OVAL_RX}" '
                       f'ry="{STAGE_OVAL_RY}" fill="{MANUSCRIPT_FILL}" '
                       f'stroke="{MANUSCRIPT_STROKE}" stroke-width="2.5" '
                       f'data-os="{MANUSCRIPT_STROKE}"/>')
            svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 18, fs=14, bold=True, fill="white"))
        svg.append('</g>')

    svg_body = "\n    ".join(svg)

    # ── Assemble HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_html.escape(plan_title)}</title>
<style>
  body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
  h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
  .graph-panel {{ flex:1; overflow:auto; position:relative; cursor:grab; }}
  svg {{ display:block; width:100%; height:100%; min-width:{svg_width}px; min-height:{svg_height}px; }}
  .node-box {{ cursor:pointer; transition:opacity .2s, filter .2s; }}
  .node-box:hover {{ filter:brightness(1.12) drop-shadow(0 0 4px rgba(0,0,0,.3)); }}
  .node-box.dimmed {{ opacity:0.25; }}
  .edge-line {{ stroke-opacity:0.3; transition:stroke-opacity .2s, stroke-width .2s; }}
  .edge-line.highlighted {{ stroke-opacity:1 !important; stroke-width:3 !important; }}
  .node-tooltip {{ pointer-events:none; position:absolute; background:#2C3E50; color:#fff; padding:5px 10px; border-radius:5px; font-size:12px; white-space:nowrap; z-index:60; box-shadow:0 2px 8px rgba(0,0,0,.2); display:none; }}
  .zoom-toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
  .zoom-btn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; color:#444; transition:all .15s; padding:0; }}
  .zoom-btn:hover {{ background:#e9ecef; }}
  .zoom-level {{ font-size:10px; text-align:center; color:#888; padding:2px 0; user-select:none; }}
  .zoom-divider {{ height:1px; background:#e0e0e0; margin:2px 0; }}
  .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
  .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
  .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
  .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
  .popup-overlay.visible {{ display:flex; }}
  .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
  @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
  .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
  .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
  .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
  .popup-close:hover {{ background:#f0f0f0; color:#333; }}
</style>
</head>
<body>
<h1>{_html.escape(plan_title)}</h1>
<div class="graph-panel" id="graphPanel">
  <div class="zoom-toolbar">
    <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
    <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
    <div class="zoom-divider"></div>
    <span class="zoom-level" id="zoomLevel">100%</span>
    <div class="zoom-divider"></div>
    <button class="zoom-btn" onclick="zoomReset()" title="Reset">&#8634;</button>
    <button class="zoom-btn" onclick="zoomFit()" title="Fit">&#11036;</button>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:2px"></div> Operation / Tool</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Output Data</div>
    <div class="legend-item"><div class="legend-swatch" style="border:2px dashed #3D8B3D;background:transparent;border-radius:0"></div> Reused (within obj)</div>
    <div class="legend-item"><div class="legend-swatch" style="border:2px dashed #E67E22;background:transparent;border-radius:0"></div> Reused (cross obj)</div>
  </div>
  <svg id="flowchartSvg" viewBox="0 0 {svg_width} {svg_height}">
    <defs>
      <marker id="mb_arrow" markerWidth="14" markerHeight="12" refX="13" refY="6" orient="auto" markerUnits="userSpaceOnUse">
        <polygon points="0 0,14 6,0 12" fill="#555"/>
      </marker>
      <marker id="mb_arrow_green" markerWidth="14" markerHeight="12" refX="13" refY="6" orient="auto" markerUnits="userSpaceOnUse">
        <polygon points="0 0,14 6,0 12" fill="#3D8B3D"/>
      </marker>
      <marker id="mb_arrow_orange" markerWidth="14" markerHeight="12" refX="13" refY="6" orient="auto" markerUnits="userSpaceOnUse">
        <polygon points="0 0,14 6,0 12" fill="#E67E22"/>
      </marker>
    </defs>
    <g id="zoomGroup">
    {svg_body}
    </g>
  </svg>
</div>

<div class="node-tooltip" id="nodeTooltip"></div>
<div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
  <div class="popup-box" onclick="event.stopPropagation()">
    <button class="popup-close" onclick="closePopup()">&times;</button>
    <div id="popupContent"></div>
  </div>
</div>

<script>
const details = """ + json.dumps(details) + """;
const nodeNames = """ + json.dumps({nid: nd.get("name","") for nid, nd in node_lookup.items()}) + """;

let scale=1, panX=0, panY=0, isPanning=false, _wasDragged=false, psX=0, psY=0, _mdX=0, _mdY=0;
const PT=4;
const MN=0.2, MX=5, ZS=0.15;
const svg=document.getElementById('flowchartSvg');
const zg=document.getElementById('zoomGroup');
const gp=document.getElementById('graphPanel');
const tooltip=document.getElementById('nodeTooltip');

function applyTransform(){
  zg.setAttribute('transform','translate('+panX+','+panY+') scale('+scale+')');
  document.getElementById('zoomLevel').textContent=Math.round(scale*100)+'%';
}
function zoomIn(){ scale=Math.min(MX,scale+ZS); applyTransform(); }
function zoomOut(){ scale=Math.max(MN,scale-ZS); applyTransform(); }
function zoomReset(){ scale=1;panX=0;panY=0;applyTransform(); }
function zoomFit(){
  var b=zg.getBBox(),v=svg.viewBox.baseVal,p=30;
  scale=Math.min((v.width-p*2)/b.width,(v.height-p*2)/b.height,MX);
  panX=(v.width/scale-b.width)/2-b.x+p/scale;
  panY=(v.height/scale-b.height)/2-b.y+p/scale;
  applyTransform();
}

gp.addEventListener('wheel',function(e){
  e.preventDefault();
  var r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
  var sx=((e.clientX-r.left)/r.width)*v.width, sy=((e.clientY-r.top)/r.height)*v.height;
  var os=scale;
  scale=e.deltaY<0?Math.min(MX,scale+ZS):Math.max(MN,scale-ZS);
  panX=sx/scale-sx/os+panX; panY=sy/scale-sy/os+panY; applyTransform();
},{passive:false});

gp.addEventListener('mousedown',function(e){
  if(e.button===0||e.button===1){
    e.preventDefault();isPanning=true;_wasDragged=false;
    psX=e.clientX;psY=e.clientY;_mdX=e.clientX;_mdY=e.clientY;
    gp.style.cursor='grabbing';
  }
});
window.addEventListener('mousemove',function(e){
  if(!isPanning)return;
  var dist=Math.abs(e.clientX-_mdX)+Math.abs(e.clientY-_mdY);
  if(dist>PT)_wasDragged=true;
  var v=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
  panX+=(e.clientX-psX)*(v.width/r.width)/scale;
  panY+=(e.clientY-psY)*(v.height/r.height)/scale;
  psX=e.clientX;psY=e.clientY;applyTransform();
});
window.addEventListener('mouseup',function(){ if(isPanning){isPanning=false;gp.style.cursor='';} });
gp.addEventListener('click',function(e){ if(_wasDragged){e.stopPropagation();_wasDragged=false;} },true);

// ── Hover highlight: direct connections only ──
function getConnected(nid){
  var edges=document.querySelectorAll('.edge-line');
  var visited={};
  var connEdges=[];
  visited[nid]=true;
  edges.forEach(function(e){
    var f=e.getAttribute('data-from'), t=e.getAttribute('data-to');
    if(f===nid){ connEdges.push(e); visited[t]=true; }
    if(t===nid){ connEdges.push(e); visited[f]=true; }
  });
  return {nodes:visited, edges:connEdges};
}

function hlNode(nid){
  var conn=getConnected(nid);
  // Highlight connected edges
  document.querySelectorAll('.edge-line').forEach(function(e){
    if(conn.edges.indexOf(e)>=0) e.classList.add('highlighted');
  });
  // Dim unconnected nodes
  document.querySelectorAll('.node-box').forEach(function(g){
    var id=g.getAttribute('data-nid');
    if(id && !conn.nodes[id]) g.classList.add('dimmed');
  });
  // Tooltip
  var name=nodeNames[nid]||'';
  if(name){
    tooltip.textContent=name;
    tooltip.style.display='block';
  }
}

function ulNode(){
  document.querySelectorAll('.edge-line.highlighted').forEach(function(e){ e.classList.remove('highlighted'); });
  document.querySelectorAll('.node-box.dimmed').forEach(function(g){ g.classList.remove('dimmed'); });
  tooltip.style.display='none';
}

// Position tooltip near cursor
document.addEventListener('mousemove',function(e){
  if(tooltip.style.display==='block'){
    tooltip.style.left=(e.clientX+12)+'px';
    tooltip.style.top=(e.clientY-28)+'px';
  }
});

function showPopup(nodeId){
  var c=document.getElementById('popupContent');
  var o=document.getElementById('popupOverlay');
  c.innerHTML=details[nodeId]||'<p style="color:#999">No details.</p>';
  o.classList.add('visible');
  document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
    el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
    el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
  });
  var cl=event.currentTarget.querySelector('ellipse,rect');
  if(cl){ cl.setAttribute('stroke-width','3.5'); cl.setAttribute('stroke','#FFD700'); }
}
function closePopup(e){
  if(e&&e.target!==document.getElementById('popupOverlay'))return;
  document.getElementById('popupOverlay').classList.remove('visible');
  document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
    el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
    el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
  });
}
document.addEventListener('keydown',function(e){ if(e.key==='Escape')closePopup(); });
window.addEventListener('DOMContentLoaded',function(){ setTimeout(zoomFit,100); });
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Model Builder workflow saved to: {output_path}")
    return output_path




def _layered_dag_layout(G, col_spacing=280, row_spacing=110, sweep_iterations=24):
    """Compute a Sugiyama-style layered layout with barycenter sweep.

    Returns ``(node_level, positions, canvas_w, canvas_h)`` where
    ``positions[node] = (x, y)``. X is fixed by topological level; Y is
    ordered to minimize edge crossings by sweeping the barycenter
    heuristic forward and backward across levels — so the initial render
    is already crossing-minimal (no force simulation needed).
    """
    import networkx as nx
    from collections import defaultdict
    from statistics import mean

    # X coordinate = topological level (longest path from any root)
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo = list(G.nodes())

    node_level = {}
    for n in topo:
        preds = list(G.predecessors(n))
        node_level[n] = 0 if not preds else max(node_level.get(p, 0) for p in preds) + 1
    max_level = max(node_level.values()) if node_level else 0

    # Group nodes by level, in topological order
    levels = defaultdict(list)
    for n in topo:
        levels[node_level[n]].append(n)
    sorted_levels = sorted(levels.keys())

    # Initial row index within each level (stable from topo order)
    y_index = {}
    for lvl in sorted_levels:
        for i, n in enumerate(levels[lvl]):
            y_index[n] = float(i)

    # Barycenter sweep: alternate forward (order by predecessors) and
    # backward (order by successors) passes until ordering stabilizes.
    for _ in range(sweep_iterations):
        for lvl in sorted_levels:
            if lvl == sorted_levels[0]:
                continue
            ordered = sorted(
                levels[lvl],
                key=lambda n: (
                    mean(y_index[p] for p in G.predecessors(n))
                    if list(G.predecessors(n))
                    else y_index[n]
                ),
            )
            levels[lvl] = ordered
            for i, n in enumerate(ordered):
                y_index[n] = float(i)
        for lvl in reversed(sorted_levels):
            if lvl == sorted_levels[-1]:
                continue
            ordered = sorted(
                levels[lvl],
                key=lambda n: (
                    mean(y_index[s] for s in G.successors(n))
                    if list(G.successors(n))
                    else y_index[n]
                ),
            )
            levels[lvl] = ordered
            for i, n in enumerate(ordered):
                y_index[n] = float(i)

    # Canvas dimensions
    max_per_level = max(len(levels[lvl]) for lvl in sorted_levels) if sorted_levels else 1
    canvas_w = max(1200, (max_level + 1) * col_spacing + 200)
    canvas_h = max(500, max_per_level * row_spacing + 120)

    # Final pixel positions — center each level vertically
    positions = {}
    for lvl in sorted_levels:
        members = levels[lvl]
        count = len(members)
        total_h = (count - 1) * row_spacing
        start_y = (canvas_h - total_h) / 2
        for i, n in enumerate(members):
            positions[n] = (140 + lvl * col_spacing, start_y + i * row_spacing)

    return node_level, positions, canvas_w, canvas_h


def show_dag_geoprocessing_workflow(G, output_path="Outputs/geoprocessing_workflow_dag.html",
                                    title="Geoprocessing Workflow", research_question=""):
    """Render a NetworkX DiGraph as an interactive 2D DAG in the style of
    ArcGIS Model Builder.

    Node positions are computed by a layered (Sugiyama-style) layout with
    a barycenter sweep — so the initial render is already crossing-minimal
    instead of relying on a force simulation that lets branches cross each
    other. Nodes stay draggable for fine-tuning; zoom, pan, fit-to-view,
    reset-to-layout, and click-to-popup are preserved.

    Node colours follow the Model Builder palette:
      - Blue ovals:   input data (root data nodes)
      - Yellow rects: operations
      - Green ovals:  intermediate / output data
      - Purple ovals: final / terminal output data

    Args:
        G: NetworkX DiGraph with node_type="data"|"operation" attributes
        output_path: where to write the HTML file
        title: displayed title
        research_question: optional RQ text shown in popup

    Returns:
        Path to saved HTML file
    """
    import json
    import os
    import html as _html

    # ── Layered layout with barycenter sweep (crossing-minimal) ──────
    col_spacing = 280
    row_spacing = 110
    node_level, positions, canvas_w, canvas_h = _layered_dag_layout(
        G, col_spacing=col_spacing, row_spacing=row_spacing,
    )
    max_level = max(node_level.values()) if node_level else 0

    # ── Classify nodes ───────────────────────────────────────────────
    root_data = set()
    final_data = set()
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "data":
            if G.in_degree(n) == 0:
                root_data.add(n)
            if G.out_degree(n) == 0:
                final_data.add(n)

    # ── Build JSON data for D3 ───────────────────────────────────────
    INPUT_FILL   = "#6BA3D6"; INPUT_STROKE  = "#2F74B5"
    TOOL_FILL    = "#F5D576"; TOOL_STROKE   = "#C4A030"
    OUTPUT_FILL  = "#6BB56A"; OUTPUT_STROKE = "#3D8B3D"
    FINAL_OUTPUT_FILL = "#9B6DBF"; FINAL_OUTPUT_STROKE = "#6C3483"   # purple — terminal output node(s)
    FONT = "Segoe UI, Arial, sans-serif"

    node_id_map = {n: i for i, n in enumerate(G.nodes())}
    nodes_json = []
    details = {}

    for n, d in G.nodes(data=True):
        nt = d.get("node_type", "data")
        desc = d.get("description", n)
        data_path = d.get("data_path", "")
        is_input = n in root_data
        is_final = n in final_data
        level = node_level.get(n, 0)

        if nt == "operation":
            fill = TOOL_FILL; stroke = TOOL_STROKE
            shape = "rect"
        elif is_input:
            fill = INPUT_FILL; stroke = INPUT_STROKE
            shape = "ellipse"
        elif is_final:
            # Terminal data node(s) — distinguish from intermediate outputs
            fill = FINAL_OUTPUT_FILL; stroke = FINAL_OUTPUT_STROKE
            shape = "ellipse"
        else:
            fill = OUTPUT_FILL; stroke = OUTPUT_STROKE
            shape = "ellipse"

        x, y = positions[n]
        nodes_json.append({
            "id": n, "idx": node_id_map[n],
            "level": level, "shape": shape,
            "fill": fill, "stroke": stroke,
            "label": desc if len(desc) < 28 else (desc[:25] + "..."),
            "isFinal": is_final,
            "x": x, "y": y,
        })

        # Popup details
        if nt == "operation":
            inp = [p for p in G.predecessors(n)]
            out = [s for s in G.successors(n)]
            details[n] = (f"<h3>{_html.escape(desc)}</h3>"
                          f"<p><b>Type:</b> Operation</p>"
                          f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in inp)}</p>"
                          f"<p><b>Outputs:</b> {', '.join(_html.escape(o) for o in out)}</p>")
        else:
            kind = "Input Data" if is_input else ("Final Output" if is_final else "Intermediate Data")
            det = f"<h3>{_html.escape(n)}</h3><p><b>Type:</b> {kind}</p>"
            if desc: det += f"<p><b>Description:</b> {_html.escape(desc)}</p>"
            if data_path: det += f"<p><b>Path:</b> <code>{_html.escape(data_path)}</code></p>"
            details[n] = det

    links_json = [{"source": node_id_map[u], "target": node_id_map[v]}
                  for u, v in G.edges()]

    # ── HTML with pre-computed layered (crossing-minimal) layout ─────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_html.escape(title)}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
  h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
  #graph {{ flex:1; overflow:hidden; }}
  svg {{ width:100%; height:100%; }}
  .link {{ fill:none; stroke:#555; stroke-width:2; }}
  .node {{ cursor:grab; }}
  .node:active {{ cursor:grabbing; }}
  .node:hover {{ filter:brightness(1.12) drop-shadow(0 0 6px rgba(0,0,0,.3)); }}
  .node-label {{ pointer-events:none; font-family:{FONT}; font-size:12px; font-weight:bold; text-anchor:middle; }}
  .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
  .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
  .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
  .toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
  .tbtn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:15px; display:flex; align-items:center; justify-content:center; color:#444; padding:0; }}
  .tbtn:hover {{ background:#e9ecef; }}
  .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
  .popup-overlay.visible {{ display:flex; }}
  .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
  @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
  .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
  .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
  .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
  .popup-close:hover {{ background:#f0f0f0; color:#333; }}
</style>
</head>
<body>
<h1>{_html.escape(title)}</h1>
<div id="graph" style="position:relative">
  <div class="toolbar">
    <button class="tbtn" id="zoomInBtn" title="Zoom in">+</button>
    <button class="tbtn" id="zoomOutBtn" title="Zoom out">&minus;</button>
    <button class="tbtn" id="fitBtn" title="Fit to view">&#9974;</button>
    <button class="tbtn" id="resetBtn" title="Reset layout">&#8634;</button>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:3px"></div> Operation / Tool</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Intermediate Data</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{FINAL_OUTPUT_FILL};border-radius:8px"></div> Final Output</div>
  </div>
</div>

<div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
  <div class="popup-box" onclick="event.stopPropagation()">
    <button class="popup-close" onclick="closePopup()">&times;</button>
    <div id="popupContent"></div>
  </div>
</div>

<script>
const nodesData = {json.dumps(nodes_json)};
const linksData = {json.dumps(links_json)};
const details   = {json.dumps(details)};

const W = {canvas_w}, H = {canvas_h};
const RW = 170, RH = 56, RX = 8;          // rect (operation)
const EX = 80,  EY = 28;                  // ellipse radii (data)

// Save the original layered-layout positions so the user can reset
// after dragging.
const ORIGINAL_POS = new Map(nodesData.map(d => [d.idx, {{x: d.x, y: d.y}}]));

const svg = d3.select('#graph').append('svg')
  .attr('viewBox', [0, 0, W, H]);

// Zoom
const zoomG = svg.append('g');
const zoomBehavior = d3.zoom()
  .scaleExtent([0.15, 5])
  .on('zoom', (e) => zoomG.attr('transform', e.transform));
svg.call(zoomBehavior);

// Arrow marker
svg.append('defs').append('marker')
  .attr('id', 'arrow').attr('viewBox', '0 0 10 7')
  .attr('refX', 10).attr('refY', 3.5)
  .attr('markerWidth', 10).attr('markerHeight', 7)
  .attr('orient', 'auto')
  .append('polygon').attr('points', '0 0,10 3.5,0 7').attr('fill', '#555');

// Resolve link endpoints to actual node objects (JSON encodes them as indices).
linksData.forEach(l => {{
  l.sourceNode = nodesData[l.source];
  l.targetNode = nodesData[l.target];
}});

function nodeAnchor(d, side) {{
  // (x, y) where an edge attaches to node d on the given side.
  if (d.shape === 'rect') {{
    return {{ x: d.x + (side === 'right' ? RW/2 : -RW/2), y: d.y }};
  }}
  return {{ x: d.x + (side === 'right' ? EX : -EX), y: d.y }};
}}

function edgePath(l) {{
  const s = nodeAnchor(l.sourceNode, 'right');
  const t = nodeAnchor(l.targetNode, 'left');
  const mx = (s.x + t.x) / 2;
  if (Math.abs(s.y - t.y) > 5) {{
    return `M${{s.x}},${{s.y}} C${{mx}},${{s.y}} ${{mx}},${{t.y}} ${{t.x}},${{t.y}}`;
  }}
  return `M${{s.x}},${{s.y}} L${{t.x}},${{t.y}}`;
}}

// Flag set by the drag handler when the user actually moved a node; the
// click handler reads it to avoid opening the popup at the end of a drag.
let dragSuppressClick = false;

// Links
const link = zoomG.append('g').selectAll('path')
  .data(linksData).join('path')
  .attr('class', 'link')
  .attr('marker-end', 'url(#arrow)')
  .attr('d', edgePath);

// Node groups — placed directly at their pre-computed layered positions.
const node = zoomG.append('g').selectAll('g')
  .data(nodesData).join('g')
  .attr('class', 'node')
  .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
  .on('click', (e, d) => {{
    if (dragSuppressClick) {{ dragSuppressClick = false; return; }}
    showPopup(d.id);
  }});

// Draw shapes
node.each(function(d) {{
  const g = d3.select(this);
  if (d.shape === 'rect') {{
    g.append('rect')
      .attr('width', RW).attr('height', RH)
      .attr('x', -RW/2).attr('y', -RH/2)
      .attr('rx', RX)
      .attr('fill', d.fill).attr('stroke', d.stroke)
      .attr('stroke-width', 2);
    g.append('text').attr('class', 'node-label')
      .attr('y', 4).attr('fill', '#333').text(d.label);
  }} else {{
    const rx = d.isFinal ? EX+5 : EX;
    const ry = d.isFinal ? EY+3 : EY;
    g.append('ellipse')
      .attr('rx', rx).attr('ry', ry)
      .attr('fill', d.fill).attr('stroke', d.stroke)
      .attr('stroke-width', d.isFinal ? 2.5 : 1.5);
    g.append('text').attr('class', 'node-label')
      .attr('y', 4).attr('fill', 'white').text(d.label);
  }}
}});

// Drag: free movement. Use a regular function so `this` is the dragged
// SVG group; only redraw the incident edges (not every edge).
const drag = d3.drag()
  .on('start', function(e, d) {{
    dragSuppressClick = false;
    d3.select(this).raise();
    d._dragStartX = d.x; d._dragStartY = d.y;
  }})
  .on('drag', function(e, d) {{
    d.x = e.x; d.y = e.y;
    d3.select(this).attr('transform', `translate(${{d.x}},${{d.y}})`);
    link.filter(l => l.sourceNode === d || l.targetNode === d).attr('d', edgePath);
  }})
  .on('end', function(e, d) {{
    if (Math.hypot(d.x - d._dragStartX, d.y - d._dragStartY) > 2) {{
      dragSuppressClick = true;
    }}
  }});
node.call(drag);

function fitToView() {{
  const b = zoomG.node().getBBox();
  const fw = W / (b.width  + 80);
  const fh = H / (b.height + 80);
  const s = Math.min(fw, fh, 1.6);
  const tx = (W - b.width  * s) / 2 - b.x * s;
  const ty = (H - b.height * s) / 2 - b.y * s;
  svg.transition().duration(400).call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(s));
}}

// Toolbar
d3.select('#zoomInBtn').on('click', () => svg.transition().call(zoomBehavior.scaleBy, 1.3));
d3.select('#zoomOutBtn').on('click', () => svg.transition().call(zoomBehavior.scaleBy, 0.7));
d3.select('#fitBtn').on('click', fitToView);
d3.select('#resetBtn').on('click', () => {{
  // Snap every node back to its layered-layout position and re-fit.
  nodesData.forEach(d => {{
    const p = ORIGINAL_POS.get(d.idx);
    if (p) {{ d.x = p.x; d.y = p.y; }}
  }});
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
  link.attr('d', edgePath);
  fitToView();
}});

// Auto-fit on first paint
requestAnimationFrame(fitToView);

// Popup
function showPopup(nodeId) {{
  document.getElementById('popupContent').innerHTML = details[nodeId] || '<p>No details.</p>';
  document.getElementById('popupOverlay').classList.add('visible');
}}
function closePopup(e) {{
  if (e && e.target !== document.getElementById('popupOverlay')) return;
  document.getElementById('popupOverlay').classList.remove('visible');
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closePopup(); }});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"DAG workflow saved to: {output_path}")
    return output_path


# def show_dag_geoprocessing_workflow(G, output_path="Outputs/geoprocessing_workflow_dag.html",
#                                     title="Geoprocessing Workflow", research_question=""):
#     """Render a NetworkX DiGraph as a true 2D DAG in the style of ArcGIS Model Builder.
#
#     Nodes are positioned using a layered (Sugiyama-style) layout:
#       - X position = topological level (longest path from any root)
#       - Y position = median heuristic within each level to reduce crossings
#
#     Node colours follow the Model Builder palette:
#       - Blue ovals:   input data (root data nodes with a data_path)
#       - Yellow rects: operations
#       - Green ovals:  intermediate / output data
#
#     Args:
#         G: NetworkX DiGraph with node_type="data"|"operation" attributes
#         output_path: where to write the HTML file
#         title: displayed title
#         research_question: optional RQ text shown in popup
#
#     Returns:
#         Path to saved HTML file
#     """
#     import networkx as nx
#     import json
#     import os
#     import html as _html
#
#     # ── Palette (same as Model Builder) ──────────────────────────────
#     INPUT_FILL   = "#6BA3D6"; INPUT_STROKE  = "#2F74B5"
#     TOOL_FILL    = "#F5D576"; TOOL_STROKE   = "#C4A030"
#     OUTPUT_FILL  = "#6BB56A"; OUTPUT_STROKE = "#3D8B3D"
#     FONT = "Segoe UI, Arial, sans-serif"
#
#     # ── Dimensions ───────────────────────────────────────────────────
#     OVAL_RX = 85;  OVAL_RY = 32
#     RECT_W  = 180; RECT_H  = 62; RECT_RX = 8
#     COL_GAP = 100          # horizontal gap between level columns
#     ROW_GAP = 30           # vertical gap between nodes in same column
#     PAD     = 60           # canvas padding
#
#     # ── Topological sort & level assignment ──────────────────────────
#     try:
#         topo = list(nx.topological_sort(G))
#     except nx.NetworkXUnfeasible:
#         topo = list(G.nodes())
#
#     node_level = {}
#     for n in topo:
#         preds = list(G.predecessors(n))
#         node_level[n] = 0 if not preds else max(node_level.get(p, 0) for p in preds) + 1
#
#     # Group nodes by level
#     levels = {}
#     for n, lv in node_level.items():
#         levels.setdefault(lv, []).append(n)
#
#     max_level = max(levels.keys()) if levels else 0
#
#     def _node_height(n):
#         nt = G.nodes[n].get("node_type", "data")
#         return RECT_H if nt == "operation" else OVAL_RY * 2
#
#     # ── Phase 1: Branch-aware initial ordering ───────────────────────
#     # Trace each root forward via BFS to assign a "lane" (branch index).
#     # Nodes reachable from the same root stay in adjacent Y positions.
#     # Convergence nodes (multiple predecessors from different branches)
#     # get placed between their input branches.
#     roots = [n for n in topo if G.in_degree(n) == 0]
#     node_branch = {}   # node -> float (branch lane, can be fractional for convergence)
#     for bi, root in enumerate(roots):
#         for n in nx.bfs_tree(G, root):
#             if n not in node_branch:
#                 node_branch[n] = float(bi)
#
#     # Re-assign convergence nodes to the average lane of their predecessors
#     for n in topo:
#         preds = list(G.predecessors(n))
#         if len(preds) > 1:
#             pred_lanes = [node_branch.get(p, 0.0) for p in preds]
#             node_branch[n] = sum(pred_lanes) / len(pred_lanes)
#             # Also shift all descendants toward this averaged lane
#             for desc in nx.descendants(G, n):
#                 if desc not in node_branch or G.in_degree(desc) <= 1:
#                     node_branch[desc] = node_branch[n]
#
#     # Sort each level by branch lane
#     for lv in levels:
#         levels[lv].sort(key=lambda n: (node_branch.get(n, 0), n))
#
#     # ── Phase 2: Initial Y coordinate assignment ─────────────────────
#     node_pos = {}  # node -> (cx, cy)
#     for lv in range(max_level + 1):
#         nodes = levels.get(lv, [])
#         col_cx = PAD + lv * (RECT_W + COL_GAP) + RECT_W / 2
#         y = PAD
#         for n in nodes:
#             h = _node_height(n)
#             node_pos[n] = (col_cx, y + h / 2)
#             y += h + ROW_GAP
#
#     # ── Phase 3: Barycenter refinement (many passes) ─────────────────
#     # Forward pass: position node at weighted average of predecessors
#     # Backward pass: position node at weighted average of successors
#     # This iteratively pulls connected nodes toward each other.
#     def _enforce_min_gap(level_nodes):
#         """Push nodes apart so they don't overlap, preserving relative order."""
#         if len(level_nodes) < 2:
#             return
#         cx = node_pos[level_nodes[0]][0]
#         # Sort by current Y
#         ordered = sorted(level_nodes, key=lambda n: node_pos[n][1])
#         # Forward: push down if too close
#         for i in range(1, len(ordered)):
#             prev = ordered[i - 1]
#             curr = ordered[i]
#             min_y = node_pos[prev][1] + _node_height(prev) / 2 + ROW_GAP + _node_height(curr) / 2
#             if node_pos[curr][1] < min_y:
#                 node_pos[curr] = (cx, min_y)
#         # Backward: push up if too close (balances the layout)
#         for i in range(len(ordered) - 2, -1, -1):
#             above = ordered[i]
#             below = ordered[i + 1]
#             max_y = node_pos[below][1] - _node_height(below) / 2 - ROW_GAP - _node_height(above) / 2
#             if node_pos[above][1] > max_y:
#                 node_pos[above] = (node_pos[above][0], max_y)
#
#     for iteration in range(24):
#         # Alternate forward and backward sweeps
#         if iteration % 2 == 0:
#             sweep = range(1, max_level + 1)       # forward
#         else:
#             sweep = range(max_level - 1, -1, -1)  # backward
#
#         for lv in sweep:
#             nodes = levels.get(lv, [])
#             if not nodes:
#                 continue
#             cx = node_pos[nodes[0]][0]
#
#             for n in nodes:
#                 # Collect connected nodes' Y positions
#                 if iteration % 2 == 0:
#                     neighbors = list(G.predecessors(n))   # forward: pulled by predecessors
#                 else:
#                     neighbors = list(G.successors(n))     # backward: pulled by successors
#                 if not neighbors:
#                     neighbors = list(G.predecessors(n)) + list(G.successors(n))
#                 ys = [node_pos[nb][1] for nb in neighbors if nb in node_pos]
#                 if ys:
#                     target = sum(ys) / len(ys)
#                     # Blend toward target (damped to avoid oscillation)
#                     damping = 0.7
#                     curr_y = node_pos[n][1]
#                     new_y = curr_y + damping * (target - curr_y)
#                     node_pos[n] = (cx, new_y)
#
#             # Enforce minimum gap after repositioning
#             _enforce_min_gap(nodes)
#
#     # ── Phase 4: Compaction — center each level's block vertically ───
#     # Find the overall vertical centre and shift each level toward it,
#     # so the whole graph is compact and centred.
#     all_cy = [node_pos[n][1] for n in node_pos]
#     global_center = (min(all_cy) + max(all_cy)) / 2 if all_cy else PAD
#
#     for lv in range(max_level + 1):
#         nodes = levels.get(lv, [])
#         if not nodes:
#             continue
#         ys = [node_pos[n][1] for n in nodes]
#         level_center = (min(ys) + max(ys)) / 2
#         shift = (global_center - level_center) * 0.3   # gentle pull toward center
#         cx = node_pos[nodes[0]][0]
#         for n in nodes:
#             node_pos[n] = (cx, node_pos[n][1] + shift)
#         _enforce_min_gap(nodes)
#
#     # ── Normalize positions (shift so nothing is above PAD) ──────────
#     all_y = [pos[1] for pos in node_pos.values()]
#     if all_y:
#         min_y = min(all_y)
#         shift = PAD + max(_node_height(n) for n in node_pos) / 2 - min_y
#         if shift > 0:
#             for n in node_pos:
#                 cx, cy = node_pos[n]
#                 node_pos[n] = (cx, cy + shift)
#
#     # ── Classify nodes ───────────────────────────────────────────────
#     root_data = set()
#     final_data = set()
#     for n, d in G.nodes(data=True):
#         if d.get("node_type") == "data":
#             if G.in_degree(n) == 0:
#                 root_data.add(n)
#             if G.out_degree(n) == 0:
#                 final_data.add(n)
#
#     # ── SVG dimensions ───────────────────────────────────────────────
#     all_x = [pos[0] for pos in node_pos.values()]
#     all_y = [pos[1] for pos in node_pos.values()]
#     svg_width  = int(max(all_x) + RECT_W / 2 + PAD * 2) if all_x else 800
#     svg_height = int(max(all_y) + OVAL_RY + PAD * 2) if all_y else 400
#
#     # ── Helper: wrap SVG text ────────────────────────────────────────
#     FS_NODE = 13
#     def wrap_svg_text(text, cx, cy, max_chars=18, fs=FS_NODE, bold=False, fill="#333"):
#         fw = ' font-weight="bold"' if bold else ""
#         safe = lambda t: _html.escape(t)
#         if len(text) <= max_chars:
#             return (f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="{fs}"{fw}'
#                     f' font-family="{FONT}" fill="{fill}">{safe(text)}</text>')
#         mid = len(text) // 2
#         best = mid
#         for off in range(mid):
#             if text[mid - off] == " ": best = mid - off; break
#             if mid + off < len(text) and text[mid + off] == " ": best = mid + off; break
#         l1 = safe(text[:best].strip()[:max_chars])
#         l2 = safe(text[best:].strip()[:max_chars])
#         dy = fs * 0.6
#         return (f'<text text-anchor="middle" font-size="{fs}"{fw} font-family="{FONT}" fill="{fill}">'
#                 f'<tspan x="{cx}" y="{cy - dy + 2}">{l1}</tspan>'
#                 f'<tspan x="{cx}" y="{cy + dy + 2}">{l2}</tspan></text>')
#
#     # ── Build SVG elements ───────────────────────────────────────────
#     svg = []
#     details = {}
#
#     # Edges first (so nodes draw on top)
#     for u, v in G.edges():
#         ux, uy = node_pos[u]
#         vx, vy = node_pos[v]
#         u_nt = G.nodes[u].get("node_type", "data")
#         v_nt = G.nodes[v].get("node_type", "data")
#         # Exit from right side of source
#         x1 = ux + (RECT_W / 2 if u_nt == "operation" else OVAL_RX)
#         y1 = uy
#         # Enter left side of target
#         x2 = vx - (RECT_W / 2 if v_nt == "operation" else OVAL_RX)
#         y2 = vy
#
#         if abs(y1 - y2) > 5:
#             mx = (x1 + x2) / 2
#             svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
#                        f'fill="none" stroke="#555" stroke-width="2" '
#                        f'marker-end="url(#mb_arrow)"/>')
#         else:
#             svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                        f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # Nodes
#     for n, d in G.nodes(data=True):
#         cx, cy = node_pos[n]
#         nid = _html.escape(n).replace('"', '&quot;')
#         safe_nid = n.replace("'", "\\'").replace('"', '\\"')
#         nt = d.get("node_type", "data")
#         desc = d.get("description", n)
#         data_path = d.get("data_path", "")
#
#         svg.append(f'<g class="node-box" onclick="showPopup(\'{safe_nid}\')">')
#
#         if nt == "operation":
#             rx = cx - RECT_W / 2; ry = cy - RECT_H / 2
#             svg.append(f'  <rect x="{rx}" y="{ry}" width="{RECT_W}" height="{RECT_H}" '
#                        f'rx="{RECT_RX}" fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
#                        f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(desc, cx, cy, 20, bold=True, fill="#333"))
#             # Popup
#             inp = [p for p in G.predecessors(n)]
#             out = [s for s in G.successors(n)]
#             details[n] = (f"<h3>{_html.escape(desc)}</h3>"
#                           f"<p><b>Type:</b> Operation</p>"
#                           f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in inp)}</p>"
#                           f"<p><b>Outputs:</b> {', '.join(_html.escape(o) for o in out)}</p>")
#         else:
#             # Data node
#             is_input = n in root_data
#             is_final = n in final_data
#             fill   = INPUT_FILL   if is_input else OUTPUT_FILL
#             stroke = INPUT_STROKE if is_input else OUTPUT_STROKE
#             orx = OVAL_RX + (5 if is_final else 0)
#             ory = OVAL_RY + (3 if is_final else 0)
#             sw  = "2.5" if is_final else "1.5"
#             svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{orx}" ry="{ory}" '
#                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
#                        f'data-os="{stroke}"/>')
#             svg.append("  " + wrap_svg_text(desc if len(desc) < 30 else n, cx, cy, 16, bold=True, fill="white"))
#             kind = "Input Data" if is_input else ("Final Output" if is_final else "Intermediate Data")
#             det = f"<h3>{_html.escape(n)}</h3><p><b>Type:</b> {kind}</p>"
#             if desc:
#                 det += f"<p><b>Description:</b> {_html.escape(desc)}</p>"
#             if data_path:
#                 det += f"<p><b>Path:</b> <code>{_html.escape(data_path)}</code></p>"
#             details[n] = det
#
#         svg.append('</g>')
#
#     svg_body = "\n    ".join(svg)
#
#     # ── HTML template ────────────────────────────────────────────────
#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>{_html.escape(title)}</title>
# <style>
#   body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
#   h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
#   .graph-panel {{ flex:1; overflow:hidden; position:relative; cursor:grab; }}
#   svg {{ display:block; width:100%; height:100%; }}
#   .node-box {{ cursor:pointer; transition:filter .15s; }}
#   .node-box:hover {{ filter:brightness(1.12) drop-shadow(0 0 4px rgba(0,0,0,.3)); }}
#   .zoom-toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
#   .zoom-btn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; color:#444; transition:all .15s; padding:0; }}
#   .zoom-btn:hover {{ background:#e9ecef; }}
#   .zoom-level {{ font-size:10px; text-align:center; color:#888; padding:2px 0; user-select:none; }}
#   .zoom-divider {{ height:1px; background:#e0e0e0; margin:2px 0; }}
#   .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
#   .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
#   .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
#   .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
#   .popup-overlay.visible {{ display:flex; }}
#   .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
#   @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
#   .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
#   .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
#   .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
#   .popup-close:hover {{ background:#f0f0f0; color:#333; }}
# </style>
# </head>
# <body>
# <h1>{_html.escape(title)}</h1>
# <div class="graph-panel" id="graphPanel">
#   <div class="zoom-toolbar">
#     <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
#     <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
#     <div class="zoom-divider"></div>
#     <span class="zoom-level" id="zoomLevel">100%</span>
#     <div class="zoom-divider"></div>
#     <button class="zoom-btn" onclick="zoomReset()" title="Reset">&#8634;</button>
#     <button class="zoom-btn" onclick="zoomFit()" title="Fit">&#11036;</button>
#   </div>
#   <div class="legend">
#     <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:3px"></div> Operation / Tool</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Output / Derived Data</div>
#   </div>
#   <svg id="flowchartSvg" viewBox="0 0 {svg_width} {svg_height}">
#     <defs>
#       <marker id="mb_arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
#         <polygon points="0 0,10 3.5,0 7" fill="#555"/>
#       </marker>
#     </defs>
#     <g id="zoomGroup">
#     {svg_body}
#     </g>
#   </svg>
# </div>
#
# <div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
#   <div class="popup-box" onclick="event.stopPropagation()">
#     <button class="popup-close" onclick="closePopup()">&times;</button>
#     <div id="popupContent"></div>
#   </div>
# </div>
#
# <script>
# const details = {json.dumps(details)};
#
# let scale=1, panX=0, panY=0, isPanning=false, _wasDragged=false, psX=0, psY=0, _mdX=0, _mdY=0;
# const PT=4, MN=0.2, MX=5, ZS=0.15;
# const svg=document.getElementById('flowchartSvg');
# const zg=document.getElementById('zoomGroup');
# const gp=document.getElementById('graphPanel');
#
# function applyTransform(){{
#   zg.setAttribute('transform','translate('+panX+','+panY+') scale('+scale+')');
#   document.getElementById('zoomLevel').textContent=Math.round(scale*100)+'%';
# }}
# function zoomIn(){{ scale=Math.min(MX,scale+ZS); applyTransform(); }}
# function zoomOut(){{ scale=Math.max(MN,scale-ZS); applyTransform(); }}
# function zoomReset(){{ scale=1;panX=0;panY=0;applyTransform(); }}
# function zoomFit(){{
#   var b=zg.getBBox(),v=svg.viewBox.baseVal,p=30;
#   scale=Math.min((v.width-p*2)/b.width,(v.height-p*2)/b.height,MX);
#   panX=(v.width/scale-b.width)/2-b.x+p/scale;
#   panY=(v.height/scale-b.height)/2-b.y+p/scale;
#   applyTransform();
# }}
#
# gp.addEventListener('wheel',function(e){{
#   e.preventDefault();
#   var r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
#   var sx=((e.clientX-r.left)/r.width)*v.width, sy=((e.clientY-r.top)/r.height)*v.height;
#   var os=scale;
#   scale=e.deltaY<0?Math.min(MX,scale+ZS):Math.max(MN,scale-ZS);
#   panX=sx/scale-sx/os+panX; panY=sy/scale-sy/os+panY; applyTransform();
# }},{{passive:false}});
#
# gp.addEventListener('mousedown',function(e){{
#   if(e.button===0||e.button===1){{
#     e.preventDefault();isPanning=true;_wasDragged=false;
#     psX=e.clientX;psY=e.clientY;_mdX=e.clientX;_mdY=e.clientY;
#     gp.style.cursor='grabbing';
#   }}
# }});
# window.addEventListener('mousemove',function(e){{
#   if(!isPanning)return;
#   var dist=Math.abs(e.clientX-_mdX)+Math.abs(e.clientY-_mdY);
#   if(dist>PT)_wasDragged=true;
#   var v=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
#   panX+=(e.clientX-psX)*(v.width/r.width)/scale;
#   panY+=(e.clientY-psY)*(v.height/r.height)/scale;
#   psX=e.clientX;psY=e.clientY;applyTransform();
# }});
# window.addEventListener('mouseup',function(){{ if(isPanning){{isPanning=false;gp.style.cursor='';}} }});
# gp.addEventListener('click',function(e){{ if(_wasDragged){{e.stopPropagation();_wasDragged=false;}} }},true);
#
# function showPopup(nodeId){{
#   var c=document.getElementById('popupContent');
#   var o=document.getElementById('popupOverlay');
#   c.innerHTML=details[nodeId]||'<p style="color:#999">No details.</p>';
#   o.classList.add('visible');
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){{
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   }});
#   var cl=event.currentTarget.querySelector('ellipse,rect');
#   if(cl){{ cl.setAttribute('stroke-width','3.5'); cl.setAttribute('stroke','#FFD700'); }}
# }}
# function closePopup(e){{
#   if(e&&e.target!==document.getElementById('popupOverlay'))return;
#   document.getElementById('popupOverlay').classList.remove('visible');
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){{
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   }});
# }}
# document.addEventListener('keydown',function(e){{ if(e.key==='Escape')closePopup(); }});
# window.addEventListener('DOMContentLoaded',function(){{ setTimeout(zoomFit,100); }});
# </script>
# </body>
# </html>"""
#
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     print(f"DAG workflow saved to: {output_path}")
#     return output_path


# def show_dag_geoprocessing_workflow1(G, output_path="Outputs/geoprocessing_workflow_dag.html",
#                                     title="Geoprocessing Workflow", research_question=""):
#     """Render a NetworkX DiGraph as a true 2D DAG in the style of ArcGIS Model Builder.
#
#     Nodes are positioned using a layered (Sugiyama-style) layout:
#       - X position = topological level (longest path from any root)
#       - Y position = median heuristic within each level to reduce crossings
#
#     Node colours follow the Model Builder palette:
#       - Blue ovals:   input data (root data nodes with a data_path)
#       - Yellow rects: operations
#       - Green ovals:  intermediate / output data
#
#     Args:
#         G: NetworkX DiGraph with node_type="data"|"operation" attributes
#         output_path: where to write the HTML file
#         title: displayed title
#         research_question: optional RQ text shown in popup
#
#     Returns:
#         Path to saved HTML file
#     """
#     import networkx as nx
#     import json
#     import os
#     import html as _html
#
#     # ── Palette (same as Model Builder) ──────────────────────────────
#     INPUT_FILL   = "#6BA3D6"; INPUT_STROKE  = "#2F74B5"
#     TOOL_FILL    = "#F5D576"; TOOL_STROKE   = "#C4A030"
#     OUTPUT_FILL  = "#6BB56A"; OUTPUT_STROKE = "#3D8B3D"
#     FONT = "Segoe UI, Arial, sans-serif"
#
#     # ── Dimensions ───────────────────────────────────────────────────
#     OVAL_RX = 85;  OVAL_RY = 32
#     RECT_W  = 180; RECT_H  = 62; RECT_RX = 8
#     COL_GAP = 100          # horizontal gap between level columns
#     ROW_GAP = 30           # vertical gap between nodes in same column
#     PAD     = 60           # canvas padding
#
#     # ── Topological sort & level assignment ──────────────────────────
#     try:
#         topo = list(nx.topological_sort(G))
#     except nx.NetworkXUnfeasible:
#         topo = list(G.nodes())
#
#     node_level = {}
#     for n in topo:
#         preds = list(G.predecessors(n))
#         node_level[n] = 0 if not preds else max(node_level.get(p, 0) for p in preds) + 1
#
#     # Group nodes by level
#     levels = {}
#     for n, lv in node_level.items():
#         levels.setdefault(lv, []).append(n)
#
#     max_level = max(levels.keys()) if levels else 0
#
#     # ── Y ordering: median heuristic (2 passes) ─────────────────────
#     # Initial order: follow topological sort within each level
#     topo_index = {n: i for i, n in enumerate(topo)}
#     for lv in levels:
#         levels[lv].sort(key=lambda n: topo_index.get(n, 0))
#
#     def _node_height(n):
#         nt = G.nodes[n].get("node_type", "data")
#         return RECT_H if nt == "operation" else OVAL_RY * 2
#
#     def _median_pos(node, pos_dict):
#         preds = list(G.predecessors(node))
#         succs = list(G.successors(node))
#         neighbors = preds + succs
#         ys = [pos_dict[nb][1] for nb in neighbors if nb in pos_dict]
#         if ys:
#             ys.sort()
#             return ys[len(ys) // 2]
#         return None
#
#     # Initial Y assignment: evenly spaced within each column
#     node_pos = {}  # node -> (cx, cy)
#     for lv in range(max_level + 1):
#         nodes = levels.get(lv, [])
#         col_cx = PAD + lv * (RECT_W + COL_GAP) + RECT_W / 2
#         total_h = sum(_node_height(n) for n in nodes) + ROW_GAP * max(len(nodes) - 1, 0)
#         y = PAD + (total_h > 0 and 0 or 0)
#         for n in nodes:
#             h = _node_height(n)
#             node_pos[n] = (col_cx, y + h / 2)
#             y += h + ROW_GAP
#
#     # Refine Y with median heuristic (forward + backward passes)
#     for _pass in range(4):
#         scan_order = range(max_level + 1) if _pass % 2 == 0 else range(max_level, -1, -1)
#         for lv in scan_order:
#             nodes = levels.get(lv, [])
#             medians = []
#             for n in nodes:
#                 med = _median_pos(n, node_pos)
#                 medians.append((n, med if med is not None else node_pos[n][1]))
#             medians.sort(key=lambda x: x[1])
#             # Re-space with minimum gap, centred on median positions
#             cx = node_pos[nodes[0]][0] if nodes else 0
#             for i, (n, _) in enumerate(medians):
#                 target_y = medians[i][1]
#                 # Ensure no overlap with previous node
#                 if i > 0:
#                     prev_n = medians[i - 1][0]
#                     min_y = node_pos[prev_n][1] + _node_height(prev_n) / 2 + ROW_GAP + _node_height(n) / 2
#                     target_y = max(target_y, min_y)
#                 node_pos[n] = (cx, target_y)
#
#     # ── Normalize positions (shift so min_y = PAD) ───────────────────
#     all_y = [pos[1] for pos in node_pos.values()]
#     if all_y:
#         min_y = min(all_y)
#         shift = PAD - min_y + max(_node_height(n) for n in node_pos) / 2
#         for n in node_pos:
#             cx, cy = node_pos[n]
#             node_pos[n] = (cx, cy + shift)
#
#     # ── Classify nodes ───────────────────────────────────────────────
#     root_data = set()
#     final_data = set()
#     for n, d in G.nodes(data=True):
#         if d.get("node_type") == "data":
#             if G.in_degree(n) == 0:
#                 root_data.add(n)
#             if G.out_degree(n) == 0:
#                 final_data.add(n)
#
#     # ── SVG dimensions ───────────────────────────────────────────────
#     all_x = [pos[0] for pos in node_pos.values()]
#     all_y = [pos[1] for pos in node_pos.values()]
#     svg_width  = int(max(all_x) + RECT_W / 2 + PAD * 2) if all_x else 800
#     svg_height = int(max(all_y) + OVAL_RY + PAD * 2) if all_y else 400
#
#     # ── Helper: wrap SVG text ────────────────────────────────────────
#     FS_NODE = 13
#     def wrap_svg_text(text, cx, cy, max_chars=18, fs=FS_NODE, bold=False, fill="#333"):
#         fw = ' font-weight="bold"' if bold else ""
#         safe = lambda t: _html.escape(t)
#         if len(text) <= max_chars:
#             return (f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="{fs}"{fw}'
#                     f' font-family="{FONT}" fill="{fill}">{safe(text)}</text>')
#         mid = len(text) // 2
#         best = mid
#         for off in range(mid):
#             if text[mid - off] == " ": best = mid - off; break
#             if mid + off < len(text) and text[mid + off] == " ": best = mid + off; break
#         l1 = safe(text[:best].strip()[:max_chars])
#         l2 = safe(text[best:].strip()[:max_chars])
#         dy = fs * 0.6
#         return (f'<text text-anchor="middle" font-size="{fs}"{fw} font-family="{FONT}" fill="{fill}">'
#                 f'<tspan x="{cx}" y="{cy - dy + 2}">{l1}</tspan>'
#                 f'<tspan x="{cx}" y="{cy + dy + 2}">{l2}</tspan></text>')
#
#     # ── Build SVG elements ───────────────────────────────────────────
#     svg = []
#     details = {}
#
#     # Edges first (so nodes draw on top)
#     for u, v in G.edges():
#         ux, uy = node_pos[u]
#         vx, vy = node_pos[v]
#         u_nt = G.nodes[u].get("node_type", "data")
#         v_nt = G.nodes[v].get("node_type", "data")
#         # Exit from right side of source
#         x1 = ux + (RECT_W / 2 if u_nt == "operation" else OVAL_RX)
#         y1 = uy
#         # Enter left side of target
#         x2 = vx - (RECT_W / 2 if v_nt == "operation" else OVAL_RX)
#         y2 = vy
#
#         if abs(y1 - y2) > 5:
#             mx = (x1 + x2) / 2
#             svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
#                        f'fill="none" stroke="#555" stroke-width="2" '
#                        f'marker-end="url(#mb_arrow)"/>')
#         else:
#             svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
#                        f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
#
#     # Nodes
#     for n, d in G.nodes(data=True):
#         cx, cy = node_pos[n]
#         nid = _html.escape(n).replace('"', '&quot;')
#         safe_nid = n.replace("'", "\\'").replace('"', '\\"')
#         nt = d.get("node_type", "data")
#         desc = d.get("description", n)
#         data_path = d.get("data_path", "")
#
#         svg.append(f'<g class="node-box" onclick="showPopup(\'{safe_nid}\')">')
#
#         if nt == "operation":
#             rx = cx - RECT_W / 2; ry = cy - RECT_H / 2
#             svg.append(f'  <rect x="{rx}" y="{ry}" width="{RECT_W}" height="{RECT_H}" '
#                        f'rx="{RECT_RX}" fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
#                        f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
#             svg.append("  " + wrap_svg_text(desc, cx, cy, 20, bold=True, fill="#333"))
#             # Popup
#             inp = [p for p in G.predecessors(n)]
#             out = [s for s in G.successors(n)]
#             details[n] = (f"<h3>{_html.escape(desc)}</h3>"
#                           f"<p><b>Type:</b> Operation</p>"
#                           f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in inp)}</p>"
#                           f"<p><b>Outputs:</b> {', '.join(_html.escape(o) for o in out)}</p>")
#         else:
#             # Data node
#             is_input = n in root_data
#             is_final = n in final_data
#             fill   = INPUT_FILL   if is_input else OUTPUT_FILL
#             stroke = INPUT_STROKE if is_input else OUTPUT_STROKE
#             orx = OVAL_RX + (5 if is_final else 0)
#             ory = OVAL_RY + (3 if is_final else 0)
#             sw  = "2.5" if is_final else "1.5"
#             svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{orx}" ry="{ory}" '
#                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
#                        f'data-os="{stroke}"/>')
#             svg.append("  " + wrap_svg_text(desc if len(desc) < 30 else n, cx, cy, 16, bold=True, fill="white"))
#             kind = "Input Data" if is_input else ("Final Output" if is_final else "Intermediate Data")
#             det = f"<h3>{_html.escape(n)}</h3><p><b>Type:</b> {kind}</p>"
#             if desc:
#                 det += f"<p><b>Description:</b> {_html.escape(desc)}</p>"
#             if data_path:
#                 det += f"<p><b>Path:</b> <code>{_html.escape(data_path)}</code></p>"
#             details[n] = det
#
#         svg.append('</g>')
#
#     svg_body = "\n    ".join(svg)
#
#     # ── HTML template ────────────────────────────────────────────────
#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>{_html.escape(title)}</title>
# <style>
#   body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
#   h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
#   .graph-panel {{ flex:1; overflow:hidden; position:relative; cursor:grab; }}
#   svg {{ display:block; width:100%; height:100%; }}
#   .node-box {{ cursor:pointer; transition:filter .15s; }}
#   .node-box:hover {{ filter:brightness(1.12) drop-shadow(0 0 4px rgba(0,0,0,.3)); }}
#   .zoom-toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
#   .zoom-btn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; color:#444; transition:all .15s; padding:0; }}
#   .zoom-btn:hover {{ background:#e9ecef; }}
#   .zoom-level {{ font-size:10px; text-align:center; color:#888; padding:2px 0; user-select:none; }}
#   .zoom-divider {{ height:1px; background:#e0e0e0; margin:2px 0; }}
#   .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
#   .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
#   .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
#   .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
#   .popup-overlay.visible {{ display:flex; }}
#   .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
#   @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
#   .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
#   .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
#   .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
#   .popup-close:hover {{ background:#f0f0f0; color:#333; }}
# </style>
# </head>
# <body>
# <h1>{_html.escape(title)}</h1>
# <div class="graph-panel" id="graphPanel">
#   <div class="zoom-toolbar">
#     <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
#     <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
#     <div class="zoom-divider"></div>
#     <span class="zoom-level" id="zoomLevel">100%</span>
#     <div class="zoom-divider"></div>
#     <button class="zoom-btn" onclick="zoomReset()" title="Reset">&#8634;</button>
#     <button class="zoom-btn" onclick="zoomFit()" title="Fit">&#11036;</button>
#   </div>
#   <div class="legend">
#     <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:3px"></div> Operation / Tool</div>
#     <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Output / Derived Data</div>
#   </div>
#   <svg id="flowchartSvg" viewBox="0 0 {svg_width} {svg_height}">
#     <defs>
#       <marker id="mb_arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
#         <polygon points="0 0,10 3.5,0 7" fill="#555"/>
#       </marker>
#     </defs>
#     <g id="zoomGroup">
#     {svg_body}
#     </g>
#   </svg>
# </div>
#
# <div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
#   <div class="popup-box" onclick="event.stopPropagation()">
#     <button class="popup-close" onclick="closePopup()">&times;</button>
#     <div id="popupContent"></div>
#   </div>
# </div>
#
# <script>
# const details = {json.dumps(details)};
#
# let scale=1, panX=0, panY=0, isPanning=false, _wasDragged=false, psX=0, psY=0, _mdX=0, _mdY=0;
# const PT=4, MN=0.2, MX=5, ZS=0.15;
# const svg=document.getElementById('flowchartSvg');
# const zg=document.getElementById('zoomGroup');
# const gp=document.getElementById('graphPanel');
#
# function applyTransform(){{
#   zg.setAttribute('transform','translate('+panX+','+panY+') scale('+scale+')');
#   document.getElementById('zoomLevel').textContent=Math.round(scale*100)+'%';
# }}
# function zoomIn(){{ scale=Math.min(MX,scale+ZS); applyTransform(); }}
# function zoomOut(){{ scale=Math.max(MN,scale-ZS); applyTransform(); }}
# function zoomReset(){{ scale=1;panX=0;panY=0;applyTransform(); }}
# function zoomFit(){{
#   var b=zg.getBBox(),v=svg.viewBox.baseVal,p=30;
#   scale=Math.min((v.width-p*2)/b.width,(v.height-p*2)/b.height,MX);
#   panX=(v.width/scale-b.width)/2-b.x+p/scale;
#   panY=(v.height/scale-b.height)/2-b.y+p/scale;
#   applyTransform();
# }}
#
# gp.addEventListener('wheel',function(e){{
#   e.preventDefault();
#   var r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
#   var sx=((e.clientX-r.left)/r.width)*v.width, sy=((e.clientY-r.top)/r.height)*v.height;
#   var os=scale;
#   scale=e.deltaY<0?Math.min(MX,scale+ZS):Math.max(MN,scale-ZS);
#   panX=sx/scale-sx/os+panX; panY=sy/scale-sy/os+panY; applyTransform();
# }},{{passive:false}});
#
# gp.addEventListener('mousedown',function(e){{
#   if(e.button===0||e.button===1){{
#     e.preventDefault();isPanning=true;_wasDragged=false;
#     psX=e.clientX;psY=e.clientY;_mdX=e.clientX;_mdY=e.clientY;
#     gp.style.cursor='grabbing';
#   }}
# }});
# window.addEventListener('mousemove',function(e){{
#   if(!isPanning)return;
#   var dist=Math.abs(e.clientX-_mdX)+Math.abs(e.clientY-_mdY);
#   if(dist>PT)_wasDragged=true;
#   var v=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
#   panX+=(e.clientX-psX)*(v.width/r.width)/scale;
#   panY+=(e.clientY-psY)*(v.height/r.height)/scale;
#   psX=e.clientX;psY=e.clientY;applyTransform();
# }});
# window.addEventListener('mouseup',function(){{ if(isPanning){{isPanning=false;gp.style.cursor='';}} }});
# gp.addEventListener('click',function(e){{ if(_wasDragged){{e.stopPropagation();_wasDragged=false;}} }},true);
#
# function showPopup(nodeId){{
#   var c=document.getElementById('popupContent');
#   var o=document.getElementById('popupOverlay');
#   c.innerHTML=details[nodeId]||'<p style="color:#999">No details.</p>';
#   o.classList.add('visible');
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){{
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   }});
#   var cl=event.currentTarget.querySelector('ellipse,rect');
#   if(cl){{ cl.setAttribute('stroke-width','3.5'); cl.setAttribute('stroke','#FFD700'); }}
# }}
# function closePopup(e){{
#   if(e&&e.target!==document.getElementById('popupOverlay'))return;
#   document.getElementById('popupOverlay').classList.remove('visible');
#   document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){{
#     el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
#     el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
#   }});
# }}
# document.addEventListener('keydown',function(e){{ if(e.key==='Escape')closePopup(); }});
# window.addEventListener('DOMContentLoaded',function(){{ setTimeout(zoomFit,100); }});
# </script>
# </body>
# </html>"""
#
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     print(f"DAG workflow saved to: {output_path}")
#     return output_path


def show_spatial_analysis_geoprocessing_workflow_plan_overview_popup(plan, output_path="Outputs/geoprocessing_workflow_plan_overview_popup.html"):
    """Render an ArcGIS Model Builder-style SVG workflow for each objective.

    Layout per objective (left-to-right chain):
        [Input Data (blue oval)] -> [Operation (yellow rect)] -> [Output Data (green oval)] -> ...

    Output nodes from earlier steps are reused when they appear as inputs to later
    steps (arrows drawn from the existing green oval), just like Model Builder.

    Clicking any node opens a popup with full details.  Supports zoom & pan.

    Args:
        plan: Research plan dict (objectives -> steps -> input_nodes / output_node)
        output_path: Path to save the HTML file

    Returns:
        Path to the saved HTML file
    """
    import json
    import os
    import html as _html

    objectives = plan.get("objectives", [])
    plan_title = plan.get("title", "Geoprocessing Workflow")

    # ── ArcGIS Model Builder palette ──────────────────────────────────
    INPUT_FILL    = "#6BA3D6";  INPUT_STROKE  = "#2F74B5"   # blue ovals
    TOOL_FILL     = "#F5D576";  TOOL_STROKE   = "#C4A030"   # yellow rects
    OUTPUT_FILL   = "#6BB56A";  OUTPUT_STROKE = "#3D8B3D"   # green ovals
    BAND_COLORS   = ["#edf2f9", "#fdf5e8", "#edf9ed", "#f9edee", "#f0edf9"]

    # ── Node dimensions ──────────────────────────────────────────────
    OVAL_RX   = 85;  OVAL_RY   = 32
    RECT_W    = 180; RECT_H    = 62; RECT_RX = 5
    GAP       = 60          # horizontal gap between connected shapes
    STACK_GAP = 14          # vertical gap between stacked input ovals
    LEFT_PAD  = 380         # room for RQ node + gap + objective label
    TOP_PAD   = 60
    ROW_GAP   = 50          # vertical space between objective bands
    FONT      = "Segoe UI, Arial, sans-serif"
    FS_NODE   = 14
    FS_OBJ    = 17

    # ── Helpers ──────────────────────────────────────────────────────
    def wrap_svg_text(text, cx, cy, max_chars=18, fs=FS_NODE, bold=False, fill="#333"):
        """Return SVG <text> element(s), wrapping into up to 2 lines."""
        fw = ' font-weight="bold"' if bold else ""
        safe = lambda t: _html.escape(t)
        if len(text) <= max_chars:
            return (f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="{fs}"{fw}'
                    f' font-family="{FONT}" fill="{fill}">{safe(text)}</text>')
        # best split near the middle at a space
        mid = len(text) // 2
        best = mid
        for off in range(mid):
            if text[mid - off] == " ":
                best = mid - off; break
            if mid + off < len(text) and text[mid + off] == " ":
                best = mid + off; break
        l1 = safe(text[:best].strip()[:max_chars])
        l2 = safe(text[best:].strip()[:max_chars])
        dy = fs * 0.6
        return (f'<text text-anchor="middle" font-size="{fs}"{fw} font-family="{FONT}" fill="{fill}">'
                f'<tspan x="{cx}" y="{cy - dy + 2}">{l1}</tspan>'
                f'<tspan x="{cx}" y="{cy + dy + 2}">{l2}</tspan></text>')

    # ── First pass: compute layout per objective row ─────────────────
    rows = []
    max_row_width = 0
    global_placed = {}       # data_name -> {cx, cy_off, node_id, row_idx}
    cross_row_edges = []     # (source_node_id, target_node_id, source_row_idx, target_row_idx)
    is_dag = plan.get("dag_mode", False)

    for i_obj, obj in enumerate(objectives):
        obj_num = obj["number"]
        steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))

        placed = {}          # data_name -> {cx, cy_off, node_id}
        nodes_list = []
        edges_list = []      # (from_id, to_id, edge_type)
        x = LEFT_PAD
        max_stack_h = RECT_H
        last_out_cx = None   # cx of the most recent output oval in this chain
        prev_step_out_name = None  # output_node name of immediately previous step

        for j, step in enumerate(steps):
            step_num  = step["step_number"]
            operation = step.get("operation", "")
            inp_names = step.get("input_nodes", [])
            out_name  = step.get("output_node", "")
            desc      = step.get("description", "")
            expected  = step.get("expected_output", "") or ""
            op_type   = step.get("operation_type", "")

            new_inputs    = [n for n in inp_names if n not in placed and n not in global_placed]
            reused_inputs = [n for n in inp_names if n in placed]
            cross_row_inputs = [n for n in inp_names if n not in placed and n in global_placed]

            # ── new input ovals ──
            input_ids = {}
            if new_inputs:
                n = len(new_inputs)
                block_h = n * (OVAL_RY * 2) + (n - 1) * STACK_GAP
                max_stack_h = max(max_stack_h, block_h)
                start_y = -(block_h / 2) + OVAL_RY
                inp_cx = x + OVAL_RX          # center so left edge is at x
                for k, inp in enumerate(new_inputs):
                    y_off = start_y + k * (OVAL_RY * 2 + STACK_GAP)
                    nid = f"o{obj_num}_s{step_num}_i{k}"
                    nodes_list.append({"id": nid, "type": "input", "name": inp,
                                       "cx": inp_cx, "cy_off": y_off})
                    placed[inp] = {"cx": inp_cx, "cy_off": y_off, "node_id": nid}
                    input_ids[inp] = nid
                x = inp_cx + OVAL_RX + GAP    # advance past right edge + gap

            # ── operation rectangle ──
            op_cx = x + RECT_W / 2
            op_id = f"o{obj_num}_s{step_num}_op"
            nodes_list.append({"id": op_id, "type": "operation", "name": operation,
                               "cx": op_cx, "cy_off": 0,
                               "step_num": step_num, "desc": desc,
                               "expected": expected, "op_type": op_type,
                               "input_nodes": inp_names, "output_node": out_name})

            for inp in new_inputs:
                edges_list.append((input_ids[inp], op_id, "direct"))

            # ── reused inputs: chain vs distant ──
            # Chain: the immediately previous step's output flows horizontally
            #        into this operation (normal arrow, no reference node needed).
            # Distant: output from an earlier (non-adjacent) step → reference
            #          node above the previous output column + diagonal arrow.
            chain_input = None
            distant_reused = []
            if not new_inputs and prev_step_out_name in reused_inputs:
                chain_input = prev_step_out_name
                distant_reused = [n for n in reused_inputs if n != chain_input]
            else:
                distant_reused = reused_inputs

            if chain_input:
                edges_list.append((placed[chain_input]["node_id"], op_id, "direct"))
            elif prev_step_out_name and prev_step_out_name in placed:
                # No data-chain but keep the visual flow continuous
                edges_list.append((placed[prev_step_out_name]["node_id"], op_id, "direct"))

            if distant_reused:
                ref_col_cx = last_out_cx if last_out_cx else (op_cx - RECT_W / 2 - GAP - OVAL_RX)
                ref_start_y = -(OVAL_RY * 2 + 16)
                for k, inp in enumerate(distant_reused):
                    ref_y = ref_start_y - k * (OVAL_RY * 2 + STACK_GAP)
                    ref_id = f"o{obj_num}_s{step_num}_ref{k}"
                    nodes_list.append({"id": ref_id, "type": "output_ref",
                                       "name": inp,
                                       "cx": ref_col_cx, "cy_off": ref_y})
                    edges_list.append((ref_id, op_id, "reference"))
                    max_stack_h = max(max_stack_h, abs(ref_y) + OVAL_RY)

            # ── cross-row inputs: data produced in a different branch ──
            for inp in cross_row_inputs:
                src = global_placed[inp]
                cross_row_edges.append((src["node_id"], op_id, src["row_idx"], i_obj))

            x = op_cx + RECT_W / 2 + GAP

            # ── output oval ──
            if out_name:
                out_id = f"o{obj_num}_s{step_num}_out"
                is_final = (j == len(steps) - 1)
                out_cx = x + OVAL_RX           # center so left edge is at x
                nodes_list.append({"id": out_id, "type": "output", "name": out_name,
                                   "cx": out_cx, "cy_off": 0, "final": is_final,
                                   "desc": expected})
                edges_list.append((op_id, out_id, "direct"))
                placed[out_name] = {"cx": out_cx, "cy_off": 0, "node_id": out_id}
                global_placed[out_name] = {"cx": out_cx, "cy_off": 0, "node_id": out_id, "row_idx": i_obj}
                last_out_cx = out_cx           # remember for reference node positioning
                prev_step_out_name = out_name  # track for chain detection
                x = out_cx + OVAL_RX + GAP     # advance past right edge + gap

        row_h = max(max_stack_h * 2 + 50, 90)
        rows.append({"obj": obj, "nodes": nodes_list, "edges": edges_list,
                     "width": x, "height": row_h})
        max_row_width = max(max_row_width, x)

    # ── Assign absolute Y positions ──────────────────────────────────
    y_cursor   = TOP_PAD
    for row in rows:
        row["cy"]  = y_cursor + row["height"] / 2
        row["top"] = y_cursor
        y_cursor  += row["height"] + ROW_GAP

    # ── Research Question node (far left, arrows diverge to each obj) ─
    RQ_RECT_W = 130; RQ_RECT_H = 56; RQ_RX = 8
    RQ_FILL = "#2C3E50"; RQ_STROKE = "#1A252F"
    OBJ_LABEL_X = 220     # x position for objective labels (after RQ node)
    rq_text = plan.get("research_question", "Research Question")
    all_cy = [row["cy"] for row in rows]
    rq_mid_y = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50
    rq_cx = 10 + RQ_RECT_W / 2   # left edge at x=10
    rq_cy = rq_mid_y
    rq_id = "rq_node"

    # Create objective title nodes and connect RQ → obj_title → first chain node
    OBJ_NODE_W = 120; OBJ_NODE_H = 42; OBJ_NODE_RX = 6
    OBJ_NODE_FILL = "#34495E"; OBJ_NODE_STROKE = "#2C3E50"
    obj_title_nodes = []
    rq_edges = []
    for row in rows:
        obj = row["obj"]
        obj_num = obj["number"]
        oid = f"obj_title_{obj_num}"
        obj_cx = OBJ_LABEL_X + OBJ_NODE_W / 2
        obj_title_nodes.append({
            "id": oid, "cx": obj_cx, "abs_cy": row["cy"],
            "type": "obj_title", "name": obj.get("title", ""),
            "obj_num": obj_num,
            "hypothesis": obj.get("hypothesis", ""),
        })
        rq_edges.append(("rq_node", oid))

    # ── Pipeline tail: Result Collation → Results Analysis →
    #    Discussion of Results → Manuscript ─────────────────────────
    STAGE_RECT_W  = 190;  STAGE_RECT_H = 60; STAGE_RX = 5
    STAGE_OVAL_RX = 90;   STAGE_OVAL_RY = 32
    STAGE_GAP     = 55
    PIPELINE_FILL   = "#8E44AD";  PIPELINE_STROKE = "#6C3483"   # purple ops
    MANUSCRIPT_FILL = "#C0392B";  MANUSCRIPT_STROKE = "#922B21" # red final

    # Collect the final output node id from each objective
    final_out_ids = []
    for row in rows:
        for nd in row["nodes"]:
            if nd.get("final"):
                final_out_ids.append(nd["id"])

    # Vertical centre across all objective rows
    all_cy = [row["cy"] for row in rows]
    mid_y  = (min(all_cy) + max(all_cy)) / 2 if all_cy else TOP_PAD + 50

    # Chain:  [Result Collation] → [Results Analysis] →
    #         [Discussion of Results] → [Manuscript]
    stage_labels = [
        ("Result Collation",       "collation"),   # yellow like operation nodes
        ("Results Analysis",       "operation"),
        ("Discussion of Results",  "operation"),
        ("Manuscript",             "final"),
    ]
    stage_nodes = []   # list of {id, cx, cy, type, name}
    stage_edges = []   # (from_id, to_id)
    sx = max_row_width + STAGE_GAP
    for idx, (label, stype) in enumerate(stage_labels):
        sid = f"stage_{idx}"
        if stype == "final":
            cx = sx + STAGE_OVAL_RX
            stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
                                "type": "stage_final", "name": label})
            sx = cx + STAGE_OVAL_RX + STAGE_GAP
        elif stype == "collation":
            cx = sx + STAGE_RECT_W / 2
            stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
                                "type": "stage_collation", "name": label})
            sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
        else:
            cx = sx + STAGE_RECT_W / 2
            stage_nodes.append({"id": sid, "cx": cx, "abs_cy": mid_y,
                                "type": "stage_op", "name": label})
            sx = cx + STAGE_RECT_W / 2 + STAGE_GAP
        if idx > 0:
            stage_edges.append((f"stage_{idx-1}", sid))

    # Edges from each objective's final output → Result Collation
    for fid in final_out_ids:
        stage_edges.append((fid, "stage_0"))

    svg_width  = max(1200, int(sx) + 60)
    svg_height = max(400, int(y_cursor) + 30)

    # ── Build node lookup & popup-details dict ───────────────────────
    node_lookup = {}
    details = {}

    for row in rows:
        obj     = row["obj"]
        obj_num = obj["number"]
        for nd in row["nodes"]:
            nd["abs_cy"] = row["cy"] + nd["cy_off"]
            node_lookup[nd["id"]] = nd

            if nd["type"] == "input":
                details[nd["id"]] = (
                    f"<h3>Input Data</h3>"
                    f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>")
            elif nd["type"] == "operation":
                d = (f"<h3>Step {obj_num}.{nd['step_num']}: {_html.escape(nd['name'])}</h3>"
                     f"<p><b>Operation Type:</b> {_html.escape(nd.get('op_type',''))}</p>"
                     f"<p><b>Inputs:</b> {', '.join(_html.escape(i) for i in nd.get('input_nodes',[]))}</p>"
                     f"<p><b>Output:</b> {_html.escape(nd.get('output_node',''))}</p>"
                     f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>")
                if nd.get("expected"):
                    d += f"<p><b>Expected Output:</b> {_html.escape(nd['expected'])}</p>"
                details[nd["id"]] = d
            elif nd["type"] in ("output", "output_ref"):
                label = "Output Data" if nd["type"] == "output" else "Reused Output"
                details[nd["id"]] = (
                    f"<h3>{label}</h3>"
                    f"<p><b>Name:</b> {_html.escape(nd['name'])}</p>"
                    + (f"<p><b>Description:</b> {_html.escape(nd.get('desc',''))}</p>"
                       if nd.get("desc") else ""))

    # Register RQ node in lookup + details
    node_lookup[rq_id] = {"id": rq_id, "type": "rq", "name": rq_text,
                          "cx": rq_cx, "abs_cy": rq_cy}
    details[rq_id] = (f"<h3>Research Question</h3>"
                      f"<p>{_html.escape(rq_text)}</p>")

    # Register pipeline stage nodes in lookup + details
    stage_descs = {
        "Result Collation":      "Collate outputs and results from all objectives into a unified dataset for analysis.",
        "Results Analysis":      "Perform statistical analysis, validation, and interpretation of the collated results.",
        "Discussion of Results": "Synthesise findings, compare with hypotheses, and discuss implications.",
        "Manuscript":            "Final research deliverable: paper, report, or document combining all findings.",
    }
    for sn in stage_nodes:
        node_lookup[sn["id"]] = sn
        details[sn["id"]] = (f"<h3>{_html.escape(sn['name'])}</h3>"
                             f"<p>{_html.escape(stage_descs.get(sn['name'], ''))}</p>")

    # Register objective text anchors in lookup (for arrow targeting only)
    for otn in obj_title_nodes:
        node_lookup[otn["id"]] = otn

    # ── Generate SVG body ────────────────────────────────────────────
    svg = []

    # background bands + objective text labels
    CONVERGE_BAND = "#fff3e0"   # warm tint for convergence rows
    for i_r, row in enumerate(rows):
        obj = row["obj"]
        branch_type = obj.get("branch_type", "")
        if is_dag and branch_type == "convergence":
            bc = CONVERGE_BAND
        else:
            bc = BAND_COLORS[i_r % len(BAND_COLORS)]
        svg.append(f'<rect x="0" y="{row["top"]-5}" width="{svg_width}" '
                   f'height="{row["height"]+10}" fill="{bc}" rx="6"/>')
        # Label: use "Branch N" / "Convergence" for DAG mode
        if is_dag and branch_type == "parallel":
            label_text = f'Branch {obj["number"]}'
        elif is_dag and branch_type == "convergence":
            label_text = "Convergence"
        else:
            label_text = f'Obj {obj["number"]}'
        svg.append(f'<text x="{OBJ_LABEL_X}" y="{row["cy"]-10}" font-size="{FS_OBJ}" '
                   f'font-weight="bold" font-family="{FONT}" fill="#333">'
                   f'{_html.escape(label_text)}</text>')
        t = obj.get("title", "")
        if t:
            short = t[:22] + (".." if len(t) > 22 else "")
            svg.append(f'<text x="{OBJ_LABEL_X}" y="{row["cy"]+8}" font-size="{FS_NODE}" '
                       f'font-family="{FONT}" fill="#666">{_html.escape(short)}</text>')

    # ── RQ node (rendered before edges so edges draw on top) ────────
    svg.append(f'<g class="node-box" onclick="showPopup(\'{rq_id}\')">')
    rq_rx_ = rq_cx - RQ_RECT_W / 2; rq_ry_ = rq_cy - RQ_RECT_H / 2
    svg.append(f'  <rect x="{rq_rx_}" y="{rq_ry_}" width="{RQ_RECT_W}" '
               f'height="{RQ_RECT_H}" rx="{RQ_RX}" '
               f'fill="{RQ_FILL}" stroke="{RQ_STROKE}" '
               f'stroke-width="2" data-os="{RQ_STROKE}"/>')
    svg.append("  " + wrap_svg_text("Research Question", rq_cx, rq_cy, 20, fs=15, bold=True, fill="white"))
    svg.append('</g>')

    # ── RQ edges (RQ → objective text labels) ────────────────────
    #   Diverging S-curves: exit RQ horizontally, arrive at obj text horizontally
    for fid, tid in rq_edges:
        fn = node_lookup[fid]; tn = node_lookup[tid]
        x1 = fn["cx"] + RQ_RECT_W / 2       # right edge of RQ rect
        y1 = fn["abs_cy"]
        x2 = OBJ_LABEL_X - 5                # just before the obj text
        y2 = tn["abs_cy"]
        # S-curve control points: leave RQ horizontally, arrive at text horizontally
        spread = (x2 - x1) * 0.5
        svg.append(f'<path d="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
                   f'fill="none" stroke="#555" stroke-width="2" '
                   f'marker-end="url(#mb_arrow)"/>')


    # edges (drawn first so nodes paint over them)
    for row in rows:
        for fid, tid, etype in row["edges"]:
            fn = node_lookup[fid]; tn = node_lookup[tid]

            if etype == "reference":
                # Reference node is above-left of the operation →
                # diagonal arrow from right edge of ref oval to left edge of op rect
                x1 = fn["cx"] + OVAL_RX              # right edge of reference oval
                y1 = fn["abs_cy"]                     # center-y of reference oval
                x2 = tn["cx"] - RECT_W / 2           # left edge of operation rect
                y2 = tn["abs_cy"]                     # center-y of operation
                svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                           f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')
            else:
                # Horizontal flow: right side of source → left side of target
                x1 = fn["cx"] + (OVAL_RX if fn["type"] != "operation" else RECT_W / 2)
                y1 = fn["abs_cy"]
                x2 = tn["cx"] - (OVAL_RX if tn["type"] != "operation" else RECT_W / 2)
                y2 = tn["abs_cy"]

                if abs(y1 - y2) > 5:
                    # Stacked inputs at different y — smooth S-curve
                    mx = (x1 + x2) / 2
                    svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
                               f'fill="none" stroke="#555" stroke-width="2" '
                               f'marker-end="url(#mb_arrow)"/>')
                else:
                    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                               f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')

    # ── cross-row edges (parallel branch outputs → convergence inputs) ─
    CROSS_STROKE = "#E67E22"   # orange for cross-row connections
    for src_id, tgt_id, src_row_idx, tgt_row_idx in cross_row_edges:
        fn = node_lookup.get(src_id)
        tn = node_lookup.get(tgt_id)
        if fn and tn:
            x1 = fn["cx"] + OVAL_RX
            y1 = fn["abs_cy"]
            x2 = tn["cx"] - RECT_W / 2
            y2 = tn["abs_cy"]
            # S-curve from source row to target row
            spread = abs(x2 - x1) * 0.4
            svg.append(f'<path d="M{x1},{y1} C{x1 + spread},{y1} {x2 - spread},{y2} {x2},{y2}" '
                       f'fill="none" stroke="{CROSS_STROKE}" stroke-width="2.5" '
                       f'stroke-dasharray="6,3" marker-end="url(#mb_arrow_cross)"/>')

    # nodes
    for row in rows:
        for nd in row["nodes"]:
            cx = nd["cx"]; cy = nd["abs_cy"]; nid = nd["id"]
            svg.append(f'<g class="node-box" onclick="showPopup(\'{nid}\')">')

            if nd["type"] == "input":
                svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
                           f'fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="1.5" '
                           f'data-os="{INPUT_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))

            elif nd["type"] == "operation":
                rx = cx - RECT_W / 2; ry_ = cy - RECT_H / 2
                svg.append(f'  <rect x="{rx}" y="{ry_}" width="{RECT_W}" height="{RECT_H}" '
                           f'rx="{RECT_RX}" fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
                           f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 18, bold=True, fill="#333"))

            elif nd["type"] == "output":
                orx = OVAL_RX + (5 if nd.get("final") else 0)
                ory = OVAL_RY + (3 if nd.get("final") else 0)
                sw  = "2.5" if nd.get("final") else "1.5"
                svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{orx}" ry="{ory}" '
                           f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="{sw}" '
                           f'data-os="{OUTPUT_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))

            elif nd["type"] == "output_ref":
                # Reused output — green oval placed above the operation
                svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{OVAL_RX}" ry="{OVAL_RY}" '
                           f'fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="1.5" '
                           f'data-os="{OUTPUT_STROKE}"/>')
                svg.append("  " + wrap_svg_text(nd["name"], cx, cy, 16, bold=True, fill="white"))

            svg.append('</g>')

    # ── Pipeline stage edges ─────────────────────────────────────────
    for fid, tid in stage_edges:
        fn = node_lookup[fid]; tn = node_lookup[tid]
        # Exit point
        if fn.get("type") in ("output", "stage_final"):
            x1 = fn["cx"] + (STAGE_OVAL_RX if fn["type"] == "stage_final" else OVAL_RX)
        elif fn.get("type") in ("stage_op", "stage_collation"):
            x1 = fn["cx"] + STAGE_RECT_W / 2
        else:
            x1 = fn["cx"] + (OVAL_RX if fn.get("type") != "operation" else RECT_W / 2)
        y1 = fn["abs_cy"]
        # Entry point
        if tn.get("type") == "stage_final":
            x2 = tn["cx"] - STAGE_OVAL_RX
        elif tn.get("type") in ("stage_op", "stage_collation"):
            x2 = tn["cx"] - STAGE_RECT_W / 2
        else:
            x2 = tn["cx"] - OVAL_RX
        y2 = tn["abs_cy"]

        if abs(y1 - y2) > 5:
            mx = (x1 + x2) / 2
            svg.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
                       f'fill="none" stroke="#555" stroke-width="2" '
                       f'marker-end="url(#mb_arrow)"/>')
        else:
            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                       f'stroke="#555" stroke-width="2" marker-end="url(#mb_arrow)"/>')

    # ── Pipeline stage nodes ─────────────────────────────────────────
    for sn in stage_nodes:
        cx = sn["cx"]; cy = sn["abs_cy"]; sid = sn["id"]
        svg.append(f'<g class="node-box" onclick="showPopup(\'{sid}\')">')
        if sn["type"] == "stage_collation":
            rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
            svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
                       f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
                       f'fill="{TOOL_FILL}" stroke="{TOOL_STROKE}" '
                       f'stroke-width="2" data-os="{TOOL_STROKE}"/>')
            svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="#333"))
        elif sn["type"] == "stage_op":
            rx_ = cx - STAGE_RECT_W / 2; ry_ = cy - STAGE_RECT_H / 2
            svg.append(f'  <rect x="{rx_}" y="{ry_}" width="{STAGE_RECT_W}" '
                       f'height="{STAGE_RECT_H}" rx="{STAGE_RX}" '
                       f'fill="{PIPELINE_FILL}" stroke="{PIPELINE_STROKE}" '
                       f'stroke-width="2" data-os="{PIPELINE_STROKE}"/>')
            svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 22, fs=14, bold=True, fill="white"))
        elif sn["type"] == "stage_final":
            svg.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{STAGE_OVAL_RX}" '
                       f'ry="{STAGE_OVAL_RY}" fill="{MANUSCRIPT_FILL}" '
                       f'stroke="{MANUSCRIPT_STROKE}" stroke-width="2.5" '
                       f'data-os="{MANUSCRIPT_STROKE}"/>')
            svg.append("  " + wrap_svg_text(sn["name"], cx, cy, 18, fs=14, bold=True, fill="white"))
        svg.append('</g>')

    svg_body = "\n    ".join(svg)

    # ── Assemble HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_html.escape(plan_title)}</title>
<style>
  body {{ margin:0; font-family:{FONT}; background:#f8f9fa; display:flex; flex-direction:column; height:100vh; }}
  h1 {{ text-align:center; padding:12px 0 4px; margin:0; font-size:18px; color:#2C3E50; }}
  .graph-panel {{ flex:1; overflow:auto; position:relative; cursor:grab; }}
  svg {{ display:block; width:100%; height:100%; min-width:{svg_width}px; min-height:{svg_height}px; }}
  .node-box {{ cursor:pointer; transition:filter .15s; }}
  .node-box:hover {{ filter:brightness(1.12) drop-shadow(0 0 4px rgba(0,0,0,.3)); }}
  .zoom-toolbar {{ position:absolute; top:10px; right:10px; display:flex; flex-direction:column; gap:4px; z-index:50; background:#fff; border-radius:8px; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; }}
  .zoom-btn {{ width:32px; height:32px; border:1px solid #ddd; border-radius:6px; background:#fff; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; color:#444; transition:all .15s; padding:0; }}
  .zoom-btn:hover {{ background:#e9ecef; }}
  .zoom-level {{ font-size:10px; text-align:center; color:#888; padding:2px 0; user-select:none; }}
  .zoom-divider {{ height:1px; background:#e0e0e0; margin:2px 0; }}
  .legend {{ position:absolute; bottom:10px; left:10px; background:#fff; border-radius:8px; padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,.15); border:1px solid #ddd; font-size:11px; z-index:50; }}
  .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
  .legend-swatch {{ width:28px; height:16px; border:1px solid #999; }}
  .popup-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.35); z-index:100; justify-content:center; align-items:center; }}
  .popup-overlay.visible {{ display:flex; }}
  .popup-box {{ background:#fff; border-radius:10px; padding:24px 28px; max-width:580px; width:92%; max-height:75vh; overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,.25); position:relative; animation:popIn .18s ease-out; }}
  @keyframes popIn {{ from{{transform:scale(.92);opacity:0}} to{{transform:scale(1);opacity:1}} }}
  .popup-box h3 {{ margin-top:0; color:#2C3E50; border-bottom:2px solid #4C72B0; padding-bottom:8px; }}
  .popup-box p {{ line-height:1.6; color:#333; font-size:13px; }}
  .popup-close {{ position:absolute; top:10px; right:14px; background:none; border:none; font-size:20px; color:#999; cursor:pointer; padding:4px 8px; border-radius:4px; }}
  .popup-close:hover {{ background:#f0f0f0; color:#333; }}
</style>
</head>
<body>
<h1>{_html.escape(plan_title)}</h1>
<div class="graph-panel" id="graphPanel">
  <div class="zoom-toolbar">
    <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
    <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
    <div class="zoom-divider"></div>
    <span class="zoom-level" id="zoomLevel">100%</span>
    <div class="zoom-divider"></div>
    <button class="zoom-btn" onclick="zoomReset()" title="Reset">&#8634;</button>
    <button class="zoom-btn" onclick="zoomFit()" title="Fit">&#11036;</button>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:{INPUT_FILL};border-radius:8px"></div> Input Data</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{TOOL_FILL};border-radius:2px"></div> Operation / Tool</div>
    <div class="legend-item"><div class="legend-swatch" style="background:{OUTPUT_FILL};border-radius:8px"></div> Output Data</div>
    {'<div class="legend-item"><div class="legend-swatch" style="background:#fff;border:2px dashed ' + CROSS_STROKE + ';border-radius:2px"></div> Cross-branch Flow</div>' if is_dag and cross_row_edges else ''}
  </div>
  <svg id="flowchartSvg" viewBox="0 0 {svg_width} {svg_height}">
    <defs>
      <marker id="mb_arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
        <polygon points="0 0,10 3.5,0 7" fill="#555"/>
      </marker>
      <marker id="mb_arrow_cross" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
        <polygon points="0 0,10 3.5,0 7" fill="{CROSS_STROKE}"/>
      </marker>
    </defs>
    <g id="zoomGroup">
    {svg_body}
    </g>
  </svg>
</div>

<div class="popup-overlay" id="popupOverlay" onclick="closePopup(event)">
  <div class="popup-box" onclick="event.stopPropagation()">
    <button class="popup-close" onclick="closePopup()">&times;</button>
    <div id="popupContent"></div>
  </div>
</div>

<script>
const details = """ + json.dumps(details) + """;

let scale=1, panX=0, panY=0, isPanning=false, _wasDragged=false, psX=0, psY=0, _mdX=0, _mdY=0;
const PT=4;
const MN=0.2, MX=5, ZS=0.15;
const svg=document.getElementById('flowchartSvg');
const zg=document.getElementById('zoomGroup');
const gp=document.getElementById('graphPanel');

function applyTransform(){
  zg.setAttribute('transform','translate('+panX+','+panY+') scale('+scale+')');
  document.getElementById('zoomLevel').textContent=Math.round(scale*100)+'%';
}
function zoomIn(){ scale=Math.min(MX,scale+ZS); applyTransform(); }
function zoomOut(){ scale=Math.max(MN,scale-ZS); applyTransform(); }
function zoomReset(){ scale=1;panX=0;panY=0;applyTransform(); }
function zoomFit(){
  var b=zg.getBBox(),v=svg.viewBox.baseVal,p=30;
  scale=Math.min((v.width-p*2)/b.width,(v.height-p*2)/b.height,MX);
  panX=(v.width/scale-b.width)/2-b.x+p/scale;
  panY=(v.height/scale-b.height)/2-b.y+p/scale;
  applyTransform();
}

gp.addEventListener('wheel',function(e){
  e.preventDefault();
  var r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
  var sx=((e.clientX-r.left)/r.width)*v.width, sy=((e.clientY-r.top)/r.height)*v.height;
  var os=scale;
  scale=e.deltaY<0?Math.min(MX,scale+ZS):Math.max(MN,scale-ZS);
  panX=sx/scale-sx/os+panX; panY=sy/scale-sy/os+panY; applyTransform();
},{passive:false});

gp.addEventListener('mousedown',function(e){
  if(e.button===0||e.button===1){
    e.preventDefault();isPanning=true;_wasDragged=false;
    psX=e.clientX;psY=e.clientY;_mdX=e.clientX;_mdY=e.clientY;
    gp.style.cursor='grabbing';
  }
});
window.addEventListener('mousemove',function(e){
  if(!isPanning)return;
  var dist=Math.abs(e.clientX-_mdX)+Math.abs(e.clientY-_mdY);
  if(dist>PT)_wasDragged=true;
  var v=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
  panX+=(e.clientX-psX)*(v.width/r.width)/scale;
  panY+=(e.clientY-psY)*(v.height/r.height)/scale;
  psX=e.clientX;psY=e.clientY;applyTransform();
});
window.addEventListener('mouseup',function(){ if(isPanning){isPanning=false;gp.style.cursor='';} });
gp.addEventListener('click',function(e){ if(_wasDragged){e.stopPropagation();_wasDragged=false;} },true);

function showPopup(nodeId){
  var c=document.getElementById('popupContent');
  var o=document.getElementById('popupOverlay');
  c.innerHTML=details[nodeId]||'<p style="color:#999">No details.</p>';
  o.classList.add('visible');
  // reset all highlights
  document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
    el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
    el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
  });
  var cl=event.currentTarget.querySelector('ellipse,rect');
  if(cl){ cl.setAttribute('stroke-width','3.5'); cl.setAttribute('stroke','#FFD700'); }
}
function closePopup(e){
  if(e&&e.target!==document.getElementById('popupOverlay'))return;
  document.getElementById('popupOverlay').classList.remove('visible');
  document.querySelectorAll('.node-box ellipse,.node-box rect').forEach(function(el){
    el.setAttribute('stroke',el.getAttribute('data-os')||'#333');
    el.setAttribute('stroke-width',el.tagName==='rect'?'2':'1.5');
  });
}
document.addEventListener('keydown',function(e){ if(e.key==='Escape')closePopup(); });
window.addEventListener('DOMContentLoaded',function(){ setTimeout(zoomFit,100); });
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Model Builder workflow saved to: {output_path}")
    return output_path


def show_plan_overview_static(plan):
    """Render a fixed-layout top-level overview: RQ → Objectives → Steps.

    Uses matplotlib with a fixed grid layout — no floating/physics.
    Objectives are arranged as parallel columns, steps flow top-to-bottom.

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    objectives = plan.get("objectives", [])
    rq = plan.get("research_question", "Research Question")
    colors_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    # Layout parameters
    col_width = 4.0        # horizontal spacing between objective columns
    row_height = 1.2       # vertical spacing between steps
    rq_y = 0               # RQ row
    obj_y = -1.8           # Objective row
    step_start_y = -3.5    # First step row

    n_obj = len(objectives)
    max_steps = max((len(obj.get("steps", [])) for obj in objectives), default=1)

    fig_width = max(10, n_obj * col_width + 2)
    fig_height = max(6, abs(step_start_y) + max_steps * row_height + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Center objectives horizontally
    total_width = (n_obj - 1) * col_width
    start_x = -total_width / 2

    def draw_box(x, y, text, color, width=3.2, height=0.9, fontsize=8, text_color="black"):
        box = mpatches.FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                       boxstyle="round,pad=0.15", facecolor=color,
                                       edgecolor="#333", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=text_color, wrap=True, fontweight="bold",
                bbox=dict(facecolor="none", edgecolor="none"))
        return (x, y)

    def draw_arrow(start, end, color="#666"):
        ax.annotate("", xy=(end[0], end[1] + 0.45), xytext=(start[0], start[1] - 0.45),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    # Draw RQ node
    rq_pos = draw_box(0, rq_y, rq[:70], "#2C3E50", width=min(fig_width - 2, 8),
                       height=1.0, fontsize=9, text_color="white")

    for i, obj in enumerate(objectives):
        obj_num = obj["number"]
        color = colors_list[(obj_num - 1) % len(colors_list)]
        x = start_x + i * col_width
        obj_title = obj.get("title", "")

        # Draw objective
        obj_pos = draw_box(x, obj_y, f"Obj {obj_num}\n{obj_title[:40]}", color,
                           fontsize=7, text_color="white")
        draw_arrow(rq_pos, obj_pos, color="#555")

        # Draw steps
        steps = sorted(obj.get("steps", []), key=lambda s: s.get("step_number", 0))
        prev_pos = obj_pos
        for j, step in enumerate(steps):
            step_num = step["step_number"]
            operation = step.get("operation", "")
            sy = step_start_y - j * row_height

            step_label = f"S{obj_num}.{step_num}\n{operation[:28]}"
            # Lighter shade for steps
            import colorsys
            r, g, b = int(color[1:3], 16) / 255, int(color[3:5], 16) / 255, int(color[5:7], 16) / 255
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            lr, lg, lb = colorsys.hls_to_rgb(h, min(l + 0.25, 0.92), s)
            light_color = f"#{int(lr*255):02x}{int(lg*255):02x}{int(lb*255):02x}"

            step_pos = draw_box(x, sy, step_label, light_color, width=3.0,
                                height=0.85, fontsize=7)
            draw_arrow(prev_pos, step_pos, color=color)
            prev_pos = step_pos

    title = plan.get("title", "Research Plan Overview")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.set_xlim(-fig_width / 2, fig_width / 2)
    ax.set_ylim(step_start_y - max_steps * row_height - 0.5, rq_y + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


def show_objective_overview_popup(ctx):
    print("DISPLAYING THE VISUAL WORKFLOW IS UNDER DEVELOPMENT")


def show_graph(G):
    """Render a NetworkX DiGraph as an interactive pyvis HTML network.

    Args:
        G: nx.DiGraph (e.g. from build_plan_dag)

    Returns:
        pyvis Network object — call nt.show('filename.html') to save/display
    """
    from pyvis.network import Network

    nt = Network(height="700px", width="100%", directed=True, notebook=True)
    nt.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    colors_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "data")
        obj_num = data.get("objective", 0)
        desc = data.get("description", "")
        label = node[:30]

        if node_type == "operation":
            operation = data.get("operation", "")
            label = f'{operation[:30]}'
            color = colors_list[(obj_num - 1) % len(colors_list)] if obj_num > 0 else "#999"
            title = f"<b>Obj {obj_num} - Step {data.get('step_number', '')}</b><br>{operation}<br><br>{desc}"
            nt.add_node(node, label=label, title=title, color=color,
                        shape="ellipse", size=25)
        else:
            title = f"<b>{node}</b><br>{desc}" if desc else node
            nt.add_node(node, label=label, title=title, color="#D4E6F1",
                        shape="box", size=20)

    for src, dst in G.edges():
        nt.add_edge(src, dst)

    return nt





def create_research_plan_flowchart(structured_plan, output_path="research_plan_flowchart_dag.html"):
    """
    Create an interactive flowchart using Mermaid.
    Shows: Title -> Objectives -> Steps

    Args:
        structured_plan: Dictionary with research plan structure
        output_path: Path to save HTML file

    Returns:
        Path to the saved flowchart
    """
    # Build Mermaid diagram syntax
    mermaid_code = ["graph TD"]

    # Title
    title = structured_plan.get('title', 'Research Plan')
    mermaid_code.append(f'    Title["{title}"]')
    mermaid_code.append('    style Title fill:#lightblue,stroke:#333,stroke-width:2px')

    # Objectives and steps
    objectives = structured_plan.get('objectives', [])
    for obj in objectives:
        obj_id = f"Obj{obj['number']}"
        obj_label = f"Objective {obj['number']}<br/>{obj['title'][:40]}..."

        mermaid_code.append(f'    {obj_id}["{obj_label}"]')
        mermaid_code.append(f'    Title --> {obj_id}')
        mermaid_code.append(f'    style {obj_id} fill:#90EE90,stroke:#333,stroke-width:2px')

        # Steps
        steps = obj.get('steps', [])
        for i, step in enumerate(steps):
            step_id = f"Step{obj['number']}{chr(97+i)}"
            step_text = step.get("description", str(step)) if isinstance(step, dict) else str(step)
            step_label = f"{chr(97+i)}. {step_text[:30]}..."

            mermaid_code.append(f'    {step_id}["{step_label}"]')
            mermaid_code.append(f'    {obj_id} --> {step_id}')
            mermaid_code.append(f'    style {step_id} fill:#FFFFE0,stroke:#333,stroke-width:1px')

    # Create HTML with Mermaid
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Research Plan Flowchart</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
</head>
<body>
    <h1 style="text-align:center;">{title}</h1>
    <div class="mermaid">
{chr(10).join(mermaid_code)}
    </div>
</body>
</html>"""

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Mermaid flowchart saved to: {output_path}")
    print(f"  Open in browser to view interactive diagram")

    return output_path




def _execute_generated_code(ctx, code):
    """
    Execute LLM-generated Python code safely.
    Tracks files created during execution.
    Returns dict with status, output, created files, and any error.
    """
    import io
    import glob as globmod
    import contextlib
    from utils.config import resolve_output_dir

    # Append to ctx logs if available
    logs = ctx.get("logs", [])

    # Snapshot files before execution to detect new and modified outputs
    output_dir = resolve_output_dir(ctx)
    os.makedirs(output_dir, exist_ok=True)
    # Final-outputs subfolder (mirrors spatial_analysis_executor convention).
    final_output_dir = os.path.join(output_dir, "final_outputs")
    os.makedirs(final_output_dir, exist_ok=True)
    _tracked_exts = ['*.csv', '*.png', '*.jpg', '*.jpeg', '*.json', '*.html', '*.txt',
                     '*.gpkg', '*.shp', '*.geojson', '*.md', '*.npy',
                     '*.tif', '*.tiff', '*.parquet', '*.xlsx', '*.xls', '*.pdf',
                     '*.svg', '*.kml', '*.gml', '*.feather', '*.pkl', '*.zip',
                     '*.nc', '*.h5', '*.hdf5', '*.dbf']
    before_mtimes = {}
    for ext in _tracked_exts:
        for fp in globmod.glob(os.path.join(output_dir, '**', ext), recursive=True):
            try:
                before_mtimes[fp] = os.path.getmtime(fp)
            except OSError:
                pass

    def _detect_created_files():
        found = []
        for ext in _tracked_exts:
            for fp in globmod.glob(os.path.join(output_dir, '**', ext), recursive=True):
                try:
                    cur_mtime = os.path.getmtime(fp)
                except OSError:
                    continue
                if fp not in before_mtimes or cur_mtime > before_mtimes[fp]:
                    found.append(fp)
        return sorted(set(found))

    stdout_capture = io.StringIO()
    # Inject OUTPUT_DIR + FINAL_OUTPUT_DIR so generated code has stable,
    # in-scope variables to write artifacts into. Also chdir as a safety net
    # for relative paths.
    # Same helpers the agents' executors inject; the debugger tests its
    # repairs through this function, so they must resolve here too.
    from agents.result_analysis_agent.figure_catalog_v2 import (
        record_figure, record_table, record_data_to_viz,
    )
    exec_globals = {
        "__builtins__": __builtins__,
        "OUTPUT_DIR": output_dir,
        "FINAL_OUTPUT_DIR": final_output_dir,
        "record_figure": record_figure,
        "record_table": record_table,
        "record_data_to_viz": record_data_to_viz,
    }
    _prev_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
    except OSError:
        pass
    try:
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, exec_globals)
            output = stdout_capture.getvalue()

            created_files = _detect_created_files()

            msg = f"Code executed successfully. Created/modified {len(created_files)} file(s)." if created_files else "Code executed successfully. No new files detected."
            logs.append(msg)
            print(msg)

            return {"status": "completed", "output": output, "code": code, "created_files": created_files}

        except Exception as err:
            import traceback
            output = stdout_capture.getvalue()
            error = str(err)
            exception_type = type(err).__name__
            msg = f"Code execution error: {exception_type}: {error}"
            logs.append(msg)
            print(msg)
            # Files written before the crash are moved out of the results
            # (see utils/failed_attempts.py), so none are reported as created.
            from utils.failed_attempts import quarantine_new_files
            quarantine_new_files(output_dir, before_mtimes, _detect_created_files(),
                                 f"{exception_type}: {error}")
            partial_files = []
            return {
                "status": "error",
                "error": error,
                "exception_type": exception_type,
                "error_message": f"{exception_type}: {error}",
                "error_traceback": traceback.format_exc(),
                "output": output,
                "code": code,
                "created_files": partial_files,
            }
    finally:
        try:
            os.chdir(_prev_cwd)
        except OSError:
            pass