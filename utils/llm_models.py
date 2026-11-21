"""Model and reasoning effort for each named LLM call, read from llm_models.toml.

    from utils.llm_models import llm_model, llm_reasoning
    llm = GIBDChat(model_name=llm_model("research_plan.generate"),
                   **llm_reasoning("research_plan.generate"))

llm_reasoning() returns {} when the file sets no effort for the call, so the
request carries no reasoning_effort at all (not reasoning_effort=None, which
the chat-completion helpers would send as null).

The file is read once per process, so edits take effect after a restart.
"""
from functools import lru_cache
from pathlib import Path

try:
    import tomllib as _toml      # Python 3.11+
except ModuleNotFoundError:      # pragma: no cover - older Pythons
    import tomli as _toml

MODELS_FILE = Path(__file__).with_name("llm_models.toml")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _entry(name, value):
    if isinstance(value, str):
        value = {"model": value}
    model = str(value.get("model") or "").strip()
    effort = value.get("reasoning_effort")
    unknown = set(value) - {"model", "reasoning_effort"}
    if not model or unknown or (effort is not None and effort not in EFFORTS):
        raise ValueError(
            f"{MODELS_FILE.name}: {name} needs a model and, optionally, a reasoning_effort "
            f"from {', '.join(EFFORTS)}; got {value!r}")
    return {"model": model, "reasoning_effort": effort}


def _flatten(table, prefix=""):
    out = {}
    for key, value in table.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and "model" not in value:
            out.update(_flatten(value, name + "."))
        else:
            out[name] = _entry(name, value)
    return out


@lru_cache(maxsize=1)
def all_settings():
    """{call name: {"model": ..., "reasoning_effort": ... or None}}."""
    with open(MODELS_FILE, "rb") as f:
        return _flatten(_toml.load(f))


def all_models():
    """{call name: model} for every LLM call listed in the file."""
    return {name: s["model"] for name, s in all_settings().items()}


def _settings(name):
    try:
        return all_settings()[name]
    except KeyError:
        raise KeyError(f"No settings for LLM call {name!r} in {MODELS_FILE}") from None


def llm_model(name):
    """The model for the LLM call called *name*, e.g. "research_plan.generate"."""
    return _settings(name)["model"]


def llm_effort(name):
    """The reasoning effort for *name*, or None when the file sets none."""
    return _settings(name)["reasoning_effort"]


def llm_reasoning(name):
    """Keyword arguments carrying *name*'s reasoning effort: {} when none is set."""
    effort = llm_effort(name)
    return {"reasoning_effort": effort} if effort else {}
