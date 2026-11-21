"""
Shared config: read environment variables and centralize defaults.
Keep this tiny; add fields only when you truly need them.
"""
import os
from dataclasses import dataclass
from pathlib import Path
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"  # utils/.env
    load_dotenv()
except ImportError:
    pass


@dataclass
class Configurations:
    def __init__(self):
        # Get project root directory (AGM folder) - go up one level from utils/
        self.root_dir: str = str(Path(__file__).parent.parent.resolve())

        # Output directory as absolute path
        # output_dir_name = os.getenv("OUTPUT_DIR", "Outputs")
        self.output_dir: str = os.path.join(self.root_dir, "Outputs")

        self.use_llm: bool = os.getenv("USE_LLM", "0").lower() in {"1", "true"}

def load_configurations():
    """Load configuration settings."""
    return Configurations()


def resolve_output_dir(ctx):
    """Resolve the output directory for this ctx, with strict enforcement
    when running inside a task/conversation.

    Rules:
      - If ctx["task_id"] is set (WebUI/task-scoped workflow), ctx["output_dir"]
        MUST also be set. Missing it raises RuntimeError — this prevents silent
        leakage of outputs into the global Outputs/ root when a ctx is
        misrouted without its per-task folder.
      - If ctx["task_id"] is not set (standalone notebook, ad-hoc script),
        fall back to the global settings.output_dir.
      - In strict mode, also verifies the configured output_dir lives under
        settings.output_dir (i.e., it's a real per-task subdirectory) and
        ends with task_id when possible.
    """
    settings = load_configurations()
    ctx = ctx or {}
    task_id = ctx.get("task_id")
    output_dir = ctx.get("output_dir")

    if task_id:
        if not output_dir:
            raise RuntimeError(
                f"ctx['output_dir'] is required when ctx['task_id']={task_id!r} is set. "
                f"Refusing to fall back to the global Outputs/ root — that would mix "
                f"artifacts across conversations."
            )
        # Guard against ctx pointing outside the configured Outputs/ root
        abs_out = os.path.abspath(output_dir)
        abs_root = os.path.abspath(settings.output_dir)
        if not (abs_out == abs_root or abs_out.startswith(abs_root + os.sep)):
            raise RuntimeError(
                f"ctx['output_dir']={output_dir!r} is outside the configured "
                f"Outputs/ root ({settings.output_dir}). Refusing to use it."
            )
        return output_dir

    # Standalone context (no task_id): fall back to global outputs dir.
    return output_dir or settings.output_dir



# Configurations = Settings()
