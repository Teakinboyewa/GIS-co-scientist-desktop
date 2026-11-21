"""
Single-agent baseline: one LLM, one tool loop, the whole research workflow.

The multi-agent Research mode splits the work across a data agent, an RQ
agent, a planner and reviewer, routed code-generation agents, a debugger, and
manuscript section writers. This mode gives one model the same instructions
(see single_agent_prompts) and the same data-inspection, skill and literature
tools, and lets it run the workflow itself. What differs between the two arms
is the orchestration, which is the variable the baseline is for.

The loop emits the same event types the Research workflow does (status,
llm_stream, step_result, manuscript_complete), so the existing cards render it.
"""

import contextlib
import glob as globmod
import io
import json
import os
import re
import traceback

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import utils.agm_helper as helper
from utils.llm_models import llm_effort, llm_model, llm_reasoning
from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from utils.config import load_configurations
from agents.single_agent.single_agent_prompts import (
    MANUSCRIPT_FILENAME, build_system_prompt, build_user_message,
)


# Model turns before the run is stopped. A full run is roughly 3 stage files,
# 10-25 plan steps with inspection and a retry or two each, literature search,
# and the manuscript -- well under this, so hitting it means the agent is stuck.
MAX_TURNS = 200
# Times the agent is told to keep going after it stops without the manuscript.
MAX_NUDGES = 2
# Tool output kept in the conversation. Long stdout is the fastest way to fill
# the context window, and the agent can always re-read a file.
TOOL_RESULT_CAP = 8000

_TRACKED_EXTS = ('csv', 'png', 'jpg', 'jpeg', 'json', 'html', 'txt', 'gpkg', 'shp',
                 'geojson', 'md', 'npy', 'tif', 'tiff', 'parquet', 'xlsx', 'xls',
                 'pdf', 'svg', 'pkl', 'zip', 'dbf')


def _cap(text, limit=TOOL_RESULT_CAP):
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    return (text[:head] + f"\n\n... [{len(text) - limit} characters omitted] ...\n\n"
            + text[-(limit - head):])


def _content_text(message):
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                          for b in content)
    return content


def _snapshot(output_dir):
    stamps = {}
    for ext in _TRACKED_EXTS:
        for fp in globmod.glob(os.path.join(output_dir, '**', f'*.{ext}'), recursive=True):
            try:
                stamps[fp] = os.path.getmtime(fp)
            except OSError:
                pass
    return stamps


def _changed_files(output_dir, before):
    after = _snapshot(output_dir)
    return sorted(fp for fp, mtime in after.items()
                  if fp not in before or mtime > before[fp])


def _inside(output_dir, relative_path):
    """Resolve a path the agent gave against output_dir, refusing escapes."""
    root = os.path.realpath(output_dir)
    target = os.path.realpath(os.path.join(root, relative_path))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"Path '{relative_path}' is outside the output directory.")
    return target


_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fence(code):
    """The code-generation prompt tells the model to wrap code in a ```python
    block, and the pipeline extracts the code from that block before running
    it. Do the same, so a fenced script is not a NameError on `python`."""
    match = _FENCE.match(code)
    return match.group(1) if match else code


def execute_python(ctx, code):
    """Run one agent script the way the pipeline runs step code: in-process,
    with the figure-catalog helpers and OUTPUT_DIR injected, cwd at the task
    folder. Returns the same shape as ModelingAgent._execute_generated_code."""
    from agents.result_analysis_agent.figure_catalog_v2 import (
        record_figure, record_table, record_data_to_viz,
    )
    code = strip_code_fence(code)
    output_dir = ctx["output_dir"]
    final_output_dir = os.path.join(output_dir, "final_outputs")
    os.makedirs(final_output_dir, exist_ok=True)
    before = _snapshot(output_dir)

    exec_globals = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "record_figure": record_figure,
        "record_table": record_table,
        "record_data_to_viz": record_data_to_viz,
        "OUTPUT_DIR": output_dir,
        "FINAL_OUTPUT_DIR": final_output_dir,
    }
    stdout = io.StringIO()
    prev_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<single_agent_step>", "exec"), exec_globals)
        return {"status": "completed", "output": stdout.getvalue(),
                "created_files": _changed_files(output_dir, before)}
    except BaseException as err:  # SystemExit from sys.exit() in agent code too
        if isinstance(err, KeyboardInterrupt):
            raise
        name = type(err).__name__
        message = f"{name}: {err}" if str(err) else name
        # Same rule as the multi-agent executors: what a failed script created
        # is moved out of the results (utils/failed_attempts.py).
        from utils.failed_attempts import quarantine_new_files
        moved = quarantine_new_files(output_dir, before, _changed_files(output_dir, before), message)
        return {"status": "error", "output": stdout.getvalue(),
                "error_message": message,
                "error_traceback": traceback.format_exc(),
                "created_files": [],
                "quarantined_files": moved}
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass


def build_tools(ctx, on_execution):
    """The agent's tool kit.

    Data inspection and skill lookup are the code-generation agents' own tools;
    the literature tools are the manuscript writers' own. run_python, write_file
    and read_file stand in for what the pipeline does between agents: running
    step code, saving stage outputs, and handing those outputs to the next
    stage.
    """
    from utils.toolbase.modeling_agent_code_generation_tools import (
        build_modeling_agent_codegen_tools,
    )
    from utils.toolbase.skill_lookup_tool import build_skill_lookup_tool
    from agents.reporting_agent.manuscript_tools import build_manuscript_tools

    output_dir = ctx["output_dir"]

    inspect_tools = [t for t in build_modeling_agent_codegen_tools(ctx)
                     if t.name != "lookup_skill_doc"]
    # Unscoped: the single agent does both the data and the modeling work, so
    # it sees every skill rather than one agent's share of them.
    skill_tool = build_skill_lookup_tool(agent=None)
    literature_tools = [t for t in build_manuscript_tools(ctx, mode="literature")
                        if t.name != "run_analysis_snippet"]

    @tool
    def run_python(code: str, purpose: str) -> str:
        """Execute a complete Python script and return its stdout, the files it
        created or modified, and any error with its traceback.

        Each call is a fresh script: variables do not persist between calls.
        OUTPUT_DIR, FINAL_OUTPUT_DIR, record_figure, record_table and
        record_data_to_viz are pre-defined; the working directory is OUTPUT_DIR.

        Args:
            code: The full Python script.
            purpose: One line naming what this script does, e.g.
                "Objective 1, step 2: join ACS tracts to flood zones".
        """
        result = execute_python(ctx, code)
        on_execution(code, purpose, result)
        files = result.get("created_files") or []
        if files:
            helper.update_data_registry(ctx, files)
        lines = [f"status: {result['status']}"]
        if files:
            lines.append("files created/modified:\n" + "\n".join(f"  - {f}" for f in files))
        lines.append("stdout:\n" + (result.get("output") or "(none)"))
        if result["status"] == "error":
            if result.get("quarantined_files"):
                lines.append("files this failed script created were moved to _failed_attempts/ "
                             "and are not results:\n"
                             + "\n".join(f"  - {f}" for f in result["quarantined_files"]))
            lines.append("error: " + result["error_message"])
            lines.append("traceback:\n" + result.get("error_traceback", ""))
        return _cap("\n".join(lines))

    @tool
    def write_file(relative_path: str, content: str) -> str:
        """Write a text file (JSON, markdown, ...) inside the output directory,
        replacing it if it exists. Use it for the stage artifacts:
        data_summary.json, rq_breakdown.json, research_plan.json and
        manuscript.md.

        Args:
            relative_path: Path relative to the output directory.
            content: The full file content.
        """
        try:
            path = _inside(output_dir, relative_path)
        except ValueError as e:
            return f"Error: {e}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        note = ""
        if path.lower().endswith(".json"):
            try:
                json.loads(content)
            except ValueError as e:
                note = f" Warning: the content is not valid JSON ({e})."
        return f"Saved {len(content)} characters to {path}.{note}"

    @tool
    def read_file(relative_path: str) -> str:
        """Read a text file from the output directory (for example a stage
        artifact you saved earlier, or a table a step wrote).

        Args:
            relative_path: Path relative to the output directory.
        """
        try:
            path = _inside(output_dir, relative_path)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return _cap(f.read())
        except (OSError, ValueError) as e:
            return f"Error: {e}"

    @tool
    def list_output_files() -> str:
        """List every file in the output directory with its size."""
        rows = []
        for dirpath, _dirs, files in os.walk(output_dir):
            for name in files:
                fp = os.path.join(dirpath, name)
                rel = os.path.relpath(fp, output_dir)
                if rel.startswith('.') or name == 'workflow_state.json':
                    continue
                try:
                    rows.append(f"{rel} ({os.path.getsize(fp)} bytes)")
                except OSError:
                    continue
        return _cap("\n".join(sorted(rows)) or "(empty)")

    return [*inspect_tools, skill_tool, run_python, write_file, read_file,
            list_output_files, *literature_tools]


def _describe_call(name, args):
    if name == "run_python":
        return f"run_python: {args.get('purpose', '')}"
    if name == "write_file":
        return f"write_file({args.get('relative_path', '')})"
    short = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{name}({short[:160] + '...' if len(short) > 160 else short})"


def run_single_agent(ctx, emit, is_cancelled=lambda: False):
    """Run the agent until it finishes, gives up, or is cancelled.

    emit(event) receives workflow events (dicts) as they happen.
    Returns a summary dict: status, final_message, turns, manuscript_path.
    """
    output_dir = ctx["output_dir"]
    model = llm_model("single_agent.agent")
    llm = GIBDChat(model_name=model, **llm_reasoning("single_agent.agent"))

    # Card bookkeeping: model text and inspection calls go to a log card; each
    # run_python gets its own card, after which a fresh log card starts. Reads
    # top to bottom like a notebook.
    state = {"log_card": 1, "log_open": False, "exec_count": 0}

    def log_step():
        return f"single_agent_log_{state['log_card']}"

    def stream(text):
        if not state["log_open"]:
            emit({"type": "status", "message": "Agent working...", "step": log_step()})
            state["log_open"] = True
        emit({"type": "llm_stream", "content": text, "step": log_step()})

    def on_execution(code, purpose, result):
        state["exec_count"] += 1
        step = f"single_agent_exec_{state['exec_count']}"
        emit({"type": "status", "message": purpose or f"Script {state['exec_count']}",
              "step": step})
        emit({"type": "llm_stream", "content": f"```python\n{code}\n```\n", "step": step})
        emit({
            "type": "step_result",
            "step": step,
            "step_index": state["exec_count"],
            "step_description": purpose,
            "status": result["status"],
            "output": result.get("output", ""),
            "code": code,
            "objective_key": "",
            "artifacts": _artifacts(result.get("created_files") or []),
            "error": result.get("error_message", ""),
            "error_traceback": result.get("error_traceback", ""),
        })
        state["log_card"] += 1
        state["log_open"] = False

    tools = build_tools(ctx, on_execution)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = build_system_prompt(output_dir)
    user_message = build_user_message(ctx["rq"], ctx.get("datasets") or [], output_dir)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    _save_prompts(output_dir, system_prompt, user_message, model)

    manuscript_path = os.path.join(output_dir, MANUSCRIPT_FILENAME)
    nudges = 0
    final_message = ""
    status = "max_turns"
    turn = 0
    transcript = _Transcript(output_dir)

    for turn in range(1, MAX_TURNS + 1):
        if is_cancelled():
            status = "cancelled"
            break

        response = None
        for chunk in llm_with_tools.stream(messages):
            token = _content_text(chunk)
            if token:
                stream(token)
            response = chunk if response is None else response + chunk
        if response is None:
            status = "empty_response"
            break
        messages.append(response)
        transcript.add_assistant(turn, response)

        if not response.tool_calls:
            final_message = _content_text(response)
            if os.path.exists(manuscript_path) or nudges >= MAX_NUDGES:
                status = "completed" if os.path.exists(manuscript_path) else "no_manuscript"
                break
            nudges += 1
            nudge = (f"You stopped before saving {MANUSCRIPT_FILENAME}. The workflow is "
                     "not finished: continue from where you left off, using your tools, "
                     f"until {MANUSCRIPT_FILENAME} is saved.")
            messages.append(HumanMessage(content=nudge))
            transcript.add_user(turn, nudge)
            continue

        for tc in response.tool_calls:
            name, args = tc["name"], tc.get("args") or {}
            if name != "run_python":
                stream(f"\n\n🔧 {_describe_call(name, args)}\n")
            fn = tool_map.get(name)
            if fn is None:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = fn.invoke(args)
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
            result = _cap(result)
            if name not in ("run_python", "write_file"):
                stream(f"```\n{_cap(result, 1500)}\n```\n")
            elif name == "write_file":
                stream(f"{result}\n")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            transcript.add_tool(turn, name, args, result)

    transcript.close(status, turn)
    return {
        "status": status,
        "final_message": final_message,
        "turns": turn,
        "manuscript_path": manuscript_path if os.path.exists(manuscript_path) else "",
    }


def _artifacts(created_files):
    root = load_configurations().output_dir
    out = []
    for fpath in created_files:
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lstrip(".").lower()
        try:
            size = os.path.getsize(fpath)
        except OSError:
            size = 0
        try:
            rel = os.path.relpath(fpath, root).replace("\\", "/")
        except ValueError:
            rel = fname
        out.append({"path": fpath, "relative_path": rel, "filename": fname,
                    "size": size, "type": ext,
                    "is_shapefile_component": ext in ("shx", "dbf", "prj", "cpg")})
    return out


def _save_prompts(output_dir, system_prompt, user_message, model):
    """Keep the exact prompt a run was given next to its outputs, so a result
    can always be traced to the instructions that produced it."""
    folder = os.path.join(output_dir, "single_agent")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "system_prompt.md"), "w", encoding="utf-8") as f:
        f.write(system_prompt)
    with open(os.path.join(folder, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"model": model, "reasoning_effort": llm_effort("single_agent.agent"),
                   "max_turns": MAX_TURNS, "user_message": user_message}, f, indent=2)


class _Transcript:
    """Append-only JSONL record of every turn, for analysing runs afterwards."""

    def __init__(self, output_dir):
        folder = os.path.join(output_dir, "single_agent")
        os.makedirs(folder, exist_ok=True)
        self._f = open(os.path.join(folder, "transcript.jsonl"), "w", encoding="utf-8")

    def _write(self, record):
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()

    def add_assistant(self, turn, message):
        self._write({"turn": turn, "role": "assistant", "content": _content_text(message),
                     "tool_calls": [{"name": tc["name"], "args": tc.get("args")}
                                    for tc in (message.tool_calls or [])]})

    def add_tool(self, turn, name, args, result):
        self._write({"turn": turn, "role": "tool", "name": name, "args": args,
                     "result": result})

    def add_user(self, turn, content):
        self._write({"turn": turn, "role": "user", "content": content})

    def close(self, status, turns):
        self._write({"role": "end", "status": status, "turns": turns})
        self._f.close()


def finalize_manuscript(ctx, task_manager=None):
    """Apply the multi-agent pipeline's deterministic assembly clean-up to the
    agent's manuscript, then convert it to DOCX/PDF.

    The pipeline strips [REF_N] markers, renumbers figures and tables in
    reading order, drops references to figures that are never embedded, and
    fixes table spacing -- none of it an LLM step. Applying the same pass keeps
    formatting differences out of the comparison. The agent's own file is kept
    as manuscript_agent_raw.md.

    Returns (markdown, path) or ("", "") if there is no manuscript.
    """
    from agents.reporting_agent.manuscript_pipeline import (
        _renumber_in_reading_order, _strip_dangling_figure_refs,
        _ensure_blank_line_before_tables,
    )
    output_dir = ctx["output_dir"]
    md_path = os.path.join(output_dir, MANUSCRIPT_FILENAME)
    if not os.path.exists(md_path):
        return "", ""
    with open(md_path, "r", encoding="utf-8") as f:
        raw = f.read()
    with open(os.path.join(output_dir, "manuscript_agent_raw.md"), "w", encoding="utf-8") as f:
        f.write(raw)

    manuscript = re.sub(r'\s*\[REF_\d+\]', '', raw)
    manuscript = _renumber_in_reading_order(manuscript)
    manuscript = _strip_dangling_figure_refs(manuscript)
    manuscript = _ensure_blank_line_before_tables(manuscript)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(manuscript)

    ctx.setdefault("artifacts", {})["manuscript_md"] = md_path
    ctx["artifacts"]["manuscript_content"] = manuscript
    if task_manager is not None:
        try:
            from agents.result_analysis_agent.report_converter import convert_md_report
            convert_md_report(md_path, manuscript, ctx, task_manager, "manuscript")
        except Exception as e:
            task_manager.log(ctx, f"[SINGLE AGENT] DOCX/PDF conversion failed (non-fatal): {e}")
    return manuscript, md_path
