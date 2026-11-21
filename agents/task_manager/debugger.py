import io
import json
import os
import contextlib
import glob as globmod

from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from utils.llm_models import llm_model, llm_reasoning
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from utils.toolbase.skill_lookup_tool import build_skill_lookup_tool

import utils.agm_helper as helper
from utils.config import load_configurations, resolve_output_dir
from utils.output_path_guard import install_output_path_guard
from utils.python_exe import python_executable
from agents.task_manager import TaskManager
task_manager = TaskManager()


# ── Error Taxonomy ────────────────────────────────────────────────────────────

ERROR_TYPES = {
    "import_error": {
        "label": "Missing or Broken Import",
        "exceptions": ["ModuleNotFoundError", "ImportError"],
        "description": "A required library or module is missing or incorrectly imported.",
    },
    "column_error": {
        "label": "Column / Key Error",
        "exceptions": ["KeyError"],
        "description": "The code references a DataFrame column or dict key that does not exist.",
    },
    "file_error": {
        "label": "File / Path Error",
        "exceptions": ["FileNotFoundError", "OSError", "PermissionError"],
        "description": "A file path is wrong, missing, or inaccessible.",
    },
    "syntax_error": {
        "label": "Syntax Error",
        "exceptions": ["SyntaxError", "IndentationError"],
        "description": "The LLM generated Python code with invalid syntax.",
    },
    "type_error": {
        "label": "Type / Value Error",
        "exceptions": ["TypeError", "ValueError"],
        "description": "Wrong types used in an operation or invalid values passed to a function.",
    },
    "logic_error": {
        "label": "Logic / Runtime Error",
        "exceptions": ["AttributeError", "IndexError", "ZeroDivisionError", "NameError"],
        "description": "The code logic is incorrect — wrong attribute, out-of-bounds index, division by zero, etc.",
    },
    "spatial_error": {
        "label": "Spatial / Geometry Error",
        "exceptions": ["CRSError", "TopologicalError", "GEOSException"],
        "description": "CRS mismatch, invalid geometry, or geopandas/shapely/pyproj-specific failures.",
    },
    "unknown": {
        "label": "Unknown Error",
        "exceptions": [],
        "description": "Error does not match any known category. Escalate for human review.",
    },
}

VALID_ERROR_TYPES = list(ERROR_TYPES.keys())


# ── Investigation Tools ──────────────────────────────────────────────────────
# These are built as closures inside error_debug so they can access ctx.

def _build_tools(ctx):
    """Build investigation tools as closures over the pipeline context."""

    @tool
    def inspect_dataset_columns(dataset_path: str) -> str:
        """Read a dataset file (gpkg, csv, shp) and return its column names, dtypes, shape, and first 3 rows.
        Use this to verify whether a column actually exists in the data."""
        try:
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext in (".gpkg", ".shp", ".geojson"):
                import geopandas as gpd
                df = gpd.read_file(dataset_path, rows=5)
            elif ext == ".csv":
                import pandas as pd
                df = pd.read_csv(dataset_path, nrows=5)
            else:
                return f"Unsupported file type: {ext}"

            info = {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "head_3_rows": df.head(3).to_string()
            }
            return json.dumps(info, indent=2, default=str)
        except Exception as e:
            return f"Error reading dataset: {e}"

    @tool
    def read_code_file(file_path: str) -> str:
        """Read and return the contents of a Python code file.
        Use this to inspect generated code from previous or failed steps."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    @tool
    def check_file_exists(file_path: str) -> str:
        """Check if a file exists and return its size. Use this to verify
        whether output files from previous steps were actually created."""
        exists = os.path.exists(file_path)
        if exists:
            size = os.path.getsize(file_path)
            return json.dumps({"exists": True, "size_bytes": size, "path": file_path})
        return json.dumps({"exists": False, "path": file_path})


    @tool
    def execute_full_code(code: str) -> str:
        """Execute a COMPLETE Python script (any length) and return its output, error,
        and list of created files. Use this to test your full fixed code before submitting."""
        result = helper._execute_generated_code(ctx, code)
        return json.dumps({
            "status": result["status"],
            "output": result.get("output", "")[:2000],
            "error": result.get("error"),
            "error_traceback": result.get("error_traceback", ""),
            "created_files": result.get("created_files", []),
        }, indent=2)

    # ── import_error tools ─────────────────────────────────────────────────

    @tool
    def check_module_installed(module_name: str) -> str:
        """Check if a Python module/package is installed and importable.
        Use this when you suspect a ModuleNotFoundError or ImportError.
        Example: check_module_installed('spreg') or check_module_installed('esda')"""
        try:
            import importlib
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            location = getattr(mod, "__file__", "unknown")
            return json.dumps({"installed": True, "module": module_name, "version": version, "location": location})
        except ImportError as e:
            return json.dumps({"installed": False, "module": module_name, "error": str(e)})

    @tool
    def install_python_package(package_name: str) -> str:
        """Install a Python package using pip. Use this when check_module_installed
        confirms a module is NOT installed and the code needs it.
        The package_name should be the pip package name (e.g., 'esda', 'spreg', 'mapclassify').
        Example: install_python_package('mapclassify')"""
        import subprocess
        import sys
        try:
            result = subprocess.run(
                [python_executable(), "-m", "pip", "install", package_name],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return json.dumps({
                    "success": True,
                    "package": package_name,
                    "message": f"Successfully installed {package_name}",
                    "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                })
            else:
                return json.dumps({
                    "success": False,
                    "package": package_name,
                    "error": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                })
        except subprocess.TimeoutExpired:
            return json.dumps({"success": False, "package": package_name, "error": "Installation timed out after 120 seconds"})
        except Exception as e:
            return json.dumps({"success": False, "package": package_name, "error": str(e)})

    # ── column_error / type_error tools ──────────────────────────────────────

    @tool
    def inspect_column_detail(dataset_path: str, column_name: str) -> str:
        """Inspect a specific column in a dataset: dtype, null count, unique count,
        min/max for numeric, and sample values. Use this to diagnose column type mismatches,
        unexpected nulls, or wrong data types.
        Example: inspect_column_detail('C:/data/file.gpkg', '%Poverty')"""
        try:
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext in (".gpkg", ".shp", ".geojson"):
                import geopandas as gpd
                df = gpd.read_file(dataset_path)
            elif ext == ".csv":
                import pandas as pd
                df = pd.read_csv(dataset_path)
            else:
                return f"Unsupported file type: {ext}"

            if column_name not in df.columns:
                # Suggest similar columns
                import difflib
                close = difflib.get_close_matches(column_name, df.columns.tolist(), n=5, cutoff=0.4)
                return json.dumps({
                    "error": f"Column '{column_name}' not found",
                    "similar_columns": close,
                    "all_columns": list(df.columns)
                }, indent=2)

            col = df[column_name]
            info = {
                "column": column_name,
                "dtype": str(col.dtype),
                "null_count": int(col.isnull().sum()),
                "non_null_count": int(col.notnull().sum()),
                "unique_count": int(col.nunique()),
                "sample_values": [str(v) for v in col.dropna().head(5).tolist()],
            }
            if col.dtype.kind in ("i", "f"):  # numeric
                info["min"] = float(col.min()) if col.notnull().any() else None
                info["max"] = float(col.max()) if col.notnull().any() else None
                info["mean"] = float(col.mean()) if col.notnull().any() else None
            return json.dumps(info, indent=2, default=str)
        except Exception as e:
            return f"Error inspecting column: {e}"

    # ── file_error tools ─────────────────────────────────────────────────────

    @tool
    def list_directory_files(directory_path: str, extension: str = "") -> str:
        """List files in a directory, optionally filtered by extension.
        Use this to find the correct file path when a FileNotFoundError occurs.
        Example: list_directory_files('C:/Users/you/project/Outputs', '.csv')"""
        try:
            if not os.path.isdir(directory_path):
                return json.dumps({"error": f"Directory does not exist: {directory_path}"})
            files = []
            for f in os.listdir(directory_path):
                full_path = os.path.join(directory_path, f)
                if os.path.isfile(full_path):
                    if not extension or f.lower().endswith(extension.lower()):
                        files.append({"name": f, "size_bytes": os.path.getsize(full_path)})
            return json.dumps({"directory": directory_path, "file_count": len(files), "files": files[:50]}, indent=2)
        except Exception as e:
            return f"Error listing directory: {e}"

    # ── spatial_error tools ──────────────────────────────────────────────────

    @tool
    def check_spatial_info(dataset_path: str) -> str:
        """Inspect spatial properties of a geospatial dataset: CRS, geometry types,
        validity counts, bounds, and geometry column name. Use this to diagnose CRS
        mismatches, invalid geometries, or missing spatial data.
        Example: check_spatial_info('C:/data/file.gpkg')"""
        try:
            import geopandas as gpd
            gdf = gpd.read_file(dataset_path, rows=100)
            geom_col = gdf.geometry.name
            info = {
                "crs": str(gdf.crs) if gdf.crs else "No CRS set",
                "crs_epsg": gdf.crs.to_epsg() if gdf.crs else None,
                "geometry_column": geom_col,
                "geometry_types": list(gdf.geom_type.unique()),
                "total_rows": len(gdf),
                "null_geometries": int(gdf.geometry.isnull().sum()),
                "invalid_geometries": int((~gdf.geometry.is_valid).sum()),
                "bounds": list(gdf.total_bounds),
            }
            return json.dumps(info, indent=2, default=str)
        except Exception as e:
            return f"Error reading spatial info: {e}"

    # ── syntax_error tools ───────────────────────────────────────────────────



    # ── logic_error tools ────────────────────────────────────────────────────

    @tool
    def get_pipeline_logs(last_n: int = 30) -> str:
        """Retrieve the last N pipeline execution logs. Use this to understand
        the sequence of events leading to the error — what agents ran, what steps
        completed, and where things went wrong.
        Example: get_pipeline_logs(20)"""
        logs = ctx.get("logs", [])
        recent = logs[-last_n:] if last_n > 0 else logs
        return "\n".join(recent) if recent else "No logs available"

    # ── model documentation lookup ──────────────────────────────────────────

    @tool
    def lookup_model_docs(model_name: str) -> str:
        """Look up documentation for a specific model/method from the curated knowledge base.
        Use this when a model (e.g., GWR, OLS, Moran's I, GM_Lag) encounters a logic error
        and you need to check the correct import path, parameters, or usage example.
        Returns the model's documentation if found, or a message that it's not in the KB.
        Example: lookup_model_docs('GWR') or lookup_model_docs('Moran')"""
        from agents.modeling_agent.model_base.curated_model_kb import CURATED_MODEL_KB
        query = model_name.lower()
        for category, models in CURATED_MODEL_KB.items():
            for m in models:
                name_lower = m["name"].lower()
                import_short = m["import_path"].split(".")[-1].lower()
                if query in name_lower or query == import_short or name_lower.startswith(query):
                    return json.dumps({
                        "found": True,
                        "name": m["name"],
                        "category": category,
                        "import_path": m["import_path"],
                        "key_params": m["key_params"],
                        "use_when": m["use_when"],
                        "example": m["example"],
                    }, indent=2)
        return json.dumps({
            "found": False,
            "message": f"No documentation for '{model_name}' in the curated KB. Use your own knowledge."
        })

    # ── Previous step context ───────────────────────────────────────────

    @tool
    def get_previous_step_outputs(objective_key: str = "", step_index: int = 0) -> str:
        """Get execution details and created files from a previous step.
        Use this when a file is missing — the previous step may have saved it
        under a different name or path than expected.

        If no arguments are given, returns ALL completed steps with their created files.
        If objective_key and step_index are given, returns that specific step's details.

        Example: get_previous_step_outputs('objective_1', 2)"""
        step_results = ctx.get("artifacts", {}).get("step_execution_results", {})

        if objective_key and step_index:
            steps = step_results.get(objective_key, [])
            for s in steps:
                if s.get("step_index") == step_index:
                    inner = s.get("result", {}) if isinstance(s.get("result"), dict) else {}
                    return json.dumps({
                        "step_index": s.get("step_index"),
                        "description": s.get("step_description", ""),
                        "status": s.get("status", ""),
                        "agent": s.get("agent_name", ""),
                        "created_files": inner.get("created_files", []),
                        "code_file": inner.get("code_file", ""),
                        "output": (inner.get("output", "") or "")[:1000],
                        "error": s.get("error_message"),
                    }, indent=2, default=str)
            return f"No step {step_index} found in {objective_key}."

        # Return summary of ALL completed steps with their created files
        summary = []
        for obj_key, steps in step_results.items():
            for s in steps:
                inner = s.get("result", {}) if isinstance(s.get("result"), dict) else {}
                created = inner.get("created_files", [])
                if created:
                    summary.append({
                        "objective": obj_key,
                        "step_index": s.get("step_index"),
                        "description": s.get("step_description", "")[:80],
                        "status": s.get("status", ""),
                        "created_files": created,
                    })
        if not summary:
            return "No completed steps with output files found."
        return json.dumps(summary, indent=2, default=str)

    return [
        # General investigation
        inspect_dataset_columns, read_code_file, check_file_exists, execute_full_code,
        # import_error
        check_module_installed, install_python_package,
        # column_error / type_error
        inspect_column_detail,
        # file_error
        list_directory_files, get_previous_step_outputs,
        # spatial_error
        check_spatial_info,
        # logic_error / general
        get_pipeline_logs,
        # model documentation
        lookup_model_docs,
        # skill documents (detailed code examples for GWR, correlation, etc.)
        build_skill_lookup_tool(),
    ]


# ── Agentic Error Debugger ────────────────────────────────────────────────────

MAX_TOOL_ITERATIONS = 20
MAX_DEBUG_ATTEMPTS = 2


def _cancelled_result():
    """Return a standard cancelled result dict."""
    return {
        "error_analysis": {
            "error_type": "unknown",
            "diagnosis": "[Cancelled by user]",
            "fix_approach": "",
            "confidence": "low",
        },
        "repair_result": {
            "status": "cancelled",
            "fixed_code": "",
            "output": "",
            "created_files": [],
            "code_file": "",
            "success": False,
            "notes": "Cancelled by user",
        },
    }


_TOOL_LABELS = {
    "inspect_dataset_columns":   "Inspecting dataset columns",
    "inspect_column_detail":     "Inspecting column details",
    "read_code_file":            "Reading source file",
    "execute_full_code":         "Executing fixed code to verify",
    "get_pipeline_logs":         "Reading pipeline logs",
    "get_previous_step_outputs": "Looking at upstream step outputs",
    "get_failed_line":           "Locating the failing line",
    "list_available_files":      "Listing available files",
    "skill_lookup":              "Looking up relevant skill",
}


def _describe_tool_call(tool_name, tool_args):
    """Build a short human-readable status line for a debugger tool call."""
    label = _TOOL_LABELS.get(tool_name, f"Running {tool_name}")
    hint_keys = ("dataset_path", "file_path", "path", "column_name",
                 "objective_key", "step_index", "query")
    hints = []
    for k in hint_keys:
        v = tool_args.get(k) if isinstance(tool_args, dict) else None
        if v not in (None, ""):
            if isinstance(v, str) and len(v) > 60:
                v = "…" + v[-57:]
            hints.append(f"{k}={v}")
            if len(hints) >= 2:
                break
    return f"{label} ({', '.join(hints)})" if hints else label


def error_debug(ctx, failed_step):
    """
    Agentic error debugger: analyzes the error AND generates fixed code in one pass.

    The LLM uses investigation tools to understand the root cause, then produces
    both a diagnosis and the complete fixed Python code. If the fix fails on execution,
    the error is fed back for another attempt.

    Args:
        ctx: dict           # Full pipeline context (artifacts, logs, etc.)
        failed_step: dict   # Full result dict for the failed step

    Returns:
        Dict with 'error_analysis' and 'repair_result'.
    """

    # ── Extract context (handle field-name variants from different callers) ──
    step_index = failed_step.get("step_index", failed_step.get("step_idx", ""))
    step_description = failed_step.get("step_description",
                                       failed_step.get("operation_description", ""))
    error_message = failed_step.get("error_message", "")
    failed_code = failed_step.get("failed_code", "")

#     # Input / output paths
    input_paths_summary = failed_step.get("input_paths_summary", "")
    output_node = failed_step.get("output_node", "")
    output_node_path = failed_step.get("output_node_path", "")

    output_dir = resolve_output_dir(ctx)

    # Optional user feedback from a previous debug round
    user_feedback = failed_step.get("_user_debug_feedback", "")

    # ── Build tools and bind to LLM ──────────────────────────────────────────
    tools = _build_tools(ctx)
    try:
        import os, hashlib
        from flask import has_request_context
        _k = (helper.load_OpenAI_key() or "") if hasattr(helper, "load_OpenAI_key") else (os.environ.get("OPENAI_API_KEY", "") or "")
        print(f"[KEYCHECK debugger] in_request_ctx={has_request_context()} tail={_k[-6:]!r} len={len(_k)} sha1={hashlib.sha1(_k.encode()).hexdigest()[:8]}")
    except Exception as _e:
        print(f"[KEYCHECK debugger] failed: {_e}")
    llm = GIBDChat(
        # gibd_user_key=helper.load_OpenAI_key(),
        model_name=llm_model("execution.debugger"), **llm_reasoning("execution.debugger")
        )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    stream_cb = ctx.get("_stream_callback")

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = f"""You are an expert code debugging and repair agent for a geospatial research pipeline.
    Your task is to analyze the error, diagnose the root cause, and generate fixed code.

    ## Workflow:

    1. **Identify the failure point.**
       - Use `get_failed_line` to locate the exact line that raised the error.
       - Read the full traceback and surrounding code for context.

    2. **Diagnose the root cause.**
       - Explain *why* the error occurred — don't guess, confirm by inspecting the data, types, or state involved.
       - Use tools only when inspection is necessary (e.g., checking a DataFrame's columns or CRS). If the cause is clear from the traceback alone, no tool call is needed.

    3. **Generate the fix.**
       - Correct only what is needed to resolve the error. Preserve the original logic and style.
       - If the fix requires a new import, include it.

    4. **ALWAYS verify the fix before responding.**
       - After generating the fixed code, you MUST call `execute_full_code` with the complete fixed script.
       - If execution succeeds, proceed to give your final response.
       - If execution fails, read the new error, fix the code, and call `execute_full_code` again.
       - Repeat until the code runs successfully or you are certain the error is unrecoverable.
       - NEVER give your final response without first verifying the fix through `execute_full_code`.

    ## Output Format:
    You MUST respond using the exact structure below. Do not deviate.

    <diagnosis>
      <failed_line>The exact line number and code that raised the error</failed_line>
      <error_type>The exception class (e.g., KeyError, CRSError, ValueError)</error_type>
      <root_cause>A concise explanation of why the error occurred</root_cause>
    </diagnosis>

    <fix>
      <description>One-sentence summary of what the fix does</description>
      <code>
      # Paste the COMPLETE fixed Python script here (all imports, all logic).
      # It must be self-contained and runnable on its own.
      # Add a brief inline comment on each changed line.
      </code>
      <changed_lines>List the line numbers that were modified, added, or removed</changed_lines>
    </fix>

    <validation>
      <check>Describe how to verify the fix works (e.g., expected output, assertion, or test case)</check>
      <risk>Note any side effects or assumptions the fix relies on (write "None" if straightforward)</risk>
    </validation>

    ## Rules:
    - ONLY fix the error in the provided code. Your job is to make the SAME code run without errors.
    - Do NOT use tools unless the error specifically requires it. If the error message tells you everything, fix it directly.
    - Never fabricate column names, file paths, or CRS codes. If uncertain, use a tool to confirm.
    - If the error has multiple possible causes, state the most likely one and note alternatives in <risk>.
    - If the error is unrecoverable (e.g., missing input file), say so clearly instead of generating a speculative fix.
    - Save output files to: {output_dir}
    - Two variables are already defined in the execution scope: `OUTPUT_DIR` (equals `{output_dir}`, used for ALL intermediate files) and `FINAL_OUTPUT_DIR` (a `final_outputs/` subfolder, used ONLY for the final answer file(s) the user wants to see as WebUI tiles). Use `os.path.join(OUTPUT_DIR, <filename>)` for intermediates and `os.path.join(FINAL_OUTPUT_DIR, <filename>)` for the 1-3 final answer artifacts (final map, final result table, optional summary). Do NOT redefine either variable and do NOT use bare relative paths. Preserve whichever directory the original failing code was using for each file unless the bug requires moving it.
    - For matplotlib figures, ALWAYS use `plt.savefig(<path>, dpi=150, bbox_inches='tight')` followed by `plt.close()`. Use FINAL_OUTPUT_DIR for the final answer map(s) and OUTPUT_DIR for diagnostic / intermediate plots. NEVER call `plt.show()` — the code is executed via `exec()` non-interactively.
    - Do NOT use `if __name__ == "__main__"` guards
  
"""


# ── Initial user message ──────────────────────────────────────────────────
    # Build operation_codes section if provided (assembly-level failures)
    operation_codes = failed_step.get("operation_codes", [])
    op_codes_section = ""
    if operation_codes:
        op_blocks = []
        for op in operation_codes:
            fn_code = op.get("code", "").strip()
            if fn_code:
                op_blocks.append(f"### {op['node_name']}\n```python\n{fn_code}\n```")
        if op_blocks:
            op_codes_section = (
                "\n\n## Operation Function Definitions\n"
                "CRITICAL: The fixed code MUST include ALL of these function definitions exactly as given. "
                "Do NOT omit any of them.\n\n"
                + "\n\n".join(op_blocks)
            )

    user_prompt = f"""Fix the error in the code below.

    ## Failed Code:
    ```python
    {failed_code if failed_code else "Not available"}
    ```

    ## Error:
    {error_message}

    ## Context:
    - **Step {step_index}**: {step_description}

    ## Paths:
    ### Inputs (read from):
    {input_paths_summary if input_paths_summary else "Not specified"}

    ### Output (save to):
    - {output_node}: {output_node_path}
    {op_codes_section}
    """


    if user_feedback:
        user_prompt += f"\n\n## User Feedback from Previous Debug Round:\n{user_feedback}"

    # ── Conversation messages (persist across attempts) ──────────────────────
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # ── Outer attempt loop: investigate → fix → execute → feedback ───────────
    error_analysis = {}
    fixed_code = ""

    for attempt in range(1, MAX_DEBUG_ATTEMPTS + 1):
        if helper.check_cancelled(ctx):
            return _cancelled_result()

        print(f"\n[DEBUGGER] Attempt {attempt}/{MAX_DEBUG_ATTEMPTS}")
        if stream_cb:
            stream_cb(f"\n🛠️ Debug attempt {attempt}/{MAX_DEBUG_ATTEMPTS} — analyzing error...\n")

        # ── Agentic tool loop: let LLM investigate using tools ────────────
        final_response = None
        for iteration in range(MAX_TOOL_ITERATIONS):
            # Heartbeat before each LLM turn — reasoning models think silently
            # for several seconds before emitting the first token, so surface
            # a progress line so the user isn't staring at a frozen card.
            if stream_cb:
                if iteration == 0:
                    stream_cb(f"\n🧠 Thinking (attempt {attempt})...\n")
                else:
                    stream_cb(f"\n🧠 Reviewing tool output (turn {iteration + 1})...\n")

            try:
                # Stream this turn (single round-trip). Content tokens flow live,
                # tool_call_chunks accumulate into structured tool_calls.
                if stream_cb:
                    accumulated = None
                    for chunk in llm_with_tools.stream(messages):
                        token = chunk.content if hasattr(chunk, "content") else ""
                        if token:
                            stream_cb(token)
                        accumulated = chunk if accumulated is None else accumulated + chunk
                    response = accumulated
                else:
                    response = llm_with_tools.invoke(messages)
            except Exception as e:
                print(f"[DEBUGGER] LLM invocation error: {e}")
                if stream_cb:
                    stream_cb(f"\n⚠️ LLM error: {e}\n")
                break

            if response is None:
                break

            if response.content:
                print(response.content)

            if response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    print(f"  >> Tool call: {tool_name}({json.dumps(tool_args, default=str)})")

                    # Friendly per-tool status so the user sees *what* is being
                    # inspected, not just a bare "[TOOL CALL]" line.
                    tool_label = _describe_tool_call(tool_name, tool_args)
                    if stream_cb:
                        stream_cb(f"\n🔧 {tool_label}\n")

                    tool_fn = tool_map.get(tool_name)
                    if tool_fn:
                        tool_result = tool_fn.invoke(tool_args)
                        print(f"  << Result: {str(tool_result)}")
                        if stream_cb:
                            stream_cb(f"   ✓ {tool_name} done\n")
                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    if stream_cb:
                        stream_cb(f"\n```\n{str(tool_result)}\n```\n")
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
            else:
                # No tool calls — this is the final response
                final_response = response
                if stream_cb:
                    stream_cb(f"\n✍️ Generating fix...\n")
                break

        if final_response is None:
            print("[DEBUGGER] Max tool iterations reached without a final response.")
            if attempt < MAX_DEBUG_ATTEMPTS:
                messages.append(HumanMessage(content="You have used all tool iterations. Please provide your response now with the diagnosis and fixed code."))
                continue
            break

        # ── Parse the final response ──────────────────────────────────────
        content = final_response.content or ""
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        print(f"\n{'='*60}\n[DEBUGGER] RAW LLM RESPONSE:\n{'='*60}\n{content}\n{'='*60}")

        # ── Extract fields from XML-tagged response ───────────────────────
        parsed = _parse_xml_debug_response(content)

        fixed_code = parsed.get("fixed_code", "")

        task_manager.log(ctx, f"[Debugger]: Diagnosis: {parsed.get('root_cause', '')}")
        print(f"\n[DEBUGGER] PARSED KEYS: {list(parsed.keys())}")
        print(f"[DEBUGGER] fixed_code present: {bool(fixed_code.strip())}, length: {len(fixed_code)}")

        # ── Return diagnosis and fixed code ───────────────────────────────
        return {
            "error_analysis": parsed,
            "repair_result": {
                "status": "fixed" if fixed_code.strip() else "failed",
                "fixed_code": fixed_code,
                "success": bool(fixed_code.strip()),
                "notes": parsed.get("fix_description", ""),
            },
        }

    # ── All attempts exhausted — return failure ───────────────────────────
    return {
        "error_analysis": error_analysis,
        "repair_result": {
            "status": "failed",
            "fixed_code": "",
            "success": False,
            "notes": f"Failed after {MAX_DEBUG_ATTEMPTS} attempts.",
        },
    }




# ── Response Parsing Helpers ──────────────────────────────────────────────────

def _parse_xml_debug_response(content):
    """
    Parse the XML-tagged debug response from the LLM.
    Extracts fields from <diagnosis>, <fix>, and <validation> blocks.
    Falls back gracefully if tags are missing.
    """
    import re

    def _extract_tag(text, tag):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return m.group(1).strip() if m else ""

    parsed = {}

    # ── Diagnosis block ──
    parsed["failed_line"] = _extract_tag(content, "failed_line")
    parsed["error_type"] = _extract_tag(content, "error_type")
    parsed["root_cause"] = _extract_tag(content, "root_cause")

    # ── Fix block ──
    parsed["fix_description"] = _extract_tag(content, "description")
    parsed["changed_lines"] = _extract_tag(content, "changed_lines")

    # Extract code from the <code> tag of the <fix> block — handle markdown
    # fences inside it. Models also write inline <code>x</code> in the
    # diagnosis prose; taking the first <code> match there made a one-line
    # snippet the "fixed script" (run dd5922b8: NameError: gpd). The script is
    # the longest <code> block inside <fix>, or anywhere if <fix> is missing.
    fix_block = _extract_tag(content, "fix") or content
    code_blocks = re.findall(r'<code>(.*?)</code>', fix_block, re.DOTALL)
    code_raw = max((b.strip() for b in code_blocks), key=len, default="")
    if code_raw:
        code_clean = _extract_code_from_markdown(code_raw)
        parsed["fixed_code"] = code_clean if code_clean else code_raw
    else:
        parsed["fixed_code"] = _extract_code_from_markdown(content)

    # ── Validation block ──
    parsed["check"] = _extract_tag(content, "check")
    parsed["risk"] = _extract_tag(content, "risk")

    return parsed


def _extract_code_from_markdown(text):
    """Extract Python code from markdown code blocks, handling common variations."""
    import re
    if not text or not isinstance(text, str):
        return ""
    # Try variations: ```python, ```Python, ```py, ``` python (with space)
    for marker in ['```python', '```Python', '```py', '``` python', '``` Python']:
        if marker in text:
            return text.split(marker, 1)[1].rsplit("```", 1)[0].strip()
    # Case-insensitive regex fallback
    m = re.search(r'```\s*[Pp](?:ython|y)\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Generic ``` block as last resort
    if "```" in text:
        return text.split("```", 1)[1].rsplit("```", 1)[0].strip()
    return ""


def _parse_debug_response(content, messages, llm_with_tools):
    """
    Robustly parse the debug agent's final response.
    Expects a JSON object with both analysis fields (error_type, diagnosis, etc.)
    and fix fields (status, fixed_code, notes).
    Handles: valid JSON, JSON in code blocks, markdown with code, and retries.
    """
    import re

    content = content.strip()

    # Attempt 1: Direct JSON parse
    try:
        clean = content
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        pass

    # Attempt 2: Find JSON object embedded in the text
    try:
        match = re.search(r'\{[\s\S]*"status"\s*:', content)
        if match:
            start = match.start()
            brace_count = 0
            for i, ch in enumerate(content[start:], start):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content[start:i+1]
                        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        pass

    # Attempt 3: LLM returned code in markdown — extract it
    extracted_code = _extract_code_from_markdown(content)
    if extracted_code and len(extracted_code) > 20:
        print("[DEBUGGER] Extracted code from markdown (LLM didn't return JSON)")
        return {
            "error_type": "unknown",
            "diagnosis": "Extracted from non-JSON response",
            "fix_approach": "",
            "confidence": "low",
            "status": "fixed",
            "fixed_code": extracted_code,
            "notes": "Code extracted from markdown response",
        }

    # Attempt 4: Ask the LLM to reformat as JSON
    print("[DEBUGGER] Asking LLM to reformat response as JSON...")
    retry_msg = HumanMessage(content="""Your response was not valid JSON. Please respond with ONLY a JSON object, no markdown:
{"error_type": "...", "diagnosis": "...", "fix_approach": "...", "confidence": "...",
 "status": "fixed", "fixed_code": "your complete python code here", "notes": "what was fixed"}
Or if you cannot fix it:
{"error_type": "unknown", "diagnosis": "...", "fix_approach": "", "confidence": "low",
 "status": "failed", "fixed_code": "", "notes": "why it failed"}""")
    if messages is not None:
        messages.append(retry_msg)

    try:
        retry_response = llm_with_tools.invoke(messages)
        retry_content = retry_response.content.strip()
        if retry_content.startswith("```"):
            retry_content = retry_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(retry_content)
    except Exception:
        pass

    # Final fallback
    print("[DEBUGGER] Could not parse response. Treating as failed.")
    return {
        "error_type": "unknown",
        "diagnosis": f"Could not parse agent response. Raw: {content[:200]}",
        "fix_approach": "",
        "confidence": "low",
        "status": "failed",
        "fixed_code": extracted_code or "",
        "notes": f"Could not parse agent response.",
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_debugger(ctx, failed_step):
    """
    Full debug pipeline: analyze the error and generate fixed code.

    Args:
        ctx: dict           # Full pipeline context
        failed_step: dict   # Failed step result from step_executor

    Returns:
        Dict with 'error_analysis' and 'repair_result'.
    """
    return error_debug(ctx, failed_step)


# ── Shared Exec Utility ───────────────────────────────────────────────────────

def _execute_fixed_code(code: str, output_dir: str = None) -> dict:
    """
    Execute fixed/repaired Python code and capture output.
    Saves the code to a real file first so tracebacks reference a navigable path.
    Returns dict with status, output, created_files, code_file, and error (if any).
    """
    import uuid
    if output_dir is None:
        settings = load_configurations()
        output_dir = settings.output_dir

    # Save fixed code to a real file so tracebacks are navigable
    code_dir = os.path.join(output_dir, "generated_code")
    os.makedirs(code_dir, exist_ok=True)
    code_file = os.path.join(code_dir, f"debug_{str(uuid.uuid4())[:8]}.py")
    with open(code_file, "w", encoding="utf-8") as f:
        f.write(code)

    # Same final-output convention as the spatial analysis executor
    final_output_dir = os.path.join(output_dir, "final_outputs")
    os.makedirs(final_output_dir, exist_ok=True)

    _tracked_exts = ["*.csv", "*.png", "*.jpg", "*.jpeg", "*.json", "*.html", "*.txt",
                     "*.gpkg", "*.shp", "*.geojson", "*.md", "*.npy",
                     "*.tif", "*.tiff", "*.parquet", "*.xlsx", "*.xls", "*.pdf",
                     "*.svg", "*.kml", "*.gml", "*.feather", "*.pkl", "*.zip",
                     "*.nc", "*.h5", "*.hdf5", "*.dbf"]
    before_mtimes = {}
    for ext in _tracked_exts:
        for fp in globmod.glob(os.path.join(output_dir, "**", ext), recursive=True):
            try:
                before_mtimes[fp] = os.path.getmtime(fp)
            except OSError:
                pass

    def _detect_created_files():
        found = []
        for ext in _tracked_exts:
            for fp in globmod.glob(os.path.join(output_dir, "**", ext), recursive=True):
                try:
                    cur_mtime = os.path.getmtime(fp)
                except OSError:
                    continue
                if fp not in before_mtimes or cur_mtime > before_mtimes[fp]:
                    found.append(fp)
        return sorted(set(found))

    stdout_capture = io.StringIO()
    # Inject OUTPUT_DIR + FINAL_OUTPUT_DIR so debugger-fixed code follows the
    # same contract as the original assembly code: intermediates → OUTPUT_DIR,
    # final answer file(s) → FINAL_OUTPUT_DIR.
    # The code-generation prompts tell agents to call the figure-catalog
    # helpers, and the agents' own executors inject them. Without them here a
    # correct repair dies on NameError: record_table at re-execution.
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
            compiled = compile(code, code_file, "exec")
            with contextlib.redirect_stdout(stdout_capture), install_output_path_guard(output_dir, final_output_dir):
                exec(compiled, exec_globals)
            output = stdout_capture.getvalue()

            created_files = _detect_created_files()

            return {"status": "completed", "output": output, "created_files": created_files, "error": None, "code_file": code_file}
        except Exception as e:
            import traceback as tb
            extracted = tb.extract_tb(e.__traceback__)
            code_frames = [f for f in extracted if f.filename.startswith("<step:") or f.filename.startswith("<debug:")]
            if not code_frames:
                code_frames = extracted[-1:] if extracted else []
            frame_lines = ["Traceback (generated code only):"]
            for f in code_frames:
                frame_lines.append(f'  File "{f.filename}", line {f.lineno}, in {f.name}')
                if f.line:
                    frame_lines.append(f'    {f.line}')
            frame_lines.append(f'{type(e).__name__}: {e}')
            code_traceback = "\n".join(frame_lines)
            exec_snapshot = {k: type(v).__name__ for k, v in exec_globals.items() if not k.startswith("__")}
            _exception_type = type(e).__name__
            # Files written before the crash are moved out of the results
            # (see utils/failed_attempts.py), so none are reported as created.
            from utils.failed_attempts import quarantine_new_files
            quarantine_new_files(output_dir, before_mtimes, _detect_created_files(),
                                 f"{_exception_type}: {e}")
            return {
                "status": "error",
                "output": stdout_capture.getvalue(),
                "created_files": [],
                "error": str(e) or f"{_exception_type}",
                "exception_type": _exception_type,
                "full_traceback": code_traceback,
                "var_snapshot": exec_snapshot,
                "code_file": code_file,
            }
    finally:
        try:
            os.chdir(_prev_cwd)
        except OSError:
            pass