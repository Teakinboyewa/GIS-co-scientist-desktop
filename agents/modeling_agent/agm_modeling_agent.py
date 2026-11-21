'''
#Changelog:


ref: agm_modeling_agent_v1.py

Changed the def final_recommendation_code_generator completely. This new version now only select the one suitable model
'''



import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from utils.llm_models import llm_model, llm_reasoning

# from agents.task_manager.task_solution_testing import data_overview

load_dotenv()
from agents.base import Agent
from agents.modeling_agent.modeling_tools import ModelingTools
from utils.toolbase.modeling_agent_code_generation_tools import build_modeling_agent_codegen_tools
import utils.agm_helper as helper


from agents.task_manager import TaskManager
task_manager = TaskManager()
from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution import Solution as BaseSolution


from utils.lazy_imports import is_installed
# ipywidgets (notebook UI) is imported only where widgets are built.
WIDGETS_AVAILABLE = is_installed("ipywidgets")


def _check_empty_data_files(created_files):
    """
    Inspect created data files and flag any that contain no actual data rows.
    Returns a list of dicts: [{"path": ..., "reason": ...}, ...] for empty files.
    Only checks tabular/vector formats — images and other binaries are skipped.
    """
    import pandas as pd
    empty = []
    _tabular_exts = {".csv", ".gpkg", ".geojson", ".shp", ".json", ".npy"}
    for fpath in created_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in _tabular_exts:
            continue
        try:
            if ext == ".csv":
                df = pd.read_csv(fpath, nrows=1)
                if df.empty:
                    empty.append({"path": fpath, "reason": "CSV has headers but 0 data rows"})
            elif ext in (".gpkg", ".geojson", ".shp"):
                import geopandas as gpd
                gdf = gpd.read_file(fpath, rows=1)
                if gdf.empty:
                    empty.append({"path": fpath, "reason": f"{ext.upper().lstrip('.')} file has 0 features"})
            elif ext == ".json":
                with open(fpath, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                if isinstance(data, list) and len(data) == 0:
                    empty.append({"path": fpath, "reason": "JSON contains empty list"})
                elif isinstance(data, dict) and len(data) == 0:
                    empty.append({"path": fpath, "reason": "JSON contains empty object"})
            elif ext == ".npy":
                import numpy as np
                arr = np.load(fpath, allow_pickle=True)
                if arr.size == 0:
                    empty.append({"path": fpath, "reason": "NumPy array is empty (size=0)"})
        except Exception:
            # If we can't read it, don't flag it — it might be a valid binary format
            pass
    return empty


class ModelingSolution(BaseSolution):
    """Modeling-specific Solution class with custom graph generation"""

    def __init__(self, task, data_overview, model_requirement=None, task_id=None, stream_callback=None):
        super().__init__(task, data_overview, model_requirement=model_requirement, task_id=task_id, stream_callback=stream_callback)
        # Add any modeling-specific initialization here

    def get_LLM_response_for_graph(self, stream=True, execute=True, yield_chunks=True):
        return super().get_LLM_response_for_graph_ModelingAgent(stream=stream, execute=execute, yield_chunks=yield_chunks)

    def load_graph_file(self, file="", interactive=True):
        return super().load_graph_file(file, interactive)
    
    
    
    
class ModelingAgent(Agent):
    name = "ModelingAgent"

    def __init__(self):
        super().__init__()
        self.tools = ModelingTools()
        # Do not cache the API key — ModelingAgent is instantiated at module
        # import time, before any Flask request. Caching here would freeze the
        # stale .env value instead of picking up g.api_key per-request.

    @property
    def api_key(self):
        return helper.load_OpenAI_key()



    @staticmethod
    def _extract_code_block(content):
        """Extract Python code from a markdown code block, handling common variations."""
        import re
        if not content or not isinstance(content, str):
            return ""
        # Try variations: ```python, ```Python, ```py, ``` python (with space)
        for marker in ['```python', '```Python', '```py', '``` python', '``` Python']:
            if marker in content:
                return content.split(marker, 1)[1].rsplit("```", 1)[0].strip()
        # Case-insensitive regex fallback for any ```python/Python/PYTHON variant
        m = re.search(r'```\s*[Pp](?:ython|y)\s*\n(.*?)```', content, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Generic ``` block as last resort
        if "```" in content:
            return content.split("```", 1)[1].rsplit("```", 1)[0].strip()
        return ""

    def create_solution(self, task, data_overview, model_requirement=None):
        """Factory method to create ModelingSolution instances"""
        return ModelingSolution(
            task=task,
            data_overview=data_overview,
            model_requirement=model_requirement
        )

    def _execute_generated_code(self, ctx, code, operation_description):
        """
        Execute LLM-generated Python code safely.
        Tracks files created during execution.
        Returns dict with status, output, created files, and any error.
        code_file: path to the saved .py file — used as the compile filename so
                   tracebacks reference the real file rather than a synthetic name.
        """
        import io
        import glob as globmod
        import contextlib
        from utils.config import load_configurations, resolve_output_dir
        settings = load_configurations()

        self.log(ctx, f"Executing generated code for: {operation_description}")

        # Snapshot files before execution to detect new and modified outputs
        output_dir = resolve_output_dir(ctx)
        _tracked_exts = ['*.csv', '*.png', '*.jpg', '*.jpeg', '*.json', '*.html', '*.txt',
                         '*.gpkg', '*.shp', '*.geojson', '*.md', '*.npy']
        before_mtimes = {}
        for ext in _tracked_exts:
            for fp in globmod.glob(os.path.join(output_dir, '**', ext), recursive=True):
                try:
                    before_mtimes[fp] = os.path.getmtime(fp)
                except OSError:
                    pass

        stdout_capture = io.StringIO()
        # Inject the figure-cataloging helper so LLM-generated code can register
        # figures for the manuscript at write time.
        from agents.result_analysis_agent.figure_catalog_v2 import (
            record_figure, record_table, record_data_to_viz,
        )
        exec_globals = {
            "__builtins__": __builtins__,
            "record_figure": record_figure,
            "record_table": record_table,
            "record_data_to_viz": record_data_to_viz,
            "OUTPUT_DIR": output_dir,
        }
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, exec_globals)
            output = stdout_capture.getvalue()

            # Detect newly created and modified files
            created_files = []
            for ext in _tracked_exts:
                for fp in globmod.glob(os.path.join(output_dir, '**', ext), recursive=True):
                    try:
                        cur_mtime = os.path.getmtime(fp)
                    except OSError:
                        continue
                    if fp not in before_mtimes or cur_mtime > before_mtimes[fp]:
                        created_files.append(fp)
            created_files = sorted(set(created_files))

            if created_files:
                self.log(ctx, f"Code executed successfully. Created/modified {len(created_files)} file(s): {created_files}")
            else:
                self.log(ctx, f"Code executed successfully. No new files detected.")

            # ── Check for empty data files ────────────────────────────────
            empty_data_files = _check_empty_data_files(created_files)
            empty_data_warning = ""
            if empty_data_files:
                details = "; ".join(f"{os.path.basename(f['path'])} ({f['reason']})" for f in empty_data_files)
                empty_data_warning = f"Warning: {len(empty_data_files)} output file(s) contain no data: {details}"
                self.log(ctx, f"[EMPTY DATA] {empty_data_warning}")

            return {
                "status": "completed", "output": output, "code": code,
                "created_files": created_files,
                "empty_data_files": empty_data_files,
                "empty_data_warning": empty_data_warning,
            }

        except Exception as err:
            import traceback
            output = stdout_capture.getvalue()
            exception_type = type(err).__name__
            # Guard against empty-message exceptions (e.g. IterationError with no text):
            # str(err) returns "" which is falsy and would skip the debugger.
            error = str(err) or f"{exception_type}"
            self.log(ctx, f"Code execution error: {error}")
            # Files this failed attempt created must not reach the catalog or
            # the manuscript (see utils/failed_attempts.py).
            from utils.failed_attempts import quarantine_new_files
            _now = [fp for ext in _tracked_exts
                    for fp in globmod.glob(os.path.join(output_dir, '**', ext), recursive=True)]
            _moved = quarantine_new_files(output_dir, before_mtimes, _now, f"{exception_type}: {error}")
            if _moved:
                self.log(ctx, f"[FAILED ATTEMPT] Moved {len(_moved)} file(s) it created to _failed_attempts/")
            return {
                "status": "error",
                "error": error,
                "exception_type": exception_type,
                "error_message": f"{exception_type}: {error}",
                "error_traceback": traceback.format_exc(),
                "output": output,
                "code": code,
                "created_files": []
            }


    def _generate_code_for_operation(self, ctx, operation_info, extra_instructions=""):
        """
        Generate Python code for an operation using LLM bound with tools.
        The LLM uses tools to inspect data (columns, dtypes, head, etc.) before
        generating final code. Does NOT execute the generated code.

        Args:
            ctx: Context dict with artifacts, data_registry, etc.
            operation_info: Step dict from the research plan (operation_type, operation,
                           input_nodes, output_node, description, expected_output).

        Returns:
            dict with status, code, code_file, operation_info.
        """
        from utils.config import load_configurations, resolve_output_dir
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        import uuid

        settings = load_configurations()
        stream_cb = ctx.get("_stream_callback")

        # ── Extract operation context from the research plan step ─────────
        rq_breakdown = ctx.get("artifacts", {}).get("rq_breakdown", {})

        # operation_type = operation_info.get("operation_type", "")
        operation_description = operation_info.get("description", "")
        operation_name = operation_info.get("operation", "")
        input_nodes = operation_info.get("input_nodes", [])
        output_node = operation_info.get("output_node", "")
        expected_output = operation_info.get("expected_output", "")

        self.log(ctx, f"[CODEGEN] Generating code for: {operation_description}")

        # ── Paths are pre-resolved by research_plan_generator._resolve_node_paths ──
        input_node_paths = operation_info.get("input_node_paths", {})
        output_node_path = operation_info.get("output_node_path", "")
        if not output_node_path and output_node:
            stem, ext = os.path.splitext(output_node)
            if not ext:
                ext = ".csv"
            safe_name = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")[:60]
            output_node_path = os.path.join(resolve_output_dir(ctx), f"{safe_name}{ext}")

        # Build input paths summary for the prompt
        input_paths_summary = ""
        for node_name, path in input_node_paths.items():
            input_paths_summary += f"  - {node_name}: {path}\n" if path else f"  - {node_name}: UNRESOLVED\n"
        print(input_paths_summary)

        # ── Build tools and bind to LLM ──────────────────────────────────
        tools = build_modeling_agent_codegen_tools(ctx)
        try:
            import hashlib as _hl
            from flask import has_request_context
            import utils.agm_helper as _h
            _k = (_h.load_OpenAI_key() or "")
            print(f"[KEYCHECK modeling] in_request_ctx={has_request_context()} tail={_k[-6:]!r} len={len(_k)} sha1={_hl.sha1(_k.encode()).hexdigest()[:8]}")
        except Exception as _e:
            print(f"[KEYCHECK modeling] failed: {_e}")
        llm = GIBDChat(
            model_name=llm_model("execution.modeling_code"), **llm_reasoning("execution.modeling_code")
            )
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        system_prompt = f"""You are an expert Geo-information scientist and Python programmer \
         You generate precise, executable Python code for individual geospatial research operations inside an automated research system.

Your role is to generate Python code for this operation:
{operation_description}

MANDATORY WORKFLOW — Execute in this exact order:
    STEP 1: INSPECT ALL INPUT DATA
        Call `get_columns` on EVERY input node listed below. This is sufficient —
        do NOT call `check_column_values` or `preview_data` unless Step 1b
        specifically requires it.

        After `get_columns` returns, you know the EXACT column names and dtypes.
        These are the ONLY column names that exist. NEVER assume, guess, or use
        column names that were not returned by `get_columns`.

        CRITICAL: If `get_columns` says a file has columns [A, B, C], then ONLY
        A, B, and C exist. Do NOT write code that references any other column.
        Do NOT add comments like "# Assuming column X exists".

        For JSON inputs: `get_columns` returns the top-level JSON schema
        (type, keys, length, and a truncated sample) instead of columns.
        Use that schema to write the correct `json.load` / key-access code.

    STEP 1b: RESOLVE ISSUES (only if needed)
        Review the `get_columns` results against what the operation needs:

        - Missing file: call `list_datasets` to find an alternative. If none,
          STOP and return a code block that raises FileNotFoundError.
        - Missing/mismatched columns: if a column the operation needs doesn't
          exist, scan the `get_columns` results for similar names. Only call
          `check_column_values` if you need to verify a join key's dtype or
          values to resolve a specific mismatch. If no match, STOP and return
          a code block that raises KeyError listing missing and available columns.
        - CRS mismatch: plan to reproject to the primary input's CRS.
        - Dtype mismatch on join keys: plan to cast both sides to str.
        - Nulls: handle with dropna/fillna as appropriate in generated code.

        If no issues, proceed directly to Step 2.

    STEP 2: LOAD AND FOLLOW THE SKILL REFERENCE
        Call `lookup_skill_doc` with a short query naming the method
        (e.g. "gwr", "ols regression", "spatial lag", "pearson correlation",
        "lisa", "correlation matrix"). Pass an optional `section` if you
        only want one part (e.g. section="paths" or "pitfalls").

        If the tool returns a "Multiple skills matched" disambiguation list,
        call it again with a more specific query.

        If no skill fits the operation, skip this step and proceed without
        a skill reference. Read the returned skill doc (when one is loaded)
        before writing code.

    STEP 3: GENERATE THE PYTHON SCRIPT
        Only after completing Steps 1-2, produce the final code block.
        Use ONLY column names confirmed by `get_columns` — no exceptions.

INPUT / OUTPUT PATHS
## Input Node Paths (where to READ data from)
{input_paths_summary}

## Output Node Path (where to SAVE the result)
- {output_node}: {output_node_path}

CODE REQUIREMENTS
    ## Data I/O
    - Read inputs from the EXACT paths listed above
    - Save the primary output to: `{output_node_path}`
    - Save secondary outputs (plots, reports) to: `{resolve_output_dir(ctx)}`
    - `os.makedirs(os.path.dirname(output_path), exist_ok=True)` before saving

    ## Manuscript artifacts (figures, tables, data-to-viz)
    Three helpers are pre-imported — call the one that matches what you saved.

    (1) `record_figure(...)` — for an image file you just saved (plt.savefig,
        folium.Map.save, plotly write_html/write_image).
            record_figure(
                output_dir=OUTPUT_DIR,
                file_path=<path you saved>,
                caption=<one-line, informative caption>,
                role='final' | 'promoted' | 'intermediate',
                objective_key=<e.g. 'objective_1'>, step_index=<int>
            )

    (2) `record_table(...)` — for a data file (.csv/.xlsx/.parquet) that
        belongs in the manuscript AS A TABLE (e.g. regression coefficients,
        model comparison, descriptive stats). Do NOT plot it — the manuscript
        pipeline will render it as a table.
        **Save manuscript-ready tables into `OUTPUT_DIR/tables/`**, not the
        task root. That directory is the opt-in signal for "this CSV is a
        table, not an intermediate dataset" — the figure catalog's orphan
        scan picks up anything there even if `record_table` is skipped.
        Intermediate/working CSVs (merged datasets, filtered subsets, join
        audits) stay at `OUTPUT_DIR/` and are kept out of the catalog.
            import os
            tables_dir = os.path.join(OUTPUT_DIR, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            table_path = os.path.join(tables_dir, "ols_coefficients.csv")
            coef_df.to_csv(table_path, index=False)
            record_table(
                output_dir=OUTPUT_DIR, file_path=table_path,
                caption="Table 1. Regression coefficients for flood exposure",
                role='promoted',
                columns=['variable','coef','std_err','p_value'],  # optional
            )

    (3) `record_data_to_viz(...)` — for a data file (.gpkg/.csv) you want the
        presentation layer to auto-render as a figure. Prefer saving the
        image yourself with matplotlib when possible; use this only when
        visualization is genuinely deferred.
            record_data_to_viz(
                output_dir=FINAL_OUTPUT_DIR, file_path=<.gpkg or .csv>,
                caption="County-level flood risk index, 2024",
                viz_hint={{"type": "choropleth", "column": "risk_idx",
                          "cmap": "OrRd", "scheme": "Quantiles"}},
                role='final',
            )

    Role guidance (applies to all three):
        * 'final'        — the answer the user asked for (always shown).
        * 'promoted'     — methods/diagnostic artifact that belongs in the paper.
        * 'intermediate' — on disk but NOT in the manuscript.

    Captions MUST describe what the artifact shows; do NOT echo the filename.
    For images: do NOT call `plt.show()`, always `plt.close()` after save.

    ## Correctness Rules
    - Use ONLY column names confirmed in Step 1
    - Reproject spatial layers to a common CRS before joins/overlays
    - Validate geometry before spatial operations:
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    ## OUTPUT FORMAT
    - Wrap code in a ```python ... ``` block. No explanation before or after.
    - Your final response MUST contain a ```python code block only.
"""
        # print(system_prompt)
        user_prompt = f"""Generate Python code for this operation:


**DESCRIPTION**: {operation_description}

**INPUT NODES**:
{json.dumps(input_nodes)}

**OUTPUT NODE**: {output_node}

**EXPECTED OUTPUT**: {expected_output}

Follow the three-step mandatory workflow from your instructions.

"""
        if extra_instructions:
            user_prompt += f"\n\n## Additional Instructions from User\n{extra_instructions}\n"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        self.log(ctx, "[CODEGEN] Agent inspecting data before code generation...")
        if stream_cb:
            stream_cb(f"Generating code for: {operation_name}\n")

        # ── Agentic tool loop ────────────────────────────────────────────
        MAX_ITERATIONS = 10
        final_response = None


        try:
            for iteration in range(MAX_ITERATIONS):
                self.log(ctx, f"[CODEGEN] Iteration {iteration + 1}/{MAX_ITERATIONS}")

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

                if response.tool_calls:
                    messages.append(response)

                    for tc in response.tool_calls:
                        tool_name = tc["name"]
                        tool_args = tc["args"]
                        self.log(ctx,
                                 f"[CODEGEN] Tool call: {tool_name}({json.dumps(tool_args, default=str)[:150]})")

                        if stream_cb:
                            args_short = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                            if len(args_short) > 120:
                                args_short = args_short[:120] + "..."
                            stream_cb(f"\n🔧 [TOOL CALL] **{tool_name}**({args_short})\n")

                        tool_fn = tool_map.get(tool_name)
                        if tool_fn:
                            try:
                                tool_result = tool_fn.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Error: {e}"
                        else:
                            tool_result = f"Unknown tool: {tool_name}"

                        if stream_cb:
                            stream_cb(f"\n```\n{str(tool_result)}\n```\n")

                        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
                else:
                    # No tool calls — LLM has generated its final response with code
                    final_response = response
                    break

        except Exception as e:
            self.log(ctx, f"[CODEGEN] Error in tool loop: {e}")
            return {"status": "error", "error": str(e)}

        if final_response is None:
            self.log(ctx, "[CODEGEN] Max iterations reached without final code.")
            return {"status": "error", "error": "Max tool iterations reached without generating code"}

        # ── Extract code from LLM response ───────────────────────────────
        content = final_response.content or ""
        # Handle list-of-content-blocks format from newer LangChain/API versions
        if isinstance(content, list):
            text_parts = [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
            content = "".join(text_parts)

        code = self._extract_code_block(content)

        # ── Retry once if no code block found ─────────────────────────────
        if not code:
            self.log(ctx, "[CODEGEN] No code block found in LLM response. Retrying with explicit instruction...")
            if stream_cb:
                stream_cb("\nNo code block found. Retrying...\n")
            retry_msg = HumanMessage(content=(
                "Your response did not contain a ```python code block. "
                "Please provide the complete Python code wrapped in a ```python ... ``` block. "
                "Do not explain — just output the code block."
            ))
            messages.append(final_response)
            messages.append(retry_msg)
            try:
                retry_response = llm_with_tools.invoke(messages)
                retry_content = retry_response.content or ""
                if isinstance(retry_content, list):
                    retry_content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block) for block in retry_content
                    )
                code = self._extract_code_block(retry_content)
            except Exception as e:
                self.log(ctx, f"[CODEGEN] Retry failed: {e}")

        if not code:
            self.log(ctx, "[CODEGEN] No code block found in LLM response after retry.")
            return {"status": "error", "error": "LLM did not return a code block", "raw_response": content}

        # ── Save generated code to file ──────────────────────────────────
        code_dir = os.path.join(resolve_output_dir(ctx), "generated_code")
        os.makedirs(code_dir, exist_ok=True)
        code_id = str(uuid.uuid4())[:8]
        # The operation name is free text from the plan ("filter invalid/empty
        # geometries"); a / \ : * ? etc. in it would be read as a path.
        safe_op = re.sub(r"[^A-Za-z0-9._-]+", "_", str(operation_name))[:30].strip("._") or "step"
        code_file = os.path.join(code_dir, f"op_{safe_op}_{code_id}.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
        self.log(ctx, f"[CODEGEN] Code saved to {code_file}")

        if stream_cb:
            stream_cb(f"\nCode generated and saved to {code_file}\n")

        # Note: output node is NOT registered here — registration happens
        # after successful execution in objective_executor.py to avoid
        # phantom entries from failed steps.

        return {
            "status": "generated",
            "code": code,
            "code_file": code_file,
            "description": operation_description,
            "operation_info": operation_info,
            "output_node_path": output_node_path,
            "input_node_paths": input_node_paths,
        }

    def handle_generate_and_execute_code_for_simple_operation(self, ctx, operation_info, extra_instructions=""):
        """Generate code and execute it immediately."""
        gen_result = self._generate_code_for_operation(ctx, operation_info, extra_instructions=extra_instructions)
        if gen_result.get("status") == "error":
            return gen_result

        code = gen_result["code"]
        code_file = gen_result["code_file"]
        operation_description = operation_info.get("description", "")


        self.log(ctx, f"Executing generated code...")
        result = self._execute_generated_code(ctx, code, operation_description)
        result["code_file"] = code_file

        # ── Inline debugger: detect errors and attempt automatic repair ─────
        exception_type = result.get("exception_type", "")
        _error = result.get("error", "")
        error_message = result.get("error_message", "")

        if result.get("status") == "error":
            self.log(ctx, f"  [DEBUG] Execution error: {_error}")

            from agents.task_manager.debugger import run_debugger

            ##Getting Context for Debugging
            step_idx = operation_info.get("step_number", 0)
            input_nodes = operation_info.get("input_nodes", [])
            output_node = operation_info.get("output_node", "")
            expected_output = operation_info.get("expected_output", "")
            # ── Paths are pre-resolved by research_plan_generator

            input_node_paths = operation_info.get("input_node_paths", {})
            output_node_path = operation_info.get("output_node_path", "")

            # Build input paths summary for the prompt
            input_paths_summary = ""
            for node_name, path in input_node_paths.items():
                input_paths_summary += f"  - {node_name}: {path}\n" if path else f"  - {node_name}: UNRESOLVED\n"

            # Build a failed-step dict matching what run_debugger expects
            failed_step = {
                "status": "error",
                "failed_code": code,
                "step_idx": step_idx,
                "operation_description": operation_description,
                "result": result,
                "error_message": error_message,
                "input_paths_summary": input_paths_summary,
                "output_node": output_node,
                "output_node_path": output_node_path,
                }

            # Forward any user debug feedback from a previous round
            if result.get("_user_debug_feedback"):
                failed_step["_user_debug_feedback"] = result["_user_debug_feedback"]

            self.log(ctx, f"  [INLINE DEBUG] Running agentic debugger (analyze + repair)...")
            stream_cb = ctx.get("_stream_callback")
            if stream_cb:
                stream_cb(f"  Error detected: {_error}\n")
                stream_cb(f"  Running debugger...\n")

            #Run the debugger
            debug_result = run_debugger(ctx, failed_step)

            repair = debug_result.get("repair_result", {})
            result["error_analysis"] = debug_result.get("error_analysis", {})
            result["debug_repair"] = repair

            if repair.get("success"):
                fixed_code = repair.get("fixed_code", "")

                # ── Re-execute fixed code to capture created_files & output ──
                # error_debug() doesn't propagate execution results into
                # repair_result, so we run the fixed code here.
                if fixed_code and not repair.get("created_files"):
                    from agents.task_manager.debugger import _execute_fixed_code
                    reexec = _execute_fixed_code(fixed_code, output_dir=ctx.get("output_dir"))
                    if reexec.get("status") == "completed":
                        repair["output"] = reexec.get("output", "")
                        repair["created_files"] = reexec.get("created_files", [])
                        repair["code_file"] = reexec.get("code_file", repair.get("code_file", ""))
                    else:
                        repair["success"] = False
                        repair["notes"] = f"Re-execution of fixed code failed: {reexec.get('error', '')}"

                # ── Interactive mode: show fixed code for review before accepting ──
                _settings = ctx.get("settings", {})
                interactive = _settings.get("interactive_code", False) if isinstance(_settings, dict) else getattr(_settings, "interactive_code", False)
                i_review_q = ctx.get("_code_review_queue")
                i_response_q = ctx.get("_code_review_response_queue")

                if interactive and i_review_q and i_response_q and fixed_code:
                    step_idx = operation_info.get("step_number", 0)
                    if stream_cb:
                        stream_cb(f"  Debugger produced fixed code. Waiting for review...\n")
                    i_review_q.put({
                        "type": "code_review",
                        "code": fixed_code,
                        "code_file": repair.get("code_file", ""),
                        "step_index": step_idx,
                        "step_description": f"[DEBUG FIX] {operation_description}",
                        "agent_name": "Debugger",
                        "step": ctx.get("_stream_step", "objective_execution"),
                        "round": 1,
                    })

                    # Wait for user decision on the fixed code
                    debug_response = None
                    while True:
                        if helper.check_cancelled(ctx):
                            break
                        try:
                            debug_response = i_response_q.get(timeout=0.5)
                            break
                        except Exception:
                            continue

                    if debug_response is None:
                        debug_response = {"action": "approve"}

                    debug_action = debug_response.get("action", "approve")

                    if debug_action == "approve":
                        user_fixed_code = debug_response.get("code", fixed_code) or fixed_code
                        if user_fixed_code != fixed_code:
                            if stream_cb:
                                stream_cb(f"  Running user-edited fix...\n")
                            from agents.task_manager.debugger import _execute_fixed_code
                            rerun = _execute_fixed_code(user_fixed_code, output_dir=ctx.get("output_dir"))
                            if rerun.get("status") == "completed":
                                repair["fixed_code"] = user_fixed_code
                                repair["output"] = rerun.get("output", "")
                                repair["created_files"] = rerun.get("created_files", [])
                                repair["code_file"] = rerun.get("code_file", repair.get("code_file", ""))
                            else:
                                repair["success"] = False
                                repair["notes"] = f"User-edited fix failed: {rerun.get('error', '')}"
                                if stream_cb:
                                    stream_cb(f"  User-edited fix also failed.\n")
                        else:
                            if stream_cb:
                                stream_cb(f"  Running debugger fix...\n")

                    elif debug_action == "skip":
                        repair["success"] = False
                        repair["notes"] = "Debug fix skipped by user"
                        if stream_cb:
                            stream_cb(f"  Debug fix skipped.\n")

                    elif debug_action == "feedback":
                        feedback = debug_response.get("feedback", "")
                        if stream_cb:
                            stream_cb(f"  Re-running debugger with feedback...\n")
                        failed_step["_user_debug_feedback"] = feedback
                        debug_result2 = run_debugger(ctx, failed_step)
                        repair = debug_result2.get("repair_result", {})
                        result["debug_repair"] = repair

                # ── Apply successful repair ──
                if repair.get("success"):
                    self.log(ctx, f"  [INLINE DEBUG] Repaired successfully.")
                    result = {
                        "status": "completed",
                        "output": repair.get("output", ""),
                        "code": repair.get("fixed_code", ""),
                        "code_file": repair.get("code_file", code_file),
                        "created_files": repair.get("created_files", []),
                        "error": None,
                        "error_analysis": debug_result.get("error_analysis", {}),
                        "debug_repair": repair,
                    }
                    if stream_cb:
                        stream_cb(f"  Debugger repaired successfully.\n")
                else:
                    self.log(ctx, f"  [INLINE DEBUG] Repair failed. {repair.get('notes', '')[:80]}")
                    if stream_cb:
                        stream_cb(f"  Debugger repair failed.\n")
            else:
                self.log(ctx, f"  [INLINE DEBUG] Repair failed. {repair.get('notes', '')[:80]}")
                if stream_cb:
                    stream_cb(f"  Debugger repair failed.\n")

            # ── Fallback: regenerate entire code from scratch if debugger failed ──
            if result.get("status") == "error" and not operation_info.get("_debug_regen_retry"):
                self.log(ctx, "[DEBUG FALLBACK] Debugger failed — regenerating code from scratch...")
                if stream_cb:
                    stream_cb("⚠ Debugger could not fix the error. Regenerating entire code from scratch...\n")

                # Build extra instructions with the error context
                error_analysis = debug_result.get("error_analysis", {}) if debug_result else {}
                root_cause = error_analysis.get("root_cause", error_message)
                regen_hint = (
                    f"\n\n## IMPORTANT — Previous Code Failed With Error\n"
                    f"The previous code for this step failed with: {error_message}\n"
                    f"Debugger diagnosis: {root_cause}\n"
                    f"Traceback:\n{result.get('error_traceback', '')[:500]}\n\n"
                    f"Generate a COMPLETELY NEW implementation that avoids this error.\n"
                    f"Do NOT reuse the same approach — try an alternative strategy.\n"
                )

                regen_info = dict(operation_info)
                regen_info["_debug_regen_retry"] = True
                regen_result = self.handle_generate_and_execute_code_for_simple_operation(
                    ctx, regen_info, extra_instructions=regen_hint
                )

                if regen_result.get("status") == "completed":
                    self.log(ctx, "[DEBUG FALLBACK] Regeneration succeeded.")
                    if stream_cb:
                        stream_cb("✔ Code regenerated and executed successfully.\n")
                    return regen_result
                else:
                    self.log(ctx, "[DEBUG FALLBACK] Regeneration also failed.")
                    if stream_cb:
                        stream_cb("⚠ Regenerated code also failed. Returning error.\n")
                    # Return the regen result so the user sees the latest error
                    return regen_result

        else:
            _output = result.get("output", "")
            if _output:
                self.log(ctx, f"  [DEBUG] Execution output: {_output[:500]}")

        # ── Empty-result detection & retry ────────────────────────────────
        # Detect two cases:
        #   1. No output AND no files at all
        #   2. Files were created but contain no actual data (0 rows / empty)
        if result.get("status") == "completed" and not operation_info.get("_empty_result_retry"):
            no_files = not result.get("created_files")
            no_output = not (result.get("output") or "").strip()
            empty_data = result.get("empty_data_files", [])
            # Check if ALL tabular data files are empty (not just some)
            tabular_exts = {".csv", ".gpkg", ".geojson", ".shp", ".json", ".npy"}
            tabular_files = [f for f in result.get("created_files", [])
                             if os.path.splitext(f)[1].lower() in tabular_exts]
            all_data_empty = bool(empty_data) and len(empty_data) >= len(tabular_files) and len(tabular_files) > 0

            should_retry = (no_files and no_output) or all_data_empty

            if should_retry:
                stream_cb = ctx.get("_stream_callback")
                prev_code = result.get("code", "")

                if all_data_empty:
                    details = "; ".join(f"{os.path.basename(f['path'])} ({f['reason']})" for f in empty_data)
                    self.log(ctx, f"[EMPTY DATA] Output files contain no data: {details}. Retrying...")
                    if stream_cb:
                        stream_cb(f"⚠ Output files are empty ({details}). Regenerating with fix...\n")
                    extra_hint = (
                        f"\n\n## IMPORTANT — Previous Attempt Produced Empty Data Files\n"
                        f"The following code ran without errors but the output files contain NO data rows:\n"
                        f"Empty files: {details}\n\n"
                        f"```python\n{prev_code}\n```\n\n"
                        f"Common causes of empty output:\n"
                        f"- Filtering/merge conditions that eliminate all rows (e.g., wrong join key, CRS mismatch in spatial join, overly strict filter)\n"
                        f"- Column name mismatch between datasets causing an empty merge result\n"
                        f"- Using inner join when left join is needed\n\n"
                        f"Fix the code so that:\n"
                        f"1. Print the shape of each intermediate DataFrame BEFORE and AFTER joins/filters (e.g., print('After merge:', df.shape))\n"
                        f"2. Verify join keys exist and have matching values in both datasets before merging\n"
                        f"3. If a spatial join is involved, ensure both layers are in the same CRS\n"
                        f"4. The final output file contains actual data rows\n"
                    )
                else:
                    self.log(ctx, "[EMPTY RESULT] Code ran successfully but produced no output and no files. Retrying...")
                    if stream_cb:
                        stream_cb("⚠ Code executed without errors but produced no output or files. Regenerating with fix...\n")
                    extra_hint = (
                        f"\n\n## IMPORTANT — Previous Attempt Produced Empty Results\n"
                        f"The following code ran without errors but generated NO output files and NO printed output:\n"
                        f"```python\n{prev_code}\n```\n"
                        f"This likely means the code loaded data but never saved any results. "
                        f"Make sure your code:\n"
                        f"1. Saves the output file to the specified output path (e.g., df.to_csv(), gdf.to_file(), plt.savefig())\n"
                        f"2. Prints a summary of the result (e.g., print(df.shape), print(df.head()))\n"
                        f"3. Does NOT skip execution due to conditional logic that silently passes\n"
                    )

                retry_info = dict(operation_info)
                retry_info["_empty_result_retry"] = True
                retry_result = self.handle_generate_and_execute_code_for_simple_operation(ctx, retry_info, extra_instructions=extra_hint)
                # Accept retry if it produced non-empty results
                retry_output = (retry_result.get("output") or "").strip()
                retry_files = retry_result.get("created_files", [])
                retry_empty_data = retry_result.get("empty_data_files", [])
                if retry_result.get("status") == "completed" and (retry_output or retry_files) and not retry_empty_data:
                    self.log(ctx, "[EMPTY RESULT] Retry succeeded — using regenerated code.")
                    if stream_cb:
                        stream_cb("✔ Retry succeeded — code now produces output.\n")
                    return retry_result
                else:
                    self.log(ctx, "[EMPTY RESULT] Retry also produced empty/no results. Returning original result.")
                    if stream_cb:
                        stream_cb("⚠ Retry also produced empty results. Returning original.\n")

        return result

    def handle_medium_step(self, ctx, operation_info):
        """Medium pipeline: delegates to handle_simple_step (same execution with model KB support)."""
        return self.handle_generate_and_execute_code_for_simple_operation(ctx, operation_info)

    def handle_complex_step(self, ctx, operation_info):
        """Complex pipeline: delegates to handle_simple_step (same execution with model KB support)."""
        return self.handle_generate_and_execute_code_for_simple_operation(ctx, operation_info)

    def handle_generate_code_for_simple_operation(self, ctx, operation_info,
                                                  extra_instructions=""):  # EQUIVALENT TO def handle_simple_step_generate_only
        """Generate code but do NOT execute. Returns code and code_file for review."""
        return self._generate_code_for_operation(ctx, operation_info, extra_instructions=extra_instructions)




    def _store_result(self, ctx, objective_key, step_number, result):
        """
        Store modeling results in context artifacts, organized by objective and step.
        Also saves to a JSON file for persistence.

        Args:
            ctx: Context dictionary
            objective_key: e.g., "objective_1"
            step_number: e.g., "1" or "2"
            result: Result dictionary from final_recommendation_and_code_generator
        """
        import os
        from utils.config import load_configurations, resolve_output_dir

        # Initialize artifacts structure if needed
        artifacts = ctx.setdefault("artifacts", {})
        modeling_results = artifacts.setdefault("modeling_results", {})
        objective_results = modeling_results.setdefault(objective_key, {})

        # Store by step
        step_key = f"step_{step_number}"
        objective_results[step_key] = result

        # Also store a flat key for easy access
        flat_key = f"{objective_key}_step_{step_number}"
        artifacts[flat_key] = result

        # Save to file for persistence
        try:
            settings = load_configurations()
            model_out_dir = resolve_output_dir(ctx)
            os.makedirs(model_out_dir, exist_ok=True)
            results_path = os.path.join(model_out_dir, "modeling_results.json")

            # Load existing results if file exists
            existing_results = {}
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)

            # Update with new result
            if objective_key not in existing_results:
                existing_results[objective_key] = {}
            existing_results[objective_key][step_key] = result

            # Save back
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)

            self.log(ctx, f"Results saved to {results_path}")

        except Exception as e:
            self.log(ctx, f"Warning: Could not save results to file: {e}")

    def final_recommendation(self, rag_recommendations: List[Dict], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the recommended models again to select the best by looking at the data overview.

        Args:
            rag_recommendations: List of model recommendations from RAG/vector search
            task: Dictionary containing:
                - step_description: Description of the modeling task
                - data_requirement: Data overview/requirements dict
                - objective: The objective of this step
                - step_number: Step number in the research plan

        Returns:
            Dictionary with:
                - recommended_model: Best models with its details
                - reasoning: Explanation for model selection
        """
        # Extract task parameters
        step_description = task.get("step_description", "")
        data_requirement = task.get("data_requirement", {})
        objective_key = task.get("objective_key", "")
        objective = task.get("objective", "")
        step_number = task.get("step_number", "")

        # Format RAG recommendations for the prompt
        rag_models_text = json.dumps(rag_recommendations, indent=2)
        data_overview_text = json.dumps(data_requirement, indent=2) if isinstance(data_requirement, dict) else str(data_requirement)

        # Create the system prompt
        system_prompt = """You are an expert data scientist and ML engineer. Your task is to:
1. Take a second look at the provided model recommendations from RAG search again and Select one most suitable model for the task description and data overview


IMPORTANT GUIDELINES:
- Consider the data types, dimensions, and characteristics when selecting the model
- Consider fittness to the task description and data overview

You MUST respond with valid JSON in this exact format:
{
    "reasoning": "Your explanation for why this model was selected...",
    "recommended_model": 
        {
            "model_name": "ModelName",
            "model_class": "full.module.path.ModelName",
            "why_selected": "Reason this model is suitable for the task...",

        }
    
}
IMPORTANT: Return ONLY ONE mopdel as a single object, NOT a list or array.
"""

        # Create the user prompt
        user_prompt = f"""## Task Description
{step_description}

## Objective
{objective}

## Data Overview
{data_overview_text}

## RAG Model Recommendations (candidates to choose from)
{rag_models_text}

Based on the above information, select the most suitable model.
Remember to respond with valid JSON only."""

        # Make the LLM call with streaming
        print("\n" + "=" * 70)
        print("FINAL MODEL RECOMMENDATION")
        print("=" * 70)
        print(f"Step {step_number}: {step_description}\n")
        print("Analyzing RAG recommendations...")
        print("-" * 70)

        user_key = helper.load_OpenAI_key()

        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = helper.client_chat_completion(
            messages=messages, 
            model=llm_model("execution.model_recommendation"), **llm_reasoning("execution.model_recommendation"),
            # reasoning_effort="low"
            )

        full_response = response.choices[0].message.content.strip()



        # Parse the JSON response
        try:
            # Extract JSON from the response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', full_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_str = full_response

            result = json.loads(json_str)
            recommended_model = result.get("recommended_model", {})
            print("\nSuccessfully parsed model recommendation!")
            print(f"[MODELING AGENT]: {recommended_model}")

            return {
                "agent": "ModelingAgent",
                "objective_key": objective_key,
                "step_number": step_number,
                "reasoning": result.get("reasoning", ""),
                "recommended_model": recommended_model
            }


        except json.JSONDecodeError as e:
            print(f"[Code Generator] JSON parse error: {e}")
            return {
                "agent": "ModelingAgent",
                "objective_key": objective_key,
                "step_number": step_number,
                "error": str(e),
                "raw_response": full_response,
                "recommended_model": "None"
            }




