import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
from utils.llm_models import llm_model, llm_reasoning
from dotenv import load_dotenv
load_dotenv()

# import task_solution_prompts as solution_prompts


from agents.base import Agent
from utils.config import load_configurations, resolve_output_dir
from utils.toolbase.data_agent_tools import sniff_format, read_vector, read_table, try_reproject
from utils.toolbase.data_agent_code_generation_tools import build_data_agent_codegen_tools
import geopandas as gpd
import utils.agm_helper as helper
import utils.prompts as prompts
import os
import logging

from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution import Solution as BaseSolution

def convert_to_serializable(obj):
    """
    Recursively convert numpy/pandas types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (pd.Timedelta, pd.Interval)):
        return str(obj)
    elif hasattr(obj, 'item'):  # Handles other numpy scalar types
        return obj.item()
    elif obj is None or isinstance(obj, (str, int, float)):
        return obj
    else:
        # For any other type, convert to string
        return str(obj)

from agents.data_agent.DataRetrieverAgent import (
    DataRetrieverAgent, parse_llm_object, format_data_request_for_display,
    summarize_downloaded_files,
)


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
            pass
    return empty


class DataSolution(BaseSolution):
    """Standalone class for handling the solution generation process for a DataAgent step."""
    def __init__(self, task, data_overview, model_requirement=None, task_id=None, stream_callback=None):
        super().__init__(task, data_overview, model_requirement=model_requirement, task_id=task_id, stream_callback=stream_callback)
        # Add any data agent-specific initialization here

    def get_LLM_response_for_graph(self, stream=True, execute=True, yield_chunks=True):
        # This method can be used to get LLM response for graph generation or other purposes
        # For now, it just calls the base method, but you can customize it for data-specific needs
        return super().get_LLM_response_for_graph_DataAgent(stream=stream, execute=execute, yield_chunks=yield_chunks)

    def load_graph_file(self, file="", interactive=True):
        return super().load_graph_file(file, interactive)
  

class DataAgent(Agent):
    name = "DataAgent"

    def __init__(self):
        super().__init__()
        # Do not cache the API key here. DataAgent is instantiated at module
        # import time (before any Flask request), so resolving the key in
        # __init__ would capture os.environ's stale value instead of the
        # per-request g.api_key from the UI. Resolve it lazily per-call.

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
        """Factory method to create Solution instances"""
        return DataSolution(
            task=task,
            data_overview=data_overview,
            model_requirement=model_requirement
        )
       

    def data_retriever(self, data_request):
        """This contain the existing LLMFInd Architecture. This calls the DataRetriever class"""


    def audit_datasets(self, ctx, task):
        self.log(ctx, f"Found {len(ctx.get('datasets', []))} dataset(s) in context.")
        settings = load_configurations()
        target_crs = task.get("params", {}).get("target_crs")  # e.g., "EPSG:2272"
        audits: List[Dict[str, Any]] = []
        stream_cb = ctx.get("_stream_callback")
        total_ds = len(ctx.get("datasets", []))

        for i, ds in enumerate(ctx.get("datasets", []), 1):
            path = ds.get("path")
            name = ds.get("name")
            rec: Dict[str, Any] = {"name": name, "path": path, "ok": False}
            ftype, ext = sniff_format(path)
            rec["type"] = ftype
            rec["ext"] = ext

            if stream_cb:
                stream_cb(f"[{i}/{total_ds}] Auditing: {name} ({ftype})\n")

            if ftype == "vector":
                if stream_cb:
                    stream_cb(f"  Reading vector data...\n")
                meta = read_vector(path, geom_col=ds.get("geom_col", "geometry"))
                rec.update(meta)
                if meta.get("ok"):
                    if stream_cb:
                        crs_info = meta.get("crs", "unknown")
                        n_rows = meta.get("n_rows", "?")
                        n_cols = meta.get("n_cols", "?")
                        stream_cb(f"  CRS: {crs_info} | Rows: {n_rows} | Columns: {n_cols}\n")
                    # optional reprojection
                    out = try_reproject(path, target_crs) if target_crs else None
                    if out:
                        rec["reprojected_to"] = {"crs": target_crs, "path": out}
                        if stream_cb:
                            stream_cb(f"  Reprojected to {target_crs}\n")
                    # key checks
                    keys = ds.get("key_cols") or []
                    if keys:
                        if stream_cb:
                            stream_cb(f"  Checking key columns: {keys}\n")
                        try:
                            gdf = gpd.read_file(path)
                            rec["keys"] = {}
                            for k in keys:
                                if k in gdf.columns:
                                    miss = int(gdf[k].isna().sum())
                                    dups = int(gdf[k].duplicated().sum())
                                    rec["keys"][k] = {"present": True, "missing": miss, "duplicates": dups}
                                else:
                                    rec["keys"][k] = {"present": False}
                        except Exception as e:
                            rec["key_check_error"] = str(e)
                    # Report missingness
                    miss_pct = meta.get("missing_pct")
                    if miss_pct and stream_cb:
                        stream_cb(f"  Missing data: {miss_pct}%\n")
                elif stream_cb:
                    stream_cb(f"  Warning: vector read failed\n")

            elif ftype == "table":
                if stream_cb:
                    stream_cb(f"  Reading tabular data...\n")
                meta = read_table(path, key_cols=ds.get("key_cols"))
                rec.update(meta)
                if meta.get("ok") and stream_cb:
                    n_rows = meta.get("n_rows", "?")
                    n_cols = meta.get("n_cols", "?")
                    stream_cb(f"  Rows: {n_rows} | Columns: {n_cols}\n")

            elif ftype == "zip":
                if stream_cb:
                    stream_cb(f"  Extracting from zip archive...\n")
                # Try vector from a zipped dataset (e.g., shapefile)
                posix = Path(path).as_posix()
                meta = read_vector(f"zip://{posix}", geom_col=ds.get("geom_col", "geometry"))
                rec.update(meta)
                # If vector read failed, try zipped CSV fallback
                if not rec.get("ok"):
                    try:
                        if stream_cb:
                            stream_cb(f"  Trying tabular fallback...\n")
                        meta2 = read_table(path, key_cols=ds.get("key_cols"))
                        rec.update(meta2)
                    except Exception as e:
                        rec["error"] = f"Zip read failed: {e}"

            else:
                rec["error"] = "Unknown or unsupported format."
                if stream_cb:
                    stream_cb(f"  Error: unsupported format ({ext})\n")

            if stream_cb:
                status = "OK" if rec.get("ok") else "FAILED"
                stream_cb(f"  Status: {status}\n\n")

            audits.append(rec)

        # Save report
        out_dir = resolve_output_dir(ctx)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data_audit.json")
        # Convert numpy/pandas types to native Python types for JSON serialization
        serializable_audits = convert_to_serializable(audits)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable_audits, f, indent=2, ensure_ascii=False)

        # Store serializable version in context and return value
        ctx.setdefault("artifacts", {})["data_audit"] = out_path
        ctx["artifacts"]["data_audit_count"] = len(serializable_audits)
        ctx["artifacts"]["data_audit_records"] = serializable_audits
        self.log(ctx, f"Data audit completed. Audited {len(serializable_audits)} dataset(s) and saved to {out_path}")

        return {"data_audit": out_path, "audit_items": len(serializable_audits), "audit_records": serializable_audits}


    def eda(self, ctx):
        """
        Agentic EDA & Preprocessing using LLM tool-calling.

        The LLM decides which EDA tools to call based on the research question
        and data context. It can iterate: inspect data, preprocess, explore,
        then summarize — all driven by tool calls.
        """
        # from agents.data_agent.eda_tools import _build_eda_tools
        from utils.toolbase.eda_tools import _build_eda_tools
        from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        MAX_TOOL_ITERATIONS = 20

        self.log(ctx, "Starting agentic EDA...")
        settings = load_configurations()
        stream_cb = ctx.get("_stream_callback")
        artifacts = ctx.get("artifacts", {})

        # Gather context — use audit records and real data only (no LLM-generated summary)
        research_question = ctx.get("rq", "")
        audit_records = artifacts.get("data_audit_records", [])

        # Get the actual file paths from ctx["datasets"] (reliable),
        # NOT from LLM-generated data_summary which may be wrong
        datasets = ctx.get("datasets", [])

        if not audit_records:
            self.log(ctx, "ERROR: No data audit records found for EDA")
            return {"error": "Data audit records are required"}

        if not datasets:
            self.log(ctx, "ERROR: No datasets found in context")
            return {"error": "No datasets found in context"}

        # Load all datasets into memory
        loaded_datasets = {}
        for ds in datasets:
            ds_path = ds.get("path", "")
            ds_name = ds.get("name", os.path.basename(ds_path))
            if not ds_path or not os.path.exists(ds_path):
                self.log(ctx, f"WARNING: Skipping missing file: {ds_path}")
                continue
            self.log(ctx, f"Loading dataset: {ds_name} from {ds_path}...")
            ext = os.path.splitext(ds_path)[1].lower()
            try:
                if ext in (".gpkg", ".shp", ".geojson"):
                    loaded_datasets[ds_name] = {"df": gpd.read_file(ds_path), "path": ds_path}
                elif ext == ".csv":
                    loaded_datasets[ds_name] = {"df": pd.read_csv(ds_path), "path": ds_path}
                elif ext in (".xlsx", ".xls"):
                    loaded_datasets[ds_name] = {"df": pd.read_excel(ds_path), "path": ds_path}
                elif ext == ".zip":
                    # Try vector first, then CSV fallback
                    from pathlib import Path
                    posix = Path(ds_path).as_posix()
                    try:
                        loaded_datasets[ds_name] = {"df": gpd.read_file(f"zip://{posix}"), "path": ds_path}
                    except Exception:
                        loaded_datasets[ds_name] = {"df": pd.read_csv(ds_path), "path": ds_path}
                else:
                    self.log(ctx, f"WARNING: Unsupported file type {ext} for {ds_name}, skipping")
                    continue
                shape = loaded_datasets[ds_name]["df"].shape
                self.log(ctx, f"  Loaded {ds_name}: {shape[0]} rows, {shape[1]} columns")
            except Exception as e:
                self.log(ctx, f"WARNING: Failed to load {ds_name}: {e}")

        if not loaded_datasets:
            self.log(ctx, "ERROR: Could not load any datasets")
            return {"error": "Could not load any datasets"}

        # No dataset is set as "primary" — all datasets are equal
        # state["df"] starts as None; LLM must use switch_dataset first
        dataset_info_lines = []
        for name, info in loaded_datasets.items():
            d = info["df"]
            dtypes_summary = {col: str(dtype) for col, dtype in d.dtypes.items()
                              if str(dtype) != "geometry"}
            dataset_info_lines.append(
                f"- {name}: {d.shape[0]} rows x {d.shape[1]} cols, "
                f"path: {info['path']}\n"
                f"  Columns & dtypes: {json.dumps(dtypes_summary)}"
            )
        dataset_info_str = "\n".join(dataset_info_lines)

        self.log(ctx, f"Loaded {len(loaded_datasets)} dataset(s): {list(loaded_datasets.keys())}")

        # Build tools with mutable state — df starts as None (no dataset selected yet)
        eda_out_dir = resolve_output_dir(ctx)
        os.makedirs(eda_out_dir, exist_ok=True)
        eda_state = {
            "df": None,
            "output_dir": eda_out_dir,
            "cleaned_data_path": None,
            "all_datasets": loaded_datasets,  # all loaded datasets available
            "_approval_queue": ctx.get("_approval_queue"),
            "_response_queue": ctx.get("_response_queue"),
        }
        tools = _build_eda_tools(eda_state)

        # Build LLM with tools
        user_key = helper.load_OpenAI_key()
        llm = GIBDChat(model_name=llm_model("data.eda"), **llm_reasoning("data.eda"))
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        # Build audit summary for LLM context (from real audit, not LLM)
        audit_summary_lines = []
        for rec in audit_records:
            name = rec.get("name", "?")
            ok = rec.get("ok", False)
            n_rows = rec.get("rows") or rec.get("n_rows", "?")
            n_cols = rec.get("cols") or rec.get("n_cols", "?")
            crs = rec.get("crs", "")
            miss_pct = rec.get("missing_pct", "")
            audit_summary_lines.append(
                f"- {name}: {'OK' if ok else 'ISSUES'}, {n_rows} rows x {n_cols} cols"
                + (f", CRS: {crs}" if crs else "")
                + (f", missing: {miss_pct}%" if miss_pct else "")
            )
        audit_summary_str = "\n".join(audit_summary_lines) if audit_summary_lines else "No audit data."

        # System prompt
        system_prompt = f"""You are an expert data analyst. Your goal is to produce a single, clean, analysis-ready dataset from the loaded data to support the research question.

## Research Question
{research_question}

## Datasets ({len(loaded_datasets)} loaded)
{dataset_info_str}

No dataset is pre-selected. All datasets are equally available.
You MUST call `switch_dataset('<name>')` to select a working dataset before using any other tool.

## Data Audit Results
{audit_summary_str}

## Your Goal
Produce a single, clean, analysis-ready dataset. You have the tools — decide what EDA and preprocessing steps are actually needed for this specific research question and data. Not every dataset needs the same treatment.

Be efficient — do only what is necessary.

When you are done, save the dataset with `save_cleaned_data` and then respond with a final text summary (no tool call) to end the EDA.

## Guardrails
- You MUST call `switch_dataset` before any other tool — no dataset is pre-loaded
- `select_relevant_columns` searches across ALL datasets automatically — use it to identify and tag relevant columns
- Verify that the core variables needed for the research question actually exist in the data before proceeding. If not found, use `request_user_approval` to inform the user
- ONLY use column names that exist in the datasets listed above — do NOT invent column names
- If the research question requires variables from multiple datasets, use `find_join_columns` to identify join keys (do NOT guess by name alone), then `merge_datasets`
- If the data is at a finer geographic scale than the research question requires, use `aggregate_spatial_scale` to aggregate to the target level — this is a standard analytical decision you should make autonomously
- If a column needed for the research question is not available, or you are unsure which column to use, use `request_user_approval` — do NOT silently substitute or mislabel columns
"""

        user_prompt = f"""Perform EDA on the loaded datasets to support the research question: "{research_question}"

You decide what exploratory data analysis and preprocessing is needed. Use the tools available to you."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        if stream_cb:
            stream_cb("\n\n## Agentic EDA\n\n")

        # Agentic loop
        try:
            for iteration in range(MAX_TOOL_ITERATIONS):
                self.log(ctx, f"[EDA] Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS}")

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
                        self.log(ctx, f"[EDA] Tool call: {tool_name}({json.dumps(tool_args, default=str)[:150]})")

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
                    # No tool calls — final response. Already streamed above.
                    break

        except Exception as e:
            self.log(ctx, f"[EDA] Error in agentic loop: {e}")
            if stream_cb:
                stream_cb(f"\n**EDA Error:** {e}\n")

        # Build and save report
        final_summary = response.content if response.content else "EDA completed."
        final_df = eda_state["df"]
        if final_df is None:
            # LLM never switched to a dataset — fall back to first loaded dataset
            first_ds = list(loaded_datasets.values())[0]
            final_df = first_ds["df"]
        selected_cols = eda_state.get("selected_columns", {})

        eda_report = {
            "summary": final_summary,
            "cleaned_data_path": eda_state.get("cleaned_data_path", ""),
            "dataset_shape_after": list(final_df.shape),
            "iterations_used": min(iteration + 1, MAX_TOOL_ITERATIONS),
            "selected_columns": selected_cols,
        }

        report_path = os.path.join(resolve_output_dir(ctx), "eda_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(eda_report, f, indent=2, ensure_ascii=False)

        # Build data_summary_data artifact from real data for downstream consumers
        # (RQ breakdown, step execution, modeling agent, result analysis)
        all_columns = list(final_df.columns)
        dtypes_map = {col: str(dtype) for col, dtype in final_df.dtypes.items()
                      if str(dtype) != "geometry"}
        numeric_cols = [c for c in all_columns if pd.api.types.is_numeric_dtype(final_df[c])]
        categorical_cols = [c for c in all_columns if final_df[c].dtype == "object" or final_df[c].dtype.name == "category"]
        has_geometry = hasattr(final_df, "geometry") and final_df.geometry is not None

        data_summary_data = {
            "dataset_overview": {
                "data_path": eda_state.get("cleaned_data_path", "") or list(loaded_datasets.values())[0]["path"],
                "total_datasets": len(loaded_datasets),
                "dataset_types": ["spatial" if has_geometry else "tabular"],
                "columns": all_columns,
                "geographic_coverage": "",
                "temporal_coverage": "",
                "overall_quality": "assessed via EDA",
            },
            "variable_categorization": {
                # Include all LLM-chosen categories (flexible, not hardcoded)
                **{k: v for k, v in selected_cols.items()},
                # Always include dtype-based categories for downstream consumers
                "numeric_variables": numeric_cols,
                "categorical_variables": categorical_cols,
            },
            "data_structure": {
                "total_variables": len(all_columns),
                "variable_types": {
                    "numeric": len(numeric_cols),
                    "categorical": len(categorical_cols),
                    "spatial": 1 if has_geometry else 0,
                },
                "units_of_observation": [],
            },
            "data_quality": {
                "completeness_score": "assessed via EDA",
            },
            "eda_summary": final_summary,
        }

        summary_path = os.path.join(resolve_output_dir(ctx), "data_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data_summary_data, f, indent=2, ensure_ascii=False)

        # Store in context for downstream consumers
        ctx.setdefault("artifacts", {})["eda_report"] = report_path
        ctx["artifacts"]["eda_report_data"] = eda_report
        ctx["artifacts"]["data_summary"] = summary_path
        ctx["artifacts"]["data_summary_data"] = data_summary_data
        if eda_state.get("cleaned_data_path"):
            ctx["artifacts"]["cleaned_data_path"] = eda_state["cleaned_data_path"]

        # ── Initialize data registry with ALL available data after EDA ──
        # Priority: EDA cleaned > other EDA output files > loaded datasets
        DATA_EXTENSIONS = {".csv", ".gpkg", ".shp", ".geojson", ".xlsx", ".xls"}
        all_sources = []
        seen_paths = set()



        # 3. All loaded datasets
        for ds_name, ds_info in loaded_datasets.items():
            ds_path = ds_info.get("path", "")
            norm = os.path.normpath(ds_path) if ds_path else ""
            if norm and norm not in seen_paths and os.path.exists(ds_path):
                info = helper._peek_file(ds_path)
                if info:
                    info["source"] = "loaded_dataset"
                    info["from_step"] = None
                    all_sources.append(info)
                    seen_paths.add(norm)

        ctx["artifacts"]["data_registry_all"] = all_sources

        # Set primary registry entry to EDA cleaned (or first available)
        if all_sources:
            primary = all_sources[0]
            ctx["artifacts"]["data_registry"] = {
                "path": primary["path"],
                "columns": list(primary["columns"].keys()),
                "dtypes": primary["columns"],
                "shape": primary["shape"],
                "produced_by": None,
            }
            self.log(ctx, f"[DATA REGISTRY] Initialized with {len(all_sources)} source(s): "
                     + ", ".join(f'{s["file"]} ({s["source"]})' for s in all_sources))

        self.log(ctx, f"EDA complete. Report saved to {report_path}")
        self.log(ctx, f"Data summary generated from real data: {summary_path}")
        return eda_report

    def eda_explore(self, ctx, user_task):
        """Run a single ad-hoc EDA task requested by the user after initial EDA is complete.

        Loads the cleaned data (or original), builds tools, runs a short agentic loop
        with streaming and interactive approval support (same pattern as eda()).
        Returns dict with result text + any new artifacts created.
        """
        # from agents.data_agent.eda_tools import _build_eda_tools
        from utils.toolbase.eda_tools import _build_eda_tools
        from utils.agm_helper import GIBDChat, get_ui_model, get_ui_reasoning_effort
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        settings = load_configurations()
        stream_cb = ctx.get("_stream_callback")

        # Find data to load
        data_path = ctx.get("artifacts", {}).get("cleaned_data_path", "")
        if not data_path:
            ds_overview = ctx.get("artifacts", {}).get("data_summary_data", {}).get("dataset_overview", {})
            data_path = ds_overview.get("data_path", "")
        if not data_path:
            datasets = ctx.get("datasets", [])
            if datasets:
                data_path = datasets[0].get("path", "")

        if not data_path or not os.path.exists(data_path):
            return {"error": f"Could not find data file: {data_path}"}

        # Load data
        ext = os.path.splitext(data_path)[1].lower()
        try:
            if ext in (".gpkg", ".shp", ".geojson"):
                df = gpd.read_file(data_path)
            elif ext == ".csv":
                df = pd.read_csv(data_path)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(data_path)
            elif ext == ".zip":
                from pathlib import Path
                try:
                    df = gpd.read_file(f"zip://{Path(data_path).as_posix()}")
                except Exception:
                    df = pd.read_csv(data_path)
            else:
                return {"error": f"Unsupported file type: {ext}"}
        except Exception as e:
            return {"error": f"Failed to load data: {e}"}

        # Record files before execution to detect new artifacts
        explore_out_dir = resolve_output_dir(ctx)
        os.makedirs(explore_out_dir, exist_ok=True)
        files_before = set(os.listdir(explore_out_dir))

        # Build tools (with approval queues for interactive support)
        eda_state = {
            "df": df,
            "output_dir": explore_out_dir,
            "cleaned_data_path": data_path,
            "all_datasets": {},
            "_approval_queue": ctx.get("_approval_queue"),
            "_response_queue": ctx.get("_response_queue"),
        }
        tools = _build_eda_tools(eda_state)
        tool_map = {t.name: t for t in tools}

        user_key = helper.load_OpenAI_key()
        llm = GIBDChat(model_name=llm_model("data.eda_explore"), **llm_reasoning("data.eda_explore"))
        llm_with_tools = llm.bind_tools(tools)

        dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()
                       if str(dtype) != "geometry"}

        system_prompt = f"""You are an expert data analyst. The user wants to perform a specific EDA task on their dataset.

## Dataset
- Path: {data_path}
- Shape: {df.shape[0]} rows x {df.shape[1]} cols
- Columns & dtypes: {json.dumps(dtypes_info)}

## Rules
- Perform ONLY what the user asked — be focused and efficient
- Use the available tools to accomplish the task
- When done, respond with a text summary (no tool call)
- ONLY use column names that exist in the dataset
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_task),
        ]

        if stream_cb:
            stream_cb(f"\n\n## Explore: {user_task}\n\n")

        MAX_ITERATIONS = 10
        response = None
        try:
            for iteration in range(MAX_ITERATIONS):
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
                    # Final content already streamed above via .stream()
                    break
        except Exception as e:
            if stream_cb:
                stream_cb(f"\n**Explore Error:** {e}\n")

        result_text = response.content if response and response.content else "Task completed."

        # Detect new artifacts
        files_after = set(os.listdir(explore_out_dir))
        new_files = files_after - files_before
        artifacts = []
        from utils.config import load_configurations as _load_cfg
        _global_out = os.path.abspath(_load_cfg().output_dir)
        for fname in new_files:
            fpath = os.path.join(explore_out_dir, fname)
            if not os.path.isfile(fpath):
                continue
            fext = os.path.splitext(fname)[1].lower().lstrip(".")
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                fsize = 0
            try:
                rel_path = os.path.relpath(os.path.abspath(fpath), _global_out).replace(os.sep, "/")
            except ValueError:
                rel_path = fname
            artifacts.append({
                "path": fpath,
                "relative_path": rel_path,
                "filename": fname,
                "size": fsize,
                "type": fext,
            })

        return {"result": result_text, "artifacts": artifacts}


    def data_summary(self, ctx):
        """
        Generates a comprehensive data summary from data audit records.
        Extracts structured metadata programmatically from audit records,
        then uses LLM only for analytical judgment (variable categorization,
        quality assessment, limitations, preprocessing recommendations).

        Args:
            ctx: Context containing data audit records

        Returns:
            Dictionary with structured data summary
        """
        self.log(ctx, "Generating data summary from audit records...")
        # Get data audit records
        audit_records = ctx.get("artifacts", {}).get("data_audit_records", [])
        if not audit_records:
            self.log(ctx, "ERROR: No data audit records found")
            return {"error": "Data audit records are required"}

        try:
            # --- Step 1: Build per-dataset metadata from audit records ---
            datasets_info = {}
            all_columns = []
            all_dataset_types = []
            first_data_path = ""

            for rec in audit_records:
                name = rec.get("name", "unknown")
                data_path = rec.get("path", "")
                if not first_data_path and data_path:
                    first_data_path = data_path

                # Determine format from audit record
                has_crs = rec.get("crs") is not None and rec.get("crs") != "None"
                fmt = "vector" if has_crs else "table"
                if fmt not in all_dataset_types:
                    all_dataset_types.append(fmt)

                columns = rec.get("columns", [])
                all_columns.extend(c for c in columns if c not in all_columns)

                ds_entry = {
                    "data_path": data_path,
                    "format": fmt,
                    "n_rows": rec.get("n_rows"),
                    "n_cols": rec.get("n_cols"),
                    "columns": columns,
                    "head": rec.get("head"),
                }

                if has_crs:
                    ds_entry["crs"] = rec.get("crs")
                    ds_entry["geometry"] = rec.get("geometry")

                if rec.get("keys"):
                    ds_entry["keys"] = rec["keys"]

                datasets_info[name] = ds_entry

            self.log(ctx, f"Extracted metadata for {len(datasets_info)} dataset(s). Requesting LLM analysis...")

            # --- Step 2: LLM for analytical judgment only ---
            system_prompt = prompts.data_summary_system_prompt
            user_prompt = f"""**Pre-extracted Dataset Metadata:**
{json.dumps(datasets_info, indent=2)}

**Raw Audit Records:**
{json.dumps(audit_records, indent=2)}
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            user_key = helper.load_OpenAI_key()


            try:
                response = helper.gibd_chat_completion_stream(
                    user_key=user_key, messages=messages, model=llm_model("data.summary"), 
                    response_format={"type": "json_object"}, **llm_reasoning("data.summary"),
                )
                stream_callback = ctx.get("_stream_callback")
                full_content = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                    if content:
                        print(content, end="", flush=True)
                        full_content += content
                        if stream_callback:
                            stream_callback(content)

                print() # newline after streaming

            except Exception as llm_err:
                self.log(ctx, f"ERROR in LLM response: {llm_err}")
                raise
            # else:
            #     response = helper.gibd_chat_completion(
            #         user_key, "gpt-4o", messages,
            #         response_format={"type": "json_object"},
            #     )
            #     print (user_key)
            #     full_content = response.choices[0].message.content

            llm_analysis = parse_llm_object(full_content)

            # --- Step 3: Merge tool-extracted + LLM-judged into final summary ---
            parsed_data = {
                "datasets": datasets_info,
                "dataset_overview": {
                    "data_path": first_data_path,
                    "total_datasets": len(datasets_info),
                    "dataset_types": all_dataset_types,
                    "columns": all_columns,
                    "geographic_coverage": llm_analysis.get("geographic_coverage", ""),
                    "temporal_coverage": llm_analysis.get("temporal_coverage", ""),
                    "overall_quality": llm_analysis.get("overall_quality", ""),
                },
                "variable_categorization": llm_analysis.get("variable_categorization", {}),
                "data_quality": llm_analysis.get("data_quality", {}),
                "data_limitations": llm_analysis.get("data_limitations", {}),
                "preprocessing_recommendations": llm_analysis.get("preprocessing_recommendations", {}),
            }

            self.log(ctx, "Data summary generated successfully")

            # Save summary to file
            settings = load_configurations()
            sum_out_dir = resolve_output_dir(ctx)
            os.makedirs(sum_out_dir, exist_ok=True)
            summary_path = os.path.join(sum_out_dir, "data_summary.json")

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)

            # Store in context
            ctx.setdefault("artifacts", {})["data_summary"] = summary_path
            ctx["artifacts"]["data_summary_data"] = parsed_data

            self.log(ctx, f"Data summary saved to {summary_path}")

            return parsed_data

        except Exception as e:
            self.log(ctx, f"ERROR in data summary generation: {str(e)}")
            return {"error": str(e)}


    def handle(self, ctx, task):
        print("MOVED TO THE audit_datasets METHOD")

        

    def _handle_execute_step(self, ctx, params):
        print("[DATA AGENT]: UNDER PROCESSING")




# ============= Handling steps (simple, medium, complex ==============================


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
        # Inject figure-cataloging helper so data-prep code can register
        # diagnostic figures (coverage maps, QA plots) for the manuscript.
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


    def _generate_step_code(self, ctx, task_spec, extra_instructions=""):
        """
        Generate Python code for a step using LLM. Does NOT execute.
        Returns dict with 'code', 'code_file', and 'step_ctx', or 'error'.
        """
        from utils.config import load_configurations, resolve_output_dir
        from utils.toolbase.data_agent_code_generation_tools import get_relevant_data_skills as get_relevant_skills
        import uuid

        settings = load_configurations()
        step_ctx = self._get_step_context(ctx, task_spec)

        self.log(ctx, f"[CODEGEN] Generating code for: {step_ctx['step_description']}")

        # ── Resolve data source deterministically ──
        resolved = helper.resolve_data_for_step(ctx)
        if resolved:
            resolved_path = resolved["path"]
            resolved_columns = resolved["columns"]  # {col: dtype}
            resolved_shape = resolved["shape"]
            resolved_source = resolved["source"]
            resolved_from_step = resolved.get("from_step")
            resolved_head = resolved.get("head", "")
            other_available = resolved.get("other_available", [])
            self.log(ctx, f"[CODEGEN] Resolved data: {resolved_path} (source={resolved_source}, cols={len(resolved_columns)}, shape={resolved_shape})")
        else:
            # Fallback to step_ctx data path
            resolved_path = step_ctx["data_path"]
            resolved_columns = {c: "unknown" for c in step_ctx.get("columns", [])}
            resolved_shape = "unknown"
            resolved_source = "fallback"
            resolved_from_step = None
            resolved_head = ""
            other_available = []
            self.log(ctx, f"[CODEGEN] No data resolved, falling back to: {resolved_path}")

        # Build dependency context strings
        next_steps_text = ""
        if step_ctx["next_steps"]:
            for ns in step_ctx["next_steps"]:
                next_steps_text += f"  - Step {ns.get('step_index')}: {ns.get('step_description', '')}\n"
        else:
            next_steps_text = "  None (this is the last step)\n"

        var_cats_text = json.dumps(step_ctx["variable_categorization"], indent=2) if step_ctx[
            "variable_categorization"] else "Not available"

        # Format other available data sources
        if other_available:
            lines = ["If the primary dataset above is not suitable for this step, these are also available:"]
            for src in other_available:
                cols_preview = ", ".join(src["columns"][:8])
                if len(src["columns"]) > 8:
                    cols_preview += f", ... ({len(src['columns'])} total)"
                lines.append(f"  - {src['file']} ({src['source']}, shape={src['shape']}): [{cols_preview}]")
                lines.append(f"    Path: {src['path']}")
            other_sources_text = "\n".join(lines)
        else:
            other_sources_text = "None — only the resolved dataset above is available."

        # Inject relevant skill docs for this step (only if keywords match)
        skill_reference = get_relevant_skills(step_ctx['step_description'])
        skill_block = ""
        if skill_reference:
            skill_block = f"""
## Skill Reference (follow these patterns closely)
{skill_reference}
"""

        system_prompt = f"""You are a concise Python code generator for a geospatial research pipeline.

        RULES:
        1. Generate a SINGLE, self-contained Python script that accomplishes the step described.
        2. The code must be concise - no unnecessary comments, no boilerplate, no error handling.
        3. Use ONLY columns listed in the "Resolved Dataset" section below — these are VERIFIED from the actual file. Do NOT invent column names.
        4. Load the dataset from the RESOLVED path below. Use geopandas if it is a spatial file (.gpkg, .shp, .geojson), or pandas for CSV.
        5. Save any outputs to: {resolve_output_dir(ctx)}
        6. For plots, use matplotlib with Agg backend: import matplotlib; matplotlib.use('Agg')
        7. Print a brief summary of results at the end (e.g., shape, head, key statistics).
        8. Return ONLY the Python code inside a ```python code block. No explanations before or after.
        9. The code will be executed via exec(), so do NOT use 'if __name__' guards.
        10. If a next step depends on this step's output, save results in a format the next step can use.
        11. If a Skill Reference section is provided below, follow its patterns for imports, data prep, and method usage.
        {skill_block}"""

        user_prompt = f"""## Research Context
        - Research Question: {step_ctx['research_question']}
        - Objective: {step_ctx['objective_title']}
        - Hypothesis: {step_ctx['hypothesis']}

        ## Current Step
        {step_ctx['step_description']}
        - Task type: {step_ctx['task_type']}
        - Task class: {step_ctx['task_class']}
        - Execution notes: {step_ctx['execution_notes']}

        ## Resolved Dataset (VERIFIED — use this data)
        - Path: {resolved_path}
        - Source: {resolved_source}{f' (from step {resolved_from_step})' if resolved_from_step else ''}
        - Shape: {resolved_shape}
        - Columns (verified): {json.dumps(resolved_columns, indent=2)}
        - Preview:
        {resolved_head}

        ## Variable Categorization
        {var_cats_text}

        ## Other Available Data Sources
        {other_sources_text}

        ## Next Steps (save results so they can use them)
        {next_steps_text}

        Generate the Python code for this step.
        """

        if extra_instructions:
            user_prompt += f"\n\n## Additional Instructions from User\n{extra_instructions}\n"

        try:
            user_key = helper.load_OpenAI_key()
            stream_cb = ctx.get("_stream_callback")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if stream_cb:
                stream_resp = helper.client_chat_completion_stream(
                    messages = messages,
                    model = llm_model("data.step_code"), **llm_reasoning("data.step_code")
                )
                content = ""
                for chunk in stream_resp:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                    if delta:
                        content += delta
                        stream_cb(delta)
                content = content.strip()
            else:
                response = helper.client_chat_completion(
                    messages = messages,
                    model = llm_model("data.step_code"), **llm_reasoning("data.step_code")
                )
                content = response.choices[0].message.content.strip()

            # Extract code from markdown code block
            code = self._extract_code_block(content) or content

            self.log(ctx, f"Code generated ({len(code.splitlines())} lines).")

            # Save the generated code
            code_dir = os.path.join(resolve_output_dir(ctx), "generated_code")
            os.makedirs(code_dir, exist_ok=True)
            code_id = str(uuid.uuid4())[:8]
            code_file = os.path.join(code_dir, f"simple_{step_ctx['objective_key']}_{code_id}.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            self.log(ctx, f"Code saved to {code_file}")

            return {"status": "generated", "code": code, "code_file": code_file, "step_ctx": step_ctx}

        except Exception as e:
            self.log(ctx, f"[CODEGEN] Failed: {e}")
            return {"status": "error", "error": str(e)}

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
        data_registry = ctx.get("artifacts", {}).get("data_registry", [])
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

        # ── Build tools and bind to LLM ──────────────────────────────────
        tools = build_data_agent_codegen_tools(ctx)
        llm = GIBDChat(
            model_name=llm_model("data.operation_code"), **llm_reasoning("data.operation_code")
            )
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        system_prompt = f"""You are an expert Geo-information scientist and Python programmer \
specializing in geospatial data pipelines. You generate precise, executable Python code \
for individual geospatial research operations inside an automated research system.

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
        Call `lookup_skill_doc` with a query describing the data operation
        (e.g., "merge datasets on FIPS", "clean nulls", "aggregate points").
        Pass an optional `section` (e.g., "paths", "pitfalls") if you only
        need one part of the doc. If the tool returns a "Multiple skills
        matched" disambiguation list, call it again with a more specific
        query. Read the returned skill doc before writing code.

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
    Data-prep rarely produces 'final' artifacts; default to 'promoted' for
    methods-section material, 'intermediate' for working files.

    (1) `record_figure(...)` — for an image you saved (coverage map, missingness
        plot, variable distribution, QA check).
            record_figure(
                output_dir=OUTPUT_DIR, file_path=<path>,
                caption=<one-line caption>,
                role='promoted' | 'intermediate',
            )

    (2) `record_table(...)` — for a data file (.csv/.xlsx) that belongs in the
        manuscript AS A TABLE (e.g. descriptive statistics, data provenance).
        **Save manuscript-ready tables into `OUTPUT_DIR/tables/`**, not the
        task root. That directory is the opt-in signal for "this CSV is a
        table, not an intermediate dataset" — the figure catalog's orphan
        scan picks up anything there even if `record_table` is skipped.
        Intermediate CSVs (merged data, filtered subsets, join audits) stay
        at `OUTPUT_DIR/` and are kept out of the catalog.
            import os
            tables_dir = os.path.join(OUTPUT_DIR, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            table_path = os.path.join(tables_dir, "descriptives_by_county.csv")
            desc_df.to_csv(table_path, index=False)
            record_table(
                output_dir=OUTPUT_DIR, file_path=table_path,
                caption="Table S1. Sample descriptives by county",
                role='promoted',
                columns=[...],  # optional subset
            )

    (3) `record_data_to_viz(...)` — for a data file (.gpkg/.csv) to be
        auto-rendered as a figure by the presentation layer. Use this only
        when visualization is deferred; prefer saving the image yourself.
            record_data_to_viz(
                output_dir=OUTPUT_DIR, file_path=<.gpkg>,
                caption="Spatial coverage of the sample",
                viz_hint={{"type": "choropleth", "column": "sample_n"}},
                role='promoted',
            )

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
{input_paths_summary}

**OUTPUT NODE**: 
{output_node}: {output_node_path}

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
                        self.log(ctx, f"[CODEGEN] Tool call: {tool_name}({json.dumps(tool_args, default=str)[:150]})")

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

                        # Print tool results for notebook visibility
                        if tool_name == "lookup_skill_doc":
                            skill_query = tool_args.get("query", "")
                            skill_preview = str(tool_result)[:500]
                            print(f"\n--- Reading skill doc: '{skill_query}' ---")
                            print(skill_preview)
                            if len(str(tool_result)) > 500:
                                print(f"... ({len(str(tool_result))} chars total)")
                            print(f"--- End skill doc ---\n")
                            if stream_cb:
                                stream_cb(f"\n--- Reading skill doc: '{skill_query}' ---\n")
                                stream_cb(f"\n{str(tool_result)}\n")
                                stream_cb(f"\n--- End skill doc ---\n")
                        else:
                            print(f"[{tool_name}] {str(tool_result)[:800]}")
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
            "description":operation_description,
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

            # Run the debugger
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
        """Medium pipeline: delegates to handle_simple_step."""
        return self.handle_generate_and_execute_code_for_simple_operation(ctx, operation_info)

    def handle_complex_step(self, ctx, operation_info):
        """Complex pipeline: delegates to handle_simple_step."""
        return self.handle_generate_and_execute_code_for_simple_operation(ctx, operation_info)


    def handle_generate_code_for_simple_operation(self, ctx, operation_info, extra_instructions=""): # EQUIVALENT TO def handle_simple_step_generate_only
        """Generate code but do NOT execute. Returns code and code_file for review."""
        return self._generate_code_for_operation(ctx, operation_info, extra_instructions=extra_instructions)







    def _store_result(self, ctx, objective_key, step_number, result):
        """
        Store DataAgent results in context artifacts and to file.
        """
        from utils.config import load_configurations, resolve_output_dir

        # Initialize artifacts structure
        artifacts = ctx.setdefault("artifacts", {})
        modeling_results = artifacts.setdefault("modeling_results", {})
        objective_results = modeling_results.setdefault(objective_key, {})

        # Store by step
        step_key = f"step_{step_number}"
        objective_results[step_key] = result

        # Also store flat key
        flat_key = f"{objective_key}_step_{step_number}"
        artifacts[flat_key] = result

        # Save to file
        try:
            settings = load_configurations()
            res_out_dir = resolve_output_dir(ctx)
            os.makedirs(res_out_dir, exist_ok=True)
            results_path = os.path.join(res_out_dir, "modeling_results.json")

            # Load existing results
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

    def data_download(self, ctx, max_attempts=3):
        data_downloaded = []
        stream_callback = ctx.get("_stream_callback")
        interactive_callback = ctx.get("_source_select_callback")
        # Separate stream for need-driven skill synthesis: tagged with its own
        # step id so the synthesis progress gets its OWN workflow card.
        synthesis_stream_callback = ctx.get("_synthesis_stream_callback")

        # Expose the caller's private contributed sources (Handbooks_user/<uid>).
        # Derive the dir straight from the API key so it works in every workflow
        # path — autonomous or manual — not just when proceed_rq_breakdown set it
        # on ctx. Merge with any per-task value already on ctx.
        extra_handbook_dirs = list(ctx.get("_user_handbook_dirs") or [])
        try:
            from agents.data_agent.user_data_sources import user_handbook_dir_for_key
            _udir = user_handbook_dir_for_key(self.api_key)
            if _udir and _udir not in extra_handbook_dirs:
                extra_handbook_dirs.append(_udir)
        except Exception as _e:
            logging.warning(f"Could not resolve user handbook dir: {_e}")

        downloader = DataRetrieverAgent(
            api_key=self.api_key,
            extra_handbook_dirs=extra_handbook_dirs,
        )
        try:
            logging.info(
                f"[DataDownload] handbook dirs: global + extra={downloader.extra_handbook_dirs}"
            )
        except Exception:
            pass
        rq_breakdown = ctx["artifacts"]["rq_breakdown"]
        total = len(rq_breakdown["data_requests"])

        for i, data_request in enumerate(rq_breakdown["data_requests"], 1):
            # Tag current request index so the workflow can route streams to per-request sub-cards
            ctx["_current_download_req"] = i
            # Re-read keys each request: the source-select callback persists keys
            # entered for an earlier request back into ctx, so later requests for
            # the same key-gated source reuse them instead of re-prompting.
            user_keys = ctx.get("_data_source_keys") or {}
            result = None
            # Carries the previous attempt's failure reason forward into the
            # next attempt's prompt (see below) so a fresh outer retry isn't
            # completely blind to what already failed — without this, each
            # retry re-selects a source and regenerates code from scratch
            # with no memory of the prior failure, and can repeat it.
            last_failure_note = None

            data_request_display = format_data_request_for_display(data_request)
            for attempt in range(1, max_attempts + 1):
                if stream_callback:
                    if attempt == 1:
                        stream_callback(f"\n\n## Data Request {i}/{total}: {data_request_display}\n")
                    else:
                        stream_callback(
                            f"\n\n## Retry {attempt - 1}/{max_attempts - 1} "
                            f"for Data Request {i}/{total}: {data_request_display}\n"
                        )

                # On a retry, append the previous failure as extra context —
                # this flows into both the source-selection and code-gen
                # prompts (both just interpolate this text), nudging the LLM
                # away from repeating a source/approach that already failed.
                request_for_llm = data_request
                if attempt > 1 and last_failure_note:
                    request_for_llm = (
                        f"{data_request}\n\n"
                        f"[Note: a previous attempt at this exact request failed "
                        f"with: {last_failure_note} If that points to a specific "
                        f"cause (e.g. wrong source, wrong endpoint), avoid "
                        f"repeating it.]"
                    )

                result = downloader.LLM_Find(
                    data_request=request_for_llm,
                    stream_callback=stream_callback,
                    interactive_callback=interactive_callback,
                    user_keys=user_keys,
                    synthesis_stream_callback=synthesis_stream_callback,
                )

                # Check success: result exists, status is success, and files were actually downloaded
                if (
                    result
                    and result.get("status") == "success"
                    and result.get("downloaded_files")
                ):
                    break  # success — move to next data request

                # Determine failure reason for logging
                if result is None:
                    reason = "data source not recognized"
                elif result.get("status") == "failed":
                    reason = result.get("error", "no files downloaded")
                else:
                    reason = "download returned no files"

                # Keep the last line only (traceback bodies can be long) and
                # cap length — this rides along in the next attempt's prompt,
                # so it should stay a short pointer, not a full dump.
                _reason_text = (reason or "").strip()
                last_failure_note = (
                    _reason_text.splitlines()[-1][:300] if _reason_text else None
                )

                # User-initiated abort should stop retries for this request
                if (
                    isinstance(result, dict)
                    and result.get("status") == "failed"
                    and "user aborted" in (result.get("error") or "").lower()
                ):
                    logging.info(
                        f"Data request {i}/{total} aborted by user. Skipping retries."
                    )
                    break

                # Invalid/unauthorized credentials can't be fixed by retrying —
                # stop now so the clear key error stands instead of looping.
                if isinstance(result, dict) and result.get("credential_error"):
                    logging.info(
                        f"Data request {i}/{total} failed: invalid credentials. Skipping retries."
                    )
                    if stream_callback:
                        stream_callback(
                            "\n⛔ **Not retrying — fix the API key and run this request again.**\n"
                        )
                    break

                # The source has no data for what was asked (period/area/
                # variable outside its coverage). A fresh attempt would
                # re-select a source and regenerate code for the SAME
                # request, then rediscover the same gap — or worse, be
                # nudged by the failure note into a source/parameters the
                # user never asked for. Stop and let the review gate below
                # put the reason in front of the user.
                if isinstance(result, dict) and result.get("data_unavailable"):
                    logging.info(
                        f"Data request {i}/{total}: source has no data for "
                        "this request. Skipping retries."
                    )
                    if stream_callback:
                        stream_callback(
                            "\n⛔ **Not retrying — the source has no data for this "
                            "request as asked. Adjust the date range, area, or "
                            "variable, or choose another source, then run it again.**\n"
                        )
                    break

                if attempt < max_attempts:
                    logging.warning(
                        f"Data request {i}/{total} attempt {attempt} failed ({reason}). Retrying..."
                    )
                    if stream_callback:
                        stream_callback(
                            f"\n⚠️ **Attempt {attempt} failed ({reason}). Retrying...**\n"
                        )
                else:
                    logging.error(
                        f"Data request {i}/{total} failed after {max_attempts} attempts ({reason})."
                    )
                    if stream_callback:
                        stream_callback(
                            f"\n❌ **All {max_attempts} attempts failed for this data request ({reason}).**\n"
                        )

            # ── Human review checkpoint ────────────────────────────────
            # Interactive runs pause here so a person can see WHAT actually
            # came back before the workflow consumes it. A download that
            # "succeeds" can still return nothing usable — a header-only CSV
            # exits 0, and the analysis then computes a confident answer from
            # an empty table. Automatic recovery (the debug loop, and the
            # one-shot re-research on an all-empty result) has already run by
            # this point, so this is the last gate before the data is used.
            #
            # Skipped entirely when no callback is installed (autonomous run)
            # or after the user chose "Always Proceed".
            # The tiles showing what actually landed on disk. Emitted BEFORE
            # the review gate below: the gate asks "is this the data you
            # wanted?", and it used to fire while the only thing on screen was
            # a text summary -- the tiles arrived after the answer, which is
            # exactly backwards. Re-emitted after any re-run so the tiles
            # always describe the result being reviewed.
            req_done_cb = ctx.get("_request_done_callback")

            def _emit_request_files(current):
                if not req_done_cb:
                    return
                try:
                    req_done_cb(i, data_request, current)
                except Exception as cb_err:
                    logging.warning(f"request_done_callback failed: {cb_err}")

            _emit_request_files(result)

            review_cb = ctx.get("_download_review_callback")
            while review_cb and not ctx.get("_skip_download_review"):
                files_now = (result or {}).get("downloaded_files") or []
                try:
                    answer = review_cb({
                        "request_index": i,
                        "total_requests": total,
                        "data_request": format_data_request_for_display(data_request),
                        "status": (result or {}).get("status", "failed"),
                        "error": (result or {}).get("error", ""),
                        # True when the failure is the source having no data
                        # for the request (not a bug): the card then says
                        # "change the request", not "the download failed".
                        "data_unavailable": bool(
                            (result or {}).get("data_unavailable")),
                        "code": (result or {}).get("code", ""),
                        "source_name": (result or {}).get("source_name", ""),
                        "source_id": (result or {}).get("source_id", ""),
                        "summary": summarize_downloaded_files(files_now),
                    })
                except Exception as cb_err:
                    logging.warning(f"download_review_callback failed: {cb_err}")
                    break
                if not isinstance(answer, dict):
                    break  # no answer possible → accept what we have

                action = (answer.get("action") or "proceed").strip().lower()
                if action == "always_proceed":
                    ctx["_skip_download_review"] = True
                    break
                if action == "finish":
                    # Accept what this request produced and stop the stage --
                    # the remaining requests are never attempted. Checked by
                    # the enclosing loop after this request is appended.
                    ctx["_finish_downloads"] = True
                    logging.info(
                        f"Download stage finished by user at request {i}/{total}; "
                        f"{max(0, total - i)} request(s) left unrun.")
                    if stream_callback and i < total:
                        stream_callback(
                            f"\n\n⏭ **Skipped the remaining {total - i} "
                            f"data request(s) at your request.**\n")
                    break
                if action in ("", "proceed"):
                    break
                if action == "skip":
                    logging.info(f"Data request {i}/{total} skipped by user.")
                    if stream_callback:
                        stream_callback(
                            "\n\n⏭️ **Dataset skipped — it will not be used "
                            "in the analysis.**\n")
                    result = {"status": "skipped",
                              "error": "Skipped by the user at review.",
                              "downloaded_files": []}
                    _emit_request_files(result)
                    break
                if action not in ("rerun", "edit_code", "feedback"):
                    logging.warning(f"Unknown review action '{action}'; proceeding.")
                    break

                # Re-run variants. The source is pinned to whatever this
                # request already resolved to, so a correction can never
                # silently swap the data source underneath the run.
                src_id = (answer.get("source_id")
                          or (result or {}).get("source_id") or "")
                if not src_id:
                    logging.warning(
                        "Cannot re-run this request: no source id was recorded.")
                    if stream_callback:
                        stream_callback(
                            "\n\n⚠️ **Cannot re-run — the source for this "
                            "request was not recorded. Proceeding.**\n")
                    break
                try:
                    rerun_result = downloader.rerun_download(
                        data_request=data_request,
                        source_ID=src_id,
                        source_name=(answer.get("source_name")
                                     or (result or {}).get("source_name") or ""),
                        user_keys=ctx.get("_data_source_keys") or user_keys,
                        edited_code=(answer.get("code")
                                     if action == "edit_code" else None),
                        feedback=(answer.get("feedback")
                                  if action == "feedback" else None),
                        stream_callback=stream_callback,
                    )
                except Exception as rerun_err:
                    logging.warning(f"Download re-run failed: {rerun_err}")
                    if stream_callback:
                        stream_callback(f"\n\n⚠️ **Re-run failed:** {rerun_err}\n")
                    rerun_result = None
                if rerun_result:
                    # Keep the source fields so the next review round (and any
                    # further re-run) still knows what this request used.
                    rerun_result.setdefault(
                        "source_name", (result or {}).get("source_name", ""))
                    rerun_result.setdefault(
                        "source_id", (result or {}).get("source_id", ""))
                    result = rerun_result
                # Loop back and show the review again for the new outcome --
                # with tiles that describe THAT outcome, not the previous one.
                _emit_request_files(result)

            data_downloaded.append(result)

            # Everything a re-run needs AFTER the review gate has closed. The
            # gate is the only moment the workflow thread is parked waiting for
            # an answer, so once it passes there is no live context to correct
            # the request through -- unless it was written down here.
            rerun_cb = ctx.get("_register_rerun_context")
            if rerun_cb:
                try:
                    rerun_cb(i, {
                        "request_index": i,
                        "total_requests": total,
                        "data_request": data_request,
                        "source_id": (result or {}).get("source_id", ""),
                        "source_name": (result or {}).get("source_name", ""),
                        "code": (result or {}).get("code", ""),
                        "status": (result or {}).get("status", ""),
                    })
                except Exception as cb_err:
                    logging.warning(f"register_rerun_context failed: {cb_err}")

            if ctx.get("_finish_downloads"):
                break

        return data_downloaded





