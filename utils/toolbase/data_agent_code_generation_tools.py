"""
Code generation tools for the DataAgent.

Provides tools for the LLM to inspect data (columns, dtypes, sample rows,
shapes) before generating code — so the generated code references actual
columns correctly — plus skill-document lookup.

Skill lookup is delegated to the unified loader in ``skill_lookup_tool``,
scoped to ``agent="data"`` via the frontmatter ``agent`` field. Adding a
new data skill = drop a ``*_skill.md`` with ``agent: data`` in the
frontmatter under ``skills/``. No edits here required.

Usage:
    from utils.toolbase.data_agent_code_generation_tools import build_data_agent_codegen_tools
    tools = build_data_agent_codegen_tools(ctx)
"""

import json
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool

from utils.toolbase.skill_lookup_tool import (
    build_skill_lookup_tool,
    get_relevant_skills,
)
from utils.toolbase.modeling_agent_code_generation_tools import _inspect_json_schema


def get_relevant_data_skills(step_description: str) -> str:
    """Backwards-compatible wrapper — scoped data-skill keyword match.

    Preserved for ``agents/data_agent/agm_data_agent.py`` which imports
    this name. Returns empty string if no ``agent: data`` skills exist.
    """
    return get_relevant_skills(step_description, agent="data")


def build_data_agent_codegen_tools(ctx):
    """Build tools for the data agent's code generation LLM.

    Args:
        ctx: Context dict with artifacts (data_registry, etc.)

    Returns:
        List of LangChain tools.
    """

    @tool
    def list_datasets() -> str:
        """List all available datasets in the data registry with their file paths, shapes, and column counts.
        This includes both original input data AND files created by previous steps."""
        data_registry = ctx.get("artifacts", {}).get("data_registry", [])
        if not data_registry:
            return "No datasets in registry."
        lines = []
        for i, src in enumerate(data_registry):
            tag = " [step output]" if src.get("source") == "step_output" else ""
            lines.append(
                f"{i+1}. {src.get('file', '?')} | path: {src.get('path', '?')} | shape: {src.get('shape', '?')}{tag}"
            )
        return "\n".join(lines)

    @tool
    def get_columns(file_path: str) -> str:
        """Get all column names and their data types for a dataset.
        For JSON inputs, returns the top-level schema (keys, length, sample)
        instead of columns — use that to write the correct json.load code.
        Pass the file path from the data registry.
        Example: get_columns('data/ACS_tract_2019_decile_cleaned.csv')"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext == '.json':
                return _inspect_json_schema(file_path)
            if ext in ['.csv']:
                df = pd.read_csv(file_path, nrows=0)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=0)
            elif ext in ['.parquet']:
                df = pd.read_parquet(file_path).head(0)
            elif ext in ['.gpkg', '.geojson', '.shp']:
                import geopandas as gpd
                df = gpd.read_file(file_path, rows=0)
            else:
                return f"Unsupported format: {ext}"
            dtypes = df.dtypes.to_dict()
            lines = [f"Columns ({len(dtypes)}):"]
            for col, dtype in dtypes.items():
                lines.append(f"  - {col}: {dtype}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    @tool
    def preview_data(file_path: str, n_rows: int = 5) -> str:
        """Preview the first N rows of a dataset (default 5).
        For JSON: returns the top-level schema (same as get_columns).
        Example: preview_data('data/ACS_tract_2019_decile_cleaned.csv', 3)"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext == '.json':
                return _inspect_json_schema(file_path, sample_chars=800)
            if ext in ['.csv']:
                df = pd.read_csv(file_path, nrows=n_rows)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=n_rows)
            elif ext in ['.parquet']:
                df = pd.read_parquet(file_path).head(n_rows)
            elif ext in ['.gpkg', '.geojson', '.shp']:
                import geopandas as gpd
                df = gpd.read_file(file_path, rows=n_rows)
                if 'geometry' in df.columns:
                    df = df.drop(columns='geometry')
            else:
                return f"Unsupported format: {ext}"
            return f"Shape: {df.shape}\n{df.to_string(max_colwidth=60)}"
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    @tool
    def check_column_values(file_path: str, column_name: str) -> str:
        """Check unique value counts, nulls, dtype, and sample values for a specific column.
        For JSON: returns the value at the given top-level key (or at each
        element's key for list-of-records JSON), plus basic stats.
        Example: check_column_values('data/ACS_tract_2019_decile_cleaned.csv', 'MedianIncome')"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext == '.json':
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if column_name not in data:
                        return f"Key '{column_name}' not found. Top-level keys: {list(data.keys())}"
                    v = data[column_name]
                    return f"Key: {column_name}\n  type: {type(v).__name__}\n  value: {json.dumps(v, default=str)[:600]}"
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    vals = [row.get(column_name) for row in data if isinstance(row, dict) and column_name in row]
                    if not vals:
                        return f"Key '{column_name}' not found in any element. First element keys: {list(data[0].keys())}"
                    series = pd.Series(vals)
                    return (
                        f"Key: {column_name} (from list[dict])\n"
                        f"  present in: {len(vals)}/{len(data)} elements\n"
                        f"  dtype: {series.dtype}\n"
                        f"  unique: {series.nunique()}\n"
                        f"  sample: {series.dropna().head(5).tolist()}"
                    )
                return f"JSON is not a dict or list-of-records; cannot look up key '{column_name}'."
            if ext in ['.csv']:
                df = pd.read_csv(file_path, usecols=[column_name])
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, usecols=[column_name])
            elif ext in ['.parquet']:
                df = pd.read_parquet(file_path, columns=[column_name])
            elif ext in ['.gpkg', '.geojson', '.shp']:
                import geopandas as gpd
                gdf = gpd.read_file(file_path)
                if column_name not in gdf.columns:
                    return f"Column '{column_name}' not found. Available: {list(gdf.columns)}"
                df = gdf[[column_name]]
            else:
                return f"Unsupported format: {ext}"

            series = df[column_name]
            n_unique = series.nunique()
            n_null = int(series.isna().sum())
            dtype = str(series.dtype)
            sample = series.dropna().head(5).tolist()

            lines = [
                f"Column: {column_name}",
                f"  dtype: {dtype}",
                f"  total rows: {len(series)}",
                f"  unique values: {n_unique}",
                f"  nulls: {n_null} ({round(n_null / len(series) * 100, 1)}%)" if len(series) > 0 else f"  nulls: {n_null}",
                f"  sample values: {sample}",
            ]
            if n_unique <= 20:
                vc = series.value_counts().head(20).to_dict()
                lines.append(f"  value counts: {vc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    lookup_skill_doc = build_skill_lookup_tool(agent="data")

    return [list_datasets, get_columns, preview_data, check_column_values, lookup_skill_doc]
