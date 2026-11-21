# ── Tools for Agentic Research Plan Review ────────────────────────────
import json
from langchain_core.tools import tool


def _build_research_plan_review_tools(ctx):
    """Build tools that let the reviewer LLM inspect the research plan and data registry."""

    plan = ctx.get("artifacts", {}).get("research_plan_first_draft", {})
    registry = ctx.get("artifacts", {}).get("data_registry", [])
    objectives = plan.get("objectives", [])

    @tool
    def get_plan_overview() -> str:
        """Get a summary of the research plan: number of objectives, steps per objective, and hypotheses."""
        if not objectives:
            return "No objectives found in the research plan."
        overview = []
        for obj in objectives:
            steps = obj.get("steps", [])
            overview.append({
                "objective": obj.get("number"),
                "title": obj.get("title"),
                "hypothesis": obj.get("hypothesis"),
                "num_steps": len(steps),
                "num_data_requirements": len(obj.get("data_requirements", [])),
            })
        return json.dumps(overview, indent=2, default=str)

    @tool
    def get_objective_steps(objective_number: int) -> str:
        """Get the full steps for a specific objective by its number."""
        for obj in objectives:
            if obj.get("number") == objective_number:
                return json.dumps(obj.get("steps", []), indent=2, default=str)
        available = [obj.get("number") for obj in objectives]
        return f"Objective {objective_number} not found. Available: {available}"

    @tool
    def get_data_requirements(objective_number: int) -> str:
        """Get the data requirements for a specific objective, including suitability scores and preprocessing needs."""
        for obj in objectives:
            if obj.get("number") == objective_number:
                return json.dumps(obj.get("data_requirements", []), indent=2, default=str)
        available = [obj.get("number") for obj in objectives]
        return f"Objective {objective_number} not found. Available: {available}"

    @tool
    def verify_step_connectivity(objective_number: int) -> str:
        """Check that steps in an objective form a connected pipeline — each step's input_nodes
        reference either a source_id or a prior step's output_node."""
        for obj in objectives:
            if obj.get("number") == objective_number:
                steps = obj.get("steps", [])
                available_nodes = set()
                # Add all source_ids from data requirements
                for dr in obj.get("data_requirements", []):
                    sid = dr.get("source_id", "")
                    if sid:
                        available_nodes.add(sid)
                issues = []
                for step in steps:
                    inputs = step.get("input_nodes", [])
                    for inp in inputs:
                        if inp not in available_nodes:
                            issues.append(f"Step {step.get('step_number')}: input_node '{inp}' not produced by any prior step or data source")
                    out = step.get("output_node", "")
                    if out:
                        available_nodes.add(out)
                if issues:
                    return json.dumps({"connected": False, "issues": issues}, indent=2)
                return json.dumps({"connected": True, "issues": []}, indent=2)
        return f"Objective {objective_number} not found."

    @tool
    def verify_data_preprocessing_resolved(objective_number: int) -> str:
        """Check that objectives with data_preprocessing.required=true have a corresponding
        data preparation step in their steps list."""
        for obj in objectives:
            if obj.get("number") == objective_number:
                data_reqs = obj.get("data_requirements", [])
                steps = obj.get("steps", [])
                step_ops = [s.get("operation_type", "").lower() for s in steps]
                issues = []
                for dr in data_reqs:
                    preprocessing = dr.get("data_preprocessing") or dr.get("data preprocessing") or {}
                    if preprocessing.get("required"):
                        has_prep_step = any("data" in op or "wrangling" in op or "preprocess" in op or "preparation" in op for op in step_ops)
                        if not has_prep_step:
                            issues.append(f"source '{dr.get('source_id')}' needs preprocessing but no data preparation step found")
                if issues:
                    return json.dumps({"resolved": False, "issues": issues}, indent=2)
                return json.dumps({"resolved": True, "issues": []}, indent=2)
        return f"Objective {objective_number} not found."

    @tool
    def check_field_in_registry(field_name: str) -> str:
        """Check if a field/column name referenced in the plan actually exists in the data registry."""
        found = []
        all_cols = set()
        for src in registry:
            cols = src.get("columns", {})
            all_cols.update(cols.keys())
            if field_name in cols:
                found.append({"dataset": src.get("file"), "dtype": cols[field_name]})
        if found:
            return json.dumps({"field": field_name, "found_in": found}, indent=2)
        similar = [c for c in all_cols if field_name.lower() in c.lower() or c.lower() in field_name.lower()]
        return json.dumps({"field": field_name, "found_in": [], "similar_fields": similar[:10]}, indent=2)

    @tool
    def inspect_column_values(dataset_name: str, column_name: str) -> str:
        """Get sample values and dtype for a specific column in a dataset.
        Useful for verifying join/merge key compatibility between datasets."""
        for src in registry:
            if src.get("file") == dataset_name:
                cols = src.get("columns", {})
                if column_name not in cols:
                    return f"Column '{column_name}' not found in '{dataset_name}'. Available: {list(cols.keys())}"
                head = src.get("head", {})
                values = head.get(column_name, [])
                return json.dumps({
                    "dataset": dataset_name,
                    "column": column_name,
                    "dtype": cols[column_name],
                    "sample_values": values
                }, indent=2, default=str)
        available = [s.get("file") for s in registry]
        return f"Dataset '{dataset_name}' not found. Available: {available}"

    from utils.toolbase.dataset_profile import profile_tool
    return [get_plan_overview, get_objective_steps, get_data_requirements,
            verify_step_connectivity, verify_data_preprocessing_resolved,
            check_field_in_registry, inspect_column_values, profile_tool(registry)]