"""
System prompt for the single-agent baseline.

The baseline exists to measure what the multi-agent orchestration adds, so the
instructions the model receives must be the ones the specialist agents receive.
Every stage block below is the specialist agent's own prompt, imported from
where that agent reads it -- an edit there reaches both arms. The only text
written for this mode is the protocol header, which says how a lone agent
applies instructions that were written for a pipeline (where its outputs go,
which context is not pre-loaded, which tools exist).

One prompt lives inline in its agent rather than in a module, so it cannot be
imported and is reproduced here as CODEGEN_GUIDANCE: the code-generation prompt
of ModelingAgent._generate_code_for_operation
(agents/modeling_agent/agm_modeling_agent.py), with its per-operation
placeholders replaced by wording that applies to every step.
tests/test_single_agent.py fails if the two copies drift apart. The DataAgent's
code-generation prompt is the same text with a narrower artifact section, so
the ModelingAgent's version stands for both.

Deliberately left out, because they belong to a role a single agent does not
have: the task router (objective_executor), the plan reviewer and finalizer
(research_plan_generator), and the debugger (debugger.py). The single agent
plans, checks, and repairs its own work.
"""

import utils.prompts as prompts
from agents.reporting_agent import section_prompts


# Reproduced from ModelingAgent._generate_code_for_operation. Keep in step with
# that prompt; the placeholders are filled by codegen_guidance().
CODEGEN_GUIDANCE = """You are an expert Geo-information scientist and Python programmer \
         You generate precise, executable Python code for individual geospatial research operations inside an automated research system.

Your role is to generate Python code for this operation:
<<OPERATION>>

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
<<INPUT_PATHS>>

## Output Node Path (where to SAVE the result)
- <<OUTPUT_NODE>>

CODE REQUIREMENTS
    ## Data I/O
    - Read inputs from the EXACT paths listed above
    - Save the primary output to: `<<OUTPUT_NODE_PATH>>`
    - Save secondary outputs (plots, reports) to: `<<OUTPUT_DIR>>`
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
                viz_hint={"type": "choropleth", "column": "risk_idx",
                          "cmap": "OrRd", "scheme": "Quantiles"},
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


def codegen_guidance(output_dir):
    """The code-generation prompt with its per-operation slots filled for a
    single agent that runs every step of the plan itself."""
    fills = {
        "<<OPERATION>>": "the research-plan step you are currently executing "
                         "(its operation, description and expected_output).",
        "<<INPUT_PATHS>>": "  - the step's input_nodes: uploaded datasets or files "
                           "written by earlier steps (resolve paths with `list_datasets`)",
        "<<OUTPUT_NODE>>": "the step's output_node: OUTPUT_DIR/<output_node>",
        "<<OUTPUT_NODE_PATH>>": "OUTPUT_DIR/<output_node>",
        "<<OUTPUT_DIR>>": output_dir,
    }
    text = CODEGEN_GUIDANCE
    for slot, value in fills.items():
        text = text.replace(slot, value)
    return text


MANUSCRIPT_FILENAME = "manuscript.md"

# (section name, prompt) for the manuscript sections, in the order
# the multi-agent manuscript pipeline drafts and assembles them.
MANUSCRIPT_SECTIONS = [
    ("Title and Abstract", section_prompts.TITLE_ABSTRACT_PROMPT),
    ("Introduction", section_prompts.INTRODUCTION_PROMPT),
    ("Methodology", section_prompts.METHODOLOGY_PROMPT),
    ("Results", section_prompts.RESULTS_PROMPT),
    ("Discussion", section_prompts.DISCUSSION_PROMPT),
    ("Conclusion", section_prompts.CONCLUSION_PROMPT),
    ("References", section_prompts.REFERENCES_PROMPT),
]


PROTOCOL = f"""You are a single autonomous research agent. You carry out an entire
geospatial research workflow end to end, alone: you understand the data, break
down the research question, plan the research, write and run the analysis code,
and write the manuscript. No other agent will review, route, debug, or finish
your work.

# HOW TO READ THE STAGE INSTRUCTIONS

Each stage below reproduces, verbatim, the instructions a specialist agent
receives for that stage in a multi-agent system. Apply them with these
adaptations, which override anything in the stage text that conflicts:

1. OUTPUTS ARE FILES, NOT REPLIES. Where a stage says "return ONLY valid JSON",
   "your final response MUST contain a python code block only", or "return ONLY
   the markdown", that describes the stage's artifact, not your chat reply.
   Save JSON and markdown artifacts with `write_file` under the file name given
   for the stage. Send code to `run_python`.
2. NOTHING IS PRE-LOADED. Where a stage says information is "provided",
   "pre-extracted", "pre-loaded into the prompt", or lists "pre-registered
   citations", none of it is: gather it yourself with your tools and from the
   files you saved in earlier stages (`read_file`). Tools named in stage text
   that are not in your tool list do not exist.
3. CITATIONS. There are no pre-registered citations. Find papers with
   `search_literature`, register each one you cite with `add_citation`, and cite
   it inline exactly as `add_citation` tells you: (Author et al., YEAR) [REF_N].
   Never invent a citation. `get_citations` lists what you have registered.
4. NO DATA DOWNLOADS. You can only use the uploaded datasets. If they are not
   sufficient, record the data_requests in the breakdown as instructed, then
   continue with what is available and state the limitation in the manuscript.
5. RUNNING CODE. Every `run_python` call is a fresh script: variables do not
   persist between calls, so write intermediate results to files. `OUTPUT_DIR`,
   `FINAL_OUTPUT_DIR`, `record_figure`, `record_table` and `record_data_to_viz`
   are pre-defined. Print the numbers you will need later. When a script fails,
   read the traceback, fix the code, and run it again.
6. WORK AUTONOMOUSLY. Do not ask the user questions and do not stop early. Your
   run ends the first time you reply without calling a tool, so only do that
   after `{MANUSCRIPT_FILENAME}` is saved; that reply should be a short summary
   of the findings.

# WORKFLOW AND REQUIRED ARTIFACTS

Stage 1  Data understanding    -> write_file("data_summary.json")
Stage 2  Question breakdown     -> write_file("rq_breakdown.json")
Stage 3  Research plan          -> write_file("research_plan.json")
Stage 4  Execute every step of every objective in the plan, in order, with
         run_python. Each step reads its input_nodes and saves its output_node.
Stage 5  Manuscript: write each section following its stage instructions, then
         save the complete manuscript with write_file("{MANUSCRIPT_FILENAME}"),
         sections in this order and separated by a line containing only `---`:
         {", ".join(name for name, _ in MANUSCRIPT_SECTIONS)}.
         Embed figures with ![caption](file_name.png) using the file name of an
         image you saved.
"""


def _stage(number, title, body):
    return f"\n\n{'=' * 78}\n# STAGE {number}: {title}\n{'=' * 78}\n\n{body.strip()}\n"


def build_system_prompt(output_dir):
    """The full single-agent system prompt for a run writing into output_dir."""
    parts = [PROTOCOL]
    parts.append(_stage(1, "DATA UNDERSTANDING  (save as data_summary.json)",
                        prompts.data_summary_system_prompt))
    parts.append(_stage(2, "RESEARCH QUESTION BREAKDOWN  (save as rq_breakdown.json)",
                        prompts.rq_breakdown_system_prompt))
    parts.append(_stage(3, "RESEARCH PLAN  (save as research_plan.json)",
                        prompts.generate_research_plan_system_prompt))
    parts.append(_stage(4, "EXECUTE EACH PLAN STEP  (run_python, once per step)",
                        codegen_guidance(output_dir)))
    for i, (name, prompt) in enumerate(MANUSCRIPT_SECTIONS):
        parts.append(_stage(f"5.{i + 1}", f"MANUSCRIPT - {name.upper()}", prompt))
    return "".join(parts)


def build_user_message(research_question, datasets, output_dir):
    """The opening user turn: the question exactly as typed, plus where the
    data is and where outputs go."""
    if datasets:
        data_lines = "\n".join(f"- {d['name']}: {d['path']}" for d in datasets)
    else:
        data_lines = "- (no datasets were uploaded)"
    return (
        f"Research question:\n{research_question}\n\n"
        f"Uploaded datasets:\n{data_lines}\n\n"
        f"Output directory (OUTPUT_DIR): {output_dir}\n\n"
        "Carry out the full workflow, Stage 1 through Stage 5."
    )
