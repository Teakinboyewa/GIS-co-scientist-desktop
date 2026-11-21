import os
import uuid
from utils.config import load_configurations, resolve_output_dir
from utils.llm_models import llm_model, llm_reasoning
import utils.prompts as prompts
import utils.agm_helper as helper
from utils.agm_helper import get_ui_model, get_ui_reasoning_effort
from utils.output_path_guard import install_output_path_guard
from utils.lazy_imports import lazy_module
nx = lazy_module("networkx")  # loads on first use


class SpatialAnalysisExecutor():
    """

    """
    name = "SpatialAnalysisExecutor"

    def __init__(self, ctx):
        # super().__init__()
        self.api_key = helper.load_OpenAI_key()
        # self.model="gpt-6-luna",
        # self.reasoning_effort="low",
        # self.model="gpt-4o",
        self.settings = load_configurations()
        self.operations = []
        self.assembly_prompt = None
        self.solution_graph = self.solution_graph_from_graphfile(ctx)



    def log(self, ctx, message):
        line = f"[{self.name}]: {message}"
        ctx.setdefault("logs", []).append(line)
        print(line)

    def solution_graph_from_graphfile(self, ctx):
        # This runs in a separate request from the workflow generation step
        # (after the approval pause / on feedback re-runs), so we must resolve
        # the graph file robustly rather than trusting the stored path. The
        # stored artifact path is absolute and tied to the machine/worker that
        # generated it; resolve_graphfile() prefers it when it exists locally
        # and otherwise falls back to the deterministic per-task location
        # (<output_dir>/graphfiles/<task_id>.graphml), repairing the artifact.
        # Imported lazily to avoid any import-time coupling.
        from agents.task_manager.geoprocessing_workflow_generator import resolve_graphfile

        solution_graph_file = resolve_graphfile(ctx)
        if not solution_graph_file:
            raise RuntimeError(
                "No geoprocessing workflow graph file could be found for this task "
                f"(checked the stored path and the per-task location "
                f"{os.path.join(ctx.get('output_dir', '?'), 'graphfiles')}). The "
                "geoprocessing workflow generation step did not complete, or its "
                "output was not carried over when the context was restored. Please "
                "regenerate the geoprocessing workflow before running operation code "
                "generation."
            )

        self.solution_graph = nx.read_graphml(solution_graph_file)
        return self.solution_graph

    def operation_node_names(self, ctx):
        operation_node_names = []
        assert self.solution_graph, "The Solution class instance has no solution graph. Please generate the graph"
        for node_name in self.solution_graph.nodes():
            node = self.solution_graph.nodes[node_name]
            if node['node_type'] == 'operation':
                operation_node_names.append(node_name)
        return operation_node_names

    def generate_function_def(self, node_name, G):
        '''
        Return a dict, includes two lines: the function definition and return line.
        parameters: operation_node
        '''
        node_dict = G.nodes[node_name]
        node_type = node_dict['node_type']

        predecessors = G.predecessors(node_name)

        # print("predecessors:", list(predecessors))

        # create parameter list with default values
        para_default_str = ''  # for the parameters with the file path
        para_str = ''  # for the parameters without the file path
        for para_name in predecessors:
            # print("para_name:", para_name)
            para_node = G.nodes[para_name]
            # print(f"para_node: {para_node}")
            # print(para_node)
            data_path = para_node.get('data_path', '')  # if there is a path, the function need to read this file

            if data_path != "":
                para_default_str = para_default_str + f"{para_name}='{data_path}', "
            else:
                para_str = para_str + f"{para_name}={para_name}, "

        all_para_str = para_str + para_default_str

        function_def = f'{node_name}({all_para_str})'
        function_def = function_def.replace(', )', ')')  # remove the last ","

        # generate the return line
        successors = G.successors(node_name)
        return_str = 'return ' + ', '.join(list(successors))

        
        return_dict = {"function_definition": function_def,
                       "return_line": return_str,
                       'description': node_dict['description'],
                       'node_name': node_name
                       }
        return return_dict


    def initial_operations(self, ctx):
        self.operations = []
        operation_node_names = self.operation_node_names(ctx)
        for node_name in operation_node_names:
            function_def_returns = self.generate_function_def(node_name, self.solution_graph)
            self.operations.append(function_def_returns)

    def get_ancestor_operations(self, ctx, node_name):
        ancestor_operation_names = []
        ancestor_node_names = nx.ancestors(self.solution_graph, node_name)
        # for ancestor_node_name in ancestor_node_names:
        operation_node_names = self.operation_node_names(ctx)
        ancestor_operation_names = [node_name for node_name in ancestor_node_names if node_name in operation_node_names]

        ancestor_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in ancestor_operation_names:
                ancestor_operation_nodes.append(oper)
        return ancestor_operation_nodes

    def get_descendant_operations(self, ctx, node_name):
        descendant__operation_names = []
        descendant_node_names = nx.descendants(self.solution_graph, node_name)
        # for descendant_node_name in descendant_node_names:
        operation_node_names = self.operation_node_names(ctx)
        descendant__operation_names = [node_name for node_name in descendant_node_names if node_name in operation_node_names]
        # descendant_codes = '\n'.join([oper['operation_code'] for oper in descendant_node_names])
        descendant_operation_nodes = []
        for operation in self.operations:
            operation_name = operation['node_name']
            if operation_name in descendant__operation_names:
                descendant_operation_nodes.append(operation)
        return descendant_operation_nodes

    def get_descendant_operations_definition(self, descendant_operations):

        keys = ['node_name', 'description', 'function_definition', 'return_line']
        operation_def_list = []
        for node in descendant_operations:
            operation_def = {key: node[key] for key in keys}
            operation_def_list.append(str(operation_def))
        defs = '\n'.join(operation_def_list)
        return defs

    def get_prompt_for_an_operation(self, ctx, operation):
        assert self.solution_graph, "Do not find solution graph!"
        # operation_dict = function_def.copy()


        #-------- Extracting information from ctx

        task = ctx.get("rq", "")

        # Build input paths summary for the prompt
        data_registry = ctx["artifacts"].get("data_registry", [])
        # data_input_summary = [f"- {s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']}" for s in data_registry]
        data_input_summary = [f"- {s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']} | columns: {s['columns']}" for s in data_registry]
        data_input_summary_str = '\n'.join(data_input_summary)

        code_for_graph = ctx["artifacts"]["geoprocessing_workflow_code"]



        node_name = operation['node_name']
        # get ancestors code
        ancestor_operations = self.get_ancestor_operations(ctx, node_name)
        ancestor_operation_codes = '\n'.join([operation['operation_code'] for operation in ancestor_operations])
        descendant_operations = self.get_descendant_operations(ctx, node_name)
        descendant_defs = self.get_descendant_operations_definition(descendant_operations)
        descendant_defs_str = str(descendant_defs)

        pre_requirements = [
            f'The function description is: {operation["description"]}',
            f'The function definition is: {operation["function_definition"]}',
            f'The function return line is: {operation["return_line"]}'
        ]

        operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            pre_requirements + prompts.operation_requirement)])

        operation_prompt = f'Your role: {prompts.operation_role} \n\n' + \
                           f'operation_task: {prompts.operation_task_prefix} {operation["description"]} \n\n' + \
                           f'This function is one step to solve the question/task: {task} \n\n' + \
                           f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{code_for_graph} \n\n" + \
                           f'Data input summary: {data_input_summary_str} \n\n' + \
                           f'Your reply example: {prompts.operation_reply_example} \n\n' + \
                           f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n\n' + \
                           f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes} \n\n" + \
                           f"The descendant function (if any) definitions for the question are (node_name is function name): \n {descendant_defs_str}"

        operation['operation_prompt'] = operation_prompt
        return operation_prompt

    def get_LLM_responses_for_operations(self, ctx, review=False):
        self.initial_operations(ctx)
        event_cb = ctx.get("_event_callback")

        for idx, operation in enumerate(self.operations):
            node_name = operation['node_name']
            step_tag = f"operation_{idx + 1}_{node_name}"

            # Update stream step so chunks land on the right card
            ctx["_stream_step"] = step_tag

            # Emit a status event to create/label the card
            if event_cb:
                event_cb({
                    "type": "status",
                    "message": f"Operation {idx + 1}/{len(self.operations)}: {node_name}",
                    "step": step_tag,
                })

            print(
                f"{idx + 1} / {len(self.operations)}, LLM is generating code for operation node: {node_name}")
            system_prompt = self.get_prompt_for_an_operation(ctx, operation)

            messages = [
                {"role": "system", "content": system_prompt}
            ]

            try:
                response = helper.client_chat_completion_stream(
                    messages=messages,
                    model=llm_model("spatial_analysis.operation_code"), **llm_reasoning("spatial_analysis.operation_code")
                    
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
                print()  # newline after streaming
                operation['response'] = full_content
                try:
                    operation_code = helper.extract_code(response_content=full_content)
                except Exception as e:
                    operation_code = ""
                operation['operation_code'] = operation_code

                # Emit step_result to reformat the card (execution log + code)
                if event_cb:
                    event_cb({
                        "type": "step_result",
                        "step": step_tag,
                        "step_index": idx + 1,
                        "step_description": operation.get('description', node_name),
                        "status": "completed" if operation_code else "error",
                        "code": operation_code,
                        "output": "",
                        "error": "" if operation_code else "No code extracted from LLM response",
                        "artifacts": [],
                    })

            except Exception as llm_err:
                self.log(ctx, f"ERROR in LLM response for {node_name}: {llm_err}")
                raise

        # Save all operation codes into ctx
        ctx["artifacts"]["operation_codes"] = [
            {"node_name": op["node_name"], "description": op.get("description", ""), "code": op.get("operation_code", "")}
            for op in self.operations
        ]

        return self.operations



    def _get_final_output_nodes(self):
        """Identify the terminal (leaf) data nodes in the workflow graph.
        These are data nodes with no successors — they represent the final
        outputs that should be saved to FINAL_OUTPUT_DIR."""
        final_nodes = []
        for node_name in self.solution_graph.nodes():
            node = self.solution_graph.nodes[node_name]
            if node.get('node_type') == 'data' and self.solution_graph.out_degree(node_name) == 0:
                final_nodes.append({
                    "name": node_name,
                    "description": node.get('description', ''),
                })
        return final_nodes

    def prompt_for_assembly_program(self, ctx):

        # Build labeled operation code blocks so the LLM sees each function clearly
        labeled_operation_blocks = []
        for idx, operation in enumerate(self.operations):
            node_name = operation.get('node_name', f'operation_{idx+1}')
            code = operation.get('operation_code', '')
            labeled_operation_blocks.append(
                f"### Function {idx+1}: {node_name}\n"
                f"Description: {operation.get('description', '')}\n"
                f"```python\n{code}\n```"
            )
        all_operation_code_str = '\n\n'.join(labeled_operation_blocks)

        task = ctx.get("rq", "")
        # Build input paths summary for the prompt
        data_registry = ctx["artifacts"].get("data_registry", [])
        data_input_summary = [f"- {s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']} | columns: {s['columns']}" for s in data_registry]
        data_input_summary_str = '\n'.join(data_input_summary)

        assembly_requirement = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(prompts.assembly_requirement)])

        # Identify the final output nodes from the workflow graph so the LLM
        # knows exactly which results must be saved to FINAL_OUTPUT_DIR.
        final_output_nodes = self._get_final_output_nodes()
        if final_output_nodes:
            final_outputs_list = '\n'.join([
                f"  - **{n['name']}**: {n['description']}" for n in final_output_nodes
            ])
            final_outputs_section = (
                f"\n\n## FINAL OUTPUT NODES (from the geoprocessing workflow graph)\n"
                f"The workflow graph has the following terminal output node(s) — these are the final results "
                f"that MUST be saved to FINAL_OUTPUT_DIR (not OUTPUT_DIR):\n"
                f"{final_outputs_list}\n\n"
                f"Each of these MUST be written to a file in FINAL_OUTPUT_DIR using os.path.join(FINAL_OUTPUT_DIR, <filename>). "
                f"ALL other intermediate data should go to OUTPUT_DIR. "
                f"Do NOT skip saving any of these final outputs — the WebUI displays only files from FINAL_OUTPUT_DIR as result tiles."
            )
        else:
            final_outputs_section = ""

        assembly_prompt = f"Your role: {prompts.assembly_role} \n\n" + \
                          f"Your task is: assemble ALL the given Python functions below into a complete, executable Python program to solve the question: \n {task}\n\n" + \
                          f"CRITICAL: You MUST include EVERY function listed below in your assembly program. " + \
                          f"Copy each function definition EXACTLY as given" + \
                          f"The assembly program must define all {len(self.operations)} functions, then call them in the correct order inside 'assembely_solution()'.\n\n" + \
                          f"Requirement: \n {assembly_requirement} \n\n" + \
                          f"Data location: \n {data_input_summary_str} \n\n" + \
                          final_outputs_section + \
                          f"\n\nThere are {len(self.operations)} functions in total. ALL must appear in your code:\n\n{all_operation_code_str}"

        self.assembly_prompt = assembly_prompt
        return self.assembly_prompt


    def get_LLM_assembly_response(self, ctx, review=False):
        event_cb = ctx.get("_event_callback")
        step_tag = "assembly_code_generation"

        # Update stream step so chunks land on the assembly card
        ctx["_stream_step"] = step_tag

        if event_cb:
            event_cb({
                "type": "status",
                "message": "Generating assembly program...",
                "step": step_tag,
            })

        
        system_prompt = self.assembly_prompt or self.prompt_for_assembly_program(ctx)
        messages = [
                {"role": "system", "content": system_prompt}
            ]

        try:
            response = helper.client_chat_completion_stream(
                messages=messages,
                model=llm_model("spatial_analysis.assembly_code"), **llm_reasoning("spatial_analysis.assembly_code"),
                # model=get_ui_model(),
                # reasoning_effort = get_ui_reasoning_effort()
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
            print()  # newline after streaming

            self.assembly_LLM_response = full_content

            try:
                code_for_assembly = helper.extract_code(response_content=full_content)
            except Exception as e:
                code_for_assembly = ""

            self.code_for_assembly = code_for_assembly

            # Save assembly code into ctx
            ctx["artifacts"]["assembly_code"] = code_for_assembly
            self.log(ctx, f"System_prompt = {code_for_assembly}")

            # Emit step_result to reformat the card (execution log + code)
            if event_cb:
                event_cb({
                    "type": "step_result",
                    "step": step_tag,
                    "step_index": "Assembly",
                    "step_description": "Assembly program — combines all operations into a single executable script",
                    "status": "completed" if code_for_assembly else "error",
                    "code": code_for_assembly,
                    "output": "",
                    "error": "" if code_for_assembly else "No assembly code extracted from LLM response",
                    "artifacts": [],
                })

        except Exception as llm_err:
            self.log(ctx, f"ERROR in assembly LLM response: {llm_err}")
            self.code_for_assembly = ""
            raise

        # return self.assembly_LLM_response

    def _execute_generated_code(self, ctx, code, operation_description):
        """
        Execute LLM-generated Python code safely.
        Tracks files created during execution.
        Returns dict with status, output, created files, and any error.
        """
        import io
        import glob as globmod
        import contextlib

        settings = load_configurations()
        self.log(ctx, f"Executing Assembly code ...")

        # Snapshot files before execution to detect new and modified outputs
        output_dir = resolve_output_dir(ctx)
        os.makedirs(output_dir, exist_ok=True)
        
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

        stdout_capture = io.StringIO()
        
        exec_globals = {
            "__builtins__": __builtins__,
            "OUTPUT_DIR": output_dir,
            "FINAL_OUTPUT_DIR": final_output_dir,
        }
        _prev_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
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

        try:
            try:
                with contextlib.redirect_stdout(stdout_capture), install_output_path_guard(output_dir, final_output_dir):
                    exec(code, exec_globals)
                output = stdout_capture.getvalue()

                created_files = _detect_created_files()

                if created_files:
                    self.log(ctx, f"Code executed successfully. Created/modified {len(created_files)} file(s): {created_files}")
                else:
                    self.log(ctx, f"Code executed successfully. No new files detected.")

                return {"status": "completed", "output": output, "code": code, "created_files": created_files}

            except Exception as err:
                import traceback
                output = stdout_capture.getvalue()
                exception_type = type(err).__name__
                # Guard against empty-message exceptions (e.g. IterationError with no text)
                error = str(err) or f"{exception_type}"
                self.log(ctx, f"Code execution error: {error}")
                # Still report any files that were written before the crash
                partial_files = _detect_created_files()
                if partial_files:
                    self.log(ctx, f"Partial output detected before error: {len(partial_files)} file(s)")
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

    def execute_assembly_code(self, ctx):
        """Execute the assembly code and run debugger if errors occur."""
        assembly_code = ctx["artifacts"].get("assembly_code", "")
        if not assembly_code:
            self.log(ctx, "No assembly code found in ctx to execute.")
            return {"status": "error", "error": "No assembly code found"}
        task = ctx.get("rq", "N/A")
        operation_description = f"Assembly program for this task: {task}"

        # Save code to file for traceability
        settings = load_configurations()
        output_dir = resolve_output_dir(ctx)
        code_file = os.path.join(output_dir, f"generated_code_assembly_{uuid.uuid4().hex[:8]}.py")
        os.makedirs(os.path.dirname(code_file), exist_ok=True)
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(assembly_code)
        self.log(ctx, f"Assembly code saved to: {code_file}")

        # Prepend only missing operation function definitions to guard against LLM omissions
        operation_codes = ctx["artifacts"].get("operation_codes", [])
        if operation_codes:
            import re
            defined_fns = set(re.findall(r"^def\s+(\w+)\s*\(", assembly_code, re.MULTILINE))
            op_blocks = []
            for op in operation_codes:
                fn_code = op.get("code", "").strip()
                if not fn_code:
                    continue
                # Extract the function name from the operation code
                fn_name_match = re.search(r"^def\s+(\w+)\s*\(", fn_code, re.MULTILINE)
                fn_name = fn_name_match.group(1) if fn_name_match else None
                if fn_name and fn_name in defined_fns:
                    continue  # already in assembly code, skip
                op_blocks.append(f"# --- {op['node_name']} ---\n{fn_code}")
            if op_blocks:
                assembly_code = "\n\n".join(op_blocks) + "\n\n" + assembly_code

        result = self._execute_generated_code(ctx, assembly_code, operation_description)
        result["code_file"] = code_file

        # ── Inline debugger: detect errors and attempt automatic repair ─────
        _error = result.get("error", "")
        error_message = result.get("error_message", "")

        if result.get("status") == "error":
            self.log(ctx, f"  [DEBUG] Execution error: {_error}")

            from agents.task_manager.debugger import run_debugger

            # Build input paths summary from data registry
            data_registry = ctx["artifacts"].get("data_registry", [])
            input_paths_summary = ""
            for s in data_registry:
                input_paths_summary += f"  - {s['file']}: {s['path']}\n"

            # Build a failed-step dict matching what run_debugger expects
            failed_step = {
                "status": "error",
                "failed_code": assembly_code,
                "step_idx": 0,
                "operation_description": operation_description,
                "result": result,
                "error_message": error_message,
                "input_paths_summary": input_paths_summary,
                "output_node": "",
                "output_node_path": output_dir,
                "operation_codes": ctx["artifacts"].get("operation_codes", []),
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
                
                if fixed_code and not repair.get("created_files"):
                    from agents.task_manager.debugger import _execute_fixed_code
                    reexec = _execute_fixed_code(fixed_code, output_dir=output_dir)
                    if reexec.get("status") == "completed":
                        repair["output"] = reexec.get("output", "")
                        repair["created_files"] = reexec.get("created_files", [])
                        repair["code_file"] = reexec.get("code_file", repair.get("code_file", ""))
                    else:
                        # Re-execution failed — mark repair as failed
                        repair["success"] = False
                        repair["notes"] = f"Re-execution of fixed code failed: {reexec.get('error', '')}"

                # ── Interactive mode: show fixed code for review before accepting ──
                _settings = ctx.get("settings", {})
                interactive = _settings.get("interactive_code", False) if isinstance(_settings, dict) else getattr(_settings, "interactive_code", False)
                i_review_q = ctx.get("_code_review_queue")
                i_response_q = ctx.get("_code_review_response_queue")

                if interactive and i_review_q and i_response_q and fixed_code:
                    if stream_cb:
                        stream_cb(f"  Debugger produced fixed code. Waiting for review...\n")
                    i_review_q.put({
                        "type": "code_review",
                        "code": fixed_code,
                        "code_file": repair.get("code_file", ""),
                        "step_index": 0,
                        "step_description": f"[DEBUG FIX] {operation_description}",
                        "agent_name": "Debugger",
                        "step": ctx.get("_stream_step", "assembly_code_generation"),
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
                            rerun = _execute_fixed_code(user_fixed_code, output_dir=output_dir)
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
                    # Update assembly code in ctx with the fixed version
                    ctx["artifacts"]["assembly_code"] = repair.get("fixed_code", "")
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
            # ── Fallback: regenerate assembly from scratch if debugger failed ──
            if result.get("status") == "error":
                self.log(ctx, "[DEBUG FALLBACK] Debugger failed — regenerating assembly code from scratch...")
                if stream_cb:
                    stream_cb("  Debugger could not fix the error. Regenerating assembly from scratch...\n")

                error_analysis = debug_result.get("error_analysis", {}) if debug_result else {}
                root_cause = error_analysis.get("root_cause", error_message)
                regen_hint = (
                    f"\n\n## IMPORTANT — Previous Code Failed With Error\n"
                    f"The previous assembly code failed with: {error_message}\n"
                    f"Debugger diagnosis: {root_cause}\n"
                    f"Traceback:\n{result.get('error_traceback', '')[:500]}\n\n"
                    f"Generate a COMPLETELY NEW assembly implementation that avoids this error.\n"
                    f"Do NOT reuse the same approach — try an alternative strategy.\n"
                )

                # Rebuild the assembly prompt with the error context
                import utils.prompts as prompts
                assembly_prompt = self.prompt_for_assembly_program(ctx)
                assembly_prompt += regen_hint
                self.assembly_prompt = assembly_prompt

                try:
                    self.get_LLM_assembly_response(ctx)
                    regen_code = ctx["artifacts"].get("assembly_code", "")
                    if regen_code:
                        from agents.task_manager.debugger import _execute_fixed_code
                        regen_exec = _execute_fixed_code(regen_code, output_dir=output_dir)
                        if regen_exec.get("status") == "completed":
                            self.log(ctx, "[DEBUG FALLBACK] Regeneration succeeded.")
                            if stream_cb:
                                stream_cb("  Regenerated code executed successfully.\n")
                            result = {
                                "status": "completed",
                                "output": regen_exec.get("output", ""),
                                "code": regen_code,
                                "code_file": regen_exec.get("code_file", code_file),
                                "created_files": regen_exec.get("created_files", []),
                                "error": None,
                            }
                        else:
                            self.log(ctx, "[DEBUG FALLBACK] Regenerated code also failed.")
                            if stream_cb:
                                stream_cb("  Regenerated code also failed.\n")
                except Exception as regen_err:
                    self.log(ctx, f"[DEBUG FALLBACK] Regeneration error: {regen_err}")
                    if stream_cb:
                        stream_cb(f"  Regeneration error: {regen_err}\n")

        else:
            _output = result.get("output", "")
            if _output:
                self.log(ctx, f"  [DEBUG] Execution output: {_output[:500]}")

        # Save execution result into ctx
        ctx["artifacts"]["assembly_execution_result"] = {
            "status": result.get("status", "error"),
            "output": result.get("output", ""),
            "code_file": result.get("code_file", ""),
            "created_files": result.get("created_files", []),
            "error": result.get("error", ""),
        }

        return result
