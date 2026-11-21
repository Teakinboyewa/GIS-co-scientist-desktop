import os
import sys
import traceback
import uuid
import time

from utils.lazy_imports import lazy_module
nx = lazy_module("networkx")  # loads on first use
import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning
from utils.agm_helper import get_ui_model, get_ui_reasoning_effort
from utils.config import load_configurations
from utils.lazy_imports import lazy_function
# pyvis loads IPython and networkx with it (several seconds), so it loads on first use.
Network = lazy_function("pyvis.network", "Network")
from utils.lazy_imports import lazy_function  # IPython loads on first use
display = lazy_function("IPython.display", "display")
IFrame = lazy_function("IPython.display", "IFrame")
clear_output = lazy_function("IPython.display", "clear_output")
import agents.task_manager.WEBUI_TASK_PIPELINE.task_solution_prompts as solution_prompts
# import task_solution_prompts as solution_prompts

from utils.lazy_imports import is_installed
# ipywidgets (notebook UI) is imported only where widgets are built.
WIDGETS_AVAILABLE = is_installed("ipywidgets")
class Solution:
    """
    Class for providing analytical solution for tasks/steps. This class help to break task into operational steps
    """

    def _llm_model(self, name):
        """The model for the LLM call *name*: the one passed to __init__, else the configured one."""
        return self.model if getattr(self, "_model_given", False) else llm_model(name)

    def _llm_reasoning(self, name):
        """Reasoning-effort kwargs for *name*; none when a model was passed to __init__."""
        return {} if getattr(self, "_model_given", False) else llm_reasoning(name)

    def __init__(self, task, data_overview, graph_role = solution_prompts.graph_role, model=None, model_requirement=None, task_id=None, stream_callback=None, output_dir=None):
        self.task = task
        self.data_overview = data_overview
        self.graph_role = graph_role
        # A model passed in wins for every call; otherwise each call takes its
        # own model and effort from utils/llm_models.toml (see _llm_model).
        self._model_given = bool(model)
        self.model = model or llm_model("agent_solution.graph")
        self.model_requirement = model_requirement
        self.task_id = task_id or str(uuid.uuid4())
        self.operational_steps = []
        self.solution_graph = None
        self.stream_callback = stream_callback  # Callback for streaming LLM responses to WebUI

        # Use per-task output directory if provided, otherwise global
        settings = load_configurations()
        self.output_dir = output_dir or settings.output_dir
        self.graph_dir = os.path.join(self.output_dir, "graphfiles")
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_file = os.path.join(self.graph_dir, f"{self.task_id}.graphml")
        self.graph_html_file = os.path.join(self.graph_dir, f"{self.task_id}.html")

        # self.graph_prompt_text = self.graph_prompt(task=task)
        self.graph_response = None
        self.api_key = helper.load_OpenAI_key()
        self.code_for_graph = ""

        self.assembly_LLM_response=""

        self.graph_prompt = ""
        self.data_overview_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(self.data_overview)])

        graph_requirement = solution_prompts.graph_requirement.copy()
        graph_requirement.append(f"The graph must be saved at: {self.graph_file}")
        self.graph_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])

        graph_prompt = f'Your role:  {self.graph_role} \n\n' + \
                          f'Your task: {solution_prompts.graph_task_prefix} \n {self.task} \n\n'+ \
                          f'Your reply needs to meet these requirements: \n {self.graph_requirement_str} \n\n' + \
                          f'your reply example: {solution_prompts.graph_reply_exmaple} \n\n' + \
                          f'Data overview for the task:\n {self.data_overview_str} \n'
        self.graph_prompt = graph_prompt

        # Human-in-the-loop feedback tracking
        self.is_approved = False
        self.revision_history = []
        self.revision_count = 0



    def get_LLM_responses_for_operations(self, stream=True, yield_chunks=False):
        """
        Generate operation codes from LLM with optional streaming
        Args:
            stream: Whether to stream from OpenAI
            yield_chunks: If True, yields streaming chunks for WebUI display
        Yields (if yield_chunks=True):
            Dict with streaming chunk info
        Returns:
            List of operations with generated code (if yield_chunks=False)
        """
        self.initial_operations()
        for idx, operation in enumerate(self.operations):
            print(f"{idx + 1} / {len(self.operations)}, LLM is generating code for operation node: {operation['node_name']}")

            # Yield status update if yielding chunks
            if yield_chunks:
                yield {
                    'type': 'llm_update',
                    'content': f"🔧 Generating code for operation {idx + 1}/{len(self.operations)}: {operation['node_name']}",
                    'step': 'operation_code_gen'
                }

            prompt = self.get_prompt_for_an_operation(operation)
            messages = [{"role": "system", "content": solution_prompts.operation_role},
                        {"role": "user", "content": prompt}]

            user_key = helper.load_OpenAI_key()
            if stream:
                response = helper.gibd_chat_completion_stream(user_key, self._llm_model("agent_solution.operations"), messages, **self._llm_reasoning("agent_solution.operations"))
            else:
                response = helper.gibd_chat_completion(user_key, self._llm_model("agent_solution.operations"), messages, **self._llm_reasoning("agent_solution.operations"))

            # Collect the full response content
            full_content = ""
            if stream:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        full_content += chunk_content
                        print(chunk_content, end='', flush=True)

                        # Yield chunk for WebUI streaming display
                        if yield_chunks:
                            yield {
                                'type': 'llm_stream',
                                'content': chunk_content,
                                'step': 'code_streaming'
                            }
                        # Send chunk to callback if provided (legacy support)
                        elif self.stream_callback:
                            self.stream_callback(chunk_content)
                print()  # New line after streaming completes
            else:
                full_content = response.choices[0].message.content
                print(full_content)

            operation['response'] = full_content

            # Extract code from the response content
            try:
                operation_code = helper.extract_code(response_content=full_content)
                if not operation_code:
                    print(f"\nWarning: No code block found in response for '{operation['node_name']}'")
                    print(f"Response preview: {full_content[:300]}..." if len(full_content) > 300 else f"Full response: {full_content}")
                else:
                    print(f"\nSuccessfully extracted code for '{operation['node_name']}' ({len(operation_code)} chars)")
            except Exception as e:
                print(f"\nFailed to extract code for '{operation['node_name']}': {e}")
                operation_code = ""
            operation['operation_code'] = operation_code

        if not yield_chunks:
            return self.operations


    def get_prompt_for_an_operation(self, operation):
        assert self.solution_graph, "Do not find solution graph!"
        # operation_dict = function_def.copy()

        node_name = operation['node_name']

        # get ancestors code
        ancestor_operations = self.get_ancestor_operations(node_name)
        ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
        descendant_operations = self.get_descendant_operations(node_name)
        descendant_defs = self.get_descendant_operations_definition(descendant_operations)
        descendant_defs_str = str(descendant_defs)

        pre_requirements = [
            f'The function description is: {operation["description"]}',
            f'The function definition is: {operation["function_definition"]}',
            f'The function return line is: {operation["return_line"]}',
            f'IMPORTANT: Save all output files (maps, CSVs, results, etc.) to the Outputs directory at: {self.output_dir}',
            f'Include the workflow ID "{self.task_id}" in output filenames to make them unique (e.g., "map_{self.task_id}.png", "results_{self.task_id}.csv")'
            ]

        operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            pre_requirements + solution_prompts.operation_requirement)])

        operation_prompt = f'Your role: {solution_prompts.operation_role} \n\n' + \
                           f'operation_task: {solution_prompts.operation_task_prefix} {operation["description"]} \n\n' + \
                           f'This function is one step to solve the question/task: {self.task} \n\n' + \
                           f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{self.code_for_graph} \n\n" + \
                           f'Your reply example: {solution_prompts.operation_reply_exmaple} \n\n' + \
                           f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n\n' + \
                           f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes} \n\n" + \
                           f"The descendant function (if any) definitions for the question are (node_name is function name): \n {descendant_defs_str}"

        operation['operation_prompt'] = operation_prompt
        return operation_prompt

    @property
    def operation_node_names(self):
        opera_node_names = []
        assert self.solution_graph, "The Solution class instance has no solution graph. Please generate the graph"
        for node_name in self.solution_graph.nodes():
            node = self.solution_graph.nodes[node_name]
            if node['node_type'] == 'operation':
                opera_node_names.append(node_name)
        return opera_node_names

    def get_ancestor_operations(self, node_name):
        ancestor_operation_names = []
        ancestor_node_names = nx.ancestors(self.solution_graph, node_name)
        # for ancestor_node_name in ancestor_node_names:
        ancestor_operation_names = [node_name for node_name in ancestor_node_names if node_name in self.operation_node_names]

        ancestor_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in ancestor_operation_names:
                ancestor_operation_nodes.append(oper)

        return ancestor_operation_nodes


    def get_descendant_operations(self, node_name):
        descendant__operation_names = []
        descendant_node_names = nx.descendants(self.solution_graph, node_name)
        # for descendant_node_name in descendant_node_names:
        descendant__operation_names = [node_name for node_name in descendant_node_names if node_name in self.operation_node_names]
        # descendant_codes = '\n'.join([oper['operation_code'] for oper in descendant_node_names])
        descendant_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in descendant__operation_names:
                descendant_operation_nodes.append(oper)

        return descendant_operation_nodes

    def get_descendant_operations_definition(self, descendant_operations):

        keys = ['node_name', 'description', 'function_definition', 'return_line']
        operation_def_list = []
        for node in descendant_operations:
            operation_def = {key: node[key] for key in keys}
            operation_def_list.append(str(operation_def))
        defs = '\n'.join(operation_def_list)
        return defs



    # def get_LLM_response_for_graph(self, stream=True, execute=True, yield_chunks=False):
    #     """
    #     Generate graph from LLM with optional streaming
    #     Args:
    #         stream: Whether to stream from OpenAI
    #         execute: Whether to execute the graph code
    #         yield_chunks: If True, yields streaming chunks for WebUI display
    #     Yields (if yield_chunks=True):
    #         Dict with streaming chunk info
    #     Returns:
    #         Graph response text (if yield_chunks=False)
    #     """
    #     messages = [{"role": "system", "content": self.graph_role},
    #                 {"role": "user", "content": self.graph_prompt}]
    #
    #     full_prompt = f"[SYSTEM]\n{self.graph_role}\n\n[USER]\n{self.graph_prompt}"
    #     print(full_prompt)
    #
    #     client = OpenAI(api_key=self.api_key)
    #     response = client.chat.completions.create(model=self.model,
    #                                                   messages=messages,
    #                                                    stream=stream)
    #
    #     full_content = ""
    #     if stream:
    #         for chunk in response:
    #             chunk_content = chunk.choices[0].delta.content
    #             if chunk_content is not None:
    #                 full_content += chunk_content
    #                 print(chunk_content, end='', flush=True)
    #
    #                 # Yield chunk for WebUI streaming display
    #                 if yield_chunks:
    #                     yield {
    #                         'type': 'llm_stream',
    #                         'content': chunk_content,
    #                         'step': 'graph_streaming'
    #                     }
    #                 # Send chunk to callback if provided (legacy support)
    #                 elif self.stream_callback:
    #                     self.stream_callback(chunk_content)
    #     else:
    #         full_content = response.choices[0].message.content
    #         print(full_content)
    #     print('\n\n')
    #     print("Got LLM reply.")
    #
    #     self.graph_response = full_content
    #     try:
    #         self.code_for_graph = helper.extract_code(response_content=self.graph_response)
    #     except Exception as e:
    #         self.code_for_graph = ""
    #         print("Extract graph Python code rom LLM failed.")
    #     if execute:
    #         self.execute_graph_code()
    #         # self.load_graph_file()
    #
    #     if not yield_chunks:
    #         return self.graph_response

    def get_LLM_response_for_graph_ModelingAgent(self, stream=True, execute=True, yield_chunks=False):
        """
        Generate graph from LLM with optional streaming specifically for Modeling Agent
        Args:
            stream: Whether to stream from OpenAI
            execute: Whether to execute the graph code
            yield_chunks: If True, yields streaming chunks for WebUI display
        Yields (if yield_chunks=True):
            Dict with streaming chunk info
        Returns:
            Graph response text (if yield_chunks=False)
        """

        messages = [{"role": "system", "content": self.graph_role},
                    {"role": "user", "content": self.graph_prompt}]

        full_prompt = f"[SYSTEM]\n{self.graph_role}\n\n[USER]\n{self.graph_prompt}"
        print(full_prompt)

        user_key = helper.load_OpenAI_key()
        if stream:
            response = helper.gibd_chat_completion_stream(user_key, self._llm_model("agent_solution.graph"), messages, **self._llm_reasoning("agent_solution.graph"))
        else:
            response = helper.gibd_chat_completion(user_key, self._llm_model("agent_solution.graph"), messages, **self._llm_reasoning("agent_solution.graph"))

        full_content = ""
        if stream:
            for chunk in response:
                if not chunk.choices:
                    continue
                chunk_content = chunk.choices[0].delta.content
                if chunk_content is not None:
                    full_content += chunk_content
                    print(chunk_content, end='', flush=True)

                    # Yield chunk for WebUI streaming display
                    if yield_chunks:
                        # print(f"[DEBUG] Yielding chunk: {chunk_content[:50]}...")  # Add this line
                        yield {
                            'type': 'llm_stream',
                            'content': chunk_content,
                            'step': 'graph_streaming'
                        }
                    # Send chunk to callback if provided (legacy support)
                    elif self.stream_callback:
                        self.stream_callback(chunk_content)
        else:
            full_content = response.choices[0].message.content
            print(full_content)
        print('\n\n')
        print("Got LLM reply.")

        self.graph_response = full_content
        try:
            self.code_for_graph = helper.extract_code(response_content=self.graph_response)
        except Exception as e:
            self.code_for_graph = ""
            print("Extract graph Python code rom LLM failed.")
        if execute:
            self.execute_graph_code()
            # self.load_graph_file()

        if not yield_chunks:
            return self.graph_response

    def get_LLM_response_for_graph_DataAgent(self, stream=True, execute=True, yield_chunks=False):
        """
        Generate graph from LLM with optional streaming specifically for Data Agent
        Args:
            stream: Whether to stream from OpenAI
            execute: Whether to execute the graph code
            yield_chunks: If True, yields streaming chunks for WebUI display
        Yields (if yield_chunks=True):
            Dict with streaming chunk info
        Returns:
            Graph response text (if yield_chunks=False)
        """

        messages = [{"role": "system", "content": self.graph_role},
                    {"role": "user", "content": self.graph_prompt}]

        full_prompt = f"[SYSTEM]\n{self.graph_role}\n\n[USER]\n{self.graph_prompt}"
        print(full_prompt)

        user_key = helper.load_OpenAI_key()
        if stream:
            response = helper.gibd_chat_completion_stream(user_key, self._llm_model("agent_solution.graph"), messages, **self._llm_reasoning("agent_solution.graph"))
        else:
            response = helper.gibd_chat_completion(user_key, self._llm_model("agent_solution.graph"), messages, **self._llm_reasoning("agent_solution.graph"))

        full_content = ""
        if stream:
            for chunk in response:
                if not chunk.choices:
                    continue
                chunk_content = chunk.choices[0].delta.content
                if chunk_content is not None:
                    full_content += chunk_content
                    print(chunk_content, end='', flush=True)

                    # Yield chunk for WebUI streaming display
                    if yield_chunks:
                        # print(f"[DEBUG] Yielding chunk: {chunk_content[:50]}...")  # Add this line
                        yield {
                            'type': 'llm_stream',
                            'content': chunk_content,
                            'step': 'graph_streaming'
                        }
                    # Send chunk to callback if provided (legacy support)
                    elif self.stream_callback:
                        self.stream_callback(chunk_content)
        else:
            full_content = response.choices[0].message.content
            print(full_content)
        print('\n\n')
        print("Got LLM reply.")

        self.graph_response = full_content
        try:
            self.code_for_graph = helper.extract_code(response_content=self.graph_response)
        except Exception as e:
            self.code_for_graph = ""
            print("Extract graph Python code rom LLM failed.")
        if execute:
            self.execute_graph_code()
            # self.load_graph_file()

        if not yield_chunks:
            return self.graph_response

    
    # def get_LLM_response_for_graph(self, stream=True, execute=True):
    #     system_role = """ You are a senior GIS expert and geoprocessing workflow architect
    #     GOAL:
    #     Given a geospatial task, generate a directed execution graph whose nodes are (1) a series of consecutive steps and (2) data that may be needed to solve the task.
        
    #     GRAPH STRUCTURE:
    #     - The graph must be fully connected (NO disconnected components).
    #     - There are exactly two node types:
    #         1) Operation nodes (data processing steps)
    #         2) Data nodes (input and output datasets or variables)
            
    #     WORKFLOW RULES:
    #     1. Each operation represents a single GIS data-processing action.
    #     2. Inputs to an operation must be:
    #        - Outputs from previous operations, OR
    #        - Data loaded or collected from a file path. 
    #     3. IF analysis model is specified, the workflow should be based on the analytical steps of performing the analysis model      
    #     3. The output of the final operation represents the final answer to the task AND must be based on the selected analysis model when provided.
    #     4. Join attributes to vector layers via a common attribute when required.
    #     6. Spatial data (vector or raster) is required to produce a map.
    #     7. Keep the graph concise — avoid unnecessary operation nodes.
    #     8. Do NOT include GraphML writing as an operation in the graph.
    #     9. Do NOT generate code that implements the GIS operations themselves.
    #     10. The output of the last operation is the final answer to the task, which in some cases may be more than 1.
        
    #     NODE ATTRIBUTES (DO NOT CHANGE ATTRIBUTE NAMES):
    #     - For all nodes:
    #       - node_type: "data" or "operation"
    #       - description: clear and concise explanation
    #     - For data nodes only:
    #       - data_path: file path string, or "" if not applicable
        
    #     Example data node attributes:
    #     {
    #       "name": "County boundary",
    #       "path": "D:\\Test\\county.shp",
    #       "columns": "County boundary for the study area",
    #       ...
    #     }
    
    #     NAMING CONSTRAINTS:
    #     - DATA NODE NAMES:
    #         1. First, check whether a name is available in the `name` field of the provided `data_overview`.
    #             - If available, use the EXACT name (no modification).
    #         2. If the `name` field is missing in the `data_overview`, derive the name from the data source
    #             (e.g., filename with extension or variable name).
    #         3. The final data node name must be human-readable and concise.
    #     - OPERATION NODE NAMES:
    #         - The operation node names must be human-readable and concise
    #         - Avoid overly long or ambiguous names.
        
    #     EDGES:
    #     - An edge represents a connection between a data node and an operation node.
    #     - All inputs and outputs must be explicitly connected via edges.
        
    #     OUTPUT REQUIREMENTS:
    #     - Add ALL nodes and edges to a NetworkX instance.
    #     - Include all required node attributes.
    #     - Do NOT use dicts or lists as attribute values (GraphML limitation).
    #     - Output ONLY Python code that constructs the NetworkX graph.
    #     Do NOT include any explanation, comments, or conversation outside the code block.
    #     - Enclose the entire reply in a Python code block:
    #       ```python
          
    #     REPLY EXAMPLE:
    #     ```python
    #     import networkx as nx
        
    #     # Create a directed graph
    #     G = nx.DiGraph()
        
    #     # Add data nodes
    #     G.add_node("Data_Name", node_type="data", description="Geopackage with various demographic and land use variables", data_path="data/data.gpkg")
        
    #     # Add operation nodes
    #     G.add_node("Load Data", node_type="operation", description="Load the spatial data from a geopackage")
    #     G.add_node("Preprocess Data", node_type="operation", description="Preprocess the data for regression")
    #     G.add_node("Run regression", node_type="operation", description="Regression analysis")
        
    #     # Add edges
    #     G.add_edge("Data_Name", "Load Data")
    #     G.add_edge("Load Data", "Preprocess Data")
    #     G.add_edge("Preprocess Data", "Run regression")
        
    #     # The final output node
    #     G.add_node("Regression Output", node_type="data", description="Output results of regression analysis", data_path="")
    #     G.add_edge("Run regression", "Regression Output")
    #     ```
       
    #     """

    #     user_prompt = f"""## Task Description

    #     {self.task}

    #     ## Data Overview
    #     {self.data_overview}
        
    #     ## Analysis model
    #     {self.model_requirement}

    #     Save the network into GraphML format, save it at:
    #     {self.graph_file}
              
    #     """
    #     messages = [{"role": "system", "content": system_role},
    #                 {"role": "user", "content": user_prompt}]

    #     full_prompt = f"[SYSTEM]\n{system_role}\n\n[USER]\n{user_prompt}"
    #     print(full_prompt)
    #     client = OpenAI(api_key=self.api_key)
    #     response = client.chat.completions.create(model=self.model,
    #                                                   messages=messages,
    #                                                    stream=stream)

    #     full_content = ""
    #     if stream:
    #         for chunk in response:
    #             chunk_content = chunk.choices[0].delta.content
    #             if chunk_content is not None:
    #                 full_content += chunk_content
    #                 print(chunk_content, end='', flush=True)
    #     else:
    #         full_content = response.choices[0].message.content
    #         print(full_content)
    #     print('\n\n')
    #     print("Got LLM reply.")

    #     self.graph_response = full_content
    #     try:
    #         self.code_for_graph = helper.extract_code(response_content=self.graph_response)
    #     except Exception as e:
    #         self.code_for_graph = ""
    #         print("Extract graph Python code rom LLM failed.")
    #     if execute:
    #         self.execute_graph_code()
    #         # self.load_graph_file()
    #     return self.graph_response

    def execute_graph_code(self):
        """Execute the generated graph code with proper namespace."""
        print(f"\nDEBUG: === execute_graph_code() START ===")
        if not self.code_for_graph:
            print("No code to execute.")
            return False
        try:
            print(f"DEBUG: Preparing execution environment...")
            print(f"DEBUG: Target graph_file = {self.graph_file}")

            # Provide necessary modules and variables in the execution namespace
            exec_globals = {
                "nx": nx,
                "networkx": nx,
                "graph_file": self.graph_file,
                "__builtins__": __builtins__
            }
            exec_locals = {}

            print(f"DEBUG: Executing graph code...")
            exec(self.code_for_graph, exec_globals, exec_locals)
            print(f"DEBUG: Code execution completed")

            # Try to find the graph object (usually named 'G') and save it
            print(f"DEBUG: Looking for graph object 'G'...")
            graph_obj = exec_locals.get("G") or exec_globals.get("G")

            if graph_obj is not None:
                print(f"DEBUG: Found graph object 'G'")
                print(f"DEBUG: Graph type = {type(graph_obj)}")
                print(f"DEBUG: Graph nodes = {graph_obj.number_of_nodes()}")
                print(f"DEBUG: Graph edges = {graph_obj.number_of_edges()}")

                # Check and fix None values in graph data (GraphML doesn't support None)
                print(f"DEBUG: Checking for None values in graph data...")
                has_none = False

                # Fix None values in node attributes
                for node in graph_obj.nodes():
                    for key, value in graph_obj.nodes[node].items():
                        if value is None:
                            print(f"WARNING: Node '{node}' has None value for attribute '{key}' - replacing with empty string")
                            graph_obj.nodes[node][key] = ""
                            has_none = True

                # Fix None values in edge attributes
                for source, target in graph_obj.edges():
                    for key, value in graph_obj.edges[source, target].items():
                        if value is None:
                            print(f"WARNING: Edge '{source}' -> '{target}' has None value for attribute '{key}' - replacing with empty string")
                            graph_obj.edges[source, target][key] = ""
                            has_none = True

                if has_none:
                    print(f"DEBUG: Fixed None values by replacing with empty strings")

                print(f"DEBUG: Writing graph to GraphML file...")
                nx.write_graphml(graph_obj, self.graph_file)
                print(f"DEBUG: nx.write_graphml() completed")

                # Verify file was written
                if os.path.exists(self.graph_file):
                    file_size = os.path.getsize(self.graph_file)
                    print(f"DEBUG: Graph file saved successfully - size = {file_size} bytes")
                    if file_size == 0:
                        print("ERROR: File size is 0 bytes!")
                else:
                    print(f"ERROR: File does not exist after write!")

                print(f"Graph saved to: {self.graph_file}")
            else:
                print(f"DEBUG: Graph object 'G' not found in exec_locals or exec_globals")
                print(f"DEBUG: exec_locals keys: {list(exec_locals.keys())}")
                print(f"DEBUG: exec_globals keys (custom): {[k for k in exec_globals.keys() if not k.startswith('__')]}")

                # Check if file was already saved by the exec'd code
                if os.path.exists(self.graph_file):
                    print(f"Graph already saved to: {self.graph_file}")
                else:
                    print("Warning: Could not find graph object 'G' in executed code.")
                    return False

            print(f"DEBUG: === execute_graph_code() END (SUCCESS) ===\n")
            return True
        except Exception as e:
            print(f"ERROR: Error executing graph code: {e}")
            import traceback
            traceback.print_exc()
            print(f"DEBUG: === execute_graph_code() END (FAILED) ===\n")
            return False

    def load_graph_file(self, file="", interactive=True):
        """
        Load a graph from file and optionally start interactive review.

        Args:
            file (str): Path to graph file. Uses self.graph_file if not specified.
            interactive (bool): If True, displays graph and starts interactive review.
                              If False, just loads and saves the graph visualization.

        Returns:
            tuple: (solution_graph, network_object) for further use
        """
        graph_path = file if file else self.graph_file

        if not os.path.exists(graph_path):
            print(f"Graph file not found: {graph_path}")
            return None, None

        if file:
            self.graph_file = file
            # Derive HTML path from the provided file path
            self.graph_html_file = graph_path.replace('.graphml', '.html')

        self.solution_graph = nx.read_graphml(graph_path)
        print(f"Graph loaded successfully: {len(self.solution_graph.nodes())} nodes, {len(self.solution_graph.edges())} edges")

        nt = self.show_graph(graph_file=self.solution_graph)
        nt.save_graph(self.graph_html_file)
        print(f"Graph visualization saved to: {self.graph_html_file}")

        # Display the graph using pyvis's native show method
        self.display_graph_html(nt=nt)

        if interactive:
            # Start interactive review after graph is displayed
            self.interactive_review()
        else:
            print("Interactive review disabled. Use solution.interactive_review() to start review later.")

        # Return the network object so it can be used further if needed
        return self.solution_graph, nt

    def display_graph_html(self, nt=None, height='800px'):
        """
        Display the graph HTML file in the notebook.

        Args:
            nt: Network object (if provided, uses nt.show() for proper rendering)
            height: Height of the display iframe
        """
        if nt is not None:
            # Use pyvis's native show method which handles JavaScript properly
            return nt.show(self.graph_html_file)
        else:
            # Fallback: use IFrame with cache-busting
            cache_buster = int(time.time() * 1000)
            display(IFrame(f"{self.graph_html_file}?v={cache_buster}", width='100%', height=height))

    # ==================== Human-in-the-Loop Feedback Methods ====================

    def list_nodes(self):
        """Display all nodes in the current graph with their attributes."""
        if self.solution_graph is None:
            print("No graph loaded. Please load a graph first.")
            return

        print("\n" + "=" * 60)
        print("CURRENT GRAPH NODES")
        print("=" * 60)

        data_nodes = []
        operation_nodes = []

        for node, attrs in self.solution_graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            if node_type == 'data':
                data_nodes.append((node, attrs))
            else:
                operation_nodes.append((node, attrs))

        print("\n📁 DATA NODES:")
        print("-" * 40)
        for node, attrs in data_nodes:
            desc = attrs.get('description', 'No description')
            path = attrs.get('data_path', '')
            print(f"  • {node}")
            print(f"    Description: {desc}")
            if path:
                print(f"    Path: {path}")

        print("\n⚙️ OPERATION NODES:")
        print("-" * 40)
        for node, attrs in operation_nodes:
            desc = attrs.get('description', 'No description')
            print(f"  • {node}")
            print(f"    Description: {desc}")

        print("\n📊 EDGES:")
        print("-" * 40)
        for source, target in self.solution_graph.edges():
            print(f"  {source} → {target}")

        print("=" * 60 + "\n")

    def get_graph_summary(self):
        """Get a text summary of the current graph for LLM context."""
        if self.solution_graph is None:
            return "No graph available."

        summary = "Current Graph Structure:\n"
        summary += f"- Total nodes: {len(self.solution_graph.nodes())}\n"
        summary += f"- Total edges: {len(self.solution_graph.edges())}\n\n"

        summary += "Nodes:\n"
        for node, attrs in self.solution_graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            desc = attrs.get('description', 'No description')
            summary += f"- {node} (type: {node_type}): {desc}\n"

        summary += "\nEdges:\n"
        for source, target in self.solution_graph.edges():
            summary += f"- {source} → {target}\n"

        return summary

    def refine_graph(self, feedback, stream=True, yield_chunks=False):
        """
        Refine the graph based on user feedback.
        Sends the current graph and feedback to LLM to generate an updated graph.

        Args:
            feedback: User feedback for refinement
            stream: Whether to stream from OpenAI
            yield_chunks: If True, yields streaming chunks for WebUI display
        """
        if self.solution_graph is None:
            print("No graph to refine. Please generate a graph first.")
            return None

        self.revision_count += 1
        self.revision_history.append({
            "revision": self.revision_count,
            "feedback": feedback
        })

        print(f"\n{'=' * 60}")
        print(f"REVISION #{self.revision_count}")
        print(f"Feedback: {feedback}")
        print(f"{'=' * 60}\n")

        # Get current graph summary
        graph_summary = self.get_graph_summary()

        system_role = """You are a senior GIS expert and geoprocessing workflow architect.

        You are refining an existing workflow graph based on user feedback.

        RULES:
        1. Modify the graph according to the user's feedback.
        2. Keep all existing valid nodes and edges unless the feedback says to remove them.
        3. Maintain graph connectivity - NO disconnected components.
        4. Node types: "data" or "operation"
        5. Node attributes: node_type, description, data_path (for data nodes only)
        6. ALL attribute values MUST be strings (not None, not empty)
        7. If an attribute would be None, use an empty string "" instead

        OUTPUT:
        - Output ONLY Python code that constructs the complete updated NetworkX graph.
        - Create a graph object named 'G'
        - Include ALL nodes and edges (both existing and new).
        - DO NOT include code to save the graph (no nx.write_graphml)
        - Enclose the reply in a Python code block: ```python ... ```
        """

        user_prompt = f"""## Original Task
        {self.task}

        ## Data Overview
        {self.data_overview}

        ## Current Graph Structure
        {graph_summary}

        ## User Feedback for Revision
        {feedback}

        ## Instructions
        Based on the user feedback, modify the graph accordingly and store it in a variable named 'G'.
        DO NOT save the graph to file - just create the NetworkX graph object.
        """

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_prompt}
        ]

        print("Generating refined graph based on feedback...")
        user_key = helper.load_OpenAI_key()
        if stream:
            response = helper.gibd_chat_completion_stream(user_key, self._llm_model("agent_solution.refine"), messages, **self._llm_reasoning("agent_solution.refine"))
        else:
            response = helper.gibd_chat_completion(user_key, self._llm_model("agent_solution.refine"), messages, **self._llm_reasoning("agent_solution.refine"))

        full_content = ""
        if stream:
            for chunk in response:
                if not chunk.choices:
                    continue
                chunk_content = chunk.choices[0].delta.content
                if chunk_content is not None:
                    full_content += chunk_content
                    print(chunk_content, end='', flush=True)

                    # Yield chunk for WebUI streaming display
                    if yield_chunks:
                        yield {
                            'type': 'llm_stream',
                            'content': chunk_content,
                            'step': 'graph_refinement'
                        }
                    # Send chunk to callback if provided (legacy support)
                    elif self.stream_callback:
                        self.stream_callback(chunk_content)
        else:
            full_content = response.choices[0].message.content
            print(full_content)

        print('\n\nGot LLM reply for revision.')

        self.graph_response = full_content

        try:
            self.code_for_graph = helper.extract_code(response_content=self.graph_response)
            self.execute_graph_code()
        except Exception as e:
            print(f"Failed to extract/execute refined graph code: {e}")
            import traceback
            traceback.print_exc()
            if not yield_chunks:
                return None

        if not yield_chunks:
            return self.graph_response

    def interactive_review(self, wait=False):
        """
        Interactive loop for human-in-the-loop graph refinement.
        Presents options to approve the graph or provide feedback for refinement.

        Args:
            wait (bool): If True, blocks execution until user clicks Approve.
                        If False, returns immediately (widgets work asynchronously).

        Usage in notebook:
            # Option 1: Automatic (recommended) - waits for approval
            solution.get_LLM_response_for_graph()
            graph, nt = solution.load_graph_file(interactive=True)  # Shows graph + waits for approval

            # Option 2: Manual control
            solution.get_LLM_response_for_graph()
            graph, nt = solution.load_graph_file(interactive=False)  # Just show graph
            solution.interactive_review(wait=True)  # Start review and wait
        """

        if WIDGETS_AVAILABLE:
            return self._interactive_review_widgets(wait=wait)
        else:
            return self._interactive_review_text()

    def _interactive_review_widgets(self, wait=False):
        """
        Widget-based interactive review for Jupyter notebooks.

        Args:
            wait (bool): If True, blocks execution until user clicks Approve.
        """
        import ipywidgets as widgets
        # Create output widget for displaying messages
        output = widgets.Output()

        # Create the main options buttons
        approve_btn = widgets.Button(
            description='Approve',
            button_style='success',
            icon='check',
            tooltip='Approve the current graph'
        )

        feedback_btn = widgets.Button(
            description='Provide Feedback',
            button_style='warning',
            icon='edit',
            tooltip='Provide feedback to refine the graph'
        )

        list_btn = widgets.Button(
            description='List Nodes',
            button_style='info',
            icon='list',
            tooltip='Show all nodes in the graph'
        )

        # Create feedback input area (initially hidden)
        feedback_text = widgets.Textarea(
            placeholder='Enter your feedback here...',
            description='Feedback:',
            layout=widgets.Layout(width='100%', height='100px')
        )

        submit_feedback_btn = widgets.Button(
            description='Submit Feedback',
            button_style='primary',
            icon='paper-plane'
        )

        cancel_btn = widgets.Button(
            description='Cancel',
            button_style='danger',
            icon='times'
        )

        # Container for feedback input (initially hidden)
        feedback_box = widgets.VBox(
            [feedback_text, widgets.HBox([submit_feedback_btn, cancel_btn])],
            layout=widgets.Layout(display='none')
        )

        # Main buttons container
        main_buttons = widgets.HBox([approve_btn, feedback_btn, list_btn])

        # Status label
        status_label = widgets.HTML(value='<h3>Review the graph above and select an option:</h3>')

        # Main container
        container = widgets.VBox([status_label, main_buttons, feedback_box, output])

        def on_approve_click(b):
            with output:
                clear_output()
                self.approve_graph()
                status_label.value = '<h3 style="color: green;">Graph Approved!</h3>'
                main_buttons.layout.display = 'none'
                feedback_box.layout.display = 'none'

        def on_feedback_click(b):
            # Show feedback input area
            feedback_box.layout.display = 'block'
            main_buttons.layout.display = 'none'
            status_label.value = '<h3>Enter your feedback below:</h3>'

        def on_list_click(b):
            with output:
                clear_output()
                self.list_nodes()

        def on_submit_feedback_click(b):
            feedback = feedback_text.value.strip()
            if not feedback:
                with output:
                    clear_output()
                    print("Please enter feedback before submitting.")
                return

            with output:
                clear_output()
                print(f"Processing feedback: {feedback}")

                # Refine the graph
                self.refine_graph(feedback)

                # Reload and display (interactive=False to avoid recursive review)
                print("\nLoading updated graph...")
                self.load_graph_file(interactive=False)

                print("\nGraph updated! Review the new graph above.")

            # Reset UI for next review cycle
            feedback_text.value = ''
            feedback_box.layout.display = 'none'
            main_buttons.layout.display = 'flex'
            status_label.value = '<h3>Review the updated graph and select an option:</h3>'

        def on_cancel_click(b):
            feedback_text.value = ''
            feedback_box.layout.display = 'none'
            main_buttons.layout.display = 'flex'
            status_label.value = '<h3>Review the graph above and select an option:</h3>'

        # Attach event handlers
        approve_btn.on_click(on_approve_click)
        feedback_btn.on_click(on_feedback_click)
        list_btn.on_click(on_list_click)
        submit_feedback_btn.on_click(on_submit_feedback_click)
        cancel_btn.on_click(on_cancel_click)

        # Display the widget
        display(container)

        # If wait=True, block until user approves
        if wait:
            import IPython
            print("Waiting for approval... (click 'Approve' button to continue)")
            while not self.is_approved:
                # Process UI events and wait
                IPython.display.display(IPython.display.Javascript(''))  # Force UI update
                time.sleep(0.5)
            print("Approval received! Continuing...")

        return self.is_approved

    def _interactive_review_text(self):
        """Text-based interactive review (fallback when widgets not available)."""
        print("\n" + "=" * 60)
        print("INTERACTIVE GRAPH REVIEW")
        print("=" * 60)
        print("Review the graph and select an option:")
        print("  1. approve  - Accept the current graph")
        print("  2. feedback - Provide feedback to refine the graph")
        print("  3. list     - Show all nodes in the graph")
        print("  4. quit     - Exit without approving")
        print("=" * 60 + "\n")

        while not self.is_approved:
            try:
                user_input = input("\nSelect option (approve/feedback/list/quit): ").strip().lower()
            except EOFError:
                print("Input stream closed. Exiting review.")
                break

            if not user_input:
                print("Please enter an option.")
                continue

            if user_input in ['approve', 'done', 'ok', 'yes', 'accept', '1']:
                self.approve_graph()
                break

            elif user_input in ['quit', 'exit', '4']:
                print("Exiting review without approval.")
                break

            elif user_input in ['list', '3']:
                self.list_nodes()
                continue

            elif user_input in ['feedback', '2']:
                # Collect feedback
                print("\nEnter your feedback (press Enter twice to submit):")
                feedback_lines = []
                while True:
                    try:
                        line = input()
                        if line == '':
                            if feedback_lines:
                                break
                        else:
                            feedback_lines.append(line)
                    except EOFError:
                        break

                feedback = '\n'.join(feedback_lines)
                if feedback.strip():
                    # Refine the graph
                    self.refine_graph(feedback)

                    # Reload and display (interactive=False to avoid recursive review)
                    print("\nLoading updated graph...")
                    self.load_graph_file(interactive=False)
                    print(f"\nGraph updated! View at: {self.graph_html_file}")
                else:
                    print("No feedback provided.")
                continue

            elif user_input == 'history':
                self.show_revision_history()
                continue

            else:
                print("Invalid option. Please select: approve, feedback, list, or quit")

        return self.is_approved

    def approve_graph(self):
        """Mark the current graph as approved and final."""
        self.is_approved = True
        print("\n" + "=" * 60)
        print("✅ GRAPH APPROVED")
        print(f"Total revisions made: {self.revision_count}")
        print(f"Final graph: {len(self.solution_graph.nodes())} nodes, {len(self.solution_graph.edges())} edges")
        print(f"Graph file: {self.graph_file}")
        print("=" * 60 + "\n")

    def show_revision_history(self):
        """Display the history of revisions made to the graph."""
        print("\n" + "=" * 60)
        print("📜 REVISION HISTORY")
        print("=" * 60)
        if not self.revision_history:
            print("No revisions made yet.")
        else:
            for rev in self.revision_history:
                print(f"\nRevision #{rev['revision']}:")
                print(f"  Feedback: {rev['feedback']}")
        print("=" * 60 + "\n")

    # ==================== End Human-in-the-Loop Methods ====================

    def show_graph(self, graph_file):
        if self.has_disconnected_components(directed_graph=graph_file):
            print("Disconnected component, please re-generate the graph!")

        nt = Network(notebook=True,
                     cdn_resources="remote",
                     directed=True,
                     # bgcolor="#222222",
                     # font_color="white",
                     height="800px",
                     # width="100%",
                     #  select_menu=True,
                     # filter_menu=True,
                     )

        nt.from_nx(graph_file)

        sinks = self.find_sink_node(G=graph_file)
        sources = self.find_source_node(G=graph_file)
        # print("sinks:", sinks)

        # Set node colors based on node type
        node_colors = []
        for node in nt.nodes:
            # print('node:', node)
            if node['node_type'] == 'data':
                # print('node:', node)
                if node['label'] in sinks:
                    node_colors.append('violet')  # lightgreen
                    # print(node)
                elif node['label'] in sources:
                    node_colors.append('lightgreen')  #
                    # print(node)
                else:
                    node_colors.append('orange')

            elif node['node_type'] == 'operation':
                node_colors.append('deepskyblue')

                # Update node colorsb
        for i, color in enumerate(node_colors):
            nt.nodes[i]['color'] = color
            # nt.nodes[i]['shape'] = 'box'
            nt.nodes[i]['shape'] = 'dot'
            # nt.set_node_style(node, shape="box")
            nt.nodes[i]['font'] = {'size': 20}  # set font size

        return nt


    def has_disconnected_components(self, directed_graph):
        # Get the weakly connected components
        weakly_connected = list(nx.weakly_connected_components(directed_graph))

        # Check if there is more than one weakly connected component
        if len(weakly_connected) > 1:
            print("component count:", len(weakly_connected))
            return True
        else:
            return False


    def find_sink_node(self, G):
        """
        Find the sink node in a NetworkX directed graph.

        :param G: A NetworkX directed graph
        :return: The sink node, or None if not found
        """
        sinks = []
        for node in G.nodes():
            if G.out_degree(node) == 0 and G.in_degree(node) > 0:
                sinks.append(node)
        return sinks

        # Function to find the source node

    def find_source_node(self, G):
        # Initialize an empty list to store potential source nodes
        source_nodes = []

        # Iterate over all nodes in the graph
        for node in G.nodes():
            # Check if the node has no incoming edges
            if G.in_degree(node) == 0:
                # Add the node to the list of source nodes
                source_nodes.append(node)

        # Return the source nodes
        return source_nodes




    def initial_operations(self):
        self.operations = []
        operation_names = self.operation_node_names
        for node_name in operation_names:
            function_def_returns = self.generate_function_def(node_name, self.solution_graph)
            self.operations.append(function_def_returns)

    def generate_function_def(self, node_name, G):
        """
        Construct a Python function signature and return line based on a node’s predecessors and successors in a graph.
        Args:
            node_name (str): Name of the node to convert into a function.
            G (networkx.DiGraph): Graph where predecessors are function inputs and successors are outputs.
        Returns:
            dict: Contains the function definition string, return line, node description, and node name.
        """

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
                # Don't add undefined default values - just the parameter name
                para_str = para_str + f"{para_name}, "

        all_para_str = para_str + para_default_str

        function_def = f'{node_name}({all_para_str})'
        function_def = function_def.replace(', )', ')')  # remove the last ","

        # generate the return line
        successors = G.successors(node_name)
        return_str = 'return ' + ', '.join(list(successors))

        # print("function_def:", function_def)  # , f"node_type:{node_type}"
        # print("return_str:", return_str)  # , f"node_type:{node_type}"
        # print(function_def, predecessors, successors)
        return_dict = {"function_definition": function_def,
                       "return_line": return_str,
                       'description': node_dict['description'],
                       'node_name': node_name
                       }
        return return_dict



    # def graph_prompt(self, task):
    #     graph_prompt = f"""
    #     You are an expert problem-solving system that converts tasks into execution graphs.
    #
    #     GOAL:
    #     Given a task, generate a directed graph where:
    #     - Each node represents a single, concrete execution step
    #     - Nodes are ordered as a logical sequence of actions
    #     - Edges represent dependency or flow between steps
    #
    #     GRAPH RULES:
    #     1. Each node must contain:
    #        - Step ID
    #        - Step title (short, action-oriented)
    #        - Step description (what is done in this step)
    #     2. Steps must be:
    #        - Atomic (no combined actions)
    #        - Consecutive and executable
    #        - Free of ambiguity
    #     3. The graph must:
    #        - Start with problem understanding
    #        - End with task completion or output validation
    #     4. Avoid meta-commentary, explanations, or advice outside the graph.
    #
    #     OUTPUT FORMAT:
    #     Return the graph in **Graphviz DOT format**. Save it at: {self.graph_file}
    #
    #     STYLE CONSTRAINTS:
    #     - Use concise language
    #     - No prose outside the graph
    #     - No assumptions beyond the task description
    #
    #
    #
    #     TASK:
    #     {task}
    #     """
    #     return graph_prompt

#%% =============================ASSEMBLY CODE FUNCTIONS ===============================================================

    def prompt_for_assembly_program(self):
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in self.operations])
        # operation_code = solution.operations[-1]['operation_code']
        # assembly_prompt = f"" + \

        assembly_requirement = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(solution_prompts.assembly_requirement)])

        assembly_prompt = f"Your role: {solution_prompts.assembly_role} \n\n" + \
                          f"Your task is: use the given Python functions, return a complete Python program to solve the question: \n {self.task}" + \
                          f"Requirement: \n {assembly_requirement} \n\n" + \
                          f"Data overview: \n  \n" + \
                          f"Code: \n {all_operation_code_str}"

        self.assembly_prompt = assembly_prompt
        return self.assembly_prompt



    def get_LLM_assembly_response(self, stream=True, yield_chunks=False):
        self.prompt_for_assembly_program()

        user_prompt = self.assembly_prompt
        system_prompt = solution_prompts.assembly_role
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        user_key = helper.load_OpenAI_key()
        if stream:
            response = helper.gibd_chat_completion_stream(user_key, self._llm_model("agent_solution.assembly"), messages, **self._llm_reasoning("agent_solution.assembly"))
        else:
            response = helper.gibd_chat_completion(user_key, self._llm_model("agent_solution.assembly"), messages, **self._llm_reasoning("agent_solution.assembly"))

        # Collect the full response content
        full_content = ""
        if stream:
            for chunk in response:
                if not chunk.choices:
                    continue
                chunk_content = chunk.choices[0].delta.content
                if chunk_content is not None:
                    full_content += chunk_content
                    print(chunk_content, end='', flush=True)
                    # Send chunk to WebUI if callback is provided
                    
                    # Yield chunk for WebUI streaming display
                    if yield_chunks:
                        yield {
                            'type': 'llm_stream',
                            'content': chunk_content,
                            'step': 'assembly_streaming'
                        }
                    # Send chunk to callback if provided (legacy support)
                    elif self.stream_callback:
                        self.stream_callback(chunk_content)
            print()  # New line after streaming completes 

        else:
            full_content = response.choices[0].message.content
            print(full_content)

        self.assembly_LLM_response = full_content

        try:
            code_for_assembly = helper.extract_code(response_content=self.assembly_LLM_response)
        except Exception as e:
            code_for_assembly = ""
            
        if not yield_chunks:
            return self.assembly_LLM_response




        self.code_for_assembly = code_for_assembly

        # if review:
        #     self.ask_LLM_to_review_assembly_code()

        return self.assembly_LLM_response




    def execute_complete_program(self, code: str, try_cnt: int = 5) -> str:

        count = 0
        while count < try_cnt:
            print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
            try:
                count += 1
                compiled_code = compile(code, 'Complete program', 'exec')
                exec(compiled_code, globals())  # #pass only globals() not locals()
                #!!!!    all variables in code will become global variables! May cause huge issues!     !!!!
                print("\n\n--------------- Done ---------------\n\n")
                return code


            except Exception as err:

                if count == try_cnt:
                    print(f"Failed to execute and debug the code within {try_cnt} times.")
                    return code

                debug_prompt = self.get_debug_prompt(exception=err, code=code)

                print("Sending error information to LLM for debugging...")
                messages = [{"role": "system", "content": solution_prompts.debug_role},
                            {"role": "user", "content": debug_prompt}]

                response = helper.gibd_chat_completion_stream(
                    helper.load_OpenAI_key(), self._llm_model("agent_solution.debug"), messages, **self._llm_reasoning("agent_solution.debug")
                )

                # Collect streamed response
                full_content = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        full_content += chunk_content
                        print(chunk_content, end='', flush=True)
                        # Send chunk to WebUI if callback is provided
                        if self.stream_callback:
                            self.stream_callback(chunk_content)
                print()  # New line after streaming completes

                # Extract code from the collected response
                code = helper.extract_code(response_content=full_content)

        return code


    def get_debug_prompt(self, exception, code):
        etype, exc, tb = sys.exc_info()
        exttb = traceback.extract_tb(tb)  # Do not quite understand this part.
        # https://stackoverflow.com/questions/39625465/how-do-i-retain-source-lines-in-tracebacks-when-running-dynamically-compiled-cod/39626362#39626362

        ## Fill the missing data:
        exttb2 = [(fn, lnnr, funcname,
                   (code.splitlines()[lnnr - 1] if fn == 'Complete program'
                    else line))
                  for fn, lnnr, funcname, line in exttb]

        # Print:
        error_info_str = 'Traceback (most recent call last):\n'
        for line in traceback.format_list(exttb2[1:]):
            error_info_str += line
        for line in traceback.format_exception_only(etype, exc):
            error_info_str += line

        print(f"Error_info_str: \n{error_info_str}")

        # print(f"traceback.format_exc():\n{traceback.format_exc()}")

        debug_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(solution_prompts.debug_requirement)])

        debug_prompt = f"Your role: {solution_prompts.debug_role} \n" + \
                          f"Your task: correct the code of a program according to the error information, then return the corrected and completed program. \n\n" + \
                          f"Requirement: \n {debug_requirement_str} \n\n" + \
                          f"The given code is used for this task: {self.task} \n\n" + \
                          f"The data location associated with the given code: \n {self.data_overview} \n\n" + \
                          f"The error information for the code is: \n{str(error_info_str)} \n\n" + \
                          f"The code is: \n{code}"

        return debug_prompt