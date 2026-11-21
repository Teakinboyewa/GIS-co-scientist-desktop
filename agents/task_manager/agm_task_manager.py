import json
import os
from typing import Dict, Any

from agents.base import Agent
import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning
from utils.agm_helper import get_ui_model, get_ui_reasoning_effort

api_key = helper.load_OpenAI_key()

# import task_solution_prompts as solution_prompts

# print(api_key)


# '''

from utils.lazy_imports import is_installed
# ipywidgets (notebook UI) is imported only where widgets are built.
WIDGETS_AVAILABLE = is_installed("ipywidgets")

class TaskManager(Agent):
    """
    Simple LLM-based router that assigns tasks to the most appropriate agent.

    Singleton: all TaskManager() calls return the same instance, so agents
    registered once are available everywhere (notebook, objective_executor, etc.).

    Responsibilities:
    1. Take research question and datasets
    2. Use LLM to determine which agent is most appropriate
    3. Route the task to that agent
    4. Return the result

    Note: DataAgent is responsible for generating the data overview from datasets.
    """
    name = "TaskManager"
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        self.api_key = api_key
        self.agent_registry = {}  # Will store available agents
        self._initialized = True
        # self._register_agents()

    def handle(self, ctx, task):
        """
        Main handler - routes the task to the most appropriate agent using LLM.

        Args:
            ctx: Context containing 'rq' (research question) and 'datasets'
            task: Task specification with optional params

        Returns:
            Dictionary with routing decision and agent result
        """
        self.log(ctx, "Starting task routing...")

        # Extract inputs
        rq = ctx.get("rq", "")
        datasets = ctx.get("datasets", [])

        if not rq:
            self.log(ctx, "ERROR: No research question provided")
            return {"error": "Research question is required", "selected_agent": None}

        self.log(ctx, f"Research question: {rq}")
        self.log(ctx, f"Datasets provided: {len(datasets)}")

        # Route to appropriate agent
        routing_decision = self._init_route_to_agent(ctx, rq, datasets)

        if not routing_decision or "error" in routing_decision:
            self.log(ctx, "ERROR: Failed to determine appropriate agent")
            return {"error": routing_decision.get("error", "Routing failed"), "selected_agent": None}

        selected_agent_name = routing_decision["agent"]
        reasoning = routing_decision.get("reasoning", "")
        task_params = routing_decision.get("params", {})

        self.log(ctx, f"Routing to: {selected_agent_name}")
        self.log(ctx, f"Reasoning: {reasoning}")

        # Get the agent instance
        agent_info = self.agent_registry.get(selected_agent_name)

        if not agent_info:
            error_msg = f"Agent '{selected_agent_name}' not registered"
            self.log(ctx, f"ERROR: {error_msg}")
            return {
                "error": error_msg,
                "selected_agent": selected_agent_name,
                "reasoning": reasoning
            }

        # Execute the task with the selected agent
        try:
            agent_instance = agent_info["instance"]
            task_spec = {"name": routing_decision.get("task", "handle"), "params": task_params}

            self.log(ctx, f"Executing {selected_agent_name}...")
            result = agent_instance.handle(ctx, task_spec)

            self.log(ctx, f"{selected_agent_name} completed successfully")

            return {
                "selected_agent": selected_agent_name,
                "reasoning": reasoning,
                "task": task_spec,
                "result": result,
                "success": True
            }

        except Exception as e:
            error_msg = f"Error executing {selected_agent_name}: {str(e)}"
            self.log(ctx, f"ERROR: {error_msg}")
            return {
                "selected_agent": selected_agent_name,
                "reasoning": reasoning,
                "error": error_msg,
                "success": False
            }

    def register_agent(self, agent_name: str, agent_instance, description: str = ""):
        """
        Register specialized agents that can be orchestrated.

        Args:
            agent_name: Name of the agent
            agent_instance: Instance of the agent
            description: Description of agent's capabilities (used for LLM routing)
        """


        self.agent_registry[agent_name] = {
            "instance": agent_instance,
            "description": description
        }


    # def register_agent(self, agent_name: str, agent_instance, description: str = "", ctx=None):
    #     """
    #     Register specialized agents that can be orchestrated.
    #
    #     Args:
    #         agent_name: Name of the agent
    #         agent_instance: Instance of the agent
    #         description: Description of agent's capabilities (used for LLM routing)
    #     """
    #     if ctx is None:
    #         ctx = {}
    #
    #     self.agent_registry[agent_name] = {
    #         "instance": agent_instance,
    #         "description": description,
    #         "ctx": ctx
    #     }
    #     self.log(ctx, f"Registered agent - {agent_name}")

    def generate_research_plan(self, ctx):
        """
        Generate a comprehensive research plan based on objectives, hypotheses, and data context.

        The research plan is built from:
        - Core: Research objectives and hypotheses (from RQUnderstandingAgent)
        - Context: Data overview providing feasibility context (from DataAgent)

        Args:
            ctx: Context containing artifacts from previous agents

        Returns:
            Dictionary with research plan and path to saved plan
        """
        self.log(ctx, "Starting research plan generation...")

        # Check prerequisites
        artifacts = ctx.get("artifacts", {})

        if "rq_parsed_data" not in artifacts:
            self.log(ctx, "ERROR: Parsed RQ not found. Run RQUnderstandingAgent first.")
            return {"error": "Parsed research question required", "success": False}

        if "data_audit_records" not in artifacts:
            self.log(ctx, "WARNING: Data audit not found. Plan will lack data context.")

        # Get the artifacts
        rq_parsed = artifacts["rq_parsed_data"]
        data_audit = artifacts.get("data_audit_records", [])
        rq = ctx.get("rq", "")

        self.log(ctx, "Generating research plan based on objectives and hypotheses...")

        # Generate plan using LLM
        research_plan = self._create_research_plan_from_objectives(ctx, rq, rq_parsed, data_audit)

        if "error" in research_plan:
            self.log(ctx, f"ERROR: Failed to generate research plan - {research_plan['error']}")
            return research_plan

        # Save plan
        plan_path = self._save_research_plan(ctx, research_plan)

        # Store in artifacts
        artifacts["research_plan"] = plan_path
        artifacts["research_plan_data"] = research_plan

        self.log(ctx, f"Research plan generated successfully → {plan_path}")

        result = {
            "research_plan_path": plan_path,
            "research_plan": research_plan,
            "success": True
        }

        # Add visualization info if available
        if 'research_plan_viz_id' in ctx:
            result['viz_id'] = ctx['research_plan_viz_id']
            result['viz_url'] = f"/api/research-plan/{ctx['research_plan_viz_id']}"

        return result



    def _init_route_to_agent(self, ctx: Dict[str, Any], rq: str, datasets: list) -> Dict[str, Any]:
        """
        Use LLM to determine which agent should handle the task.

        Args:
            ctx: Context dictionary
            rq: Research question
            datasets: List of available datasets

        Returns:
            Dictionary with agent name, task, params, and reasoning
        """
        try:
            self.log(ctx, "Using LLM to route task...")

            # Prepare agent descriptions
            available_agents = []
            for agent_name, agent_info in self.agent_registry.items():
                available_agents.append({
                    "name": agent_name,
                    "description": agent_info.get("description", "No description provided")
                })

            # Check what has already been done
            artifacts = ctx.get("artifacts", {})
            rq_parsed = "rq_parsed" in artifacts or "rq_parsed_data" in artifacts
            data_audited = "data_audit" in artifacts or "data_audit_records" in artifacts

            # Create prompt
            prompt = f"""You are a task router for a multi-agent geospatial modeling system.

Research Question: {rq}

Available Datasets: {json.dumps(datasets, indent=2) if datasets else "No datasets provided"}

Current Workflow State:
- Research question parsed: {"Yes" if rq_parsed else "No"}
- Data audited: {"Yes" if data_audited else "No"}

Available Agents:
{json.dumps(available_agents, indent=2)}

Your task: Analyze the research question, available datasets, and current workflow state, then decide which agent is most appropriate to handle this task.

Return ONLY valid JSON in this exact format:
{{
  "agent": "AgentName",
  "task": "task_name",
  "params": {{}},
  "reasoning": "Brief explanation of why this agent is most appropriate"
}}

CRITICAL WORKFLOW RULES (Priority Order):
1. If data has NOT been audited yet → Choose DataAgent
   - This must be done FIRST to validate data, harmonize CRS, generate data overview
   - The data overview will inform subsequent agents about available fields and variables

2. If data has been audited but research question has NOT been parsed yet → Choose RQUnderstandingAgent
   - This must be done SECOND to understand objectives, variables, hypotheses, study units, spatial extent, and time frame
   - RQUnderstandingAgent will use the data overview to suggest appropriate variables/fields from the actual data

3. If both data is audited AND RQ is parsed:
   - If modeling/analysis needs to be performed → Choose ModelingAgent
   - If model results exist and need interpretation → Choose ResultInterpretationAgent
   - If final report is needed → Choose ReportingAgent

- Always provide clear reasoning for your choice based on the current workflow state
"""

            # Call GIBD API
            messages = [
                {"role": "system", "content": "You are a task routing assistant. Output only valid JSON."},
                {"role": "user", "content": prompt}]
            
            response = helper.client_chat_completion(
                messages = messages,
                model = llm_model("task_manager.init_route"),
                
                response_format={"type": "json_object"}, **llm_reasoning("task_manager.init_route"),
            )

            # Parse response
            content = response.choices[0].message.content
            routing_data = json.loads(content)

            self.log(ctx, f"LLM selected: {routing_data.get('agent')}")

            return routing_data

        except Exception as e:
            self.log(ctx, f"ERROR in LLM routing: {str(e)}")
            return {"error": str(e)}

    def _create_research_plan_from_objectives(self, ctx: Dict[str, Any], rq: str,
                                               rq_parsed: Dict[str, Any],
                                               data_audit: list) -> Dict[str, Any]:
        """
        Create research plan using LLM, centered on objectives and hypotheses.

        Args:
            ctx: Context dictionary
            rq: Original research question
            rq_parsed: Parsed RQ with objectives, hypotheses, variables
            data_audit: Data audit records for context

        Returns:
            Dictionary with structured research plan
        """
        try:
            self.log(ctx, "Using LLM to create research plan...")

            # Extract key components from parsed RQ
            objectives = rq_parsed.get("objectives", [])
            hypotheses = rq_parsed.get("hypotheses", [])
            variables = rq_parsed.get("variables", {})
            study_units = rq_parsed.get("study_units", [])
            spatial_extent = rq_parsed.get("spatial_extent", "")
            time_frame = rq_parsed.get("time_frame", "")
            methods_suggested = rq_parsed.get("methods_suggested", [])

            # Extract data context
            available_datasets = len(data_audit)
            total_records = sum([d.get("rows", 0) for d in data_audit if d.get("ok")])

            SYSTEM_PROMPT = """You are an expert research planner for geospatial analysis.
Create a comprehensive, actionable research plan based on each research objectives and hypotheses.

Your plan should be structured as JSON with these sections:

1. research_breakdown: Brief summary of the research (2-3 sentences)

2. objectives: List of research objectives (from input)

3. hypotheses: List of testable hypotheses (from input)

4. methodology (for each objectives): Dictionary with keys "objective_1", "objective_2", etc. for each objective, containing:
   - analytical_approach: Overall analysis strategy
   - spatial_analysis_methods: Specific spatial methods to use
   - statistical_methods: Statistical tests/models to apply
   - variables_mapping: Map conceptual variables to data fields

5. data_requirements (for each objectives): Dictionary with keys "objective_1", "objective_2", etc. for each objective, containing:
   - primary_variables: Key variables needed
   - control_variables: Control/confounding variables
   - spatial_identifiers: Geographic keys (e.g., GEOID, tract)
   - temporal_scope: Time period coverage

6. analysis_steps (for each objectives): Dictionary with keys "objective_1", "objective_2", etc. for each objective, containing ordered list of analysis steps with:
   - step_number
   - description
   - expected_output
   - dependencies (what must be done before this)

7. expected_outcomes: What results/insights are expected

8. limitations_assumptions: Key limitations and assumptions

Rules:
- Center the plan around the OBJECTIVES and HYPOTHESES
- Methodology, data requirements, and analysis steps should be keyed by "objective_1", "objective_2", "objective_3", etc.
- Use data context to ensure feasibility
- Be specific and actionable
- Return ONLY valid JSON
"""

            USER_PROMPT = f"""Original Research Question:
{rq}

Research Objectives:
{json.dumps(objectives, indent=2)}

Hypotheses to Test:
{json.dumps(hypotheses, indent=2)}

Variables Identified:
{json.dumps(variables, indent=2)}

Study Context:
- Study Units: {study_units}
- Spatial Extent: {spatial_extent}
- Time Frame: {time_frame}
- Suggested Methods: {methods_suggested}

Data Context (for feasibility):
- Available Datasets: {available_datasets}
- Total Records: {total_records}
- Data Quality: {"Good" if all(d.get("ok") for d in data_audit) else "Mixed"}

Create a comprehensive research plan centered on achieving each of the objectives and testing the hypotheses.
The plan should be actionable and account for the available data.
"""

            # Call GIBD API
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}]
            
            response = helper.client_chat_completion(
                messages = messages,
                model=llm_model("task_manager.plan_from_objectives"),
                
                response_format={"type": "json_object"}, **llm_reasoning("task_manager.plan_from_objectives"),
            )

            # Parse response
            content = response.choices[0].message.content
            plan_data = json.loads(content)

            self.log(ctx, f"Research plan created with {len(plan_data.get('analysis_steps', []))} analysis steps")

            return plan_data

        except Exception as e:
            self.log(ctx, f"ERROR in research plan creation: {str(e)}")
            return {"error": str(e)}

    def _save_research_plan(self, ctx: Dict[str, Any], plan_data: Dict[str, Any]) -> str:
        """Save research plan to file and generate visualization."""
        import os
        from utils.config import resolve_output_dir
        from utils.research_plan_viz import save_research_plan_visualization

        tm_out_dir = resolve_output_dir(ctx)
        os.makedirs(tm_out_dir, exist_ok=True)
        plan_path = os.path.join(tm_out_dir, "research_plan.json")

        # Save JSON version
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)

        self.log(ctx, f"Research plan saved to {plan_path}")

        # Generate and save visualization
        try:
            task_id = ctx.get('task_id', 'default')
            viz_path = save_research_plan_visualization(plan_data, tm_out_dir, task_id)
            self.log(ctx, f"Research plan visualization saved to {viz_path}")

            # Store visualization info in context for frontend
            ctx['research_plan_viz_path'] = viz_path
            ctx['research_plan_viz_id'] = task_id
        except Exception as e:
            self.log(ctx, f"Warning: Could not generate research plan visualization: {str(e)}")

        return plan_path


    
    



    def execute_research_plan(self, ctx):
        """
        Execute the entire research plan by executing each objective sequentially.

        Args:
            ctx: Context containing artifacts with research_plan_data

        Returns:
            Dictionary with execution results for all objectives
        """
        print("=" * 70)
        self.log(ctx, "Starting Research Plan Execution")
        print("=" * 70)


        # Check prerequisites
        artifacts = ctx.get("artifacts", {})

        if "research_plan_data" not in artifacts:
            self.log(ctx, "ERROR: Research plan not found. Generate plan first.")
            return {"error": "Research plan required", "success": False}

        research_plan = artifacts["research_plan_data"]
        objectives = research_plan.get("objectives", [])

        if not objectives:
            self.log(ctx, "ERROR: No objectives found in research plan")
            return {"error": "No objectives to execute", "success": False}

        self.log(ctx, f"Found {len(objectives)} objectives to execute")

        # Execute each objective
        for i, objective in enumerate(objectives, 1):
            print(f"\n{'=' * 70}")
            self.log(ctx, f"Objective {i}/{len(objectives)}: {objective}")
            # print(f"{'=' * 70}")
            self.execute_objective(ctx, objective, research_plan)

            # result = self.execute_objective(ctx, objective, research_plan)
        #
        #     execution_summary["results_by_objective"][objective] = result
        #
        #     if result.get("success"):
        #         execution_summary["objectives_completed"] += 1
        #         execution_summary["total_steps_executed"] += result.get("steps_completed", 0)
        #         self.log(ctx, f"✓ Objective {i} completed successfully")
        #     else:
        #         execution_summary["objectives_failed"] += 1
        #         self.log(ctx, f"✗ Objective {i} failed: {result.get('error', 'Unknown error')}")
        #
        # # Final summary
        # self.log(ctx, f"\n{'=' * 70}")
        # self.log(ctx, "Research Plan Execution Summary")
        # self.log(ctx, f"{'=' * 70}")
        # self.log(ctx, f"Objectives Completed: {execution_summary['objectives_completed']}/{len(objectives)}")
        # self.log(ctx, f"Total Steps Executed: {execution_summary['total_steps_executed']}")
        # self.log(ctx, f"{'=' * 70}")
        #
        # # Store summary in artifacts
        # artifacts["execution_summary"] = execution_summary
        #
        # return {
        #     "success": execution_summary["objectives_failed"] == 0,
        #     "execution_summary": execution_summary
        # }


    def extract_research_object_for_objective(self, ctx, research_plan):
        """Extract the research object for a given objective from the research plan."""

        research_object = {}
        objectives = research_plan.get("objectives", [])
        for obj in objectives:
            num = obj.get("number", 1)
            obj_key = f"objective_{num}"

            # Convert steps to dicts for downstream compatibility
            # Handles both old format (plain strings) and new format (dicts with depends_on)
            steps = []
            for i, step_raw in enumerate(obj.get("steps", []), start=1):
                if isinstance(step_raw, dict):
                    steps.append({
                        "step_number": step_raw.get("step_number", i),
                        "description": step_raw.get("description", ""),
                        "expected_output": step_raw.get("expected_output", ""),
                        "dependencies": step_raw.get("depends_on", [])
                    })
                else:
                    steps.append({
                        "step_number": i,
                        "description": step_raw,
                        "expected_output": "",
                        "dependencies": []
                    })

            research_object[obj_key] = {
                "title": obj.get("title", ""),
                "hypothesis": obj.get("hypothesis", ""),
                "steps": steps,
                "suggested_variables": obj.get("primary_variables", []),
            }

        self.log(ctx, f"Found {len(objectives)} objectives")
        return research_object
    





          


    def _route_step_to_agent(self, ctx, step):
        """
        Use LLM to determine which agent should handle a specific analysis step.

        Uses keyword matching and methodology context to route steps to appropriate agents.

        Args:
            ctx: Context dictionary
            step: Step dictionary with description, expected_output, etc.
            methodology: Methodology dict for this objective (provides context)

        Returns:
            Dictionary with agent name and reasoning
        """
        description = step.get("description", "").lower()
        expected_output = step.get("expected_output", "").lower()

        # Combine for better matching
        step_text = f"{description} {expected_output}"
        print(step_text)

        try:
            self.log(ctx, "Using LLM to route task...")

            # Prepare agent descriptions
            available_agents = []
            # print("agent_registry:", self.agent_registry)
            # print("agent_registry keys:", list(self.agent_registry.keys()) if self.agent_registry else "EMPTY")

            for agent_name, agent_info in self.agent_registry.items():
                available_agents.append({
                    "name": agent_name,
                    "description": agent_info.get("description", "No description provided")
                })


            prompt = f""" You are a task router whose role for for a multi-agent spatial modeling system. 
             
                 Step: {step_text}
                 
                 Available Agents:
                 {json.dumps(available_agents, indent=2)}
                 
                 Your task: Analyze the objective step description, and decide which agent is most appropriate to handle this objective step.
                 
                 Return ONLY valid JSON in this exact format:
                {{
                  "agent": "AgentName",
                  "task": "task_name",
                  "params": {{}},
                  "reasoning": "Brief explanation of why this agent is most appropriate"
                }}
                """
            # Call GIBD API
            messages = [
                {"role": "system", "content": "You are a task routing assistant. Output only valid JSON."},
                {"role": "user", "content": prompt}]
            response = helper.client_chat_completion(
                messages=messages,
                model=llm_model("task_manager.route_step"),
                
                response_format={"type": "json_object"}, **llm_reasoning("task_manager.route_step"),
            )

            # Parse response
            content = response.choices[0].message.content
            routing_data = json.loads(content)
            #
            self.log(ctx, f"LLM selected: {routing_data.get('agent')}")
            #
            # print(routing_data)
            return routing_data


        except Exception as e:
            self.log(ctx, f"ERROR in routing task: {str(e)}")
            return {"error": str(e)}







    def execute_all_codes(self, try_cnt=10):
        from utils.config import load_configurations
        settings = load_configurations()
        results_path = os.path.join(settings.output_dir, "modeling_results.json")
        print(results_path)
        # Open and load the JSON file
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Organize codes by objectives, sorted by step number
        objectives_organized = {}

        for objective_key, steps_dict in data.items():
            # Sort steps by step_number
            sorted_steps = sorted(steps_dict.items(),
                                  key=lambda x: int(x[1].get('step_number', 0)))

            objectives_organized[objective_key] = sorted_steps

            # print(f"\n{objective_key}:")
            for step_key, step_data in sorted_steps:
                step_num = step_data.get('step_number', 'N/A')
                agent = step_data.get('agent', 'N/A')
                # print(f"  Step {step_num} (Agent: {agent})")
                # Create combined code for each objective (organized by step)
                combined_codes = {}

                for objective_key, sorted_steps in objectives_organized.items():
                    # Combine all python_code for this objective in step order
                    combined_code = f"""# {objective_key.replace('_', ' ').title()} - Combined Code
                # This file contains all code for {objective_key}, organized by steps

                """

                    for step_key, step_data in sorted_steps:
                        step_num = step_data.get('step_number', 'N/A')
                        agent = step_data.get('agent', 'N/A')
                        python_code = step_data.get('python_code', '')

                        combined_code += f"\n{'#' + '=' * 80}\n"
                        combined_code += f"# STEP {step_num} - Agent: {agent}\n"
                        combined_code += f"{'#' + '=' * 80}\n\n"
                        combined_code += python_code
                        combined_code += f"\n\n"

                    combined_codes[objective_key] = combined_code


                print(f"✓ Created combined code for {len(combined_codes)} objectives")

                print("[EXECUTOR]: Executing operation code for {objective_key}...")

                count = 0
                while count < try_cnt:
                    print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
                    try:
                        count += 1
                        compiled_code = compile(combined_code, 'Complete program', 'exec')
                        exec(compiled_code, globals())  # #pass only globals() not locals()
                        # !!!!    all variables in code will become global variables! May cause huge issues!     !!!!
                        print("\n\n--------------- Done ---------------\n\n")
                        return combined_code

                    except Exception as err:

                        if count == try_cnt:
                            print(f"Failed to execute and debug the code within {try_cnt} times.")
                            return combined_code

                        # debug_prompt = self.get_debug_prompt(exception=err, code=code)
                        # print("Sending error information to LLM for debugging...")
                        # # print("Prompt:\n", debug_prompt)
                        # response = helper.get_LLM_reply(prompt=debug_prompt,
                        #                                 system_role=constants.debug_role,
                        #                                 model=self.model,
                        #                                 verbose=True,
                        #                                 stream=True,
                        #                                 retry_cnt=5,
                        #                                 )
                        # code = helper.extract_code(response)

                return combined_code









