import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import utils.agm_helper as helper
from utils.llm_models import llm_model, llm_reasoning
from utils.agm_helper import get_ui_model, get_ui_reasoning_effort
import utils.research_plan_viz as research_plan_viz
from agents.data_agent import DataAgent
from agents.rq_understanding_agent import RQUnderstandingAgent
from agents.modeling_agent import ModelingAgent
import utils.prompts as prompts
from agents.task_manager import TaskManager, research_plan_generator

task_manager = TaskManager()
data_agent = DataAgent()
rq_agent = RQUnderstandingAgent()
modeling_agent = ModelingAgent()

# ===== Workflow State Management =====
import json
import os
from utils.config import load_configurations
settings = load_configurations()

# Global workflow state storage (also persisted to disk)
workflow_states = {}


def _get_task_output_dir(task_id):
    """Get per-task output directory: Outputs/<task_id>/"""
    settings = load_configurations()
    task_dir = os.path.join(settings.output_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def _get_state_dir(task_id=None):
    """Per-task state directory: Outputs/<task_id>/.

    If task_id is not provided, returns the root outputs dir so callers
    iterating over states can scan per-task subdirectories.
    """
    settings = load_configurations()
    if task_id:
        state_dir = os.path.join(settings.output_dir, task_id)
    else:
        state_dir = settings.output_dir
    os.makedirs(state_dir, exist_ok=True)
    return state_dir

def _get_state_file(task_id):
    """Per-task state file: Outputs/<task_id>/workflow_state.json.

    For reads, falls back to the legacy Outputs/workflow_states/<task_id>.json
    location so previously saved states remain accessible. Writes should use
    the new path via _get_state_write_file().
    """
    new_path = os.path.join(_get_state_dir(task_id), "workflow_state.json")
    if not os.path.exists(new_path):
        settings = load_configurations()
        legacy = os.path.join(settings.output_dir, "workflow_states", f"{task_id}.json")
        if os.path.exists(legacy):
            return legacy
    return new_path

def _get_state_write_file(task_id):
    """Per-task state write path — always the new location."""
    return os.path.join(_get_state_dir(task_id), "workflow_state.json")

def _serialize_state(state):
    """Serialize state for JSON storage, handling non-serializable objects"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, date
    
    def convert(obj):
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        else:
            # For complex objects, store class name as placeholder
            return f"<{type(obj).__name__}>"
    
    return convert(state)

def save_workflow_state(task_id, state):
    """Save workflow state for continuation.

    Always writes to Outputs/<task_id>/workflow_state.json (never the legacy
    global workflow_states/ directory).
    """
    # Save to memory for fast access
    workflow_states[task_id] = state
    # Save to disk for persistence
    try:
        state_file = _get_state_write_file(task_id)
        serializable_state = _serialize_state(state)
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
        print(f"[State] Saved workflow state to {state_file}")
    except Exception as e:
        print(f"[State] Warning: Could not save state to disk: {e}")

    return state

def get_workflow_state(task_id):
    """Retrieve workflow state from memory or disk"""
    # Try memory first (fastest)
    if task_id in workflow_states:
        return workflow_states[task_id]
    
    # Try loading from disk
    try:
        state_file = _get_state_file(task_id)
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Cache in memory
            workflow_states[task_id] = state
            print(f"[State] Loaded workflow state from {state_file}")
            return state
    except Exception as e:
        print(f"[State] Warning: Could not load state from disk: {e}")
    
    return None



def clear_workflow_state(task_id):
    """Clear workflow state from both memory and disk (new + legacy paths)."""
    # Clear from memory
    if task_id in workflow_states:
        del workflow_states[task_id]

    # Delete from disk — both new per-task path and legacy global path
    settings = load_configurations()
    candidates = [
        os.path.join(settings.output_dir, task_id, "workflow_state.json"),
        os.path.join(settings.output_dir, "workflow_states", f"{task_id}.json"),
    ]
    for state_file in candidates:
        try:
            if os.path.exists(state_file):
                os.remove(state_file)
                print(f"[State] Deleted workflow state file {state_file}")
        except Exception as e:
            print(f"[State] Warning: Could not delete state file {state_file}: {e}")

def list_workflow_states():
    """List all task IDs that have a workflow state on disk.

    Scans per-task directories (Outputs/<task_id>/workflow_state.json) and
    includes any legacy global state files so older workflows remain visible.
    """
    settings = load_configurations()
    task_ids = set()
    try:
        root = settings.output_dir
        if os.path.isdir(root):
            for entry in os.listdir(root):
                entry_path = os.path.join(root, entry)
                if os.path.isdir(entry_path) and os.path.exists(
                    os.path.join(entry_path, "workflow_state.json")
                ):
                    task_ids.add(entry)
    except Exception as e:
        print(f"[State] Warning: Could not scan per-task state dirs: {e}")
    try:
        legacy_dir = os.path.join(settings.output_dir, "workflow_states")
        if os.path.isdir(legacy_dir):
            for f in os.listdir(legacy_dir):
                if f.endswith('.json'):
                    task_ids.add(f[:-5])
    except Exception as e:
        print(f"[State] Warning: Could not scan legacy state dir: {e}")
    return sorted(task_ids)
    
def cleanup_old_states(days=7):
    """Delete workflow state files older than specified days.

    Scans both the new per-task location (Outputs/<task_id>/workflow_state.json)
    and the legacy global directory (Outputs/workflow_states/<task_id>.json).
    """
    import time

    settings = load_configurations()
    current_time = time.time()
    deleted = 0

    # New per-task layout
    try:
        root = settings.output_dir
        if os.path.isdir(root):
            for entry in os.listdir(root):
                entry_path = os.path.join(root, entry)
                state_file = os.path.join(entry_path, "workflow_state.json")
                if os.path.isdir(entry_path) and os.path.exists(state_file):
                    file_age_days = (current_time - os.path.getmtime(state_file)) / 86400
                    if file_age_days > days:
                        os.remove(state_file)
                        if entry in workflow_states:
                            del workflow_states[entry]
                        deleted += 1
    except Exception as e:
        print(f"[State] Warning: Error during per-task cleanup: {e}")

    # Legacy global layout
    try:
        legacy_dir = os.path.join(settings.output_dir, "workflow_states")
        if os.path.isdir(legacy_dir):
            for filename in os.listdir(legacy_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(legacy_dir, filename)
                    file_age_days = (current_time - os.path.getmtime(filepath)) / 86400
                    if file_age_days > days:
                        os.remove(filepath)
                        task_id = filename[:-5]
                        if task_id in workflow_states:
                            del workflow_states[task_id]
                        deleted += 1
    except Exception as e:
        print(f"[State] Warning: Error during legacy cleanup: {e}")

    if deleted > 0:
        print(f"[State] Cleaned up {deleted} old workflow state(s)")
    
    

model = None  # Set dynamically via get_ui_model()
chat_model = None  # Set dynamically via get_ui_model()



# Register Available Agents with descriptions
def register_agent():
    """Register all agents with optional context for logging"""

    # Clear and re-register agents to ensure fresh registration
    # task_manager.agent_registry = {}

    task_manager.register_agent(
        "RQUnderstandingAgent", RQUnderstandingAgent(), description="Interprets user-defined research questions and translates them into operational objectives, variables, hypotheses, study units, spatial extent, and time frame.")

    task_manager.register_agent("DataAgent",     DataAgent(), description="Generates data overview by auditing datasets. Validates data inputs, harmonizes CRS, resolves keys (GEOID, CBG, tract), generates metadata & data dictionary. Performs exploratory data analysis (EDA).")

    # Register ModelingAgent - uses FAISS-powered semantic search to select top 3 models
    task_manager.register_agent("ModelingAgent", ModelingAgent(), description="Selects, configures, and executes appropriate statistical and machine learning models from the system's model base using intelligent model selection based on task requirements."

        # Note: As you implement other agents, register them here:
        #     # task_manager.register_agent("SpatialAnalysisAgent", SpatialAnalysisAgent(), description="Performs spatial analysis, GIS mapping, and geographically weighted regression.")
        #     # task_manager.register_agent("ResultInterpretationAgent", ResultInterpretationAgent(), description="Synthesizes model outputs, analyzes and explains model results in geographic and theoretical context.")
        #     # task_manager.register_agent("ReportingAgent", ReportingAgent(), description="Synthesizes the full analytical process into a structured scientific report.")

    )



def audit_uploaded_data(file_paths, file_names, ctx=None):
    """
    Audit uploaded data files and generate data overview
    Args:
        file_paths: List of file paths to uploaded data files
        file_names: List of original file names
    Returns:
        List of data overview dictionaries (audit records)
    """
    from agents.data_agent.agm_data_agent import DataAgent

    # Use provided context for logging if available, otherwise create a new one
    if ctx is None:
        ctx = {}

    # Prepare datasets list for DataAgent
    datasets = []
    for path, name in zip(file_paths, file_names):
        datasets.append({
            "path": path,
            "name": name,
        })

    # Create context with datasets
    ctx["datasets"] = datasets


    # Initialize DataAgent and run audit
    data_agent = DataAgent()
    result = data_agent.audit_datasets(ctx, {"params": {}})

    # Return the audit records (data_overview)
    return result.get("audit_records", [])


def rq_parsing(research_question, data_overview=None, ctx=None):
    """
    Parse and understand the research question using RQUnderstandingAgent
    
    Args:
        research_question: The research question text
        data_overview: Optional data overview from audit (for context)
        ctx: Context dict for logging (optional)
    
    Returns:
        Dictionary with parsed RQ data and file path
    """
    from agents.rq_understanding_agent import RQUnderstandingAgent
    
    # Use provided context or create new one
    if ctx is None:
        ctx = {}
    
    # Set up context with research question
    ctx["rq"] = research_question
    
    # Add data overview to artifacts if available
    if data_overview:
        ctx.setdefault("artifacts", {})["data_audit_records"] = data_overview
    
    # Initialize RQUnderstandingAgent and parse
    rq_agent = RQUnderstandingAgent()
    result = rq_agent.rq_parsing(ctx)
    
    # Return the parsed data
    return result

def chat_with_ai(message):
    """
    Chat with an AI agent
    Args:
        message: The message to send to the AI agent
    Returns:
        The response from the AI agent
    """
    
    user_key = helper.load_OpenAI_key()
    response = helper.gibd_chat_completion(user_key, llm_model("legacy.chat"), [{"role": "user", "content": message}], **llm_reasoning("legacy.chat"))
    return response.choices[0].message.content


def detect_user_intent(message):
    """
    Detect whether the user wants to perform a task or just chat
    Args:
        message: The user's message
    Returns:
        Dictionary with intent classification:
        {
            "intent": "task" or "chat",
            "confidence": float (0-1),
            "reasoning": string
        }
    """


    system_prompt = prompts.user_intent_prompts

    user_key = helper.load_OpenAI_key()
    response = helper.gibd_chat_completion(
        user_key, llm_model("intent.detect"),
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": message}], **llm_reasoning("intent.detect")
    )

    result_text = response.choices[0].message.content.strip()

    # Parse the JSON response
    try:
        result = json.loads(result_text)
        return result
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "intent": "chat",
            "confidence": 0.5,
            "reasoning": "Could not parse intent, defaulting to chat"
        }

#=================================== REAL WORKFLOW STARTS ==========================================
def smart_message_handler_stream(message, file_paths=None, file_names=None, active_solutions=None):
    """
    Streaming version of smart_message_handler that yields progress updates
    Args:
        message: User's message
        file_paths: List of uploaded file paths (optional)
        file_names: List of uploaded file names (optional)
        active_solutions: Dictionary of active solutions (optional)
    Yields:
        Dictionary with update information for streaming to frontend
    """

    # Yield initial status
    yield {
        'type': 'llm_update',
        'content': '🔍 Analyzing your message...',
        'step': 'intent_detection'
    }

    # Detect user intent
    print(f"\n[DEBUG] smart_message_handler_stream called")
    print(f"[DEBUG] Message: {message}")
    print(f"[DEBUG] Files uploaded: {len(file_paths) if file_paths else 0}")

    intent_result = detect_user_intent(message)
    print(f"[DEBUG] Intent detected: {intent_result}")

    yield {
        'type': 'llm_update',
        'content': f'📊 Intent detected: {intent_result["intent"].upper()} (confidence: {intent_result["confidence"]})',
        'step': 'intent_detected',
        'intent': intent_result
    }



    # If it's just chat, respond normally
    if intent_result["intent"] == "chat":
        print(f"[DEBUG] Routing to: CHAT")

        # Use yield from to propagate the generator yields to the WebUI
        yield from just_chat_response(message, intent_result, stream=True)
        return  # Exit after handling chat
    

    # If it's a research question or task but no files uploaded
    if intent_result["intent"] in ["task", "research"] and (not file_paths or len(file_paths) == 0):
        print(f"[DEBUG] Routing to: {intent_result['intent'].upper()}_WITHOUT_DATA")

        response_msg = f"I understand you want to perform a {intent_result['intent']} but no data files were uploaded. Please upload your data file(s) so I can help you with the analysis."

        yield {
            'type': 'llm_update',
            'content': f'⚠️ {intent_result["intent"].upper()} detected but no data files uploaded',
            'step': f'{intent_result["intent"]}_no_data'
        }

        yield {
            'type': 'result',
            'response': response_msg,
            'intent': intent_result,
            'result_type': f'{intent_result["intent"]}_without_data'
        }
        return
    



    # If it's a task and files are uploaded, generate solution
    if intent_result["intent"] == "task" and file_paths and len(file_paths) > 0:
        print(f"[DEBUG] Routing to: TASK_WITH_DATA")

        # Use yield from to propagate the generator yields to the WebUI
        yield from agm_workflow_handler(message, intent_result, file_paths, file_names, active_solutions)
        return  # Exit after handling task with data
    

    if intent_result["intent"] == "research":
        print(f"[DEBUG] Routing to: RESEARCH_WITH_DATA")

        # Use yield from to propagate the generator yields to the WebUI
        # yield from
        yield from agm_workflow_handler(message, intent_result, file_paths, file_names)
        return  # Exit after handling research with data


    
    


    # Fallback
    yield {
        'type': 'result',
        'response': "I'm not sure how to handle this request. Could you please clarify?",
        'intent': intent_result,
        'result_type': 'unknown'
    }





def just_chat_response(message, intent_result, chat_model=chat_model, stream=True):
    """
    Generate chat response with optional streaming
    Args:
        message: User's message
        intent_result: Intent detection result
        chat_model: OpenAI model to use
        stream: If True, stream the response; if False, return complete response at once
    Yields:
        Dictionary with response updates
    """
    yield {
        'type': 'llm_update',
        'content': '💬 Generating conversational response...',
        'step': 'chat_response'
    }

    user_key = helper.load_OpenAI_key()

    if stream:
        response = helper.gibd_chat_completion_stream(user_key, chat_model, [{"role": "user", "content": message}])
    else:
        response = helper.gibd_chat_completion(user_key, chat_model, [{"role": "user", "content": message}])

    if stream:
        full_response = ""
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield {
                    'type': 'llm_stream',
                    'content': content,
                    'step': 'streaming_chat'
                }

        yield {
            'type': 'result',
            'response': full_response,
            'intent': intent_result,
            'result_type': 'chat'
        }
    else:
        full_response = response.choices[0].message.content

        yield {
            'type': 'result',
            'response': full_response,
            'intent': intent_result,
            'result_type': 'chat'
        }

    return



def agm_workflow_handler(message, intent_result, file_paths=None, file_names=None, active_solutions=None):
    """
        Main handler for processing user messages and managing the workflow
        Args:
            message: User's message
            file_paths: List of uploaded file paths (optional)
            file_names: List of uploaded file names (optional)
        Returns:
            Dictionary with final response and metadata
        """
    #============== INITIALIZATIONS ===========================
    # Prepare datasets list for DataAgent
    datasets = []
    for path, name in zip(file_paths, file_names):
        datasets.append({
            "path": path,
            "name": name,
        })

    ctx = {
        "rq": message,
        "datasets": datasets,


        "artifacts": {},
        "logs": []
    }
    log_count = 0  # To track new logs for streaming updates



    # ===== STEP 1: Register Available Agents =====
    yield from helper.yield_status_update('Registering agents...', 'agent_registration')
    register_agent()

    # Debug: Log registered agents
    registered_agents = list(task_manager.agent_registry.keys()) if task_manager.agent_registry else []
    task_manager.log(ctx, f"📋 Registered agents: {registered_agents}")

    yield from helper.yield_new_logs(ctx, log_count, 'agent_registration')
    log_count = len(ctx.get('logs', []))


    # ===== STEP 2: Audit uploaded Data ====
    yield from helper.yield_status_update(f'📂 Auditing {len(file_paths)} uploaded file(s)...', 'data_audit')
    data_overview = data_agent.audit_datasets(ctx, {})
    yield from helper.yield_new_logs(ctx, log_count, 'data_audit')
    log_count = len(ctx.get('logs', []))

    # ===== STEP 3: SUMMARIZE THE AUDITED DATA ======
    yield from helper.yield_status_update(f'📂 Summarizing the audited data... ', 'data_summary')
    data_summary = data_agent.data_summary(ctx)
    yield from helper.yield_new_logs(ctx, log_count, 'data_summary')
    yield from helper.yield_status_update("Data summary generated successfully", "data_summary_completed")


    # ===== STEP 4: Research question breakdown =====
    yield from helper.yield_status_update('🔍 Understanding your research question in context of the data summary ...', 'rq_breakdown')
    rq_breakdown = rq_agent.rq_breakdown(ctx)
    yield from helper.yield_new_logs(ctx, log_count, 'rq_parsing')
    log_count = len(ctx.get('logs', []))
    yield from helper.yield_status_update('✅ Research question breakdown generated', 'rq_breakdown_complete')


    # # ===== STEP 4: Generate Research Plan =====
    yield from helper.yield_status_update('🧠 Generating research plan...', 'research_plan_generation')

    # Run langgraph in a thread, stream LLM chunks to frontend via queue
    import queue, threading
    chunk_queue = queue.Queue()
    ctx["_stream_callback"] = lambda content: chunk_queue.put({"type": "research_plan_stream", "content": content})
    graph_error = [None]

    def _run_graph():
        try:
            research_plan_generator.build_serial_graph(ctx)
        except Exception as e:
            graph_error[0] = e
        finally:
            chunk_queue.put(None)

    t = threading.Thread(target=_run_graph, daemon=True)
    t.start()
    while True:
        try:
            item = chunk_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        yield item
    t.join()
    ctx.pop("_stream_callback", None)
    if graph_error[0]:
        raise graph_error[0]

    yield from helper.yield_new_logs(ctx, log_count, 'research_plan_generation')
    log_count = len(ctx.get('logs', []))

    # Access the structured plan and save both visualizations
    structured_plan = ctx["artifacts"]["research_plan_final"]
    task_id = ctx.get('task_id', 'default')
    task_output_dir = _get_task_output_dir(task_id)
    viz_paths = research_plan_viz.save_research_plan_visualization(
        structured_plan, task_output_dir, task_id
    )
    # Store ctx for research plan feedback loop
    research_plan_generator.store_context(task_id, ctx)

    yield from helper.yield_status_update('✅ Research plan generated successfully',
                                               'research_plan_generation_complete')

    # Send research plan visualization info to frontend
    yield {
        'type': 'research_plan_viz',
        'viz_id': task_id,
        'viz_url': f"/api/research-plan/{task_id}"
    }

    # ===== STEP 5: Execute Research Plan =====
    ## retrieve the objectives first
    yield from helper.yield_status_update('🎯 Extracting research objectives from research plan...',
                                          'extract_objectives')

    research_plan = ctx["artifacts"]["research_plan_final"]
    objective_research_objects = task_manager.extract_research_object_for_objective(ctx, research_plan)

    if objective_research_objects:
        yield from helper.yield_status_update(f'✅ Extracted research objects for {len(objective_research_objects)} objectives', 'objectives_object_extracted')
        # print(f"[DEBUG] Extracted objectives: {objectives}")

    total_objectives = len(objective_research_objects)
    completed_objectives = 0

    # Executing each objective
    for obj_index, (obj_key, obj_data) in enumerate(objective_research_objects.items(), start=1):
        # Ensure agents are registered before routing
        if not task_manager.agent_registry:
            task_manager.log(ctx, "⚠️ Agent registry empty, re-registering agents...")
            register_agent(ctx)


        # START OBJECTIVE
        task_manager.log(ctx, f"📋 OBJECTIVE {obj_index}/{total_objectives}: {obj_key}")
        task_manager.log(ctx, f"\nExecuting steps for {obj_key}: {obj_data['title']}")
        yield from helper.yield_status_update(f'🎯 Starting {obj_key} ({obj_index}/{total_objectives})', f'objective_{obj_key}_start')
        yield from helper.yield_new_logs(ctx, log_count, f'objective_{obj_key}_start')
        log_count = len(ctx.get('logs', []))
        
        
        steps = obj_data.get("steps", [])
        total_steps = len(steps)
        completed_steps = 0

        # Iterate through steps of the objective
        for step_index, step in enumerate(steps, start=1):
            step_number = step.get("step_number", step_index)
            step_description = step.get("description", "")

            # START STEP        
            task_manager.log(ctx, f"\n  ▶ STEP {step_index}/{total_steps} (Step {step_number}): {step_description}")
            yield from helper.yield_status_update(f'🚀 {obj_key} - Step {step_index}/{total_steps}', f'step_{obj_key}_{step_number}_start')
            
            # Route and execute the step
            routing = task_manager._route_step_to_agent(ctx, step)
            agent_name = routing.get("agent")
            reasoning = routing.get("reasoning", "")
            task_manager.log(ctx, f"    → Routing to: {agent_name}")
            task_manager.log(ctx, f"    → Reasoning: {reasoning}")

            # Debug: Log registry state
            available_agents = list(task_manager.agent_registry.keys()) if task_manager.agent_registry else []
            task_manager.log(ctx, f"🔍 DEBUG: Available agents in registry: {available_agents}")
            task_manager.log(ctx, f"🔍 DEBUG: Looking for agent: '{agent_name}' (type: {type(agent_name).__name__})")

            # Check if agent is registered
            agent_info = task_manager.agent_registry.get(agent_name)

            if not agent_info:
                task_manager.log(ctx, f"⚠ Agent '{agent_name}' not yet implemented")
                task_manager.log(ctx, f"🔍 DEBUG: Registry has {len(task_manager.agent_registry)} agents but '{agent_name}' not found")
                yield from helper.yield_status_update(f'⚠️ Step {step_number} skipped - agent not available', f'step_{obj_key}_{step_number}_skipped')
                continue  # Skip to next step instead of failing
                

            # declaration of some parameters

            step_number = step.get("step_number", "") #step["step_number"]
            step_description = step.get("step_description", "")
            step_expected_output = step.get("expected_output", "")
            step_dependencies = step.get("dependencies", [])


            # Prepare task (step) for the agent
            task_spec = {
                "name": "execute_step",
                "params_dictionary": {
                    "objective_key": obj_key,
                    "objective": obj_data['title'],
                    "step_number": step_number,
                    "step_description": step_description,
                    "step_expected_output":  step_expected_output,
                    "dependencies": step_dependencies,
                    "routing_info": routing,
                    "related_data_columns": obj_data['suggested_variables']

                }
            }



            # # Execute the step with the selected agent
            try :
                # Generate solution for step by assigned agent with streaming updates 
                agent_instance = agent_info["instance"]

                # yield from helper.yield_status_update ()
                yield from helper.yield_status_update(f"🚀 Executing {obj_key} step {step_number} with {agent_name}...")
                yield from helper.yield_new_logs(ctx, log_count, f'step_execution_{obj_key}_step{step_number}')
                log_count = len(ctx.get('logs', []))

                # Create Solution object of agent instance
                
                solution  = agent_instance.create_solution(task = step_description, data_overview=data_overview, model_requirement=None)
                # Store solution in active_solutions for later use
                if active_solutions is not None:
                    active_solutions[solution.task_id] = solution
                # Generate solution graph
                task_manager.log(ctx, f"🔧 Generating solution graph for {obj_key} /  step {step_number}")

                yield from helper.yield_new_logs(ctx, log_count, f'graph_generation_{obj_key}_step{step_number}')
                log_count = len(ctx.get('logs', []))
                
                DataAgent().log(ctx, f"🔧 TESTING")
                # Stream the LLM response for graph generation
                print(f"[DEBUG] About to call get_LLM_response_for_graph with yield_chunks=True")  # Add this
                for chunk in solution.get_LLM_response_for_graph(stream=True, execute=True, yield_chunks=True):
                    # print(f"[DEBUG] Got chunk from get_LLM_response_for_graph: {chunk.get('type')}")  # Add this
                    yield chunk
                
                task_manager.log(ctx, f"    ✅ Solution graph generated successfully")
                yield from helper.yield_new_logs(ctx, log_count, f'graph_complete_{obj_key}_step{step_number}')
                log_count = len(ctx.get('logs', []))


                yield {
                    'type': 'llm_update',
                    'content': '✅ Workflow graph generated successfully',
                    'step': 'graph_generated'
                }

                yield {
                    'type': 'llm_update',
                    'content': '📝 Loading and validating graph structure...',
                    'step': 'graph_loading'
                }

                solution_graph, nt = solution.load_graph_file()

                 # ... (your graph processing code here) ...

                # Convert graph to frontend format
                graph_data = {
                    'nodes': [
                        {
                            'id': node,
                            'label': solution_graph.nodes[node].get('label', node),
                            'shape': solution_graph.nodes[node].get('shape', 'box'),
                            'color': solution_graph.nodes[node].get('color', '#97C2FC')
                        }
                        for node in solution_graph.nodes()
                    ],
                    'edges': [
                        {
                            'from': edge[0],
                            'to': edge[1],
                            'label': solution_graph.edges[edge].get('label', '')
                        }
                        for edge in solution_graph.edges()
                    ]
                }
                result = {
                    'graph_html_path': f"/api/graph/{solution.task_id}",
                    'graph_data': graph_data,
                    'graph_response': solution.graph_response,
                    'data_overview': data_overview,
                    '_task_id': solution.task_id
                }
                
                yield {
                    'type': 'llm_update',
                    'content': '🎉 Workflow graph complete!',
                    'step': 'graph_complete'
                }

                
                # ===== SAVE WORKFLOW STATE FOR CONTINUATION =====
                workflow_state = {
                    'message': message,
                    'intent_result': intent_result,
                    'file_paths': file_paths,
                    'file_names': file_names,
                    'data_overview': data_overview,
                    'objective_research_objects': objective_research_objects,
                    'current_obj_key': obj_key,
                    'current_obj_index': obj_index,
                    'current_obj_data': obj_data,
                    'current_step_index': step_index,
                    'current_step_number': step_number,
                    'current_step': step,
                    'total_objectives': total_objectives,
                    'total_steps': total_steps,
                    'completed_objectives': completed_objectives,
                    'completed_steps': completed_steps,
                    'agent_name': agent_name,
                    'ctx': ctx,
                    'log_count': log_count
                }
                save_workflow_state(solution.task_id, workflow_state)
                
                
                
                # ==== RETURN GRAPH FOR THIS STEP - WAIT FOR USER INTERACTION ====
                yield {
                    'type': 'result',
                    'response': f"Graph generated for {obj_key} - Step {step_number}. Please review and approve to proceed.",
                    'intent': intent_result,
                    'solution': result,
                    'workflow_state': {
                        'task_id': solution.task_id,
                        'current_objective': obj_key,
                        'current_step': step_number,
                        'obj_index': obj_index,
                        'step_index': step_index,
                        'total_objectives': total_objectives,
                        'total_steps': total_steps,
                        'completed_steps': completed_steps
                    },
                    'result_type': 'step_graph_awaiting_approval'
                }
                return  # ← EXIT HERE - Wait for user approval

                # # COMPLETE STEP
                # completed_steps += 1
                # task_manager.log(ctx, f"  ✅ STEP {step_index}/{total_steps} COMPLETED")
                # yield from helper.yield_status_update(
                #     f'✅ {obj_key} - Step {step_index}/{total_steps} complete', 
                #     f'step_{obj_key}_{step_number}_complete'
                # )
                # yield from helper.yield_new_logs(ctx, log_count, f'step_{obj_key}_{step_number}_complete')
                # log_count = len(ctx.get('logs', []))
                
            except Exception as e:
                error_msg = f"Error executing step {step_number}: {str(e)}"
                task_manager.log(ctx, f"    ❌ ERROR: {error_msg}")
                yield from helper.yield_status_update(f'❌ Step {step_number} failed', f'step_{obj_key}_{step_number}_error')
                # Continue to next step instead of failing entire workflow
                continue
        
    #     # COMPLETE OBJECTIVE
    #     completed_objectives += 1
    #     task_manager.log(ctx, f"\n{'='*60}")
    #     task_manager.log(ctx, f"✅ OBJECTIVE {obj_index}/{total_objectives} COMPLETED: {obj_key}")
    #     task_manager.log(ctx, f"   Completed {completed_steps}/{total_steps} steps")
    #     task_manager.log(ctx, f"{'='*60}\n")
    #     yield from helper.yield_status_update(
    #         f'✅ Objective {obj_index}/{total_objectives} complete', 
    #         f'objective_{obj_key}_complete'
    #     )
    #     yield from helper.yield_new_logs(ctx, log_count, f'objective_{obj_key}_complete')
    #     log_count = len(ctx.get('logs', []))

    # # ===== ALL OBJECTIVES COMPLETE =====
    # task_manager.log(ctx, f"\n🎉 ALL OBJECTIVES COMPLETED: {completed_objectives}/{total_objectives}")
    # yield from helper.yield_status_update('🎉 Research plan execution complete!', 'all_objectives_complete')

    # # ===== STEP 6: Return Final Result =====
    # yield {
    #     'type': 'result',
    #     'response': f"Research plan execution complete! Completed {completed_objectives} objectives.",
    #     'intent': intent_result,
    #     'objectives_completed': completed_objectives,
    #     'total_objectives': total_objectives,
    #     'result_type': 'research_complete'
    # }


#=================================== REAL WORKFLOW ENDS =============================================
def get_workflow_nodes_info(task_id):
    """
    Get detailed information about all nodes and edges in a workflow graph
    Args:
        task_id: The unique identifier for the workflow
    Returns:
        Dictionary with nodes, edges, and graph statistics
    """
    import networkx as nx
    import os
    from utils.config import load_configurations

    settings = load_configurations()
    task_output_dir = _get_task_output_dir(task_id)
    graph_dir = os.path.join(task_output_dir, "graphfiles")
    graph_file = os.path.join(graph_dir, f"{task_id}.graphml")

    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    # Load the graph
    G = nx.read_graphml(graph_file)

    # Extract nodes information
    data_nodes = []
    operation_nodes = []

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_info = {
            'name': node,
            'type': node_type,
            'description': attrs.get('description', 'No description')
        }

        if node_type == 'data':
            node_info['data_path'] = attrs.get('data_path', '')
            data_nodes.append(node_info)
        else:
            operation_nodes.append(node_info)

    # Extract edges
    edges = [{'source': source, 'target': target} for source, target in G.edges()]

    return {
        'data_nodes': data_nodes,
        'operation_nodes': operation_nodes,
        'edges': edges,
        'total_nodes': len(G.nodes()),
        'total_edges': len(G.edges()),
        '_task_id': task_id  # Internal use only, not for display
    }



def refine_workflow_graph_stream(task_id, feedback, task=None, data_overview=None, active_solutions=None):
    """
    Streaming version of refine_workflow_graph
    Yields LLM streaming chunks in real-time
    Args:
        task_id: The unique identifier for the workflow
        feedback: User feedback text for refinement
        task: Original task description (optional, for new sessions)
        data_overview: Original data overview (optional, for new sessions)
        active_solutions: Dictionary of active Solution instances (for session management)
    Yields:
        Dictionary with streaming chunk info
    """
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution import Solution
    import networkx as nx
    import os
    from utils.config import load_configurations

    settings = load_configurations()
    task_output_dir = _get_task_output_dir(task_id)
    graph_dir = os.path.join(task_output_dir, "graphfiles")
    graph_file = os.path.join(graph_dir, f"{task_id}.graphml")

    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    # Get or create Solution instance
    if active_solutions is not None and task_id in active_solutions:
        solution = active_solutions[task_id]
    else:
        # Load existing graph and create Solution instance
        G = nx.read_graphml(graph_file)
        solution = Solution(
            task=task or 'Workflow refinement',
            data_overview=data_overview or '',
            task_id=task_id
        )
        solution.solution_graph = G

        # Store in active_solutions if provided
        if active_solutions is not None:
            active_solutions[task_id] = solution

    # Send initial status update
    yield {
        'type': 'llm_update',
        'content': f'🔄 Refining workflow graph based on your feedback...',
        'step': 'graph_refinement_start'
    }

    # Stream the graph refinement process with yield_chunks=True
    for chunk in solution.refine_graph(feedback, stream=True, yield_chunks=True):
        yield chunk

    # Reload the graph and regenerate HTML
    solution.solution_graph = nx.read_graphml(solution.graph_file)
    nt = solution.show_graph(graph_file=solution.solution_graph)
    nt.save_graph(solution.graph_html_file)

    yield {
        'type': 'llm_update',
        'content': '✅ Workflow graph refined successfully',
        'step': 'graph_refined'
    }

    # Send the final result
    yield {
        'type': 'result',
        'revision_count': solution.revision_count,
        'graph_file': solution.graph_file,
        'graph_html': solution.graph_html_file,
        'task_id': task_id
    }


def approve_workflow_graph(task_id, active_solutions=None):
    """
    Mark a workflow as approved by the user
    Args:
        task_id: The unique identifier for the workflow
        active_solutions: Dictionary of active Solution instances (for session management)
    Returns:
        Dictionary with approval status
    """
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution import Solution
    import networkx as nx
    import os
    from utils.config import load_configurations

    settings = load_configurations()
    task_output_dir = _get_task_output_dir(task_id)
    graph_dir = os.path.join(task_output_dir, "graphfiles")
    graph_file = os.path.join(graph_dir, f"{task_id}.graphml")

    # Get or create Solution instance
    if active_solutions is not None and task_id in active_solutions:
        solution = active_solutions[task_id]
    else:
        # Load existing graph and create Solution instance
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        G = nx.read_graphml(graph_file)
        solution = Solution(
            task='Workflow approval',
            data_overview='',
            task_id=task_id
        )
        solution.solution_graph = G

        # Store in active_solutions if provided
        if active_solutions is not None:
            active_solutions[task_id] = solution

    # Approve the workflow
    solution.approve_graph()
    revision_count = solution.revision_count

    return {
        'revision_count': revision_count,
        'approved': True,
        '_task_id': task_id  # Internal use only, not for display
    }





def refine_workflow_code(task_id, feedback, code_type='assembly', active_solutions=None):
    """
    Refine generated code based on user feedback
    Args:
        task_id: The unique identifier for the workflow
        feedback: User feedback text for code refinement
        code_type: Type of code to refine ('assembly' or 'operation')
        active_solutions: Dictionary of active Solution instances (for session management)
    Returns:
        Dictionary with refined code
    """
    import utils.agm_helper as helper
    import os
    from utils.config import load_configurations

    settings = load_configurations()
    task_output_dir = _get_task_output_dir(task_id)
    code_dir = os.path.join(task_output_dir, "generated_code")
    code_file = os.path.join(code_dir, f"{task_id}.py")

    if not os.path.exists(code_file):
        raise FileNotFoundError(f"Code file not found. Please generate code first.")

    # Get Solution instance
    if active_solutions is not None and task_id in active_solutions:
        solution = active_solutions[task_id]
    else:
        raise ValueError("No active solution found. Please generate code first.")

    # Read current code
    with open(code_file, 'r', encoding='utf-8') as f:
        current_code = f.read()

    print(f"Refining code for workflow {task_id} with feedback: {feedback}")

    # Use LLM to refine the code based on feedback
    system_prompt = """You are a senior Python developer and GIS expert.
Your task is to refine existing Python code based on user feedback.

RULES:
1. Maintain the overall structure and logic of the code
2. Apply the user's feedback to improve the code
3. Ensure the code is correct, efficient, and follows best practices
4. Keep all necessary imports and function definitions
5. Return ONLY the refined Python code in a code block

OUTPUT:
- Output ONLY Python code enclosed in ```python ... ```
- Do not include explanations outside the code block
"""

    user_prompt = f"""## Current Code:
```python
{current_code}
```

## User Feedback:
{feedback}

## Task:
Refine the code based on the user feedback. Return the complete refined code.
"""

    user_key = helper.load_OpenAI_key()
    response = helper.gibd_chat_completion(
        user_key, llm_model("spatial_analysis.workflow_code_refine"),
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
         **llm_reasoning("spatial_analysis.workflow_code_refine"),
    )

    refined_code_response = response.choices[0].message.content

    # Extract code from response
    refined_code = helper.extract_code(response_content=refined_code_response)

    if not refined_code:
        refined_code = refined_code_response  # Fallback if no code block found

    # Save refined code
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(refined_code)

    print(f"Refined code saved to: {code_file}")

    return {
        'refined_code': refined_code,
        'code_file': code_file,
        '_task_id': task_id  # Internal use only, not for display
    }


def generate_code_stream(task_id, active_solutions=None):
    """
    Streaming version for code generation (without execution)
    Called when user approves the workflow
    Args:
        task_id: The unique identifier for the workflow
        active_solutions: Dictionary of active Solution instances
    Yields:
        Dictionary with update information for streaming to frontend
    """
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution import Solution
    import networkx as nx
    import os
    from utils.config import load_configurations

    settings = load_configurations()
    task_output_dir = _get_task_output_dir(task_id)
    graph_dir = os.path.join(task_output_dir, "graphfiles")
    graph_file = os.path.join(graph_dir, f"{task_id}.graphml")

    if not os.path.exists(graph_file):
        yield {
            'type': 'error',
            'error': f"Graph file not found: {graph_file}"
        }
        return

    # Get or create Solution instance
    if active_solutions is not None and task_id in active_solutions:
        solution = active_solutions[task_id]
    else:
        # Load existing graph and create Solution instance
        G = nx.read_graphml(graph_file)
        solution = Solution(
            task='Code generation and execution',
            data_overview='',
            task_id=task_id
        )
        solution.solution_graph = G

        if active_solutions is not None:
            active_solutions[task_id] = solution

    # Set up streaming callback to collect chunks
    streaming_chunks = []

    def stream_callback(chunk):
        """Callback to capture streaming chunks"""
        streaming_chunks.append(chunk)

    solution.stream_callback = stream_callback

    yield {
        'type': 'llm_update',
        'content': '✅ Approving workflow graph...',
        'step': 'workflow_approval'
    }

    # Approve the workflow
    solution.approve_graph()

    yield {
        'type': 'llm_update',
        'content': '✅ Workflow approved successfully',
        'step': 'workflow_approved'
    }

    # Generate code for the workflow
    yield {
        'type': 'llm_update',
        'content': '💻 Starting code generation...',
        'step': 'code_generation_start'
    }

    try:
        # STEP 1: Generate operation codes for each node
        # num_nodes = len(solution.solution_graph.nodes())
        # yield {
        #     'type': 'llm_update',
        #     'content': f'🔧 Generating code for {num_nodes} workflow operations...',
        #     'step': 'operations_code_generation'
        # }

        # Stream the LLM responses for operation code generation
        for chunk in solution.get_LLM_responses_for_operations(stream=True, yield_chunks=True):
            yield chunk

        # Extract all operation codes from solution.operations
        operation_codes = []
        for operation in solution.operations:
            if 'operation_code' in operation and operation['operation_code']:
                operation_codes.append({
                    'node_name': operation['node_name'],
                    'description': operation['description'],
                    'code': operation['operation_code']
                })

                yield {
                    'type': 'llm_update',
                    'content': f'✅ Generated code for operation: {operation["node_name"]}',
                    'step': 'operation_code_generated'
                }

        all_operation_code_str = '\n\n'.join([op['code'] for op in operation_codes])

        # STEP 2: Generate assembly code (complete program)
        yield {
            'type': 'llm_update',
            'content': '🔨 Assembling complete workflow program...',
            'step': 'assembly_code_generation'
        }

        # Stream assembly code generation directly
        for chunk in solution.get_LLM_assembly_response(stream=True, yield_chunks=True):
            yield chunk

        assembly_code = solution.code_for_assembly

        yield {
            'type': 'llm_update',
            'content': '✅ Assembly code generated successfully',
            'step': 'assembly_complete'
        }

        # Combine all code
        complete_code = all_operation_code_str + '\n\n' + assembly_code

        # Save code to file
        code_dir = os.path.join(task_output_dir, "generated_code")
        os.makedirs(code_dir, exist_ok=True)
        code_file = os.path.join(code_dir, f"{task_id}.py")

        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(complete_code)

        yield {
            'type': 'llm_update',
            'content': f'💾 Code saved to: {os.path.basename(code_file)}',
            'step': 'code_saved'
        }

        yield {
            'type': 'llm_update',
            'content': '🎉 Code generation complete!',
            'step': 'code_generation_complete'
        }

        # Final result - code is generated but NOT executed yet
        yield {
            'type': 'result',
            'generated_code': complete_code,
            'operation_count': len(operation_codes),
            'code_file': code_file,
            'message': f"Generated {len(operation_codes)} operations and assembled the code successfully. Click 'Execute Code' to run it.",
            '_task_id': task_id  # Internal use only, not for display
        }

    except Exception as code_error:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during code generation: {error_details}")

        yield {
            'type': 'llm_update',
            'content': f'⚠️ Error: {str(code_error)}',
            'step': 'code_error'
        }

        yield {
            'type': 'error',
            'error': str(code_error),
            'details': error_details
        }


def execute_code_stream(task_id, code, active_solutions=None):
    """
    Streaming version for code execution - streams debugging output in REAL-TIME
    Called when user clicks "Execute Code" button
    Args:
        task_id: The unique identifier for the workflow
        code: The Python code to execute
        active_solutions: Dictionary of active Solution instances
    Yields:
        Dictionary with update information for streaming to frontend
    """
    from agents.task_manager.WEBUI_TASK_PIPELINE.task_solution import Solution
    from agents.task_manager.WEBUI_TASK_PIPELINE import task_solution_prompts as solution_prompts
    import utils.agm_helper as helper
    import os
    from utils.config import load_configurations
    import glob

    settings = load_configurations()

    # Get or create Solution instance
    if active_solutions is not None and task_id in active_solutions:
        solution = active_solutions[task_id]
    else:
        # Create a new Solution instance
        solution = Solution(
            task='Code execution',
            data_overview='',
            task_id=task_id
        )
        if active_solutions is not None:
            active_solutions[task_id] = solution
            
    # Check if workflow state exists, if not, try to recover it from active_solutions
    if task_id not in workflow_states and active_solutions and task_id in active_solutions:
        solution = active_solutions[task_id]
        # Create a minimal workflow state to enable continuation
        # Note: This is a simplified state - ideally load from database/file
        workflow_states[task_id] = {
            'task_id': task_id,
            'solution': solution,
            # Add other required state fields if available
        }
        print(f"[DEBUG] Recovered workflow state for task_id: {task_id}")

    yield {
        'type': 'llm_update',
        'content': '🚀 Starting code execution with auto-debugging...',
        'step': 'executing'
    }

    print(f"Executing code for workflow {task_id}...")

    # Track files before execution to detect new and modified artifacts
    outputs_dir = settings.output_dir
    _tracked_exts = ['*.png', '*.jpg', '*.jpeg', '*.csv', '*.html', '*.json', '*.geojson',
                     '*.shp', '*.gpkg', '*.txt', '*.md', '*.npy']
    before_mtimes = {}
    for ext in _tracked_exts:
        for fp in glob.glob(os.path.join(outputs_dir, '**', ext), recursive=True):
            try:
                before_mtimes[fp] = os.path.getmtime(fp)
            except OSError:
                pass

    try:
        # Execute the code with debugging support - INLINE to enable real-time streaming
        code_to_exec = code
        executed_code = code  # Initialize in case all attempts fail
        count = 0
        try_cnt = 3
        execution_successful = False
        last_error = None

        while count < try_cnt:
            print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
            try:
                count += 1
                compiled_code = compile(code_to_exec, 'Complete program', 'exec')
                exec(compiled_code, globals())
                print("\n\n--------------- Done ---------------\n\n")
                executed_code = code_to_exec
                execution_successful = True
                break  # Success! Exit the retry loop

            except Exception as err:
                executed_code = code_to_exec  # Save the last attempted version
                last_error = err

                if count == try_cnt:
                    print(f"Failed to execute and debug the code within {try_cnt} times.")
                    yield {
                        'type': 'llm_update',
                        'content': f'❌ Failed to fix code after {try_cnt} attempts.',
                        'step': 'debugging'
                    }
                    break  # Exit loop, will handle error in result

                # Debugging needed - yield status update
                yield {
                    'type': 'llm_update',
                    'content': f'❌ Execution error detected. Attempting to debug (attempt {count}/{try_cnt})...',
                    'step': 'debugging'
                }

                debug_prompt = solution.get_debug_prompt(exception=err, code=code_to_exec)

                print("Sending error information to LLM for debugging...")

                # Yield another update before streaming starts
                yield {
                    'type': 'llm_update',
                    'content': '🔧 Analyzing error and generating fix...',
                    'step': 'debugging'
                }

                messages = [
                    {"role": "system", "content": solution_prompts.debug_role},
                    {"role": "user", "content": debug_prompt}
                ]

                response = helper.gibd_chat_completion_stream(
                    helper.load_OpenAI_key(), solution._llm_model("agent_solution.debug"), messages, **solution._llm_reasoning("agent_solution.debug")
                )

                # Stream LLM debugging response in REAL-TIME
                full_content = ""
                for chunk_response in response:
                    if not chunk_response.choices:
                        continue
                    chunk_content = chunk_response.choices[0].delta.content
                    if chunk_content is not None:
                        full_content += chunk_content
                        print(chunk_content, end='', flush=True)
                        # Yield streaming chunk in REAL-TIME
                        yield {
                            'type': 'llm_stream',
                            'content': chunk_content,
                            'step': 'debugging'
                        }
                print()  # New line after streaming completes

                # Extract corrected code from the LLM response
                code_to_exec = helper.extract_code(response_content=full_content)

                # Yield status update
                yield {
                    'type': 'llm_update',
                    'content': f'✓ Code updated by debugger. Retrying execution...',
                    'step': 'executing'
                }

        # If code was debugged (different from original), save the fixed version back to file
        if executed_code != code:
            code_dir = os.path.join(settings.output_dir, "generated_code")
            os.makedirs(code_dir, exist_ok=True)
            code_file = os.path.join(code_dir, f"{task_id}.py")
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(executed_code)
            print(f"Debugged code saved to: {code_file}")

        # Detect new files created during execution
        yield {
            'type': 'llm_update',
            'content': '🔍 Detecting generated artifacts...',
            'step': 'detecting_artifacts'
        }

        new_files = set()
        for ext in _tracked_exts:
            for fp in glob.glob(os.path.join(outputs_dir, '**', ext), recursive=True):
                try:
                    cur_mtime = os.path.getmtime(fp)
                except OSError:
                    continue
                if fp not in before_mtimes or cur_mtime > before_mtimes[fp]:
                    new_files.add(fp)
        artifacts = []
        shapefile_components_added = set()  # Track which component files we've already added

        for file_path in sorted(new_files):
            rel_path = os.path.relpath(file_path, outputs_dir)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            artifacts.append({
                'path': file_path,
                'relative_path': rel_path,
                'filename': os.path.basename(file_path),
                'size': file_size,
                'type': file_ext[1:] if file_ext else 'unknown'
            })

            # If this is a shapefile (.shp), automatically find and include component files
            if file_ext == '.shp':
                base_path = os.path.splitext(file_path)[0]  # Remove .shp extension
                component_extensions = ['.shx', '.dbf', '.prj', '.cpg']

                for comp_ext in component_extensions:
                    comp_path = base_path + comp_ext
                    if os.path.exists(comp_path) and comp_path not in shapefile_components_added:
                        comp_rel_path = os.path.relpath(comp_path, outputs_dir)
                        comp_size = os.path.getsize(comp_path)

                        artifacts.append({
                            'path': comp_path,
                            'relative_path': comp_rel_path,
                            'filename': os.path.basename(comp_path),
                            'size': comp_size,
                            'type': comp_ext[1:],  # shx, dbf, prj, cpg
                            'is_shapefile_component': True  # Mark as component so UI can handle it
                        })
                        shapefile_components_added.add(comp_path)

        if artifacts:
            yield {
                'type': 'llm_update',
                'content': f'📊 Generated {len(artifacts)} artifact(s): {", ".join([a["filename"] for a in artifacts])}',
                'step': 'artifacts_generated'
            }
        else:
            yield {
                'type': 'llm_update',
                'content': '📊 No new artifacts detected',
                'step': 'no_artifacts'
            }

        # Check if execution was successful and yield appropriate result
        if execution_successful:
            yield {
                'type': 'llm_update',
                'content': '✅ Code executed successfully!',
                'step': 'code_executed'
            }

            yield {
                'type': 'llm_update',
                'content': '🎉 Execution complete!',
                'step': 'execution_complete'
            }

            # Final success result
            yield {
                'type': 'result',
                'success': True,
                'artifacts': artifacts,
                'executed_code': executed_code,
                'message': f"Code executed successfully! {len(artifacts)} artifact(s) were created." if artifacts else "Code executed successfully!",
                '_task_id': task_id
            }
        else:
            # Execution failed even after debugging attempts
            yield {
                'type': 'result',
                'success': False,
                'error': str(last_error) if last_error else 'Unknown error',
                'message': f'Code execution failed after {try_cnt} attempts: {str(last_error) if last_error else "Unknown error"}',
                'executed_code': executed_code,
                'artifacts': artifacts,
                '_task_id': task_id
            }

    except Exception as exec_error:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during code execution: {error_details}")

        yield {
            'type': 'llm_update',
            'content': f'❌ Execution error: {str(exec_error)}',
            'step': 'execution_error'
        }

        yield {
            'type': 'error',
            'error': str(exec_error),
            'details': error_details
        }
        
        
        
def continue_to_next_step(task_id):
    """
    Continue workflow to the next step after code execution.
    Called after execute_code_stream() completes.
    
    Yields:
        Status updates and next step graph (or completion message)
    """
    # Retrieve saved workflow state
    state = get_workflow_state(task_id)
    
    if not state:
        yield {
            'type': 'error',
            'message': 'Workflow state not found. Please restart the workflow.'
        }
        return
    
    # Extract state variables
    objective_research_objects = state['objective_research_objects']
    current_obj_key = state['current_obj_key']
    current_step_index = state['current_step_index']
    total_steps = state['total_steps']
    completed_steps = state['completed_steps'] + 1
    obj_data = state['current_obj_data']
    data_overview = state['data_overview']
    intent_result = state['intent_result']
    ctx = state['ctx']
    log_count = state['log_count']

    # Ensure agents are registered (critical when state is loaded from disk after server restart)
    if not task_manager.agent_registry:
        task_manager.log(ctx, "⚠️ Agent registry empty, re-registering agents...")
        register_agent(ctx)
        yield from helper.yield_new_logs(ctx, log_count, 'agent_reregistration')
        log_count = len(ctx.get('logs', []))

    # Log current step completion
    task_manager.log(ctx, f"  ✅ STEP {current_step_index}/{total_steps} COMPLETED")
    yield from helper.yield_status_update(
        f'✅ Step {current_step_index}/{total_steps} complete',
        f'step_{current_step_index}_complete'
    )
    yield from helper.yield_new_logs(ctx, log_count, f'step_complete')
    log_count = len(ctx.get('logs', []))
    
    # Get steps for current objective
    steps = obj_data.get("steps", [])
    
    # Check if there are more steps in this objective
    if current_step_index < len(steps):
        # ===== MOVE TO NEXT STEP =====
        next_step_index = current_step_index + 1
        next_step = steps[next_step_index - 1]  # 0-indexed list
        next_step_number = next_step.get("step_number", next_step_index)
        next_step_description = next_step.get("description", "")
        
        task_manager.log(ctx, f"\n  ▶ STEP {next_step_index}/{total_steps} (Step {next_step_number}): {next_step_description}")
        yield from helper.yield_status_update(
            f'🚀 {current_obj_key} - Step {next_step_index}/{total_steps}',
            f'step_{next_step_index}_start'
        )

        # Send status before blocking LLM call to keep connection alive
        yield from helper.yield_status_update(
            f'🔍 Analyzing step requirements...',
            f'routing_agent_step_{next_step_index}'
        )

        # Route to agent (this makes a blocking LLM call)
        routing = task_manager._route_step_to_agent(ctx, next_step)
        agent_name = routing.get("agent")
        task_manager.log(ctx, f"    → Routing to: {agent_name}")

        # Yield logs immediately after routing completes
        yield from helper.yield_new_logs(ctx, log_count, f'routing_complete_step_{next_step_index}')
        log_count = len(ctx.get('logs', []))
        
        agent_info = task_manager.agent_registry.get(agent_name)
        
        if not agent_info:
            task_manager.log(ctx, f"⚠ Agent '{agent_name}' not available - SKIPPING")
            yield from helper.yield_status_update(f'⚠️ Step skipped - agent not available', 'step_skipped')
            # Update state and recursively continue
            state['current_step_index'] = next_step_index
            state['completed_steps'] = completed_steps
            save_workflow_state(task_id, state)
            yield from continue_to_next_step(task_id)
            return
        
        # Create solution for next step
        agent_instance = agent_info["instance"]
        solution = agent_instance.create_solution(
            task=next_step_description,
            data_overview=data_overview,
            model_requirement=None
        )
        
        task_manager.log(ctx, f"🔧 Generating solution graph for {current_obj_key} / step {next_step_number}")
        yield from helper.yield_new_logs(ctx, log_count, f'graph_generation_step_{next_step_index}')
        log_count = len(ctx.get('logs', []))
        
        # Stream graph generation
        for chunk in solution.get_LLM_response_for_graph(stream=True, execute=True, yield_chunks=True):
            yield chunk
        
        task_manager.log(ctx, f"✅ Solution graph generated successfully")
        yield from helper.yield_new_logs(ctx, log_count, f'graph_complete_step_{next_step_index}')
        log_count = len(ctx.get('logs', []))
        
        # Load and prepare graph
        solution_graph, nt = solution.load_graph_file()
        
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    'label': solution_graph.nodes[node].get('label', node),
                    'shape': solution_graph.nodes[node].get('shape', 'box'),
                    'color': solution_graph.nodes[node].get('color', '#97C2FC')
                }
                for node in solution_graph.nodes()
            ],
            'edges': [
                {
                    'from': edge[0],
                    'to': edge[1],
                    'label': solution_graph.edges[edge].get('label', '')
                }
                for edge in solution_graph.edges()
            ]
        }
        
        result = {
            'graph_html_path': f"/api/graph/{solution.task_id}",
            'graph_data': graph_data,
            'graph_response': solution.graph_response,
            'data_overview': data_overview,
            '_task_id': solution.task_id
        }
        
        # Update state for next iteration
        state['current_step_index'] = next_step_index
        state['current_step_number'] = next_step_number
        state['current_step'] = next_step
        state['completed_steps'] = completed_steps
        state['log_count'] = log_count
        save_workflow_state(solution.task_id, state)
        
        # Return next step's graph
        yield {
            'type': 'result',
            'response': f"Step {current_step_index} complete! Generated graph for Step {next_step_index}/{total_steps}. Please review and approve.",
            'intent': intent_result,
            'solution': result,
            'workflow_state': {
                'task_id': solution.task_id,
                'current_objective': current_obj_key,
                'current_step': next_step_number,
                'step_index': next_step_index,
                'total_steps': total_steps,
                'completed_steps': completed_steps
            },
            'result_type': 'step_graph_awaiting_approval'
        }
        
    else:
        # ===== NO MORE STEPS - OBJECTIVE COMPLETE =====
        task_manager.log(ctx, f"\n{'='*60}")
        task_manager.log(ctx, f"✅ OBJECTIVE COMPLETED: {current_obj_key}")
        task_manager.log(ctx, f"   Completed {completed_steps}/{total_steps} steps")
        task_manager.log(ctx, f"{'='*60}\n")
        
        yield from helper.yield_status_update(
            f'✅ Objective {current_obj_key} complete',
            'objective_complete'
        )
        
        # Clear state
        clear_workflow_state(task_id)
        
        yield {
            'type': 'result',
            'response': f"🎉 Objective {current_obj_key} completed! All {total_steps} steps finished successfully.",
            'result_type': 'objective_complete'
        }


# if __name__ == "__main__":
#     # Test detect_user_intent function
#     message = "Fit a random forest model to predict poverty"

#     print("\n" + "=" * 70)
#     print("TESTING detect_user_intent FUNCTION")
#     print("=" * 70)

#     result = detect_user_intent(message)

#     print(f"\nMessage: '{message}'")
#     print(f"\nIntent:     {result['intent']}")
#     print(f"Confidence: {result['confidence']}")
#     print(f"Reasoning:  {result['reasoning']}")

#     print("\n" + "=" * 70)
#     print("TESTING COMPLETE")
#     print("=" * 70 + "\n")

