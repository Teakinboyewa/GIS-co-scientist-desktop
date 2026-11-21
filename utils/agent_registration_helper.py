from agents.task_manager import TaskManager
from agents.data_agent import DataAgent
from agents.rq_understanding_agent import RQUnderstandingAgent
from agents.modeling_agent import ModelingAgent
import agents.task_manager.research_plan_generator as research_plan_generator
import agents.task_manager.objective_executor as objective_executor

task_manager = TaskManager()
data_agent = DataAgent()
rq_agent = RQUnderstandingAgent()

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

    print("Registered agents: RQUnderstandingAgent, DataAgent, ModelingAgent")

