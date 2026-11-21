"""
Code snippet to add to agm_demo.ipynb to register the ModelingAgent.

Add this to the import cell (after the existing imports):
"""

# Import the ModelingAgent
from agents.modeling_agent import ModelingAgent

# Then add this registration call to the cell where you register agents
# (after registering DataAgent and RQUnderstandingAgent):

# Register ModelingAgent
task_manager.register_agent(
    "ModelingAgent",
    ModelingAgent(),
    description="Selects, configures, and executes appropriate statistical and machine learning models from the system's model base using intelligent model selection based on task requirements."
)

"""
This will enable the TaskManager to:
1. Route statistical modeling steps to the ModelingAgent
2. Automatically select top 3 relevant models based on step descriptions
3. Use FAISS-powered semantic search to find the best sklearn models for each task

Example steps that will route to ModelingAgent:
- "Conduct paired t-tests to assess changes in visitation"
- "Perform multivariate regression analysis"
- "Conduct ANOVA to test differences among demographic groups"
"""