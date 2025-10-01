# afiyapal_multi_tool_agent/agent.py

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
# This import executes the top-level code in rag_tool.py (loading the index)
# and brings the function 'first_aid_rag_search' into scope.
from .rag_tool import first_aid_rag_search 

# =================================================================
# 1. First Aid Expert Agent (The Sub-Agent)
# =================================================================

# Define the First Aid Sub-Agent
first_aid_expert_agent = LlmAgent(
    name="FirstAidExpertAgent",
    model="gemini-2.5-flash", 
    description=(
        "A highly specialized agent for providing detailed, professional, and "
        "reference-based first aid advice for common injuries. You MUST use "
        "the 'first_aid_rag_search' tool to find specific, grounded information "
        "from the provided medical books before formulating your answer. "
    ),
    instruction=(
        "Your sole role is to provide step-by-step, professional first aid instructions. "
        "ALWAYS follow the procedure from the RETRIEVED KNOWLEDGE. If the search returns "
        "context, use that context (the procedures from the books) as the basis of your reply. "
        "Present the information found in a clear, numbered, professional format. "
        "If the search fails or returns empty context, state clearly that you are using general knowledge."
    ),
    # The agent is connected directly to the imported function
    tools=[first_aid_rag_search],
)


# =================================================================
# 2. Health Coordinator Agent (The Root Agent)
# =================================================================

# Wrap the sub-agent as a tool for the root agent
first_aid_tool = AgentTool(first_aid_expert_agent)


# Define the Health Coordinator (Root) Agent
root_agent = LlmAgent(
    name="HealthCoordinatorAgent",
    model="gemini-2.5-flash",
    description=(
        "The primary professional Health Assistant. This agent handles all mental "
        "health guidance, counseling, and triage of health questions."
    ),
    instruction=(
        "You are a compassionate, professional Health Assistant. "
        "1. For mental health and counseling questions, provide empathetic, non-diagnostic guidance, always "
        "recommending consulting a professional for serious issues. "
        "2. For ALL first aid or injury-related questions (e.g., 'cut finger', 'sprained ankle', 'burn'), you MUST delegate the question to the 'FirstAidExpertAgent' tool. "
        "3. Maintain a professional, detailed, and non-alarmist tone at all times."
    ),
    # Connect the sub-agent as a tool
    tools=[first_aid_tool],
)