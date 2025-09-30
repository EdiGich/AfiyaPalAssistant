# agent.py

from google.adk.agents import Agent, LlmAgent
from google.adk.tools import AgentTool
from rag_tool import first_aid_rag_search # <--- Import the RAG tool

# =================================================================
# 1. First Aid Expert Agent (The Sub-Agent)
# =================================================================

# Define the First Aid Sub-Agent
first_aid_expert_agent = LlmAgent(
    name="FirstAidExpertAgent",
    model="gemini-2.0-flash", 
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
        "Present the information found in a clear, numbered, professional format."
    ),
    # Connect the RAG function as a tool
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
    model="gemini-2.0-flash",
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

# Keep the default agent definition for running (ADK uses the last defined Agent as default if no name is specified)
# You can remove the old get_weather and get_current_time functions as they are no longer used.