# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Graph definition for ClimateGuajira agent."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from src.agents.climate_guajira.configuration import Configuration
from src.agents.climate_guajira.tools import create_tools
from src.agents.climate_guajira.prompts import SYSTEM_PROMPT

# Load environment variables
load_dotenv()


def create_graph(config: Configuration | None = None, checkpointer=None):
    """Create and return the ClimateGuajira agent graph.
    
    This function builds the LangGraph agent with:
    - Configured LLM model
    - RAG tools for Atlas Eólico
    - Database tools for historical data
    - Visualization tools for plots
    - System prompt for climate expertise
    - Optional checkpointer for state persistence
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        checkpointer: Optional checkpointer for state persistence (e.g., SqliteSaver).
                     Required for maintaining conversation history across sessions.
        
    Returns:
        Compiled LangGraph agent ready for invocation.
    
    Example:
        >>> # Without persistence
        >>> graph = create_graph()
        >>> response = graph.invoke({"messages": [...]})
        
        >>> # With persistence (Telegram bot)
        >>> from langgraph.checkpoint.sqlite import SqliteSaver
        >>> checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        >>> graph = create_graph(checkpointer=checkpointer)
        >>> config = {"configurable": {"thread_id": "user_123"}}
        >>> response = graph.invoke({"messages": [...]}, config=config)
    """
    if config is None:
        config = Configuration()
    
    # Get model and tools
    model = config.get_model()
    tools = create_tools(config)
    
    # Create the ReAct agent with optional checkpointer
    graph = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    return graph


# Default agent instance for LangGraph Studio
graph = create_graph()