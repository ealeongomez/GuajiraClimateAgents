# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
ClimateGuajira Agent Package.

This package provides a LangGraph agent specialized in wind energy
and climate data for La Guajira, Colombia.

The agent uses RAG (Retrieval Augmented Generation) to answer questions
based on the Atlas Eólico de Colombia.

Example usage:
    >>> from src.agents.climate_guajira import graph
    >>> response = graph.invoke({"messages": [("user", "¿Cuál es el potencial eólico de La Guajira?")]})
    >>> print(response["messages"][-1].content)

For LangGraph Studio:
    The `graph` object is exported and can be referenced in langgraph.json
"""

from src.agents.climate_guajira.graph import graph, create_graph
from src.agents.climate_guajira.configuration import Configuration
from src.agents.climate_guajira.state import AgentState
from src.agents.climate_guajira.tools import create_tools
from src.agents.climate_guajira.prompts import SYSTEM_PROMPT, RAG_PROMPT

__all__ = [
    "graph",
    "create_graph",
    "Configuration",
    "AgentState",
    "create_tools",
    "SYSTEM_PROMPT",
    "RAG_PROMPT",
]

