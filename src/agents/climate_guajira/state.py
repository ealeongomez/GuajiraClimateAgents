# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""State definitions for ClimateGuajira agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class AgentState:
    """State for the ClimateGuajira agent.
    
    This state is passed between nodes in the graph and maintains
    the conversation history and any additional context.
    
    Attributes:
        messages: The conversation messages with add_messages reducer.
    """
    
    messages: Annotated[Sequence[BaseMessage], add_messages] = field(
        default_factory=list
    )
    """Messages in the conversation. Uses add_messages reducer to append."""

