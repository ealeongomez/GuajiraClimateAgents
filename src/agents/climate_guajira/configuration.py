# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Configuration for ClimateGuajira agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_openai import ChatOpenAI


@dataclass(kw_only=True)
class Configuration:
    """Configuration for the ClimateGuajira agent.
    
    This class defines all configurable parameters for the agent,
    including model settings and retrieval parameters.
    """
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    """The name of the language model to use."""
    
    temperature: float = 0.0
    """Temperature for model generation (0 = deterministic)."""
    
    # Retrieval configuration
    retrieval_k: int = 4
    """Number of documents to retrieve from vector store."""
    
    # Vector store configuration
    collection_name: str = "Atlas_eolico_Colombia"
    """Name of the ChromaDB collection."""
    
    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model to use."""
    
    def get_model(self) -> ChatOpenAI:
        """Get a configured ChatOpenAI instance.
        
        Returns:
            Configured ChatOpenAI model.
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
        )

