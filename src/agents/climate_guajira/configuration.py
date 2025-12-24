# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Configuration for ClimateGuajira agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Dict

from langchain_openai import ChatOpenAI


@dataclass(kw_only=True)
class Configuration:
    """Configuration for the ClimateGuajira agent.
    
    This class defines all configurable parameters for the agent,
    including model settings, retrieval parameters, and database connection.
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
    
    # Database configuration
    db_server: str = os.getenv('DB_SERVER', 'localhost')
    """SQL Server hostname or IP address."""
    
    db_port: str = os.getenv('DB_PORT', '1433')
    """SQL Server port."""
    
    db_user: str = os.getenv('DB_USER', 'sa')
    """Database username."""
    
    db_password: str = os.getenv('DB_PASSWORD', '')
    """Database password."""
    
    db_name: str = os.getenv('DB_NAME', 'ClimateDB')
    """Database name."""
    
    def get_model(self) -> ChatOpenAI:
        """Get a configured ChatOpenAI instance.
        
        Returns:
            Configured ChatOpenAI model.
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
        )
    
    def get_db_config(self) -> Dict[str, str]:
        """Get database configuration dictionary for pymssql.
        
        Returns:
            Dictionary with database connection parameters.
        """
        return {
            'server': self.db_server,
            'port': self.db_port,
            'user': self.db_user,
            'password': self.db_password,
            'database': self.db_name
        }

