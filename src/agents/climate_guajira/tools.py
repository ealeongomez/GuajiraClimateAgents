# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Tools for ClimateGuajira agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.vector_store import VectorStore
from src.agents.climate_guajira.prompts import RAG_PROMPT
from src.agents.climate_guajira.configuration import Configuration


def get_vector_store(config: Configuration) -> VectorStore:
    """Initialize and return the vector store.
    
    Args:
        config: Agent configuration.
        
    Returns:
        Configured VectorStore instance.
    """
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    return VectorStore(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(
            PROJECT_ROOT / "data" / "embeddings" / config.collection_name
        )
    )


def create_tools(config: Configuration | None = None) -> List:
    """Create and return the list of tools for the agent.
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        
    Returns:
        List of tool functions.
    """
    if config is None:
        config = Configuration()
    
    # Initialize vector store
    vector_store = get_vector_store(config)
    rag_llm = config.get_model()
    
    @tool
    def consultar_atlas_eolico(pregunta: str) -> str:
        """Consulta el Atlas Eólico de Colombia sobre energía eólica.
        
        Usa esta herramienta para preguntas sobre:
        - Potencial eólico en Colombia y La Guajira
        - Velocidad y dirección del viento
        - Zonas aptas para parques eólicos
        - Capacidad de generación eólica
        - Mapas y datos del recurso eólico
        
        Args:
            pregunta: Pregunta sobre energía eólica en Colombia.
        
        Returns:
            Respuesta basada en el Atlas Eólico de Colombia.
        """
        # Retrieval
        docs = vector_store.similarity_search(pregunta, k=config.retrieval_k)
        
        if not docs:
            return "No encontré información relevante en el Atlas Eólico."
        
        # Format context with page references
        context = "\n\n".join(
            f"[Página {doc.metadata.get('page', '?')}]: {doc.page_content}"
            for doc in docs
        )
        
        # Generation with RAG chain
        chain = RAG_PROMPT | rag_llm | StrOutputParser()
        return chain.invoke({"context": context, "question": pregunta})
    
    @tool
    def buscar_documentos(query: str) -> str:
        """Busca documentos relevantes en el Atlas Eólico sin generar respuesta.
        
        Usa esta herramienta cuando necesites ver los documentos originales
        sin procesamiento adicional.
        
        Args:
            query: Términos de búsqueda.
        
        Returns:
            Fragmentos de documentos encontrados con referencias de página.
        """
        docs = vector_store.similarity_search(query, k=config.retrieval_k)
        
        if not docs:
            return "No se encontraron documentos relevantes."
        
        results = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', '?')
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            results.append(f"📄 Resultado {i} (Página {page}):\n{content}")
        
        return "\n\n---\n\n".join(results)
    
    return [consultar_atlas_eolico, buscar_documentos]

