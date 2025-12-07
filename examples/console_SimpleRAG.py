# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Ejemplo de RAG usando la base de datos vectorial como Tool."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.vector_store import VectorStore

load_dotenv()

# ================================================================
# Inicializar componentes
# ================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = VectorStore(collection_name="climate_docs", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================================================
# Definir Tool
# ================================================================
@tool
def search_documents(query: str) -> str:
    """Busca información relevante en la base de datos de documentos sobre clima.
    
    Args:
        query: La consulta de búsqueda para encontrar documentos relevantes.
    
    Returns:
        Contenido de los documentos más relevantes encontrados.
    """
    docs = store.similarity_search(query, k=4)
    if not docs:
        return "No se encontraron documentos relevantes."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ================================================================
# Crear agente con la tool
# ================================================================
tools = [search_documents]
agent = create_react_agent(llm, tools)

print(f"📚 Base de datos: {store.get_collection_count()} documentos")
print("🔧 Tool disponible: search_documents")
print("Escribe 'salir' para terminar.\n")

# ================================================================
# Main loop
# ================================================================
while True:
    question = input("❓ Pregunta: ").strip()

    if question.lower() in ("salir", "exit", "q"):
        break

    if question:
        response = agent.invoke({"messages": [("user", question)]})
        answer = response["messages"][-1].content
        print(f"\n💬 {answer}\n")
