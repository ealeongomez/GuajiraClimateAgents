# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Ejemplo de RAG usando la base de datos vectorial como Tool."""

import sys
from pathlib import Path

from colorama import init, Fore, Style
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.vector_store import VectorStore

init(autoreset=True)  # Inicializar colorama
load_dotenv()

# ================================================================
# Inicializar componentes
# ================================================================
# Embeddings locales con Ollama (nomic-embed-text o mxbai-embed-large)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
store = VectorStore(collection_name="climate_docs_ollama", embedding_function=embeddings)

# LLM local con Ollama (opciones: llama3.2:1b, deepseek-r1:1.5b, llama3.2)
llm = ChatOllama(
    model="llama3.2:1b",  # Modelo rápido
    temperature=0,
    num_predict=256,
    num_ctx=2048
)


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

print(f"{Fore.CYAN}📚 Base de datos: {store.get_collection_count()} documentos")
print(f"{Fore.CYAN}🔧 Tool disponible: search_documents")
print(f"{Fore.YELLOW}Escribe 'salir' para terminar.\n")

# ================================================================
# Main loop
# ================================================================
while True:
    question = input(f"{Fore.GREEN}❓ Pregunta: {Style.RESET_ALL}").strip()

    if question.lower() in ("salir", "exit", "q"):
        print(f"{Fore.YELLOW}👋 ¡Hasta luego!")
        break

    if question:
        response = agent.invoke({"messages": [("user", question)]})
        answer = response["messages"][-1].content
        print(f"\n{Fore.BLUE}💬 {answer}{Style.RESET_ALL}\n")
