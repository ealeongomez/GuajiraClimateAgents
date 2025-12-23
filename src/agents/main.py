# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Agente LangGraph con RAG Tool sobre el Atlas Eólico."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.vector_store import VectorStore

load_dotenv()

# ================================================================
# Configurar Vector Store
# ================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = VectorStore(
    collection_name="Atlas_eolico_Colombia",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_ROOT / "data" / "embeddings" / "Atlas_eolico_Colombia")
)

# ================================================================
# Configurar RAG Chain
# ================================================================
rag_prompt = ChatPromptTemplate.from_template("""
Eres un experto en energía eólica. Responde basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, indica que no está disponible en el Atlas.

Contexto del Atlas Eólico de Colombia:
{context}

Pregunta: {question}

Respuesta detallada:
""")

rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================================================
# Tool RAG
# ================================================================
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
    docs = vector_store.similarity_search(pregunta, k=4)
    
    if not docs:
        return "No encontré información relevante en el Atlas Eólico."
    
    # Formatear contexto
    context = "\n\n".join(
        f"[Pág. {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in docs
    )
    
    # Generation
    chain = rag_prompt | rag_llm | StrOutputParser()
    return chain.invoke({"context": context, "question": pregunta})


# ================================================================
# Crear Agente
# ================================================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=model,
    tools=[consultar_atlas_eolico],
)

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print(f"\n🌬️ Agente Atlas Eólico de Colombia")
    print(f"📚 Documentos cargados: {vector_store.get_collection_count()}")
    print("=" * 50)
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        question = input("❓ Pregunta: ").strip()
        
        if question.lower() in ("salir", "exit", "q"):
            print("👋 ¡Hasta luego!")
            break
        
        if question:
            response = agent.invoke({"messages": [("user", question)]})
            answer = response["messages"][-1].content
            print(f"\n💬 {answer}\n")