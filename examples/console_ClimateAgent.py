# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Agente de clima con acceso a base de datos SQL y vectorial."""

import os
import sys
from pathlib import Path

import pymssql
from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.vector_store import VectorStore

# Inicializar
init(autoreset=True)
load_dotenv(PROJECT_ROOT / ".env")

# ================================================================
# CONFIGURACIÓN
# ================================================================
DB_CONFIG = {
    'server': os.getenv('DB_SERVER', 'localhost'),
    'port': os.getenv('DB_PORT', '1433'),
    'user': os.getenv('DB_USER', 'sa'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'ClimateDB')
}

# ================================================================
# INICIALIZAR COMPONENTES
# ================================================================
# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Vector Store (Atlas Eólico)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = VectorStore(
    collection_name="Atlas_eolico_Colombia",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_ROOT / "data" / "embeddings" / "Atlas_eolico_Colombia")
)

# RAG Prompt
RAG_PROMPT = ChatPromptTemplate.from_template("""
Eres un experto en energía eólica. Responde basándote en el contexto proporcionado.

Contexto del Atlas Eólico:
{context}

Pregunta: {question}

Respuesta detallada:
""")


# ================================================================
# TOOLS - BASE DE DATOS SQL
# ================================================================
@tool
def obtener_estadisticas_municipio(municipio: str) -> str:
    """Obtiene estadísticas climáticas de un municipio de La Guajira.
    
    Usa esta herramienta para obtener promedios, máximos y mínimos de
    variables climáticas como velocidad del viento, temperatura, etc.
    
    Municipios disponibles: albania, barrancas, distraccion, el_molino,
    fonseca, hatonuevo, la_jagua_del_pilar, maicao, manaure, mingueo,
    riohacha, san_juan_del_cesar, uribia.
    
    Args:
        municipio: Nombre del municipio (ej: 'riohacha', 'maicao').
    
    Returns:
        Estadísticas del municipio.
    """
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                COUNT(*) as total_registros,
                MIN(datetime) as fecha_inicio,
                MAX(datetime) as fecha_fin,
                AVG(wind_speed_10m) as velocidad_promedio_viento,
                MAX(wind_speed_10m) as velocidad_maxima_viento,
                AVG(temperature_2m) as temperatura_promedio,
                AVG(precipitation) as precipitacion_promedio
            FROM climate_observations
            WHERE municipio = %s
        """
        
        cursor.execute(query, (municipio.lower().replace(' ', '_'),))
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] == 0:
            return f"No se encontraron datos para el municipio: {municipio}"
        
        return f"""
📊 Estadísticas de {municipio.title()}:
• Total de registros: {row[0]:,}
• Periodo: {row[1]} a {row[2]}
• Velocidad promedio del viento: {row[3]:.2f} km/h
• Velocidad máxima del viento: {row[4]:.2f} km/h
• Temperatura promedio: {row[5]:.2f} °C
• Precipitación promedio: {row[6]:.2f} mm
"""
    except Exception as e:
        return f"Error al consultar base de datos: {str(e)}"


@tool
def comparar_municipios_viento(municipios: str) -> str:
    """Compara la velocidad del viento entre varios municipios.
    
    Args:
        municipios: Municipios separados por comas (ej: 'riohacha,maicao,uribia').
    
    Returns:
        Comparación de velocidad del viento.
    """
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        munis = [m.strip().lower().replace(' ', '_') for m in municipios.split(',')]
        placeholders = ', '.join(['%s'] * len(munis))
        
        query = f"""
            SELECT 
                municipio,
                AVG(wind_speed_10m) as promedio,
                MIN(wind_speed_10m) as minimo,
                MAX(wind_speed_10m) as maximo
            FROM climate_observations
            WHERE municipio IN ({placeholders})
            GROUP BY municipio
            ORDER BY promedio DESC
        """
        
        cursor.execute(query, tuple(munis))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "No se encontraron datos para los municipios especificados."
        
        result = "🌬️ Comparación de velocidad del viento:\n\n"
        for row in rows:
            result += f"• {row[0].title()}: promedio={row[1]:.2f} km/h, "
            result += f"min={row[2]:.2f}, max={row[3]:.2f}\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def listar_municipios_disponibles() -> str:
    """Lista todos los municipios disponibles en la base de datos.
    
    Returns:
        Lista de municipios con cantidad de registros.
    """
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                municipio,
                COUNT(*) as registros,
                AVG(wind_speed_10m) as viento_promedio
            FROM climate_observations
            GROUP BY municipio
            ORDER BY viento_promedio DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        result = "📍 Municipios disponibles en La Guajira:\n\n"
        for row in rows:
            result += f"• {row[0].title()}: {row[1]:,} registros "
            result += f"(viento promedio: {row[2]:.2f} km/h)\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# ================================================================
# TOOLS - BASE VECTORIAL (ATLAS EÓLICO)
# ================================================================
@tool
def consultar_atlas_eolico(pregunta: str) -> str:
    """Consulta el Atlas Eólico de Colombia sobre energía eólica.
    
    Usa esta herramienta para preguntas sobre:
    - Potencial eólico en Colombia y La Guajira
    - Zonas aptas para parques eólicos
    - Capacidad de generación eólica
    - Mapas y datos del recurso eólico
    - Información técnica sobre energía eólica
    
    Args:
        pregunta: Pregunta sobre energía eólica.
    
    Returns:
        Respuesta basada en el Atlas Eólico de Colombia.
    """
    try:
        # Retrieval
        docs = vector_store.similarity_search(pregunta, k=4)
        
        if not docs:
            return "No encontré información relevante en el Atlas Eólico."
        
        # Format context
        context = "\n\n".join(
            f"[Página {doc.metadata.get('page', '?')}]: {doc.page_content}"
            for doc in docs
        )
        
        # Generation
        chain = RAG_PROMPT | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": pregunta})
    except Exception as e:
        return f"Error al consultar Atlas Eólico: {str(e)}"


# ================================================================
# CREAR AGENTE
# ================================================================
tools = [
    obtener_estadisticas_municipio,
    comparar_municipios_viento,
    listar_municipios_disponibles,
    consultar_atlas_eolico
]

agent = create_react_agent(llm, tools)

# ================================================================
# INTERFAZ DE CONSOLA
# ================================================================
def print_header():
    """Imprime el encabezado del chatbot."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}🌬️  AGENTE DE CLIMA - LA GUAJIRA")
    print(f"{Fore.CYAN}{'='*70}")
    print(f"\n{Fore.GREEN}Capacidades:")
    print(f"{Fore.GREEN}  📊 Consultar base de datos con datos históricos climáticos")
    print(f"{Fore.GREEN}  📚 Consultar Atlas Eólico de Colombia")
    print(f"{Fore.GREEN}  🔍 Comparar municipios")
    print(f"{Fore.GREEN}  📈 Obtener estadísticas detalladas")
    print(f"\n{Fore.YELLOW}Escribe 'salir' para terminar.\n")


def main():
    """Función principal del chatbot."""
    print_header()
    
    # Verificar conexión a DB
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM climate_observations")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"{Fore.CYAN}✅ Conectado a base de datos: {count:,} registros")
    except Exception as e:
        print(f"{Fore.RED}⚠️  Error de conexión a DB: {e}")
    
    # Verificar vector store
    try:
        doc_count = vector_store.get_collection_count()
        print(f"{Fore.CYAN}✅ Atlas Eólico cargado: {doc_count} documentos\n")
    except Exception as e:
        print(f"{Fore.RED}⚠️  Error al cargar Atlas Eólico: {e}\n")
    
    # Loop principal
    while True:
        question = input(f"{Fore.GREEN}❓ Pregunta: {Style.RESET_ALL}").strip()
        
        if question.lower() in ("salir", "exit", "q", "quit"):
            print(f"\n{Fore.YELLOW}👋 ¡Hasta luego!\n")
            break
        
        if not question:
            continue
        
        try:
            print(f"\n{Fore.CYAN}🤔 Pensando...\n")
            response = agent.invoke({"messages": [("user", question)]})
            answer = response["messages"][-1].content
            print(f"{Fore.BLUE}💬 {answer}{Style.RESET_ALL}\n")
            print(f"{Fore.CYAN}{'-'*70}\n")
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()

