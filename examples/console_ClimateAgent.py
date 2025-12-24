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

# Nota: La tabla climate_observations incluye columnas temporales optimizadas:
# year, month, day, hour - extraídas automáticamente de datetime
# Estas columnas están indexadas para consultas eficientes

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


@tool
def obtener_estadisticas_por_mes(municipio: str, anio: int) -> str:
    """Obtiene estadísticas climáticas mensuales de un municipio para un año específico.
    
    Usa las columnas temporales optimizadas (year, month) para consultas eficientes.
    
    Args:
        municipio: Nombre del municipio (ej: 'riohacha', 'maicao').
        anio: Año a consultar (ej: 2024, 2023).
    
    Returns:
        Estadísticas mensuales del municipio para el año especificado.
    """
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                month,
                COUNT(*) as registros,
                AVG(wind_speed_10m) as velocidad_promedio_viento,
                AVG(temperature_2m) as temperatura_promedio,
                SUM(precipitation) as precipitacion_total
            FROM climate_observations
            WHERE municipio = %s AND year = %s
            GROUP BY month
            ORDER BY month
        """
        
        cursor.execute(query, (municipio.lower().replace(' ', '_'), anio))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return f"No se encontraron datos para {municipio} en el año {anio}"
        
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        result = f"📅 Estadísticas mensuales de {municipio.title()} - {anio}:\n\n"
        for row in rows:
            mes_num = row[0]
            mes_nombre = meses[mes_num - 1] if 1 <= mes_num <= 12 else str(mes_num)
            result += f"• {mes_nombre}: viento={row[2]:.2f} km/h, "
            result += f"temp={row[3]:.2f}°C, precip={row[4]:.2f}mm\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def obtener_estadisticas_por_hora(municipio: str, anio: int, mes: int) -> str:
    """Obtiene estadísticas climáticas por hora del día para un mes específico.
    
    Usa la columna temporal 'hour' para análisis por hora del día (0-23).
    
    Args:
        municipio: Nombre del municipio (ej: 'riohacha', 'maicao').
        anio: Año a consultar (ej: 2024).
        mes: Mes a consultar (1-12).
    
    Returns:
        Estadísticas por hora del día para el mes especificado.
    """
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                hour,
                COUNT(*) as registros,
                AVG(wind_speed_10m) as velocidad_promedio_viento,
                AVG(temperature_2m) as temperatura_promedio
            FROM climate_observations
            WHERE municipio = %s AND year = %s AND month = %s
            GROUP BY hour
            ORDER BY hour
        """
        
        cursor.execute(query, (municipio.lower().replace(' ', '_'), anio, mes))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return f"No se encontraron datos para {municipio} en {mes}/{anio}"
        
        meses = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        mes_nombre = meses[mes] if 1 <= mes <= 12 else str(mes)
        
        result = f"🕐 Estadísticas por hora - {municipio.title()} ({mes_nombre} {anio}):\n\n"
        
        # Mostrar resumen de horas pico y valle
        max_wind = max(rows, key=lambda x: x[2])
        min_wind = min(rows, key=lambda x: x[2])
        
        result += f"⬆️  Hora con más viento: {max_wind[0]:02d}:00 ({max_wind[2]:.2f} km/h)\n"
        result += f"⬇️  Hora con menos viento: {min_wind[0]:02d}:00 ({min_wind[2]:.2f} km/h)\n\n"
        
        result += "Promedios por hora:\n"
        for row in rows[:8]:  # Mostrar solo primeras 8 horas para no saturar
            result += f"• {row[0]:02d}:00 - viento: {row[2]:.2f} km/h, temp: {row[3]:.2f}°C\n"
        
        if len(rows) > 8:
            result += f"\n... ({len(rows) - 8} horas más)\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def comparar_anios(municipio: str, anio1: int, anio2: int) -> str:
    """Compara estadísticas climáticas entre dos años para un municipio.
    
    Usa la columna temporal 'year' para comparaciones eficientes entre años.
    
    Args:
        municipio: Nombre del municipio (ej: 'riohacha', 'maicao').
        anio1: Primer año a comparar.
        anio2: Segundo año a comparar.
    
    Returns:
        Comparación de estadísticas entre los dos años.
    """
    try:
        conn = pymssql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                year,
                COUNT(*) as registros,
                AVG(wind_speed_10m) as velocidad_promedio_viento,
                MAX(wind_speed_10m) as velocidad_maxima_viento,
                AVG(temperature_2m) as temperatura_promedio,
                SUM(precipitation) as precipitacion_total
            FROM climate_observations
            WHERE municipio = %s AND year IN (%s, %s)
            GROUP BY year
            ORDER BY year
        """
        
        cursor.execute(query, (municipio.lower().replace(' ', '_'), anio1, anio2))
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 2:
            return f"No hay suficientes datos para comparar {anio1} y {anio2} en {municipio}"
        
        result = f"📊 Comparación {anio1} vs {anio2} - {municipio.title()}:\n\n"
        
        data = {row[0]: row for row in rows}
        
        for year in [anio1, anio2]:
            if year in data:
                row = data[year]
                result += f"Año {year}:\n"
                result += f"  • Registros: {row[1]:,}\n"
                result += f"  • Viento promedio: {row[2]:.2f} km/h\n"
                result += f"  • Viento máximo: {row[3]:.2f} km/h\n"
                result += f"  • Temperatura promedio: {row[4]:.2f}°C\n"
                result += f"  • Precipitación total: {row[5]:.2f} mm\n\n"
        
        # Calcular diferencias
        if anio1 in data and anio2 in data:
            diff_viento = data[anio2][2] - data[anio1][2]
            diff_temp = data[anio2][4] - data[anio1][4]
            
            result += "Diferencias:\n"
            result += f"  • Viento: {diff_viento:+.2f} km/h\n"
            result += f"  • Temperatura: {diff_temp:+.2f}°C\n"
        
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
    # Herramientas básicas
    obtener_estadisticas_municipio,
    comparar_municipios_viento,
    listar_municipios_disponibles,
    
    # Herramientas con columnas temporales optimizadas
    obtener_estadisticas_por_mes,
    obtener_estadisticas_por_hora,
    comparar_anios,
    
    # Herramienta RAG
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
    print(f"{Fore.GREEN}  🔍 Comparar municipios y años")
    print(f"{Fore.GREEN}  📈 Obtener estadísticas detalladas")
    print(f"{Fore.GREEN}  📅 Analizar datos por mes y hora del día")
    print(f"{Fore.GREEN}  ⚡ Consultas optimizadas con índices temporales")
    print(f"\n{Fore.YELLOW}Ejemplos de preguntas:")
    print(f"{Fore.YELLOW}  • ¿Cómo fue el viento en Riohacha durante 2024?")
    print(f"{Fore.YELLOW}  • Compara el viento entre 2023 y 2024 en Maicao")
    print(f"{Fore.YELLOW}  • ¿A qué hora del día hay más viento en Manaure?")
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
        
        # Verificar columnas temporales
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'climate_observations' 
            AND COLUMN_NAME IN ('year', 'month', 'day', 'hour')
        """)
        temp_cols = cursor.fetchone()[0]
        
        conn.close()
        print(f"{Fore.CYAN}✅ Conectado a base de datos: {count:,} registros")
        if temp_cols == 4:
            print(f"{Fore.CYAN}✅ Columnas temporales optimizadas detectadas")
        else:
            print(f"{Fore.YELLOW}⚠️  Advertencia: No se detectaron columnas temporales")
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

