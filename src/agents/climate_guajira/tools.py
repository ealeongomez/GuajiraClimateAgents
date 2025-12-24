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

import pymssql
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
    db_config = config.get_db_config()
    
    # ================================================================
    # RAG TOOLS - ATLAS EÓLICO
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
    
    # ================================================================
    # DATABASE TOOLS - DATOS HISTÓRICOS CLIMÁTICOS
    # ================================================================
    
    @tool
    def obtener_estadisticas_municipio(municipio: str) -> str:
        """Obtiene estadísticas climáticas históricas de un municipio de La Guajira.
        
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
            conn = pymssql.connect(**db_config)
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
            conn = pymssql.connect(**db_config)
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
            conn = pymssql.connect(**db_config)
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
    # OPTIMIZED TOOLS - COLUMNAS TEMPORALES (year, month, day, hour)
    # ================================================================
    
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
            conn = pymssql.connect(**db_config)
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
        Útil para identificar patrones diarios y optimizar generación eólica.
        
        Args:
            municipio: Nombre del municipio (ej: 'riohacha', 'maicao').
            anio: Año a consultar (ej: 2024).
            mes: Mes a consultar (1-12).
        
        Returns:
            Estadísticas por hora del día para el mes especificado.
        """
        try:
            conn = pymssql.connect(**db_config)
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
            for row in rows[:8]:  # Mostrar solo primeras 8 horas
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
            conn = pymssql.connect(**db_config)
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
    
    # Return all tools
    return [
        # RAG tools
        consultar_atlas_eolico,
        buscar_documentos,
        
        # Database basic tools
        obtener_estadisticas_municipio,
        comparar_municipios_viento,
        listar_municipios_disponibles,
        
        # Optimized temporal tools
        obtener_estadisticas_por_mes,
        obtener_estadisticas_por_hora,
        comparar_anios,
    ]

