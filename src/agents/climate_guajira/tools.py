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
from datetime import datetime

import pymssql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configurar matplotlib para no usar GUI
matplotlib.use('Agg')


def _save_plot(fig, filename: str, project_root: Path) -> str:
    """Guarda la figura en disco.
    
    Args:
        fig: Figura de matplotlib
        filename: Nombre del archivo
        project_root: Ruta raíz del proyecto
    
    Returns:
        Ruta relativa del archivo guardado
    """
    filepath = project_root / "src" / "agents" / "climate_guajira" / "images" / filename
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return str(filepath.relative_to(project_root))

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
        """Consulta el Atlas Eólico sobre potencial eólico, zonas aptas y capacidad de generación.
        
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
        """Busca documentos originales en el Atlas Eólico con referencias de página.
        
        Args:
            query: Términos de búsqueda.
        
        Returns:
            Fragmentos de documentos encontrados.
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
        """Estadísticas climáticas de un municipio: viento, temperatura, precipitación.
        
        Args:
            municipio: Nombre del municipio (ej: 'riohacha', 'maicao', 'uribia').
        
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
        """Compara velocidad del viento entre municipios (separados por comas).
        
        Args:
            municipios: Municipios separados por comas.
        
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
        """Lista los 13 municipios con cantidad de registros y viento promedio.
        
        Returns:
            Lista de municipios disponibles.
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
        """Estadísticas mensuales (12 meses) de un municipio para un año.
        
        Args:
            municipio: Nombre del municipio.
            anio: Año a consultar (ej: 2024).
        
        Returns:
            Estadísticas mensuales del año.
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
        """Estadísticas por hora del día (0-23) para un mes y año.
        
        Args:
            municipio: Nombre del municipio.
            anio: Año (ej: 2024).
            mes: Mes (1-12).
        
        Returns:
            Estadísticas horarias del mes.
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
        """Compara estadísticas climáticas entre dos años.
        
        Args:
            municipio: Nombre del municipio.
            anio1: Primer año.
            anio2: Segundo año.
        
        Returns:
            Comparación entre años.
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
    
    # ================================================================
    # VISUALIZATION TOOLS - GRÁFICAS Y ANÁLISIS VISUAL
    # ================================================================
    
    @tool
    def graficar_serie_temporal_municipio(municipio: str, fecha_inicio: str, fecha_fin: str) -> str:
        """Gráfica de serie temporal de viento (formato YYYY-MM-DD).
        
        Args:
            municipio: Nombre del municipio.
            fecha_inicio: Fecha inicio (YYYY-MM-DD).
            fecha_fin: Fecha fin (YYYY-MM-DD).
        
        Returns:
            Resumen y ruta de imagen.
        """
        try:
            conn = pymssql.connect(**db_config)
            
            query = """
                SELECT datetime, wind_speed_10m, temperature_2m
                FROM climate_observations
                WHERE municipio = %s
                AND datetime >= %s
                AND datetime <= %s
                ORDER BY datetime
            """
            
            df = pd.read_sql(query, conn, params=(
                municipio.lower().replace(' ', '_'), 
                fecha_inicio, 
                fecha_fin
            ))
            conn.close()
            
            if df.empty:
                return f"No se encontraron datos para {municipio} entre {fecha_inicio} y {fecha_fin}"
            
            # Crear gráfica
            fig, ax = plt.subplots(figsize=(14, 5))
            
            ax.plot(df['datetime'], df['wind_speed_10m'], 
                   color='#2E86AB', linewidth=0.8, alpha=0.8)
            ax.fill_between(df['datetime'], df['wind_speed_10m'], 
                           alpha=0.3, color='#2E86AB')
            
            ax.set_xlabel('Fecha', fontsize=12)
            ax.set_ylabel('Velocidad del Viento (km/h)', fontsize=12)
            ax.set_title(f'🌬️ Velocidad del Viento - {municipio.title()} ({fecha_inicio} a {fecha_fin})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"serie_temporal_{municipio}_{timestamp}.png"
            filepath_rel = _save_plot(fig, filename, PROJECT_ROOT)
            
            # Calcular estadísticas
            promedio = df['wind_speed_10m'].mean()
            maximo = df['wind_speed_10m'].max()
            minimo = df['wind_speed_10m'].min()
            
            return f"""✅ Gráfica generada

📊 Resumen:
• Municipio: {municipio.title()}
• Periodo: {fecha_inicio} a {fecha_fin}
• Registros: {len(df):,}
• Viento: promedio={promedio:.2f}, max={maximo:.2f}, min={minimo:.2f} km/h

📁 Imagen: {filepath_rel}
"""
        except Exception as e:
            return f"Error al generar gráfica: {str(e)}"
    
    @tool
    def graficar_comparacion_municipios(municipios: str, variable: str = "wind_speed_10m") -> str:
        """Gráfica de barras comparando municipios (separados por comas).
        
        Args:
            municipios: Municipios separados por comas.
            variable: wind_speed_10m, temperature_2m, o precipitation.
        
        Returns:
            Resumen y ruta de imagen.
        """
        try:
            conn = pymssql.connect(**db_config)
            
            munis = [m.strip().lower().replace(' ', '_') for m in municipios.split(',')]
            placeholders = ', '.join(['%s'] * len(munis))
            
            query = f"""
                SELECT 
                    municipio, 
                    AVG({variable}) as promedio
                FROM climate_observations
                WHERE municipio IN ({placeholders})
                GROUP BY municipio
                ORDER BY promedio DESC
            """
            
            df = pd.read_sql(query, conn, params=tuple(munis))
            conn.close()
            
            if df.empty:
                return "No se encontraron datos para los municipios especificados."
            
            # Crear gráfica
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
            bars = ax.barh(df['municipio'], df['promedio'], color=colors)
            
            # Labels según la variable
            labels_map = {
                'wind_speed_10m': ('Velocidad del Viento (km/h)', '🌬️'),
                'temperature_2m': ('Temperatura (°C)', '🌡️'),
                'precipitation': ('Precipitación (mm)', '💧')
            }
            
            label, emoji = labels_map.get(variable, ('Valor', '📊'))
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_title(f'{emoji} Comparación de {label} por Municipio', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Agregar valores en las barras
            for bar, val in zip(bars, df['promedio']):
                ax.text(val + (val * 0.01), bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparacion_{variable}_{timestamp}.png"
            filepath_rel = _save_plot(fig, filename, PROJECT_ROOT)
            
            # Construir resultado
            result = "✅ Comparación generada\n\n📊 Resultados:\n"
            for _, row in df.iterrows():
                result += f"• {row['municipio'].title()}: {row['promedio']:.2f}\n"
            
            result += f"\n📁 Imagen: {filepath_rel}"
            
            return result
        except Exception as e:
            return f"Error al generar comparación: {str(e)}"
    
    @tool
    def graficar_patron_horario(municipio: str, anio: int, mes: int) -> str:
        """Gráfica polar de 24 horas del viento (patrón diario).
        
        Args:
            municipio: Nombre del municipio.
            anio: Año (ej: 2024).
            mes: Mes (1-12).
        
        Returns:
            Análisis y ruta de imagen.
        """
        try:
            conn = pymssql.connect(**db_config)
            
            query = """
                SELECT 
                    hour,
                    AVG(wind_speed_10m) as velocidad_promedio
                FROM climate_observations
                WHERE municipio = %s AND year = %s AND month = %s
                GROUP BY hour
                ORDER BY hour
            """
            
            df = pd.read_sql(query, conn, params=(
                municipio.lower().replace(' ', '_'), 
                anio, 
                mes
            ))
            conn.close()
            
            if df.empty:
                return f"No se encontraron datos para {municipio} en {mes}/{anio}"
            
            # Crear gráfica polar
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Convertir horas a radianes
            theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            values = df['velocidad_promedio'].values
            
            # Cerrar el círculo
            theta = np.append(theta, theta[0])
            values = np.append(values, values[0])
            
            ax.plot(theta, values, color='#2E86AB', linewidth=2)
            ax.fill(theta, values, alpha=0.3, color='#2E86AB')
            
            # Configurar etiquetas
            ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
            ax.set_xticklabels([f'{h}h' for h in range(24)])
            
            meses = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
            mes_nombre = meses[mes] if 1 <= mes <= 12 else str(mes)
            
            ax.set_title(f'🕐 Patrón Horario del Viento\n{municipio.title()} - {mes_nombre} {anio}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patron_horario_{municipio}_{anio}_{mes}_{timestamp}.png"
            filepath_rel = _save_plot(fig, filename, PROJECT_ROOT)
            
            # Encontrar hora pico
            max_hour = df.loc[df['velocidad_promedio'].idxmax()]
            min_hour = df.loc[df['velocidad_promedio'].idxmin()]
            
            return f"""✅ Patrón horario generado

📊 Análisis {mes_nombre} {anio}:
• Municipio: {municipio.title()}
• Hora pico: {int(max_hour['hour']):02d}:00 ({max_hour['velocidad_promedio']:.2f} km/h)
• Hora valle: {int(min_hour['hour']):02d}:00 ({min_hour['velocidad_promedio']:.2f} km/h)
• Variación: {max_hour['velocidad_promedio'] - min_hour['velocidad_promedio']:.2f} km/h

📁 Imagen: {filepath_rel}
"""
        except Exception as e:
            return f"Error al generar patrón horario: {str(e)}"
    
    @tool
    def graficar_viento_temperatura(municipio: str, fecha_inicio: str, fecha_fin: str) -> str:
        """Gráfica con doble eje: viento vs temperatura (YYYY-MM-DD).
        
        Args:
            municipio: Nombre del municipio.
            fecha_inicio: Fecha inicio (YYYY-MM-DD).
            fecha_fin: Fecha fin (YYYY-MM-DD).
        
        Returns:
            Resumen y ruta de imagen.
        """
        try:
            conn = pymssql.connect(**db_config)
            
            query = """
                SELECT 
                    CAST(datetime AS DATE) as fecha,
                    AVG(wind_speed_10m) as velocidad_viento,
                    AVG(temperature_2m) as temperatura
                FROM climate_observations
                WHERE municipio = %s
                AND datetime >= %s
                AND datetime <= %s
                GROUP BY CAST(datetime AS DATE)
                ORDER BY fecha
            """
            
            df = pd.read_sql(query, conn, params=(
                municipio.lower().replace(' ', '_'),
                fecha_inicio,
                fecha_fin
            ))
            conn.close()
            
            if df.empty:
                return f"No se encontraron datos para {municipio} entre {fecha_inicio} y {fecha_fin}"
            
            # Crear gráfica con doble eje
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            # Eje izquierdo - Viento
            color1 = '#2E86AB'
            ax1.plot(df['fecha'], df['velocidad_viento'], 
                    color=color1, linewidth=2, label='Viento')
            ax1.fill_between(df['fecha'], df['velocidad_viento'], 
                            alpha=0.2, color=color1)
            ax1.set_xlabel('Fecha', fontsize=12)
            ax1.set_ylabel('Velocidad del Viento (km/h)', color=color1, fontsize=12)
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # Eje derecho - Temperatura
            ax2 = ax1.twinx()
            color2 = '#E94F37'
            ax2.plot(df['fecha'], df['temperatura'], 
                    color=color2, linewidth=2, label='Temperatura')
            ax2.set_ylabel('Temperatura (°C)', color=color2, fontsize=12)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            plt.title(f'🌬️ Viento vs 🌡️ Temperatura - {municipio.title()}', 
                     fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"viento_temp_{municipio}_{timestamp}.png"
            filepath_rel = _save_plot(fig, filename, PROJECT_ROOT)
            
            # Calcular estadísticas
            promedio_viento = df['velocidad_viento'].mean()
            promedio_temp = df['temperatura'].mean()
            
            return f"""✅ Comparación generada

📊 Promedios ({fecha_inicio} a {fecha_fin}):
• Municipio: {municipio.title()}
• Días: {len(df)}
• Viento: {promedio_viento:.2f} km/h
• Temperatura: {promedio_temp:.2f} °C

📁 Imagen: {filepath_rel}
"""
        except Exception as e:
            return f"Error al generar gráfica: {str(e)}"
    
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
        
        # Visualization tools
        graficar_serie_temporal_municipio,
        graficar_comparacion_municipios,
        graficar_patron_horario,
        graficar_viento_temperatura,
    ]

