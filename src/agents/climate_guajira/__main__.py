# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Main entry point for ClimateGuajira agent CLI."""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pymssql
from src.agents.climate_guajira import graph, create_tools, Configuration


def main():
    """Run the ClimateGuajira agent in interactive mode."""
    config = Configuration()
    tools = create_tools(config)
    
    print("\n" + "=" * 70)
    print("🌬️  ClimateGuajira - Agente Inteligente de Clima y Energía Eólica")
    print("=" * 70)
    print(f"\n📦 Modelo: {config.model_name}")
    print(f"🗄️  Base de datos: {config.db_name}")
    
    # Verificar conexión a base de datos
    try:
        db_config = config.get_db_config()
        conn = pymssql.connect(**db_config)
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
        print(f"✅ Base de datos conectada: {count:,} registros climáticos")
        if temp_cols == 4:
            print(f"✅ Columnas temporales optimizadas disponibles")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo conectar a la base de datos: {e}")
    
    print(f"\n🔧 Herramientas disponibles ({len(tools)}):")
    
    # Agrupar herramientas por categoría
    rag_tools = [t for t in tools if 'atlas' in t.name or 'documento' in t.name]
    db_basic = [t for t in tools if t.name in ['obtener_estadisticas_municipio', 
                                                 'comparar_municipios_viento', 
                                                 'listar_municipios_disponibles']]
    db_temporal = [t for t in tools if t.name in ['obtener_estadisticas_por_mes',
                                                   'obtener_estadisticas_por_hora',
                                                   'comparar_anios']]
    
    if rag_tools:
        print("\n  📚 Atlas Eólico (RAG):")
        for tool in rag_tools:
            print(f"     • {tool.name}")
    
    if db_basic:
        print("\n  📊 Base de Datos (Estadísticas generales):")
        for tool in db_basic:
            print(f"     • {tool.name}")
    
    if db_temporal:
        print("\n  ⚡ Base de Datos (Análisis temporal optimizado):")
        for tool in db_temporal:
            print(f"     • {tool.name}")
    
    print("\n" + "=" * 70)
    print("💡 Ejemplos de preguntas:")
    print("   • ¿Cuál es el potencial eólico de La Guajira?")
    print("   • ¿Cómo fue el viento en Riohacha durante 2024?")
    print("   • Compara el viento entre Maicao y Manaure")
    print("   • ¿A qué hora del día hay más viento en Uribia?")
    print("\nEscribe 'salir' para terminar.\n")
    
    while True:
        try:
            question = input("❓ Pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 ¡Hasta luego!")
            break
        
        if question.lower() in ("salir", "exit", "q", "quit"):
            print("\n👋 ¡Hasta luego!")
            break
        
        if not question:
            continue
        
        print("\n⏳ Procesando...\n")
        
        try:
            response = graph.invoke({"messages": [("user", question)]})
            answer = response["messages"][-1].content
            print(f"💬 {answer}\n")
            print("-" * 70 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()

