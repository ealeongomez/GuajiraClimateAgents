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

from src.agents.climate_guajira import graph, create_tools, Configuration


def main():
    """Run the ClimateGuajira agent in interactive mode."""
    config = Configuration()
    tools = create_tools(config)
    
    print("\n🌬️  ClimateGuajira - Agente del Atlas Eólico de Colombia")
    print("=" * 60)
    print(f"📦 Modelo: {config.model_name}")
    print(f"🔧 Herramientas disponibles:")
    for tool in tools:
        print(f"   • {tool.name}")
    print("\nEscribe tu pregunta sobre energía eólica en Colombia.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        try:
            question = input("❓ Pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 ¡Hasta luego!")
            break
        
        if question.lower() in ("salir", "exit", "q", "quit"):
            print("👋 ¡Hasta luego!")
            break
        
        if not question:
            continue
        
        print("\n⏳ Procesando...")
        
        try:
            response = graph.invoke({"messages": [("user", question)]})
            answer = response["messages"][-1].content
            print(f"\n💬 {answer}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

