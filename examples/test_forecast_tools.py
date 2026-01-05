# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Test script for forecast tools."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.climate_guajira.configuration import Configuration
from src.agents.climate_guajira.tools import create_tools


def test_forecast_tools():
    """Test the new forecast tools."""
    print("\n" + "=" * 80)
    print("🧪 TESTING FORECAST TOOLS")
    print("=" * 80 + "\n")
    
    # Initialize configuration and tools
    config = Configuration()
    tools = create_tools(config)
    
    # Find the forecast tools
    obtener_prediccion = None
    graficar_prediccion = None
    
    for tool in tools:
        if tool.name == "obtener_prediccion_municipio":
            obtener_prediccion = tool
        elif tool.name == "graficar_prediccion_municipio":
            graficar_prediccion = tool
    
    if not obtener_prediccion or not graficar_prediccion:
        print("❌ Forecast tools not found!")
        return
    
    print("✅ Forecast tools loaded\n")
    
    # Test municipalities
    test_municipios = ["riohacha", "maicao", "uribia"]
    
    for municipio in test_municipios:
        print("\n" + "=" * 80)
        print(f"📍 Testing: {municipio.upper()}")
        print("=" * 80)
        
        # Test 1: Get prediction
        print("\n1️⃣  Testing obtener_prediccion_municipio...")
        print("-" * 80)
        result = obtener_prediccion.invoke({"municipio": municipio})
        print(result)
        
        # Test 2: Generate plot
        print("\n2️⃣  Testing graficar_prediccion_municipio...")
        print("-" * 80)
        result = graficar_prediccion.invoke({"municipio": municipio})
        print(result)
        
        print("\n" + "=" * 80)
        input("Press Enter to continue with next municipality...")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_forecast_tools()

