#!/usr/bin/env python3
# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Script simple para actualización rápida de la base de datos.

Uso:
    python scripts/update_db_simple.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db_updater import update_database_from_env


def main():
    """Ejecuta actualización de base de datos."""
    print("🚀 Iniciando actualización de base de datos...")
    print("=" * 80)
    
    try:
        results = update_database_from_env()
        
        print("\n" + "=" * 80)
        print("✅ ACTUALIZACIÓN COMPLETADA")
        print("=" * 80)
        print(f"📥 Total descargado: {results['total_downloaded']:,} registros")
        print(f"💾 Total insertado: {results['total_inserted']:,} registros")
        print(f"✅ Exitosos: {results['successful']}/{results['total']}")
        print("=" * 80)
        
        # Exit with success
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


