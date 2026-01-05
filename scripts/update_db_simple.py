#!/usr/bin/env python3
# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Script para actualización completa: datos históricos + predicciones.

Uso:
    python scripts/update_db_simple.py                # Ejecución única
    python scripts/update_db_simple.py --forecast-only
    python scripts/update_db_simple.py --data-only
    python scripts/update_db_simple.py --daemon        # Modo continuo (cada hora a las XX:05)
"""

import sys
import argparse
import time
import schedule
from pathlib import Path
from datetime import datetime, timedelta, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db_updater import update_database_from_env
from src.models.forecast_generator import ForecastGenerator
from src.utils.forecast_db_updater import ForecastDBUpdater
import pymssql
import os
from dotenv import load_dotenv

 
def update_historical_data():
    """Actualiza datos históricos."""
    print("📊 ACTUALIZACIÓN DE DATOS HISTÓRICOS")
    print("=" * 80)
    
    results = update_database_from_env()
    
    print(f"📥 Total descargado: {results['total_downloaded']:,} registros")
    print(f"💾 Total insertado: {results['total_inserted']:,} registros")
    print(f"✅ Exitosos: {results['successful']}/{results['total']}")
    
    return results


def update_forecasts():
    """Genera y guarda predicciones."""
    print("\n🔮 GENERACIÓN Y ACTUALIZACIÓN DE PREDICCIONES")
    print("=" * 80)
    
    load_dotenv()
    
    # Conectar a BD
    conn = pymssql.connect(
        server=os.getenv("DB_SERVER", "localhost"),
        user=os.getenv("DB_USER", "sa"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME", "ClimateDB"),
        port=os.getenv("DB_PORT", "1433")
    )
    
    try:
        # Inicializar generador
        generator = ForecastGenerator(models_dir=str(PROJECT_ROOT / "data/models/LSTM"))
        generator.load_models()
        generator.load_normalization_params(conn)
        
        # Generar predicciones
        forecasts = generator.generate_all_forecasts(conn)
        
        # Guardar en BD
        updater = ForecastDBUpdater(conn)
        updater.clear_old_forecasts(older_than_hours=48)
        total_inserted = updater.insert_forecasts(forecasts)
        updater.close()
        
        return {
            'municipalities': len(forecasts),
            'total_predictions': total_inserted
        }
        
    finally:
        conn.close()


def run_update(forecast_only=False, data_only=False):
    """
    Ejecuta una actualización completa del sistema.
    
    Por defecto (sin argumentos) ejecuta AMBOS procesos:
    1. Descarga y almacena datos históricos de Open-Meteo
    2. Genera y almacena predicciones con modelos LSTM
    
    Args:
        forecast_only: Solo actualizar predicciones (omite datos históricos)
        data_only: Solo actualizar datos históricos (omite predicciones)
        
    Returns:
        0 si exitoso, 1 si hay error
    """
    print("\n" + "=" * 80)
    print("🚀 ACTUALIZACIÓN DE BASE DE DATOS")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if forecast_only:
        print("📋 Modo: Solo predicciones")
    elif data_only:
        print("📋 Modo: Solo datos históricos")
    else:
        print("📋 Modo: Actualización completa (datos + predicciones)")
    
    print("=" * 80 + "\n")
    
    try:
        data_results = None
        forecast_results = None
        
        # PASO 1: Actualizar datos históricos
        if not forecast_only:
            print("📊 PASO 1/2: Actualizando datos históricos...")
            data_results = update_historical_data()
            print("✅ Datos históricos actualizados\n")
        
        # PASO 2: Generar y actualizar predicciones
        if not data_only:
            print("🔮 PASO 2/2: Generando predicciones...")
            forecast_results = update_forecasts()
            print("✅ Predicciones actualizadas\n")
        
        # Resumen final
        print("\n" + "=" * 80)
        print("✅ ACTUALIZACIÓN COMPLETADA")
        print("=" * 80)
        
        if data_results:
            print(f"📥 Datos históricos: {data_results['total_inserted']:,} registros insertados")
        
        if forecast_results:
            print(f"🔮 Predicciones: {forecast_results['total_predictions']:,} registros")
            print(f"📍 Municipios: {forecast_results['municipalities']}")
        
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def run_daemon():
    """
    Ejecuta en modo daemon: actualiza cada hora automáticamente.
    
    - Descarga datos históricos nuevos de Open-Meteo
    - Genera predicciones con modelos LSTM
    - Inserta ambos en la base de datos
    
    Se ejecuta cada hora a los 2 minutos (00:02, 01:02, 02:02, etc.)
    """
    print("\n" + "=" * 80)
    print("🔄 MODO DAEMON INICIADO")
    print("=" * 80)
    print(f"📅 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print("⏰ Programación:")
    print("   • Actualización COMPLETA cada hora a las XX:02")
    print("   • Descarga datos históricos + Genera predicciones")
    print("   • Se ejecuta automáticamente sin intervención")
    print("")
    print("📊 En cada ejecución:")
    print("   1. Descarga datos de Open-Meteo (últimas horas)")
    print("   2. Inserta en climate_observations")
    print("   3. Carga modelos LSTM")
    print("   4. Genera predicciones (24h futuras)")
    print("   5. Actualiza tabla Forecast")
    print("")
    print("💡 Para detener: Ctrl+C")
    print("=" * 80 + "\n")
    
    # Programar actualización a los 2 minutos de cada hora (00:02, 01:02, 02:02, etc.)
    schedule.every().hour.at(":02").do(run_update)
    
    # Ejecutar inmediatamente al iniciar
    print("🔄 Ejecutando actualización inicial...")
    result = run_update()
    
    if result == 0:
        print("\n✅ Primera actualización completada")
        # Calcular próxima ejecución (próximo minuto 02)
        now = datetime.now()
        if now.minute < 2:
            next_run = now.replace(minute=2, second=0, microsecond=0)
        else:
            next_run = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
        print(f"⏰ Próxima ejecución: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n⚠️  Primera actualización tuvo errores, pero el daemon continuará")
    
    # Loop principal
    print("\n⏳ Daemon activo, esperando próxima hora (ejecuta a las XX:02)...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Revisar cada 30 segundos (más eficiente)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("🛑 DAEMON DETENIDO")
        print("=" * 80)
        print(f"📅 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        sys.exit(0)


def main():
    """Ejecuta actualización según modo seleccionado."""
    parser = argparse.ArgumentParser(
        description="Actualiza datos históricos y predicciones"
    )
    parser.add_argument(
        '--forecast-only',
        action='store_true',
        help='Solo actualizar predicciones'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Solo actualizar datos históricos'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Ejecutar en modo continuo (cada hora a las XX:02)'
    )
    
    args = parser.parse_args()
    
    # Modo daemon
    if args.daemon:
        run_daemon()
        return 0
    
    # Ejecución única
    return run_update(
        forecast_only=args.forecast_only,
        data_only=args.data_only
    )


if __name__ == "__main__":
    sys.exit(main())


