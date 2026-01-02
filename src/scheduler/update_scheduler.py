# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Scheduler para actualización automática de base de datos climáticos.

Este script ejecuta actualizaciones periódicas de la base de datos,
descargando datos nuevos de la API Open-Meteo e insertándolos en
la base de datos ClimateDB.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db_updater import ClimateDBUpdater
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger(
    name="ClimateScheduler",
    console_level=logging.INFO,
    file_level=logging.DEBUG
)


def update_database_task():
    """
    Tarea principal de actualización de base de datos.
    
    Esta función se ejecuta cada hora para actualizar la base de datos
    con los datos climáticos más recientes.
    """
    logger.info("=" * 80)
    logger.info("🚀 SCHEDULED DATABASE UPDATE STARTED")
    logger.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        # Create updater and run update
        with ClimateDBUpdater(
            server=os.getenv("DB_SERVER"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "1433")
        ) as updater:
            results = updater.update_all_municipalities()
            
            # Log summary
            logger.info("\n" + "=" * 80)
            logger.info("📊 UPDATE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"📥 Total downloaded: {results['total_downloaded']:,} records")
            logger.info(f"💾 Total inserted: {results['total_inserted']:,} records")
            logger.info(f"✅ Successful: {results['successful']}/{results['total']}")
            logger.info("=" * 80)
            
            return results
            
    except Exception as e:
        logger.error(f"❌ Error in scheduled update: {str(e)}", exc_info=True)
        raise


def run_scheduler(
    cron_expression: str = "5 * * * *",  # Every hour at :05
    run_immediately: bool = False
):
    """
    Ejecuta el scheduler para actualizaciones periódicas.
    
    Args:
        cron_expression: Expresión cron para la frecuencia de actualización.
                        Default: "5 * * * *" (cada hora en el minuto 5)
        run_immediately: Si True, ejecuta una actualización inmediatamente
                        al iniciar. Default: False
    
    Examples:
        >>> # Ejecutar cada hora en el minuto 5
        >>> run_scheduler("5 * * * *")
        
        >>> # Ejecutar cada 30 minutos
        >>> run_scheduler("*/30 * * * *")
        
        >>> # Ejecutar cada día a las 6 AM
        >>> run_scheduler("0 6 * * *")
    """
    # Create logs directory
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 CLIMATE DATABASE SCHEDULER STARTED")
    logger.info("=" * 80)
    logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏰ Schedule: {cron_expression}")
    logger.info(f"📂 Log directory: {log_dir}")
    logger.info("=" * 80)
    
    # Run immediately if requested
    if run_immediately:
        logger.info("\n🔄 Running initial update...")
        try:
            update_database_task()
        except Exception as e:
            logger.error(f"Initial update failed: {e}")
    
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Parse cron expression
    cron_parts = cron_expression.split()
    if len(cron_parts) != 5:
        raise ValueError(
            "Invalid cron expression. Format: 'minute hour day month day_of_week'"
        )
    
    # Add job
    scheduler.add_job(
        update_database_task,
        CronTrigger(
            minute=cron_parts[0],
            hour=cron_parts[1],
            day=cron_parts[2],
            month=cron_parts[3],
            day_of_week=cron_parts[4]
        ),
        id='update_climate_db',
        name='Actualización de base de datos climáticos',
        replace_existing=True
    )
    
    logger.info("\n✅ Scheduler configured successfully")
    logger.info("⏰ Next run: Check schedule above")
    logger.info("\n⚠️  Press Ctrl+C to stop the scheduler\n")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n🛑 Scheduler stopped by user")
        scheduler.shutdown()
        logger.info("✅ Shutdown complete")


def main():
    """Punto de entrada principal del script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scheduler for automatic climate database updates"
    )
    parser.add_argument(
        "--cron",
        type=str,
        default="5 * * * *",
        help="Cron expression for update frequency (default: '5 * * * *' - every hour at :05)"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run an update immediately before starting the scheduler"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run update once and exit (no scheduler)"
    )
    
    args = parser.parse_args()
    
    if args.once:
        logger.info("🔄 Running single update (no scheduler)...")
        update_database_task()
        logger.info("✅ Single update completed")
    else:
        run_scheduler(
            cron_expression=args.cron,
            run_immediately=args.run_now
        )


if __name__ == "__main__":
    main()

