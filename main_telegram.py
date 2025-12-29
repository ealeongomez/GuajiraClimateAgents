# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Entry point para el bot de Telegram en producción.

Este script inicia el bot de Telegram que proporciona acceso
al agente ClimateGuajira con:
- Historial de conversación persistente por usuario
- Generación y envío de gráficas climáticas
- Manejo robusto de errores
- Estadísticas de uso

Uso:
    python main_telegram.py

Variables de entorno requeridas (.env):
    - TELEGRAM_BOT_TOKEN: Token del bot de Telegram
    - OPENAI_API_KEY: API key de OpenAI
    - DB_SERVER, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME: Conexión a SQL Server
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANTE: Cargar .env ANTES de importar cualquier módulo
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

# Configurar logging
from src.utils.logger import setup_logger
logger = setup_logger("MainTelegram", log_dir=PROJECT_ROOT / "logs")

logger.info("=" * 70)
logger.info(f"Cargando variables de entorno desde: {env_path}")
if env_path.exists():
    logger.info("✅ Archivo .env encontrado")
else:
    logger.error("❌ Archivo .env NO ENCONTRADO")

from src.bot import ClimateBot


def main():
    """Función principal para iniciar el bot."""
    import os
    
    logger.info("=" * 70)
    logger.info("🌬️  ClimateGuajira Telegram Bot")
    logger.info("=" * 70)
    
    # Verificar variables de entorno críticas
    required_vars = {
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'DB_SERVER': os.getenv('DB_SERVER', 'localhost'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD')
    }
    
    logger.info("🔍 Verificando variables de entorno:")
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if var_value:
            if 'TOKEN' in var_name or 'KEY' in var_name or 'PASSWORD' in var_name:
                display = f"***{var_value[-4:]}" if len(var_value) > 4 else "***"
            else:
                display = var_value
            logger.info(f"   ✅ {var_name}: {display}")
        else:
            logger.error(f"   ❌ {var_name}: NO CONFIGURADA")
            missing_vars.append(var_name)
    
    if missing_vars:
        logger.error(f"Error: Faltan variables de entorno: {', '.join(missing_vars)}")
        logger.error("💡 Asegúrate de tener un archivo .env con todas las variables necesarias")
        sys.exit(1)
    
    logger.info("Todas las variables de entorno están configuradas correctamente")
    
    try:
        # Inicializar y ejecutar bot
        logger.info("Inicializando bot de Telegram...")
        bot = ClimateBot()
        logger.info("Bot inicializado correctamente")
        logger.info("Iniciando polling... (Presiona Ctrl+C para detener)")
        bot.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Bot detenido por el usuario")
    except Exception as e:
        logger.critical(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

