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
print(f"🔧 Cargando variables de entorno desde: {env_path}")
print(f"✅ Archivo .env {'encontrado' if env_path.exists() else 'NO ENCONTRADO'}")

from src.bot import ClimateBot


def main():
    """Función principal para iniciar el bot."""
    import os
    
    print("=" * 70)
    print("🌬️  ClimateGuajira Telegram Bot")
    print("=" * 70)
    print()
    
    # Verificar variables de entorno críticas
    required_vars = {
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'DB_SERVER': os.getenv('DB_SERVER', 'localhost'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD')
    }
    
    print("🔍 Verificando variables de entorno:")
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if var_value:
            if 'TOKEN' in var_name or 'KEY' in var_name or 'PASSWORD' in var_name:
                display = f"***{var_value[-4:]}" if len(var_value) > 4 else "***"
            else:
                display = var_value
            print(f"   ✅ {var_name}: {display}")
        else:
            print(f"   ❌ {var_name}: NO CONFIGURADA")
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"\n❌ Error: Faltan variables de entorno: {', '.join(missing_vars)}")
        print("💡 Asegúrate de tener un archivo .env con todas las variables necesarias")
        sys.exit(1)
    
    print()
    
    try:
        # Inicializar y ejecutar bot
        bot = ClimateBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Bot detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

