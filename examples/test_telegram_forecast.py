# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Script de prueba para verificar integración de forecast con Telegram."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.agents.climate_guajira.configuration import Configuration
from src.agents.climate_guajira.tools import create_tools


def test_telegram_integration():
    """Simula el flujo de Telegram para verificar que las imágenes se detecten."""
    
    print("\n" + "="*80)
    print("🧪 TEST DE INTEGRACIÓN CON TELEGRAM")
    print("="*80 + "\n")
    
    # 1. Inicializar tools
    print("1️⃣  Inicializando tools...")
    config = Configuration()
    tools = create_tools(config)
    
    # 2. Encontrar tool de gráficas
    graficar_tool = [t for t in tools if t.name == 'graficar_prediccion_municipio'][0]
    print("   ✅ Tool encontrada: graficar_prediccion_municipio\n")
    
    # 3. Ejecutar tool
    print("2️⃣  Ejecutando: graficar_prediccion_municipio('riohacha')")
    result = graficar_tool.invoke({'municipio': 'riohacha'})
    
    # 4. Verificar que IMG_PATH esté presente
    print("3️⃣  Verificando presencia de IMG_PATH...")
    if 'IMG_PATH:' in result:
        print("   ✅ IMG_PATH encontrado\n")
        
        # Extraer ruta (simular bot)
        lines = result.split('\n')
        img_line = [line for line in lines if 'IMG_PATH:' in line][0]
        img_path = img_line.split('IMG_PATH:')[1].strip()
        print(f"   📁 Ruta extraída: {img_path}\n")
        
        # Verificar que el archivo existe
        if Path(img_path).exists():
            print("   ✅ Archivo de imagen existe\n")
        else:
            print(f"   ❌ Archivo no encontrado: {img_path}\n")
            return False
    else:
        print("   ❌ IMG_PATH NO encontrado")
        print("   ⚠️  El bot de Telegram NO podrá enviar la imagen\n")
        return False
    
    # 5. Simular limpieza del mensaje (como lo hace el bot)
    print("4️⃣  Simulando limpieza del mensaje (como Telegram bot)...")
    clean_lines = [line for line in lines if 'IMG_PATH:' not in line]
    clean_message = '\n'.join(clean_lines).strip()
    
    if 'IMG_PATH:' not in clean_message:
        print("   ✅ Línea IMG_PATH eliminada del mensaje\n")
    else:
        print("   ❌ IMG_PATH aún visible en el mensaje\n")
        return False
    
    # 6. Mostrar resultado final
    print("5️⃣  Mensaje que verá el usuario:")
    print("   " + "-"*76)
    for line in clean_message.split('\n')[:10]:
        print(f"   {line}")
    print("   ...")
    print("   " + "-"*76 + "\n")
    
    print("=" * 80)
    print("✅ PRUEBA EXITOSA - La integración con Telegram funciona correctamente")
    print("=" * 80 + "\n")
    
    print("📱 Para probar en Telegram real:")
    print("   1. Inicia el bot: python main_telegram.py")
    print("   2. Envía mensaje: 'Muéstrame una gráfica de predicción para Riohacha'")
    print("   3. El bot debería enviar el texto + la imagen por separado\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_telegram_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

