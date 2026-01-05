#!/usr/bin/env python3
# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Ejemplo de cómo consultar y usar las predicciones guardadas.

ESTRUCTURA ACTUAL (1 fila por municipio):
- wind_speed_input: JSON array con 48 valores (históricos)
- wind_speed_output: JSON array con 24 valores (predicciones)
"""

import pymssql
import json
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

# Conectar a BD
conn = pymssql.connect(
    server=os.getenv("DB_SERVER", "localhost"),
    user=os.getenv("DB_USER", "sa"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "ClimateDB"),
    port=os.getenv("DB_PORT", "1433")
)

cursor = conn.cursor()

# Ejemplo 1: Contar total de municipios con predicciones
print("=" * 80)
print("📊 EJEMPLO 1: Total de municipios con predicciones")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM Forecast")
total = cursor.fetchone()[0]
print(f"Total de municipios con predicciones: {total}")

# Ejemplo 2: Ver última predicción para Riohacha
print("\n" + "=" * 80)
print("📊 EJEMPLO 2: Última predicción para Riohacha")
print("=" * 80)

query = """
SELECT 
    datetime_inicio,
    wind_speed_input,
    wind_speed_output,
    created_at
FROM Forecast
WHERE municipio = 'riohacha'
ORDER BY created_at DESC
OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY
"""

cursor.execute(query)
row = cursor.fetchone()

if row:
    datetime_inicio = row[0]
    input_array = json.loads(row[1])    # 48 valores históricos
    output_array = json.loads(row[2])   # 24 predicciones
    created_at = row[3]
    
    print(f"Predicción generada: {created_at}")
    print(f"Primera hora predicha: {datetime_inicio}")
    print(f"\nDatos de entrada: {len(input_array)} valores históricos")
    print(f"  • Primeros 3: {input_array[:3]}")
    print(f"  • Últimos 3: {input_array[-3:]}")
    
    print(f"\nPredicciones: {len(output_array)} horas")
    print(f"  • Primeras 3: {output_array[:3]}")
    print(f"  • Últimas 3: {output_array[-3:]}")
    
    # Mostrar todas las predicciones con sus fechas
    print("\n📅 Predicciones detalladas:")
    for i, pred_value in enumerate(output_array):
        pred_time = datetime_inicio + timedelta(hours=i)
        print(f"  {pred_time}: {pred_value:.2f} m/s")
else:
    print("No hay predicciones para Riohacha")

# Ejemplo 3: Comparar predicciones entre municipios
print("\n" + "=" * 80)
print("📊 EJEMPLO 3: Comparación entre municipios")
print("=" * 80)

query = """
SELECT 
    municipio,
    datetime_inicio,
    wind_speed_output
FROM Forecast
ORDER BY created_at DESC
"""

cursor.execute(query)
rows = cursor.fetchall()

print(f"{'Municipio':<20} {'Hora inicio':<20} {'Promedio':<10} {'Máximo':<10} {'Mínimo':<10}")
print("-" * 70)

for row in rows:
    municipio = row[0]
    datetime_inicio = row[1]
    output_array = json.loads(row[2])
    
    avg_wind = sum(output_array) / len(output_array)
    max_wind = max(output_array)
    min_wind = min(output_array)
    
    print(f"{municipio:<20} {str(datetime_inicio):<20} {avg_wind:>8.2f}   {max_wind:>8.2f}   {min_wind:>8.2f}")

# Ejemplo 4: Obtener predicción para una hora específica
print("\n" + "=" * 80)
print("📊 EJEMPLO 4: Predicción para hora específica")
print("=" * 80)

municipio_query = "maicao"
hora_deseada = 5  # 5 horas después del inicio

query = """
SELECT 
    datetime_inicio,
    wind_speed_output
FROM Forecast
WHERE municipio = %s
ORDER BY created_at DESC
OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY
"""

cursor.execute(query, (municipio_query,))
row = cursor.fetchone()

if row:
    datetime_inicio = row[0]
    output_array = json.loads(row[1])
    
    if hora_deseada < len(output_array):
        hora_predicha = datetime_inicio + timedelta(hours=hora_deseada)
        valor_predicho = output_array[hora_deseada]
        print(f"Municipio: {municipio_query}")
        print(f"Hora: {hora_predicha}")
        print(f"Velocidad del viento predicha: {valor_predicho:.2f} m/s")
    else:
        print(f"Hora {hora_deseada} fuera de rango (solo hay {len(output_array)} predicciones)")
else:
    print(f"No hay predicciones para {municipio_query}")

# Ejemplo 5: Obtener todas las predicciones para múltiples municipios
print("\n" + "=" * 80)
print("📊 EJEMPLO 5: Predicciones para todos los municipios")
print("=" * 80)

municipios_interes = ['riohacha', 'maicao', 'uribia']

for mun in municipios_interes:
    query = """
    SELECT 
        datetime_inicio,
        wind_speed_output,
        created_at
    FROM Forecast
    WHERE municipio = %s
    ORDER BY created_at DESC
    OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY
    """
    
    cursor.execute(query, (mun,))
    row = cursor.fetchone()
    
    if row:
        datetime_inicio = row[0]
        output_array = json.loads(row[1])
        created_at = row[2]
        
        avg_wind = sum(output_array) / len(output_array)
        
        print(f"\n{mun.upper()}")
        print(f"  Generado: {created_at}")
        print(f"  Inicio: {datetime_inicio}")
        print(f"  Promedio: {avg_wind:.2f} m/s")
        print(f"  Rango: {min(output_array):.2f} - {max(output_array):.2f} m/s")
    else:
        print(f"\n{mun.upper()}: Sin predicciones")

# Ejemplo 6: Para uso en bot de Telegram
print("\n" + "=" * 80)
print("📊 EJEMPLO 6: Formato para Bot de Telegram")
print("=" * 80)

def get_forecast_for_bot(municipio: str) -> str:
    """
    Obtiene la predicción en formato listo para enviar a un usuario.
    """
    query = """
    SELECT 
        datetime_inicio,
        wind_speed_output,
        created_at
    FROM Forecast
    WHERE municipio = %s
    ORDER BY created_at DESC
    OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY
    """
    
    cursor.execute(query, (municipio.lower(),))
    row = cursor.fetchone()
    
    if not row:
        return f"❌ No hay predicciones disponibles para {municipio}"
    
    datetime_inicio = row[0]
    output_array = json.loads(row[1])
    created_at = row[2]
    
    # Crear mensaje
    mensaje = f"🌬️ **Predicción de Viento - {municipio.title()}**\n\n"
    mensaje += f"📅 Generado: {created_at.strftime('%Y-%m-%d %H:%M')}\n"
    mensaje += f"⏰ Desde: {datetime_inicio.strftime('%Y-%m-%d %H:%M')}\n\n"
    mensaje += "📊 **Próximas 24 horas:**\n"
    
    # Mostrar cada 3 horas (8 valores de 24)
    for i in range(0, len(output_array), 3):
        hora = datetime_inicio + timedelta(hours=i)
        valor = output_array[i]
        mensaje += f"  • {hora.strftime('%H:%M')}: {valor:.1f} m/s\n"
    
    # Estadísticas
    avg_wind = sum(output_array) / len(output_array)
    mensaje += f"\n📈 Promedio: {avg_wind:.2f} m/s\n"
    mensaje += f"⬆️ Máximo: {max(output_array):.2f} m/s\n"
    mensaje += f"⬇️ Mínimo: {min(output_array):.2f} m/s\n"
    
    return mensaje

# Probar formato de bot
ejemplo_bot = get_forecast_for_bot("riohacha")
print(ejemplo_bot)

# Cerrar conexión
cursor.close()
conn.close()

print("\n" + "=" * 80)
print("✅ Ejemplos completados")
print("=" * 80)
