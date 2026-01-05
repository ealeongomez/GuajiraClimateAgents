#!/usr/bin/env python3
# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Actualiza la tabla Forecast con nuevas predicciones.
"""

import pymssql
import pandas as pd
import json
from typing import Dict
from datetime import datetime


class ForecastDBUpdater:
    """Maneja la inserción de predicciones en la BD."""
    
    def __init__(self, conn):
        """
        Args:
            conn: Conexión pymssql activa
        """
        self.conn = conn
        self.cursor = conn.cursor()
        
    def clear_old_forecasts(self, older_than_hours: int = 24):
        """
        Elimina predicciones antiguas manteniendo solo las más recientes por municipio.
        
        Estrategia:
        1. Mantiene solo la predicción más reciente para cada municipio
        2. Elimina predicciones más antiguas que N horas como respaldo
        
        Args:
            older_than_hours: Eliminar predicciones más antiguas que N horas
        """
        # Eliminar predicciones antiguas (más de N horas)
        query_old = f"""
        DELETE FROM Forecast
        WHERE created_at < DATEADD(HOUR, -{older_than_hours}, GETDATE())
        """
        self.cursor.execute(query_old)
        self.conn.commit()
        deleted_old = self.cursor.rowcount
        
        # Mantener solo la predicción más reciente por municipio
        # (elimina predicciones duplicadas del mismo municipio)
        query_duplicates = """
        WITH RankedForecasts AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY municipio 
                       ORDER BY created_at DESC
                   ) as rn
            FROM Forecast
        )
        DELETE FROM Forecast
        WHERE id IN (
            SELECT id FROM RankedForecasts WHERE rn > 1
        )
        """
        self.cursor.execute(query_duplicates)
        self.conn.commit()
        deleted_duplicates = self.cursor.rowcount
        
        total_deleted = deleted_old + deleted_duplicates
        if total_deleted > 0:
            print(f"🗑️  Eliminadas {deleted_old} predicciones antiguas y {deleted_duplicates} duplicadas\n")
        else:
            print(f"🗑️  No hay predicciones para eliminar\n")
        
    def insert_forecasts(
        self, 
        forecasts: Dict[str, pd.DataFrame]
    ) -> int:
        """
        Inserta predicciones en la tabla Forecast.
        Elimina predicciones existentes para la misma fecha antes de insertar.
        
        Args:
            forecasts: Dict con DataFrames de predicciones
            
        Returns:
            Número total de registros insertados
        """
        print("💾 Insertando predicciones en la base de datos...")
        total_inserted = 0
        
        delete_query = """
        DELETE FROM Forecast
        WHERE municipio = %s AND datetime_inicio = %s
        """
        
        insert_query = """
        INSERT INTO Forecast (
            municipio, datetime_inicio, wind_speed_input, wind_speed_output
        )
        VALUES (%s, %s, %s, %s)
        """
        
        for mun, df in forecasts.items():
            try:
                # Solo debe haber 1 fila por municipio
                if len(df) != 1:
                    print(f"  ⚠️  {mun}: Se esperaba 1 fila, encontradas {len(df)}")
                    continue
                
                row = df.iloc[0]
                
                # Convertir datetime de pandas a datetime de Python
                dt = row['datetime_inicio']
                if hasattr(dt, 'to_pydatetime'):
                    dt = dt.to_pydatetime()
                
                # Eliminar predicciones existentes para este municipio y fecha
                self.cursor.execute(delete_query, (str(row['municipio']), dt))
                deleted = self.cursor.rowcount
                
                # Convertir arrays a JSON
                input_data = row['wind_speed_input']
                output_data = row['wind_speed_output']
                
                if not isinstance(input_data, list):
                    input_data = list(input_data) if hasattr(input_data, '__iter__') else [input_data]
                if not isinstance(output_data, list):
                    output_data = list(output_data) if hasattr(output_data, '__iter__') else [output_data]
                
                # Insertar registro
                self.cursor.execute(insert_query, (
                    str(row['municipio']),
                    dt,
                    json.dumps(input_data),   # Array de 48 valores
                    json.dumps(output_data)   # Array de 24 valores
                ))
                self.conn.commit()
                
                total_inserted += 1
                status = "actualizado" if deleted > 0 else "insertado"
                print(f"  ✅ {mun}: {len(output_data)} predicciones ({status})")
                
            except Exception as e:
                print(f"  ❌ {mun}: Error - {str(e)}")
                self.conn.rollback()
                
        print(f"\n✅ Total procesado: {total_inserted} municipios\n")
        return total_inserted
        
    def close(self):
        """Cierra el cursor."""
        self.cursor.close()

