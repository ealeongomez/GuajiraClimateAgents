# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Database updater for climate observations.

This module provides functionality to update the climate_observations table
with new data from the Open-Meteo API, handling incremental updates and
avoiding duplicates.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pymssql

from .climate_data import ClimateDataFetcher


logger = logging.getLogger(__name__)


class ClimateDBUpdater:
    """
    Handles database updates for climate observations.
    
    This class manages the connection to the database, fetches new data
    from the API, and inserts it efficiently using batch operations and
    MERGE statements to avoid duplicates.
    
    Example:
        >>> updater = ClimateDBUpdater(
        ...     server="localhost",
        ...     database="ClimateDB",
        ...     user="sa",
        ...     password="password"
        ... )
        >>> results = updater.update_all_municipalities()
        >>> print(f"Inserted {results['total_inserted']} new records")
    """
    
    DEFAULT_START_DATE = datetime(2015, 12, 21)
    BATCH_SIZE = 1000
    
    def __init__(
        self,
        server: str,
        database: str,
        user: str,
        password: str,
        port: str = "1433",
        autocommit: bool = True
    ):
        """
        Initialize the database updater.
        
        Args:
            server: Database server address.
            database: Database name.
            user: Database user.
            password: Database password.
            port: Database port (default: 1433).
            autocommit: Enable autocommit mode (default: True).
        """
        self.server = server
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = autocommit
        
        self._conn = None
        self._cursor = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            self._conn = pymssql.connect(
                server=self.server,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=self.autocommit
            )
            self._cursor = self._conn.cursor()
            logger.info(f"✅ Connected to database: {self.database}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
        logger.info("✅ Database connection closed")
    
    def get_last_dates(self) -> Dict[str, datetime]:
        """
        Get the last recorded date for each municipality.
        
        Returns:
            Dictionary mapping municipality names to their last recorded datetime.
        """
        if not self._cursor:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        self._cursor.execute("""
            SELECT 
                municipio,
                MAX(datetime) as ultima_fecha,
                COUNT(*) as total_registros
            FROM climate_observations
            GROUP BY municipio
            ORDER BY municipio
        """)
        
        result = {}
        logger.info("📅 Last date per municipality:")
        logger.info("-" * 70)
        
        for row in self._cursor.fetchall():
            municipio, ultima_fecha, total_registros = row
            result[municipio] = ultima_fecha
            logger.info(
                f"  • {municipio:20s} → {ultima_fecha} "
                f"({total_registros:,} records)"
            )
        
        logger.info("-" * 70)
        return result
    
    def bulk_insert_climate_data(
        self,
        df: pd.DataFrame,
        municipio: str
    ) -> int:
        """
        Insert climate data into the database using MERGE to avoid duplicates.
        
        Args:
            df: DataFrame with climate data to insert.
            municipio: Municipality name.
            
        Returns:
            Number of records inserted or updated.
        """
        if not self._cursor:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        if df.empty:
            logger.warning(f"⚠️  No new data for {municipio}")
            return 0
        
        # Prepare data
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Ensure required columns
        required_cols = [
            'datetime', 'wind_speed_10m', 'wind_direction_10m',
            'temperature_2m', 'relative_humidity_2m', 'precipitation'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Insert in batches
        total_inserted = 0
        
        for i in range(0, len(df), self.BATCH_SIZE):
            batch = df.iloc[i:i+self.BATCH_SIZE]
            inserted = self._insert_batch(batch, municipio)
            total_inserted += inserted
        
        return total_inserted
    
    def _insert_batch(self, batch: pd.DataFrame, municipio: str) -> int:
        """
        Insert a batch of records using temporary table and MERGE.
        
        Args:
            batch: Batch of records to insert.
            municipio: Municipality name.
            
        Returns:
            Number of records affected.
        """
        # Create temporary table
        self._cursor.execute("""
            IF OBJECT_ID('tempdb..#TempClimateData') IS NOT NULL
                DROP TABLE #TempClimateData
                
            CREATE TABLE #TempClimateData (
                municipio NVARCHAR(50),
                datetime DATETIME2,
                wind_speed_10m FLOAT,
                wind_direction_10m INT,
                temperature_2m FLOAT,
                relative_humidity_2m INT,
                precipitation FLOAT
            )
        """)
        
        # Prepare values
        insert_query = """
            INSERT INTO #TempClimateData 
            (municipio, datetime, wind_speed_10m, wind_direction_10m, 
             temperature_2m, relative_humidity_2m, precipitation)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        values = []
        for _, row in batch.iterrows():
            values.append((
                municipio,
                row['datetime'],
                float(row['wind_speed_10m']) if pd.notna(row['wind_speed_10m']) else None,
                int(row['wind_direction_10m']) if pd.notna(row['wind_direction_10m']) else None,
                float(row['temperature_2m']) if pd.notna(row['temperature_2m']) else None,
                int(row['relative_humidity_2m']) if pd.notna(row['relative_humidity_2m']) else None,
                float(row['precipitation']) if pd.notna(row['precipitation']) else None,
            ))
        
        # Insert into temporary table
        self._cursor.executemany(insert_query, values)
        
        # Merge into main table
        self._cursor.execute("""
            MERGE climate_observations AS target
            USING #TempClimateData AS source
            ON target.municipio = source.municipio 
               AND target.datetime = source.datetime
            WHEN NOT MATCHED THEN
                INSERT (municipio, datetime, wind_speed_10m, wind_direction_10m,
                        temperature_2m, relative_humidity_2m, precipitation, created_at)
                VALUES (source.municipio, source.datetime, source.wind_speed_10m, 
                        source.wind_direction_10m, source.temperature_2m, 
                        source.relative_humidity_2m, source.precipitation, GETDATE())
            WHEN MATCHED THEN
                UPDATE SET
                    wind_speed_10m = source.wind_speed_10m,
                    wind_direction_10m = source.wind_direction_10m,
                    temperature_2m = source.temperature_2m,
                    relative_humidity_2m = source.relative_humidity_2m,
                    precipitation = source.precipitation;
                    
            SELECT @@ROWCOUNT as affected_rows
        """)
        
        result = self._cursor.fetchone()
        inserted = result[0] if result else 0
        
        # Clean up temporary table
        self._cursor.execute("DROP TABLE #TempClimateData")
        
        return inserted
    
    def update_municipality(
        self,
        municipio: str,
        last_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Update data for a single municipality.
        
        Args:
            municipio: Municipality name.
            last_date: Last recorded date. If None, will query database.
            end_date: End date for update. If None, uses current time.
            
        Returns:
            Dictionary with update statistics.
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Determine start date
        if last_date is None:
            last_dates = self.get_last_dates()
            if municipio in last_dates:
                start_date = last_dates[municipio] + timedelta(hours=1)
            else:
                start_date = self.DEFAULT_START_DATE
        else:
            start_date = last_date + timedelta(hours=1)
        
        logger.info(f"📍 Processing {municipio.upper()}")
        logger.info(f"   📅 Downloading from: {start_date}")
        
        # Check if update is needed
        if start_date >= end_date:
            logger.info(f"   ℹ️  Already up to date")
            return {
                "municipio": municipio,
                "downloaded": 0,
                "inserted": 0,
                "status": "already_updated"
            }
        
        try:
            # Fetch data from API
            fetcher = ClimateDataFetcher(
                municipio=municipio,
                start_date=start_date,
                end_date=end_date,
                wind_only=False
            )
            
            df = fetcher.fetch(block_days=180)
            
            if df.empty:
                logger.warning(f"   ⚠️  No new data downloaded")
                return {
                    "municipio": municipio,
                    "downloaded": 0,
                    "inserted": 0,
                    "status": "no_new_data"
                }
            
            logger.info(f"   ✅ Downloaded {len(df):,} records")
            
            # Insert into database
            logger.info(f"   💾 Inserting into database...")
            inserted = self.bulk_insert_climate_data(df, municipio)
            
            logger.info(f"   ✅ {inserted:,} records inserted/updated")
            
            return {
                "municipio": municipio,
                "downloaded": len(df),
                "inserted": inserted,
                "start_date": df['datetime'].min(),
                "end_date": df['datetime'].max(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"   ❌ Error: {str(e)}")
            return {
                "municipio": municipio,
                "downloaded": 0,
                "inserted": 0,
                "status": f"error: {str(e)[:50]}"
            }
    
    def update_all_municipalities(
        self,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Update data for all municipalities.
        
        Args:
            end_date: End date for update. If None, uses current time.
            
        Returns:
            Dictionary with overall statistics and per-municipality results.
        """
        if end_date is None:
            end_date = datetime.now()
        
        logger.info("🚀 Starting database update for all municipalities")
        logger.info("=" * 80)
        logger.info(f"📅 Update until: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get last dates
        last_dates = self.get_last_dates()
        
        # Get all municipalities
        municipios = ClimateDataFetcher.get_available_municipios()
        
        logger.info(f"📊 Municipalities to process: {len(municipios)}")
        logger.info("=" * 80)
        
        # Update each municipality
        results = []
        total_downloaded = 0
        total_inserted = 0
        
        for idx, municipio in enumerate(municipios, 1):
            logger.info(f"\n[{idx}/{len(municipios)}] Processing {municipio}...")
            
            last_date = last_dates.get(municipio)
            result = self.update_municipality(municipio, last_date, end_date)
            
            results.append(result)
            total_downloaded += result.get('downloaded', 0)
            total_inserted += result.get('inserted', 0)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ UPDATE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"📥 Total downloaded: {total_downloaded:,} records")
        logger.info(f"💾 Total inserted: {total_inserted:,} records")
        
        successful = len([r for r in results if r['status'] == 'success'])
        logger.info(f"✅ Successful: {successful}/{len(municipios)}")
        
        return {
            "total_downloaded": total_downloaded,
            "total_inserted": total_inserted,
            "successful": successful,
            "total": len(municipios),
            "results": results
        }
    
    def get_database_status(self) -> pd.DataFrame:
        """
        Get current status of the database.
        
        Returns:
            DataFrame with statistics per municipality.
        """
        if not self._cursor:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        self._cursor.execute("""
            SELECT 
                municipio,
                MIN(datetime) as fecha_inicio,
                MAX(datetime) as ultima_fecha,
                COUNT(*) as total_registros
            FROM climate_observations
            GROUP BY municipio
            ORDER BY municipio
        """)
        
        data = []
        for row in self._cursor.fetchall():
            data.append({
                'municipio': row[0],
                'fecha_inicio': row[1],
                'ultima_fecha': row[2],
                'total_registros': row[3]
            })
        
        return pd.DataFrame(data)


def update_database_from_env() -> Dict:
    """
    Convenience function to update database using environment variables.
    
    Requires the following environment variables:
    - DB_SERVER
    - DB_PORT
    - DB_USER
    - DB_PASSWORD
    - DB_NAME
    
    Returns:
        Dictionary with update statistics.
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    with ClimateDBUpdater(
        server=os.getenv("DB_SERVER"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "1433")
    ) as updater:
        return updater.update_all_municipalities()

