# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Climate data fetcher using Open-Meteo API.

This module provides a class for downloading and managing historical
and forecast climate data for municipalities in La Guajira, Colombia.
"""

import time
import random
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytz
import pandas as pd
import requests
from requests.exceptions import HTTPError


logger = logging.getLogger(__name__)


@dataclass
class DateRange:
    """Represents a date range for data fetching."""
    start: datetime
    end: datetime

    def to_str_tuple(self) -> Tuple[str, str]:
        """Return tuple of ISO date strings (YYYY-MM-DD)."""
        return self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d")


class ClimateDataFetcher:
    """
    A class to fetch and manage climate data from Open-Meteo API.

    This class provides methods to download historical and forecast weather
    data for municipalities in La Guajira, Colombia, with support for
    incremental updates and rate limiting.

    Attributes:
        municipio: Name of the municipality to fetch data for.
        date_range: DateRange object specifying the time window.
        data_dir: Directory where CSV files are stored.
        timezone: Timezone for datetime handling.

    Example:
        >>> from datetime import datetime, timedelta
        >>> fetcher = ClimateDataFetcher(
        ...     municipio="riohacha",
        ...     start_date=datetime.now() - timedelta(days=30),
        ...     end_date=datetime.now()
        ... )
        >>> df = fetcher.fetch()
        >>> fetcher.save()
    """

    TIMEZONE = pytz.timezone("America/Bogota")
    USER_AGENT = "GuajiraWindForecast/1.0 (Academic Research)"
    
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    HOURLY_FIELDS_ALL = (
        "wind_speed_10m,wind_direction_10m,temperature_2m,"
        "relative_humidity_2m,precipitation"
    )
    HOURLY_FIELDS_WIND = "wind_speed_10m,wind_direction_10m"
    
    MUNICIPIOS: Dict[str, Tuple[float, float]] = {
        "riohacha": (11.5447, -72.9072),
        "maicao": (11.3776, -72.2391),
        "uribia": (11.7147, -72.2652),
        "manaure": (11.7794, -72.4469),
        "fonseca": (10.8306, -72.8517),
        "san_juan_del_cesar": (10.7695, -73.0030),
        "albania": (11.1608, -72.5922),
        "barrancas": (10.9577, -72.7947),
        "distraccion": (10.8958, -72.8869),
        "el_molino": (10.6528, -72.9247),
        "hatonuevo": (11.0694, -72.7647),
        "la_jagua_del_pilar": (10.5108, -73.0714),
        "mingueo": (11.2000, -73.3667),
    }

    DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"

    def __init__(
        self,
        municipio: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        data_dir: Optional[Path] = None,
        wind_only: bool = False,
        hour_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize the ClimateDataFetcher.

        Args:
            municipio: Name of the municipality (must be in MUNICIPIOS).
            start_date: Start date for data fetching.
            end_date: End date for data fetching. Defaults to now.
            data_dir: Optional custom path for data storage.
                Defaults to data/raw/.
            wind_only: If True, fetch only wind-related fields.
            hour_range: Optional tuple (start_hour, end_hour) to filter data.
                Example: (6, 18) for daytime hours only.

        Raises:
            ValueError: If municipio is not recognized.
        """
        self.municipio = self._normalize_municipio(municipio)
        
        if self.municipio not in self.MUNICIPIOS:
            available = ", ".join(sorted(self.MUNICIPIOS.keys()))
            raise ValueError(
                f"Municipio '{municipio}' no reconocido. "
                f"Opciones disponibles: {available}"
            )
        
        self.lat, self.lon = self.MUNICIPIOS[self.municipio]
        self.end_date = end_date or datetime.now(self.TIMEZONE)
        self.date_range = DateRange(start=start_date, end=self.end_date)
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_PATH
        self.wind_only = wind_only
        self.hour_range = hour_range
        
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.USER_AGENT})
        
        self._data: Optional[pd.DataFrame] = None
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Create the data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_municipio(name: str) -> str:
        """Normalize municipality name to standard format."""
        return name.strip().lower().replace(" ", "_")

    @property
    def hourly_fields(self) -> str:
        """Return the appropriate hourly fields based on configuration."""
        return self.HOURLY_FIELDS_WIND if self.wind_only else self.HOURLY_FIELDS_ALL

    @property
    def csv_path(self) -> Path:
        """Return the path to the CSV file for this municipality."""
        return self.data_dir / f"open_meteo_{self.municipio}.csv"

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Return the current data DataFrame."""
        return self._data

    def _fetch_from_api(
        self,
        url: str,
        params: Dict,
        retries: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch data from Open-Meteo API with retry logic.

        Args:
            url: API endpoint URL.
            params: Query parameters for the request.
            retries: Maximum number of retry attempts.

        Returns:
            DataFrame with hourly weather data.

        Raises:
            HTTPError: If all retries fail.
        """
        for attempt in range(retries):
            try:
                response = self._session.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if "hourly" not in data or "time" not in data["hourly"]:
                    logger.warning(f"No hourly data in response from {url}")
                    return pd.DataFrame()
                
                df = pd.DataFrame({"datetime": data["hourly"]["time"]})
                for key, values in data["hourly"].items():
                    if key != "time":
                        df[key] = values
                
                return df
                
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait_time = (attempt + 1) * random.uniform(3, 6)
                    logger.warning(
                        f"Rate limit alcanzado. Reintentando en {wait_time:.1f}s... "
                        f"(intento {attempt + 1}/{retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise
        
        logger.error("Error persistente al descargar datos después de reintentos.")
        return pd.DataFrame()

    def _fetch_archive(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Open-Meteo Archive API.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with historical weather data.
        """
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": self.hourly_fields,
            "timezone": "America/Bogota",
        }
        return self._fetch_from_api(self.ARCHIVE_URL, params)

    def _fetch_forecast(
        self,
        past_days: int = 3,
        forecast_days: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch recent and forecast data from Open-Meteo Forecast API.

        Args:
            past_days: Number of past days to include.
            forecast_days: Number of forecast days to include.

        Returns:
            DataFrame with recent and forecast weather data.
        """
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": self.hourly_fields,
            "timezone": "America/Bogota",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
        return self._fetch_from_api(self.FORECAST_URL, params)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame with additional columns and filtering.

        Args:
            df: Raw DataFrame from API.

        Returns:
            Normalized DataFrame with additional metadata columns.
        """
        if df.empty:
            return df
        
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour"] = df["datetime"].dt.hour
        df["date"] = df["datetime"].dt.date
        df["municipio"] = self.municipio
        
        # Apply hour filter if specified
        if self.hour_range:
            start_hour, end_hour = self.hour_range
            df = df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)]
        
        return df

    def fetch(self, block_days: int = 180) -> pd.DataFrame:
        """
        Fetch climate data for the configured date range.

        Downloads data in blocks to avoid API limits. Automatically
        combines archive (historical) and forecast (recent) data.

        Args:
            block_days: Number of days per request block. Defaults to 180.

        Returns:
            DataFrame with all fetched climate data.

        Example:
            >>> fetcher = ClimateDataFetcher("riohacha", start, end)
            >>> df = fetcher.fetch()
            >>> print(f"Downloaded {len(df)} records")
        """
        all_data: List[pd.DataFrame] = []
        start_str, end_str = self.date_range.to_str_tuple()
        
        now_local = datetime.now(self.TIMEZONE)
        yesterday = (now_local - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Fetch archive data in blocks
        current = self.date_range.start
        while current < self.date_range.end:
            block_end = min(
                current + timedelta(days=block_days),
                self.date_range.end
            )
            block_end_str = block_end.strftime("%Y-%m-%d")
            
            # Only fetch from archive if the date is before yesterday
            if current.strftime("%Y-%m-%d") <= yesterday:
                arch_end = min(block_end_str, yesterday)
                logger.info(
                    f"Descargando archivo: {current.strftime('%Y-%m-%d')} "
                    f"a {arch_end} para {self.municipio}"
                )
                df = self._fetch_archive(
                    current.strftime("%Y-%m-%d"),
                    arch_end
                )
                if not df.empty:
                    df = self._normalize_dataframe(df)
                    all_data.append(df)
                
                time.sleep(1)  # Rate limiting pause
            
            current = block_end + timedelta(days=1)
        
        # Fetch recent data from forecast API
        logger.info(f"Descargando datos recientes para {self.municipio}")
        forecast_df = self._fetch_forecast(past_days=3, forecast_days=1)
        if not forecast_df.empty:
            forecast_df = self._normalize_dataframe(forecast_df)
            all_data.append(forecast_df)
        
        if not all_data:
            self._data = pd.DataFrame()
            return self._data
        
        # Combine and deduplicate
        self._data = pd.concat(all_data, ignore_index=True)
        self._data.sort_values("datetime", inplace=True)
        self._data.drop_duplicates(
            subset=["municipio", "datetime"],
            keep="last",
            inplace=True
        )
        self._data.reset_index(drop=True, inplace=True)
        
        logger.info(f"Total de registros descargados: {len(self._data)}")
        return self._data

    def load_existing(self) -> pd.DataFrame:
        """
        Load existing data from CSV file.

        Returns:
            DataFrame with existing data, or empty DataFrame if file doesn't exist.
        """
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        return pd.DataFrame()

    def save(self) -> Optional[Path]:
        """
        Save current data to CSV file using atomic write.

        Returns:
            Path to the saved file, or None if no data to save.
        """
        if self._data is None or self._data.empty:
            logger.warning("No hay datos para guardar.")
            return None
        
        tmp_path = self.csv_path.with_suffix(".tmp.csv")
        self._data.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, self.csv_path)
        
        logger.info(f"Datos guardados en: {self.csv_path}")
        return self.csv_path

    def fetch_incremental(self) -> Dict:
        """
        Perform incremental fetch, only downloading new data.

        Loads existing data, determines the last timestamp, and only
        fetches data that is newer.

        Returns:
            Dict with operation summary including new_rows and success status.

        Example:
            >>> fetcher = ClimateDataFetcher("riohacha", start, end)
            >>> result = fetcher.fetch_incremental()
            >>> print(f"Added {result['new_rows']} new records")
        """
        existing = self.load_existing()
        
        if not existing.empty:
            last_ts = pd.to_datetime(existing["datetime"]).max()
            # Update start date to fetch only new data
            self.date_range = DateRange(
                start=last_ts.to_pydatetime() + timedelta(hours=1),
                end=self.end_date
            )
        
        new_data = self.fetch()
        
        if new_data.empty:
            self._data = existing
            new_rows = 0
        else:
            self._data = pd.concat([existing, new_data], ignore_index=True)
            self._data.drop_duplicates(
                subset=["municipio", "datetime"],
                keep="last",
                inplace=True
            )
            new_rows = len(self._data) - len(existing)
        
        self.save()
        
        return {
            "municipio": self.municipio,
            "new_rows": int(new_rows),
            "total_rows": len(self._data) if self._data is not None else 0,
            "file": str(self.csv_path),
            "last_timestamp": (
                self._data["datetime"].max().strftime("%Y-%m-%d %H:%M")
                if self._data is not None and not self._data.empty
                else None
            ),
            "success": True,
        }

    @classmethod
    def get_available_municipios(cls) -> List[str]:
        """
        Return list of available municipality names.

        Returns:
            Sorted list of municipality identifiers.
        """
        return sorted(cls.MUNICIPIOS.keys())

    def __repr__(self) -> str:
        """Return string representation of the fetcher."""
        start_str, end_str = self.date_range.to_str_tuple()
        return (
            f"ClimateDataFetcher(municipio='{self.municipio}', "
            f"range='{start_str}' to '{end_str}')"
        )