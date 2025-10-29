"""
Weather enrichment module for NYC Taxi Pipeline
"""

import pandas as pd
import requests
from pathlib import Path

from source.config.settings import Config
from source.utils.logger import logger
from source.utils.exceptions import EnrichmentError


class WeatherEnricher:
    """Fetches and adds weather data from Open-Meteo API."""

    def __init__(self):
        """Initialize weather enricher."""
        logger.info("WeatherEnricher initialized")

    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch hourly NYC weather data from Open-Meteo API.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with datetime, temperature, precipitation, and weather codes.

        Raises:
            EnrichmentError: If weather API request fails
        """
        url = (
            f"{Config.WEATHER_API_URL}"
            f"?latitude={Config.NYC_LATITUDE}&longitude={Config.NYC_LONGITUDE}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,precipitation,weathercode"
            f"&timezone={Config.WEATHER_TIMEZONE}"
        )

        try:
            logger.info(f"Fetching weather data from {start_date} to {end_date}")
            response = requests.get(url, timeout=60)

            if not response.ok:
                raise EnrichmentError(
                    f"Weather API failed ({response.status_code}): {response.text[:200]}"
                )

            data = response.json()

            if "hourly" not in data or "time" not in data["hourly"]:
                raise EnrichmentError(f"Unexpected weather API response: {data}")

            weather_df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "temperature_c": data["hourly"]["temperature_2m"],
                "weather_code": data["hourly"]["weathercode"]
            })

            logger.info(f"Fetched {len(weather_df):,} hourly weather records")

            # Attach human-readable labels if lookup exists
            weather_df = self._add_weather_descriptions(weather_df)

            return weather_df

        except requests.exceptions.RequestException as e:
            raise EnrichmentError(f"Failed to fetch weather data: {e}")
        except Exception as e:
            raise EnrichmentError(f"Unexpected error fetching weather: {e}")

    def add_weather_info(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather info based on pickup_datetime.

        Args:
            df: Trip data DataFrame
            weather_df: Weather data DataFrame

        Returns:
            DataFrame with weather columns added.
        """
        try:
            pickup_col = self._find_pickup_column(df)
            if pickup_col is None:
                raise EnrichmentError("Pickup datetime column not found")

            # Round pickup time to the nearest hour
            df["pickup_hour"] = pd.to_datetime(df[pickup_col]).dt.floor("h")

            merged = pd.merge(
                df,
                weather_df,
                how="left",
                left_on="pickup_hour",
                right_on="datetime"
            )

            merged.drop(columns=["datetime"], inplace=True, errors='ignore')

            logger.info("Added weather information successfully")
            return merged

        except Exception as e:
            raise EnrichmentError(f"Failed to add weather info: {e}")

    def _add_weather_descriptions(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Attach human-readable weather descriptions from CSV lookup."""
        lookup_path = Path("data/lookup/weather_codes.csv")

        if lookup_path.exists():
            try:
                logger.info(f"Reading weather code lookup from: {lookup_path}")
                code_lookup = pd.read_csv(lookup_path)
                logger.info(f"Lookup CSV columns: {code_lookup.columns.tolist()}")

                # Ensure merge key types match
                code_lookup["weather_code"] = code_lookup["weather_code"].astype(int)
                weather_df["weather_code"] = weather_df["weather_code"].astype(int)

                weather_df = weather_df.merge(code_lookup, how="left", on="weather_code")
                logger.info("Mapped weather codes to descriptions")
            except Exception as e:
                logger.error("Failed to merge weather code descriptions", exc_info=True)
        else:
            logger.warning(f"Weather code lookup not found: {lookup_path}")

        return weather_df

    def _find_pickup_column(self, df: pd.DataFrame) -> str:
        """Find pickup datetime column name."""
        for col in df.columns:
            if "pickup" in col.lower() and "datetime" in col.lower():
                return col
        return None
