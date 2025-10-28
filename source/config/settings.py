"""
Configuration settings for NYC Taxi Pipeline

All constants, URLs, and threshold values are defined here.
"""

from pathlib import Path


class Config:
    """Main configuration class"""

    # API URLs
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    BOROUGH_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Supported taxi types
    SUPPORTED_TAXI_TYPES = ['yellow', 'green', 'fhv', 'fhvhv']

    # Data quality thresholds
    MIN_TRIP_DISTANCE = 0.0
    MAX_TRIP_DISTANCE = 100.0
    MIN_FARE_AMOUNT = 0.0
    MAX_FARE_AMOUNT = 1000.0
    MAX_TIP_AMOUNT = 1000.0
    MAX_TOLLS_AMOUNT = 1000.0

    # Directory paths
    DEFAULT_RAW_DIR = "data/raw"
    DEFAULT_PROCESSED_DIR = "data/processed"

    # HTTP settings
    HTTP_TIMEOUT = 300
    RETRY_TOTAL = 3
    RETRY_BACKOFF_FACTOR = 1
    RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]

    # Download settings
    CHUNK_SIZE = 8192
    PROGRESS_LOG_INTERVAL = 1024 * 1024 * 10  # 10 MB

    # Parquet settings
    DEFAULT_COMPRESSION = "snappy"
    DEFAULT_ENGINE = "pyarrow"

    # Weather API settings
    NYC_LATITUDE = 40.7128
    NYC_LONGITUDE = -74.0060
    WEATHER_TIMEZONE = "America/New_York"

    # Datetime features
    TIME_PERIOD_BINS = [0, 6, 12, 18, 24]
    TIME_PERIOD_LABELS = ['Night', 'Morning', 'Afternoon', 'Evening']

    # Validation thresholds
    HIGH_MISSING_VALUE_THRESHOLD = 50  # percentage

    @staticmethod
    def get_filename(taxi_type: str, year: int, month: int) -> str:
        """Generate standardized filename"""
        return f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"

    @staticmethod
    def get_processed_filename(taxi_type: str, year: int, month: int) -> str:
        """Generate processed filename"""
        return f"{taxi_type}_taxi_{year}_{month:02d}_processed.parquet"