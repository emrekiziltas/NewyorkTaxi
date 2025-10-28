"""
Data downloader module for NYC Taxi Pipeline
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config.settings import Config
from utils.logger import logger
from utils.http_client import HTTPClient
from utils.exceptions import DownloadError


class DataDownloader:
    """Handles downloading NYC Taxi data from official sources."""

    def __init__(self, raw_dir: str = Config.DEFAULT_RAW_DIR):
        """
        Initialize the data downloader.

        Args:
            raw_dir: Directory to store raw data files
        """
        self.data_dir = Path(raw_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.http_client = HTTPClient()

        logger.info(f"DataDownloader initialized: raw_dir={raw_dir}")

    def download_parquet_file(
            self,
            year: int,
            month: int,
            taxi_type: str = "yellow"
    ) -> Optional[pd.DataFrame]:
        """
        Download NYC Taxi data for specified period.

        Args:
            year: Year of data
            month: Month of data (1-12)
            taxi_type: Type of taxi (yellow, green, fhv, fhvhv)

        Returns:
            DataFrame containing the trip data, or None if not available

        Raises:
            DownloadError: If download fails
            ValueError: If invalid parameters
        """
        # Validate inputs
        if taxi_type not in Config.SUPPORTED_TAXI_TYPES:
            raise ValueError(f"Unsupported taxi type: {taxi_type}. Must be one of {Config.SUPPORTED_TAXI_TYPES}")

        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}. Must be between 1 and 12")

        # Generate filename and paths
        filename = Config.get_filename(taxi_type, year, month)
        url = f"{Config.BASE_URL}/{filename}"
        local_path = self.data_dir / filename

        # Return cached data if available
        if local_path.exists():
            logger.info(f"Loading cached data: {local_path}")
            try:
                return pd.read_parquet(local_path)
            except Exception as e:
                logger.warning(f"Failed to load cached file: {e}. Re-downloading...")
                local_path.unlink()

        # Download file
        try:
            logger.info(f"Downloading: {url}")
            response = self.http_client.get(url, stream=True, timeout=Config.HTTP_TIMEOUT)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Write file with progress tracking
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=Config.CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10 MB
                    if total_size > 0 and downloaded % Config.PROGRESS_LOG_INTERVAL == 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Download completed: {local_path} ({total_size / 1024 / 1024:.2f} MB)")

            # Load and return DataFrame
            df = pd.read_parquet(local_path)
            logger.info(f"Loaded {len(df):,} records")
            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Data not available for {year}-{month:02d}")
                return None
            raise DownloadError(f"HTTP error downloading data: {e}")

        except requests.exceptions.RequestException as e:
            raise DownloadError(f"Network error downloading data: {e}")

        except Exception as e:
            raise DownloadError(f"Unexpected error during download: {e}")

    def close(self):
        """Close HTTP client connection."""
        self.http_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()