from pathlib import Path
from typing import Optional
import pandas as pd
import requests

from source.config.settings import Config
from source.utils.logger import logger
from source.utils.http_client import HTTPClient
from source.utils.exceptions import DownloadError


class DataDownloader:
    """Handles downloading NYC Taxi data from official sources."""

    def __init__(self, raw_dir: str = Config.DEFAULT_RAW_DIR, processed_dir: str = "data/processed"):
        self.data_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.http_client = HTTPClient()

        logger.info(f"DataDownloader initialized: raw_dir={raw_dir}")

    def download_parquet_file(self, year: int, month: int, taxi_type: str = "yellow") -> Optional[pd.DataFrame]:
        """Download NYC Taxi data for specified period."""
        if taxi_type not in Config.SUPPORTED_TAXI_TYPES:
            raise ValueError(f"Unsupported taxi type: {taxi_type}. Must be one of {Config.SUPPORTED_TAXI_TYPES}")
        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}. Must be between 1 and 12")

        filename = Config.get_filename(taxi_type, year, month)
        url = f"{Config.BASE_URL}/{filename}"
        local_path = self.data_dir / filename

        if local_path.exists():
            logger.info(f"Loading cached data: {local_path}")
            try:
                return pd.read_parquet(local_path)
            except Exception as e:
                logger.warning(f"Failed to load cached file: {e}. Re-downloading...")
                local_path.unlink()

        try:
            logger.info(f"Downloading: {url}")
            response = self.http_client.get(url, stream=True, timeout=Config.HTTP_TIMEOUT)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=Config.CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % Config.PROGRESS_LOG_INTERVAL == 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Download completed: {local_path} ({total_size / 1024 / 1024:.2f} MB)")
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
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
