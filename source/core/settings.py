"""
Data storage module for NYC Taxi Pipeline
"""

from pathlib import Path

import pandas as pd

from ..config.settings import Config
from ..utils.logger import logger
from ..utils.exceptions import StorageError


class DataStorage:
    """Handles saving data to various formats."""

    def __init__(self, processed_dir: str = Config.DEFAULT_PROCESSED_DIR):
        """
        Initialize data storage.

        Args:
            processed_dir: Directory to store processed data
        """
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataStorage initialized: processed_dir={processed_dir}")

    def save_to_parquet(
            self,
            df: pd.DataFrame,
            output_path: Path,
            compression: str = Config.DEFAULT_COMPRESSION
    ) -> None:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame to save
            output_path: Output file path
            compression: Compression algorithm (snappy, gzip, etc.)

        Raises:
            StorageError: If save operation fails
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(
                output_path,
                index=False,
                compression=compression,
                engine=Config.DEFAULT_ENGINE
            )

            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(
                f"Saved to Parquet: {output_path} ({len(df):,} records, {file_size_mb:.2f} MB)"
            )

        except Exception as e:
            raise StorageError(f"Failed to save Parquet file: {e}")

    def save_to_csv(
            self,
            df: pd.DataFrame,
            output_path: Path,
            **kwargs
    ) -> None:
        """
        Save DataFrame to CSV format.

        Args:
            df: DataFrame to save
            output_path: Output file path
            **kwargs: Additional arguments for to_csv()

        Raises:
            StorageError: If save operation fails
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, **kwargs)

            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(
                f"Saved to CSV: {output_path} ({len(df):,} records, {file_size_mb:.2f} MB)"
            )

        except Exception as e:
            raise StorageError(f"Failed to save CSV file: {e}")

    def get_processed_path(self, taxi_type: str, year: int, month: int) -> Path:
        """
        Get the path for a processed file.

        Args:
            taxi_type: Type of taxi
            year: Year
            month: Month

        Returns:
            Path to processed file
        """
        filename = Config.get_processed_filename(taxi_type, year, month)
        return self.processed_dir / filename