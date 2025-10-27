"""
NYC Taxi Data Ingestion Pipeline

A production-ready ETL pipeline for downloading, processing, and storing
NYC Taxi & Limousine Commission trip record data.

Author: Emre Kiziltas
License: MIT
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging with professional formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass


class NYCTaxiDataIngestion:
    """
    NYC Taxi Trip data ingestion and processing pipeline.

    This class handles downloading, validating, and transforming NYC TLC
    trip record data from official sources.

    Attributes:
        base_url (str): Base URL for NYC TLC data
        data_dir (Path): Directory for raw data storage
        processed_dir (Path): Directory for processed data storage
        session (requests.Session): HTTP session with retry logic
    """

    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    SUPPORTED_TAXI_TYPES = ['yellow', 'green', 'fhv', 'fhvhv']

    # Data quality thresholds
    MIN_TRIP_DISTANCE = 0.0
    MAX_TRIP_DISTANCE = 100.0
    MIN_FARE_AMOUNT = 0.0
    MAX_FARE_AMOUNT = 1000.0

    def __init__(
        self,
        raw_dir: str = 'data/raw',
        processed_dir: str = 'data/processed'
    ):
        """
        Initialize the data ingestion pipeline.

        Args:
            raw_dir: Directory path for storing raw downloaded data
            processed_dir: Directory path for storing processed data
        """
        self.data_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Configure HTTP session with retry logic
        self.session = self._create_session()

        logger.info(f"Pipeline initialized: raw_dir={raw_dir}, processed_dir={processed_dir}")

    def _create_session(self) -> requests.Session:
        """
        Create HTTP session with retry strategy.

        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def download_parquet_file(
        self,
        year: int,
        month: int,
        taxi_type: str = 'yellow'
    ) -> Optional[pd.DataFrame]:
        """
        Download NYC Taxi data for specified period.

        Args:
            year: Year of data (e.g., 2024)
            month: Month of data (1-12)
            taxi_type: Type of taxi ('yellow', 'green', 'fhv', 'fhvhv')

        Returns:
            DataFrame containing trip data, or None if download fails

        Raises:
            ValueError: If taxi_type is not supported
            DataIngestionError: If download fails after retries
        """
        if taxi_type not in self.SUPPORTED_TAXI_TYPES:
            raise ValueError(
                f"Unsupported taxi type: {taxi_type}. "
                f"Must be one of {self.SUPPORTED_TAXI_TYPES}"
            )

        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}. Must be between 1 and 12")

        filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
        url = f"{self.BASE_URL}/{filename}"
        local_path = self.data_dir / filename

        # Return cached data if available
        if local_path.exists():
            logger.info(f"Loading cached data: {local_path}")
            try:
                return pd.read_parquet(local_path)
            except Exception as e:
                logger.warning(f"Failed to load cached file: {e}. Re-downloading...")
                local_path.unlink()  # Remove corrupted file

        # Download data
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Save file with progress indication
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
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
            raise DataIngestionError(f"HTTP error downloading data: {e}")

        except requests.exceptions.RequestException as e:
            raise DataIngestionError(f"Network error downloading data: {e}")

        except Exception as e:
            raise DataIngestionError(f"Unexpected error: {e}")

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive data quality checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary containing validation results and metrics
        """
        if df is None or df.empty:
            return {
                "status": "failed",
                "reason": "Empty DataFrame",
                "passed": False
            }

        report = {
            "total_records": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_records": df.duplicated().sum(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "passed": True,
            "warnings": []
        }

        # Check for datetime columns
        date_columns = [col for col in df.columns if 'datetime' in col.lower()]
        for col in date_columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    report[f"{col}_range"] = {
                        "min": str(df[col].min()),
                        "max": str(df[col].max())
                    }
                else:
                    report["warnings"].append(f"{col} is not datetime type")
            except Exception as e:
                logger.warning(f"Could not analyze {col}: {e}")

        # Check missing value percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 50].to_dict()
        if high_missing:
            report["warnings"].append(f"High missing values: {high_missing}")

        # Check for numeric anomalies
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col in ['trip_distance', 'fare_amount', 'total_amount']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    report["warnings"].append(
                        f"{col}: {negative_count} negative values found"
                    )

        logger.info(f"Data quality check completed: {len(report['warnings'])} warnings")
        return report

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations and feature engineering.

        Args:
            df: Raw DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if df is None or df.empty:
            logger.warning("Cannot transform empty DataFrame")
            return df

        logger.info("Starting data transformations...")
        df_transformed = df.copy()
        initial_rows = len(df_transformed)

        # 1. Convert datetime columns
        date_columns = [col for col in df_transformed.columns if 'datetime' in col.lower()]
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df_transformed[col]):
                logger.info(f"Converting {col} to datetime")
                df_transformed[col] = pd.to_datetime(df_transformed[col], errors='coerce')

        # 2. Remove invalid records
        numeric_validations = {
            'trip_distance': (self.MIN_TRIP_DISTANCE, self.MAX_TRIP_DISTANCE),
            'fare_amount': (self.MIN_FARE_AMOUNT, self.MAX_FARE_AMOUNT),
            'total_amount': (self.MIN_FARE_AMOUNT, self.MAX_FARE_AMOUNT),
            'tip_amount': (0, self.MAX_FARE_AMOUNT),
            'tolls_amount': (0, self.MAX_FARE_AMOUNT)
        }

        for col, (min_val, max_val) in numeric_validations.items():
            if col in df_transformed.columns:
                before = len(df_transformed)
                df_transformed = df_transformed[
                    (df_transformed[col] >= min_val) &
                    (df_transformed[col] <= max_val)
                ]
                removed = before - len(df_transformed)
                if removed > 0:
                    logger.info(f"Removed {removed:,} invalid {col} records")

        # 3. Feature engineering: Extract datetime features
        pickup_col = self._get_pickup_column(df_transformed)

        if pickup_col and pd.api.types.is_datetime64_any_dtype(df_transformed[pickup_col]):
            logger.info(f"Extracting datetime features from {pickup_col}")
            df_transformed['pickup_hour'] = df_transformed[pickup_col].dt.hour
            df_transformed['pickup_day'] = df_transformed[pickup_col].dt.day
            df_transformed['pickup_weekday'] = df_transformed[pickup_col].dt.dayofweek
            df_transformed['pickup_month'] = df_transformed[pickup_col].dt.month
            df_transformed['pickup_year'] = df_transformed[pickup_col].dt.year
            df_transformed['is_weekend'] = df_transformed['pickup_weekday'].isin([5, 6])

            # Add time period categories
            df_transformed['time_period'] = pd.cut(
                df_transformed['pickup_hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
        else:
            logger.warning("Pickup datetime column not found or invalid")

        # 4. Calculate derived metrics
        if 'trip_distance' in df_transformed.columns and 'total_amount' in df_transformed.columns:
            df_transformed['cost_per_mile'] = (
                df_transformed['total_amount'] /
                df_transformed['trip_distance'].replace(0, pd.NA)
            )

        if 'tip_amount' in df_transformed.columns and 'fare_amount' in df_transformed.columns:
            df_transformed['tip_percentage'] = (
                df_transformed['tip_amount'] /
                df_transformed['fare_amount'].replace(0, pd.NA) * 100
            )

        final_rows = len(df_transformed)
        removed_pct = ((initial_rows - final_rows) / initial_rows * 100)
        logger.info(
            f"Transformation completed: {final_rows:,} records "
            f"({removed_pct:.2f}% filtered)"
        )

        return df_transformed

    def _get_pickup_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the pickup datetime column name."""
        pickup_columns = ['tpep_pickup_datetime', 'lpep_pickup_datetime', 'pickup_datetime']
        for col in pickup_columns:
            if col in df.columns:
                return col
        return None

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: Path,
        compression: str = 'snappy'
    ) -> None:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame to save
            output_path: Output file path
            compression: Compression algorithm ('snappy', 'gzip', 'brotli')
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(
                output_path,
                index=False,
                compression=compression,
                engine='pyarrow'
            )

            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(
                f"Saved to Parquet: {output_path} "
                f"({len(df):,} records, {file_size_mb:.2f} MB)"
            )

        except Exception as e:
            raise DataIngestionError(f"Failed to save Parquet file: {e}")

    def process_month(
        self,
        year: int,
        month: int,
        taxi_type: str = 'yellow',
        skip_if_exists: bool = True
    ) -> Optional[Path]:
        """
        Complete processing pipeline for one month of data.

        Args:
            year: Year to process
            month: Month to process (1-12)
            taxi_type: Type of taxi data
            skip_if_exists: Skip processing if output file exists

        Returns:
            Path to processed file, or None if processing failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {taxi_type.upper()} Taxi - {year}-{month:02d}")
        logger.info(f"{'='*60}")

        output_path = self.processed_dir / f"{taxi_type}_taxi_{year}_{month:02d}_processed.parquet"

        # Skip if already processed
        if skip_if_exists and output_path.exists():
            logger.info(f"Already processed: {output_path}")
            return output_path

        try:
            # Step 1: Download
            df = self.download_parquet_file(year, month, taxi_type)
            if df is None:
                return None

            # Step 2: Validate
            quality_report = self.validate_data_quality(df)
            if not quality_report['passed']:
                logger.error(f"Data quality check failed: {quality_report['reason']}")
                return None

            # Step 3: Transform
            df_transformed = self.transform_data(df)
            if df_transformed is None or df_transformed.empty:
                logger.error("Transformation resulted in empty DataFrame")
                return None

            # Step 4: Save
            self.save_to_parquet(df_transformed, output_path)

            logger.info(f"✓ Successfully processed {year}-{month:02d}")
            return output_path

        except Exception as e:
            logger.error(f"✗ Failed to process {year}-{month:02d}: {e}", exc_info=True)
            return None


def merge_parquet_files(
    input_pattern: str = "data/processed/yellow_taxi_*_processed.parquet",
    output_path: str = "data/processed/yellow_taxi_merged.parquet"
) -> Optional[Path]:
    """
    Merge multiple processed Parquet files into a single file.

    Args:
        input_pattern: Glob pattern for input files
        output_path: Path for merged output file

    Returns:
        Path to merged file, or None if merge failed
    """
    from pathlib import Path
    import pandas as pd

    logger.info(f"\n{'='*60}")
    logger.info("Starting file merge process")
    logger.info(f"{'='*60}")

    parquet_files = sorted(Path().glob(input_pattern))

    if not parquet_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return None

    logger.info(f"Found {len(parquet_files)} files to merge")

    try:
        dfs = []
        for file_path in parquet_files:
            logger.info(f"Loading: {file_path.name}")
            df = pd.read_parquet(file_path)
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged {len(merged_df):,} total records")

        # Save merged file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        merged_df.to_parquet(output_path_obj, index=False, compression='snappy')
        file_size_mb = output_path_obj.stat().st_size / 1024 / 1024

        logger.info(f"✓ Merged file saved: {output_path_obj} ({file_size_mb:.2f} MB)")
        return output_path_obj

    except Exception as e:
        logger.error(f"✗ Merge failed: {e}", exc_info=True)
        return None


def main():
    """Main pipeline execution function."""

    start_time = datetime.now()

    logger.info("\n" + "="*60)
    logger.info("NYC TAXI DATA PIPELINE")
    logger.info("="*60 + "\n")

    # Initialize pipeline
    pipeline = NYCTaxiDataIngestion(
        raw_dir='data/raw',
        processed_dir='data/processed'
    )

    # Configuration
    YEARS = range(2023, 2025)
    MONTHS = range(1, 13)
    TAXI_TYPE = 'yellow'

    # Process all months
    processed_files = []
    failed_files = []

    for year in YEARS:
        for month in MONTHS:
            result = pipeline.process_month(year, month, TAXI_TYPE)
            if result:
                processed_files.append(result)
            else:
                failed_files.append(f"{year}-{month:02d}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"✓ Successfully processed: {len(processed_files)} files")
    if failed_files:
        logger.warning(f"✗ Failed to process: {len(failed_files)} files")
        logger.warning(f"  Failed periods: {', '.join(failed_files)}")

    # Merge files
    if processed_files:
        logger.info("\nStarting merge process...")
        merged_file = merge_parquet_files(
            input_pattern=f"data/processed/{TAXI_TYPE}_taxi_*_processed.parquet",
            output_path=f"data/processed/{TAXI_TYPE}_taxi_merged.parquet"
        )
        if merged_file:
            logger.info(f"✓ All files successfully merged: {merged_file}")

    # Final statistics
    elapsed_time = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed_time}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)