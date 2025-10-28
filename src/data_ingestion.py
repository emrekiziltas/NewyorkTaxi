"""
NYC Taxi Data Ingestion Pipeline

A production-ready ETL pipeline for downloading, processing, and storing
NYC Taxi & Limousine Commission trip record data.

Author: Emre Kiziltas
License: MIT
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging with professional formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class NYCTaxiDataIngestion:
    """
    NYC Taxi Trip data ingestion and processing pipeline.

    This class handles downloading, validating, and transforming NYC TLC
    trip record data from official sources.
    """

    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    SUPPORTED_TAXI_TYPES = ['yellow', 'green', 'fhv', 'fhvhv']

    # Data quality thresholds
    MIN_TRIP_DISTANCE = 0.0
    MAX_TRIP_DISTANCE = 100.0
    MIN_FARE_AMOUNT = 0.0
    MAX_FARE_AMOUNT = 1000.0

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """Initialize the data ingestion pipeline."""
        self.data_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Configure HTTP session with retry logic
        self.session = self._create_session()

        logger.info(f"Pipeline initialized: raw_dir={raw_dir}, processed_dir={processed_dir}")

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
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
            self, year: int, month: int, taxi_type: str = "yellow"
    ) -> Optional[pd.DataFrame]:
        """Download NYC Taxi data for specified period."""
        if taxi_type not in self.SUPPORTED_TAXI_TYPES:
            raise ValueError(f"Unsupported taxi type: {taxi_type}")

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
                local_path.unlink()

        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0:
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
            raise DataIngestionError(f"HTTP error downloading data: {e}")

        except requests.exceptions.RequestException as e:
            raise DataIngestionError(f"Network error downloading data: {e}")

        except Exception as e:
            raise DataIngestionError(f"Unexpected error: {e}")

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Perform comprehensive data quality checks."""
        if df is None or df.empty:
            return {"status": "failed", "reason": "Empty DataFrame", "passed": False}

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

        # Check datetime columns
        date_columns = [c for c in df.columns if "datetime" in c.lower()]
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

        # Missing values > 50%
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50].to_dict()
        if high_missing:
            report["warnings"].append(f"High missing values: {high_missing}")

        # Numeric anomalies
        for col in df.select_dtypes(include=['number']).columns:
            if col in ['trip_distance', 'fare_amount', 'total_amount']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    report["warnings"].append(f"{col}: {negative_count} negative values found")

        logger.info(f"Data quality check completed: {len(report['warnings'])} warnings")
        return report

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations and feature engineering."""
        if df is None or df.empty:
            logger.warning("Cannot transform empty DataFrame")
            return df

        logger.info("Starting data transformations...")
        df_transformed = df.copy()
        initial_rows = len(df_transformed)

        # Convert datetime columns
        date_columns = [c for c in df_transformed.columns if "datetime" in c.lower()]
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df_transformed[col]):
                logger.info(f"Converting {col} to datetime")
                df_transformed[col] = pd.to_datetime(df_transformed[col], errors="coerce")

        # Remove invalid numeric values
        validations = {
            "trip_distance": (self.MIN_TRIP_DISTANCE, self.MAX_TRIP_DISTANCE),
            "fare_amount": (self.MIN_FARE_AMOUNT, self.MAX_FARE_AMOUNT),
            "total_amount": (self.MIN_FARE_AMOUNT, self.MAX_FARE_AMOUNT),
            "tip_amount": (0, self.MAX_FARE_AMOUNT),
            "tolls_amount": (0, self.MAX_FARE_AMOUNT)
        }

        for col, (min_val, max_val) in validations.items():
            if col in df_transformed.columns:
                before = len(df_transformed)
                df_transformed = df_transformed[
                    (df_transformed[col] >= min_val) & (df_transformed[col] <= max_val)
                    ]
                removed = before - len(df_transformed)
                if removed > 0:
                    logger.info(f"Removed {removed:,} invalid {col} records")

        # Extract datetime features
        pickup_col = self._get_pickup_column(df_transformed)
        if pickup_col and pd.api.types.is_datetime64_any_dtype(df_transformed[pickup_col]):
            df_transformed['pickup_hour'] = df_transformed[pickup_col].dt.hour
            df_transformed['pickup_day'] = df_transformed[pickup_col].dt.day
            df_transformed['pickup_weekday'] = df_transformed[pickup_col].dt.dayofweek
            df_transformed['pickup_month'] = df_transformed[pickup_col].dt.month
            df_transformed['pickup_year'] = df_transformed[pickup_col].dt.year
            df_transformed['is_weekend'] = df_transformed['pickup_weekday'].isin([5, 6])
            df_transformed['time_period'] = pd.cut(
                df_transformed['pickup_hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )

        # Derived metrics
        if 'trip_distance' in df_transformed.columns and 'total_amount' in df_transformed.columns:
            df_transformed['cost_per_mile'] = (
                    df_transformed['total_amount'] / df_transformed['trip_distance'].replace(0, pd.NA)
            )

        if 'tip_amount' in df_transformed.columns and 'fare_amount' in df_transformed.columns:
            df_transformed['tip_percentage'] = (
                    df_transformed['tip_amount'] / df_transformed['fare_amount'].replace(0, pd.NA) * 100
            )

        final_rows = len(df_transformed)
        removed_pct = ((initial_rows - final_rows) / initial_rows * 100)
        logger.info(f"Transformation completed: {final_rows:,} records ({removed_pct:.2f}% filtered)")

        return df_transformed

    def _get_pickup_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the pickup datetime column name."""
        for col in ['tpep_pickup_datetime', 'lpep_pickup_datetime', 'pickup_datetime']:
            if col in df.columns:
                return col
        return None

    def save_to_parquet(self, df: pd.DataFrame, output_path: Path, compression: str = "snappy") -> None:
        """Save DataFrame to Parquet format."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False, compression=compression, engine="pyarrow")
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"Saved to Parquet: {output_path} ({len(df):,} records, {file_size_mb:.2f} MB)")
        except Exception as e:
            raise DataIngestionError(f"Failed to save Parquet file: {e}")

    def add_borough_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pickup and dropoff borough information based on TLC zone lookup.

        Args:
            df: DataFrame containing PULocationID and DOLocationID

        Returns:
            DataFrame with new columns: pickup_borough, dropoff_borough
        """
        lookup_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
        try:
            lookup_df = pd.read_csv(lookup_url)
            logger.info(f"Loaded TLC zone lookup table ({len(lookup_df)} rows)")
            lookup_path = Path("data/raw/taxi_zone_lookup.csv")
            if not lookup_path.exists():
                lookup_path.parent.mkdir(parents=True, exist_ok=True)
                pd.read_csv(lookup_url).to_csv(lookup_path, index=False)
            lookup_df = pd.read_csv(lookup_path)
        except Exception as e:
            logger.error(f"Failed to load borough lookup: {e}")
            return df

        # Create mapping dictionaries
        id_to_borough = lookup_df.set_index("LocationID")["Borough"].to_dict()

        # Add borough columns if possible
        if "PULocationID" in df.columns:
            df["pickup_borough"] = df["PULocationID"].map(id_to_borough)
        if "DOLocationID" in df.columns:
            df["dropoff_borough"] = df["DOLocationID"].map(id_to_borough)

        return df

    def process_month(
            self, year: int, month: int, taxi_type: str = "yellow", skip_if_exists: bool = True
    ) -> Optional[Path]:
        """Complete processing pipeline for one month of data."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {taxi_type.upper()} Taxi - {year}-{month:02d}")
        logger.info(f"{'=' * 60}")

        output_path = self.processed_dir / f"{taxi_type}_taxi_{year}_{month:02d}_processed.parquet"

        if skip_if_exists and output_path.exists():
            logger.info(f"Already processed: {output_path}")
            return output_path

        try:
            df = self.download_parquet_file(year, month, taxi_type)
            if df is None:
                return None

            quality_report = self.validate_data_quality(df)
            if not quality_report["passed"]:
                logger.error(f"Data quality check failed: {quality_report['reason']}")
                return None

            df_transformed = self.transform_data(df)

            df_transformed = self.add_borough_info(df_transformed)

            if df_transformed is None or df_transformed.empty:
                logger.error("Transformation resulted in empty DataFrame")
                return None

            self.save_to_parquet(df_transformed, output_path)
            logger.info(f" Successfully processed {year}-{month:02d}")
            return output_path

        except Exception as e:
            logger.error(f"✗ Failed to process {year}-{month:02d}: {e}", exc_info=True)
            return None

def fetch_weather_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly NYC weather data from Open-Meteo API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=40.7128&longitude=-74.0060"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,precipitation"
        "&timezone=America%2FNew_York"
    )

    response = requests.get(url, timeout=60)
    if not response.ok:
        raise RuntimeError(f"Weather API failed ({response.status_code}): {response.text[:200]}")

    data = response.json()
    if "hourly" not in data or "time" not in data["hourly"]:
        raise RuntimeError(f"Unexpected weather API response: {data}")

    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
        "precipitation_mm": data["hourly"]["precipitation"]
    })
    return weather_df


def add_weather_info(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Add weather info based on pickup_datetime."""
    pickup_col = next((c for c in df.columns if "pickup" in c and "datetime" in c), None)
    if pickup_col is None:
        raise ValueError("pickup_datetime column not found")

    df["pickup_hour"] = pd.to_datetime(df[pickup_col]).dt.floor("H")
    merged = pd.merge(df, weather_df, how="left", left_on="pickup_hour", right_on="datetime")
    merged.drop(columns=["datetime"], inplace=True)
    return merged



def main():
    """Main pipeline execution function."""
    start_time = datetime.now()

    logger.info("\n" + "=" * 60)
    logger.info("NYC TAXI DATA PIPELINE")
    logger.info("=" * 60 + "\n")

    pipeline = NYCTaxiDataIngestion(raw_dir="data/raw", processed_dir="data/processed")

    YEARS = range(2023, 2025)
    MONTHS = range(1, 13)
    TAXI_TYPE = "yellow"

    processed_files, failed_files = [], []

    for year in YEARS:
        for month in MONTHS:
            result = pipeline.process_month(year, month, TAXI_TYPE)
            if result:
                processed_files.append(result)
            else:
                failed_files.append(f"{year}-{month:02d}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    logger.info(f"[OK] Successfully processed: {len(processed_files)} files")
    logger.error(f"[X] Failed to process some files")
    if failed_files:
        logger.warning(f"✗ Failed to process: {len(failed_files)} files")
        logger.warning(f"  Failed periods: {', '.join(failed_files)}")

    weather_df = fetch_weather_data("2024-01-01", "2024-12-31")

    elapsed_time = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed_time}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
