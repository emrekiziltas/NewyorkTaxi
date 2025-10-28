"""
Data transformation module for NYC Taxi Pipeline
"""

from typing import Optional

import pandas as pd

from ..config.settings import Config
from ..utils.logger import logger
from ..utils.exceptions import TransformationError


class DataTransformer:
    """Handles data transformations and feature engineering."""

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations and feature engineering.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            TransformationError: If transformation fails
        """
        if df is None or df.empty:
            logger.warning("Cannot transform empty DataFrame")
            return df

        logger.info("Starting data transformations...")
        df_transformed = df.copy()
        initial_rows = len(df_transformed)

        try:
            # Convert datetime columns
            df_transformed = self._convert_datetime_columns(df_transformed)

            # Remove invalid numeric values
            df_transformed = self._filter_invalid_values(df_transformed)

            # Extract datetime features
            df_transformed = self._extract_datetime_features(df_transformed)

            # Create derived metrics
            df_transformed = self._create_derived_metrics(df_transformed)

            final_rows = len(df_transformed)
            removed_pct = ((initial_rows - final_rows) / initial_rows * 100)
            logger.info(
                f"Transformation completed: {final_rows:,} records ({removed_pct:.2f}% filtered)"
            )

            return df_transformed

        except Exception as e:
            raise TransformationError(f"Transformation failed: {e}")

    def _convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to proper datetime type."""
        date_columns = [c for c in df.columns if "datetime" in c.lower()]

        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info(f"Converting {col} to datetime")
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def _filter_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid numeric values."""
        validations = {
            "trip_distance": (Config.MIN_TRIP_DISTANCE, Config.MAX_TRIP_DISTANCE),
            "fare_amount": (Config.MIN_FARE_AMOUNT, Config.MAX_FARE_AMOUNT),
            "total_amount": (Config.MIN_FARE_AMOUNT, Config.MAX_FARE_AMOUNT),
            "tip_amount": (0, Config.MAX_TIP_AMOUNT),
            "tolls_amount": (0, Config.MAX_TOLLS_AMOUNT)
        }

        for col, (min_val, max_val) in validations.items():
            if col in df.columns:
                before = len(df)
                df = df[
                    (df[col] >= min_val) & (df[col] <= max_val)
                    ]
                removed = before - len(df)
                if removed > 0:
                    logger.info(f"Removed {removed:,} invalid {col} records")

        return df

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime features from pickup time."""
        pickup_col = self._get_pickup_column(df)

        if pickup_col and pd.api.types.is_datetime64_any_dtype(df[pickup_col]):
            df['pickup_hour'] = df[pickup_col].dt.hour
            df['pickup_day'] = df[pickup_col].dt.day
            df['pickup_weekday'] = df[pickup_col].dt.dayofweek
            df['pickup_month'] = df[pickup_col].dt.month
            df['pickup_year'] = df[pickup_col].dt.year
            df['is_weekend'] = df['pickup_weekday'].isin([5, 6])

            # Time period categorization
            df['time_period'] = pd.cut(
                df['pickup_hour'],
                bins=Config.TIME_PERIOD_BINS,
                labels=Config.TIME_PERIOD_LABELS,
                include_lowest=True
            )

            logger.info("Extracted datetime features")

        return df

    def _create_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived metrics."""
        # Cost per mile
        if 'trip_distance' in df.columns and 'total_amount' in df.columns:
            df['cost_per_mile'] = (
                    df['total_amount'] / df['trip_distance'].replace(0, pd.NA)
            )

        # Tip percentage
        if 'tip_amount' in df.columns and 'fare_amount' in df.columns:
            df['tip_percentage'] = (
                    df['tip_amount'] / df['fare_amount'].replace(0, pd.NA) * 100
            )

        logger.info("Created derived metrics")
        return df

    def _get_pickup_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the pickup datetime column name."""
        for col in ['tpep_pickup_datetime', 'lpep_pickup_datetime', 'pickup_datetime']:
            if col in df.columns:
                return col
        return None