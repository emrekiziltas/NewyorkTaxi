"""
Data validation module for NYC Taxi Pipeline
"""

from typing import Dict, Any

import pandas as pd

from config.settings import Config
from utils.logger import logger
from utils.exceptions import ValidationError


class DataValidator:
    """Performs comprehensive data quality checks."""

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary containing validation report

        Raises:
            ValidationError: If validation encounters critical errors
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

        # Check datetime columns
        self._validate_datetime_columns(df, report)

        # Check for high missing values
        self._check_missing_values(df, report)

        # Check numeric anomalies
        self._check_numeric_anomalies(df, report)

        logger.info(f"Data quality check completed: {len(report['warnings'])} warnings")
        return report

    def _validate_datetime_columns(self, df: pd.DataFrame, report: Dict[str, Any]):
        """Validate datetime columns."""
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

    def _check_missing_values(self, df: pd.DataFrame, report: Dict[str, Any]):
        """Check for columns with high percentage of missing values."""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > Config.HIGH_MISSING_VALUE_THRESHOLD].to_dict()

        if high_missing:
            report["warnings"].append(
                f"High missing values (>{Config.HIGH_MISSING_VALUE_THRESHOLD}%): {high_missing}"
            )

    def _check_numeric_anomalies(self, df: pd.DataFrame, report: Dict[str, Any]):
        """Check for negative values in columns that should be positive."""
        numeric_columns = ['trip_distance', 'fare_amount', 'total_amount', 'tip_amount']

        for col in numeric_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    report["warnings"].append(
                        f"{col}: {negative_count:,} negative values found"
                    )