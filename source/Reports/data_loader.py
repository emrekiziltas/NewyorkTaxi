"""
Data loading and file management module
"""
import re
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from config import config
from database import db


class DataLoader:
    """Handles data file discovery and loading"""

    @staticmethod
    @st.cache_resource
    def get_parquet_files_by_year() -> Dict[str, List[Path]]:
        """
        Find Parquet files in processed folder and group by year

        Returns:
            Dictionary mapping years to list of file paths
        """
        try:
            st.set_page_config(f" emre: {config.DATA_DIR}")

            if not config.DATA_DIR.exists():
                return {}

            parquet_files = list(config.DATA_DIR.glob("*_processed.parquet"))
            files_by_year = {}

            for file in parquet_files:
                match = re.search(r'_(\d{4})_', file.name)
                if match:
                    year = match.group(1)
                    if year not in files_by_year:
                        files_by_year[year] = []
                    files_by_year[year].append(file)

            # Sort files within each year
            for year in files_by_year:
                files_by_year[year] = sorted(files_by_year[year])

            return files_by_year

        except Exception as e:
            st.error(f" Error scanning data directory: {e}")
            return {}

    @staticmethod
    def load_data_summary(file_paths: List[str]) -> Optional[pd.DataFrame]:
        """
        Load summary statistics from selected files

        Args:
            file_paths: List of file paths to load

        Returns:
            DataFrame with summary statistics
        """
        try:
            file_list = "', '".join(file_paths)
            query = f"""
            SELECT 
                COUNT(*) as total_records,
                MIN(tpep_pickup_datetime) as min_date,
                MAX(tpep_pickup_datetime) as max_date
            FROM read_parquet(['{file_list}'])
            """
            return db.execute_query(query, "Failed to load data summary")

        except Exception as e:
            st.error(f" Error loading data summary: {e}")
            return None

    @staticmethod
    def filter_files_by_month_range(
            files: List[Path],
            start_month: int,
            end_month: int
    ) -> List[str]:
        """
        Filter files by month range

        Args:
            files: List of file paths
            start_month: Start month (1-12)
            end_month: End month (1-12)

        Returns:
            List of filtered file names
        """
        filtered = []
        for file in files:
            match = re.search(r'_(\d{2})_', file.name)
            if match:
                month = int(match.group(1))
                if start_month <= month <= end_month:
                    filtered.append(file.name)
        return filtered


class MemoryEstimator:
    """Estimates and warns about memory usage"""

    @staticmethod
    def estimate_size(
            row_count: int,
            column_count: int = 20
    ) -> tuple[float, str]:
        """
        Estimate the memory size of a DataFrame

        Args:
            row_count: Number of rows
            column_count: Number of columns

        Returns:
            Tuple of (size_in_mb, warning_level)
        """
        # Rough estimation: ~100 bytes per cell on average
        estimated_bytes = row_count * column_count * 100
        estimated_mb = estimated_bytes / (1024 * 1024)

        if estimated_mb < config.MEMORY_WARNING_THRESHOLD:
            return estimated_mb, "safe"
        elif estimated_mb < config.MEMORY_DANGER_THRESHOLD:
            return estimated_mb, "warning"
        else:
            return estimated_mb, "danger"

    @staticmethod
    def show_warning(row_count: int, column_count: int = 20):
        """Display memory warning if dataset is large"""
        size_mb, level = MemoryEstimator.estimate_size(row_count, column_count)

        if level == "warning":
            st.warning(
                f"️ Large dataset detected (~{size_mb:.1f} MB, {row_count:,} rows). "
                "Export may take some time."
            )
        elif level == "danger":
            st.error(
                f" Very large dataset (~{size_mb:.1f} MB, {row_count:,} rows). "
                "Consider adding filters to reduce data size."
            )
            st.info(
                " **Tip:** Use date range filters or increase minimum "
                "fare/distance to reduce the dataset size."
            )


# Global instances
data_loader = DataLoader()
memory_estimator = MemoryEstimator()