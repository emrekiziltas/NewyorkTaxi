"""
Database connection and query execution module
"""
import duckdb
import pandas as pd
import streamlit as st
import traceback
from typing import Optional
from config import config


class DatabaseManager:
    """Manages DuckDB connections and query execution"""

    def __init__(self):
        self._connection = None

    @property
    def connection(self):
        """Get or create database connection"""
        if self._connection is None:
            self._connection = self._create_connection()
        return self._connection

    @staticmethod
    @st.cache_resource
    def _create_connection():
        """Create cached DuckDB connection"""
        try:
            return duckdb.connect(':memory:')
        except Exception as e:
            st.error(f" Failed to create database connection: {e}")
            st.stop()

    def execute_query(
            self,
            query: str,
            error_message: str = "Query execution failed"
    ) -> Optional[pd.DataFrame]:
        """
        Execute a DuckDB query with comprehensive error handling

        Args:
            query: SQL query string
            error_message: Custom error message to display

        Returns:
            DataFrame if successful, None if failed
        """
        try:
            result = self.connection.execute(query).fetchdf()

            if result is None:
                st.warning(f"️ {error_message}: Query returned no results")
                return None

            if result.empty:
                st.info("ℹ No data found matching the current filters")
                return None

            return result

        except duckdb.CatalogException as e:
            st.error(f" Database error: Column or table not found\n{str(e)}")
            return None

        except duckdb.ParserException as e:
            st.error(f" Query syntax error: {str(e)}")
            return None

        except duckdb.BinderException as e:
            st.error(f" Query binding error: {str(e)}")
            return None

        except MemoryError:
            st.error(
                f" {error_message}: Out of memory. "
                "Try reducing the date range or adding more filters."
            )
            return None

        except Exception as e:
            st.error(f" {error_message}: {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
            return None

    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None


# Global database manager instance
db = DatabaseManager()