"""
Custom exceptions for NYC Taxi Pipeline
"""


class DataIngestionError(Exception):
    """Base exception for data ingestion errors."""
    pass


class DownloadError(DataIngestionError):
    """Exception raised when download fails."""
    pass


class ValidationError(DataIngestionError):
    """Exception raised when data validation fails."""
    pass


class TransformationError(DataIngestionError):
    """Exception raised when data transformation fails."""
    pass


class StorageError(DataIngestionError):
    """Exception raised when data storage fails."""
    pass


class EnrichmentError(DataIngestionError):
    """Exception raised when data enrichment fails."""
    pass