"""
HTTP client with retry logic for NYC Taxi Pipeline
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config.settings import Config


class HTTPClient:
    """HTTP client with automatic retry logic"""

    def __init__(
            self,
            total_retries: int = Config.RETRY_TOTAL,
            backoff_factor: int = Config.RETRY_BACKOFF_FACTOR,
            status_forcelist: list = None
    ):
        """
        Initialize HTTP client with retry strategy.

        Args:
            total_retries: Total number of retries
            backoff_factor: Backoff factor for retries
            status_forcelist: List of HTTP status codes to retry
        """
        self.session = self._create_session(
            total_retries,
            backoff_factor,
            status_forcelist or Config.RETRY_STATUS_FORCELIST
        )

    def _create_session(
            self,
            total_retries: int,
            backoff_factor: int,
            status_forcelist: list
    ) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Perform GET request with retry logic.

        Args:
            url: URL to request
            **kwargs: Additional arguments for requests.get()

        Returns:
            Response object
        """
        return self.session.get(url, **kwargs)

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()