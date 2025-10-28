"""
Logging configuration for NYC Taxi Pipeline
"""

import sys
import logging
from pathlib import Path


def setup_logger(name: str = __name__, log_file: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

    return logger


# Create default logger
logger = setup_logger("nyc_taxi_pipeline")