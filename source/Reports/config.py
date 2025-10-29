"""
Configuration module for NYC Taxi Analytics Dashboard
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AppConfig:
    """Application configuration settings"""

    # Page configuration
    PAGE_TITLE: str = "NYC Taxi Analytics Dashboard"
    PAGE_ICON: str = "🚕"
    LAYOUT: str = "wide"

    # Data configuration
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data/processed")).resolve()

    # Query limits
    DEFAULT_EXPORT_LIMIT: int = 100000
    MAX_EXPORT_LIMIT: int = 1000000
    MIN_EXPORT_LIMIT: int = 100
    SAMPLE_LIMIT: int = 100
    CORRELATION_SAMPLE_LIMIT: int = 10000

    # Memory thresholds (in MB)
    MEMORY_WARNING_THRESHOLD: int = 50
    MEMORY_DANGER_THRESHOLD: int = 200

    # Default filter values
    DEFAULT_HOUR_RANGE: tuple = (0, 23)
    DEFAULT_FARE_RANGE: tuple = (0.0, 500.0)
    DEFAULT_DISTANCE_RANGE: tuple = (0.0, 50.0)
    DEFAULT_PASSENGER_RANGE: tuple = (1, 6)

    # Weekday mapping
    WEEKDAY_MAP: Dict[str, int] = field(default_factory=lambda: {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    })

    WEEKDAY_NAMES: List[str] = field(default_factory=lambda: [
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    # Month names
    MONTH_NAMES: List[str] = field(default_factory=lambda: [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])

    # Custom CSS
    CUSTOM_CSS: str = """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1f77b4;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
            }
            .report-container {
                background-color: #E8F1FA;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                margin-bottom: 25px;
            }
            .report-title {
                color: #0A3D62;
                font-weight: 600;
            }
        </style>
    """


# Global config instance
config = AppConfig()
