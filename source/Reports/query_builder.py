"""
SQL query builder module
"""
from typing import List, Dict, Any


class QueryBuilder:
    """Builds SQL queries for taxi data analysis"""

    def __init__(self, file_paths: List[str]):
        """
        Initialize query builder with data source

        Args:
            file_paths: List of parquet file paths
        """
        self.file_paths = file_paths
        self.from_clause = self._build_from_clause()

    def _build_from_clause(self) -> str:
        """Build the FROM clause for reading parquet files"""
        file_list = "', '".join(self.file_paths)
        return f"read_parquet(['{file_list}'])"

    def build_where_clause(self, filters: Dict[str, Any]) -> str:
        """
        Build WHERE clause from filters

        Args:
            filters: Dictionary of filter conditions containing:
                - hour_range: tuple of (min_hour, max_hour)
                - fare_range: tuple of (min_fare, max_fare)
                - distance_range: tuple of (min_distance, max_distance)
                - passenger_range: tuple of (min_passengers, max_passengers)
                - weekdays: list of weekday names (optional)
                - weekday_map: dict mapping weekday names to numbers (optional)
                - tip_percentage: minimum tip percentage (optional)

        Returns:
            WHERE clause string
        """
        # Default weekday mapping
        default_weekday_map = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6, 'Sunday': 0
        }

        # Ensure numeric values for hour_range (convert from any type)
        hour_min = int(filters['hour_range'][0])
        hour_max = int(filters['hour_range'][1])

        conditions = [
            f"EXTRACT(HOUR FROM pickup_hour) BETWEEN {hour_min} AND {hour_max}",
            f"CAST(total_amount AS DOUBLE) BETWEEN {filters['fare_range'][0]} AND {filters['fare_range'][1]}",
            f"CAST(trip_distance AS DOUBLE) BETWEEN {filters['distance_range'][0]} AND {filters['distance_range'][1]}",
            f"passenger_count BETWEEN {filters['passenger_range'][0]} AND {filters['passenger_range'][1]}"
        ]

        # Only add weekday filter if weekdays are provided and not empty
        weekdays = filters.get('weekdays', [])
        if weekdays and len(weekdays) > 0:
            # Use provided weekday_map or default
            weekday_map = filters.get('weekday_map', default_weekday_map)
            weekday_nums = [weekday_map[day] for day in weekdays if day in weekday_map]
            if weekday_nums:
                conditions.append(f"pickup_weekday IN ({','.join(map(str, weekday_nums))})")

        if filters.get('tip_percentage', 0) > 0:
            tip_pct = filters['tip_percentage']
            conditions.append(
                f"(tip_amount / NULLIF(fare_amount, 0) * 100) >= {tip_pct}"
            )

        return " AND ".join(conditions)

    def get_kpi_query(self, where_clause: str) -> str:
        """Build KPI summary query"""
        return f"""
        SELECT
            COUNT(*) as total_trips,
            ROUND(AVG(total_amount), 2) as avg_fare,
            ROUND(SUM(total_amount), 2) as total_revenue,
            ROUND(AVG(trip_distance), 2) as avg_distance,
            ROUND(AVG(tip_amount), 2) as avg_tip,
            ROUND(AVG(passenger_count), 1) as avg_passengers,
            ROUND(AVG(CASE WHEN fare_amount > 0 
                THEN tip_amount / fare_amount * 100 ELSE 0 END), 1) as avg_tip_percentage
        FROM {self.from_clause}
        WHERE {where_clause}
        """

    def get_hourly_query(self, where_clause: str) -> str:
        """Build hourly analysis query"""
        return f"""
        SELECT
            EXTRACT(HOUR FROM pickup_hour) as Hour,
            COUNT(*) as Trips,
            ROUND(AVG(total_amount), 2) as Avg_Fare
        FROM {self.from_clause}
        WHERE {where_clause}
        GROUP BY pickup_hour
        ORDER BY Hour
        """

    def get_weekday_query(self, where_clause: str) -> str:
        """Build weekday analysis query"""
        return f"""
        SELECT
            pickup_weekday,
            COUNT(*) as trips,
            ROUND(AVG(total_amount), 2) as avg_fare,
            ROUND(AVG(trip_distance), 2) as avg_distance
        FROM {self.from_clause}
        WHERE {where_clause}
        GROUP BY pickup_weekday
        ORDER BY pickup_weekday
        """

    def get_fare_distribution_query(self, where_clause: str) -> str:
        """Build fare distribution query"""
        return f"""
        SELECT
            CAST(total_amount / 5 AS INTEGER) * 5 as fare_bucket,
            COUNT(*) as trip_count
        FROM {self.from_clause}
        WHERE {where_clause} AND total_amount BETWEEN 0 AND 100
        GROUP BY fare_bucket
        ORDER BY fare_bucket
        """

    def get_tip_analysis_query(self, where_clause: str) -> str:
        """Build tip analysis query"""
        return f"""
        SELECT
            ROUND((tip_amount / NULLIF(fare_amount, 0) * 100) / 5) * 5 as tip_percent_bucket,
            COUNT(*) as count
        FROM {self.from_clause}
        WHERE {where_clause} 
            AND fare_amount > 0 
            AND tip_amount >= 0
            AND (tip_amount / fare_amount * 100) BETWEEN 0 AND 50
        GROUP BY tip_percent_bucket
        ORDER BY tip_percent_bucket
        """

    def get_revenue_trends_query(self, where_clause: str) -> str:
        """Build revenue trends query"""
        return f"""
        SELECT
            pickup_month as month,
            pickup_day as day,
            ROUND(SUM(total_amount), 2) as daily_revenue,
            COUNT(*) as daily_trips
        FROM {self.from_clause}
        WHERE {where_clause}
        GROUP BY pickup_month, pickup_day
        ORDER BY pickup_month, pickup_day
        """

    def get_distance_distribution_query(self, where_clause: str) -> str:
        """Build distance distribution query"""
        return f"""
        SELECT
            CAST(trip_distance as INTEGER) as distance_bucket,
            COUNT(*) as count
        FROM {self.from_clause}
        WHERE {where_clause} AND trip_distance BETWEEN 0 AND 30
        GROUP BY distance_bucket
        ORDER BY distance_bucket
        LIMIT 30
        """

    def get_passenger_query(self, where_clause: str) -> str:
        """Build passenger count query"""
        return f"""
        SELECT
            passenger_count,
            COUNT(*) as trips,
            ROUND(AVG(total_amount), 2) as avg_fare
        FROM {self.from_clause}
        WHERE {where_clause} AND passenger_count BETWEEN 1 AND 6
        GROUP BY passenger_count
        ORDER BY passenger_count
        """

    def get_statistics_query(self, where_clause: str) -> str:
        """Build statistical summary query"""
        return f"""
        SELECT
            'Fare Amount' as metric,
            ROUND(MIN(total_amount), 2) as min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_amount), 2) as q1,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount), 2) as median,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_amount), 2) as q3,
            ROUND(MAX(total_amount), 2) as max,
            ROUND(AVG(total_amount), 2) as mean,
            ROUND(STDDEV(total_amount), 2) as std_dev
        FROM {self.from_clause}
        WHERE {where_clause} AND total_amount BETWEEN 0 AND 500

        UNION ALL

        SELECT
            'Trip Distance' as metric,
            ROUND(MIN(trip_distance), 2),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trip_distance), 2),
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trip_distance), 2),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trip_distance), 2),
            ROUND(MAX(trip_distance), 2),
            ROUND(AVG(trip_distance), 2),
            ROUND(STDDEV(trip_distance), 2)
        FROM {self.from_clause}
        WHERE {where_clause} AND trip_distance BETWEEN 0 AND 100

        UNION ALL

        SELECT
            'Tip Amount' as metric,
            ROUND(MIN(tip_amount), 2),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY tip_amount), 2),
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tip_amount), 2),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tip_amount), 2),
            ROUND(MAX(tip_amount), 2),
            ROUND(AVG(tip_amount), 2),
            ROUND(STDDEV(tip_amount), 2)
        FROM {self.from_clause}
        WHERE {where_clause} AND tip_amount >= 0
        """

    def get_correlation_query(self, where_clause: str, limit: int = 10000) -> str:
        """Build correlation analysis query"""
        return f"""
        SELECT
            trip_distance,
            total_amount,
            tip_amount,
            passenger_count
        FROM {self.from_clause}
        WHERE {where_clause}
            AND trip_distance BETWEEN 0 AND 50
            AND total_amount BETWEEN 0 AND 200
        LIMIT {limit}
        """

    def get_airport_query(self, where_clause: str) -> str:
        """Build airport analysis query"""
        return f"""
        SELECT
            strftime(tpep_pickup_datetime, '%A') AS day_of_week,
            COUNT(*) AS total_trips,
            SUM(CASE WHEN airport_fee > 0 THEN 1 ELSE 0 END) AS airport_trips,
            ROUND(SUM(CASE WHEN airport_fee > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS airport_trip_pct
        FROM {self.from_clause}
        WHERE {where_clause}
        GROUP BY day_of_week
        ORDER BY
            CASE day_of_week
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
            END
        """

    def get_cost_tip_query(self, where_clause: str) -> str:
        """Build cost per mile and tip percentage query"""
        return f"""
        SELECT
            is_weekend,
            time_period,
            ROUND(AVG(cost_per_mile), 2) AS avg_cost_per_mile,
            ROUND(AVG(tip_percentage), 2) AS avg_tip_percentage
        FROM {self.from_clause}
        WHERE {where_clause}
        GROUP BY is_weekend, time_period
        ORDER BY is_weekend, time_period
        """

    def get_export_query(self, where_clause: str, limit: int) -> str:
        """Build export query"""
        return f"""
        SELECT *
        FROM {self.from_clause}
        WHERE {where_clause}
        LIMIT {limit}
        """

    def get_sample_query(self, where_clause: str, limit: int = 100) -> str:
        """Build sample data query"""
        return f"""
        SELECT 
            EXTRACT(HOUR FROM pickup_hour),
            pickup_weekday,
            trip_distance,
            total_amount,
            tip_amount,
            passenger_count,
            fare_amount
        FROM {self.from_clause}
        WHERE {where_clause}
        LIMIT {limit}
        """

    def get_data_summary_query(self) -> str:
        """Build data summary query for loading statistics"""
        return f"""
        SELECT 
            COUNT(*) as total_records,
            MIN(tpep_pickup_datetime) as min_date,
            MAX(tpep_pickup_datetime) as max_date
        FROM {self.from_clause}
        """

    def get_count_query(self, where_clause: str) -> str:
        """Build count query for export estimation"""
        return f"""
        SELECT COUNT(*) as total_count
        FROM {self.from_clause}
        WHERE {where_clause}
        """

    def get_top_revenue_days_query(self, where_clause: str, limit: int = 10) -> str:
        """Build top revenue days query"""
        return f"""
        SELECT
            pickup_month || '-' || pickup_day as date,
            COUNT(*) as trips,
            ROUND(SUM(total_amount), 2) as revenue,
            ROUND(AVG(total_amount), 2) as avg_fare
        FROM {self.from_clause}
        WHERE {where_clause}
        GROUP BY pickup_month, pickup_day
        ORDER BY revenue DESC
        LIMIT {limit}
        """
