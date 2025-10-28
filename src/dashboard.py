import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path
import re
import plotly.express as px
import plotly.graph_objects as go
import os
import traceback
from typing import Optional, Tuple

st.set_page_config(
    page_title="NYC Taxi Analytics Dashboard",
    layout="wide",
    page_icon="🚕",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Try to read from environment variable, otherwise default to local folder
DATA_DIR = Path(os.getenv("DATA_DIR", "src/data/processed")).resolve()

# Create directory if it doesn't exist (useful when first running pipeline)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- DuckDB Connection (Must be defined early) ---
@st.cache_resource
def get_db_connection():
    """Cache DuckDB connection"""
    try:
        return duckdb.connect(':memory:')
    except Exception as e:
        st.error(f"❌ Failed to create database connection: {e}")
        st.stop()


# --- Error Handling Utility Functions ---
def safe_query_execution(conn, query: str, error_message: str = "Query execution failed") -> Optional[pd.DataFrame]:
    """
    Safely execute a DuckDB query with comprehensive error handling.

    Args:
        conn: DuckDB connection
        query: SQL query string
        error_message: Custom error message to display

    Returns:
        DataFrame if successful, None if failed
    """
    try:
        result = conn.execute(query).fetchdf()

        if result is None:
            st.warning(f"⚠️ {error_message}: Query returned no results")
            return None

        if result.empty:
            st.info(f"ℹ️ No data found matching the current filters")
            return None

        return result

    except duckdb.CatalogException as e:
        st.error(f"❌ Database error: Column or table not found\n{str(e)}")
        return None
    except duckdb.ParserException as e:
        st.error(f"❌ Query syntax error: {str(e)}")
        return None
    except duckdb.BinderException as e:
        st.error(f"❌ Query binding error: {str(e)}")
        return None
    except MemoryError:
        st.error(f"❌ {error_message}: Out of memory. Try reducing the date range or adding more filters.")
        return None
    except Exception as e:
        st.error(f"❌ {error_message}: {str(e)}")
        with st.expander("🔍 View Error Details"):
            st.code(traceback.format_exc())
        return None


def estimate_dataframe_size(row_count: int, column_count: int = 20) -> Tuple[float, str]:
    """
    Estimate the memory size of a DataFrame.

    Args:
        row_count: Number of rows
        column_count: Number of columns (default: 20)

    Returns:
        Tuple of (size_in_mb, warning_level)
    """
    # Rough estimation: ~100 bytes per cell on average
    estimated_bytes = row_count * column_count * 100
    estimated_mb = estimated_bytes / (1024 * 1024)

    if estimated_mb < 50:
        return estimated_mb, "safe"
    elif estimated_mb < 200:
        return estimated_mb, "warning"
    else:
        return estimated_mb, "danger"


def show_memory_warning(row_count: int, column_count: int = 20):
    """Display memory warning if dataset is large."""
    size_mb, level = estimate_dataframe_size(row_count, column_count)

    if level == "warning":
        st.warning(f"⚠️ Large dataset detected (~{size_mb:.1f} MB, {row_count:,} rows). Export may take some time.")
    elif level == "danger":
        st.error(
            f"🚨 Very large dataset (~{size_mb:.1f} MB, {row_count:,} rows). Consider adding filters to reduce data size.")
        st.info("💡 **Tip:** Use date range filters or increase minimum fare/distance to reduce the dataset size.")


# --- Find Data Files and Group by Year ---
@st.cache_resource
def get_parquet_files_by_year():
    """Find Parquet files in processed folder and group by year"""
    try:
        if not DATA_DIR.exists():
            return {}

        parquet_files = list(DATA_DIR.glob("*_processed.parquet"))

        files_by_year = {}
        for file in parquet_files:
            match = re.search(r'_(\d{4})_', file.name)
            if match:
                year = match.group(1)
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(file)

        for year in files_by_year:
            files_by_year[year] = sorted(files_by_year[year])

        return files_by_year

    except Exception as e:
        st.error(f"❌ Error scanning data directory: {e}")
        return {}


def load_data_summary(conn, file_paths):
    """Load summary statistics from selected files"""
    try:
        file_list = "', '".join(file_paths)
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            MIN(tpep_pickup_datetime) as min_date,
            MAX(tpep_pickup_datetime) as max_date
        FROM read_parquet(['{file_list}'])
        """
        return safe_query_execution(conn, query, "Failed to load data summary")
    except Exception as e:
        st.error(f"❌ Error loading data summary: {e}")
        return None


files_by_year = get_parquet_files_by_year()

# --- File Check ---
if not files_by_year:
    st.error(f"❌ No data files found in: {DATA_DIR}")
    st.warning("""
    **Please run the data pipeline first:**
    ```bash
    python process_taxi_data.py
    ```
    This command will create Parquet files in the `data/processed/` folder.
    """)
    st.stop()

# --- Sidebar - Year and File Selection ---
st.sidebar.header("📁 Data Source")

# Year selection
available_years = sorted(files_by_year.keys(), reverse=True)
selected_year = st.sidebar.selectbox(
    "📅 Select Year:",
    options=available_years,
    index=0
)

# Show files for selected year
files_for_year = files_by_year[selected_year]

# Month selection options
selection_mode = st.sidebar.radio(
    "Selection Mode:",
    ["All Months", "Specific Months", "Date Range"],
    index=0
)

if selection_mode == "All Months":
    selected_files = [f.name for f in files_for_year]
elif selection_mode == "Specific Months":
    selected_files = st.sidebar.multiselect(
        f"📄 Select {selected_year} Files:",
        options=[f.name for f in files_for_year],
        default=[f.name for f in files_for_year[:3]]
    )
else:  # Date Range
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    start_month = st.sidebar.selectbox("Start Month:", months, index=0)
    end_month = st.sidebar.selectbox("End Month:", months, index=11)

    start_idx = months.index(start_month) + 1
    end_idx = months.index(end_month) + 1

    selected_files = [f.name for f in files_for_year
                      if start_idx <= int(re.search(r'_(\d{2})_', f.name).group(1)) <= end_idx]

if not selected_files:
    st.warning("⚠️ Please select at least one file.")
    st.stop()

selected_paths = [str(DATA_DIR / f) for f in selected_files]

# Initialize DuckDB connection (needed for data_summary)
conn = get_db_connection()

# Load data summary
data_summary = load_data_summary(conn, selected_paths)
if data_summary is not None and not data_summary.empty:
    total_records = data_summary['total_records'].iloc[0]
    st.sidebar.success(f"""
    📊 **Data Loaded:**
    - Files: {len(selected_files)}
    - Records: {total_records:,}
    - Date Range: {data_summary['min_date'].iloc[0]} to {data_summary['max_date'].iloc[0]}
    """)

    # Show memory estimation
    size_mb, level = estimate_dataframe_size(total_records)
    if level == "warning":
        st.sidebar.warning(f"⚠️ Dataset size: ~{size_mb:.1f} MB")
    elif level == "danger":
        st.sidebar.error(f"🚨 Large dataset: ~{size_mb:.1f} MB")
else:
    st.sidebar.info(f"📊 {len(selected_files)} files selected from **{selected_year}**")

# --- Advanced Filters ---
st.sidebar.header("🔍 Filters")

with st.sidebar.expander("⏰ Time Filters", expanded=True):
    hour_range = st.slider(
        "Hour of Day:",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

    weekday_options = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    selected_weekdays = st.multiselect(
        "Days of Week:",
        options=list(weekday_options.keys()),
        default=list(weekday_options.keys())
    )

with st.sidebar.expander("💰 Financial Filters", expanded=True):
    fare_range = st.slider(
        "Fare Range ($):",
        min_value=0.0,
        max_value=500.0,
        value=(0.0, 500.0),
        step=5.0
    )

    tip_percentage = st.slider(
        "Minimum Tip % (of fare):",
        min_value=0,
        max_value=50,
        value=0,
        step=5
    )

with st.sidebar.expander("🚗 Trip Filters", expanded=True):
    distance_range = st.slider(
        "Trip Distance (miles):",
        min_value=0.0,
        max_value=50.0,
        value=(0.0, 50.0),
        step=1.0
    )

    passenger_count = st.slider(
        "Passenger Count:",
        min_value=1,
        max_value=6,
        value=(1, 6)
    )

# Export/Download option
st.sidebar.header("💾 Export Data")
export_format = st.sidebar.selectbox(
    "Export Format:",
    ["CSV", "Excel", "Parquet"]
)

# Export limit with warning
export_limit = st.sidebar.number_input(
    "Export Row Limit:",
    min_value=100,
    max_value=1000000,
    value=100000,
    step=10000,
    help="Limit the number of rows to export to prevent memory issues"
)


# --- Query Preparation ---
def build_query_conditions():
    """Build filter conditions"""
    conditions = [
        f"pickup_hour BETWEEN {hour_range[0]} AND {hour_range[1]}",
        f"total_amount BETWEEN {fare_range[0]} AND {fare_range[1]}",
        f"trip_distance BETWEEN {distance_range[0]} AND {distance_range[1]}",
        f"passenger_count BETWEEN {passenger_count[0]} AND {passenger_count[1]}"
    ]

    if selected_weekdays:
        weekday_nums = [weekday_options[day] for day in selected_weekdays]
        conditions.append(f"pickup_weekday IN ({','.join(map(str, weekday_nums))})")

    if tip_percentage > 0:
        conditions.append(f"(tip_amount / NULLIF(fare_amount, 0) * 100) >= {tip_percentage}")

    return " AND ".join(conditions)


where_clause = build_query_conditions()
file_list = "', '".join(selected_paths)
from_clause = f"read_parquet(['{file_list}'])"

# --- Main Dashboard ---
st.markdown(f'<h1 class="main-header">🚕 NYC Taxi Analytics - {selected_year}</h1>', unsafe_allow_html=True)

# --- KPIs ---
kpi_query = f"""
SELECT
    COUNT(*) as total_trips,
    ROUND(AVG(total_amount), 2) as avg_fare,
    ROUND(SUM(total_amount), 2) as total_revenue,
    ROUND(AVG(trip_distance), 2) as avg_distance,
    ROUND(AVG(tip_amount), 2) as avg_tip,
    ROUND(AVG(passenger_count), 1) as avg_passengers,
    ROUND(AVG(CASE WHEN fare_amount > 0 THEN tip_amount / fare_amount * 100 ELSE 0 END), 1) as avg_tip_percentage
FROM {from_clause}
WHERE {where_clause}
"""

kpi_result = safe_query_execution(conn, kpi_query, "Failed to calculate KPIs")

if kpi_result is not None and not kpi_result.empty and kpi_result['total_trips'].iloc[0] > 0:
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    with col1:
        st.metric("🚕 Total Trips", f"{kpi_result['total_trips'].iloc[0]:,}")
    with col2:
        st.metric("💵 Avg Fare", f"${kpi_result['avg_fare'].iloc[0]:.2f}")
    with col3:
        st.metric("💰 Total Revenue", f"${kpi_result['total_revenue'].iloc[0]:,.0f}")
    with col4:
        st.metric("📏 Avg Distance", f"{kpi_result['avg_distance'].iloc[0]:.2f} mi")
    with col5:
        st.metric("🎁 Avg Tip", f"${kpi_result['avg_tip'].iloc[0]:.2f}")
    with col6:
        st.metric("📊 Avg Tip %", f"{kpi_result['avg_tip_percentage'].iloc[0]:.1f}%")
    with col7:
        st.metric("👥 Avg Passengers", f"{kpi_result['avg_passengers'].iloc[0]:.1f}")
    with col8:
        revenue_per_trip = kpi_result['total_revenue'].iloc[0] / kpi_result['total_trips'].iloc[0]
        st.metric("💲 Revenue/Trip", f"${revenue_per_trip:.2f}")
else:
    st.warning("⚠️ No data found for these filters. Please adjust your filter settings.")
    st.stop()

st.divider()

# --- Advanced Analytics Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⏰ Temporal Analysis",
    "💰 Financial Insights",
    "🗺️ Trip Patterns",
    "📊 Statistical Analysis",
    "🔍 Deep Dive",
    "Airport"
])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trips by Hour")
        hourly_query = f"""
        SELECT
            pickup_hour as Hour,
            COUNT(*) as Trips,
            ROUND(AVG(total_amount), 2) as Avg_Fare
        FROM {from_clause}
        WHERE {where_clause}
        GROUP BY pickup_hour
        ORDER BY pickup_hour
        """
        hourly_df = safe_query_execution(conn, hourly_query, "Failed to load hourly data")

        if hourly_df is not None and not hourly_df.empty:
            fig = px.line(hourly_df, x='Hour', y='Trips',
                          title='Hourly Trip Volume',
                          markers=True)
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Hour")
        if hourly_df is not None and not hourly_df.empty:
            fig = px.bar(hourly_df, x='Hour', y='Avg_Fare',
                         title='Average Fare by Hour',
                         color='Avg_Fare',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # Day of week analysis
    st.subheader("Weekly Patterns")
    weekday_query = f"""
    SELECT
        pickup_weekday,
        COUNT(*) as trips,
        ROUND(AVG(total_amount), 2) as avg_fare,
        ROUND(AVG(trip_distance), 2) as avg_distance
    FROM {from_clause}
    WHERE {where_clause}
    GROUP BY pickup_weekday
    ORDER BY pickup_weekday
    """
    weekday_df = safe_query_execution(conn, weekday_query, "Failed to load weekly data")

    if weekday_df is not None and not weekday_df.empty:
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_df['Day'] = weekday_df['pickup_weekday'].map(
            lambda x: weekday_names[x] if x < len(weekday_names) else str(x)
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(x=weekday_df['Day'], y=weekday_df['trips'],
                             name='Trips', yaxis='y'))
        fig.add_trace(go.Scatter(x=weekday_df['Day'], y=weekday_df['avg_fare'],
                                 name='Avg Fare', yaxis='y2', mode='lines+markers'))

        fig.update_layout(
            title='Weekly Trip Volume and Average Fare',
            yaxis=dict(title='Number of Trips'),
            yaxis2=dict(title='Average Fare ($)', overlaying='y', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fare Distribution")
        fare_dist_query = f"""
        SELECT
            CAST(total_amount / 5 AS INTEGER) * 5 as fare_bucket,
            COUNT(*) as trip_count
        FROM {from_clause}
        WHERE {where_clause} AND total_amount BETWEEN 0 AND 100
        GROUP BY fare_bucket
        ORDER BY fare_bucket
        """
        fare_dist_df = safe_query_execution(conn, fare_dist_query, "Failed to load fare distribution")

        if fare_dist_df is not None and not fare_dist_df.empty:
            fig = px.histogram(fare_dist_df, x='fare_bucket', y='trip_count',
                               title='Fare Distribution',
                               labels={'fare_bucket': 'Fare ($)', 'trip_count': 'Trips'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tip Analysis")
        tip_query = f"""
        SELECT
            ROUND((tip_amount / NULLIF(fare_amount, 0) * 100) / 5) * 5 as tip_percent_bucket,
            COUNT(*) as count
        FROM {from_clause}
        WHERE {where_clause} 
            AND fare_amount > 0 
            AND tip_amount >= 0
            AND (tip_amount / fare_amount * 100) BETWEEN 0 AND 50
        GROUP BY tip_percent_bucket
        ORDER BY tip_percent_bucket
        """
        tip_df = safe_query_execution(conn, tip_query, "Failed to load tip analysis")

        if tip_df is not None and not tip_df.empty:
            fig = px.bar(tip_df, x='tip_percent_bucket', y='count',
                         title='Tip Percentage Distribution',
                         labels={'tip_percent_bucket': 'Tip %', 'count': 'Trips'})
            st.plotly_chart(fig, use_container_width=True)

    # Revenue trends
    st.subheader("Revenue Trends")
    revenue_query = f"""
    SELECT
        pickup_month as month,
        pickup_day as day,
        ROUND(SUM(total_amount), 2) as daily_revenue,
        COUNT(*) as daily_trips
    FROM {from_clause}
    WHERE {where_clause}
    GROUP BY pickup_month, pickup_day
    ORDER BY pickup_month, pickup_day
    """
    revenue_df = safe_query_execution(conn, revenue_query, "Failed to load revenue trends")

    if revenue_df is not None and not revenue_df.empty:
        revenue_df['date'] = revenue_df['month'].astype(str) + '-' + revenue_df['day'].astype(str)
        fig = px.line(revenue_df, x='date', y='daily_revenue',
                      title='Daily Revenue Trend',
                      labels={'daily_revenue': 'Revenue ($)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distance Distribution")
        distance_query = f"""
        SELECT
            CAST(trip_distance as INTEGER) as distance_bucket,
            COUNT(*) as count
        FROM {from_clause}
        WHERE {where_clause} AND trip_distance BETWEEN 0 AND 30
        GROUP BY distance_bucket
        ORDER BY distance_bucket
        LIMIT 30
        """
        distance_df = safe_query_execution(conn, distance_query, "Failed to load distance distribution")

        if distance_df is not None and not distance_df.empty:
            fig = px.area(distance_df, x='distance_bucket', y='count',
                          title='Trip Distance Distribution',
                          labels={'distance_bucket': 'Distance (miles)', 'count': 'Trips'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Passenger Count")
        passenger_query = f"""
        SELECT
            passenger_count,
            COUNT(*) as trips,
            ROUND(AVG(total_amount), 2) as avg_fare
        FROM {from_clause}
        WHERE {where_clause} AND passenger_count BETWEEN 1 AND 6
        GROUP BY passenger_count
        ORDER BY passenger_count
        """
        passenger_df = safe_query_execution(conn, passenger_query, "Failed to load passenger data")

        if passenger_df is not None and not passenger_df.empty:
            fig = px.bar(passenger_df, x='passenger_count', y='trips',
                         title='Trips by Passenger Count',
                         color='avg_fare',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📊 Statistical Summary")

    stats_query = f"""
    SELECT
        'Fare Amount' as metric,
        ROUND(MIN(total_amount), 2) as min,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_amount), 2) as q1,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount), 2) as median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_amount), 2) as q3,
        ROUND(MAX(total_amount), 2) as max,
        ROUND(AVG(total_amount), 2) as mean,
        ROUND(STDDEV(total_amount), 2) as std_dev
    FROM {from_clause}
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
    FROM {from_clause}
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
    FROM {from_clause}
    WHERE {where_clause} AND tip_amount >= 0
    """

    stats_df = safe_query_execution(conn, stats_query, "Failed to calculate statistics")

    if stats_df is not None and not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True)

        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        corr_query = f"""
        SELECT
            trip_distance,
            total_amount,
            tip_amount,
            passenger_count
        FROM {from_clause}
        WHERE {where_clause}
            AND trip_distance BETWEEN 0 AND 50
            AND total_amount BETWEEN 0 AND 200
        LIMIT 10000
        """
        corr_df = safe_query_execution(conn, corr_query, "Failed to calculate correlations")

        if corr_df is not None and not corr_df.empty:
            correlation_matrix = corr_df.corr()
            fig = px.imshow(correlation_matrix,
                            text_auto=True,
                            aspect='auto',
                            color_continuous_scale='RdBu_r',
                            title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🔍 Custom Query Builder")

    # Query templates
    query_template = st.selectbox(
        "Select Analysis Type:",
        [
            "Top Revenue Days",
            "Peak Hours by Day",
            "Long Distance Trips",
            "High Tip Trips",
            "Custom Query"
        ]
    )

    if query_template == "Top Revenue Days":
        top_days_query = f"""
        SELECT
            pickup_month || '-' || pickup_day as date,
            COUNT(*) as trips,
            ROUND(SUM(total_amount), 2) as revenue,
            ROUND(AVG(total_amount), 2) as avg_fare
        FROM {from_clause}
        WHERE {where_clause}
        GROUP BY pickup_month, pickup_day
        ORDER BY revenue DESC
        LIMIT 10
        """
        result_df = safe_query_execution(conn, top_days_query, "Failed to load top revenue days")

        if result_df is not None:
            st.dataframe(result_df, use_container_width=True)

    elif query_template == "Custom Query":
        custom_query = st.text_area(
            "Enter your DuckDB query:",
            f"SELECT * FROM {from_clause} WHERE {where_clause} LIMIT 100",
            height=150
        )

        if st.button("Execute Query"):
            result_df = safe_query_execution(conn, custom_query, "Custom query execution failed")

            if result_df is not None:
                st.dataframe(result_df, use_container_width=True)
                st.success(f"✅ Query executed successfully. Returned {len(result_df):,} rows.")

    # Data Export
    st.subheader("💾 Export Filtered Data")

    # Count records before export
    count_query = f"""
    SELECT COUNT(*) as total_count
    FROM {from_clause}
    WHERE {where_clause}
    """
    count_result = safe_query_execution(conn, count_query, "Failed to count records")

    if count_result is not None and not count_result.empty:
        total_count = count_result['total_count'].iloc[0]

        # Show memory warning
        show_memory_warning(min(total_count, export_limit))

        st.info(f"📊 Total matching records: {total_count:,} | Export limit: {export_limit:,}")

        if total_count > export_limit:
            st.warning(f"⚠️ Dataset exceeds export limit. Only the first {export_limit:,} records will be exported.")

    if st.button("Generate Export"):
        with st.spinner("Generating export file..."):
            export_query = f"""
            SELECT *
            FROM {from_clause}
            WHERE {where_clause}
            LIMIT {export_limit}
            """
            export_df = safe_query_execution(conn, export_query, "Failed to generate export")

            if export_df is not None and not export_df.empty:
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"nyc_taxi_{selected_year}_export.csv",
                        mime="text/csv"
                    )
                    st.success(f"✅ CSV file ready! ({len(export_df):,} rows)")

                elif export_format == "Parquet":
                    # Save to buffer for download
                    import io

                    buffer = io.BytesIO()
                    export_df.to_parquet(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="📥 Download Parquet",
                        data=buffer,
                        file_name=f"nyc_taxi_{selected_year}_export.parquet",
                        mime="application/octet-stream"
                    )
                    st.success(f"✅ Parquet file ready! ({len(export_df):,} rows)")

                elif export_format == "Excel":
                    try:
                        import io

                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            export_df.to_excel(writer, index=False, sheet_name='Taxi Data')
                        buffer.seek(0)

                        st.download_button(
                            label="📥 Download Excel",
                            data=buffer,
                            file_name=f"nyc_taxi_{selected_year}_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success(f"✅ Excel file ready! ({len(export_df):,} rows)")
                    except ImportError:
                        st.error("❌ Excel export requires openpyxl. Install with: `pip install openpyxl`")

# --- Sample Data Table ---
st.divider()
with st.expander("🔍 Filtered Raw Data (Sample)", expanded=False):
    sample_query = f"""
    SELECT 
        pickup_hour,
        pickup_weekday,
        trip_distance,
        total_amount,
        tip_amount,
        passenger_count,
        fare_amount
    FROM {from_clause}
    WHERE {where_clause}
    LIMIT 100
    """
    sample_df = safe_query_execution(conn, sample_query, "Failed to load sample data")

    if sample_df is not None and not sample_df.empty:
        st.dataframe(sample_df, use_container_width=True)

    with tab6:

        # --- Global style ---
        st.markdown("""
            <style>
            .report-container {
                background-color: #E8F1FA;  /* soft blue background */
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
        """, unsafe_allow_html=True)

        # --- Section title ---
        st.subheader("🔍 Airport Analysis Dashboard")

        col1, col2 = st.columns(2)

        # -------------------- COLUMN 1 --------------------
        with col1:
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown('<h4 class="report-title">Airport Trips by Day of Week</h4>', unsafe_allow_html=True)

            airport_query = f"""
            SELECT
                strftime(tpep_pickup_datetime, '%A') AS day_of_week,
                COUNT(*) AS total_trips,
                SUM(CASE WHEN airport_fee > 0 THEN 1 ELSE 0 END) AS airport_trips,
                ROUND(SUM(CASE WHEN airport_fee > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS airport_trip_pct
            FROM {from_clause}
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

            airport_df = safe_query_execution(conn, airport_query, "Failed to load airport trip data")

            if airport_df is not None and not airport_df.empty:
                fig = px.bar(
                    airport_df,
                    x="day_of_week",
                    y="airport_trip_pct",
                    title="Airport Trips Percentage by Day of Week",
                    color="airport_trip_pct",
                    color_continuous_scale="Blues",
                    labels={"day_of_week": "Day of Week", "airport_trip_pct": "Airport Trip %"}
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(airport_df, use_container_width=True)
            else:
                st.warning("No data found for airport trips.")

            st.markdown('</div>', unsafe_allow_html=True)

        # -------------------- COLUMN 2 --------------------
        with col2:
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown('<h4 class="report-title">Cost per Mile and Tip Percentage by Weekend & Time Period</h4>',
                        unsafe_allow_html=True)

            query = f"""
            SELECT
                is_weekend,
                time_period,
                ROUND(AVG(cost_per_mile), 2) AS avg_cost_per_mile,
                ROUND(AVG(tip_percentage), 2) AS avg_tip_percentage
            FROM {from_clause}
            WHERE {where_clause}
            GROUP BY is_weekend, time_period
            ORDER BY is_weekend, time_period
            """

            df = safe_query_execution(conn, query, "Failed to load cost/tip data")

            if df is not None and not df.empty:
                # Use the same blue scale as column 1
                blue_scale = "Blues"  # continuous blue scale

                # Avg Cost per Mile
                fig1 = px.bar(
                    df,
                    x="time_period",
                    y="avg_cost_per_mile",
                    color="avg_cost_per_mile",  # continuous coloring like col1
                    barmode="group",
                    title="Average Cost per Mile by Time Period",
                    labels={
                        "time_period": "Time of Day",
                        "avg_cost_per_mile": "Avg Cost per Mile ($)"
                    },
                    color_continuous_scale=blue_scale
                )
                st.plotly_chart(fig1, use_container_width=True)

                # Avg Tip Percentage
                fig2 = px.bar(
                    df,
                    x="time_period",
                    y="avg_tip_percentage",
                    color="avg_tip_percentage",  # continuous coloring
                    barmode="group",
                    title="Average Tip Percentage by Time Period",
                    labels={
                        "time_period": "Time of Day",
                        "avg_tip_percentage": "Avg Tip (%)"
                    },
                    color_continuous_scale=blue_scale
                )
                st.plotly_chart(fig2, use_container_width=True)


    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.sidebar.divider()
st.sidebar.info("""
**💡 Dashboard Features:**
- Real-time data filtering
- Interactive visualizations
- Statistical analysis
- Custom query support
- Data export capabilities
- Multi-year comparison
- Error handling & memory warnings
""")

st.sidebar.caption("Built with Streamlit, DuckDB & Plotly")
