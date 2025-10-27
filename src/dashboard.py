import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path
import re
import plotly.express as px
import plotly.graph_objects as go

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

DATA_DIR = Path(r"C:\Users\ek675\PycharmProjects\PythonProject\NewyorkTaxi\src\data\processed")


# --- Find Data Files and Group by Year ---
@st.cache_resource
def get_parquet_files_by_year():
    """Find Parquet files in processed folder and group by year"""
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


@st.cache_data
def load_data_summary(file_paths):
    """Load summary statistics from selected files"""
    try:
        conn = get_db_connection()
        file_list = "', '".join(file_paths)
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            MIN(tpep_pickup_datetime) as min_date,
            MAX(tpep_pickup_datetime) as max_date
        FROM read_parquet(['{file_list}'])
        """
        return conn.execute(query).fetchdf()
    except:
        return None


files_by_year = get_parquet_files_by_year()

# --- File Check ---
if not files_by_year:
    st.error(f"❌ No data files found: {DATA_DIR}")
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

# Load data summary
data_summary = load_data_summary(selected_paths)
if data_summary is not None and not data_summary.empty:
    st.sidebar.success(f"""
    📊 **Data Loaded:**
    - Files: {len(selected_files)}
    - Records: {data_summary['total_records'].iloc[0]:,}
    - Date Range: {data_summary['min_date'].iloc[0]} to {data_summary['max_date'].iloc[0]}
    """)
else:
    st.sidebar.info(f"📊 {len(selected_files)} files selected from **{selected_year}**")


# --- DuckDB Connection ---
@st.cache_resource
def get_db_connection():
    """Cache DuckDB connection"""
    return duckdb.connect(':memory:')


conn = get_db_connection()

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
try:
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

    kpi_result = conn.execute(kpi_query).fetchdf()

    if not kpi_result.empty and kpi_result['total_trips'].iloc[0] > 0:
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
        st.warning("⚠️ No data found for these filters.")
        st.stop()

except Exception as e:
    st.error(f"❌ KPI calculation error: {e}")
    st.stop()

st.divider()

# --- Advanced Analytics Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⏰ Temporal Analysis",
    "💰 Financial Insights",
    "🗺️ Trip Patterns",
    "📊 Statistical Analysis",
    "🔍 Deep Dive"
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
        hourly_df = conn.execute(hourly_query).fetchdf()

        if not hourly_df.empty:
            fig = px.line(hourly_df, x='Hour', y='Trips',
                          title='Hourly Trip Volume',
                          markers=True)
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")

    with col2:
        st.subheader("Revenue by Hour")
        if not hourly_df.empty:
            fig = px.bar(hourly_df, x='Hour', y='Avg_Fare',
                         title='Average Fare by Hour',
                         color='Avg_Fare',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")

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
    weekday_df = conn.execute(weekday_query).fetchdf()

    if not weekday_df.empty:
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
        fare_dist_df = conn.execute(fare_dist_query).fetchdf()

        if not fare_dist_df.empty:
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
        tip_df = conn.execute(tip_query).fetchdf()

        if not tip_df.empty:
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
    revenue_df = conn.execute(revenue_query).fetchdf()

    if not revenue_df.empty:
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
        distance_df = conn.execute(distance_query).fetchdf()

        if not distance_df.empty:
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
        passenger_df = conn.execute(passenger_query).fetchdf()

        if not passenger_df.empty:
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

    stats_df = conn.execute(stats_query).fetchdf()
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
    corr_df = conn.execute(corr_query).fetchdf()

    if not corr_df.empty:
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
        result_df = conn.execute(top_days_query).fetchdf()
        st.dataframe(result_df, use_container_width=True)

    elif query_template == "Custom Query":
        custom_query = st.text_area(
            "Enter your DuckDB query:",
            f"SELECT * FROM {from_clause} WHERE {where_clause} LIMIT 100",
            height=150
        )

        if st.button("Execute Query"):
            try:
                result_df = conn.execute(custom_query).fetchdf()
                st.dataframe(result_df, use_container_width=True)
                st.success(f"✅ Query executed successfully. Returned {len(result_df)} rows.")
            except Exception as e:
                st.error(f"❌ Query error: {e}")

    # Data Export
    st.subheader("💾 Export Filtered Data")
    if st.button("Generate Export"):
        export_query = f"""
        SELECT *
        FROM {from_clause}
        WHERE {where_clause}
        LIMIT 100000
        """
        export_df = conn.execute(export_query).fetchdf()

        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"nyc_taxi_{selected_year}_export.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            st.info("Excel export requires openpyxl. Install with: pip install openpyxl")

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
    sample_df = conn.execute(sample_query).fetchdf()
    st.dataframe(sample_df, use_container_width=True)

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
""")

st.sidebar.caption("Built with Streamlit, DuckDB & Plotly")
