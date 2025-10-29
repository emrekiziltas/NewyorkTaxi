"""
NYC Taxi Analytics Dashboard - Main Application
"""
import streamlit as st
import pandas as pd
import io
from pathlib import Path

from config import config
from database import db
from data_loader import data_loader, memory_estimator
from query_builder import QueryBuilder
from ui_components import ui, filter_panel


def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        layout=config.LAYOUT,
        page_icon=config.PAGE_ICON,
        initial_sidebar_state="expanded"
    )
    st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_data_source():
    """Render data source selection in sidebar"""
    st.sidebar.header("📁 Data Source")

    # Get available files
    files_by_year = data_loader.get_parquet_files_by_year()

    if not files_by_year:
        st.error(f"❌ No data files found in: {config.DATA_DIR}")
        st.warning("""
        **Please run the data pipeline first:**
        ```bash
        python process_taxi_data.py
        ```
        This command will create Parquet files in the `data/processed/` folder.
        """)
        st.stop()

    # Year selection
    available_years = sorted(files_by_year.keys(), reverse=True)
    selected_year = st.sidebar.selectbox(
        "📅 Select Year:",
        options=available_years,
        index=0
    )

    files_for_year = files_by_year[selected_year]

    # Month selection mode
    selection_mode = st.sidebar.radio(
        "Selection Mode:",
        ["All Months", "Specific Months", "Date Range"],
        index=0
    )

    # Select files based on mode
    if selection_mode == "All Months":
        selected_files = [f.name for f in files_for_year]

    elif selection_mode == "Specific Months":
        selected_files = st.sidebar.multiselect(
            f"📄 Select {selected_year} Files:",
            options=[f.name for f in files_for_year],
            default=[f.name for f in files_for_year[:3]]
        )

    else:  # Date Range
        start_month = st.sidebar.selectbox(
            "Start Month:",
            config.MONTH_NAMES,
            index=0
        )
        end_month = st.sidebar.selectbox(
            "End Month:",
            config.MONTH_NAMES,
            index=11
        )

        start_idx = config.MONTH_NAMES.index(start_month) + 1
        end_idx = config.MONTH_NAMES.index(end_month) + 1

        selected_files = data_loader.filter_files_by_month_range(
            files_for_year,
            start_idx,
            end_idx
        )

    if not selected_files:
        st.warning("⚠️ Please select at least one file.")
        st.stop()

    selected_paths = [str(config.DATA_DIR / f) for f in selected_files]

    # Load and display data summary
    data_summary = data_loader.load_data_summary(selected_paths)
    if data_summary is not None and not data_summary.empty:
        total_records = data_summary['total_records'].iloc[0]
        st.sidebar.success(f"""
        📊 **Data Loaded:**
        - Files: {len(selected_files)}
        - Records: {total_records:,}
        - Date Range: {data_summary['min_date'].iloc[0]} to {data_summary['max_date'].iloc[0]}
        """)

        # Show memory estimation
        size_mb, level = memory_estimator.estimate_size(total_records)
        if level == "warning":
            st.sidebar.warning(f"⚠️ Dataset size: ~{size_mb:.1f} MB")
        elif level == "danger":
            st.sidebar.error(f"🚨 Large dataset: ~{size_mb:.1f} MB")
    else:
        st.sidebar.info(f"📊 {len(selected_files)} files selected from **{selected_year}**")

    return selected_year, selected_paths


def render_export_section(query_builder: QueryBuilder, where_clause: str):
    """Render data export section"""
    st.sidebar.header("💾 Export Data")

    export_format = st.sidebar.selectbox(
        "Export Format:",
        ["CSV", "Excel", "Parquet"]
    )

    export_limit = st.sidebar.number_input(
        "Export Row Limit:",
        min_value=config.MIN_EXPORT_LIMIT,
        max_value=config.MAX_EXPORT_LIMIT,
        value=config.DEFAULT_EXPORT_LIMIT,
        step=10000,
        help="Limit the number of rows to export to prevent memory issues"
    )

    return export_format, export_limit


def render_temporal_analysis_tab(query_builder: QueryBuilder, where_clause: str):
    """Render temporal analysis tab"""
    # Hourly charts
    hourly_df = db.execute_query(
        query_builder.get_hourly_query(where_clause),
        "Failed to load hourly data"
    )
    ui.render_hourly_charts(hourly_df)

    # Weekly patterns
    weekday_df = db.execute_query(
        query_builder.get_weekday_query(where_clause),
        "Failed to load weekly data"
    )
    ui.render_weekly_patterns(weekday_df)


def render_financial_insights_tab(query_builder: QueryBuilder, where_clause: str):
    """Render financial insights tab"""
    col1, col2 = st.columns(2)

    with col1:
        fare_df = db.execute_query(
            query_builder.get_fare_distribution_query(where_clause),
            "Failed to load fare distribution"
        )
        ui.render_fare_distribution(fare_df)

    with col2:
        tip_df = db.execute_query(
            query_builder.get_tip_analysis_query(where_clause),
            "Failed to load tip analysis"
        )
        ui.render_tip_analysis(tip_df)

    # Revenue trends
    revenue_df = db.execute_query(
        query_builder.get_revenue_trends_query(where_clause),
        "Failed to load revenue trends"
    )
    ui.render_revenue_trends(revenue_df)


def render_trip_patterns_tab(query_builder: QueryBuilder, where_clause: str):
    """Render trip patterns tab"""
    col1, col2 = st.columns(2)

    with col1:
        distance_df = db.execute_query(
            query_builder.get_distance_distribution_query(where_clause),
            "Failed to load distance distribution"
        )
        ui.render_distance_distribution(distance_df)

    with col2:
        passenger_df = db.execute_query(
            query_builder.get_passenger_query(where_clause),
            "Failed to load passenger data"
        )
        ui.render_passenger_analysis(passenger_df)


def render_statistical_analysis_tab(query_builder: QueryBuilder, where_clause: str):
    """Render statistical analysis tab"""
    stats_df = db.execute_query(
        query_builder.get_statistics_query(where_clause),
        "Failed to calculate statistics"
    )
    ui.render_statistics(stats_df)

    # Correlation analysis
    corr_df = db.execute_query(
        query_builder.get_correlation_query(where_clause),
        "Failed to calculate correlations"
    )
    ui.render_correlation_matrix(corr_df)


def render_deep_dive_tab(query_builder: QueryBuilder, where_clause: str,
                         export_format: str, export_limit: int, selected_year: str):
    """Render deep dive tab"""
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
        FROM {query_builder.from_clause}
        WHERE {where_clause}
        GROUP BY pickup_month, pickup_day
        ORDER BY revenue DESC
        LIMIT 10
        """
        result_df = db.execute_query(top_days_query, "Failed to load top revenue days")

        if result_df is not None:
            st.dataframe(result_df, use_container_width=True)

    elif query_template == "Custom Query":
        custom_query = st.text_area(
            "Enter your DuckDB query:",
            f"SELECT * FROM {query_builder.from_clause} WHERE {where_clause} LIMIT 100",
            height=150
        )

        if st.button("Execute Query"):
            result_df = db.execute_query(custom_query, "Custom query execution failed")

            if result_df is not None:
                st.dataframe(result_df, use_container_width=True)
                st.success(f"✅ Query executed successfully. Returned {len(result_df):,} rows.")

    # Data Export
    st.subheader("💾 Export Filtered Data")

    # Count records
    count_query = f"""
    SELECT COUNT(*) as total_count
    FROM {query_builder.from_clause}
    WHERE {where_clause}
    """
    count_result = db.execute_query(count_query, "Failed to count records")

    if count_result is not None and not count_result.empty:
        total_count = count_result['total_count'].iloc[0]
        memory_estimator.show_warning(min(total_count, export_limit))
        st.info(f"📊 Total matching records: {total_count:,} | Export limit: {export_limit:,}")

        if total_count > export_limit:
            st.warning(
                f"⚠️ Dataset exceeds export limit. "
                f"Only the first {export_limit:,} records will be exported."
            )

    if st.button("Generate Export"):
        with st.spinner("Generating export file..."):
            export_query = query_builder.get_export_query(where_clause, export_limit)
            export_df = db.execute_query(export_query, "Failed to generate export")

            if export_df is not None and not export_df.empty:
                handle_export(export_df, export_format, selected_year)


def handle_export(export_df: pd.DataFrame, export_format: str, selected_year: str):
    """Handle file export based on format"""
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
        buffer = io.BytesIO()
        export_df.to_parquet(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Parquet",
            data=buffer,
            file_name=f"nyc_taxi_{selected_year}_export.parquet",
            mime="application/octet-stream"
        )
        st.success(f" Parquet file ready! ({len(export_df):,} rows)")

    elif export_format == "Excel":
        try:
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
            st.success(f" Excel file ready! ({len(export_df):,} rows)")
        except ImportError:
            st.error(" Excel export requires openpyxl. Install with: `pip install openpyxl`")


def render_airport_tab(query_builder: QueryBuilder, where_clause: str):
    """Render airport analysis tab"""
    st.subheader("🔍 Airport Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        airport_df = db.execute_query(
            query_builder.get_airport_query(where_clause),
            "Failed to load airport trip data"
        )
        ui.render_airport_analysis(airport_df)

    with col2:
        cost_tip_df = db.execute_query(
            query_builder.get_cost_tip_query(where_clause),
            "Failed to load cost/tip data"
        )
        ui.render_cost_tip_analysis(cost_tip_df)


def render_footer():
    """Render sidebar footer"""
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


def main():
    """Main application entry point"""
    # Setup
    setup_page()

    # Sidebar - Data Source
    selected_year, selected_paths = render_sidebar_data_source()

    # Sidebar - Filters
    filters = filter_panel.render()

    # Sidebar - Export Options
    export_format, export_limit = render_export_section(None, "")

    # Initialize query builder
    query_builder = QueryBuilder(selected_paths)
    where_clause = query_builder.build_where_clause(filters)

    # Main Dashboard Header
    st.markdown(
        f'<h1 class="main-header">🚕 NYC Taxi Analytics - {selected_year}</h1>',
        unsafe_allow_html=True
    )

    # KPIs
    kpi_result = db.execute_query(
        query_builder.get_kpi_query(where_clause),
        "Failed to calculate KPIs"
    )

    if not ui.render_kpis(kpi_result):
        st.stop()

    st.divider()

    # Analytics Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⏰ Temporal Analysis",
        "💰 Financial Insights",
        "🗺️ Trip Patterns",
        "📊 Statistical Analysis",
        "🔍 Deep Dive",
        "✈️ Airport"
    ])

    with tab1:
        render_temporal_analysis_tab(query_builder, where_clause)

    with tab2:
        render_financial_insights_tab(query_builder, where_clause)

    with tab3:
        render_trip_patterns_tab(query_builder, where_clause)

    with tab4:
        render_statistical_analysis_tab(query_builder, where_clause)

    with tab5:
        render_deep_dive_tab(
            query_builder, where_clause,
            export_format, export_limit, selected_year
        )

    with tab6:
        render_airport_tab(query_builder, where_clause)

    # Sample Data Table
    st.divider()
    with st.expander("🔍 Filtered Raw Data (Sample)", expanded=False):
        sample_df = db.execute_query(
            query_builder.get_sample_query(where_clause),
            "Failed to load sample data"
        )

        if sample_df is not None and not sample_df.empty:
            st.dataframe(sample_df, use_container_width=True)

    # Footer
    render_footer()


if __name__ == "__main__":
    main()