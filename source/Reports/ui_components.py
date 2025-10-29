"""
UI components and rendering module
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import pandas as pd
from config import config


class UIComponents:
    """Reusable UI components for the dashboard"""

    @staticmethod
    def render_kpis(kpi_data: pd.DataFrame):
        """Render KPI metrics"""
        if kpi_data is None or kpi_data.empty or kpi_data['total_trips'].iloc[0] == 0:
            st.warning(" No data found for these filters. Please adjust your filter settings.")
            return False

        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)

        data = kpi_data.iloc[0]

        with col1:
            st.metric("🚕 Total Trips", f"{data['total_trips']:,}")
        with col2:
            st.metric("💵 Avg Fare", f"${data['avg_fare']:.2f}")
        with col3:
            st.metric("💰 Total Revenue", f"${data['total_revenue']:,.0f}")
        with col4:
            st.metric("📏 Avg Distance", f"{data['avg_distance']:.2f} mi")
        with col5:
            st.metric("🎁 Avg Tip", f"${data['avg_tip']:.2f}")
        with col6:
            st.metric("📊 Avg Tip %", f"{data['avg_tip_percentage']:.1f}%")
        with col7:
            st.metric("👥 Avg Passengers", f"{data['avg_passengers']:.1f}")
        with col8:
            revenue_per_trip = data['total_revenue'] / data['total_trips']
            st.metric("💲 Revenue/Trip", f"${revenue_per_trip:.2f}")

        return True

    @staticmethod
    def render_hourly_charts(hourly_df: Optional[pd.DataFrame]):
        """Render hourly analysis charts"""
        if hourly_df is None or hourly_df.empty:
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Trips by Hour")
            fig = px.line(
                hourly_df, x='Hour', y='Trips',
                title='Hourly Trip Volume',
                markers=True
            )
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Revenue by Hour")
            fig = px.bar(
                hourly_df, x='Hour', y='Avg_Fare',
                title='Average Fare by Hour',
                color='Avg_Fare',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_weekly_patterns(weekday_df: Optional[pd.DataFrame]):
        """Render weekly pattern analysis"""
        if weekday_df is None or weekday_df.empty:
            return

        st.subheader("Weekly Patterns")

        # Map weekday numbers to names
        weekday_df['Day'] = weekday_df['pickup_weekday'].map(
            lambda x: config.WEEKDAY_NAMES[x] if x < len(config.WEEKDAY_NAMES) else str(x)
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekday_df['Day'],
            y=weekday_df['trips'],
            name='Trips',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=weekday_df['Day'],
            y=weekday_df['avg_fare'],
            name='Avg Fare',
            yaxis='y2',
            mode='lines+markers'
        ))

        fig.update_layout(
            title='Weekly Trip Volume and Average Fare',
            yaxis=dict(title='Number of Trips'),
            yaxis2=dict(title='Average Fare ($)', overlaying='y', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_fare_distribution(fare_df: Optional[pd.DataFrame]):
        """Render fare distribution chart"""
        if fare_df is None or fare_df.empty:
            return

        st.subheader("Fare Distribution")
        fig = px.histogram(
            fare_df, x='fare_bucket', y='trip_count',
            title='Fare Distribution',
            labels={'fare_bucket': 'Fare ($)', 'trip_count': 'Trips'}
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_tip_analysis(tip_df: Optional[pd.DataFrame]):
        """Render tip analysis chart"""
        if tip_df is None or tip_df.empty:
            return

        st.subheader("Tip Analysis")
        fig = px.bar(
            tip_df, x='tip_percent_bucket', y='count',
            title='Tip Percentage Distribution',
            labels={'tip_percent_bucket': 'Tip %', 'count': 'Trips'}
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_revenue_trends(revenue_df: Optional[pd.DataFrame]):
        """Render revenue trends chart"""
        if revenue_df is None or revenue_df.empty:
            return

        st.subheader("Revenue Trends")
        revenue_df['date'] = revenue_df['month'].astype(str) + '-' + revenue_df['day'].astype(str)
        fig = px.line(
            revenue_df, x='date', y='daily_revenue',
            title='Daily Revenue Trend',
            labels={'daily_revenue': 'Revenue ($)', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_distance_distribution(distance_df: Optional[pd.DataFrame]):
        """Render distance distribution chart"""
        if distance_df is None or distance_df.empty:
            return

        st.subheader("Distance Distribution")
        fig = px.area(
            distance_df, x='distance_bucket', y='count',
            title='Trip Distance Distribution',
            labels={'distance_bucket': 'Distance (miles)', 'count': 'Trips'}
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_passenger_analysis(passenger_df: Optional[pd.DataFrame]):
        """Render passenger count analysis"""
        if passenger_df is None or passenger_df.empty:
            return

        st.subheader("Passenger Count")
        fig = px.bar(
            passenger_df, x='passenger_count', y='trips',
            title='Trips by Passenger Count',
            color='avg_fare',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_statistics(stats_df: Optional[pd.DataFrame]):
        """Render statistical summary table"""
        if stats_df is None or stats_df.empty:
            return

        st.subheader("📊 Statistical Summary")
        st.dataframe(stats_df, use_container_width=True)

    @staticmethod
    def render_correlation_matrix(corr_df: Optional[pd.DataFrame]):
        """Render correlation matrix heatmap"""
        if corr_df is None or corr_df.empty:
            return

        st.subheader("🔗 Correlation Analysis")
        correlation_matrix = corr_df.corr()
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_airport_analysis(airport_df: Optional[pd.DataFrame]):
        """Render airport trip analysis"""
        if airport_df is None or airport_df.empty:
            st.warning("No data found for airport trips.")
            return

        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown('<h4 class="report-title">Airport Trips by Day of Week</h4>', unsafe_allow_html=True)

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

        st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def render_cost_tip_analysis(cost_tip_df: Optional[pd.DataFrame]):
        """Render cost per mile and tip percentage analysis"""
        if cost_tip_df is None or cost_tip_df.empty:
            return

        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(
            '<h4 class="report-title">Cost per Mile and Tip Percentage by Time Period</h4>',
            unsafe_allow_html=True
        )

        # Average Cost per Mile
        fig1 = px.bar(
            cost_tip_df,
            x="time_period",
            y="avg_cost_per_mile",
            color="avg_cost_per_mile",
            barmode="group",
            title="Average Cost per Mile by Time Period",
            labels={
                "time_period": "Time of Day",
                "avg_cost_per_mile": "Avg Cost per Mile ($)"
            },
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Average Tip Percentage
        fig2 = px.bar(
            cost_tip_df,
            x="time_period",
            y="avg_tip_percentage",
            color="avg_tip_percentage",
            barmode="group",
            title="Average Tip Percentage by Time Period",
            labels={
                "time_period": "Time of Day",
                "avg_tip_percentage": "Avg Tip (%)"
            },
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)


class FilterPanel:
    """Sidebar filter panel"""

    @staticmethod
    def render() -> Dict[str, Any]:
        """
        Render filter panel in sidebar

        Returns:
            Dictionary of filter values
        """
        st.sidebar.header("🔍 Filters")
        filters = {}

        # Time Filters
        with st.sidebar.expander("⏰ Time Filters", expanded=True):
            filters['hour_range'] = st.slider(
                "Hour of Day:",
                min_value=0,
                max_value=23,
                value=config.DEFAULT_HOUR_RANGE
            )

            filters['weekdays'] = st.multiselect(
                "Days of Week:",
                options=list(config.WEEKDAY_MAP.keys()),
                default=list(config.WEEKDAY_MAP.keys())
            )

        # Financial Filters
        with st.sidebar.expander("💰 Financial Filters", expanded=True):
            filters['fare_range'] = st.slider(
                "Fare Range ($):",
                min_value=0.0,
                max_value=500.0,
                value=config.DEFAULT_FARE_RANGE,
                step=5.0
            )

            filters['tip_percentage'] = st.slider(
                "Minimum Tip % (of fare):",
                min_value=0,
                max_value=50,
                value=0,
                step=5
            )

        # Trip Filters
        with st.sidebar.expander("🚗 Trip Filters", expanded=True):
            filters['distance_range'] = st.slider(
                "Trip Distance (miles):",
                min_value=0.0,
                max_value=50.0,
                value=config.DEFAULT_DISTANCE_RANGE,
                step=1.0
            )

            filters['passenger_range'] = st.slider(
                "Passenger Count:",
                min_value=1,
                max_value=6,
                value=config.DEFAULT_PASSENGER_RANGE
            )

        return filters


# Global UI components instance
ui = UIComponents()
filter_panel = FilterPanel()