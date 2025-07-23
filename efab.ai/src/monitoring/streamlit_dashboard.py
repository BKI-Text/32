#!/usr/bin/env python3
"""
Streamlit-based Performance Monitoring Dashboard
Beverly Knits AI Supply Chain Planner
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.monitoring import get_performance_dashboard, start_monitoring

# Page configuration
st.set_page_config(
    page_title="ML Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .alert-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_dashboard_data(hours: int = 24):
    """Get dashboard data with caching"""
    dashboard = get_performance_dashboard()
    return dashboard.get_dashboard_data(hours=hours)

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_model_report(model_id: str, hours: int = 24):
    """Get model performance report with caching"""
    dashboard = get_performance_dashboard()
    return dashboard.get_model_performance_report(model_id, hours=hours)

def display_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Display a metric card"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label=title, value=value, delta=delta, help=help_text)

def plot_system_metrics(system_metrics: list):
    """Plot system performance metrics"""
    if not system_metrics:
        st.warning("No system metrics available")
        return
    
    df = pd.DataFrame(system_metrics)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)', 'Active Connections'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CPU Usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Memory Usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %', line=dict(color='green')),
        row=1, col=2
    )
    
    # Disk Usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['disk_usage'], name='Disk %', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Active Connections
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['active_connections'], name='Connections', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(
        title="System Performance Metrics",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_model_metrics(model_metrics: dict):
    """Plot model performance metrics"""
    if not model_metrics:
        st.info("No model metrics available")
        return
    
    # Model selector
    selected_model = st.selectbox("Select Model", list(model_metrics.keys()))
    
    if selected_model and model_metrics[selected_model]:
        df = pd.DataFrame(model_metrics[selected_model])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create tabs for different metrics
        tab1, tab2, tab3, tab4 = st.tabs(["Latency & Throughput", "Accuracy & Error Rate", "Resource Usage", "Request Counts"])
        
        with tab1:
            # Latency and Throughput
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Prediction Latency (ms)', 'Throughput (req/s)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['prediction_latency']*1000, name='Latency (ms)', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['throughput'], name='Throughput', line=dict(color='green')),
                row=1, col=2
            )
            
            fig.update_layout(title=f"Performance Metrics - {selected_model}", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Accuracy and Error Rate
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy', 'Error Rate (%)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            if df['accuracy'].notna().any():
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['accuracy'], name='Accuracy', line=dict(color='blue')),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['error_rate']*100, name='Error Rate %', line=dict(color='red')),
                row=1, col=2
            )
            
            fig.update_layout(title=f"Quality Metrics - {selected_model}", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Resource Usage
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Memory Usage (MB)', 'CPU Usage (%)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['memory_usage'], name='Memory MB', line=dict(color='orange')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['cpu_usage'], name='CPU %', line=dict(color='red')),
                row=1, col=2
            )
            
            fig.update_layout(title=f"Resource Usage - {selected_model}", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Request Counts
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Prediction Count', 'Error Count'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['prediction_count'], name='Predictions', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['error_count'], name='Errors', line=dict(color='red')),
                row=1, col=2
            )
            
            fig.update_layout(title=f"Request Counts - {selected_model}", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def display_alerts(active_alerts: list, alert_history: list):
    """Display alerts section"""
    st.subheader("🚨 Alerts")
    
    # Active alerts
    if active_alerts:
        st.error(f"**{len(active_alerts)} Active Alerts**")
        for alert in active_alerts:
            st.markdown(f"""
            <div class="alert-high">
                <strong>{alert['rule_name']}</strong><br>
                {alert['metric_name']} {alert['condition']} {alert['threshold']} 
                (Current: {alert['current_value']:.2f})<br>
                Triggered: {alert['triggered_at']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No active alerts")
    
    # Alert history
    with st.expander("Alert History", expanded=False):
        if alert_history:
            df_alerts = pd.DataFrame(alert_history)
            df_alerts['triggered_at'] = pd.to_datetime(df_alerts['triggered_at'])
            df_alerts = df_alerts.sort_values('triggered_at', ascending=False)
            
            st.dataframe(
                df_alerts[['rule_name', 'metric_name', 'current_value', 'threshold', 'triggered_at', 'status']],
                use_container_width=True
            )
        else:
            st.info("No alert history available")

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<div class="main-header">🎯 ML Performance Monitoring Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Settings")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=2
    )
    
    hours_map = {
        "Last 1 Hour": 1,
        "Last 6 Hours": 6,
        "Last 24 Hours": 24,
        "Last 7 Days": 168
    }
    
    selected_hours = hours_map[time_range]
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        st.sidebar.info("Dashboard refreshes every 30 seconds")
        # Force refresh every 30 seconds
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Start monitoring if not already running
    try:
        start_monitoring()
    except Exception as e:
        st.sidebar.error(f"Error starting monitoring: {e}")
    
    # Get dashboard data
    try:
        dashboard_data = get_dashboard_data(hours=selected_hours)
        
        if 'error' in dashboard_data:
            st.error(f"Error loading dashboard data: {dashboard_data['error']}")
            return
        
        # Summary metrics
        st.subheader("📊 Summary Metrics")
        
        if 'summary' in dashboard_data:
            summary = dashboard_data['summary']
            
            # System metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'system' in summary:
                    display_metric_card(
                        "Average CPU",
                        f"{summary['system']['avg_cpu']:.1f}%",
                        help_text="Average CPU usage across the time period"
                    )
            
            with col2:
                if 'system' in summary:
                    display_metric_card(
                        "Average Memory",
                        f"{summary['system']['avg_memory']:.1f}%",
                        help_text="Average memory usage across the time period"
                    )
            
            with col3:
                if 'models' in summary:
                    display_metric_card(
                        "Total Models",
                        str(summary['models']['total_models']),
                        help_text="Number of models being monitored"
                    )
            
            with col4:
                if 'models' in summary:
                    display_metric_card(
                        "Total Predictions",
                        str(summary['models']['total_predictions']),
                        help_text="Total predictions made by all models"
                    )
            
            # Model metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'models' in summary:
                    display_metric_card(
                        "Avg Latency",
                        f"{summary['models']['avg_latency']*1000:.1f}ms",
                        help_text="Average prediction latency"
                    )
            
            with col2:
                if 'models' in summary:
                    display_metric_card(
                        "Avg Throughput",
                        f"{summary['models']['avg_throughput']:.1f} req/s",
                        help_text="Average request throughput"
                    )
            
            with col3:
                if 'models' in summary:
                    display_metric_card(
                        "Error Rate",
                        f"{summary['models']['error_rate']:.2f}%",
                        help_text="Overall error rate across all models"
                    )
            
            with col4:
                if 'system' in summary:
                    display_metric_card(
                        "Active Connections",
                        str(summary['system']['active_connections']),
                        help_text="Current number of active connections"
                    )
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖥️ System Performance")
            plot_system_metrics(dashboard_data.get('system_metrics', []))
        
        with col2:
            st.subheader("🤖 Model Performance")
            plot_model_metrics(dashboard_data.get('model_metrics', {}))
        
        # Alerts section
        display_alerts(
            dashboard_data.get('active_alerts', []),
            dashboard_data.get('alert_history', [])
        )
        
        # Detailed model reports
        st.subheader("📋 Detailed Model Reports")
        
        if dashboard_data.get('model_metrics'):
            model_list = list(dashboard_data['model_metrics'].keys())
            
            if model_list:
                selected_model_report = st.selectbox(
                    "Select Model for Detailed Report",
                    model_list,
                    key="model_report_selector"
                )
                
                if selected_model_report:
                    with st.expander(f"Detailed Report - {selected_model_report}", expanded=True):
                        report = get_model_report(selected_model_report, hours=selected_hours)
                        
                        if 'error' in report:
                            st.error(f"Error generating report: {report['error']}")
                        else:
                            # Report summary
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Data Points", report['total_data_points'])
                            
                            with col2:
                                if report['performance_stats']['accuracy']['avg'] is not None:
                                    st.metric("Avg Accuracy", f"{report['performance_stats']['accuracy']['avg']:.3f}")
                                else:
                                    st.metric("Avg Accuracy", "N/A")
                            
                            with col3:
                                st.metric("P95 Latency", f"{report['performance_stats']['latency']['p95']*1000:.1f}ms")
                            
                            # Performance statistics table
                            st.subheader("Performance Statistics")
                            
                            perf_data = []
                            for metric_name, stats in report['performance_stats'].items():
                                if stats and isinstance(stats, dict):
                                    for stat_name, value in stats.items():
                                        if value is not None:
                                            perf_data.append({
                                                'Metric': f"{metric_name}_{stat_name}",
                                                'Value': f"{value:.4f}" if isinstance(value, float) else str(value)
                                            })
                            
                            if perf_data:
                                st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
        
        # Status information
        st.subheader("ℹ️ Monitoring Status")
        
        monitoring_status = dashboard_data.get('monitoring_status', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if monitoring_status.get('collector_running', False) else "🔴"
            st.write(f"{status_color} Collector Running: {monitoring_status.get('collector_running', False)}")
        
        with col2:
            st.write(f"📊 Total Models: {monitoring_status.get('total_models', 0)}")
        
        with col3:
            st.write(f"📈 Data Points: {monitoring_status.get('data_points', 0)}")
        
        # Footer
        st.markdown("---")
        st.markdown(f"*Last Updated: {dashboard_data.get('timestamp', 'Unknown')}*")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()