#!/usr/bin/env python3
"""
Real-time Performance Monitoring Dashboard
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import threading
from queue import Queue, Empty
import psutil
import traceback

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    metric_name: str
    metric_value: float
    metric_type: str  # 'gauge', 'counter', 'histogram'
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'labels': self.labels,
            'metadata': self.metadata
        }

@dataclass
class ModelPerformanceData:
    """Model performance data"""
    model_id: str
    model_name: str
    timestamp: datetime
    prediction_latency: float
    throughput: float
    accuracy: Optional[float] = None
    error_rate: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    prediction_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'prediction_latency': self.prediction_latency,
            'throughput': self.throughput,
            'accuracy': self.accuracy,
            'error_rate': self.error_rate,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'prediction_count': self.prediction_count,
            'error_count': self.error_count
        }

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    response_time: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'active_connections': self.active_connections,
            'response_time': self.response_time,
            'error_rate': self.error_rate
        }

class MetricsCollector:
    """Collect performance metrics from various sources"""
    
    def __init__(self):
        self.metrics_queue = Queue()
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.system_metrics: deque = deque(maxlen=1000)
        self.running = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
        
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
        
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Process queued metrics
                self._process_queued_metrics()
                
                # Sleep for collection interval
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/').percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Network connections
            active_connections = len(psutil.net_connections())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_info.percent,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections,
                response_time=0.0,  # Will be updated by application
                error_rate=0.0      # Will be updated by application
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                response_time=0.0,
                error_rate=0.0
            )
    
    def _process_queued_metrics(self):
        """Process metrics from queue"""
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                if isinstance(metric, ModelPerformanceData):
                    self.model_metrics[metric.model_id].append(metric)
                elif isinstance(metric, PerformanceMetric):
                    # Store custom metrics
                    self.model_metrics[f"custom_{metric.metric_name}"].append(metric)
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error processing queued metric: {e}")
    
    def add_model_metric(self, metric: ModelPerformanceData):
        """Add model performance metric"""
        self.metrics_queue.put(metric)
    
    def add_custom_metric(self, metric: PerformanceMetric):
        """Add custom metric"""
        self.metrics_queue.put(metric)
    
    def get_model_metrics(self, model_id: str, minutes: int = 60) -> List[ModelPerformanceData]:
        """Get model metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        metrics = self.model_metrics.get(model_id, [])
        
        return [
            metric for metric in metrics
            if isinstance(metric, ModelPerformanceData) and metric.timestamp > cutoff_time
        ]
    
    def get_system_metrics(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            metric for metric in self.system_metrics
            if metric.timestamp > cutoff_time
        ]
    
    def get_all_model_ids(self) -> List[str]:
        """Get all model IDs being monitored"""
        return [
            key for key in self.model_metrics.keys()
            if not key.startswith('custom_')
        ]

class AlertSystem:
    """Alert system for performance monitoring"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
    def add_alert_rule(self, rule_name: str, metric_name: str, condition: str, 
                      threshold: float, duration_minutes: int = 5):
        """
        Add alert rule
        
        Args:
            rule_name: Name of the alert rule
            metric_name: Metric to monitor
            condition: Condition ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
            duration_minutes: Duration before triggering alert
        """
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'condition': condition,
            'threshold': threshold,
            'duration_minutes': duration_minutes,
            'created_at': datetime.now()
        }
        
        logger.info(f"Added alert rule: {rule_name}")
    
    def check_alerts(self, metrics_collector: MetricsCollector):
        """Check alert conditions"""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Get recent metrics
                if rule['metric_name'] == 'cpu_percent':
                    recent_metrics = metrics_collector.get_system_metrics(minutes=rule['duration_minutes'])
                    values = [m.cpu_percent for m in recent_metrics]
                elif rule['metric_name'] == 'memory_percent':
                    recent_metrics = metrics_collector.get_system_metrics(minutes=rule['duration_minutes'])
                    values = [m.memory_percent for m in recent_metrics]
                elif rule['metric_name'] == 'response_time':
                    recent_metrics = metrics_collector.get_system_metrics(minutes=rule['duration_minutes'])
                    values = [m.response_time for m in recent_metrics]
                else:
                    continue
                
                if not values:
                    continue
                
                # Check condition
                avg_value = np.mean(values)
                condition_met = self._evaluate_condition(avg_value, rule['condition'], rule['threshold'])
                
                if condition_met:
                    if rule_name not in self.active_alerts:
                        # New alert
                        alert = {
                            'rule_name': rule_name,
                            'metric_name': rule['metric_name'],
                            'current_value': avg_value,
                            'threshold': rule['threshold'],
                            'condition': rule['condition'],
                            'triggered_at': current_time,
                            'status': 'active'
                        }
                        
                        self.active_alerts[rule_name] = alert
                        self.alert_history.append(alert.copy())
                        
                        logger.warning(f"Alert triggered: {rule_name} - {rule['metric_name']} {rule['condition']} {rule['threshold']} (current: {avg_value:.2f})")
                else:
                    if rule_name in self.active_alerts:
                        # Resolve alert
                        alert = self.active_alerts[rule_name]
                        alert['status'] = 'resolved'
                        alert['resolved_at'] = current_time
                        
                        del self.active_alerts[rule_name]
                        
                        logger.info(f"Alert resolved: {rule_name}")
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return value == threshold
        elif condition == '!=':
            return value != threshold
        else:
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert['triggered_at'] > cutoff_time
        ]

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, db_path: str = "data/monitoring/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector()
        self.alert_system = AlertSystem()
        
        self._init_database()
        self._setup_default_alerts()
        
        logger.info("Performance dashboard initialized")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Model performance metrics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prediction_latency REAL,
                    throughput REAL,
                    accuracy REAL,
                    error_rate REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    prediction_count INTEGER,
                    error_count INTEGER
                )
            ''')
            
            # System metrics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage REAL,
                    network_io TEXT,
                    active_connections INTEGER,
                    response_time REAL,
                    error_rate REAL
                )
            ''')
            
            # Custom metrics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS custom_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_type TEXT,
                    labels TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model_timestamp ON model_metrics(model_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_custom_timestamp ON custom_metrics(metric_name, timestamp)')
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High CPU usage
        self.alert_system.add_alert_rule(
            'high_cpu_usage',
            'cpu_percent',
            '>',
            80.0,
            duration_minutes=5
        )
        
        # High memory usage
        self.alert_system.add_alert_rule(
            'high_memory_usage',
            'memory_percent',
            '>',
            85.0,
            duration_minutes=5
        )
        
        # High response time
        self.alert_system.add_alert_rule(
            'high_response_time',
            'response_time',
            '>',
            2.0,
            duration_minutes=3
        )
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.metrics_collector.start_collection()
        
        # Start alert checking in background
        asyncio.create_task(self._alert_checking_loop())
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.metrics_collector.stop_collection()
        logger.info("Performance monitoring stopped")
    
    async def _alert_checking_loop(self):
        """Background alert checking loop"""
        while True:
            try:
                self.alert_system.check_alerts(self.metrics_collector)
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert checking: {e}")
                await asyncio.sleep(60)
    
    def record_model_performance(self, model_id: str, model_name: str, 
                                prediction_latency: float, throughput: float,
                                accuracy: Optional[float] = None, error_rate: float = 0.0,
                                memory_usage: float = 0.0, cpu_usage: float = 0.0,
                                prediction_count: int = 0, error_count: int = 0):
        """Record model performance metrics"""
        metric = ModelPerformanceData(
            model_id=model_id,
            model_name=model_name,
            timestamp=datetime.now(),
            prediction_latency=prediction_latency,
            throughput=throughput,
            accuracy=accuracy,
            error_rate=error_rate,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            prediction_count=prediction_count,
            error_count=error_count
        )
        
        # Add to collector
        self.metrics_collector.add_model_metric(metric)
        
        # Store in database
        self._store_model_metric(metric)
    
    def _store_model_metric(self, metric: ModelPerformanceData):
        """Store model metric in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_metrics (
                        model_id, model_name, timestamp, prediction_latency, throughput,
                        accuracy, error_rate, memory_usage, cpu_usage, prediction_count, error_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.model_id,
                    metric.model_name,
                    metric.timestamp.isoformat(),
                    metric.prediction_latency,
                    metric.throughput,
                    metric.accuracy,
                    metric.error_rate,
                    metric.memory_usage,
                    metric.cpu_usage,
                    metric.prediction_count,
                    metric.error_count
                ))
        except Exception as e:
            logger.error(f"Error storing model metric: {e}")
    
    def record_custom_metric(self, metric_name: str, metric_value: float, 
                           metric_type: str = 'gauge', labels: Dict[str, str] = None,
                           metadata: Dict[str, Any] = None):
        """Record custom metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        # Add to collector
        self.metrics_collector.add_custom_metric(metric)
        
        # Store in database
        self._store_custom_metric(metric)
    
    def _store_custom_metric(self, metric: PerformanceMetric):
        """Store custom metric in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO custom_metrics (
                        timestamp, metric_name, metric_value, metric_type, labels, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.metric_value,
                    metric.metric_type,
                    json.dumps(metric.labels),
                    json.dumps(metric.metadata)
                ))
        except Exception as e:
            logger.error(f"Error storing custom metric: {e}")
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get model metrics
            model_metrics = {}
            for model_id in self.metrics_collector.get_all_model_ids():
                metrics = self.metrics_collector.get_model_metrics(model_id, minutes=hours*60)
                if metrics:
                    model_metrics[model_id] = [m.to_dict() for m in metrics]
            
            # Get system metrics
            system_metrics = self.metrics_collector.get_system_metrics(minutes=hours*60)
            system_data = [m.to_dict() for m in system_metrics]
            
            # Get alerts
            active_alerts = self.alert_system.get_active_alerts()
            alert_history = self.alert_system.get_alert_history(hours=hours)
            
            # Calculate summary statistics
            summary = self._calculate_summary_stats(system_metrics, model_metrics)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'model_metrics': model_metrics,
                'system_metrics': system_data,
                'active_alerts': active_alerts,
                'alert_history': alert_history,
                'monitoring_status': {
                    'collector_running': self.metrics_collector.running,
                    'total_models': len(model_metrics),
                    'data_points': len(system_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'monitoring_status': {
                    'collector_running': self.metrics_collector.running,
                    'total_models': 0,
                    'data_points': 0
                }
            }
    
    def _calculate_summary_stats(self, system_metrics: List[SystemMetrics], 
                               model_metrics: Dict[str, List]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not system_metrics:
            return {}
        
        # System stats
        cpu_values = [m.cpu_percent for m in system_metrics]
        memory_values = [m.memory_percent for m in system_metrics]
        
        # Model stats
        total_predictions = 0
        total_errors = 0
        avg_latency = 0
        avg_throughput = 0
        
        model_count = 0
        for model_id, metrics in model_metrics.items():
            if metrics:
                model_count += 1
                latest_metric = metrics[-1]
                total_predictions += latest_metric.get('prediction_count', 0)
                total_errors += latest_metric.get('error_count', 0)
                avg_latency += latest_metric.get('prediction_latency', 0)
                avg_throughput += latest_metric.get('throughput', 0)
        
        if model_count > 0:
            avg_latency /= model_count
            avg_throughput /= model_count
        
        return {
            'system': {
                'avg_cpu': np.mean(cpu_values) if cpu_values else 0,
                'max_cpu': np.max(cpu_values) if cpu_values else 0,
                'avg_memory': np.mean(memory_values) if memory_values else 0,
                'max_memory': np.max(memory_values) if memory_values else 0,
                'active_connections': system_metrics[-1].active_connections if system_metrics else 0
            },
            'models': {
                'total_models': model_count,
                'total_predictions': total_predictions,
                'total_errors': total_errors,
                'avg_latency': avg_latency,
                'avg_throughput': avg_throughput,
                'error_rate': (total_errors / total_predictions * 100) if total_predictions > 0 else 0
            }
        }
    
    def get_model_performance_report(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance report for a specific model"""
        try:
            metrics = self.metrics_collector.get_model_metrics(model_id, minutes=hours*60)
            
            if not metrics:
                return {
                    'model_id': model_id,
                    'error': 'No metrics found for this model',
                    'report_time': datetime.now().isoformat()
                }
            
            # Calculate statistics
            latencies = [m.prediction_latency for m in metrics]
            throughputs = [m.throughput for m in metrics]
            accuracies = [m.accuracy for m in metrics if m.accuracy is not None]
            error_rates = [m.error_rate for m in metrics]
            
            return {
                'model_id': model_id,
                'model_name': metrics[0].model_name,
                'report_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'total_data_points': len(metrics),
                'performance_stats': {
                    'latency': {
                        'avg': np.mean(latencies),
                        'min': np.min(latencies),
                        'max': np.max(latencies),
                        'p50': np.percentile(latencies, 50),
                        'p95': np.percentile(latencies, 95),
                        'p99': np.percentile(latencies, 99)
                    },
                    'throughput': {
                        'avg': np.mean(throughputs),
                        'min': np.min(throughputs),
                        'max': np.max(throughputs)
                    },
                    'accuracy': {
                        'avg': np.mean(accuracies) if accuracies else None,
                        'min': np.min(accuracies) if accuracies else None,
                        'max': np.max(accuracies) if accuracies else None
                    },
                    'error_rate': {
                        'avg': np.mean(error_rates),
                        'max': np.max(error_rates)
                    }
                },
                'recent_metrics': [m.to_dict() for m in metrics[-10:]]  # Last 10 metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating model performance report: {e}")
            return {
                'model_id': model_id,
                'error': str(e),
                'report_time': datetime.now().isoformat()
            }
    
    def export_metrics(self, output_file: str, hours: int = 24):
        """Export metrics to JSON file"""
        try:
            dashboard_data = self.get_dashboard_data(hours=hours)
            
            with open(output_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            logger.info(f"Metrics exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old metrics data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Clean up old model metrics
                result = conn.execute(
                    "DELETE FROM model_metrics WHERE timestamp < ?",
                    (cutoff_str,)
                )
                model_deleted = result.rowcount
                
                # Clean up old system metrics
                result = conn.execute(
                    "DELETE FROM system_metrics WHERE timestamp < ?",
                    (cutoff_str,)
                )
                system_deleted = result.rowcount
                
                # Clean up old custom metrics
                result = conn.execute(
                    "DELETE FROM custom_metrics WHERE timestamp < ?",
                    (cutoff_str,)
                )
                custom_deleted = result.rowcount
                
                logger.info(f"Cleaned up old data: {model_deleted} model metrics, {system_deleted} system metrics, {custom_deleted} custom metrics")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

# Global dashboard instance
performance_dashboard = PerformanceDashboard()

def get_performance_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard instance"""
    return performance_dashboard

def start_monitoring():
    """Start global performance monitoring"""
    performance_dashboard.start_monitoring()

def stop_monitoring():
    """Stop global performance monitoring"""
    performance_dashboard.stop_monitoring()

def record_model_performance(*args, **kwargs):
    """Record model performance using global dashboard"""
    performance_dashboard.record_model_performance(*args, **kwargs)

def record_custom_metric(*args, **kwargs):
    """Record custom metric using global dashboard"""
    performance_dashboard.record_custom_metric(*args, **kwargs)