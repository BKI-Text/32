"""
Comprehensive Performance Monitoring System for Beverly Knits AI Supply Chain Planner

This module provides real-time performance monitoring, metrics collection,
and alerting capabilities for production systems.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import asyncio
from pathlib import Path
import psutil
import gc
from functools import wraps
import statistics

from ..utils.error_handling import handle_errors, ErrorCategory

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

@dataclass
class PerformanceAlert:
    """Performance alert"""
    metric_name: str
    alert_type: str  # 'warning', 'critical', 'recovery'
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    severity: str
    affected_components: List[str] = field(default_factory=list)

@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str  # 'healthy', 'warning', 'critical'
    score: float  # 0-100
    components: Dict[str, str]
    alerts: List[PerformanceAlert]
    last_updated: datetime
    uptime: float
    performance_summary: Dict[str, Any]

class PerformanceCollector:
    """Collects system and application performance metrics"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.is_collecting = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start background metrics collection"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("Performance collection started")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Performance collection stopped")
    
    def _collect_metrics_loop(self):
        """Background thread for collecting metrics"""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_history['cpu_usage'].append(
            PerformanceMetric('cpu_usage', cpu_percent, 'percent', timestamp, 'system')
        )
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_history['memory_usage'].append(
            PerformanceMetric('memory_usage', memory.percent, 'percent', timestamp, 'system')
        )
        
        self.metrics_history['memory_available'].append(
            PerformanceMetric('memory_available', memory.available / (1024**3), 'GB', timestamp, 'system')
        )
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics_history['disk_usage'].append(
            PerformanceMetric('disk_usage', disk.percent, 'percent', timestamp, 'system')
        )
        
        # Network metrics
        network = psutil.net_io_counters()
        self.metrics_history['network_bytes_sent'].append(
            PerformanceMetric('network_bytes_sent', network.bytes_sent, 'bytes', timestamp, 'system')
        )
        
        self.metrics_history['network_bytes_recv'].append(
            PerformanceMetric('network_bytes_recv', network.bytes_recv, 'bytes', timestamp, 'system')
        )
        
        # Process metrics
        process = psutil.Process()
        self.metrics_history['process_memory'].append(
            PerformanceMetric('process_memory', process.memory_info().rss / (1024**2), 'MB', timestamp, 'application')
        )
        
        self.metrics_history['process_cpu'].append(
            PerformanceMetric('process_cpu', process.cpu_percent(), 'percent', timestamp, 'application')
        )
        
        # Garbage collection metrics
        gc_stats = gc.get_stats()
        if gc_stats:
            self.metrics_history['gc_collections'].append(
                PerformanceMetric('gc_collections', sum(stat['collections'] for stat in gc_stats), 'count', timestamp, 'application')
            )
    
    def get_latest_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get the latest metric values"""
        latest_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_metrics[metric_name] = history[-1]
        return latest_metrics
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[PerformanceMetric]:
        """Get metric history for specified duration"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [metric for metric in self.metrics_history[metric_name] if metric.timestamp > cutoff_time]

class ApplicationMetricsCollector:
    """Collects application-specific metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = defaultdict(float)
        
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.counters[name] += value
        self.metrics[name].append(
            PerformanceMetric(name, self.counters[name], 'count', datetime.now(), 'counter', tags or {})
        )
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        self.timers[name].append(duration)
        self.metrics[name].append(
            PerformanceMetric(name, duration, 'seconds', datetime.now(), 'timer', tags or {})
        )
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        self.metrics[name].append(
            PerformanceMetric(name, value, 'value', datetime.now(), 'gauge', tags or {})
        )
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timer metric"""
        if name not in self.timers or not self.timers[name]:
            return {}
        
        values = self.timers[name]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, 
                 metrics_path: str = "logs/metrics/",
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 enable_alerts: bool = True):
        self.metrics_path = Path(metrics_path)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
        self.system_collector = PerformanceCollector()
        self.app_collector = ApplicationMetricsCollector()
        
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        self.enable_alerts = enable_alerts
        self.active_alerts = []
        self.alert_callbacks = []
        
        self.monitoring_start_time = datetime.now()
        self.is_monitoring = False
        
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default alerting thresholds"""
        return {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'disk_usage': {'warning': 80.0, 'critical': 95.0},
            'process_memory': {'warning': 1024.0, 'critical': 2048.0},  # MB
            'response_time': {'warning': 2.0, 'critical': 5.0},  # seconds
            'error_rate': {'warning': 0.05, 'critical': 0.10},  # 5% and 10%
            'throughput': {'warning': 100.0, 'critical': 50.0}  # requests/min (lower is worse)
        }
    
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            self.system_collector.start_collection()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            self.system_collector.stop_collection()
            logger.info("Performance monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def record_request(self, endpoint: str, method: str, duration: float, status_code: int):
        """Record HTTP request metrics"""
        tags = {'endpoint': endpoint, 'method': method, 'status': str(status_code)}
        
        # Record timing
        self.app_collector.record_timer('request_duration', duration, tags)
        
        # Increment request counter
        self.app_collector.increment_counter('requests_total', 1, tags)
        
        # Record error if status code indicates error
        if status_code >= 400:
            self.app_collector.increment_counter('requests_errors', 1, tags)
        
        # Check for alerts
        if self.enable_alerts:
            self._check_request_alerts(endpoint, duration, status_code)
    
    def record_database_query(self, query_type: str, duration: float, success: bool):
        """Record database query metrics"""
        tags = {'query_type': query_type, 'success': str(success)}
        
        self.app_collector.record_timer('db_query_duration', duration, tags)
        self.app_collector.increment_counter('db_queries_total', 1, tags)
        
        if not success:
            self.app_collector.increment_counter('db_queries_errors', 1, tags)
    
    def record_ml_inference(self, model_name: str, duration: float, success: bool):
        """Record ML inference metrics"""
        tags = {'model': model_name, 'success': str(success)}
        
        self.app_collector.record_timer('ml_inference_duration', duration, tags)
        self.app_collector.increment_counter('ml_inferences_total', 1, tags)
        
        if not success:
            self.app_collector.increment_counter('ml_inference_errors', 1, tags)
    
    def record_planning_cycle(self, duration: float, items_processed: int, success: bool):
        """Record planning cycle metrics"""
        tags = {'success': str(success)}
        
        self.app_collector.record_timer('planning_cycle_duration', duration, tags)
        self.app_collector.set_gauge('planning_items_processed', items_processed, tags)
        self.app_collector.increment_counter('planning_cycles_total', 1, tags)
        
        if not success:
            self.app_collector.increment_counter('planning_cycles_errors', 1, tags)
    
    def _check_request_alerts(self, endpoint: str, duration: float, status_code: int):
        """Check for request-related alerts"""
        # Check response time
        if 'response_time' in self.alert_thresholds:
            thresholds = self.alert_thresholds['response_time']
            if duration > thresholds.get('critical', float('inf')):
                self._trigger_alert('response_time', 'critical', duration, thresholds['critical'], 
                                  f"Critical response time on {endpoint}: {duration:.2f}s")
            elif duration > thresholds.get('warning', float('inf')):
                self._trigger_alert('response_time', 'warning', duration, thresholds['warning'],
                                  f"Slow response time on {endpoint}: {duration:.2f}s")
        
        # Check error rate
        if status_code >= 400:
            error_rate = self._calculate_error_rate()
            if 'error_rate' in self.alert_thresholds:
                thresholds = self.alert_thresholds['error_rate']
                if error_rate > thresholds.get('critical', float('inf')):
                    self._trigger_alert('error_rate', 'critical', error_rate, thresholds['critical'],
                                      f"Critical error rate: {error_rate:.2%}")
                elif error_rate > thresholds.get('warning', float('inf')):
                    self._trigger_alert('error_rate', 'warning', error_rate, thresholds['warning'],
                                      f"High error rate: {error_rate:.2%}")
    
    def _calculate_error_rate(self, window_minutes: int = 5) -> float:
        """Calculate error rate over a time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        total_requests = 0
        error_requests = 0
        
        if 'requests_total' in self.app_collector.metrics:
            for metric in self.app_collector.metrics['requests_total']:
                if metric.timestamp > cutoff_time:
                    total_requests += metric.value
        
        if 'requests_errors' in self.app_collector.metrics:
            for metric in self.app_collector.metrics['requests_errors']:
                if metric.timestamp > cutoff_time:
                    error_requests += metric.value
        
        return error_requests / total_requests if total_requests > 0 else 0.0
    
    def _trigger_alert(self, metric_name: str, alert_type: str, current_value: float, 
                      threshold_value: float, message: str):
        """Trigger a performance alert"""
        alert = PerformanceAlert(
            metric_name=metric_name,
            alert_type=alert_type,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=datetime.now(),
            severity=alert_type
        )
        
        self.active_alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Performance alert: {message}")
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""
        latest_metrics = self.system_collector.get_latest_metrics()
        
        # Calculate health score
        health_score = 100.0
        component_status = {}
        
        # Check system metrics
        for metric_name, metric in latest_metrics.items():
            if metric_name in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric_name]
                
                if metric.value > thresholds.get('critical', float('inf')):
                    component_status[metric_name] = 'critical'
                    health_score -= 25
                elif metric.value > thresholds.get('warning', float('inf')):
                    component_status[metric_name] = 'warning'
                    health_score -= 10
                else:
                    component_status[metric_name] = 'healthy'
        
        # Check application metrics
        error_rate = self._calculate_error_rate()
        if error_rate > 0.1:
            component_status['error_rate'] = 'critical'
            health_score -= 20
        elif error_rate > 0.05:
            component_status['error_rate'] = 'warning'
            health_score -= 5
        else:
            component_status['error_rate'] = 'healthy'
        
        # Determine overall status
        if health_score >= 80:
            overall_status = 'healthy'
        elif health_score >= 60:
            overall_status = 'warning'
        else:
            overall_status = 'critical'
        
        # Calculate uptime
        uptime = (datetime.now() - self.monitoring_start_time).total_seconds()
        
        # Performance summary
        performance_summary = {
            'avg_response_time': self._get_avg_response_time(),
            'requests_per_minute': self._get_requests_per_minute(),
            'error_rate': error_rate,
            'active_alerts': len(self.active_alerts)
        }
        
        return SystemHealth(
            status=overall_status,
            score=max(0, health_score),
            components=component_status,
            alerts=self.active_alerts[-10:],  # Last 10 alerts
            last_updated=datetime.now(),
            uptime=uptime,
            performance_summary=performance_summary
        )
    
    def _get_avg_response_time(self, window_minutes: int = 5) -> float:
        """Get average response time over a time window"""
        if 'request_duration' not in self.app_collector.timers:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_times = []
        
        if 'request_duration' in self.app_collector.metrics:
            for metric in self.app_collector.metrics['request_duration']:
                if metric.timestamp > cutoff_time:
                    recent_times.append(metric.value)
        
        return statistics.mean(recent_times) if recent_times else 0.0
    
    def _get_requests_per_minute(self, window_minutes: int = 5) -> float:
        """Get requests per minute over a time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        request_count = 0
        
        if 'requests_total' in self.app_collector.metrics:
            for metric in self.app_collector.metrics['requests_total']:
                if metric.timestamp > cutoff_time:
                    request_count += 1
        
        return request_count / window_minutes if window_minutes > 0 else 0.0
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        if format == 'json':
            return self._export_json()
        elif format == 'prometheus':
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {},
            'application_metrics': {},
            'health': self.get_system_health().__dict__
        }
        
        # System metrics
        for metric_name, metric in self.system_collector.get_latest_metrics().items():
            data['system_metrics'][metric_name] = {
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp.isoformat()
            }
        
        # Application metrics
        for metric_name, counter_value in self.app_collector.counters.items():
            data['application_metrics'][metric_name] = {
                'value': counter_value,
                'type': 'counter'
            }
        
        for metric_name, gauge_value in self.app_collector.gauges.items():
            data['application_metrics'][metric_name] = {
                'value': gauge_value,
                'type': 'gauge'
            }
        
        for metric_name, timer_values in self.app_collector.timers.items():
            if timer_values:
                stats = self.app_collector.get_timer_stats(metric_name)
                data['application_metrics'][metric_name] = {
                    'type': 'timer',
                    'stats': stats
                }
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # System metrics
        for metric_name, metric in self.system_collector.get_latest_metrics().items():
            lines.append(f"# HELP {metric_name} {metric.category} metric")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {metric.value}")
        
        # Application counters
        for metric_name, value in self.app_collector.counters.items():
            lines.append(f"# HELP {metric_name} Application counter")
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")
        
        # Application gauges
        for metric_name, value in self.app_collector.gauges.items():
            lines.append(f"# HELP {metric_name} Application gauge")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")
        
        return '\n'.join(lines)

def performance_timer(monitor: PerformanceMonitor, metric_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                monitor.app_collector.record_timer(metric_name, duration, {'success': str(success)})
        return wrapper
    return decorator

def performance_counter(monitor: PerformanceMonitor, metric_name: str):
    """Decorator to count function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                monitor.app_collector.increment_counter(metric_name, 1, {'success': str(success)})
        return wrapper
    return decorator

# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return global_performance_monitor