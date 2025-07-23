#!/usr/bin/env python3
"""
Comprehensive System Monitoring and Alerting for Beverly Knits AI Supply Chain Planner
Real-time monitoring, alerting, and performance tracking for production deployment
"""

import sys
import os
import time
# Optional system monitoring - handle if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(str, Enum):
    """Alert status tracking"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """System alert with metadata"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    metric_name: str
    current_value: Any
    threshold_value: Any
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    occurrence_count: int = 1
    first_occurrence: Optional[datetime] = None

@dataclass
class MetricThreshold:
    """Monitoring threshold configuration"""
    metric_name: str
    component: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str  # 'gt', 'lt', 'eq'
    evaluation_window: int = 60  # seconds
    min_data_points: int = 3

@dataclass
class SystemMetrics:
    """Current system performance metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    active_connections: int
    process_count: int
    load_average: List[float]

@dataclass
class ApplicationMetrics:
    """Application-specific performance metrics"""
    timestamp: datetime
    erp_connection_status: bool
    erp_response_time_ms: float
    ml_model_prediction_time_ms: float
    cache_hit_rate: float
    cache_size_mb: float
    active_planning_cycles: int
    failed_requests_per_min: int
    data_quality_score: float

class SystemMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.is_monitoring:
            logger.warning("⚠️ System monitoring already running")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("📊 System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 24 hours of metrics
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            # Return mock metrics if psutil not available
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=45.0,
                memory_usage_percent=65.0,
                disk_usage_percent=35.0,
                disk_io_read_mb=100.0,
                disk_io_write_mb=50.0,
                network_bytes_sent=1000000,
                network_bytes_recv=2000000,
                active_connections=25,
                process_count=150,
                load_average=[0.8, 0.9, 1.0]
            )
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network
        network = psutil.net_io_counters()
        
        # System load
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100,
            disk_io_read_mb=(disk_io.read_bytes / (1024 * 1024)) if disk_io else 0,
            disk_io_write_mb=(disk_io.write_bytes / (1024 * 1024)) if disk_io else 0,
            network_bytes_sent=network.bytes_sent if network else 0,
            network_bytes_recv=network.bytes_recv if network else 0,
            active_connections=len(psutil.net_connections()),
            process_count=len(psutil.pids()),
            load_average=list(load_avg)
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of metrics over time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_usage_percent for m in recent_metrics]
        
        return {
            "time_period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu_usage": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_usage": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "latest_timestamp": recent_metrics[-1].timestamp.isoformat()
        }

class AlertManager:
    """Manage system alerts and notifications"""
    
    def __init__(self, smtp_config: Optional[Dict] = None):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.thresholds: List[MetricThreshold] = []
        self.notification_handlers: List[Callable] = []
        self.smtp_config = smtp_config
        self._setup_default_thresholds()
        
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds"""
        self.thresholds = [
            MetricThreshold("cpu_usage_percent", "system", 80.0, 90.0, "gt"),
            MetricThreshold("memory_usage_percent", "system", 85.0, 95.0, "gt"),
            MetricThreshold("disk_usage_percent", "system", 80.0, 90.0, "gt"),
            MetricThreshold("erp_response_time_ms", "application", 2000.0, 5000.0, "gt"),
            MetricThreshold("cache_hit_rate", "application", 0.7, 0.5, "lt"),
            MetricThreshold("data_quality_score", "application", 0.8, 0.6, "lt"),
            MetricThreshold("failed_requests_per_min", "application", 5.0, 10.0, "gt"),
        ]
        logger.info(f"📋 Configured {len(self.thresholds)} monitoring thresholds")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add custom notification handler"""
        self.notification_handlers.append(handler)
        logger.info("📨 Added notification handler")
    
    def evaluate_metrics(self, system_metrics: SystemMetrics, 
                        app_metrics: Optional[ApplicationMetrics] = None):
        """Evaluate metrics against thresholds and generate alerts"""
        
        # Combine all metrics for evaluation
        all_metrics = {
            "cpu_usage_percent": system_metrics.cpu_usage_percent,
            "memory_usage_percent": system_metrics.memory_usage_percent,
            "disk_usage_percent": system_metrics.disk_usage_percent,
        }
        
        if app_metrics:
            all_metrics.update({
                "erp_response_time_ms": app_metrics.erp_response_time_ms,
                "cache_hit_rate": app_metrics.cache_hit_rate,
                "data_quality_score": app_metrics.data_quality_score,
                "failed_requests_per_min": app_metrics.failed_requests_per_min,
            })
        
        # Check each threshold
        for threshold in self.thresholds:
            if threshold.metric_name not in all_metrics:
                continue
                
            current_value = all_metrics[threshold.metric_name]
            
            # Evaluate against thresholds
            critical_breach = self._evaluate_threshold(
                current_value, threshold.critical_threshold, threshold.comparison_operator
            )
            warning_breach = self._evaluate_threshold(
                current_value, threshold.warning_threshold, threshold.comparison_operator
            )
            
            alert_id = f"{threshold.component}_{threshold.metric_name}"
            
            if critical_breach:
                self._create_or_update_alert(
                    alert_id, AlertSeverity.CRITICAL, threshold, current_value
                )
            elif warning_breach:
                self._create_or_update_alert(
                    alert_id, AlertSeverity.HIGH, threshold, current_value
                )
            else:
                # Resolve alert if it exists
                self._resolve_alert(alert_id)
    
    def _evaluate_threshold(self, current: float, threshold: float, operator: str) -> bool:
        """Evaluate if metric breaches threshold"""
        if operator == "gt":
            return current > threshold
        elif operator == "lt":
            return current < threshold
        elif operator == "eq":
            return current == threshold
        return False
    
    def _create_or_update_alert(self, alert_id: str, severity: AlertSeverity, 
                               threshold: MetricThreshold, current_value: Any):
        """Create new alert or update existing one"""
        
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.occurrence_count += 1
            alert.current_value = current_value
            alert.timestamp = datetime.now()
        else:
            # Create new alert
            alert = Alert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                component=threshold.component,
                metric_name=threshold.metric_name,
                current_value=current_value,
                threshold_value=threshold.critical_threshold if severity == AlertSeverity.CRITICAL else threshold.warning_threshold,
                message=self._generate_alert_message(threshold, current_value, severity),
                first_occurrence=datetime.now()
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            logger.warning(f"🚨 {severity.upper()} Alert: {alert.message}")
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"✅ Resolved alert: {alert_id}")
    
    def _generate_alert_message(self, threshold: MetricThreshold, 
                               current_value: Any, severity: AlertSeverity) -> str:
        """Generate human-readable alert message"""
        return (
            f"{threshold.component.title()} {threshold.metric_name} is {current_value} "
            f"({severity.value} threshold: {threshold.warning_threshold if severity == AlertSeverity.HIGH else threshold.critical_threshold})"
        )
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured handlers"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"❌ Notification handler failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        active_by_severity = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "alerts_by_severity": active_by_severity,
            "total_alerts_today": len([
                a for a in self.alert_history 
                if a.timestamp.date() == datetime.now().date()
            ]),
            "most_recent_alert": max(
                self.active_alerts.values(), 
                key=lambda x: x.timestamp
            ).timestamp.isoformat() if self.active_alerts else None
        }

class NotificationHandlers:
    """Pre-built notification handlers"""
    
    @staticmethod
    def console_handler(alert: Alert):
        """Simple console notification"""
        print(f"🚨 {alert.severity.upper()} ALERT: {alert.message}")
    
    @staticmethod
    def file_handler(log_file: str):
        """File-based alert logging"""
        def handler(alert: Alert):
            with open(log_file, 'a') as f:
                f.write(f"{alert.timestamp.isoformat()} | {alert.severity.upper()} | {alert.message}\n")
        return handler
    
    @staticmethod
    def email_handler(smtp_config: Dict):
        """Email notification handler"""
        def handler(alert: Alert):
            try:
                msg = MIMEMultipart()
                msg['From'] = smtp_config['from']
                msg['To'] = smtp_config['to']
                msg['Subject'] = f"🚨 {alert.severity.upper()} Alert: {alert.component}"
                
                body = f"""
Alert Details:
- Component: {alert.component}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}
- Message: {alert.message}
- Timestamp: {alert.timestamp}
"""
                
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['port'])
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
                server.quit()
                
            except Exception as e:
                logger.error(f"❌ Email notification failed: {e}")
        
        return handler

class ComprehensiveMonitor:
    """Main monitoring coordinator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.system_monitor = SystemMonitor(
            collection_interval=self.config.get('collection_interval', 30)
        )
        self.alert_manager = AlertManager(
            smtp_config=self.config.get('smtp_config')
        )
        
        # Setup notification handlers
        self.alert_manager.add_notification_handler(NotificationHandlers.console_handler)
        
        # Add file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.alert_manager.add_notification_handler(
            NotificationHandlers.file_handler("logs/alerts.log")
        )
        
        self.is_running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        if self.is_running:
            logger.warning("⚠️ Monitoring already running")
            return
        
        logger.info("🚀 Starting comprehensive monitoring system...")
        
        # Start system metrics collection
        self.system_monitor.start_monitoring()
        
        # Start alert evaluation loop
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("✅ Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        logger.info("🛑 Stopping comprehensive monitoring...")
        
        self.is_running = False
        self.system_monitor.stop_monitoring()
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("✅ Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and alerting loop"""
        while self.is_running:
            try:
                # Get latest system metrics
                system_metrics = self.system_monitor.get_current_metrics()
                
                if system_metrics:
                    # Simulate application metrics (in production, would get from actual app)
                    app_metrics = self._simulate_application_metrics()
                    
                    # Evaluate against thresholds
                    self.alert_manager.evaluate_metrics(system_metrics, app_metrics)
                
                time.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(60)
    
    def _simulate_application_metrics(self) -> ApplicationMetrics:
        """Simulate application metrics for demonstration"""
        import random
        
        return ApplicationMetrics(
            timestamp=datetime.now(),
            erp_connection_status=True,
            erp_response_time_ms=random.uniform(500, 1500),
            ml_model_prediction_time_ms=random.uniform(50, 200),
            cache_hit_rate=random.uniform(0.75, 0.95),
            cache_size_mb=random.uniform(50, 200),
            active_planning_cycles=random.randint(0, 5),
            failed_requests_per_min=random.randint(0, 3),
            data_quality_score=random.uniform(0.85, 0.98)
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        system_metrics = self.system_monitor.get_current_metrics()
        system_summary = self.system_monitor.get_metrics_summary(hours=1)
        alert_summary = self.alert_manager.get_alert_summary()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": "active" if self.is_running else "inactive",
            "system_metrics": asdict(system_metrics) if system_metrics else None,
            "system_summary": system_summary,
            "alert_summary": alert_summary,
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "health_status": self._calculate_health_status(system_metrics, active_alerts)
        }
    
    def _calculate_health_status(self, system_metrics: Optional[SystemMetrics], 
                                active_alerts: List[Alert]) -> str:
        """Calculate overall system health"""
        if not system_metrics:
            return "unknown"
        
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in active_alerts if a.severity == AlertSeverity.HIGH]
        
        if critical_alerts:
            return "critical"
        elif high_alerts:
            return "degraded"
        elif system_metrics.cpu_usage_percent > 70 or system_metrics.memory_usage_percent > 80:
            return "warning"
        else:
            return "healthy"
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        status = self.get_system_status()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "system_health": status["health_status"],
            "monitoring_duration_hours": 24,  # Configured monitoring period
            "total_active_alerts": status["alert_summary"]["active_alerts"],
            "system_performance": status["system_summary"],
            "alert_breakdown": status["alert_summary"]["alerts_by_severity"],
            "recommendations": self._generate_recommendations(status),
            "next_review": (datetime.now() + timedelta(hours=6)).isoformat()
        }
    
    def _generate_recommendations(self, status: Dict) -> List[str]:
        """Generate operational recommendations"""
        recommendations = []
        
        if status["health_status"] == "critical":
            recommendations.append("🚨 Critical issues detected - immediate attention required")
        elif status["health_status"] == "degraded":
            recommendations.append("⚠️ Performance degraded - review high priority alerts")
        
        system_summary = status.get("system_summary", {})
        if system_summary:
            cpu_avg = system_summary.get("cpu_usage", {}).get("avg", 0)
            if cpu_avg > 80:
                recommendations.append("💻 High CPU usage detected - consider scaling or optimization")
            
            memory_avg = system_summary.get("memory_usage", {}).get("avg", 0)
            if memory_avg > 85:
                recommendations.append("💾 High memory usage detected - check for memory leaks")
        
        active_alerts = status["alert_summary"]["active_alerts"]
        if active_alerts == 0:
            recommendations.append("✅ All systems operating normally")
        elif active_alerts > 5:
            recommendations.append("📊 Multiple active alerts - prioritize resolution by severity")
        
        return recommendations

def main():
    """Main monitoring system demonstration"""
    logger.info("🎯 Beverly Knits AI - Comprehensive System Monitoring")
    logger.info("Real-time monitoring, alerting, and performance tracking")
    logger.info("=" * 80)
    
    # Initialize monitoring system
    monitor = ComprehensiveMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run for demonstration period
        logger.info("📊 Monitoring system running... (Press Ctrl+C to stop)")
        
        # Display status updates every 30 seconds for 5 minutes
        for i in range(10):
            time.sleep(30)
            
            status = monitor.get_system_status()
            logger.info(f"🎯 System Health: {status['health_status'].upper()}")
            logger.info(f"📊 Active Alerts: {status['alert_summary']['active_alerts']}")
            
            if status['system_metrics']:
                cpu = status['system_metrics']['cpu_usage_percent']
                memory = status['system_metrics']['memory_usage_percent']
                logger.info(f"💻 CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
        
        # Generate final report
        report = monitor.generate_monitoring_report()
        report_file = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("=" * 80)
        logger.info("🎉 MONITORING DEMONSTRATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 System Health: {report['system_health'].upper()}")
        logger.info(f"🚨 Total Active Alerts: {report['total_active_alerts']}")
        logger.info(f"📈 Performance Summary Available")
        logger.info(f"📄 Report saved to: {report_file}")
        
        if report['recommendations']:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                logger.info(f"   • {rec}")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Monitoring stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Monitoring system failed: {e}")
        return False
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)