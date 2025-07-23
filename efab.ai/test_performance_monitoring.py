#!/usr/bin/env python3
"""
Test Real-time Performance Monitoring Dashboard
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import sys
import os
import logging
import tempfile
import shutil
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.monitoring import (
    PerformanceMetric,
    ModelPerformanceData,
    SystemMetrics,
    MetricsCollector,
    AlertSystem,
    PerformanceDashboard,
    start_monitoring,
    stop_monitoring,
    record_model_performance,
    record_custom_metric
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitoringTester:
    """Test performance monitoring system"""
    
    def __init__(self):
        self.temp_dir = None
        self.dashboard = None
        
    def setup_test_environment(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(self.temp_dir, "test_performance.db")
        
        self.dashboard = PerformanceDashboard(db_path=db_path)
        logger.info(f"Test environment setup at: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.dashboard:
            self.dashboard.stop_monitoring()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
    
    def test_metrics_collector(self) -> bool:
        """Test metrics collector"""
        logger.info("🚀 Testing metrics collector")
        
        try:
            collector = MetricsCollector()
            
            # Start collection
            collector.start_collection()
            
            # Add some test metrics
            test_metric = ModelPerformanceData(
                model_id="test_model_1",
                model_name="Test Model",
                timestamp=datetime.now(),
                prediction_latency=0.15,
                throughput=100.0,
                accuracy=0.85,
                error_rate=0.05,
                prediction_count=1000,
                error_count=50
            )
            
            collector.add_model_metric(test_metric)
            
            # Wait for processing
            time.sleep(2)
            
            # Check metrics
            metrics = collector.get_model_metrics("test_model_1", minutes=5)
            
            if not metrics:
                logger.error("No metrics found after adding")
                return False
            
            if len(metrics) != 1:
                logger.error(f"Expected 1 metric, got {len(metrics)}")
                return False
            
            retrieved_metric = metrics[0]
            if retrieved_metric.model_id != "test_model_1":
                logger.error("Retrieved metric has wrong model ID")
                return False
            
            # Test system metrics
            system_metrics = collector.get_system_metrics(minutes=5)
            if not system_metrics:
                logger.error("No system metrics collected")
                return False
            
            # Stop collection
            collector.stop_collection()
            
            logger.info("✅ Metrics collector test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Metrics collector test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_alert_system(self) -> bool:
        """Test alert system"""
        logger.info("🚀 Testing alert system")
        
        try:
            alert_system = AlertSystem()
            
            # Add test alert rule
            alert_system.add_alert_rule(
                'test_high_cpu',
                'cpu_percent',
                '>',
                50.0,
                duration_minutes=1
            )
            
            # Create mock metrics collector with high CPU
            collector = MetricsCollector()
            
            # Add high CPU metrics
            high_cpu_metric = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=75.0,
                memory_percent=50.0,
                disk_usage=30.0,
                network_io={},
                active_connections=10,
                response_time=0.1,
                error_rate=0.0
            )
            
            collector.system_metrics.append(high_cpu_metric)
            
            # Check alerts
            alert_system.check_alerts(collector)
            
            # Verify alert was triggered
            active_alerts = alert_system.get_active_alerts()
            if not active_alerts:
                logger.error("No alerts triggered for high CPU")
                return False
            
            if len(active_alerts) != 1:
                logger.error(f"Expected 1 alert, got {len(active_alerts)}")
                return False
            
            alert = active_alerts[0]
            if alert['rule_name'] != 'test_high_cpu':
                logger.error("Wrong alert rule triggered")
                return False
            
            # Add normal CPU metric
            normal_cpu_metric = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=30.0,
                memory_percent=50.0,
                disk_usage=30.0,
                network_io={},
                active_connections=10,
                response_time=0.1,
                error_rate=0.0
            )
            
            collector.system_metrics.append(normal_cpu_metric)
            
            # Check alerts again
            alert_system.check_alerts(collector)
            
            # Verify alert was resolved
            active_alerts = alert_system.get_active_alerts()
            if active_alerts:
                logger.error("Alert should have been resolved")
                return False
            
            logger.info("✅ Alert system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Alert system test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_performance_dashboard(self) -> bool:
        """Test performance dashboard"""
        logger.info("🚀 Testing performance dashboard")
        
        try:
            # Start monitoring
            self.dashboard.start_monitoring()
            
            # Record test metrics
            for i in range(5):
                self.dashboard.record_model_performance(
                    model_id=f"test_model_{i%2}",
                    model_name=f"Test Model {i%2}",
                    prediction_latency=0.1 + i * 0.02,
                    throughput=100.0 + i * 10,
                    accuracy=0.8 + i * 0.02,
                    error_rate=0.01 + i * 0.005,
                    prediction_count=100 + i * 50,
                    error_count=1 + i
                )
                
                time.sleep(0.1)  # Small delay between metrics
            
            # Record custom metrics
            for i in range(3):
                self.dashboard.record_custom_metric(
                    metric_name="custom_test_metric",
                    metric_value=50.0 + i * 10,
                    metric_type="gauge",
                    labels={"source": "test", "iteration": str(i)}
                )
            
            # Wait for processing
            time.sleep(2)
            
            # Get dashboard data
            dashboard_data = self.dashboard.get_dashboard_data(hours=1)
            
            # Verify dashboard data structure
            required_keys = ['timestamp', 'summary', 'model_metrics', 'system_metrics', 'monitoring_status']
            for key in required_keys:
                if key not in dashboard_data:
                    logger.error(f"Dashboard data missing key: {key}")
                    return False
            
            # Check model metrics
            model_metrics = dashboard_data['model_metrics']
            if not model_metrics:
                logger.error("No model metrics in dashboard data")
                return False
            
            expected_models = ['test_model_0', 'test_model_1']
            for model_id in expected_models:
                if model_id not in model_metrics:
                    logger.error(f"Model {model_id} not found in dashboard data")
                    return False
                
                if not model_metrics[model_id]:
                    logger.error(f"No metrics for model {model_id}")
                    return False
            
            # Check system metrics
            system_metrics = dashboard_data['system_metrics']
            if not system_metrics:
                logger.error("No system metrics in dashboard data")
                return False
            
            # Check monitoring status
            monitoring_status = dashboard_data['monitoring_status']
            if not monitoring_status.get('collector_running', False):
                logger.error("Collector should be running")
                return False
            
            # Test model performance report
            report = self.dashboard.get_model_performance_report('test_model_0', hours=1)
            
            if 'error' in report:
                logger.error(f"Error in model performance report: {report['error']}")
                return False
            
            if report['model_id'] != 'test_model_0':
                logger.error("Wrong model ID in performance report")
                return False
            
            if 'performance_stats' not in report:
                logger.error("Performance stats missing from report")
                return False
            
            logger.info("✅ Performance dashboard test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance dashboard test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_dashboard_data_export(self) -> bool:
        """Test dashboard data export"""
        logger.info("🚀 Testing dashboard data export")
        
        try:
            # Record some test data
            for i in range(3):
                self.dashboard.record_model_performance(
                    model_id="export_test_model",
                    model_name="Export Test Model",
                    prediction_latency=0.1 + i * 0.01,
                    throughput=100.0 + i * 5,
                    accuracy=0.85 + i * 0.01,
                    error_rate=0.02,
                    prediction_count=100,
                    error_count=2
                )
            
            # Export to file
            export_file = os.path.join(self.temp_dir, "test_export.json")
            self.dashboard.export_metrics(export_file, hours=1)
            
            # Verify file was created
            if not os.path.exists(export_file):
                logger.error("Export file was not created")
                return False
            
            # Verify file has content
            with open(export_file, 'r') as f:
                import json
                data = json.load(f)
            
            if not data:
                logger.error("Export file is empty")
                return False
            
            if 'model_metrics' not in data:
                logger.error("Export file missing model_metrics")
                return False
            
            if 'export_test_model' not in data['model_metrics']:
                logger.error("Export test model not found in export")
                return False
            
            logger.info("✅ Dashboard data export test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dashboard data export test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_data_cleanup(self) -> bool:
        """Test data cleanup functionality"""
        logger.info("🚀 Testing data cleanup")
        
        try:
            # Record some old data (simulate by directly inserting into database)
            import sqlite3
            old_timestamp = (datetime.now() - timedelta(days=45)).isoformat()
            
            with sqlite3.connect(self.dashboard.db_path) as conn:
                # Insert old model metric
                conn.execute('''
                    INSERT INTO model_metrics (
                        model_id, model_name, timestamp, prediction_latency, throughput,
                        accuracy, error_rate, memory_usage, cpu_usage, prediction_count, error_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'old_model',
                    'Old Model',
                    old_timestamp,
                    0.1, 100.0, 0.8, 0.01, 0.0, 0.0, 100, 1
                ))
                
                # Insert old system metric
                conn.execute('''
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, disk_usage, network_io,
                        active_connections, response_time, error_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    old_timestamp, 50.0, 60.0, 70.0, '{}', 10, 0.1, 0.0
                ))
            
            # Record recent data
            self.dashboard.record_model_performance(
                model_id="recent_model",
                model_name="Recent Model",
                prediction_latency=0.1,
                throughput=100.0,
                accuracy=0.85,
                error_rate=0.01,
                prediction_count=100,
                error_count=1
            )
            
            # Count records before cleanup
            with sqlite3.connect(self.dashboard.db_path) as conn:
                model_count_before = conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]
                system_count_before = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
            
            # Cleanup old data (keep last 30 days)
            self.dashboard.cleanup_old_data(days_to_keep=30)
            
            # Count records after cleanup
            with sqlite3.connect(self.dashboard.db_path) as conn:
                model_count_after = conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]
                system_count_after = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
            
            # Verify cleanup worked
            if model_count_after >= model_count_before:
                logger.error("Model metrics cleanup did not work")
                return False
            
            if system_count_after >= system_count_before:
                logger.error("System metrics cleanup did not work")
                return False
            
            # Verify recent data is still there
            with sqlite3.connect(self.dashboard.db_path) as conn:
                recent_model = conn.execute(
                    "SELECT * FROM model_metrics WHERE model_id = 'recent_model'"
                ).fetchone()
                
                if not recent_model:
                    logger.error("Recent model data was incorrectly deleted")
                    return False
                
                old_model = conn.execute(
                    "SELECT * FROM model_metrics WHERE model_id = 'old_model'"
                ).fetchone()
                
                if old_model:
                    logger.error("Old model data was not deleted")
                    return False
            
            logger.info("✅ Data cleanup test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data cleanup test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_concurrent_metrics_recording(self) -> bool:
        """Test concurrent metrics recording"""
        logger.info("🚀 Testing concurrent metrics recording")
        
        try:
            # Start monitoring
            self.dashboard.start_monitoring()
            
            # Function to record metrics in thread
            def record_metrics_thread(thread_id: int, num_metrics: int):
                for i in range(num_metrics):
                    self.dashboard.record_model_performance(
                        model_id=f"thread_model_{thread_id}",
                        model_name=f"Thread Model {thread_id}",
                        prediction_latency=0.1 + i * 0.01,
                        throughput=100.0 + i * 5,
                        accuracy=0.8 + i * 0.01,
                        error_rate=0.01,
                        prediction_count=10,
                        error_count=0
                    )
                    time.sleep(0.05)  # Small delay
            
            # Create multiple threads
            threads = []
            num_threads = 3
            metrics_per_thread = 5
            
            for i in range(num_threads):
                thread = threading.Thread(
                    target=record_metrics_thread,
                    args=(i, metrics_per_thread)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Wait for processing
            time.sleep(2)
            
            # Verify all metrics were recorded
            dashboard_data = self.dashboard.get_dashboard_data(hours=1)
            model_metrics = dashboard_data['model_metrics']
            
            expected_models = [f"thread_model_{i}" for i in range(num_threads)]
            
            for model_id in expected_models:
                if model_id not in model_metrics:
                    logger.error(f"Model {model_id} not found in dashboard data")
                    return False
                
                if len(model_metrics[model_id]) != metrics_per_thread:
                    logger.error(f"Expected {metrics_per_thread} metrics for {model_id}, got {len(model_metrics[model_id])}")
                    return False
            
            logger.info("✅ Concurrent metrics recording test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Concurrent metrics recording test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """Run all monitoring tests"""
        logger.info("🚀 Running Performance Monitoring Tests")
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            tests = [
                ("Metrics Collector", self.test_metrics_collector),
                ("Alert System", self.test_alert_system),
                ("Performance Dashboard", self.test_performance_dashboard),
                ("Dashboard Data Export", self.test_dashboard_data_export),
                ("Data Cleanup", self.test_data_cleanup),
                ("Concurrent Metrics Recording", self.test_concurrent_metrics_recording)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Running test: {test_name}")
                    logger.info(f"{'='*60}")
                    
                    result = test_func()
                    
                    if result:
                        passed += 1
                        logger.info(f"✅ {test_name}: PASSED")
                    else:
                        logger.error(f"❌ {test_name}: FAILED")
                        
                except Exception as e:
                    logger.error(f"❌ {test_name}: ERROR - {e}")
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"PERFORMANCE MONITORING TESTS SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Passed: {passed}/{total}")
            logger.info(f"Success rate: {passed/total*100:.1f}%")
            
            if passed == total:
                logger.info("🎉 All performance monitoring tests passed!")
                return True
            else:
                logger.error("💥 Some performance monitoring tests failed!")
                return False
                
        finally:
            # Always cleanup
            self.cleanup_test_environment()

def main():
    """Main test execution"""
    tester = PerformanceMonitoringTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()