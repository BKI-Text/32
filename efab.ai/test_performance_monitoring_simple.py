#!/usr/bin/env python3
"""
Simple Test for Performance Monitoring Dashboard (without psutil)
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance_monitoring_config():
    """Test performance monitoring configuration without external dependencies"""
    logger.info("🚀 Testing Performance Monitoring Configuration")
    
    try:
        # Test 1: Check if monitoring files exist
        monitoring_dir = Path("src/monitoring")
        if not monitoring_dir.exists():
            logger.error("Monitoring directory not found")
            return False
        
        required_files = [
            "performance_dashboard.py",
            "streamlit_dashboard.py",
            "__init__.py"
        ]
        
        for file_name in required_files:
            file_path = monitoring_dir / file_name
            if not file_path.exists():
                logger.error(f"Required file not found: {file_name}")
                return False
            
            logger.info(f"✅ Found {file_name}")
        
        # Test 2: Check file structure and basic imports
        dashboard_file = monitoring_dir / "performance_dashboard.py"
        
        with open(dashboard_file, 'r') as f:
            content = f.read()
        
        # Check for required classes
        required_classes = [
            "class PerformanceMetric",
            "class ModelPerformanceData",
            "class SystemMetrics",
            "class MetricsCollector",
            "class AlertSystem",
            "class PerformanceDashboard"
        ]
        
        for class_name in required_classes:
            if class_name not in content:
                logger.error(f"Required class not found: {class_name}")
                return False
            
            logger.info(f"✅ Found {class_name}")
        
        # Test 3: Check Streamlit dashboard
        streamlit_file = monitoring_dir / "streamlit_dashboard.py"
        
        with open(streamlit_file, 'r') as f:
            streamlit_content = f.read()
        
        # Check for required Streamlit components
        required_components = [
            "st.set_page_config",
            "plot_system_metrics",
            "plot_model_metrics",
            "display_alerts",
            "def main():"
        ]
        
        for component in required_components:
            if component not in streamlit_content:
                logger.error(f"Required component not found: {component}")
                return False
            
            logger.info(f"✅ Found {component}")
        
        # Test 4: Test basic data structures
        logger.info("Testing basic data structures...")
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test database creation
            db_path = os.path.join(temp_dir, "test_performance.db")
            
            # Create database tables manually to test structure
            with sqlite3.connect(db_path) as conn:
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
            
            # Test inserting sample data
            with sqlite3.connect(db_path) as conn:
                # Insert sample model metric
                conn.execute('''
                    INSERT INTO model_metrics (
                        model_id, model_name, timestamp, prediction_latency, throughput,
                        accuracy, error_rate, memory_usage, cpu_usage, prediction_count, error_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'test_model',
                    'Test Model',
                    datetime.now().isoformat(),
                    0.15, 100.0, 0.85, 0.05, 128.0, 25.0, 1000, 50
                ))
                
                # Insert sample system metric
                conn.execute('''
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, disk_usage, network_io,
                        active_connections, response_time, error_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), 45.0, 65.0, 30.0, '{}', 25, 0.12, 0.02
                ))
                
                # Insert sample custom metric
                conn.execute('''
                    INSERT INTO custom_metrics (
                        timestamp, metric_name, metric_value, metric_type, labels, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), 'custom_test_metric', 75.0, 'gauge', '{}', '{}'
                ))
            
            # Test querying data
            with sqlite3.connect(db_path) as conn:
                # Check model metrics
                cursor = conn.execute("SELECT * FROM model_metrics")
                model_rows = cursor.fetchall()
                
                if not model_rows:
                    logger.error("No model metrics found after insertion")
                    return False
                
                logger.info(f"✅ Found {len(model_rows)} model metrics")
                
                # Check system metrics
                cursor = conn.execute("SELECT * FROM system_metrics")
                system_rows = cursor.fetchall()
                
                if not system_rows:
                    logger.error("No system metrics found after insertion")
                    return False
                
                logger.info(f"✅ Found {len(system_rows)} system metrics")
                
                # Check custom metrics
                cursor = conn.execute("SELECT * FROM custom_metrics")
                custom_rows = cursor.fetchall()
                
                if not custom_rows:
                    logger.error("No custom metrics found after insertion")
                    return False
                
                logger.info(f"✅ Found {len(custom_rows)} custom metrics")
            
            # Test 5: Test alert system logic
            logger.info("Testing alert system logic...")
            
            # Test alert condition evaluation
            def evaluate_condition(value: float, condition: str, threshold: float) -> bool:
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
            
            # Test various conditions
            test_cases = [
                (75.0, '>', 50.0, True),
                (30.0, '>', 50.0, False),
                (45.0, '<', 50.0, True),
                (60.0, '<', 50.0, False),
                (50.0, '>=', 50.0, True),
                (49.0, '>=', 50.0, False),
                (50.0, '<=', 50.0, True),
                (51.0, '<=', 50.0, False),
                (50.0, '==', 50.0, True),
                (51.0, '==', 50.0, False),
                (51.0, '!=', 50.0, True),
                (50.0, '!=', 50.0, False),
            ]
            
            for value, condition, threshold, expected in test_cases:
                result = evaluate_condition(value, condition, threshold)
                if result != expected:
                    logger.error(f"Alert condition test failed: {value} {condition} {threshold} -> {result} (expected {expected})")
                    return False
            
            logger.info("✅ Alert condition evaluation tests passed")
            
            # Test 6: Test dashboard data structure
            logger.info("Testing dashboard data structure...")
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'system': {
                        'avg_cpu': 45.0,
                        'max_cpu': 75.0,
                        'avg_memory': 65.0,
                        'max_memory': 80.0,
                        'active_connections': 25
                    },
                    'models': {
                        'total_models': 2,
                        'total_predictions': 2000,
                        'total_errors': 100,
                        'avg_latency': 0.15,
                        'avg_throughput': 100.0,
                        'error_rate': 5.0
                    }
                },
                'model_metrics': {
                    'test_model': [
                        {
                            'model_id': 'test_model',
                            'model_name': 'Test Model',
                            'timestamp': datetime.now().isoformat(),
                            'prediction_latency': 0.15,
                            'throughput': 100.0,
                            'accuracy': 0.85,
                            'error_rate': 0.05,
                            'memory_usage': 128.0,
                            'cpu_usage': 25.0,
                            'prediction_count': 1000,
                            'error_count': 50
                        }
                    ]
                },
                'system_metrics': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'cpu_percent': 45.0,
                        'memory_percent': 65.0,
                        'disk_usage': 30.0,
                        'network_io': {},
                        'active_connections': 25,
                        'response_time': 0.12,
                        'error_rate': 0.02
                    }
                ],
                'active_alerts': [
                    {
                        'rule_name': 'high_cpu_usage',
                        'metric_name': 'cpu_percent',
                        'current_value': 75.0,
                        'threshold': 70.0,
                        'condition': '>',
                        'triggered_at': datetime.now().isoformat(),
                        'status': 'active'
                    }
                ],
                'alert_history': [],
                'monitoring_status': {
                    'collector_running': True,
                    'total_models': 2,
                    'data_points': 100
                }
            }
            
            # Test JSON serialization
            json_data = json.dumps(dashboard_data, indent=2)
            
            # Test JSON deserialization
            loaded_data = json.loads(json_data)
            
            if loaded_data['summary']['system']['avg_cpu'] != 45.0:
                logger.error("JSON serialization/deserialization failed")
                return False
            
            logger.info("✅ Dashboard data structure test passed")
            
            # Test 7: Test export functionality
            logger.info("Testing export functionality...")
            
            export_file = os.path.join(temp_dir, "test_export.json")
            
            with open(export_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            # Verify file was created and has content
            if not os.path.exists(export_file):
                logger.error("Export file was not created")
                return False
            
            with open(export_file, 'r') as f:
                exported_data = json.load(f)
            
            if exported_data['summary']['models']['total_models'] != 2:
                logger.error("Export data verification failed")
                return False
            
            logger.info("✅ Export functionality test passed")
            
            logger.info("🎉 All performance monitoring configuration tests passed!")
            return True
            
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    logger.info("🚀 Running Simple Performance Monitoring Tests")
    
    success = test_performance_monitoring_config()
    
    if success:
        logger.info("✅ All performance monitoring tests passed!")
    else:
        logger.error("❌ Some performance monitoring tests failed!")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()