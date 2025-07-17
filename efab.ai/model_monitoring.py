#!/usr/bin/env python3
"""
Model Monitoring and Alerting System for Beverly Knits AI
Monitors model performance and provides alerts for degradation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor ML model performance and alert on degradation"""
    
    def __init__(self, models_path: str = "models/trained/", data_path: str = "data/live/"):
        self.models_path = Path(models_path)
        self.data_path = Path(data_path)
        self.monitoring_path = Path("monitoring/")
        self.monitoring_path.mkdir(parents=True, exist_ok=True)
        
        # Performance thresholds
        self.performance_thresholds = {
            'demand_forecasting': {'mape': 50.0, 'mae': 2000.0},
            'price_prediction': {'mape': 25.0, 'mae': 2.0},
            'anomaly_detection': {'detection_rate': 0.05}
        }
        
        # Load baseline performance
        self.baseline_performance = self.load_baseline_performance()
    
    def load_baseline_performance(self) -> Dict[str, Any]:
        """Load baseline model performance metrics"""
        try:
            results_path = self.models_path / "training_results_basic.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    return json.load(f)
            
            # Try enhanced results
            enhanced_path = self.models_path / "enhanced_training_results.json"
            if enhanced_path.exists():
                with open(enhanced_path, 'r') as f:
                    return json.load(f)
            
            logger.warning("No baseline performance data found")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading baseline performance: {e}")
            return {}
    
    def load_current_data(self) -> pd.DataFrame:
        """Load current sales data for monitoring"""
        try:
            sales_file = self.data_path / "Sales Activity Report.csv"
            if not sales_file.exists():
                return pd.DataFrame()
            
            sales_data = pd.read_csv(sales_file, encoding='utf-8-sig')
            
            # Convert date column
            sales_data['date'] = pd.to_datetime(sales_data['Invoice Date'], errors='coerce')
            sales_data = sales_data.dropna(subset=['date'])
            
            # Clean numeric columns
            sales_data['Yds_ordered'] = pd.to_numeric(sales_data['Yds_ordered'].astype(str).str.replace(',', ''), errors='coerce')
            sales_data['Unit Price'] = pd.to_numeric(sales_data['Unit Price'].astype(str).str.replace('$', ''), errors='coerce')
            sales_data['Document'] = pd.to_numeric(sales_data['Document'], errors='coerce')
            
            # Remove invalid data
            sales_data = sales_data.dropna(subset=['Yds_ordered', 'Unit Price'])
            sales_data = sales_data[sales_data['Yds_ordered'] > 0]
            sales_data = sales_data[sales_data['Unit Price'] > 0]
            
            # Group by date
            daily_data = sales_data.groupby('date').agg({
                'Yds_ordered': 'sum',
                'Unit Price': 'mean',
                'Document': 'count'
            }).reset_index()
            
            daily_data.columns = ['date', 'demand', 'Unit Price', 'Document']
            daily_data = daily_data.set_index('date').sort_index()
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error loading current data: {e}")
            return pd.DataFrame()
    
    def test_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Test current model performance"""
        try:
            # Load model
            model_path = self.models_path / f"{model_name}_model.pkl"
            if not model_path.exists():
                return {'status': 'error', 'error': 'Model not found'}
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load current data
            current_data = self.load_current_data()
            if current_data.empty:
                return {'status': 'error', 'error': 'No current data available'}
            
            # Test demand forecasting
            if model_name == 'demand_forecasting':
                return self.test_demand_forecasting(model_data, current_data)
            
            # Test price prediction
            elif model_name == 'price_prediction':
                return self.test_price_prediction(model_data, current_data)
            
            # Test anomaly detection
            elif model_name == 'anomaly_detection':
                return self.test_anomaly_detection(model_data)
            
            else:
                return {'status': 'error', 'error': 'Unknown model type'}
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_demand_forecasting(self, model_data: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Test demand forecasting model performance"""
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Prepare features
            features_df = data.copy()
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_month'] = features_df.index.day
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
            
            # Lag features
            features_df['demand_lag_1'] = features_df['demand'].shift(1)
            features_df['demand_lag_7'] = features_df['demand'].shift(7)
            features_df['demand_rolling_7'] = features_df['demand'].rolling(7).mean()
            features_df['demand_rolling_30'] = features_df['demand'].rolling(30).mean()
            
            # Drop NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < 10:
                return {'status': 'error', 'error': 'Insufficient data for testing'}
            
            # Use last 20% for testing
            test_size = max(5, int(len(features_df) * 0.2))
            test_data = features_df.tail(test_size)
            
            X_test = test_data[feature_columns]
            y_test = test_data['demand']
            
            # Make predictions
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_test - predictions))
            rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Compare with baseline
            baseline_mape = self.baseline_performance.get('demand_forecasting', {}).get('mape', 100)
            performance_change = ((mape - baseline_mape) / baseline_mape) * 100
            
            return {
                'status': 'success',
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'baseline_mape': baseline_mape,
                'performance_change': performance_change,
                'test_samples': len(y_test),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_price_prediction(self, model_data: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Test price prediction model performance"""
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Prepare features
            features_df = data.copy()
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_month'] = features_df.index.day
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            features_df['Yds_ordered'] = features_df['demand']
            features_df['demand_rolling_7'] = features_df['demand'].rolling(7).mean()
            features_df['demand_rolling_30'] = features_df['demand'].rolling(30).mean()
            
            # Drop NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < 10:
                return {'status': 'error', 'error': 'Insufficient data for testing'}
            
            # Use last 20% for testing
            test_size = max(5, int(len(features_df) * 0.2))
            test_data = features_df.tail(test_size)
            
            X_test = test_data[feature_columns]
            y_test = test_data['Unit Price']
            
            # Make predictions
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_test - predictions))
            rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Compare with baseline
            baseline_mape = self.baseline_performance.get('price_prediction', {}).get('mape', 50)
            performance_change = ((mape - baseline_mape) / baseline_mape) * 100
            
            return {
                'status': 'success',
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'baseline_mape': baseline_mape,
                'performance_change': performance_change,
                'test_samples': len(y_test),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_anomaly_detection(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test anomaly detection model performance"""
        try:
            # Load supplier data
            inventory_file = self.data_path / "Yarn_ID_Current_Inventory.csv"
            if not inventory_file.exists():
                return {'status': 'error', 'error': 'Inventory data not found'}
            
            inventory_data = pd.read_csv(inventory_file, encoding='utf-8-sig')
            
            # Prepare supplier data
            supplier_data = pd.DataFrame({
                'supplier_id': inventory_data['Supplier'].astype(str),
                'cost_per_unit': pd.to_numeric(inventory_data['Cost_Pound'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0),
                'inventory_level': pd.to_numeric(inventory_data['Inventory'].astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', ''), errors='coerce').fillna(0),
                'on_order': pd.to_numeric(inventory_data['On_Order'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            })
            
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Prepare features
            features_df = pd.DataFrame()
            features_df['cost_per_pound'] = supplier_data['cost_per_unit']
            features_df['inventory_level'] = supplier_data['inventory_level']
            features_df['on_order'] = supplier_data['on_order']
            features_df['inventory_ratio'] = features_df['inventory_level'] / (features_df['on_order'] + 1e-8)
            features_df['cost_inventory_ratio'] = features_df['cost_per_pound'] * abs(features_df['inventory_level'])
            
            # Fill missing values
            features_df = features_df.fillna(0)
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Detect anomalies
            anomaly_predictions = model.predict(features_scaled)
            anomaly_count = np.sum(anomaly_predictions == -1)
            detection_rate = anomaly_count / len(features_scaled)
            
            # Compare with baseline
            baseline_detection_rate = self.baseline_performance.get('anomaly_detection', {}).get('anomaly_percentage', 10) / 100
            detection_change = ((detection_rate - baseline_detection_rate) / baseline_detection_rate) * 100
            
            return {
                'status': 'success',
                'anomalies_detected': int(anomaly_count),
                'total_samples': len(features_scaled),
                'detection_rate': detection_rate,
                'baseline_detection_rate': baseline_detection_rate,
                'detection_change': detection_change,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_alert_conditions(self, performance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if alert conditions are met"""
        alerts = []
        
        for model_name, results in performance_results.items():
            if results.get('status') != 'success':
                continue
            
            thresholds = self.performance_thresholds.get(model_name, {})
            
            # Check MAPE threshold
            if 'mape' in results and 'mape' in thresholds:
                if results['mape'] > thresholds['mape']:
                    alerts.append({
                        'model': model_name,
                        'type': 'performance_degradation',
                        'metric': 'mape',
                        'current_value': results['mape'],
                        'threshold': thresholds['mape'],
                        'severity': 'high' if results['mape'] > thresholds['mape'] * 1.5 else 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Check MAE threshold
            if 'mae' in results and 'mae' in thresholds:
                if results['mae'] > thresholds['mae']:
                    alerts.append({
                        'model': model_name,
                        'type': 'performance_degradation',
                        'metric': 'mae',
                        'current_value': results['mae'],
                        'threshold': thresholds['mae'],
                        'severity': 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Check performance change
            if 'performance_change' in results:
                if results['performance_change'] > 20:  # 20% degradation
                    alerts.append({
                        'model': model_name,
                        'type': 'performance_degradation',
                        'metric': 'performance_change',
                        'current_value': results['performance_change'],
                        'threshold': 20,
                        'severity': 'high',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        logger.info("🔍 Generating model monitoring report...")
        
        # Test all models
        models_to_test = ['demand_forecasting', 'price_prediction', 'anomaly_detection']
        performance_results = {}
        
        for model_name in models_to_test:
            logger.info(f"Testing {model_name} model...")
            performance_results[model_name] = self.test_model_performance(model_name)
        
        # Check for alerts
        alerts = self.check_alert_conditions(performance_results)
        
        # Create report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'performance_results': performance_results,
            'alerts': alerts,
            'summary': {
                'total_models_tested': len(models_to_test),
                'models_healthy': sum(1 for r in performance_results.values() if r.get('status') == 'success'),
                'total_alerts': len(alerts),
                'high_severity_alerts': len([a for a in alerts if a.get('severity') == 'high'])
            }
        }
        
        # Save report
        report_path = self.monitoring_path / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Monitoring report saved to {report_path}")
        return report
    
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle"""
        logger.info("🚀 Starting Model Monitoring Cycle")
        
        try:
            # Generate report
            report = self.generate_monitoring_report()
            
            # Print summary
            print("\n" + "="*60)
            print("📊 MODEL MONITORING REPORT")
            print("="*60)
            
            for model_name, results in report['performance_results'].items():
                print(f"\n🔍 {model_name.upper()}:")
                if results.get('status') == 'success':
                    if 'mape' in results:
                        print(f"  MAPE: {results['mape']:.2f}%")
                    if 'mae' in results:
                        print(f"  MAE: {results['mae']:.2f}")
                    if 'performance_change' in results:
                        change = results['performance_change']
                        symbol = "📈" if change > 0 else "📉"
                        print(f"  Performance Change: {symbol} {change:.1f}%")
                else:
                    print(f"  Status: ❌ {results.get('error', 'Unknown error')}")
            
            # Print alerts
            if report['alerts']:
                print(f"\n🚨 ALERTS ({len(report['alerts'])}):")
                for alert in report['alerts']:
                    severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                    print(f"  {severity_icon} {alert['model']} - {alert['metric']}: {alert['current_value']:.2f} > {alert['threshold']}")
            else:
                print("\n✅ No alerts detected")
            
            print(f"\n📈 SUMMARY:")
            print(f"  Models tested: {report['summary']['total_models_tested']}")
            print(f"  Healthy models: {report['summary']['models_healthy']}")
            print(f"  Total alerts: {report['summary']['total_alerts']}")
            
            logger.info("✅ Monitoring cycle completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.run_monitoring_cycle()