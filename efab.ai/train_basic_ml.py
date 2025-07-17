#!/usr/bin/env python3
"""
Basic ML Training Script using Available Dependencies
Uses only sklearn and pandas for basic ML functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Available ML libraries
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training_basic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BasicMLTrainer:
    """Basic ML training using sklearn and available dependencies"""
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = Path(data_path)
        self.models_path = Path("models/trained/")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Trained models
        self.trained_models = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data from Beverly Knits sources"""
        logger.info("Loading Beverly Knits data sources...")
        
        # Load sales data
        sales_path = self.data_path / "Sales Activity Report.csv"
        sales_data = pd.read_csv(sales_path, encoding='utf-8-sig')
        logger.info(f"Loaded sales data: {len(sales_data)} records")
        
        # Clean and prepare sales data
        sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])
        sales_data['Yds_ordered'] = pd.to_numeric(sales_data['Yds_ordered'], errors='coerce')
        sales_data['Unit Price'] = sales_data['Unit Price'].str.replace('$', '').str.replace(',', '')
        sales_data['Unit Price'] = pd.to_numeric(sales_data['Unit Price'], errors='coerce')
        
        # Create time series with features
        daily_sales = sales_data.groupby('Invoice Date').agg({
            'Yds_ordered': 'sum',
            'Unit Price': 'mean',
            'Document': 'count'
        }).reset_index()
        
        # Add time features
        daily_sales['day_of_week'] = daily_sales['Invoice Date'].dt.dayofweek
        daily_sales['day_of_month'] = daily_sales['Invoice Date'].dt.day
        daily_sales['month'] = daily_sales['Invoice Date'].dt.month
        daily_sales['quarter'] = daily_sales['Invoice Date'].dt.quarter
        daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
        
        # Create lag features
        daily_sales = daily_sales.sort_values('Invoice Date')
        daily_sales['demand_lag_1'] = daily_sales['Yds_ordered'].shift(1)
        daily_sales['demand_lag_7'] = daily_sales['Yds_ordered'].shift(7)
        daily_sales['demand_rolling_7'] = daily_sales['Yds_ordered'].rolling(window=7).mean()
        daily_sales['demand_rolling_30'] = daily_sales['Yds_ordered'].rolling(window=30).mean()
        
        # Fill missing values
        daily_sales = daily_sales.fillna(method='bfill').fillna(0)
        
        logger.info(f"Prepared time series data: {len(daily_sales)} records")
        logger.info(f"Date range: {daily_sales['Invoice Date'].min()} to {daily_sales['Invoice Date'].max()}")
        
        # Load inventory data for supplier analysis
        inventory_path = self.data_path / "Yarn_ID_Current_Inventory.csv"
        inventory_data = pd.read_csv(inventory_path, encoding='utf-8-sig')
        
        # Prepare supplier data for risk analysis
        supplier_features = []
        for _, row in inventory_data.iterrows():
            try:
                cost_str = str(row.get('Cost_Pound', '$0')).replace('$', '').replace(',', '')
                cost = float(cost_str) if cost_str else 0.0
                
                inventory_str = str(row.get('Inventory', '0')).replace(',', '').replace('(', '-').replace(')', '')
                inventory = float(inventory_str) if inventory_str else 0.0
                
                on_order_str = str(row.get('On_Order', '0')).replace(',', '')
                on_order = float(on_order_str) if on_order_str else 0.0
                
                supplier_info = {
                    'supplier': str(row.get('Supplier', 'Unknown')),
                    'yarn_id': str(row.get('Yarn_ID', 'Unknown')),
                    'cost_per_pound': cost,
                    'inventory_level': inventory,
                    'on_order': on_order,
                    'has_negative_inventory': 1 if inventory < 0 else 0,
                    'high_cost': 1 if cost > 5.0 else 0,
                    'out_of_stock': 1 if inventory <= 0 else 0,
                    'cost_category': 'high' if cost > 5.0 else 'medium' if cost > 2.0 else 'low'
                }
                supplier_features.append(supplier_info)
            except Exception as e:
                logger.warning(f"Error processing supplier row: {e}")
                continue
        
        supplier_df = pd.DataFrame(supplier_features)
        logger.info(f"Prepared supplier data: {len(supplier_df)} records")
        
        return daily_sales, supplier_df
    
    def train_demand_forecasting_model(self, time_series_data: pd.DataFrame):
        """Train demand forecasting using Random Forest"""
        logger.info("Training demand forecasting model...")
        
        try:
            # Prepare features and target
            feature_columns = [
                'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
                'demand_lag_1', 'demand_lag_7', 'demand_rolling_7', 'demand_rolling_30',
                'Unit Price', 'Document'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in time_series_data.columns]
            
            X = time_series_data[available_features]
            y = time_series_data['Yds_ordered']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            
            # Save model
            model_data = {
                'model': rf_model,
                'scaler': scaler,
                'feature_columns': available_features,
                'metrics': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape
                }
            }
            
            model_path = self.models_path / "demand_forecasting_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.trained_models['demand_forecasting'] = model_data
            
            logger.info(f"✅ Demand forecasting model trained successfully")
            logger.info(f"   MAE: {mae:.2f}")
            logger.info(f"   RMSE: {rmse:.2f}")
            logger.info(f"   MAPE: {mape:.2f}%")
            
            return {
                'status': 'success',
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'features_used': available_features,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"❌ Demand forecasting training failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def train_price_prediction_model(self, time_series_data: pd.DataFrame):
        """Train price prediction model"""
        logger.info("Training price prediction model...")
        
        try:
            # Prepare features for price prediction
            feature_columns = [
                'day_of_week', 'day_of_month', 'month', 'quarter',
                'Yds_ordered', 'demand_rolling_7', 'demand_rolling_30'
            ]
            
            available_features = [col for col in feature_columns if col in time_series_data.columns]
            
            # Filter out rows with missing price data
            price_data = time_series_data[time_series_data['Unit Price'] > 0]
            
            if len(price_data) < 20:
                logger.warning("Insufficient price data for training")
                return {'status': 'insufficient_data'}
            
            X = price_data[available_features]
            y = price_data['Unit Price']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Linear Regression model
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = lr_model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            
            # Save model
            model_data = {
                'model': lr_model,
                'scaler': scaler,
                'feature_columns': available_features,
                'metrics': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape
                }
            }
            
            model_path = self.models_path / "price_prediction_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.trained_models['price_prediction'] = model_data
            
            logger.info(f"✅ Price prediction model trained successfully")
            logger.info(f"   MAE: {mae:.2f}")
            logger.info(f"   RMSE: {rmse:.2f}")
            logger.info(f"   MAPE: {mape:.2f}%")
            
            return {
                'status': 'success',
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'features_used': available_features,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"❌ Price prediction training failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def train_anomaly_detection_model(self, supplier_data: pd.DataFrame):
        """Train anomaly detection for supplier analysis"""
        logger.info("Training anomaly detection model...")
        
        try:
            # Prepare numeric features
            numeric_features = ['cost_per_pound', 'inventory_level', 'on_order']
            
            # Filter and prepare data
            X = supplier_data[numeric_features].fillna(0)
            
            # Add engineered features
            X['inventory_ratio'] = X['inventory_level'] / (X['on_order'] + 1e-8)
            X['cost_inventory_ratio'] = X['cost_per_pound'] * abs(X['inventory_level'])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            iso_forest.fit(X_scaled)
            
            # Predict anomalies
            anomaly_scores = iso_forest.decision_function(X_scaled)
            anomaly_predictions = iso_forest.predict(X_scaled)
            
            # Calculate statistics
            anomaly_count = np.sum(anomaly_predictions == -1)
            anomaly_percentage = (anomaly_count / len(X)) * 100
            
            # Save model
            model_data = {
                'model': iso_forest,
                'scaler': scaler,
                'feature_columns': list(X.columns),
                'anomaly_count': int(anomaly_count),
                'anomaly_percentage': float(anomaly_percentage),
                'total_samples': len(X)
            }
            
            model_path = self.models_path / "anomaly_detection_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.trained_models['anomaly_detection'] = model_data
            
            logger.info(f"✅ Anomaly detection model trained successfully")
            logger.info(f"   Total samples: {len(X)}")
            logger.info(f"   Anomalies detected: {anomaly_count} ({anomaly_percentage:.1f}%)")
            
            return {
                'status': 'success',
                'total_samples': len(X),
                'anomalies_detected': int(anomaly_count),
                'anomaly_percentage': float(anomaly_percentage),
                'features_used': list(X.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection training failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def save_training_results(self, results: dict):
        """Save training results and create summary"""
        logger.info("Saving training results...")
        
        # Save JSON results
        results_path = self.models_path / "training_results_basic.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        report_path = self.models_path / "training_report_basic.txt"
        with open(report_path, 'w') as f:
            f.write("Beverly Knits Basic ML Training Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Using available dependencies: sklearn, pandas, numpy\n\n")
            
            f.write("TRAINED MODELS:\n")
            f.write("-" * 20 + "\n")
            
            if 'demand_forecasting' in results:
                result = results['demand_forecasting']
                f.write(f"Demand Forecasting: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"  Algorithm: Random Forest\n")
                    f.write(f"  MAPE: {result['mape']:.2f}%\n")
                    f.write(f"  MAE: {result['mae']:.2f}\n")
                    f.write(f"  RMSE: {result['rmse']:.2f}\n")
                    f.write(f"  Features: {len(result['features_used'])}\n")
                f.write("\n")
            
            if 'price_prediction' in results:
                result = results['price_prediction']
                f.write(f"Price Prediction: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"  Algorithm: Linear Regression\n")
                    f.write(f"  MAPE: {result['mape']:.2f}%\n")
                    f.write(f"  MAE: {result['mae']:.2f}\n")
                    f.write(f"  RMSE: {result['rmse']:.2f}\n")
                f.write("\n")
            
            if 'anomaly_detection' in results:
                result = results['anomaly_detection']
                f.write(f"Anomaly Detection: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"  Algorithm: Isolation Forest\n")
                    f.write(f"  Samples: {result['total_samples']}\n")
                    f.write(f"  Anomalies: {result['anomalies_detected']} ({result['anomaly_percentage']:.1f}%)\n")
                f.write("\n")
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Report saved to {report_path}")
    
    def run_training(self):
        """Run complete basic ML training"""
        logger.info("🚀 Starting Basic ML Training for Beverly Knits")
        logger.info("=" * 60)
        
        try:
            # Load data
            time_series_data, supplier_data = self.load_and_prepare_data()
            
            # Train models
            results = {}
            
            # Train demand forecasting
            results['demand_forecasting'] = self.train_demand_forecasting_model(time_series_data)
            
            # Train price prediction
            results['price_prediction'] = self.train_price_prediction_model(time_series_data)
            
            # Train anomaly detection
            results['anomaly_detection'] = self.train_anomaly_detection_model(supplier_data)
            
            # Add metadata
            results['training_completed'] = datetime.now().isoformat()
            results['time_series_records'] = len(time_series_data)
            results['supplier_records'] = len(supplier_data)
            results['dependencies_used'] = ['sklearn', 'pandas', 'numpy']
            
            # Save results
            self.save_training_results(results)
            
            # Print summary
            self.print_summary(results)
            
            logger.info("✅ Basic ML Training Completed Successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def print_summary(self, results: dict):
        """Print training summary"""
        print("\n" + "=" * 60)
        print("🎉 BASIC ML TRAINING SUMMARY")
        print("=" * 60)
        
        print(f"📊 Data Sources: Beverly Knits Sales & Inventory Data")
        print(f"📈 Time Series Records: {results.get('time_series_records', 0)}")
        print(f"🏭 Supplier Records: {results.get('supplier_records', 0)}")
        print(f"🔧 Dependencies: {', '.join(results.get('dependencies_used', []))}")
        
        print("\n🤖 TRAINED MODELS:")
        print("-" * 30)
        
        if 'demand_forecasting' in results:
            result = results['demand_forecasting']
            if result['status'] == 'success':
                print(f"  ✅ Demand Forecasting (Random Forest): MAPE {result['mape']:.2f}%")
            else:
                print(f"  ❌ Demand Forecasting: {result['status']}")
        
        if 'price_prediction' in results:
            result = results['price_prediction']
            if result['status'] == 'success':
                print(f"  ✅ Price Prediction (Linear Regression): MAPE {result['mape']:.2f}%")
            else:
                print(f"  ❌ Price Prediction: {result['status']}")
        
        if 'anomaly_detection' in results:
            result = results['anomaly_detection']
            if result['status'] == 'success':
                print(f"  ✅ Anomaly Detection (Isolation Forest): {result['anomalies_detected']} anomalies")
            else:
                print(f"  ❌ Anomaly Detection: {result['status']}")
        
        print("\n📁 TRAINED MODELS SAVED:")
        print("-" * 30)
        models_dir = Path("models/trained/")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pkl"):
                print(f"  📄 {model_file.name}")
        
        print("\n🚀 NEXT STEPS:")
        print("-" * 30)
        print("  1. Models trained with available dependencies")
        print("  2. Run: streamlit run main.py")
        print("  3. Test ML features in Beverly Knits app")
        print("  4. Install additional dependencies for advanced models")
        print("=" * 60)

if __name__ == "__main__":
    trainer = BasicMLTrainer()
    trainer.run_training()