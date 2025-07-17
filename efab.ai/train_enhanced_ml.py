#!/usr/bin/env python3
"""
Enhanced ML Training Script with Advanced Models
Attempts to use advanced ML libraries if available, falls back to basic sklearn
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """Enhanced ML trainer with fallback capabilities"""
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = Path(data_path)
        self.models_path = Path("models/trained/")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Check for advanced ML libraries
        self.has_statsmodels = self._check_library('statsmodels')
        self.has_prophet = self._check_library('prophet')
        self.has_tensorflow = self._check_library('tensorflow')
        
        logger.info(f"Available advanced libraries: statsmodels={self.has_statsmodels}, prophet={self.has_prophet}, tensorflow={self.has_tensorflow}")
    
    def _check_library(self, library_name: str) -> bool:
        """Check if a library is available"""
        try:
            __import__(library_name)
            return True
        except ImportError:
            return False
    
    def load_sales_data(self) -> pd.DataFrame:
        """Load and preprocess sales data"""
        sales_file = self.data_path / "Sales Activity Report.csv"
        if not sales_file.exists():
            logger.error(f"Sales file not found: {sales_file}")
            return pd.DataFrame()
        
        try:
            # Read sales data
            sales_data = pd.read_csv(sales_file, encoding='utf-8-sig')
            logger.info(f"Loaded {len(sales_data)} sales records")
            
            # Convert date column
            sales_data['date'] = pd.to_datetime(sales_data['Invoice Date'], errors='coerce')
            sales_data = sales_data.dropna(subset=['date'])
            
            # Clean numeric columns
            sales_data['Yds_ordered'] = pd.to_numeric(sales_data['Yds_ordered'].astype(str).str.replace(',', ''), errors='coerce')
            sales_data['Unit Price'] = pd.to_numeric(sales_data['Unit Price'].astype(str).str.replace('$', ''), errors='coerce')
            sales_data['Document'] = pd.to_numeric(sales_data['Document'], errors='coerce')
            
            # Remove outliers and invalid data
            sales_data = sales_data.dropna(subset=['Yds_ordered', 'Unit Price'])
            sales_data = sales_data[sales_data['Yds_ordered'] > 0]
            sales_data = sales_data[sales_data['Unit Price'] > 0]
            
            # Group by date for time series
            daily_data = sales_data.groupby('date').agg({
                'Yds_ordered': 'sum',
                'Unit Price': 'mean',
                'Document': 'count'
            }).reset_index()
            
            daily_data.columns = ['date', 'demand', 'Unit Price', 'Document']
            daily_data = daily_data.set_index('date').sort_index()
            
            logger.info(f"Processed to {len(daily_data)} daily records")
            return daily_data
            
        except Exception as e:
            logger.error(f"Error loading sales data: {e}")
            return pd.DataFrame()
    
    def train_arima_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA model if statsmodels is available"""
        if not self.has_statsmodels:
            logger.warning("statsmodels not available, skipping ARIMA training")
            return {'status': 'skipped', 'reason': 'statsmodels not available'}
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            
            # Use demand data
            demand_series = data['demand'].fillna(method='ffill')
            
            # Check stationarity
            adf_result = adfuller(demand_series)
            is_stationary = adf_result[1] < 0.05
            
            # Auto-determine ARIMA parameters
            if is_stationary:
                order = (1, 0, 1)  # AR(1), no differencing, MA(1)
            else:
                order = (1, 1, 1)  # AR(1), 1st differencing, MA(1)
            
            # Train ARIMA model
            model = ARIMA(demand_series, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=30)
            
            # Calculate performance metrics
            train_size = int(len(demand_series) * 0.8)
            train_data = demand_series[:train_size]
            test_data = demand_series[train_size:]
            
            train_model = ARIMA(train_data, order=order).fit()
            predictions = train_model.forecast(steps=len(test_data))
            
            mae = np.mean(np.abs(test_data - predictions))
            rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
            
            # Save model
            model_path = self.models_path / "arima_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': fitted_model,
                    'order': order,
                    'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape},
                    'forecast': forecast.tolist()
                }, f)
            
            logger.info(f"✅ ARIMA model trained - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return {
                'status': 'success',
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'order': order,
                'forecast_samples': len(forecast)
            }
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def train_prophet_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model if available"""
        if not self.has_prophet:
            logger.warning("Prophet not available, skipping Prophet training")
            return {'status': 'skipped', 'reason': 'prophet not available'}
        
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_data = data.reset_index()
            prophet_data = prophet_data.rename(columns={'date': 'ds', 'demand': 'y'})
            
            # Create and train Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Calculate performance metrics
            train_size = int(len(prophet_data) * 0.8)
            train_data = prophet_data[:train_size]
            test_data = prophet_data[train_size:]
            
            train_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            train_model.fit(train_data)
            
            test_future = train_model.make_future_dataframe(periods=len(test_data))
            test_forecast = train_model.predict(test_future)
            
            test_predictions = test_forecast['yhat'].tail(len(test_data))
            actual_values = test_data['y']
            
            mae = np.mean(np.abs(actual_values - test_predictions))
            rmse = np.sqrt(np.mean((actual_values - test_predictions) ** 2))
            mape = np.mean(np.abs((actual_values - test_predictions) / actual_values)) * 100
            
            # Save model
            model_path = self.models_path / "prophet_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape},
                    'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
                }, f)
            
            logger.info(f"✅ Prophet model trained - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return {
                'status': 'success',
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'forecast_samples': len(forecast)
            }
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def train_lstm_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model if TensorFlow is available"""
        if not self.has_tensorflow:
            logger.warning("TensorFlow not available, skipping LSTM training")
            return {'status': 'skipped', 'reason': 'tensorflow not available'}
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler
            
            # Prepare data for LSTM
            demand_data = data['demand'].values.reshape(-1, 1)
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(demand_data)
            
            # Create sequences
            def create_sequences(data, seq_length=10):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:(i + seq_length), 0])
                    y.append(data[i + seq_length, 0])
                return np.array(X), np.array(y)
            
            X, y = create_sequences(scaled_data)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Inverse transform
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            mae = np.mean(np.abs(y_test_actual - predictions))
            rmse = np.sqrt(np.mean((y_test_actual - predictions) ** 2))
            mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
            
            # Save model
            model_path = self.models_path / "lstm_model.h5"
            model.save(str(model_path))
            
            scaler_path = self.models_path / "lstm_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"✅ LSTM model trained - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            return {
                'status': 'success',
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'epochs': 50,
                'sequence_length': 10
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def train_enhanced_ensemble(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble of available models"""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
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
            
            # Features and target
            feature_columns = [
                'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
                'demand_lag_1', 'demand_lag_7', 'demand_rolling_7', 'demand_rolling_30',
                'Unit Price', 'Document'
            ]
            
            X = features_df[feature_columns]
            y = features_df['demand']
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            trained_models = {}
            model_predictions = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                    
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                    
                    trained_models[name] = {
                        'model': model,
                        'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape}
                    }
                    model_predictions[name] = predictions
                    
                    logger.info(f"✅ {name} trained - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
            
            # Create ensemble prediction (simple average)
            if model_predictions:
                ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
                
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
                
                logger.info(f"✅ Ensemble model - MAE: {ensemble_mae:.2f}, RMSE: {ensemble_rmse:.2f}, MAPE: {ensemble_mape:.2f}%")
                
                # Save ensemble model
                ensemble_path = self.models_path / "ensemble_model.pkl"
                with open(ensemble_path, 'wb') as f:
                    pickle.dump({
                        'models': trained_models,
                        'scaler': scaler,
                        'feature_columns': feature_columns,
                        'ensemble_metrics': {
                            'mae': ensemble_mae,
                            'rmse': ensemble_rmse, 
                            'mape': ensemble_mape
                        }
                    }, f)
                
                return {
                    'status': 'success',
                    'models_trained': list(trained_models.keys()),
                    'ensemble_metrics': {
                        'mae': ensemble_mae,
                        'rmse': ensemble_rmse,
                        'mape': ensemble_mape
                    },
                    'individual_models': {name: model['metrics'] for name, model in trained_models.items()}
                }
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_enhanced_training(self):
        """Run enhanced ML training with all available models"""
        logger.info("🚀 Starting Enhanced ML Training")
        
        # Load data
        data = self.load_sales_data()
        if data.empty:
            logger.error("No data available for training")
            return
        
        training_results = {
            'training_started': datetime.now().isoformat(),
            'data_records': len(data),
            'available_libraries': {
                'statsmodels': self.has_statsmodels,
                'prophet': self.has_prophet,
                'tensorflow': self.has_tensorflow
            }
        }
        
        # Train ARIMA model
        logger.info("Training ARIMA model...")
        training_results['arima'] = self.train_arima_model(data)
        
        # Train Prophet model
        logger.info("Training Prophet model...")
        training_results['prophet'] = self.train_prophet_model(data)
        
        # Train LSTM model
        logger.info("Training LSTM model...")
        training_results['lstm'] = self.train_lstm_model(data)
        
        # Train ensemble
        logger.info("Training ensemble model...")
        training_results['ensemble'] = self.train_enhanced_ensemble(data)
        
        # Save results
        training_results['training_completed'] = datetime.now().isoformat()
        
        results_path = self.models_path / "enhanced_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info("🎉 Enhanced ML Training Complete!")
        
        # Print summary
        successful_models = []
        for model_name, result in training_results.items():
            if isinstance(result, dict) and result.get('status') == 'success':
                successful_models.append(model_name)
        
        logger.info(f"Successfully trained models: {successful_models}")
        
        return training_results

if __name__ == "__main__":
    trainer = EnhancedMLTrainer()
    trainer.run_enhanced_training()