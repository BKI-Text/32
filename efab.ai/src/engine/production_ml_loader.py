"""
Production ML Model Loader for Beverly Knits AI Supply Chain Planner

This module loads trained models and provides a unified interface for production use.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProductionMLLoader:
    """Load and manage trained ML models for production use"""
    
    def __init__(self, models_path: str = "models/trained/"):
        self.models_path = Path(models_path)
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Load available models
        self.load_available_models()
    
    def load_available_models(self):
        """Load all available trained models"""
        logger.info("Loading available ML models for production...")
        
        # Load demand forecasting model
        demand_model_path = self.models_path / "demand_forecasting_model.pkl"
        if demand_model_path.exists():
            try:
                with open(demand_model_path, 'rb') as f:
                    self.loaded_models['demand_forecasting'] = pickle.load(f)
                logger.info("✅ Demand forecasting model loaded")
            except Exception as e:
                logger.error(f"❌ Error loading demand forecasting model: {e}")
        
        # Load price prediction model
        price_model_path = self.models_path / "price_prediction_model.pkl"
        if price_model_path.exists():
            try:
                with open(price_model_path, 'rb') as f:
                    self.loaded_models['price_prediction'] = pickle.load(f)
                logger.info("✅ Price prediction model loaded")
            except Exception as e:
                logger.error(f"❌ Error loading price prediction model: {e}")
        
        # Load anomaly detection model
        anomaly_model_path = self.models_path / "anomaly_detection_model.pkl"
        if anomaly_model_path.exists():
            try:
                with open(anomaly_model_path, 'rb') as f:
                    self.loaded_models['anomaly_detection'] = pickle.load(f)
                logger.info("✅ Anomaly detection model loaded")
            except Exception as e:
                logger.error(f"❌ Error loading anomaly detection model: {e}")
        
        # Load metadata
        results_path = self.models_path / "training_results_basic.json"
        if results_path.exists():
            try:
                import json
                with open(results_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("✅ Model metadata loaded")
            except Exception as e:
                logger.error(f"❌ Error loading model metadata: {e}")
        
        logger.info(f"Loaded {len(self.loaded_models)} production models")
    
    def predict_demand(self, 
                      historical_data: pd.DataFrame, 
                      periods: int = 30) -> List[Dict[str, Any]]:
        """Predict demand using trained forecasting model"""
        try:
            if 'demand_forecasting' not in self.loaded_models:
                logger.warning("Demand forecasting model not available")
                return []
            
            model_data = self.loaded_models['demand_forecasting']
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Prepare features for prediction
            latest_data = historical_data.tail(1).copy()
            
            # Create future predictions
            predictions = []
            current_date = historical_data.index.max()
            
            for i in range(periods):
                future_date = current_date + timedelta(days=i+1)
                
                # Create feature vector
                features = {}
                features['day_of_week'] = future_date.dayofweek
                features['day_of_month'] = future_date.day
                features['month'] = future_date.month
                features['quarter'] = future_date.quarter
                features['is_weekend'] = 1 if future_date.dayofweek >= 5 else 0
                
                # Use last available values for other features
                if len(historical_data) > 0:
                    features['demand_lag_1'] = historical_data['demand'].iloc[-1]
                    features['demand_lag_7'] = historical_data['demand'].iloc[-7] if len(historical_data) >= 7 else historical_data['demand'].iloc[-1]
                    features['demand_rolling_7'] = historical_data['demand'].tail(7).mean()
                    features['demand_rolling_30'] = historical_data['demand'].tail(30).mean()
                    features['Unit Price'] = historical_data['Unit Price'].iloc[-1] if 'Unit Price' in historical_data.columns else 8.0
                    features['Document'] = historical_data['Document'].iloc[-1] if 'Document' in historical_data.columns else 1
                else:
                    # Default values
                    features.update({
                        'demand_lag_1': 1000,
                        'demand_lag_7': 1000,
                        'demand_rolling_7': 1000,
                        'demand_rolling_30': 1000,
                        'Unit Price': 8.0,
                        'Document': 1
                    })
                
                # Create feature vector
                feature_vector = []
                for col in feature_columns:
                    feature_vector.append(features.get(col, 0))
                
                # Scale features
                feature_vector = np.array(feature_vector).reshape(1, -1)
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Make prediction
                prediction = model.predict(feature_vector_scaled)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                
                predictions.append({
                    'date': future_date,
                    'predicted_demand': float(prediction),
                    'confidence': 0.7,  # Basic confidence score
                    'model': 'random_forest'
                })
            
            logger.info(f"Generated {len(predictions)} demand predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting demand: {e}")
            return []
    
    def predict_price(self, demand_value: float, date: datetime) -> float:
        """Predict price based on demand and date"""
        try:
            if 'price_prediction' not in self.loaded_models:
                logger.warning("Price prediction model not available")
                return 8.0  # Default price
            
            model_data = self.loaded_models['price_prediction']
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Create feature vector
            features = {}
            features['day_of_week'] = date.dayofweek
            features['day_of_month'] = date.day
            features['month'] = date.month
            features['quarter'] = date.quarter
            features['Yds_ordered'] = demand_value
            features['demand_rolling_7'] = demand_value  # Simplified
            features['demand_rolling_30'] = demand_value  # Simplified
            
            # Create feature vector
            feature_vector = []
            for col in feature_columns:
                feature_vector.append(features.get(col, 0))
            
            # Scale features
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Make prediction
            prediction = model.predict(feature_vector_scaled)[0]
            prediction = max(1.0, prediction)  # Ensure minimum price
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return 8.0  # Default price
    
    def detect_anomalies(self, supplier_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in supplier data"""
        try:
            if 'anomaly_detection' not in self.loaded_models:
                logger.warning("Anomaly detection model not available")
                return []
            
            model_data = self.loaded_models['anomaly_detection']
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Prepare features
            features_df = pd.DataFrame()
            
            # Map supplier data to expected features
            features_df['cost_per_pound'] = supplier_data.get('cost_per_unit', 0)
            features_df['inventory_level'] = supplier_data.get('inventory_level', 0)
            features_df['on_order'] = supplier_data.get('on_order', 0)
            
            # Calculate engineered features
            features_df['inventory_ratio'] = features_df['inventory_level'] / (features_df['on_order'] + 1e-8)
            features_df['cost_inventory_ratio'] = features_df['cost_per_pound'] * abs(features_df['inventory_level'])
            
            # Fill missing values
            features_df = features_df.fillna(0)
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Detect anomalies
            anomaly_scores = model.decision_function(features_scaled)
            anomaly_predictions = model.predict(features_scaled)
            
            # Create anomaly results
            anomalies = []
            for i, (score, prediction) in enumerate(zip(anomaly_scores, anomaly_predictions)):
                is_anomaly = prediction == -1
                
                if is_anomaly:
                    anomaly_info = {
                        'supplier_id': supplier_data.get('supplier_id', [f'SUP_{i}'])[i] if hasattr(supplier_data, 'get') else f'SUP_{i}',
                        'anomaly_score': float(score),
                        'is_anomaly': True,
                        'anomaly_type': 'statistical_outlier',
                        'severity': 'high' if score < -0.5 else 'medium',
                        'description': f'Supplier shows anomalous behavior pattern (score: {score:.2f})'
                    }
                    anomalies.append(anomaly_info)
            
            logger.info(f"Detected {len(anomalies)} anomalies in {len(supplier_data)} suppliers")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {
            'models_loaded': len(self.loaded_models),
            'available_models': list(self.loaded_models.keys()),
            'model_details': {}
        }
        
        # Add model performance metrics
        for model_name, model_data in self.loaded_models.items():
            if 'metrics' in model_data:
                status['model_details'][model_name] = {
                    'loaded': True,
                    'metrics': model_data['metrics']
                }
        
        # Add metadata
        if self.model_metadata:
            status['training_info'] = {
                'training_completed': self.model_metadata.get('training_completed'),
                'time_series_records': self.model_metadata.get('time_series_records'),
                'supplier_records': self.model_metadata.get('supplier_records')
            }
        
        return status
    
    def generate_forecast_report(self, 
                               historical_data: pd.DataFrame, 
                               periods: int = 30) -> Dict[str, Any]:
        """Generate comprehensive forecast report"""
        try:
            # Generate demand predictions
            demand_predictions = self.predict_demand(historical_data, periods)
            
            # Calculate summary statistics
            if demand_predictions:
                predicted_values = [p['predicted_demand'] for p in demand_predictions]
                
                report = {
                    'forecast_period': periods,
                    'total_predicted_demand': sum(predicted_values),
                    'average_daily_demand': np.mean(predicted_values),
                    'peak_demand': max(predicted_values),
                    'minimum_demand': min(predicted_values),
                    'demand_volatility': np.std(predicted_values),
                    'predictions': demand_predictions,
                    'model_used': 'random_forest_regressor',
                    'confidence_level': 0.7
                }
                
                # Add price predictions
                for prediction in demand_predictions:
                    predicted_price = self.predict_price(
                        prediction['predicted_demand'], 
                        prediction['date']
                    )
                    prediction['predicted_price'] = predicted_price
                    prediction['estimated_revenue'] = predicted_price * prediction['predicted_demand']
                
                # Calculate revenue summary
                total_revenue = sum(p['estimated_revenue'] for p in demand_predictions)
                report['total_estimated_revenue'] = total_revenue
                report['average_price'] = np.mean([p['predicted_price'] for p in demand_predictions])
                
                return report
            else:
                return {'error': 'No predictions generated'}
                
        except Exception as e:
            logger.error(f"Error generating forecast report: {e}")
            return {'error': str(e)}
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        return model_name in self.loaded_models
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        if model_name not in self.loaded_models:
            return {'error': 'Model not available'}
        
        model_data = self.loaded_models[model_name]
        return model_data.get('metrics', {})

# Global instance for easy access
production_ml_loader = ProductionMLLoader()