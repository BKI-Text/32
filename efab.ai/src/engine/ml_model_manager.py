"""
ML Model Manager for Beverly Knits AI Supply Chain Planner

This module provides centralized management for all ML models including
ARIMA, Prophet, LSTM, and advanced risk scoring models.
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from ..config.settings import PLANNING_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mape: float
    mae: float
    rmse: float
    training_time: float
    last_updated: datetime
    training_samples: int

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    enabled: bool
    parameters: Dict[str, Any]
    retrain_interval_days: int
    min_training_samples: int

class MLModelManager:
    """Centralized ML model management system"""
    
    def __init__(self, model_cache_path: str = "models/cache/"):
        self.model_cache_path = Path(model_cache_path)
        self.model_cache_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Load configurations
        self._load_model_configurations()
        
        # Initialize models
        self._initialize_models()
        
    def _load_model_configurations(self):
        """Load model configurations from settings"""
        try:
            config_path = self.model_cache_path / "model_configs.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    configs = json.load(f)
                    for name, config in configs.items():
                        self.model_configs[name] = ModelConfig(**config)
            else:
                # Default configurations
                self.model_configs = {
                    "arima": ModelConfig(
                        name="arima",
                        enabled=True,
                        parameters={"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
                        retrain_interval_days=30,
                        min_training_samples=50
                    ),
                    "prophet": ModelConfig(
                        name="prophet",
                        enabled=True,
                        parameters={"seasonality_mode": "multiplicative", "yearly_seasonality": True},
                        retrain_interval_days=30,
                        min_training_samples=100
                    ),
                    "lstm": ModelConfig(
                        name="lstm",
                        enabled=True,
                        parameters={"sequence_length": 30, "hidden_units": 50, "epochs": 100},
                        retrain_interval_days=7,
                        min_training_samples=200
                    ),
                    "xgboost": ModelConfig(
                        name="xgboost",
                        enabled=True,
                        parameters={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                        retrain_interval_days=14,
                        min_training_samples=100
                    )
                }
                self._save_model_configurations()
                
        except Exception as e:
            logger.error(f"Error loading model configurations: {e}")
            
    def _save_model_configurations(self):
        """Save model configurations to file"""
        try:
            config_path = self.model_cache_path / "model_configs.json"
            configs = {name: {
                "name": config.name,
                "enabled": config.enabled,
                "parameters": config.parameters,
                "retrain_interval_days": config.retrain_interval_days,
                "min_training_samples": config.min_training_samples
            } for name, config in self.model_configs.items()}
            
            with open(config_path, 'w') as f:
                json.dump(configs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving model configurations: {e}")
            
    def _initialize_models(self):
        """Initialize all ML models"""
        logger.info("Initializing ML models...")
        
        # Import model classes
        try:
            from .forecasting.arima_forecaster import ARIMAForecaster
            from .forecasting.prophet_forecaster import ProphetForecaster
            from .forecasting.lstm_forecaster import LSTMForecaster
            from .forecasting.xgboost_forecaster import XGBoostForecaster
            
            # Initialize enabled models
            if self.model_configs["arima"].enabled:
                self.models["arima"] = ARIMAForecaster(
                    **self.model_configs["arima"].parameters
                )
                
            if self.model_configs["prophet"].enabled:
                self.models["prophet"] = ProphetForecaster(
                    **self.model_configs["prophet"].parameters
                )
                
            if self.model_configs["lstm"].enabled:
                self.models["lstm"] = LSTMForecaster(
                    **self.model_configs["lstm"].parameters
                )
                
            if self.model_configs["xgboost"].enabled:
                self.models["xgboost"] = XGBoostForecaster(
                    **self.model_configs["xgboost"].parameters
                )
                
            logger.info(f"Initialized {len(self.models)} ML models")
            
        except ImportError as e:
            logger.warning(f"Some ML models not available: {e}")
            logger.info("Will create model implementations...")
            
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a specific model by name"""
        return self.models.get(model_name)
        
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())
        
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled"""
        config = self.model_configs.get(model_name)
        return config.enabled if config else False
        
    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get performance metrics for a model"""
        return self.model_metrics.get(model_name)
        
    def should_retrain_model(self, model_name: str) -> bool:
        """Check if a model should be retrained based on interval"""
        config = self.model_configs.get(model_name)
        metrics = self.model_metrics.get(model_name)
        
        if not config or not metrics:
            return True
            
        days_since_training = (datetime.now() - metrics.last_updated).days
        return days_since_training >= config.retrain_interval_days
        
    def train_model(self, model_name: str, data: pd.DataFrame, target_column: str) -> bool:
        """Train a specific model"""
        try:
            model = self.models.get(model_name)
            config = self.model_configs.get(model_name)
            
            if not model or not config:
                logger.error(f"Model {model_name} not found or not configured")
                return False
                
            if len(data) < config.min_training_samples:
                logger.warning(f"Insufficient data for {model_name}: {len(data)} < {config.min_training_samples}")
                return False
                
            logger.info(f"Training {model_name} model...")
            start_time = datetime.now()
            
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Train model
            model.fit(X, y)
            
            # Calculate metrics
            predictions = model.predict(X)
            mape = mean_absolute_percentage_error(y, predictions)
            mae = np.mean(np.abs(y - predictions))
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store metrics
            self.model_metrics[model_name] = ModelMetrics(
                mape=mape,
                mae=mae,
                rmse=rmse,
                training_time=training_time,
                last_updated=datetime.now(),
                training_samples=len(data)
            )
            
            # Save model
            self._save_model(model_name, model)
            
            logger.info(f"Model {model_name} trained successfully. MAPE: {mape:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return False
            
    def predict(self, model_name: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Make predictions using a specific model"""
        try:
            model = self.models.get(model_name)
            if not model:
                logger.error(f"Model {model_name} not found")
                return None
                
            return model.predict(data)
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {e}")
            return None
            
    def get_best_model(self, metric: str = "mape") -> Optional[str]:
        """Get the best performing model based on a metric"""
        if not self.model_metrics:
            return None
            
        best_model = None
        best_score = float('inf')
        
        for model_name, metrics in self.model_metrics.items():
            score = getattr(metrics, metric, float('inf'))
            if score < best_score:
                best_score = score
                best_model = model_name
                
        return best_model
        
    def ensemble_predict(self, data: pd.DataFrame, models: Optional[List[str]] = None) -> np.ndarray:
        """Make ensemble predictions using multiple models"""
        if models is None:
            models = list(self.models.keys())
            
        predictions = []
        weights = []
        
        for model_name in models:
            if model_name in self.models:
                pred = self.predict(model_name, data)
                if pred is not None:
                    predictions.append(pred)
                    # Weight by inverse MAPE (better models get higher weight)
                    metrics = self.model_metrics.get(model_name)
                    weight = 1.0 / (metrics.mape + 0.001) if metrics else 1.0
                    weights.append(weight)
                    
        if not predictions:
            return np.array([])
            
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
        
    def _save_model(self, model_name: str, model: Any):
        """Save model to disk"""
        try:
            model_path = self.model_cache_path / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            # Save metrics
            metrics_path = self.model_cache_path / f"{model_name}_metrics.json"
            if model_name in self.model_metrics:
                metrics = self.model_metrics[model_name]
                metrics_data = {
                    "mape": metrics.mape,
                    "mae": metrics.mae,
                    "rmse": metrics.rmse,
                    "training_time": metrics.training_time,
                    "last_updated": metrics.last_updated.isoformat(),
                    "training_samples": metrics.training_samples
                }
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            
    def load_model(self, model_name: str) -> bool:
        """Load model from disk"""
        try:
            model_path = self.model_cache_path / f"{model_name}_model.pkl"
            if not model_path.exists():
                return False
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.models[model_name] = model
                
            # Load metrics
            metrics_path = self.model_cache_path / f"{model_name}_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    self.model_metrics[model_name] = ModelMetrics(
                        mape=metrics_data["mape"],
                        mae=metrics_data["mae"],
                        rmse=metrics_data["rmse"],
                        training_time=metrics_data["training_time"],
                        last_updated=datetime.fromisoformat(metrics_data["last_updated"]),
                        training_samples=metrics_data["training_samples"]
                    )
                    
            logger.info(f"Loaded model {model_name} from disk")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
            
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        for model_name in self.model_configs.keys():
            config = self.model_configs[model_name]
            metrics = self.model_metrics.get(model_name)
            
            status[model_name] = {
                "enabled": config.enabled,
                "loaded": model_name in self.models,
                "trained": metrics is not None,
                "needs_retraining": self.should_retrain_model(model_name),
                "metrics": {
                    "mape": metrics.mape if metrics else None,
                    "mae": metrics.mae if metrics else None,
                    "rmse": metrics.rmse if metrics else None,
                    "last_updated": metrics.last_updated.isoformat() if metrics else None,
                    "training_samples": metrics.training_samples if metrics else None
                }
            }
            
        return status
        
    def cleanup_old_models(self, days_old: int = 30):
        """Remove old model files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for file_path in self.model_cache_path.glob("*_model.pkl"):
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    logger.info(f"Removed old model file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")