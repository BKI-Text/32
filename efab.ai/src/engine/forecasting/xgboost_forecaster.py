"""
XGBoost Forecasting Model for Beverly Knits AI Supply Chain Planner

This module implements XGBoost ensemble forecasting for demand prediction
with automatic feature engineering and hyperparameter optimization.
"""

import logging
import warnings
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from ...core.domain.entities import Forecast, ForecastSource
from ...core.domain.value_objects import Quantity, SkuId

logger = logging.getLogger(__name__)

class XGBoostForecaster:
    """
    XGBoost ensemble forecasting model with comprehensive feature engineering
    for supply chain demand prediction.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 random_state: int = 42,
                 feature_engineering: bool = True,
                 hyperopt_enabled: bool = False):
        """
        Initialize XGBoost forecaster.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed for reproducibility
            feature_engineering: Whether to create additional features
            hyperopt_enabled: Whether to use hyperparameter optimization
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.feature_engineering = feature_engineering
        self.hyperopt_enabled = hyperopt_enabled
        
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = []
        self.feature_importance = {}
        self.best_params = None
        
    def _create_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Create comprehensive features for XGBoost model.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        if not self.feature_engineering:
            return df[[target_column]]
            
        # Lag features
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
        # Rolling statistics
        for window in [3, 7, 14, 30, 90]:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window).std()
            df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window).min()
            df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window).max()
            df[f'{target_column}_rolling_median_{window}'] = df[target_column].rolling(window).median()
            
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'{target_column}_ema_{alpha}'] = df[target_column].ewm(alpha=alpha).mean()
            
        # Temporal features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
            
        # Cyclical encoding for temporal features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            df['day_of_month_sin'] = np.sin(2 * np.pi * df.index.day / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * df.index.day / 31)
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            
        # Trend and momentum features
        df[f'{target_column}_trend_3'] = df[target_column].diff(3)
        df[f'{target_column}_trend_7'] = df[target_column].diff(7)
        df[f'{target_column}_trend_30'] = df[target_column].diff(30)
        
        # Momentum
        df[f'{target_column}_momentum_3'] = df[target_column] / df[target_column].shift(3) - 1
        df[f'{target_column}_momentum_7'] = df[target_column] / df[target_column].shift(7) - 1
        df[f'{target_column}_momentum_30'] = df[target_column] / df[target_column].shift(30) - 1
        
        # Volatility features
        df[f'{target_column}_volatility_7'] = df[target_column].rolling(7).std()
        df[f'{target_column}_volatility_30'] = df[target_column].rolling(30).std()
        
        # Relative position features
        for window in [7, 14, 30]:
            rolling_min = df[target_column].rolling(window).min()
            rolling_max = df[target_column].rolling(window).max()
            df[f'{target_column}_rel_pos_{window}'] = (df[target_column] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            
        # Seasonal features
        if len(df) >= 52:  # Weekly seasonality
            df[f'{target_column}_seasonal_7'] = df[target_column].shift(7)
            df[f'{target_column}_seasonal_14'] = df[target_column].shift(14)
            
        if len(df) >= 365:  # Yearly seasonality
            df[f'{target_column}_seasonal_365'] = df[target_column].shift(365)
            
        # Interaction features
        if f'{target_column}_lag_1' in df.columns and f'{target_column}_rolling_mean_7' in df.columns:
            df[f'{target_column}_lag1_x_mean7'] = df[f'{target_column}_lag_1'] * df[f'{target_column}_rolling_mean_7']
            
        # Statistical features
        df[f'{target_column}_skew_30'] = df[target_column].rolling(30).skew()
        df[f'{target_column}_kurt_30'] = df[target_column].rolling(30).kurt()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        self.feature_names = [col for col in df.columns if col != target_column]
        
        return df
        
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with best parameters
        """
        try:
            import optuna
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'random_state': self.random_state
                }
                
                model = xgb.XGBRegressor(**params)
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_percentage_error')
                return scores.mean()
                
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            return study.best_params
            
        except ImportError:
            logger.warning("Optuna not available for hyperparameter optimization")
            return {}
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
            
    def fit(self, data: pd.DataFrame, target_column: str = 'demand') -> 'XGBoostForecaster':
        """
        Fit XGBoost model to historical data.
        
        Args:
            data: DataFrame with datetime index and target column
            target_column: Name of the target column to forecast
            
        Returns:
            Self for method chaining
        """
        try:
            import xgboost as xgb
            
            logger.info("Fitting XGBoost model...")
            
            # Prepare data
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                else:
                    raise ValueError("Data must have datetime index or 'date' column")
                    
            # Create features
            featured_data = self._create_features(data, target_column)
            
            if len(featured_data) < 30:
                raise ValueError("Insufficient data for XGBoost modeling (need at least 30 points)")
                
            # Separate features and target
            X = featured_data.drop(columns=[target_column])
            y = featured_data[target_column]
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Optimize hyperparameters if enabled
            if self.hyperopt_enabled:
                self.best_params = self._optimize_hyperparameters(X_scaled, y)
                if self.best_params:
                    logger.info(f"Best parameters found: {self.best_params}")
                    params = self.best_params
                else:
                    params = {
                        'n_estimators': self.n_estimators,
                        'max_depth': self.max_depth,
                        'learning_rate': self.learning_rate,
                        'subsample': self.subsample,
                        'colsample_bytree': self.colsample_bytree,
                        'reg_alpha': self.reg_alpha,
                        'reg_lambda': self.reg_lambda,
                        'random_state': self.random_state
                    }
            else:
                params = {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample_bytree,
                    'reg_alpha': self.reg_alpha,
                    'reg_lambda': self.reg_lambda,
                    'random_state': self.random_state
                }
                
            # Train model
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_scaled, y)
            
            # Store feature importance
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            self.is_fitted = True
            
            logger.info(f"XGBoost model fitted successfully with {len(X.columns)} features")
            
            return self
            
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {e}")
            raise
            
    def predict(self, data: pd.DataFrame, target_column: str = 'demand') -> pd.DataFrame:
        """
        Generate forecasts using the fitted model.
        
        Args:
            data: DataFrame with same structure as training data
            target_column: Name of target column
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            logger.info("Generating XGBoost predictions...")
            
            # Create features
            featured_data = self._create_features(data, target_column)
            
            # Separate features
            X = featured_data.drop(columns=[target_column])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'ds': X.index,
                'forecast': predictions
            })
            
            # Calculate confidence intervals using quantile regression approach
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            prediction_std = np.std(predictions) * 0.1
            result_df['lower_ci'] = predictions - 1.96 * prediction_std
            result_df['upper_ci'] = predictions + 1.96 * prediction_std
            
            # Calculate confidence scores
            result_df['confidence_score'] = np.clip(
                1.0 - (result_df['upper_ci'] - result_df['lower_ci']) / result_df['forecast'],
                0.3, 1.0
            )
            
            # Ensure non-negative forecasts
            result_df['forecast'] = result_df['forecast'].clip(lower=0)
            result_df['lower_ci'] = result_df['lower_ci'].clip(lower=0)
            result_df['upper_ci'] = result_df['upper_ci'].clip(lower=0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating XGBoost predictions: {e}")
            raise
            
    def forecast_demand(self, 
                       historical_data: pd.DataFrame, 
                       periods: int = 30) -> List[Forecast]:
        """
        Generate domain forecast objects for supply chain planning.
        
        Args:
            historical_data: Historical demand data
            periods: Number of periods to forecast
            
        Returns:
            List of Forecast domain objects
        """
        try:
            # Fit model if not already fitted
            if not self.is_fitted:
                self.fit(historical_data)
                
            # Create future dates
            last_date = historical_data.index.max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            # Create future dataframe with last known values
            future_data = pd.DataFrame(index=future_dates)
            future_data['demand'] = historical_data['demand'].iloc[-1]  # Use last known value
            
            # Combine with historical data for feature creation
            extended_data = pd.concat([historical_data, future_data])
            
            # Generate predictions
            forecast_df = self.predict(extended_data)
            
            # Take only future predictions
            future_forecasts = forecast_df[forecast_df['ds'].isin(future_dates)]
            
            # Convert to domain objects
            forecasts = []
            
            for _, row in future_forecasts.iterrows():
                forecast = Forecast(
                    sku_id=SkuId(value="aggregate"),  # Aggregate forecast
                    forecast_qty=Quantity(
                        amount=max(0, row['forecast']),  # Ensure non-negative
                        unit="unit"
                    ),
                    forecast_date=row['ds'].date(),
                    source=ForecastSource.PROJECTION,
                    confidence_score=float(row['confidence_score']),
                    created_at=datetime.now()
                )
                
                forecasts.append(forecast)
                
            logger.info(f"Generated {len(forecasts)} XGBoost demand forecasts")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating demand forecasts: {e}")
            return []
            
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from fitted model.
        
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_fitted:
            return {}
            
        return dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
        
    def evaluate_model(self, test_data: pd.DataFrame, target_column: str = 'demand') -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset
            target_column: Name of target column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        try:
            # Generate predictions
            predictions_df = self.predict(test_data, target_column)
            
            # Get actual values
            actual_values = test_data[target_column].values
            predicted_values = predictions_df['forecast'].values
            
            # Align arrays
            min_len = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_len]
            predicted_values = predicted_values[:min_len]
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(actual_values, predicted_values)
            mae = np.mean(np.abs(actual_values - predicted_values))
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            
            return {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(np.corrcoef(actual_values, predicted_values)[0, 1] ** 2)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating XGBoost model: {e}")
            return {}
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and parameters.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
            
        return {
            "model_type": "XGBoost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "best_params": self.best_params,
            "hyperopt_enabled": self.hyperopt_enabled
        }
        
    def save_model(self, filepath: str):
        """Save fitted model to file"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        try:
            import pickle
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'best_params': self.best_params,
                'config': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample_bytree,
                    'reg_alpha': self.reg_alpha,
                    'reg_lambda': self.reg_lambda,
                    'random_state': self.random_state,
                    'feature_engineering': self.feature_engineering,
                    'hyperopt_enabled': self.hyperopt_enabled
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"XGBoost model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, filepath: str):
        """Load fitted model from file"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.best_params = model_data['best_params']
            
            # Restore config
            config = model_data['config']
            self.n_estimators = config['n_estimators']
            self.max_depth = config['max_depth']
            self.learning_rate = config['learning_rate']
            self.subsample = config['subsample']
            self.colsample_bytree = config['colsample_bytree']
            self.reg_alpha = config['reg_alpha']
            self.reg_lambda = config['reg_lambda']
            self.random_state = config['random_state']
            self.feature_engineering = config['feature_engineering']
            self.hyperopt_enabled = config['hyperopt_enabled']
            
            self.is_fitted = True
            logger.info(f"XGBoost model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise