"""
ARIMA Forecasting Model for Beverly Knits AI Supply Chain Planner

This module implements ARIMA (AutoRegressive Integrated Moving Average) 
time series forecasting for demand prediction with seasonality detection.
"""

import logging
import warnings
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from ...core.domain.entities import Forecast, ForecastSource
from ...core.domain.value_objects import Quantity, SkuId

logger = logging.getLogger(__name__)

# Suppress statsmodels warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ARIMAForecaster:
    """
    ARIMA time series forecasting model with automatic parameter selection
    and seasonality detection.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 auto_arima: bool = True,
                 seasonal: bool = True,
                 max_p: int = 3,
                 max_q: int = 3,
                 max_d: int = 2):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: Non-seasonal ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_arima: Whether to automatically select best parameters
            seasonal: Whether to include seasonal components
            max_p: Maximum AR order for auto selection
            max_q: Maximum MA order for auto selection
            max_d: Maximum differencing order for auto selection
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima
        self.seasonal = seasonal
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.best_params = None
        self.aic_score = None
        self.bic_score = None
        
    def _check_stationarity(self, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (is_stationary, test_results)
        """
        try:
            result = adfuller(data.dropna())
            
            test_results = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
            
            return result[1] < 0.05, test_results
            
        except Exception as e:
            logger.error(f"Error checking stationarity: {e}")
            return False, {}
            
    def _difference_series(self, data: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Difference the series to make it stationary.
        
        Args:
            data: Time series data
            max_diff: Maximum number of differencing operations
            
        Returns:
            Tuple of (differenced_series, num_differences)
        """
        current_data = data.copy()
        num_diff = 0
        
        for i in range(max_diff):
            is_stationary, _ = self._check_stationarity(current_data)
            if is_stationary:
                break
                
            current_data = current_data.diff().dropna()
            num_diff += 1
            
        return current_data, num_diff
        
    def _detect_seasonality(self, data: pd.Series) -> Tuple[bool, int]:
        """
        Detect seasonality in time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (has_seasonality, seasonal_period)
        """
        try:
            # Try different seasonal periods
            periods_to_test = [12, 4, 7, 52]  # Monthly, Quarterly, Weekly, Yearly
            
            for period in periods_to_test:
                if len(data) >= 2 * period:
                    try:
                        decomposition = seasonal_decompose(
                            data, 
                            model='additive', 
                            period=period
                        )
                        
                        # Check if seasonal component has significant variation
                        seasonal_var = decomposition.seasonal.var()
                        total_var = data.var()
                        
                        if seasonal_var > 0.1 * total_var:
                            logger.info(f"Detected seasonality with period {period}")
                            return True, period
                            
                    except Exception as e:
                        logger.debug(f"Error testing seasonality with period {period}: {e}")
                        continue
                        
            return False, 12  # Default to monthly if no seasonality detected
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return False, 12
            
    def _auto_select_parameters(self, data: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically select best ARIMA parameters using AIC/BIC.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of best (p, d, q) parameters
        """
        logger.info("Auto-selecting ARIMA parameters...")
        
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        # Determine differencing order
        _, d = self._difference_series(data, self.max_d)
        
        # Grid search for p and q
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    temp_model = ARIMA(data, order=(p, d, q))
                    temp_fitted = temp_model.fit()
                    
                    if temp_fitted.aic < best_aic:
                        best_aic = temp_fitted.aic
                        best_params = (p, d, q)
                        
                except Exception as e:
                    logger.debug(f"Error fitting ARIMA({p},{d},{q}): {e}")
                    continue
                    
        logger.info(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        return best_params
        
    def fit(self, data: pd.DataFrame, target_column: str = 'demand') -> 'ARIMAForecaster':
        """
        Fit ARIMA model to historical data.
        
        Args:
            data: DataFrame with datetime index and target column
            target_column: Name of the target column to forecast
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info("Fitting ARIMA model...")
            
            # Prepare data
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                else:
                    raise ValueError("Data must have datetime index or 'date' column")
                    
            series = data[target_column].dropna()
            
            if len(series) < 10:
                raise ValueError("Insufficient data for ARIMA modeling (need at least 10 points)")
                
            # Check for seasonality
            has_seasonality, seasonal_period = self._detect_seasonality(series)
            
            # Auto-select parameters if enabled
            if self.auto_arima:
                self.best_params = self._auto_select_parameters(series)
                order_to_use = self.best_params
            else:
                order_to_use = self.order
                
            # Fit ARIMA model
            if self.seasonal and has_seasonality:
                seasonal_order = (
                    self.seasonal_order[0],
                    self.seasonal_order[1], 
                    self.seasonal_order[2],
                    seasonal_period
                )
                self.model = ARIMA(
                    series, 
                    order=order_to_use,
                    seasonal_order=seasonal_order
                )
            else:
                self.model = ARIMA(series, order=order_to_use)
                
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # Store model statistics
            self.aic_score = self.fitted_model.aic
            self.bic_score = self.fitted_model.bic
            
            logger.info(f"ARIMA model fitted successfully. AIC: {self.aic_score:.2f}, BIC: {self.bic_score:.2f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
            
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Generate forecasts for specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            logger.info(f"Generating {periods} period ARIMA forecast...")
            
            # Generate forecasts
            forecast_result = self.fitted_model.get_forecast(steps=periods)
            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast': forecast_mean,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1]
            })
            
            # Add confidence scores based on confidence interval width
            ci_width = forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]
            max_width = ci_width.max()
            confidence_scores = 1.0 - (ci_width / max_width) * 0.5  # 0.5 to 1.0 scale
            forecast_df['confidence_score'] = confidence_scores
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating ARIMA predictions: {e}")
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
                
            # Generate predictions
            forecast_df = self.predict(periods)
            
            # Convert to domain objects
            forecasts = []
            base_date = historical_data.index.max()
            
            for i, (date, row) in enumerate(forecast_df.iterrows()):
                forecast_date = base_date + timedelta(days=i + 1)
                
                forecast = Forecast(
                    sku_id=SkuId(value="aggregate"),  # Aggregate forecast
                    forecast_qty=Quantity(
                        amount=max(0, row['forecast']),  # Ensure non-negative
                        unit="unit"
                    ),
                    forecast_date=forecast_date.date(),
                    source=ForecastSource.PROJECTION,
                    confidence_score=float(row['confidence_score']),
                    created_at=datetime.now()
                )
                
                forecasts.append(forecast)
                
            logger.info(f"Generated {len(forecasts)} ARIMA demand forecasts")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating demand forecasts: {e}")
            return []
            
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostic information.
        
        Returns:
            Dictionary with model diagnostics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
            
        try:
            residuals = self.fitted_model.resid
            
            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            diagnostics = {
                "model_order": self.fitted_model.model.order,
                "seasonal_order": getattr(self.fitted_model.model, 'seasonal_order', None),
                "aic": self.aic_score,
                "bic": self.bic_score,
                "log_likelihood": self.fitted_model.llf,
                "residual_stats": {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "skewness": float(residuals.skew()),
                    "kurtosis": float(residuals.kurtosis())
                },
                "ljung_box_test": {
                    "p_values": lb_test['lb_pvalue'].tolist(),
                    "residuals_white_noise": (lb_test['lb_pvalue'] > 0.05).all()
                }
            }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting model diagnostics: {e}")
            return {"error": str(e)}
            
    def evaluate_forecast_accuracy(self, 
                                 actual_data: pd.Series, 
                                 forecast_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using multiple metrics.
        
        Args:
            actual_data: Actual observed values
            forecast_data: Forecasted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            # Align data
            common_index = actual_data.index.intersection(forecast_data.index)
            actual_aligned = actual_data.loc[common_index]
            forecast_aligned = forecast_data.loc[common_index]
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(actual_aligned, forecast_aligned)
            mae = np.mean(np.abs(actual_aligned - forecast_aligned))
            rmse = np.sqrt(mean_squared_error(actual_aligned, forecast_aligned))
            
            # Additional metrics
            mse = mean_squared_error(actual_aligned, forecast_aligned)
            bias = np.mean(forecast_aligned - actual_aligned)
            
            metrics = {
                "mape": float(mape),
                "mae": float(mae),
                "rmse": float(rmse),
                "mse": float(mse),
                "bias": float(bias),
                "r2": float(np.corrcoef(actual_aligned, forecast_aligned)[0, 1] ** 2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating forecast accuracy: {e}")
            return {}
            
    def save_model(self, filepath: str):
        """Save fitted model to file"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        try:
            self.fitted_model.save(filepath)
            logger.info(f"ARIMA model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, filepath: str):
        """Load fitted model from file"""
        try:
            from statsmodels.tsa.arima.model import ARIMAResults
            self.fitted_model = ARIMAResults.load(filepath)
            self.is_fitted = True
            logger.info(f"ARIMA model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise