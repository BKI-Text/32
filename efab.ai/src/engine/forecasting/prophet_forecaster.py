"""
Prophet Forecasting Model for Beverly Knits AI Supply Chain Planner

This module implements Facebook Prophet forecasting model for demand prediction
with automatic seasonality detection and holiday effects.
"""

import logging
import warnings
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from ...core.domain.entities import Forecast, ForecastSource
from ...core.domain.value_objects import Quantity, SkuId

logger = logging.getLogger(__name__)

# Suppress Prophet warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ProphetForecaster:
    """
    Facebook Prophet forecasting model with automatic seasonality detection
    and holiday effects for supply chain demand forecasting.
    """
    
    def __init__(self,
                 seasonality_mode: str = 'multiplicative',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 holidays: Optional[pd.DataFrame] = None,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 changepoint_prior_scale: float = 0.05,
                 mcmc_samples: int = 0,
                 interval_width: float = 0.80,
                 uncertainty_samples: int = 1000):
        """
        Initialize Prophet forecaster.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative' seasonality
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            holidays: DataFrame with holiday dates and names
            seasonality_prior_scale: Prior scale for seasonality
            holidays_prior_scale: Prior scale for holidays
            changepoint_prior_scale: Prior scale for changepoints
            mcmc_samples: Number of MCMC samples (0 for MAP estimation)
            interval_width: Width of uncertainty intervals
            uncertainty_samples: Number of samples for uncertainty
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        
        self.model = None
        self.is_fitted = False
        self.training_data = None
        
    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        
        Args:
            data: Input DataFrame with datetime index
            target_column: Name of target column
            
        Returns:
            DataFrame formatted for Prophet
        """
        # Reset index to get datetime as column
        if isinstance(data.index, pd.DatetimeIndex):
            prepared_data = data.reset_index()
            date_column = prepared_data.columns[0]
        else:
            prepared_data = data.copy()
            date_column = 'date'
            
        # Rename columns for Prophet
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(prepared_data[date_column]),
            'y': prepared_data[target_column]
        })
        
        # Remove any NaN values
        prophet_data = prophet_data.dropna()
        
        # Ensure positive values (Prophet works better with positive data)
        prophet_data['y'] = prophet_data['y'].clip(lower=0)
        
        return prophet_data
        
    def _create_business_holidays(self) -> pd.DataFrame:
        """
        Create business-relevant holidays for textile manufacturing.
        
        Returns:
            DataFrame with holiday dates and names
        """
        holidays = pd.DataFrame({
            'holiday': [
                'New Year', 'Labor Day', 'Independence Day', 'Christmas',
                'Thanksgiving', 'Black Friday', 'Cyber Monday'
            ],
            'ds': pd.to_datetime([
                '2023-01-01', '2023-09-04', '2023-07-04', '2023-12-25',
                '2023-11-23', '2023-11-24', '2023-11-27'
            ])
        })
        
        # Add more years
        for year in range(2024, 2027):
            yearly_holidays = pd.DataFrame({
                'holiday': [
                    'New Year', 'Labor Day', 'Independence Day', 'Christmas'
                ],
                'ds': pd.to_datetime([
                    f'{year}-01-01', f'{year}-09-04', f'{year}-07-04', f'{year}-12-25'
                ])
            })
            holidays = pd.concat([holidays, yearly_holidays], ignore_index=True)
            
        return holidays
        
    def fit(self, data: pd.DataFrame, target_column: str = 'demand') -> 'ProphetForecaster':
        """
        Fit Prophet model to historical data.
        
        Args:
            data: DataFrame with datetime index and target column
            target_column: Name of the target column to forecast
            
        Returns:
            Self for method chaining
        """
        try:
            # Import Prophet (lazy import to avoid dependency issues)
            from prophet import Prophet
            
            logger.info("Fitting Prophet model...")
            
            # Prepare data
            prophet_data = self._prepare_data(data, target_column)
            
            if len(prophet_data) < 10:
                raise ValueError("Insufficient data for Prophet modeling (need at least 10 points)")
                
            # Create holidays if not provided
            if self.holidays is None:
                self.holidays = self._create_business_holidays()
                
            # Initialize Prophet model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=self.holidays,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                changepoint_prior_scale=self.changepoint_prior_scale,
                mcmc_samples=self.mcmc_samples,
                interval_width=self.interval_width,
                uncertainty_samples=self.uncertainty_samples
            )
            
            # Add custom seasonalities for textile business
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            self.model.add_seasonality(
                name='quarterly',
                period=365.25/4,
                fourier_order=3
            )
            
            # Fit the model
            self.model.fit(prophet_data)
            self.is_fitted = True
            self.training_data = prophet_data
            
            logger.info("Prophet model fitted successfully")
            
            return self
            
        except ImportError:
            logger.error("Prophet not installed. Install with: pip install prophet")
            raise
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
            
    def predict(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate forecasts for specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            logger.info(f"Generating {periods} period Prophet forecast...")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate forecasts
            forecast = self.model.predict(future)
            
            # Extract forecast period only (not historical)
            forecast_period = forecast.tail(periods)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'ds': forecast_period['ds'],
                'forecast': forecast_period['yhat'],
                'lower_ci': forecast_period['yhat_lower'],
                'upper_ci': forecast_period['yhat_upper'],
                'trend': forecast_period['trend'],
                'seasonal': forecast_period.get('seasonal', 0),
                'yearly': forecast_period.get('yearly', 0),
                'weekly': forecast_period.get('weekly', 0),
                'monthly': forecast_period.get('monthly', 0),
                'quarterly': forecast_period.get('quarterly', 0)
            })
            
            # Calculate confidence scores based on uncertainty interval width
            ci_width = result_df['upper_ci'] - result_df['lower_ci']
            max_width = ci_width.max()
            
            # Confidence score: narrower intervals = higher confidence
            result_df['confidence_score'] = np.clip(
                1.0 - (ci_width / max_width) * 0.5, 0.3, 1.0
            )
            
            # Ensure forecasts are non-negative
            result_df['forecast'] = result_df['forecast'].clip(lower=0)
            result_df['lower_ci'] = result_df['lower_ci'].clip(lower=0)
            result_df['upper_ci'] = result_df['upper_ci'].clip(lower=0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating Prophet predictions: {e}")
            raise
            
    def forecast_demand(self, 
                       historical_data: pd.DataFrame, 
                       periods: int = 30,
                       freq: str = 'D') -> List[Forecast]:
        """
        Generate domain forecast objects for supply chain planning.
        
        Args:
            historical_data: Historical demand data
            periods: Number of periods to forecast
            freq: Frequency of predictions
            
        Returns:
            List of Forecast domain objects
        """
        try:
            # Fit model if not already fitted
            if not self.is_fitted:
                self.fit(historical_data)
                
            # Generate predictions
            forecast_df = self.predict(periods, freq)
            
            # Convert to domain objects
            forecasts = []
            
            for _, row in forecast_df.iterrows():
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
                
            logger.info(f"Generated {len(forecasts)} Prophet demand forecasts")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating demand forecasts: {e}")
            return []
            
    def get_forecast_components(self, periods: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get detailed forecast components (trend, seasonality, holidays).
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with component DataFrames
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing components")
            
        try:
            # Generate future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            
            # Extract components
            components = {
                'trend': forecast[['ds', 'trend']].copy(),
                'seasonal': forecast[['ds', 'yearly', 'weekly']].copy(),
                'holidays': forecast[['ds', 'holidays']].copy() if 'holidays' in forecast.columns else None
            }
            
            # Add custom seasonalities if present
            if 'monthly' in forecast.columns:
                components['monthly'] = forecast[['ds', 'monthly']].copy()
            if 'quarterly' in forecast.columns:
                components['quarterly'] = forecast[['ds', 'quarterly']].copy()
                
            return components
            
        except Exception as e:
            logger.error(f"Error getting forecast components: {e}")
            return {}
            
    def detect_changepoints(self) -> pd.DataFrame:
        """
        Detect significant changepoints in the time series.
        
        Returns:
            DataFrame with changepoint dates and their effects
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting changepoints")
            
        try:
            # Get changepoints
            changepoints = self.model.changepoints
            
            # Get changepoint effects
            future = self.model.make_future_dataframe(periods=0)
            forecast = self.model.predict(future)
            
            # Calculate changepoint effects
            changepoint_effects = []
            for i, cp in enumerate(changepoints):
                effect = forecast.loc[forecast['ds'] == cp, 'trend'].iloc[0]
                changepoint_effects.append({
                    'changepoint': cp,
                    'effect': effect
                })
                
            return pd.DataFrame(changepoint_effects)
            
        except Exception as e:
            logger.error(f"Error detecting changepoints: {e}")
            return pd.DataFrame()
            
    def evaluate_forecast_accuracy(self, 
                                 actual_data: pd.DataFrame,
                                 target_column: str = 'demand') -> Dict[str, float]:
        """
        Evaluate forecast accuracy using cross-validation.
        
        Args:
            actual_data: Actual observed data
            target_column: Name of target column
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Prepare data
            prophet_data = self._prepare_data(actual_data, target_column)
            
            # Perform cross-validation
            cv_results = cross_validation(
                self.model, 
                horizon='30 days',
                period='15 days',
                initial='90 days'
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            # Return summary metrics
            return {
                'mape': float(metrics['mape'].mean()),
                'mae': float(metrics['mae'].mean()),
                'rmse': float(metrics['rmse'].mean()),
                'coverage': float(metrics['coverage'].mean())
            }
            
        except Exception as e:
            logger.error(f"Error evaluating forecast accuracy: {e}")
            # Fallback to simple accuracy calculation
            return self._simple_accuracy_evaluation(actual_data, target_column)
            
    def _simple_accuracy_evaluation(self, 
                                   actual_data: pd.DataFrame,
                                   target_column: str) -> Dict[str, float]:
        """
        Simple accuracy evaluation without cross-validation.
        
        Args:
            actual_data: Actual observed data
            target_column: Name of target column
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            # Use last 30 days for evaluation
            test_data = actual_data.tail(30)
            train_data = actual_data.iloc[:-30]
            
            # Fit on training data
            temp_model = ProphetForecaster(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality
            )
            temp_model.fit(train_data, target_column)
            
            # Generate predictions
            predictions = temp_model.predict(30)
            
            # Calculate accuracy
            actual_values = test_data[target_column].values
            predicted_values = predictions['forecast'].values[:len(actual_values)]
            
            mape = mean_absolute_percentage_error(actual_values, predicted_values)
            mae = np.mean(np.abs(actual_values - predicted_values))
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            
            return {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse),
                'coverage': 0.8  # Default coverage
            }
            
        except Exception as e:
            logger.error(f"Error in simple accuracy evaluation: {e}")
            return {}
            
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters and configuration.
        
        Returns:
            Dictionary with model parameters
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
            
        try:
            return {
                "seasonality_mode": self.seasonality_mode,
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "daily_seasonality": self.daily_seasonality,
                "seasonality_prior_scale": self.seasonality_prior_scale,
                "holidays_prior_scale": self.holidays_prior_scale,
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "interval_width": self.interval_width,
                "training_data_points": len(self.training_data) if self.training_data is not None else 0,
                "changepoints_detected": len(self.model.changepoints) if self.model else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            return {"error": str(e)}
            
    def plot_forecast(self, periods: int = 30) -> Any:
        """
        Plot forecast with components (requires matplotlib).
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Matplotlib figure object
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        try:
            import matplotlib.pyplot as plt
            
            # Generate forecast
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            
            # Create plot
            fig = self.model.plot(forecast)
            plt.title("Prophet Demand Forecast")
            plt.xlabel("Date")
            plt.ylabel("Demand")
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            return None