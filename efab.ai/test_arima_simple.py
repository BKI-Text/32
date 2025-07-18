#!/usr/bin/env python3
"""
Simple ARIMA Model Test - Beverly Knits AI Supply Chain Planner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.engine.forecasting.arima_forecaster import ARIMAForecaster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_arima_with_synthetic_data():
    """Test ARIMA model with synthetic data"""
    
    logger.info("🚀 Testing ARIMA Model with Synthetic Data")
    
    # Create synthetic time series data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate synthetic demand with trend and seasonality
    trend = np.linspace(1000, 1500, 100)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 50, 100)
    demand = trend + seasonal + noise
    
    # Create DataFrame in correct format
    df = pd.DataFrame({
        'date': dates,
        'demand': demand
    })
    
    logger.info(f"Created synthetic data: {len(df)} records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Average demand: {df['demand'].mean():.2f}")
    
    # Show first few rows
    logger.info("First 5 rows of training data:")
    logger.info(df.head())
    
    try:
        # Initialize ARIMA forecaster
        forecaster = ARIMAForecaster()
        
        # Train the model
        logger.info("Training ARIMA model...")
        forecaster.fit(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = forecaster.predict(periods=7)
        
        logger.info("✅ ARIMA model training successful!")
        logger.info(f"Predictions for next 7 days: {predictions}")
        
        # Save the model
        model_path = "models/arima_test_model.pkl"
        os.makedirs("models", exist_ok=True)
        forecaster.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ARIMA training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_arima_with_synthetic_data()
    sys.exit(0 if success else 1)