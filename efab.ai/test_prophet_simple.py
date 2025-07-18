#!/usr/bin/env python3
"""
Simple Prophet Model Test - Beverly Knits AI Supply Chain Planner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.engine.forecasting.prophet_forecaster import ProphetForecaster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prophet_with_synthetic_data():
    """Test Prophet model with synthetic data"""
    
    logger.info("🚀 Testing Prophet Model with Synthetic Data")
    
    # Create synthetic time series data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate synthetic demand with trend and seasonality
    trend = np.linspace(1000, 1500, 100)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 50, 100)
    demand = trend + seasonal + noise
    
    # Create DataFrame in correct format (date, demand)
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
        # Initialize Prophet forecaster
        forecaster = ProphetForecaster()
        
        # Train the model
        logger.info("Training Prophet model...")
        forecaster.fit(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = forecaster.predict(periods=7)
        
        logger.info("✅ Prophet model training successful!")
        logger.info(f"Predictions for next 7 days:")
        logger.info(predictions)
        
        # Model trained successfully
        logger.info("Prophet model is ready for predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prophet training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prophet_with_synthetic_data()
    sys.exit(0 if success else 1)