#!/usr/bin/env python3
"""
Simple LSTM Model Test - Beverly Knits AI Supply Chain Planner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.engine.forecasting.lstm_forecaster import LSTMForecaster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lstm_with_synthetic_data():
    """Test LSTM model with synthetic data"""
    
    logger.info("🚀 Testing LSTM Model with Synthetic Data")
    
    # Create synthetic time series data (need plenty of data for LSTM)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    # Generate synthetic demand with trend and seasonality
    trend = np.linspace(1000, 1500, 300)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(300) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 50, 300)
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
        # Initialize LSTM forecaster
        forecaster = LSTMForecaster(
            sequence_length=30,  # Use 30 days of history
            epochs=5,  # Quick training for test
            batch_size=32
        )
        
        # Train the model
        logger.info("Training LSTM model...")
        forecaster.fit(df)
        
        # Make predictions
        logger.info("Making predictions...")
        # For LSTM, we need the last sequence from training data
        # Let's use the last 30 days of training data and reshape properly
        last_sequence = df['demand'].values[-30:].reshape(30, 1)  # Shape: (30, 1)
        predictions = forecaster.predict(periods=7, last_sequence=last_sequence)
        
        logger.info("✅ LSTM model training successful!")
        logger.info(f"Predictions for next 7 days:")
        logger.info(predictions)
        
        # Model trained successfully
        logger.info("LSTM model is ready for predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LSTM training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lstm_with_synthetic_data()
    sys.exit(0 if success else 1)