#!/usr/bin/env python3
"""
Test ARIMA Model Training - Beverly Knits AI Supply Chain Planner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.engine.forecasting.arima_forecaster import ARIMAForecaster
from src.data.beverly_knits_live_data_integrator import BeverlyKnitsLiveDataIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_arima_training():
    """Test ARIMA model training with proper data format"""
    
    logger.info("🚀 Testing ARIMA Model Training")
    
    # Load data
    integrator = BeverlyKnitsLiveDataIntegrator(data_path="data/live/")
    
    try:
        # Load raw data
        sales_data = integrator.load_sales_data()
        demand_data = integrator.load_demand_data()
        
        logger.info(f"Loaded sales data: {len(sales_data)} records")
        logger.info(f"Loaded demand data: {len(demand_data)} records")
        
        # Prepare time series data correctly
        time_series_data = []
        
        # Process sales data
        if not sales_data.empty:
            for _, row in sales_data.iterrows():
                time_series_data.append({
                    'date': row['Transaction_Date'],
                    'demand': row['Quantity']
                })
        
        # Process demand data
        if not demand_data.empty:
            for _, row in demand_data.iterrows():
                time_series_data.append({
                    'date': row['Date'],
                    'demand': row['Quantity']
                })
        
        # Create DataFrame
        df = pd.DataFrame(time_series_data)
        
        if df.empty:
            logger.error("No time series data available")
            return
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate daily data
        daily_data = df.groupby(df['date'].dt.date).agg({
            'demand': 'sum'
        }).reset_index()
        
        # Rename columns correctly
        daily_data.columns = ['ds', 'y']
        daily_data['ds'] = pd.to_datetime(daily_data['ds'])
        
        # Sort by date
        daily_data = daily_data.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"Prepared time series data: {len(daily_data)} records")
        logger.info(f"Date range: {daily_data['ds'].min()} to {daily_data['ds'].max()}")
        logger.info(f"Average daily demand: {daily_data['y'].mean():.2f}")
        
        # Show first few rows
        logger.info("First 5 rows of training data:")
        logger.info(daily_data.head())
        
        # Initialize ARIMA forecaster
        forecaster = ARIMAForecaster()
        
        # Train the model
        logger.info("Training ARIMA model...")
        forecaster.fit(daily_data)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = forecaster.predict(steps=7)
        
        logger.info("✅ ARIMA model training successful!")
        logger.info(f"Predictions for next 7 days: {predictions}")
        
        # Save the model
        model_path = "models/arima_model.pkl"
        forecaster.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ARIMA training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_arima_training()
    sys.exit(0 if success else 1)