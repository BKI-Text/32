#!/usr/bin/env python3
"""
Test Automated Model Retraining Scheduler
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.scheduler import (
    ModelRetrainingScheduler,
    RetrainingConfig,
    ModelMetrics,
    RetrainingTrigger,
    get_model_scheduler,
    start_model_scheduler,
    stop_model_scheduler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockMLModel:
    """Mock ML model for testing"""
    
    def __init__(self, model_id: str, model_type: str):
        self.model_id = model_id
        self.model_type = model_type
        self.performance = 0.85  # Initial performance
        self.training_count = 0
        
    def predict(self, X):
        """Mock prediction with some noise"""
        # Add some noise to simulate model drift
        noise = np.random.normal(0, 0.1, size=len(X))
        return X + noise
        
    def evaluate(self, X, y) -> ModelMetrics:
        """Mock evaluation"""
        predictions = self.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        # Simulate performance degradation over time
        self.performance = max(0.5, self.performance - 0.02)
        
        return ModelMetrics(
            model_id=self.model_id,
            model_type=self.model_type,
            accuracy=self.performance,
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2_score=r2,
            last_updated=datetime.now(),
            data_points=len(X)
        )
        
    def retrain(self) -> ModelMetrics:
        """Mock retraining"""
        logger.info(f"🔄 Retraining model {self.model_id}")
        self.training_count += 1
        
        # Simulate training time
        time.sleep(1)
        
        # Improve performance after retraining
        self.performance = min(0.95, self.performance + 0.15)
        
        # Generate synthetic data for evaluation
        X = np.random.randn(100)
        y = X + np.random.normal(0, 0.1, 100)
        
        return self.evaluate(X, y)

def create_mock_retraining_function(model: MockMLModel):
    """Create a mock retraining function"""
    def retrain_model():
        return model.retrain()
    return retrain_model

async def test_model_retraining_scheduler():
    """Test the model retraining scheduler"""
    logger.info("🚀 Testing Automated Model Retraining Scheduler")
    
    try:
        # Create scheduler
        scheduler = ModelRetrainingScheduler(data_dir="data/test_models")
        
        # Create mock models
        arima_model = MockMLModel("ARIMA_001", "ARIMA")
        prophet_model = MockMLModel("PROPHET_001", "Prophet")
        lstm_model = MockMLModel("LSTM_001", "LSTM")
        
        # Create retraining configurations
        arima_config = RetrainingConfig(
            model_id="ARIMA_001",
            model_type="ARIMA",
            performance_threshold=0.7,
            max_age_days=1,  # Short for testing
            retraining_interval_hours=1,
            enabled=True
        )
        
        prophet_config = RetrainingConfig(
            model_id="PROPHET_001",
            model_type="Prophet",
            performance_threshold=0.75,
            data_drift_threshold=0.2,
            max_age_days=2,
            enabled=True
        )
        
        lstm_config = RetrainingConfig(
            model_id="LSTM_001",
            model_type="LSTM",
            performance_threshold=0.8,
            data_drift_threshold=0.15,
            max_age_days=3,
            enabled=True
        )
        
        # Register models with scheduler
        scheduler.register_model(arima_config, create_mock_retraining_function(arima_model))
        scheduler.register_model(prophet_config, create_mock_retraining_function(prophet_model))
        scheduler.register_model(lstm_config, create_mock_retraining_function(lstm_model))
        
        logger.info("Registered 3 models with scheduler")
        
        # Start scheduler
        scheduler.start()
        
        logger.info("✅ Scheduler started successfully")
        
        # Simulate model performance updates
        logger.info("Simulating model performance updates...")
        
        # Generate synthetic data
        X = np.random.randn(100)
        y = X + np.random.normal(0, 0.1, 100)
        
        # Update metrics for each model (simulate degradation)
        for i in range(5):
            logger.info(f"Performance update cycle {i+1}")
            
            # Evaluate models and update metrics
            arima_metrics = arima_model.evaluate(X, y)
            prophet_metrics = prophet_model.evaluate(X, y)
            lstm_metrics = lstm_model.evaluate(X, y)
            
            # Update scheduler with metrics
            scheduler.update_model_metrics("ARIMA_001", arima_metrics)
            scheduler.update_model_metrics("PROPHET_001", prophet_metrics)
            scheduler.update_model_metrics("LSTM_001", lstm_metrics)
            
            logger.info(f"ARIMA performance: {arima_metrics.accuracy:.3f}")
            logger.info(f"Prophet performance: {prophet_metrics.accuracy:.3f}")
            logger.info(f"LSTM performance: {lstm_metrics.accuracy:.3f}")
            
            # Wait between updates
            await asyncio.sleep(2)
            
        # Check scheduler status
        status = scheduler.get_status()
        logger.info(f"Scheduler status: {status}")
        
        # Wait for scheduler to detect degradation and trigger retraining
        logger.info("Waiting for scheduler to detect performance degradation...")
        await asyncio.sleep(5)
        
        # Check for scheduled jobs
        job_history = scheduler.get_job_history()
        logger.info(f"Job history: {len(job_history)} jobs")
        
        for job in job_history:
            logger.info(f"Job {job['job_id']}: {job['status']} (trigger: {job['trigger']})")
            
        # Force trigger retraining for testing
        logger.info("Manually triggering retraining...")
        
        scheduler.schedule_retraining("ARIMA_001", RetrainingTrigger.MANUAL)
        scheduler.schedule_retraining("PROPHET_001", RetrainingTrigger.PERFORMANCE_DEGRADATION)
        scheduler.schedule_retraining("LSTM_001", RetrainingTrigger.DATA_DRIFT)
        
        # Wait for jobs to complete
        await asyncio.sleep(10)
        
        # Check final status
        final_status = scheduler.get_status()
        logger.info(f"Final scheduler status: {final_status}")
        
        final_job_history = scheduler.get_job_history()
        logger.info(f"Final job history: {len(final_job_history)} jobs")
        
        completed_jobs = [job for job in final_job_history if job['status'] == 'completed']
        logger.info(f"Completed jobs: {len(completed_jobs)}")
        
        # Stop scheduler
        scheduler.stop()
        
        logger.info("✅ Model retraining scheduler test completed successfully!")
        
        # Show final results
        logger.info("\n=== TEST RESULTS ===")
        logger.info(f"Registered models: {final_status['registered_models']}")
        logger.info(f"Completed jobs: {final_status['completed_jobs']}")
        logger.info(f"Failed jobs: {final_status['failed_jobs']}")
        
        for model_id, model_info in final_status['models'].items():
            logger.info(f"\nModel {model_id}:")
            if model_info['last_metrics']:
                logger.info(f"  Last accuracy: {model_info['last_metrics']['accuracy']:.3f}")
                logger.info(f"  Last updated: {model_info['last_metrics']['last_updated']}")
            logger.info(f"  Needs retraining: {model_info['needs_retraining']}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Model retraining scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_global_scheduler():
    """Test the global scheduler instance"""
    logger.info("🚀 Testing Global Model Scheduler")
    
    try:
        # Get global scheduler
        global_scheduler = get_model_scheduler()
        
        # Create a simple model for testing
        test_model = MockMLModel("GLOBAL_TEST", "Test")
        
        # Create configuration
        config = RetrainingConfig(
            model_id="GLOBAL_TEST",
            model_type="Test",
            performance_threshold=0.8,
            max_age_days=1,
            enabled=True
        )
        
        # Register model
        global_scheduler.register_model(config, create_mock_retraining_function(test_model))
        
        # Start global scheduler
        start_model_scheduler()
        
        logger.info("✅ Global scheduler started")
        
        # Add some metrics
        X = np.random.randn(50)
        y = X + np.random.normal(0, 0.1, 50)
        
        for i in range(3):
            metrics = test_model.evaluate(X, y)
            global_scheduler.update_model_metrics("GLOBAL_TEST", metrics)
            logger.info(f"Updated metrics - accuracy: {metrics.accuracy:.3f}")
            await asyncio.sleep(1)
            
        # Check status
        status = global_scheduler.get_status()
        logger.info(f"Global scheduler status: {status}")
        
        # Stop global scheduler
        stop_model_scheduler()
        
        logger.info("✅ Global scheduler test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Global scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Running Model Retraining Scheduler Tests")
    
    # Test 1: Model retraining scheduler
    test1_success = await test_model_retraining_scheduler()
    
    # Test 2: Global scheduler
    test2_success = await test_global_scheduler()
    
    # Overall result
    if test1_success and test2_success:
        logger.info("✅ All model retraining scheduler tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)