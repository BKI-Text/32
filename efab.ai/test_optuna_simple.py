#!/usr/bin/env python3
"""
Simple Optuna Hyperparameter Optimization Test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import optuna
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    """Simple objective function for Optuna testing"""
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

def test_optuna_optimization():
    """Test Optuna hyperparameter optimization"""
    
    logger.info("🚀 Testing Optuna Hyperparameter Optimization")
    
    try:
        # Create study
        study = optuna.create_study(direction='minimize')
        
        # Optimize
        logger.info("Running optimization...")
        study.optimize(objective, n_trials=10)
        
        # Get best result
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info("✅ Optuna optimization successful!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optuna optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optuna_optimization()
    sys.exit(0 if success else 1)