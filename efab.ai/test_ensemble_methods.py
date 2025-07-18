#!/usr/bin/env python3
"""
Test Advanced Ensemble Methods
Beverly Knits AI Supply Chain Planner
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ensemble import (
    StackingEnsemble,
    BlendingEnsemble,
    VotingEnsemble,
    EnsembleOptimizer,
    evaluate_ensemble,
    create_demand_forecasting_ensemble
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_supply_chain_data(n_samples: int = 1000, n_features: int = 10, 
                                      noise: float = 0.1, random_state: int = 42):
    """Create synthetic supply chain data for testing"""
    np.random.seed(random_state)
    
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Add supply chain specific patterns
    # Seasonal patterns
    seasonal_component = np.sin(np.arange(n_samples) * 2 * np.pi / 12) * 50
    
    # Trend component
    trend_component = np.arange(n_samples) * 0.5
    
    # Add to target
    y = y + seasonal_component + trend_component
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X, y, feature_names

def test_stacking_ensemble():
    """Test stacking ensemble"""
    logger.info("🚀 Testing Stacking Ensemble")
    
    try:
        # Create synthetic data
        X, y, feature_names = create_synthetic_supply_chain_data(n_samples=500)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create base models
        base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        # Create stacking ensemble
        stacking_ensemble = StackingEnsemble(
            base_models=base_models,
            cv_folds=3,
            use_features=True,
            random_state=42
        )
        
        # Train ensemble
        logger.info("Training stacking ensemble...")
        stacking_ensemble.fit(X_train, y_train)
        
        # Make predictions
        predictions = stacking_ensemble.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Stacking Ensemble - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Get feature importance
        importance = stacking_ensemble.get_feature_importance()
        logger.info(f"Feature importance: {importance}")
        
        # Test individual base models for comparison
        base_scores = {}
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            base_scores[name] = {
                'mse': mean_squared_error(y_test, pred),
                'r2': r2_score(y_test, pred)
            }
            
        logger.info("Base model scores:")
        for name, scores in base_scores.items():
            logger.info(f"  {name}: MSE={scores['mse']:.4f}, R²={scores['r2']:.4f}")
            
        logger.info("✅ Stacking ensemble test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Stacking ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_blending_ensemble():
    """Test blending ensemble"""
    logger.info("🚀 Testing Blending Ensemble")
    
    try:
        # Create synthetic data
        X, y, feature_names = create_synthetic_supply_chain_data(n_samples=500)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create base models
        base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        # Test different blending methods
        blend_methods = ['linear', 'weighted', 'stacking']
        
        for blend_method in blend_methods:
            logger.info(f"Testing blending method: {blend_method}")
            
            # Create blending ensemble
            blending_ensemble = BlendingEnsemble(
                base_models=base_models,
                blend_method=blend_method,
                holdout_ratio=0.2,
                random_state=42
            )
            
            # Train ensemble
            blending_ensemble.fit(X_train, y_train)
            
            # Make predictions
            predictions = blending_ensemble.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            logger.info(f"  {blend_method} - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Get blend weights
            weights = blending_ensemble.get_blend_weights()
            logger.info(f"  Blend weights: {weights}")
            
        logger.info("✅ Blending ensemble test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Blending ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voting_ensemble():
    """Test voting ensemble"""
    logger.info("🚀 Testing Voting Ensemble")
    
    try:
        # Create synthetic data
        X, y, feature_names = create_synthetic_supply_chain_data(n_samples=500)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create base models
        base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        # Test different voting methods
        voting_methods = ['soft', 'hard', 'weighted']
        
        for voting_method in voting_methods:
            logger.info(f"Testing voting method: {voting_method}")
            
            # Create voting ensemble
            voting_ensemble = VotingEnsemble(
                base_models=base_models,
                voting_method=voting_method
            )
            
            # Train ensemble
            voting_ensemble.fit(X_train, y_train)
            
            # Make predictions
            predictions = voting_ensemble.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            logger.info(f"  {voting_method} - MSE: {mse:.4f}, R²: {r2:.4f}")
            
        logger.info("✅ Voting ensemble test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Voting ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_optimizer():
    """Test ensemble optimizer"""
    logger.info("🚀 Testing Ensemble Optimizer")
    
    try:
        # Create synthetic data
        X, y, feature_names = create_synthetic_supply_chain_data(n_samples=300)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create optimizer
        optimizer = EnsembleOptimizer(random_state=42)
        
        # Create base models
        base_models = optimizer.create_optimized_base_models()
        
        logger.info(f"Created {len(base_models)} base models")
        
        # Optimize ensemble
        logger.info("Optimizing ensemble methods...")
        optimization_result = optimizer.optimize_ensemble(
            X_train, y_train, base_models,
            ensemble_types=['stacking', 'blending', 'voting'],
            cv_folds=3
        )
        
        # Get best ensemble
        best_ensemble = optimization_result['best_ensemble']
        best_name = optimization_result['best_ensemble_name']
        best_score = optimization_result['best_score']
        
        logger.info(f"Best ensemble: {best_name} (CV MSE: {best_score:.4f})")
        
        # Train and evaluate best ensemble
        best_ensemble.fit(X_train, y_train)
        test_metrics = evaluate_ensemble(best_ensemble, X_test, y_test)
        
        logger.info("Test metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        # Show all results
        logger.info("\nAll ensemble results:")
        for name, result in optimization_result['all_results'].items():
            logger.info(f"  {name}: CV MSE={result['cv_score']:.4f} (±{result['cv_std']:.4f})")
            
        logger.info("✅ Ensemble optimizer test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ensemble optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demand_forecasting_ensemble():
    """Test demand forecasting ensemble"""
    logger.info("🚀 Testing Demand Forecasting Ensemble")
    
    try:
        # Create synthetic demand data
        X, y, feature_names = create_synthetic_supply_chain_data(
            n_samples=800, n_features=15, noise=0.15
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create demand forecasting ensemble
        logger.info("Creating demand forecasting ensemble...")
        result = create_demand_forecasting_ensemble(
            X_train, y_train, X_test, y_test
        )
        
        ensemble = result['ensemble']
        ensemble_name = result['ensemble_name']
        cv_score = result['cv_score']
        test_metrics = result['test_metrics']
        
        logger.info(f"Created ensemble: {ensemble_name}")
        logger.info(f"CV MSE: {cv_score:.4f}")
        logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
        logger.info(f"Test R²: {test_metrics['r2_score']:.4f}")
        logger.info(f"Test MAPE: {test_metrics['mape']:.2f}%")
        
        # Test predictions
        sample_predictions = ensemble.predict(X_test[:5])
        logger.info(f"Sample predictions: {sample_predictions}")
        logger.info(f"Sample actuals: {y_test[:5]}")
        
        # Performance comparison
        logger.info("\nPerformance comparison:")
        for name, ensemble_result in result['all_results'].items():
            logger.info(f"  {name}: CV MSE={ensemble_result['cv_score']:.4f}")
            
        logger.info("✅ Demand forecasting ensemble test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demand forecasting ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_performance_comparison():
    """Test ensemble performance comparison"""
    logger.info("🚀 Testing Ensemble Performance Comparison")
    
    try:
        # Create synthetic data
        X, y, feature_names = create_synthetic_supply_chain_data(n_samples=600)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create base models
        base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
        }
        
        # Create different ensemble types
        ensembles = {
            'stacking': StackingEnsemble(base_models, cv_folds=3, random_state=42),
            'blending_linear': BlendingEnsemble(base_models, blend_method='linear', random_state=42),
            'blending_weighted': BlendingEnsemble(base_models, blend_method='weighted', random_state=42),
            'voting_soft': VotingEnsemble(base_models, voting_method='soft'),
            'voting_hard': VotingEnsemble(base_models, voting_method='hard')
        }
        
        results = {}
        
        # Test each ensemble
        for ensemble_name, ensemble in ensembles.items():
            logger.info(f"Testing {ensemble_name}...")
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            test_metrics = evaluate_ensemble(ensemble, X_test, y_test)
            results[ensemble_name] = test_metrics
            
            logger.info(f"  {ensemble_name} - MSE: {test_metrics['mse']:.4f}, R²: {test_metrics['r2_score']:.4f}")
            
        # Compare with individual base models
        logger.info("\nBase model comparison:")
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            logger.info(f"  {name} - MSE: {mse:.4f}, R²: {r2:.4f}")
            
        # Find best ensemble
        best_ensemble = min(results.keys(), key=lambda x: results[x]['mse'])
        logger.info(f"\nBest ensemble: {best_ensemble}")
        logger.info(f"Best MSE: {results[best_ensemble]['mse']:.4f}")
        logger.info(f"Best R²: {results[best_ensemble]['r2_score']:.4f}")
        
        logger.info("✅ Ensemble performance comparison test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ensemble performance comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all ensemble tests"""
    logger.info("🚀 Running Advanced Ensemble Methods Tests")
    
    tests = [
        test_stacking_ensemble,
        test_blending_ensemble,
        test_voting_ensemble,
        test_ensemble_optimizer,
        test_demand_forecasting_ensemble,
        test_ensemble_performance_comparison
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
            logger.info(f"Test {test.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
            
    # Overall result
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n=== ENSEMBLE TESTS SUMMARY ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("✅ All ensemble method tests passed!")
        return True
    else:
        logger.error("❌ Some ensemble tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)