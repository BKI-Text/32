#!/usr/bin/env python3
"""
Test Automated Feature Selection Pipeline
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import tempfile
import shutil
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_selection import (
    AutomatedFeatureSelector,
    FeatureSelectionPipeline,
    run_feature_selection
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for feature selection"""
    # Generate synthetic data
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Add some constant and highly correlated features
    X_df['constant_feature'] = 1.0
    X_df['correlated_feature'] = X_df['feature_0'] * 0.95 + np.random.normal(0, 0.01, len(X_df))
    
    return X_df, y_series

def test_automated_feature_selector():
    """Test AutomatedFeatureSelector"""
    logger.info("🚀 Testing AutomatedFeatureSelector")
    
    try:
        # Create test data
        X, y = create_test_data()
        
        logger.info(f"Created test data: {X.shape[1]} features, {X.shape[0]} samples")
        
        # Initialize selector
        selector = AutomatedFeatureSelector(problem_type='regression', scoring='r2')
        
        # Fit selector
        results = selector.fit(X, y)
        
        # Check results structure
        required_keys = ['best_method', 'best_score', 'selected_features', 'n_selected', 'n_original']
        for key in required_keys:
            if key not in results:
                logger.error(f"Missing key in results: {key}")
                return False
        
        logger.info(f"Best method: {results['best_method']}")
        logger.info(f"Best score: {results['best_score']:.4f}")
        logger.info(f"Selected features: {results['n_selected']}/{results['n_original']}")
        
        # Check if features were actually selected
        if results['n_selected'] == 0:
            logger.error("No features were selected")
            return False
        
        if results['n_selected'] >= results['n_original']:
            logger.error("Feature selection didn't reduce feature count")
            return False
        
        # Test transform
        X_transformed = selector.transform(X)
        
        if X_transformed.shape[1] != results['n_selected']:
            logger.error("Transform didn't return correct number of features")
            return False
        
        # Test feature importance summary
        importance_df = selector.get_feature_importance_summary()
        
        if importance_df.empty:
            logger.error("Feature importance summary is empty")
            return False
        
        if len(importance_df) != results['n_original']:
            logger.error("Feature importance summary has wrong number of features")
            return False
        
        logger.info("✅ AutomatedFeatureSelector test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ AutomatedFeatureSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_selection_pipeline():
    """Test FeatureSelectionPipeline"""
    logger.info("🚀 Testing FeatureSelectionPipeline")
    
    try:
        # Create test data
        X, y = create_test_data()
        
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Configure pipeline
            config = {
                'problem_type': 'regression',
                'scoring': 'r2',
                'remove_correlated': True,
                'correlation_threshold': 0.95,
                'remove_constant': True,
                'scale_features': True,
                'save_results': True,
                'output_dir': temp_dir
            }
            
            # Initialize pipeline
            pipeline = FeatureSelectionPipeline(config)
            
            # Run pipeline
            results = pipeline.run(X, y)
            
            # Check results
            required_keys = ['original_features', 'selected_features', 'feature_reduction', 'final_score']
            for key in required_keys:
                if key not in results:
                    logger.error(f"Missing key in pipeline results: {key}")
                    return False
            
            logger.info(f"Original features: {results['original_features']}")
            logger.info(f"Selected features: {results['selected_features']}")
            logger.info(f"Feature reduction: {results['feature_reduction']:.2%}")
            logger.info(f"Final score: {results['final_score']:.4f}")
            
            # Check if features were reduced
            if results['feature_reduction'] <= 0:
                logger.error("Pipeline didn't reduce feature count")
                return False
            
            # Test transform on new data
            X_new = X.iloc[:10].copy()  # Take first 10 rows
            X_transformed = pipeline.transform(X_new)
            
            if X_transformed.shape[1] != results['selected_features']:
                logger.error("Transform didn't return correct number of features")
                return False
            
            if X_transformed.shape[0] != 10:
                logger.error("Transform didn't return correct number of samples")
                return False
            
            # Check if output files were created
            output_files = list(Path(temp_dir).glob("*.json")) + list(Path(temp_dir).glob("*.csv"))
            
            if len(output_files) == 0:
                logger.error("No output files were created")
                return False
            
            logger.info(f"Created {len(output_files)} output files")
            
            logger.info("✅ FeatureSelectionPipeline test passed")
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        logger.error(f"❌ FeatureSelectionPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_selection_methods():
    """Test different feature selection methods"""
    logger.info("🚀 Testing feature selection methods")
    
    try:
        # Create test data with known informative features
        X, y = make_regression(
            n_samples=200,
            n_features=15,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Test selector
        selector = AutomatedFeatureSelector(problem_type='regression')
        results = selector.fit(X_df, y_series)
        
        # Check that multiple methods were tested
        all_results = results['all_results']
        
        expected_methods = ['variance_threshold', 'univariate_k10', 'mutual_info_k10', 'rfe_cv', 'random_forest']
        
        found_methods = 0
        for method in expected_methods:
            if method in all_results:
                found_methods += 1
                logger.info(f"✅ Found method: {method}")
            else:
                logger.warning(f"⚠️ Method not found: {method}")
        
        if found_methods < 3:
            logger.error(f"Only {found_methods} methods found, expected at least 3")
            return False
        
        # Check that each method has proper results
        for method_name, method_result in all_results.items():
            if 'selected_features' not in method_result:
                logger.error(f"Method {method_name} missing selected_features")
                return False
            
            if 'n_features' not in method_result:
                logger.error(f"Method {method_name} missing n_features")
                return False
            
            if method_result['n_features'] != len(method_result['selected_features']):
                logger.error(f"Method {method_name} has inconsistent feature count")
                return False
        
        logger.info("✅ Feature selection methods test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature selection methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_run_feature_selection_function():
    """Test run_feature_selection function"""
    logger.info("🚀 Testing run_feature_selection function")
    
    try:
        # Create test data
        X, y = create_test_data()
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run feature selection
            config = {
                'problem_type': 'regression',
                'scoring': 'r2',
                'output_dir': temp_dir
            }
            
            results = run_feature_selection(X, y, config)
            
            # Check results
            if 'selected_features' not in results:
                logger.error("Results missing selected_features")
                return False
            
            if 'final_score' not in results:
                logger.error("Results missing final_score")
                return False
            
            if results['selected_features'] >= results['original_features']:
                logger.error("Feature selection didn't reduce feature count")
                return False
            
            logger.info(f"Function reduced features from {results['original_features']} to {results['selected_features']}")
            
            logger.info("✅ run_feature_selection function test passed")
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        logger.error(f"❌ run_feature_selection function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_steps():
    """Test preprocessing steps"""
    logger.info("🚀 Testing preprocessing steps")
    
    try:
        # Create data with problematic features
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        # Add constant feature
        X = np.column_stack([X, np.ones(X.shape[0])])
        
        # Add highly correlated feature
        X = np.column_stack([X, X[:, 0] * 0.99 + np.random.normal(0, 0.01, X.shape[0])])
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        original_features = X_df.shape[1]
        
        # Test pipeline with preprocessing
        config = {
            'problem_type': 'regression',
            'remove_constant': True,
            'remove_correlated': True,
            'correlation_threshold': 0.95,
            'scale_features': True,
            'save_results': False
        }
        
        pipeline = FeatureSelectionPipeline(config)
        results = pipeline.run(X_df, y_series)
        
        # Check that preprocessing removed problematic features
        if results['original_features'] >= original_features:
            logger.error("Preprocessing didn't remove problematic features")
            return False
        
        logger.info(f"Preprocessing reduced features from {original_features} to {results['original_features']}")
        
        logger.info("✅ Preprocessing steps test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing steps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all feature selection tests"""
    logger.info("🚀 Running Automated Feature Selection Tests")
    
    tests = [
        ("AutomatedFeatureSelector", test_automated_feature_selector),
        ("FeatureSelectionPipeline", test_feature_selection_pipeline),
        ("Feature Selection Methods", test_feature_selection_methods),
        ("run_feature_selection Function", test_run_feature_selection_function),
        ("Preprocessing Steps", test_preprocessing_steps)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*60}")
            
            result = test_func()
            
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE SELECTION TESTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All feature selection tests passed!")
        return True
    else:
        logger.error("💥 Some feature selection tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)