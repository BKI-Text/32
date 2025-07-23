#!/usr/bin/env python3
"""
Test ML Model Versioning System
Beverly Knits AI Supply Chain Planner
"""

import os
import sys
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.versioning import (
    ModelVersioningSystem,
    ModelStatus,
    ModelType,
    ModelMetadata,
    register_model,
    load_model,
    get_latest_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelVersioningTester:
    """Test model versioning system"""
    
    def __init__(self):
        self.temp_dir = None
        self.versioning_system = None
        
    def setup_test_environment(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.versioning_system = ModelVersioningSystem(
            storage_path=os.path.join(self.temp_dir, "models"),
            db_path=os.path.join(self.temp_dir, "models", "test_versions.db")
        )
        logger.info(f"Test environment setup at: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
    
    def create_test_models(self):
        """Create test models"""
        # Generate test data
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        
        # Create different models
        models = {
            'linear_model': LinearRegression(),
            'ridge_model': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            trained_models[name] = {
                'model': model,
                'metrics': {'mse': mse, 'r2': r2},
                'features': [f'feature_{i}' for i in range(X.shape[1])],
                'targets': ['target']
            }
        
        return trained_models
    
    def test_model_registration(self) -> bool:
        """Test model registration"""
        logger.info("🚀 Testing model registration")
        
        try:
            # Create test models
            trained_models = self.create_test_models()
            
            registered_model_ids = []
            
            # Register models
            for model_name, model_data in trained_models.items():
                model_id = self.versioning_system.register_model(
                    model=model_data['model'],
                    model_name=model_name,
                    model_type=ModelType.CUSTOM,
                    description=f"Test {model_name}",
                    input_features=model_data['features'],
                    output_targets=model_data['targets'],
                    training_metrics=model_data['metrics'],
                    validation_metrics=model_data['metrics'],
                    test_metrics=model_data['metrics'],
                    framework='sklearn',
                    tags=['test', 'regression'],
                    labels={'environment': 'test', 'dataset': 'synthetic'}
                )
                
                registered_model_ids.append(model_id)
                logger.info(f"Registered model: {model_name} with ID: {model_id}")
            
            # Verify registration
            for model_id in registered_model_ids:
                metadata = self.versioning_system.get_model_metadata(model_id)
                if not metadata:
                    logger.error(f"Model metadata not found: {model_id}")
                    return False
                
                logger.info(f"Model {model_id} registered successfully")
            
            logger.info("✅ Model registration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model registration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_loading(self) -> bool:
        """Test model loading"""
        logger.info("🚀 Testing model loading")
        
        try:
            # Create and register a test model
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            model = LinearRegression()
            model.fit(X, y)
            
            model_id = self.versioning_system.register_model(
                model=model,
                model_name="test_loading_model",
                model_type=ModelType.CUSTOM,
                description="Test model for loading",
                input_features=['feature_0', 'feature_1', 'feature_2'],
                output_targets=['target'],
                training_metrics={'mse': 0.1, 'r2': 0.9}
            )
            
            # Load the model
            loaded_model, metadata = self.versioning_system.load_model(model_id)
            
            # Verify model works
            test_X = np.random.randn(10, 3)
            original_predictions = model.predict(test_X)
            loaded_predictions = loaded_model.predict(test_X)
            
            # Check if predictions match
            if not np.allclose(original_predictions, loaded_predictions):
                logger.error("Loaded model predictions don't match original")
                return False
            
            # Verify metadata
            if metadata.model_id != model_id:
                logger.error("Metadata model ID doesn't match")
                return False
            
            logger.info("✅ Model loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_versioning(self) -> bool:
        """Test model versioning"""
        logger.info("🚀 Testing model versioning")
        
        try:
            # Create multiple versions of the same model
            model_name = "versioned_model"
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            
            model_ids = []
            
            # Create 3 versions
            for i in range(3):
                model = LinearRegression()
                model.fit(X, y)
                
                model_id = self.versioning_system.register_model(
                    model=model,
                    model_name=model_name,
                    model_type=ModelType.CUSTOM,
                    description=f"Version {i+1} of {model_name}",
                    input_features=['feature_0', 'feature_1', 'feature_2'],
                    output_targets=['target'],
                    training_metrics={'mse': 0.1 - i*0.01, 'r2': 0.9 + i*0.01}
                )
                
                model_ids.append(model_id)
                logger.info(f"Created version {i+1} with ID: {model_id}")
            
            # Test listing models
            models = self.versioning_system.list_models(model_name=model_name)
            if len(models) != 3:
                logger.error(f"Expected 3 models, got {len(models)}")
                return False
            
            # Test getting latest model
            latest_model = self.versioning_system.get_latest_model(model_name)
            if not latest_model:
                logger.error("Latest model not found")
                return False
            
            # Verify versions are correct
            versions = [model.version for model in models]
            expected_versions = ['1.0.0', '1.0.1', '1.0.2']
            
            if sorted(versions) != sorted(expected_versions):
                logger.error(f"Version mismatch: expected {expected_versions}, got {versions}")
                return False
            
            logger.info("✅ Model versioning test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model versioning test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_status_management(self) -> bool:
        """Test model status management"""
        logger.info("🚀 Testing model status management")
        
        try:
            # Create a test model
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            model = LinearRegression()
            model.fit(X, y)
            
            model_id = self.versioning_system.register_model(
                model=model,
                model_name="status_test_model",
                model_type=ModelType.CUSTOM,
                description="Test model for status management"
            )
            
            # Test initial status
            metadata = self.versioning_system.get_model_metadata(model_id)
            if metadata.status != ModelStatus.TRAINED:
                logger.error(f"Initial status should be TRAINED, got {metadata.status}")
                return False
            
            # Update status to VALIDATED
            self.versioning_system.update_model_status(model_id, ModelStatus.VALIDATED)
            
            # Verify status update
            metadata = self.versioning_system.get_model_metadata(model_id)
            if metadata.status != ModelStatus.VALIDATED:
                logger.error(f"Status should be VALIDATED, got {metadata.status}")
                return False
            
            # Update status to DEPLOYED
            self.versioning_system.update_model_status(
                model_id, 
                ModelStatus.DEPLOYED,
                deployment_environment="production",
                deployment_config={"replicas": 3, "memory": "2Gi"}
            )
            
            # Verify deployment status
            metadata = self.versioning_system.get_model_metadata(model_id)
            if metadata.status != ModelStatus.DEPLOYED:
                logger.error(f"Status should be DEPLOYED, got {metadata.status}")
                return False
            
            if metadata.deployment_environment != "production":
                logger.error("Deployment environment not set correctly")
                return False
            
            if not metadata.deployment_date:
                logger.error("Deployment date not set")
                return False
            
            logger.info("✅ Model status management test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model status management test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_tagging(self) -> bool:
        """Test model tagging"""
        logger.info("🚀 Testing model tagging")
        
        try:
            # Create a test model
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            model = LinearRegression()
            model.fit(X, y)
            
            model_id = self.versioning_system.register_model(
                model=model,
                model_name="tagging_test_model",
                model_type=ModelType.CUSTOM,
                description="Test model for tagging",
                tags=['initial', 'test']
            )
            
            # Verify initial tags
            metadata = self.versioning_system.get_model_metadata(model_id)
            if set(metadata.tags) != {'initial', 'test'}:
                logger.error(f"Initial tags incorrect: {metadata.tags}")
                return False
            
            # Add new tags
            self.versioning_system.add_model_tags(model_id, ['production', 'validated'])
            
            # Verify tags added
            metadata = self.versioning_system.get_model_metadata(model_id)
            expected_tags = {'initial', 'test', 'production', 'validated'}
            if set(metadata.tags) != expected_tags:
                logger.error(f"Tags after addition incorrect: {metadata.tags}")
                return False
            
            # Remove tags
            self.versioning_system.remove_model_tags(model_id, ['initial', 'test'])
            
            # Verify tags removed
            metadata = self.versioning_system.get_model_metadata(model_id)
            expected_tags = {'production', 'validated'}
            if set(metadata.tags) != expected_tags:
                logger.error(f"Tags after removal incorrect: {metadata.tags}")
                return False
            
            logger.info("✅ Model tagging test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model tagging test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_comparison(self) -> bool:
        """Test model comparison"""
        logger.info("🚀 Testing model comparison")
        
        try:
            # Create two models to compare
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            
            # Model 1
            model1 = LinearRegression()
            model1.fit(X, y)
            pred1 = model1.predict(X)
            
            model_id1 = self.versioning_system.register_model(
                model=model1,
                model_name="comparison_model_1",
                model_type=ModelType.CUSTOM,
                description="First model for comparison",
                training_metrics={'mse': mean_squared_error(y, pred1), 'r2': r2_score(y, pred1)}
            )
            
            # Model 2
            model2 = Ridge(alpha=1.0)
            model2.fit(X, y)
            pred2 = model2.predict(X)
            
            model_id2 = self.versioning_system.register_model(
                model=model2,
                model_name="comparison_model_2",
                model_type=ModelType.CUSTOM,
                description="Second model for comparison",
                training_metrics={'mse': mean_squared_error(y, pred2), 'r2': r2_score(y, pred2)}
            )
            
            # Compare models
            comparison = self.versioning_system.compare_models(model_id1, model_id2)
            
            # Verify comparison structure
            required_keys = ['model1', 'model2', 'differences']
            for key in required_keys:
                if key not in comparison:
                    logger.error(f"Comparison missing key: {key}")
                    return False
            
            # Verify model information
            if comparison['model1']['id'] != model_id1:
                logger.error("Model 1 ID mismatch in comparison")
                return False
            
            if comparison['model2']['id'] != model_id2:
                logger.error("Model 2 ID mismatch in comparison")
                return False
            
            # Verify differences
            if 'training_metrics' not in comparison['differences']:
                logger.error("Training metrics differences not found")
                return False
            
            logger.info("✅ Model comparison test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model comparison test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_lineage(self) -> bool:
        """Test model lineage"""
        logger.info("🚀 Testing model lineage")
        
        try:
            # Create multiple versions of the same model
            model_name = "lineage_test_model"
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            
            model_ids = []
            
            # Create 3 versions
            for i in range(3):
                model = LinearRegression()
                model.fit(X, y)
                
                model_id = self.versioning_system.register_model(
                    model=model,
                    model_name=model_name,
                    model_type=ModelType.CUSTOM,
                    description=f"Version {i+1} for lineage test",
                    training_metrics={'mse': 0.1 - i*0.01, 'r2': 0.9 + i*0.01}
                )
                
                model_ids.append(model_id)
            
            # Get lineage for the latest model
            lineage = self.versioning_system.get_model_lineage(model_ids[-1])
            
            # Verify lineage structure
            required_keys = ['model_name', 'current_version', 'total_versions', 'versions']
            for key in required_keys:
                if key not in lineage:
                    logger.error(f"Lineage missing key: {key}")
                    return False
            
            # Verify lineage data
            if lineage['model_name'] != model_name:
                logger.error("Lineage model name mismatch")
                return False
            
            if lineage['total_versions'] != 3:
                logger.error(f"Expected 3 versions in lineage, got {lineage['total_versions']}")
                return False
            
            if len(lineage['versions']) != 3:
                logger.error(f"Expected 3 version entries, got {len(lineage['versions'])}")
                return False
            
            logger.info("✅ Model lineage test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model lineage test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_registry_statistics(self) -> bool:
        """Test registry statistics"""
        logger.info("🚀 Testing registry statistics")
        
        try:
            # Create some test models
            X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
            
            # Create models of different types
            models = [
                (LinearRegression(), "linear_model", ModelType.CUSTOM),
                (Ridge(alpha=1.0), "ridge_model", ModelType.CUSTOM),
                (RandomForestRegressor(n_estimators=10, random_state=42), "rf_model", ModelType.CUSTOM)
            ]
            
            for model, name, model_type in models:
                model.fit(X, y)
                self.versioning_system.register_model(
                    model=model,
                    model_name=name,
                    model_type=model_type,
                    description=f"Test {name} for statistics"
                )
            
            # Get statistics
            stats = self.versioning_system.get_registry_stats()
            
            # Verify statistics structure
            required_keys = ['total_models', 'models_by_type', 'models_by_status', 'storage']
            for key in required_keys:
                if key not in stats:
                    logger.error(f"Statistics missing key: {key}")
                    return False
            
            # Verify counts
            if stats['total_models'] < 3:
                logger.error(f"Expected at least 3 models, got {stats['total_models']}")
                return False
            
            if 'custom' not in stats['models_by_type']:
                logger.error("Custom model type not found in statistics")
                return False
            
            if 'trained' not in stats['models_by_status']:
                logger.error("Trained status not found in statistics")
                return False
            
            logger.info("✅ Registry statistics test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Registry statistics test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """Run all versioning tests"""
        logger.info("🚀 Running ML Model Versioning Tests")
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            tests = [
                ("Model Registration", self.test_model_registration),
                ("Model Loading", self.test_model_loading),
                ("Model Versioning", self.test_model_versioning),
                ("Model Status Management", self.test_model_status_management),
                ("Model Tagging", self.test_model_tagging),
                ("Model Comparison", self.test_model_comparison),
                ("Model Lineage", self.test_model_lineage),
                ("Registry Statistics", self.test_registry_statistics)
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
            logger.info(f"MODEL VERSIONING TESTS SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Passed: {passed}/{total}")
            logger.info(f"Success rate: {passed/total*100:.1f}%")
            
            if passed == total:
                logger.info("🎉 All model versioning tests passed!")
                return True
            else:
                logger.error("💥 Some model versioning tests failed!")
                return False
                
        finally:
            # Always cleanup
            self.cleanup_test_environment()

def main():
    """Main test execution"""
    tester = ModelVersioningTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()