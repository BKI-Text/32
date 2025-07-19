#!/usr/bin/env python3
"""
Enhanced ML Training Pipeline for Beverly Knits AI Supply Chain Planner
Advanced ML training with model validation, versioning, and performance monitoring
"""

import sys
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import joblib
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# Optional plotting libraries - will handle if not available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model versioning information"""
    version: str
    timestamp: str
    model_type: str
    dataset_version: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    validation_scores: List[float]
    
@dataclass
class ModelValidationResult:
    """Model validation results"""
    model_name: str
    validation_score: float
    test_score: float
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[Dict]
    is_valid: bool
    validation_notes: List[str]

class EnhancedMLTrainer:
    """Enhanced ML trainer with validation and versioning"""
    
    def __init__(self, model_registry_path: str = "models/registry"):
        self.model_registry_path = Path(model_registry_path)
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        self.trained_models_path = Path("models/trained")
        self.trained_models_path.mkdir(parents=True, exist_ok=True)
        
        self.validation_results = []
        self.model_versions = {}
        
        # Performance thresholds for validation
        self.performance_thresholds = {
            'regression_r2': 0.85,
            'regression_rmse_max': 100,
            'classification_accuracy': 0.80,
            'cv_std_max': 0.10  # Max standard deviation for cross-validation scores
        }
    
    def train_demand_forecasting_models_enhanced(self, X: pd.DataFrame, y: pd.Series) -> List[ModelValidationResult]:
        """Train and validate demand forecasting models with enhanced features"""
        logger.info("🚀 Training enhanced demand forecasting models...")
        
        results = []
        
        # Models to train with hyperparameter tuning
        model_configs = [
            {
                'name': 'xgboost_demand_enhanced',
                'model_class': 'XGBRegressor',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            {
                'name': 'random_forest_demand_enhanced',
                'model_class': 'RandomForestRegressor',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            {
                'name': 'gradient_boost_demand_enhanced',
                'model_class': 'GradientBoostingRegressor',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9]
                }
            }
        ]
        
        for config in model_configs:
            result = self._train_and_validate_model(config, X, y, 'regression')
            if result:
                results.append(result)
        
        return results
    
    def train_inventory_optimization_models_enhanced(self, X: pd.DataFrame, y: pd.Series) -> List[ModelValidationResult]:
        """Train and validate inventory optimization models"""
        logger.info("📦 Training enhanced inventory optimization models...")
        
        results = []
        
        model_configs = [
            {
                'name': 'xgboost_inventory_enhanced',
                'model_class': 'XGBRegressor',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.1, 0.2]
                }
            },
            {
                'name': 'random_forest_inventory_enhanced',
                'model_class': 'RandomForestRegressor',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5]
                }
            }
        ]
        
        for config in model_configs:
            result = self._train_and_validate_model(config, X, y, 'regression')
            if result:
                results.append(result)
        
        return results
    
    def train_supplier_performance_models_enhanced(self, X: pd.DataFrame, y: pd.Series) -> List[ModelValidationResult]:
        """Train and validate supplier performance models"""
        logger.info("🏭 Training enhanced supplier performance models...")
        
        results = []
        
        # Convert continuous target to categories for classification
        y_categorical = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
        
        model_configs = [
            {
                'name': 'random_forest_supplier_enhanced',
                'model_class': 'RandomForestClassifier',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5]
                }
            },
            {
                'name': 'gradient_boost_supplier_enhanced', 
                'model_class': 'GradientBoostingClassifier',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6],
                    'learning_rate': [0.1, 0.2]
                }
            }
        ]
        
        for config in model_configs:
            result = self._train_and_validate_model(config, X, y_categorical, 'classification')
            if result:
                results.append(result)
        
        return results
    
    def _train_and_validate_model(self, config: Dict, X: pd.DataFrame, y: pd.Series, task_type: str) -> Optional[ModelValidationResult]:
        """Train and validate a single model with hyperparameter tuning"""
        model_name = config['name']
        logger.info(f"🔧 Training {model_name}...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features if needed
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            model_class = self._get_model_class(config['model_class'])
            if model_class is None:
                logger.error(f"❌ Unknown model class: {config['model_class']}")
                return None
            
            # Hyperparameter tuning
            base_model = model_class(random_state=42)
            
            # Use appropriate scoring metric
            scoring = 'r2' if task_type == 'regression' else 'accuracy'
            
            grid_search = GridSearchCV(
                base_model,
                config['param_grid'],
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
            # Fit with grid search
            start_time = datetime.now()
            grid_search.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_scaled)
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring=scoring)
            
            # Calculate metrics
            if task_type == 'regression':
                test_score = r2_score(y_test, y_pred)
                validation_score = grid_search.best_score_
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Validation checks
                is_valid = (
                    test_score >= self.performance_thresholds['regression_r2'] and
                    rmse <= self.performance_thresholds['regression_rmse_max'] and
                    np.std(cv_scores) <= self.performance_thresholds['cv_std_max']
                )
                
                validation_notes = []
                if test_score < self.performance_thresholds['regression_r2']:
                    validation_notes.append(f"R² score {test_score:.3f} below threshold {self.performance_thresholds['regression_r2']}")
                if rmse > self.performance_thresholds['regression_rmse_max']:
                    validation_notes.append(f"RMSE {rmse:.2f} above threshold {self.performance_thresholds['regression_rmse_max']}")
                
                confusion_matrix = None
                classification_report_dict = None
                
            else:  # classification
                test_score = accuracy_score(y_test, y_pred)
                validation_score = grid_search.best_score_
                
                # Classification metrics
                from sklearn.metrics import confusion_matrix as cm
                confusion_matrix = cm(y_test, y_pred)
                classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
                
                # Validation checks
                is_valid = (
                    test_score >= self.performance_thresholds['classification_accuracy'] and
                    np.std(cv_scores) <= self.performance_thresholds['cv_std_max']
                )
                
                validation_notes = []
                if test_score < self.performance_thresholds['classification_accuracy']:
                    validation_notes.append(f"Accuracy {test_score:.3f} below threshold {self.performance_thresholds['classification_accuracy']}")
            
            # Feature importance
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            elif hasattr(best_model, 'coef_'):
                feature_importance = dict(zip(X.columns, abs(best_model.coef_)))
            
            # Save model if valid
            if is_valid:
                model_path = self.trained_models_path / f"{model_name}.pkl"
                joblib.dump({
                    'model': best_model,
                    'scaler': scaler,
                    'metadata': {
                        'model_name': model_name,
                        'model_type': config['model_class'],
                        'training_time': training_time,
                        'best_params': grid_search.best_params_,
                        'feature_columns': list(X.columns),
                        'trained_at': datetime.now().isoformat()
                    }
                }, model_path)
                
                # Save to registry
                self._save_model_version(model_name, best_model, grid_search.best_params_, 
                                       feature_importance, cv_scores, task_type)
                
                logger.info(f"✅ {model_name} trained successfully - Score: {test_score:.3f}")
            else:
                logger.warning(f"⚠️ {model_name} failed validation - Score: {test_score:.3f}")
            
            return ModelValidationResult(
                model_name=model_name,
                validation_score=validation_score,
                test_score=test_score,
                cross_validation_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
                confusion_matrix=confusion_matrix,
                classification_report=classification_report_dict,
                is_valid=is_valid,
                validation_notes=validation_notes
            )
            
        except Exception as e:
            logger.error(f"❌ Training failed for {model_name}: {e}")
            return None
    
    def _get_model_class(self, class_name: str):
        """Get model class by name"""
        try:
            if class_name == 'XGBRegressor':
                import xgboost as xgb
                return xgb.XGBRegressor
            elif class_name == 'RandomForestRegressor':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor
            elif class_name == 'GradientBoostingRegressor':
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor
            elif class_name == 'RandomForestClassifier':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier
            elif class_name == 'GradientBoostingClassifier':
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier
            else:
                return None
        except ImportError as e:
            logger.error(f"❌ Failed to import {class_name}: {e}")
            return None
    
    def _save_model_version(self, model_name: str, model, hyperparameters: Dict, 
                          feature_importance: Dict, cv_scores: List[float], task_type: str):
        """Save model version information"""
        version = f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_info = ModelVersion(
            version=version,
            timestamp=datetime.now().isoformat(),
            model_type=type(model).__name__,
            dataset_version="erp_v1.0",
            performance_metrics={
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'task_type': task_type
            },
            hyperparameters=hyperparameters,
            feature_importance=feature_importance,
            validation_scores=cv_scores
        )
        
        self.model_versions[model_name] = version_info
        
        # Save to registry
        registry_file = self.model_registry_path / f"{model_name}_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(asdict(version_info), f, indent=2, default=str)
    
    def generate_training_report(self, all_results: List[ModelValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        valid_models = [r for r in all_results if r.is_valid]
        invalid_models = [r for r in all_results if not r.is_valid]
        
        # Performance summary
        if valid_models:
            best_model = max(valid_models, key=lambda x: x.test_score)
            avg_performance = np.mean([r.test_score for r in valid_models])
        else:
            best_model = None
            avg_performance = 0
        
        report = {
            "training_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_models_trained": len(all_results),
                "valid_models": len(valid_models),
                "invalid_models": len(invalid_models),
                "success_rate": len(valid_models) / len(all_results) if all_results else 0,
                "average_performance": avg_performance,
                "best_model": best_model.model_name if best_model else None,
                "best_performance": best_model.test_score if best_model else 0
            },
            "valid_models": [
                {
                    "model_name": r.model_name,
                    "test_score": r.test_score,
                    "validation_score": r.validation_score,
                    "cv_mean": np.mean(r.cross_validation_scores),
                    "cv_std": np.std(r.cross_validation_scores),
                    "top_features": dict(sorted(r.feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True)[:5])
                }
                for r in valid_models
            ],
            "invalid_models": [
                {
                    "model_name": r.model_name,
                    "test_score": r.test_score,
                    "validation_notes": r.validation_notes
                }
                for r in invalid_models
            ],
            "model_versions": {k: asdict(v) for k, v in self.model_versions.items()},
            "performance_thresholds": self.performance_thresholds,
            "recommendations": self._generate_training_recommendations(all_results)
        }
        
        return report
    
    def _generate_training_recommendations(self, results: List[ModelValidationResult]) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        valid_models = [r for r in results if r.is_valid]
        invalid_models = [r for r in results if not r.is_valid]
        
        if len(valid_models) == 0:
            recommendations.append("⚠️ No models passed validation. Consider lowering thresholds or improving data quality.")
        elif len(valid_models) < len(results) / 2:
            recommendations.append("⚠️ Less than 50% of models passed validation. Review training data and feature engineering.")
        
        # Performance recommendations
        avg_scores = [r.test_score for r in valid_models] if valid_models else []
        if avg_scores and np.mean(avg_scores) < 0.9:
            recommendations.append("📈 Average model performance below 90%. Consider feature engineering or ensemble methods.")
        
        # Cross-validation stability
        unstable_models = [r for r in valid_models if np.std(r.cross_validation_scores) > 0.05]
        if unstable_models:
            model_names = [r.model_name for r in unstable_models]
            recommendations.append(f"📊 High CV variance in models: {', '.join(model_names)}. Consider regularization.")
        
        return recommendations

def main():
    """Main enhanced training pipeline"""
    logger.info("🚀 Beverly Knits AI - Enhanced ML Training Pipeline")
    logger.info("Advanced ML training with validation, versioning, and monitoring")
    logger.info("=" * 80)
    
    trainer = EnhancedMLTrainer()
    
    try:
        # Connect to ERP and extract data (reusing previous pipeline)
        erp = EfabERPIntegration(username='psytz', password='big$cat')
        if not erp.connect():
            logger.error("❌ Failed to connect to ERP")
            return False
        
        # Load previous training data for demonstration
        try:
            from erp_ml_training_pipeline import ERPDataExtractor
            extractor = ERPDataExtractor()
            
            if extractor.connect():
                # Extract datasets
                demand_dataset = extractor.extract_demand_forecasting_data()
                inventory_dataset = extractor.extract_inventory_optimization_data() 
                supplier_dataset = extractor.extract_supplier_performance_data()
                
                all_results = []
                
                # Train demand forecasting models
                if demand_dataset:
                    # Prepare data
                    df = demand_dataset.data
                    df_encoded = pd.get_dummies(df[['yarn_type']], prefix='yarn')
                    numeric_features = ['month', 'historical_avg', 'seasonal_factor', 'supplier_count']
                    X = pd.concat([df[numeric_features], df_encoded], axis=1)
                    y = df['demand_quantity']
                    
                    demand_results = trainer.train_demand_forecasting_models_enhanced(X, y)
                    all_results.extend(demand_results)
                
                # Train inventory optimization models
                if inventory_dataset:
                    df = inventory_dataset.data
                    X = df[['current_stock', 'lead_time', 'demand_variance', 'cost_per_unit', 'supplier_reliability']]
                    y = df['optimal_stock_level']
                    
                    inventory_results = trainer.train_inventory_optimization_models_enhanced(X, y)
                    all_results.extend(inventory_results)
                
                # Train supplier performance models
                if supplier_dataset:
                    df = supplier_dataset.data
                    df_encoded = pd.get_dummies(df[['supplier']], prefix='supplier')
                    numeric_features = ['delivery_time', 'quality_rating', 'cost_variance', 'order_frequency', 'payment_terms']
                    X = pd.concat([df[numeric_features], df_encoded], axis=1)
                    y = df['reliability_score']
                    
                    supplier_results = trainer.train_supplier_performance_models_enhanced(X, y)
                    all_results.extend(supplier_results)
                
                # Generate comprehensive report
                report = trainer.generate_training_report(all_results)
                
                # Save report
                report_file = f"enhanced_ml_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # Display results
                logger.info("\n" + "=" * 80)
                logger.info("🎉 ENHANCED ML TRAINING COMPLETE")
                logger.info("=" * 80)
                logger.info(f"📊 Models Trained: {report['summary']['total_models_trained']}")
                logger.info(f"✅ Valid Models: {report['summary']['valid_models']}")
                logger.info(f"❌ Invalid Models: {report['summary']['invalid_models']}")
                logger.info(f"📈 Success Rate: {report['summary']['success_rate']:.1%}")
                logger.info(f"🏆 Best Model: {report['summary']['best_model']} ({report['summary']['best_performance']:.3f})")
                
                if report['recommendations']:
                    logger.info("\n💡 RECOMMENDATIONS:")
                    for rec in report['recommendations']:
                        logger.info(f"   • {rec}")
                
                logger.info(f"\n📄 Report saved to: {report_file}")
                
                return True
            else:
                logger.error("❌ Failed to connect to ERP for data extraction")
                return False
                
        except ImportError:
            logger.error("❌ Could not import ERP data extractor. Using synthetic data for demo.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)