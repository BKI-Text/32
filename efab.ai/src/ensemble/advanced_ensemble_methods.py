#!/usr/bin/env python3
"""
Advanced Ensemble Methods (Stacking, Blending, Voting)
Beverly Knits AI Supply Chain Planner
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Advanced stacking ensemble with multiple base learners and meta-learner
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator], 
                 meta_model: BaseEstimator = None,
                 cv_folds: int = 5,
                 use_features: bool = True,
                 random_state: int = 42):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: Dictionary of base models {name: model}
            meta_model: Meta-learner model (default: Ridge regression)
            cv_folds: Number of cross-validation folds
            use_features: Whether to include original features in meta-learner
            random_state: Random state for reproducibility
        """
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.use_features = use_features
        self.random_state = random_state
        
        # Trained models
        self.trained_base_models = {}
        self.trained_meta_model = None
        self.feature_scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the stacking ensemble
        
        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Training stacking ensemble with {len(self.base_models)} base models")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Create cross-validation folds
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Generate meta-features using cross-validation
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.debug(f"Processing fold {fold_idx + 1}/{self.cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            # Train base models on fold
            for model_idx, (model_name, model) in enumerate(self.base_models.items()):
                try:
                    # Clone and fit model
                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_train_fold, y_train_fold)
                    
                    # Generate predictions for validation set
                    val_predictions = model_clone.predict(X_val_fold)
                    meta_features[val_idx, model_idx] = val_predictions
                    
                except Exception as e:
                    logger.warning(f"Error training {model_name} on fold {fold_idx}: {e}")
                    meta_features[val_idx, model_idx] = np.mean(y_train_fold)
                    
        # Train base models on full dataset
        for model_name, model in self.base_models.items():
            try:
                logger.debug(f"Training {model_name} on full dataset")
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X, y)
                self.trained_base_models[model_name] = model_clone
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                # Use a simple mean predictor as fallback
                self.trained_base_models[model_name] = _MeanPredictor(np.mean(y))
                
        # Prepare meta-features for meta-learner
        if self.use_features:
            # Scale original features
            X_scaled = self.feature_scaler.fit_transform(X)
            meta_X = np.hstack([meta_features, X_scaled])
        else:
            meta_X = meta_features
            
        # Train meta-learner
        logger.debug("Training meta-learner")
        self.trained_meta_model = self.meta_model
        self.trained_meta_model.fit(meta_X, y)
        
        logger.info("Stacking ensemble training completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        X = np.array(X)
        
        # Generate base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.trained_base_models)))
        
        for model_idx, (model_name, model) in enumerate(self.trained_base_models.items()):
            try:
                base_predictions[:, model_idx] = model.predict(X)
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {e}")
                base_predictions[:, model_idx] = 0
                
        # Prepare meta-features
        if self.use_features:
            X_scaled = self.feature_scaler.transform(X)
            meta_X = np.hstack([base_predictions, X_scaled])
        else:
            meta_X = base_predictions
            
        # Make final predictions
        return self.trained_meta_model.predict(meta_X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner"""
        if hasattr(self.trained_meta_model, 'coef_'):
            # Linear models
            coefficients = self.trained_meta_model.coef_
            importance = {}
            
            # Base model importance
            for i, model_name in enumerate(self.trained_base_models.keys()):
                importance[f"base_{model_name}"] = abs(coefficients[i])
                
            # Original feature importance (if used)
            if self.use_features and len(coefficients) > len(self.trained_base_models):
                for i in range(len(self.trained_base_models), len(coefficients)):
                    importance[f"feature_{i-len(self.trained_base_models)}"] = abs(coefficients[i])
                    
            return importance
        else:
            return {}

class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """
    Blending ensemble using holdout validation
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator],
                 blend_method: str = 'linear',
                 holdout_ratio: float = 0.2,
                 random_state: int = 42):
        """
        Initialize blending ensemble
        
        Args:
            base_models: Dictionary of base models {name: model}
            blend_method: Blending method ('linear', 'weighted', 'stacking')
            holdout_ratio: Ratio of data to use for blending
            random_state: Random state for reproducibility
        """
        self.base_models = base_models
        self.blend_method = blend_method
        self.holdout_ratio = holdout_ratio
        self.random_state = random_state
        
        # Trained models
        self.trained_base_models = {}
        self.blend_weights = {}
        self.blend_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the blending ensemble
        
        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Training blending ensemble with {len(self.base_models)} base models")
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data for training and blending
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        holdout_size = int(n_samples * self.holdout_ratio)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[holdout_size:]
        holdout_indices = indices[:holdout_size]
        
        X_train, X_holdout = X[train_indices], X[holdout_indices]
        y_train, y_holdout = y[train_indices], y[holdout_indices]
        
        # Train base models
        for model_name, model in self.base_models.items():
            try:
                logger.debug(f"Training {model_name}")
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)
                self.trained_base_models[model_name] = model_clone
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                self.trained_base_models[model_name] = _MeanPredictor(np.mean(y_train))
                
        # Generate predictions on holdout set
        holdout_predictions = np.zeros((len(holdout_indices), len(self.trained_base_models)))
        
        for model_idx, (model_name, model) in enumerate(self.trained_base_models.items()):
            try:
                holdout_predictions[:, model_idx] = model.predict(X_holdout)
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {e}")
                holdout_predictions[:, model_idx] = np.mean(y_train)
                
        # Learn blending weights
        if self.blend_method == 'linear':
            # Simple linear blending
            self.blend_model = LinearRegression()
            self.blend_model.fit(holdout_predictions, y_holdout)
            
        elif self.blend_method == 'weighted':
            # Weighted blending based on individual model performance
            self.blend_weights = {}
            total_weight = 0
            
            for model_idx, model_name in enumerate(self.trained_base_models.keys()):
                pred = holdout_predictions[:, model_idx]
                mse = mean_squared_error(y_holdout, pred)
                weight = 1.0 / (1.0 + mse)  # Inverse MSE weighting
                self.blend_weights[model_name] = weight
                total_weight += weight
                
            # Normalize weights
            for model_name in self.blend_weights:
                self.blend_weights[model_name] /= total_weight
                
        elif self.blend_method == 'stacking':
            # Use ridge regression for stacking
            self.blend_model = Ridge(alpha=1.0)
            self.blend_model.fit(holdout_predictions, y_holdout)
            
        logger.info("Blending ensemble training completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the blending ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        X = np.array(X)
        
        # Generate base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.trained_base_models)))
        
        for model_idx, (model_name, model) in enumerate(self.trained_base_models.items()):
            try:
                base_predictions[:, model_idx] = model.predict(X)
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {e}")
                base_predictions[:, model_idx] = 0
                
        # Blend predictions
        if self.blend_method == 'weighted':
            # Weighted average
            final_predictions = np.zeros(X.shape[0])
            for model_idx, model_name in enumerate(self.trained_base_models.keys()):
                weight = self.blend_weights[model_name]
                final_predictions += weight * base_predictions[:, model_idx]
                
        else:
            # Linear or stacking blending
            final_predictions = self.blend_model.predict(base_predictions)
            
        return final_predictions
        
    def get_blend_weights(self) -> Dict[str, float]:
        """Get blending weights"""
        if self.blend_method == 'weighted':
            return self.blend_weights.copy()
        elif hasattr(self.blend_model, 'coef_'):
            return {name: coef for name, coef in 
                   zip(self.trained_base_models.keys(), self.blend_model.coef_)}
        else:
            return {}

class VotingEnsemble(BaseEstimator, RegressorMixin):
    """
    Voting ensemble with multiple voting strategies
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator],
                 voting_method: str = 'soft',
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize voting ensemble
        
        Args:
            base_models: Dictionary of base models {name: model}
            voting_method: Voting method ('soft', 'hard', 'weighted')
            weights: Optional weights for weighted voting
        """
        self.base_models = base_models
        self.voting_method = voting_method
        self.weights = weights if weights is not None else {}
        
        # Trained models
        self.trained_base_models = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the voting ensemble
        
        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Training voting ensemble with {len(self.base_models)} base models")
        
        X = np.array(X)
        y = np.array(y)
        
        # Train base models
        for model_name, model in self.base_models.items():
            try:
                logger.debug(f"Training {model_name}")
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X, y)
                self.trained_base_models[model_name] = model_clone
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                self.trained_base_models[model_name] = _MeanPredictor(np.mean(y))
                
        # Set equal weights if not provided
        if not self.weights:
            self.weights = {name: 1.0 / len(self.trained_base_models) 
                          for name in self.trained_base_models.keys()}
            
        logger.info("Voting ensemble training completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the voting ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        X = np.array(X)
        
        # Generate base model predictions
        predictions = []
        weights = []
        
        for model_name, model in self.trained_base_models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights.get(model_name, 1.0))
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {e}")
                
        if not predictions:
            return np.zeros(X.shape[0])
            
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Voting
        if self.voting_method == 'soft':
            # Weighted average
            return np.average(predictions, weights=weights, axis=0)
        elif self.voting_method == 'hard':
            # Simple average
            return np.mean(predictions, axis=0)
        elif self.voting_method == 'weighted':
            # Weighted average (same as soft)
            return np.average(predictions, weights=weights, axis=0)
        else:
            return np.mean(predictions, axis=0)

class _MeanPredictor:
    """Simple mean predictor for fallback"""
    
    def __init__(self, mean_value: float):
        self.mean_value = mean_value
        
    def predict(self, X):
        return np.full(len(X), self.mean_value)

class EnsembleOptimizer:
    """
    Optimizer for ensemble methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def optimize_ensemble(self, X: np.ndarray, y: np.ndarray,
                         base_models: Dict[str, BaseEstimator],
                         ensemble_types: List[str] = None,
                         cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize ensemble methods and return the best one
        
        Args:
            X: Training features
            y: Training targets
            base_models: Base models to ensemble
            ensemble_types: Types of ensembles to try
            cv_folds: Cross-validation folds
            
        Returns:
            Dictionary with best ensemble and performance metrics
        """
        if ensemble_types is None:
            ensemble_types = ['stacking', 'blending', 'voting']
            
        logger.info(f"Optimizing ensemble methods: {ensemble_types}")
        
        results = {}
        
        # Test stacking ensemble
        if 'stacking' in ensemble_types:
            stacking_ensemble = StackingEnsemble(
                base_models=base_models,
                cv_folds=cv_folds,
                random_state=self.random_state
            )
            
            scores = cross_val_score(stacking_ensemble, X, y, cv=cv_folds, 
                                   scoring='neg_mean_squared_error')
            results['stacking'] = {
                'ensemble': stacking_ensemble,
                'cv_score': -scores.mean(),
                'cv_std': scores.std(),
                'type': 'stacking'
            }
            
        # Test blending ensemble
        if 'blending' in ensemble_types:
            for blend_method in ['linear', 'weighted', 'stacking']:
                blending_ensemble = BlendingEnsemble(
                    base_models=base_models,
                    blend_method=blend_method,
                    random_state=self.random_state
                )
                
                scores = cross_val_score(blending_ensemble, X, y, cv=cv_folds,
                                       scoring='neg_mean_squared_error')
                results[f'blending_{blend_method}'] = {
                    'ensemble': blending_ensemble,
                    'cv_score': -scores.mean(),
                    'cv_std': scores.std(),
                    'type': f'blending_{blend_method}'
                }
                
        # Test voting ensemble
        if 'voting' in ensemble_types:
            for voting_method in ['soft', 'hard', 'weighted']:
                voting_ensemble = VotingEnsemble(
                    base_models=base_models,
                    voting_method=voting_method
                )
                
                scores = cross_val_score(voting_ensemble, X, y, cv=cv_folds,
                                       scoring='neg_mean_squared_error')
                results[f'voting_{voting_method}'] = {
                    'ensemble': voting_ensemble,
                    'cv_score': -scores.mean(),
                    'cv_std': scores.std(),
                    'type': f'voting_{voting_method}'
                }
                
        # Find best ensemble
        best_ensemble_name = min(results.keys(), key=lambda x: results[x]['cv_score'])
        best_result = results[best_ensemble_name]
        
        logger.info(f"Best ensemble: {best_ensemble_name} (CV MSE: {best_result['cv_score']:.4f})")
        
        return {
            'best_ensemble': best_result['ensemble'],
            'best_ensemble_name': best_ensemble_name,
            'best_score': best_result['cv_score'],
            'all_results': results
        }
        
    def create_optimized_base_models(self) -> Dict[str, BaseEstimator]:
        """Create a set of optimized base models"""
        base_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        return base_models

def evaluate_ensemble(ensemble: BaseEstimator, X_test: np.ndarray, 
                     y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ensemble performance
    
    Args:
        ensemble: Trained ensemble model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with performance metrics
    """
    predictions = ensemble.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2_score': r2_score(y_test, predictions),
        'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
    }
    
    return metrics

# Example usage functions
def create_demand_forecasting_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Create and optimize ensemble for demand forecasting
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with ensemble and performance metrics
    """
    logger.info("Creating demand forecasting ensemble")
    
    # Create optimizer
    optimizer = EnsembleOptimizer(random_state=42)
    
    # Create base models
    base_models = optimizer.create_optimized_base_models()
    
    # Optimize ensemble
    optimization_result = optimizer.optimize_ensemble(
        X_train, y_train, base_models,
        ensemble_types=['stacking', 'blending', 'voting']
    )
    
    # Train best ensemble
    best_ensemble = optimization_result['best_ensemble']
    best_ensemble.fit(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = evaluate_ensemble(best_ensemble, X_test, y_test)
    
    result = {
        'ensemble': best_ensemble,
        'ensemble_name': optimization_result['best_ensemble_name'],
        'cv_score': optimization_result['best_score'],
        'test_metrics': test_metrics,
        'all_results': optimization_result['all_results']
    }
    
    logger.info(f"Ensemble created: {result['ensemble_name']}")
    logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
    logger.info(f"Test R²: {test_metrics['r2_score']:.4f}")
    
    return result