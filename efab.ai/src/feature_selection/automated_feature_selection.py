#!/usr/bin/env python3
"""
Automated Feature Selection Pipeline
Beverly Knits AI Supply Chain Planner
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AutomatedFeatureSelector:
    """Automated feature selection with multiple algorithms"""
    
    def __init__(self, problem_type: str = 'regression', scoring: str = None):
        """
        Initialize feature selector
        
        Args:
            problem_type: 'regression' or 'classification'
            scoring: Scoring method for evaluation
        """
        self.problem_type = problem_type
        self.scoring = scoring or ('r2' if problem_type == 'regression' else 'accuracy')
        
        # Feature selection results
        self.selection_results = {}
        self.selected_features = []
        self.feature_scores = {}
        self.best_selector = None
        self.best_score = -np.inf
        
        # Configure selectors based on problem type
        self._configure_selectors()
    
    def _configure_selectors(self):
        """Configure feature selectors based on problem type"""
        if self.problem_type == 'regression':
            self.base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scoring_func = f_regression
            self.mutual_info_func = mutual_info_regression
            self.lasso_estimator = Lasso(alpha=0.1, random_state=42)
        else:
            self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scoring_func = f_classif
            self.mutual_info_func = mutual_info_classif
            self.lasso_estimator = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Fit feature selection methods
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with selection results
        """
        logger.info(f"Starting automated feature selection with {X.shape[1]} features")
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy for sklearn compatibility
        X_array = X.values
        y_array = y.values
        
        # Apply multiple feature selection methods
        self._apply_variance_threshold(X_array, X.columns)
        self._apply_univariate_selection(X_array, y_array, X.columns)
        self._apply_mutual_info_selection(X_array, y_array, X.columns)
        self._apply_recursive_feature_elimination(X_array, y_array, X.columns)
        self._apply_model_based_selection(X_array, y_array, X.columns)
        
        # Find best feature subset
        self._find_best_features(X_array, y_array)
        
        logger.info(f"Feature selection completed. Best method: {self.best_selector}")
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return self.get_results()
    
    def _apply_variance_threshold(self, X: np.ndarray, feature_names: List[str]):
        """Apply variance threshold selection"""
        try:
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            
            selected_features = [feature_names[i] for i in range(len(feature_names)) 
                               if selector.get_support()[i]]
            
            self.selection_results['variance_threshold'] = {
                'selected_features': selected_features,
                'n_features': len(selected_features),
                'selector': selector,
                'method': 'variance_threshold'
            }
            
            logger.info(f"Variance threshold: {len(selected_features)} features")
            
        except Exception as e:
            logger.error(f"Error in variance threshold selection: {e}")
    
    def _apply_univariate_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Apply univariate statistical selection"""
        try:
            # Select top k features
            for k in [10, 20, 50]:
                if k > X.shape[1]:
                    continue
                    
                selector = SelectKBest(score_func=self.scoring_func, k=k)
                X_selected = selector.fit_transform(X, y)
                
                selected_features = [feature_names[i] for i in range(len(feature_names)) 
                                   if selector.get_support()[i]]
                
                scores = selector.scores_
                feature_scores = {feature_names[i]: scores[i] for i in range(len(feature_names))}
                
                self.selection_results[f'univariate_k{k}'] = {
                    'selected_features': selected_features,
                    'n_features': len(selected_features),
                    'selector': selector,
                    'scores': feature_scores,
                    'method': 'univariate'
                }
                
                logger.info(f"Univariate selection (k={k}): {len(selected_features)} features")
                
        except Exception as e:
            logger.error(f"Error in univariate selection: {e}")
    
    def _apply_mutual_info_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Apply mutual information selection"""
        try:
            # Select top k features based on mutual information
            for k in [10, 20, 50]:
                if k > X.shape[1]:
                    continue
                    
                selector = SelectKBest(score_func=self.mutual_info_func, k=k)
                X_selected = selector.fit_transform(X, y)
                
                selected_features = [feature_names[i] for i in range(len(feature_names)) 
                                   if selector.get_support()[i]]
                
                scores = selector.scores_
                feature_scores = {feature_names[i]: scores[i] for i in range(len(feature_names))}
                
                self.selection_results[f'mutual_info_k{k}'] = {
                    'selected_features': selected_features,
                    'n_features': len(selected_features),
                    'selector': selector,
                    'scores': feature_scores,
                    'method': 'mutual_info'
                }
                
                logger.info(f"Mutual info selection (k={k}): {len(selected_features)} features")
                
        except Exception as e:
            logger.error(f"Error in mutual info selection: {e}")
    
    def _apply_recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Apply recursive feature elimination"""
        try:
            # RFE with cross-validation
            selector = RFECV(
                estimator=self.base_estimator,
                step=1,
                cv=3,
                scoring=self.scoring,
                n_jobs=-1
            )
            
            X_selected = selector.fit_transform(X, y)
            
            selected_features = [feature_names[i] for i in range(len(feature_names)) 
                               if selector.get_support()[i]]
            
            # Get feature rankings
            rankings = selector.ranking_
            feature_rankings = {feature_names[i]: rankings[i] for i in range(len(feature_names))}
            
            self.selection_results['rfe_cv'] = {
                'selected_features': selected_features,
                'n_features': len(selected_features),
                'selector': selector,
                'rankings': feature_rankings,
                'method': 'rfe_cv',
                'cv_scores': selector.cv_results_['mean_test_score']
            }
            
            logger.info(f"RFE-CV selection: {len(selected_features)} features")
            
        except Exception as e:
            logger.error(f"Error in RFE selection: {e}")
    
    def _apply_model_based_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Apply model-based feature selection"""
        try:
            # Random Forest feature importance
            rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42) if self.problem_type == 'regression' else RandomForestClassifier(n_estimators=100, random_state=42)
            rf_selector = SelectFromModel(
                rf_estimator,
                threshold='mean'
            )
            
            X_selected = rf_selector.fit_transform(X, y)
            
            selected_features = [feature_names[i] for i in range(len(feature_names)) 
                               if rf_selector.get_support()[i]]
            
            # Get feature importances
            importances = rf_estimator.feature_importances_
            feature_importances = {feature_names[i]: importances[i] for i in range(len(feature_names))}
            
            self.selection_results['random_forest'] = {
                'selected_features': selected_features,
                'n_features': len(selected_features),
                'selector': rf_selector,
                'importances': feature_importances,
                'method': 'random_forest'
            }
            
            logger.info(f"Random Forest selection: {len(selected_features)} features")
            
            # Lasso-based selection (for regression)
            if self.problem_type == 'regression' and self.lasso_estimator:
                lasso_selector = SelectFromModel(
                    self.lasso_estimator,
                    threshold='mean'
                )
                
                X_selected = lasso_selector.fit_transform(X, y)
                
                selected_features = [feature_names[i] for i in range(len(feature_names)) 
                                   if lasso_selector.get_support()[i]]
                
                # Get Lasso coefficients
                coefficients = self.lasso_estimator.coef_
                feature_coefficients = {feature_names[i]: coefficients[i] for i in range(len(feature_names))}
                
                self.selection_results['lasso'] = {
                    'selected_features': selected_features,
                    'n_features': len(selected_features),
                    'selector': lasso_selector,
                    'coefficients': feature_coefficients,
                    'method': 'lasso'
                }
                
                logger.info(f"Lasso selection: {len(selected_features)} features")
                
        except Exception as e:
            logger.error(f"Error in model-based selection: {e}")
    
    def _find_best_features(self, X: np.ndarray, y: np.ndarray):
        """Find the best feature subset by evaluating each method"""
        logger.info("Evaluating feature selection methods...")
        
        # Evaluate each selection method
        for method_name, result in self.selection_results.items():
            try:
                selected_features = result['selected_features']
                
                if len(selected_features) == 0:
                    continue
                
                # Get indices of selected features
                feature_indices = [self.feature_names.index(feat) for feat in selected_features]
                X_selected = X[:, feature_indices]
                
                # Evaluate using cross-validation
                scores = cross_val_score(
                    self.base_estimator,
                    X_selected,
                    y,
                    cv=3,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                # Store evaluation results
                result['cv_score'] = mean_score
                result['cv_std'] = std_score
                result['cv_scores'] = scores.tolist()
                
                logger.info(f"{method_name}: {mean_score:.4f} (±{std_score:.4f})")
                
                # Update best method
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_selector = method_name
                    self.selected_features = selected_features
                    
            except Exception as e:
                logger.error(f"Error evaluating {method_name}: {e}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using selected features
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.selected_features:
            logger.warning("No features selected. Returning original data.")
            return X
        
        # Select only the chosen features
        available_features = [feat for feat in self.selected_features if feat in X.columns]
        
        if len(available_features) != len(self.selected_features):
            logger.warning(f"Some selected features not found in data. Using {len(available_features)} features.")
        
        return X[available_features]
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive feature selection results"""
        return {
            'best_method': self.best_selector,
            'best_score': self.best_score,
            'selected_features': self.selected_features,
            'n_selected': len(self.selected_features),
            'n_original': len(self.feature_names),
            'reduction_ratio': 1 - len(self.selected_features) / len(self.feature_names),
            'all_results': self.selection_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance summary across all methods"""
        if not self.selection_results:
            return pd.DataFrame()
        
        # Collect feature scores from all methods
        all_scores = {}
        
        for method_name, result in self.selection_results.items():
            method_scores = {}
            
            if 'scores' in result:
                method_scores = result['scores']
            elif 'importances' in result:
                method_scores = result['importances']
            elif 'coefficients' in result:
                method_scores = {k: abs(v) for k, v in result['coefficients'].items()}
            elif 'rankings' in result:
                # Convert rankings to scores (lower rank = higher score)
                rankings = result['rankings']
                max_rank = max(rankings.values())
                method_scores = {k: max_rank - v + 1 for k, v in rankings.items()}
            
            # Normalize scores
            if method_scores:
                max_score = max(method_scores.values())
                if max_score > 0:
                    method_scores = {k: v / max_score for k, v in method_scores.items()}
            
            all_scores[method_name] = method_scores
        
        # Create summary DataFrame
        summary_data = []
        for feature in self.feature_names:
            feature_data = {'feature': feature}
            
            # Add scores from each method
            for method_name, scores in all_scores.items():
                feature_data[method_name] = scores.get(feature, 0.0)
            
            # Calculate average score
            method_scores = [scores.get(feature, 0.0) for scores in all_scores.values()]
            feature_data['average_score'] = np.mean(method_scores)
            
            # Check if feature was selected by best method
            if self.best_selector and self.best_selector in self.selection_results:
                best_selected = self.selection_results[self.best_selector]['selected_features']
                feature_data['selected_by_best'] = feature in best_selected
            else:
                feature_data['selected_by_best'] = False
            
            summary_data.append(feature_data)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('average_score', ascending=False)
        
        return df
    
    def save_results(self, output_path: str):
        """Save feature selection results"""
        results = self.get_results()
        
        # Convert numpy arrays to lists for JSON serialization
        for method_name, result in results['all_results'].items():
            if 'cv_scores' in result:
                result['cv_scores'] = [float(x) for x in result['cv_scores']]
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Feature selection results saved to {output_path}")

class FeatureSelectionPipeline:
    """Complete feature selection pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.selector = None
        self.scaler = StandardScaler()
        self.results = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'problem_type': 'regression',
            'scoring': 'r2',
            'remove_correlated': True,
            'correlation_threshold': 0.95,
            'remove_constant': True,
            'scale_features': True,
            'save_results': True,
            'output_dir': 'data/feature_selection'
        }
    
    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Run complete feature selection pipeline
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Pipeline results
        """
        logger.info("Starting feature selection pipeline")
        
        # Create output directory if specified
        if self.config.get('save_results', True):
            output_dir = Path(self.config.get('output_dir', 'data/feature_selection'))
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Preprocessing
        X_processed = self._preprocess_features(X)
        
        # Step 2: Feature selection
        self.selector = AutomatedFeatureSelector(
            problem_type=self.config.get('problem_type', 'regression'),
            scoring=self.config.get('scoring', 'r2')
        )
        
        selection_results = self.selector.fit(X_processed, y)
        
        # Step 3: Get selected features
        X_selected = self.selector.transform(X_processed)
        
        # Step 4: Final evaluation
        final_score = self._evaluate_final_features(X_selected, y)
        
        # Step 5: Generate reports
        self.results = {
            'original_features': X.shape[1],
            'selected_features': X_selected.shape[1],
            'feature_reduction': 1 - X_selected.shape[1] / X.shape[1],
            'final_score': final_score,
            'selected_feature_names': X_selected.columns.tolist(),
            'selection_results': selection_results,
            'config': self.config
        }
        
        # Save results
        if self.config.get('save_results', True):
            output_dir = Path(self.config.get('output_dir', 'data/feature_selection'))
            self._save_results(output_dir)
        
        logger.info(f"Feature selection pipeline completed")
        logger.info(f"Reduced features from {X.shape[1]} to {X_selected.shape[1]}")
        logger.info(f"Final score: {final_score:.4f}")
        
        return self.results
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features"""
        logger.info("Preprocessing features...")
        
        X_processed = X.copy()
        
        # Remove constant features
        if self.config.get('remove_constant', True):
            constant_features = X_processed.columns[X_processed.nunique() <= 1]
            if len(constant_features) > 0:
                X_processed = X_processed.drop(columns=constant_features)
                logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        if self.config.get('remove_correlated', True):
            correlation_matrix = X_processed.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > self.config.get('correlation_threshold', 0.95))
            ]
            
            if len(high_corr_features) > 0:
                X_processed = X_processed.drop(columns=high_corr_features)
                logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Scale features
        if self.config.get('scale_features', True):
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
            logger.info("Features scaled")
        
        return X_processed
    
    def _evaluate_final_features(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate final selected features"""
        if X.shape[1] == 0:
            return 0.0
        
        # Use the same estimator as feature selection
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=5,
            scoring=self.config.get('scoring', 'r2'),
            n_jobs=-1
        )
        
        return scores.mean()
    
    def _save_results(self, output_dir: Path):
        """Save pipeline results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = output_dir / f"feature_selection_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save feature importance summary
        if self.selector:
            importance_df = self.selector.get_feature_importance_summary()
            importance_file = output_dir / f"feature_importance_{timestamp}.csv"
            importance_df.to_csv(importance_file, index=False)
            
            # Save selected features list
            selected_features_file = output_dir / f"selected_features_{timestamp}.json"
            with open(selected_features_file, 'w') as f:
                json.dump(self.results['selected_feature_names'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using selected features"""
        if not self.selector:
            raise ValueError("Pipeline not fitted. Call run() first.")
        
        # Apply preprocessing
        X_processed = self._preprocess_features(X)
        
        # Apply feature selection
        X_selected = self.selector.transform(X_processed)
        
        return X_selected

# Global feature selection instance
feature_selection_pipeline = FeatureSelectionPipeline()

def run_feature_selection(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run feature selection pipeline"""
    pipeline = FeatureSelectionPipeline(config)
    return pipeline.run(X, y)

def get_feature_selection_pipeline() -> FeatureSelectionPipeline:
    """Get global feature selection pipeline"""
    return feature_selection_pipeline