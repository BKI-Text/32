"""
Advanced ML Risk Assessment System for Beverly Knits AI Supply Chain Planner

This module implements state-of-the-art machine learning models for:
- Multi-dimensional supplier risk assessment
- Real-time anomaly detection
- Predictive supply chain disruption modeling
- Dynamic risk scoring with confidence intervals
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesClassifier
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

from ..core.domain.entities import RiskLevel, Supplier, SupplierMaterial
from ..core.domain.value_objects import SupplierId, MaterialId
from ..utils.error_handling import handle_errors, ErrorCategory

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRiskScore:
    """Enhanced risk score with detailed components and confidence metrics"""
    overall_score: float
    financial_risk: float
    operational_risk: float
    quality_risk: float
    delivery_risk: float
    market_risk: float
    compliance_risk: float
    risk_level: RiskLevel
    factors: List[str]
    confidence: float
    confidence_interval: Tuple[float, float]
    trend: str  # 'improving', 'stable', 'degrading'
    prediction_horizon: int  # days
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AdvancedAnomalyDetection:
    """Advanced anomaly detection with multiple detection methods"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    description: str
    severity: str
    detection_method: str
    confidence: float
    impact_assessment: str
    recommended_actions: List[str]
    historical_context: Dict[str, Any]

@dataclass
class SupplyChainInsight:
    """Supply chain insights and recommendations"""
    insight_type: str
    title: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    urgency: str  # 'immediate', 'short_term', 'long_term'
    recommendations: List[str]
    affected_suppliers: List[str]
    affected_materials: List[str]
    confidence: float

class AdvancedMLRiskAssessor:
    """
    Advanced ML-based risk assessment system with multiple algorithms
    and comprehensive supplier evaluation capabilities.
    """
    
    def __init__(self,
                 model_path: str = "models/risk_assessment/",
                 random_state: int = 42,
                 anomaly_contamination: float = 0.1,
                 risk_threshold_low: float = 0.3,
                 risk_threshold_medium: float = 0.7,
                 confidence_threshold: float = 0.8):
        """
        Initialize Advanced ML Risk Assessor.
        
        Args:
            model_path: Path to save/load trained models
            random_state: Random seed for reproducibility
            anomaly_contamination: Expected proportion of anomalies
            risk_threshold_low: Threshold for low risk classification
            risk_threshold_medium: Threshold for medium risk classification
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.random_state = random_state
        self.anomaly_contamination = anomaly_contamination
        self.risk_threshold_low = risk_threshold_low
        self.risk_threshold_medium = risk_threshold_medium
        self.confidence_threshold = confidence_threshold
        
        # Initialize models ensemble
        self.risk_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=1000
            )
        }
        
        self.anomaly_detectors = {
            'isolation_forest': IsolationForest(
                contamination=anomaly_contamination, random_state=random_state, n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(nu=anomaly_contamination),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        
        # Specialized predictors
        self.quality_predictor = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        self.delivery_predictor = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        self.financial_predictor = GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        )
        
        # Feature processing
        self.risk_scaler = RobustScaler()
        self.anomaly_scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = SelectKBest(f_classif, k=20)
        
        # Training status
        self.training_status = {
            'risk_models': False,
            'anomaly_detectors': False,
            'quality_predictor': False,
            'delivery_predictor': False,
            'financial_predictor': False
        }
        
        # Performance metrics
        self.model_performance = {}
        
        # Load existing models if available
        self._load_models()
    
    @handle_errors(ErrorCategory.ML_PROCESSING)
    def _prepare_advanced_features(self, suppliers_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare advanced feature engineering for supplier risk assessment.
        
        Args:
            suppliers_data: DataFrame with supplier information
            
        Returns:
            DataFrame with engineered features
        """
        features = suppliers_data.copy()
        
        # Financial stability features
        if 'revenue' in features.columns and 'expenses' in features.columns:
            features['profit_margin'] = (features['revenue'] - features['expenses']) / features['revenue']
            features['revenue_growth'] = features['revenue'].pct_change()
            features['expense_ratio'] = features['expenses'] / features['revenue']
        
        # Operational efficiency features
        if 'on_time_delivery_rate' in features.columns:
            features['delivery_consistency'] = features['on_time_delivery_rate'].rolling(window=12).std()
            features['delivery_trend'] = features['on_time_delivery_rate'].rolling(window=6).mean()
        
        if 'quality_score' in features.columns:
            features['quality_stability'] = features['quality_score'].rolling(window=12).std()
            features['quality_improvement'] = features['quality_score'].diff()
        
        # Market and compliance features
        if 'market_share' in features.columns:
            features['market_position'] = pd.qcut(features['market_share'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        if 'compliance_violations' in features.columns:
            features['compliance_trend'] = features['compliance_violations'].rolling(window=6).mean()
            features['compliance_severity'] = features['compliance_violations'] * features.get('violation_severity', 1)
        
        # Relationship and dependency features
        if 'years_of_partnership' in features.columns:
            features['partnership_strength'] = np.log1p(features['years_of_partnership'])
        
        if 'dependency_score' in features.columns:
            features['mutual_dependency'] = features['dependency_score'] * features.get('supplier_dependency', 1)
        
        # Geographic and supply chain features
        if 'geographic_risk' in features.columns:
            features['geo_diversification'] = features.groupby('region')['geographic_risk'].transform('count')
        
        # Interaction features
        if 'quality_score' in features.columns and 'on_time_delivery_rate' in features.columns:
            features['performance_index'] = features['quality_score'] * features['on_time_delivery_rate']
        
        # Temporal features
        features['assessment_month'] = pd.to_datetime(features.get('assessment_date', datetime.now())).dt.month
        features['assessment_quarter'] = pd.to_datetime(features.get('assessment_date', datetime.now())).dt.quarter
        
        # Handle missing values with advanced imputation
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if features[col].isnull().any():
                features[col] = features[col].fillna(features[col].median())
        
        categorical_columns = features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if features[col].isnull().any():
                features[col] = features[col].fillna(features[col].mode()[0] if not features[col].mode().empty else 'Unknown')
        
        return features
    
    @handle_errors(ErrorCategory.ML_PROCESSING)
    def train_risk_models(self, 
                         training_data: pd.DataFrame,
                         target_column: str = 'risk_level',
                         validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train ensemble of risk assessment models.
        
        Args:
            training_data: DataFrame with supplier data and risk labels
            target_column: Name of the target column
            validation_split: Proportion of data for validation
            
        Returns:
            Dictionary with training results and performance metrics
        """
        logger.info("Starting advanced risk model training")
        
        # Prepare features
        features = self._prepare_advanced_features(training_data)
        
        # Prepare target variable
        if target_column not in features.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(features[target_column])
        
        # Select features (exclude target and ID columns)
        feature_columns = [col for col in features.columns 
                          if col != target_column and not col.endswith('_id')]
        X = features[feature_columns]
        
        # Handle categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Scale features
        X_scaled = self.risk_scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Train ensemble models
        training_results = {}
        
        for name, model in self.risk_models.items():
            logger.info(f"Training {name} model")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = model.score(X_test, y_test)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                
                training_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'trained': True
                }
                
                # AUC score for multi-class
                if y_pred_proba is not None and len(np.unique(y)) > 2:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                        training_results[name]['auc'] = auc
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC for {name}: {e}")
                
                logger.info(f"{name} model trained successfully - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name} model: {e}")
                training_results[name] = {'trained': False, 'error': str(e)}
        
        # Save models and metadata
        self._save_models()
        self._save_training_metadata(training_results, target_encoder, feature_columns)
        
        self.training_status['risk_models'] = True
        self.model_performance = training_results
        
        return training_results
    
    @handle_errors(ErrorCategory.ML_PROCESSING)
    def train_anomaly_detectors(self, 
                               training_data: pd.DataFrame,
                               contamination: Optional[float] = None) -> Dict[str, Any]:
        """
        Train ensemble of anomaly detection models.
        
        Args:
            training_data: DataFrame with normal supplier data
            contamination: Expected proportion of anomalies
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting anomaly detector training")
        
        if contamination is not None:
            self.anomaly_contamination = contamination
        
        # Prepare features
        features = self._prepare_advanced_features(training_data)
        
        # Select numeric features for anomaly detection
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.anomaly_scaler.fit_transform(X)
        
        # Train anomaly detectors
        training_results = {}
        
        for name, detector in self.anomaly_detectors.items():
            logger.info(f"Training {name} anomaly detector")
            
            try:
                if name == 'dbscan':
                    # DBSCAN doesn't use contamination parameter
                    detector.fit(X_scaled)
                    labels = detector.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    training_results[name] = {
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': n_noise / len(X_scaled),
                        'trained': True
                    }
                else:
                    detector.fit(X_scaled)
                    
                    # Evaluate on training data
                    anomaly_scores = detector.decision_function(X_scaled)
                    predictions = detector.predict(X_scaled)
                    
                    n_anomalies = np.sum(predictions == -1)
                    anomaly_ratio = n_anomalies / len(X_scaled)
                    
                    training_results[name] = {
                        'anomaly_ratio': anomaly_ratio,
                        'score_mean': np.mean(anomaly_scores),
                        'score_std': np.std(anomaly_scores),
                        'trained': True
                    }
                
                logger.info(f"{name} anomaly detector trained successfully")
                
            except Exception as e:
                logger.error(f"Failed to train {name} anomaly detector: {e}")
                training_results[name] = {'trained': False, 'error': str(e)}
        
        # Save models
        self._save_models()
        
        self.training_status['anomaly_detectors'] = True
        
        return training_results
    
    @handle_errors(ErrorCategory.ML_PROCESSING)
    def assess_supplier_risk(self, 
                           supplier_data: Dict[str, Any],
                           include_trends: bool = True,
                           prediction_horizon: int = 90) -> EnhancedRiskScore:
        """
        Assess supplier risk using trained ensemble models.
        
        Args:
            supplier_data: Dictionary with supplier information
            include_trends: Whether to include trend analysis
            prediction_horizon: Days ahead for prediction
            
        Returns:
            EnhancedRiskScore object with detailed risk assessment
        """
        if not self.training_status['risk_models']:
            raise ValueError("Risk models not trained. Please train models first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([supplier_data])
        
        # Prepare features
        features = self._prepare_advanced_features(df)
        
        # Select and encode features
        feature_columns = [col for col in features.columns 
                          if not col.endswith('_id') and col in self.label_encoders or col in features.select_dtypes(include=[np.number]).columns]
        X = features[feature_columns]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                # Handle unknown categories
                try:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                except ValueError:
                    # Unknown category, use most frequent
                    X[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])
        
        # Feature selection and scaling
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.risk_scaler.transform(X_selected)
        
        # Ensemble predictions
        predictions = {}
        probabilities = {}
        
        for name, model in self.risk_models.items():
            if self.model_performance.get(name, {}).get('trained', False):
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = pred
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[0]
                        probabilities[name] = prob
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")
        
        # Ensemble prediction (majority vote)
        if predictions:
            risk_prediction = max(set(predictions.values()), key=list(predictions.values()).count)
            
            # Calculate confidence
            confidence = list(predictions.values()).count(risk_prediction) / len(predictions)
            
            # Average probabilities if available
            if probabilities:
                avg_probs = np.mean(list(probabilities.values()), axis=0)
                overall_score = np.max(avg_probs)
            else:
                overall_score = risk_prediction / (len(self.risk_models) - 1)  # Normalize
        else:
            risk_prediction = 1  # Medium risk as default
            overall_score = 0.5
            confidence = 0.0
        
        # Calculate component risks
        financial_risk = self._calculate_financial_risk(supplier_data)
        operational_risk = self._calculate_operational_risk(supplier_data)
        quality_risk = self._calculate_quality_risk(supplier_data)
        delivery_risk = self._calculate_delivery_risk(supplier_data)
        market_risk = self._calculate_market_risk(supplier_data)
        compliance_risk = self._calculate_compliance_risk(supplier_data)
        
        # Determine risk level
        if overall_score < self.risk_threshold_low:
            risk_level = RiskLevel.LOW
        elif overall_score < self.risk_threshold_medium:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.HIGH
        
        # Generate risk factors
        factors = self._identify_risk_factors(supplier_data, overall_score)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(overall_score, confidence)
        
        # Determine trend
        trend = self._determine_trend(supplier_data) if include_trends else 'stable'
        
        return EnhancedRiskScore(
            overall_score=overall_score,
            financial_risk=financial_risk,
            operational_risk=operational_risk,
            quality_risk=quality_risk,
            delivery_risk=delivery_risk,
            market_risk=market_risk,
            compliance_risk=compliance_risk,
            risk_level=risk_level,
            factors=factors,
            confidence=confidence,
            confidence_interval=confidence_interval,
            trend=trend,
            prediction_horizon=prediction_horizon
        )
    
    def _calculate_financial_risk(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate financial risk component"""
        financial_indicators = [
            'profit_margin', 'revenue_growth', 'debt_to_equity',
            'current_ratio', 'cash_flow_stability'
        ]
        
        risk_score = 0.0
        count = 0
        
        for indicator in financial_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if indicator == 'profit_margin':
                    risk_score += max(0, (0.1 - value) * 10)  # Higher risk if profit margin < 10%
                elif indicator == 'revenue_growth':
                    risk_score += max(0, (0.05 - value) * 5)  # Higher risk if growth < 5%
                elif indicator == 'debt_to_equity':
                    risk_score += min(1, value / 2)  # Higher risk with more debt
                elif indicator == 'current_ratio':
                    risk_score += max(0, (1.5 - value) * 0.5)  # Higher risk if ratio < 1.5
                elif indicator == 'cash_flow_stability':
                    risk_score += (1 - value)  # Higher risk with unstable cash flow
                count += 1
        
        return min(1.0, risk_score / max(1, count))
    
    def _calculate_operational_risk(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate operational risk component"""
        operational_indicators = [
            'capacity_utilization', 'equipment_age', 'workforce_stability',
            'process_automation', 'inventory_turnover'
        ]
        
        risk_score = 0.0
        count = 0
        
        for indicator in operational_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if indicator == 'capacity_utilization':
                    # Risk increases at very high or very low utilization
                    optimal_range = (0.7, 0.9)
                    if value < optimal_range[0]:
                        risk_score += (optimal_range[0] - value) * 2
                    elif value > optimal_range[1]:
                        risk_score += (value - optimal_range[1]) * 3
                elif indicator == 'equipment_age':
                    risk_score += min(1, value / 20)  # Higher risk with older equipment
                elif indicator == 'workforce_stability':
                    risk_score += (1 - value)  # Higher risk with unstable workforce
                elif indicator == 'process_automation':
                    risk_score += (1 - value) * 0.5  # Moderate risk with low automation
                elif indicator == 'inventory_turnover':
                    risk_score += max(0, (6 - value) / 6)  # Higher risk with low turnover
                count += 1
        
        return min(1.0, risk_score / max(1, count))
    
    def _calculate_quality_risk(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate quality risk component"""
        quality_indicators = [
            'quality_score', 'defect_rate', 'certification_status',
            'quality_improvement_trend', 'customer_complaints'
        ]
        
        risk_score = 0.0
        count = 0
        
        for indicator in quality_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if indicator == 'quality_score':
                    risk_score += max(0, (0.9 - value) * 2)  # Higher risk if quality < 90%
                elif indicator == 'defect_rate':
                    risk_score += min(1, value * 100)  # Higher risk with higher defects
                elif indicator == 'certification_status':
                    risk_score += (1 - value)  # Higher risk without certifications
                elif indicator == 'quality_improvement_trend':
                    risk_score += max(0, -value)  # Higher risk if trend is negative
                elif indicator == 'customer_complaints':
                    risk_score += min(1, value / 10)  # Higher risk with more complaints
                count += 1
        
        return min(1.0, risk_score / max(1, count))
    
    def _calculate_delivery_risk(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate delivery risk component"""
        delivery_indicators = [
            'on_time_delivery_rate', 'lead_time_variability',
            'transportation_reliability', 'buffer_inventory'
        ]
        
        risk_score = 0.0
        count = 0
        
        for indicator in delivery_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if indicator == 'on_time_delivery_rate':
                    risk_score += max(0, (0.95 - value) * 2)  # Higher risk if OTD < 95%
                elif indicator == 'lead_time_variability':
                    risk_score += min(1, value * 2)  # Higher risk with variable lead times
                elif indicator == 'transportation_reliability':
                    risk_score += (1 - value)  # Higher risk with unreliable transport
                elif indicator == 'buffer_inventory':
                    risk_score += max(0, (0.2 - value) * 2)  # Higher risk with low buffer
                count += 1
        
        return min(1.0, risk_score / max(1, count))
    
    def _calculate_market_risk(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate market risk component"""
        market_indicators = [
            'market_share', 'competitive_position', 'industry_growth',
            'regulatory_environment', 'economic_indicators'
        ]
        
        risk_score = 0.0
        count = 0
        
        for indicator in market_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if indicator == 'market_share':
                    risk_score += max(0, (0.1 - value) * 5)  # Higher risk with low market share
                elif indicator == 'competitive_position':
                    risk_score += (1 - value)  # Higher risk with weak position
                elif indicator == 'industry_growth':
                    risk_score += max(0, -value)  # Higher risk in declining industry
                elif indicator == 'regulatory_environment':
                    risk_score += (1 - value)  # Higher risk with unfavorable regulations
                elif indicator == 'economic_indicators':
                    risk_score += (1 - value)  # Higher risk with poor economic conditions
                count += 1
        
        return min(1.0, risk_score / max(1, count))
    
    def _calculate_compliance_risk(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate compliance risk component"""
        compliance_indicators = [
            'compliance_score', 'audit_results', 'certification_status',
            'regulatory_violations', 'ethical_standards'
        ]
        
        risk_score = 0.0
        count = 0
        
        for indicator in compliance_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if indicator == 'compliance_score':
                    risk_score += max(0, (0.9 - value) * 2)  # Higher risk if compliance < 90%
                elif indicator == 'audit_results':
                    risk_score += (1 - value)  # Higher risk with poor audit results
                elif indicator == 'certification_status':
                    risk_score += (1 - value)  # Higher risk without certifications
                elif indicator == 'regulatory_violations':
                    risk_score += min(1, value / 5)  # Higher risk with more violations
                elif indicator == 'ethical_standards':
                    risk_score += (1 - value)  # Higher risk with poor ethics
                count += 1
        
        return min(1.0, risk_score / max(1, count))
    
    def _identify_risk_factors(self, supplier_data: Dict[str, Any], overall_score: float) -> List[str]:
        """Identify key risk factors contributing to the overall score"""
        factors = []
        
        # Financial factors
        if supplier_data.get('profit_margin', 1) < 0.1:
            factors.append("Low profit margin")
        if supplier_data.get('revenue_growth', 0) < 0:
            factors.append("Declining revenue")
        if supplier_data.get('debt_to_equity', 0) > 2:
            factors.append("High debt levels")
        
        # Operational factors
        if supplier_data.get('capacity_utilization', 0.8) > 0.95:
            factors.append("Operating at capacity limits")
        if supplier_data.get('equipment_age', 0) > 15:
            factors.append("Aging equipment")
        if supplier_data.get('workforce_stability', 1) < 0.8:
            factors.append("High workforce turnover")
        
        # Quality factors
        if supplier_data.get('quality_score', 1) < 0.9:
            factors.append("Quality performance below standards")
        if supplier_data.get('defect_rate', 0) > 0.02:
            factors.append("High defect rate")
        
        # Delivery factors
        if supplier_data.get('on_time_delivery_rate', 1) < 0.95:
            factors.append("Poor on-time delivery performance")
        if supplier_data.get('lead_time_variability', 0) > 0.2:
            factors.append("Inconsistent lead times")
        
        # Market factors
        if supplier_data.get('market_share', 0.5) < 0.1:
            factors.append("Low market share")
        if supplier_data.get('industry_growth', 0) < 0:
            factors.append("Declining industry")
        
        # Compliance factors
        if supplier_data.get('compliance_score', 1) < 0.9:
            factors.append("Compliance issues")
        if supplier_data.get('regulatory_violations', 0) > 0:
            factors.append("Regulatory violations")
        
        return factors[:5]  # Return top 5 factors
    
    def _calculate_confidence_interval(self, score: float, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for risk score"""
        margin = (1 - confidence) * 0.2  # 20% margin when confidence is 0
        lower = max(0, score - margin)
        upper = min(1, score + margin)
        return (lower, upper)
    
    def _determine_trend(self, supplier_data: Dict[str, Any]) -> str:
        """Determine risk trend based on historical data"""
        # This would typically analyze historical data
        # For now, return based on improvement indicators
        improvement_indicators = [
            'quality_improvement_trend', 'delivery_improvement_trend',
            'financial_improvement_trend', 'performance_trend'
        ]
        
        positive_trends = 0
        negative_trends = 0
        
        for indicator in improvement_indicators:
            if indicator in supplier_data:
                value = supplier_data[indicator]
                if value > 0.1:
                    positive_trends += 1
                elif value < -0.1:
                    negative_trends += 1
        
        if positive_trends > negative_trends:
            return 'improving'
        elif negative_trends > positive_trends:
            return 'degrading'
        else:
            return 'stable'
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save risk models
            for name, model in self.risk_models.items():
                if hasattr(model, 'fit'):
                    joblib.dump(model, self.model_path / f'risk_model_{name}.joblib')
            
            # Save anomaly detectors
            for name, detector in self.anomaly_detectors.items():
                if hasattr(detector, 'fit'):
                    joblib.dump(detector, self.model_path / f'anomaly_detector_{name}.joblib')
            
            # Save scalers and encoders
            joblib.dump(self.risk_scaler, self.model_path / 'risk_scaler.joblib')
            joblib.dump(self.anomaly_scaler, self.model_path / 'anomaly_scaler.joblib')
            joblib.dump(self.feature_selector, self.model_path / 'feature_selector.joblib')
            
            # Save label encoders
            with open(self.model_path / 'label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            # Load risk models
            for name in self.risk_models.keys():
                model_path = self.model_path / f'risk_model_{name}.joblib'
                if model_path.exists():
                    self.risk_models[name] = joblib.load(model_path)
                    self.training_status['risk_models'] = True
            
            # Load anomaly detectors
            for name in self.anomaly_detectors.keys():
                detector_path = self.model_path / f'anomaly_detector_{name}.joblib'
                if detector_path.exists():
                    self.anomaly_detectors[name] = joblib.load(detector_path)
                    self.training_status['anomaly_detectors'] = True
            
            # Load scalers and encoders
            scaler_path = self.model_path / 'risk_scaler.joblib'
            if scaler_path.exists():
                self.risk_scaler = joblib.load(scaler_path)
            
            anomaly_scaler_path = self.model_path / 'anomaly_scaler.joblib'
            if anomaly_scaler_path.exists():
                self.anomaly_scaler = joblib.load(anomaly_scaler_path)
            
            feature_selector_path = self.model_path / 'feature_selector.joblib'
            if feature_selector_path.exists():
                self.feature_selector = joblib.load(feature_selector_path)
            
            # Load label encoders
            encoders_path = self.model_path / 'label_encoders.pkl'
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load some models: {e}")
    
    def _save_training_metadata(self, training_results: Dict[str, Any], 
                              target_encoder: LabelEncoder, 
                              feature_columns: List[str]):
        """Save training metadata"""
        metadata = {
            'training_date': datetime.now().isoformat(),
            'training_results': training_results,
            'target_classes': target_encoder.classes_.tolist(),
            'feature_columns': feature_columns,
            'model_parameters': {
                'random_state': self.random_state,
                'risk_threshold_low': self.risk_threshold_low,
                'risk_threshold_medium': self.risk_threshold_medium,
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        with open(self.model_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            'training_status': self.training_status,
            'model_performance': self.model_performance,
            'model_path': str(self.model_path),
            'parameters': {
                'random_state': self.random_state,
                'anomaly_contamination': self.anomaly_contamination,
                'risk_threshold_low': self.risk_threshold_low,
                'risk_threshold_medium': self.risk_threshold_medium,
                'confidence_threshold': self.confidence_threshold
            }
        }