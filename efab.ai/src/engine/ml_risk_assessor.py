"""
ML Risk Assessor for Beverly Knits AI Supply Chain Planner

This module implements advanced machine learning models for supplier risk assessment
and anomaly detection in supply chain operations.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans

from ..core.domain.entities import RiskLevel, Supplier, SupplierMaterial
from ..core.domain.value_objects import SupplierId, MaterialId

logger = logging.getLogger(__name__)

@dataclass
class RiskScore:
    """Risk score with components"""
    overall_score: float
    financial_risk: float
    operational_risk: float
    quality_risk: float
    delivery_risk: float
    risk_level: RiskLevel
    factors: List[str]
    confidence: float

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    description: str
    severity: str

class MLRiskAssessor:
    """
    Advanced ML-based risk assessment system for supplier evaluation
    and supply chain anomaly detection.
    """
    
    def __init__(self,
                 random_state: int = 42,
                 anomaly_contamination: float = 0.1,
                 risk_threshold_low: float = 0.3,
                 risk_threshold_medium: float = 0.7):
        """
        Initialize ML Risk Assessor.
        
        Args:
            random_state: Random seed for reproducibility
            anomaly_contamination: Expected proportion of anomalies
            risk_threshold_low: Threshold for low risk classification
            risk_threshold_medium: Threshold for medium risk classification
        """
        self.random_state = random_state
        self.anomaly_contamination = anomaly_contamination
        self.risk_threshold_low = risk_threshold_low
        self.risk_threshold_medium = risk_threshold_medium
        
        # Models
        self.risk_model = None
        self.anomaly_detector = None
        self.quality_predictor = None
        self.delivery_predictor = None
        
        # Scalers and encoders
        self.risk_scaler = StandardScaler()
        self.anomaly_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Training status
        self.is_risk_model_trained = False
        self.is_anomaly_detector_trained = False
        
    def _prepare_supplier_features(self, suppliers_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for supplier risk assessment.
        
        Args:
            suppliers_data: DataFrame with supplier information
            
        Returns:
            DataFrame with engineered features
        """
        features = suppliers_data.copy()
        
        # Financial features
        features['price_volatility'] = features.groupby('supplier_id')['cost_per_unit'].transform(
            lambda x: x.rolling(window=min(30, len(x))).std()
        )
        
        features['price_trend'] = features.groupby('supplier_id')['cost_per_unit'].transform(
            lambda x: x.diff().rolling(window=min(10, len(x))).mean()
        )
        
        # Operational features
        features['delivery_consistency'] = features.groupby('supplier_id')['lead_time_days'].transform(
            lambda x: 1.0 / (x.rolling(window=min(20, len(x))).std() + 1e-8)
        )
        
        features['order_frequency'] = features.groupby('supplier_id').cumcount() + 1
        
        # Quality features
        if 'quality_score' in features.columns:
            features['quality_trend'] = features.groupby('supplier_id')['quality_score'].transform(
                lambda x: x.diff().rolling(window=min(10, len(x))).mean()
            )
        else:
            features['quality_score'] = 0.8  # Default quality score
            features['quality_trend'] = 0.0
            
        # Relationship features
        features['supplier_age_days'] = (datetime.now() - pd.to_datetime(features['created_at'])).dt.days
        features['materials_count'] = features.groupby('supplier_id')['material_id'].transform('count')
        
        # Risk indicators
        features['high_moq_flag'] = (features['moq_amount'] > features['moq_amount'].quantile(0.8)).astype(int)
        features['long_lead_time_flag'] = (features['lead_time_days'] > features['lead_time_days'].quantile(0.8)).astype(int)
        
        # Categorical encoding
        categorical_columns = ['supplier_id', 'material_id']
        for col in categorical_columns:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features[col].astype(str))
                else:
                    features[f'{col}_encoded'] = self.label_encoders[col].transform(features[col].astype(str))
                    
        return features
        
    def _calculate_risk_components(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate individual risk components.
        
        Args:
            features: DataFrame with prepared features
            
        Returns:
            DataFrame with risk components
        """
        risk_components = features.copy()
        
        # Financial risk (0-1, higher is more risky)
        risk_components['financial_risk'] = np.clip(
            (features['price_volatility'].fillna(0) * 0.4 + 
             np.abs(features['price_trend'].fillna(0)) * 0.3 +
             (features['cost_per_unit'] / features['cost_per_unit'].max()) * 0.3), 0, 1
        )
        
        # Operational risk
        risk_components['operational_risk'] = np.clip(
            ((1.0 / features['delivery_consistency'].fillna(1)) * 0.5 +
             (features['lead_time_days'] / features['lead_time_days'].max()) * 0.3 +
             features['long_lead_time_flag'] * 0.2), 0, 1
        )
        
        # Quality risk
        risk_components['quality_risk'] = np.clip(
            ((1.0 - features['quality_score']) * 0.6 +
             np.abs(features['quality_trend'].fillna(0)) * 0.4), 0, 1
        )
        
        # Delivery risk  
        risk_components['delivery_risk'] = np.clip(
            (risk_components['operational_risk'] * 0.6 +
             features['high_moq_flag'] * 0.2 +
             (1.0 / (features['supplier_age_days'] / 365 + 1)) * 0.2), 0, 1
        )
        
        return risk_components
        
    def train_risk_model(self, historical_data: pd.DataFrame, target_column: str = 'risk_level'):
        """
        Train machine learning model for risk assessment.
        
        Args:
            historical_data: Historical supplier performance data
            target_column: Name of target risk level column
        """
        try:
            logger.info("Training ML risk assessment model...")
            
            # Prepare features
            features = self._prepare_supplier_features(historical_data)
            risk_components = self._calculate_risk_components(features)
            
            # Select feature columns
            feature_columns = [
                'price_volatility', 'price_trend', 'delivery_consistency',
                'order_frequency', 'quality_score', 'quality_trend',
                'supplier_age_days', 'materials_count', 'high_moq_flag',
                'long_lead_time_flag', 'financial_risk', 'operational_risk',
                'quality_risk', 'delivery_risk'
            ]
            
            # Filter available columns
            available_columns = [col for col in feature_columns if col in risk_components.columns]
            X = risk_components[available_columns].fillna(0)
            
            # Create target variable if not exists
            if target_column not in historical_data.columns:
                # Create synthetic target based on risk components
                overall_risk = (
                    risk_components['financial_risk'] * 0.3 +
                    risk_components['operational_risk'] * 0.25 +
                    risk_components['quality_risk'] * 0.25 +
                    risk_components['delivery_risk'] * 0.2
                )
                
                y = pd.cut(overall_risk, 
                          bins=[0, self.risk_threshold_low, self.risk_threshold_medium, 1.0],
                          labels=['low', 'medium', 'high'])
            else:
                y = historical_data[target_column]
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.risk_scaler.fit_transform(X_train)
            X_test_scaled = self.risk_scaler.transform(X_test)
            
            # Train Random Forest classifier
            self.risk_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            self.risk_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.risk_model.predict(X_test_scaled)
            y_pred_proba = self.risk_model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Multi-class ROC AUC
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                auc_score = 0.5
                
            logger.info(f"Risk model trained successfully. Accuracy: {report['accuracy']:.3f}, AUC: {auc_score:.3f}")
            
            self.is_risk_model_trained = True
            
        except Exception as e:
            logger.error(f"Error training risk model: {e}")
            raise
            
    def train_anomaly_detector(self, historical_data: pd.DataFrame):
        """
        Train anomaly detection model.
        
        Args:
            historical_data: Historical supply chain data
        """
        try:
            logger.info("Training anomaly detection model...")
            
            # Prepare features for anomaly detection
            features = self._prepare_supplier_features(historical_data)
            
            # Select numerical features for anomaly detection
            numerical_columns = [
                'cost_per_unit', 'lead_time_days', 'moq_amount', 'reliability_score',
                'price_volatility', 'price_trend', 'delivery_consistency',
                'order_frequency', 'quality_score', 'supplier_age_days'
            ]
            
            # Filter available columns
            available_columns = [col for col in numerical_columns if col in features.columns]
            X = features[available_columns].fillna(features[available_columns].median())
            
            # Scale features
            X_scaled = self.anomaly_scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.anomaly_detector = IsolationForest(
                contamination=self.anomaly_contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0
            )
            
            self.anomaly_detector.fit(X_scaled)
            
            # Evaluate anomaly detection
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            anomaly_predictions = self.anomaly_detector.predict(X_scaled)
            
            anomaly_count = np.sum(anomaly_predictions == -1)
            anomaly_percentage = (anomaly_count / len(X)) * 100
            
            logger.info(f"Anomaly detector trained successfully. Detected {anomaly_count} anomalies ({anomaly_percentage:.1f}%)")
            
            self.is_anomaly_detector_trained = True
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            raise
            
    def predict_supplier_risk(self, supplier_data: pd.DataFrame) -> List[RiskScore]:
        """
        Predict risk scores for suppliers.
        
        Args:
            supplier_data: DataFrame with supplier information
            
        Returns:
            List of RiskScore objects
        """
        if not self.is_risk_model_trained:
            logger.warning("Risk model not trained. Using rule-based assessment.")
            return self._rule_based_risk_assessment(supplier_data)
            
        try:
            # Prepare features
            features = self._prepare_supplier_features(supplier_data)
            risk_components = self._calculate_risk_components(features)
            
            # Select feature columns
            feature_columns = [
                'price_volatility', 'price_trend', 'delivery_consistency',
                'order_frequency', 'quality_score', 'quality_trend',
                'supplier_age_days', 'materials_count', 'high_moq_flag',
                'long_lead_time_flag', 'financial_risk', 'operational_risk',
                'quality_risk', 'delivery_risk'
            ]
            
            # Filter available columns
            available_columns = [col for col in feature_columns if col in risk_components.columns]
            X = risk_components[available_columns].fillna(0)
            
            # Scale features
            X_scaled = self.risk_scaler.transform(X)
            
            # Predict risk
            risk_predictions = self.risk_model.predict(X_scaled)
            risk_probabilities = self.risk_model.predict_proba(X_scaled)
            
            # Create RiskScore objects
            risk_scores = []
            
            for i, (_, row) in enumerate(supplier_data.iterrows()):
                risk_level_str = risk_predictions[i]
                risk_level = RiskLevel(risk_level_str)
                
                # Get confidence (max probability)
                confidence = np.max(risk_probabilities[i])
                
                # Overall risk score (0-1)
                overall_score = (
                    risk_components.iloc[i]['financial_risk'] * 0.3 +
                    risk_components.iloc[i]['operational_risk'] * 0.25 +
                    risk_components.iloc[i]['quality_risk'] * 0.25 +
                    risk_components.iloc[i]['delivery_risk'] * 0.2
                )
                
                # Identify key risk factors
                factors = []
                if risk_components.iloc[i]['financial_risk'] > 0.5:
                    factors.append("High financial risk")
                if risk_components.iloc[i]['operational_risk'] > 0.5:
                    factors.append("Operational concerns")
                if risk_components.iloc[i]['quality_risk'] > 0.5:
                    factors.append("Quality issues")
                if risk_components.iloc[i]['delivery_risk'] > 0.5:
                    factors.append("Delivery reliability")
                    
                risk_score = RiskScore(
                    overall_score=float(overall_score),
                    financial_risk=float(risk_components.iloc[i]['financial_risk']),
                    operational_risk=float(risk_components.iloc[i]['operational_risk']),
                    quality_risk=float(risk_components.iloc[i]['quality_risk']),
                    delivery_risk=float(risk_components.iloc[i]['delivery_risk']),
                    risk_level=risk_level,
                    factors=factors,
                    confidence=float(confidence)
                )
                
                risk_scores.append(risk_score)
                
            return risk_scores
            
        except Exception as e:
            logger.error(f"Error predicting supplier risk: {e}")
            return self._rule_based_risk_assessment(supplier_data)
            
    def _rule_based_risk_assessment(self, supplier_data: pd.DataFrame) -> List[RiskScore]:
        """
        Fallback rule-based risk assessment.
        
        Args:
            supplier_data: DataFrame with supplier information
            
        Returns:
            List of RiskScore objects
        """
        risk_scores = []
        
        for _, row in supplier_data.iterrows():
            # Simple rule-based assessment
            financial_risk = min(row.get('cost_per_unit', 100) / 1000, 1.0)
            operational_risk = min(row.get('lead_time_days', 30) / 60, 1.0)
            quality_risk = 1.0 - row.get('quality_score', 0.8)
            delivery_risk = 1.0 - row.get('reliability_score', 0.8)
            
            overall_score = (financial_risk * 0.3 + operational_risk * 0.25 + 
                           quality_risk * 0.25 + delivery_risk * 0.2)
            
            if overall_score < self.risk_threshold_low:
                risk_level = RiskLevel.LOW
            elif overall_score < self.risk_threshold_medium:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.HIGH
                
            risk_score = RiskScore(
                overall_score=float(overall_score),
                financial_risk=float(financial_risk),
                operational_risk=float(operational_risk),
                quality_risk=float(quality_risk),
                delivery_risk=float(delivery_risk),
                risk_level=risk_level,
                factors=["Rule-based assessment"],
                confidence=0.7
            )
            
            risk_scores.append(risk_score)
            
        return risk_scores
        
    def detect_anomalies(self, current_data: pd.DataFrame) -> List[AnomalyDetection]:
        """
        Detect anomalies in current supply chain data.
        
        Args:
            current_data: Current supply chain data
            
        Returns:
            List of AnomalyDetection objects
        """
        if not self.is_anomaly_detector_trained:
            logger.warning("Anomaly detector not trained. Using rule-based detection.")
            return self._rule_based_anomaly_detection(current_data)
            
        try:
            # Prepare features
            features = self._prepare_supplier_features(current_data)
            
            # Select numerical features
            numerical_columns = [
                'cost_per_unit', 'lead_time_days', 'moq_amount', 'reliability_score',
                'price_volatility', 'price_trend', 'delivery_consistency',
                'order_frequency', 'quality_score', 'supplier_age_days'
            ]
            
            # Filter available columns
            available_columns = [col for col in numerical_columns if col in features.columns]
            X = features[available_columns].fillna(features[available_columns].median())
            
            # Scale features
            X_scaled = self.anomaly_scaler.transform(X)
            
            # Detect anomalies
            anomaly_predictions = self.anomaly_detector.predict(X_scaled)
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            
            # Create AnomalyDetection objects
            anomaly_detections = []
            
            for i, (_, row) in enumerate(current_data.iterrows()):
                is_anomaly = anomaly_predictions[i] == -1
                anomaly_score = float(anomaly_scores[i])
                
                if is_anomaly:
                    # Determine anomaly type
                    anomaly_type = self._classify_anomaly_type(row, features.iloc[i])
                    
                    # Determine severity
                    if anomaly_score < -0.5:
                        severity = "High"
                    elif anomaly_score < -0.2:
                        severity = "Medium"
                    else:
                        severity = "Low"
                        
                    description = f"Anomaly detected in supplier {row.get('supplier_id', 'unknown')}: {anomaly_type}"
                    
                else:
                    anomaly_type = "Normal"
                    severity = "None"
                    description = "No anomaly detected"
                    
                anomaly_detection = AnomalyDetection(
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    anomaly_type=anomaly_type,
                    description=description,
                    severity=severity
                )
                
                anomaly_detections.append(anomaly_detection)
                
            return anomaly_detections
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return self._rule_based_anomaly_detection(current_data)
            
    def _classify_anomaly_type(self, row: pd.Series, features: pd.Series) -> str:
        """
        Classify the type of anomaly detected.
        
        Args:
            row: Original data row
            features: Engineered features row
            
        Returns:
            Anomaly type description
        """
        # Simple anomaly type classification
        if features.get('price_volatility', 0) > 0.5:
            return "High price volatility"
        elif row.get('lead_time_days', 0) > 60:
            return "Unusually long lead time"
        elif row.get('cost_per_unit', 0) > row.get('cost_per_unit', 0) * 2:
            return "Price spike"
        elif features.get('delivery_consistency', 1) < 0.3:
            return "Inconsistent delivery"
        else:
            return "General anomaly"
            
    def _rule_based_anomaly_detection(self, current_data: pd.DataFrame) -> List[AnomalyDetection]:
        """
        Fallback rule-based anomaly detection.
        
        Args:
            current_data: Current supply chain data
            
        Returns:
            List of AnomalyDetection objects
        """
        anomaly_detections = []
        
        for _, row in current_data.iterrows():
            # Simple rule-based anomaly detection
            is_anomaly = False
            anomaly_type = "Normal"
            severity = "None"
            description = "No anomaly detected"
            
            # Check for obvious anomalies
            if row.get('lead_time_days', 0) > 90:
                is_anomaly = True
                anomaly_type = "Long lead time"
                severity = "High"
                description = f"Lead time of {row.get('lead_time_days')} days is unusually long"
            elif row.get('cost_per_unit', 0) > 1000:
                is_anomaly = True
                anomaly_type = "High cost"
                severity = "Medium"
                description = f"Cost of {row.get('cost_per_unit')} is unusually high"
            elif row.get('reliability_score', 1) < 0.3:
                is_anomaly = True
                anomaly_type = "Low reliability"
                severity = "Medium"
                description = f"Reliability score of {row.get('reliability_score')} is very low"
                
            anomaly_detection = AnomalyDetection(
                is_anomaly=is_anomaly,
                anomaly_score=-0.5 if is_anomaly else 0.1,
                anomaly_type=anomaly_type,
                description=description,
                severity=severity
            )
            
            anomaly_detections.append(anomaly_detection)
            
        return anomaly_detections
        
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all ML models.
        
        Returns:
            Dictionary with model status information
        """
        return {
            "risk_model_trained": self.is_risk_model_trained,
            "anomaly_detector_trained": self.is_anomaly_detector_trained,
            "risk_thresholds": {
                "low": self.risk_threshold_low,
                "medium": self.risk_threshold_medium
            },
            "anomaly_contamination": self.anomaly_contamination,
            "models_available": {
                "risk_model": self.risk_model is not None,
                "anomaly_detector": self.anomaly_detector is not None
            }
        }
        
    def save_models(self, filepath: str):
        """Save all trained models to file"""
        try:
            import pickle
            
            models_data = {
                'risk_model': self.risk_model,
                'anomaly_detector': self.anomaly_detector,
                'risk_scaler': self.risk_scaler,
                'anomaly_scaler': self.anomaly_scaler,
                'label_encoders': self.label_encoders,
                'config': {
                    'random_state': self.random_state,
                    'anomaly_contamination': self.anomaly_contamination,
                    'risk_threshold_low': self.risk_threshold_low,
                    'risk_threshold_medium': self.risk_threshold_medium
                },
                'training_status': {
                    'risk_model_trained': self.is_risk_model_trained,
                    'anomaly_detector_trained': self.is_anomaly_detector_trained
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(models_data, f)
                
            logger.info(f"ML risk models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
            
    def load_models(self, filepath: str):
        """Load trained models from file"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                models_data = pickle.load(f)
                
            self.risk_model = models_data['risk_model']
            self.anomaly_detector = models_data['anomaly_detector']
            self.risk_scaler = models_data['risk_scaler']
            self.anomaly_scaler = models_data['anomaly_scaler']
            self.label_encoders = models_data['label_encoders']
            
            # Restore config
            config = models_data['config']
            self.random_state = config['random_state']
            self.anomaly_contamination = config['anomaly_contamination']
            self.risk_threshold_low = config['risk_threshold_low']
            self.risk_threshold_medium = config['risk_threshold_medium']
            
            # Restore training status
            training_status = models_data['training_status']
            self.is_risk_model_trained = training_status['risk_model_trained']
            self.is_anomaly_detector_trained = training_status['anomaly_detector_trained']
            
            logger.info(f"ML risk models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise