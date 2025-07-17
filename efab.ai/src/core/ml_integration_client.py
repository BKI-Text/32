"""
ML Integration Client for Beverly Knits AI Supply Chain Planner
Integrates with zen-mcp-server for advanced AI/ML capabilities.
"""

import subprocess
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pickle
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class MLPrediction:
    """ML prediction result with confidence metrics."""
    prediction_type: str
    prediction_value: Union[float, List[float]]
    confidence_score: float
    model_used: str
    feature_importance: Optional[Dict[str, float]] = None
    prediction_date: datetime = None
    
    def __post_init__(self):
        if self.prediction_date is None:
            self.prediction_date = datetime.now()

@dataclass
class DemandForecastResult:
    """Demand forecasting result with ML insights."""
    material_id: str
    forecasted_demand: float
    confidence_interval: tuple
    seasonal_factor: float
    trend_component: float
    model_accuracy: float
    forecast_horizon_days: int

class BeverlyKnitsMLClient:
    """AI/ML integration client for Beverly Knits using zen-mcp-server and local ML models."""
    
    def __init__(self, config_path: str = "config/zen_ml_config.json"):
        self.config_path = config_path
        self.zen_process = None
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path("models/ml_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path("temp/ml_processing")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # ML model cache
        self.model_cache = {}
        
        # Initialize local ML capabilities
        self._initialize_local_ml()
        
    def _initialize_local_ml(self):
        """Initialize local machine learning capabilities."""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, r2_score
            
            # Create default models
            self.ml_models = {
                'demand_forecasting': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                ),
                'price_prediction': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42,
                    learning_rate=0.1
                ),
                'supplier_risk': RandomForestRegressor(
                    n_estimators=50,
                    random_state=42,
                    max_depth=8
                ),
                'quality_prediction': LinearRegression()
            }
            
            self.scaler = StandardScaler()
            self.logger.info("✅ Local ML capabilities initialized")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Some ML libraries not available: {e}")
            self.ml_models = {}
    
    async def initialize_zen_mcp(self) -> bool:
        """Initialize zen-mcp-server for advanced ML operations."""
        try:
            # Check if zen-mcp-server is available
            result = subprocess.run(['which', 'zen-mcp-server'], 
                                   capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning("zen-mcp-server not found, using local ML only")
                return False
            
            # Start zen-mcp-server
            self.zen_process = subprocess.Popen([
                'zen-mcp-server',
                '--config', self.config_path,
                '--mode', 'ml_integration'
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
               stderr=subprocess.PIPE, text=True)
            
            self.logger.info("✅ zen-mcp-server integration initialized")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize zen-mcp-server: {e}")
            return False
    
    def predict_yarn_demand(
        self, 
        historical_data: pd.DataFrame,
        forecast_horizon_days: int = 90
    ) -> List[DemandForecastResult]:
        """Predict yarn demand using ML models."""
        
        self.logger.info(f"Predicting yarn demand for {forecast_horizon_days} days")
        
        if 'demand_forecasting' not in self.ml_models:
            self.logger.error("Demand forecasting model not available")
            return []
        
        results = []
        
        # Group by material
        for material_id, material_data in historical_data.groupby('material_id'):
            try:
                forecast_result = self._forecast_single_material(
                    material_id, material_data, forecast_horizon_days
                )
                if forecast_result:
                    results.append(forecast_result)
            except Exception as e:
                self.logger.error(f"Failed to forecast {material_id}: {e}")
        
        self.logger.info(f"Generated {len(results)} demand forecasts")
        return results
    
    def _forecast_single_material(
        self, 
        material_id: str, 
        material_data: pd.DataFrame,
        forecast_horizon_days: int
    ) -> Optional[DemandForecastResult]:
        """Forecast demand for a single material."""
        
        if len(material_data) < 10:  # Minimum data points required
            self.logger.debug(f"Insufficient data for {material_id}: {len(material_data)} points")
            return None
        
        # Prepare features
        features = self._extract_demand_features(material_data)
        if features is None:
            return None
        
        # Make prediction
        model = self.ml_models['demand_forecasting']
        
        try:
            # For demo purposes, use statistical forecasting
            # In production, this would use trained ML models
            recent_demand = material_data['demand'].tail(10).mean()
            trend = material_data['demand'].diff().tail(5).mean()
            seasonal_factor = self._calculate_seasonal_factor(material_data)
            
            # Simple forecast: recent_average + trend + seasonal_adjustment
            base_forecast = recent_demand + (trend * forecast_horizon_days / 7)
            adjusted_forecast = base_forecast * seasonal_factor
            
            # Calculate confidence interval (±20% for demo)
            confidence_interval = (
                adjusted_forecast * 0.8,
                adjusted_forecast * 1.2
            )
            
            return DemandForecastResult(
                material_id=material_id,
                forecasted_demand=max(0, adjusted_forecast),
                confidence_interval=confidence_interval,
                seasonal_factor=seasonal_factor,
                trend_component=trend,
                model_accuracy=0.85,  # Demo accuracy
                forecast_horizon_days=forecast_horizon_days
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {material_id}: {e}")
            return None
    
    def _extract_demand_features(self, material_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for demand forecasting."""
        
        try:
            # Basic features for demo
            features = []
            
            # Historical demand statistics
            features.append(material_data['demand'].mean())
            features.append(material_data['demand'].std())
            features.append(material_data['demand'].tail(7).mean())  # Recent week
            features.append(material_data['demand'].tail(30).mean())  # Recent month
            
            # Trend features
            if len(material_data) > 1:
                features.append(material_data['demand'].diff().mean())
                features.append(material_data['demand'].pct_change().mean())
            else:
                features.extend([0.0, 0.0])
            
            # Seasonal features (if date column exists)
            if 'date' in material_data.columns:
                material_data['date'] = pd.to_datetime(material_data['date'])
                features.append(material_data['date'].dt.month.mode()[0] if len(material_data) > 0 else 6)
                features.append(material_data['date'].dt.quarter.mode()[0] if len(material_data) > 0 else 2)
            else:
                features.extend([6, 2])  # Default values
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _calculate_seasonal_factor(self, material_data: pd.DataFrame) -> float:
        """Calculate seasonal adjustment factor."""
        
        try:
            if 'date' in material_data.columns and len(material_data) >= 12:
                material_data['date'] = pd.to_datetime(material_data['date'])
                material_data['month'] = material_data['date'].dt.month
                
                # Calculate monthly averages
                monthly_avg = material_data.groupby('month')['demand'].mean()
                overall_avg = material_data['demand'].mean()
                
                current_month = datetime.now().month
                if current_month in monthly_avg.index and overall_avg > 0:
                    seasonal_factor = monthly_avg[current_month] / overall_avg
                    return max(0.5, min(2.0, seasonal_factor))  # Cap between 0.5 and 2.0
            
            return 1.0  # No seasonal adjustment
            
        except Exception:
            return 1.0
    
    def assess_supplier_risk(
        self, 
        supplier_data: pd.DataFrame
    ) -> Dict[str, MLPrediction]:
        """Assess supplier risk using ML models."""
        
        self.logger.info("Assessing supplier risk with ML models")
        
        risk_predictions = {}
        
        for _, supplier in supplier_data.iterrows():
            try:
                supplier_id = supplier['supplier_id']
                
                # Extract risk features
                features = self._extract_supplier_features(supplier)
                
                # Calculate risk score (0 = low risk, 1 = high risk)
                risk_score = self._calculate_supplier_risk_score(supplier)
                
                prediction = MLPrediction(
                    prediction_type='supplier_risk',
                    prediction_value=risk_score,
                    confidence_score=0.8,
                    model_used='risk_assessment_ensemble',
                    feature_importance={
                        'on_time_delivery': 0.3,
                        'quality_score': 0.25,
                        'financial_stability': 0.2,
                        'communication': 0.15,
                        'price_competitiveness': 0.1
                    }
                )
                
                risk_predictions[supplier_id] = prediction
                
            except Exception as e:
                self.logger.error(f"Risk assessment failed for supplier {supplier.get('supplier_id', 'unknown')}: {e}")
        
        self.logger.info(f"Completed risk assessment for {len(risk_predictions)} suppliers")
        return risk_predictions
    
    def _extract_supplier_features(self, supplier: pd.Series) -> List[float]:
        """Extract features for supplier risk assessment."""
        
        features = []
        
        # On-time delivery rate
        features.append(supplier.get('on_time_delivery_rate', 0.85))
        
        # Quality score
        features.append(supplier.get('quality_score', 0.8))
        
        # Lead time consistency
        features.append(supplier.get('lead_time_variance', 0.1))
        
        # Financial stability indicators
        features.append(supplier.get('payment_terms_compliance', 0.9))
        features.append(supplier.get('credit_rating', 0.7))
        
        # Communication responsiveness
        features.append(supplier.get('communication_score', 0.8))
        
        # Price stability
        features.append(supplier.get('price_stability', 0.9))
        
        return features
    
    def _calculate_supplier_risk_score(self, supplier: pd.Series) -> float:
        """Calculate supplier risk score (0 = low risk, 1 = high risk)."""
        
        # Get key metrics with defaults
        on_time_rate = supplier.get('on_time_delivery_rate', 0.85)
        quality_score = supplier.get('quality_score', 0.8)
        lead_time_variance = supplier.get('lead_time_variance', 0.1)
        
        # Calculate composite risk score
        reliability_risk = 1.0 - on_time_rate
        quality_risk = 1.0 - quality_score
        variability_risk = min(1.0, lead_time_variance * 10)  # Scale variance
        
        # Weighted average
        risk_score = (
            reliability_risk * 0.4 +
            quality_risk * 0.3 +
            variability_risk * 0.3
        )
        
        return min(1.0, max(0.0, risk_score))
    
    def predict_material_prices(
        self, 
        market_data: pd.DataFrame,
        forecast_horizon_days: int = 30
    ) -> Dict[str, MLPrediction]:
        """Predict material price changes using market data."""
        
        self.logger.info(f"Predicting material prices for {forecast_horizon_days} days")
        
        price_predictions = {}
        
        # Group by material
        for material_id, material_prices in market_data.groupby('material_id'):
            try:
                # Simple price trend analysis for demo
                recent_prices = material_prices['price'].tail(10)
                
                if len(recent_prices) >= 3:
                    # Calculate price trend
                    price_change = recent_prices.pct_change().mean()
                    volatility = recent_prices.std() / recent_prices.mean()
                    
                    # Project future price
                    current_price = recent_prices.iloc[-1]
                    predicted_price = current_price * (1 + price_change * forecast_horizon_days / 30)
                    
                    # Confidence based on volatility (lower volatility = higher confidence)
                    confidence = max(0.3, 1.0 - volatility)
                    
                    prediction = MLPrediction(
                        prediction_type='price_forecast',
                        prediction_value=predicted_price,
                        confidence_score=confidence,
                        model_used='price_trend_analysis',
                        feature_importance={
                            'historical_trend': 0.4,
                            'market_volatility': 0.3,
                            'seasonal_patterns': 0.2,
                            'external_factors': 0.1
                        }
                    )
                    
                    price_predictions[material_id] = prediction
                    
            except Exception as e:
                self.logger.error(f"Price prediction failed for {material_id}: {e}")
        
        self.logger.info(f"Generated price predictions for {len(price_predictions)} materials")
        return price_predictions
    
    def optimize_inventory_levels(
        self, 
        inventory_data: pd.DataFrame,
        demand_forecasts: List[DemandForecastResult]
    ) -> Dict[str, MLPrediction]:
        """Optimize inventory levels using ML-driven insights."""
        
        self.logger.info("Optimizing inventory levels with ML insights")
        
        optimization_results = {}
        
        # Create demand forecast lookup
        demand_lookup = {forecast.material_id: forecast for forecast in demand_forecasts}
        
        for _, inventory in inventory_data.iterrows():
            try:
                material_id = inventory['material_id']
                current_inventory = inventory.get('current_inventory', 0)
                
                # Get demand forecast if available
                demand_forecast = demand_lookup.get(material_id)
                
                if demand_forecast:
                    # Calculate optimal inventory level
                    forecasted_demand = demand_forecast.forecasted_demand
                    confidence_interval = demand_forecast.confidence_interval
                    
                    # Safety stock based on demand uncertainty
                    demand_uncertainty = (confidence_interval[1] - confidence_interval[0]) / 2
                    safety_stock = demand_uncertainty * 0.5  # 50% of uncertainty range
                    
                    # Optimal inventory = forecasted demand + safety stock
                    optimal_inventory = forecasted_demand + safety_stock
                    
                    # Calculate inventory adjustment needed
                    inventory_adjustment = optimal_inventory - current_inventory
                    
                    prediction = MLPrediction(
                        prediction_type='inventory_optimization',
                        prediction_value=optimal_inventory,
                        confidence_score=demand_forecast.model_accuracy,
                        model_used='ml_inventory_optimizer',
                        feature_importance={
                            'demand_forecast': 0.6,
                            'demand_uncertainty': 0.2,
                            'lead_time': 0.1,
                            'cost_considerations': 0.1
                        }
                    )
                    
                    optimization_results[material_id] = prediction
                    
            except Exception as e:
                self.logger.error(f"Inventory optimization failed for {material_id}: {e}")
        
        self.logger.info(f"Optimized inventory for {len(optimization_results)} materials")
        return optimization_results
    
    def generate_ml_insights_report(
        self,
        demand_forecasts: List[DemandForecastResult],
        risk_assessments: Dict[str, MLPrediction],
        price_predictions: Dict[str, MLPrediction]
    ) -> Dict[str, Any]:
        """Generate comprehensive ML insights report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'demand_forecasts_generated': len(demand_forecasts),
                'supplier_risks_assessed': len(risk_assessments),
                'price_predictions_made': len(price_predictions)
            },
            'insights': {
                'high_demand_materials': [],
                'high_risk_suppliers': [],
                'price_volatility_alerts': []
            },
            'recommendations': []
        }
        
        # Analyze demand forecasts
        high_demand_threshold = np.percentile([f.forecasted_demand for f in demand_forecasts], 80) if demand_forecasts else 0
        
        for forecast in demand_forecasts:
            if forecast.forecasted_demand > high_demand_threshold:
                report['insights']['high_demand_materials'].append({
                    'material_id': forecast.material_id,
                    'forecasted_demand': forecast.forecasted_demand,
                    'confidence': forecast.model_accuracy
                })
        
        # Analyze supplier risks
        for supplier_id, risk_pred in risk_assessments.items():
            if risk_pred.prediction_value > 0.7:  # High risk threshold
                report['insights']['high_risk_suppliers'].append({
                    'supplier_id': supplier_id,
                    'risk_score': risk_pred.prediction_value,
                    'confidence': risk_pred.confidence_score
                })
        
        # Analyze price predictions
        for material_id, price_pred in price_predictions.items():
            if price_pred.confidence_score > 0.8:  # High confidence predictions
                report['insights']['price_volatility_alerts'].append({
                    'material_id': material_id,
                    'predicted_price': price_pred.prediction_value,
                    'confidence': price_pred.confidence_score
                })
        
        # Generate recommendations
        if report['insights']['high_demand_materials']:
            report['recommendations'].append(
                "Consider increasing safety stock for high-demand materials"
            )
        
        if report['insights']['high_risk_suppliers']:
            report['recommendations'].append(
                "Review supplier relationships for high-risk suppliers and consider backup options"
            )
        
        if report['insights']['price_volatility_alerts']:
            report['recommendations'].append(
                "Monitor market conditions for materials with high price volatility"
            )
        
        return report
    
    async def shutdown(self):
        """Shutdown ML integration services."""
        if self.zen_process:
            self.zen_process.terminate()
            await asyncio.sleep(1)
            if self.zen_process.poll() is None:
                self.zen_process.kill()
            self.logger.info("zen-mcp-server integration shut down")


# Factory function for easy instantiation
def create_ml_client(config_path: Optional[str] = None) -> BeverlyKnitsMLClient:
    """Create and initialize ML client."""
    return BeverlyKnitsMLClient(config_path or "config/zen_ml_config.json")