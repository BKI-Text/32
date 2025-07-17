from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import logging
from collections import defaultdict
import math
import pandas as pd
import numpy as np

from ..core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast, 
    ProcurementRecommendation, MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)
from ..config.settings import PLANNING_CONFIG
from .eoq_optimizer import EOQOptimizer
from .multi_supplier_optimizer import MultiSupplierOptimizer
from .sales_forecasting_engine import SalesForecastingEngine
from .ml_model_manager import MLModelManager
from .ml_risk_assessor import MLRiskAssessor

logger = logging.getLogger(__name__)

class PlanningEngine:
    def __init__(self):
        self.config = PLANNING_CONFIG
        self.eoq_optimizer = EOQOptimizer()
        self.multi_supplier_optimizer = MultiSupplierOptimizer(
            cost_weight=self.config.COST_WEIGHT,
            reliability_weight=self.config.RELIABILITY_WEIGHT,
            lead_time_weight=1.0 - self.config.COST_WEIGHT - self.config.RELIABILITY_WEIGHT
        )
        self.sales_forecasting_engine = SalesForecastingEngine()
        
        # Initialize ML components
        self.ml_model_manager = MLModelManager()
        self.ml_risk_assessor = MLRiskAssessor()
        
    def execute_planning_cycle(
        self, 
        forecasts: List[Forecast], 
        boms: List[BOM], 
        inventory: List[Inventory],
        suppliers: List[SupplierMaterial]
    ) -> List[ProcurementRecommendation]:
        
        logger.info("Starting 6-phase planning cycle")
        
        # Phase 1: Forecast Unification
        unified_forecasts = self._unify_forecasts(forecasts)
        logger.info(f"Phase 1 complete: Unified {len(unified_forecasts)} forecasts")
        
        # Phase 2: BOM Explosion
        material_requirements = self._explode_boms(unified_forecasts, boms)
        logger.info(f"Phase 2 complete: Generated {len(material_requirements)} material requirements")
        
        # Phase 3: Inventory Netting
        net_requirements = self._net_inventory(material_requirements, inventory)
        logger.info(f"Phase 3 complete: Calculated {len(net_requirements)} net requirements")
        
        # Phase 4: Procurement Optimization
        optimized_requirements = self._optimize_procurement(net_requirements, suppliers)
        logger.info(f"Phase 4 complete: Optimized {len(optimized_requirements)} requirements")
        
        # Phase 5: Supplier Selection
        supplier_selections = self._select_suppliers(optimized_requirements, suppliers)
        logger.info(f"Phase 5 complete: Selected suppliers for {len(supplier_selections)} materials")
        
        # Phase 6: Output Generation
        recommendations = self._generate_recommendations(supplier_selections, suppliers)
        logger.info(f"Phase 6 complete: Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def _unify_forecasts(self, forecasts: List[Forecast]) -> Dict[str, Quantity]:
        """Phase 1: Unify forecasts with source reliability weighting"""
        unified = defaultdict(lambda: Quantity(amount=Decimal("0"), unit="unit"))
        
        for forecast in forecasts:
            weight = self.config.SOURCE_WEIGHTS.get(forecast.source.value, 0.5)
            weighted_qty = forecast.forecast_qty.amount * Decimal(str(weight)) * Decimal(str(forecast.confidence_score))
            
            if forecast.sku_id.value not in unified:
                unified[forecast.sku_id.value] = Quantity(
                    amount=weighted_qty, 
                    unit=forecast.forecast_qty.unit
                )
            else:
                unified[forecast.sku_id.value] = unified[forecast.sku_id.value] + Quantity(
                    amount=weighted_qty, 
                    unit=forecast.forecast_qty.unit
                )
        
        return dict(unified)
    
    def _explode_boms(self, forecasts: Dict[str, Quantity], boms: List[BOM]) -> Dict[str, Quantity]:
        """Phase 2: Explode BOMs to translate SKU forecasts to material requirements"""
        material_requirements = defaultdict(lambda: Quantity(amount=Decimal("0"), unit="unit"))
        
        bom_lookup = defaultdict(list)
        for bom in boms:
            bom_lookup[bom.sku_id.value].append(bom)
        
        for sku_id, forecast_qty in forecasts.items():
            if sku_id in bom_lookup:
                for bom in bom_lookup[sku_id]:
                    required_qty = bom.calculate_requirement(forecast_qty)
                    
                    if bom.material_id.value not in material_requirements:
                        material_requirements[bom.material_id.value] = required_qty
                    else:
                        material_requirements[bom.material_id.value] = (
                            material_requirements[bom.material_id.value] + required_qty
                        )
        
        return dict(material_requirements)
    
    def _net_inventory(self, requirements: Dict[str, Quantity], inventory: List[Inventory]) -> Dict[str, Quantity]:
        """Phase 3: Net inventory against requirements"""
        net_requirements = {}
        
        inventory_lookup = {inv.material_id.value: inv for inv in inventory}
        
        for material_id, required_qty in requirements.items():
            if material_id in inventory_lookup:
                inv = inventory_lookup[material_id]
                available_qty = inv.get_available_qty()
                
                net_amount = required_qty.amount - available_qty.amount
                if net_amount > 0:
                    net_requirements[material_id] = Quantity(
                        amount=net_amount, 
                        unit=required_qty.unit
                    )
            else:
                net_requirements[material_id] = required_qty
        
        return net_requirements
    
    def _optimize_procurement(self, requirements: Dict[str, Quantity], suppliers: List[SupplierMaterial]) -> Dict[str, Quantity]:
        """Phase 4: Apply safety stock, MOQ, and EOQ optimization"""
        optimized_requirements = {}
        
        for material_id, required_qty in requirements.items():
            # Apply safety stock
            safety_buffer = Decimal(str(self.config.SAFETY_STOCK_PERCENTAGE))
            buffered_qty = required_qty.amount * (1 + safety_buffer)
            
            # Find supplier options for this material
            material_suppliers = [s for s in suppliers if s.material_id.value == material_id]
            
            if material_suppliers:
                # Use the first supplier for MOQ/EOQ calculations
                supplier = material_suppliers[0]
                
                # Apply MOQ
                if buffered_qty < supplier.moq.amount:
                    buffered_qty = supplier.moq.amount
                
                # Apply EOQ if enabled
                if self.config.ENABLE_EOQ_OPTIMIZATION:
                    eoq = self._calculate_eoq(buffered_qty, supplier)
                    if eoq > buffered_qty:
                        buffered_qty = eoq
            
            optimized_requirements[material_id] = Quantity(
                amount=buffered_qty, 
                unit=required_qty.unit
            )
        
        return optimized_requirements
    
    def _calculate_eoq(self, annual_demand: Decimal, supplier: SupplierMaterial) -> Decimal:
        """Calculate Economic Order Quantity"""
        try:
            # EOQ = sqrt(2 * D * S / H)
            # D = annual demand, S = ordering cost, H = holding cost per unit
            
            ordering_cost = supplier.ordering_cost.amount
            unit_cost = supplier.cost_per_unit.amount
            holding_cost_per_unit = unit_cost * Decimal(str(supplier.holding_cost_rate))
            
            if holding_cost_per_unit <= 0:
                return annual_demand
            
            eoq_squared = (2 * annual_demand * ordering_cost) / holding_cost_per_unit
            eoq = Decimal(str(math.sqrt(float(eoq_squared))))
            
            return eoq
        except Exception as e:
            logger.warning(f"EOQ calculation failed: {e}")
            return annual_demand
    
    def _select_suppliers(self, requirements: Dict[str, Quantity], suppliers: List[SupplierMaterial]) -> Dict[str, Any]:
        """Phase 5: Select optimal suppliers using multi-supplier optimization"""
        selections = {}
        
        supplier_lookup = defaultdict(list)
        for supplier in suppliers:
            supplier_lookup[supplier.material_id.value].append(supplier)
        
        for material_id, required_qty in requirements.items():
            if material_id in supplier_lookup:
                material_suppliers = supplier_lookup[material_id]
                
                # Use multi-supplier optimizer for advanced sourcing decisions
                sourcing_recommendation = self.multi_supplier_optimizer.optimize_sourcing(
                    material_id=MaterialId(value=material_id),
                    demand=required_qty,
                    suppliers=material_suppliers
                )
                
                selections[material_id] = {
                    'recommendation': sourcing_recommendation,
                    'suppliers': material_suppliers,
                    'required_qty': required_qty
                }
        
        return selections
    
    def _generate_recommendations(self, selections: Dict[str, Any], suppliers: List[SupplierMaterial]) -> List[ProcurementRecommendation]:
        """Phase 6: Generate final procurement recommendations with EOQ optimization"""
        recommendations = []
        
        for material_id, selection_data in selections.items():
            sourcing_recommendation = selection_data['recommendation']
            material_suppliers = selection_data['suppliers']
            required_qty = selection_data['required_qty']
            
            # Generate recommendations for each supplier allocation
            for supplier_id, allocated_qty in sourcing_recommendation.allocations.items():
                # Find the supplier object
                supplier = next((s for s in material_suppliers if s.supplier_id == supplier_id), None)
                
                if supplier:
                    # Apply EOQ optimization
                    eoq_result = self.eoq_optimizer.calculate_eoq(
                        material_id=MaterialId(value=material_id),
                        quarterly_demand=allocated_qty,
                        supplier=supplier
                    )
                    
                    # Create recommendation
                    recommendation = ProcurementRecommendation(
                        material_id=MaterialId(value=material_id),
                        supplier_id=supplier_id,
                        recommended_order_qty=eoq_result.eoq_quantity,
                        unit_cost=supplier.cost_per_unit,
                        total_cost=Money(
                            amount=supplier.cost_per_unit.amount * eoq_result.eoq_quantity.amount,
                            currency=supplier.cost_per_unit.currency
                        ),
                        expected_lead_time=supplier.lead_time,
                        risk_flag=sourcing_recommendation.risk_assessment,
                        reasoning=f"{sourcing_recommendation.reasoning} EOQ optimized: {eoq_result.eoq_quantity.amount} {eoq_result.eoq_quantity.unit} (orders {eoq_result.order_frequency:.1f} times/year)",
                        urgency_score=self._calculate_urgency_score(supplier.lead_time)
                    )
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _assess_risk(self, reliability_score: float) -> RiskLevel:
        """Assess risk level based on reliability score"""
        if reliability_score >= self.config.RISK_THRESHOLDS["medium"]:
            return RiskLevel.LOW
        elif reliability_score >= self.config.RISK_THRESHOLDS["high"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _calculate_urgency_score(self, lead_time: LeadTime) -> float:
        """Calculate urgency score based on lead time"""
        # Higher urgency for longer lead times
        max_lead_time = 30  # days
        urgency = min(lead_time.days / max_lead_time, 1.0)
        return urgency
    
    def execute_sales_based_planning_cycle(
        self,
        sales_data: Any,  # pandas DataFrame
        style_bom_data: Any,  # pandas DataFrame
        inventory: List[Inventory],
        suppliers: List[SupplierMaterial],
        planning_date: Optional[date] = None
    ) -> List[ProcurementRecommendation]:
        """Execute planning cycle using sales-based forecasting."""
        
        logger.info("Starting sales-based planning cycle")
        
        try:
            # Generate sales-based forecasts
            sales_forecasts = self.sales_forecasting_engine.generate_sales_based_forecasts(
                sales_data=sales_data,
                style_bom_data=style_bom_data,
                planning_date=planning_date
            )
            
            # Convert to domain forecasts
            domain_forecasts = self.sales_forecasting_engine.convert_sales_forecasts_to_domain_forecasts(
                sales_forecasts
            )
            
            logger.info(f"Generated {len(domain_forecasts)} sales-based forecasts")
            
            # Execute standard planning cycle with sales-based forecasts
            # Create empty BOM list since sales forecasts already include material-level demand
            empty_boms = []
            
            recommendations = self.execute_planning_cycle(
                forecasts=domain_forecasts,
                boms=empty_boms,
                inventory=inventory,
                suppliers=suppliers
            )
            
            logger.info(f"Sales-based planning cycle completed with {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Sales-based planning cycle failed: {e}")
            raise
    
    def enhance_forecasts_with_sales_data(
        self,
        existing_forecasts: List[Forecast],
        sales_data: Any,  # pandas DataFrame
        style_bom_data: Any  # pandas DataFrame
    ) -> List[Forecast]:
        """Enhance existing forecasts with sales-based insights."""
        
        try:
            # Generate sales-based forecasts
            sales_forecasts = self.sales_forecasting_engine.generate_sales_based_forecasts(
                sales_data=sales_data,
                style_bom_data=style_bom_data
            )
            
            # Convert to domain forecasts
            sales_domain_forecasts = self.sales_forecasting_engine.convert_sales_forecasts_to_domain_forecasts(
                sales_forecasts
            )
            
            # Combine with existing forecasts
            enhanced_forecasts = existing_forecasts + sales_domain_forecasts
            
            logger.info(f"Enhanced {len(existing_forecasts)} forecasts with {len(sales_domain_forecasts)} sales-based forecasts")
            return enhanced_forecasts
            
        except Exception as e:
            logger.error(f"Failed to enhance forecasts with sales data: {e}")
            return existing_forecasts
    
    def generate_ml_forecasts(self, 
                            historical_data: pd.DataFrame, 
                            periods: int = 30,
                            models: List[str] = None) -> List[Forecast]:
        """Generate ML-based demand forecasts using multiple models."""
        try:
            if models is None:
                models = ['arima', 'prophet', 'lstm', 'xgboost']
            
            logger.info(f"Generating ML forecasts using models: {models}")
            
            # Generate forecasts using ML Model Manager
            # For now, create a simple forecast using historical data
            # This will be expanded as the ML models become available
            forecasts = []
            
            # Generate basic forecasts for demonstration
            from ..core.domain.entities import Forecast, ForecastSource
            from ..core.domain.value_objects import SkuId, Quantity
            
            # Use the last known value as a simple forecast
            if not historical_data.empty:
                last_value = historical_data['demand'].iloc[-1]
                last_date = historical_data.index[-1]
                
                for i in range(periods):
                    forecast_date = last_date + timedelta(days=i+1)
                    forecast_qty = max(0, last_value + np.random.normal(0, last_value * 0.1))
                    
                    forecast = Forecast(
                        sku_id=SkuId(value="aggregate"),
                        forecast_qty=Quantity(amount=forecast_qty, unit="unit"),
                        forecast_date=forecast_date.date(),
                        source=ForecastSource.PROJECTION,
                        confidence_score=0.7,
                        created_at=datetime.now()
                    )
                    forecasts.append(forecast)
            
            logger.info(f"Generated {len(forecasts)} ML forecasts")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating ML forecasts: {e}")
            return []
    
    def assess_supplier_risk_with_ml(self, 
                                   supplier_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess supplier risk using ML-based risk assessor."""
        try:
            logger.info("Assessing supplier risk using ML models")
            
            # Train or load ML risk models
            if not self.ml_risk_assessor.is_risk_model_trained:
                self.ml_risk_assessor.train_risk_model(supplier_data)
                
            if not self.ml_risk_assessor.is_anomaly_detector_trained:
                self.ml_risk_assessor.train_anomaly_detector(supplier_data)
            
            # Predict supplier risk
            risk_scores = self.ml_risk_assessor.predict_supplier_risk(supplier_data)
            
            # Detect anomalies
            anomalies = self.ml_risk_assessor.detect_anomalies(supplier_data)
            
            return {
                'risk_scores': risk_scores,
                'anomalies': anomalies,
                'model_status': self.ml_risk_assessor.get_model_status()
            }
            
        except Exception as e:
            logger.error(f"Error assessing supplier risk with ML: {e}")
            return {'risk_scores': [], 'anomalies': [], 'model_status': {}}
    
    def execute_ml_enhanced_planning_cycle(self,
                                         historical_demand_data: pd.DataFrame,
                                         supplier_data: pd.DataFrame,
                                         boms: List[BOM],
                                         inventory: List[Inventory],
                                         suppliers: List[SupplierMaterial],
                                         periods: int = 30,
                                         models: List[str] = None) -> List[ProcurementRecommendation]:
        """Execute ML-enhanced planning cycle with advanced forecasting and risk assessment."""
        try:
            logger.info("Starting ML-enhanced planning cycle")
            
            # Phase 1: Generate ML forecasts
            ml_forecasts = self.generate_ml_forecasts(
                historical_data=historical_demand_data,
                periods=periods,
                models=models
            )
            
            # Phase 2: Assess supplier risk with ML
            ml_risk_assessment = self.assess_supplier_risk_with_ml(supplier_data)
            
            # Phase 3: Execute standard planning cycle with ML forecasts
            recommendations = self.execute_planning_cycle(
                forecasts=ml_forecasts,
                boms=boms,
                inventory=inventory,
                suppliers=suppliers
            )
            
            # Phase 4: Enhance recommendations with ML risk insights
            enhanced_recommendations = self._enhance_recommendations_with_ml_risk(
                recommendations, ml_risk_assessment
            )
            
            logger.info(f"ML-enhanced planning cycle completed with {len(enhanced_recommendations)} recommendations")
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"ML-enhanced planning cycle failed: {e}")
            return []
    
    def _enhance_recommendations_with_ml_risk(self,
                                            recommendations: List[ProcurementRecommendation],
                                            ml_risk_assessment: Dict[str, Any]) -> List[ProcurementRecommendation]:
        """Enhance recommendations with ML risk assessment results."""
        try:
            risk_scores = ml_risk_assessment.get('risk_scores', [])
            anomalies = ml_risk_assessment.get('anomalies', [])
            
            # Create lookup dictionaries
            risk_lookup = {score.supplier_id: score for score in risk_scores if hasattr(score, 'supplier_id')}
            anomaly_lookup = {anom.supplier_id: anom for anom in anomalies if hasattr(anom, 'supplier_id')}
            
            enhanced_recommendations = []
            
            for rec in recommendations:
                # Get ML risk score if available
                ml_risk_score = risk_lookup.get(rec.supplier_id)
                anomaly_info = anomaly_lookup.get(rec.supplier_id)
                
                # Create enhanced recommendation
                enhanced_reasoning = rec.reasoning
                
                if ml_risk_score:
                    enhanced_reasoning += f" ML Risk Score: {ml_risk_score.overall_score:.2f}"
                    if ml_risk_score.factors:
                        enhanced_reasoning += f" Risk Factors: {', '.join(ml_risk_score.factors)}"
                
                if anomaly_info and anomaly_info.is_anomaly:
                    enhanced_reasoning += f" Anomaly Detected: {anomaly_info.description}"
                
                # Update risk flag based on ML assessment
                updated_risk_flag = rec.risk_flag
                if ml_risk_score:
                    updated_risk_flag = ml_risk_score.risk_level
                
                enhanced_rec = ProcurementRecommendation(
                    material_id=rec.material_id,
                    supplier_id=rec.supplier_id,
                    recommended_order_qty=rec.recommended_order_qty,
                    unit_cost=rec.unit_cost,
                    total_cost=rec.total_cost,
                    expected_lead_time=rec.expected_lead_time,
                    risk_flag=updated_risk_flag,
                    reasoning=enhanced_reasoning,
                    urgency_score=rec.urgency_score
                )
                
                enhanced_recommendations.append(enhanced_rec)
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"Error enhancing recommendations with ML risk: {e}")
            return recommendations
    
    def get_ml_model_status(self) -> Dict[str, Any]:
        """Get status of all ML models."""
        try:
            model_manager_status = self.ml_model_manager.get_model_status()
            risk_assessor_status = self.ml_risk_assessor.get_model_status()
            
            return {
                'model_manager': model_manager_status,
                'risk_assessor': risk_assessor_status,
                'ml_enabled': True
            }
            
        except Exception as e:
            logger.error(f"Error getting ML model status: {e}")
            return {'ml_enabled': False, 'error': str(e)}