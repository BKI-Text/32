"""
Supply Chain Planning Service
Application service orchestrating the core supply chain planning workflow.
"""

from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import logging

from ..domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast, 
    ProcurementRecommendation, MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)
from ...engine.planning_engine import PlanningEngine
from ...data.beverly_knits_live_data_integrator import BeverlyKnitsLiveDataIntegrator
from ...utils.error_handling import handle_errors, ErrorCategory

logger = logging.getLogger(__name__)

class SupplyChainPlanningService:
    """
    Application service for supply chain planning operations.
    Orchestrates the complete planning workflow from data integration to recommendations.
    """
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = data_path
        self.data_integrator = BeverlyKnitsLiveDataIntegrator(data_path)
        self.planning_engine = PlanningEngine()
        
    @handle_errors(category=ErrorCategory.PLANNING_ENGINE)
    def execute_complete_planning_cycle(self) -> Dict[str, Any]:
        """
        Execute the complete supply chain planning cycle.
        
        Returns:
            Dict containing recommendations, metrics, and execution details
        """
        logger.info("Starting complete supply chain planning cycle")
        
        # Step 1: Integrate live data
        logger.info("Step 1: Integrating live data")
        domain_objects = self.data_integrator.integrate_live_data()
        
        # Step 2: Execute planning engine
        logger.info("Step 2: Executing planning engine")
        recommendations = self.planning_engine.execute_planning_cycle(
            forecasts=domain_objects['forecasts'],
            boms=domain_objects['boms'],
            inventory=domain_objects['inventory'],
            suppliers=domain_objects['supplier_materials']
        )
        
        # Step 3: Calculate planning metrics
        logger.info("Step 3: Calculating planning metrics")
        metrics = self._calculate_planning_metrics(recommendations, domain_objects)
        
        # Step 4: Generate executive summary
        logger.info("Step 4: Generating executive summary")
        summary = self._generate_executive_summary(recommendations, metrics)
        
        result = {
            'recommendations': recommendations,
            'metrics': metrics,
            'summary': summary,
            'execution_timestamp': datetime.now().isoformat(),
            'data_quality_score': domain_objects.get('quality_score', 0.0),
            'planning_horizon_days': self.planning_engine.config.PLANNING_HORIZON_DAYS
        }
        
        logger.info(f"Planning cycle completed successfully. Generated {len(recommendations)} recommendations")
        return result
    
    @handle_errors(category=ErrorCategory.PLANNING_ENGINE)
    def get_planning_recommendations(
        self, 
        material_ids: Optional[List[str]] = None,
        supplier_ids: Optional[List[str]] = None,
        risk_level_filter: Optional[RiskLevel] = None
    ) -> List[ProcurementRecommendation]:
        """
        Get filtered planning recommendations based on criteria.
        
        Args:
            material_ids: Filter by specific material IDs
            supplier_ids: Filter by specific supplier IDs
            risk_level_filter: Filter by risk level
            
        Returns:
            Filtered list of procurement recommendations
        """
        logger.info("Retrieving filtered planning recommendations")
        
        # Execute planning cycle to get latest recommendations
        result = self.execute_complete_planning_cycle()
        recommendations = result['recommendations']
        
        # Apply filters
        filtered_recommendations = recommendations
        
        if material_ids:
            filtered_recommendations = [
                rec for rec in filtered_recommendations 
                if rec.material_id.value in material_ids
            ]
        
        if supplier_ids:
            filtered_recommendations = [
                rec for rec in filtered_recommendations 
                if rec.supplier_id.value in supplier_ids
            ]
        
        if risk_level_filter:
            filtered_recommendations = [
                rec for rec in filtered_recommendations 
                if rec.risk_flag == risk_level_filter
            ]
        
        logger.info(f"Filtered recommendations: {len(filtered_recommendations)} of {len(recommendations)}")
        return filtered_recommendations
    
    @handle_errors(category=ErrorCategory.DATA_INTEGRATION)
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and return quality report.
        
        Returns:
            Dictionary containing data quality metrics and issues
        """
        logger.info("Validating data quality")
        
        # Use data integrator to get quality information
        domain_objects = self.data_integrator.integrate_live_data()
        
        quality_report = {
            'overall_score': domain_objects.get('quality_score', 0.0),
            'issues_found': self.data_integrator.quality_issues,
            'fixes_applied': self.data_integrator.fixes_applied,
            'data_completeness': self._calculate_data_completeness(domain_objects),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Data quality validation completed. Score: {quality_report['overall_score']:.2f}")
        return quality_report
    
    def _calculate_planning_metrics(
        self, 
        recommendations: List[ProcurementRecommendation], 
        domain_objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate key planning metrics."""
        
        if not recommendations:
            return {
                'total_cost': 0.0,
                'total_materials': 0,
                'total_suppliers': 0,
                'average_lead_time': 0.0,
                'risk_distribution': {'low': 0, 'medium': 0, 'high': 0}
            }
        
        total_cost = sum(rec.total_cost.amount for rec in recommendations)
        unique_materials = len(set(rec.material_id.value for rec in recommendations))
        unique_suppliers = len(set(rec.supplier_id.value for rec in recommendations))
        
        # Calculate average lead time
        total_lead_time = sum(rec.lead_time.days for rec in recommendations)
        average_lead_time = total_lead_time / len(recommendations) if recommendations else 0
        
        # Risk distribution
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
        for rec in recommendations:
            risk_distribution[rec.risk_flag.value] += 1
        
        return {
            'total_cost': float(total_cost),
            'total_materials': unique_materials,
            'total_suppliers': unique_suppliers,
            'total_recommendations': len(recommendations),
            'average_lead_time': average_lead_time,
            'risk_distribution': risk_distribution,
            'cost_per_material': float(total_cost / unique_materials) if unique_materials > 0 else 0.0
        }
    
    def _generate_executive_summary(
        self, 
        recommendations: List[ProcurementRecommendation], 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive-level summary of planning results."""
        
        high_risk_count = metrics['risk_distribution']['high']
        total_recommendations = metrics['total_recommendations']
        
        # Determine overall status
        if high_risk_count > total_recommendations * 0.2:  # > 20% high risk
            overall_status = "attention_required"
            status_message = f"{high_risk_count} high-risk recommendations require attention"
        elif metrics['total_cost'] > 100000:  # Large procurement value
            overall_status = "review_recommended"
            status_message = "Large procurement value - management review recommended"
        else:
            overall_status = "ready_to_execute"
            status_message = "Recommendations ready for execution"
        
        # Key insights
        insights = []
        
        if metrics['total_suppliers'] < metrics['total_materials'] * 0.5:
            insights.append("Consider diversifying supplier base to reduce risk")
        
        if metrics['average_lead_time'] > 30:
            insights.append("Long average lead times may impact stock availability")
        
        if metrics['risk_distribution']['low'] > total_recommendations * 0.8:
            insights.append("Strong supplier reliability across recommendations")
        
        return {
            'overall_status': overall_status,
            'status_message': status_message,
            'key_insights': insights,
            'procurement_value': metrics['total_cost'],
            'materials_covered': metrics['total_materials'],
            'suppliers_involved': metrics['total_suppliers'],
            'execution_priority': 'high' if high_risk_count > 0 else 'normal'
        }
    
    def _calculate_data_completeness(self, domain_objects: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data completeness metrics."""
        
        completeness = {}
        
        # Forecasts completeness
        forecasts = domain_objects.get('forecasts', [])
        if forecasts:
            valid_forecasts = sum(1 for f in forecasts if f.forecast_qty.amount > 0)
            completeness['forecasts'] = (valid_forecasts / len(forecasts)) * 100
        else:
            completeness['forecasts'] = 0.0
        
        # Inventory completeness
        inventory = domain_objects.get('inventory', [])
        if inventory:
            valid_inventory = sum(1 for inv in inventory if inv.on_hand_qty.amount >= 0)
            completeness['inventory'] = (valid_inventory / len(inventory)) * 100
        else:
            completeness['inventory'] = 0.0
        
        # Supplier materials completeness
        supplier_materials = domain_objects.get('supplier_materials', [])
        if supplier_materials:
            valid_suppliers = sum(
                1 for sm in supplier_materials 
                if sm.cost_per_unit.amount > 0 and sm.reliability_score > 0
            )
            completeness['supplier_materials'] = (valid_suppliers / len(supplier_materials)) * 100
        else:
            completeness['supplier_materials'] = 0.0
        
        # BOMs completeness
        boms = domain_objects.get('boms', [])
        if boms:
            valid_boms = sum(1 for bom in boms if bom.qty_per_unit.amount > 0)
            completeness['boms'] = (valid_boms / len(boms)) * 100
        else:
            completeness['boms'] = 0.0
        
        return completeness