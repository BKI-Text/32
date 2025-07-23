"""Planning Router for Beverly Knits AI Supply Chain Planner API"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, List
from datetime import datetime, timedelta
import time
import asyncio
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..models.request_models import PlanningExecuteRequest
from ..models.response_models import PlanningResultResponse, ProcurementRecommendationResponse, BaseResponse
from ...src.auth.dependencies import get_current_user

# Import planning engine
try:
    from src.engine.planning_engine import PlanningEngine
    from src.data.beverly_knits_live_data_integrator import BeverlyKnitsLiveDataIntegrator
    PLANNING_ENGINE_AVAILABLE = True
except ImportError:
    PLANNING_ENGINE_AVAILABLE = False

router = APIRouter()

# Planning execution status storage
planning_status = {}

@router.post("/execute", response_model=PlanningResultResponse)
async def execute_planning(
    planning_request: PlanningExecuteRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute planning cycle
    
    Runs the 6-phase planning engine with the provided parameters.
    
    **6-Phase Process:**
    1. Forecast Unification - Weighted consolidation of demand signals
    2. BOM Explosion - SKU-to-material requirement conversion
    3. Inventory Netting - Current stock and open PO accounting
    4. Procurement Optimization - EOQ and safety stock calculations
    5. Supplier Selection - Multi-criteria optimization
    6. Output Generation - Actionable recommendations
    
    **Note:** This may take several minutes for large datasets.
    """
    if not PLANNING_ENGINE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Planning engine is not available. Please check system configuration."
        )
    
    execution_id = f"planning_{int(time.time())}"
    start_time = time.time()
    
    # Store execution status
    planning_status[execution_id] = {
        "status": "running",
        "started_at": datetime.now(),
        "user_id": current_user["id"],
        "parameters": planning_request.dict()
    }
    
    try:
        # Initialize components
        integrator = BeverlyKnitsLiveDataIntegrator(data_path="data/live/")
        engine = PlanningEngine()
        
        # Load and process data
        domain_objects = integrator.integrate_live_data()
        
        # Execute planning cycle
        recommendations = engine.execute_planning_cycle(
            forecasts=domain_objects.get('forecasts', []),
            boms=domain_objects.get('boms', []),
            inventory=domain_objects.get('inventory', []),
            suppliers=domain_objects.get('supplier_materials', [])
        )
        
        execution_time = time.time() - start_time
        
        # Convert recommendations to response format
        recommendation_responses = []
        total_cost = 0
        
        for rec in recommendations:
            rec_response = ProcurementRecommendationResponse(
                material_id=rec.material_id.value,
                supplier_id=rec.supplier_id.value,
                recommended_order_qty=rec.recommended_order_qty.amount,
                unit=rec.recommended_order_qty.unit,
                unit_cost=rec.unit_cost.amount,
                total_cost=rec.total_cost.amount,
                currency=rec.total_cost.currency,
                expected_lead_time_days=rec.expected_lead_time.days,
                risk_level=rec.risk_flag.value,
                urgency_score=rec.urgency_score,
                reasoning=rec.reasoning
            )
            recommendation_responses.append(rec_response)
            total_cost += float(rec.total_cost.amount)
        
        # Calculate statistics
        statistics = {
            "total_materials_processed": len(set(rec.material_id for rec in recommendations)),
            "unique_suppliers_selected": len(set(rec.supplier_id for rec in recommendations)),
            "high_risk_recommendations": len([rec for rec in recommendations if rec.risk_flag.value == "HIGH"]),
            "medium_risk_recommendations": len([rec for rec in recommendations if rec.risk_flag.value == "MEDIUM"]),
            "low_risk_recommendations": len([rec for rec in recommendations if rec.risk_flag.value == "LOW"]),
            "average_urgency_score": sum(rec.urgency_score for rec in recommendations) / len(recommendations) if recommendations else 0,
            "average_lead_time_days": sum(rec.expected_lead_time.days for rec in recommendations) / len(recommendations) if recommendations else 0
        }
        
        # Update planning status
        planning_status[execution_id] = {
            "status": "completed",
            "started_at": planning_status[execution_id]["started_at"],
            "completed_at": datetime.now(),
            "user_id": current_user["id"],
            "parameters": planning_request.dict(),
            "results": {
                "total_recommendations": len(recommendations),
                "total_cost": total_cost,
                "execution_time": execution_time
            }
        }
        
        return PlanningResultResponse(
            success=True,
            message=f"Planning cycle completed successfully. Generated {len(recommendations)} recommendations.",
            recommendations=recommendation_responses,
            total_recommendations=len(recommendations),
            total_cost=total_cost,
            planning_horizon_days=planning_request.planning_horizon_days,
            execution_time_seconds=execution_time,
            statistics=statistics
        )
        
    except Exception as e:
        # Update planning status with error
        planning_status[execution_id] = {
            "status": "failed",
            "started_at": planning_status[execution_id]["started_at"],
            "failed_at": datetime.now(),
            "user_id": current_user["id"],
            "parameters": planning_request.dict(),
            "error": str(e)
        }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Planning execution failed: {str(e)}"
        )

@router.get("/status/{execution_id}", response_model=Dict[str, Any])
async def get_planning_status(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get planning execution status
    
    Returns the current status of a planning execution.
    """
    status_info = planning_status.get(execution_id)
    
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Planning execution {execution_id} not found"
        )
    
    # Check if user has access to this execution
    if status_info["user_id"] != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this planning execution"
        )
    
    return status_info

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_planning_history(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get planning execution history
    
    Returns recent planning executions for the current user.
    """
    user_executions = [
        {
            "execution_id": exec_id,
            **exec_info
        }
        for exec_id, exec_info in planning_status.items()
        if exec_info["user_id"] == current_user["id"] or current_user["role"] == "admin"
    ]
    
    # Sort by started_at descending
    user_executions.sort(key=lambda x: x["started_at"], reverse=True)
    
    return user_executions[:limit]

@router.post("/validate", response_model=BaseResponse)
async def validate_planning_parameters(
    planning_request: PlanningExecuteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Validate planning parameters
    
    Validates planning parameters without executing the planning cycle.
    """
    validation_errors = []
    
    # Validate weight combinations
    if planning_request.cost_weight + planning_request.reliability_weight > 1.0:
        validation_errors.append("Cost weight and reliability weight must sum to 1.0 or less")
    
    # Validate planning horizon
    if planning_request.planning_horizon_days > 365:
        validation_errors.append("Planning horizon cannot exceed 365 days")
    
    # Validate safety stock percentage
    if planning_request.safety_stock_percentage > 0.5:
        validation_errors.append("Safety stock percentage should not exceed 50%")
    
    # Check data availability
    if not PLANNING_ENGINE_AVAILABLE:
        validation_errors.append("Planning engine is not available")
    else:
        try:
            integrator = BeverlyKnitsLiveDataIntegrator(data_path="data/live/")
            domain_objects = integrator.integrate_live_data()
            
            if not domain_objects.get('forecasts'):
                validation_errors.append("No forecast data available")
            if not domain_objects.get('boms'):
                validation_errors.append("No BOM data available")
            if not domain_objects.get('inventory'):
                validation_errors.append("No inventory data available")
            if not domain_objects.get('supplier_materials'):
                validation_errors.append("No supplier material data available")
                
        except Exception as e:
            validation_errors.append(f"Data validation failed: {str(e)}")
    
    if validation_errors:
        return BaseResponse(
            success=False,
            message=f"Validation failed: {'; '.join(validation_errors)}"
        )
    
    return BaseResponse(
        success=True,
        message="Planning parameters are valid"
    )

@router.post("/sales-based", response_model=PlanningResultResponse)
async def execute_sales_based_planning(
    planning_request: PlanningExecuteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute sales-based planning cycle
    
    Runs the planning engine using historical sales data for demand forecasting.
    
    **Note:** This method uses actual sales patterns to generate more accurate forecasts.
    """
    if not PLANNING_ENGINE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Planning engine is not available. Please check system configuration."
        )
    
    execution_id = f"sales_planning_{int(time.time())}"
    start_time = time.time()
    
    try:
        # Initialize components
        integrator = BeverlyKnitsLiveDataIntegrator(data_path="data/live/")
        engine = PlanningEngine()
        
        # Load and process data
        domain_objects = integrator.integrate_live_data()
        
        # Check if sales data is available
        sales_data = getattr(integrator, 'sales_data', None)
        style_bom_data = getattr(integrator, 'style_bom_data', None)
        
        if sales_data is None or style_bom_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sales data not available for sales-based planning"
            )
        
        # Execute sales-based planning cycle
        recommendations = engine.execute_sales_based_planning_cycle(
            sales_data=sales_data,
            style_bom_data=style_bom_data,
            inventory=domain_objects.get('inventory', []),
            suppliers=domain_objects.get('supplier_materials', [])
        )
        
        execution_time = time.time() - start_time
        
        # Convert recommendations to response format
        recommendation_responses = []
        total_cost = 0
        
        for rec in recommendations:
            rec_response = ProcurementRecommendationResponse(
                material_id=rec.material_id.value,
                supplier_id=rec.supplier_id.value,
                recommended_order_qty=rec.recommended_order_qty.amount,
                unit=rec.recommended_order_qty.unit,
                unit_cost=rec.unit_cost.amount,
                total_cost=rec.total_cost.amount,
                currency=rec.total_cost.currency,
                expected_lead_time_days=rec.expected_lead_time.days,
                risk_level=rec.risk_flag.value,
                urgency_score=rec.urgency_score,
                reasoning=rec.reasoning
            )
            recommendation_responses.append(rec_response)
            total_cost += float(rec.total_cost.amount)
        
        # Calculate statistics
        statistics = {
            "planning_type": "sales_based",
            "total_materials_processed": len(set(rec.material_id for rec in recommendations)),
            "unique_suppliers_selected": len(set(rec.supplier_id for rec in recommendations)),
            "high_risk_recommendations": len([rec for rec in recommendations if rec.risk_flag.value == "HIGH"]),
            "medium_risk_recommendations": len([rec for rec in recommendations if rec.risk_flag.value == "MEDIUM"]),
            "low_risk_recommendations": len([rec for rec in recommendations if rec.risk_flag.value == "LOW"]),
            "average_urgency_score": sum(rec.urgency_score for rec in recommendations) / len(recommendations) if recommendations else 0,
            "average_lead_time_days": sum(rec.expected_lead_time.days for rec in recommendations) / len(recommendations) if recommendations else 0
        }
        
        return PlanningResultResponse(
            success=True,
            message=f"Sales-based planning cycle completed successfully. Generated {len(recommendations)} recommendations.",
            recommendations=recommendation_responses,
            total_recommendations=len(recommendations),
            total_cost=total_cost,
            planning_horizon_days=planning_request.planning_horizon_days,
            execution_time_seconds=execution_time,
            statistics=statistics
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sales-based planning execution failed: {str(e)}"
        )

@router.get("/configuration", response_model=Dict[str, Any])
async def get_planning_configuration(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get planning configuration
    
    Returns the current planning configuration and available options.
    """
    return {
        "default_parameters": {
            "safety_stock_percentage": 0.15,
            "planning_horizon_days": 90,
            "cost_weight": 0.6,
            "reliability_weight": 0.4,
            "max_suppliers_per_material": 3,
            "enable_eoq_optimization": True,
            "enable_multi_supplier": True,
            "enable_risk_assessment": True
        },
        "parameter_ranges": {
            "safety_stock_percentage": {"min": 0.0, "max": 0.5},
            "planning_horizon_days": {"min": 1, "max": 365},
            "cost_weight": {"min": 0.0, "max": 1.0},
            "reliability_weight": {"min": 0.0, "max": 1.0},
            "max_suppliers_per_material": {"min": 1, "max": 10}
        },
        "available_features": {
            "eoq_optimization": True,
            "multi_supplier_sourcing": True,
            "risk_assessment": True,
            "sales_based_planning": True,
            "ml_enhanced_planning": PLANNING_ENGINE_AVAILABLE
        }
    }

@router.delete("/history/{execution_id}", response_model=BaseResponse)
async def delete_planning_execution(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete planning execution
    
    Removes a planning execution from the history.
    """
    status_info = planning_status.get(execution_id)
    
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Planning execution {execution_id} not found"
        )
    
    # Check if user has access to this execution
    if status_info["user_id"] != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this planning execution"
        )
    
    # Delete execution
    del planning_status[execution_id]
    
    return BaseResponse(
        success=True,
        message=f"Planning execution {execution_id} deleted successfully"
    )