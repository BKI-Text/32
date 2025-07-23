"""Analytics Router for Beverly Knits AI Supply Chain Planner API"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
from decimal import Decimal
import json

from ..models.request_models import AnalyticsRequest
from ..models.response_models import AnalyticsResponse, AnalyticsMetric, BaseResponse
from ...src.auth.dependencies import get_current_user

router = APIRouter()

@router.get("/dashboard", response_model=AnalyticsResponse)
async def get_dashboard_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get dashboard analytics
    
    Returns key performance indicators and metrics for the executive dashboard.
    """
    # In production, this would query actual data from the database
    # For now, return comprehensive demo data
    
    metrics = [
        AnalyticsMetric(
            name="Total Materials Managed",
            value=1247,
            unit="items",
            trend="up",
            comparison={"previous_month": 1198, "change": 49}
        ),
        AnalyticsMetric(
            name="Active Suppliers",
            value=87,
            unit="suppliers",
            trend="stable",
            comparison={"previous_month": 85, "change": 2}
        ),
        AnalyticsMetric(
            name="Total Inventory Value",
            value=2847635.50,
            unit="USD",
            trend="up",
            comparison={"previous_month": 2654321.00, "change": 193314.50}
        ),
        AnalyticsMetric(
            name="Average Lead Time",
            value=18.5,
            unit="days",
            trend="down",
            comparison={"previous_month": 21.2, "change": -2.7}
        ),
        AnalyticsMetric(
            name="Procurement Recommendations",
            value=156,
            unit="recommendations",
            trend="up",
            comparison={"previous_week": 142, "change": 14}
        ),
        AnalyticsMetric(
            name="High-Risk Suppliers",
            value=8,
            unit="suppliers",
            trend="down",
            comparison={"previous_month": 12, "change": -4}
        ),
        AnalyticsMetric(
            name="Forecast Accuracy",
            value=87.3,
            unit="percentage",
            trend="up",
            comparison={"previous_month": 84.1, "change": 3.2}
        ),
        AnalyticsMetric(
            name="Cost Savings",
            value=45287.30,
            unit="USD",
            trend="up",
            comparison={"previous_month": 38965.20, "change": 6322.10}
        )
    ]
    
    # Sample chart data
    charts = [
        {
            "id": "cost_trends",
            "type": "line",
            "title": "Cost Trends Over Time",
            "data": [
                {"date": "2025-01-01", "total_cost": 125000},
                {"date": "2025-01-02", "total_cost": 128000},
                {"date": "2025-01-03", "total_cost": 122000},
                {"date": "2025-01-04", "total_cost": 135000},
                {"date": "2025-01-05", "total_cost": 130000}
            ]
        },
        {
            "id": "supplier_risk_distribution",
            "type": "pie",
            "title": "Supplier Risk Distribution",
            "data": [
                {"category": "Low Risk", "value": 65, "percentage": 74.7},
                {"category": "Medium Risk", "value": 14, "percentage": 16.1},
                {"category": "High Risk", "value": 8, "percentage": 9.2}
            ]
        },
        {
            "id": "material_type_breakdown",
            "type": "bar",
            "title": "Material Type Breakdown",
            "data": [
                {"material_type": "Yarn", "count": 487, "value": 1245000},
                {"material_type": "Fabric", "count": 312, "value": 987000},
                {"material_type": "Thread", "count": 198, "value": 245000},
                {"material_type": "Accessory", "count": 156, "value": 189000},
                {"material_type": "Trim", "count": 94, "value": 87000}
            ]
        }
    ]
    
    summary = {
        "total_value": 2847635.50,
        "total_suppliers": 87,
        "total_materials": 1247,
        "active_recommendations": 156,
        "cost_savings_ytd": 245876.50,
        "performance_score": 87.3
    }
    
    return AnalyticsResponse(
        success=True,
        message="Dashboard analytics retrieved successfully",
        metrics=metrics,
        charts=charts,
        summary=summary,
        date_range={
            "start": date.today() - timedelta(days=30),
            "end": date.today()
        }
    )

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    start_date: Optional[date] = Query(None, description="Start date for analysis"),
    end_date: Optional[date] = Query(None, description="End date for analysis"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get performance metrics
    
    Returns detailed performance metrics for the supply chain planning system.
    """
    # Set default date range if not provided
    if not start_date:
        start_date = date.today() - timedelta(days=30)
    if not end_date:
        end_date = date.today()
    
    return {
        "planning_performance": {
            "total_planning_cycles": 127,
            "average_execution_time": 4.2,
            "success_rate": 98.4,
            "recommendations_generated": 3847,
            "total_cost_optimized": 1247685.30
        },
        "forecasting_performance": {
            "forecast_accuracy": 87.3,
            "mean_absolute_error": 12.5,
            "mean_absolute_percentage_error": 0.125,
            "total_forecasts_generated": 2156,
            "ml_model_performance": {
                "arima": {"accuracy": 84.2, "mae": 14.1},
                "prophet": {"accuracy": 89.7, "mae": 11.3},
                "lstm": {"accuracy": 91.2, "mae": 9.8},
                "xgboost": {"accuracy": 86.5, "mae": 13.2}
            }
        },
        "supplier_performance": {
            "average_reliability_score": 0.892,
            "on_time_delivery_rate": 94.7,
            "quality_score": 96.2,
            "cost_competitiveness": 88.9,
            "risk_distribution": {
                "low": 65,
                "medium": 14,
                "high": 8
            }
        },
        "inventory_performance": {
            "inventory_turnover": 8.4,
            "stockout_rate": 2.3,
            "excess_inventory_rate": 5.1,
            "carrying_cost_reduction": 15.7
        },
        "cost_performance": {
            "total_cost_savings": 245876.50,
            "procurement_cost_reduction": 12.3,
            "inventory_cost_reduction": 18.9,
            "average_cost_per_unit": 23.45
        },
        "date_range": {
            "start": start_date,
            "end": end_date
        }
    }

@router.get("/suppliers", response_model=Dict[str, Any])
async def get_supplier_analytics(
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get supplier analytics
    
    Returns detailed analytics about supplier performance and risk assessment.
    """
    return {
        "supplier_summary": {
            "total_suppliers": 87,
            "active_suppliers": 82,
            "new_suppliers_this_month": 3,
            "average_reliability_score": 0.892
        },
        "risk_analysis": {
            "risk_distribution": {
                "low": {"count": 65, "percentage": 74.7},
                "medium": {"count": 14, "percentage": 16.1},
                "high": {"count": 8, "percentage": 9.2}
            },
            "risk_factors": [
                {"factor": "Lead Time Variability", "impact": "High", "affected_suppliers": 12},
                {"factor": "Quality Issues", "impact": "Medium", "affected_suppliers": 8},
                {"factor": "Financial Stability", "impact": "Low", "affected_suppliers": 5}
            ]
        },
        "performance_metrics": {
            "on_time_delivery": {
                "overall": 94.7,
                "by_risk_level": {
                    "low": 97.8,
                    "medium": 91.2,
                    "high": 84.3
                }
            },
            "quality_scores": {
                "overall": 96.2,
                "by_risk_level": {
                    "low": 98.1,
                    "medium": 94.7,
                    "high": 89.2
                }
            },
            "cost_competitiveness": {
                "overall": 88.9,
                "savings_generated": 45287.30
            }
        },
        "top_suppliers": [
            {"name": "Premium Textile Suppliers Inc.", "score": 97.8, "materials": 15},
            {"name": "Eco-Friendly Fibers Ltd.", "score": 96.4, "materials": 12},
            {"name": "Global Yarn Trading Co.", "score": 94.2, "materials": 18}
        ],
        "recommendations": [
            "Review contracts with 3 high-risk suppliers",
            "Diversify sourcing for critical materials",
            "Implement supplier development program"
        ]
    }

@router.get("/inventory", response_model=Dict[str, Any])
async def get_inventory_analytics(
    material_type: Optional[str] = Query(None, description="Filter by material type"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get inventory analytics
    
    Returns detailed analytics about inventory levels, turnover, and optimization.
    """
    return {
        "inventory_summary": {
            "total_materials": 1247,
            "total_value": 2847635.50,
            "average_inventory_level": 2284.50,
            "critical_materials": 156
        },
        "turnover_analysis": {
            "overall_turnover": 8.4,
            "by_material_type": {
                "yarn": 9.2,
                "fabric": 7.8,
                "thread": 12.1,
                "accessory": 6.5,
                "trim": 5.9
            },
            "fast_moving_items": 287,
            "slow_moving_items": 94
        },
        "stock_status": {
            "in_stock": 1089,
            "low_stock": 127,
            "out_of_stock": 31,
            "excess_stock": 64
        },
        "cost_analysis": {
            "carrying_cost": 284763.55,
            "carrying_cost_rate": 0.10,
            "cost_savings_ytd": 58947.20,
            "optimization_opportunities": 127
        },
        "safety_stock": {
            "materials_with_safety_stock": 894,
            "average_safety_stock_days": 12.5,
            "safety_stock_value": 456821.30
        },
        "alerts": [
            {"type": "low_stock", "count": 127, "urgency": "medium"},
            {"type": "stockout", "count": 31, "urgency": "high"},
            {"type": "excess_inventory", "count": 64, "urgency": "low"}
        ]
    }

@router.get("/costs", response_model=Dict[str, Any])
async def get_cost_analytics(
    start_date: Optional[date] = Query(None, description="Start date for analysis"),
    end_date: Optional[date] = Query(None, description="End date for analysis"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get cost analytics
    
    Returns detailed cost analysis and savings opportunities.
    """
    return {
        "cost_summary": {
            "total_procurement_cost": 1247685.30,
            "total_inventory_cost": 2847635.50,
            "total_carrying_cost": 284763.55,
            "total_ordering_cost": 45287.30
        },
        "cost_savings": {
            "total_savings_ytd": 245876.50,
            "procurement_savings": 147523.20,
            "inventory_savings": 78947.30,
            "optimization_savings": 19406.00
        },
        "cost_breakdown": {
            "by_material_type": {
                "yarn": 1245000,
                "fabric": 987000,
                "thread": 245000,
                "accessory": 189000,
                "trim": 87000
            },
            "by_supplier": {
                "Premium Textile Suppliers Inc.": 456789.50,
                "Global Yarn Trading Co.": 389456.20,
                "Eco-Friendly Fibers Ltd.": 234567.80
            }
        },
        "cost_trends": [
            {"period": "2025-01", "cost": 125000, "savings": 8547.30},
            {"period": "2025-02", "cost": 128000, "savings": 9123.45},
            {"period": "2025-03", "cost": 122000, "savings": 10234.56}
        ],
        "optimization_opportunities": [
            {"opportunity": "Consolidate suppliers", "potential_savings": 15000},
            {"opportunity": "Negotiate better terms", "potential_savings": 12000},
            {"opportunity": "Optimize order quantities", "potential_savings": 8000}
        ]
    }

@router.get("/forecasts", response_model=Dict[str, Any])
async def get_forecast_analytics(
    source: Optional[str] = Query(None, description="Filter by forecast source"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get forecast analytics
    
    Returns detailed analytics about forecast accuracy and performance.
    """
    return {
        "forecast_summary": {
            "total_forecasts": 2156,
            "active_forecasts": 1847,
            "overall_accuracy": 87.3,
            "average_confidence": 0.82
        },
        "accuracy_by_source": {
            "sales_order": {"accuracy": 92.1, "count": 547},
            "prod_plan": {"accuracy": 85.7, "count": 623},
            "projection": {"accuracy": 78.9, "count": 445},
            "sales_history": {"accuracy": 88.2, "count": 541}
        },
        "ml_model_performance": {
            "arima": {
                "accuracy": 84.2,
                "mae": 14.1,
                "rmse": 18.7,
                "predictions": 1245
            },
            "prophet": {
                "accuracy": 89.7,
                "mae": 11.3,
                "rmse": 15.2,
                "predictions": 1156
            },
            "lstm": {
                "accuracy": 91.2,
                "mae": 9.8,
                "rmse": 12.9,
                "predictions": 987
            },
            "xgboost": {
                "accuracy": 86.5,
                "mae": 13.2,
                "rmse": 17.1,
                "predictions": 1098
            }
        },
        "accuracy_trends": [
            {"date": "2025-01-01", "accuracy": 85.2},
            {"date": "2025-01-02", "accuracy": 86.1},
            {"date": "2025-01-03", "accuracy": 87.8},
            {"date": "2025-01-04", "accuracy": 88.5},
            {"date": "2025-01-05", "accuracy": 87.3}
        ],
        "error_analysis": {
            "mean_absolute_error": 12.5,
            "mean_absolute_percentage_error": 0.125,
            "root_mean_square_error": 16.8,
            "bias": -0.05
        },
        "improvement_opportunities": [
            "Incorporate more external data sources",
            "Improve seasonal pattern recognition",
            "Enhance model ensemble techniques"
        ]
    }

@router.post("/custom", response_model=AnalyticsResponse)
async def get_custom_analytics(
    analytics_request: AnalyticsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get custom analytics
    
    Returns customized analytics based on the provided parameters.
    """
    # Generate custom metrics based on request
    metrics = []
    
    if analytics_request.include_forecasts:
        metrics.append(
            AnalyticsMetric(
                name="Forecast Accuracy",
                value=87.3,
                unit="percentage",
                trend="up"
            )
        )
    
    if analytics_request.include_recommendations:
        metrics.append(
            AnalyticsMetric(
                name="Active Recommendations",
                value=156,
                unit="recommendations",
                trend="up"
            )
        )
    
    # Generate charts based on grouping
    charts = []
    if "material_type" in analytics_request.group_by:
        charts.append({
            "id": "material_type_analysis",
            "type": "bar",
            "title": "Analysis by Material Type",
            "data": [
                {"category": "Yarn", "value": 487},
                {"category": "Fabric", "value": 312},
                {"category": "Thread", "value": 198}
            ]
        })
    
    return AnalyticsResponse(
        success=True,
        message="Custom analytics generated successfully",
        metrics=metrics,
        charts=charts,
        summary={"custom_analysis": True},
        date_range={
            "start": analytics_request.start_date or date.today() - timedelta(days=30),
            "end": analytics_request.end_date or date.today()
        }
    )

@router.get("/export", response_model=BaseResponse)
async def export_analytics(
    format: str = Query("csv", description="Export format (csv, excel, json)"),
    report_type: str = Query("dashboard", description="Type of report to export"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Export analytics data
    
    Exports analytics data in the specified format.
    
    **Note:** This is a placeholder implementation.
    Production version will generate actual export files.
    """
    if format not in ["csv", "excel", "json"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid export format. Supported formats: csv, excel, json"
        )
    
    if report_type not in ["dashboard", "performance", "suppliers", "inventory", "costs", "forecasts"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report type"
        )
    
    # In production, this would generate the actual export file
    return BaseResponse(
        success=True,
        message=f"Analytics export for {report_type} in {format} format has been queued for generation"
    )

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_analytics_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get analytics alerts
    
    Returns system alerts and notifications based on analytics data.
    """
    alerts = [
        {
            "id": "ALERT_001",
            "type": "inventory",
            "severity": "high",
            "title": "Low Stock Alert",
            "message": "31 materials are out of stock",
            "created_at": datetime.now() - timedelta(hours=2),
            "resolved": False
        },
        {
            "id": "ALERT_002",
            "type": "supplier",
            "severity": "medium",
            "title": "Supplier Risk Alert",
            "message": "8 suppliers have high risk ratings",
            "created_at": datetime.now() - timedelta(hours=5),
            "resolved": False
        },
        {
            "id": "ALERT_003",
            "type": "cost",
            "severity": "low",
            "title": "Cost Optimization",
            "message": "Potential savings of $15,000 identified",
            "created_at": datetime.now() - timedelta(hours=8),
            "resolved": False
        },
        {
            "id": "ALERT_004",
            "type": "forecast",
            "severity": "medium",
            "title": "Forecast Accuracy",
            "message": "Forecast accuracy dropped to 82.1%",
            "created_at": datetime.now() - timedelta(days=1),
            "resolved": True
        }
    ]
    
    # Filter by severity if specified
    if severity:
        alerts = [alert for alert in alerts if alert["severity"] == severity]
    
    return alerts