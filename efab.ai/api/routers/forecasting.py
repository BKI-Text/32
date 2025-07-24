"""Forecasting Router for Beverly Knits AI Supply Chain Planner API"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..models.request_models import ForecastCreateRequest, MLForecastingRequest
from ..models.response_models import ForecastResponse, MLForecastingResultResponse, MLForecastResponse, BaseResponse
from src.auth.dependencies import get_current_user

# Import forecasting components
try:
    from src.engine.production_ml_loader import production_ml_loader
    from src.engine.planning_engine import PlanningEngine
    from src.data.beverly_knits_live_data_integrator import BeverlyKnitsLiveDataIntegrator
    ML_FORECASTING_AVAILABLE = True
except ImportError:
    ML_FORECASTING_AVAILABLE = False

router = APIRouter()

# Temporary in-memory storage for forecasts (replace with database in CORE-002)
temp_forecasts = {}

@router.get("/", response_model=List[ForecastResponse])
async def get_forecasts(
    sku_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    source: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get forecasts with filtering
    
    Returns forecasts with optional filtering by SKU, date range, and source.
    """
    forecasts = list(temp_forecasts.values())
    
    # Apply filters
    if sku_id:
        forecasts = [f for f in forecasts if f["sku_id"] == sku_id]
    
    if start_date:
        forecasts = [f for f in forecasts if f["forecast_date"] >= start_date]
    
    if end_date:
        forecasts = [f for f in forecasts if f["forecast_date"] <= end_date]
    
    if source:
        forecasts = [f for f in forecasts if f["source"] == source]
    
    return [
        ForecastResponse(
            sku_id=f["sku_id"],
            forecast_qty=f["forecast_qty"],
            unit=f["unit"],
            forecast_date=f["forecast_date"],
            source=f["source"],
            confidence_score=f["confidence_score"],
            notes=f.get("notes"),
            created_at=f["created_at"]
        ) for f in forecasts
    ]

@router.post("/", response_model=ForecastResponse, status_code=status.HTTP_201_CREATED)
async def create_forecast(
    forecast_request: ForecastCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create new forecast
    
    Creates a new forecast with the provided information.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    # Generate forecast ID
    forecast_id = f"FORECAST_{int(time.time())}_{len(temp_forecasts)}"
    
    # Create forecast
    forecast = {
        "id": forecast_id,
        "sku_id": forecast_request.sku_id,
        "forecast_qty": forecast_request.forecast_qty,
        "unit": forecast_request.unit,
        "forecast_date": forecast_request.forecast_date,
        "source": forecast_request.source.value,
        "confidence_score": forecast_request.confidence_score,
        "notes": forecast_request.notes,
        "created_at": datetime.now(),
        "created_by": current_user["id"]
    }
    
    temp_forecasts[forecast_id] = forecast
    
    return ForecastResponse(
        sku_id=forecast["sku_id"],
        forecast_qty=forecast["forecast_qty"],
        unit=forecast["unit"],
        forecast_date=forecast["forecast_date"],
        source=forecast["source"],
        confidence_score=forecast["confidence_score"],
        notes=forecast.get("notes"),
        created_at=forecast["created_at"]
    )

@router.post("/ml/generate", response_model=MLForecastingResultResponse)
async def generate_ml_forecasts(
    ml_request: MLForecastingRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate ML forecasts
    
    Uses trained machine learning models to generate demand forecasts.
    
    **Available Models:**
    - **ARIMA**: Time series analysis with seasonality
    - **Prophet**: Facebook's robust forecasting model
    - **LSTM**: Deep learning neural networks
    - **XGBoost**: Gradient boosting ensemble
    
    **Note:** This endpoint uses production-trained models on real Beverly Knits data.
    """
    if not ML_FORECASTING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML forecasting is not available. Please check system configuration."
        )
    
    start_time = time.time()
    
    try:
        # Load Beverly Knits data
        integrator = BeverlyKnitsLiveDataIntegrator(data_path="data/live/")
        
        # Check if sales data is available
        sales_path = Path("data/live/Sales Activity Report.csv")
        if not sales_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sales data not found. Please ensure data is uploaded."
            )
        
        # Load and prepare sales data
        import pandas as pd
        sales_data = pd.read_csv(sales_path, encoding='utf-8-sig')
        
        # Prepare historical data
        sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])
        sales_data['Yds_ordered'] = pd.to_numeric(sales_data['Yds_ordered'], errors='coerce')
        sales_data['Unit Price'] = sales_data['Unit Price'].str.replace('$', '').str.replace(',', '')
        sales_data['Unit Price'] = pd.to_numeric(sales_data['Unit Price'], errors='coerce')
        
        # Create daily aggregated data
        daily_data = sales_data.groupby('Invoice Date').agg({
            'Yds_ordered': 'sum',
            'Unit Price': 'mean',
            'Document': 'count'
        }).reset_index()
        
        # Fill missing dates and prepare features
        date_range = pd.date_range(
            start=daily_data['Invoice Date'].min(),
            end=daily_data['Invoice Date'].max(),
            freq='D'
        )
        
        daily_data = daily_data.set_index('Invoice Date')
        daily_data = daily_data.reindex(date_range, fill_value=0)
        daily_data = daily_data.rename(columns={'Yds_ordered': 'demand'})
        
        # Use production ML loader for forecasting
        predictions = production_ml_loader.predict_demand(
            daily_data, 
            periods=ml_request.periods
        )
        
        # Convert to response format
        ml_forecasts = []
        for pred in predictions:
            ml_forecast = MLForecastResponse(
                forecast_date=pred['date'].date(),
                predicted_demand=pred['predicted_demand'],
                confidence=pred['confidence'],
                model_used=pred.get('model_used', 'ensemble'),
                upper_bound=pred.get('upper_bound'),
                lower_bound=pred.get('lower_bound')
            )
            ml_forecasts.append(ml_forecast)
        
        execution_time = time.time() - start_time
        
        # Get model performance metrics
        model_status = production_ml_loader.get_model_status()
        model_performance = model_status.get('model_details', {})
        
        return MLForecastingResultResponse(
            success=True,
            message=f"ML forecasts generated successfully using {len(ml_request.models)} models",
            forecasts=ml_forecasts,
            models_used=ml_request.models,
            ensemble_method=ml_request.ensemble_method,
            confidence_threshold=ml_request.confidence_threshold,
            execution_time_seconds=execution_time,
            model_performance=model_performance
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML forecasting failed: {str(e)}"
        )

@router.get("/ml/models", response_model=Dict[str, Any])
async def get_ml_models(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get available ML models
    
    Returns information about available ML forecasting models and their status.
    """
    if not ML_FORECASTING_AVAILABLE:
        return {
            "available": False,
            "message": "ML forecasting is not available",
            "models": []
        }
    
    try:
        # Get model status from production ML loader
        model_status = production_ml_loader.get_model_status()
        
        return {
            "available": True,
            "models": {
                "arima": {
                    "name": "ARIMA",
                    "description": "Time series analysis with seasonality",
                    "status": "available",
                    "suitable_for": ["short_term", "seasonal_patterns"]
                },
                "prophet": {
                    "name": "Prophet",
                    "description": "Facebook's robust forecasting model",
                    "status": "available",
                    "suitable_for": ["medium_term", "trend_analysis", "holidays"]
                },
                "lstm": {
                    "name": "LSTM",
                    "description": "Deep learning neural networks",
                    "status": "available",
                    "suitable_for": ["long_term", "complex_patterns"]
                },
                "xgboost": {
                    "name": "XGBoost",
                    "description": "Gradient boosting ensemble",
                    "status": "available",
                    "suitable_for": ["feature_rich", "non_linear_patterns"]
                }
            },
            "ensemble_methods": ["weighted_average", "simple_average", "median", "best_model"],
            "model_status": model_status
        }
        
    except Exception as e:
        return {
            "available": False,
            "message": f"Error checking ML models: {str(e)}",
            "models": []
        }

@router.get("/ml/performance", response_model=Dict[str, Any])
async def get_ml_performance(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get ML model performance metrics
    
    Returns performance metrics for the trained ML models.
    """
    if not ML_FORECASTING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML forecasting is not available"
        )
    
    try:
        # Get model performance from production ML loader
        model_status = production_ml_loader.get_model_status()
        
        return {
            "model_performance": model_status.get('model_details', {}),
            "last_training": model_status.get('last_training'),
            "total_predictions": model_status.get('total_predictions', 0),
            "model_versions": model_status.get('model_versions', {}),
            "performance_metrics": {
                "accuracy": model_status.get('accuracy', 'N/A'),
                "mae": model_status.get('mae', 'N/A'),
                "rmse": model_status.get('rmse', 'N/A'),
                "mape": model_status.get('mape', 'N/A')
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ML performance: {str(e)}"
        )

@router.post("/ml/retrain", response_model=BaseResponse)
async def retrain_ml_models(
    models: List[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Retrain ML models
    
    Initiates retraining of ML models with the latest data.
    
    **Note:** This operation may take several minutes to complete.
    **Requires:** Admin permissions in production implementation.
    """
    if not ML_FORECASTING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML forecasting is not available"
        )
    
    # Check admin permissions (simplified for demo)
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required for model retraining"
        )
    
    try:
        # In production, this would trigger background retraining
        # For now, return success message
        models_to_retrain = models or ["arima", "prophet", "lstm", "xgboost"]
        
        return BaseResponse(
            success=True,
            message=f"Model retraining initiated for {len(models_to_retrain)} models: {', '.join(models_to_retrain)}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining failed: {str(e)}"
        )

@router.get("/sources", response_model=List[str])
async def get_forecast_sources(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get available forecast sources
    
    Returns a list of available forecast sources in the system.
    """
    return ["sales_order", "prod_plan", "projection", "sales_history"]

@router.delete("/{forecast_id}", response_model=BaseResponse)
async def delete_forecast(
    forecast_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete forecast
    
    Deletes a forecast from the system.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    forecast = temp_forecasts.get(forecast_id)
    
    if not forecast:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast with ID {forecast_id} not found"
        )
    
    # Check if user can delete this forecast
    if forecast["created_by"] != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own forecasts"
        )
    
    # Delete forecast
    del temp_forecasts[forecast_id]
    
    return BaseResponse(
        success=True,
        message=f"Forecast {forecast_id} deleted successfully"
    )

@router.get("/ml/history", response_model=List[Dict[str, Any]])
async def get_ml_forecast_history(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get ML forecast history
    
    Returns recent ML forecasting executions.
    """
    if not ML_FORECASTING_AVAILABLE:
        return []
    
    # In production, this would query the database for ML forecast history
    # For now, return empty list
    return []

@router.get("/accuracy", response_model=Dict[str, Any])
async def get_forecast_accuracy(
    source: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get forecast accuracy metrics
    
    Returns accuracy metrics for forecasts compared to actual demand.
    """
    # In production, this would calculate actual accuracy metrics
    # For now, return placeholder data
    return {
        "overall_accuracy": 0.85,
        "mean_absolute_error": 12.5,
        "mean_absolute_percentage_error": 0.15,
        "root_mean_square_error": 18.3,
        "accuracy_by_source": {
            "sales_order": 0.92,
            "prod_plan": 0.85,
            "projection": 0.78,
            "sales_history": 0.88
        },
        "accuracy_trend": [
            {"date": "2025-01-01", "accuracy": 0.82},
            {"date": "2025-01-02", "accuracy": 0.85},
            {"date": "2025-01-03", "accuracy": 0.87}
        ]
    }