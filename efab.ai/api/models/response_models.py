"""Response Models for Beverly Knits AI Supply Chain Planner API"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

# Base Response Models
class BaseResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class ErrorResponse(BaseResponse):
    error_code: str = Field(..., description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    success: bool = Field(default=False)

class PaginatedResponse(BaseModel):
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")

# Authentication Response Models
class TokenResponse(BaseResponse):
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

class UserResponse(BaseModel):
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    role: str = Field(..., description="User role")
    is_active: bool = Field(..., description="User active status")
    created_at: datetime = Field(..., description="User creation date")
    last_login: Optional[datetime] = Field(None, description="Last login date")

# Material Response Models
class MaterialResponse(BaseModel):
    id: str = Field(..., description="Material ID")
    name: str = Field(..., description="Material name")
    type: str = Field(..., description="Material type")
    description: Optional[str] = Field(None, description="Material description")
    specifications: Dict[str, str] = Field(default_factory=dict)
    is_critical: bool = Field(..., description="Critical material flag")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

class MaterialListResponse(BaseResponse):
    materials: List[MaterialResponse] = Field(..., description="List of materials")
    total: int = Field(..., description="Total number of materials")

# Supplier Response Models
class SupplierResponse(BaseModel):
    id: str = Field(..., description="Supplier ID")
    name: str = Field(..., description="Supplier name")
    contact_info: Optional[str] = Field(None, description="Contact information")
    lead_time_days: int = Field(..., description="Default lead time in days")
    reliability_score: float = Field(..., description="Reliability score")
    risk_level: str = Field(..., description="Risk level")
    is_active: bool = Field(..., description="Active supplier flag")
    created_at: datetime = Field(..., description="Creation timestamp")
    material_count: int = Field(default=0, description="Number of materials supplied")

class SupplierListResponse(BaseResponse):
    suppliers: List[SupplierResponse] = Field(..., description="List of suppliers")
    total: int = Field(..., description="Total number of suppliers")

class SupplierMaterialResponse(BaseModel):
    supplier_id: str = Field(..., description="Supplier ID")
    material_id: str = Field(..., description="Material ID")
    cost_per_unit: Decimal = Field(..., description="Cost per unit")
    currency: str = Field(..., description="Currency")
    moq_amount: Decimal = Field(..., description="Minimum order quantity")
    moq_unit: str = Field(..., description="MOQ unit")
    lead_time_days: int = Field(..., description="Lead time in days")
    reliability_score: float = Field(..., description="Reliability score")
    ordering_cost: Decimal = Field(..., description="Ordering cost")
    holding_cost_rate: float = Field(..., description="Holding cost rate")

# Inventory Response Models
class InventoryResponse(BaseModel):
    material_id: str = Field(..., description="Material ID")
    on_hand_qty: Decimal = Field(..., description="On hand quantity")
    unit: str = Field(..., description="Unit of measurement")
    open_po_qty: Decimal = Field(..., description="Open PO quantity")
    po_expected_date: Optional[date] = Field(None, description="Expected PO delivery date")
    safety_stock: Decimal = Field(..., description="Safety stock")
    available_qty: Decimal = Field(..., description="Available quantity")
    last_updated: datetime = Field(..., description="Last update timestamp")

# BOM Response Models
class BOMResponse(BaseModel):
    sku_id: str = Field(..., description="SKU ID")
    material_id: str = Field(..., description="Material ID")
    qty_per_unit: Decimal = Field(..., description="Quantity per unit")
    unit: str = Field(..., description="Unit of measurement")
    material_name: Optional[str] = Field(None, description="Material name")
    material_type: Optional[str] = Field(None, description="Material type")

# Forecast Response Models
class ForecastResponse(BaseModel):
    sku_id: str = Field(..., description="SKU ID")
    forecast_qty: Decimal = Field(..., description="Forecast quantity")
    unit: str = Field(..., description="Unit of measurement")
    forecast_date: date = Field(..., description="Forecast date")
    source: str = Field(..., description="Forecast source")
    confidence_score: float = Field(..., description="Confidence score")
    notes: Optional[str] = Field(None, description="Additional notes")
    created_at: datetime = Field(..., description="Creation timestamp")

# Planning Response Models
class ProcurementRecommendationResponse(BaseModel):
    material_id: str = Field(..., description="Material ID")
    supplier_id: str = Field(..., description="Supplier ID")
    recommended_order_qty: Decimal = Field(..., description="Recommended order quantity")
    unit: str = Field(..., description="Unit of measurement")
    unit_cost: Decimal = Field(..., description="Unit cost")
    total_cost: Decimal = Field(..., description="Total cost")
    currency: str = Field(..., description="Currency")
    expected_lead_time_days: int = Field(..., description="Expected lead time in days")
    risk_level: str = Field(..., description="Risk level")
    urgency_score: float = Field(..., description="Urgency score")
    reasoning: str = Field(..., description="Recommendation reasoning")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    material_name: Optional[str] = Field(None, description="Material name")

class PlanningResultResponse(BaseResponse):
    recommendations: List[ProcurementRecommendationResponse] = Field(..., description="Procurement recommendations")
    total_recommendations: int = Field(..., description="Total number of recommendations")
    total_cost: Decimal = Field(..., description="Total cost of all recommendations")
    currency: str = Field(default="USD", description="Currency")
    planning_horizon_days: int = Field(..., description="Planning horizon used")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Planning statistics")

# ML Forecasting Response Models
class MLForecastResponse(BaseModel):
    forecast_date: date = Field(..., description="Forecast date")
    predicted_demand: Decimal = Field(..., description="Predicted demand")
    confidence: float = Field(..., description="Confidence score")
    model_used: str = Field(..., description="Model used for prediction")
    upper_bound: Optional[Decimal] = Field(None, description="Upper confidence bound")
    lower_bound: Optional[Decimal] = Field(None, description="Lower confidence bound")

class MLForecastingResultResponse(BaseResponse):
    forecasts: List[MLForecastResponse] = Field(..., description="ML forecasts")
    models_used: List[str] = Field(..., description="Models used")
    ensemble_method: str = Field(..., description="Ensemble method used")
    confidence_threshold: float = Field(..., description="Confidence threshold")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    model_performance: Dict[str, Any] = Field(default_factory=dict, description="Model performance metrics")

# Analytics Response Models
class AnalyticsMetric(BaseModel):
    name: str = Field(..., description="Metric name")
    value: Union[int, float, str] = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    trend: Optional[str] = Field(None, description="Trend direction")
    comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison data")

class AnalyticsResponse(BaseResponse):
    metrics: List[AnalyticsMetric] = Field(..., description="Analytics metrics")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="Chart data")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    date_range: Dict[str, date] = Field(..., description="Date range analyzed")

# System Response Models
class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Service statuses")
    uptime_seconds: int = Field(..., description="Uptime in seconds")

class SystemInfoResponse(BaseModel):
    api_name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    documentation: Dict[str, str] = Field(..., description="Documentation links")

# Data Upload Response Models
class DataUploadResponse(BaseResponse):
    file_name: str = Field(..., description="Uploaded file name")
    file_type: str = Field(..., description="File type")
    records_processed: int = Field(..., description="Number of records processed")
    records_created: int = Field(..., description="Number of records created")
    records_updated: int = Field(..., description="Number of records updated")
    records_failed: int = Field(..., description="Number of records failed")
    quality_fixes_applied: List[str] = Field(default_factory=list, description="Quality fixes applied")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")

# Validation Response Models
class ValidationErrorDetail(BaseModel):
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    value: Any = Field(..., description="Invalid value")

class ValidationErrorResponse(ErrorResponse):
    validation_errors: List[ValidationErrorDetail] = Field(..., description="Validation errors")
    error_code: str = Field(default="VALIDATION_ERROR")

# Generic Success Response
class GenericSuccessResponse(BaseResponse):
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    success: bool = Field(default=True)