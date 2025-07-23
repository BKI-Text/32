"""Request Models for Beverly Knits AI Supply Chain Planner API"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

# Enums for request validation
class MaterialTypeRequest(str, Enum):
    YARN = "yarn"
    FABRIC = "fabric"
    THREAD = "thread"
    ACCESSORY = "accessory"
    TRIM = "trim"

class ForecastSourceRequest(str, Enum):
    SALES_ORDER = "sales_order"
    PROD_PLAN = "prod_plan"
    PROJECTION = "projection"
    SALES_HISTORY = "sales_history"

class RiskLevelRequest(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Authentication Requests
class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Refresh token")

class UserCreateRequest(BaseModel):
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    full_name: str = Field(..., description="Full name")
    role: str = Field(default="user", description="User role")

class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., description="New password")

# Material Requests
class MaterialCreateRequest(BaseModel):
    name: str = Field(..., description="Material name")
    type: MaterialTypeRequest = Field(..., description="Material type")
    description: Optional[str] = Field(None, description="Material description")
    specifications: Dict[str, str] = Field(default_factory=dict)
    is_critical: bool = Field(default=False, description="Critical material flag")

class MaterialUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Material name")
    type: Optional[MaterialTypeRequest] = Field(None, description="Material type")
    description: Optional[str] = Field(None, description="Material description")
    specifications: Optional[Dict[str, str]] = Field(None)
    is_critical: Optional[bool] = Field(None, description="Critical material flag")

# Supplier Requests
class SupplierCreateRequest(BaseModel):
    name: str = Field(..., description="Supplier name")
    contact_info: Optional[str] = Field(None, description="Contact information")
    lead_time_days: int = Field(..., description="Default lead time in days")
    reliability_score: float = Field(..., ge=0, le=1, description="Reliability score 0-1")
    risk_level: RiskLevelRequest = Field(..., description="Risk assessment")
    is_active: bool = Field(default=True, description="Active supplier flag")

class SupplierUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Supplier name")
    contact_info: Optional[str] = Field(None, description="Contact information")
    lead_time_days: Optional[int] = Field(None, description="Default lead time in days")
    reliability_score: Optional[float] = Field(None, ge=0, le=1, description="Reliability score 0-1")
    risk_level: Optional[RiskLevelRequest] = Field(None, description="Risk assessment")
    is_active: Optional[bool] = Field(None, description="Active supplier flag")

# Supplier Material Requests
class SupplierMaterialRequest(BaseModel):
    supplier_id: str = Field(..., description="Supplier ID")
    material_id: str = Field(..., description="Material ID")
    cost_per_unit: Decimal = Field(..., description="Cost per unit")
    currency: str = Field(default="USD", description="Currency")
    moq_amount: Decimal = Field(..., description="Minimum order quantity")
    moq_unit: str = Field(..., description="MOQ unit")
    lead_time_days: int = Field(..., description="Lead time in days")
    reliability_score: float = Field(..., ge=0, le=1)
    ordering_cost: Decimal = Field(default=Decimal("100.0"), description="Ordering cost")
    holding_cost_rate: float = Field(default=0.25, ge=0, le=1)

# Inventory Requests
class InventoryUpdateRequest(BaseModel):
    material_id: str = Field(..., description="Material ID")
    on_hand_qty: Decimal = Field(..., description="On hand quantity")
    unit: str = Field(..., description="Unit of measurement")
    open_po_qty: Decimal = Field(default=Decimal("0"), description="Open PO quantity")
    po_expected_date: Optional[date] = Field(None, description="Expected PO delivery date")
    safety_stock: Decimal = Field(default=Decimal("0"), description="Safety stock")

# BOM Requests
class BOMCreateRequest(BaseModel):
    sku_id: str = Field(..., description="SKU ID")
    material_id: str = Field(..., description="Material ID")
    qty_per_unit: Decimal = Field(..., description="Quantity per unit")
    unit: str = Field(..., description="Unit of measurement")

# Forecast Requests
class ForecastCreateRequest(BaseModel):
    sku_id: str = Field(..., description="SKU ID")
    forecast_qty: Decimal = Field(..., description="Forecast quantity")
    unit: str = Field(..., description="Unit of measurement")
    forecast_date: date = Field(..., description="Forecast date")
    source: ForecastSourceRequest = Field(..., description="Forecast source")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")
    notes: Optional[str] = Field(None, description="Additional notes")

# Planning Requests
class PlanningExecuteRequest(BaseModel):
    safety_stock_percentage: float = Field(default=0.15, ge=0, le=1, description="Safety stock percentage")
    planning_horizon_days: int = Field(default=90, ge=1, le=365, description="Planning horizon in days")
    cost_weight: float = Field(default=0.6, ge=0, le=1, description="Cost weight in supplier selection")
    reliability_weight: float = Field(default=0.4, ge=0, le=1, description="Reliability weight")
    max_suppliers_per_material: int = Field(default=3, ge=1, le=10, description="Max suppliers per material")
    enable_eoq_optimization: bool = Field(default=True, description="Enable EOQ optimization")
    enable_multi_supplier: bool = Field(default=True, description="Enable multi-supplier sourcing")
    enable_risk_assessment: bool = Field(default=True, description="Enable risk assessment")

# ML Forecasting Requests
class MLForecastingRequest(BaseModel):
    periods: int = Field(default=30, ge=1, le=365, description="Number of periods to forecast")
    models: List[str] = Field(default=["arima", "prophet"], description="ML models to use")
    ensemble_method: str = Field(default="weighted_average", description="Ensemble method")
    confidence_threshold: float = Field(default=0.8, ge=0.5, le=1.0, description="Confidence threshold")

# Analytics Requests
class AnalyticsRequest(BaseModel):
    start_date: Optional[date] = Field(None, description="Start date for analytics")
    end_date: Optional[date] = Field(None, description="End date for analytics")
    include_forecasts: bool = Field(default=True, description="Include forecast data")
    include_recommendations: bool = Field(default=True, description="Include recommendations")
    group_by: List[str] = Field(default=["material_type"], description="Group by fields")

# Data Upload Requests
class DataUploadRequest(BaseModel):
    file_type: str = Field(..., description="Type of data file")
    file_name: str = Field(..., description="File name")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing data")
    apply_quality_fixes: bool = Field(default=True, description="Apply automatic quality fixes")

# Pagination Request
class PaginationRequest(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort by field")
    sort_order: str = Field(default="asc", description="Sort order (asc/desc)")

# Filter Request
class FilterRequest(BaseModel):
    field: str = Field(..., description="Field to filter")
    operator: str = Field(..., description="Filter operator (eq, ne, gt, lt, in, contains)")
    value: Any = Field(..., description="Filter value")

# Validation helpers
class RequestValidator:
    @staticmethod
    def validate_weights(cost_weight: float, reliability_weight: float) -> bool:
        """Validate that weights sum to reasonable values"""
        return abs(cost_weight + reliability_weight - 1.0) <= 0.1

    @staticmethod
    def validate_date_range(start_date: Optional[date], end_date: Optional[date]) -> bool:
        """Validate date range"""
        if start_date and end_date:
            return start_date <= end_date
        return True