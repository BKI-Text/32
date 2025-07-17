from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator
from .value_objects import Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime

class MaterialType(str, Enum):
    YARN = "yarn"
    FABRIC = "fabric"
    THREAD = "thread"
    ACCESSORY = "accessory"
    TRIM = "trim"

class ForecastSource(str, Enum):
    SALES_ORDER = "sales_order"
    PROD_PLAN = "prod_plan"
    PROJECTION = "projection"
    SALES_HISTORY = "sales_history"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Material(BaseModel):
    id: MaterialId
    name: str = Field(..., description="Material name")
    type: MaterialType = Field(..., description="Material type")
    description: Optional[str] = Field(None, description="Material description")
    specifications: Dict[str, str] = Field(default_factory=dict)
    is_critical: bool = Field(default=False, description="Critical material flag")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def update_specifications(self, specs: Dict[str, str]):
        self.specifications.update(specs)
        self.updated_at = datetime.now()

class Supplier(BaseModel):
    id: SupplierId
    name: str = Field(..., description="Supplier name")
    contact_info: Optional[str] = Field(None, description="Contact information")
    lead_time: LeadTime = Field(..., description="Default lead time")
    reliability_score: float = Field(..., ge=0, le=1, description="Reliability score 0-1")
    risk_level: RiskLevel = Field(..., description="Risk assessment")
    is_active: bool = Field(default=True, description="Active supplier flag")
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('reliability_score')
    def validate_reliability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Reliability score must be between 0 and 1')
        return v

class SupplierMaterial(BaseModel):
    supplier_id: SupplierId
    material_id: MaterialId
    cost_per_unit: Money
    moq: Quantity
    lead_time: LeadTime
    contract_qty_limit: Optional[Quantity] = Field(None, description="Contract quantity limit")
    reliability_score: float = Field(..., ge=0, le=1)
    ordering_cost: Money = Field(default=Money(amount=Decimal("100.0"), currency="USD"))
    holding_cost_rate: float = Field(default=0.25, ge=0, le=1)
    
    @validator('reliability_score')
    def validate_reliability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Reliability score must be between 0 and 1')
        return v

class Inventory(BaseModel):
    material_id: MaterialId
    on_hand_qty: Quantity
    open_po_qty: Quantity = Field(default=Quantity(amount=Decimal("0"), unit="unit"))
    po_expected_date: Optional[date] = Field(None, description="Expected PO delivery date")
    safety_stock: Quantity = Field(default=Quantity(amount=Decimal("0"), unit="unit"))
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def get_available_qty(self) -> Quantity:
        return self.on_hand_qty + self.open_po_qty

class BOM(BaseModel):
    sku_id: SkuId
    material_id: MaterialId
    qty_per_unit: Quantity
    unit: str = Field(..., description="Unit of measurement")
    
    def calculate_requirement(self, sku_qty: Quantity) -> Quantity:
        if self.qty_per_unit.unit != self.unit:
            raise ValueError(f"Unit mismatch: {self.qty_per_unit.unit} vs {self.unit}")
        
        required_amount = self.qty_per_unit.amount * sku_qty.amount
        return Quantity(amount=required_amount, unit=self.unit)

class Forecast(BaseModel):
    sku_id: SkuId
    forecast_qty: Quantity
    forecast_date: date
    source: ForecastSource
    confidence_score: float = Field(default=0.8, ge=0, le=1)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v

class ProcurementRecommendation(BaseModel):
    material_id: MaterialId
    supplier_id: SupplierId
    recommended_order_qty: Quantity
    unit_cost: Money
    total_cost: Money
    expected_lead_time: LeadTime
    risk_flag: RiskLevel
    reasoning: str
    urgency_score: float = Field(..., ge=0, le=1)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('urgency_score')
    def validate_urgency(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Urgency score must be between 0 and 1')
        return v