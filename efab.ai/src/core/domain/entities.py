from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from .value_objects import Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime

class MaterialType(str, Enum):
    YARN = "yarn"
    FABRIC = "fabric"
    THREAD = "thread"
    ACCESSORY = "accessory"
    TRIM = "trim"

class FabricInventoryStatus(str, Enum):
    """Beverly Knits fabric inventory flow status codes"""
    G00_GREIGE_GOODS = "G00"  # Greige goods - raw undyed fabric
    G02_INTERNAL_MANUFACTURE = "G02"  # Internally manufactured fabric being finished, flow from G00
    G04_EXTERNAL_MANUFACTURE = "G04"  # Externally manufactured greige fabric being finished
    G09_SECOND_QUALITY_GREIGE = "G09"  # Second quality greige fabric
    I01_AWAITING_INSPECTION = "I01"  # Finished fabric waiting for final quality inspection from G02/G04
    F01_FINISHED_INVENTORY = "F01"  # Finished goods inventory ready to ship from I01
    F02_EXTERNAL_FINISHED = "F02"  # Externally purchased finished fabric
    F08_QUARANTINED_QUALITY = "F08"  # Questionable quality fabric that has been quarantined
    F09_SECOND_QUALITY = "F09"  # Second quality fabric that is quarantined
    P01_ALLOCATED = "P01"  # Fabric that has been picked and allocated to a sales order
    T01_AWAITING_TEST = "T01"  # Fabric that is awaiting test results before it can be added to F01
    BH_BILLED_HELD = "BH"  # Fabric that has been billed to customer but being held at facility

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
    
    @field_validator('reliability_score')
    @classmethod
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
    
    @field_validator('reliability_score')
    @classmethod
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

class FabricInventory(BaseModel):
    """Enhanced fabric inventory with Beverly Knits status tracking"""
    material_id: MaterialId
    status: FabricInventoryStatus = Field(..., description="Current fabric inventory status")
    quantity: Quantity = Field(..., description="Quantity in current status")
    location: Optional[str] = Field(None, description="Physical location or facility")
    lot_number: Optional[str] = Field(None, description="Fabric lot/batch number")
    quality_grade: Optional[str] = Field(None, description="Quality grade (First, Second, etc.)")
    allocated_to: Optional[str] = Field(None, description="Sales order or customer allocation")
    test_results: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Quality test results")
    quarantine_reason: Optional[str] = Field(None, description="Reason for quarantine if applicable")
    expected_release_date: Optional[date] = Field(None, description="Expected release from current status")
    last_status_change: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def can_ship(self) -> bool:
        """Check if fabric is ready for shipment"""
        return self.status == FabricInventoryStatus.F01_FINISHED_INVENTORY
    
    def is_available_for_allocation(self) -> bool:
        """Check if fabric is available for sales order allocation"""
        available_statuses = {
            FabricInventoryStatus.F01_FINISHED_INVENTORY,
            FabricInventoryStatus.F02_EXTERNAL_FINISHED
        }
        return self.status in available_statuses and self.allocated_to is None
    
    def is_in_production(self) -> bool:
        """Check if fabric is currently in production/processing"""
        production_statuses = {
            FabricInventoryStatus.G00_GREIGE_GOODS,
            FabricInventoryStatus.G02_INTERNAL_MANUFACTURE,
            FabricInventoryStatus.G04_EXTERNAL_MANUFACTURE,
            FabricInventoryStatus.I01_AWAITING_INSPECTION,
            FabricInventoryStatus.T01_AWAITING_TEST
        }
        return self.status in production_statuses
    
    def is_quarantined(self) -> bool:
        """Check if fabric is quarantined"""
        quarantine_statuses = {
            FabricInventoryStatus.F08_QUARANTINED_QUALITY,
            FabricInventoryStatus.F09_SECOND_QUALITY,
            FabricInventoryStatus.G09_SECOND_QUALITY_GREIGE
        }
        return self.status in quarantine_statuses
    
    def update_status(self, new_status: FabricInventoryStatus, reason: Optional[str] = None):
        """Update fabric inventory status with audit trail"""
        self.status = new_status
        self.last_status_change = datetime.now()
        if reason:
            self.quarantine_reason = reason if new_status.value.endswith('9') else None

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
    
    @field_validator('confidence_score')
    @classmethod
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
    
    @field_validator('urgency_score')
    @classmethod
    def validate_urgency(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Urgency score must be between 0 and 1')
        return v