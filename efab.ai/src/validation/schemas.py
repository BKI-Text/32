"""Business Entity Validation Schemas for Beverly Knits AI Supply Chain Planner"""

from typing import Dict, List, Optional, Union
from pydantic import validator, Field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
import re

from .base import (
    ValidatedModel, BusinessRule, ValidationLevel, ValidationContext, ValidationPatterns,
    PositiveNumberRule, NonNegativeNumberRule, FutureDateRule, ValidLeadTimeRule,
    ReasonableQuantityRule, create_validation_decorator
)

# Business Rule Implementations

class ValidSkuRule(BusinessRule):
    """Rule to validate SKU format"""
    
    def __init__(self):
        super().__init__(
            name="valid_sku",
            description="SKU must be 3-20 alphanumeric characters",
            severity=ValidationLevel.ERROR
        )
    
    def validate(self, value: str, context: ValidationContext) -> bool:
        return bool(ValidationPatterns.SKU.match(value))

class ValidSupplierNameRule(BusinessRule):
    """Rule to validate supplier name"""
    
    def __init__(self):
        super().__init__(
            name="valid_supplier_name",
            description="Supplier name must be 2-100 characters",
            severity=ValidationLevel.ERROR
        )
    
    def validate(self, value: str, context: ValidationContext) -> bool:
        return isinstance(value, str) and 2 <= len(value.strip()) <= 100

class ReasonableCostRule(BusinessRule):
    """Rule to validate reasonable cost"""
    
    def __init__(self, max_cost: Decimal = Decimal('100000')):
        self.max_cost = max_cost
        super().__init__(
            name="reasonable_cost",
            description=f"Cost should be reasonable (less than {max_cost})",
            severity=ValidationLevel.WARNING
        )
    
    def validate(self, value: Decimal, context: ValidationContext) -> bool:
        try:
            return Decimal(str(value)) <= self.max_cost
        except:
            return False

class ValidReliabilityScoreRule(BusinessRule):
    """Rule to validate reliability score"""
    
    def __init__(self):
        super().__init__(
            name="valid_reliability_score",
            description="Reliability score must be between 0 and 1",
            severity=ValidationLevel.ERROR
        )
    
    def validate(self, value: float, context: ValidationContext) -> bool:
        return ValidationPatterns.is_ratio(value)

class MinimumOrderQuantityRule(BusinessRule):
    """Rule to validate minimum order quantity"""
    
    def __init__(self):
        super().__init__(
            name="minimum_order_quantity",
            description="Order quantity must meet minimum requirements",
            severity=ValidationLevel.ERROR
        )
    
    def validate(self, value: Decimal, context: ValidationContext) -> bool:
        return ValidationPatterns.is_positive_decimal(value)

class ForecastDateRule(BusinessRule):
    """Rule to validate forecast date"""
    
    def __init__(self):
        super().__init__(
            name="forecast_date",
            description="Forecast date should be in the future",
            severity=ValidationLevel.WARNING
        )
    
    def validate(self, value: date, context: ValidationContext) -> bool:
        return ValidationPatterns.is_future_date(value)

class SafetyStockRule(BusinessRule):
    """Rule to validate safety stock levels"""
    
    def __init__(self):
        super().__init__(
            name="safety_stock",
            description="Safety stock must be non-negative",
            severity=ValidationLevel.ERROR
        )
    
    def validate(self, value: Decimal, context: ValidationContext) -> bool:
        return ValidationPatterns.is_non_negative_decimal(value)

# Validation Schemas

class MaterialValidationSchema(ValidatedModel):
    """Validation schema for Material entities"""
    
    id: str = Field(..., description="Material ID")
    name: str = Field(..., min_length=2, max_length=255, description="Material name")
    type: str = Field(..., description="Material type")
    description: Optional[str] = Field(None, max_length=1000, description="Material description")
    specifications: Dict[str, str] = Field(default_factory=dict, description="Material specifications")
    is_critical: bool = Field(default=False, description="Critical material flag")
    
    @validator('id')
    def validate_id(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("Material ID must be valid SKU format")
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Material name must be at least 2 characters")
        return v.strip()
    
    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['yarn', 'fabric', 'thread', 'accessory', 'trim']
        if v not in allowed_types:
            raise ValueError(f"Material type must be one of: {', '.join(allowed_types)}")
        return v
    
    @validator('specifications')
    def validate_specifications(cls, v):
        if v:
            for key, value in v.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("Specifications must be string key-value pairs")
                if len(key) > 50 or len(value) > 200:
                    raise ValueError("Specification keys/values are too long")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "id": [ValidSkuRule()],
            "name": [ValidSupplierNameRule()],
        }

class SupplierValidationSchema(ValidatedModel):
    """Validation schema for Supplier entities"""
    
    id: str = Field(..., description="Supplier ID")
    name: str = Field(..., min_length=2, max_length=255, description="Supplier name")
    contact_info: Optional[str] = Field(None, max_length=500, description="Contact information")
    lead_time_days: int = Field(..., ge=1, le=365, description="Lead time in days")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Reliability score")
    risk_level: str = Field(..., description="Risk level")
    is_active: bool = Field(default=True, description="Active status")
    
    @validator('id')
    def validate_id(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("Supplier ID must be valid SKU format")
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Supplier name must be at least 2 characters")
        return v.strip()
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        allowed_levels = ['low', 'medium', 'high']
        if v not in allowed_levels:
            raise ValueError(f"Risk level must be one of: {', '.join(allowed_levels)}")
        return v
    
    @validator('contact_info')
    def validate_contact_info(cls, v):
        if v and len(v.strip()) < 5:
            raise ValueError("Contact info must be at least 5 characters if provided")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "id": [ValidSkuRule()],
            "name": [ValidSupplierNameRule()],
            "lead_time_days": [ValidLeadTimeRule()],
            "reliability_score": [ValidReliabilityScoreRule()],
        }

class SupplierMaterialValidationSchema(ValidatedModel):
    """Validation schema for SupplierMaterial relationships"""
    
    supplier_id: str = Field(..., description="Supplier ID")
    material_id: str = Field(..., description="Material ID")
    cost_per_unit: Decimal = Field(..., gt=0, description="Cost per unit")
    currency: str = Field(default="USD", description="Currency code")
    moq_amount: Decimal = Field(..., gt=0, description="Minimum order quantity")
    moq_unit: str = Field(default="units", description="MOQ unit")
    lead_time_days: int = Field(..., ge=1, le=365, description="Lead time in days")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Reliability score")
    ordering_cost: Decimal = Field(default=Decimal('100'), ge=0, description="Ordering cost")
    holding_cost_rate: float = Field(default=0.25, ge=0, le=1, description="Holding cost rate")
    contract_qty_limit: Optional[Decimal] = Field(None, ge=0, description="Contract quantity limit")
    
    @validator('supplier_id', 'material_id')
    def validate_ids(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("IDs must be valid SKU format")
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        if not ValidationPatterns.CURRENCY.match(v):
            raise ValueError("Currency must be valid 3-letter code")
        return v
    
    @validator('moq_unit')
    def validate_moq_unit(cls, v):
        allowed_units = ['units', 'kg', 'lbs', 'meters', 'yards', 'liters']
        if v not in allowed_units:
            raise ValueError(f"MOQ unit must be one of: {', '.join(allowed_units)}")
        return v
    
    @validator('contract_qty_limit')
    def validate_contract_limit(cls, v, values):
        if v is not None and 'moq_amount' in values:
            if v < values['moq_amount']:
                raise ValueError("Contract quantity limit cannot be less than MOQ")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "supplier_id": [ValidSkuRule()],
            "material_id": [ValidSkuRule()],
            "cost_per_unit": [PositiveNumberRule("cost"), ReasonableCostRule()],
            "moq_amount": [PositiveNumberRule("minimum order quantity")],
            "lead_time_days": [ValidLeadTimeRule()],
            "reliability_score": [ValidReliabilityScoreRule()],
            "ordering_cost": [NonNegativeNumberRule("ordering cost")],
        }

class InventoryValidationSchema(ValidatedModel):
    """Validation schema for Inventory entities"""
    
    material_id: str = Field(..., description="Material ID")
    on_hand_qty: Decimal = Field(..., ge=0, description="On-hand quantity")
    unit: str = Field(default="units", description="Unit of measure")
    open_po_qty: Decimal = Field(default=0, ge=0, description="Open PO quantity")
    po_expected_date: Optional[date] = Field(None, description="Expected PO date")
    safety_stock: Decimal = Field(default=0, ge=0, description="Safety stock level")
    
    @validator('material_id')
    def validate_material_id(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("Material ID must be valid SKU format")
        return v
    
    @validator('unit')
    def validate_unit(cls, v):
        allowed_units = ['units', 'kg', 'lbs', 'meters', 'yards', 'liters', 'pieces']
        if v not in allowed_units:
            raise ValueError(f"Unit must be one of: {', '.join(allowed_units)}")
        return v
    
    @validator('po_expected_date')
    def validate_po_date(cls, v, values):
        if v is not None:
            if v < date.today():
                raise ValueError("PO expected date cannot be in the past")
            if 'open_po_qty' in values and values['open_po_qty'] == 0:
                raise ValueError("PO expected date set but no open PO quantity")
        return v
    
    @validator('safety_stock')
    def validate_safety_stock(cls, v, values):
        if v > 0 and 'on_hand_qty' in values:
            if v > values['on_hand_qty'] * 2:
                raise ValueError("Safety stock seems unusually high compared to on-hand quantity")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "material_id": [ValidSkuRule()],
            "on_hand_qty": [NonNegativeNumberRule("on-hand quantity")],
            "open_po_qty": [NonNegativeNumberRule("open PO quantity")],
            "safety_stock": [SafetyStockRule()],
        }

class BOMValidationSchema(ValidatedModel):
    """Validation schema for BOM entities"""
    
    sku_id: str = Field(..., description="SKU ID")
    material_id: str = Field(..., description="Material ID")
    qty_per_unit: Decimal = Field(..., gt=0, description="Quantity per unit")
    unit: str = Field(default="units", description="Unit of measure")
    waste_percentage: Optional[Decimal] = Field(None, ge=0, le=50, description="Waste percentage")
    efficiency_factor: Optional[Decimal] = Field(None, gt=0, le=2, description="Efficiency factor")
    
    @validator('sku_id', 'material_id')
    def validate_ids(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("IDs must be valid SKU format")
        return v
    
    @validator('unit')
    def validate_unit(cls, v):
        allowed_units = ['units', 'kg', 'lbs', 'meters', 'yards', 'liters', 'pieces']
        if v not in allowed_units:
            raise ValueError(f"Unit must be one of: {', '.join(allowed_units)}")
        return v
    
    @validator('waste_percentage')
    def validate_waste_percentage(cls, v):
        if v is not None and v > 25:
            raise ValueError("Waste percentage over 25% seems unusually high")
        return v
    
    @validator('efficiency_factor')
    def validate_efficiency_factor(cls, v):
        if v is not None and v < 0.5:
            raise ValueError("Efficiency factor below 0.5 seems unusually low")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "sku_id": [ValidSkuRule()],
            "material_id": [ValidSkuRule()],
            "qty_per_unit": [PositiveNumberRule("quantity per unit")],
            "waste_percentage": [NonNegativeNumberRule("waste percentage")],
            "efficiency_factor": [PositiveNumberRule("efficiency factor")],
        }

class ForecastValidationSchema(ValidatedModel):
    """Validation schema for Forecast entities"""
    
    sku_id: str = Field(..., description="SKU ID")
    forecast_qty: Decimal = Field(..., gt=0, description="Forecast quantity")
    unit: str = Field(default="units", description="Unit of measure")
    forecast_date: date = Field(..., description="Forecast date")
    source: str = Field(..., description="Forecast source")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    notes: Optional[str] = Field(None, max_length=1000, description="Forecast notes")
    model_used: Optional[str] = Field(None, max_length=50, description="ML model used")
    upper_bound: Optional[Decimal] = Field(None, ge=0, description="Upper confidence bound")
    lower_bound: Optional[Decimal] = Field(None, ge=0, description="Lower confidence bound")
    
    @validator('sku_id')
    def validate_sku_id(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("SKU ID must be valid SKU format")
        return v
    
    @validator('unit')
    def validate_unit(cls, v):
        allowed_units = ['units', 'kg', 'lbs', 'meters', 'yards', 'liters', 'pieces']
        if v not in allowed_units:
            raise ValueError(f"Unit must be one of: {', '.join(allowed_units)}")
        return v
    
    @validator('source')
    def validate_source(cls, v):
        allowed_sources = ['sales_order', 'prod_plan', 'projection', 'sales_history']
        if v not in allowed_sources:
            raise ValueError(f"Source must be one of: {', '.join(allowed_sources)}")
        return v
    
    @validator('forecast_date')
    def validate_forecast_date(cls, v):
        if v < date.today() - timedelta(days=30):
            raise ValueError("Forecast date cannot be more than 30 days in the past")
        return v
    
    @validator('upper_bound')
    def validate_upper_bound(cls, v, values):
        if v is not None and 'forecast_qty' in values:
            if v < values['forecast_qty']:
                raise ValueError("Upper bound cannot be less than forecast quantity")
        return v
    
    @validator('lower_bound')
    def validate_lower_bound(cls, v, values):
        if v is not None and 'forecast_qty' in values:
            if v > values['forecast_qty']:
                raise ValueError("Lower bound cannot be greater than forecast quantity")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "sku_id": [ValidSkuRule()],
            "forecast_qty": [PositiveNumberRule("forecast quantity"), ReasonableQuantityRule()],
            "confidence_score": [ValidReliabilityScoreRule()],
            "forecast_date": [ForecastDateRule()],
        }

class ProcurementRecommendationValidationSchema(ValidatedModel):
    """Validation schema for ProcurementRecommendation entities"""
    
    material_id: str = Field(..., description="Material ID")
    supplier_id: str = Field(..., description="Supplier ID")
    recommended_qty: Decimal = Field(..., gt=0, description="Recommended quantity")
    unit: str = Field(default="units", description="Unit of measure")
    recommendation_date: date = Field(..., description="Recommendation date")
    required_by_date: date = Field(..., description="Required by date")
    urgency_level: str = Field(..., description="Urgency level")
    estimated_cost: Decimal = Field(..., gt=0, description="Estimated cost")
    currency: str = Field(default="USD", description="Currency code")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reason: Optional[str] = Field(None, max_length=1000, description="Recommendation reason")
    status: str = Field(default="pending", description="Recommendation status")
    
    @validator('material_id', 'supplier_id')
    def validate_ids(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("IDs must be valid SKU format")
        return v
    
    @validator('unit')
    def validate_unit(cls, v):
        allowed_units = ['units', 'kg', 'lbs', 'meters', 'yards', 'liters', 'pieces']
        if v not in allowed_units:
            raise ValueError(f"Unit must be one of: {', '.join(allowed_units)}")
        return v
    
    @validator('urgency_level')
    def validate_urgency_level(cls, v):
        allowed_levels = ['low', 'medium', 'high', 'critical']
        if v not in allowed_levels:
            raise ValueError(f"Urgency level must be one of: {', '.join(allowed_levels)}")
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        if not ValidationPatterns.CURRENCY.match(v):
            raise ValueError("Currency must be valid 3-letter code")
        return v
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['pending', 'approved', 'rejected', 'ordered']
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v
    
    @validator('required_by_date')
    def validate_required_by_date(cls, v, values):
        if 'recommendation_date' in values and v < values['recommendation_date']:
            raise ValueError("Required by date cannot be before recommendation date")
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "material_id": [ValidSkuRule()],
            "supplier_id": [ValidSkuRule()],
            "recommended_qty": [PositiveNumberRule("recommended quantity"), ReasonableQuantityRule()],
            "estimated_cost": [PositiveNumberRule("estimated cost"), ReasonableCostRule()],
            "confidence_score": [ValidReliabilityScoreRule()],
        }

class UserValidationSchema(ValidatedModel):
    """Validation schema for User entities"""
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")
    role: str = Field(..., description="User role")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(default=True, description="Active status")
    department: Optional[str] = Field(None, max_length=100, description="Department")
    position: Optional[str] = Field(None, max_length=100, description="Position")
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v
    
    @validator('email')
    def validate_email(cls, v):
        if not ValidationPatterns.EMAIL.match(v):
            raise ValueError("Invalid email format")
        return v
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['admin', 'manager', 'user', 'viewer']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v
    
    @validator('permissions')
    def validate_permissions(cls, v):
        valid_permissions = [
            'view_materials', 'edit_materials', 'delete_materials',
            'view_suppliers', 'edit_suppliers', 'delete_suppliers',
            'view_inventory', 'edit_inventory',
            'view_bom', 'edit_bom', 'delete_bom',
            'view_forecasts', 'edit_forecasts', 'delete_forecasts',
            'execute_planning', 'view_planning_results',
            'view_analytics', 'export_data',
            'manage_users', 'manage_permissions', 'system_admin'
        ]
        
        for permission in v:
            if permission not in valid_permissions:
                raise ValueError(f"Invalid permission: {permission}")
        
        return v
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "username": [ValidSupplierNameRule()],
            "full_name": [ValidSupplierNameRule()],
        }