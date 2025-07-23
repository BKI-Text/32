"""BOM (Bill of Materials) Model for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import Column, String, ForeignKey, Numeric, Index
from sqlalchemy.orm import relationship
from decimal import Decimal

from .base import BaseModel

class BOMModel(BaseModel):
    """Bill of Materials database model"""
    __tablename__ = "boms"
    
    # Foreign keys
    sku_id = Column(String, nullable=False, index=True)
    material_id = Column(String, ForeignKey("materials.id"), nullable=False, index=True)
    
    # BOM details
    qty_per_unit = Column(Numeric(12, 4), nullable=False)
    unit = Column(String(20), nullable=False, default="units")
    
    # Optional fields
    waste_percentage = Column(Numeric(5, 2), nullable=True, default=0)  # Waste factor
    efficiency_factor = Column(Numeric(5, 4), nullable=True, default=1.0)  # Efficiency factor
    
    # Relationships
    material = relationship("MaterialModel", back_populates="bom_entries")
    
    # Indexes
    __table_args__ = (
        Index('idx_bom_sku_material', 'sku_id', 'material_id'),
        Index('idx_bom_sku', 'sku_id'),
        Index('idx_bom_material', 'material_id'),
    )
    
    def __repr__(self):
        return f"<BOMModel(sku_id={self.sku_id}, material_id={self.material_id}, qty={self.qty_per_unit})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import BOM
        from ...core.domain.value_objects import SkuId, MaterialId, Quantity
        
        return BOM(
            sku_id=SkuId(value=self.sku_id),
            material_id=MaterialId(value=self.material_id),
            qty_per_unit=Quantity(amount=self.qty_per_unit, unit=self.unit),
            unit=self.unit
        )
    
    @classmethod
    def from_domain_entity(cls, bom):
        """Create from domain entity"""
        return cls(
            sku_id=bom.sku_id.value,
            material_id=bom.material_id.value,
            qty_per_unit=bom.qty_per_unit.amount,
            unit=bom.unit
        )
    
    def calculate_material_requirement(self, sku_quantity: Decimal) -> Decimal:
        """Calculate material requirement for given SKU quantity"""
        base_requirement = self.qty_per_unit * sku_quantity
        
        # Apply efficiency factor
        if self.efficiency_factor:
            base_requirement = base_requirement / self.efficiency_factor
        
        # Apply waste percentage
        if self.waste_percentage:
            waste_factor = 1 + (self.waste_percentage / 100)
            base_requirement = base_requirement * waste_factor
        
        return base_requirement
    
    def get_effective_qty_per_unit(self) -> Decimal:
        """Get effective quantity per unit including waste and efficiency factors"""
        effective_qty = self.qty_per_unit
        
        # Apply efficiency factor
        if self.efficiency_factor and self.efficiency_factor > 0:
            effective_qty = effective_qty / self.efficiency_factor
        
        # Apply waste percentage
        if self.waste_percentage:
            waste_factor = 1 + (self.waste_percentage / 100)
            effective_qty = effective_qty * waste_factor
        
        return effective_qty
    
    def update_qty_per_unit(self, new_qty: Decimal):
        """Update quantity per unit"""
        self.qty_per_unit = new_qty
    
    def update_waste_percentage(self, new_waste: Decimal):
        """Update waste percentage"""
        self.waste_percentage = max(Decimal('0'), new_waste)
    
    def update_efficiency_factor(self, new_efficiency: Decimal):
        """Update efficiency factor"""
        self.efficiency_factor = max(Decimal('0.01'), new_efficiency)
    
    def is_critical_component(self) -> bool:
        """Check if this BOM entry is for a critical material"""
        return self.material and self.material.is_critical
    
    def get_material_name(self) -> str:
        """Get the name of the material"""
        return self.material.name if self.material else "Unknown"
    
    def get_material_type(self) -> str:
        """Get the type of the material"""
        return self.material.type if self.material else "Unknown"
    
    def calculate_cost_per_unit(self, material_cost: Decimal) -> Decimal:
        """Calculate cost per unit for this BOM component"""
        effective_qty = self.get_effective_qty_per_unit()
        return effective_qty * material_cost
    
    def get_usage_percentage(self, total_sku_qty: Decimal) -> float:
        """Calculate what percentage of total SKU quantity this component represents"""
        if total_sku_qty == 0:
            return 0.0
        
        component_qty = self.qty_per_unit * total_sku_qty
        return float(component_qty / total_sku_qty) * 100
    
    def is_substitute_available(self) -> bool:
        """Check if substitute materials are available (placeholder)"""
        # This would check for substitute materials in a real implementation
        return False
    
    def get_lead_time_impact(self) -> int:
        """Get the lead time impact of this component"""
        if self.material and self.material.supplier_materials:
            # Return the minimum lead time among all suppliers
            lead_times = [sm.lead_time_days for sm in self.material.supplier_materials]
            return min(lead_times) if lead_times else 30
        return 30
    
    def validate_bom_entry(self) -> list:
        """Validate BOM entry and return list of issues"""
        issues = []
        
        if self.qty_per_unit <= 0:
            issues.append("Quantity per unit must be positive")
        
        if self.waste_percentage and self.waste_percentage < 0:
            issues.append("Waste percentage cannot be negative")
        
        if self.efficiency_factor and self.efficiency_factor <= 0:
            issues.append("Efficiency factor must be positive")
        
        if not self.material:
            issues.append("Material reference is missing")
        
        if not self.sku_id:
            issues.append("SKU ID is required")
        
        return issues
    
    def get_bom_summary(self) -> dict:
        """Get summary information about this BOM entry"""
        return {
            "sku_id": self.sku_id,
            "material_id": self.material_id,
            "material_name": self.get_material_name(),
            "material_type": self.get_material_type(),
            "qty_per_unit": float(self.qty_per_unit),
            "unit": self.unit,
            "effective_qty_per_unit": float(self.get_effective_qty_per_unit()),
            "waste_percentage": float(self.waste_percentage) if self.waste_percentage else 0,
            "efficiency_factor": float(self.efficiency_factor) if self.efficiency_factor else 1.0,
            "is_critical": self.is_critical_component(),
            "lead_time_days": self.get_lead_time_impact()
        }