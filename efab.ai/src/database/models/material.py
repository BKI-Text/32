"""Material Model for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import Column, String, Boolean, JSON, Index
from sqlalchemy.orm import relationship
from typing import Dict, Any

from .base import BaseModel

class MaterialModel(BaseModel):
    """Material database model"""
    __tablename__ = "materials"
    
    # Core material fields
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)  # yarn, fabric, thread, accessory, trim
    description = Column(String(1000), nullable=True)
    specifications = Column(JSON, nullable=True, default=dict)
    is_critical = Column(Boolean, default=False, nullable=False, index=True)
    
    # Relationships
    supplier_materials = relationship("SupplierMaterialModel", back_populates="material", cascade="all, delete-orphan")
    inventory_records = relationship("InventoryModel", back_populates="material", cascade="all, delete-orphan")
    bom_entries = relationship("BOMModel", back_populates="material", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_material_name_type', 'name', 'type'),
        Index('idx_material_critical', 'is_critical'),
    )
    
    def __repr__(self):
        return f"<MaterialModel(id={self.id}, name={self.name}, type={self.type})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import Material, MaterialType
        from ...core.domain.value_objects import MaterialId
        
        return Material(
            id=MaterialId(value=self.id),
            name=self.name,
            type=MaterialType(self.type),
            description=self.description,
            specifications=self.specifications or {},
            is_critical=self.is_critical,
            created_at=self.created_at,
            updated_at=self.updated_at
        )
    
    @classmethod
    def from_domain_entity(cls, material):
        """Create from domain entity"""
        return cls(
            id=material.id.value,
            name=material.name,
            type=material.type.value,
            description=material.description,
            specifications=material.specifications,
            is_critical=material.is_critical,
            created_at=material.created_at,
            updated_at=material.updated_at
        )
    
    def get_specifications(self) -> Dict[str, Any]:
        """Get material specifications"""
        return self.specifications or {}
    
    def update_specifications(self, new_specs: Dict[str, Any]):
        """Update material specifications"""
        current_specs = self.specifications or {}
        current_specs.update(new_specs)
        self.specifications = current_specs
    
    def add_specification(self, key: str, value: Any):
        """Add single specification"""
        if self.specifications is None:
            self.specifications = {}
        self.specifications[key] = value
    
    def remove_specification(self, key: str):
        """Remove specification"""
        if self.specifications and key in self.specifications:
            del self.specifications[key]
    
    def is_yarn(self) -> bool:
        """Check if material is yarn"""
        return self.type == "yarn"
    
    def is_fabric(self) -> bool:
        """Check if material is fabric"""
        return self.type == "fabric"
    
    def is_thread(self) -> bool:
        """Check if material is thread"""
        return self.type == "thread"
    
    def get_supplier_count(self) -> int:
        """Get number of suppliers for this material"""
        return len(self.supplier_materials)
    
    def get_current_inventory(self):
        """Get current inventory for this material"""
        if self.inventory_records:
            return self.inventory_records[0]  # Assuming one inventory record per material
        return None
    
    def has_active_suppliers(self) -> bool:
        """Check if material has active suppliers"""
        return any(sm.supplier.is_active for sm in self.supplier_materials if sm.supplier)
    
    def get_lowest_cost_supplier(self):
        """Get supplier with lowest cost for this material"""
        if not self.supplier_materials:
            return None
        
        active_suppliers = [sm for sm in self.supplier_materials if sm.supplier and sm.supplier.is_active]
        if not active_suppliers:
            return None
        
        return min(active_suppliers, key=lambda sm: sm.cost_per_unit)
    
    def get_highest_reliability_supplier(self):
        """Get supplier with highest reliability for this material"""
        if not self.supplier_materials:
            return None
        
        active_suppliers = [sm for sm in self.supplier_materials if sm.supplier and sm.supplier.is_active]
        if not active_suppliers:
            return None
        
        return max(active_suppliers, key=lambda sm: sm.reliability_score)