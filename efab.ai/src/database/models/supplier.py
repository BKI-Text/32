"""Supplier Models for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import Column, String, Boolean, Float, Integer, ForeignKey, Numeric, Index
from sqlalchemy.orm import relationship
from decimal import Decimal

from .base import BaseModel

class SupplierModel(BaseModel):
    """Supplier database model"""
    __tablename__ = "suppliers"
    
    # Core supplier fields
    name = Column(String(255), nullable=False, index=True)
    contact_info = Column(String(500), nullable=True)
    lead_time_days = Column(Integer, nullable=False, default=30)
    reliability_score = Column(Float, nullable=False, default=0.5)
    risk_level = Column(String(20), nullable=False, default="medium", index=True)  # low, medium, high
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Relationships
    supplier_materials = relationship("SupplierMaterialModel", back_populates="supplier", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_supplier_name', 'name'),
        Index('idx_supplier_active_risk', 'is_active', 'risk_level'),
        Index('idx_supplier_reliability', 'reliability_score'),
    )
    
    def __repr__(self):
        return f"<SupplierModel(id={self.id}, name={self.name}, risk_level={self.risk_level})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import Supplier, RiskLevel
        from ...core.domain.value_objects import SupplierId, LeadTime
        
        return Supplier(
            id=SupplierId(value=self.id),
            name=self.name,
            contact_info=self.contact_info,
            lead_time=LeadTime(days=self.lead_time_days),
            reliability_score=self.reliability_score,
            risk_level=RiskLevel(self.risk_level),
            is_active=self.is_active,
            created_at=self.created_at
        )
    
    @classmethod
    def from_domain_entity(cls, supplier):
        """Create from domain entity"""
        return cls(
            id=supplier.id.value,
            name=supplier.name,
            contact_info=supplier.contact_info,
            lead_time_days=supplier.lead_time.days,
            reliability_score=supplier.reliability_score,
            risk_level=supplier.risk_level.value,
            is_active=supplier.is_active,
            created_at=supplier.created_at
        )
    
    def get_material_count(self) -> int:
        """Get number of materials supplied by this supplier"""
        return len(self.supplier_materials)
    
    def get_active_materials(self):
        """Get active materials for this supplier"""
        return [sm for sm in self.supplier_materials if sm.material]
    
    def supplies_material(self, material_id: str) -> bool:
        """Check if supplier supplies a specific material"""
        return any(sm.material_id == material_id for sm in self.supplier_materials)
    
    def get_material_cost(self, material_id: str) -> Decimal:
        """Get cost per unit for a specific material"""
        for sm in self.supplier_materials:
            if sm.material_id == material_id:
                return sm.cost_per_unit
        return Decimal('0')
    
    def is_high_risk(self) -> bool:
        """Check if supplier is high risk"""
        return self.risk_level == "high"
    
    def is_low_risk(self) -> bool:
        """Check if supplier is low risk"""
        return self.risk_level == "low"
    
    def is_reliable(self, threshold: float = 0.8) -> bool:
        """Check if supplier is reliable above threshold"""
        return self.reliability_score >= threshold
    
    def deactivate(self):
        """Deactivate supplier"""
        self.is_active = False
    
    def activate(self):
        """Activate supplier"""
        self.is_active = True
    
    def update_risk_level(self, new_risk_level: str):
        """Update supplier risk level"""
        if new_risk_level in ["low", "medium", "high"]:
            self.risk_level = new_risk_level
    
    def update_reliability_score(self, new_score: float):
        """Update supplier reliability score"""
        if 0 <= new_score <= 1:
            self.reliability_score = new_score

class SupplierMaterialModel(BaseModel):
    """Supplier-Material relationship model"""
    __tablename__ = "supplier_materials"
    
    # Foreign keys
    supplier_id = Column(String, ForeignKey("suppliers.id"), nullable=False, index=True)
    material_id = Column(String, ForeignKey("materials.id"), nullable=False, index=True)
    
    # Cost and procurement info
    cost_per_unit = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    moq_amount = Column(Numeric(10, 2), nullable=False)  # Minimum order quantity
    moq_unit = Column(String(20), nullable=False, default="units")
    lead_time_days = Column(Integer, nullable=False, default=30)
    
    # Quality and reliability
    reliability_score = Column(Float, nullable=False, default=0.5)
    ordering_cost = Column(Numeric(10, 2), nullable=False, default=100.0)
    holding_cost_rate = Column(Float, nullable=False, default=0.25)
    
    # Optional contract constraints
    contract_qty_limit = Column(Numeric(10, 2), nullable=True)
    contract_qty_unit = Column(String(20), nullable=True)
    
    # Relationships
    supplier = relationship("SupplierModel", back_populates="supplier_materials")
    material = relationship("MaterialModel", back_populates="supplier_materials")
    
    # Indexes
    __table_args__ = (
        Index('idx_supplier_material', 'supplier_id', 'material_id'),
        Index('idx_supplier_material_cost', 'supplier_id', 'material_id', 'cost_per_unit'),
        Index('idx_material_supplier_reliability', 'material_id', 'reliability_score'),
    )
    
    def __repr__(self):
        return f"<SupplierMaterialModel(supplier_id={self.supplier_id}, material_id={self.material_id}, cost={self.cost_per_unit})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import SupplierMaterial
        from ...core.domain.value_objects import SupplierId, MaterialId, Money, Quantity, LeadTime
        
        return SupplierMaterial(
            supplier_id=SupplierId(value=self.supplier_id),
            material_id=MaterialId(value=self.material_id),
            cost_per_unit=Money(amount=self.cost_per_unit, currency=self.currency),
            moq=Quantity(amount=self.moq_amount, unit=self.moq_unit),
            lead_time=LeadTime(days=self.lead_time_days),
            reliability_score=self.reliability_score,
            ordering_cost=Money(amount=self.ordering_cost, currency=self.currency),
            holding_cost_rate=self.holding_cost_rate,
            contract_qty_limit=Quantity(amount=self.contract_qty_limit, unit=self.contract_qty_unit) if self.contract_qty_limit else None
        )
    
    @classmethod
    def from_domain_entity(cls, supplier_material):
        """Create from domain entity"""
        return cls(
            supplier_id=supplier_material.supplier_id.value,
            material_id=supplier_material.material_id.value,
            cost_per_unit=supplier_material.cost_per_unit.amount,
            currency=supplier_material.cost_per_unit.currency,
            moq_amount=supplier_material.moq.amount,
            moq_unit=supplier_material.moq.unit,
            lead_time_days=supplier_material.lead_time.days,
            reliability_score=supplier_material.reliability_score,
            ordering_cost=supplier_material.ordering_cost.amount,
            holding_cost_rate=supplier_material.holding_cost_rate,
            contract_qty_limit=supplier_material.contract_qty_limit.amount if supplier_material.contract_qty_limit else None,
            contract_qty_unit=supplier_material.contract_qty_limit.unit if supplier_material.contract_qty_limit else None
        )
    
    def get_total_cost(self, quantity: Decimal) -> Decimal:
        """Calculate total cost for a given quantity"""
        return self.cost_per_unit * quantity
    
    def meets_moq(self, quantity: Decimal) -> bool:
        """Check if quantity meets minimum order quantity"""
        return quantity >= self.moq_amount
    
    def is_within_contract_limit(self, quantity: Decimal) -> bool:
        """Check if quantity is within contract limits"""
        if self.contract_qty_limit is None:
            return True
        return quantity <= self.contract_qty_limit
    
    def can_fulfill_order(self, quantity: Decimal) -> bool:
        """Check if supplier can fulfill order"""
        return self.meets_moq(quantity) and self.is_within_contract_limit(quantity)
    
    def get_eoq_quantity(self, annual_demand: Decimal) -> Decimal:
        """Calculate Economic Order Quantity"""
        from decimal import Decimal
        import math
        
        # EOQ = sqrt(2 * D * S / H)
        # D = annual demand
        # S = ordering cost
        # H = holding cost per unit per year
        
        if annual_demand <= 0:
            return Decimal('0')
        
        holding_cost_per_unit = self.cost_per_unit * Decimal(str(self.holding_cost_rate))
        
        eoq_squared = (2 * annual_demand * self.ordering_cost) / holding_cost_per_unit
        eoq = Decimal(str(math.sqrt(float(eoq_squared))))
        
        # Ensure EOQ meets MOQ
        return max(eoq, self.moq_amount)
    
    def update_cost(self, new_cost: Decimal):
        """Update cost per unit"""
        self.cost_per_unit = new_cost
    
    def update_moq(self, new_moq: Decimal):
        """Update minimum order quantity"""
        self.moq_amount = new_moq
    
    def update_reliability(self, new_reliability: float):
        """Update reliability score"""
        if 0 <= new_reliability <= 1:
            self.reliability_score = new_reliability