"""Inventory Model for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import Column, String, ForeignKey, Numeric, Date, Index
from sqlalchemy.orm import relationship
from decimal import Decimal
from datetime import date

from .base import BaseModel

class InventoryModel(BaseModel):
    """Inventory database model"""
    __tablename__ = "inventory"
    
    # Foreign key
    material_id = Column(String, ForeignKey("materials.id"), nullable=False, index=True)
    
    # Inventory quantities
    on_hand_qty = Column(Numeric(12, 2), nullable=False, default=0)
    unit = Column(String(20), nullable=False, default="units")
    open_po_qty = Column(Numeric(12, 2), nullable=False, default=0)
    po_expected_date = Column(Date, nullable=True)
    safety_stock = Column(Numeric(12, 2), nullable=False, default=0)
    
    # Relationships
    material = relationship("MaterialModel", back_populates="inventory_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_inventory_material', 'material_id'),
        Index('idx_inventory_on_hand', 'material_id', 'on_hand_qty'),
        Index('idx_inventory_po_date', 'po_expected_date'),
    )
    
    def __repr__(self):
        return f"<InventoryModel(material_id={self.material_id}, on_hand={self.on_hand_qty})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import Inventory
        from ...core.domain.value_objects import MaterialId, Quantity
        
        return Inventory(
            material_id=MaterialId(value=self.material_id),
            on_hand_qty=Quantity(amount=self.on_hand_qty, unit=self.unit),
            open_po_qty=Quantity(amount=self.open_po_qty, unit=self.unit),
            po_expected_date=self.po_expected_date,
            safety_stock=Quantity(amount=self.safety_stock, unit=self.unit),
            last_updated=self.updated_at
        )
    
    @classmethod
    def from_domain_entity(cls, inventory):
        """Create from domain entity"""
        return cls(
            material_id=inventory.material_id.value,
            on_hand_qty=inventory.on_hand_qty.amount,
            unit=inventory.on_hand_qty.unit,
            open_po_qty=inventory.open_po_qty.amount,
            po_expected_date=inventory.po_expected_date,
            safety_stock=inventory.safety_stock.amount
        )
    
    def get_available_qty(self) -> Decimal:
        """Calculate available quantity (on hand + open PO)"""
        return self.on_hand_qty + self.open_po_qty
    
    def get_net_available_qty(self) -> Decimal:
        """Calculate net available quantity (available - safety stock)"""
        return self.get_available_qty() - self.safety_stock
    
    def is_low_stock(self) -> bool:
        """Check if inventory is below safety stock"""
        return self.on_hand_qty < self.safety_stock
    
    def is_out_of_stock(self) -> bool:
        """Check if inventory is out of stock"""
        return self.on_hand_qty <= 0
    
    def has_pending_po(self) -> bool:
        """Check if there are pending purchase orders"""
        return self.open_po_qty > 0
    
    def days_until_po_arrival(self) -> int:
        """Calculate days until PO arrival"""
        if not self.po_expected_date:
            return 0
        
        today = date.today()
        if self.po_expected_date <= today:
            return 0
        
        return (self.po_expected_date - today).days
    
    def update_on_hand_qty(self, new_qty: Decimal):
        """Update on-hand quantity"""
        self.on_hand_qty = max(Decimal('0'), new_qty)
    
    def adjust_inventory(self, adjustment: Decimal, reason: str = None):
        """Adjust inventory quantity"""
        new_qty = self.on_hand_qty + adjustment
        self.update_on_hand_qty(new_qty)
    
    def consume_inventory(self, quantity: Decimal) -> bool:
        """Consume inventory quantity"""
        if self.on_hand_qty >= quantity:
            self.on_hand_qty -= quantity
            return True
        return False
    
    def receive_inventory(self, quantity: Decimal):
        """Receive inventory quantity"""
        self.on_hand_qty += quantity
    
    def update_po_info(self, po_qty: Decimal, expected_date: date = None):
        """Update purchase order information"""
        self.open_po_qty = po_qty
        self.po_expected_date = expected_date
    
    def clear_po_info(self):
        """Clear purchase order information"""
        self.open_po_qty = Decimal('0')
        self.po_expected_date = None
    
    def update_safety_stock(self, new_safety_stock: Decimal):
        """Update safety stock level"""
        self.safety_stock = max(Decimal('0'), new_safety_stock)
    
    def get_stock_status(self) -> str:
        """Get stock status description"""
        if self.is_out_of_stock():
            return "out_of_stock"
        elif self.is_low_stock():
            return "low_stock"
        elif self.on_hand_qty > (self.safety_stock * 2):
            return "excess_stock"
        else:
            return "normal"
    
    def get_reorder_point(self, lead_time_days: int, daily_usage: Decimal) -> Decimal:
        """Calculate reorder point"""
        lead_time_demand = daily_usage * Decimal(str(lead_time_days))
        return lead_time_demand + self.safety_stock
    
    def needs_reorder(self, lead_time_days: int, daily_usage: Decimal) -> bool:
        """Check if inventory needs reordering"""
        reorder_point = self.get_reorder_point(lead_time_days, daily_usage)
        return self.on_hand_qty <= reorder_point
    
    def get_inventory_turnover(self, annual_usage: Decimal) -> Decimal:
        """Calculate inventory turnover ratio"""
        if self.on_hand_qty == 0:
            return Decimal('0')
        
        return annual_usage / self.on_hand_qty
    
    def get_carrying_cost(self, unit_cost: Decimal, carrying_cost_rate: float = 0.25) -> Decimal:
        """Calculate carrying cost"""
        inventory_value = self.on_hand_qty * unit_cost
        return inventory_value * Decimal(str(carrying_cost_rate))
    
    def get_stockout_risk(self, daily_usage: Decimal, lead_time_days: int) -> float:
        """Calculate stockout risk based on current inventory and usage"""
        if daily_usage <= 0:
            return 0.0
        
        days_of_supply = float(self.on_hand_qty) / float(daily_usage)
        
        if days_of_supply <= 0:
            return 1.0  # Already out of stock
        elif days_of_supply <= lead_time_days:
            return 0.8  # High risk
        elif days_of_supply <= lead_time_days * 1.5:
            return 0.5  # Medium risk
        else:
            return 0.1  # Low risk