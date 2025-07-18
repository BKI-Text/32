"""Procurement Recommendation Model for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import Column, String, Numeric, Date, Text, Index
from decimal import Decimal
from datetime import date

from .base import BaseModel

class ProcurementRecommendationModel(BaseModel):
    """Procurement recommendation database model"""
    __tablename__ = "procurement_recommendations"
    
    # Core recommendation fields
    material_id = Column(String, nullable=False, index=True)
    supplier_id = Column(String, nullable=False, index=True)
    recommended_qty = Column(Numeric(12, 2), nullable=False)
    unit = Column(String(20), nullable=False, default="units")
    
    # Timing and urgency
    recommendation_date = Column(Date, nullable=False, default=date.today)
    required_by_date = Column(Date, nullable=False)
    urgency_level = Column(String(20), nullable=False, default="medium", index=True)  # low, medium, high, critical
    
    # Cost and business impact
    estimated_cost = Column(Numeric(12, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    cost_savings = Column(Numeric(12, 2), nullable=True, default=0)
    
    # Reasoning and context
    reason = Column(Text, nullable=True)
    confidence_score = Column(Numeric(3, 2), nullable=False, default=0.5)
    
    # Processing status
    status = Column(String(20), nullable=False, default="pending", index=True)  # pending, approved, rejected, ordered
    approved_by = Column(String, nullable=True)
    approved_date = Column(Date, nullable=True)
    
    # Risk assessment
    risk_level = Column(String(20), nullable=False, default="medium", index=True)  # low, medium, high
    risk_factors = Column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_procurement_material_urgency', 'material_id', 'urgency_level'),
        Index('idx_procurement_supplier_status', 'supplier_id', 'status'),
        Index('idx_procurement_date_status', 'recommendation_date', 'status'),
        Index('idx_procurement_required_by', 'required_by_date'),
        Index('idx_procurement_confidence', 'confidence_score'),
    )
    
    def __repr__(self):
        return f"<ProcurementRecommendationModel(material_id={self.material_id}, supplier_id={self.supplier_id}, qty={self.recommended_qty})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import ProcurementRecommendation, UrgencyLevel, RecommendationStatus
        from ...core.domain.value_objects import MaterialId, SupplierId, Quantity, Money
        
        return ProcurementRecommendation(
            material_id=MaterialId(value=self.material_id),
            supplier_id=SupplierId(value=self.supplier_id),
            recommended_qty=Quantity(amount=self.recommended_qty, unit=self.unit),
            recommendation_date=self.recommendation_date,
            required_by_date=self.required_by_date,
            urgency_level=UrgencyLevel(self.urgency_level),
            estimated_cost=Money(amount=self.estimated_cost, currency=self.currency),
            cost_savings=Money(amount=self.cost_savings or 0, currency=self.currency),
            reason=self.reason,
            confidence_score=self.confidence_score,
            status=RecommendationStatus(self.status),
            created_at=self.created_at
        )
    
    @classmethod
    def from_domain_entity(cls, recommendation):
        """Create from domain entity"""
        return cls(
            material_id=recommendation.material_id.value,
            supplier_id=recommendation.supplier_id.value,
            recommended_qty=recommendation.recommended_qty.amount,
            unit=recommendation.recommended_qty.unit,
            recommendation_date=recommendation.recommendation_date,
            required_by_date=recommendation.required_by_date,
            urgency_level=recommendation.urgency_level.value,
            estimated_cost=recommendation.estimated_cost.amount,
            currency=recommendation.estimated_cost.currency,
            cost_savings=recommendation.cost_savings.amount if recommendation.cost_savings else 0,
            reason=recommendation.reason,
            confidence_score=recommendation.confidence_score,
            status=recommendation.status.value,
            created_at=recommendation.created_at
        )
    
    def is_high_urgency(self) -> bool:
        """Check if recommendation is high urgency"""
        return self.urgency_level in ["high", "critical"]
    
    def is_critical(self) -> bool:
        """Check if recommendation is critical"""
        return self.urgency_level == "critical"
    
    def is_pending(self) -> bool:
        """Check if recommendation is pending"""
        return self.status == "pending"
    
    def is_approved(self) -> bool:
        """Check if recommendation is approved"""
        return self.status == "approved"
    
    def is_rejected(self) -> bool:
        """Check if recommendation is rejected"""
        return self.status == "rejected"
    
    def is_ordered(self) -> bool:
        """Check if recommendation has been ordered"""
        return self.status == "ordered"
    
    def days_until_required(self) -> int:
        """Calculate days until required date"""
        today = date.today()
        if self.required_by_date <= today:
            return 0
        return (self.required_by_date - today).days
    
    def is_overdue(self) -> bool:
        """Check if recommendation is overdue"""
        return self.required_by_date < date.today()
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if recommendation has high confidence"""
        return float(self.confidence_score) >= threshold
    
    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        """Check if recommendation has low confidence"""
        return float(self.confidence_score) < threshold
    
    def approve(self, approved_by: str):
        """Approve recommendation"""
        self.status = "approved"
        self.approved_by = approved_by
        self.approved_date = date.today()
    
    def reject(self, approved_by: str):
        """Reject recommendation"""
        self.status = "rejected"
        self.approved_by = approved_by
        self.approved_date = date.today()
    
    def mark_as_ordered(self):
        """Mark recommendation as ordered"""
        self.status = "ordered"
    
    def update_urgency(self, new_urgency: str):
        """Update urgency level"""
        if new_urgency in ["low", "medium", "high", "critical"]:
            self.urgency_level = new_urgency
    
    def update_confidence(self, new_confidence: float):
        """Update confidence score"""
        self.confidence_score = max(0.0, min(1.0, new_confidence))
    
    def update_required_date(self, new_date: date):
        """Update required by date"""
        self.required_by_date = new_date
    
    def update_quantity(self, new_qty: Decimal):
        """Update recommended quantity"""
        self.recommended_qty = max(Decimal('0'), new_qty)
    
    def calculate_cost_per_unit(self) -> Decimal:
        """Calculate cost per unit"""
        if self.recommended_qty == 0:
            return Decimal('0')
        return self.estimated_cost / self.recommended_qty
    
    def get_priority_score(self) -> int:
        """Get priority score for sorting (higher = more urgent)"""
        urgency_scores = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25
        }
        
        base_score = urgency_scores.get(self.urgency_level, 50)
        
        # Add confidence bonus
        confidence_bonus = int(float(self.confidence_score) * 20)
        
        # Add overdue penalty
        overdue_penalty = -50 if self.is_overdue() else 0
        
        return base_score + confidence_bonus + overdue_penalty
    
    def get_recommendation_summary(self) -> dict:
        """Get summary information about this recommendation"""
        return {
            "material_id": self.material_id,
            "supplier_id": self.supplier_id,
            "recommended_qty": float(self.recommended_qty),
            "unit": self.unit,
            "estimated_cost": float(self.estimated_cost),
            "currency": self.currency,
            "cost_per_unit": float(self.calculate_cost_per_unit()),
            "urgency_level": self.urgency_level,
            "confidence_score": float(self.confidence_score),
            "status": self.status,
            "days_until_required": self.days_until_required(),
            "is_overdue": self.is_overdue(),
            "priority_score": self.get_priority_score(),
            "recommendation_date": self.recommendation_date.isoformat(),
            "required_by_date": self.required_by_date.isoformat()
        }
    
    def validate_recommendation(self) -> list:
        """Validate recommendation and return list of issues"""
        issues = []
        
        if self.recommended_qty <= 0:
            issues.append("Recommended quantity must be positive")
        
        if self.estimated_cost < 0:
            issues.append("Estimated cost cannot be negative")
        
        if not (0 <= float(self.confidence_score) <= 1):
            issues.append("Confidence score must be between 0 and 1")
        
        if self.urgency_level not in ["low", "medium", "high", "critical"]:
            issues.append("Invalid urgency level")
        
        if self.status not in ["pending", "approved", "rejected", "ordered"]:
            issues.append("Invalid status")
        
        if not self.material_id:
            issues.append("Material ID is required")
        
        if not self.supplier_id:
            issues.append("Supplier ID is required")
        
        if self.required_by_date < self.recommendation_date:
            issues.append("Required by date cannot be before recommendation date")
        
        return issues