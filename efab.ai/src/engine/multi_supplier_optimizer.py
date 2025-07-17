from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.domain import (
    SupplierMaterial, Money, Quantity, MaterialId, SupplierId, LeadTime, RiskLevel
)

logger = logging.getLogger(__name__)

class SourcingStrategy(Enum):
    SINGLE_SOURCE = "single_source"
    DUAL_SOURCE = "dual_source"
    MULTI_SOURCE = "multi_source"

@dataclass
class SupplierScore:
    supplier_id: SupplierId
    cost_score: float
    reliability_score: float
    lead_time_score: float
    total_score: float
    
@dataclass
class SourcingRecommendation:
    material_id: MaterialId
    strategy: SourcingStrategy
    allocations: Dict[SupplierId, Quantity]
    total_cost: Money
    risk_assessment: RiskLevel
    reasoning: str

class MultiSupplierOptimizer:
    def __init__(
        self, 
        cost_weight: float = 0.4,
        reliability_weight: float = 0.3,
        lead_time_weight: float = 0.3,
        risk_diversification_threshold: float = 0.5
    ):
        self.cost_weight = cost_weight
        self.reliability_weight = reliability_weight
        self.lead_time_weight = lead_time_weight
        self.risk_diversification_threshold = risk_diversification_threshold
        
        # Validate weights sum to 1.0
        total_weight = cost_weight + reliability_weight + lead_time_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def score_suppliers(self, suppliers: List[SupplierMaterial]) -> List[SupplierScore]:
        """Score suppliers based on cost, reliability, and lead time"""
        
        if not suppliers:
            return []
        
        # Extract metrics for normalization
        costs = [float(s.cost_per_unit.amount) for s in suppliers]
        reliabilities = [s.reliability_score for s in suppliers]
        lead_times = [s.lead_time.days for s in suppliers]
        
        # Normalize scores (0-1 range)
        min_cost, max_cost = min(costs), max(costs)
        min_reliability, max_reliability = min(reliabilities), max(reliabilities)
        min_lead_time, max_lead_time = min(lead_times), max(lead_times)
        
        scores = []
        
        for supplier in suppliers:
            # Cost score (lower cost = higher score)
            if max_cost > min_cost:
                cost_score = 1.0 - (float(supplier.cost_per_unit.amount) - min_cost) / (max_cost - min_cost)
            else:
                cost_score = 1.0
            
            # Reliability score (higher reliability = higher score)
            if max_reliability > min_reliability:
                reliability_score = (supplier.reliability_score - min_reliability) / (max_reliability - min_reliability)
            else:
                reliability_score = 1.0
            
            # Lead time score (lower lead time = higher score)
            if max_lead_time > min_lead_time:
                lead_time_score = 1.0 - (supplier.lead_time.days - min_lead_time) / (max_lead_time - min_lead_time)
            else:
                lead_time_score = 1.0
            
            # Calculate weighted total score
            total_score = (
                cost_score * self.cost_weight +
                reliability_score * self.reliability_weight +
                lead_time_score * self.lead_time_weight
            )
            
            scores.append(SupplierScore(
                supplier_id=supplier.supplier_id,
                cost_score=cost_score,
                reliability_score=reliability_score,
                lead_time_score=lead_time_score,
                total_score=total_score
            ))
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return scores
    
    def determine_sourcing_strategy(
        self, 
        material_id: MaterialId,
        demand: Quantity,
        suppliers: List[SupplierMaterial]
    ) -> SourcingStrategy:
        """Determine optimal sourcing strategy based on demand and supplier characteristics"""
        
        if len(suppliers) <= 1:
            return SourcingStrategy.SINGLE_SOURCE
        
        # Score suppliers
        supplier_scores = self.score_suppliers(suppliers)
        
        if len(supplier_scores) <= 1:
            return SourcingStrategy.SINGLE_SOURCE
        
        # Check if top supplier is significantly better than others
        top_score = supplier_scores[0].total_score
        second_score = supplier_scores[1].total_score
        
        score_gap = top_score - second_score
        
        # If top supplier is significantly better, use single source
        if score_gap > 0.3:
            return SourcingStrategy.SINGLE_SOURCE
        
        # If demand is high or suppliers are similar, consider diversification
        if len(supplier_scores) >= 3 and score_gap < 0.1:
            return SourcingStrategy.MULTI_SOURCE
        else:
            return SourcingStrategy.DUAL_SOURCE
    
    def optimize_sourcing(
        self, 
        material_id: MaterialId,
        demand: Quantity,
        suppliers: List[SupplierMaterial]
    ) -> SourcingRecommendation:
        """Optimize sourcing strategy for a material"""
        
        if not suppliers:
            return SourcingRecommendation(
                material_id=material_id,
                strategy=SourcingStrategy.SINGLE_SOURCE,
                allocations={},
                total_cost=Money(amount=Decimal("0"), currency="USD"),
                risk_assessment=RiskLevel.HIGH,
                reasoning="No suppliers available"
            )
        
        # Determine strategy
        strategy = self.determine_sourcing_strategy(material_id, demand, suppliers)
        
        # Score suppliers
        supplier_scores = self.score_suppliers(suppliers)
        
        # Generate allocations based on strategy
        allocations = self._generate_allocations(strategy, demand, suppliers, supplier_scores)
        
        # Calculate total cost
        total_cost = self._calculate_total_cost(allocations, suppliers)
        
        # Assess risk
        risk_assessment = self._assess_sourcing_risk(strategy, supplier_scores)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(strategy, supplier_scores, allocations)
        
        return SourcingRecommendation(
            material_id=material_id,
            strategy=strategy,
            allocations=allocations,
            total_cost=total_cost,
            risk_assessment=risk_assessment,
            reasoning=reasoning
        )
    
    def _generate_allocations(
        self, 
        strategy: SourcingStrategy,
        demand: Quantity,
        suppliers: List[SupplierMaterial],
        supplier_scores: List[SupplierScore]
    ) -> Dict[SupplierId, Quantity]:
        """Generate quantity allocations based on sourcing strategy"""
        
        allocations = {}
        
        if strategy == SourcingStrategy.SINGLE_SOURCE:
            # Allocate all demand to top supplier
            if supplier_scores:
                top_supplier = supplier_scores[0]
                allocations[top_supplier.supplier_id] = demand
        
        elif strategy == SourcingStrategy.DUAL_SOURCE:
            # Split between top 2 suppliers
            if len(supplier_scores) >= 2:
                primary_supplier = supplier_scores[0]
                secondary_supplier = supplier_scores[1]
                
                # Allocate based on relative scores
                total_score = primary_supplier.total_score + secondary_supplier.total_score
                primary_ratio = primary_supplier.total_score / total_score
                
                # Ensure primary gets at least 60%
                primary_ratio = max(primary_ratio, 0.6)
                secondary_ratio = 1.0 - primary_ratio
                
                allocations[primary_supplier.supplier_id] = Quantity(
                    amount=demand.amount * Decimal(str(primary_ratio)),
                    unit=demand.unit
                )
                allocations[secondary_supplier.supplier_id] = Quantity(
                    amount=demand.amount * Decimal(str(secondary_ratio)),
                    unit=demand.unit
                )
        
        elif strategy == SourcingStrategy.MULTI_SOURCE:
            # Distribute among top 3 suppliers
            top_suppliers = supplier_scores[:3]
            
            if len(top_suppliers) >= 3:
                # Distribute: 50%, 30%, 20%
                ratios = [0.5, 0.3, 0.2]
                
                for i, supplier_score in enumerate(top_suppliers):
                    ratio = ratios[i] if i < len(ratios) else 0.0
                    allocations[supplier_score.supplier_id] = Quantity(
                        amount=demand.amount * Decimal(str(ratio)),
                        unit=demand.unit
                    )
        
        return allocations
    
    def _calculate_total_cost(
        self, 
        allocations: Dict[SupplierId, Quantity],
        suppliers: List[SupplierMaterial]
    ) -> Money:
        """Calculate total cost for allocations"""
        
        total_cost = Decimal("0")
        currency = "USD"
        
        # Create supplier lookup
        supplier_lookup = {s.supplier_id: s for s in suppliers}
        
        for supplier_id, quantity in allocations.items():
            if supplier_id in supplier_lookup:
                supplier = supplier_lookup[supplier_id]
                cost = supplier.cost_per_unit.amount * quantity.amount
                total_cost += cost
                currency = supplier.cost_per_unit.currency
        
        return Money(amount=total_cost, currency=currency)
    
    def _assess_sourcing_risk(
        self, 
        strategy: SourcingStrategy,
        supplier_scores: List[SupplierScore]
    ) -> RiskLevel:
        """Assess risk level based on sourcing strategy and supplier reliability"""
        
        if not supplier_scores:
            return RiskLevel.HIGH
        
        # Calculate average reliability
        avg_reliability = sum(score.reliability_score for score in supplier_scores) / len(supplier_scores)
        
        # Adjust risk based on strategy
        if strategy == SourcingStrategy.SINGLE_SOURCE:
            # Single source is riskier
            if avg_reliability >= 0.9:
                return RiskLevel.MEDIUM
            elif avg_reliability >= 0.7:
                return RiskLevel.HIGH
            else:
                return RiskLevel.HIGH
        
        elif strategy == SourcingStrategy.DUAL_SOURCE:
            # Dual source reduces risk
            if avg_reliability >= 0.8:
                return RiskLevel.LOW
            elif avg_reliability >= 0.6:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.HIGH
        
        elif strategy == SourcingStrategy.MULTI_SOURCE:
            # Multi-source is lowest risk
            if avg_reliability >= 0.7:
                return RiskLevel.LOW
            elif avg_reliability >= 0.5:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.HIGH
        
        return RiskLevel.MEDIUM
    
    def _generate_reasoning(
        self, 
        strategy: SourcingStrategy,
        supplier_scores: List[SupplierScore],
        allocations: Dict[SupplierId, Quantity]
    ) -> str:
        """Generate reasoning for sourcing decision"""
        
        if not supplier_scores:
            return "No suppliers available for analysis"
        
        top_supplier = supplier_scores[0]
        
        if strategy == SourcingStrategy.SINGLE_SOURCE:
            return f"Single source recommended. Top supplier {top_supplier.supplier_id.value} has strong performance (score: {top_supplier.total_score:.2f})"
        
        elif strategy == SourcingStrategy.DUAL_SOURCE:
            if len(supplier_scores) >= 2:
                second_supplier = supplier_scores[1]
                return f"Dual sourcing recommended for risk mitigation. Primary: {top_supplier.supplier_id.value} (score: {top_supplier.total_score:.2f}), Secondary: {second_supplier.supplier_id.value} (score: {second_supplier.total_score:.2f})"
            else:
                return "Dual sourcing planned but only one supplier available"
        
        elif strategy == SourcingStrategy.MULTI_SOURCE:
            supplier_list = [f"{score.supplier_id.value} ({score.total_score:.2f})" for score in supplier_scores[:3]]
            return f"Multi-source strategy for maximum risk diversification. Suppliers: {', '.join(supplier_list)}"
        
        return "Sourcing strategy determined based on supplier analysis"