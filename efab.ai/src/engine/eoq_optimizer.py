from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import math
import logging
from dataclasses import dataclass

from ..core.domain import (
    SupplierMaterial, Money, Quantity, MaterialId, SupplierId, LeadTime
)

logger = logging.getLogger(__name__)

@dataclass
class EOQResult:
    material_id: MaterialId
    supplier_id: SupplierId
    eoq_quantity: Quantity
    total_cost: Money
    annual_holding_cost: Money
    annual_ordering_cost: Money
    order_frequency: float
    
class EOQOptimizer:
    def __init__(self, annual_demand_multiplier: float = 4.0):
        self.annual_demand_multiplier = annual_demand_multiplier
        
    def calculate_eoq(
        self, 
        material_id: MaterialId,
        quarterly_demand: Quantity,
        supplier: SupplierMaterial
    ) -> EOQResult:
        """Calculate Economic Order Quantity for a material-supplier combination"""
        
        # Convert quarterly demand to annual
        annual_demand = quarterly_demand.amount * Decimal(str(self.annual_demand_multiplier))
        
        # Extract cost parameters
        unit_cost = supplier.cost_per_unit.amount
        ordering_cost = supplier.ordering_cost.amount
        holding_cost_rate = Decimal(str(supplier.holding_cost_rate))
        
        # Calculate holding cost per unit per year
        holding_cost_per_unit = unit_cost * holding_cost_rate
        
        if holding_cost_per_unit <= 0:
            logger.warning(f"Invalid holding cost for material {material_id.value}")
            return self._create_fallback_result(material_id, supplier, quarterly_demand)
        
        try:
            # EOQ formula: sqrt(2 * D * S / H)
            # Where D = annual demand, S = ordering cost, H = holding cost per unit
            eoq_squared = (2 * annual_demand * ordering_cost) / holding_cost_per_unit
            eoq = Decimal(str(math.sqrt(float(eoq_squared))))
            
            # Apply minimum order quantity constraint
            final_quantity = max(eoq, supplier.moq.amount)
            
            # Calculate associated costs
            annual_holding_cost = (final_quantity / 2) * holding_cost_per_unit
            annual_ordering_cost = (annual_demand / final_quantity) * ordering_cost
            total_annual_cost = annual_holding_cost + annual_ordering_cost + (annual_demand * unit_cost)
            
            # Calculate order frequency (orders per year)
            order_frequency = float(annual_demand / final_quantity)
            
            return EOQResult(
                material_id=material_id,
                supplier_id=supplier.supplier_id,
                eoq_quantity=Quantity(amount=final_quantity, unit=quarterly_demand.unit),
                total_cost=Money(amount=total_annual_cost, currency=supplier.cost_per_unit.currency),
                annual_holding_cost=Money(amount=annual_holding_cost, currency=supplier.cost_per_unit.currency),
                annual_ordering_cost=Money(amount=annual_ordering_cost, currency=supplier.cost_per_unit.currency),
                order_frequency=order_frequency
            )
            
        except Exception as e:
            logger.error(f"EOQ calculation failed for material {material_id.value}: {e}")
            return self._create_fallback_result(material_id, supplier, quarterly_demand)
    
    def _create_fallback_result(
        self, 
        material_id: MaterialId, 
        supplier: SupplierMaterial, 
        demand: Quantity
    ) -> EOQResult:
        """Create fallback result when EOQ calculation fails"""
        fallback_quantity = max(demand.amount, supplier.moq.amount)
        fallback_cost = fallback_quantity * supplier.cost_per_unit.amount
        
        return EOQResult(
            material_id=material_id,
            supplier_id=supplier.supplier_id,
            eoq_quantity=Quantity(amount=fallback_quantity, unit=demand.unit),
            total_cost=Money(amount=fallback_cost, currency=supplier.cost_per_unit.currency),
            annual_holding_cost=Money(amount=Decimal("0"), currency=supplier.cost_per_unit.currency),
            annual_ordering_cost=Money(amount=Decimal("0"), currency=supplier.cost_per_unit.currency),
            order_frequency=1.0
        )
    
    def optimize_multiple_suppliers(
        self, 
        material_id: MaterialId,
        demand: Quantity,
        suppliers: List[SupplierMaterial],
        max_suppliers: int = 3
    ) -> List[EOQResult]:
        """Optimize procurement across multiple suppliers"""
        
        if not suppliers:
            return []
        
        # Calculate EOQ for each supplier
        eoq_results = []
        for supplier in suppliers:
            result = self.calculate_eoq(material_id, demand, supplier)
            eoq_results.append(result)
        
        # Sort by total cost (ascending)
        eoq_results.sort(key=lambda x: x.total_cost.amount)
        
        # Return top suppliers up to max_suppliers limit
        return eoq_results[:max_suppliers]
    
    def calculate_multi_supplier_allocation(
        self, 
        material_id: MaterialId,
        total_demand: Quantity,
        suppliers: List[SupplierMaterial],
        risk_diversification: bool = True
    ) -> Dict[SupplierId, Quantity]:
        """Calculate optimal allocation across multiple suppliers"""
        
        if not suppliers:
            return {}
        
        if len(suppliers) == 1:
            return {suppliers[0].supplier_id: total_demand}
        
        # Calculate EOQ for each supplier
        eoq_results = self.optimize_multiple_suppliers(
            material_id, 
            total_demand, 
            suppliers, 
            max_suppliers=len(suppliers)
        )
        
        if not eoq_results:
            return {}
        
        # Simple allocation strategy: distribute based on cost efficiency
        allocation = {}
        
        if risk_diversification and len(eoq_results) > 1:
            # Diversify risk by splitting demand
            primary_supplier = eoq_results[0]
            secondary_supplier = eoq_results[1]
            
            # Allocate 70% to primary, 30% to secondary
            primary_allocation = total_demand.amount * Decimal("0.7")
            secondary_allocation = total_demand.amount * Decimal("0.3")
            
            allocation[primary_supplier.supplier_id] = Quantity(
                amount=primary_allocation, 
                unit=total_demand.unit
            )
            allocation[secondary_supplier.supplier_id] = Quantity(
                amount=secondary_allocation, 
                unit=total_demand.unit
            )
        else:
            # Single supplier allocation
            best_supplier = eoq_results[0]
            allocation[best_supplier.supplier_id] = total_demand
        
        return allocation
    
    def generate_procurement_schedule(
        self, 
        eoq_result: EOQResult,
        planning_horizon_days: int = 90
    ) -> List[Dict[str, Any]]:
        """Generate procurement schedule based on EOQ results"""
        
        # Calculate order intervals
        days_per_year = 365
        order_interval_days = days_per_year / eoq_result.order_frequency
        
        # Generate schedule for planning horizon
        schedule = []
        current_day = 0
        
        while current_day < planning_horizon_days:
            schedule.append({
                'day': int(current_day),
                'date': f"Day {int(current_day)}",
                'quantity': eoq_result.eoq_quantity.amount,
                'unit': eoq_result.eoq_quantity.unit,
                'cost': eoq_result.eoq_quantity.amount * eoq_result.total_cost.amount / (eoq_result.eoq_quantity.amount * 4),  # Quarterly cost
                'supplier_id': eoq_result.supplier_id.value
            })
            
            current_day += order_interval_days
        
        return schedule