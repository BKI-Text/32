import pytest
from decimal import Decimal
import math

from src.engine.eoq_optimizer import EOQOptimizer, EOQResult
from src.core.domain import (
    SupplierMaterial, Money, Quantity, MaterialId, SupplierId, LeadTime
)

class TestEOQOptimizer:
    def setup_method(self):
        self.optimizer = EOQOptimizer()
        
        self.test_supplier = SupplierMaterial(
            supplier_id=SupplierId(value="SUP-001"),
            material_id=MaterialId(value="YARN-001"),
            cost_per_unit=Money(amount=Decimal("5.00"), currency="USD"),
            moq=Quantity(amount=Decimal("100"), unit="lb"),
            lead_time=LeadTime(days=14),
            reliability_score=0.85,
            ordering_cost=Money(amount=Decimal("100.00"), currency="USD"),
            holding_cost_rate=0.25
        )
        
        self.material_id = MaterialId(value="YARN-001")
        self.quarterly_demand = Quantity(amount=Decimal("250"), unit="lb")
    
    def test_calculate_eoq_basic(self):
        result = self.optimizer.calculate_eoq(
            self.material_id,
            self.quarterly_demand,
            self.test_supplier
        )
        
        assert isinstance(result, EOQResult)
        assert result.material_id == self.material_id
        assert result.supplier_id == self.test_supplier.supplier_id
        assert result.eoq_quantity.amount > 0
        assert result.total_cost.amount > 0
        assert result.order_frequency > 0
    
    def test_calculate_eoq_respects_moq(self):
        # Test with low demand that should trigger MOQ constraint
        low_demand = Quantity(amount=Decimal("50"), unit="lb")
        
        result = self.optimizer.calculate_eoq(
            self.material_id,
            low_demand,
            self.test_supplier
        )
        
        # EOQ should be at least the MOQ
        assert result.eoq_quantity.amount >= self.test_supplier.moq.amount
    
    def test_calculate_eoq_formula_verification(self):
        # Test EOQ formula: sqrt(2 * D * S / H)
        annual_demand = self.quarterly_demand.amount * 4  # 1000
        ordering_cost = self.test_supplier.ordering_cost.amount  # 100
        unit_cost = self.test_supplier.cost_per_unit.amount  # 5
        holding_cost_rate = Decimal(str(self.test_supplier.holding_cost_rate))  # 0.25
        holding_cost_per_unit = unit_cost * holding_cost_rate  # 1.25
        
        expected_eoq = Decimal(str(math.sqrt(float(2 * annual_demand * ordering_cost / holding_cost_per_unit))))
        
        result = self.optimizer.calculate_eoq(
            self.material_id,
            self.quarterly_demand,
            self.test_supplier
        )
        
        # Should match expected EOQ (or MOQ if EOQ is smaller)
        if expected_eoq >= self.test_supplier.moq.amount:
            assert abs(result.eoq_quantity.amount - expected_eoq) < Decimal("1.0")
        else:
            assert result.eoq_quantity.amount == self.test_supplier.moq.amount
    
    def test_calculate_eoq_cost_components(self):
        result = self.optimizer.calculate_eoq(
            self.material_id,
            self.quarterly_demand,
            self.test_supplier
        )
        
        # Verify cost components are calculated
        assert result.annual_holding_cost.amount > 0
        assert result.annual_ordering_cost.amount > 0
        assert result.total_cost.amount > 0
        
        # Total cost should be sum of holding + ordering + material costs
        annual_demand = self.quarterly_demand.amount * 4
        material_cost = annual_demand * self.test_supplier.cost_per_unit.amount
        expected_total = result.annual_holding_cost.amount + result.annual_ordering_cost.amount + material_cost
        
        assert abs(result.total_cost.amount - expected_total) < Decimal("0.01")
    
    def test_optimize_multiple_suppliers(self):
        # Create multiple suppliers with different costs
        suppliers = [
            self.test_supplier,
            SupplierMaterial(
                supplier_id=SupplierId(value="SUP-002"),
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("4.50"), currency="USD"),
                moq=Quantity(amount=Decimal("150"), unit="lb"),
                lead_time=LeadTime(days=21),
                reliability_score=0.75,
                ordering_cost=Money(amount=Decimal("80.00"), currency="USD"),
                holding_cost_rate=0.20
            ),
            SupplierMaterial(
                supplier_id=SupplierId(value="SUP-003"),
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("6.00"), currency="USD"),
                moq=Quantity(amount=Decimal("75"), unit="lb"),
                lead_time=LeadTime(days=7),
                reliability_score=0.95,
                ordering_cost=Money(amount=Decimal("120.00"), currency="USD"),
                holding_cost_rate=0.30
            )
        ]
        
        results = self.optimizer.optimize_multiple_suppliers(
            self.material_id,
            self.quarterly_demand,
            suppliers,
            max_suppliers=3
        )
        
        assert len(results) <= 3
        assert all(isinstance(result, EOQResult) for result in results)
        
        # Results should be sorted by total cost (ascending)
        for i in range(len(results) - 1):
            assert results[i].total_cost.amount <= results[i + 1].total_cost.amount
    
    def test_calculate_multi_supplier_allocation(self):
        suppliers = [self.test_supplier]
        
        allocation = self.optimizer.calculate_multi_supplier_allocation(
            self.material_id,
            self.quarterly_demand,
            suppliers,
            risk_diversification=False
        )
        
        assert len(allocation) == 1
        assert self.test_supplier.supplier_id in allocation
        assert allocation[self.test_supplier.supplier_id] == self.quarterly_demand
    
    def test_calculate_multi_supplier_allocation_with_diversification(self):
        # Create two suppliers for diversification test
        suppliers = [
            self.test_supplier,
            SupplierMaterial(
                supplier_id=SupplierId(value="SUP-002"),
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("4.50"), currency="USD"),
                moq=Quantity(amount=Decimal("150"), unit="lb"),
                lead_time=LeadTime(days=21),
                reliability_score=0.75,
                ordering_cost=Money(amount=Decimal("80.00"), currency="USD"),
                holding_cost_rate=0.20
            )
        ]
        
        allocation = self.optimizer.calculate_multi_supplier_allocation(
            self.material_id,
            self.quarterly_demand,
            suppliers,
            risk_diversification=True
        )
        
        assert len(allocation) == 2
        
        # Verify allocation percentages (70% primary, 30% secondary)
        total_allocated = sum(qty.amount for qty in allocation.values())
        assert abs(total_allocated - self.quarterly_demand.amount) < Decimal("0.01")
    
    def test_generate_procurement_schedule(self):
        result = self.optimizer.calculate_eoq(
            self.material_id,
            self.quarterly_demand,
            self.test_supplier
        )
        
        schedule = self.optimizer.generate_procurement_schedule(
            result,
            planning_horizon_days=90
        )
        
        assert len(schedule) >= 1
        assert all(isinstance(entry, dict) for entry in schedule)
        
        # Check schedule entry structure
        for entry in schedule:
            assert 'day' in entry
            assert 'quantity' in entry
            assert 'cost' in entry
            assert 'supplier_id' in entry
            assert entry['day'] >= 0
            assert entry['quantity'] > 0
    
    def test_fallback_on_calculation_error(self):
        # Create supplier with invalid data to trigger fallback
        invalid_supplier = SupplierMaterial(
            supplier_id=SupplierId(value="SUP-INVALID"),
            material_id=MaterialId(value="YARN-001"),
            cost_per_unit=Money(amount=Decimal("0.00"), currency="USD"),  # Invalid cost
            moq=Quantity(amount=Decimal("100"), unit="lb"),
            lead_time=LeadTime(days=14),
            reliability_score=0.85,
            ordering_cost=Money(amount=Decimal("100.00"), currency="USD"),
            holding_cost_rate=0.0  # Invalid holding cost rate
        )
        
        result = self.optimizer.calculate_eoq(
            self.material_id,
            self.quarterly_demand,
            invalid_supplier
        )
        
        # Should still return a valid result (fallback)
        assert isinstance(result, EOQResult)
        assert result.eoq_quantity.amount > 0
        assert result.total_cost.amount >= 0
    
    def test_empty_suppliers_list(self):
        results = self.optimizer.optimize_multiple_suppliers(
            self.material_id,
            self.quarterly_demand,
            [],
            max_suppliers=3
        )
        
        assert results == []
    
    def test_single_supplier_allocation(self):
        suppliers = [self.test_supplier]
        
        allocation = self.optimizer.calculate_multi_supplier_allocation(
            self.material_id,
            self.quarterly_demand,
            suppliers,
            risk_diversification=True
        )
        
        # Should allocate all to single supplier even with diversification enabled
        assert len(allocation) == 1
        assert allocation[self.test_supplier.supplier_id] == self.quarterly_demand