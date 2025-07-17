import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, patch

from src.engine.planning_engine import PlanningEngine
from src.core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast,
    ProcurementRecommendation, MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

class TestPlanningEngine:
    def setup_method(self):
        self.engine = PlanningEngine()
        
        # Create test data
        self.test_forecasts = [
            Forecast(
                sku_id=SkuId(value="SKU-001"),
                forecast_qty=Quantity(amount=Decimal("100"), unit="unit"),
                forecast_date=date(2024, 1, 1),
                source=ForecastSource.SALES_ORDER,
                confidence_score=0.9
            ),
            Forecast(
                sku_id=SkuId(value="SKU-001"),
                forecast_qty=Quantity(amount=Decimal("50"), unit="unit"),
                forecast_date=date(2024, 1, 1),
                source=ForecastSource.PROD_PLAN,
                confidence_score=0.8
            )
        ]
        
        self.test_boms = [
            BOM(
                sku_id=SkuId(value="SKU-001"),
                material_id=MaterialId(value="YARN-001"),
                qty_per_unit=Quantity(amount=Decimal("0.5"), unit="lb"),
                unit="lb"
            ),
            BOM(
                sku_id=SkuId(value="SKU-001"),
                material_id=MaterialId(value="YARN-002"),
                qty_per_unit=Quantity(amount=Decimal("0.3"), unit="lb"),
                unit="lb"
            )
        ]
        
        self.test_inventory = [
            Inventory(
                material_id=MaterialId(value="YARN-001"),
                on_hand_qty=Quantity(amount=Decimal("50"), unit="lb"),
                open_po_qty=Quantity(amount=Decimal("20"), unit="lb")
            ),
            Inventory(
                material_id=MaterialId(value="YARN-002"),
                on_hand_qty=Quantity(amount=Decimal("30"), unit="lb"),
                open_po_qty=Quantity(amount=Decimal("10"), unit="lb")
            )
        ]
        
        self.test_suppliers = [
            SupplierMaterial(
                supplier_id=SupplierId(value="SUP-001"),
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("5.00"), currency="USD"),
                moq=Quantity(amount=Decimal("100"), unit="lb"),
                lead_time=LeadTime(days=14),
                reliability_score=0.85
            ),
            SupplierMaterial(
                supplier_id=SupplierId(value="SUP-002"),
                material_id=MaterialId(value="YARN-002"),
                cost_per_unit=Money(amount=Decimal("4.50"), currency="USD"),
                moq=Quantity(amount=Decimal("50"), unit="lb"),
                lead_time=LeadTime(days=10),
                reliability_score=0.90
            )
        ]
    
    def test_unify_forecasts(self):
        unified = self.engine._unify_forecasts(self.test_forecasts)
        
        assert "SKU-001" in unified
        # Expected: 100 * 1.0 * 0.9 + 50 * 0.9 * 0.8 = 90 + 36 = 126
        assert unified["SKU-001"].amount == Decimal("126")
        assert unified["SKU-001"].unit == "unit"
    
    def test_explode_boms(self):
        unified_forecasts = {"SKU-001": Quantity(amount=Decimal("100"), unit="unit")}
        
        requirements = self.engine._explode_boms(unified_forecasts, self.test_boms)
        
        assert "YARN-001" in requirements
        assert "YARN-002" in requirements
        assert requirements["YARN-001"].amount == Decimal("50")  # 100 * 0.5
        assert requirements["YARN-002"].amount == Decimal("30")  # 100 * 0.3
    
    def test_net_inventory(self):
        requirements = {
            "YARN-001": Quantity(amount=Decimal("100"), unit="lb"),
            "YARN-002": Quantity(amount=Decimal("50"), unit="lb")
        }
        
        net_requirements = self.engine._net_inventory(requirements, self.test_inventory)
        
        # YARN-001: 100 - (50 + 20) = 30
        assert net_requirements["YARN-001"].amount == Decimal("30")
        # YARN-002: 50 - (30 + 10) = 10
        assert net_requirements["YARN-002"].amount == Decimal("10")
    
    def test_optimize_procurement(self):
        requirements = {
            "YARN-001": Quantity(amount=Decimal("50"), unit="lb"),
            "YARN-002": Quantity(amount=Decimal("30"), unit="lb")
        }
        
        optimized = self.engine._optimize_procurement(requirements, self.test_suppliers)
        
        # Should apply safety stock (15% default)
        assert optimized["YARN-001"].amount >= Decimal("57.5")  # 50 * 1.15
        assert optimized["YARN-002"].amount >= Decimal("34.5")  # 30 * 1.15
    
    def test_execute_planning_cycle(self):
        recommendations = self.engine.execute_planning_cycle(
            self.test_forecasts,
            self.test_boms,
            self.test_inventory,
            self.test_suppliers
        )
        
        assert len(recommendations) >= 1
        assert all(isinstance(rec, ProcurementRecommendation) for rec in recommendations)
        
        # Check that recommendations have required fields
        for rec in recommendations:
            assert rec.material_id is not None
            assert rec.supplier_id is not None
            assert rec.recommended_order_qty.amount > 0
            assert rec.total_cost.amount > 0
            assert rec.reasoning is not None
    
    def test_calculate_eoq(self):
        # Test EOQ calculation
        annual_demand = Decimal("1000")
        supplier = self.test_suppliers[0]
        
        eoq = self.engine._calculate_eoq(annual_demand, supplier)
        
        # EOQ should be positive
        assert eoq > 0
        
        # Should be at least the MOQ
        assert eoq >= supplier.moq.amount
    
    def test_assess_risk(self):
        # Test risk assessment
        high_risk = self.engine._assess_risk(0.6)
        medium_risk = self.engine._assess_risk(0.8)
        low_risk = self.engine._assess_risk(0.9)
        
        assert high_risk == RiskLevel.HIGH
        assert medium_risk == RiskLevel.MEDIUM
        assert low_risk == RiskLevel.LOW
    
    def test_calculate_urgency_score(self):
        # Test urgency score calculation
        short_lead_time = LeadTime(days=7)
        medium_lead_time = LeadTime(days=15)
        long_lead_time = LeadTime(days=30)
        
        short_urgency = self.engine._calculate_urgency_score(short_lead_time)
        medium_urgency = self.engine._calculate_urgency_score(medium_lead_time)
        long_urgency = self.engine._calculate_urgency_score(long_lead_time)
        
        # Shorter lead times should have lower urgency scores
        assert short_urgency < medium_urgency
        assert medium_urgency < long_urgency
        assert all(0 <= score <= 1 for score in [short_urgency, medium_urgency, long_urgency])
    
    def test_empty_inputs(self):
        # Test with empty inputs
        recommendations = self.engine.execute_planning_cycle([], [], [], [])
        assert recommendations == []
    
    def test_missing_inventory(self):
        # Test with missing inventory for some materials
        partial_inventory = [self.test_inventory[0]]  # Only YARN-001
        
        recommendations = self.engine.execute_planning_cycle(
            self.test_forecasts,
            self.test_boms,
            partial_inventory,
            self.test_suppliers
        )
        
        # Should still generate recommendations
        assert len(recommendations) >= 1
    
    def test_missing_suppliers(self):
        # Test with missing suppliers for some materials
        partial_suppliers = [self.test_suppliers[0]]  # Only YARN-001
        
        recommendations = self.engine.execute_planning_cycle(
            self.test_forecasts,
            self.test_boms,
            self.test_inventory,
            partial_suppliers
        )
        
        # Should generate recommendations only for materials with suppliers
        material_ids = [rec.material_id.value for rec in recommendations]
        assert "YARN-001" in material_ids
        # YARN-002 should not be in recommendations as it has no supplier