import pytest
from decimal import Decimal
from datetime import datetime, date

from src.core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast,
    ProcurementRecommendation, MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

class TestValueObjects:
    def test_money_creation(self):
        money = Money(amount=Decimal("100.50"), currency="USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"
    
    def test_money_addition(self):
        money1 = Money(amount=Decimal("100.00"), currency="USD")
        money2 = Money(amount=Decimal("50.00"), currency="USD")
        result = money1 + money2
        assert result.amount == Decimal("150.00")
        assert result.currency == "USD"
    
    def test_money_multiplication(self):
        money = Money(amount=Decimal("10.00"), currency="USD")
        result = money * 3
        assert result.amount == Decimal("30.00")
        assert result.currency == "USD"
    
    def test_quantity_creation(self):
        qty = Quantity(amount=Decimal("100"), unit="lb")
        assert qty.amount == Decimal("100")
        assert qty.unit == "lb"
    
    def test_quantity_addition(self):
        qty1 = Quantity(amount=Decimal("100"), unit="lb")
        qty2 = Quantity(amount=Decimal("50"), unit="lb")
        result = qty1 + qty2
        assert result.amount == Decimal("150")
        assert result.unit == "lb"

class TestMaterial:
    def test_material_creation(self):
        material = Material(
            id=MaterialId(value="YARN-001"),
            name="Cotton Yarn",
            type=MaterialType.YARN,
            description="High-quality cotton yarn"
        )
        assert material.id.value == "YARN-001"
        assert material.name == "Cotton Yarn"
        assert material.type == MaterialType.YARN
        assert material.description == "High-quality cotton yarn"
    
    def test_update_specifications(self):
        material = Material(
            id=MaterialId(value="YARN-001"),
            name="Cotton Yarn",
            type=MaterialType.YARN
        )
        
        material.update_specifications({"weight": "200g", "color": "natural"})
        assert material.specifications["weight"] == "200g"
        assert material.specifications["color"] == "natural"

class TestSupplier:
    def test_supplier_creation(self):
        supplier = Supplier(
            id=SupplierId(value="SUP-001"),
            name="ABC Textiles",
            lead_time=LeadTime(days=14),
            reliability_score=0.85,
            risk_level=RiskLevel.LOW
        )
        assert supplier.id.value == "SUP-001"
        assert supplier.name == "ABC Textiles"
        assert supplier.lead_time.days == 14
        assert supplier.reliability_score == 0.85
        assert supplier.risk_level == RiskLevel.LOW
    
    def test_supplier_reliability_validation(self):
        with pytest.raises(ValueError):
            Supplier(
                id=SupplierId(value="SUP-001"),
                name="ABC Textiles",
                lead_time=LeadTime(days=14),
                reliability_score=1.5,  # Invalid - greater than 1
                risk_level=RiskLevel.LOW
            )

class TestSupplierMaterial:
    def test_supplier_material_creation(self):
        supplier_material = SupplierMaterial(
            supplier_id=SupplierId(value="SUP-001"),
            material_id=MaterialId(value="YARN-001"),
            cost_per_unit=Money(amount=Decimal("5.00"), currency="USD"),
            moq=Quantity(amount=Decimal("100"), unit="lb"),
            lead_time=LeadTime(days=14),
            reliability_score=0.85
        )
        assert supplier_material.supplier_id.value == "SUP-001"
        assert supplier_material.material_id.value == "YARN-001"
        assert supplier_material.cost_per_unit.amount == Decimal("5.00")
        assert supplier_material.moq.amount == Decimal("100")

class TestInventory:
    def test_inventory_creation(self):
        inventory = Inventory(
            material_id=MaterialId(value="YARN-001"),
            on_hand_qty=Quantity(amount=Decimal("500"), unit="lb"),
            open_po_qty=Quantity(amount=Decimal("200"), unit="lb")
        )
        assert inventory.material_id.value == "YARN-001"
        assert inventory.on_hand_qty.amount == Decimal("500")
        assert inventory.open_po_qty.amount == Decimal("200")
    
    def test_get_available_qty(self):
        inventory = Inventory(
            material_id=MaterialId(value="YARN-001"),
            on_hand_qty=Quantity(amount=Decimal("500"), unit="lb"),
            open_po_qty=Quantity(amount=Decimal("200"), unit="lb")
        )
        available = inventory.get_available_qty()
        assert available.amount == Decimal("700")
        assert available.unit == "lb"

class TestBOM:
    def test_bom_creation(self):
        bom = BOM(
            sku_id=SkuId(value="SKU-001"),
            material_id=MaterialId(value="YARN-001"),
            qty_per_unit=Quantity(amount=Decimal("0.5"), unit="lb"),
            unit="lb"
        )
        assert bom.sku_id.value == "SKU-001"
        assert bom.material_id.value == "YARN-001"
        assert bom.qty_per_unit.amount == Decimal("0.5")
    
    def test_calculate_requirement(self):
        bom = BOM(
            sku_id=SkuId(value="SKU-001"),
            material_id=MaterialId(value="YARN-001"),
            qty_per_unit=Quantity(amount=Decimal("0.5"), unit="lb"),
            unit="lb"
        )
        
        sku_qty = Quantity(amount=Decimal("100"), unit="unit")
        requirement = bom.calculate_requirement(sku_qty)
        assert requirement.amount == Decimal("50")
        assert requirement.unit == "lb"

class TestForecast:
    def test_forecast_creation(self):
        forecast = Forecast(
            sku_id=SkuId(value="SKU-001"),
            forecast_qty=Quantity(amount=Decimal("100"), unit="unit"),
            forecast_date=date(2024, 1, 1),
            source=ForecastSource.SALES_ORDER,
            confidence_score=0.9
        )
        assert forecast.sku_id.value == "SKU-001"
        assert forecast.forecast_qty.amount == Decimal("100")
        assert forecast.source == ForecastSource.SALES_ORDER
        assert forecast.confidence_score == 0.9
    
    def test_forecast_confidence_validation(self):
        with pytest.raises(ValueError):
            Forecast(
                sku_id=SkuId(value="SKU-001"),
                forecast_qty=Quantity(amount=Decimal("100"), unit="unit"),
                forecast_date=date(2024, 1, 1),
                source=ForecastSource.SALES_ORDER,
                confidence_score=1.5  # Invalid - greater than 1
            )

class TestProcurementRecommendation:
    def test_procurement_recommendation_creation(self):
        recommendation = ProcurementRecommendation(
            material_id=MaterialId(value="YARN-001"),
            supplier_id=SupplierId(value="SUP-001"),
            recommended_order_qty=Quantity(amount=Decimal("200"), unit="lb"),
            unit_cost=Money(amount=Decimal("5.00"), currency="USD"),
            total_cost=Money(amount=Decimal("1000.00"), currency="USD"),
            expected_lead_time=LeadTime(days=14),
            risk_flag=RiskLevel.LOW,
            reasoning="Optimal cost/reliability ratio",
            urgency_score=0.3
        )
        assert recommendation.material_id.value == "YARN-001"
        assert recommendation.supplier_id.value == "SUP-001"
        assert recommendation.recommended_order_qty.amount == Decimal("200")
        assert recommendation.total_cost.amount == Decimal("1000.00")
        assert recommendation.risk_flag == RiskLevel.LOW
        assert recommendation.urgency_score == 0.3