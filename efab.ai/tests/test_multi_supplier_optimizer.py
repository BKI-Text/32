import pytest
from decimal import Decimal

from src.engine.multi_supplier_optimizer import (
    MultiSupplierOptimizer, SupplierScore, SourcingRecommendation, SourcingStrategy
)
from src.core.domain import (
    SupplierMaterial, Money, Quantity, MaterialId, SupplierId, LeadTime, RiskLevel
)

class TestMultiSupplierOptimizer:
    def setup_method(self):
        self.optimizer = MultiSupplierOptimizer(
            cost_weight=0.4,
            reliability_weight=0.3,
            lead_time_weight=0.3
        )
        
        self.material_id = MaterialId(value="YARN-001")
        self.demand = Quantity(amount=Decimal("500"), unit="lb")
        
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
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("4.50"), currency="USD"),
                moq=Quantity(amount=Decimal("150"), unit="lb"),
                lead_time=LeadTime(days=21),
                reliability_score=0.75
            ),
            SupplierMaterial(
                supplier_id=SupplierId(value="SUP-003"),
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("6.00"), currency="USD"),
                moq=Quantity(amount=Decimal("75"), unit="lb"),
                lead_time=LeadTime(days=7),
                reliability_score=0.95
            )
        ]
    
    def test_weight_validation(self):
        # Test that weights must sum to 1.0
        with pytest.raises(ValueError):
            MultiSupplierOptimizer(
                cost_weight=0.5,
                reliability_weight=0.3,
                lead_time_weight=0.5  # Total = 1.3, should fail
            )
    
    def test_score_suppliers(self):
        scores = self.optimizer.score_suppliers(self.test_suppliers)
        
        assert len(scores) == 3
        assert all(isinstance(score, SupplierScore) for score in scores)
        
        # Check that all scores are between 0 and 1
        for score in scores:
            assert 0 <= score.cost_score <= 1
            assert 0 <= score.reliability_score <= 1
            assert 0 <= score.lead_time_score <= 1
            assert 0 <= score.total_score <= 1
        
        # Scores should be sorted by total score (descending)
        for i in range(len(scores) - 1):
            assert scores[i].total_score >= scores[i + 1].total_score
    
    def test_score_suppliers_single_supplier(self):
        scores = self.optimizer.score_suppliers([self.test_suppliers[0]])
        
        assert len(scores) == 1
        # With single supplier, all scores should be 1.0
        assert scores[0].cost_score == 1.0
        assert scores[0].reliability_score == 1.0
        assert scores[0].lead_time_score == 1.0
        assert scores[0].total_score == 1.0
    
    def test_score_suppliers_empty_list(self):
        scores = self.optimizer.score_suppliers([])
        assert scores == []
    
    def test_determine_sourcing_strategy_single_supplier(self):
        strategy = self.optimizer.determine_sourcing_strategy(
            self.material_id,
            self.demand,
            [self.test_suppliers[0]]
        )
        
        assert strategy == SourcingStrategy.SINGLE_SOURCE
    
    def test_determine_sourcing_strategy_significant_gap(self):
        # Create suppliers with significant performance gap
        dominant_supplier = SupplierMaterial(
            supplier_id=SupplierId(value="SUP-DOMINANT"),
            material_id=MaterialId(value="YARN-001"),
            cost_per_unit=Money(amount=Decimal("1.00"), currency="USD"),  # Much cheaper
            moq=Quantity(amount=Decimal("50"), unit="lb"),
            lead_time=LeadTime(days=5),  # Much faster
            reliability_score=0.99  # Much more reliable
        )
        
        suppliers = [dominant_supplier, self.test_suppliers[0]]
        
        strategy = self.optimizer.determine_sourcing_strategy(
            self.material_id,
            self.demand,
            suppliers
        )
        
        assert strategy == SourcingStrategy.SINGLE_SOURCE
    
    def test_determine_sourcing_strategy_multi_source(self):
        # Create suppliers with similar performance
        similar_suppliers = []
        for i in range(3):
            supplier = SupplierMaterial(
                supplier_id=SupplierId(value=f"SUP-{i+1}"),
                material_id=MaterialId(value="YARN-001"),
                cost_per_unit=Money(amount=Decimal("5.00") + Decimal("0.1") * i, currency="USD"),
                moq=Quantity(amount=Decimal("100"), unit="lb"),
                lead_time=LeadTime(days=14 + i),
                reliability_score=0.85 + 0.01 * i
            )
            similar_suppliers.append(supplier)
        
        strategy = self.optimizer.determine_sourcing_strategy(
            self.material_id,
            self.demand,
            similar_suppliers
        )
        
        assert strategy == SourcingStrategy.MULTI_SOURCE
    
    def test_optimize_sourcing_single_source(self):
        # Use only one supplier to force single source
        recommendation = self.optimizer.optimize_sourcing(
            self.material_id,
            self.demand,
            [self.test_suppliers[0]]
        )
        
        assert isinstance(recommendation, SourcingRecommendation)
        assert recommendation.strategy == SourcingStrategy.SINGLE_SOURCE
        assert len(recommendation.allocations) == 1
        assert self.test_suppliers[0].supplier_id in recommendation.allocations
        assert recommendation.allocations[self.test_suppliers[0].supplier_id] == self.demand
    
    def test_optimize_sourcing_dual_source(self):
        # Use two suppliers with moderate performance gap
        recommendation = self.optimizer.optimize_sourcing(
            self.material_id,
            self.demand,
            self.test_suppliers[:2]
        )
        
        assert isinstance(recommendation, SourcingRecommendation)
        assert len(recommendation.allocations) == 2
        
        # Verify allocations sum to total demand
        total_allocated = sum(qty.amount for qty in recommendation.allocations.values())
        assert abs(total_allocated - self.demand.amount) < Decimal("0.01")
    
    def test_optimize_sourcing_empty_suppliers(self):
        recommendation = self.optimizer.optimize_sourcing(
            self.material_id,
            self.demand,
            []
        )
        
        assert recommendation.strategy == SourcingStrategy.SINGLE_SOURCE
        assert len(recommendation.allocations) == 0
        assert recommendation.risk_assessment == RiskLevel.HIGH
        assert "No suppliers available" in recommendation.reasoning
    
    def test_calculate_total_cost(self):
        allocations = {
            self.test_suppliers[0].supplier_id: Quantity(amount=Decimal("300"), unit="lb"),
            self.test_suppliers[1].supplier_id: Quantity(amount=Decimal("200"), unit="lb")
        }
        
        total_cost = self.optimizer._calculate_total_cost(allocations, self.test_suppliers)
        
        expected_cost = (
            Decimal("300") * self.test_suppliers[0].cost_per_unit.amount +
            Decimal("200") * self.test_suppliers[1].cost_per_unit.amount
        )
        
        assert total_cost.amount == expected_cost
        assert total_cost.currency == "USD"
    
    def test_assess_sourcing_risk_single_source(self):
        # Test risk assessment for single source
        high_reliability_scores = [
            SupplierScore(
                supplier_id=SupplierId(value="SUP-001"),
                cost_score=0.8,
                reliability_score=0.95,
                lead_time_score=0.7,
                total_score=0.8
            )
        ]
        
        risk = self.optimizer._assess_sourcing_risk(
            SourcingStrategy.SINGLE_SOURCE,
            high_reliability_scores
        )
        
        assert risk == RiskLevel.MEDIUM  # Single source with high reliability
    
    def test_assess_sourcing_risk_multi_source(self):
        # Test risk assessment for multi source
        medium_reliability_scores = [
            SupplierScore(
                supplier_id=SupplierId(value="SUP-001"),
                cost_score=0.8,
                reliability_score=0.75,
                lead_time_score=0.7,
                total_score=0.8
            ),
            SupplierScore(
                supplier_id=SupplierId(value="SUP-002"),
                cost_score=0.7,
                reliability_score=0.80,
                lead_time_score=0.6,
                total_score=0.7
            )
        ]
        
        risk = self.optimizer._assess_sourcing_risk(
            SourcingStrategy.MULTI_SOURCE,
            medium_reliability_scores
        )
        
        assert risk == RiskLevel.LOW  # Multi-source reduces risk
    
    def test_generate_reasoning(self):
        supplier_scores = [
            SupplierScore(
                supplier_id=SupplierId(value="SUP-001"),
                cost_score=0.8,
                reliability_score=0.85,
                lead_time_score=0.7,
                total_score=0.8
            )
        ]
        
        allocations = {SupplierId(value="SUP-001"): Quantity(amount=Decimal("500"), unit="lb")}
        
        reasoning = self.optimizer._generate_reasoning(
            SourcingStrategy.SINGLE_SOURCE,
            supplier_scores,
            allocations
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "SUP-001" in reasoning
        assert "Single source" in reasoning
    
    def test_generate_allocations_single_source(self):
        supplier_scores = [
            SupplierScore(
                supplier_id=SupplierId(value="SUP-001"),
                cost_score=0.8,
                reliability_score=0.85,
                lead_time_score=0.7,
                total_score=0.8
            )
        ]
        
        allocations = self.optimizer._generate_allocations(
            SourcingStrategy.SINGLE_SOURCE,
            self.demand,
            self.test_suppliers,
            supplier_scores
        )
        
        assert len(allocations) == 1
        assert supplier_scores[0].supplier_id in allocations
        assert allocations[supplier_scores[0].supplier_id] == self.demand
    
    def test_generate_allocations_dual_source(self):
        supplier_scores = [
            SupplierScore(
                supplier_id=SupplierId(value="SUP-001"),
                cost_score=0.8,
                reliability_score=0.85,
                lead_time_score=0.7,
                total_score=0.8
            ),
            SupplierScore(
                supplier_id=SupplierId(value="SUP-002"),
                cost_score=0.7,
                reliability_score=0.75,
                lead_time_score=0.6,
                total_score=0.7
            )
        ]
        
        allocations = self.optimizer._generate_allocations(
            SourcingStrategy.DUAL_SOURCE,
            self.demand,
            self.test_suppliers,
            supplier_scores
        )
        
        assert len(allocations) == 2
        
        # Verify allocations sum to total demand
        total_allocated = sum(qty.amount for qty in allocations.values())
        assert abs(total_allocated - self.demand.amount) < Decimal("0.01")
        
        # Primary supplier should get at least 60%
        primary_allocation = allocations[supplier_scores[0].supplier_id]
        assert primary_allocation.amount >= self.demand.amount * Decimal("0.6")
    
    def test_generate_allocations_multi_source(self):
        supplier_scores = [
            SupplierScore(
                supplier_id=SupplierId(value="SUP-001"),
                cost_score=0.8,
                reliability_score=0.85,
                lead_time_score=0.7,
                total_score=0.8
            ),
            SupplierScore(
                supplier_id=SupplierId(value="SUP-002"),
                cost_score=0.7,
                reliability_score=0.75,
                lead_time_score=0.6,
                total_score=0.7
            ),
            SupplierScore(
                supplier_id=SupplierId(value="SUP-003"),
                cost_score=0.6,
                reliability_score=0.65,
                lead_time_score=0.5,
                total_score=0.6
            )
        ]
        
        allocations = self.optimizer._generate_allocations(
            SourcingStrategy.MULTI_SOURCE,
            self.demand,
            self.test_suppliers,
            supplier_scores
        )
        
        assert len(allocations) == 3
        
        # Verify allocations sum to total demand
        total_allocated = sum(qty.amount for qty in allocations.values())
        assert abs(total_allocated - self.demand.amount) < Decimal("0.01")
        
        # Verify allocation ratios (50%, 30%, 20%)
        expected_ratios = [0.5, 0.3, 0.2]
        for i, (supplier_id, quantity) in enumerate(allocations.items()):
            expected_amount = self.demand.amount * Decimal(str(expected_ratios[i]))
            assert abs(quantity.amount - expected_amount) < Decimal("0.01")