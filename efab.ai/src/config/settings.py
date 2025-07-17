from decimal import Decimal
from typing import Dict, Any
from datetime import timedelta

class PlanningConfig:
    SOURCE_WEIGHTS = {
        "sales_order": 1.0,
        "prod_plan": 0.9,
        "projection": 0.7,
        "sales_history": 0.8
    }
    
    SAFETY_STOCK_PERCENTAGE = 0.15
    PLANNING_HORIZON_DAYS = 90
    FORECAST_LOOKBACK_DAYS = 30
    
    COST_WEIGHT = 0.6
    RELIABILITY_WEIGHT = 0.4
    MAX_SUPPLIERS_PER_MATERIAL = 3
    
    ENABLE_EOQ_OPTIMIZATION = True
    ENABLE_MULTI_SUPPLIER = True
    ENABLE_RISK_ASSESSMENT = True
    
    DEFAULT_ORDERING_COST = Decimal("100.0")
    DEFAULT_HOLDING_COST_RATE = 0.25
    
    RISK_THRESHOLDS = {
        "high": 0.7,
        "medium": 0.85,
        "low": 1.0
    }
    
    MATERIAL_CATEGORIES = {
        "yarn": {"safety_buffer": 0.15, "critical_threshold": 0.2},
        "fabric": {"safety_buffer": 0.10, "critical_threshold": 0.15},
        "thread": {"safety_buffer": 0.20, "critical_threshold": 0.25},
        "accessory": {"safety_buffer": 0.05, "critical_threshold": 0.10},
        "trim": {"safety_buffer": 0.10, "critical_threshold": 0.15}
    }
    
    SUPPLIER_PERFORMANCE_TIERS = {
        "premium": {"reliability_min": 0.95, "cost_multiplier": 1.1},
        "standard": {"reliability_min": 0.85, "cost_multiplier": 1.0},
        "budget": {"reliability_min": 0.70, "cost_multiplier": 0.9}
    }
    
    SEASONAL_ADJUSTMENTS = {
        "Q1": 0.9,
        "Q2": 1.1,
        "Q3": 1.0,
        "Q4": 1.2
    }
    
    UNIT_CONVERSIONS = {
        ("lb", "kg"): 0.453592,
        ("kg", "lb"): 2.20462,
        ("yard", "meter"): 0.9144,
        ("meter", "yard"): 1.09361
    }

PLANNING_CONFIG = PlanningConfig()