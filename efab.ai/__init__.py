"""
Beverly Knits AI Supply Chain Optimization Planner

An intelligent, AI-driven supply chain planning system for textile manufacturing.
Transform your raw material procurement from reactive guesswork to proactive, 
data-driven optimization.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Beverly Knits AI Team"
__description__ = "AI-driven supply chain optimization for textile manufacturing"

# Core imports for easy access
from src.engine import PlanningEngine
from src.data import DataIntegrator
from src.utils import generate_sample_data
from src.config import get_config

__all__ = [
    "PlanningEngine",
    "DataIntegrator", 
    "generate_sample_data",
    "get_config"
]