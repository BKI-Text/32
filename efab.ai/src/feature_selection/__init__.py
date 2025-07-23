"""
Automated Feature Selection Module
Beverly Knits AI Supply Chain Planner
"""

from .automated_feature_selection import (
    AutomatedFeatureSelector,
    FeatureSelectionPipeline,
    feature_selection_pipeline,
    run_feature_selection,
    get_feature_selection_pipeline
)

__all__ = [
    'AutomatedFeatureSelector',
    'FeatureSelectionPipeline',
    'feature_selection_pipeline',
    'run_feature_selection',
    'get_feature_selection_pipeline'
]