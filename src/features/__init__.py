"""
Phase 2: Feature Engineering with Polars
==========================================

Main exports:
- FeatureEngineer: Core feature engineering class
- quick_feature_engineering: One-liner pipeline
- FeatureValidator: Validation class
"""

from feature_engineer import (
    FeatureEngineer,
    quick_feature_engineering
)

from validate_features import (
    FeatureValidator,
    validate_features_file
)


__all__ = [
    'FeatureEngineer',
    'quick_feature_engineering',
    'FeatureValidator',
    'validate_features_file',
]