"""
Phase 3: Diagnostic Script
Location: phase3_diagnostics.py

Investigates:
1. Feature data quality
2. Data leakage issues
3. Feature correlations with target
4. Class balance
5. Why model performance is poor
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Config
FEATURES_TRAIN_CSV = DATA_PROCESSED / "features_train.csv"
FEATURES_TEST_CSV = DATA_PROCESSED / "features_test.csv"
EXCLUDE_COLUMNS = {'game_id_unique', 'shot_id', 'target_home_win'}

def load_data():
    """Load Phase 2 features."""
    try:
        import polars as pl
        train_df = pl.read_csv(FEATURES_TRAIN_CSV).to_pandas()
        test_df = pl.read_csv(FEATURES_TEST_CSV).to_pandas()
    except ImportError:
        train_df = pd.read_csv(FEATURES_TRAIN_CSV)
        test_df = pd.read_csv(FEATURES_TEST_CSV)
    
    return train_df, test_df

def diagnose_features(train_df):
    """Diagnose feature quality."""
    logger.info("\n" + "=" * 100)
    logger.info("FEATURE DIAGNOSTICS")
    logger.info("=" * 100)
    
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLUMNS]
    X = train_df[feature_cols]
    y = train_df['target_home_win']
    
    # 1. Feature types
    logger.info("\n1. FEATURE DATA TYPES")
    logger.info("-" * 100)
    dtype_summary = X.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        logger.info(f"  {dtype}: {count} features")
    
    # 2. Null values
    logger.info("\n2. NULL VALUES")
    logger.info("-" * 100)
    null_count = X.isnull().sum().sum()
    logger.info(f"  Total nulls: {null_count}")
    if null_count > 0:
        logger.warning(f"  ⚠ WARNING: Nulls detected! This will break sklearn models")
        logger.info(f"  Columns with nulls:")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                logger.info(f"    {col}: {X[col].isnull().sum()} nulls")
    else:
        logger.info(f"  ✓ No null values")
    
    # 3. Feature variance
    logger.info("\n3. FEATURE VARIANCE (detecting zero-variance features)")
    logger.info("-" * 100)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    variances = X[numeric_cols].var()
    zero_var = variances[variances == 0]
    if len(zero_var) > 0:
        logger.warning(f"  ⚠ WARNING: {len(zero_var)} features with zero variance:")
        for col in zero_var.index:
            logger.warning(f"    {col}")
    else:
        logger.info(f"  ✓ All features have non-zero variance")
    
    # 4. Class balance
    logger.info("\n4. CLASS BALANCE (Target Distribution)")
    logger.info("-" * 100)
    home_win_pct = y.mean()
    logger.info(f"  Home wins: {home_win_pct:.1%}")
    logger.info(f"  Away wins: {(1-home_win_pct):.1%}")
    if abs(home_win_pct - 0.5) < 0.05:
        logger.warning(f"  ⚠ WARNING: Nearly balanced target suggests models may struggle")
    
    # 5. Feature correlation with target
    logger.info("\n5. FEATURE CORRELATION WITH TARGET")
    logger.info("-" * 100)
    correlations = {}
    
    # Encode categorical first
    X_encoded = X.copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
    
    # Compute correlations
    for col in X_encoded.columns:
        corr = X_encoded[col].corr(y)
        correlations[col] = corr
    
    corr_df = pd.DataFrame(list(correlations.items()), columns=['feature', 'correlation'])
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_corr', ascending=False)
    
    logger.info(f"  Top 10 features by correlation with target:")
    for idx, row in corr_df.head(10).iterrows():
        logger.info(f"    {row['feature']:30s}: {row['correlation']:+.4f}")
    
    weak_correlations = corr_df[corr_df['abs_corr'] < 0.01]
    if len(weak_correlations) > 0:
        logger.warning(f"  ⚠ WARNING: {len(weak_correlations)} features with |corr| < 0.01 (very weak)")
        logger.info(f"  These features may not be predictive:")
        for idx, row in weak_correlations.iterrows():
            logger.info(f"    {row['feature']}: {row['correlation']:+.4f}")
    
    return corr_df

def diagnose_data_leakage(train_df, test_df):
    """Check for data leakage issues."""
    logger.info("\n" + "=" * 100)
    logger.info("DATA LEAKAGE DIAGNOSTICS")
    logger.info("=" * 100)
    
    # 1. Check for duplicate games
    logger.info("\n1. CHECKING FOR DUPLICATE GAMES")
    logger.info("-" * 100)
    
    train_games = set(train_df['game_id_unique'].unique())
    test_games = set(test_df['game_id_unique'].unique())
    overlap = train_games & test_games
    
    if len(overlap) > 0:
        logger.error(f"  ✗ CRITICAL: {len(overlap)} games appear in both train and test!")
        logger.error(f"  This is data leakage - models will overfit")
    else:
        logger.info(f"  ✓ No game overlap between train and test")
    
    # 2. Check temporal ordering
    logger.info("\n2. CHECKING TEMPORAL ORDERING")
    logger.info("-" * 100)
    
    train_seasons = sorted(train_df['season'].unique())
    test_seasons = sorted(test_df['season'].unique())
    
    logger.info(f"  Train seasons: {train_seasons[0]}-{train_seasons[-1]}")
    logger.info(f"  Test seasons:  {test_seasons[0]}-{test_seasons[-1]}")
    
    if train_seasons[-1] >= test_seasons[0]:
        logger.warning(f"  ⚠ WARNING: Train and test seasons overlap!")
        logger.warning(f"  Train max season: {train_seasons[-1]}")
        logger.warning(f"  Test min season:  {test_seasons[0]}")
    else:
        logger.info(f"  ✓ Proper temporal split (train before test)")
    
    # 3. Check target distribution shift
    logger.info("\n3. CHECKING TARGET DISTRIBUTION SHIFT")
    logger.info("-" * 100)
    
    train_home_pct = train_df['target_home_win'].mean()
    test_home_pct = test_df['target_home_win'].mean()
    
    logger.info(f"  Train: {train_home_pct:.1%} home wins")
    logger.info(f"  Test:  {test_home_pct:.1%} home wins")
    logger.info(f"  Diff:  {abs(train_home_pct - test_home_pct):.1%}")
    
    if abs(train_home_pct - test_home_pct) > 0.05:
        logger.warning(f"  ⚠ WARNING: Target distribution shift detected")
        logger.warning(f"  This can indicate temporal/seasonal effects")

def diagnose_feature_space(train_df):
    """Analyze feature space."""
    logger.info("\n" + "=" * 100)
    logger.info("FEATURE SPACE DIAGNOSTICS")
    logger.info("=" * 100)
    
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLUMNS]
    X = train_df[feature_cols]
    
    # 1. Feature scales
    logger.info("\n1. FEATURE SCALES (Min/Max)")
    logger.info("-" * 100)
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:10]:  # Show first 10
        min_val = X[col].min()
        max_val = X[col].max()
        logger.info(f"  {col:30s}: [{min_val:10.2f}, {max_val:10.2f}]")
    
    # 2. Categorical features
    logger.info("\n2. CATEGORICAL FEATURES")
    logger.info("-" * 100)
    
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        unique_vals = X[col].unique()
        logger.info(f"  {col}: {len(unique_vals)} unique values")
        logger.info(f"    Values: {sorted(unique_vals)}")

def main():
    """Run all diagnostics."""
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 3: FEATURE & MODEL DIAGNOSTICS")
    logger.info("=" * 100)
    
    try:
        # Load data
        logger.info("\nLoading data...")
        train_df, test_df = load_data()
        logger.info(f"✓ Train: {len(train_df):,} rows")
        logger.info(f"✓ Test:  {len(test_df):,} rows")
        
        # Diagnostics
        diagnose_features(train_df)
        diagnose_data_leakage(train_df, test_df)
        diagnose_feature_space(train_df)
        
        logger.info("\n" + "=" * 100)
        logger.info("DIAGNOSTICS COMPLETE")
        logger.info("=" * 100)
        
        logger.info("\nIMPORTANT FINDINGS:")
        logger.info("-" * 100)
        logger.info("1. Check feature correlations with target above")
        logger.info("2. If many features have |corr| < 0.05, features may be weak")
        logger.info("3. If no data leakage detected, issue is feature quality")
        logger.info("4. If ROC-AUC ≈ 0.50, model is basically guessing")
        logger.info("\nNext steps:")
        logger.info("1. Check if features are computed correctly in Phase 2")
        logger.info("2. Verify xG values are present in original data")
        logger.info("3. Check score_differential calculation")
        logger.info("4. Confirm temporal split is correct")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)