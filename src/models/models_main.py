"""
Phase 3: Baseline Models - Main Orchestrator
Location: phase3_main.py

Single entry point that:
1. Loads Phase 2 features (features_train.csv, features_test.csv)
2. Trains three baseline models (LR, XGBoost, LightGBM)
3. Evaluates and compares performance
4. Saves model artifacts
5. Generates comparison reports

Usage:
    python models_main.py

Expected output:
    - Trained models in phase3_models/models/
    - Comparison metrics in phase3_models/results/
    - Training logs in phase3_models/logs/
"""

import logging
import sys
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

# Import model training functions
# (In production, update these import paths to match your project structure)
try:
    from baseline import train_logistic_regression, LogisticRegressionBaseline
    from xgboost_model import train_xgboost, XGBoostModel
    from lightgbm_model import train_lightgbm, LightGBMModel
    from evaluate import ModelComparison, validate_model_performance, generate_report
except ImportError as e:
    print(f"ERROR: Could not import model modules: {e}")
    print("Make sure baseline.py, xgboost_model.py, lightgbm_model.py, and evaluate.py are in the same directory")
    sys.exit(1)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Uncomment to also log to file:
        # logging.FileHandler('phase3_training.log')
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (Replace with actual config imports in production)
# ============================================================================

# Input paths (Phase 2 outputs)
FEATURES_TRAIN_CSV = Path("data/processed/features_train.csv")
FEATURES_TEST_CSV = Path("data/processed/features_test.csv")

# Output paths
OUTPUT_DIR = Path("phase3_models")
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories
for d in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced',
}

XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
}

LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}

# Feature engineering
EXCLUDE_COLUMNS = {'game_id', 'shot_id', 'target_home_win'}
CATEGORICAL_FEATURES = ['strength_state']  # Will be label-encoded

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'log_loss': 0.69,
    'roc_auc': 0.60,
    'brier_score': 0.25,
    'accuracy': 0.58,
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load Phase 2 feature matrices.
    
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 100)
    logger.info("PHASE 3: BASELINE MODELS")
    logger.info("=" * 100)
    logger.info(f"\nLoading Phase 2 features from:\n  {FEATURES_TRAIN_CSV}\n  {FEATURES_TEST_CSV}")
    
    # Verify files exist
    if not FEATURES_TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training features not found: {FEATURES_TRAIN_CSV}")
    if not FEATURES_TEST_CSV.exists():
        raise FileNotFoundError(f"Test features not found: {FEATURES_TEST_CSV}")
    
    # Load with polars or pandas
    try:
        import polars as pl
        train_df = pl.read_csv(FEATURES_TRAIN_CSV).to_pandas()
        test_df = pl.read_csv(FEATURES_TEST_CSV).to_pandas()
        logger.info("✓ Features loaded with Polars")
    except ImportError:
        train_df = pd.read_csv(FEATURES_TRAIN_CSV)
        test_df = pd.read_csv(FEATURES_TEST_CSV)
        logger.info("✓ Features loaded with Pandas")
    
    # Verify shape
    logger.info(f"\nTrain set: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    logger.info(f"Test set:  {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
    
    # Separate features and target
    target_col = 'target_home_win'
    
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Drop excluded columns
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLUMNS]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Verify alignment
    assert len(X_train) == len(y_train), "Train features/target mismatch"
    assert len(X_test) == len(y_test), "Test features/target mismatch"
    
    # Data summary
    logger.info(f"\nFeatures: {len(feature_cols)}")
    logger.info(f"Target balance (train): {y_train.mean():.1%} home wins (expected: ~54.7%)")
    logger.info(f"Target balance (test):  {y_test.mean():.1%} home wins")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_all_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> ModelComparison:
    """
    Train all three baseline models.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        ModelComparison instance with all models and metrics
    """
    comparison = ModelComparison()
    
    # ========================================================================
    # Model 1: Logistic Regression
    # ========================================================================
    try:
        logger.info("\n" + "=" * 100)
        model_lr, metrics_lr = train_logistic_regression(
            X_train, y_train, X_test, y_test,
            hyperparams=LOGISTIC_REGRESSION_PARAMS
        )
        comparison.add_model("Logistic Regression", model_lr, metrics_lr)
        
        # Validate performance
        validate_model_performance(metrics_lr, PERFORMANCE_THRESHOLDS, "Logistic Regression")
        
        # Save model
        model_lr.save(
            str(MODELS_DIR / "logistic_regression.pkl"),
            str(MODELS_DIR / "scaler_logistic_regression.pkl")
        )
    except Exception as e:
        logger.error(f"Failed to train Logistic Regression: {e}", exc_info=True)
    
    # ========================================================================
    # Model 2: XGBoost
    # ========================================================================
    try:
        logger.info("\n" + "=" * 100)
        model_xgb, metrics_xgb = train_xgboost(
            X_train, y_train, X_test, y_test,
            hyperparams=XGBOOST_PARAMS,
            categorical_features=CATEGORICAL_FEATURES
        )
        comparison.add_model("XGBoost", model_xgb, metrics_xgb)
        
        # Validate performance
        validate_model_performance(metrics_xgb, PERFORMANCE_THRESHOLDS, "XGBoost")
        
        # Save model
        model_xgb.save(
            str(MODELS_DIR / "xgboost_model.json"),
            str(MODELS_DIR / "xgboost_metadata.pkl")
        )
    except Exception as e:
        logger.error(f"Failed to train XGBoost: {e}", exc_info=True)
    
    # ========================================================================
    # Model 3: LightGBM
    # ========================================================================
    try:
        logger.info("\n" + "=" * 100)
        model_lgb, metrics_lgb = train_lightgbm(
            X_train, y_train, X_test, y_test,
            hyperparams=LIGHTGBM_PARAMS,
            categorical_features=CATEGORICAL_FEATURES
        )
        comparison.add_model("LightGBM", model_lgb, metrics_lgb)
        
        # Validate performance
        validate_model_performance(metrics_lgb, PERFORMANCE_THRESHOLDS, "LightGBM")
        
        # Save model
        model_lgb.save(
            str(MODELS_DIR / "lightgbm_model.txt"),
            str(MODELS_DIR / "lightgbm_metadata.pkl")
        )
    except Exception as e:
        logger.error(f"Failed to train LightGBM: {e}", exc_info=True)
    
    return comparison


# ============================================================================
# REPORTING
# ============================================================================

def generate_reports(comparison: ModelComparison) -> None:
    """
    Generate comparison reports and save artifacts.
    
    Args:
        comparison: ModelComparison instance
    """
    logger.info("\n" + "=" * 100)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 100)
    
    # Print summary
    comparison.print_summary()
    
    # Save comparison CSV
    comparison.save_comparison(str(RESULTS_DIR / "model_comparison.csv"))
    
    # Save metrics JSON
    comparison.save_metrics_json(str(RESULTS_DIR / "model_metrics.json"))
    
    # Save feature importance
    comparison.save_feature_importance(str(RESULTS_DIR / "feature_importance.csv"))
    
    # Generate text report
    comparison_df = comparison.get_comparison_table()
    best_model_name = comparison_df['log_loss'].idxmin()  # Best by log loss
    generate_report(comparison, best_model_name, str(RESULTS_DIR))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main orchestration function."""
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_features()
        
        # Train models
        comparison = train_all_models(X_train, y_train, X_test, y_test)
        
        # Generate reports
        generate_reports(comparison)
        
        # Final summary
        logger.info("\n" + "=" * 100)
        logger.info("PHASE 3 COMPLETE")
        logger.info("=" * 100)
        logger.info(f"\nOutput files:")
        logger.info(f"  Models:           {MODELS_DIR}")
        logger.info(f"  Comparison CSV:   {RESULTS_DIR / 'model_comparison.csv'}")
        logger.info(f"  Metrics JSON:     {RESULTS_DIR / 'model_metrics.json'}")
        logger.info(f"  Feature Import:   {RESULTS_DIR / 'feature_importance.csv'}")
        logger.info(f"  Full Report:      {RESULTS_DIR / 'phase3_report.txt'}")
        logger.info("\n✓ Ready for Phase 5: Calibration")
        logger.info("=" * 100)
        
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error in Phase 3: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)