"""
Phase 3: Baseline Models Configuration
Add these sections to your existing config.py
"""

from pathlib import Path

# ============================================================================
# PHASE 3: PATHS (add to config.py)
# ============================================================================

PHASE3_DIR = PROJECT_ROOT / "phase3_models"
PHASE3_DATA_DIR = PHASE3_DIR / "data"
PHASE3_MODELS_DIR = PHASE3_DIR / "models"
PHASE3_RESULTS_DIR = PHASE3_DIR / "results"
PHASE3_LOGS_DIR = PHASE3_DIR / "logs"

# Create directories on import
for p in [PHASE3_DIR, PHASE3_DATA_DIR, PHASE3_MODELS_DIR, PHASE3_RESULTS_DIR, PHASE3_LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Input: Phase 2 outputs
FEATURES_TRAIN_CSV = Path("data/processed/features_train.csv")
FEATURES_TEST_CSV = Path("data/processed/features_test.csv")

# Output: Trained models
MODEL_LR_PATH = PHASE3_MODELS_DIR / "logistic_regression.pkl"
MODEL_XGBOOST_PATH = PHASE3_MODELS_DIR / "xgboost_model.pkl"
MODEL_LIGHTGBM_PATH = PHASE3_MODELS_DIR / "lightgbm_model.pkl"

# Output: Scalers and encoders
SCALER_LR_PATH = PHASE3_MODELS_DIR / "scaler_logistic_regression.pkl"
LABEL_ENCODER_PATH = PHASE3_MODELS_DIR / "label_encoder_strength_state.pkl"

# Output: Results and reports
RESULTS_METRICS_JSON = PHASE3_RESULTS_DIR / "model_metrics.json"
RESULTS_COMPARISON_CSV = PHASE3_RESULTS_DIR / "model_comparison.csv"
RESULTS_FEATURE_IMPORTANCE_CSV = PHASE3_RESULTS_DIR / "feature_importance.csv"
RESULTS_LOG_FILE = PHASE3_LOGS_DIR / "phase3_training.log"

# ============================================================================
# PHASE 3: FEATURE LISTS (add to config.py)
# ============================================================================

# Exclude from modeling (identifiers and target)
EXCLUDE_COLUMNS = {'game_id', 'shot_id', 'target_home_win'}

# All numeric features (after encoding categorical)
NUMERIC_FEATURES = [
    'season', 'period', 'is_playoff_game',
    'time_elapsed', 'game_seconds_remaining',
    'score_differential', 'xg_differential', 'shot_differential',
    'is_even_strength', 'is_power_play', 'is_empty_net',
    'time_remaining_normalized',
    'is_late_game', 'is_close_game',
    'is_period_1', 'is_period_2', 'is_period_3plus',
    'game_phase',
    'score_time_interaction', 'xg_time_interaction',
    'is_clutch_moment', 'powerplay_score_interaction'
]

# Categorical features (need encoding)
CATEGORICAL_FEATURES = ['strength_state']

# Baseline minimal feature set (for quick experiments)
BASELINE_FEATURES = [
    'score_differential',
    'xg_differential',
    'time_remaining_normalized',
    'is_close_game',
    'is_late_game'
]

# Standard feature set (recommended)
STANDARD_FEATURES = [
    'score_differential', 'xg_differential', 'shot_differential',
    'time_remaining_normalized', 'is_late_game', 'is_close_game',
    'is_period_1', 'is_period_2', 'is_period_3plus',
    'score_time_interaction', 'xg_time_interaction',
    'is_clutch_moment'
]

# ============================================================================
# PHASE 3: MODEL HYPERPARAMETERS (add to config.py)
# ============================================================================

LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced',
    'solver': 'lbfgs',
    'n_jobs': -1,  # parallel
}

XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'verbose': 0,
}

LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# ============================================================================
# PHASE 3: EVALUATION METRICS (add to config.py)
# ============================================================================

# Primary metrics for comparison
PRIMARY_METRICS = ['log_loss', 'roc_auc', 'brier_score', 'accuracy']

# Thresholds for "good" performance
PERFORMANCE_THRESHOLDS = {
    'log_loss': 0.69,        # Better than random (log loss = 0.693 for random)
    'roc_auc': 0.60,         # Meaningful discrimination
    'brier_score': 0.25,     # Better than naive prediction
    'accuracy': 0.58,        # Better than baseline (54.7% home wins)
}

# ============================================================================
# PHASE 3: CROSS-VALIDATION (add to config.py)
# ============================================================================

CV_FOLDS = 5
CV_RANDOM_STATE = 42

# ============================================================================
# PHASE 3: LOGGING (add to config.py)
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PHASE3_LOGS_DIR / "phase3_training.log"

