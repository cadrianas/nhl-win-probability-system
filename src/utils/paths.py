"""
Centralized path management for the NHL Win Probability project.
This module provides a single source of truth for all file paths.
Ensures the project works from any working directory.

Usage:
    from src.utils.paths import DATA_PROCESSED, RESULTS_MODELS, ensure_directories
    ensure_directories()  # Create all dirs on startup
"""

from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================

# Project root: src/utils/paths.py -> src -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# ============================================================================
# DATA DIRECTORIES
# ============================================================================

DATA = PROJECT_ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"

# ============================================================================
# RESULTS DIRECTORIES
# ============================================================================

RESULTS = PROJECT_ROOT / "results"
RESULTS_MODELS = RESULTS / "models"           # Trained models & calibration
RESULTS_LOGS = RESULTS / "logs"               # Training logs
RESULTS_PLOTS = RESULTS / "plots"             # Visualizations & diagrams
RESULTS_METRICS = RESULTS / "metrics"         # Evaluation metrics

# ============================================================================
# PROJECT STRUCTURE DIRECTORIES
# ============================================================================

NOTEBOOKS = PROJECT_ROOT / "notebooks"
CONFIGS = PROJECT_ROOT / "configs"
SRC = PROJECT_ROOT / "src"

# ============================================================================
# SOURCE CODE MODULES
# ============================================================================

SRC_PREPROCESSING = SRC / "preprocessing"
SRC_FEATURES = SRC / "features"
SRC_MODELS = SRC / "models"
SRC_TRAINING = SRC / "training"
SRC_CALIBRATION = SRC / "calibration"
SRC_VISUALIZATION = SRC / "visualization"
SRC_UTILITIES = SRC / "utils"

# ============================================================================
# SPECIFIC FILE PATHS (Phase 4 onwards)
# ============================================================================

# Phase 4: Model Training
MODEL_XGBOOST_BEST = RESULTS_MODELS / "xgboost_best_model.pkl"
MODEL_XGBOOST_FINAL = RESULTS_MODELS / "xgboost_final.pkl"
MODEL_LR_BASELINE = RESULTS_MODELS / "logistic_regression_baseline.pkl"
MODEL_LGBM = RESULTS_MODELS / "lightgbm_model.pkl"

# Phase 5: Calibration
MODEL_XGBOOST_CALIBRATED_ISOTONIC = RESULTS_MODELS / "xgboost_calibrated_isotonic.pkl"
MODEL_XGBOOST_CALIBRATED_SIGMOID = RESULTS_MODELS / "xgboost_calibrated_sigmoid.pkl"

CALIBRATION_METRICS = RESULTS_METRICS / "calibration_metrics.json"
CALIBRATION_METRICS_CONTEXT = RESULTS_METRICS / "calibration_metrics_by_context.json"
CALIBRATION_PLOT = RESULTS_PLOTS / "calibration_curves_comparison.png"

# Phase 3: Feature Engineering
FEATURES_TRAIN = DATA_PROCESSED / "features_train.csv"
FEATURES_TEST = DATA_PROCESSED / "features_test.csv"
FEATURES_VALIDATION = DATA_PROCESSED / "features_validation.csv"
FEATURES_METADATA = DATA_PROCESSED / "features_metadata.json"

# Phase 2: Game States
GAME_STATES = DATA_INTERIM / "game_states.csv"

# Phase 1: Raw Data
SHOTS_RAW = DATA_RAW / "shots.csv"
PLAYERS_RAW = DATA_RAW / "players.csv"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_directories() -> None:
    """
    Create all necessary directories if they don't exist.
    Call this at the start of any pipeline to ensure structure is ready.
    
    Example:
        from src.utils.paths import ensure_directories
        ensure_directories()  # Creates all dirs before pipeline starts
    """
    directories = [
        # Data directories
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        # Results directories
        RESULTS_MODELS,
        RESULTS_LOGS,
        RESULTS_PLOTS,
        RESULTS_METRICS,
        # Source directories
        SRC_PREPROCESSING,
        SRC_FEATURES,
        SRC_MODELS,
        SRC_TRAINING,
        SRC_CALIBRATION,
        SRC_VISUALIZATION,
        SRC_UTILITIES,
        # Project directories
        NOTEBOOKS,
        CONFIGS,
    ]
    
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)


def print_structure() -> None:
    """Print the current project structure."""
    print(f"Project Structure: {PROJECT_ROOT}\n")
    
    print("📊 Data Directories:")
    print(f"  RAW:        {DATA_RAW}")
    print(f"  INTERIM:    {DATA_INTERIM}")
    print(f"  PROCESSED:  {DATA_PROCESSED}\n")
    
    print("📈 Results Directories:")
    print(f"  MODELS:     {RESULTS_MODELS}")
    print(f"  LOGS:       {RESULTS_LOGS}")
    print(f"  PLOTS:      {RESULTS_PLOTS}")
    print(f"  METRICS:    {RESULTS_METRICS}\n")
    
    print("💻 Source Code Modules:")
    print(f"  PREPROCESSING:  {SRC_PREPROCESSING}")
    print(f"  FEATURES:       {SRC_FEATURES}")
    print(f"  MODELS:         {SRC_MODELS}")
    print(f"  TRAINING:       {SRC_TRAINING}")
    print(f"  CALIBRATION:    {SRC_CALIBRATION}")
    print(f"  VISUALIZATION:  {SRC_VISUALIZATION}\n")


# ============================================================================
# MAIN: Initialize on import or direct execution
# ============================================================================

if __name__ == "__main__":
    ensure_directories()
    print("✅ Project structure initialized\n")
    print_structure()
    
    print("\n📋 Key Files (Phase 5):")
    print(f"  Best Model:           {MODEL_XGBOOST_BEST}")
    print(f"  Calibrated (Isotonic): {MODEL_XGBOOST_CALIBRATED_ISOTONIC}")
    print(f"  Calibrated (Sigmoid):  {MODEL_XGBOOST_CALIBRATED_SIGMOID}")
    print(f"  Calibration Metrics:  {CALIBRATION_METRICS}")
    print(f"  Calibration Plot:     {CALIBRATION_PLOT}")
    print(f"  Test Features:        {FEATURES_TEST}")