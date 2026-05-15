"""
Centralized path management for the NHL Win Probability project.

This module provides a single source of truth for all file paths.
Ensures the project works from any working directory.
"""

from pathlib import Path

# Project root: src/utils/paths.py -> src -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Results directories
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
RESULTS_PLOTS = PROJECT_ROOT / "results" / "plots"

# Notebooks
NOTEBOOKS = PROJECT_ROOT / "notebooks"

# Configs
CONFIGS = PROJECT_ROOT / "configs"

# Source code
SRC = PROJECT_ROOT / "src"


def ensure_directories() -> None:
    """
    Create all necessary directories if they don't exist.
    
    Call this at the start of any pipeline to ensure structure is ready.
    """
    directories = [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        RESULTS_MODELS,
        RESULTS_LOGS,
        RESULTS_PLOTS,
        NOTEBOOKS,
        CONFIGS,
        SRC,
    ]
    
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"✅ Project structure initialized at {PROJECT_ROOT}")
    print(f"\nKey directories:")
    print(f"  DATA_RAW:        {DATA_RAW}")
    print(f"  DATA_PROCESSED:  {DATA_PROCESSED}")
    print(f"  RESULTS_MODELS:  {RESULTS_MODELS}")
    print(f"  RESULTS_LOGS:    {RESULTS_LOGS}")
