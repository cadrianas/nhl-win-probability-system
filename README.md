# NHL Win Probability Project 🏒

An end-to-end machine learning pipeline to predict the win probability of NHL games in real-time based on game states and shot-level data.

## Project Overview

This project transforms raw NHL shot data (MoneyPuck) into a calibrated win probability model. It follows a structured multi-phase pipeline from data cleaning to an interactive dashboard.

## Project Structure

The project follows a standardized directory structure managed via `src/utils/paths.py`:

```text
├── data/               # Data storage (Git ignored)
│   ├── raw/            # Original shots.csv files
│   ├── interim/        # Intermediate game states
│   └── processed/      # Cleaned shots and engineered features
├── results/            # Model artifacts and analysis
│   ├── models/         # Trained .pkl and .json models
│   ├── metrics/        # Performance reports and JSON metrics
│   ├── plots/          # Calibration curves and visualizations
│   ├── phase2/         # Feature engineering statistics
│   └── tuning/         # Hyperparameter optimization results
├── src/                # Source code
│   ├── preprocessing/  # Data cleaning and state generation
│   ├── features/       # Feature engineering with Polars
│   ├── models/         # Baseline model training and evaluation
│   ├── hyperparameter_tuning/ # Optuna-based optimization
│   ├── calibration/    # Probability calibration (Isotonic/Sigmoid)
│   ├── dashboard/      # Streamlit interactive dashboard
│   └── utils/          # Centralized path management
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: Additional dependencies like `xgboost`, `lightgbm`, `optuna`, and `shap` may be required for specific phases.

## Execution Pipeline

The project is executed in sequential phases:

### Phase 1: Data Preprocessing
Cleans raw shot data and generates game state snapshots.
- `python src/preprocessing/clean_shots.py`
- `python src/preprocessing/create_game_states.py`

### Phase 2: Feature Engineering
Uses Polars for high-performance feature creation, including context and interaction features.
- `python src/features/main.py`

### Phase 3: Baseline Models
Trains and compares initial models (Logistic Regression, XGBoost, LightGBM).
- `python src/models/models_main.py`

### Phase 4: Hyperparameter Tuning
Optimizes model performance using Bayesian search with Optuna.
- `python src/hyperparameter_tuning/hyperparameter_tuning.py`

### Phase 5: Probability Calibration
Ensures predicted probabilities reflect true frequencies using Isotonic Regression.
- `python src/calibration/calibration.py`

### Phase 7: Interactive Dashboard
Interactive visualization of win probability and feature importance (SHAP).
- `streamlit run src/dashboard/streamlit_app.py`

## Path Management

This project uses a centralized path management system in `src/utils/paths.py`. Always import paths from this module to ensure the project remains portable:

```python
from src.utils.paths import DATA_PROCESSED, RESULTS_MODELS, ensure_directories

ensure_directories()  # Call at the start of scripts
```

*Note project is not finalised, the next step is integrating Snakefile and LSTM*

*Note: Google Jules was used in this project*
