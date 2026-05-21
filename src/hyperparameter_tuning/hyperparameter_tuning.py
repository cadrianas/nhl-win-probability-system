"""
Phase 4: Hyperparameter Tuning with Optuna

This module provides Bayesian optimization for XGBoost and LightGBM hyperparameters.
Uses Optuna with pruning to efficiently search the hyperparameter space.

Usage:
    python phase4_hyperparameter_tuning.py --model xgboost --n-trials 100
    python phase4_hyperparameter_tuning.py --model lightgbm --n-trials 100
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import sys

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

# Ensure project root is on path so src.utils.paths resolves
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.paths import RESULTS_MODELS

def preprocess_features(X):
    """Drop non-numeric columns."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    return X[numeric_cols]

    
class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization for XGBoost and LightGBM.
    
    Attributes:
        model_type: 'xgboost' or 'lightgbm'
        X_train, y_train: Training data
        X_val, y_val: Validation data
        study: Optuna Study object
        best_params: Best parameters found
        best_score: Best validation metric achieved
    """
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = 'xgboost',
        metric: str = 'roc_auc',
        output_dir: Path = Path('tuning_results')
    ):
        """
        Initialize tuner.
        
        Args:
            X_train, y_train: Training features and target
            X_val, y_val: Validation features and target
            model_type: 'xgboost' or 'lightgbm'
            metric: Optimization metric ('roc_auc', 'log_loss')
            output_dir: Directory to save results
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type.lower()
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        assert self.model_type in ['xgboost', 'lightgbm'], \
            f"model_type must be 'xgboost' or 'lightgbm', got {model_type}"
        assert self.metric in ['roc_auc', 'log_loss'], \
            f"metric must be 'roc_auc' or 'log_loss', got {metric}"
        
        # Optuna study setup
        # For ROC-AUC: maximize (higher is better)
        # For log_loss: minimize (lower is better)
        direction = 'maximize' if self.metric == 'roc_auc' else 'minimize'
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5)
        )
        
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.trial_history = []
    
    def _objective_xgboost(self, trial: optuna.Trial) -> float:
        """
        Objective function for XGBoost optimization.
        
        Args:
            trial: Optuna Trial object
            
        Returns:
            Validation metric score
        """
        # Hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 0.001, 5, log=True),
            'alpha': trial.suggest_float('alpha', 0.001, 5, log=True),
            'random_state': 42,
            'verbosity': 0,
            'eval_metric': 'logloss'
        }
        
        try:
            # Convert to numeric only
            X_train_numeric = preprocess_features(self.X_train)
            X_val_numeric = preprocess_features(self.X_val)

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_numeric, self.y_train)
            # Get predictions
            y_pred_proba = model.predict_proba(X_val_numeric)[:, 1]
            
            # Calculate metric
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.y_val, y_pred_proba)
            else:  # log_loss
                score = log_loss(self.y_val, y_pred_proba)
            
            # Report to trial for pruning
            trial.report(score, step=model.n_estimators)
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf') if self.metric == 'roc_auc' else float('inf')
    
    def _objective_lightgbm(self, trial: optuna.Trial) -> float:
        """
        Objective function for LightGBM optimization.

        Args:
            trial: Optuna Trial object

        Returns:
            Validation metric score
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 10, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 10, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'random_state': 42,
            'verbose': -1,
            'metric': 'binary_logloss'
        }

        try:
            # Convert to numeric only
            X_train_numeric = preprocess_features(self.X_train)
            X_val_numeric = preprocess_features(self.X_val)

            # Train model
            model = lgb.LGBMClassifier(**params)

            model.fit(
                X_train_numeric, self.y_train,
                eval_set=[(X_val_numeric, self.y_val)],
                eval_metric='auc' if self.metric == 'roc_auc' else 'binary_logloss',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

            # Get predictions
            y_pred_proba = model.predict_proba(X_val_numeric)[:, 1]

            # Calculate metric
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.y_val, y_pred_proba)
            else:  # log_loss
                score = log_loss(self.y_val, y_pred_proba)

            return score

        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf') if self.metric == 'roc_auc' else float('inf')
    
    def optimize(self, n_trials: int = 100, n_jobs: int = 1):
        """
        Run Bayesian optimization.
        
        Args:
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        print(f"\n{'='*60}")
        print(f"Optimizing {self.model_type.upper()} Hyperparameters")
        print(f"{'='*60}")
        print(f"Metric: {self.metric}")
        print(f"Training set size: {len(self.X_train)}")
        print(f"Validation set size: {len(self.X_val)}")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"N trials: {n_trials}")
        print(f"Parallel jobs: {n_jobs}")
        print(f"{'='*60}\n")
        
        # Choose objective function
        if self.model_type == 'xgboost':
            objective = self._objective_xgboost
        else:
            objective = self._objective_lightgbm
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Extract best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete")
        print(f"{'='*60}")
        print(f"Best {self.metric}: {self.best_score:.6f}")
        print(f"Best Hyperparameters:\n")
        for key, value in self.best_params.items():
            print(f"  {key:25s}: {value}")
        print(f"{'='*60}\n")
        
        return self.best_params, self.best_score
    
    def train_best_model(self) -> Any:
        """
        Train final model with best hyperparameters.

        Returns:
            Trained model object
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")

        print(f"\nTraining final {self.model_type.upper()} model with best params...")

        # Preprocess features to remove categorical columns
        X_train_numeric = preprocess_features(self.X_train)

        if self.model_type == 'xgboost':
            model = xgb.XGBClassifier(**self.best_params, random_state=42, verbosity=0)
        else:
            model = lgb.LGBMClassifier(**self.best_params, random_state=42, verbose=-1)

        model.fit(X_train_numeric, self.y_train)
        self.best_model = model

        return model
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get all trials as a pandas DataFrame."""
        trials_data = []
        
        for trial in self.study.trials:
            trial_dict = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
            }
            trial_dict.update(trial.params)
            trials_data.append(trial_dict)
        
        return pd.DataFrame(trials_data)
    
    def save_results(self, name: str = None):
        """
        Save tuning results to disk.
        
        Args:
            name: Prefix for output files (default: model_type + timestamp)
        """
        if name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"{self.model_type}_{timestamp}"
        
        # Save best parameters
        params_file = self.output_dir / f"{name}_best_params.json"
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"✓ Best params saved to {params_file}")
        
        # Save all trials
        trials_df = self.get_trials_dataframe()
        trials_file = self.output_dir / f"{name}_all_trials.csv"
        trials_df.to_csv(trials_file, index=False)
        print(f"✓ All trials saved to {trials_file}")
        
        # Save study object
        study_file = self.output_dir / f"{name}_study.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        print(f"✓ Optuna study saved to {study_file}")
        
        # Save best model
        if self.best_model is not None:
            model_file = self.output_dir / f"{name}_best_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"✓ Best model saved to {model_file}")
        
        # Save summary report
        summary = {
            'model_type': self.model_type,
            'metric': self.metric,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat()
        }
        summary_file = self.output_dir / f"{name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to {summary_file}")
        
        return params_file, trials_file, model_file, summary_file


def load_data(features_file: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix and target variable.
    
    Handles multiple file naming conventions:
        - Single file: features.csv, features_train.csv, etc.
        - Split files: features_train.csv + features_test.csv
    
    Assumes target column is: 'home_team_won', 'target', or 'y'
    
    Args:
        features_file: Path to features CSV (or directory containing split files)
        
    Returns:
        (X, y) tuple
    """
    features_file = Path(features_file)
    
    # If it's a directory, look for train/test split
    if features_file.is_dir():
        train_file = features_file / 'features_train.csv'
        if train_file.exists():
            features_file = train_file
            print(f"Found split files. Using {train_file} for tuning.")
        else:
            raise ValueError(f"No features_train.csv found in {features_file}")
    
    # If file doesn't exist, try common alternatives
    if not features_file.exists():
        alternatives = [
            features_file.parent / 'features_train.csv',
            features_file.parent / 'features.csv',
            Path('data/processed/features_train.csv'),
            Path('data/processed/features.csv'),
        ]
        
        found = False
        for alt in alternatives:
            if alt.exists():
                print(f"File not found: {features_file}")
                print(f"Using alternative: {alt}")
                features_file = alt
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"Could not find features file.\n"
                f"Tried: {features_file}, {[str(a) for a in alternatives]}"
            )
    
    df = pd.read_csv(features_file)
    
    # Identify target column
    target_cols = ['target_home_win', 'home_team_won', 'target', 'y']
    target_col = next((col for col in target_cols if col in df.columns), None)
    
    if target_col is None:
        raise ValueError(
            f"Could not find target column in {features_file}.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected one of: {target_cols}"
        )
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"✓ Loaded {len(X):,} samples with {X.shape[1]} features")
    print(f"  Target distribution: {dict(y.value_counts())}")
    
    return X, y


def run_tuning_pipeline(
    features_file: Path,
    model_type: str = 'xgboost',
    n_trials: int = 100,
    val_size: float = 0.2,
    output_dir: Path = None,
    metric: str = 'roc_auc',
    n_jobs: int = 1
):
    """
    Complete tuning pipeline: load data → split → optimize → save.
    
    Args:
        features_file: Path to features CSV
        model_type: 'xgboost' or 'lightgbm'
        n_trials: Number of optimization trials
        val_size: Validation set fraction
        output_dir: Directory to save results (defaults to results/models/)
        metric: 'roc_auc' or 'log_loss'
        n_jobs: Parallel jobs for optimization
    """
    if output_dir is None:
        output_dir = RESULTS_MODELS
    # Load data
    print(f"\nLoading features from {features_file}")
    X, y = load_data(features_file)
    
    # Train-val split (temporal if possible, else random)
    # For production, you'd want temporal split to avoid leakage
    val_idx = int(len(X) * (1 - val_size))
    
    X_train, X_val = X.iloc[:val_idx], X.iloc[val_idx:]
    y_train, y_val = y.iloc[:val_idx], y.iloc[val_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        X_train, y_train, X_val, y_val,
        model_type=model_type,
        metric=metric,
        output_dir=output_dir
    )
    
    # Run optimization
    best_params, best_score = tuner.optimize(n_trials=n_trials, n_jobs=n_jobs)
    
    # Train final model
    best_model = tuner.train_best_model()
    
    # Save results
    tuner.save_results(name=f"{model_type}")
    
    # Print trials summary
    trials_df = tuner.get_trials_dataframe()
    print(f"\nTrial Summary:")
    print(f"  Total trials: {len(trials_df)}")
    print(f"  Completed: {(trials_df['state'] == 'COMPLETE').sum()}")
    print(f"  Pruned: {(trials_df['state'] == 'PRUNED').sum()}")
    print(f"  Best score: {best_score:.6f}")
    
    # Show top 5 trials
    print(f"\nTop 5 Trials:")
    if tuner.metric == 'roc_auc':
        top_trials = trials_df.nlargest(5, 'value')
    else:
        top_trials = trials_df.nsmallest(5, 'value')
    
    for idx, row in top_trials.iterrows():
        print(f"  Trial {row['trial_number']:3d}: {metric} = {row['value']:.6f}")
    
    return tuner, best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for NHL Win Probability models'
    )
    parser.add_argument('--features', type=Path, default='data/processed/features.csv',
                        help='Path to features CSV file')
    parser.add_argument('--model', type=str, choices=['xgboost', 'lightgbm'],
                        default='xgboost', help='Model to tune')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--metric', type=str, choices=['roc_auc', 'log_loss'],
                        default='roc_auc', help='Optimization metric')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory to save results (default: results/models/)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel jobs (-1 for all cores)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Validation set fraction')
    
    args = parser.parse_args()
    
    # Run tuning pipeline
    tuner, best_model = run_tuning_pipeline(
        features_file=args.features,
        model_type=args.model,
        n_trials=args.n_trials,
        val_size=args.val_size,
        output_dir=args.output_dir,
        metric=args.metric,
        n_jobs=args.n_jobs
    )