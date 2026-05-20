"""
Phase 3: LightGBM Baseline Model
Location: src/models/lightgbm_model.py

LightGBM is a fast gradient boosting framework:
- Often faster than XGBoost
- Handles categorical features natively
- Lower memory usage
- Great for large datasets
"""

import pickle
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss, 
    accuracy_score, confusion_matrix
)

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    LightGBM model wrapper with training, evaluation, and artifact management.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize LightGBM model with hyperparameters.
        
        Args:
            **hyperparams: lightgbm.LGBMClassifier parameters
                Common: n_estimators, learning_rate, num_leaves, max_depth
        """
        self.hyperparams = hyperparams
        self.model = lgb.LGBMClassifier(**hyperparams)
        self.feature_names = None
        self.categorical_features = None
        self.is_fitted = False
        
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        categorical_features: Optional[list] = None,
        verbose: bool = True
    ) -> None:
        """
        Train LightGBM model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
            X_val: Validation features (for early stopping)
            y_val: Validation target
            categorical_features: List of categorical column names
            verbose: Print training progress
        
        Returns:
            None (modifies self)
        """
        logger.info("Preparing LightGBM training data...")
        
        self.feature_names = X_train.columns.tolist()
        self.categorical_features = categorical_features
        
        # Prepare validation set if provided
        eval_set = None
        eval_names = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            eval_names = ['validation']
            logger.info(f"Training with validation set ({len(X_val)} samples)")
        
        logger.info(f"Training LightGBM with {len(X_train)} samples, {X_train.shape[1]} features")
        logger.info(f"  n_estimators: {self.hyperparams.get('n_estimators', 100)}")
        logger.info(f"  learning_rate: {self.hyperparams.get('learning_rate', 0.1)}")
        logger.info(f"  num_leaves: {self.hyperparams.get('num_leaves', 31)}")
        
        # Train with early stopping if eval set provided
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_names=eval_names,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=50) if verbose else lgb.log_evaluation(0)
                ]
            )
            logger.info(f"✓ Best iteration: {self.model.best_iteration_}")
        else:
            self.model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
        
        self.is_fitted = True
        logger.info(f"✓ Model fitted")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Probabilities for class 1 (home win)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels (0 or 1).
        
        Args:
            X: Features
        
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dict with metrics: log_loss, roc_auc, brier_score, accuracy, etc.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Evaluating on {len(X_test)} test samples...")
        
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        metrics = {
            'log_loss': log_loss(y_test, y_pred_proba),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
        }
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'split', top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: 'split' (frequency) or 'gain' (contribution)
            top_n: Return top N features
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get feature importance
        importance_array = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_array
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).reset_index(drop=True)
    
    def save(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save model and metadata to disk.
        
        Args:
            model_path: Path to save model
            metadata_path: Path to save metadata (hyperparams, etc.)
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model")
        
        self.model.booster_.save_model(model_path)
        logger.info(f"✓ Model saved to {model_path}")
        
        if metadata_path:
            metadata = {
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features,
                'hyperparameters': self.hyperparams,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"✓ Metadata saved to {metadata_path}")
    
    @staticmethod
    def load(model_path: str, metadata_path: Optional[str] = None) -> 'LightGBMModel':
        """
        Load model and metadata from disk.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to metadata file
        
        Returns:
            Loaded LightGBMModel instance
        """
        booster = lgb.Booster(model_file=model_path)
        instance = LightGBMModel()
        instance.model = lgb.LGBMClassifier()
        instance.model.booster_ = booster
        instance.is_fitted = True
        
        if metadata_path:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            instance.feature_names = metadata.get('feature_names')
            instance.categorical_features = metadata.get('categorical_features')
            instance.hyperparams = metadata.get('hyperparameters', {})
        
        logger.info(f"✓ Model loaded from {model_path}")
        return instance
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of model configuration and state.
        
        Returns:
            Dict with model info
        """
        return {
            'model_type': 'LightGBM',
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'hyperparameters': self.hyperparams,
            'top_5_features': self.get_feature_importance('split', 5).to_dict('records') if self.is_fitted else None,
        }


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    hyperparams: Dict[str, Any] = None,
    categorical_features: Optional[list] = None
) -> Tuple[LightGBMModel, Dict[str, float]]:
    """
    Convenience function: Train LightGBM and evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        X_val: Validation features (for early stopping)
        y_val: Validation target
        hyperparams: Model hyperparameters (default: standard)
        categorical_features: List of categorical column names
    
    Returns:
        Tuple: (fitted model, evaluation metrics dict)
    """
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
    
    logger.info("=" * 80)
    logger.info("LIGHTGBM MODEL")
    logger.info("=" * 80)
    
    model = LightGBMModel(**hyperparams)
    model.fit(X_train, y_train, X_val, y_val, categorical_features)
    metrics = model.evaluate(X_test, y_test)
    
    logger.info("=" * 80)
    
    return model, metrics


if __name__ == "__main__":
    # Example usage (requires Phase 2 features)
    logging.basicConfig(level=logging.INFO)
    
    print("This module is imported by phase3_main.py")
    print("See phase3_main.py for full training pipeline")