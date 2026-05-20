"""
Phase 3: XGBoost Baseline Model
Location: src/models/xgboost_model.py

XGBoost is a modern gradient boosting framework:
- Handles nonlinear relationships automatically
- Feature interactions learned implicitly
- Fast training, production-ready
- Excellent feature importance metrics
"""

import pickle
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss, 
    accuracy_score, confusion_matrix
)

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost model wrapper with training, evaluation, and artifact management.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize XGBoost model with hyperparameters.
        
        Args:
            **hyperparams: xgboost.XGBClassifier parameters
                Common: n_estimators, learning_rate, max_depth, subsample, colsample_bytree
        """
        self.hyperparams = hyperparams
        self.model = xgb.XGBClassifier(**hyperparams)
        self.label_encoders = {}  # For categorical features
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
        Train XGBoost model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
            X_val: Validation features (for early stopping)
            y_val: Validation target
            categorical_features: List of categorical column names to encode
            verbose: Print training progress
        
        Returns:
            None (modifies self)
        """
        logger.info("Preparing XGBoost training data...")
        
        # Encode categorical features
        X_train_encoded = X_train.copy()
        if categorical_features:
            logger.info(f"Encoding {len(categorical_features)} categorical features: {categorical_features}")
            for col in categorical_features:
                le = LabelEncoder()
                X_train_encoded[col] = le.fit_transform(X_train[col])
                self.label_encoders[col] = le
            self.categorical_features = categorical_features
        
        # Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_encoded = X_val.copy()
            if categorical_features:
                for col in categorical_features:
                    X_val_encoded[col] = self.label_encoders[col].transform(X_val[col])
            eval_set = [(X_val_encoded.values, y_val)]
            logger.info(f"Training with validation set ({len(X_val)} samples)")
        
        logger.info(f"Training XGBoost with {len(X_train)} samples, {X_train_encoded.shape[1]} features")
        logger.info(f"  n_estimators: {self.hyperparams.get('n_estimators', 100)}")
        logger.info(f"  learning_rate: {self.hyperparams.get('learning_rate', 0.1)}")
        logger.info(f"  max_depth: {self.hyperparams.get('max_depth', 6)}")
        
        # Train with early stopping if eval set provided
        if eval_set:
            self.model.fit(
                X_train_encoded.values, y_train,
                eval_set=eval_set,
                verbose=verbose
            )
            logger.info(f"✓ Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train_encoded.values, y_train, verbose=False)
        
        self.feature_names = X_train_encoded.columns.tolist()
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
        
        X_encoded = X.copy()
        if self.categorical_features:
            for col in self.categorical_features:
                X_encoded[col] = self.label_encoders[col].transform(X[col])
        
        return self.model.predict_proba(X_encoded.values)[:, 1]
    
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
        
        X_encoded = X.copy()
        if self.categorical_features:
            for col in self.categorical_features:
                X_encoded[col] = self.label_encoders[col].transform(X[col])
        
        return self.model.predict(X_encoded.values)
    
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
    
    def get_feature_importance(self, importance_type: str = 'weight', top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: 'weight', 'gain', or 'cover'
            top_n: Return top N features
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get feature importance from booster
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).reset_index(drop=True)
    
    def save(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save model and metadata to disk.
        
        Args:
            model_path: Path to save model
            metadata_path: Path to save metadata (hyperparams, encoders, etc.)
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model")
        
        self.model.save_model(model_path)
        logger.info(f"✓ Model saved to {model_path}")
        
        if metadata_path:
            metadata = {
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features,
                'label_encoders': self.label_encoders,
                'hyperparameters': self.hyperparams,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"✓ Metadata saved to {metadata_path}")
    
    @staticmethod
    def load(model_path: str, metadata_path: Optional[str] = None) -> 'XGBoostModel':
        """
        Load model and metadata from disk.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to metadata file
        
        Returns:
            Loaded XGBoostModel instance
        """
        instance = XGBoostModel()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(model_path)
        instance.is_fitted = True
        
        if metadata_path:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            instance.feature_names = metadata.get('feature_names')
            instance.categorical_features = metadata.get('categorical_features')
            instance.label_encoders = metadata.get('label_encoders', {})
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
            'model_type': 'XGBoost',
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'hyperparameters': self.hyperparams,
            'top_5_features': self.get_feature_importance('weight', 5).to_dict('records') if self.is_fitted else None,
        }


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    hyperparams: Dict[str, Any] = None,
    categorical_features: Optional[list] = None
) -> Tuple[XGBoostModel, Dict[str, float]]:
    """
    Convenience function: Train XGBoost and evaluate on test set.
    
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
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
        }
    
    logger.info("=" * 80)
    logger.info("XGBOOST MODEL")
    logger.info("=" * 80)
    
    model = XGBoostModel(**hyperparams)
    model.fit(X_train, y_train, X_val, y_val, categorical_features)
    metrics = model.evaluate(X_test, y_test)
    
    logger.info("=" * 80)
    
    return model, metrics


if __name__ == "__main__":
    # Example usage (requires Phase 2 features)
    logging.basicConfig(level=logging.INFO)
    
    print("This module is imported by phase3_main.py")
    print("See models_main.py for full training pipeline")