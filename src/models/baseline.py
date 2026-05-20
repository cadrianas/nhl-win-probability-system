"""
Phase 3: Logistic Regression Baseline Model
Location: src/models/baseline.py

Logistic regression is the gold standard baseline:
- Interpretable (coefficients show feature direction/magnitude)
- Fast to train
- Perfect for calibration analysis
- Demonstrates statistical thinking
"""

import pickle
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss, 
    accuracy_score, confusion_matrix, roc_curve, auc
)

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """
    Logistic Regression model wrapper with training, evaluation, and artifact management.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize LR model with hyperparameters.
        
        Args:
            **hyperparams: sklearn LogisticRegression parameters
                Default: max_iter=1000, random_state=42, class_weight='balanced'
        """
        self.hyperparams = hyperparams
        self.model = LogisticRegression(**hyperparams)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.coefficients = None
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
        
        Returns:
            None (modifies self)
        """
        logger.info("Standardizing training features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        logger.info(f"Training Logistic Regression with {len(X_train)} samples, {X_train.shape[1]} features")
        self.model.fit(X_train_scaled, y_train)
        
        self.feature_names = X_train.columns.tolist()
        self.coefficients = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        self.is_fitted = True
        logger.info(f"✓ Model fitted. Intercept: {self.model.intercept_[0]:.4f}")
    
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
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
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature coefficients (logistic regression 'importance').
        
        Args:
            top_n: Return top N features by absolute coefficient
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.coefficients.head(top_n).copy()
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save model and scaler to disk.
        
        Args:
            model_path: Path to save model pickle
            scaler_path: Path to save scaler pickle
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"✓ Model saved to {model_path}")
        logger.info(f"✓ Scaler saved to {scaler_path}")
    
    @staticmethod
    def load(model_path: str, scaler_path: str) -> 'LogisticRegressionBaseline':
        """
        Load model and scaler from disk.
        
        Args:
            model_path: Path to model pickle
            scaler_path: Path to scaler pickle
        
        Returns:
            Loaded LogisticRegressionBaseline instance
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        instance = LogisticRegressionBaseline()
        instance.model = model
        instance.scaler = scaler
        instance.is_fitted = True
        
        logger.info(f"✓ Model loaded from {model_path}")
        return instance
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of model configuration and state.
        
        Returns:
            Dict with model info
        """
        return {
            'model_type': 'LogisticRegression',
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparams,
            'intercept': float(self.model.intercept_[0]) if self.is_fitted else None,
            'top_5_features': self.get_feature_importance(5).to_dict('records') if self.is_fitted else None,
        }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    hyperparams: Dict[str, Any] = None
) -> Tuple[LogisticRegressionBaseline, Dict[str, float]]:
    """
    Convenience function: Train LR and evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        hyperparams: Model hyperparameters (default: standard)
    
    Returns:
        Tuple: (fitted model, evaluation metrics dict)
    """
    if hyperparams is None:
        hyperparams = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced',
        }
    
    logger.info("=" * 80)
    logger.info("LOGISTIC REGRESSION BASELINE")
    logger.info("=" * 80)
    
    model = LogisticRegressionBaseline(**hyperparams)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    logger.info("=" * 80)
    
    return model, metrics


if __name__ == "__main__":
    # Example usage (requires Phase 2 features)
    logging.basicConfig(level=logging.INFO)
    
    print("This module is imported by phase3_main.py")
    print("See models_main.py for full training pipeline")