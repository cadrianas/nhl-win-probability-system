"""
Phase 3: Model Evaluation & Comparison
Location: src/training/evaluate.py

Comprehensive evaluation, metrics comparison, and report generation.
"""

import json
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple trained models and generate comparison reports.
    """
    
    def __init__(self):
        """Initialize comparison tracker."""
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
    
    def add_model(self, name: str, model: Any, metrics: Dict[str, float]) -> None:
        """
        Add a model and its metrics to comparison.
        
        Args:
            name: Model name (e.g., 'Logistic Regression')
            model: Trained model instance
            metrics: Dict of computed metrics
        """
        self.models[name] = model
        self.metrics[name] = metrics
        
        # Extract feature importance
        try:
            if hasattr(model, 'get_feature_importance'):
                self.feature_importance[name] = model.get_feature_importance(top_n=50)
            else:
                logger.warning(f"Model {name} does not support get_feature_importance()")
        except Exception as e:
            logger.warning(f"Could not extract feature importance for {name}: {e}")
    
    def get_comparison_table(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Generate comparison table across all models.
        
        Args:
            metrics: List of metric names to compare. 
                    Default: ['log_loss', 'roc_auc', 'brier_score', 'accuracy']
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if metrics is None:
            metrics = ['log_loss', 'roc_auc', 'brier_score', 'accuracy']
        
        comparison_data = {}
        for model_name, model_metrics in self.metrics.items():
            comparison_data[model_name] = {
                m: model_metrics.get(m, np.nan) for m in metrics
            }
        
        df = pd.DataFrame(comparison_data).T
        
        # Format for readability
        df = df.round(4)
        
        return df
    
    def get_feature_importance_comparison(self, top_n: int = 20) -> pd.DataFrame:
        """
        Compare feature importance across models.
        
        Args:
            top_n: Top N features to return
        
        Returns:
            DataFrame with feature importance from each model
        """
        # Get union of top features from all models
        all_features = set()
        for model_name, importance_df in self.feature_importance.items():
            all_features.update(importance_df['feature'].head(top_n).tolist())
        
        all_features = sorted(list(all_features))
        
        # Build comparison
        comparison = {}
        for feature in all_features:
            comparison[feature] = {}
            for model_name, importance_df in self.feature_importance.items():
                feature_rows = importance_df[importance_df['feature'] == feature]
                if len(feature_rows) > 0:
                    comparison[feature][model_name] = feature_rows.iloc[0]['importance']
                else:
                    comparison[feature][model_name] = 0
        
        df = pd.DataFrame(comparison).T
        return df.sort_values(by=df.columns[0], ascending=False).head(top_n)
    
    def print_summary(self) -> None:
        """Print formatted summary of all models."""
        logger.info("\n" + "=" * 100)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 100)
        
        # Metrics comparison
        logger.info("\nMetrics Comparison:")
        logger.info("-" * 100)
        
        comparison_df = self.get_comparison_table()
        logger.info("\n" + str(comparison_df))
        
        # Best model per metric
        logger.info("\n" + "-" * 100)
        logger.info("Best Model per Metric:")
        logger.info("-" * 100)
        
        for col in comparison_df.columns:
            if col in ['log_loss', 'brier_score']:
                # Lower is better
                best_model = comparison_df[col].idxmin()
                best_value = comparison_df[col].min()
            else:
                # Higher is better
                best_model = comparison_df[col].idxmax()
                best_value = comparison_df[col].max()
            
            logger.info(f"  {col:20s}: {best_model:20s} ({best_value:.4f})")
        
        logger.info("\n" + "=" * 100)
    
    def save_comparison(self, output_csv: str) -> None:
        """
        Save comparison table to CSV.
        
        Args:
            output_csv: Path to output CSV file
        """
        comparison_df = self.get_comparison_table()
        comparison_df.to_csv(output_csv)
        logger.info(f"✓ Comparison saved to {output_csv}")
    
    def save_metrics_json(self, output_json: str) -> None:
        """
        Save all metrics to JSON file.
        
        Args:
            output_json: Path to output JSON file
        """
        # Convert to JSON-serializable format
        metrics_dict = {}
        for model_name, metrics in self.metrics.items():
            metrics_dict[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        
        with open(output_json, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"✓ Metrics saved to {output_json}")
    
    def save_feature_importance(self, output_csv: str) -> None:
        """
        Save feature importance comparison to CSV.
        
        Args:
            output_csv: Path to output CSV file
        """
        importance_df = self.get_feature_importance_comparison(top_n=50)
        importance_df.to_csv(output_csv)
        logger.info(f"✓ Feature importance saved to {output_csv}")


def validate_model_performance(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    model_name: str = "Model"
) -> Tuple[bool, List[str]]:
    """
    Check if model performance meets minimum thresholds.
    
    Args:
        metrics: Dict of computed metrics
        thresholds: Dict of minimum acceptable values
        model_name: Name for logging
    
    Returns:
        Tuple: (passes_all_thresholds: bool, list of failed metrics)
    """
    failed = []
    
    for metric, threshold in thresholds.items():
        if metric not in metrics:
            logger.warning(f"Metric {metric} not found in results")
            continue
        
        value = metrics[metric]
        
        # Lower is better for log_loss and brier_score
        if metric in ['log_loss', 'brier_score']:
            if value > threshold:
                failed.append(f"{metric}: {value:.4f} > {threshold} (FAIL)")
            else:
                logger.info(f"✓ {metric}: {value:.4f} <= {threshold} (PASS)")
        else:
            # Higher is better for roc_auc, accuracy, etc.
            if value < threshold:
                failed.append(f"{metric}: {value:.4f} < {threshold} (FAIL)")
            else:
                logger.info(f"✓ {metric}: {value:.4f} >= {threshold} (PASS)")
    
    passes_all = len(failed) == 0
    
    if passes_all:
        logger.info(f"✓ {model_name} passes all performance thresholds")
    else:
        logger.warning(f"✗ {model_name} fails {len(failed)} thresholds:")
        for f in failed:
            logger.warning(f"  - {f}")
    
    return passes_all, failed


def compute_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute calibration-related metrics (for Phase 5).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for reliability diagram
    
    Returns:
        Dict with calibration metrics
    """
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    # Expected calibration error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Maximum calibration error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred))
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
    }


def generate_report(
    comparison: ModelComparison,
    best_model_name: str,
    output_dir: str
) -> None:
    """
    Generate comprehensive text report.
    
    Args:
        comparison: ModelComparison instance
        best_model_name: Name of best model
        output_dir: Directory to save report
    """
    output_path = Path(output_dir) / "phase3_report.txt"
    
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PHASE 3: BASELINE MODELS - COMPREHENSIVE REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        # Executive summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"Best Model: {best_model_name}\n\n")
        
        # Metrics comparison
        f.write("METRICS COMPARISON\n")
        f.write("-" * 100 + "\n")
        comparison_df = comparison.get_comparison_table()
        f.write(comparison_df.to_string())
        f.write("\n\n")
        
        # Model details
        f.write("MODEL DETAILS\n")
        f.write("-" * 100 + "\n")
        for model_name, model in comparison.models.items():
            f.write(f"\n{model_name}:\n")
            if hasattr(model, 'summary'):
                summary = model.summary()
                for key, value in summary.items():
                    if key not in ['feature_names']:
                        f.write(f"  {key}: {value}\n")
        
        # Feature importance
        f.write("\n\nTOP FEATURES BY MODEL\n")
        f.write("-" * 100 + "\n")
        importance_df = comparison.get_feature_importance_comparison(top_n=20)
        f.write(importance_df.to_string())
        
        f.write("\n\n" + "=" * 100 + "\n")
    
    logger.info(f"✓ Report saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("This module is imported by phase3_main.py")
    print("See phase3_main.py for full training pipeline")