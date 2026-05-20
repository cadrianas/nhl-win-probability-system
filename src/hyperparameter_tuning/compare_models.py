"""
Phase 4: Baseline vs Tuned Model Comparison

Compares baseline (untunned) models against tuned hyperparameters
on validation and test sets.

Usage:
    python compare_models.py --baseline models/xgb_baseline.pkl --tuned tuning_results/xgboost_best_model.pkl --features data/processed/features.csv
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.calibration import calibration_curve
import seaborn as sns


class ModelComparator:
    """Compare baseline vs tuned models."""
    
    def __init__(self, baseline_model, tuned_model, X_val, y_val, X_test, y_test):
        """
        Initialize comparator.
        
        Args:
            baseline_model: Baseline (untunned) model
            tuned_model: Tuned model
            X_val, y_val: Validation set
            X_test, y_test: Test set
        """
        self.baseline_model = baseline_model
        self.tuned_model = tuned_model
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        # Generate predictions
        self.baseline_val_proba = baseline_model.predict_proba(X_val)[:, 1]
        self.tuned_val_proba = tuned_model.predict_proba(X_val)[:, 1]
        
        self.baseline_test_proba = baseline_model.predict_proba(X_test)[:, 1]
        self.tuned_test_proba = tuned_model.predict_proba(X_test)[:, 1]
    
    def compute_metrics(self, y_true, y_pred_proba, model_name=''):
        """Compute comprehensive evaluation metrics."""
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'Model': model_name,
            'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
            'Log Loss': log_loss(y_true, y_pred_proba),
            'Brier Score': brier_score_loss(y_true, y_pred_proba),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        return metrics
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get side-by-side comparison of metrics."""
        metrics_val_baseline = self.compute_metrics(self.y_val, self.baseline_val_proba, 'Baseline (Val)')
        metrics_val_tuned = self.compute_metrics(self.y_val, self.tuned_val_proba, 'Tuned (Val)')
        metrics_test_baseline = self.compute_metrics(self.y_test, self.baseline_test_proba, 'Baseline (Test)')
        metrics_test_tuned = self.compute_metrics(self.y_test, self.tuned_test_proba, 'Tuned (Test)')
        
        # Combine into single table
        all_metrics = [
            metrics_val_baseline,
            metrics_val_tuned,
            metrics_test_baseline,
            metrics_test_tuned
        ]
        
        df_metrics = pd.DataFrame(all_metrics)
        return df_metrics.set_index('Model')
    
    def print_comparison(self):
        """Print formatted comparison table."""
        df = self.get_comparison_table()
        
        print("\n" + "="*100)
        print("MODEL COMPARISON: BASELINE vs TUNED")
        print("="*100)
        print(df.round(6).to_string())
        print("="*100)
        
        # Calculate improvements
        baseline_test_auc = self.compute_metrics(self.y_test, self.baseline_test_proba)['ROC-AUC']
        tuned_test_auc = self.compute_metrics(self.y_test, self.tuned_test_proba)['ROC-AUC']
        improvement = tuned_test_auc - baseline_test_auc
        improvement_pct = (improvement / baseline_test_auc) * 100
        
        print(f"\nKEY IMPROVEMENT (Test Set):")
        print(f"  Baseline ROC-AUC: {baseline_test_auc:.6f}")
        print(f"  Tuned ROC-AUC:    {tuned_test_auc:.6f}")
        print(f"  Absolute:         {improvement:+.6f}")
        print(f"  Relative:         {improvement_pct:+.2f}%")
        print("="*100 + "\n")
    
    def plot_roc_curves(self, output_file='comparison_roc_curves.png'):
        """Plot ROC curves for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validation set
        fpr_baseline, tpr_baseline, _ = roc_curve(self.y_val, self.baseline_val_proba)
        fpr_tuned, tpr_tuned, _ = roc_curve(self.y_val, self.tuned_val_proba)
        
        auc_baseline = auc(fpr_baseline, tpr_baseline)
        auc_tuned = auc(fpr_tuned, tpr_tuned)
        
        axes[0].plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC={auc_baseline:.4f})', lw=2)
        axes[0].plot(fpr_tuned, tpr_tuned, label=f'Tuned (AUC={auc_tuned:.4f})', lw=2)
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random', lw=1)
        axes[0].set_xlabel('False Positive Rate', fontsize=11)
        axes[0].set_ylabel('True Positive Rate', fontsize=11)
        axes[0].set_title('Validation Set ROC Curves', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Test set
        fpr_baseline, tpr_baseline, _ = roc_curve(self.y_test, self.baseline_test_proba)
        fpr_tuned, tpr_tuned, _ = roc_curve(self.y_test, self.tuned_test_proba)
        
        auc_baseline = auc(fpr_baseline, tpr_baseline)
        auc_tuned = auc(fpr_tuned, tpr_tuned)
        
        axes[1].plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC={auc_baseline:.4f})', lw=2)
        axes[1].plot(fpr_tuned, tpr_tuned, label=f'Tuned (AUC={auc_tuned:.4f})', lw=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random', lw=1)
        axes[1].set_xlabel('False Positive Rate', fontsize=11)
        axes[1].set_ylabel('True Positive Rate', fontsize=11)
        axes[1].set_title('Test Set ROC Curves', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to {output_file}")
        plt.close()
    
    def plot_calibration_curves(self, output_file='comparison_calibration_curves.png'):
        """Plot calibration curves for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validation set
        prob_true_baseline, prob_pred_baseline = calibration_curve(
            self.y_val, self.baseline_val_proba, n_bins=10
        )
        prob_true_tuned, prob_pred_tuned = calibration_curve(
            self.y_val, self.tuned_val_proba, n_bins=10
        )
        
        axes[0].plot(prob_pred_baseline, prob_true_baseline, 'o-', label='Baseline', lw=2, markersize=6)
        axes[0].plot(prob_pred_tuned, prob_true_tuned, 's-', label='Tuned', lw=2, markersize=6)
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', lw=1)
        axes[0].set_xlabel('Mean Predicted Probability', fontsize=11)
        axes[0].set_ylabel('Fraction of Positives', fontsize=11)
        axes[0].set_title('Validation Set Calibration', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Test set
        prob_true_baseline, prob_pred_baseline = calibration_curve(
            self.y_test, self.baseline_test_proba, n_bins=10
        )
        prob_true_tuned, prob_pred_tuned = calibration_curve(
            self.y_test, self.tuned_test_proba, n_bins=10
        )
        
        axes[1].plot(prob_pred_baseline, prob_true_baseline, 'o-', label='Baseline', lw=2, markersize=6)
        axes[1].plot(prob_pred_tuned, prob_true_tuned, 's-', label='Tuned', lw=2, markersize=6)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', lw=1)
        axes[1].set_xlabel('Mean Predicted Probability', fontsize=11)
        axes[1].set_ylabel('Fraction of Positives', fontsize=11)
        axes[1].set_title('Test Set Calibration', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved calibration curves to {output_file}")
        plt.close()
    
    def plot_metric_comparison(self, output_file='comparison_metrics.png'):
        """Plot comparison of key metrics."""
        df = self.get_comparison_table()
        
        # Select key metrics
        key_metrics = ['ROC-AUC', 'Log Loss', 'Brier Score', 'Accuracy']
        df_plot = df[key_metrics].copy()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        df_plot.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Baseline vs Tuned: Key Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.legend(title='Metric', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metric comparison to {output_file}")
        plt.close()
    
    def plot_prediction_distributions(self, output_file='comparison_distributions.png'):
        """Plot distributions of predicted probabilities."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Validation - Baseline
        axes[0, 0].hist(self.baseline_val_proba[self.y_val == 0], bins=30, alpha=0.6, label='Away Win', color='blue')
        axes[0, 0].hist(self.baseline_val_proba[self.y_val == 1], bins=30, alpha=0.6, label='Home Win', color='orange')
        axes[0, 0].set_title('Validation: Baseline Predictions', fontweight='bold')
        axes[0, 0].set_xlabel('Predicted Home Win Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Validation - Tuned
        axes[0, 1].hist(self.tuned_val_proba[self.y_val == 0], bins=30, alpha=0.6, label='Away Win', color='blue')
        axes[0, 1].hist(self.tuned_val_proba[self.y_val == 1], bins=30, alpha=0.6, label='Home Win', color='orange')
        axes[0, 1].set_title('Validation: Tuned Predictions', fontweight='bold')
        axes[0, 1].set_xlabel('Predicted Home Win Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Test - Baseline
        axes[1, 0].hist(self.baseline_test_proba[self.y_test == 0], bins=30, alpha=0.6, label='Away Win', color='blue')
        axes[1, 0].hist(self.baseline_test_proba[self.y_test == 1], bins=30, alpha=0.6, label='Home Win', color='orange')
        axes[1, 0].set_title('Test: Baseline Predictions', fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Home Win Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Test - Tuned
        axes[1, 1].hist(self.tuned_test_proba[self.y_test == 0], bins=30, alpha=0.6, label='Away Win', color='blue')
        axes[1, 1].hist(self.tuned_test_proba[self.y_test == 1], bins=30, alpha=0.6, label='Home Win', color='orange')
        axes[1, 1].set_title('Test: Tuned Predictions', fontweight='bold')
        axes[1, 1].set_xlabel('Predicted Home Win Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction distributions to {output_file}")
        plt.close()
    
    def save_comparison_report(self, output_file='comparison_report.txt'):
        """Save comprehensive text report."""
        with open(output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("BASELINE vs TUNED MODEL COMPARISON REPORT\n")
            f.write("="*100 + "\n\n")
            
            df = self.get_comparison_table()
            f.write("METRICS COMPARISON:\n")
            f.write(df.round(6).to_string())
            f.write("\n\n")
            
            # Improvements
            baseline_test_auc = self.compute_metrics(self.y_test, self.baseline_test_proba)['ROC-AUC']
            tuned_test_auc = self.compute_metrics(self.y_test, self.tuned_test_proba)['ROC-AUC']
            improvement = tuned_test_auc - baseline_test_auc
            improvement_pct = (improvement / baseline_test_auc) * 100
            
            f.write("TEST SET IMPROVEMENTS:\n")
            f.write(f"  Baseline ROC-AUC: {baseline_test_auc:.6f}\n")
            f.write(f"  Tuned ROC-AUC:    {tuned_test_auc:.6f}\n")
            f.write(f"  Absolute gain:    {improvement:+.6f}\n")
            f.write(f"  Relative gain:    {improvement_pct:+.2f}%\n")
            f.write("\n" + "="*100 + "\n")
        
        print(f"✓ Saved comparison report to {output_file}")


def load_data(features_file: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and target."""
    df = pd.read_csv(features_file)
    
    target_cols = ['home_team_won', 'target', 'y']
    target_col = next((col for col in target_cols if col in df.columns), None)
    
    if target_col is None:
        raise ValueError(f"Could not find target column in {features_file}")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return X, y


def run_comparison(
    baseline_model_path: Path,
    tuned_model_path: Path,
    features_file: Path,
    val_size: float = 0.2,
    output_dir: Path = Path('comparison_results')
):
    """Run complete comparison pipeline."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print(f"\nLoading models...")
    with open(baseline_model_path, 'rb') as f:
        baseline_model = pickle.load(f)
    print(f"✓ Loaded baseline model from {baseline_model_path}")
    
    with open(tuned_model_path, 'rb') as f:
        tuned_model = pickle.load(f)
    print(f"✓ Loaded tuned model from {tuned_model_path}")
    
    # Load data
    print(f"\nLoading features...")
    X, y = load_data(features_file)
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Split data (same as tuning)
    val_idx = int(len(X) * (1 - val_size))
    X_train, X_test = X.iloc[:val_idx], X.iloc[val_idx:]
    y_train, y_test = y.iloc[:val_idx], y.iloc[val_idx:]
    
    # Use a subset for validation
    val_split = int(len(X_train) * 0.2)
    X_val, X_train_rest = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_val, y_train_rest = y_train.iloc[:val_split], y_train.iloc[val_split:]
    
    print(f"  Train: {len(X_train_rest)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize comparator
    comparator = ModelComparator(
        baseline_model, tuned_model,
        X_val, y_val,
        X_test, y_test
    )
    
    # Print comparison
    comparator.print_comparison()
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    comparator.plot_roc_curves(output_dir / 'roc_curves.png')
    comparator.plot_calibration_curves(output_dir / 'calibration_curves.png')
    comparator.plot_metric_comparison(output_dir / 'metric_comparison.png')
    comparator.plot_prediction_distributions(output_dir / 'prediction_distributions.png')
    
    # Save report
    comparator.save_comparison_report(output_dir / 'comparison_report.txt')
    
    # Save metrics as CSV
    df_metrics = comparator.get_comparison_table()
    df_metrics.to_csv(output_dir / 'metrics_table.csv')
    print(f"✓ Saved metrics table to {output_dir / 'metrics_table.csv'}")
    
    print(f"\n✓ All comparisons saved to {output_dir}/")
    
    return comparator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare baseline vs tuned models')
    parser.add_argument('--baseline', type=Path, default='models/xgb_baseline.pkl',
                        help='Path to baseline model')
    parser.add_argument('--tuned', type=Path, default='tuning_results/xgboost_best_model.pkl',
                        help='Path to tuned model')
    parser.add_argument('--features', type=Path, default='data/processed/features.csv',
                        help='Path to features CSV')
    parser.add_argument('--output-dir', type=Path, default='comparison_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    comparator = run_comparison(
        baseline_model_path=args.baseline,
        tuned_model_path=args.tuned,
        features_file=args.features,
        output_dir=args.output_dir
    )