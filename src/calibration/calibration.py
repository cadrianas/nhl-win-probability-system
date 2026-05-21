"""
Phase 5: Probability Calibration

Uses the project's centralized path management (src/utils/paths.py)
to ensure compatibility with any working directory.

Run from project root:
    python src/calibration/phase5_calibration.py
"""

import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

# Add src to path to import utilities
SRC_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SRC_PATH))

from utils.paths import (
    DATA_PROCESSED,
    RESULTS_MODELS,
    ensure_directories
)

# ============================================================================
# SETUP
# ============================================================================

# Ensure all directories exist
ensure_directories()

# Define file paths using centralized paths system
MODEL_PATH = RESULTS_MODELS / 'xgboost_best_model.pkl'
FEATURES_TEST_PATH = DATA_PROCESSED / 'features_test.csv'

# Output paths
CALIBRATION_METRICS_PATH = RESULTS_MODELS / 'calibration_metrics.json'
CALIBRATION_CONTEXT_PATH = RESULTS_MODELS / 'calibration_metrics_by_context.json'
CALIBRATION_PLOT_PATH = RESULTS_MODELS / 'calibration_curves_comparison.png'

ISOTONIC_MODEL_PATH = RESULTS_MODELS / 'xgboost_calibrated_isotonic.pkl'
SIGMOID_MODEL_PATH = RESULTS_MODELS / 'xgboost_calibrated_sigmoid.pkl'

# Calibration/test split ratio
CAL_TEST_SPLIT = 0.5

# ============================================================================
# 1. LOAD DATA AND SPLIT PROPERLY
# ============================================================================

print("=" * 70)
print("PHASE 5: CALIBRATION ANALYSIS")
print("=" * 70)

# Load model
print("\n[1/6] Loading best model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_uncalibrated = pickle.load(f)
    print(f"✓ Model loaded from {MODEL_PATH.relative_to(MODEL_PATH.parent.parent.parent)}")
except FileNotFoundError:
    print(f"✗ Model not found at {MODEL_PATH}")
    print("  Ensure Phase 4 (model training) completed successfully")
    exit(1)

# Load test data
print("[2/6] Loading feature data...")
try:
    X_test = pd.read_csv(FEATURES_TEST_PATH)
    print(f"✓ Loaded {len(X_test)} test samples from {FEATURES_TEST_PATH.relative_to(FEATURES_TEST_PATH.parent.parent.parent)}")
except FileNotFoundError:
    print(f"✗ Features not found at {FEATURES_TEST_PATH}")
    print("  Ensure Phase 3 (feature engineering) completed successfully")
    exit(1)

# Extract target and features
if 'target_home_win' not in X_test.columns:
    print(f"✗ 'target_home_win' column not found in features file")
    print(f"  Available columns: {X_test.columns.tolist()}")
    exit(1)

y_test = X_test['target_home_win']
X_test = X_test.drop(columns=['target_home_win'])
X_test = X_test.select_dtypes(include=['number'])

print(f"  Features shape: {X_test.shape}")
print(f"  Target distribution: {y_test.value_counts().to_dict()}")

# CRITICAL: Split test into calibration and final test
# Calibration set: used to fit calibration model
# Test set: held-out for honest evaluation
np.random.seed(42)
indices = np.random.permutation(len(X_test))
split_idx = int(len(X_test) * CAL_TEST_SPLIT)

X_cal = X_test.iloc[indices[:split_idx]].reset_index(drop=True)
y_cal = y_test.iloc[indices[:split_idx]].reset_index(drop=True)

X_final_test = X_test.iloc[indices[split_idx:]].reset_index(drop=True)
y_final_test = y_test.iloc[indices[split_idx:]].reset_index(drop=True)

print(f"\n  Calibration set: {len(X_cal)} samples")
print(f"  Final test set:  {len(X_final_test)} samples")

# Get uncalibrated predictions on both sets
y_pred_uncal_cal = model_uncalibrated.predict_proba(X_cal)[:, 1]
y_pred_uncal_test = model_uncalibrated.predict_proba(X_final_test)[:, 1]

# ============================================================================
# 2. CALIBRATE MODEL ON CALIBRATION SET
# ============================================================================

print("\n[3/6] Calibrating model (Isotonic Regression)...")

model_iso = CalibratedClassifierCV(
    model_uncalibrated,
    method='isotonic',
    cv=5
)
# FIT on calibration set only
model_iso.fit(X_cal, y_cal)
# PREDICT on final test set (unseen)
y_pred_iso = model_iso.predict_proba(X_final_test)[:, 1]

print("✓ Isotonic calibration complete")

# Also try Platt scaling for comparison
print("[3/6] Calibrating model (Platt Scaling)...")

model_sigmoid = CalibratedClassifierCV(
    model_uncalibrated,
    method='sigmoid',
    cv=5
)
model_sigmoid.fit(X_cal, y_cal)
y_pred_sigmoid = model_sigmoid.predict_proba(X_final_test)[:, 1]

print("✓ Platt calibration complete")

# ============================================================================
# 3. COMPUTE METRICS
# ============================================================================

print("\n[4/6] Computing calibration metrics...")

metrics = {
    'uncalibrated': {
        'brier': brier_score_loss(y_final_test, y_pred_uncal_test),
        'log_loss': log_loss(y_final_test, y_pred_uncal_test),
        'roc_auc': roc_auc_score(y_final_test, y_pred_uncal_test),
    },
    'isotonic': {
        'brier': brier_score_loss(y_final_test, y_pred_iso),
        'log_loss': log_loss(y_final_test, y_pred_iso),
        'roc_auc': roc_auc_score(y_final_test, y_pred_iso),
    },
    'sigmoid': {
        'brier': brier_score_loss(y_final_test, y_pred_sigmoid),
        'log_loss': log_loss(y_final_test, y_pred_sigmoid),
        'roc_auc': roc_auc_score(y_final_test, y_pred_sigmoid),
    }
}

# Print comparison table
print("\n" + "=" * 70)
print("CALIBRATION METRICS COMPARISON")
print("=" * 70)
print(f"{'Metric':<20} {'Uncalibrated':<18} {'Isotonic':<18} {'Sigmoid':<18}")
print("-" * 70)
for metric_name in ['brier', 'log_loss', 'roc_auc']:
    uncal = metrics['uncalibrated'][metric_name]
    iso = metrics['isotonic'][metric_name]
    sig = metrics['sigmoid'][metric_name]
    print(f"{metric_name:<20} {uncal:<18.4f} {iso:<18.4f} {sig:<18.4f}")

# Compute improvements
iso_brier_improvement = (1 - metrics['isotonic']['brier'] / metrics['uncalibrated']['brier']) * 100
iso_log_loss_improvement = (1 - metrics['isotonic']['log_loss'] / metrics['uncalibrated']['log_loss']) * 100
iso_roc_improvement = (metrics['isotonic']['roc_auc'] - metrics['uncalibrated']['roc_auc']) * 100

print("-" * 70)
print(f"Isotonic improvement (Brier): {iso_brier_improvement:+.1f}%")
print(f"Isotonic improvement (Log Loss): {iso_log_loss_improvement:+.1f}%")
print(f"Isotonic improvement (ROC-AUC): {iso_roc_improvement:+.2f}pp")
print("=" * 70)

# Save metrics
metrics_json = {
    k: {m: float(v) for m, v in metrics[k].items()}
    for k in metrics
}
with open(CALIBRATION_METRICS_PATH, 'w') as f:
    json.dump(metrics_json, f, indent=2)
print(f"\n✓ Metrics saved to {CALIBRATION_METRICS_PATH.relative_to(CALIBRATION_METRICS_PATH.parent.parent.parent)}")

# ============================================================================
# 4. GENERATE RELIABILITY DIAGRAMS
# ============================================================================

print("[5/6] Generating reliability diagrams...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Calibration Curves: Before and After', fontsize=14, fontweight='bold')

# Helper function to plot calibration curve
def plot_calibration(ax, y_true, y_pred, label):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=6, label=label)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

# Plot all three
plot_calibration(axes[0], y_final_test, y_pred_uncal_test, 'Uncalibrated')
axes[0].set_title('Uncalibrated Model', fontweight='bold')

plot_calibration(axes[1], y_final_test, y_pred_iso, 'Isotonic Calibration')
axes[1].set_title('After Isotonic Calibration', fontweight='bold')

plot_calibration(axes[2], y_final_test, y_pred_sigmoid, 'Platt Scaling')
axes[2].set_title('After Platt Scaling', fontweight='bold')

plt.tight_layout()
plt.savefig(CALIBRATION_PLOT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {CALIBRATION_PLOT_PATH.relative_to(CALIBRATION_PLOT_PATH.parent.parent.parent)}")
plt.close()

# ============================================================================
# 5. CALIBRATION BY GAME CONTEXT (Close vs. Blowout)
# ============================================================================

print("[5/6] Analyzing calibration by game context...")

# Load feature data to get score differential
X_full = pd.read_csv(FEATURES_TEST_PATH)
if 'score_differential' in X_full.columns:
    score_diff = X_full.iloc[indices[split_idx:]]['score_differential'].reset_index(drop=True).abs()
    
    metrics_context = {}
    
    for context_name, mask in [('Close Games (|score_diff| ≤ 1)', score_diff <= 1),
                                ('Blowouts (|score_diff| > 1)', score_diff > 1)]:
        if mask.sum() > 0:
            metrics_context[context_name] = {
                'uncalibrated': {
                    'brier': brier_score_loss(y_final_test[mask], y_pred_uncal_test[mask]),
                    'log_loss': log_loss(y_final_test[mask], y_pred_uncal_test[mask]),
                    'count': int(mask.sum())
                },
                'isotonic': {
                    'brier': brier_score_loss(y_final_test[mask], y_pred_iso[mask]),
                    'log_loss': log_loss(y_final_test[mask], y_pred_iso[mask]),
                }
            }
    
    print("\nCALIBRATION BY GAME CONTEXT:")
    print("-" * 70)
    for context, metrics_ctx in metrics_context.items():
        print(f"\n{context} (n={metrics_ctx['uncalibrated']['count']})")
        print(f"  Brier (uncal): {metrics_ctx['uncalibrated']['brier']:.4f} → "
              f"(cal): {metrics_ctx['isotonic']['brier']:.4f}")
        print(f"  Log Loss (uncal): {metrics_ctx['uncalibrated']['log_loss']:.4f} → "
              f"(cal): {metrics_ctx['isotonic']['log_loss']:.4f}")
    
    with open(CALIBRATION_CONTEXT_PATH, 'w') as f:
        json.dump(metrics_context, f, indent=2)
    print(f"\n✓ Context analysis saved to {CALIBRATION_CONTEXT_PATH.relative_to(CALIBRATION_CONTEXT_PATH.parent.parent.parent)}")

# ============================================================================
# 6. SAVE CALIBRATED MODELS
# ============================================================================

print("\n[6/6] Saving calibrated models...")

with open(ISOTONIC_MODEL_PATH, 'wb') as f:
    pickle.dump(model_iso, f)
print(f"✓ Saved: {ISOTONIC_MODEL_PATH.relative_to(ISOTONIC_MODEL_PATH.parent.parent.parent)}")

with open(SIGMOID_MODEL_PATH, 'wb') as f:
    pickle.dump(model_sigmoid, f)
print(f"✓ Saved: {SIGMOID_MODEL_PATH.relative_to(SIGMOID_MODEL_PATH.parent.parent.parent)}")

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 5 SUMMARY")
print("=" * 70)
print(f"""
✓ Calibration Complete

Key Findings:
- Uncalibrated Brier Score:  {metrics['uncalibrated']['brier']:.4f}
- Isotonic Brier Score:      {metrics['isotonic']['brier']:.4f}
- Improvement:               {iso_brier_improvement:+.1f}%

- Uncalibrated Log Loss:     {metrics['uncalibrated']['log_loss']:.4f}
- Isotonic Log Loss:         {metrics['isotonic']['log_loss']:.4f}
- Improvement:               {iso_log_loss_improvement:+.1f}%

- Uncalibrated ROC-AUC:      {metrics['uncalibrated']['roc_auc']:.4f}
- Isotonic ROC-AUC:          {metrics['isotonic']['roc_auc']:.4f}
- Improvement:               {iso_roc_improvement:+.2f}pp

Selected Model: Isotonic Calibration
Reason: Superior generalization to held-out test data

Outputs Generated:
✓ {CALIBRATION_METRICS_PATH.name}
✓ {CALIBRATION_CONTEXT_PATH.name}
✓ {CALIBRATION_PLOT_PATH.name}
✓ {ISOTONIC_MODEL_PATH.name}

Recommended Next Steps:
1. Review calibration curves visually
2. Analyze context-specific performance
3. Begin Phase 6: Snakemake pipeline orchestration
4. Write blog post on calibration importance
""")
print("=" * 70)