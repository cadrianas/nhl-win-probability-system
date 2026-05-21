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
CALIBRATION_ECE_PLOT_PATH = RESULTS_MODELS / 'calibration_error_metrics.png'

ISOTONIC_MODEL_PATH = RESULTS_MODELS / 'xgboost_calibrated_isotonic.pkl'
SIGMOID_MODEL_PATH = RESULTS_MODELS / 'xgboost_calibrated_sigmoid.pkl'

# Calibration/test split ratio
CAL_TEST_SPLIT = 0.5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_ece(y_true, y_pred, n_bins=10):
    """
    Expected Calibration Error: weighted average of |accuracy - confidence|
    across probability bins. Lower is better; 0 = perfect calibration.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        accuracy   = np.mean(y_true[mask] == np.round(y_pred[mask]))
        confidence = np.mean(y_pred[mask])
        ece += (mask.sum() / len(y_true)) * abs(accuracy - confidence)
    return ece


def compute_mce(y_true, y_pred, n_bins=10):
    """
    Maximum Calibration Error: largest |actual - predicted| across bins.
    Lower is better; 0 = perfect calibration.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    return float(np.max(np.abs(prob_true - prob_pred)))


def context_metrics(y_true, y_pred_uncal, y_pred_cal, mask):
    """Return a compact metrics dict for a boolean mask over the test set."""
    yt = y_true[mask]
    yu = y_pred_uncal[mask]
    yc = y_pred_cal[mask]
    return {
        'count': int(mask.sum()),
        'uncalibrated': {
            'brier':    float(brier_score_loss(yt, yu)),
            'log_loss': float(log_loss(yt, yu)),
        },
        'isotonic': {
            'brier':    float(brier_score_loss(yt, yc)),
            'log_loss': float(log_loss(yt, yc)),
        },
    }

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
    X_test_full = pd.read_csv(FEATURES_TEST_PATH)  # keep full df for context slicing
    print(f"✓ Loaded {len(X_test_full)} test samples from {FEATURES_TEST_PATH.relative_to(FEATURES_TEST_PATH.parent.parent.parent)}")
except FileNotFoundError:
    print(f"✗ Features not found at {FEATURES_TEST_PATH}")
    print("  Ensure Phase 3 (feature engineering) completed successfully")
    exit(1)

# Extract target and numeric features for model
if 'target_home_win' not in X_test_full.columns:
    print(f"✗ 'target_home_win' column not found in features file")
    print(f"  Available columns: {X_test_full.columns.tolist()}")
    exit(1)

y_test = X_test_full['target_home_win']
X_test = X_test_full.drop(columns=['target_home_win']).select_dtypes(include=['number'])

print(f"  Features shape: {X_test.shape}")
print(f"  Target distribution: {y_test.value_counts().to_dict()}")

# CRITICAL: Split test into calibration and final test
np.random.seed(42)
indices = np.random.permutation(len(X_test))
split_idx = int(len(X_test) * CAL_TEST_SPLIT)

X_cal = X_test.iloc[indices[:split_idx]].reset_index(drop=True)
y_cal = y_test.iloc[indices[:split_idx]].reset_index(drop=True)

X_final_test = X_test.iloc[indices[split_idx:]].reset_index(drop=True)
y_final_test = y_test.iloc[indices[split_idx:]].reset_index(drop=True)

# Keep the full (non-numeric) rows for the final test set for context slicing
X_context = X_test_full.iloc[indices[split_idx:]].reset_index(drop=True)

print(f"\n  Calibration set: {len(X_cal)} samples")
print(f"  Final test set:  {len(X_final_test)} samples")

# Uncalibrated predictions
y_pred_uncal_cal  = model_uncalibrated.predict_proba(X_cal)[:, 1]
y_pred_uncal_test = model_uncalibrated.predict_proba(X_final_test)[:, 1]

# ============================================================================
# 2. CALIBRATE MODEL ON CALIBRATION SET
# ============================================================================

print("\n[3/6] Calibrating model (Isotonic Regression)...")
model_iso = CalibratedClassifierCV(model_uncalibrated, method='isotonic', cv=5)
model_iso.fit(X_cal, y_cal)
y_pred_iso = model_iso.predict_proba(X_final_test)[:, 1]
print("✓ Isotonic calibration complete")

print("[3/6] Calibrating model (Platt Scaling)...")
model_sigmoid = CalibratedClassifierCV(model_uncalibrated, method='sigmoid', cv=5)
model_sigmoid.fit(X_cal, y_cal)
y_pred_sigmoid = model_sigmoid.predict_proba(X_final_test)[:, 1]
print("✓ Platt calibration complete")

# ============================================================================
# 3. COMPUTE METRICS  (Brier, Log Loss, AUC, ECE, MCE)
# ============================================================================

print("\n[4/6] Computing calibration metrics...")

metrics = {
    'uncalibrated': {
        'brier':    brier_score_loss(y_final_test, y_pred_uncal_test),
        'log_loss': log_loss(y_final_test, y_pred_uncal_test),
        'roc_auc':  roc_auc_score(y_final_test, y_pred_uncal_test),
        'ece':      compute_ece(y_final_test, y_pred_uncal_test),
        'mce':      compute_mce(y_final_test, y_pred_uncal_test),
    },
    'isotonic': {
        'brier':    brier_score_loss(y_final_test, y_pred_iso),
        'log_loss': log_loss(y_final_test, y_pred_iso),
        'roc_auc':  roc_auc_score(y_final_test, y_pred_iso),
        'ece':      compute_ece(y_final_test, y_pred_iso),
        'mce':      compute_mce(y_final_test, y_pred_iso),
    },
    'sigmoid': {
        'brier':    brier_score_loss(y_final_test, y_pred_sigmoid),
        'log_loss': log_loss(y_final_test, y_pred_sigmoid),
        'roc_auc':  roc_auc_score(y_final_test, y_pred_sigmoid),
        'ece':      compute_ece(y_final_test, y_pred_sigmoid),
        'mce':      compute_mce(y_final_test, y_pred_sigmoid),
    },
}

# Print comparison table
print("\n" + "=" * 70)
print("CALIBRATION METRICS COMPARISON")
print("=" * 70)
print(f"{'Metric':<20} {'Uncalibrated':<18} {'Isotonic':<18} {'Sigmoid':<18}")
print("-" * 70)
for metric_name in ['brier', 'log_loss', 'roc_auc', 'ece', 'mce']:
    uncal = metrics['uncalibrated'][metric_name]
    iso   = metrics['isotonic'][metric_name]
    sig   = metrics['sigmoid'][metric_name]
    print(f"{metric_name:<20} {uncal:<18.4f} {iso:<18.4f} {sig:<18.4f}")

iso_brier_improvement    = (1 - metrics['isotonic']['brier']    / metrics['uncalibrated']['brier'])    * 100
iso_log_loss_improvement = (1 - metrics['isotonic']['log_loss'] / metrics['uncalibrated']['log_loss']) * 100
iso_roc_improvement      = (metrics['isotonic']['roc_auc'] - metrics['uncalibrated']['roc_auc']) * 100
iso_ece_improvement      = (1 - metrics['isotonic']['ece'] / metrics['uncalibrated']['ece']) * 100

print("-" * 70)
print(f"Isotonic improvement (Brier):    {iso_brier_improvement:+.1f}%")
print(f"Isotonic improvement (Log Loss): {iso_log_loss_improvement:+.1f}%")
print(f"Isotonic improvement (ROC-AUC):  {iso_roc_improvement:+.2f}pp")
print(f"Isotonic improvement (ECE):      {iso_ece_improvement:+.1f}%")
print("=" * 70)

# Save metrics
with open(CALIBRATION_METRICS_PATH, 'w') as f:
    json.dump({k: {m: float(v) for m, v in metrics[k].items()} for k in metrics}, f, indent=2)
print(f"\n✓ Metrics saved to {CALIBRATION_METRICS_PATH.relative_to(CALIBRATION_METRICS_PATH.parent.parent.parent)}")

# ============================================================================
# 4. RELIABILITY DIAGRAMS + ECE/MCE BAR CHART
# ============================================================================

print("[5/6] Generating reliability diagrams...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Calibration Curves: Before and After', fontsize=14, fontweight='bold')

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

# ECE / MCE bar chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Calibration Error Metrics', fontsize=13, fontweight='bold')

methods = ['Uncalibrated', 'Isotonic', 'Sigmoid']
colors  = ['steelblue', 'coral', 'seagreen']

for ax, metric_key, title in [
    (axes[0], 'ece', 'Expected Calibration Error (lower = better)'),
    (axes[1], 'mce', 'Maximum Calibration Error (lower = better)'),
]:
    vals = [metrics[m][metric_key] for m in ['uncalibrated', 'isotonic', 'sigmoid']]
    bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim([0, max(vals) * 1.2])
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(CALIBRATION_ECE_PLOT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {CALIBRATION_ECE_PLOT_PATH.relative_to(CALIBRATION_ECE_PLOT_PATH.parent.parent.parent)}")
plt.close()

# ============================================================================
# 5. CALIBRATION BY GAME CONTEXT
# ============================================================================

print("[5/6] Analyzing calibration by game context...")

context_results = {}

# --- Score differential (original) ---
if 'score_differential' in X_context.columns:
    abs_diff = X_context['score_differential'].abs()
    context_results['by_score_differential'] = {
        'Close (|diff| ≤ 1)':   context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, abs_diff <= 1),
        'Blowout (|diff| > 1)': context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, abs_diff > 1),
    }

# --- Period ---
if 'period' in X_context.columns:
    period_results = {}
    for p in sorted(X_context['period'].dropna().unique()):
        mask = (X_context['period'] == p).values
        if mask.sum() > 50:
            label = f'Period {int(p)}' if p <= 3 else 'OT'
            period_results[label] = context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, mask)
    context_results['by_period'] = period_results

# --- Time remaining (within-period) ---
if 'time_remaining_seconds' in X_context.columns:
    t = X_context['time_remaining_seconds']
    context_results['by_time_remaining'] = {
        'Final 5 min (≤ 300s)':    context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, (t <= 300).values),
        'Mid period (300–900s)':   context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, ((t > 300) & (t <= 900)).values),
        'Early period (> 900s)':   context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, (t > 900).values),
    }

# --- Strength state ---
if 'strength_state' in X_context.columns:
    strength_results = {}
    for state in X_context['strength_state'].dropna().unique():
        mask = (X_context['strength_state'] == state).values
        if mask.sum() > 200:  # skip tiny samples
            strength_results[str(state)] = context_metrics(y_final_test, y_pred_uncal_test, y_pred_iso, mask)
    context_results['by_strength_state'] = strength_results

# Print context summary
print("\nCALIBRATION BY GAME CONTEXT:")
for group_name, group in context_results.items():
    print(f"\n  {group_name.replace('_', ' ').title()}")
    print(f"  {'Context':<28} {'n':>7}  {'Brier (uncal)':>14}  {'Brier (iso)':>12}  {'Δ':>8}")
    print(f"  {'-'*75}")
    for ctx_label, ctx_vals in group.items():
        n        = ctx_vals['count']
        b_uncal  = ctx_vals['uncalibrated']['brier']
        b_cal    = ctx_vals['isotonic']['brier']
        delta    = b_cal - b_uncal
        print(f"  {ctx_label:<28} {n:>7,}  {b_uncal:>14.4f}  {b_cal:>12.4f}  {delta:>+8.4f}")

with open(CALIBRATION_CONTEXT_PATH, 'w') as f:
    json.dump(context_results, f, indent=2, default=str)
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
  Brier:    {metrics['uncalibrated']['brier']:.4f} → {metrics['isotonic']['brier']:.4f}  ({iso_brier_improvement:+.1f}%)
  Log Loss: {metrics['uncalibrated']['log_loss']:.4f} → {metrics['isotonic']['log_loss']:.4f}  ({iso_log_loss_improvement:+.1f}%)
  ROC-AUC:  {metrics['uncalibrated']['roc_auc']:.4f} → {metrics['isotonic']['roc_auc']:.4f}  ({iso_roc_improvement:+.2f}pp)
  ECE:      {metrics['uncalibrated']['ece']:.4f} → {metrics['isotonic']['ece']:.4f}  ({iso_ece_improvement:+.1f}%)

Selected Model: Isotonic Calibration

Outputs:
  ✓ {CALIBRATION_METRICS_PATH.name}
  ✓ {CALIBRATION_CONTEXT_PATH.name}
  ✓ {CALIBRATION_PLOT_PATH.name}
  ✓ {CALIBRATION_ECE_PLOT_PATH.name}
  ✓ {ISOTONIC_MODEL_PATH.name}
""")
print("=" * 70)