#!/usr/bin/env python3
"""
Test if model probabilities are inverted (FIXED: proper dtype handling).
"""
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, 'src')
from utils.paths import DATA_PROCESSED, RESULTS_MODELS

print("=" * 70)
print("INVERSION DIAGNOSTIC (V2 - Fixed dtype)")
print("=" * 70)

# 1. Check target correlation with score_differential
print("\n1. Training data correlation:")
features_train = pd.read_csv(DATA_PROCESSED / 'features_train.csv', low_memory=False, nrows=10000)

# Convert to numeric
for col in features_train.columns:
    features_train[col] = pd.to_numeric(features_train[col], errors='coerce')

if 'target_home_win' in features_train.columns and 'score_differential' in features_train.columns:
    corr = features_train[['score_differential', 'target_home_win']].corr().iloc[0, 1]
    print(f"   Corr(score_differential, target_home_win) = {corr:.4f}")
    print(f"   → Higher score diff should correlate with more home wins")
else:
    print("   ⚠️  Could not find required columns")

# 2. Load model and test with REALISTIC feature values
print("\n2. Model prediction test (with realistic features):")
try:
    model = pickle.load(open(RESULTS_MODELS / 'xgboost_calibrated_isotonic.pkl', 'rb'))
    print(f"   ✓ Model loaded")
    
    # Get feature names from model
    base = model.calibrated_classifiers_[0].estimator
    booster = base.get_booster()
    feat_names = booster.feature_names
    print(f"   Model expects {len(feat_names)} features")
    
    # Use MEAN feature values from training data
    feature_means = features_train[feat_names].mean()
    
    # Create test cases: use mean features, vary only score_differential
    X_test = pd.DataFrame(index=range(5), columns=feat_names)
    for col in feat_names:
        X_test[col] = feature_means[col]
    
    X_test['score_differential'] = [-2, -1, 0, 1, 2]
    X_test = X_test.astype(float)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n   Score Diff → Home Win Prob:")
    for sd, prob in zip(X_test['score_differential'], probs):
        print(f"      {sd:+.0f}  →  {prob:.4f}")
    
    # Check monotonicity
    is_increasing = all(probs[i] <= probs[i+1] for i in range(len(probs)-1))
    is_decreasing = all(probs[i] >= probs[i+1] for i in range(len(probs)-1))
    
    print()
    if is_increasing:
        print("   ✓ CORRECT: Probs increase with score_differential")
        print("   → Model is working correctly. No flip needed.")
    elif is_decreasing:
        print("   ✗ INVERTED: Probs DECREASE with score_differential")
        print("   → Model is flipped. Fix: return 1 - probs in dashboard")
    else:
        print("   ⚠️  Non-monotonic (neither increasing nor decreasing)")
        print("   → Something else is wrong")
    
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 3. Real data spot check
print("\n3. Real game spot check:")
try:
    features_test = pd.read_csv(DATA_PROCESSED / 'features_test.csv', low_memory=False, nrows=2000)
    
    # Convert to numeric
    for col in features_test.columns:
        features_test[col] = pd.to_numeric(features_test[col], errors='coerce')
    
    # Find a game where home won decisively (score_diff > 1 at end)
    home_win_games = features_test[
        (features_test['score_differential'] > 1) & 
        (features_test['target_home_win'] == 1)
    ].dropna()
    
    away_win_games = features_test[
        (features_test['score_differential'] < -1) & 
        (features_test['target_home_win'] == 0)
    ].dropna()
    
    if not home_win_games.empty:
        sample = home_win_games.iloc[-1]
        print(f"   Home won with score_diff={sample['score_differential']:.0f}")
        
        X_sample = sample[feat_names].to_frame().T.astype(float)
        prob = model.predict_proba(X_sample)[0, 1]
        print(f"   Model predicts home win prob = {prob:.4f}")
        if prob > 0.5:
            print(f"   ✓ CORRECT: High prob when home wins")
        else:
            print(f"   ✗ INVERTED: Low prob when home wins")
    
    if not away_win_games.empty:
        sample = away_win_games.iloc[-1]
        print(f"   Away won with score_diff={sample['score_differential']:.0f}")
        
        X_sample = sample[feat_names].to_frame().T.astype(float)
        prob = model.predict_proba(X_sample)[0, 1]
        print(f"   Model predicts home win prob = {prob:.4f}")
        if prob < 0.5:
            print(f"   ✓ CORRECT: Low prob when away wins")
        else:
            print(f"   ✗ INVERTED: High prob when away wins")
        
except Exception as e:
    print(f"   ⚠️  Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)