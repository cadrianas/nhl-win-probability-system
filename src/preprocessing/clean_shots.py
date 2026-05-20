"""
Main entry point for shot cleaning.

Orchestrates:
1. Load raw data
2. Standardize columns
3. Enforce types
4. Handle missing values
5. Filter OT/shootouts
6. Create game state classification
7. CREATE TIME FEATURES ← THIS WAS MISSING!
8. Create strength states
8b. CREATE UNIQUE GAME_ID ← THIS WAS MISSING! (fixes duplicate game_ids across seasons)
9. Add target variable
10. Save cleaned parquet
"""

import sys
from pathlib import Path

# Ensure project root is on the path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.utils.paths import DATA_RAW, DATA_PROCESSED
from src.preprocessing.shot_cleaners import (
    standardize_column_names,
    enforce_dtypes,
    handle_missing_values,
)
from src.preprocessing.ot_and_shootout_logic import (
    filter_out_shootouts,
    add_game_state_column,
)
from src.preprocessing.continuous_time import create_time_features  # ← ADD THIS IMPORT
from src.preprocessing.strength_state import (
    add_strength_state_column,
    create_strength_features,
)
from src.preprocessing.target_variable import add_target_to_shots


def main():
    """
    Load raw MoneyPuck shots, clean, and save.
    """
    print("=" * 70)
    print("PHASE 1: CLEAN SHOTS")
    print("=" * 70)

    # Load all seasons
    print("\n1. Loading raw data...")
    dfs = []
    for csv_file in sorted(DATA_RAW.glob("shots_*.csv")):
        print(f"   Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        dfs.append(df)

    shots_raw = pd.concat(dfs, ignore_index=True)
    print(f"   Total rows: {len(shots_raw):,}")

    # Clean
    print("\n2. Standardizing column names...")
    shots = standardize_column_names(shots_raw)
    print(f"   Columns: {len(shots.columns)}")

    print("\n3. Enforcing data types...")
    shots = enforce_dtypes(shots)
    print(f"   Memory: {shots.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\n4. Handling missing values...")
    shots = handle_missing_values(shots)

    # OT/Shootout
    print("\n5. Filtering shootout games...")
    shots = filter_out_shootouts(shots, keep_shootouts=False)

    print("\n6. Adding game state (REG/RS_OT_3V3/PLAYOFF_OT)...")
    shots = add_game_state_column(shots)

    # TIME FEATURES ← THIS STEP WAS MISSING!
    print("\n7. Creating time features (CRITICAL STEP)...")
    print("   ⚠️  Make sure you're using continuous_time_PHASE1.py")
    print("   ⚠️  NOT the original continuous_time.py")
    shots = create_time_features(shots)
    
    # Verify time features were created
    time_cols = ['game_seconds_elapsed', 'game_seconds_remaining', 'time_decay_factor']
    for col in time_cols:
        if col not in shots.columns:
            print(f"   ❌ ERROR: {col} not created!")
            print(f"   This means continuous_time.py is WRONG")
            return None
        print(f"   ✓ {col} added")

    # Strength
    print("\n8. Adding strength state...")
    shots = add_strength_state_column(shots)
    shots = create_strength_features(shots)

    # CREATE UNIQUE GAME_ID ACROSS ALL SEASONS
    print("\n8b. Creating unique game_id (fixing duplicate IDs across seasons)...")
    print("   ⚠️  CRITICAL FIX: game_id in MoneyPuck repeats across seasons!")
    print("   Creating game_id_unique = season * 100000 + game_id")
    
    shots['game_id_unique'] = (shots['season'].astype('int64') * 100000) + shots['game_id'].astype('int64')
    
    # Verify uniqueness
    total_games_before = shots['game_id'].nunique()
    total_games_after = shots['game_id_unique'].nunique()
    print(f"   Before: {total_games_before:,} unique game_ids (BROKEN - repeated across seasons)")
    print(f"   After:  {total_games_after:,} unique game_id_unique (FIXED - truly unique)")
    
    if total_games_after >= 13000:
        print(f"   ✓ PASS! ~{total_games_after:,} games across 12 seasons is correct")
    else:
        print(f"   ⚠️  WARNING: Still low at {total_games_after:,}. Check raw data...")

    # Target
    print("\n9. Adding target variable...")
    shots = add_target_to_shots(shots, exclude_shootouts=True)

    # Save
    print("\n10. Saving cleaned shots...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(
        DATA_PROCESSED / "shots_cleaned.parquet",
        compression="snappy",
        index=False
    )
    print(f"   Saved to: {DATA_PROCESSED / 'shots_cleaned.parquet'}")
    print(f"   Size: {(DATA_PROCESSED / 'shots_cleaned.parquet').stat().st_size / 1e6:.1f} MB")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Rows: {len(shots):,}")
    print(f"Unique games (old game_id): {shots['game_id'].nunique():,}")
    print(f"Unique games (new game_id_unique): {shots['game_id_unique'].nunique():,}")
    
    if 'game_state' in shots.columns:
        print(f"\nGame state distribution:")
        print(f"  Regulation: {(shots['game_state'] == 'REG').sum() / len(shots) * 100:.1f}%")
        print(f"  Playoff OT: {(shots['game_state'] == 'PLAYOFF_OT').sum() / len(shots) * 100:.1f}%")
        print(f"  RS OT: {(shots['game_state'] == 'RS_OT_3V3').sum() / len(shots) * 100:.1f}%")

    if 'target_home_win' in shots.columns:
        print(f"\nTarget variable:")
        print(f"  Home win rate: {shots['target_home_win'].mean():.1%}")

    # Verify all critical columns exist
    print(f"\nCritical columns:")
    critical = [
        'game_id', 'game_id_unique', 'period', 'time_elapsed',
        'game_seconds_elapsed', 'game_seconds_remaining',
        'game_state', 'strength_state', 'target_home_win'
    ]
    all_present = True
    for col in critical:
        if col in shots.columns:
            print(f"   ✓ {col}")
        else:
            print(f"   ✗ {col} MISSING!")
            all_present = False
    
    if not all_present:
        print("\n  ERROR: Some critical columns are missing!")
        print("   This will cause errors in create_game_states.py")
        return None

    print("\n" + "=" * 70)
    print("CLEAN SHOTS COMPLETE")
    print("=" * 70)
    print(f"\nNext step:")
    print(f"  python src/preprocessing/create_game_states.py\n")

    return shots


if __name__ == "__main__":
    shots = main()