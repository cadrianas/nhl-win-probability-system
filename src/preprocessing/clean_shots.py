"""
Main entry point for shot cleaning.

Orchestrates:
1. Load raw data
2. Standardize columns
3. Enforce types
4. Handle missing values
5. Filter OT/shootouts
6. Create time features
7. Create strength states
8. Add target variable
9. Save cleaned parquet
"""

import pandas as pd
from pathlib import Path

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
from src.preprocessing.continuous_time import create_time_features
from src.preprocessing.strength_state import (
    add_strength_state_column,
    create_strength_features,
)
from src.preprocessing.target_variable import add_target_to_shots


def main():
    """
    Load raw MoneyPuck shots, clean, and save.
    """
    print("=" * 60)
    print("PHASE 1: CLEAN SHOTS")
    print("=" * 60)

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

    # Time
    print("\n7. Creating continuous time features...")
    shots = create_time_features(shots)

    # Strength
    print("\n8. Adding strength state...")
    shots = add_strength_state_column(shots)
    shots = create_strength_features(shots)

    # Target
    print("\n9. Adding target variable...")
    shots = add_target_to_shots(shots, exclude_shootouts=True)

    # Save
    print("\n10. Saving cleaned shots...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(DATA_PROCESSED / "shots_cleaned.parquet", compression="snappy")
    print(f"   Saved to: {DATA_PROCESSED / 'shots_cleaned.parquet'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Rows: {len(shots):,}")
    print(f"Games: {shots['game_id'].nunique():,}")
    print(
        f"Regulation: {(shots['game_state'] == 'REG').sum() / len(shots) * 100:.1f}%"
    )
    print(
        f"Playoff OT: {(shots['game_state'] == 'PLAYOFF_OT').sum() / len(shots) * 100:.1f}%"
    )
    print(
        f"RS OT: {(shots['game_state'] == 'RS_OT_3V3').sum() / len(shots) * 100:.1f}%"
    )
    print(f"Home win rate: {shots['target_home_win'].mean():.1%}")

    print("\n✅ CLEAN SHOTS COMPLETE\n")

    return shots


if __name__ == "__main__":
    main()
