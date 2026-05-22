"""
Diagnostic script to check what columns exist in shots_cleaned.parquet
and help fix the game state builder.
"""
import pandas as pd
from pathlib import Path

import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import DATA_PROCESSED

# Adjust path to your local project
shots_path = DATA_PROCESSED / "shots_cleaned.parquet"

if shots_path.exists():
    shots = pd.read_parquet(shots_path)
    
    print("=" * 70)
    print(f"SHOTS_CLEANED.PARQUET DIAGNOSTICS")
    print("=" * 70)
    print(f"\nShape: {shots.shape}")
    print(f"Columns: {len(shots.columns)}")
    
    print("\n--- ALL COLUMNS ---")
    for i, col in enumerate(sorted(shots.columns), 1):
        print(f"  {i:3d}. {col}")
    
    # Check for time-related columns
    print("\n--- TIME-RELATED COLUMNS ---")
    time_cols = [c for c in shots.columns if 'time' in c.lower()]
    print(f"Found {len(time_cols)}: {time_cols}")
    
    # Check for coordinate columns
    print("\n--- COORDINATE COLUMNS ---")
    coord_cols = [c for c in shots.columns if any(x in c.lower() for x in ['coord', 'x_', 'y_', 'distance', 'angle'])]
    print(f"Found {len(coord_cols)}:")
    for col in sorted(coord_cols):
        print(f"  - {col}")
    
    # Check what's missing from GAME_STATE_COLUMNS
    print("\n--- CHECKING AGAINST GAME_STATE_COLUMNS ---")
    
    required = [
        'time_elapsed', 'time_in_period_seconds', 'time_in_period_minutes',
        'x_coord', 'y_coord', 'shot_distance', 'shot_angle',
        'cumulative_home_xg', 'cumulative_away_xg', 'xg_differential',
        'cumulative_home_shots', 'cumulative_away_shots', 'shot_differential',
        'shots_last_2min_home', 'shots_last_2min_away',
        'shots_last_5min_home', 'shots_last_5min_away',
        'game_seconds_elapsed', 'game_seconds_remaining',
        'score_differential', 'strength_state', 'is_even_strength',
        'is_power_play', 'is_empty_net', 'is_3v3', 'game_state',
        'target_home_win'
    ]
    
    missing = [col for col in required if col not in shots.columns]
    existing = [col for col in required if col in shots.columns]
    
    print(f"\nExisting in shots_cleaned ({len(existing)}):")
    for col in sorted(existing):
        print(f"  ✓ {col}")
    
    print(f"\nMissing from shots_cleaned ({len(missing)}):")
    for col in sorted(missing):
        print(f"  ✗ {col}")
    
    print("\n" + "=" * 70)
    
else:
    print(f"File not found: {shots_path}")
    print("\nTry running with the correct path to your shots_cleaned.parquet file")