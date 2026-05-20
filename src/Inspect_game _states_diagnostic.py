"""
Debug: Check raw game_states.parquet cumulative values
This will show if the problem is in Phase 1 (cumulative computation)
"""

import polars as pl
import pandas as pd
from pathlib import Path

game_states_path = Path("data/processed/game_states.parquet")

if not game_states_path.exists():
    print(f"ERROR: {game_states_path} not found!")
    exit(1)

print("\n" + "=" * 100)
print("RAW GAME_STATES.PARQUET INSPECTION")
print("=" * 100)

# Load lazily first
df_lazy = pl.scan_parquet(game_states_path)

print(f"\nLoading sample of 1000 rows...")
sample = df_lazy.limit(1000).collect()

print(f"✓ Loaded {len(sample)} rows")

# Check cumulative columns
print("\n" + "-" * 100)
print("CUMULATIVE COLUMNS ANALYSIS:")
print("-" * 100)

if 'cumulative_home_xg' in sample.columns:
    home_xg = sample['cumulative_home_xg']
    print(f"\ncumulative_home_xg:")
    print(f"  Min: {home_xg.min()}")
    print(f"  Max: {home_xg.max()}")
    print(f"  Mean: {home_xg.mean()}")
    print(f"  Unique values: {home_xg.n_unique()}")
    print(f"  Zeros: {(home_xg == 0).sum()}/{len(home_xg)}")
    
    if home_xg.max() == 0:
        print(f"  ⚠️  WARNING: All values are zero!")
else:
    print(f"  ✗ NOT FOUND")

if 'cumulative_away_xg' in sample.columns:
    away_xg = sample['cumulative_away_xg']
    print(f"\ncumulative_away_xg:")
    print(f"  Min: {away_xg.min()}")
    print(f"  Max: {away_xg.max()}")
    print(f"  Mean: {away_xg.mean()}")
    print(f"  Unique values: {away_xg.n_unique()}")
    print(f"  Zeros: {(away_xg == 0).sum()}/{len(away_xg)}")
    
    if away_xg.max() == 0:
        print(f"  ⚠️  WARNING: All values are zero!")
else:
    print(f"  ✗ NOT FOUND")

# Check if they're equal (would result in xg_differential = 0)
if 'cumulative_home_xg' in sample.columns and 'cumulative_away_xg' in sample.columns:
    equal_count = (sample['cumulative_home_xg'] == sample['cumulative_away_xg']).sum()
    print(f"\ncumulative_home_xg == cumulative_away_xg:")
    print(f"  Rows where equal: {equal_count}/{len(sample)}")
    
    if equal_count == len(sample):
        print(f"  ⚠️  CRITICAL: They're always equal! (xg_differential will always be 0)")

# Check raw xG values if available
print("\n" + "-" * 100)
print("RAW xG COLUMNS (if available):")
print("-" * 100)

if 'xGoal' in sample.columns:
    xg = sample['xGoal']
    print(f"\nxGoal:")
    print(f"  Min: {xg.min()}")
    print(f"  Max: {xg.max()}")
    print(f"  Mean: {xg.mean()}")
    print(f"  Non-zero: {(xg > 0).sum()}/{len(xg)}")
else:
    print(f"✗ xGoal column not found")

if 'xG' in sample.columns:
    xg = sample['xG']
    print(f"\nxG:")
    print(f"  Min: {xg.min()}")
    print(f"  Max: {xg.max()}")
    print(f"  Mean: {xg.mean()}")
    print(f"  Non-zero: {(xg > 0).sum()}/{len(xg)}")
else:
    print(f"✗ xG column not found")

# Sample a few rows to see structure
print("\n" + "-" * 100)
print("SAMPLE ROWS (first 5):")
print("-" * 100)

if 'cumulative_home_xg' in sample.columns and 'cumulative_away_xg' in sample.columns:
    cols_to_show = ['game_id', 'shot_id', 'cumulative_home_xg', 'cumulative_away_xg', 'xg_differential']
    available_cols = [c for c in cols_to_show if c in sample.columns]
    
    print(f"\n{available_cols}")
    for i in range(min(10, len(sample))):
        row = sample.select(available_cols).slice(i, 1)
        print(f"{row.to_dict(as_series=False)}")

print("\n" + "=" * 100)
print("DIAGNOSIS:")
print("=" * 100)

# Final assessment
if 'cumulative_home_xg' in sample.columns and 'cumulative_away_xg' in sample.columns:
    home_max = sample['cumulative_home_xg'].max()
    away_max = sample['cumulative_away_xg'].max()
    
    if home_max == 0 and away_max == 0:
        print("\n❌ CRITICAL ISSUE: cumulative_home_xg and cumulative_away_xg are ALL ZEROS")
        print("   This is a Phase 1 bug - cumulative xG was never computed")
        print("\n   FIX: Need to fix Phase 1 to properly compute cumulative xG values")
    elif home_max == away_max and home_max != 0:
        print("\n❌ ISSUE: cumulative values are identical for both teams")
        print("   This suggests they might be total game xG instead of cumulative shot-by-shot")
    else:
        print("\n✓ cumulative values look reasonable")
        print("   The problem might be in how Phase 2 is recomputing them")
else:
    print("\n❌ CRITICAL: cumulative columns not found in source data")
    print("   Phase 1 did not create these columns")

print("=" * 100)