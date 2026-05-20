"""
Debug: Check if cumulative xG was actually computed in game_states.parquet
"""

import polars as pl
from pathlib import Path

gs_path = Path("data/processed/game_states.parquet")

print("\n" + "=" * 100)
print("CHECKING GAME_STATES.PARQUET CUMULATIVE VALUES")
print("=" * 100)

# Load sample
df = pl.read_parquet(gs_path, n_rows=1000)

print(f"\nLoaded {len(df)} sample rows")
print(f"\nColumns: {df.columns}")

# Check cumulative columns
print("\n" + "-" * 100)
print("CUMULATIVE xG INSPECTION:")
print("-" * 100)

if 'cumulative_home_xg' in df.columns:
    col = df['cumulative_home_xg']
    print(f"\ncumulative_home_xg:")
    print(f"  Min: {col.min()}")
    print(f"  Max: {col.max()}")
    print(f"  Mean: {col.mean()}")
    print(f"  Unique values: {col.n_unique()}")
    
    # Show sample values
    print(f"\n  Sample values (first 20):")
    for val in col.head(20):
        print(f"    {val}")
else:
    print("\n✗ cumulative_home_xg NOT FOUND")

if 'cumulative_away_xg' in df.columns:
    col = df['cumulative_away_xg']
    print(f"\ncumulative_away_xg:")
    print(f"  Min: {col.min()}")
    print(f"  Max: {col.max()}")
    print(f"  Mean: {col.mean()}")
    print(f"  Unique values: {col.n_unique()}")
else:
    print("\n✗ cumulative_away_xg NOT FOUND")

# Check x_goal (raw)
if 'x_goal' in df.columns:
    col = df['x_goal']
    print(f"\nx_goal (raw shot xG):")
    print(f"  Min: {col.min()}")
    print(f"  Max: {col.max()}")
    print(f"  Mean: {col.mean()}")
else:
    print("\n✗ x_goal NOT FOUND")

# Check by game_id_unique
print("\n" + "-" * 100)
print("SAMPLE BY GAME:")
print("-" * 100)

if 'game_id_unique' in df.columns:
    sample_game = df['game_id_unique'].unique()[0]
    game_data = df.filter(pl.col('game_id_unique') == sample_game)
    
    print(f"\nSample game_id_unique: {sample_game}")
    print(f"Shots in game: {len(game_data)}")
    
    cols_to_show = ['shot_id', 'cumulative_home_xg', 'cumulative_away_xg', 'x_goal']
    available = [c for c in cols_to_show if c in game_data.columns]
    
    if available:
        print(f"\n{available}:")
        for row in game_data.select(available).head(15).to_dicts():
            print(f"  {row}")

print("\n" + "=" * 100)
print("DIAGNOSIS:")
print("=" * 100)

# Check if cumulative values are actually being computed
if 'cumulative_home_xg' in df.columns:
    max_val = df['cumulative_home_xg'].max()
    if max_val == 0:
        print("\n❌ CRITICAL: cumulative_home_xg is ALL ZEROS in game_states.parquet")
        print("   This means game_state_builders.py fix did NOT work")
        print("   The cumulative computation is still broken in Phase 1")
    else:
        print(f"\n✓ cumulative_home_xg has values (max={max_val})")
        print("   The Phase 1 fix appears to have worked")
        print("   The problem must be in Phase 2 (feature engineering)")
else:
    print("\n✗ cumulative_home_xg column not found at all")

print("\n" + "=" * 100)