"""
Debug: Check if x_goal is in shots_cleaned.parquet when create_game_states is called
"""

import pandas as pd
from pathlib import Path

shots_path = Path("data/processed/shots_cleaned.parquet")

print("\n" + "=" * 100)
print("CHECKING SHOTS_CLEANED.PARQUET FOR x_goal")
print("=" * 100)

shots = pd.read_parquet(shots_path)

print(f"\nShape: {shots.shape}")
print(f"Columns: {len(shots.columns)}")

# Check for xG-related columns
xg_cols = [col for col in shots.columns if 'xg' in col.lower() or 'x_g' in col.lower()]
print(f"\nxG-related columns found:")
for col in xg_cols:
    print(f"  - {col}")
    print(f"    Sample values: {shots[col].head(5).tolist()}")
    print(f"    Min: {shots[col].min()}, Max: {shots[col].max()}")

# Check if x_goal specifically exists
if 'x_goal' in shots.columns:
    print(f"\n✓ x_goal EXISTS")
    print(f"  Type: {shots['x_goal'].dtype}")
    print(f"  Non-null: {shots['x_goal'].notna().sum():,}")
    print(f"  Min: {shots['x_goal'].min()}")
    print(f"  Max: {shots['x_goal'].max()}")
    print(f"  Mean: {shots['x_goal'].mean():.4f}")
else:
    print(f"\n✗ x_goal NOT FOUND!")

# Check for team columns
team_cols = [col for col in shots.columns if 'team' in col.lower()]
print(f"\nTeam-related columns:")
for col in team_cols[:5]:
    print(f"  - {col}")
    if shots[col].dtype == 'object':
        print(f"    Sample values: {shots[col].unique()[:3]}")
    else:
        print(f"    Type: {shots[col].dtype}")

print("\n" + "=" * 100)