"""
Check what columns exist in shots_cleaned.parquet
"""

import pandas as pd
from pathlib import Path
import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import DATA_PROCESSED

shots_path = DATA_PROCESSED / "shots_cleaned.parquet"

if not shots_path.exists():
    print(f"❌ {shots_path} not found")
    exit(1)

print("\n" + "=" * 100)
print("SHOTS_CLEANED.PARQUET COLUMN INSPECTION")
print("=" * 100)

shots = pd.read_parquet(shots_path)

print(f"\nTotal rows: {len(shots):,}")
print(f"Total columns: {len(shots.columns)}")

print("\n" + "-" * 100)
print("ALL COLUMNS:")
print("-" * 100)

for i, col in enumerate(shots.columns, 1):
    dtype = str(shots[col].dtype)
    non_null = shots[col].notna().sum()
    print(f"{i:2d}. {col:30s} | dtype: {dtype:10s} | non-null: {non_null:,}")

print("\n" + "-" * 100)
print("LOOKING FOR xG-RELATED COLUMNS:")
print("-" * 100)

xg_cols = [col for col in shots.columns if 'xg' in col.lower() or 'x_g' in col.lower()]

if xg_cols:
    print(f"\n✓ Found {len(xg_cols)} xG-related columns:")
    for col in xg_cols:
        print(f"  - {col}")
        print(f"    Min: {shots[col].min()}")
        print(f"    Max: {shots[col].max()}")
        print(f"    Mean: {shots[col].mean():.4f}")
        print(f"    Non-zero: {(shots[col] > 0).sum():,}")
else:
    print(f"\n✗ No xG-related columns found!")

print("\n" + "-" * 100)
print("LOOKING FOR 'x_goal' (what the code references):")
print("-" * 100)

if 'x_goal' in shots.columns:
    print(f"✓ 'x_goal' column EXISTS")
    xg = shots['x_goal']
    print(f"  Min: {xg.min()}")
    print(f"  Max: {xg.max()}")
    print(f"  Mean: {xg.mean():.4f}")
    print(f"  Non-zero: {(xg > 0).sum():,}")
    print(f"  Zeros: {(xg == 0).sum():,}")
else:
    print(f"✗ 'x_goal' column NOT FOUND")
    
print("\n" + "-" * 100)
print("SAMPLE DATA (first 10 rows, xG columns only):")
print("-" * 100)

if xg_cols:
    print(shots[xg_cols].head(10))
elif 'x_goal' in shots.columns:
    print(shots[['game_id', 'shot_id', 'team', 'x_goal']].head(10))
else:
    print("No xG columns found to display")

print("\n" + "=" * 100)