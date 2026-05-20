"""
FIX: Create unique game_id that includes season

The original game_id is NOT unique across seasons.
game_id 20001 appears in every season (2014, 2015, ..., 2025).

This breaks the temporal train/test split.

Solution: Create a composite unique_game_id
"""

import pandas as pd
from pathlib import Path

shots_path = Path("data/processed/shots_cleaned.parquet")

print("=" * 100)
print("FIXING GAME_ID UNIQUENESS")
print("=" * 100)

shots = pd.read_parquet(shots_path)

print(f"\nLoaded {len(shots):,} rows")
print(f"Seasons: {sorted(shots['season'].unique())}")

# Current issue: game_id not unique across seasons
print(f"\nBEFORE FIX:")
print(f"  Unique game_ids: {shots['game_id'].nunique()}")
print(f"  game_ids in multiple seasons: {sum(shots.groupby('game_id')['season'].nunique() > 1)}")

# CREATE TRULY UNIQUE GAME_ID
# Format: SSGGGG where SS = last 2 digits of season, GGGG = game number
# Cast to int64 to avoid overflow (season is int16)
shots['game_id_unique'] = (shots['season'].astype('int64') * 100000) + shots['game_id'].astype('int64')

print(f"\nAFTER FIX:")
print(f"  Unique game_ids_unique: {shots['game_id_unique'].nunique()}")
print(f"  game_ids_unique in multiple seasons: {sum(shots.groupby('game_id_unique')['season'].nunique() > 1)}")

# Verify
print(f"\nVERIFICATION:")
print(f"  Expected unique games: ~15,204")
print(f"  Actual unique games: {shots['game_id_unique'].nunique()}")

if shots['game_id_unique'].nunique() == shots.groupby(['season', 'game_id']).ngroups:
    print(f"  ✓ PASS: Each (season, game_id) pair is unique")
else:
    print(f"  ✗ FAIL: Still have duplicates")

# Save back
print(f"\nSaving...")
shots.to_parquet(shots_path, compression='snappy')

print(f"✓ Fixed shots_cleaned.parquet")
print(f"\nNOTE: You now have two game ID columns:")
print(f"  - game_id: Original (NOT unique across seasons)")
print(f"  - game_id_unique: NEW (truly unique across all seasons)")

print("\n" + "=" * 100)
print("NEXT STEPS:")
print("=" * 100)
print("\n1. Update Phase 1 to use 'game_id_unique' instead of 'game_id'")
print("   (Or keep both columns and use game_id_unique for grouping)")
print("\n2. Re-run Phase 1:")
print("   python src/preprocessing/phase1_main.py")
print("\n3. Re-run diagnostics:")
print("   python diagnose_game_coverage.py")
print("\n4. Verify temporal split works correctly")