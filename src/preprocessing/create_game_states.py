"""
Main entry point for creating game states.

Orchestrates:
1. Load cleaned shots
2. Create game state snapshots
3. Run quality checks
4. Save game states
"""

import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on the path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import DATA_PROCESSED
from src.preprocessing.game_state_builders import create_game_states
from src.preprocessing.validators.temporal_validator import (
    check_monotonic_elapsed_time,
    check_period_consistency,
    check_time_remaining_valid,
)


def main():
    """
    Transform shots into game states.
    """
    print("=" * 60)
    print("PHASE 1: CREATE GAME STATES")
    print("=" * 60)

    # Load
    print("\n1. Loading cleaned shots...")
    shots = pd.read_parquet(DATA_PROCESSED / "shots_cleaned.parquet")
    print(f"   Rows: {len(shots):,}")
    print(f"   Unique games: {shots['game_id'].nunique():,}")

    # Create states
    print("\n2. Creating game state snapshots...")
    states = create_game_states(shots)
    print(f"   States created: {len(states):,}")

    # Validate
    print("\n3. Running quality checks...")
    temporal_issues = check_monotonic_elapsed_time(states)
    period_issues = check_period_consistency(states)
    time_rem_issues = check_time_remaining_valid(states)

    print(f"   Temporal issues: {len(temporal_issues)}")
    print(f"   Period issues: {len(period_issues)}")
    print(f"   Time remaining issues: {len(time_rem_issues)}")

    # Save
    print("\n4. Saving game states...")
    states.to_parquet(DATA_PROCESSED / "game_states.parquet", compression="snappy")
    print(f"   Saved to: {DATA_PROCESSED / 'game_states.parquet'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total states: {len(states):,}")
    print(f"Total games: {states['game_id'].nunique():,}")
    print(f"Avg states per game: {len(states) / states['game_id'].nunique():.1f}")
    print(f"Home win rate: {states['target_home_win'].mean():.1%}")
    print(
        f"Regular season: {(~states['is_playoff_game']).sum() / states['game_id'].nunique() * 100:.1f}%"
    )

    print("\n✅ CREATE GAME STATES COMPLETE\n")

    return states


if __name__ == "__main__":
    main()
