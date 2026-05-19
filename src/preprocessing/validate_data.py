"""
Run all validation checks on game states and generate report.
This is the "Quality Gate" for Phase 1.

FIXED VERSION:
- Uses is_empty_net instead of home_empty_net/away_empty_net (which aren't in game_states)
- Fixes the "Regular season: 87906.1%" bug in summary stats calculation
"""
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on the path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import DATA_PROCESSED, RESULTS_LOGS
from src.preprocessing.validators.temporal_validator import (
    check_monotonic_elapsed_time,
    check_period_consistency,
    check_time_remaining_valid,
)
from src.preprocessing.validators.strength_validator import (
    check_strength_state_validity,
    check_skater_count_range,
)
from src.preprocessing.validators.score_validator import (
    check_score_consistency,
    check_score_monotonicity,
    check_lead_changes,
)
from src.preprocessing.validators.report_generator import (
    generate_validation_report,
    print_validation_summary,
)


def main():
    """
    Load game states, run all validators, produce report.
    """
    print("=" * 60)
    print("PHASE 1: VALIDATE DATA")
    print("=" * 60)
    
    # Load
    print("\n1. Loading game states...")
    states = pd.read_parquet(DATA_PROCESSED / "game_states.parquet")
    print(f"   Rows: {len(states):,}")
    print(f"   Games: {states['game_id'].nunique():,}")
    print(f"   Columns: {len(states.columns)}")
    
    # Run checks
    print("\n2. Running validation checks...")
    checks = {}
    
    print("   - Temporal monotonicity...")
    checks["temporal_monotonicity"] = check_monotonic_elapsed_time(states)
    
    print("   - Period consistency...")
    checks["period_consistency"] = check_period_consistency(states)
    
    print("   - Time remaining validity...")
    checks["time_remaining"] = check_time_remaining_valid(states)
    
    print("   - Strength state validity...")
    checks["strength_state"] = check_strength_state_validity(states)
    
    print("   - Skater count range...")
    checks["skater_count"] = check_skater_count_range(states)
    
    print("   - Score consistency...")
    checks["score_consistency"] = check_score_consistency(states)
    
    print("   - Score monotonicity...")
    checks["score_monotonicity"] = check_score_monotonicity(states)
    
    # Summary stats (FIXED: correct calculation)
    total_games = states["game_id"].nunique()
    total_states = len(states)
    home_wins = states['target_home_win'].sum()
    away_wins = (~states['target_home_win']).sum()
    
    # Game-level stats (not state-level)
    games_df = states.groupby('game_id').agg({
        'is_playoff_game': 'first',
        'target_home_win': 'first'
    }).reset_index()
    
    playoff_games = games_df['is_playoff_game'].sum()
    regular_season_games = (~games_df['is_playoff_game']).sum()
    
    stats = {
        "total_games": total_games,
        "total_states": total_states,
        "avg_states_per_game": f"{total_states / total_games:.1f}",
        "home_win_rate": f"{home_wins / total_states:.1%}",
        "away_win_rate": f"{away_wins / total_states:.1%}",
        "regular_season_games": regular_season_games,
        "playoff_games": playoff_games,
        "regular_season_pct": f"{regular_season_games / total_games * 100:.1f}%",
        "playoff_pct": f"{playoff_games / total_games * 100:.1f}%",
    }
    
    # Extra analysis
    print("\n3. Running analysis...")
    lead_changes = check_lead_changes(states)
    
    extra_analysis = {
        "lead_changes": lead_changes,
    }
    
    # Generate report
    print("\n4. Generating report...")
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_LOGS / "validation_report.json"
    report = generate_validation_report(checks, stats, extra_analysis, report_path)
    
    # Print summary
    print_validation_summary(report)
    
    # Extra validation info
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total games: {total_games:,}")
    print(f"  - Regular season: {regular_season_games:,} ({regular_season_games/total_games*100:.1f}%)")
    print(f"  - Playoff: {playoff_games:,} ({playoff_games/total_games*100:.1f}%)")
    print(f"\nTotal game states: {total_states:,}")
    print(f"  Avg per game: {total_states/total_games:.1f}")
    print(f"\nHome team outcomes:")
    print(f"  - Wins: {home_wins:,} ({home_wins/total_states:.1%})")
    print(f"  - Losses: {away_wins:,} ({away_wins/total_states:.1%})")
    
    # Check for missing columns
    print(f"\nColumns in game_states.parquet:")
    expected_cols = [
        'game_id', 'shot_id', 'season', 'period',
        'time_elapsed', 'score_differential',
        'cumulative_home_xg', 'cumulative_home_shots',
        'shots_last_5min_home', 'is_playoff_game',
        'strength_state', 'target_home_win'
    ]
    
    missing = [col for col in expected_cols if col not in states.columns]
    if missing:
        print(f"  ⚠️  Missing: {missing}")
    else:
        print(f"  ✅ All expected columns present")
    
    print("\n" + "=" * 60)
    print("✅ VALIDATE DATA COMPLETE")
    print("=" * 60 + "\n")
    
    return report


if __name__ == "__main__":
    main()