"""
Run all validation checks on game states and generate report.

This is the "Quality Gate" for Phase 1.
"""

import pandas as pd

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
from src.preprocessing.validators.goalie_validator import (
    check_empty_net_consistency,
    check_goalie_pulled_vs_outcome,
    check_empty_net_vs_score,
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

    print("   - Empty net consistency...")
    checks["empty_net"] = check_empty_net_consistency(states)

    print("   - Empty net vs score logic...")
    checks["empty_net_vs_score"] = check_empty_net_vs_score(states)

    # Summary stats
    stats = {
        "total_games": states["game_id"].nunique(),
        "total_states": len(states),
        "home_win_rate": f"{states['target_home_win'].mean():.1%}",
        "away_win_rate": f"{(1 - states['target_home_win']).mean():.1%}",
        "regular_season_pct": f"{(~states['is_playoff_game']).sum() / states['game_id'].nunique() * 100:.1f}%",
        "playoff_ot_pct": f"{(states['is_playoff_game']).sum() / states['game_id'].nunique() * 100:.1f}%",
    }

    # Extra analysis
    print("\n3. Running analysis...")
    lead_changes = check_lead_changes(states)
    goalie_analysis = check_goalie_pulled_vs_outcome(states)

    extra_analysis = {
        "lead_changes": lead_changes,
        "goalie_pulled_analysis": goalie_analysis,
    }

    # Generate report
    print("\n4. Generating report...")
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_LOGS / "validation_report.json"
    report = generate_validation_report(checks, stats, extra_analysis, report_path)

    # Print summary
    print_validation_summary(report)

    print("\n✅ VALIDATE DATA COMPLETE\n")

    return report


if __name__ == "__main__":
    main()
