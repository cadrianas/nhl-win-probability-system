"""
Validate temporal consistency in game states.

Single responsibility: Check temporal logic only.
"""

import pandas as pd


def check_monotonic_elapsed_time(df: pd.DataFrame) -> list:
    """
    Verify game_seconds_elapsed increases within each game.

    Returns:
        List of game_ids with issues
    """
    issues = []

    for game_id, game_group in df.groupby("game_id"):
        if not game_group["game_seconds_elapsed"].is_monotonic_increasing:
            issues.append(
                {
                    "type": "non_monotonic_time",
                    "game_id": game_id,
                    "message": "game_seconds_elapsed not monotonically increasing",
                }
            )

    return issues


def check_period_consistency(df: pd.DataFrame) -> list:
    """
    Verify period transitions are valid.

    Returns:
        List of issues
    """
    issues = []

    for game_id, game_group in df.groupby("game_id"):
        periods = game_group["period"].unique()

        # Periods should be 1, 2, 3, optionally 4, 5, etc.
        expected = set(range(1, periods.max() + 1))
        actual = set(periods)

        if actual != expected:
            issues.append(
                {
                    "type": "missing_periods",
                    "game_id": game_id,
                    "missing": list(expected - actual),
                }
            )

    return issues


def check_time_remaining_valid(df: pd.DataFrame) -> list:
    """
    Verify game_seconds_remaining decreases as time progresses.

    Returns:
        List of issues
    """
    issues = []

    for game_id, game_group in df.groupby("game_id"):
        if not game_group["game_seconds_remaining"].is_monotonic_decreasing:
            issues.append(
                {
                    "type": "invalid_time_remaining",
                    "game_id": game_id,
                }
            )

    return issues
