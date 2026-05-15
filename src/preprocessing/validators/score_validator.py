"""
Validate score consistency in game states.

Single responsibility: Check score logic only.
"""

import pandas as pd


def check_score_consistency(df: pd.DataFrame) -> list:
    """
    Verify score_differential matches home_score - away_score.

    Returns:
        List of issues
    """
    issues = []

    score_mismatch = df[
        df["score_differential"] != (df["home_team_goals"] - df["away_team_goals"])
    ]

    if len(score_mismatch) > 0:
        issues.append(
            {
                "type": "score_differential_mismatch",
                "count": len(score_mismatch),
            }
        )

    return issues


def check_score_monotonicity(df: pd.DataFrame) -> list:
    """
    Verify scores never decrease within a game.

    Returns:
        List of issues
    """
    issues = []

    for game_id, game_group in df.groupby("game_id"):
        if not game_group["home_team_goals"].is_monotonic_increasing:
            issues.append(
                {
                    "type": "home_score_decrease",
                    "game_id": game_id,
                }
            )

        if not game_group["away_team_goals"].is_monotonic_increasing:
            issues.append(
                {
                    "type": "away_score_decrease",
                    "game_id": game_id,
                }
            )

    return issues


def check_score_reasonableness(df: pd.DataFrame) -> list:
    """
    Verify final scores are reasonable (not 25-0, etc.).

    Returns:
        List of issues
    """
    issues = []

    max_scores = df.groupby("game_id")[["home_team_goals", "away_team_goals"]].max()

    unreasonable = (max_scores > 15).any(axis=1).sum()

    if unreasonable > 0:
        issues.append(
            {
                "type": "unreasonable_scores",
                "count": unreasonable,
                "message": "Some games have final scores > 15 goals",
            }
        )

    return issues


def check_lead_changes(df: pd.DataFrame) -> dict:
    """
    Analyze lead changes during games (not a failure, just analysis).

    Returns:
        dict with statistics
    """
    stats = {}

    for game_id, game_group in df.groupby("game_id"):
        game_group = game_group.sort_values("game_seconds_elapsed")
        leads = game_group["score_differential"].apply(
            lambda x: "home" if x > 0 else ("away" if x < 0 else "tied")
        )

        # Count lead changes
        lead_changes = (leads != leads.shift()).sum() - 1  # Exclude initial state
        stats[game_id] = lead_changes

    avg_lead_changes = sum(stats.values()) / len(stats) if stats else 0

    return {
        "total_games": len(stats),
        "avg_lead_changes_per_game": f"{avg_lead_changes:.1f}",
        "max_lead_changes": max(stats.values()) if stats else 0,
    }
