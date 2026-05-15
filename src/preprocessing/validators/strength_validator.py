"""
Validate strength state consistency.

Single responsibility: Check strength state validity.
"""

import pandas as pd


def check_strength_state_validity(df: pd.DataFrame) -> list:
    """
    Verify strength states are not 'unknown' except in specific cases.

    Returns:
        List of issues (should be minimal if preprocessing worked)
    """
    issues = []

    unknown_count = (df["strength_state"] == "UNKNOWN").sum()

    if unknown_count > 0:
        pct = unknown_count / len(df) * 100
        if pct > 1.0:  # Threshold: more than 1% unknown is concerning
            issues.append(
                {
                    "type": "high_unknown_strength_state",
                    "count": unknown_count,
                    "pct": f"{pct:.2f}%",
                }
            )

    return issues


def check_skater_count_range(df: pd.DataFrame) -> list:
    """
    Verify skater counts are in valid range (3-6).

    Returns:
        List of issues
    """
    issues = []

    # Check raw skater counts if available
    if "home_skaters_on_ice" in df.columns:
        invalid_home = (
            (df["home_skaters_on_ice"] < 3) | (df["home_skaters_on_ice"] > 6)
        ).sum()
        if invalid_home > 0:
            issues.append(
                {
                    "type": "invalid_home_skaters",
                    "count": invalid_home,
                }
            )

    if "away_skaters_on_ice" in df.columns:
        invalid_away = (
            (df["away_skaters_on_ice"] < 3) | (df["away_skaters_on_ice"] > 6)
        ).sum()
        if invalid_away > 0:
            issues.append(
                {
                    "type": "invalid_away_skaters",
                    "count": invalid_away,
                }
            )

    return issues


def check_power_play_consistency(df: pd.DataFrame) -> list:
    """
    Verify power play feature matches strength state.

    Returns:
        List of issues
    """
    issues = []

    # If is_power_play=True, strength_state should indicate PP or SH
    pp_rows = df[df["is_power_play"]]

    for idx, row in pp_rows.iterrows():
        if row["strength_state"] not in [
            "5v4_HOME",
            "5v4_AWAY",
            "4v5_HOME",
            "4v5_AWAY",
            "6v5_HOME",
            "5v6_AWAY",
        ]:
            issues.append(
                {
                    "type": "power_play_mismatch",
                    "row": idx,
                    "strength_state": row["strength_state"],
                }
            )

    return issues[:10]  # Return first 10 issues
