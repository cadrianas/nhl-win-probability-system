"""
Validate goalie-related features and empty net logic.

Single responsibility: Check goalie state logic.
"""

import pandas as pd


def check_empty_net_consistency(df: pd.DataFrame) -> list:
    """
    Verify empty net flags are used only in high-leverage situations.

    Returns:
        List of issues
    """
    issues = []

    # Empty net should only occur in final minutes of close games
    empty_net_rows = df[df["is_empty_net"]]

    if len(empty_net_rows) > 0:
        # Check: are most empty net situations in last 5 minutes?
        in_final_5 = (empty_net_rows["game_seconds_remaining"] < 300).sum()
        total_empty = len(empty_net_rows)

        if in_final_5 / total_empty < 0.7:
            issues.append(
                {
                    "type": "unexpected_empty_net_timing",
                    "pct_in_final_5min": f"{in_final_5 / total_empty * 100:.1f}%",
                    "message": "Most empty net situations should be in final 5 minutes",
                }
            )

    return issues


def check_goalie_pulled_vs_outcome(df: pd.DataFrame) -> dict:
    """
    Analyze relationship between goalie pull and eventual outcome.

    Returns:
        dict with statistics
    """
    stats = {}

    empty_net_rows = df[df["is_empty_net"]]

    if len(empty_net_rows) > 0:
        # For home empty net, check home team outcome
        home_empty = empty_net_rows[empty_net_rows["home_empty_net"] == 1]
        if len(home_empty) > 0:
            home_win_when_pulled = home_empty["target_home_win"].mean()
            stats["home_win_rate_when_goalie_pulled"] = f"{home_win_rate_when_pulled:.1%}"

        # For away empty net
        away_empty = empty_net_rows[empty_net_rows["away_empty_net"] == 1]
        if len(away_empty) > 0:
            away_win_when_pulled = (1 - away_empty["target_home_win"]).mean()
            stats["away_win_rate_when_goalie_pulled"] = f"{away_win_when_pulled:.1%}"

    return stats


def check_empty_net_vs_score(df: pd.DataFrame) -> list:
    """
    Verify empty net situations make sense given game situation.

    Returns:
        List of issues
    """
    issues = []

    empty_net_rows = df[df["is_empty_net"]]

    if len(empty_net_rows) > 0:
        # Home empty net: should usually be when home is trailing
        home_empty = empty_net_rows[empty_net_rows["home_empty_net"] == 1]
        if len(home_empty) > 0:
            trailing = (home_empty["score_differential"] < 0).sum()
            if trailing / len(home_empty) < 0.5:
                issues.append(
                    {
                        "type": "home_empty_net_when_ahead",
                        "message": "Home pulled goalie while ahead or tied",
                        "pct": f"{(1 - trailing / len(home_empty)) * 100:.1f}%",
                    }
                )

        # Away empty net: should usually be when away is trailing
        away_empty = empty_net_rows[empty_net_rows["away_empty_net"] == 1]
        if len(away_empty) > 0:
            trailing = (away_empty["score_differential"] > 0).sum()
            if trailing / len(away_empty) < 0.5:
                issues.append(
                    {
                        "type": "away_empty_net_when_ahead",
                        "message": "Away pulled goalie while ahead or tied",
                        "pct": f"{(1 - trailing / len(away_empty)) * 100:.1f}%",
                    }
                )

    return issues
