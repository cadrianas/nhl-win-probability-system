"""
Compute rolling-window statistics within a game.

Examples:
- Shots last 2 minutes
- xG last 5 minutes
- Momentum indicators

These features capture recent team performance and pressure.
"""

import pandas as pd


def calculate_rolling_shots(
    game_group: pd.DataFrame,
    window_seconds: int,
) -> pd.DataFrame:
    """
    For each row, count shots in the preceding window_seconds.

    Args:
        game_group: DataFrame of all shots in one game, sorted by game_seconds_elapsed
        window_seconds: e.g., 120 (2 min) or 300 (5 min)

    Returns:
        Same group with new columns 'shots_home' and 'shots_away' for the window
    """
    results = []

    for idx, row in game_group.iterrows():
        current_time = row["game_seconds_elapsed"]
        window_start = current_time - window_seconds

        # Shots in this window
        in_window = game_group[
            (game_group["game_seconds_elapsed"] >= window_start)
            & (game_group["game_seconds_elapsed"] <= current_time)
            & (game_group.index <= idx)  # Only past events
        ]

        home_team = row["home_team_code"]
        shots_home = len(in_window[in_window["team"] == home_team])
        shots_away = len(in_window[in_window["team"] != home_team])

        results.append(
            {
                "idx": idx,
                "shots_home": shots_home,
                "shots_away": shots_away,
            }
        )

    return pd.DataFrame(results).set_index("idx")


def add_rolling_windows(game_group: pd.DataFrame) -> pd.DataFrame:
    """
    Add 2-minute and 5-minute rolling shot counts.

    Args:
        game_group: All shots in one game, sorted by game_seconds_elapsed

    Returns:
        game_group with new columns:
        - shots_last_2min_home, shots_last_2min_away
        - shots_last_5min_home, shots_last_5min_away
    """
    rolling_2min = calculate_rolling_shots(game_group, window_seconds=120)
    rolling_5min = calculate_rolling_shots(game_group, window_seconds=300)

    game_group["shots_last_2min_home"] = rolling_2min["shots_home"]
    game_group["shots_last_2min_away"] = rolling_2min["shots_away"]
    game_group["shots_last_5min_home"] = rolling_5min["shots_home"]
    game_group["shots_last_5min_away"] = rolling_5min["shots_away"]

    return game_group


def calculate_rolling_xg(
    game_group: pd.DataFrame,
    window_seconds: int,
) -> pd.DataFrame:
    """
    For each row, sum xG in the preceding window_seconds.

    Args:
        game_group: DataFrame of all shots in one game
        window_seconds: e.g., 120 (2 min) or 300 (5 min)

    Returns:
        DataFrame with columns 'xg_home' and 'xg_away'
    """
    results = []

    for idx, row in game_group.iterrows():
        current_time = row["game_seconds_elapsed"]
        window_start = current_time - window_seconds

        # Shots in this window
        in_window = game_group[
            (game_group["game_seconds_elapsed"] >= window_start)
            & (game_group["game_seconds_elapsed"] <= current_time)
            & (game_group.index <= idx)
        ]

        home_team = row["home_team_code"]
        xg_home = in_window[in_window["team"] == home_team]["x_goal"].sum()
        xg_away = in_window[in_window["team"] != home_team]["x_goal"].sum()

        results.append(
            {
                "idx": idx,
                "xg_home": xg_home,
                "xg_away": xg_away,
            }
        )

    return pd.DataFrame(results).set_index("idx")
