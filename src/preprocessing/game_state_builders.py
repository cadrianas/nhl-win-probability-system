"""
Assemble game states from cleaned shots and utility functions.

This orchestrates the transformation from shot-level to game-state-level data.
"""

import pandas as pd
from src.preprocessing.continuous_time import create_time_features
from src.preprocessing.strength_state import (
    add_strength_state_column,
    create_strength_features,
)
from src.preprocessing.rolling_windows import add_rolling_windows
from src.preprocessing.schema import GAME_STATE_COLUMNS


def build_single_game_state(row: pd.Series, cumulative_stats: dict) -> dict:
    """
    Create one game state snapshot from a shot row.

    Args:
        row: One shot from a game
        cumulative_stats: dict tracking cumulative xG, shots, etc.

    Returns:
        dict ready to become a DataFrame row
    """
    state = {
        "game_id": row["game_id"],
        "shot_id": row["shot_id"],
        "period": row["period"],
        "game_seconds_elapsed": row["game_seconds_elapsed"],
        "game_seconds_remaining": row["game_seconds_remaining"],
        "time_decay_factor": row["time_decay_factor"],
        "season": row["season"],
        "is_playoff_game": row["is_playoff_game"],
        "game_state": row["game_state"],
        "home_team_goals": row["home_team_goals"],
        "away_team_goals": row["away_team_goals"],
        "score_differential": row["home_team_goals"] - row["away_team_goals"],
        "cumulative_home_xg": cumulative_stats["home_xg"],
        "cumulative_away_xg": cumulative_stats["away_xg"],
        "xg_differential": cumulative_stats["home_xg"] - cumulative_stats["away_xg"],
        "cumulative_home_shots": cumulative_stats["home_shots"],
        "cumulative_away_shots": cumulative_stats["away_shots"],
        "shot_differential": cumulative_stats["home_shots"]
        - cumulative_stats["away_shots"],
        "shots_last_2min_home": row.get("shots_last_2min_home", 0),
        "shots_last_2min_away": row.get("shots_last_2min_away", 0),
        "shots_last_5min_home": row.get("shots_last_5min_home", 0),
        "shots_last_5min_away": row.get("shots_last_5min_away", 0),
        "strength_state": row["strength_state"],
        "is_even_strength": row["is_even_strength"],
        "is_power_play": row["is_power_play"],
        "is_empty_net": row["is_empty_net"],
        "is_3v3": row["is_3v3"],
        "target_home_win": row["target_home_win"],
    }

    return state


def create_game_states(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform shot-level data into game state snapshots.

    Args:
        shots_df: Cleaned shots dataframe

    Returns:
        DataFrame with one row per game state snapshot
    """
    shots_df = shots_df.sort_values(["game_id", "game_seconds_elapsed"])

    states = []

    for game_id, game_group in shots_df.groupby("game_id"):
        game_group = game_group.reset_index(drop=True)

        # Add rolling windows for this game
        game_group = add_rolling_windows(game_group)

        # Build states
        cumulative_stats = {
            "home_xg": 0.0,
            "away_xg": 0.0,
            "home_shots": 0,
            "away_shots": 0,
        }
        home_team = game_group.iloc[0]["home_team_code"]

        for idx, row in game_group.iterrows():
            # Update cumulative stats
            if row["team"] == home_team:
                cumulative_stats["home_xg"] += row["x_goal"]
                cumulative_stats["home_shots"] += 1
            else:
                cumulative_stats["away_xg"] += row["x_goal"]
                cumulative_stats["away_shots"] += 1

            # Build state
            state = build_single_game_state(row, cumulative_stats)
            states.append(state)

    result = pd.DataFrame(states)

    # Ensure column order
    result = result[GAME_STATE_COLUMNS]

    return result
