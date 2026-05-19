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


def create_game_states(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw shots into game state snapshots.
    Creates missing derived columns in sequence.
    """
    import numpy as np
    
    df = shots.copy()
    df = df.sort_values(['game_id', 'time_elapsed'])
    
    # 1. CREATE SCORE DIFFERENTIAL (if missing)
    if 'score_differential' not in df.columns:
        df['score_differential'] = df['home_team_goals'] - df['away_team_goals']
    
    # 2. CREATE CUMULATIVE STATS
    df['cumulative_home_shots'] = 0
    df['cumulative_away_shots'] = 0
    df['cumulative_home_xg'] = 0.0
    df['cumulative_away_xg'] = 0.0
    
    for game_id in df['game_id'].unique():
        game_mask = df['game_id'] == game_id
        game_df = df.loc[game_mask].copy()
        
        home_team = game_df.iloc[0]['home_team_code']
        away_team = game_df.iloc[0]['away_team_code']
        
        is_home = game_df['team'] == home_team
        is_away = game_df['team'] == away_team
        
        # Cumulative shots
        df.loc[game_mask, 'cumulative_home_shots'] = is_home.cumsum().values
        df.loc[game_mask, 'cumulative_away_shots'] = is_away.cumsum().values
        
        # Cumulative xG
        df.loc[game_mask, 'cumulative_home_xg'] = (is_home * game_df['x_goal']).cumsum().values
        df.loc[game_mask, 'cumulative_away_xg'] = (is_away * game_df['x_goal']).cumsum().values
    
    # Differentials
    df['xg_differential'] = df['cumulative_home_xg'] - df['cumulative_away_xg']
    df['shot_differential'] = df['cumulative_home_shots'] - df['cumulative_away_shots']
    
    # 3. CREATE ROLLING FEATURES (last 2/5 minutes)
    df['shots_last_2min_home'] = 0
    df['shots_last_2min_away'] = 0
    df['shots_last_5min_home'] = 0
    df['shots_last_5min_away'] = 0
    
    for game_id in df['game_id'].unique():
        game_mask = df['game_id'] == game_id
        game_df = df.loc[game_mask].copy()
        
        home_team = game_df.iloc[0]['home_team_code']
        away_team = game_df.iloc[0]['away_team_code']
        
        times = game_df['time_elapsed'].values
        teams = game_df['team'].values
        
        for idx, (time, team) in enumerate(zip(times, teams)):
            # Last 2 minutes (120 seconds)
            mask_2min = (times >= time - 120) & (times <= time)
            df.loc[game_mask, 'shots_last_2min_home'].iloc[idx] = np.sum(teams[mask_2min] == home_team)
            df.loc[game_mask, 'shots_last_2min_away'].iloc[idx] = np.sum(teams[mask_2min] == away_team)
            
            # Last 5 minutes (300 seconds)
            mask_5min = (times >= time - 300) & (times <= time)
            df.loc[game_mask, 'shots_last_5min_home'].iloc[idx] = np.sum(teams[mask_5min] == home_team)
            df.loc[game_mask, 'shots_last_5min_away'].iloc[idx] = np.sum(teams[mask_5min] == away_team)
    
    # 4. SELECT FINAL COLUMNS
    final_columns = [
        'game_id', 'shot_id', 'season', 'is_playoff_game', 'period',
        'time_elapsed', 'game_seconds_elapsed', 'time_in_period_seconds',
        'time_in_period_minutes', 'game_seconds_remaining',
        'home_team_code', 'away_team_code', 'home_team_goals', 'away_team_goals',
        'score_differential', 'strength_state', 'game_state',
        'is_even_strength', 'is_power_play', 'is_empty_net', 'is_3v3',
        'x_coord', 'y_coord', 'shot_distance', 'shot_angle',
        'cumulative_home_xg', 'cumulative_away_xg', 'xg_differential',
        'cumulative_home_shots', 'cumulative_away_shots', 'shot_differential',
        'shots_last_2min_home', 'shots_last_2min_away',
        'shots_last_5min_home', 'shots_last_5min_away',
        'target_home_win'
    ]
    
    # Only select columns that exist
    cols_to_select = [col for col in final_columns if col in df.columns]
    
    return df[cols_to_select]
