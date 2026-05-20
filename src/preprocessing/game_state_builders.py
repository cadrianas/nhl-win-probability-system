"""
Assemble game states from cleaned shots and utility functions.

FIXED VERSION: Proper indexing for cumulative xG calculation
"""

import pandas as pd
import numpy as np
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
    
    FIXED: 
    1. Use game_id_unique (truly unique across seasons) for grouping
    2. Proper indexing for cumulative xG/shots calculation
    """
    
    df = shots.copy()
    
    # Use game_id_unique if available (newly fixed in Phase 1)
    # Otherwise fall back to game_id (old broken version)
    groupby_col = 'game_id_unique' if 'game_id_unique' in df.columns else 'game_id'
    print(f"   Using '{groupby_col}' for grouping games")
    
    df = df.sort_values([groupby_col, 'time_elapsed']).reset_index(drop=True)
    
    # 1. CREATE SCORE DIFFERENTIAL (if missing)
    if 'score_differential' not in df.columns:
        df['score_differential'] = df['home_team_goals'] - df['away_team_goals']
    
    # 2. CREATE CUMULATIVE STATS - FIXED VERSION
    # Initialize columns
    df['cumulative_home_shots'] = 0
    df['cumulative_away_shots'] = 0
    df['cumulative_home_xg'] = 0.0
    df['cumulative_away_xg'] = 0.0
    
    print("   Computing cumulative xG and shots...")
    
    for i, game_id in enumerate(df[groupby_col].unique()):
        if (i + 1) % 200 == 0:
            print(f"   Processing game {i + 1}...")
        
        game_mask = df[groupby_col] == game_id
        game_indices = df[game_mask].index  # Get actual indices
        game_df = df.loc[game_indices].copy()  # Use indices, not mask
        
        # FIXED: 'team' column contains 'HOME' or 'AWAY', not team codes!
        # Don't compare to home_team_code - that's 'TOR', 'BOS', etc.
        # Instead, check if team is 'HOME' or 'AWAY'
        
        # Create boolean masks (these will have the correct indices)
        is_home = game_df['team'] == 'HOME'
        is_away = game_df['team'] == 'AWAY'
        
        # Compute cumulative values with proper indexing
        home_xg_values = (is_home * game_df['x_goal']).cumsum()
        away_xg_values = (is_away * game_df['x_goal']).cumsum()
        
        home_shots_values = is_home.cumsum()
        away_shots_values = is_away.cumsum()
        
        # Assign back using the same indices (this ensures alignment)
        df.loc[game_indices, 'cumulative_home_xg'] = home_xg_values.values
        df.loc[game_indices, 'cumulative_away_xg'] = away_xg_values.values
        df.loc[game_indices, 'cumulative_home_shots'] = home_shots_values.values
        df.loc[game_indices, 'cumulative_away_shots'] = away_shots_values.values
    
    # Compute differentials
    df['xg_differential'] = df['cumulative_home_xg'] - df['cumulative_away_xg']
    df['shot_differential'] = df['cumulative_home_shots'] - df['cumulative_away_shots']
    
    print("   ✓ Cumulative stats computed")
    
    # 3. CREATE ROLLING FEATURES (last 2/5 minutes)
    print("   Computing rolling window features...")
    
    df['shots_last_2min_home'] = 0
    df['shots_last_2min_away'] = 0
    df['shots_last_5min_home'] = 0
    df['shots_last_5min_away'] = 0
    
    for i, game_id in enumerate(df[groupby_col].unique()):
        if (i + 1) % 200 == 0:
            print(f"   Processing game {i + 1}...")
        
        game_mask = df[groupby_col] == game_id
        game_indices = df[game_mask].index
        game_df = df.loc[game_indices].copy()
        
        times = game_df['time_elapsed'].values
        teams = game_df['team'].values
        
        # Vectorized approach for rolling windows
        # Fixed: teams contain 'HOME' or 'AWAY', not team codes!
        for idx, (time, team) in enumerate(zip(times, teams)):
            actual_idx = game_indices[idx]  # Get the actual index in df
            
            # Last 2 minutes (120 seconds)
            mask_2min = (times >= time - 120) & (times <= time)
            df.loc[actual_idx, 'shots_last_2min_home'] = np.sum(teams[mask_2min] == 'HOME')
            df.loc[actual_idx, 'shots_last_2min_away'] = np.sum(teams[mask_2min] == 'AWAY')
            
            # Last 5 minutes (300 seconds)
            mask_5min = (times >= time - 300) & (times <= time)
            df.loc[actual_idx, 'shots_last_5min_home'] = np.sum(teams[mask_5min] == 'HOME')
            df.loc[actual_idx, 'shots_last_5min_away'] = np.sum(teams[mask_5min] == 'AWAY')
    
    print("   ✓ Rolling window features computed")
    
    # 4. SELECT FINAL COLUMNS
    final_columns = [
        'game_id', 'game_id_unique', 'shot_id', 'season', 'is_playoff_game', 'period',
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