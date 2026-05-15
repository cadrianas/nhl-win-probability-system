"""
Canonical schema definitions for shots and game states.

This is the single source of truth for all data transformations.
Any change to the schema should be documented here first.
"""

# Raw MoneyPuck columns → standardized names (snake_case)
COLUMN_MAPPING = {
    "shotID": "shot_id",
    "gameId": "game_id",
    "season": "season",
    "isPlayoffGame": "is_playoff_game",
    "period": "period",
    "time": "time_in_period",
    "homeTeamCode": "home_team_code",
    "awayTeamCode": "away_team_code",
    "team": "team",
    "homeTeamGoals": "home_team_goals",
    "awayTeamGoals": "away_team_goals",
    "homeTeamWon": "home_team_won",
    "goal": "is_goal",
    "xGoal": "x_goal",
    "homeSkatersOnIce": "home_skaters_on_ice",
    "awaySkatersOnIce": "away_skaters_on_ice",
    "homeEmptyNet": "home_empty_net",
    "awayEmptyNet": "away_empty_net",
    "shotType": "shot_type",
}

# Data types for cleaned shots
SHOT_DTYPES = {
    "shot_id": "int32",
    "game_id": "int32",
    "season": "int16",
    "is_playoff_game": "bool",
    "period": "int8",
    "time_in_period": "float32",
    "home_team_code": "category",
    "away_team_code": "category",
    "team": "category",
    "home_team_goals": "int8",
    "away_team_goals": "int8",
    "home_team_won": "bool",
    "is_goal": "bool",
    "x_goal": "float32",
    "home_skaters_on_ice": "int8",
    "away_skaters_on_ice": "int8",
    "home_empty_net": "bool",
    "away_empty_net": "bool",
    "shot_type": "category",
    "game_seconds_elapsed": "int32",
    "game_seconds_remaining": "int32",
    "game_state": "category",
    "strength_state": "category",
    "target_home_win": "bool",
    "time_decay_factor": "float32",
    "is_even_strength": "bool",
    "is_power_play": "bool",
    "is_empty_net": "bool",
    "is_3v3": "bool",
}

# Game state output columns (in order)
GAME_STATE_COLUMNS = [
    "game_id",
    "shot_id",
    "season",
    "is_playoff_game",
    # TIME (continuous)
    "game_seconds_elapsed",
    "game_seconds_remaining",
    "time_decay_factor",
    # GAME TYPE
    "period",
    "game_state",
    # SCORE
    "home_team_goals",
    "away_team_goals",
    "score_differential",
    # STRENGTH
    "strength_state",
    "is_even_strength",
    "is_power_play",
    "is_empty_net",
    "is_3v3",
    # xG
    "cumulative_home_xg",
    "cumulative_away_xg",
    "xg_differential",
    # SHOTS
    "cumulative_home_shots",
    "cumulative_away_shots",
    "shot_differential",
    "shots_last_2min_home",
    "shots_last_2min_away",
    "shots_last_5min_home",
    "shots_last_5min_away",
    # TARGET
    "target_home_win",
]


def get_columns_to_keep() -> list:
    """Return list of raw columns to retain from MoneyPuck data."""
    return list(COLUMN_MAPPING.keys())
