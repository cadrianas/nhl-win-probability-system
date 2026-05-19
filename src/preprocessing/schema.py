"""
schema.py - IMPROVED VERSION

Separate schema definitions for each phase:
1. RAW_SHOT_DTYPES - Raw MoneyPuck columns (before any transforms)
2. SHOTS_CLEANED_DTYPES - After clean_shots.py (includes derived time, state, target)
3. GAME_STATE_DTYPES - After game_state_builders.py (cumulative, rolling features)

This makes it clear what columns should exist at each pipeline stage.
"""

import pandas as pd

# ============================================================================
# COLUMN MAPPING: Raw MoneyPuck (camelCase) → Standardized (snake_case)
# ============================================================================

COLUMN_MAPPING = {
    # Core identifiers
    "shotID": "shot_id",
    "game_id": "game_id",
    "id": "event_id",
    "season": "season",
    "isPlayoffGame": "is_playoff_game",
    "homeTeamWon": "home_team_won",
    
    # Time
    "time": "time_elapsed",
    "period": "period",
    "timeUntilNextEvent": "time_until_next_event",
    "timeSinceLastEvent": "time_since_last_event",
    "timeSinceFaceoff": "time_since_faceoff",
    
    # Teams
    "homeTeamCode": "home_team_code",
    "awayTeamCode": "away_team_code",
    "team": "team",
    "teamCode": "team_code",
    
    # Score state
    "homeTeamGoals": "home_team_goals",
    "awayTeamGoals": "away_team_goals",
    
    # Shot event
    "event": "event",
    "goal": "is_goal",
    "shotType": "shot_type",
    "shotOnEmptyNet": "shot_on_empty_net",
    "shotRebound": "shot_rebound",
    "shotRush": "shot_rush",
    "offWing": "off_wing",
    
    # Shot location
    "xCord": "x_coord",
    "yCord": "y_coord",
    "xCordAdjusted": "x_coord_adjusted",
    "yCordAdjusted": "y_coord_adjusted",
    "arenaAdjustedXCord": "arena_adjusted_x_coord",
    "arenaAdjustedYCord": "arena_adjusted_y_coord",
    "arenaAdjustedYCordAbs": "arena_adjusted_y_coord_abs",
    "arenaAdjustedShotDistance": "arena_adjusted_shot_distance",
    "shotDistance": "shot_distance",
    "lastEventxCord": "last_event_x_coord",
    "lastEventyCord": "last_event_y_coord",
    "lastEventxCord_adjusted": "last_event_x_coord_adjusted",
    "lastEventyCord_adjusted": "last_event_y_coord_adjusted",
    
    # Angles
    "shotAngle": "shot_angle",
    "shotAngleAdjusted": "shot_angle_adjusted",
    "shotAnglePlusRebound": "shot_angle_plus_rebound",
    "shotAnglePlusReboundSpeed": "shot_angle_plus_rebound_speed",
    "shotAngleReboundRoyalRoad": "shot_angle_rebound_royal_road",
    "lastEventShotAngle": "last_event_shot_angle",
    "lastEventShotDistance": "last_event_shot_distance",
    
    # Expected goals
    "xGoal": "x_goal",
    "xFroze": "x_froze",
    "xRebound": "x_rebound",
    "xPlayContinuedInZone": "x_play_continued_in_zone",
    "xPlayContinuedOutsideZone": "x_play_continued_outside_zone",
    "xPlayStopped": "x_play_stopped",
    "xShotWasOnGoal": "x_shot_was_on_goal",
    
    # Play continuation flags
    "shotPlayContinuedInZone": "play_continued_in_zone",
    "shotPlayContinuedOutsideZone": "play_continued_outside_zone",
    "shotGoalieFroze": "goalie_froze",
    "shotPlayStopped": "play_stopped",
    "shotGeneratedRebound": "generated_rebound",
    
    # Location zone
    "location": "location",
    "locationZone": "location_zone",
    
    # Game state - skaters
    "homeSkatersOnIce": "home_skaters_on_ice",
    "awaySkatersOnIce": "away_skaters_on_ice",
    "homeEmptyNet": "home_empty_net",
    "awayEmptyNet": "away_empty_net",
    
    # Penalties
    "homePenalty1TimeLeft": "home_penalty_1_time_left",
    "homePenalty1Length": "home_penalty_1_length",
    "awayPenalty1TimeLeft": "away_penalty_1_time_left",
    "awayPenalty1Length": "away_penalty_1_length",
    
    # Shooter information
    "shooterPlayerId": "shooter_player_id",
    "shooterName": "shooter_name",
    "shooterLeftRight": "shooter_left_right",
    "shooterTimeOnIce": "shooter_time_on_ice",
    "shooterTimeOnIceSinceFaceoff": "shooter_time_on_ice_since_faceoff",
    "playerPositionThatDidEvent": "player_position",
    "playerNumThatDidEvent": "player_number",
    
    # Goaltender information
    "goalieIdForShot": "goaltender_player_id",
    "goalieNameForShot": "goaltender_name",
    
    # Last event information
    "lastEventCategory": "last_event_category",
    "lastEventTeam": "last_event_team",
    "distanceFromLastEvent": "distance_from_last_event",
    "speedFromLastEvent": "speed_from_last_event",
    "playerNumThatDidLastEvent": "last_event_player_number",
    
    # Team situation (shooting)
    "shootingTeamForwardsOnIce": "shooting_team_forwards_on_ice",
    "shootingTeamDefencemenOnIce": "shooting_team_defencemen_on_ice",
    "shootingTeamAverageTimeOnIce": "shooting_team_avg_toi",
    "shootingTeamAverageTimeOnIceOfForwards": "shooting_team_avg_toi_forwards",
    "shootingTeamAverageTimeOnIceOfDefencemen": "shooting_team_avg_toi_defencemen",
    "shootingTeamMaxTimeOnIce": "shooting_team_max_toi",
    "shootingTeamMaxTimeOnIceOfForwards": "shooting_team_max_toi_forwards",
    "shootingTeamMaxTimeOnIceOfDefencemen": "shooting_team_max_toi_defencemen",
    "shootingTeamMinTimeOnIce": "shooting_team_min_toi",
    "shootingTeamMinTimeOnIceOfForwards": "shooting_team_min_toi_forwards",
    "shootingTeamMinTimeOnIceOfDefencemen": "shooting_team_min_toi_defencemen",
    
    # Team situation (defending)
    "defendingTeamForwardsOnIce": "defending_team_forwards_on_ice",
    "defendingTeamDefencemenOnIce": "defending_team_defencemen_on_ice",
    "defendingTeamAverageTimeOnIce": "defending_team_avg_toi",
    "defendingTeamAverageTimeOnIceOfForwards": "defending_team_avg_toi_forwards",
    "defendingTeamAverageTimeOnIceOfDefencemen": "defending_team_avg_toi_defencemen",
    "defendingTeamMaxTimeOnIce": "defending_team_max_toi",
    "defendingTeamMaxTimeOnIceOfForwards": "defending_team_max_toi_forwards",
    "defendingTeamMaxTimeOnIceOfDefencemen": "defending_team_max_toi_defencemen",
    "defendingTeamMinTimeOnIce": "defending_team_min_toi",
    "defendingTeamMinTimeOnIceOfForwards": "defending_team_min_toi_forwards",
    "defendingTeamMinTimeOnIceOfDefencemen": "defending_team_min_toi_defencemen",
    
    # Rest/fatigue
    "timeDifferenceSinceChange": "time_difference_since_change",
    "averageRestDifference": "average_rest_difference",
    
    # Boolean flags
    "isHomeTeam": "is_home_team",
    "shotWasOnGoal": "shot_was_on_goal",
}


# ============================================================================
# PHASE 1 OUTPUT: Raw shots after standardization (before feature creation)
# ============================================================================
# These are columns that exist in raw MoneyPuck data after column renaming
# Used by enforce_dtypes() in shot_cleaners.py
# This schema is checked DURING data cleaning (not all columns exist yet)

RAW_SHOT_DTYPES = {
    # IDs
    "shot_id": "int32",
    "game_id": "int32",
    "event_id": "int32",
    "season": "int16",
    
    # Boolean flags (raw)
    "is_playoff_game": "bool",
    "is_goal": "bool",
    "is_home_team": "bool",
    "shot_on_empty_net": "bool",
    "shot_was_on_goal": "bool",
    "shot_rebound": "bool",
    "shot_rush": "bool",
    "off_wing": "bool",
    "home_empty_net": "bool",
    "away_empty_net": "bool",
    "play_continued_in_zone": "bool",
    "play_continued_outside_zone": "bool",
    "goalie_froze": "bool",
    "play_stopped": "bool",
    "generated_rebound": "bool",
    "shot_angle_rebound_royal_road": "bool",
    "home_team_won": "bool",
    
    # Counts
    "period": "int8",
    "home_team_goals": "int8",
    "away_team_goals": "int8",
    "home_skaters_on_ice": "int8",
    "away_skaters_on_ice": "int8",
    "shooting_team_forwards_on_ice": "int8",
    "shooting_team_defencemen_on_ice": "int8",
    "defending_team_forwards_on_ice": "int8",
    "defending_team_defencemen_on_ice": "int8",
    
    # Time in seconds
    "time_elapsed": "int32",
    "time_until_next_event": "float32",
    "time_since_last_event": "float32",
    "time_since_faceoff": "float32",
    "home_penalty_1_time_left": "int16",
    "home_penalty_1_length": "int16",
    "away_penalty_1_time_left": "int16",
    "away_penalty_1_length": "int16",
    "shooter_time_on_ice": "float32",
    "shooter_time_on_ice_since_faceoff": "float32",
    "time_difference_since_change": "float32",
    "average_rest_difference": "float32",
    
    # Coordinates and distances
    "x_coord": "float32",
    "y_coord": "float32",
    "x_coord_adjusted": "float32",
    "y_coord_adjusted": "float32",
    "arena_adjusted_x_coord": "float32",
    "arena_adjusted_y_coord": "float32",
    "arena_adjusted_y_coord_abs": "float32",
    "shot_distance": "float32",
    "arena_adjusted_shot_distance": "float32",
    "last_event_x_coord": "float32",
    "last_event_y_coord": "float32",
    "last_event_x_coord_adjusted": "float32",
    "last_event_y_coord_adjusted": "float32",
    "distance_from_last_event": "float32",
    
    # Angles
    "shot_angle": "float32",
    "shot_angle_adjusted": "float32",
    "shot_angle_plus_rebound": "float32",
    "shot_angle_plus_rebound_speed": "float32",
    "last_event_shot_angle": "float32",
    "last_event_shot_distance": "float32",
    "speed_from_last_event": "float32",
    
    # Probabilities
    "x_goal": "float32",
    "x_froze": "float32",
    "x_rebound": "float32",
    "x_play_continued_in_zone": "float32",
    "x_play_continued_outside_zone": "float32",
    "x_play_stopped": "float32",
    "x_shot_was_on_goal": "float32",
    
    # TOI stats
    "shooting_team_avg_toi": "float32",
    "shooting_team_avg_toi_forwards": "float32",
    "shooting_team_avg_toi_defencemen": "float32",
    "shooting_team_max_toi": "float32",
    "shooting_team_max_toi_forwards": "float32",
    "shooting_team_max_toi_defencemen": "float32",
    "shooting_team_min_toi": "float32",
    "shooting_team_min_toi_forwards": "float32",
    "shooting_team_min_toi_defencemen": "float32",
    "defending_team_avg_toi": "float32",
    "defending_team_avg_toi_forwards": "float32",
    "defending_team_avg_toi_defencemen": "float32",
    "defending_team_max_toi": "float32",
    "defending_team_max_toi_forwards": "float32",
    "defending_team_max_toi_defencemen": "float32",
    "defending_team_min_toi": "float32",
    "defending_team_min_toi_forwards": "float32",
    "defending_team_min_toi_defencemen": "float32",
    
    # Categorical (raw)
    "home_team_code": "category",
    "away_team_code": "category",
    "team_code": "category",
    "team": "category",
    "event": "category",
    "location": "category",
    "location_zone": "category",
    "shot_type": "category",
    "shooter_name": "object",
    "shooter_left_right": "category",
    "player_position": "category",
    "goaltender_name": "object",
    "last_event_category": "category",
    "last_event_team": "category",
}


# ============================================================================
# PHASE 1 OUTPUT: shots_cleaned.parquet (after clean_shots.py)
# ============================================================================
# Includes:
# - All raw columns (from RAW_SHOT_DTYPES)
# - Derived time features (from create_time_features)
# - Game state classification (from add_game_state_column)
# - Strength state (from add_strength_state_column, create_strength_features)
# - Target variable (from add_target_to_shots)
# - Data quality flags (from handle_missing_values)
# Total: ~118 columns

SHOTS_CLEANED_DTYPES = {
    **RAW_SHOT_DTYPES,  # Include all raw columns
    
    # Derived time features (created by create_time_features)
    "game_seconds_elapsed": "int32",
    "game_seconds_remaining": "int32",
    "time_in_period_seconds": "int32",
    "time_in_period_minutes": "float32",
    "time_decay_factor": "float32",
    
    # Game state classification (created by add_game_state_column)
    "game_state": "category",  # REG, RS_OT_3V3, PLAYOFF_OT
    
    # Strength state features (created by add_strength_state_column, create_strength_features)
    "strength_state": "category",
    "is_even_strength": "bool",
    "is_power_play": "bool",
    "is_empty_net": "bool",
    "is_3v3": "bool",
    
    # Target variable (created by add_target_to_shots)
    "target_home_win": "bool",
    
    # Data quality flag (created by handle_missing_values)
    "is_toi_data_unavailable": "bool",
}


# ============================================================================
# PHASE 2 OUTPUT: game_states.parquet (after game_state_builders.py)
# ============================================================================
# Derived features created by:
# - Cumulative stats (cumulative_home_xg, cumulative_home_shots, etc.)
# - Rolling window features (shots_last_2min_home, shots_last_5min_home, etc.)
# - Score differential
# Total: ~35 columns (subset of shots_cleaned + derived features)

GAME_STATE_DTYPES = {
    # Identifiers
    "game_id": "int32",
    "shot_id": "int32",
    "season": "int16",
    "is_playoff_game": "bool",
    "period": "int8",
    
    # Time features
    "time_elapsed": "int32",
    "game_seconds_elapsed": "int32",
    "time_in_period_seconds": "int32",
    "time_in_period_minutes": "float32",
    "game_seconds_remaining": "int32",
    "time_decay_factor": "float32",
    
    # Game classification
    "game_state": "category",
    
    # Score
    "home_team_goals": "int8",
    "away_team_goals": "int8",
    "score_differential": "int8",
    
    # Strength state
    "strength_state": "category",
    "is_even_strength": "bool",
    "is_power_play": "bool",
    "is_empty_net": "bool",
    "is_3v3": "bool",
    
    # Shot location
    "x_coord": "float32",
    "y_coord": "float32",
    "shot_distance": "float32",
    "shot_angle": "float32",
    
    # Cumulative stats (DERIVED in game_state_builders)
    "cumulative_home_xg": "float32",
    "cumulative_away_xg": "float32",
    "xg_differential": "float32",
    "cumulative_home_shots": "int32",
    "cumulative_away_shots": "int32",
    "shot_differential": "int32",
    
    # Momentum (rolling windows, DERIVED in game_state_builders)
    "shots_last_2min_home": "int32",
    "shots_last_2min_away": "int32",
    "shots_last_5min_home": "int32",
    "shots_last_5min_away": "int32",
    
    # Target
    "target_home_win": "bool",
}


# ============================================================================
# GAME STATE COLUMN LIST (For Selection)
# ============================================================================
# List of expected columns in game_states.parquet
# Used by game_state_builders.py to select final columns

GAME_STATE_COLUMNS = [
    # Identifiers
    "game_id", "shot_id", "season", "is_playoff_game", "period",
    
    # Time features
    "time_elapsed", "game_seconds_elapsed", "time_in_period_seconds",
    "time_in_period_minutes", "game_seconds_remaining", "time_decay_factor",
    
    # Game classification
    "game_state",
    
    # Score
    "home_team_goals", "away_team_goals", "score_differential",
    
    # Strength state
    "strength_state", "is_even_strength", "is_power_play", "is_empty_net", "is_3v3",
    
    # Shot location
    "x_coord", "y_coord", "shot_distance", "shot_angle",
    
    # Cumulative stats
    "cumulative_home_xg", "cumulative_away_xg", "xg_differential",
    "cumulative_home_shots", "cumulative_away_shots", "shot_differential",
    
    # Momentum
    "shots_last_2min_home", "shots_last_2min_away",
    "shots_last_5min_home", "shots_last_5min_away",
    
    # Target
    "target_home_win",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_columns_to_keep() -> list:
    """Return list of raw MoneyPuck columns to retain during standardization."""
    return list(COLUMN_MAPPING.keys())


def validate_shots_cleaned(df: pd.DataFrame) -> dict:
    """
    Validate that shots_cleaned.parquet has expected columns and types.
    
    Returns:
        dict with validation results
    """
    messages = []
    
    # Check required columns exist
    for col in SHOTS_CLEANED_DTYPES.keys():
        if col not in df.columns:
            messages.append(f"WARNING: Missing expected column: {col}")
    
    return {
        "valid": len([m for m in messages if m.startswith("ERROR")]) == 0,
        "messages": messages,
        "columns_present": sum(1 for col in SHOTS_CLEANED_DTYPES.keys() if col in df.columns),
        "columns_expected": len(SHOTS_CLEANED_DTYPES),
    }


def validate_game_state(df: pd.DataFrame) -> dict:
    """
    Validate that game_states.parquet has all expected columns.
    
    Returns:
        dict with validation results
    """
    messages = []
    
    for col in GAME_STATE_COLUMNS:
        if col not in df.columns:
            messages.append(f"ERROR: Missing column: {col}")
    
    return {
        "valid": len([m for m in messages if m.startswith("ERROR")]) == 0,
        "messages": messages,
        "columns_present": sum(1 for col in GAME_STATE_COLUMNS if col in df.columns),
        "columns_expected": len(GAME_STATE_COLUMNS),
    }