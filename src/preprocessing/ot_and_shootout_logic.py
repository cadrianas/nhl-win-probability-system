"""
Define game state types and handle OT/shootout logic.

Key concepts:
- Regulation: periods 1-3
- Regular season OT: period 4, 3v3 format (max 5 minutes, then shootout)
- Playoff OT: period 4+, 5v5 format (20 minute periods, no shootout)
- Shootout: period 5 (only in regular season)

Classification logic:
- is_playoff_game=0 + period>4 → shootout game (excluded from modeling)
- is_playoff_game=0 + period=4 → regular season OT
- is_playoff_game=1 + period>=4 → playoff OT
"""
from enum import Enum
import pandas as pd


class GameState(Enum):
    """
    Enumeration of game state types.
    
    Different game types have different dynamics and scoring hazards:
    - REGULATION: Normal 5v5 play
    - REGULAR_SEASON_OT: 3v3 overtime (5 minute sudden death format)
    - PLAYOFF_OT: Full strength overtime (20 minute periods, could be multiple periods)
    """
    REGULATION = "REG"
    REGULAR_SEASON_OT = "RS_OT_3V3"
    PLAYOFF_OT = "PLAYOFF_OT"
    
    @staticmethod
    def from_period_and_playoff_flag(period: int, is_playoff: bool) -> str:
        """
        Determine game state from period number and playoff flag.
        
        Args:
            period: Period number (1, 2, 3, 4, 5, etc.)
            is_playoff: True if playoff game, False if regular season
            
        Returns:
            Game state as string (REG, RS_OT_3V3, or PLAYOFF_OT)
            
        Logic:
        - period <= 3 → REGULATION (normal play)
        - period > 3 AND is_playoff → PLAYOFF_OT
        - period > 3 AND NOT is_playoff → REGULAR_SEASON_OT
        
        Note: Period 5 (shootout) should be filtered out before modeling.
        """
        if period <= 3:
            return GameState.REGULATION.value
        elif is_playoff:
            # Playoff: period 4, 5, 6, etc. all count as OT
            return GameState.PLAYOFF_OT.value
        else:
            # Regular season: period 4+ (including period 5 shootout)
            return GameState.REGULAR_SEASON_OT.value


def is_shootout_game(df: pd.DataFrame) -> pd.Series:
    """
    Identify games that contain shootout shots.
    
    A shootout game is:
    - Regular season (is_playoff_game == 0)
    - AND has shots from period 5 (shootout period)
    
    Args:
        df: DataFrame with 'game_id', 'period', 'is_playoff_game' columns
        
    Returns:
        pd.Series of booleans: True if the shot is from a shootout game, False otherwise
        
    Note: This marks EVERY SHOT in a shootout game as True, not just period 5 shots.
    To filter out shootout games entirely, use filter_out_shootouts().
    """
    # Find games that went to period 5 (shootout)
    max_periods = df.groupby("game_id")["period"].max()
    games_with_shootout = max_periods[max_periods >= 5].index
    
    # Mark all shots from shootout games
    return df["game_id"].isin(games_with_shootout)


def filter_out_shootouts(df: pd.DataFrame, keep_shootouts: bool = False) -> pd.DataFrame:
    """
    Filter out or keep shootout games based on configuration.
    
    Args:
        df: Shots dataframe with 'game_id', 'period', 'is_playoff_game'
        keep_shootouts: If True, keep all games. If False, exclude shootout games.
        
    Returns:
        Filtered dataframe (or original if keep_shootouts=True)
        
    Why filter shootouts?
    - Shootout shooters are selected plays, not continuous game situations
    - Shootout has different win probability dynamics (direct 1v1 vs. team play)
    - Only regular season games have shootouts
    """
    if keep_shootouts:
        return df
    
    # Identify games that went to shootout (period 5)
    shootout_mask = is_shootout_game(df)
    
    rows_before = len(df)
    df_filtered = df[~shootout_mask].copy()
    rows_after = len(df_filtered)
    
    games_removed = shootout_mask.groupby(df["game_id"]).first().sum()
    rows_removed = rows_before - rows_after
    
    print(f"   Removed {int(games_removed):.0f} shootout games ({rows_removed:,} shots)")
    
    return df_filtered


def add_game_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'game_state' column classifying each shot's game context.
    
    Adds a column with values: 'REG', 'RS_OT_3V3', or 'PLAYOFF_OT'
    
    Args:
        df: Shots dataframe with 'period' and 'is_playoff_game' columns
        
    Returns:
        DataFrame with new 'game_state' column
    """
    df = df.copy()
    
    # Apply game state classification to each row
    df["game_state"] = df.apply(
        lambda row: GameState.from_period_and_playoff_flag(
            period=row["period"],
            is_playoff=bool(row["is_playoff_game"])
        ),
        axis=1
    )
    
    return df


def get_game_state_summary(df: pd.DataFrame) -> dict:
    """
    Return summary statistics of game states in the dataset.
    
    Args:
        df: DataFrame with 'game_state' column
        
    Returns:
        Dict with counts and percentages of each game state
    """
    if "game_state" not in df.columns:
        return {"error": "game_state column not found"}
    
    total = len(df)
    summary = {}
    
    for state in ["REG", "RS_OT_3V3", "PLAYOFF_OT"]:
        count = (df["game_state"] == state).sum()
        pct = 100 * count / total if total > 0 else 0
        summary[state] = {
            "count": count,
            "percentage": pct
        }
    
    return summary