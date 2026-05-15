"""
Define game state types and handle OT/shootout logic.

Key decisions:
- Regular season games ending in shootout are EXCLUDED initially
- Playoff OT games are INCLUDED
- Game type is explicitly encoded as categorical
"""

from enum import Enum
import pandas as pd


class GameState(Enum):
    """
    Enumeration of game state types.

    This is crucial for understanding underlying dynamics.
    Different game types have different scoring hazards.
    """

    REGULATION = "REG"
    REGULAR_SEASON_OT = "RS_OT_3V3"  # 3v3, 5 min then shootout
    PLAYOFF_OT = "PLAYOFF_OT"  # 5v5, 20 min periods

    @classmethod
    def from_period_and_playoff_flag(cls, period: int, is_playoff: bool) -> str:
        """
        Determine game state from period and playoff status.

        Args:
            period: 1, 2, 3, 4+
            is_playoff: bool

        Returns:
            GameState enum value as string

        Logic:
        - Period <= 3 → REGULATION
        - Period > 3 & playoff=1 → PLAYOFF_OT
        - Period > 3 & playoff=0 → REGULAR_SEASON_OT
        """
        if period <= 3:
            return cls.REGULATION.value
        elif period > 3 and is_playoff:
            return cls.PLAYOFF_OT.value
        else:  # period > 3 and not playoff
            return cls.REGULAR_SEASON_OT.value


def is_shootout_game(df: pd.DataFrame) -> pd.Series:
    """
    Identify games that went to shootout.

    Logic:
    - Regular season game (is_playoff_game == 0)
    - AND max period > 4
    - This means the game went beyond 5-min OT → shootout

    Args:
        df: DataFrame with 'game_id', 'period', 'is_playoff_game'

    Returns:
        pd.Series of booleans, True if game went to shootout
    """
    # Group by game, get max period
    max_periods = df.groupby("game_id")["period"].max()

    # Identify games that are regular season AND went to period 5+
    shootout_games = (df.groupby("game_id")["is_playoff_game"].first() == 0) & (
        max_periods > 4
    )

    return df["game_id"].isin(shootout_games[shootout_games].index)


def filter_out_shootouts(df: pd.DataFrame, keep_shootouts: bool = False) -> pd.DataFrame:
    """
    Remove or keep shootout games based on configuration.

    Args:
        df: Shots dataframe
        keep_shootouts: If True, keep all games. If False, exclude shootout games.

    Returns:
        Filtered dataframe
    """
    if keep_shootouts:
        return df

    shootout_mask = is_shootout_game(df)
    rows_before = len(df)
    df_filtered = df[~shootout_mask].copy()
    rows_after = len(df_filtered)

    games_removed = shootout_mask.groupby(df["game_id"]).first().sum()

    print(
        f"Filtered out {games_removed:.0f} shootout games: "
        f"{rows_before - rows_after:,} rows removed"
    )

    return df_filtered


def add_game_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit game_state column (REG, RS_OT_3V3, PLAYOFF_OT).

    Args:
        df: Shots dataframe with 'period' and 'is_playoff_game'

    Returns:
        DataFrame with new 'game_state' column
    """
    df["game_state"] = df.apply(
        lambda row: GameState.from_period_and_playoff_flag(
            row["period"], row["is_playoff_game"]
        ),
        axis=1,
    )

    return df
