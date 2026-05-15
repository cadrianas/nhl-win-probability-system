"""
Define target variable for win probability modeling.

Key decision:
- home_team_win_before_shootout (exclude shootout games initially)
- This is more defensible statistically and avoids contaminating the model
- Later we can build separate shootout models
"""

import pandas as pd


def identify_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, determine:
    1. Did home team win?
    2. Did game go to shootout?
    3. Target: home_team_win_before_shootout

    Args:
        df: Shots dataframe with game outcomes

    Returns:
        DataFrame with target variable added
    """
    # Group by game
    game_outcomes = (
        df.groupby("game_id")
        .agg(
            {
                "home_team_won": "first",  # Should be same for all rows in game
                "period": "max",
                "is_playoff_game": "first",
            }
        )
        .reset_index()
    )

    # Determine if game went to shootout
    game_outcomes["went_to_shootout"] = (
        (game_outcomes["is_playoff_game"] == 0) & (game_outcomes["period"] > 4)
    )

    # Target: home team won
    # (We filter out shootout games separately)
    game_outcomes["home_team_win_before_shootout"] = game_outcomes["home_team_won"].astype(
        bool
    )

    return game_outcomes


def add_target_to_shots(
    df: pd.DataFrame,
    exclude_shootouts: bool = True,
) -> pd.DataFrame:
    """
    Add target variable to shot-level data.

    Args:
        df: Shots dataframe
        exclude_shootouts: If True, remove shootout games before returning

    Returns:
        DataFrame with target variable; optionally filtered
    """
    # Ensure target exists
    if "home_team_won" not in df.columns:
        raise ValueError("home_team_won not found in dataframe")

    # Add target (same for all rows in a game)
    df["target_home_win"] = df.groupby("game_id")["home_team_won"].transform("first")

    if exclude_shootouts:
        # Identify and remove shootout games
        max_periods = df.groupby("game_id")["period"].transform("max")
        is_shootout = (df["is_playoff_game"] == 0) & (max_periods > 4)

        rows_before = len(df)
        df = df[~is_shootout].copy()
        rows_after = len(df)

        games_removed = is_shootout.groupby(df["game_id"]).first().sum()

        print(f"Excluded {games_removed:.0f} shootout games: {rows_before - rows_after:,} rows")

    return df
