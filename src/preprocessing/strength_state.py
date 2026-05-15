"""
Map skater counts to categorical strength states.

This is crucial because:
- 5v5 vs 5v4 have wildly different scoring dynamics
- Empty net is a regime change
- 3v3 OT is fundamentally different from 5v5 regulation
- These need to be explicit features for calibration and interpretability
"""

from enum import Enum
import pandas as pd


class StrengthState(Enum):
    """Categorical strength states in hockey."""

    EVEN_STRENGTH = "5v5"
    POWER_PLAY_HOME = "5v4_HOME"
    POWER_PLAY_AWAY = "5v4_AWAY"
    SHORTHANDED_HOME = "4v5_HOME"
    SHORTHANDED_AWAY = "4v5_AWAY"
    DOUBLE_MINOR = "4v4"
    DEEP_OT = "3v3"
    EMPTY_NET_HOME = "6v5_HOME"
    EMPTY_NET_AWAY = "5v6_AWAY"
    UNKNOWN = "UNKNOWN"


def get_strength_state(
    home_skaters: int,
    away_skaters: int,
    perspective: str = "overall",
) -> str:
    """
    Map skater counts to categorical strength state.

    Args:
        home_skaters: 3-6
        away_skaters: 3-6
        perspective: 'overall', 'home', or 'away'

    Returns:
        StrengthState enum value as string

    Logic:
    - First check for valid range (3-6)
    - Map combinations to explicit categories
    """
    # Validation
    if not (3 <= home_skaters <= 6) or not (3 <= away_skaters <= 6):
        return StrengthState.UNKNOWN.value

    # Handle empty net (6 skaters)
    if home_skaters == 6:
        return StrengthState.EMPTY_NET_HOME.value
    if away_skaters == 6:
        return StrengthState.EMPTY_NET_AWAY.value

    # Handle 3v3 (deep OT)
    if home_skaters == 3 and away_skaters == 3:
        return StrengthState.DEEP_OT.value

    # Even strength
    if home_skaters == away_skaters:
        if home_skaters == 4:
            return StrengthState.DOUBLE_MINOR.value
        elif home_skaters == 5:
            return StrengthState.EVEN_STRENGTH.value

    # Power play / shorthanded
    if home_skaters > away_skaters:
        if perspective == "home":
            return StrengthState.POWER_PLAY_HOME.value
        elif perspective == "away":
            return StrengthState.SHORTHANDED_AWAY.value
        else:
            return StrengthState.POWER_PLAY_HOME.value

    if away_skaters > home_skaters:
        if perspective == "away":
            return StrengthState.POWER_PLAY_AWAY.value
        elif perspective == "home":
            return StrengthState.SHORTHANDED_HOME.value
        else:
            return StrengthState.POWER_PLAY_AWAY.value

    return StrengthState.UNKNOWN.value


def add_strength_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit strength_state column to shots dataframe.

    Args:
        df: Shots with 'home_skaters_on_ice' and 'away_skaters_on_ice'

    Returns:
        DataFrame with new 'strength_state' column
    """
    df["strength_state"] = df.apply(
        lambda row: get_strength_state(
            row["home_skaters_on_ice"],
            row["away_skaters_on_ice"],
        ),
        axis=1,
    ).astype("category")

    return df


def create_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create boolean features for each strength state.

    Examples:
    - is_even_strength: bool
    - is_power_play: bool
    - is_empty_net: bool
    - is_3v3: bool

    These are useful for modeling because categorical variables
    interact with probability in non-linear ways.

    Args:
        df: Shots with 'strength_state' column

    Returns:
        DataFrame with new boolean strength features
    """
    df["is_even_strength"] = (df["strength_state"] == StrengthState.EVEN_STRENGTH.value)

    df["is_power_play"] = df["strength_state"].isin(
        [
            StrengthState.POWER_PLAY_HOME.value,
            StrengthState.POWER_PLAY_AWAY.value,
        ]
    )

    df["is_empty_net"] = df["strength_state"].isin(
        [
            StrengthState.EMPTY_NET_HOME.value,
            StrengthState.EMPTY_NET_AWAY.value,
        ]
    )

    df["is_3v3"] = (df["strength_state"] == StrengthState.DEEP_OT.value)

    return df
