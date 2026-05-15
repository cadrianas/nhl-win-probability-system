"""
Convert period-based time to continuous game seconds.

Key principle:
- No time resets at period boundaries
- OT extends naturally as additional 1200-second blocks
- This is ESSENTIAL for sequential models (LSTM, Transformers)

Why this matters:
- Time decay features work correctly across periods
- Momentum features don't reset arbitrarily
- Deep learning models learn continuous temporal patterns
"""

import pandas as pd

SECONDS_PER_PERIOD = 1200  # 20 minutes


def compute_game_seconds_elapsed(period: int, time_in_period: float) -> float:
    """
    Convert (period, time_in_period) → total elapsed seconds from puck drop.

    Args:
        period: 1, 2, 3, 4, 5, ... (extends into OT)
        time_in_period: minutes into period (float, e.g., 15.5)

    Returns:
        Total elapsed seconds from game start

    Examples:
        period=1, time_in_period=15.5  → 930 seconds
        period=2, time_in_period=0.0   → 1200 seconds (exactly start of P2)
        period=3, time_in_period=20.0  → 3600 seconds (exactly end of P3)
        period=4, time_in_period=2.5   → 3750 seconds (2.5 min into OT)

    Logic:
        For periods 1-3: standard
        For OT periods (4+): each is a full 1200-second block
    """
    time_in_seconds = time_in_period * 60

    if period <= 3:
        # Regulation: periods are sequential
        elapsed = (period - 1) * SECONDS_PER_PERIOD + time_in_seconds
    else:
        # OT: all 3 regulation periods + additional OT periods
        regulation_seconds = 3 * SECONDS_PER_PERIOD
        ot_period_number = period - 3  # OT period 1 is period 4, etc.
        ot_seconds = (ot_period_number - 1) * SECONDS_PER_PERIOD + time_in_seconds
        elapsed = regulation_seconds + ot_seconds

    return elapsed


def compute_game_seconds_remaining(period: int, time_in_period: float, is_playoff: bool) -> float:
    """
    Compute seconds remaining in the game.

    Args:
        period: Current period
        time_in_period: Minutes into current period
        is_playoff: Whether playoff (relevant for OT rules)

    Returns:
        Seconds remaining (or approximation for OT)

    Note:
    - For regulation: simple 3600 - elapsed
    - For OT: approximation; assume game could go to ~5 OT periods
    """
    elapsed = compute_game_seconds_elapsed(period, time_in_period)

    if period <= 3:
        # Regulation: always 3600 seconds total
        remaining = 3600 - elapsed
    else:
        # OT: approximation. Assume max of 5 OT periods
        max_game_seconds = 3600 + (5 * SECONDS_PER_PERIOD)
        remaining = max_game_seconds - elapsed
        remaining = max(0, remaining)  # Never negative

    return remaining


def create_time_features(df: pd.DataFrame, use_continuous_time: bool = True) -> pd.DataFrame:
    """
    Add comprehensive time features to shots dataframe.

    Args:
        df: Shots dataframe with 'period' and 'time_in_period'
        use_continuous_time: If True, use global elapsed seconds

    Returns:
        DataFrame with new time columns
    """
    df["game_seconds_elapsed"] = df.apply(
        lambda row: compute_game_seconds_elapsed(row["period"], row["time_in_period"]),
        axis=1,
    ).astype("int32")

    df["game_seconds_remaining"] = df.apply(
        lambda row: compute_game_seconds_remaining(
            row["period"], row["time_in_period"], row["is_playoff_game"]
        ),
        axis=1,
    ).astype("int32")

    # Time decay factor: goes from 1.0 (game start) to 0.0 (game end)
    # Used to model "teams protect leads as time dwindles"
    df["time_decay_factor"] = (df["game_seconds_remaining"] / 3600).astype("float32")
    df["time_decay_factor"] = df["time_decay_factor"].clip(0, 1)  # Bound to [0, 1]

    return df
