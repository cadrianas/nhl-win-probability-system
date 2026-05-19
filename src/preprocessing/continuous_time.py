"""
Convert period-based time to continuous game seconds.

Key principle:
- MoneyPuck's 'time' column is already total elapsed seconds from puck drop
- We derive 'time_in_period' from (time_elapsed, period) for interpretability
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


def compute_time_in_period(time_elapsed: float, period: int) -> float:
    """
    Extract time within current period from total elapsed seconds.
    
    Inverse of: time_elapsed = (period-1) * 1200 + time_in_period_seconds
    
    Args:
        time_elapsed: Total seconds from game start (from MoneyPuck 'time' column)
        period: Current period (1, 2, 3, 4, ...)
        
    Returns:
        Seconds into current period (0 to 1200)
        
    Examples:
        time_elapsed=930, period=1   → 930 seconds (15.5 min into P1)
        time_elapsed=1200, period=2  → 0 seconds (start of P2)
        time_elapsed=3600, period=3  → 600 seconds (10 min into P3)
        time_elapsed=3750, period=4  → 150 seconds (2.5 min into OT)
    """
    if period <= 3:
        # Regulation: straightforward
        period_start_seconds = (period - 1) * SECONDS_PER_PERIOD
        time_in_period = time_elapsed - period_start_seconds
    else:
        # OT: all 3 regulation periods + additional OT periods
        regulation_seconds = 3 * SECONDS_PER_PERIOD
        ot_period_number = period - 3  # Period 4 is OT #1, etc.
        ot_period_start = (ot_period_number - 1) * SECONDS_PER_PERIOD
        time_in_period = time_elapsed - regulation_seconds - ot_period_start
    
    # Clip to valid range (shouldn't happen with good data)
    time_in_period = max(0, min(time_in_period, SECONDS_PER_PERIOD))
    
    return time_in_period


def compute_game_seconds_remaining(time_elapsed: float, period: int, is_playoff: bool) -> float:
    """
    Compute seconds remaining in the game.

    Args:
        time_elapsed: Total seconds from game start
        period: Current period
        is_playoff: Whether playoff (relevant for OT rules)

    Returns:
        Seconds remaining (or approximation for OT)

    Note:
    - For regulation: simple 3600 - elapsed
    - For OT: approximation; assume game could go to ~5 OT periods
    """
    if period <= 3:
        # Regulation: always 3600 seconds total (3 × 1200)
        remaining = 3600 - time_elapsed
    else:
        # OT: approximation. Assume max of 5 OT periods (10 × 1200 = 12000 seconds total)
        max_game_seconds = 3600 + (5 * SECONDS_PER_PERIOD)
        remaining = max_game_seconds - time_elapsed
        remaining = max(0, remaining)  # Never negative

    return remaining


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive time features to shots dataframe.
    
    INPUT REQUIREMENTS:
    - 'time_elapsed': Total seconds from game start (from MoneyPuck 'time' column)
    - 'period': Current period (1, 2, 3, 4, ...)
    - 'is_playoff_game': Boolean flag

    OUTPUT:
    - 'game_seconds_elapsed': Copy of time_elapsed (total elapsed seconds)
    - 'time_in_period_seconds': Seconds into current period (0-1200)
    - 'time_in_period_minutes': Minutes into current period (0-20)
    - 'game_seconds_remaining': Seconds until game end (approximation for OT)
    - 'time_decay_factor': Normalized remaining time (1.0 at start, 0.0 at regulation end)

    Args:
        df: Shots dataframe with required columns

    Returns:
        DataFrame with new time columns
    """
    df = df.copy()
    
    # ✅ THIS WAS MISSING! Create game_seconds_elapsed from time_elapsed
    # game_seconds_elapsed is just the total elapsed time (same as time_elapsed)
    # We create it as a separate column for clarity in downstream code
    df["game_seconds_elapsed"] = df["time_elapsed"].astype("int32")
    
    # Derive time_in_period from total elapsed time
    df["time_in_period_seconds"] = df.apply(
        lambda row: compute_time_in_period(row["time_elapsed"], row["period"]),
        axis=1,
    ).astype("int32")
    
    # Convert to minutes for interpretability
    df["time_in_period_minutes"] = (df["time_in_period_seconds"] / 60).astype("float32")

    # Compute remaining seconds
    df["game_seconds_remaining"] = df.apply(
        lambda row: compute_game_seconds_remaining(
            row["time_elapsed"], 
            row["period"], 
            row["is_playoff_game"]
        ),
        axis=1,
    ).astype("int32")

    # Time decay factor: goes from 1.0 (game start) to 0.0 (regulation end, ~3600s)
    # Useful for modeling "teams protect leads as time dwindles"
    # Clipped at 0 for OT (after regulation, stays at 0)
    df["time_decay_factor"] = (df["game_seconds_remaining"] / 3600).astype("float32")
    df["time_decay_factor"] = df["time_decay_factor"].clip(0, 1)

    return df