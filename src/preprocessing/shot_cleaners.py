"""
Column standardization and type enforcement for raw shots.

This module handles:
- Renaming camelCase → snake_case
- Converting to optimal dtypes
- Standardizing missing values
"""

import pandas as pd
from src.preprocessing.schema import COLUMN_MAPPING, SHOT_DTYPES


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename camelCase columns to snake_case.

    Args:
        df: Raw MoneyPuck dataframe

    Returns:
        DataFrame with standardized column names
    """
    # Keep only mapped columns, rename them
    cols_to_keep = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    df_subset = df[cols_to_keep].copy()

    # Rename
    df_subset.rename(columns=COLUMN_MAPPING, inplace=True)

    return df_subset


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to optimal dtypes.

    Reduces memory footprint ~60% vs. default dtypes.

    Args:
        df: DataFrame with standardized column names

    Returns:
        DataFrame with optimized types
    """
    for col, dtype in SHOT_DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                # Skip columns that can't be converted (not yet computed)
                pass

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize missing value handling.

    Rules:
    - time_in_period: NaN is invalid → drop
    - shot_type: missing → 'unknown'
    - skater counts: NaN → impute 5 (full strength assumption)

    Args:
        df: DataFrame with enforced types

    Returns:
        DataFrame with explicit missing value handling
    """
    missing_before = df.isnull().sum().sum()

    # Drop if critical time data missing
    df = df.dropna(subset=["time_in_period"])

    # Fill categorical with 'unknown'
    if "shot_type" in df.columns:
        df["shot_type"] = df["shot_type"].fillna("unknown")

    # Skater counts: assume full strength if missing (5 vs 5)
    if "home_skaters_on_ice" in df.columns:
        df["home_skaters_on_ice"] = df["home_skaters_on_ice"].fillna(5).astype("int8")
    if "away_skaters_on_ice" in df.columns:
        df["away_skaters_on_ice"] = df["away_skaters_on_ice"].fillna(5).astype("int8")

    missing_after = df.isnull().sum().sum()

    print(f"Missing values: {missing_before:,} → {missing_after:,}")

    return df
