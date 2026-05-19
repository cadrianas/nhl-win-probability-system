"""
Column standardization and type enforcement for raw shots.

This module handles:
- Renaming camelCase → snake_case using schema
- Converting to optimal dtypes
- Standardizing missing values
- Data quality validation
"""
import pandas as pd
from src.preprocessing.schema import COLUMN_MAPPING, RAW_SHOT_DTYPES

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename camelCase columns to snake_case.
    
    Args:
        df: Raw MoneyPuck dataframe
        
    Returns:
        DataFrame with standardized column names (subset of original)
        
    Note: Only keeps columns that exist in COLUMN_MAPPING
    """
    # Get columns to keep (those defined in schema)
    cols_to_keep = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    
    # Subset and rename
    df_subset = df[cols_to_keep].copy()
    df_subset.rename(columns=COLUMN_MAPPING, inplace=True)
    
    print(f"   Kept {len(cols_to_keep)}/{len(COLUMN_MAPPING)} mapped columns")
    
    return df_subset


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to optimal dtypes using schema.
    
    Reduces memory footprint ~60% vs. default dtypes.
    
    Args:
        df: DataFrame with standardized column names
        
    Returns:
        DataFrame with optimized types
    """
    df = df.copy()
    
    # Separate by dtype to handle type conversion carefully
    categorical_cols = []
    object_cols = []
    bool_cols = []
    numeric_cols = []
    
    for col, dtype in RAW_SHOT_DTYPES.items():
        if col not in df.columns:
            continue
            
        if dtype == "category":
            categorical_cols.append(col)
        elif dtype == "object":
            object_cols.append(col)
        elif dtype == "bool":
            bool_cols.append(col)
        else:
            numeric_cols.append((col, dtype))
    
    # Convert numeric first
    for col, dtype in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        except Exception as e:
            print(f"   WARNING: Could not convert {col} to {dtype}: {e}")
    
    # Convert booleans
    for col in bool_cols:
        try:
            # Convert to numeric first, then bool
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("bool")
        except Exception as e:
            print(f"   WARNING: Could not convert {col} to bool: {e}")
    
    # Convert strings to object
    for col in object_cols:
        df[col] = df[col].astype(str)
    
    # Convert to categorical last (after filling NaN)
    for col in categorical_cols:
        try:
            df[col] = df[col].astype("category")
        except Exception as e:
            print(f"   WARNING: Could not convert {col} to category: {e}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize missing value handling.
    
    Rules:
    - time_elapsed: NaN is critical → drop rows
    - game_id: NaN is critical → drop rows
    - shot_type: missing → fill with 'unknown'
    - skater counts: NaN → impute 5 (full strength assumption)
    - TOI data: value == 999 (min) or 0 (max) when unavailable → flag
    - xGoal and probabilities: leave as NaN for now (handle in features)
    
    Args:
        df: DataFrame with enforced types
        
    Returns:
        DataFrame with explicit missing value handling
    """
    df = df.copy()
    
    initial_rows = len(df)
    
    # Drop rows with critical missing values
    critical_cols = ["time_elapsed", "game_id", "period"]
    critical_cols_exist = [col for col in critical_cols if col in df.columns]
    if critical_cols_exist:
        df = df.dropna(subset=critical_cols_exist)
    
    # Handle categorical missing values
    if "shot_type" in df.columns:
        # Add 'unknown' as a valid category if needed
        if df["shot_type"].isnull().any():
            if df["shot_type"].dtype.name == "category":
                df["shot_type"] = df["shot_type"].cat.add_categories("unknown")
            df["shot_type"] = df["shot_type"].fillna("unknown")
    
    if "event" in df.columns:
        if df["event"].isnull().any():
            if df["event"].dtype.name == "category":
                df["event"] = df["event"].cat.add_categories("unknown")
            df["event"] = df["event"].fillna("unknown")
    
    # Handle skater counts (assume full strength if missing)
    if "home_skaters_on_ice" in df.columns:
        df["home_skaters_on_ice"] = df["home_skaters_on_ice"].fillna(5).astype("int8")
    if "away_skaters_on_ice" in df.columns:
        df["away_skaters_on_ice"] = df["away_skaters_on_ice"].fillna(5).astype("int8")
    
    # Flag TOI data quality issues (2007-2008)
    # MoneyPuck sets min=999, max=0 when unavailable
    toi_min_cols = [col for col in df.columns if "min_toi" in col]
    df["is_toi_data_unavailable"] = False
    for col in toi_min_cols:
        if col in df.columns:
            df.loc[df[col] == 999, "is_toi_data_unavailable"] = True
    
    # Leave xGoal and probabilities as NaN (handle in feature engineering)
    # but log the count
    prob_cols = ["x_goal", "x_froze", "x_rebound", "x_play_continued_in_zone",
                 "x_play_continued_outside_zone", "x_play_stopped", "x_shot_was_on_goal"]
    xg_null_counts = {}
    for col in prob_cols:
        if col in df.columns:
            xg_null_counts[col] = df[col].isnull().sum()
    
    rows_removed = initial_rows - len(df)
    missing_after = df.isnull().sum().sum()
    
    print(f"   Rows removed: {rows_removed:,}")
    print(f"   Missing values remaining: {missing_after:,}")
    
    if xg_null_counts:
        print(f"   xGoal missing values: {xg_null_counts.get('x_goal', 0):,}")
    
    if df["is_toi_data_unavailable"].any():
        print(f"   TOI data unavailable in: {df['is_toi_data_unavailable'].sum():,} shots")
    
    return df