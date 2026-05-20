"""
DEBUG VERSION: game_state_builders with print statements
This will show us exactly where the cumulative computation is failing
"""

import pandas as pd
import numpy as np

def create_game_states_DEBUG(shots: pd.DataFrame) -> pd.DataFrame:
    """Debug version with print statements."""
    
    df = shots.copy()
    
    # Use game_id_unique if available
    groupby_col = 'game_id_unique' if 'game_id_unique' in df.columns else 'game_id'
    print(f"Using '{groupby_col}' for grouping")
    print(f"x_goal in df: {'x_goal' in df.columns}")
    print(f"team in df: {'team' in df.columns}")
    
    df = df.sort_values([groupby_col, 'time_elapsed']).reset_index(drop=True)
    
    # Initialize columns
    df['cumulative_home_shots'] = 0
    df['cumulative_away_shots'] = 0
    df['cumulative_home_xg'] = 0.0
    df['cumulative_away_xg'] = 0.0
    
    print("\n" + "=" * 100)
    print("DEBUGGING CUMULATIVE COMPUTATION")
    print("=" * 100)
    
    # Process first 3 games as debug
    for i, game_id in enumerate(df[groupby_col].unique()):
        if i >= 3:  # Only do first 3 games for debugging
            break
        
        print(f"\n--- GAME {i+1}: game_id_unique={game_id} ---")
        
        game_mask = df[groupby_col] == game_id
        game_indices = df[game_mask].index
        game_df = df.loc[game_indices].copy()
        
        print(f"  Shots in game: {len(game_df)}")
        print(f"  Indices: {game_indices[:5].tolist()}... (first 5)")
        
        home_team = game_df.iloc[0]['home_team_code']
        away_team = game_df.iloc[0]['away_team_code']
        
        print(f"  Home team: {home_team}, Away team: {away_team}")
        
        # Check team column
        print(f"  Teams in game_df['team']: {game_df['team'].unique().tolist()}")
        
        # Create boolean masks
        is_home = game_df['team'] == home_team
        is_away = game_df['team'] == away_team
        
        print(f"  is_home value counts: {is_home.value_counts().to_dict()}")
        print(f"  is_away value counts: {is_away.value_counts().to_dict()}")
        
        # Check x_goal
        print(f"  x_goal values in game: {game_df['x_goal'].head(5).tolist()}")
        print(f"  x_goal mean: {game_df['x_goal'].mean():.4f}")
        
        # Compute cumulative
        home_xg_product = is_home * game_df['x_goal']
        print(f"  (is_home * x_goal) sample: {home_xg_product.head(5).tolist()}")
        
        home_xg_values = home_xg_product.cumsum()
        print(f"  cumsum result (first 10): {home_xg_values.head(10).tolist()}")
        print(f"  cumsum result (max): {home_xg_values.max()}")
        
        away_xg_values = (is_away * game_df['x_goal']).cumsum()
        print(f"  away cumsum (max): {away_xg_values.max()}")
        
        # Try assignment
        print(f"\n  Attempting assignment...")
        print(f"  df.loc[game_indices, 'cumulative_home_xg'] before: {df.loc[game_indices, 'cumulative_home_xg'].head(3).tolist()}")
        
        df.loc[game_indices, 'cumulative_home_xg'] = home_xg_values.values
        df.loc[game_indices, 'cumulative_away_xg'] = away_xg_values.values
        
        print(f"  df.loc[game_indices, 'cumulative_home_xg'] after: {df.loc[game_indices, 'cumulative_home_xg'].head(3).tolist()}")
        print(f"  Success!")
    
    print("\n" + "=" * 100)
    print("DEBUG COMPLETE")
    print("=" * 100)
    
    return df

# Test it
if __name__ == '__main__':
    print("\nLoading shots...")
    shots = pd.read_parquet("data/processed/shots_cleaned.parquet")
    print(f"Loaded {len(shots):,} shots")
    
    print("\nRunning debug version...")
    result = create_game_states_DEBUG(shots)
    
    print(f"\nResult cumulative_home_xg max: {result['cumulative_home_xg'].max()}")
    print(f"Result cumulative_home_xg min: {result['cumulative_home_xg'].min()}")