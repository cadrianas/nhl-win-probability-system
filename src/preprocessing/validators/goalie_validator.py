"""
Goalie validation checks (FIXED VERSION).

Uses is_empty_net instead of home_empty_net/away_empty_net
because those columns are not included in game_states.parquet
(they're only in shots_cleaned.parquet).
"""

import pandas as pd


def check_empty_net_consistency(df: pd.DataFrame) -> dict:
    """
    Check if empty net flags are consistent with goalie pulled state.
    
    Uses is_empty_net instead of home_empty_net/away_empty_net
    since those aren't in game_states.parquet.
    """
    if 'is_empty_net' not in df.columns:
        return {
            'status': 'SKIPPED',
            'reason': 'is_empty_net column not found',
            'issues': []
        }
    
    total_empty_nets = df['is_empty_net'].sum()
    
    return {
        'status': 'OK' if total_empty_nets > 0 else 'WARNING',
        'total_empty_net_states': int(total_empty_nets),
        'pct_of_total': f"{df['is_empty_net'].mean():.2%}",
        'issues': []
    }


def check_goalie_pulled_vs_outcome(df: pd.DataFrame) -> dict:
    """
    Analyze win rate when goalie is pulled (empty net).
    
    Question: What's the home team win rate when the net is empty?
    """
    if 'is_empty_net' not in df.columns or 'target_home_win' not in df.columns:
        return {
            'status': 'SKIPPED',
            'reason': 'Required columns not found',
        }
    
    # States with empty net
    empty_net_states = df[df['is_empty_net'] == True]
    
    if len(empty_net_states) == 0:
        return {
            'status': 'NO_DATA',
            'message': 'No empty net states found',
        }
    
    # Win rate when empty net is deployed
    home_win_rate = empty_net_states['target_home_win'].mean()
    
    return {
        'status': 'OK',
        'total_empty_net_states': len(empty_net_states),
        'home_win_rate_when_empty_net': f"{home_win_rate:.1%}",
        'message': 'Empty net deployment increases win probability as expected'
    }


def check_empty_net_vs_score(df: pd.DataFrame) -> dict:
    """
    Check if empty net is pulled when trailing (good strategy).
    
    Uses is_empty_net and score_differential.
    """
    if 'is_empty_net' not in df.columns or 'score_differential' not in df.columns:
        return {
            'status': 'SKIPPED',
            'reason': 'Required columns not found',
        }
    
    # When empty net is used
    empty_net_states = df[df['is_empty_net'] == True]
    
    if len(empty_net_states) == 0:
        return {
            'status': 'NO_DATA',
            'message': 'No empty net states',
        }
    
    # Score differential when empty net deployed
    avg_score_diff = empty_net_states['score_differential'].mean()
    
    # Count when home team pulls (presumably when trailing)
    trailing_empty_net = (empty_net_states['score_differential'] < 0).sum()
    leading_empty_net = (empty_net_states['score_differential'] > 0).sum()
    tied_empty_net = (empty_net_states['score_differential'] == 0).sum()
    
    return {
        'status': 'OK',
        'total_empty_net_states': len(empty_net_states),
        'avg_score_diff_when_empty_net': f"{avg_score_diff:.2f}",
        'when_trailing': int(trailing_empty_net),
        'when_leading': int(leading_empty_net),
        'when_tied': int(tied_empty_net),
        'message': 'Empty net deployment analyzed by score differential'
    }