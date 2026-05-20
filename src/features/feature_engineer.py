"""
Phase 2: Feature Engineering with Polars (CORRECTED)
=====================================================

This version works with the ACTUAL Phase 1 output schema:
- Input: 37 columns from game_states.parquet
- Output: Core + Context + Interaction features

Fixed: core_features() now recomputes xg_differential and shot_differential
from the working cumulative columns instead of using the broken zeros.
"""

import polars as pl
from pathlib import Path
from typing import Tuple, Dict, List, Union
import json
from datetime import datetime


class FeatureEngineer:
    """
    Feature engineering orchestrator (corrected for actual Phase 1 schema).
    
    Available input columns from Phase 1:
    - game_id, shot_id, season, period, is_playoff_game
    - time_elapsed, game_seconds_remaining
    - score_differential, strength_state
    - is_even_strength, is_power_play, is_empty_net
    - cumulative_home_xg, cumulative_away_xg (working!)
    - cumulative_home_shots, cumulative_away_shots (working!)
    - xg_differential (broken - all zeros), shot_differential (broken - all zeros)
    - target_home_win
    """
    
    def __init__(self, input_path: Union[Path, str], config: Dict = None):
        """Initialize feature engineer."""
        self.input_path = Path(input_path)
        self.config = config or self._default_config()
        self.states = None
        self.features = None
        self.feature_stats = {}
        
    @staticmethod
    def _default_config() -> Dict:
        """Default configuration."""
        return {
            'train_test_split_season': 2024,
            'feature_groups': ['core', 'context', 'interaction']
        }
    
    def load_data(self, lazy: bool = True) -> 'FeatureEngineer':
        """Load game states from parquet."""
        print(f"📥 Loading game states from {self.input_path}")
        
        if lazy:
            self.states = pl.scan_parquet(self.input_path)
        else:
            self.states = pl.read_parquet(self.input_path)
        
        print(f"   ✅ Loaded successfully (lazy={lazy})")
        return self
    
    def core_features(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Core features: select from Phase 1 + RECOMPUTE broken differentials.
        
        Phase 1 has cumulative_home_xg and cumulative_away_xg (working!)
        But xg_differential and shot_differential are 0 (broken!)
        
        Solution: Recompute from the working cumulative values.
        """
        
        core = self.states.select([
            'game_id',
            'shot_id',
            'season',
            'period',
            'time_elapsed',
            'game_seconds_remaining',
            'is_playoff_game',
            'score_differential',  # This one works fine
            'strength_state',
            'is_even_strength',
            'is_power_play',
            'is_empty_net',
            'target_home_win',
            'cumulative_home_xg',  
            'cumulative_away_xg',  
            'cumulative_home_shots', 
            'cumulative_away_shots',  
        ])
        
        # RECOMPUTE the differentials from cumulative values
        # (Phase 1 computed these as 0, which is wrong)
        core = core.with_columns([
            (pl.col('cumulative_home_xg') - pl.col('cumulative_away_xg'))
            .alias('xg_differential'),
            
            (pl.col('cumulative_home_shots') - pl.col('cumulative_away_shots'))
            .alias('shot_differential'),
        ])
        
        # Drop the cumulative columns (we don't need them anymore)
        # This matches the expected output schema
        core = core.drop([
            'cumulative_home_xg',
            'cumulative_away_xg', 
            'cumulative_home_shots',
            'cumulative_away_shots',
        ])
        
        return core
    
    def context_features(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Context features: game situation and progression.
        
        Creates:
        - time_remaining_normalized
        - is_late_game (< 5 min)
        - is_close_game (within 1 goal)
        - period dummies
        - game_phase
        """
        core = self.core_features()
        
        context = (
            core
            .with_columns([
                # Time pressure: how much time left? (0=none, 1=full game = 3600s)
                (pl.col('game_seconds_remaining') / 3600.0)
                .alias('time_remaining_normalized'),
                
                # Is it late game? (last 5 minutes = 300 seconds)
                (pl.col('game_seconds_remaining') < 300).cast(pl.Int32)
                .alias('is_late_game'),
                
                # Is it close? (within 1 goal)
                (pl.col('score_differential').abs() <= 1).cast(pl.Int32)
                .alias('is_close_game'),
                
                # Period effects
                (pl.col('period') == 1).cast(pl.Int32).alias('is_period_1'),
                (pl.col('period') == 2).cast(pl.Int32).alias('is_period_2'),
                (pl.col('period') >= 3).cast(pl.Int32).alias('is_period_3plus'),
                
                # Game phase (0=start, 1=end of regulation at 3600s)
                # Approximate: time_elapsed / 3600 (rough estimate)
                pl.when(pl.col('time_elapsed') > 0)
                .then(pl.col('time_elapsed') / 3600.0)
                .otherwise(0.0)
                .alias('game_phase'),
            ])
        )
        
        return context
    
    def interaction_features(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Interaction features: multiplicative effects.
        
        Creates:
        - score_time_interaction: score × time (protect leads late)
        - xg_time_interaction: xG advantage × time
        - is_clutch_moment: late game × close game
        - powerplay_score_interaction: power play × score
        """
        context = self.context_features()
        
        interactions = (
            context
            .with_columns([
                # Score × Time remaining
                (
                    pl.col('score_differential') * 
                    pl.col('time_remaining_normalized')
                ).alias('score_time_interaction'),
                
                # xG advantage × Time
                (
                    pl.col('xg_differential') * 
                    pl.col('time_remaining_normalized')
                ).alias('xg_time_interaction'),
                
                # Late game × Close game (clutch moments)
                (
                    pl.col('is_late_game') * 
                    pl.col('is_close_game')
                ).alias('is_clutch_moment'),
                
                # Power play × Score (power play more dangerous when ahead)
                (
                    pl.col('is_power_play').cast(pl.Int32) * 
                    pl.col('score_differential')
                ).alias('powerplay_score_interaction'),
            ])
        )
        
        return interactions
    
    def engineer_all_features(self) -> 'FeatureEngineer':
        """Execute full feature engineering pipeline."""
        print("🔨 Engineering features...")
        print("   - Core features (score, time, xG)")
        print("   - Context features (game phase, situation)")
        print("   - Interaction features (multiplicative effects)")
        
        self.features = self.interaction_features()
        print("   ✅ Feature engineering complete")
        
        return self
    
    def create_train_test_split(
        self, 
        split_season: int = None
    ) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Union[pl.LazyFrame, pl.DataFrame]]:
        """
        Temporal train/test split (avoid data leakage).
        
        Split on season boundary:
        - Train: seasons before split_season
        - Test: seasons >= split_season
        """
        split_season = split_season or self.config['train_test_split_season']
        
        if self.features is None:
            raise ValueError("Must call engineer_all_features() first")
        
        print(f"📊 Creating temporal train/test split at season {split_season}")
        
        train = self.features.filter(pl.col('season') < split_season)
        test = self.features.filter(pl.col('season') >= split_season)
        
        # Get counts
        train_count = train.collect().height if hasattr(train, 'collect') else len(train)
        test_count = test.collect().height if hasattr(test, 'collect') else len(test)
        
        print(f"   Train: {train_count:,} rows (seasons < {split_season})")
        print(f"   Test:  {test_count:,} rows (seasons >= {split_season})")
        
        return train, test
    
    def export_features(
        self,
        output_dir: Union[Path, str],
        format: str = 'csv'
    ) -> Dict[str, Path]:
        """Export train/test features to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train, test = self.create_train_test_split()
        
        print(f"💾 Exporting features to {output_dir}")
        
        # Collect if lazy
        if hasattr(train, 'collect'):
            train = train.collect()
            test = test.collect()
        
        # Export
        if format == 'csv':
            train_path = output_dir / 'features_train.csv'
            test_path = output_dir / 'features_test.csv'
            train.write_csv(train_path)
            test.write_csv(test_path)
        else:  # parquet
            train_path = output_dir / 'features_train.parquet'
            test_path = output_dir / 'features_test.parquet'
            train.write_parquet(train_path)
            test.write_parquet(test_path)
        
        print(f"   ✅ Train: {train_path}")
        print(f"   ✅ Test:  {test_path}")
        
        return {
            'train': train_path,
            'test': test_path,
            'train_df': train,
            'test_df': test
        }
    
    def compute_statistics(self) -> Dict:
        """Compute feature statistics for analysis."""
        if self.features is None:
            raise ValueError("Must call engineer_all_features() first")
        
        print("📈 Computing feature statistics...")
        
        # Collect if lazy
        features = (self.features.collect() 
                   if hasattr(self.features, 'collect') 
                   else self.features)
        
        stats = {}
        for col in features.columns:
            col_data = features[col]
            
            try:
                stats[col] = {
                    'dtype': str(col_data.dtype),
                    'null_count': col_data.null_count(),
                    'null_pct': round(100 * col_data.null_count() / len(col_data), 2),
                }
                
                # Numeric stats
                if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8, pl.Int16]:
                    stats[col].update({
                        'mean': round(float(col_data.mean()), 4) if col_data.mean() is not None else None,
                        'std': round(float(col_data.std()), 4) if col_data.std() is not None else None,
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                    })
            except:
                pass
        
        self.feature_stats = stats
        print(f"   ✅ Computed stats for {len(stats)} columns")
        
        return stats
    
    def save_statistics(self, output_path: Union[Path, str]):
        """Save feature statistics to JSON."""
        output_path = Path(output_path)
        
        stats_doc = {
            'timestamp': datetime.now().isoformat(),
            'description': 'Phase 2 Feature Statistics',
            'features': self.feature_stats
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats_doc, f, indent=2)
        
        print(f"💾 Statistics saved to {output_path}")
    
    def validate_features(self) -> Dict[str, bool]:
        """Validate engineered features."""
        if self.features is None:
            raise ValueError("Must call engineer_all_features() first")
        
        print("✅ Validating features...")
        features = (self.features.collect() 
                   if hasattr(self.features, 'collect') 
                   else self.features)
        
        results = {}
        
        # Check 1: Has rows
        results['has_rows'] = len(features) > 0
        print(f"   - Has rows: {results['has_rows']} ({len(features):,})")
        
        # Check 2: No critical nulls
        critical_cols = ['game_id', 'target_home_win', 'score_differential']
        has_critical_nulls = any(
            features[col].null_count() > 0 
            for col in critical_cols if col in features.columns
        )
        results['no_critical_nulls'] = not has_critical_nulls
        print(f"   - No critical nulls: {results['no_critical_nulls']}")
        
        # Check 3: Target balance
        target = features['target_home_win']
        home_win_pct = round(100 * target.sum() / len(target), 1)
        results['balanced_target'] = 0.45 <= (home_win_pct / 100) <= 0.60
        print(f"   - Target balance (home %): {home_win_pct}% - {results['balanced_target']}")
        
        # Check 4: Unique games
        n_games = features['game_id'].n_unique()
        results['expected_game_count'] = n_games >= 1000  # At least 1000 games
        print(f"   - Game count: {n_games} games")
        
        return results


def quick_feature_engineering(
    input_path: Union[Path, str],
    output_dir: Union[Path, str],
    split_season: int = 2024,
    export_format: str = 'csv'
) -> Dict:
    """One-liner: Full feature engineering pipeline."""
    engineer = (
        FeatureEngineer(input_path)
        .load_data(lazy=True)
        .engineer_all_features()
    )
    
    # Validate
    validation = engineer.validate_features()
    if not all(validation.values()):
        print("⚠️  Validation warnings detected")
    
    # Compute stats
    stats = engineer.compute_statistics()
    
    # Export
    exports = engineer.export_features(output_dir, format=export_format)
    
    # Save stats
    stats_path = Path(output_dir) / 'feature_statistics.json'
    engineer.save_statistics(stats_path)
    
    return {
        'validation': validation,
        'statistics': stats,
        'exports': exports,
        'engineer': engineer
    }


if __name__ == '__main__':
    # Example usage
    results = quick_feature_engineering(
        'data/processed/game_states.parquet',
        'data/processed',
        split_season=2024
    )
    print("\n✅ Phase 2 Complete!")
    print(f"   Train: {results['exports']['train']}")
    print(f"   Test:  {results['exports']['test']}")