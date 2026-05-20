"""
Phase 2: Feature Validation Module
===================================

Comprehensive validation checks for engineered features.

Validates:
- Data completeness
- Value ranges
- Target balance
- Feature correlations
- Temporal properties
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
from datetime import datetime


class FeatureValidator:
    """
    Comprehensive feature validation.
    
    Checks:
    - Data integrity (nulls, dtypes, row counts)
    - Value ranges (reasonable min/max for each feature)
    - Target balance (home win distribution)
    - Temporal properties (no time reversals, monotonic ordering)
    - Feature diversity (avoid constant/near-constant features)
    """
    
    def __init__(self, features_path: Union[Path, str]):
        """
        Initialize validator.
        
        Args:
            features_path: Path to features CSV/parquet file
        """
        self.features_path = Path(features_path)
        self.features = None
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
    def load_features(self) -> 'FeatureValidator':
        """Load features from disk."""
        if self.features_path.suffix == '.parquet':
            self.features = pl.read_parquet(self.features_path)
        else:
            self.features = pl.read_csv(self.features_path)
        
        print(f"📥 Loaded {len(self.features):,} rows, {len(self.features.columns)} columns")
        return self
    
    def check_data_integrity(self) -> Dict[str, bool]:
        """
        Check basic data integrity.
        
        Returns:
            Dict with pass/fail for each check
        """
        print("\n✓ Checking data integrity...")
        checks = {}
        
        # Check 1: Non-empty
        checks['non_empty'] = len(self.features) > 0
        if not checks['non_empty']:
            self.errors.append("Dataset is empty!")
        
        # Check 2: Expected columns
        required_cols = [
            'game_id', 'target_home_win', 'score_differential', 
            'xg_differential', 'shot_differential'
        ]
        missing = [c for c in required_cols if c not in self.features.columns]
        checks['has_required_columns'] = len(missing) == 0
        if missing:
            self.errors.append(f"Missing columns: {missing}")
        
        # Check 3: No critical nulls
        critical_nulls = {
            col: self.features[col].null_count() 
            for col in required_cols 
            if col in self.features.columns
        }
        checks['no_critical_nulls'] = all(n == 0 for n in critical_nulls.values())
        if not checks['no_critical_nulls']:
            self.errors.append(f"Nulls in critical columns: {critical_nulls}")
        
        # Check 4: Correct dtypes
        float_cols = ['xg_differential', 'shot_momentum_ratio_5min']
        int_cols = ['game_id', 'score_differential', 'target_home_win']
        
        dtype_ok = True
        for col in float_cols:
            if col in self.features.columns:
                if self.features[col].dtype not in [pl.Float32, pl.Float64]:
                    dtype_ok = False
        
        checks['correct_dtypes'] = dtype_ok
        if not dtype_ok:
            self.warnings.append("Some float columns have wrong dtype")
        
        # Print results
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
        
        self.validation_results['data_integrity'] = checks
        return checks
    
    def check_value_ranges(self) -> Dict[str, bool]:
        """
        Check feature values are in reasonable ranges.
        
        Returns:
            Dict with pass/fail for each range check
        """
        print("\n✓ Checking value ranges...")
        checks = {}
        
        # Score differential: -6 to +6 (max goal differentials)
        sd_min = self.features['score_differential'].min()
        sd_max = self.features['score_differential'].max()
        checks['score_diff_range'] = -6 <= sd_min and sd_max <= 6
        if not checks['score_diff_range']:
            self.warnings.append(
                f"Score differential range [{sd_min}, {sd_max}] is unusual"
            )
        
        # xG differential: should be bounded
        xg_min = self.features['xg_differential'].min()
        xg_max = self.features['xg_differential'].max()
        checks['xg_diff_range'] = -15 <= xg_min and xg_max <= 15
        if not checks['xg_diff_range']:
            self.warnings.append(
                f"xG differential range [{xg_min}, {xg_max}] is unusual"
            )
        
        # Shot differential: 0-100 range (max ~200 shots per team)
        shot_min = self.features['shot_differential'].min()
        shot_max = self.features['shot_differential'].max()
        checks['shot_diff_range'] = -100 <= shot_min and shot_max <= 100
        if not checks['shot_diff_range']:
            self.warnings.append(
                f"Shot differential range [{shot_min}, {shot_max}] is unusual"
            )
        
        # Game seconds remaining: 0-3600 range
        if 'game_seconds_remaining' in self.features.columns:
            gsr_min = self.features['game_seconds_remaining'].min()
            gsr_max = self.features['game_seconds_remaining'].max()
            checks['time_range'] = 0 <= gsr_min and gsr_max <= 5400
            if not checks['time_range']:
                self.warnings.append(
                    f"Time range [{gsr_min}, {gsr_max}] is unusual (expected 0-5400)"
                )
        
        # Momentum ratios: bounded by physics
        if 'shot_momentum_ratio_5min' in self.features.columns:
            mr = self.features['shot_momentum_ratio_5min']
            mr_min = mr.min()
            mr_max = mr.max()
            # Ratio of (a+0.1)/(b+0.1) should be reasonable
            checks['momentum_ratio_range'] = 0.1 <= mr_min and mr_max <= 10
            if not checks['momentum_ratio_range']:
                self.warnings.append(
                    f"Momentum ratio range [{mr_min}, {mr_max}] is unusual"
                )
        
        # Print results
        for check, passed in checks.items():
            status = "✅" if passed else "⚠️ " if not passed else "✅"
            print(f"   {status} {check}")
        
        self.validation_results['value_ranges'] = checks
        return checks
    
    def check_target_balance(self) -> Dict[str, bool]:
        """
        Check target variable (home_win) balance.
        
        Home teams win ~54% of NHL games historically.
        Balance should be: 50-60% home wins.
        
        Returns:
            Dict with balance metrics
        """
        print("\n✓ Checking target balance...")
        
        target = self.features['target_home_win']
        n_total = len(target)
        n_home_wins = int(target.sum())
        n_away_wins = n_total - n_home_wins
        
        home_win_pct = 100 * n_home_wins / n_total
        away_win_pct = 100 * n_away_wins / n_total
        
        is_balanced = 48 <= home_win_pct <= 58
        
        print(f"   Total games: {n_total:,}")
        print(f"   Home wins:   {n_home_wins:,} ({home_win_pct:.1f}%)")
        print(f"   Away wins:   {n_away_wins:,} ({away_win_pct:.1f}%)")
        print(f"   Balanced:    {'✅' if is_balanced else '⚠️ '} ({is_balanced})")
        
        if not is_balanced and home_win_pct > 60:
            self.warnings.append(
                f"Imbalanced target: {home_win_pct:.1f}% home wins "
                "(expected ~54%)"
            )
        
        checks = {
            'balanced': is_balanced,
            'home_win_pct': round(home_win_pct, 1),
            'away_win_pct': round(away_win_pct, 1),
        }
        
        self.validation_results['target_balance'] = checks
        return checks
    
    def check_temporal_properties(self) -> Dict[str, bool]:
        """
        Check temporal ordering within games.
        
        For each game, time should be monotonic (or equal).
        
        Returns:
            Dict with temporal checks
        """
        print("\n✓ Checking temporal properties...")
        checks = {}
        
        if 'game_id' not in self.features.columns or 'time_elapsed' not in self.features.columns:
            self.warnings.append("Missing game_id or time_elapsed for temporal checks")
            return checks
        
        # Sort by game and time (should already be sorted)
        sorted_features = self.features.sort(['game_id', 'time_elapsed'])
        
        # Check if already sorted
        is_sorted = (self.features == sorted_features).all()
        checks['is_sorted_by_time'] = is_sorted
        
        # Check for time reversals within games
        time_reversals = 0
        for game_id in self.features['game_id'].unique():
            game_data = self.features.filter(pl.col('game_id') == game_id)
            times = game_data['time_elapsed'].to_list()
            for i in range(1, len(times)):
                if times[i] < times[i-1]:
                    time_reversals += 1
        
        checks['no_time_reversals'] = time_reversals == 0
        if time_reversals > 0:
            self.errors.append(f"Found {time_reversals} time reversals!")
        
        # Check game count matches expected (~1,419 games)
        n_games = self.features['game_id'].n_unique()
        checks['expected_game_count'] = 1300 <= n_games <= 1500
        print(f"   Total unique games: {n_games}")
        
        if not checks['expected_game_count']:
            self.warnings.append(f"Unusual game count: {n_games} (expected ~1,419)")
        
        # Print results
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
        
        self.validation_results['temporal'] = checks
        return checks
    
    def check_feature_variance(self) -> Dict[str, float]:
        """
        Check features have reasonable variance.
        
        Avoid constant or near-constant features that don't help models.
        
        Returns:
            Dict with variance metrics
        """
        print("\n✓ Checking feature variance...")
        
        variance_dict = {}
        low_variance = []
        
        numeric_cols = [
            col for col in self.features.columns
            if self.features[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
        
        for col in numeric_cols:
            try:
                var = float(self.features[col].var())
                variance_dict[col] = round(var, 4)
                
                # Flag very low variance
                if var < 0.01:
                    low_variance.append((col, var))
            except:
                pass
        
        if low_variance:
            self.warnings.append(
                f"Low variance features: {[c for c, _ in low_variance]}"
            )
        
        print(f"   Checked {len(variance_dict)} numeric columns")
        print(f"   Low variance (<0.01): {len(low_variance)}")
        
        self.validation_results['variance'] = variance_dict
        return variance_dict
    
    def generate_report(self) -> Dict:
        """
        Run all validation checks and generate report.
        
        Returns:
            Dict with all validation results
        """
        print("=" * 70)
        print("  FEATURE VALIDATION REPORT")
        print("=" * 70)
        print(f"File: {self.features_path}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Run all checks
        self.check_data_integrity()
        self.check_value_ranges()
        self.check_target_balance()
        self.check_temporal_properties()
        self.check_feature_variance()
        
        # Summary
        print("\n" + "=" * 70)
        print("  VALIDATION SUMMARY")
        print("=" * 70)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        else:
            print("\n✅ No critical errors")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        else:
            print("\n✅ No warnings")
        
        is_valid = len(self.errors) == 0
        print(f"\n{'✅ VALID' if is_valid else '❌ INVALID'}")
        
        return {
            'is_valid': is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'detailed_results': self.validation_results
        }
    
    def save_report(self, output_path: Union[Path, str]):
        """Save validation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'features_file': str(self.features_path),
            'is_valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'validation_results': self.validation_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to {output_path}")


def validate_features_file(
    features_path: Union[Path, str],
    output_report: Union[Path, str, None] = None
) -> Dict:
    """
    One-liner: validate features file.
    
    Usage:
        result = validate_features_file('data/processed/features_train.csv')
    
    Returns:
        Dict with validation results
    """
    validator = FeatureValidator(features_path).load_features()
    result = validator.generate_report()
    
    if output_report:
        validator.save_report(output_report)
    
    return result


if __name__ == '__main__':
    # Example: validate features_train.csv
    import sys
    
    if len(sys.argv) > 1:
        features_file = sys.argv[1]
    else:
        features_file = 'data/processed/features_train.csv'
    
    result = validate_features_file(
        features_file,
        output_report='results/phase2/validation_report.json'
    )