"""
Phase 2: Main Orchestration Script
===================================

One-command execution of complete feature engineering pipeline.

Usage:
    python src/features/main.py

Or with custom paths:
    python src/features/main.py \
        --input data/processed/game_states.parquet \
        --output data/processed \
        --split-season 2024 \
        --format csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import DATA_PROCESSED, RESULTS_PHASE2, PROJECT_ROOT, ensure_directories

# Import feature engineer
from src.features.feature_engineer import FeatureEngineer, quick_feature_engineering


def setup_paths(config: dict = None) -> dict:
    """
    Initialize and validate all paths.
    
    Returns:
        Dict with validated paths
    """
    config = config or {}
    
    ensure_directories()
    
    input_file = config.get('input') or (DATA_PROCESSED / 'game_states.parquet')
    output_dir = config.get('output') or DATA_PROCESSED
    
    # Validate input exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    RESULTS_PHASE2.mkdir(parents=True, exist_ok=True)
    
    paths = {
        'project_root': PROJECT_ROOT,
        'input_file': Path(input_file),
        'output_dir': Path(output_dir),
        'results_dir': RESULTS_PHASE2,
    }
    
    return paths


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_summary(results: dict, paths: dict):
    """Print execution summary."""
    print_header("SUMMARY")
    
    validation = results.get('validation', {})
    exports = results.get('exports', {})
    
    # Validation summary
    print("\n✅ Validation Results:")
    for check, passed in validation.items():
        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] {check}")
    
    # File summary
    print("\n📁 Output Files:")
    for key in ['train', 'test']:
        if key in exports:
            path = exports[key]
            size_mb = path.stat().st_size / 1e6
            print(f"   {path.name} ({size_mb:.1f} MB)")
    
    print("\n✅ Phase 2 Complete! Ready for Phase 3 (Baseline Models)")


def main(args=None):
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Phase 2: Feature Engineering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/features/main.py
  python src/features/main.py --input data/processed/game_states.parquet --output data/processed
  python src/features/main.py --split-season 2023 --format parquet
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=str(DATA_PROCESSED / 'game_states.parquet'),
        help='Path to game_states.parquet'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=str(DATA_PROCESSED),
        help='Output directory for feature files'
    )
    
    parser.add_argument(
        '--split-season',
        type=int,
        default=2024,
        help='Season to use for train/test split (train < split, test >= split)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='Export format (csv or parquet)'
    )
    
    parser.add_argument(
        '--lazy',
        action='store_true',
        default=True,
        help='Use lazy evaluation in Polars (default: True)'
    )
    
    args = parser.parse_args(args)
    
    # Setup
    print_header("PHASE 2: FEATURE ENGINEERING WITH POLARS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Validate paths
    print("\n Setting up paths...")
    try:
        paths = setup_paths({
            'input': args.input,
            'output': args.output
        })
        print(f"    Input:  {paths['input_file']}")
        print(f"    Output: {paths['output_dir']}")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return 1
    
    # Execute pipeline
    print_header("EXECUTING PIPELINE")
    
    try:
        # Load data
        print("\n1️⃣  Loading game states...")
        engineer = FeatureEngineer(paths['input_file'])
        engineer.load_data(lazy=args.lazy)
        
        # Engineer features
        print("\n2️⃣  Engineering features...")
        engineer.engineer_all_features()
        
        # Validate
        print("\n3️⃣  Validating features...")
        validation = engineer.validate_features()
        
        # Compute statistics
        print("\n4️⃣  Computing statistics...")
        stats = engineer.compute_statistics()
        
        # Export features
        print(f"\n5️⃣  Exporting features ({args.format})...")
        exports = engineer.export_features(
            paths['output_dir'],
            format=args.format
        )
        
        # Save statistics
        stats_path = paths['output_dir'] / 'feature_statistics.json'
        engineer.save_statistics(stats_path)
        
        # Print summary
        results = {
            'validation': validation,
            'statistics': stats,
            'exports': exports
        }
        print_summary(results, paths)
        
        return 0
        
    except Exception as e:
        print(f"\n Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())