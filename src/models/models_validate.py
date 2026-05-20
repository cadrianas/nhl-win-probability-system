"""
Phase 3: Project Validation Script
Location: phase3_validate.py

Performs health checks:
1. Verifies Phase 2 outputs exist and are valid
2. Checks Python dependencies (sklearn, xgboost, lightgbm, etc.)
3. Validates feature data structure
4. Tests model imports
5. Reports on system readiness for Phase 3

Usage:
    python phase3_validate.py
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required Python packages are installed.
    
    Returns:
        Tuple: (all_ok: bool, missing_packages: list)
    """
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'polars': 'polars (optional, for faster CSV loading)',
    }
    
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            if import_name != 'polars':  # Polars is optional
                missing.append(package_name)
    
    return len(missing) == 0, missing


# ============================================================================
# DATA CHECKS
# ============================================================================

def check_phase2_outputs() -> Tuple[bool, List[str]]:
    """
    Check if Phase 2 feature matrices exist and are valid.
    
    Returns:
        Tuple: (all_ok: bool, issues: list)
    """
    issues = []
    
    train_path = Path("data/processed/features_train.csv")
    test_path = Path("data/processed/features_test.csv")
    
    # Check existence
    if not train_path.exists():
        issues.append(f"✗ Training features not found: {train_path}")
        print(f"✗ Training features not found: {train_path}")
    else:
        print(f"✓ Training features found: {train_path}")
    
    if not test_path.exists():
        issues.append(f"✗ Test features not found: {test_path}")
        print(f"✗ Test features not found: {test_path}")
    else:
        print(f"✓ Test features found: {test_path}")
    
    # Try to load and validate
    if train_path.exists() and test_path.exists():
        try:
            import pandas as pd
            
            train_df = pd.read_csv(train_path, nrows=100)  # Sample load
            test_df = pd.read_csv(test_path, nrows=100)
            
            print(f"✓ Training data shape: {train_df.shape}")
            print(f"✓ Test data shape: {test_df.shape}")
            
            # Check for target column
            if 'target_home_win' not in train_df.columns:
                issues.append("✗ Target column 'target_home_win' not found in training data")
                print("✗ Target column 'target_home_win' not found in training data")
            else:
                print("✓ Target column 'target_home_win' present")
            
            # Check for feature columns
            exclude_cols = {'game_id', 'shot_id', 'target_home_win'}
            feature_cols = [c for c in train_df.columns if c not in exclude_cols]
            print(f"✓ Number of features: {len(feature_cols)}")
            
            # Check for nulls
            null_count = train_df[feature_cols].isnull().sum().sum()
            if null_count > 0:
                issues.append(f"⚠ Warning: {null_count} null values in training features")
                print(f"⚠ Warning: {null_count} null values in training features")
            else:
                print("✓ No null values in features")
        
        except Exception as e:
            issues.append(f"✗ Error loading Phase 2 data: {e}")
            print(f"✗ Error loading Phase 2 data: {e}")
    
    return len(issues) == 0, issues


# ============================================================================
# CODE CHECKS
# ============================================================================

def check_phase3_modules() -> Tuple[bool, List[str]]:
    """
    Check if Phase 3 modules can be imported.
    
    Returns:
        Tuple: (all_ok: bool, issues: list)
    """
    issues = []
    
    modules_to_check = [
        ('baseline', 'Logistic Regression model'),
        ('xgboost_model', 'XGBoost model'),
        ('lightgbm_model', 'LightGBM model'),
        ('evaluate', 'Evaluation utilities'),
    ]
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"✓ {description} ({module_name}.py)")
        except ImportError as e:
            issues.append(f"✗ {description} not found: {module_name}.py")
            print(f"✗ {description} not found or import error: {e}")
    
    # Check phase3_main.py
    main_path = Path("phase3_main.py")
    if main_path.exists():
        print(f"✓ Main orchestrator found: {main_path}")
    else:
        issues.append("✗ Main orchestrator not found: phase3_main.py")
        print(f"✗ Main orchestrator not found: {main_path}")
    
    return len(issues) == 0, issues


# ============================================================================
# DIRECTORY CHECKS
# ============================================================================

def check_output_directories() -> Tuple[bool, List[str]]:
    """
    Check if output directories can be created.
    
    Returns:
        Tuple: (all_ok: bool, issues: list)
    """
    issues = []
    
    output_dir = Path("phase3_models")
    subdirs = [
        output_dir / "models",
        output_dir / "results",
        output_dir / "logs",
    ]
    
    for subdir in subdirs:
        try:
            subdir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory ready: {subdir}")
        except Exception as e:
            issues.append(f"✗ Cannot create output directory: {subdir}: {e}")
            print(f"✗ Cannot create output directory: {subdir}: {e}")
    
    return len(issues) == 0, issues


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main():
    """Run all validation checks."""
    print("\n" + "=" * 100)
    print("PHASE 3 VALIDATION")
    print("=" * 100 + "\n")
    
    all_checks_pass = True
    
    # Check 1: Dependencies
    print("1. Checking Python Dependencies...")
    print("-" * 100)
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"\n⚠ Missing packages: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
        all_checks_pass = False
    print()
    
    # Check 2: Phase 2 outputs
    print("2. Checking Phase 2 Outputs...")
    print("-" * 100)
    phase2_ok, phase2_issues = check_phase2_outputs()
    if not phase2_ok:
        print(f"\n⚠ Phase 2 data issues:")
        for issue in phase2_issues:
            print(f"  {issue}")
        all_checks_pass = False
    print()
    
    # Check 3: Phase 3 modules
    print("3. Checking Phase 3 Code Modules...")
    print("-" * 100)
    code_ok, code_issues = check_phase3_modules()
    if not code_ok:
        print(f"\n⚠ Missing code modules:")
        for issue in code_issues:
            print(f"  {issue}")
        all_checks_pass = False
    print()
    
    # Check 4: Output directories
    print("4. Checking Output Directories...")
    print("-" * 100)
    dirs_ok, dir_issues = check_output_directories()
    if not dirs_ok:
        print(f"\n⚠ Directory issues:")
        for issue in dir_issues:
            print(f"  {issue}")
        all_checks_pass = False
    print()
    
    # Summary
    print("=" * 100)
    if all_checks_pass:
        print("✓ ALL CHECKS PASSED - READY FOR PHASE 3")
        print("\nNext steps:")
        print("  1. python phase3_main.py          # Run Phase 3 training")
        print("  2. Check results in phase3_models/results/")
        print("=" * 100)
        return 0
    else:
        print("✗ SOME CHECKS FAILED - FIX ISSUES BEFORE RUNNING PHASE 3")
        print("\nCommon fixes:")
        print("  - pip install -r requirements.txt")
        print("  - Run Phase 2 first to generate features_train.csv and features_test.csv")
        print("  - Ensure baseline.py, xgboost_model.py, etc. are in project root")
        print("=" * 100)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit_code = main()
    sys.exit(exit_code)