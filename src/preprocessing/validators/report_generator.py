"""
Report generation for validation results (FIXED VERSION).

Handles numpy int64 and other non-JSON-serializable types.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Any object (dict, list, numpy type, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_validation_report(
    checks: Dict,
    stats: Dict,
    extra_analysis: Dict,
    report_path: Path
) -> Dict:
    """
    Generate and save validation report.
    
    Args:
        checks: Dict of validation check results
        stats: Summary statistics
        extra_analysis: Extra analysis results
        report_path: Path to save JSON report
        
    Returns:
        Report dict
    """
    report = {
        "validation_checks": checks,
        "summary_stats": stats,
        "extra_analysis": extra_analysis,
    }
    
    # Convert all numpy types to native Python types
    report = convert_to_serializable(report)
    
    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    print(f"   Saved to: {report_path}")
    
    return report


def print_validation_summary(report: Dict) -> None:
    """
    Print human-readable validation summary.
    
    Args:
        report: Validation report dict
    """
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    checks = report.get("validation_checks", {})
    stats = report.get("summary_stats", {})
    
    # Print check results
    print("\nValidation Checks:")
    for check_name, result in checks.items():
        if isinstance(result, dict):
            status = result.get('status', 'UNKNOWN')
            issues = result.get('issues', [])
            
            if status == 'OK':
                print(f"  ✅ {check_name}: PASS")
            elif status == 'SKIPPED':
                print(f"  ⊘ {check_name}: SKIPPED")
            elif status == 'WARNING':
                print(f"  ⚠️  {check_name}: {len(issues)} issues")
            else:
                print(f"  ? {check_name}: {status}")
    
    # Print summary stats
    if stats:
        print("\nSummary Statistics:")
        for key, value in stats.items():
            # Format key nicely
            key_display = key.replace('_', ' ').title()
            print(f"  {key_display}: {value}")
    
    print("\n" + "=" * 60)