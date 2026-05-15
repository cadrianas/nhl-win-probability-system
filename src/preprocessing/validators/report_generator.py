"""
Aggregate validation results into a single report.

Single responsibility: Generate clean validation reports.
"""

import json
from pathlib import Path
from typing import Dict, List


def generate_validation_report(
    checks: Dict[str, List[Dict]],
    stats: Dict,
    extra_analysis: Dict = None,
    output_path: Path = None,
) -> None:
    """
    Combine validation results into a clean JSON report.

    Args:
        checks: Dict of {check_name: [issues]}
        stats: Dict of summary statistics
        extra_analysis: Additional analysis results
        output_path: Where to save JSON
    """
    report = {
        "summary": {
            "total_games": stats.get("total_games", 0),
            "total_states": stats.get("total_states", 0),
            "home_win_rate": stats.get("home_win_rate", "N/A"),
            "away_win_rate": stats.get("away_win_rate", "N/A"),
            "regular_season_pct": stats.get("regular_season_pct", "N/A"),
            "playoff_ot_pct": stats.get("playoff_ot_pct", "N/A"),
            "rs_ot_pct": stats.get("rs_ot_pct", "N/A"),
        },
        "validation_results": {},
    }

    # Add each check
    for check_name, issues in checks.items():
        report["validation_results"][check_name] = {
            "passed": len(issues) == 0,
            "issues_found": len(issues),
            "details": issues[:10],  # Show first 10
        }

    # Add extra analysis if provided
    if extra_analysis:
        report["analysis"] = extra_analysis

    # Write
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved to: {output_path}")

    return report


def print_validation_summary(report: dict) -> None:
    """
    Pretty print validation report to console.

    Args:
        report: Report dict from generate_validation_report
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION SUMMARY")
    print("=" * 60)

    summary = report.get("summary", {})
    print(f"\nGames: {summary.get('total_games', 'N/A'):,}")
    print(f"States: {summary.get('total_states', 'N/A'):,}")
    print(f"Home Win Rate: {summary.get('home_win_rate', 'N/A')}")
    print(f"Regular Season: {summary.get('regular_season_pct', 'N/A')}")
    print(f"Playoff OT: {summary.get('playoff_ot_pct', 'N/A')}")
    print(f"RS OT: {summary.get('rs_ot_pct', 'N/A')}")

    print("\n" + "-" * 60)
    print("VALIDATION CHECKS")
    print("-" * 60)

    results = report.get("validation_results", {})
    for check_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status}: {check_name} ({result['issues_found']} issues)")

    print("\n" + "=" * 60)
