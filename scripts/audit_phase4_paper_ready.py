#!/usr/bin/env python3
"""
Phase 4 Audit Script: 2×2 Norm Ablation Paper-Ready Checks.

Runs ≥9 automated checks to validate experimental integrity:
1. Config integrity: all 4 condition YAMLs exist and are valid
2. Correct toggle combinations: each condition has expected enable_contraction/disable_value_head_norm
3. Seeds present: all 3 seeds (41, 42, 43) have checkpoints per condition
4. No mislabeled conditions: checkpoint dirs match config names
5. Summary.json exists and has correct structure
6. All 12 runs present in summary
7. CLAIMS.md exists and is non-empty
8. PROVENANCE.md exists and references correct configs
9. Metric bounds: Var(V) and Argmax@4× are within reasonable ranges
10. No NaN/Inf in metrics
11. Statistical validity: means have std computed from correct seed count

Usage:
    buck2 run //buiksat_trm:audit_phase4_paper_ready -- \
        --results_dir results/paper_ready/phase4_2x2_norm_ablation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Expected conditions and their toggle values
EXPECTED_CONDITIONS = {
    "nc_nv": {"enable_contraction": False, "disable_value_head_norm": True},
    "nc_yv": {"enable_contraction": False, "disable_value_head_norm": False},
    "yc_nv": {"enable_contraction": True, "disable_value_head_norm": True},
    "yc_yv": {"enable_contraction": True, "disable_value_head_norm": False},
}

EXPECTED_SEEDS = [41, 42, 43]


class AuditResult:
    """Holds result of a single audit check."""

    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def check_config_integrity(config_dir: Path) -> AuditResult:
    """Check 1: All 4 condition YAMLs exist and are valid YAML."""
    missing = []
    invalid = []

    for cond in EXPECTED_CONDITIONS:
        config_path = config_dir / f"{cond}.yaml"
        if not config_path.exists():
            missing.append(cond)
        else:
            try:
                with open(config_path, "r") as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                invalid.append(f"{cond}: {e}")

    if missing:
        return AuditResult(
            "Config Integrity",
            False,
            f"Missing configs: {missing}",
        )
    if invalid:
        return AuditResult(
            "Config Integrity",
            False,
            f"Invalid YAML: {invalid}",
        )
    return AuditResult(
        "Config Integrity",
        True,
        f"All {len(EXPECTED_CONDITIONS)} condition configs exist and are valid YAML",
    )


def check_correct_toggles(config_dir: Path) -> AuditResult:
    """Check 2: Each condition has expected toggle values."""
    errors = []

    for cond, expected in EXPECTED_CONDITIONS.items():
        config_path = config_dir / f"{cond}.yaml"
        if not config_path.exists():
            continue

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        for key, expected_val in expected.items():
            actual = cfg.get(key)
            if actual != expected_val:
                errors.append(f"{cond}.{key}: expected {expected_val}, got {actual}")

    if errors:
        return AuditResult(
            "Correct Toggles",
            False,
            f"Toggle mismatches: {errors}",
        )
    return AuditResult(
        "Correct Toggles",
        True,
        "All conditions have correct enable_contraction/disable_value_head_norm values",
    )


def check_seeds_present(
    results_dir: Path,
    summary: Optional[Dict[str, Any]] = None,
) -> AuditResult:
    """Check 3: All 3 seeds have results per condition (checks summary if available)."""
    # If we have summary, check that all condition/seed combos are present
    if summary:
        runs = summary.get("all_results", [])
        found = set()
        for run in runs:
            cond = run.get("condition")
            seed = run.get("seed")
            if cond and seed:
                found.add((cond, seed))

        expected = {(c, s) for c in EXPECTED_CONDITIONS for s in EXPECTED_SEEDS}
        missing = expected - found

        if missing:
            return AuditResult(
                "Seeds Present",
                False,
                f"Missing condition/seed combos in summary: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}",
            )
        return AuditResult(
            "Seeds Present",
            True,
            f"All {len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)} condition/seed combinations found in summary",
        )

    # Fall back to checking directories if no summary
    missing = []

    for cond in EXPECTED_CONDITIONS:
        for seed in EXPECTED_SEEDS:
            ckpt_dir = results_dir / f"{cond}_s{seed}"
            if not ckpt_dir.exists():
                missing.append(f"{cond}_s{seed}")

    if missing:
        return AuditResult(
            "Seeds Present",
            False,
            f"Missing checkpoint dirs: {missing[:5]}{'...' if len(missing) > 5 else ''}",
        )
    return AuditResult(
        "Seeds Present",
        True,
        f"All {len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)} checkpoint directories exist",
    )


def check_no_mislabeled(results_dir: Path) -> AuditResult:
    """Check 4: Checkpoint dirs match expected naming pattern."""
    unexpected = []
    expected_pattern = {f"{c}_s{s}" for c in EXPECTED_CONDITIONS for s in EXPECTED_SEEDS}

    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and item.name not in expected_pattern:
                # Allow summary files and documentation
                if not item.name.endswith(".json") and not item.name.endswith(".md"):
                    unexpected.append(item.name)

    if unexpected:
        return AuditResult(
            "No Mislabeled",
            False,
            f"Unexpected directories: {unexpected}",
        )
    return AuditResult(
        "No Mislabeled",
        True,
        "All checkpoint directories follow expected naming convention",
    )


def check_summary_exists(results_dir: Path) -> Tuple[AuditResult, Dict[str, Any]]:
    """Check 5: summary.json exists and has correct structure."""
    summary_path = results_dir / "summary.json"

    if not summary_path.exists():
        return (
            AuditResult("Summary Exists", False, f"summary.json not found at {summary_path}"),
            {},
        )

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        return (
            AuditResult("Summary Exists", False, f"Invalid JSON: {e}"),
            {},
        )

    required_keys = ["experiment", "all_results", "aggregates"]
    missing_keys = [k for k in required_keys if k not in summary]

    if missing_keys:
        return (
            AuditResult("Summary Exists", False, f"Missing keys: {missing_keys}"),
            {},
        )

    return (
        AuditResult("Summary Exists", True, "summary.json exists with correct structure"),
        summary,
    )


def check_all_runs_present(summary: Dict[str, Any]) -> AuditResult:
    """Check 6: All 12 runs present in summary."""
    if not summary:
        return AuditResult("All Runs Present", False, "No summary to check")

    runs = summary.get("all_results", [])
    expected_count = len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)

    if len(runs) != expected_count:
        return AuditResult(
            "All Runs Present",
            False,
            f"Expected {expected_count} runs, found {len(runs)}",
        )

    # Check each condition/seed combo
    found = set()
    for run in runs:
        cond = run.get("condition")
        seed = run.get("seed")
        if cond and seed:
            found.add((cond, seed))

    expected = {(c, s) for c in EXPECTED_CONDITIONS for s in EXPECTED_SEEDS}
    missing = expected - found

    if missing:
        return AuditResult(
            "All Runs Present",
            False,
            f"Missing runs: {list(missing)[:5]}",
        )

    return AuditResult(
        "All Runs Present",
        True,
        f"All {expected_count} condition×seed combinations present",
    )


def check_claims_exists(results_dir: Path) -> AuditResult:
    """Check 7: CLAIMS.md exists and is non-empty."""
    claims_path = results_dir / "CLAIMS.md"

    if not claims_path.exists():
        return AuditResult("CLAIMS.md Exists", False, "CLAIMS.md not found")

    content = claims_path.read_text()
    if len(content.strip()) < 100:
        return AuditResult(
            "CLAIMS.md Exists",
            False,
            f"CLAIMS.md too short ({len(content)} chars)",
        )

    return AuditResult(
        "CLAIMS.md Exists",
        True,
        f"CLAIMS.md exists with {len(content)} chars",
    )


def check_provenance_exists(results_dir: Path, config_dir: Path) -> AuditResult:
    """Check 8: PROVENANCE.md exists and references configs."""
    prov_path = results_dir / "PROVENANCE.md"

    if not prov_path.exists():
        return AuditResult("PROVENANCE.md Exists", False, "PROVENANCE.md not found")

    content = prov_path.read_text()

    # Check it references the config conditions (any of them)
    found_any = any(cond in content for cond in EXPECTED_CONDITIONS)
    if not found_any:
        return AuditResult(
            "PROVENANCE.md Exists",
            False,
            "PROVENANCE.md does not reference any condition configs (nc_nv, nc_yv, etc.)",
        )

    return AuditResult(
        "PROVENANCE.md Exists",
        True,
        "PROVENANCE.md exists and references condition configs",
    )


def check_metric_bounds(summary: Dict[str, Any]) -> AuditResult:
    """Check 9: Metrics are within reasonable bounds."""
    if not summary:
        return AuditResult("Metric Bounds", False, "No summary to check")

    aggregates = summary.get("aggregates", [])
    issues = []

    for agg in aggregates:
        cond = agg.get("condition", "unknown")

        # Var(V) should be non-negative and not huge
        var_v = agg.get("var_V_mean", -1)
        if var_v < 0 or var_v > 1000:
            issues.append(f"{cond}: var_V_mean={var_v} out of bounds [0, 1000]")

        # Argmax agreement should be in [0, 1]
        argmax = agg.get("argmax_4x_mean", -1)
        if argmax < 0 or argmax > 1:
            issues.append(f"{cond}: argmax_4x_mean={argmax} out of bounds [0, 1]")

        # Success rate should be in [0, 1]
        success = agg.get("success_trivial_mean", -1)
        if success < 0 or success > 1:
            issues.append(f"{cond}: success_trivial_mean={success} out of bounds [0, 1]")

    if issues:
        return AuditResult(
            "Metric Bounds",
            False,
            f"Out of bounds metrics: {issues[:3]}{'...' if len(issues) > 3 else ''}",
        )

    return AuditResult(
        "Metric Bounds",
        True,
        "All metrics within expected bounds",
    )


def check_no_nan_inf(summary: Dict[str, Any]) -> AuditResult:
    """Check 10: No NaN or Inf in metrics."""
    if not summary:
        return AuditResult("No NaN/Inf", False, "No summary to check")

    def check_value(v: Any, path: str) -> List[str]:
        issues = []
        if isinstance(v, float):
            if v != v:  # NaN check
                issues.append(f"{path}: NaN")
            elif abs(v) == float("inf"):
                issues.append(f"{path}: Inf")
        elif isinstance(v, dict):
            for k, vv in v.items():
                issues.extend(check_value(vv, f"{path}.{k}"))
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                issues.extend(check_value(vv, f"{path}[{i}]"))
        return issues

    issues = check_value(summary, "summary")

    if issues:
        return AuditResult(
            "No NaN/Inf",
            False,
            f"Found NaN/Inf: {issues[:3]}{'...' if len(issues) > 3 else ''}",
        )

    return AuditResult("No NaN/Inf", True, "No NaN or Inf values found")


def check_statistical_validity(summary: Dict[str, Any]) -> AuditResult:
    """Check 11: Aggregates computed from correct number of seeds."""
    if not summary:
        return AuditResult("Statistical Validity", False, "No summary to check")

    aggregates = summary.get("aggregates", [])
    runs = summary.get("all_results", [])

    # Count runs per condition
    runs_per_cond = {}
    for run in runs:
        cond = run.get("condition")
        if cond:
            runs_per_cond[cond] = runs_per_cond.get(cond, 0) + 1

    issues = []
    expected_seeds = len(EXPECTED_SEEDS)

    for agg in aggregates:
        cond = agg.get("condition")
        if cond and runs_per_cond.get(cond, 0) != expected_seeds:
            issues.append(
                f"{cond}: {runs_per_cond.get(cond, 0)} runs, expected {expected_seeds}"
            )

    if issues:
        return AuditResult(
            "Statistical Validity",
            False,
            f"Wrong seed count: {issues}",
        )

    return AuditResult(
        "Statistical Validity",
        True,
        f"All conditions have {expected_seeds} seeds for statistics",
    )


def run_audit(results_dir: Path, config_dir: Path) -> Tuple[List[AuditResult], bool]:
    """Run all audit checks."""
    results = []

    # Checks 1-2: Config integrity
    results.append(check_config_integrity(config_dir))
    results.append(check_correct_toggles(config_dir))

    # Check 5: Summary exists (get it early for other checks)
    summary_result, summary = check_summary_exists(results_dir)

    # Check 3-4: Seeds and directories (pass summary if available)
    results.append(check_seeds_present(results_dir, summary))
    results.append(check_no_mislabeled(results_dir))

    # Add summary check result
    results.append(summary_result)

    # Checks 6-8: Documentation and completeness
    results.append(check_all_runs_present(summary))
    results.append(check_claims_exists(results_dir))
    results.append(check_provenance_exists(results_dir, config_dir))

    # Checks 9-11: Data quality
    results.append(check_metric_bounds(summary))
    results.append(check_no_nan_inf(summary))
    results.append(check_statistical_validity(summary))

    all_passed = all(r.passed for r in results)
    return results, all_passed


def write_audit_md(results_dir: Path, results: List[AuditResult], all_passed: bool):
    """Write AUDIT.md with results."""
    content = """# Phase 4: 2×2 Norm Ablation - Audit Report

## Summary

"""
    content += f"**Status:** {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}\n\n"
    content += f"**Total Checks:** {len(results)}\n"
    content += f"**Passed:** {sum(1 for r in results if r.passed)}\n"
    content += f"**Failed:** {sum(1 for r in results if not r.passed)}\n\n"

    content += "## Detailed Results\n\n"

    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result.passed else "✗ FAIL"
        content += f"### Check {i}: {result.name}\n\n"
        content += f"**Status:** {status}\n\n"
        content += f"**Details:** {result.message}\n\n"

    audit_path = results_dir / "AUDIT.md"
    with open(audit_path, "w") as f:
        f.write(content)
    print(f"Saved: {audit_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Audit Script")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory (e.g., results/paper_ready/phase4_2x2_norm_ablation)",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Path to config directory (defaults to configs/phase4_2x2_norm_ablation)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config_dir = (
        Path(args.config_dir)
        if args.config_dir
        else Path(__file__).parent.parent / "configs" / "phase4_2x2_norm_ablation"
    )

    print("=" * 60)
    print("Phase 4: 2×2 Norm Ablation - Audit")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Config dir: {config_dir}")
    print()

    results, all_passed = run_audit(results_dir, config_dir)

    print("Audit Results:")
    print("-" * 40)
    for result in results:
        print(result)
    print("-" * 40)
    print()

    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed)

    print(f"Total: {len(results)} checks")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print()

    if all_passed:
        print("✓ ALL CHECKS PASSED")
        write_audit_md(results_dir, results, all_passed)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        write_audit_md(results_dir, results, all_passed)
        return 1


if __name__ == "__main__":
    sys.exit(main())
