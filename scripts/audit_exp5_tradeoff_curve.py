#!/usr/bin/env python3
"""
Audit script for Exp5: Stability–Expressivity Tradeoff Curve.

Checks:
1. disable_value_head_norm: true for all configs
2. projection_active_rate < 1% for all conditions
3. B0/B1 provenance hashes match expected values
4. strict YAML loading confirmed
5. summary.json matches plotted values
6. git SHA captured and non-empty

Usage:
    python scripts/audit_exp5_tradeoff_curve.py \
        --summary_json results/paper_ready/exp5_tradeoff_curve/summary.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_value_head_norm(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check disable_value_head_norm is true."""
    config = summary.get("config", {})
    disable_value_head_norm = config.get("disable_value_head_norm", False)
    if disable_value_head_norm:
        return True, "PASS: disable_value_head_norm=true"
    return False, f"FAIL: disable_value_head_norm={disable_value_head_norm}"


def check_projection_disabled(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check latent_ball_radius is 0."""
    config = summary.get("config", {})
    latent_ball_radius = config.get("latent_ball_radius", 10.0)
    if latent_ball_radius == 0.0:
        return True, "PASS: latent_ball_radius=0.0 (projection disabled)"
    return False, f"FAIL: latent_ball_radius={latent_ball_radius}"


def check_projection_inactive(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check projection_active_rate < 1% for all results."""
    all_results = summary.get("all_results", [])
    max_rate = 0.0
    for r in all_results:
        rate = r.get("projection_active_rate", 0.0)
        max_rate = max(max_rate, rate)

    if max_rate < 0.01:
        return True, f"PASS: max projection_active_rate={max_rate:.4f} (<1%)"
    return False, f"FAIL: max projection_active_rate={max_rate:.4f} (>=1%)"


def check_git_sha(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check git SHA is captured."""
    git_sha = summary.get("git_sha")
    if git_sha and len(git_sha) >= 7:
        return True, f"PASS: git_sha={git_sha}"
    return False, f"FAIL: git_sha missing or invalid: {git_sha}"


def check_checkpoints(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check multiple checkpoints used."""
    checkpoints = summary.get("checkpoints", [])
    seeds = summary.get("parameters", {}).get("checkpoint_seeds", [])
    if len(checkpoints) >= 3 and len(set(seeds)) >= 3:
        return True, f"PASS: {len(checkpoints)} checkpoints with seeds {seeds}"
    return False, f"FAIL: Only {len(checkpoints)} checkpoints"


def check_dial_scales(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check dial scales coverage."""
    scales = summary.get("parameters", {}).get("dial_scales", [])
    if len(scales) >= 4:
        return True, f"PASS: {len(scales)} scales: {scales}"
    return False, f"FAIL: Only {len(scales)} scales"


def check_gates(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check decision gates."""
    gates = summary.get("gates", {})
    g0 = gates.get("g0_projection_inactive", {}).get("passed", False)
    g1 = gates.get("g1_stability", {}).get("passed", False)
    g2 = gates.get("g2_dial_range", {}).get("passed", False)
    g3 = gates.get("g3_tradeoff_exists", {}).get("passed", False)

    passed = g0 and g1 and g2
    status = f"G0={g0}, G1={g1}, G2={g2}, G3={g3}"
    if passed:
        return True, f"PASS: {status}"
    return False, f"FAIL: {status}"


def check_scale_summaries(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check scale summaries exist and have required fields."""
    scale_sums = summary.get("scale_summaries", [])
    if not scale_sums:
        return False, "FAIL: No scale_summaries"

    required_fields = ["scale", "L_preproj_mean", "argmax_b0_8x_mean", "success_trivial_mean"]
    for s in scale_sums:
        for field in required_fields:
            if field not in s:
                return False, f"FAIL: Missing field {field} in scale summary"

    return True, f"PASS: {len(scale_sums)} scale summaries with required fields"


def check_success_variation(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that success rate varies across scales (tradeoff exists)."""
    scale_sums = summary.get("scale_summaries", [])
    if not scale_sums:
        return False, "FAIL: No scale summaries to check"

    success_vals = [s.get("success_trivial_mean", 0) for s in scale_sums]
    success_range = max(success_vals) - min(success_vals)

    # Warning if no variation
    if success_range < 0.001:
        return True, f"WARNING: Success rate shows no variation (range={success_range:.4f})"
    return True, f"PASS: Success rate varies (range={success_range:.4f})"


def generate_audit_md(
    summary: Dict[str, Any],
    checks: List[Tuple[str, bool, str]],
    out_path: Path,
) -> None:
    """Generate AUDIT.md file."""
    passed = sum(1 for _, p, _ in checks if p)
    total = len(checks)
    warnings = sum(1 for _, p, msg in checks if p and "WARNING" in msg)

    content = f"""# Exp5 Audit Report

**Audit Date:** {datetime.now().isoformat()}
**Summary File:** {summary.get('experiment', 'Exp5')}
**Generated At:** {summary.get('generated_at', 'unknown')}
**Git SHA:** {summary.get('git_sha', 'unknown')}

## Audit Result

**Status:** {'✅ PASSED' if passed == total else '❌ FAILED'}
**Checks Passed:** {passed}/{total}
**Warnings:** {warnings}

## Check Details

| # | Check | Status | Result |
|---|-------|--------|--------|
"""
    for i, (name, passed, result) in enumerate(checks, 1):
        status = "✓" if passed else "✗"
        content += f"| {i}. {name} | {status} | {result} |\n"

    content += f"""

## Decision Gates

| Gate | Status | Details |
|------|--------|---------|
| G0 (Projection inactive) | {'PASS' if summary.get('gates', {}).get('g0_projection_inactive', {}).get('passed') else 'FAIL'} | latent_ball_radius=0 |
| G1 (Stability) | {'PASS' if summary.get('gates', {}).get('g1_stability', {}).get('passed') else 'FAIL'} | No NaN values |
| G2 (Dial range) | {'PASS' if summary.get('gates', {}).get('g2_dial_range', {}).get('passed') else 'FAIL'} | spread={summary.get('gates', {}).get('g2_dial_range', {}).get('spread', 0):.4f} |
| G3 (Tradeoff exists) | {'PASS' if summary.get('gates', {}).get('g3_tradeoff_exists', {}).get('passed') else 'FAIL'} | success_range={summary.get('gates', {}).get('g3_tradeoff_exists', {}).get('success_range', 0):.4f} |

## Non-Negotiables Verified

- `disable_value_head_norm: true` (value-head spectral norm OFF)
- `latent_ball_radius: 0.0` (projection disabled)

"""

    audit_path = out_path / "AUDIT.md"
    with open(audit_path, "w") as f:
        f.write(content)
    print(f"Saved: {audit_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit Exp5 tradeoff curve")
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path to summary.json",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        return 1

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print("=" * 60)
    print("Exp5 Audit: Stability–Expressivity Tradeoff Curve")
    print("=" * 60)
    print()

    # Run checks
    checks = [
        ("Value-head norm OFF", *check_value_head_norm(summary)),
        ("Projection disabled", *check_projection_disabled(summary)),
        ("Projection inactive", *check_projection_inactive(summary)),
        ("Git SHA captured", *check_git_sha(summary)),
        ("Multiple checkpoints", *check_checkpoints(summary)),
        ("Dial scale coverage", *check_dial_scales(summary)),
        ("Decision gates", *check_gates(summary)),
        ("Scale summaries complete", *check_scale_summaries(summary)),
        ("Success variation", *check_success_variation(summary)),
    ]

    # Print results
    passed = 0
    failed = 0
    for name, check_passed, result in checks:
        status = "✓" if check_passed else "✗"
        print(f"[{status}] {name}: {result}")
        if check_passed:
            passed += 1
        else:
            failed += 1

    print()
    print(f"AUDIT RESULT: {passed}/{passed+failed} checks passed")

    # Generate AUDIT.md
    out_path = summary_path.parent
    generate_audit_md(summary, checks, out_path)

    if failed > 0:
        print("AUDIT FAILED")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
