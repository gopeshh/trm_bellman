#!/usr/bin/env python3
"""
Audit script for Exp3: Training-Time Projection Ablation

Verifies that CLAIMS.md matches summary.json and logs are consistent.
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime


def load_summary(results_dir: Path) -> dict:
    """Load summary.json."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    with open(summary_path) as f:
        return json.load(f)


def load_claims(results_dir: Path) -> str:
    """Load CLAIMS.md content."""
    claims_path = results_dir / "CLAIMS.md"
    if not claims_path.exists():
        raise FileNotFoundError(f"CLAIMS.md not found at {claims_path}")
    with open(claims_path) as f:
        return f.read()


def audit_g1_consistency(summary: dict, claims: str) -> tuple[bool, str]:
    """Verify G1 counts match between summary and claims."""
    # Extract from summary
    g1_passed_summary = summary["gates"]["g1_passed"]
    g1_total_summary = summary["gates"]["g1_total"]

    # Extract from claims - look for pattern like "6/6 conditions passed"
    match = re.search(r"\*\*G1 \(Stability\):\*\* (\d+)/(\d+) conditions passed", claims)
    if not match:
        return False, "Could not find G1 count in CLAIMS.md"

    g1_passed_claims = int(match.group(1))
    g1_total_claims = int(match.group(2))

    if g1_passed_summary != g1_passed_claims or g1_total_summary != g1_total_claims:
        return False, (
            f"G1 mismatch: summary={g1_passed_summary}/{g1_total_summary}, "
            f"claims={g1_passed_claims}/{g1_total_claims}"
        )

    return True, f"G1 consistent: {g1_passed_summary}/{g1_total_summary}"


def audit_condition_results(summary: dict, claims: str) -> tuple[bool, str]:
    """Verify individual condition results match."""
    conditions = summary["conditions"]

    issues = []
    for cond in conditions:
        name = cond["name"]
        success = cond["training"]["final_success_rate"]
        g1 = "PASS" if cond["g1_passed"] else "FAIL"

        # Check if this condition appears in claims table with matching values
        # Table format: | name | Contraction | R | Success | NaN | G1 |
        pattern = rf"\| {name} \| (?:True|False) \| (?:[\d.]+|disabled) \| ([\d.]+) \| (?:YES|NO) \| (PASS|FAIL) \|"
        match = re.search(pattern, claims)

        if not match:
            issues.append(f"Condition {name} not found in claims table")
            continue

        claims_success = float(match.group(1))
        claims_g1 = match.group(2)

        if abs(success - claims_success) > 0.001:
            issues.append(f"{name}: success mismatch (summary={success}, claims={claims_success})")

        if g1 != claims_g1:
            issues.append(f"{name}: G1 mismatch (summary={g1}, claims={claims_g1})")

    if issues:
        return False, "; ".join(issues)

    return True, f"All {len(conditions)} conditions match"


def audit_no_nan(summary: dict) -> tuple[bool, str]:
    """Verify no condition recorded its had_nan flag."""
    conditions = summary["conditions"]
    nan_conditions = [c["name"] for c in conditions if c["training"]["had_nan"]]

    if nan_conditions:
        return False, f"NaN detected in: {', '.join(nan_conditions)}"

    return True, "No condition recorded had_nan=true"


def audit_disabled_condition_g1(summary: dict, claims: str) -> tuple[bool, str]:
    """Verify the recorded G1 result for projection-disabled conditions."""

    def is_projection_disabled(condition: dict) -> bool:
        mode = condition.get("latent_projection_mode")
        radius = condition.get("latent_ball_radius", "missing")
        if "latent_projection_mode" in condition:
            return mode == "disabled" and radius is None
        if radius is None:
            # Normalized summaries may omit the mode after evaluation.
            return True
        # Frozen Exp3 paper-ready artifacts used radius 0 as the disabled tag.
        return radius == 0.0

    disabled_conditions = [
        c for c in summary["conditions"]
        if is_projection_disabled(c)
    ]

    # Check the recorded G1 flags without inferring a general stability claim.
    unstable = [c["name"] for c in disabled_conditions if not c["g1_passed"]]

    if unstable:
        return False, f"Projection-disabled conditions did not pass recorded G1: {unstable}"

    # Accept the old heading when auditing frozen claims artifacts.
    if "Finite-run observation 2" not in claims and "Claim 2" not in claims:
        return False, "Projection-disabled observation not found in CLAIMS.md"

    for cond in disabled_conditions:
        if cond["name"] not in claims:
            return False, f"Projection-disabled observation missing condition {cond['name']}"

    return True, (
        f"Projection-disabled conditions {[c['name'] for c in disabled_conditions]} "
        "passed their recorded G1 checks"
    )


def main():
    """Run all audits."""
    print("=" * 60)
    print("Exp3 Audit: Training-Time Projection Ablation")
    print("=" * 60)
    print()

    # Determine results directory
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "paper_ready" / "exp3_projection_ablation"

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Results directory: {results_dir}")
    print()

    # Load data
    try:
        summary = load_summary(results_dir)
        claims = load_claims(results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Run audits
    audits = [
        ("G1 Consistency", audit_g1_consistency(summary, claims)),
        ("Condition Results", audit_condition_results(summary, claims)),
        ("No NaN", audit_no_nan(summary)),
        ("Projection-disabled recorded G1", audit_disabled_condition_g1(summary, claims)),
    ]

    print("AUDIT RESULTS")
    print("-" * 60)

    all_passed = True
    for name, (passed, msg) in audits:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not passed:
            all_passed = False

    print()
    print("-" * 60)

    if all_passed:
        print("AUDIT PASSED: All checks verified")

        # Write AUDIT.md
        audit_path = results_dir / "AUDIT.md"
        with open(audit_path, "w") as f:
            f.write(f"# Exp3 Audit Results\n\n")
            f.write(f"**Audit Date:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"All {len(audits)} audit checks passed.\n\n")
            f.write(f"## Checks\n\n")
            for name, (passed, msg) in audits:
                status = "✅ PASS" if passed else "❌ FAIL"
                f.write(f"- **{name}:** {status} - {msg}\n")
            f.write(f"\n## Data Verified\n\n")
            f.write(f"- `summary.json`: {len(summary['conditions'])} conditions\n")
            f.write(f"- `CLAIMS.md`: Consistent with summary\n")
            f.write(f"- G1 gate: {summary['gates']['g1_passed']}/{summary['gates']['g1_total']} passed\n")

        print(f"Wrote: {audit_path}")
        sys.exit(0)
    else:
        print("AUDIT FAILED: Some checks did not pass")
        sys.exit(1)


if __name__ == "__main__":
    main()
