#!/usr/bin/env python3
"""
Audit script for Exp4: Projection-free Contraction Dial Range Test

Verifies that CLAIMS.md matches summary.json and gates are correctly evaluated.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


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


def audit_g0_projection_inactive(summary: dict) -> tuple:
    """Verify G0: projection_active_rate < 1%."""
    g0 = summary["gates"]["g0_projection_inactive"]
    passed = g0["passed"]
    rate = g0["rate"]

    if rate >= 0.01:
        return False, f"G0 failed: projection_active_rate={rate*100:.1f}% >= 1%"
    if not passed:
        return False, "G0 marked as failed but rate < 1%"

    return True, f"G0 passed: projection_active_rate={rate*100:.1f}%"


def audit_g1_stability(summary: dict) -> tuple:
    """Verify G1: no NaN in L_preproj values."""
    g1 = summary["gates"]["g1_stability"]
    passed = g1["passed"]

    # Check results for NaN
    results = summary["results"]
    nan_found = any(r["L_preproj"] != r["L_preproj"] for r in results)  # NaN check

    if nan_found and passed:
        return False, "G1 marked as passed but NaN found in L_preproj"
    if not nan_found and not passed:
        return False, "G1 marked as failed but no NaN found"

    return True, f"G1 {'passed' if passed else 'failed'}: stability check"


def audit_g2_dial_range(summary: dict) -> tuple:
    """Verify G2: L_preproj spread >= 0.10."""
    g2 = summary["gates"]["g2_dial_range"]
    passed = g2["passed"]
    threshold = g2["threshold"]
    spread = g2["L_preproj_spread"]

    if spread < threshold and passed:
        return False, f"G2 marked as passed but spread={spread:.4f} < threshold={threshold}"
    if spread >= threshold and not passed:
        return False, f"G2 marked as failed but spread={spread:.4f} >= threshold={threshold}"

    return True, f"G2 {'passed' if passed else 'failed'}: spread={spread:.4f} (threshold={threshold})"


def audit_g3_monotonicity(summary: dict) -> tuple:
    """Verify G3: Spearman rho reported correctly."""
    g3 = summary["gates"]["g3_monotonic_linkage"]
    rho = g3["spearman_rho"]
    status = g3["status"]

    if abs(rho) > 0.5 and status != "PASS":
        return False, f"G3 status mismatch: |rho|={abs(rho):.2f} > 0.5 but status={status}"
    if abs(rho) <= 0.5 and status == "PASS":
        return False, f"G3 status mismatch: |rho|={abs(rho):.2f} <= 0.5 but status=PASS"

    return True, f"G3 {status}: Spearman ρ={rho:.4f}"


def audit_value_head_norm_off(summary: dict) -> tuple:
    """Verify disable_value_head_norm: true."""
    config = summary["config"]
    disabled = config.get("disable_value_head_norm", False)

    if not disabled:
        return False, "disable_value_head_norm is not True (violation of non-negotiable)"

    return True, "disable_value_head_norm: true (verified)"


def audit_projection_disabled(summary: dict) -> tuple:
    """Verify latent_ball_radius: 0 (projection disabled)."""
    config = summary["config"]
    R = config.get("latent_ball_radius", -1)

    if R != 0.0:
        return False, f"latent_ball_radius={R} != 0.0 (projection not disabled)"

    return True, "latent_ball_radius: 0.0 (projection disabled)"


def audit_no_overclaim(summary: dict, claims: str) -> tuple:
    """Verify claims match actual gate results."""
    g2_passed = summary["gates"]["g2_dial_range"]["passed"]
    decision = summary["decision"]

    if not g2_passed:
        # Should explicitly state dial-range failure
        if "insufficient range" not in decision.lower() and "insufficient range" not in claims.lower():
            return False, "G2 failed but 'insufficient range' not mentioned in claims"
    else:
        # Should NOT claim negative if G2 passed
        if "NEGATIVE" in decision:
            return False, "G2 passed but decision is NEGATIVE"

    return True, f"No overclaim detected (G2={'passed' if g2_passed else 'failed'})"


def main():
    """Run all audits."""
    print("=" * 60)
    print("Exp4 Audit: Projection-free Contraction Dial Range Test")
    print("=" * 60)
    print()

    # Determine results directory
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "paper_ready" / "exp4_projection_free_dial"

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
        ("G0 (Projection inactive)", audit_g0_projection_inactive(summary)),
        ("G1 (Stability)", audit_g1_stability(summary)),
        ("G2 (Dial range)", audit_g2_dial_range(summary)),
        ("G3 (Monotonicity)", audit_g3_monotonicity(summary)),
        ("Value-head norm OFF", audit_value_head_norm_off(summary)),
        ("Projection disabled", audit_projection_disabled(summary)),
        ("No overclaim", audit_no_overclaim(summary, claims)),
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

    # Summary statistics
    g2_spread = summary["gates"]["g2_dial_range"]["L_preproj_spread"]
    g3_rho = summary["gates"]["g3_monotonic_linkage"]["spearman_rho"]
    decision = summary["decision"]

    print(f"\nKEY METRICS:")
    print(f"  L_preproj spread: {g2_spread:.4f}")
    print(f"  Spearman ρ:       {g3_rho:.4f}")
    print(f"  Decision:         {decision}")
    print()

    if all_passed:
        print("AUDIT PASSED: All checks verified")

        # Write AUDIT.md
        audit_path = results_dir / "AUDIT.md"
        with open(audit_path, "w") as f:
            f.write(f"# Exp4 Audit Results\n\n")
            f.write(f"**Audit Date:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"All {len(audits)} audit checks passed.\n\n")
            f.write(f"## Checks\n\n")
            for name, (passed, msg) in audits:
                status = "✅ PASS" if passed else "❌ FAIL"
                f.write(f"- **{name}:** {status} - {msg}\n")
            f.write(f"\n## Key Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| L_preproj spread | {g2_spread:.4f} |\n")
            f.write(f"| Spearman ρ | {g3_rho:.4f} |\n")
            f.write(f"| G2 (dial range) | {'PASS' if summary['gates']['g2_dial_range']['passed'] else 'FAIL'} |\n")
            f.write(f"| G3 (monotonicity) | {summary['gates']['g3_monotonic_linkage']['status']} |\n")
            f.write(f"\n## Decision\n\n")
            f.write(f"{decision}\n")
            f.write(f"\n## Proceed to Full Sweep?\n\n")
            f.write(f"{'YES' if summary['proceed_to_full_sweep'] else 'NO'} - G2 {'passed' if summary['gates']['g2_dial_range']['passed'] else 'failed'}\n")

        print(f"Wrote: {audit_path}")
        sys.exit(0)
    else:
        print("AUDIT FAILED: Some checks did not pass")
        sys.exit(1)


if __name__ == "__main__":
    main()
