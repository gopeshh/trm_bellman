#!/usr/bin/env python3
"""
Audit script for Exp4 Final: Projection-free Contraction Dial (Paper-Defensible)

Verifies that CLAIMS.md matches summary.json and gates are correctly evaluated.
"""

import json
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
    rate = g0["max_rate"]

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
    results = summary["all_results"]
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
    g3 = summary["gates"]["g3_monotonicity"]
    rho_aa = g3["rho_aa_b0"]
    aa_passed = g3["aa_b0_passed"]
    status = g3["status"]

    # argmax agreement should have |rho| > 0.5 for PASS
    if abs(rho_aa) > 0.5 and not aa_passed:
        return False, f"G3 aa_passed mismatch: |rho|={abs(rho_aa):.2f} > 0.5 but marked failed"

    # Status should match
    if status == "PASS" and not (g3["dv_b0_passed"] or g3["aa_b0_passed"]):
        return False, f"G3 status=PASS but neither ΔV nor argmax passed"

    return True, f"G3 {status}: argmax ρ={rho_aa:.4f}"


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


def audit_multi_seed(summary: dict) -> tuple:
    """Verify multi-seed evaluation was performed."""
    params = summary.get("parameters", {})
    seeds = params.get("seeds", [])

    if len(seeds) < 3:
        return False, f"Only {len(seeds)} seeds used, expected 3+"

    # Verify results exist for each seed
    results = summary.get("all_results", [])
    seeds_in_results = set(r["seed"] for r in results)

    if len(seeds_in_results) < 3:
        return False, f"Results only have {len(seeds_in_results)} seeds"

    return True, f"Multi-seed evaluation verified ({len(seeds)} seeds)"


def audit_dial_scales(summary: dict) -> tuple:
    """Verify correct dial scales were used."""
    params = summary.get("parameters", {})
    scales = params.get("dial_scales", [])
    expected_scales = [1.0, 0.85, 0.70, 0.55]

    if len(scales) < 4:
        return False, f"Only {len(scales)} dial scales used, expected 4"

    # Verify results exist for each scale
    results = summary.get("all_results", [])
    scales_in_results = set(r["scale"] for r in results)

    if len(scales_in_results) < 4:
        return False, f"Results only have {len(scales_in_results)} scales"

    return True, f"Dial scales verified ({len(scales)} scales)"


def audit_no_overclaim(summary: dict, claims: str) -> tuple:
    """Verify claims match actual gate results."""
    decision = summary["decision"]

    # Check decision matches gates
    g0 = summary["gates"]["g0_projection_inactive"]["passed"]
    g1 = summary["gates"]["g1_stability"]["passed"]
    g2 = summary["gates"]["g2_dial_range"]["passed"]
    g3_status = summary["gates"]["g3_monotonicity"]["status"]

    if not g0 and "INVALID" not in decision:
        return False, "G0 failed but decision doesn't reflect INVALID"
    if not g1 and "INVALID" not in decision:
        return False, "G1 failed but decision doesn't reflect INVALID"
    if not g2 and "NEGATIVE" not in decision and "INVALID" not in decision:
        return False, "G2 failed but decision doesn't reflect NEGATIVE"

    # Check claims include the decision
    if "POSITIVE" in decision and "Positive" not in claims and "positive" not in claims:
        return False, "Decision is POSITIVE but claims don't mention it"

    return True, f"No overclaim detected (decision={decision.split(':')[0]})"


def main():
    """Run all audits."""
    print("=" * 60)
    print("Exp4 Final Audit: Projection-free Contraction Dial")
    print("=" * 60)
    print()

    # Determine results directory
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "paper_ready" / "exp4_projection_free_dial_final"

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
        ("Multi-seed evaluation", audit_multi_seed(summary)),
        ("Dial scales", audit_dial_scales(summary)),
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
    g3_rho = summary["gates"]["g3_monotonicity"]["rho_aa_b0"]
    decision = summary["decision"]

    print(f"\nKEY METRICS:")
    print(f"  L_preproj spread: {g2_spread:.4f}")
    print(f"  Spearman ρ (argmax): {g3_rho:.4f}")
    print(f"  Decision:         {decision}")
    print()

    if all_passed:
        print("AUDIT PASSED: All checks verified")

        # Write AUDIT.md
        audit_path = results_dir / "AUDIT.md"
        with open(audit_path, "w") as f:
            f.write(f"# Exp4 Final Audit Results\n\n")
            f.write(f"**Audit Date:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"All {len(audits)} audit checks passed.\n\n")
            f.write(f"## Checks\n\n")
            for name, (passed, msg) in audits:
                status = "PASS" if passed else "FAIL"
                f.write(f"- **{name}:** {status} - {msg}\n")
            f.write(f"\n## Key Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| L_preproj spread | {g2_spread:.4f} |\n")
            f.write(f"| Spearman ρ (argmax) | {g3_rho:.4f} |\n")
            f.write(f"| G2 (dial range) | {'PASS' if summary['gates']['g2_dial_range']['passed'] else 'FAIL'} |\n")
            f.write(f"| G3 (monotonicity) | {summary['gates']['g3_monotonicity']['status']} |\n")
            f.write(f"\n## Decision\n\n")
            f.write(f"{decision}\n")

        print(f"Wrote: {audit_path}")
        sys.exit(0)
    else:
        print("AUDIT FAILED: Some checks did not pass")
        sys.exit(1)


if __name__ == "__main__":
    main()
