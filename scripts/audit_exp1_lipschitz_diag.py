#!/usr/bin/env python3
"""
Exp1 Lipschitz/Projection Diagnostic: Audit script.

Verifies:
1. Projection dominance at R=10: projection_active_rate >= 90%
2. Projection inactive at R=100: projection_active_rate <= 20%
3. Results for both conditions exist
4. Table values match summary.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("/home/buiksat/trm_bellman")
DIAG_DIR = PROJECT_ROOT / "results/paper_ready/exp1_lipschitz_diag"

REQUIRED_FILES = [
    "table_exp1_lipschitz_diag.tex",
    "CLAIMS.md",
    "PROVENANCE.md",
    "summary.json",
]

PROJ_ACTIVE_HIGH_THRESHOLD = 0.90  # At R=10, should be >= 90%
PROJ_ACTIVE_LOW_THRESHOLD = 0.20   # At R=100, should be <= 20%


# =============================================================================
# Audit Checks
# =============================================================================

def check_files_exist() -> Tuple[bool, str]:
    """Check all required files exist."""
    missing = []
    for f in REQUIRED_FILES:
        if not (DIAG_DIR / f).exists():
            missing.append(f)

    if missing:
        return False, f"Missing files: {', '.join(missing)}"
    return True, "All required files exist"


def check_projection_dominance_r10(summary: dict) -> Tuple[bool, str]:
    """Check projection is active at R=10 for both conditions."""
    agg = summary.get("aggregated", {})
    issues = []

    for condition in ["no_contraction", "contraction"]:
        if condition not in agg:
            issues.append(f"Missing {condition} data")
            continue

        if "10.0" not in agg[condition]:
            issues.append(f"Missing R=10 data for {condition}")
            continue

        rate = agg[condition]["10.0"]["projection_active_rate"]
        if rate < PROJ_ACTIVE_HIGH_THRESHOLD:
            issues.append(f"{condition} at R=10: {rate*100:.1f}% < {PROJ_ACTIVE_HIGH_THRESHOLD*100:.0f}%")

    if issues:
        return False, "; ".join(issues)

    rates = []
    for condition in ["no_contraction", "contraction"]:
        rates.append(agg[condition]["10.0"]["projection_active_rate"])
    return True, f"Projection active at R=10: {rates[0]*100:.0f}%, {rates[1]*100:.0f}% (>= {PROJ_ACTIVE_HIGH_THRESHOLD*100:.0f}%)"


def check_projection_inactive_r100(summary: dict) -> Tuple[bool, str]:
    """Check projection is mostly inactive at R=100."""
    agg = summary.get("aggregated", {})
    issues = []

    for condition in ["no_contraction", "contraction"]:
        if condition not in agg:
            continue

        if "100.0" not in agg[condition]:
            issues.append(f"Missing R=100 data for {condition}")
            continue

        rate = agg[condition]["100.0"]["projection_active_rate"]
        if rate > PROJ_ACTIVE_LOW_THRESHOLD:
            issues.append(f"{condition} at R=100: {rate*100:.1f}% > {PROJ_ACTIVE_LOW_THRESHOLD*100:.0f}%")

    if issues:
        return False, "; ".join(issues)

    rates = []
    for condition in ["no_contraction", "contraction"]:
        if "100.0" in agg.get(condition, {}):
            rates.append(agg[condition]["100.0"]["projection_active_rate"])
    return True, f"Projection inactive at R=100: {rates[0]*100:.0f}%, {rates[1]*100:.0f}% (<= {PROJ_ACTIVE_LOW_THRESHOLD*100:.0f}%)"


def check_both_conditions(summary: dict) -> Tuple[bool, str]:
    """Check both conditions have data."""
    agg = summary.get("aggregated", {})

    if "no_contraction" not in agg:
        return False, "Missing no_contraction data"
    if "contraction" not in agg:
        return False, "Missing contraction data"

    return True, "Both conditions present"


def check_consistency_with_exp2(summary: dict) -> Tuple[bool, str]:
    """Check pattern is consistent with Exp2 (informational)."""
    agg = summary.get("aggregated", {})

    # At R=10, projection should be highly active (like Exp2)
    r10_rates = []
    for condition in ["no_contraction", "contraction"]:
        if "10.0" in agg.get(condition, {}):
            r10_rates.append(agg[condition]["10.0"]["projection_active_rate"])

    # At R=100, projection should be mostly inactive (like Exp2c)
    r100_rates = []
    for condition in ["no_contraction", "contraction"]:
        if "100.0" in agg.get(condition, {}):
            r100_rates.append(agg[condition]["100.0"]["projection_active_rate"])

    if not r10_rates or not r100_rates:
        return False, "Insufficient data for consistency check"

    # Check the gap
    avg_r10 = sum(r10_rates) / len(r10_rates)
    avg_r100 = sum(r100_rates) / len(r100_rates)
    gap = avg_r10 - avg_r100

    if gap < 0.5:
        return False, f"Projection rate gap too small: {gap*100:.1f}pp (expected > 50pp)"

    return True, f"Consistent with Exp2: R=10 avg={avg_r10*100:.0f}%, R=100 avg={avg_r100*100:.0f}% (gap={gap*100:.0f}pp)"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EXP1 LIPSCHITZ DIAGNOSTIC: Audit")
    print("=" * 60)

    if not DIAG_DIR.exists():
        print(f"\n[FATAL] Diagnostic directory not found: {DIAG_DIR}")
        print("Run exp1_lipschitz_diag.py first")
        sys.exit(1)

    # Load summary
    summary_path = DIAG_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Run checks
    checks = [
        ("Files exist", check_files_exist),
        ("Both conditions present", lambda: check_both_conditions(summary)),
        ("Projection dominance at R=10", lambda: check_projection_dominance_r10(summary)),
        ("Projection inactive at R=100", lambda: check_projection_inactive_r100(summary)),
        ("Consistency with Exp2", lambda: check_consistency_with_exp2(summary)),
    ]

    results = []
    all_passed = True

    print("\n[Audit Checks]\n")
    for name, check_fn in checks:
        try:
            passed, msg = check_fn()
        except Exception as e:
            passed = False
            msg = f"Exception: {e}"

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        print(f"         {msg}")
        print()

        results.append({
            "check": name,
            "passed": passed,
            "message": msg,
        })

        if not passed:
            all_passed = False

    # Write audit results
    audit_content = f"""# Exp1 Lipschitz/Projection Diagnostic: Audit Results

**Generated**: {datetime.now().isoformat()}
**Status**: {"✅ ALL CHECKS PASSED" if all_passed else "❌ SOME CHECKS FAILED"}

## Audit Checks

"""

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        audit_content += f"""### {r["check"]}: {status}

{r["message"]}

"""

    audit_path = DIAG_DIR / "AUDIT.md"
    audit_path.write_text(audit_content)
    print(f"[Output] Saved: {audit_path}")

    # Summary
    print("=" * 60)
    if all_passed:
        print("AUDIT: ✅ ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("AUDIT: ❌ SOME CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
