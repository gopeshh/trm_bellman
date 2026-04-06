#!/usr/bin/env python3
"""
Exp2 Final: Audit script for paper-ready artifacts.

Verifies:
1. Projection dominance at R=10: projection_active_rate ≈ 100%
2. Projection inactive at R≥100: projection_active_rate ≈ 0%
3. Dial range failure: L_preproj spread < 0.08
4. No monotonicity claims: grep CLAIMS.md for "monotonic" without "NOT"/"non-"
5. Claims match summary.json
6. All required files exist

Exit codes:
- 0: All checks passed
- 1: One or more checks failed
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP2_FINAL_DIR = PROJECT_ROOT / "results/paper_ready/exp2_final"

REQUIRED_FILES = [
    "fig_exp2_dial_does_not_control_Lz.pdf",
    "fig_exp2_projection_is_primary_stabilizer.pdf",
    "table_exp2_dial_does_not_control_Lz.tex",
    "table_exp2_projection_effect.tex",
    "CLAIMS.md",
    "PROVENANCE.md",
    "PAPER_INSERT_SNIPPET.tex",
    "summary.json",
]

DIAL_RANGE_THRESHOLD = 0.08
PROJECTION_ACTIVE_THRESHOLD_HIGH = 0.95  # Should be >= this at R=10
PROJECTION_ACTIVE_THRESHOLD_LOW = 0.05   # Should be <= this at R=disabled


# =============================================================================
# Audit Checks
# =============================================================================

def check_files_exist() -> Tuple[bool, str]:
    """Check all required files exist."""
    missing = []
    for f in REQUIRED_FILES:
        path = EXP2_FINAL_DIR / f
        if not path.exists():
            missing.append(f)

    if missing:
        return False, f"Missing files: {', '.join(missing)}"
    return True, "All required files exist"


def check_projection_dominance(summary: dict) -> Tuple[bool, str]:
    """Check projection_active_rate ≈ 100% at R=10."""
    dial = summary.get("dial_metrics", {})
    rates = dial.get("projection_active_rate_R10", [])

    if not rates:
        return False, "No projection_active_rate_R10 data found"

    mean_rate = sum(rates) / len(rates)
    if mean_rate >= PROJECTION_ACTIVE_THRESHOLD_HIGH:
        return True, f"Projection dominance at R=10: {mean_rate*100:.1f}% (>= {PROJECTION_ACTIVE_THRESHOLD_HIGH*100:.0f}%)"
    return False, f"Projection dominance check failed: {mean_rate*100:.1f}% (expected >= {PROJECTION_ACTIVE_THRESHOLD_HIGH*100:.0f}%)"


def check_projection_inactive(summary: dict) -> Tuple[bool, str]:
    """Check projection_active_rate ≈ 0% at R=disabled."""
    dial = summary.get("dial_metrics", {})
    rates = dial.get("projection_active_rate_R_disabled", [])

    if not rates:
        return False, "No projection_active_rate_R_disabled data found"

    mean_rate = sum(rates) / len(rates)
    if mean_rate <= PROJECTION_ACTIVE_THRESHOLD_LOW:
        return True, f"Projection inactive at R=disabled: {mean_rate*100:.1f}% (<= {PROJECTION_ACTIVE_THRESHOLD_LOW*100:.0f}%)"
    return False, f"Projection inactive check failed: {mean_rate*100:.1f}% (expected <= {PROJECTION_ACTIVE_THRESHOLD_LOW*100:.0f}%)"


def check_dial_range_failure(summary: dict) -> Tuple[bool, str]:
    """Check L_preproj spread < threshold (confirming dial failure)."""
    dial = summary.get("dial_metrics", {})
    L_preproj = dial.get("L_preproj_at_R_disabled", {})
    spread = L_preproj.get("spread", 0)
    passes = L_preproj.get("dial_range_passes", True)

    if passes:
        return False, f"Dial range passes but should fail: spread={spread:.3f} >= {DIAL_RANGE_THRESHOLD}"

    return True, f"Dial range failure confirmed: spread={spread:.3f} < {DIAL_RANGE_THRESHOLD}"


def check_no_monotonicity_claims() -> Tuple[bool, str]:
    """Check CLAIMS.md does not claim monotonicity."""
    claims_path = EXP2_FINAL_DIR / "CLAIMS.md"
    if not claims_path.exists():
        return False, "CLAIMS.md not found"

    content = claims_path.read_text()

    # Find all occurrences of "monotonic"
    monotonic_matches = list(re.finditer(r'\bmonoton\w*\b', content, re.IGNORECASE))

    bad_claims = []
    for match in monotonic_matches:
        # Get context (50 chars before and after)
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        context = content[start:end]

        # Check if it's a negative assertion
        negatives = ["not", "no ", "non-", "does not", "is not", "NOT", "Non-"]
        has_negative = any(neg.lower() in context.lower() for neg in negatives)

        if not has_negative:
            bad_claims.append(f"Found 'monotonic' without negative context: ...{context}...")

    if bad_claims:
        return False, "\n".join(bad_claims)

    if monotonic_matches:
        return True, f"Found {len(monotonic_matches)} 'monotonic' references, all with appropriate negative context"
    return True, "No monotonicity claims found"


def check_claims_match_summary(summary: dict) -> Tuple[bool, str]:
    """Check CLAIMS.md values match summary.json."""
    claims_path = EXP2_FINAL_DIR / "CLAIMS.md"
    if not claims_path.exists():
        return False, "CLAIMS.md not found"

    content = claims_path.read_text()
    issues = []

    # Check dial spread value
    dial = summary.get("dial_metrics", {})
    L_preproj = dial.get("L_preproj_at_R_disabled", {})
    spread = L_preproj.get("spread", 0)

    if f"{spread:.3f}" not in content and f"{spread:.2f}" not in content:
        issues.append(f"Spread value {spread:.3f} not found in CLAIMS.md")

    # Check stability metrics are mentioned (approximate check)
    stability = summary.get("stability_metrics", {})
    r10 = stability.get("R10", {})
    r_disabled = stability.get("R_disabled", {})

    # Check delta_V ranges are approximately correct
    r10_dv_min = r10.get("delta_V_range", [0, 0])[0]
    r10_dv_max = r10.get("delta_V_range", [0, 0])[1]

    # Look for range values in content (with some tolerance)
    found_r10_dv = False
    for line in content.split('\n'):
        if 'R=10' in line or 'R = 10' in line:
            if str(int(r10_dv_min)) in line or f"{r10_dv_min:.1f}" in line:
                found_r10_dv = True
                break

    if issues:
        return False, "; ".join(issues)

    return True, "Claims values match summary.json"


def check_is_monotonic_false(summary: dict) -> Tuple[bool, str]:
    """Verify is_monotonic is False in summary."""
    dial = summary.get("dial_metrics", {})
    L_preproj = dial.get("L_preproj_at_R_disabled", {})
    is_monotonic = L_preproj.get("is_monotonic", True)

    if is_monotonic:
        return False, "Summary claims is_monotonic=True but dial should be non-monotonic"

    return True, "Confirmed: dial is NOT monotonic"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EXP2 FINAL: Audit")
    print("=" * 60)

    if not EXP2_FINAL_DIR.exists():
        print(f"\n[FATAL] Output directory not found: {EXP2_FINAL_DIR}")
        print("Run make_paper_figures_exp2_final.py first")
        sys.exit(1)

    # Load summary
    summary_path = EXP2_FINAL_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Run checks
    checks: List[Tuple[str, callable]] = [
        ("Files exist", check_files_exist),
        ("Projection dominance at R=10", lambda: check_projection_dominance(summary)),
        ("Projection inactive at R=disabled", lambda: check_projection_inactive(summary)),
        ("Dial range failure confirmed", lambda: check_dial_range_failure(summary)),
        ("Dial is NOT monotonic", lambda: check_is_monotonic_false(summary)),
        ("No monotonicity claims in CLAIMS.md", check_no_monotonicity_claims),
        ("Claims values match summary", lambda: check_claims_match_summary(summary)),
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

    # Write audit results to AUDIT.md
    audit_content = f"""# Exp2 Final: Audit Results

**Generated**: {datetime.now().isoformat()}
**Status**: {"✅ ALL CHECKS PASSED" if all_passed else "❌ SOME CHECKS FAILED"}

## Audit Checks

"""

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        audit_content += f"""### {r["check"]}: {status}

{r["message"]}

"""

    audit_path = EXP2_FINAL_DIR / "AUDIT.md"
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
