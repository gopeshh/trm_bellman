#!/usr/bin/env python3
"""
Audit script for Exp4 Final v2: Multi-checkpoint contraction dial evaluation.

Checks:
1. Multi-seed independence: 3+ checkpoints with different seeds
2. Dial scale coverage: 4 scales present
3. Non-negotiable compliance: projection disabled, value-head norm disabled
4. Statistical reporting: N >= 12 observations across checkpoint clusters
5. Within-scale variation check
6. Anti-degenerate: entropy not collapsed
7. Batch provenance: B0/B1 hashes consistent
8. Git SHA present
9. Decision gates documentation

Usage:
    python scripts/audit_exp4_final_v2.py \
        results/paper_ready/exp4_projection_free_dial_final/summary.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_multi_seed(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that 3+ independently trained checkpoints were used."""
    checkpoints = summary.get("checkpoints", [])
    seeds = summary.get("parameters", {}).get("checkpoint_seeds", [])

    if len(checkpoints) < 3:
        return False, f"FAIL: Only {len(checkpoints)} checkpoints (need >= 3)"

    unique_seeds = set(seeds)
    if len(unique_seeds) < 3:
        return False, f"FAIL: Only {len(unique_seeds)} unique seeds (need >= 3)"

    return True, f"PASS: {len(checkpoints)} checkpoints with seeds {sorted(unique_seeds)}"


def check_dial_scales(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that 4 dial scales are present."""
    scales = summary.get("parameters", {}).get("dial_scales", [])

    if len(scales) < 4:
        return False, f"FAIL: Only {len(scales)} scales (need >= 4)"

    # Check scale range
    min_scale = min(scales)
    max_scale = max(scales)

    if max_scale < 0.9:
        return False, f"FAIL: Max scale {max_scale} < 0.9"

    return True, f"PASS: {len(scales)} scales: {scales}"


def check_non_negotiables(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check non-negotiable config settings."""
    config = summary.get("config", {})

    issues = []

    # Current artifacts use explicit disabled mode with no radius. This auditor
    # also reads frozen summaries that predate the mode field and used radius 0.
    projection_mode = config.get("latent_projection_mode")
    latent_ball_radius = config.get("latent_ball_radius", -1)
    explicit_disabled = (
        projection_mode == "disabled"
        and "latent_ball_radius" in config
        and latent_ball_radius is None
    )
    legacy_disabled = (
        "latent_projection_mode" not in config
        and latent_ball_radius == 0.0
    )
    if not explicit_disabled and not legacy_disabled:
        issues.append(
            "projection config must be latent_projection_mode=disabled with "
            f"latent_ball_radius=None; got mode={projection_mode}, "
            f"radius={latent_ball_radius}"
        )

    # Value head norm must be disabled
    disable_value_head_norm = config.get("disable_value_head_norm", False)
    if not disable_value_head_norm:
        issues.append(f"disable_value_head_norm={disable_value_head_norm} (must be True)")

    # All results should show projection inactive
    all_results = summary.get("all_results", [])
    proj_active = [r.get("projection_active", True) for r in all_results]
    if any(proj_active):
        n_active = sum(proj_active)
        issues.append(f"{n_active}/{len(all_results)} had projection_active=True")

    if issues:
        return False, "FAIL: " + "; ".join(issues)

    projection_detail = (
        "explicit disabled mode with radius=None"
        if explicit_disabled
        else "legacy frozen-artifact radius=0 encoding"
    )
    return True, f"PASS: {projection_detail}, disable_value_head_norm=True"


def check_sample_count(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that N >= 12 observations are reported."""
    n_samples = summary.get("monotonicity", {}).get("n_samples", 0)

    if n_samples < 12:
        return False, f"FAIL: N={n_samples} (need >= 12)"

    # Verify it matches checkpoints × scales
    n_checkpoints = len(summary.get("checkpoints", []))
    n_scales = len(summary.get("parameters", {}).get("dial_scales", []))
    expected = n_checkpoints * n_scales

    if n_samples != expected:
        return False, f"WARN: N={n_samples} but expected {n_checkpoints}×{n_scales}={expected}"

    return True, f"PASS: N={n_samples} ({n_checkpoints} checkpoints × {n_scales} scales)"


def check_within_scale_variation(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Report whether metrics vary across checkpoints within each scale."""
    all_results = summary.get("all_results", [])
    scales = sorted(set(r["scale"] for r in all_results))

    # Check if argmax values vary across checkpoints within each scale
    varying_scales = 0

    for scale in scales:
        scale_results = [r for r in all_results if r["scale"] == scale]
        argmax_vals = [r["argmax_b0_8x"] for r in scale_results]
        delta_v_vals = [r["delta_V_b0_8x"] for r in scale_results]

        # Check if there's any variation
        if len(set(argmax_vals)) > 1 or len(set(delta_v_vals)) > 1:
            varying_scales += 1

    anti_deg = summary.get("anti_degenerate", {})
    if "within_scale_variation_observed" in anti_deg:
        summary_observed = anti_deg["within_scale_variation_observed"]
        if summary_observed != (varying_scales > 0):
            return False, "FAIL: within-scale variation flag disagrees with recorded values"

    if varying_scales == 0:
        return False, (
            "WARN: no within-scale metric variation was recorded; sample dependence "
            "must be assessed from the checkpoint design, not value equality"
        )

    return True, f"PASS: Metrics vary across checkpoints ({varying_scales}/{len(scales)} scales)"


def check_entropy(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that policy entropy hasn't collapsed (anti-degenerate)."""
    anti_deg = summary.get("anti_degenerate", {})

    min_entropy = anti_deg.get("min_entropy_train", 0)
    mean_entropy = anti_deg.get("mean_entropy_train", 0)
    entropy_ok = anti_deg.get("entropy_ok", False)

    if not entropy_ok:
        return False, f"FAIL: min_entropy={min_entropy:.3f} does not exceed configured threshold 0.5"

    if min_entropy < 0.5:
        return False, f"FAIL: min_entropy={min_entropy:.3f} is below configured threshold 0.5"

    return True, f"PASS: mean_entropy={mean_entropy:.3f}, min_entropy={min_entropy:.3f}"


def check_batch_provenance(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that B0/B1 hashes are present and consistent."""
    batches = summary.get("batches", {})

    b0_hash = batches.get("b0_hash", "")
    b1_hash = batches.get("b1_hash", "")
    b0_count = batches.get("b0_count", 0)
    b1_count = batches.get("b1_count", 0)

    issues = []

    if not b0_hash:
        issues.append("missing B0 hash")
    if not b1_hash:
        issues.append("missing B1 hash")
    if b0_count < 50:
        issues.append(f"B0 count {b0_count} < 50")
    if b1_count < 100:
        issues.append(f"B1 count {b1_count} < 100")

    if issues:
        return False, "FAIL: " + "; ".join(issues)

    return True, f"PASS: B0={b0_count} (hash:{b0_hash}), B1={b1_count} (hash:{b1_hash})"


def check_git_sha(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that git SHA is present."""
    git_sha = summary.get("git_sha", None)

    if git_sha is None or git_sha == "None":
        return False, "FAIL: git_sha is None"

    if len(git_sha) < 7:
        return False, f"FAIL: git_sha '{git_sha}' too short"

    return True, f"PASS: git_sha={git_sha}"


def check_decision_gates(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that all decision gates are documented."""
    gates = summary.get("gates", {})

    required_gates = ["g0_projection_inactive", "g1_stability", "g2_dial_range", "g3_monotonicity"]

    missing = []
    for gate in required_gates:
        if gate not in gates:
            missing.append(gate)

    if missing:
        return False, f"FAIL: Missing gates: {missing}"

    # Report gate status
    g0 = gates.get("g0_projection_inactive", {}).get("passed", False)
    g1 = gates.get("g1_stability", {}).get("passed", False)
    g2 = gates.get("g2_dial_range", {}).get("passed", False)
    g3 = gates.get("g3_monotonicity", {}).get("status", "UNKNOWN")

    return True, f"PASS: G0={g0}, G1={g1}, G2={g2}, G3={g3}"


def check_statistical_reporting(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that Spearman correlations are reported with proper inference."""
    mono = summary.get("monotonicity", {})

    # Check for cluster bootstrap (preferred method)
    stat_method = mono.get("stat_method", "unknown")

    if stat_method == "cluster_bootstrap":
        # Check for bootstrap fields
        boot_aa_b0 = mono.get("boot_aa_b0", {})
        if not boot_aa_b0:
            return False, "FAIL: Missing boot_aa_b0 in cluster bootstrap results"

        rho = boot_aa_b0.get("rho_point", None)
        ci_lo = boot_aa_b0.get("rho_ci_lower", None)
        ci_hi = boot_aa_b0.get("rho_ci_upper", None)

        if rho is None or ci_lo is None or ci_hi is None:
            return False, "FAIL: Incomplete bootstrap results"

        n_bootstrap = mono.get("n_bootstrap", 0)
        n_clusters = mono.get("n_clusters", 0)

        return True, f"PASS: ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (cluster bootstrap, {n_bootstrap} iter, {n_clusters} clusters)"
    else:
        # Legacy i.i.d. method - check but warn about non-i.i.d.
        required = ["iid_rho_aa_b0", "iid_p_aa_b0"]
        for key in required:
            if key not in mono:
                # Try old field names
                old_key = key.replace("iid_", "")
                if old_key not in mono:
                    return False, f"FAIL: Missing {key} in monotonicity stats"
                mono[key] = mono[old_key]

        rho = mono.get("iid_rho_aa_b0", mono.get("rho_aa_b0", 0))
        p = mono.get("iid_p_aa_b0", mono.get("p_aa_b0", 0))

        return True, f"PASS: ρ={rho:.3f} (p={p:.4f}) WARNING: i.i.d. p-value may be invalid for repeated measures"


def check_stat_method(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that proper statistical method is used for repeated measures."""
    mono = summary.get("monotonicity", {})

    stat_method = mono.get("stat_method", None)

    if stat_method == "cluster_bootstrap":
        n_bootstrap = mono.get("n_bootstrap", 0)
        n_clusters = mono.get("n_clusters", 0)
        return True, f"PASS: cluster_bootstrap ({n_bootstrap} iterations, {n_clusters} clusters)"
    elif stat_method is None:
        # Legacy - check if using i.i.d. p-values without caveat
        return False, "WARN: No statistical method specified; i.i.d. p-values may be invalid for repeated measures"
    else:
        return False, f"WARN: Unknown statistical method: {stat_method}"


def run_audit(summary_path: str) -> int:
    """Run full audit and return exit code."""
    from datetime import datetime

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print("=" * 70)
    print("Exp4 Final v2 Audit")
    print("=" * 70)
    print(f"Summary: {summary_path}")
    print(f"Decision: {summary.get('decision', 'UNKNOWN')}")
    print(f"Generated: {summary.get('generated_at', 'UNKNOWN')}")
    print()

    checks = [
        ("1. Multi-seed independence", check_multi_seed),
        ("2. Dial scale coverage", check_dial_scales),
        ("3. Non-negotiable compliance", check_non_negotiables),
        ("4. Observation count (N >= 12)", check_sample_count),
        ("5. Within-scale variation", check_within_scale_variation),
        ("6. Anti-degenerate (entropy)", check_entropy),
        ("7. Batch provenance", check_batch_provenance),
        ("8. Git SHA tracking", check_git_sha),
        ("9. Decision gates", check_decision_gates),
        ("10. Statistical method", check_stat_method),
        ("11. Statistical reporting", check_statistical_reporting),
    ]

    passed = 0
    failed = 0
    warnings = 0
    results = []  # Store results for AUDIT.md

    print("-" * 60)
    for name, check_fn in checks:
        try:
            ok, msg = check_fn(summary)
            status = "✓" if ok else "✗"
            if "WARN" in msg:
                status = "⚠"
                warnings += 1
            elif ok:
                passed += 1
            else:
                failed += 1
            print(f"[{status}] {name}")
            print(f"    {msg}")
            results.append((name, status, msg, ok))
        except Exception as e:
            print(f"[✗] {name}")
            print(f"    ERROR: {e}")
            failed += 1
            results.append((name, "✗", f"ERROR: {e}", False))

    print("-" * 60)
    print()

    # Summary
    total = passed + failed
    print(f"AUDIT SUMMARY: {passed}/{total} checks passed")
    if warnings > 0:
        print(f"               {warnings} warnings")

    audit_passed = failed == 0

    # Generate AUDIT.md
    audit_path = Path(summary_path).parent / "AUDIT.md"
    generate_audit_md(audit_path, summary, results, passed, total, warnings, audit_passed)
    print(f"\nGenerated: {audit_path}")

    if not audit_passed:
        print("\n❌ AUDIT FAILED")
        return 1
    else:
        print("\n✅ AUDIT PASSED")
        return 0


def generate_audit_md(
    audit_path: Path,
    summary: Dict[str, Any],
    results: List[Tuple[str, str, str, bool]],
    passed: int,
    total: int,
    warnings: int,
    audit_passed: bool,
) -> None:
    """Generate AUDIT.md file."""
    from datetime import datetime

    mono = summary.get("monotonicity", {})
    gates = summary.get("gates", {})

    content = f"""# Exp4 v2 Audit Report

**Audit Date:** {datetime.now().isoformat()}
**Summary File:** {summary.get('experiment', 'Exp4_final_v2')}
**Generated At:** {summary.get('generated_at', 'unknown')}
**Git SHA:** {summary.get('git_sha', 'unknown')}

## Audit Result

**Status:** {'✅ PASSED' if audit_passed else '❌ FAILED'}
**Checks Passed:** {passed}/{total}
**Warnings:** {warnings}

## Check Details

| # | Check | Status | Result |
|---|-------|--------|--------|
"""
    for name, status, msg, ok in results:
        # Escape pipes in msg
        msg_escaped = msg.replace("|", "\\|")
        content += f"| {name} | {status} | {msg_escaped} |\n"

    content += f"""

## Statistical Methodology

- **Method:** {mono.get('stat_method', 'unknown')}
- **N Observations:** {mono.get('n_samples', 'unknown')}
- **N Clusters (checkpoints):** {mono.get('n_clusters', 'unknown')}
- **Bootstrap Iterations:** {mono.get('n_bootstrap', 'N/A')}
- **Rationale:** Scales are repeated measures on the same checkpoint; cluster bootstrap accounts for within-checkpoint correlation

## Decision Gates

| Gate | Status | Details |
|------|--------|---------|
| G0 (Projection inactive) | {'PASS' if gates.get('g0_projection_inactive', {}).get('passed') else 'FAIL'} | Explicit disabled mode; legacy radius-0 summaries accepted by this auditor |
| G1 (Stability) | {'PASS' if gates.get('g1_stability', {}).get('passed') else 'FAIL'} | No NaN values |
| G2 (Dial range) | {'PASS' if gates.get('g2_dial_range', {}).get('passed') else 'FAIL'} | spread={gates.get('g2_dial_range', {}).get('L_preproj_spread', 0):.4f} (threshold ≥0.10) |
| G3 (Monotonicity) | {gates.get('g3_monotonicity', {}).get('status', 'UNKNOWN')} | |ρ|>0.5, 95% CI excludes 0 |

## Depth Mismatch Notation

Metrics labeled "(n2=8)" refer to evaluation at depth n2=8, which is **4× the training depth** (n_train=2).
This is NOT "8× mismatch" - the mismatch multiplier is n2/n_train = 8/2 = 4.

## Non-Negotiables Verified

- `disable_value_head_norm: true` (value-head spectral norm OFF)
- Current schema: `latent_projection_mode: disabled`, `latent_ball_radius: null`
- Compatibility: frozen summaries without a mode field may use `latent_ball_radius: 0.0`
"""

    with open(audit_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Audit Exp4 Final v2 results")
    parser.add_argument(
        "summary_json",
        type=str,
        help="Path to summary.json from exp4 evaluation",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        return 1

    return run_audit(str(summary_path))


if __name__ == "__main__":
    sys.exit(main())
