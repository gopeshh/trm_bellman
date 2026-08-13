#!/usr/bin/env python3
"""
Phase 4 Audit Script: 2×2 Norm Ablation Paper-Ready Checks.

Runs ≥9 automated checks to validate experimental integrity:
1. Config integrity: all 4 condition YAMLs exist and are valid
2. Correct toggle combinations: each condition has expected enable_contraction/disable_value_head_norm
3. Seeds present: all 3 seeds (41, 42, 43) have full checkpoints per condition
4. No mislabeled conditions: checkpoint dirs match config names
5. Summary.json exists and is readable JSON
6. Summary uses the strict publication schema version 3
7. All 12 runs present in summary
8. CLAIMS.md exists and is non-empty
9. PROVENANCE.md exists and references correct configs
10. Metric bounds: measured stability metrics are within valid ranges
11. No non-finite values in measured metrics
12. Statistical validity: means have std computed from correct seed count
13. Checkpoint identity: every record revalidates against its full checkpoint
14. Diagnostic input identity: the exact ordered input bytes are revalidated
15. Evaluator source identity: runtime bytes match the claimed clean checkout

Usage:
    buck2 run //buiksat_trm:audit_phase4_paper_ready -- \
        --results_dir results/paper_ready/phase4_2x2_norm_ablation/v3 \
        --checkpoint_dir results/phase4_2x2_norm_ablation
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from scripts.phase4_result_schema import (  # noqa: E402
    Phase4SummaryValidationError,
    validate_phase4_summary,
)
from scripts.phase4_checkpoint import (  # noqa: E402
    Phase4CheckpointError,
    phase4_checkpoint_relpath,
    verify_phase4_summary_checkpoints,
)
from scripts.phase4_diagnostic_inputs import (  # noqa: E402
    Phase4DiagnosticInputError,
    verify_phase4_diagnostic_inputs,
)
from scripts.phase4_source import (  # noqa: E402
    PHASE4_AUDIT_SOURCE_PROFILE,
    Phase4SourceError,
    phase4_evaluator_source_manifest_sha256,
    require_phase4_runtime_attestation,
    resolve_phase4_source_roots,
    verify_phase4_producer_source,
    verify_phase4_runtime_sources,
)
from utils.run_identity import (  # noqa: E402
    RunIdentityError,
    discover_clean_git_source,
)


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
    checkpoint_dir: Path,
) -> AuditResult:
    """Check 3: Every design cell has its exact full checkpoint path."""
    missing = []

    for cond in EXPECTED_CONDITIONS:
        for seed in EXPECTED_SEEDS:
            checkpoint_path = checkpoint_dir / phase4_checkpoint_relpath(cond, seed)
            if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
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


def check_no_mislabeled(checkpoint_dir: Path) -> AuditResult:
    """Check 4: Checkpoint dirs match expected naming pattern."""
    unexpected = []
    expected_pattern = {f"{c}_s{s}" for c in EXPECTED_CONDITIONS for s in EXPECTED_SEEDS}

    if checkpoint_dir.exists():
        for item in checkpoint_dir.iterdir():
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
    """Check 5: summary.json exists and is readable JSON."""
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

    return (
        AuditResult("Summary Exists", True, "summary.json exists and is readable JSON"),
        summary,
    )


def check_publication_schema(summary: Dict[str, Any]) -> AuditResult:
    """Check 6: summary is a publishable Phase 4 schema-v3 artifact."""
    try:
        validate_phase4_summary(summary)
    except Phase4SummaryValidationError as error:
        return AuditResult(
            "Publication Schema",
            False,
            "Historical/schema-less summaries are non-publishable and are not "
            f"migrated: {error}",
        )
    return AuditResult(
        "Publication Schema",
        True,
        "Strict schema version 3 is valid; checkpoint identity is explicit",
    )


def check_all_runs_present(summary: Dict[str, Any]) -> AuditResult:
    """Check 7: All 12 runs present in summary."""
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
    """Check 8: CLAIMS.md exists and is non-empty."""
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
    """Check 9: PROVENANCE.md exists and references configs."""
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
    """Check 10: Measured metrics are within their valid ranges."""
    if not summary:
        return AuditResult("Metric Bounds", False, "No summary to check")

    aggregates = summary.get("aggregates", [])
    issues = []

    for agg in aggregates:
        cond = agg.get("condition", "unknown")

        # Var(V) should be non-negative and not huge
        var_v = agg["var_V_mean"]
        if var_v < 0 or var_v > 1000:
            issues.append(f"{cond}: var_V_mean={var_v} out of bounds [0, 1000]")

        # Argmax agreement should be in [0, 1]
        argmax = agg["argmax_n4_mean"]
        if argmax < 0 or argmax > 1:
            issues.append(f"{cond}: argmax_n4_mean={argmax} out of bounds [0, 1]")

        projection_rate = agg["projection_active_rate_mean"]
        if projection_rate < 0 or projection_rate > 1:
            issues.append(
                f"{cond}: projection_active_rate_mean={projection_rate} "
                "out of bounds [0, 1]"
            )

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
    """Check 11: No non-finite values in measured metrics."""
    if not summary:
        return AuditResult("Finite Metrics", False, "No summary to check")

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
            "Finite Metrics",
            False,
            f"Found NaN/Inf: {issues[:3]}{'...' if len(issues) > 3 else ''}",
        )

    return AuditResult("Finite Metrics", True, "All measured values are finite")


def check_statistical_validity(summary: Dict[str, Any]) -> AuditResult:
    """Check 12: Aggregates computed from correct number of seeds."""
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


def check_checkpoint_identities(
    summary: Dict[str, Any],
    checkpoint_dir: Path,
    config_dir: Path,
    expected_producer_source: Dict[str, Any],
) -> AuditResult:
    """Check 13: Rehash and strictly reload every recorded full checkpoint."""

    try:
        count = verify_phase4_summary_checkpoints(
            summary,
            checkpoint_dir,
            config_dir,
            expected_producer_source=expected_producer_source,
            device="cpu",
        )
    except (OSError, Phase4CheckpointError) as error:
        return AuditResult("Checkpoint Identity", False, str(error))
    return AuditResult(
        "Checkpoint Identity",
        True,
        f"Revalidated {count} checkpoint, model-state, seed, and config identities",
    )


def check_diagnostic_input_identity(
    summary: Dict[str, Any],
    data_dir: Path,
) -> AuditResult:
    """Check 14: Rehash the exact ordered finite-diagnostic input population."""

    try:
        count = verify_phase4_diagnostic_inputs(summary, data_dir)
    except (OSError, Phase4DiagnosticInputError) as error:
        return AuditResult("Diagnostic Input Identity", False, str(error))
    return AuditResult(
        "Diagnostic Input Identity",
        True,
        f"Revalidated the exact ordered {count}-state diagnostic population",
    )


def check_evaluator_source_identity(
    summary: Dict[str, Any],
    project_root: Path,
    expected_runtime_sha256: str,
) -> AuditResult:
    """Check 15: Bind evaluator bytes and commit to the clean checkout."""

    try:
        source_identity = discover_clean_git_source(project_root)
        expected_digest = phase4_evaluator_source_manifest_sha256(project_root)
    except (OSError, Phase4SourceError, RunIdentityError) as error:
        return AuditResult("Evaluator Source Identity", False, str(error))
    if summary.get("evaluator_git_commit") != source_identity["git_commit"]:
        return AuditResult(
            "Evaluator Source Identity",
            False,
            "Summary evaluator commit differs from the clean project checkout",
        )
    if summary.get("evaluator_source_manifest_sha256") != expected_digest:
        return AuditResult(
            "Evaluator Source Identity",
            False,
            "Summary evaluator source digest differs from the project checkout",
        )
    if summary.get("evaluator_runtime_artifact_sha256") != expected_runtime_sha256:
        return AuditResult(
            "Evaluator Source Identity",
            False,
            "Summary evaluator runtime differs from the authorized PAR digest",
        )
    return AuditResult(
        "Evaluator Source Identity",
        True,
        "Evaluator source digest and commit match the clean project checkout",
    )


def run_audit(
    results_dir: Path,
    config_dir: Path,
    checkpoint_dir: Path,
    data_dir: Path,
    project_root: Path,
    expected_producer_source: Dict[str, Any],
    expected_evaluator_runtime_sha256: str,
) -> Tuple[List[AuditResult], bool]:
    """Run all audit checks."""
    results = []

    # Checks 1-2: Config integrity
    results.append(check_config_integrity(config_dir))
    results.append(check_correct_toggles(config_dir))

    # Checks 5-6: Load the summary, then establish publication eligibility.
    summary_result, summary = check_summary_exists(results_dir)
    publication_schema_result = check_publication_schema(summary)

    # Check 3-4: Check the distinct full-checkpoint root, not publication output.
    results.append(check_seeds_present(checkpoint_dir))
    results.append(check_no_mislabeled(checkpoint_dir))

    # Add summary check result
    results.append(summary_result)
    results.append(publication_schema_result)

    # Checks 7-9: Documentation and completeness
    results.append(check_all_runs_present(summary))
    results.append(check_claims_exists(results_dir))
    results.append(check_provenance_exists(results_dir, config_dir))

    # Checks 10-12: Only a validated schema may enter data-quality consumers.
    if publication_schema_result.passed:
        results.append(check_metric_bounds(summary))
        results.append(check_no_nan_inf(summary))
        results.append(check_statistical_validity(summary))
        results.append(
            check_checkpoint_identities(
                summary,
                checkpoint_dir,
                config_dir,
                expected_producer_source,
            )
        )
        results.append(check_diagnostic_input_identity(summary, data_dir))
        results.append(
            check_evaluator_source_identity(
                summary,
                project_root,
                expected_evaluator_runtime_sha256,
            )
        )
    else:
        for name in (
            "Metric Bounds",
            "Finite Metrics",
            "Statistical Validity",
            "Checkpoint Identity",
            "Diagnostic Input Identity",
            "Evaluator Source Identity",
        ):
            results.append(
                AuditResult(
                    name,
                    False,
                    "Publication schema is invalid; measured fields were not consumed",
                )
            )

    all_passed = all(r.passed for r in results)
    return results, all_passed


def write_audit_md(results_dir: Path, results: List[AuditResult], all_passed: bool):
    """Write AUDIT.md with results."""
    publication_schema = next(
        (result for result in results if result.name == "Publication Schema"),
        None,
    )
    if publication_schema is None or not publication_schema.passed:
        print(
            "Not writing AUDIT.md: the input is a historical or invalid "
            "publication schema"
        )
        return

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


def main(*, runtime_attestation: Dict[str, Any] | None = None):
    try:
        attestation = require_phase4_runtime_attestation(
            runtime_attestation,
            PHASE4_AUDIT_SOURCE_PROFILE,
        )
    except Phase4SourceError as error:
        print(f"ERROR: Audit runtime source is not authenticated: {error}")
        return 1
    parser = argparse.ArgumentParser(description="Phase 4 Audit Script")
    parser.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="Exact clean implementation checkout used by the evaluator",
    )
    parser.add_argument(
        "--fbcode_root",
        type=str,
        required=True,
        help="fbcode root whose buiksat_trm cell resolves to project_root",
    )
    parser.add_argument(
        "--producer_project_root",
        type=str,
        required=True,
        help="Exact clean implementation checkout used for Phase 4 training",
    )
    parser.add_argument(
        "--expected_producer_git_commit",
        type=str,
        required=True,
        help="Authorized Phase 4 training commit",
    )
    parser.add_argument(
        "--expected_evaluator_runtime_sha256",
        required=True,
        help="Externally authorized evaluator PAR SHA-256",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help=(
            "Path to results directory "
            "(e.g., results/paper_ready/phase4_2x2_norm_ablation/v3)"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Trusted root containing the 12 full Phase 4 checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Trusted root containing the recorded Phase 4 diagnostic arrays",
    )

    args = parser.parse_args()
    if re.fullmatch(
        r"[0-9a-f]{64}",
        args.expected_evaluator_runtime_sha256,
    ) is None:
        print("ERROR: Expected evaluator runtime SHA-256 is invalid.")
        return 1

    try:
        project_root, _ = resolve_phase4_source_roots(
            args.project_root,
            args.fbcode_root,
        )
        runtime_source_digest = verify_phase4_runtime_sources(
            project_root,
            PHASE4_AUDIT_SOURCE_PROFILE,
        )
        project_identity = discover_clean_git_source(project_root)
        if (
            runtime_source_digest != attestation["source_manifest_sha256"]
            or project_identity["git_commit"]
            != attestation["source_git_commit"]
        ):
            raise Phase4SourceError(
                "Audit checkout differs from the pre-import runtime attestation."
            )
        producer_project_root = Path(
            args.producer_project_root
        ).expanduser().resolve(strict=True)
        expected_producer_source = verify_phase4_producer_source(
            producer_project_root,
            args.expected_producer_git_commit,
        )
    except (OSError, Phase4SourceError) as error:
        print(f"ERROR: Audit runtime source is not authenticated: {error}")
        return 1

    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    config_dir = (
        producer_project_root / "configs" / "phase4_2x2_norm_ablation"
    )

    print("=" * 60)
    print("Phase 4: 2×2 Norm Ablation - Audit")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Config dir: {config_dir}")
    print()

    results, all_passed = run_audit(
        results_dir,
        config_dir,
        checkpoint_dir,
        data_dir,
        project_root,
        expected_producer_source,
        args.expected_evaluator_runtime_sha256,
    )

    try:
        final_runtime_source_digest = verify_phase4_runtime_sources(
            project_root,
            PHASE4_AUDIT_SOURCE_PROFILE,
        )
        if final_runtime_source_digest != runtime_source_digest:
            raise Phase4SourceError(
                "Audit runtime source changed during validation."
            )
        if verify_phase4_producer_source(
            producer_project_root,
            args.expected_producer_git_commit,
        ) != expected_producer_source:
            raise Phase4SourceError(
                "Producer source identity changed during audit."
            )
        final_identity = discover_clean_git_source(project_root)
        summary_result, final_summary = check_summary_exists(results_dir)
        if not summary_result.passed or final_summary.get(
            "evaluator_git_commit"
        ) != final_identity["git_commit"]:
            raise Phase4SourceError(
                "Project or summary source identity changed during audit."
            )
        results.append(
            AuditResult(
                "Audit Runtime Source",
                True,
                "Audit runtime bytes match the explicit project checkout",
            )
        )
    except (OSError, Phase4SourceError, RunIdentityError) as error:
        results.append(AuditResult("Audit Runtime Source", False, str(error)))
    all_passed = all(result.passed for result in results)

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
