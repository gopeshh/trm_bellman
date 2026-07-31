#!/usr/bin/env python3
"""
Post-process Experiment 1 v4 Results for ICML Reviewer Defensibility.

Fixes:
A) Saturation metric (column name + semantics)
B) MISMATCH vs INCREMENTAL delta definitions
C) Wilson 95% CI for argmax agreement
D) B1 closure batch reporting
E) Consistency validation checks

Usage:
    python scripts/postprocess_exp1_v4_1.py --results_dir results/validation/exp1_v4 \
        --out_dir results/plot_data/exp1_v4.1
"""

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Wilson Score Confidence Interval
# =============================================================================

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Compute Wilson score 95% CI for a binomial proportion.

    Args:
        successes: Number of successes
        n: Total number of trials
        z: Z-score for CI (1.96 = 95%, 2.576 = 99%)

    Returns:
        (lower, upper) bounds of CI, both in [0, 1]
    """
    if n == 0:
        return 0.0, 1.0

    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def wilson_from_values(values: List[int]) -> Tuple[float, float, float]:
    """
    Compute Wilson CI from a list of 0/1 values.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if not values:
        return 0.0, 0.0, 1.0

    n = len(values)
    successes = sum(values)
    mean = successes / n
    lower, upper = wilson_ci(successes, n)

    return mean, lower, upper


# =============================================================================
# Data Loading
# =============================================================================

def load_per_state_csv(csv_path: str) -> List[Dict[str, float | str]]:
    """Load per-state CSV with proper type handling."""
    rows: List[Dict[str, float | str]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float | str] = {}
            for k, v in row.items():
                if k is None or v is None:
                    continue
                key = str(k)
                try:
                    parsed[key] = float(v)
                except ValueError:
                    parsed[key] = v
            rows.append(parsed)
    return rows


def collect_data(
    base_dir: Path,
    seeds: List[int],
    batch: str,
    radius_dirs: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Collect per-state data from CSV files.

    Args:
        base_dir: Base results directory
        seeds: List of seed values
        batch: "b0" or "b1"
        radius_dirs: Optional list of radius subdirs (e.g., ["R10", "R100", "R0"])

    Returns:
        Dict mapping model -> (n1, n2) -> metric -> list of values
    """
    data = {"model_a": {}, "model_b": {}}

    for model in ["model_a", "model_b"]:
        for seed in seeds:
            if radius_dirs:
                for r_dir in radius_dirs:
                    csv_path = base_dir / f"seed{seed}" / r_dir / f"{model}_{batch}_per_state.csv"
                    if csv_path.exists():
                        _load_rows_into_data(csv_path, data[model])
            else:
                csv_path = base_dir / f"seed{seed}" / f"{model}_{batch}_per_state.csv"
                if csv_path.exists():
                    _load_rows_into_data(csv_path, data[model])

    return data


def _load_rows_into_data(csv_path: Path, model_data: Dict):
    """Load rows from a CSV into the model data structure."""
    rows = load_per_state_csv(str(csv_path))

    for row in rows:
        n1 = int(row.get("n1", 0))
        n2 = int(row.get("n2", 0))
        if n1 == 0 or n2 == 0:
            continue

        key = (n1, n2)
        if key not in model_data:
            model_data[key] = {
                "delta_V": [], "delta_pi": [], "delta_z": [],
                "argmax_agree": [], "saturated": [],
                "z_pre_norm": [], "z_post_norm": [],
            }

        for metric in model_data[key]:
            # Handle column name variations
            col_name = metric
            if col_name not in row and metric == "saturated" and "saturation" in row:
                col_name = "saturation"

            if col_name in row:
                val = float(row[col_name])
                # Skip N/A saturation values (R=0 case)
                if metric == "saturated" and val < 0:
                    continue
                model_data[key][metric].append(val)


# =============================================================================
# Delta Definition Filters
# =============================================================================

def filter_mismatch_drift(data: Dict, n_train: int) -> Dict:
    """
    Filter to MISMATCH DRIFT: Δ(n_train, n2) only.

    This measures how much predictions drift from training depth to
    evaluation depth.
    """
    filtered = {}
    for (n1, n2), metrics in data.items():
        if n1 == n_train:
            filtered[(n1, n2)] = metrics
    return filtered


def filter_incremental_drift(data: Dict, n_train: int) -> Dict:
    """
    Filter to INCREMENTAL DRIFT: Δ(n_prev, n_next) for consecutive depths only.

    For depths [2, 4, 8, 16], this gives (2,4), (4,8), (8,16).
    """
    # Get all unique n2 values
    all_n2 = sorted(set(n2 for (n1, n2) in data.keys()))
    all_depths = sorted(set([n_train] + all_n2))

    # Build consecutive pairs
    consecutive_pairs = []
    for i in range(len(all_depths) - 1):
        consecutive_pairs.append((all_depths[i], all_depths[i + 1]))

    filtered = {}
    for (n1, n2), metrics in data.items():
        if (n1, n2) in consecutive_pairs:
            filtered[(n1, n2)] = metrics

    return filtered


# =============================================================================
# Aggregation with Proper Statistics
# =============================================================================

@dataclass
class MetricStats:
    mean: float
    std: float
    n: int
    # For argmax_agree (Wilson CI)
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    # For z norms (p95/max)
    p95: Optional[float] = None
    max_val: Optional[float] = None


def compute_stats(values: List[float], is_binary: bool = False) -> MetricStats:
    """Compute statistics for a metric."""
    if not values:
        return MetricStats(0.0, 0.0, 0)

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0

    if is_binary:
        # Use Wilson CI for binary metrics
        int_values = [int(v) for v in values]
        _, ci_lower, ci_upper = wilson_from_values(int_values)
        return MetricStats(mean, std, n, ci_lower, ci_upper)

    # Compute p95 and max for norm metrics
    sorted_vals = sorted(values)
    p95_idx = int(0.95 * n)
    p95 = sorted_vals[p95_idx] if p95_idx < n else sorted_vals[-1]
    max_val = max(values)

    return MetricStats(mean, std, n, p95=p95, max_val=max_val)


def aggregate_by_n2(
    data: Dict[Tuple[int, int], Dict],
    n_train: int,
) -> Dict[int, Dict[str, MetricStats]]:
    """
    Aggregate by n2 (final depth) for comparison across models.
    """
    by_n2 = {}

    for (n1, n2), metrics in data.items():
        if n2 not in by_n2:
            by_n2[n2] = {m: [] for m in metrics.keys()}

        for metric, values in metrics.items():
            by_n2[n2][metric].extend(values)

    # Compute stats
    result = {}
    for n2, metrics in by_n2.items():
        result[n2] = {}
        for metric, values in metrics.items():
            is_binary = metric == "argmax_agree"
            result[n2][metric] = compute_stats(values, is_binary)

    return result


# =============================================================================
# Consistency Checks
# =============================================================================

class ConsistencyChecker:
    """Run consistency checks on the data."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def check_kl_nonnegative(self, data: Dict):
        """KL divergence should be >= 0."""
        for (n1, n2), metrics in data.items():
            for val in metrics.get("delta_pi", []):
                if val < 0:
                    self.errors.append(f"KL divergence < 0: {val} at ({n1}, {n2})")
                if not math.isfinite(val):
                    self.errors.append(f"KL divergence not finite: {val} at ({n1}, {n2})")

    def check_deltas_nonnegative(self, data: Dict):
        """Δ_V and Δ_z should be >= 0."""
        for (n1, n2), metrics in data.items():
            for val in metrics.get("delta_V", []):
                if val < 0:
                    self.errors.append(f"delta_V < 0: {val} at ({n1}, {n2})")
            for val in metrics.get("delta_z", []):
                if val < 0:
                    self.errors.append(f"delta_z < 0: {val} at ({n1}, {n2})")

    def check_argmax_in_range(self, data: Dict):
        """argmax_agree should be in [0, 1]."""
        for (n1, n2), metrics in data.items():
            for val in metrics.get("argmax_agree", []):
                if val < 0 or val > 1:
                    self.errors.append(f"argmax_agree not in [0,1]: {val} at ({n1}, {n2})")

    def check_saturation_consistency(self, data: Dict, radius: float):
        """If sat_mean is near 1 then z_post_norm mean should ≈ R."""
        for (n1, n2), metrics in data.items():
            sat_vals = metrics.get("saturated", [])
            z_post_vals = metrics.get("z_post_norm", [])

            if not sat_vals or not z_post_vals:
                continue

            sat_mean = statistics.mean(sat_vals) if sat_vals else 0
            z_post_mean = statistics.mean(z_post_vals) if z_post_vals else 0

            if radius > 0:
                if sat_mean > 0.9:
                    # High saturation: z_post should be ≈ R
                    if abs(z_post_mean - radius) > radius * 0.1:
                        self.warnings.append(
                            f"High saturation ({sat_mean:.2f}) but z_post_norm "
                            f"({z_post_mean:.2f}) not near R ({radius}) at ({n1}, {n2})"
                        )
                elif sat_mean < 0.1:
                    # Low saturation: z_pre ≈ z_post
                    z_pre_vals = metrics.get("z_pre_norm", [])
                    if z_pre_vals:
                        z_pre_mean = statistics.mean(z_pre_vals)
                        if abs(z_pre_mean - z_post_mean) > 1.0:
                            self.warnings.append(
                                f"Low saturation ({sat_mean:.2f}) but z_pre "
                                f"({z_pre_mean:.2f}) != z_post ({z_post_mean:.2f}) "
                                f"at ({n1}, {n2})"
                            )

    def run_all(self, data: Dict, radius: float = 10.0):
        """Run all consistency checks."""
        self.check_kl_nonnegative(data)
        self.check_deltas_nonnegative(data)
        self.check_argmax_in_range(data)
        self.check_saturation_consistency(data, radius)

        return len(self.errors) == 0

    def report(self) -> str:
        """Generate a report of all issues."""
        lines = []
        if self.errors:
            lines.append("## ERRORS (hard fail):")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("## WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if not self.errors and not self.warnings:
            lines.append("All consistency checks passed.")
        return "\n".join(lines)


# =============================================================================
# Output Writers
# =============================================================================

def write_mismatch_csv(
    data_a: Dict[int, Dict[str, MetricStats]],
    data_b: Dict[int, Dict[str, MetricStats]],
    out_path: Path,
    n_train: int,
):
    """Write MISMATCH DRIFT aggregated CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [
            "model", "n_train", "n2", "metric",
            "mean", "std", "n", "ci_lower", "ci_upper", "p95", "max"
        ]
        writer.writerow(headers)

        for model, data in [("model_a", data_a), ("model_b", data_b)]:
            for n2 in sorted(data.keys()):
                for metric, stats in data[n2].items():
                    row = [
                        model, n_train, n2, metric,
                        f"{stats.mean:.6f}", f"{stats.std:.6f}", stats.n,
                        f"{stats.ci_lower:.4f}" if stats.ci_lower is not None else "",
                        f"{stats.ci_upper:.4f}" if stats.ci_upper is not None else "",
                        f"{stats.p95:.4f}" if stats.p95 is not None else "",
                        f"{stats.max_val:.4f}" if stats.max_val is not None else "",
                    ]
                    writer.writerow(row)

    print(f"[Write] {out_path}")


def write_mismatch_markdown(
    data_a: Dict[int, Dict[str, MetricStats]],
    data_b: Dict[int, Dict[str, MetricStats]],
    out_path: Path,
    n_train: int,
    seeds: List[int],
    batch: str,
):
    """Write MISMATCH DRIFT markdown summary."""
    with open(out_path, "w") as f:
        f.write(f"# Unroll Sensitivity: MISMATCH DRIFT ({batch.upper()})\n\n")
        f.write(f"**Delta definition**: Δ(n_train={n_train}, n₂) for n₂ ∈ depths\n\n")
        f.write(f"**Seeds**: {seeds}\n\n")
        f.write("This measures how much predictions drift from training depth to evaluation depth.\n\n")

        n2_values = sorted(set(data_a.keys()) & set(data_b.keys()))

        # Delta metrics with mean±std
        for metric in ["delta_V", "delta_pi", "delta_z"]:
            f.write(f"## {metric}\n\n")
            f.write("| n₂ | Model A (mean±std) | Model B (mean±std) |\n")
            f.write("|----|--------------------|--------------------|")

            for n2 in n2_values:
                stats_a = data_a.get(n2, {}).get(metric, MetricStats(0, 0, 0))
                stats_b = data_b.get(n2, {}).get(metric, MetricStats(0, 0, 0))
                f.write(f"\n| {n2} | {stats_a.mean:.4f}±{stats_a.std:.4f} | {stats_b.mean:.4f}±{stats_b.std:.4f} |")

            f.write("\n\n")

        # Argmax agreement with Wilson CI
        f.write("## argmax_agree (with Wilson 95% CI)\n\n")
        f.write("| n₂ | Model A [mean, 95% CI] | Model B [mean, 95% CI] |\n")
        f.write("|----|------------------------|------------------------|")

        for n2 in n2_values:
            stats_a = data_a.get(n2, {}).get("argmax_agree", MetricStats(0, 0, 0, 0, 1))
            stats_b = data_b.get(n2, {}).get("argmax_agree", MetricStats(0, 0, 0, 0, 1))
            a_str = f"{stats_a.mean:.4f} [{stats_a.ci_lower:.4f}, {stats_a.ci_upper:.4f}]"
            b_str = f"{stats_b.mean:.4f} [{stats_b.ci_lower:.4f}, {stats_b.ci_upper:.4f}]"
            f.write(f"\n| {n2} | {a_str} | {b_str} |")

        f.write("\n\n")

        # Z norm stats
        f.write("## z_pre_norm (mean / p95 / max)\n\n")
        f.write("| n₂ | Model A | Model B |\n")
        f.write("|----|---------|---------|")

        for n2 in n2_values:
            stats_a = data_a.get(n2, {}).get("z_pre_norm", MetricStats(0, 0, 0))
            stats_b = data_b.get(n2, {}).get("z_pre_norm", MetricStats(0, 0, 0))
            a_str = f"{stats_a.mean:.2f} / {stats_a.p95 or 0:.2f} / {stats_a.max_val or 0:.2f}"
            b_str = f"{stats_b.mean:.2f} / {stats_b.p95 or 0:.2f} / {stats_b.max_val or 0:.2f}"
            f.write(f"\n| {n2} | {a_str} | {b_str} |")

        f.write("\n\n")

        f.write("## z_post_norm (mean / p95 / max)\n\n")
        f.write("| n₂ | Model A | Model B |\n")
        f.write("|----|---------|---------|")

        for n2 in n2_values:
            stats_a = data_a.get(n2, {}).get("z_post_norm", MetricStats(0, 0, 0))
            stats_b = data_b.get(n2, {}).get("z_post_norm", MetricStats(0, 0, 0))
            a_str = f"{stats_a.mean:.2f} / {stats_a.p95 or 0:.2f} / {stats_a.max_val or 0:.2f}"
            b_str = f"{stats_b.mean:.2f} / {stats_b.p95 or 0:.2f} / {stats_b.max_val or 0:.2f}"
            f.write(f"\n| {n2} | {a_str} | {b_str} |")

        f.write("\n\n")

        # Saturation rate
        f.write("## saturation (mean ± std)\n\n")
        f.write("| n₂ | Model A | Model B |\n")
        f.write("|----|---------|---------|")

        for n2 in n2_values:
            stats_a = data_a.get(n2, {}).get("saturated", MetricStats(0, 0, 0))
            stats_b = data_b.get(n2, {}).get("saturated", MetricStats(0, 0, 0))
            if stats_a.n > 0:
                a_str = f"{stats_a.mean:.4f}±{stats_a.std:.4f} (n={stats_a.n})"
            else:
                a_str = "N/A (projection disabled)"
            if stats_b.n > 0:
                b_str = f"{stats_b.mean:.4f}±{stats_b.std:.4f} (n={stats_b.n})"
            else:
                b_str = "N/A (projection disabled)"
            f.write(f"\n| {n2} | {a_str} | {b_str} |")

        f.write("\n")

    print(f"[Write] {out_path}")


def write_incremental_markdown(
    data_a: Dict[int, Dict[str, MetricStats]],
    data_b: Dict[int, Dict[str, MetricStats]],
    out_path: Path,
    n_train: int,
    seeds: List[int],
    batch: str,
    pairs: List[Tuple[int, int]],
):
    """Write INCREMENTAL DRIFT markdown summary."""
    with open(out_path, "w") as f:
        f.write(f"# Unroll Sensitivity: INCREMENTAL DRIFT ({batch.upper()})\n\n")
        f.write("**Delta definition**: Δ(n_prev, n_next) for consecutive depth pairs\n\n")
        f.write(f"**Seeds**: {seeds}\n\n")
        f.write("This measures the drift between consecutive unroll steps.\n\n")

        # Delta metrics with mean±std
        for metric in ["delta_V", "delta_pi", "delta_z"]:
            f.write(f"## {metric}\n\n")
            f.write("| Pair | Model A (mean±std) | Model B (mean±std) |\n")
            f.write("|------|--------------------|--------------------|")

            for (n1, n2) in pairs:
                # Use n2 as key since we aggregate by destination depth
                stats_a = data_a.get(n2, {}).get(metric, MetricStats(0, 0, 0))
                stats_b = data_b.get(n2, {}).get(metric, MetricStats(0, 0, 0))
                f.write(f"\n| {n1}→{n2} | {stats_a.mean:.4f}±{stats_a.std:.4f} | {stats_b.mean:.4f}±{stats_b.std:.4f} |")

            f.write("\n\n")

        # Argmax agreement with Wilson CI
        f.write("## argmax_agree (with Wilson 95% CI)\n\n")
        f.write("| Pair | Model A [mean, 95% CI] | Model B [mean, 95% CI] |\n")
        f.write("|------|------------------------|------------------------|")

        for (n1, n2) in pairs:
            stats_a = data_a.get(n2, {}).get("argmax_agree", MetricStats(0, 0, 0, 0, 1))
            stats_b = data_b.get(n2, {}).get("argmax_agree", MetricStats(0, 0, 0, 0, 1))
            a_str = f"{stats_a.mean:.4f} [{stats_a.ci_lower:.4f}, {stats_a.ci_upper:.4f}]"
            b_str = f"{stats_b.mean:.4f} [{stats_b.ci_lower:.4f}, {stats_b.ci_upper:.4f}]"
            f.write(f"\n| {n1}→{n2} | {a_str} | {b_str} |")

        f.write("\n")

    print(f"[Write] {out_path}")


def write_radius_sweep_markdown(
    data: Dict[float, Dict[str, Dict[str, MetricStats]]],
    out_path: Path,
    seeds: List[int],
    batch: str,
    n_train: int,
    radius_n2: int,
):
    """Write radius sweep markdown with fixed saturation."""
    with open(out_path, "w") as f:
        f.write(f"# Radius Sweep Summary ({batch.upper()})\n\n")
        f.write(f"**Seeds**: {seeds}\n\n")
        depth_mult = radius_n2 // n_train if radius_n2 % n_train == 0 else radius_n2 / n_train
        f.write(
            f"**Delta definition**: fixed mismatch Δ(n_train={n_train}, n₂={radius_n2}) "
            f"({depth_mult}× depth), pooled across all per-state rows.\n\n"
        )

        radii = sorted(data.keys())

        for metric in ["delta_V", "delta_z", "argmax_agree"]:
            f.write(f"## {metric}\n\n")
            if metric == "argmax_agree":
                f.write("| Radius | Model A [mean, 95% CI] | Model B [mean, 95% CI] |\n")
                f.write("|--------|------------------------|------------------------|")
            else:
                f.write("| Radius | Model A (mean±std) | Model B (mean±std) |\n")
                f.write("|--------|--------------------|--------------------|")

            for R in radii:
                R_label = f"R={int(R)}" if R > 0 else "disabled"
                stats_a = data[R]["model_a"].get(metric, MetricStats(0, 0, 0))
                stats_b = data[R]["model_b"].get(metric, MetricStats(0, 0, 0))

                if metric == "argmax_agree":
                    a_str = f"{stats_a.mean:.4f} [{stats_a.ci_lower:.4f}, {stats_a.ci_upper:.4f}]"
                    b_str = f"{stats_b.mean:.4f} [{stats_b.ci_lower:.4f}, {stats_b.ci_upper:.4f}]"
                else:
                    a_str = f"{stats_a.mean:.4f}±{stats_a.std:.4f}"
                    b_str = f"{stats_b.mean:.4f}±{stats_b.std:.4f}"

                f.write(f"\n| {R_label} | {a_str} | {b_str} |")

            f.write("\n\n")

        # Saturation with proper n counts
        f.write("## saturation (with sample count)\n\n")
        f.write("| Radius | Model A (mean±std, n) | Model B (mean±std, n) |\n")
        f.write("|--------|----------------------|----------------------|")

        for R in radii:
            R_label = f"R={int(R)}" if R > 0 else "disabled"
            stats_a = data[R]["model_a"].get("saturated", MetricStats(0, 0, 0))
            stats_b = data[R]["model_b"].get("saturated", MetricStats(0, 0, 0))

            if stats_a.n > 0:
                a_str = f"{stats_a.mean:.4f}±{stats_a.std:.4f} (n={stats_a.n})"
            else:
                a_str = "N/A (disabled)"
            if stats_b.n > 0:
                b_str = f"{stats_b.mean:.4f}±{stats_b.std:.4f} (n={stats_b.n})"
            else:
                b_str = "N/A (disabled)"

            f.write(f"\n| {R_label} | {a_str} | {b_str} |")

        f.write("\n\n")

        # Z norms
        f.write("## z_pre_norm (mean / p95 / max)\n\n")
        f.write("| Radius | Model A | Model B |\n")
        f.write("|--------|---------|---------|")

        for R in radii:
            R_label = f"R={int(R)}" if R > 0 else "disabled"
            stats_a = data[R]["model_a"].get("z_pre_norm", MetricStats(0, 0, 0))
            stats_b = data[R]["model_b"].get("z_pre_norm", MetricStats(0, 0, 0))
            a_str = f"{stats_a.mean:.2f} / {stats_a.p95 or 0:.2f} / {stats_a.max_val or 0:.2f}"
            b_str = f"{stats_b.mean:.2f} / {stats_b.p95 or 0:.2f} / {stats_b.max_val or 0:.2f}"
            f.write(f"\n| {R_label} | {a_str} | {b_str} |")

        f.write("\n\n")

        f.write("## z_post_norm (mean / p95 / max)\n\n")
        f.write("| Radius | Model A | Model B |\n")
        f.write("|--------|---------|---------|")

        for R in radii:
            R_label = f"R={int(R)}" if R > 0 else "disabled"
            stats_a = data[R]["model_a"].get("z_post_norm", MetricStats(0, 0, 0))
            stats_b = data[R]["model_b"].get("z_post_norm", MetricStats(0, 0, 0))
            a_str = f"{stats_a.mean:.2f} / {stats_a.p95 or 0:.2f} / {stats_a.max_val or 0:.2f}"
            b_str = f"{stats_b.mean:.2f} / {stats_b.p95 or 0:.2f} / {stats_b.max_val or 0:.2f}"
            f.write(f"\n| {R_label} | {a_str} | {b_str} |")

        f.write("\n")

    print(f"[Write] {out_path}")


# =============================================================================
# Main Processing
# =============================================================================

def process_unroll_sensitivity(
    base_dir: Path,
    seeds: List[int],
    n_train: int,
    out_dir: Path,
    batch: str,
):
    """Process unroll sensitivity data for a batch."""
    print(f"\n=== Processing Unroll Sensitivity ({batch.upper()}) ===")

    # Collect raw data
    data = collect_data(base_dir, seeds, batch)

    # Run consistency checks
    print("[Check] Running consistency validation...")
    checker = ConsistencyChecker()
    for model in ["model_a", "model_b"]:
        checker.run_all(data[model], radius=10.0)

    if checker.errors:
        print("[FAIL] Consistency check failed!")
        print(checker.report())
        return False

    if checker.warnings:
        print("[WARN] Consistency warnings:")
        print(checker.report())

    # MISMATCH DRIFT
    print("[Process] Computing MISMATCH DRIFT...")
    mismatch_a = filter_mismatch_drift(data["model_a"], n_train)
    mismatch_b = filter_mismatch_drift(data["model_b"], n_train)

    agg_mismatch_a = aggregate_by_n2(mismatch_a, n_train)
    agg_mismatch_b = aggregate_by_n2(mismatch_b, n_train)

    write_mismatch_csv(agg_mismatch_a, agg_mismatch_b,
                       out_dir / f"unroll_sensitivity_{batch}_mismatch.csv", n_train)
    write_mismatch_markdown(agg_mismatch_a, agg_mismatch_b,
                            out_dir / f"unroll_sensitivity_{batch}_mismatch.md",
                            n_train, seeds, batch)

    # INCREMENTAL DRIFT
    print("[Process] Computing INCREMENTAL DRIFT...")
    incr_a = filter_incremental_drift(data["model_a"], n_train)
    incr_b = filter_incremental_drift(data["model_b"], n_train)

    # Get the consecutive pairs
    all_depths = sorted(set([n_train] + [n2 for (n1, n2) in data["model_a"].keys()]))
    pairs = [(all_depths[i], all_depths[i+1]) for i in range(len(all_depths)-1)]

    agg_incr_a = aggregate_by_n2(incr_a, n_train)
    agg_incr_b = aggregate_by_n2(incr_b, n_train)

    write_incremental_markdown(agg_incr_a, agg_incr_b,
                               out_dir / f"unroll_sensitivity_{batch}_incremental.md",
                               n_train, seeds, batch, pairs)

    return True


def process_radius_sweep(
    base_dir: Path,
    seeds: List[int],
    radii: List[float],
    n_train: int,
    radius_n2: int,
    out_dir: Path,
    batch: str,
):
    """Process radius sweep data for a batch."""
    print(f"\n=== Processing Radius Sweep ({batch.upper()}) ===")
    print(f"[Process] Using fixed mismatch Δ(n_train={n_train}, n₂={radius_n2})")

    # Collect data per radius
    all_data = {}

    for R in radii:
        R_str = f"R{int(R)}" if R == int(R) else f"R{R}"
        print(f"[Load] Loading R={R}...")

        data = {"model_a": {}, "model_b": {}}

        for model in ["model_a", "model_b"]:
            for seed in seeds:
                csv_path = base_dir / f"seed{seed}" / R_str / f"{model}_{batch}_per_state.csv"
                if csv_path.exists():
                    _load_rows_into_data(csv_path, data[model])

        # Run consistency checks
        checker = ConsistencyChecker()
        for model in ["model_a", "model_b"]:
            checker.run_all(data[model], radius=R)

        if checker.errors:
            print(f"[FAIL] Consistency check failed for R={R}!")
            print(checker.report())

        # Filter to mismatch drift only
        mismatch_a = filter_mismatch_drift(data["model_a"], n_train)
        mismatch_b = filter_mismatch_drift(data["model_b"], n_train)

        # Radius sweep is reported for one fixed mismatch: Δ(n_train, radius_n2).
        target_key = (n_train, radius_n2)
        metrics_list = [
            "delta_V",
            "delta_pi",
            "delta_z",
            "argmax_agree",
            "saturated",
            "z_pre_norm",
            "z_post_norm",
        ]
        if target_key not in mismatch_a or target_key not in mismatch_b:
            print(f"[WARN] Missing target key {target_key} at R={R}")

        flat_a = {
            metric: list(mismatch_a.get(target_key, {}).get(metric, []))
            for metric in metrics_list
        }
        flat_b = {
            metric: list(mismatch_b.get(target_key, {}).get(metric, []))
            for metric in metrics_list
        }

        all_data[R] = {
            "model_a": {m: compute_stats(v, m == "argmax_agree") for m, v in flat_a.items()},
            "model_b": {m: compute_stats(v, m == "argmax_agree") for m, v in flat_b.items()},
        }

    # Write outputs
    write_radius_sweep_markdown(all_data, out_dir / f"radius_sweep_{batch}_summary.md",
                                seeds, batch, n_train, radius_n2)

    # Write CSV
    csv_path = out_dir / f"radius_sweep_{batch}_aggregated.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "radius", "metric", "mean", "std", "n", "ci_lower", "ci_upper", "p95", "max"])

        for R in sorted(all_data.keys()):
            for model in ["model_a", "model_b"]:
                for metric, stats in all_data[R][model].items():
                    row = [
                        model, R, metric,
                        f"{stats.mean:.6f}", f"{stats.std:.6f}", stats.n,
                        f"{stats.ci_lower:.4f}" if stats.ci_lower is not None else "",
                        f"{stats.ci_upper:.4f}" if stats.ci_upper is not None else "",
                        f"{stats.p95:.4f}" if stats.p95 is not None else "",
                        f"{stats.max_val:.4f}" if stats.max_val is not None else "",
                    ]
                    writer.writerow(row)

    print(f"[Write] {csv_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Post-process Exp1 v4 for ICML")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base results directory (e.g., results/validation/exp1_v4)")
    parser.add_argument("--out_dir", type=str, default="results/tables",
                        help="Output directory for tables")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50",
                        help="Comma-separated list of seeds")
    parser.add_argument("--radii", type=str, default="10,100,0",
                        help="Comma-separated list of radii")
    parser.add_argument("--n_train", type=int, default=2,
                        help="Training depth")
    parser.add_argument("--radius_n2", type=int, default=8,
                        help="Fixed evaluation depth n2 for paper-facing radius sweep aggregation")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    radii = [float(r) for r in args.radii.split(",")]
    base_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Results dir: {base_dir}")
    print(f"[Config] Output dir: {out_dir}")
    print(f"[Config] Seeds: {seeds}")
    print(f"[Config] Radii: {radii}")
    print(f"[Config] n_train: {args.n_train}")

    success = True

    # Process both B0 and B1 for unroll sensitivity
    for batch in ["b0", "b1"]:
        ok = process_unroll_sensitivity(base_dir, seeds, args.n_train, out_dir, batch)
        success = success and ok

    # Process both B0 and B1 for radius sweep
    for batch in ["b0", "b1"]:
        ok = process_radius_sweep(
            base_dir,
            seeds,
            radii,
            args.n_train,
            args.radius_n2,
            out_dir,
            batch,
        )
        success = success and ok

    # Print summary
    print("\n" + "=" * 60)
    print("POST-PROCESSING COMPLETE")
    print("=" * 60)

    output_files = sorted(out_dir.glob("*.md")) + sorted(out_dir.glob("*.csv"))
    print("\nOutput files:")
    for f in output_files:
        print(f"  {f}")

    if success:
        print("\n[SUCCESS] All consistency checks passed")
        return 0
    else:
        print("\n[FAIL] Some consistency checks failed - review output")
        return 1


if __name__ == "__main__":
    sys.exit(main())
