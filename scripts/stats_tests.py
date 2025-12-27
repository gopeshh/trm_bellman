#!/usr/bin/env python3
"""
Statistical significance tests for ICML 2026 paper.

Performs paired bootstrap and Welch t-tests to compare methods,
with Holm-Bonferroni correction for multiple comparisons.

Usage:
    python scripts/stats_tests.py --input artifacts/summary.parquet --output paper/tables/significance.tex
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def paired_bootstrap_test(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Perform paired bootstrap test for difference in means.

    Args:
        x: First sample (e.g., UPI-TRM results per seed)
        y: Second sample (e.g., baseline results per seed)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level

    Returns:
        Dict with mean_diff, ci_lower, ci_upper, p_value
    """
    assert len(x) == len(y), "Samples must have same length for paired test"

    n = len(x)
    observed_diff = np.mean(x) - np.mean(y)

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_x = x[indices]
        boot_y = y[indices]
        diffs.append(np.mean(boot_x) - np.mean(boot_y))

    diffs = np.array(diffs)

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, alpha / 2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)

    # P-value (two-tailed): proportion of bootstrap samples with opposite sign
    if observed_diff > 0:
        p_value = 2 * np.mean(diffs <= 0)
    else:
        p_value = 2 * np.mean(diffs >= 0)

    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": min(p_value, 1.0),
    }


def welch_t_test(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Perform Welch's t-test (unequal variance t-test).

    Args:
        x: First sample
        y: Second sample

    Returns:
        Dict with t_statistic, p_value, mean_diff
    """
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_diff": np.mean(x) - np.mean(y),
    }


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values
        alpha: Significance level

    Returns:
        List of (adjusted_p, is_significant) tuples
    """
    n = len(p_values)
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    results = [None] * n
    reject_all = True

    for rank, (p, orig_idx) in enumerate(indexed):
        threshold = alpha / (n - rank)

        if reject_all and p <= threshold:
            # Reject null hypothesis
            adjusted_p = min(p * (n - rank), 1.0)
            results[orig_idx] = (adjusted_p, True)
        else:
            # Accept null hypothesis (and all remaining)
            reject_all = False
            adjusted_p = min(p * (n - rank), 1.0)
            results[orig_idx] = (adjusted_p, False)

    return results


def get_final_metrics_by_seed(
    df: pd.DataFrame,
    metric: str,
    step_percentile: float = 0.95,
) -> Dict[str, Dict[int, float]]:
    """
    Get final metric values organized by config and seed.

    Args:
        df: Aggregated data
        metric: Metric column name
        step_percentile: Use metrics from this percentile of training

    Returns:
        Dict mapping config_name -> {seed: value}
    """
    results = {}

    for config in df["config_name"].unique():
        config_data = df[df["config_name"] == config]

        if metric not in config_data.columns:
            continue

        max_step = config_data["step"].max()
        step_threshold = max_step * step_percentile

        final_data = config_data[config_data["step"] >= step_threshold]

        seed_values = {}
        for seed in final_data["seed"].unique():
            seed_data = final_data[final_data["seed"] == seed]
            values = seed_data[metric].dropna()
            if len(values) > 0:
                seed_values[seed] = values.mean()

        if seed_values:
            results[config] = seed_values

    return results


def run_pairwise_tests(
    df: pd.DataFrame,
    reference_config: str,
    metric: str = "eval_success_rate",
    test_type: str = "bootstrap",
) -> pd.DataFrame:
    """
    Run pairwise tests comparing reference config to all others.

    Args:
        df: Aggregated data
        reference_config: Config to compare against (e.g., "shaped_theory_exact")
        metric: Metric column
        test_type: "bootstrap" or "welch"

    Returns:
        DataFrame with comparison results
    """
    seed_data = get_final_metrics_by_seed(df, metric)

    if reference_config not in seed_data:
        print(f"Error: Reference config '{reference_config}' not found")
        return pd.DataFrame()

    ref_values = seed_data[reference_config]
    ref_seeds = set(ref_values.keys())

    results = []

    for config, values in seed_data.items():
        if config == reference_config:
            continue

        # Find common seeds
        common_seeds = ref_seeds.intersection(values.keys())
        if len(common_seeds) < 2:
            print(f"Warning: Not enough common seeds for {config}")
            continue

        # Get paired values
        x = np.array([ref_values[s] for s in sorted(common_seeds)])
        y = np.array([values[s] for s in sorted(common_seeds)])

        if test_type == "bootstrap":
            test_result = paired_bootstrap_test(x, y)
        else:
            test_result = welch_t_test(x, y)

        results.append({
            "config": config,
            "reference": reference_config,
            "n_seeds": len(common_seeds),
            "ref_mean": np.mean(x),
            "config_mean": np.mean(y),
            **test_result,
        })

    results_df = pd.DataFrame(results)

    # Apply Holm-Bonferroni correction
    if len(results_df) > 0:
        p_values = results_df["p_value"].tolist()
        corrected = holm_bonferroni_correction(p_values)
        results_df["p_adjusted"] = [c[0] for c in corrected]
        results_df["significant"] = [c[1] for c in corrected]

    return results_df


def generate_significance_table(
    results: pd.DataFrame,
    output_path: str,
    alpha: float = 0.05,
) -> None:
    """
    Generate LaTeX table showing significance test results.

    Args:
        results: DataFrame from run_pairwise_tests
        output_path: Path to save .tex file
        alpha: Significance level for marking
    """
    if results.empty:
        print("No results to generate table from")
        return

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Statistical Significance Tests. P-values with Holm-Bonferroni correction. " +
        f"* indicates $p < {alpha}$.}}",
        "\\label{tab:significance}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Comparison & Mean Diff & p-value & Sig. \\\\",
        "\\midrule",
    ]

    # Sort by p-value
    results = results.sort_values("p_adjusted")

    for _, row in results.iterrows():
        config = row["config"]
        # Shorten config name
        if len(config) > 25:
            config = config[:22] + "..."

        mean_diff = row["mean_diff"]
        p_adj = row["p_adjusted"]
        sig = row["significant"]

        # Format p-value
        if p_adj < 0.001:
            p_str = "$< 0.001$"
        else:
            p_str = f"{p_adj:.3f}"

        sig_str = "*" if sig else ""

        # Color diff based on direction
        if mean_diff > 0:
            diff_str = f"\\textcolor{{green}}{{+{mean_diff:.1f}}}"
        else:
            diff_str = f"\\textcolor{{red}}{{{mean_diff:.1f}}}"

        lines.append(f"vs {config} & {diff_str} & {p_str} & {sig_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")


def print_summary(results: pd.DataFrame) -> None:
    """Print human-readable summary of test results."""
    if results.empty:
        print("No results to summarize")
        return

    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("=" * 60)

    ref = results["reference"].iloc[0]
    print(f"\nReference: {ref}")
    print(f"Total comparisons: {len(results)}")
    print(f"Significant (after correction): {results['significant'].sum()}")

    print("\nDetailed results (sorted by p-value):")
    print("-" * 60)

    for _, row in results.sort_values("p_adjusted").iterrows():
        config = row["config"]
        diff = row["mean_diff"]
        p_adj = row["p_adjusted"]
        sig = "***" if row["significant"] else ""

        direction = "better" if diff > 0 else "worse"
        print(f"  vs {config[:35]:35s}: {diff:+6.2f} ({direction:6s}) p={p_adj:.4f} {sig}")


def main():
    parser = argparse.ArgumentParser(description="Run statistical significance tests")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/summary.parquet",
        help="Input parquet file from aggregate_runs.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/tables/significance.tex",
        help="Output .tex file path",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference config to compare against (default: auto-detect theory-exact)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_success_rate",
        help="Metric to test",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["bootstrap", "welch"],
        default="bootstrap",
        help="Statistical test type",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Auto-detect reference config
    reference = args.reference
    if reference is None:
        for config in df["config_name"].unique():
            if "shaped_theory_exact" in config.lower():
                reference = config
                break
        if reference is None:
            for config in df["config_name"].unique():
                if "theory_exact" in config.lower():
                    reference = config
                    break

    if reference is None:
        print("Error: Could not auto-detect reference config. Specify with --reference")
        return

    print(f"\nUsing reference config: {reference}")
    print(f"Metric: {args.metric}")
    print(f"Test type: {args.test}")

    # Run tests
    results = run_pairwise_tests(
        df,
        reference_config=reference,
        metric=args.metric,
        test_type=args.test,
    )

    if results.empty:
        print("No comparisons could be made")
        return

    # Print summary
    print_summary(results)

    # Save CSV
    csv_path = args.output.replace(".tex", ".csv")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Generate LaTeX table
    generate_significance_table(results, args.output, alpha=args.alpha)


if __name__ == "__main__":
    main()
