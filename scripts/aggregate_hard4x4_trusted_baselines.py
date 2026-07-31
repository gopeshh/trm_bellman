#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
UPI_STEP = 20_000
EXPECTED_EVAL_SAMPLES = 50
UPI_METHOD = "upi_trm"
REQUIRED_EXTERNAL_METHODS = ("sb3_ppo", "sb3_a2c")
OPTIONAL_EXTERNAL_METHODS = ("sb3_dqn", "sb3_dqn_nstep5")
EXTERNAL_METHOD_DIRS = {
    "sb3_ppo": "ppo",
    "sb3_a2c": "a2c",
    "sb3_dqn": "dqn",
    "sb3_dqn_nstep5": "dqn_nstep5",
}
DISPLAY_NAMES = {
    UPI_METHOD: "UPI-TRM",
    "sb3_ppo": "SB3 PPO",
    "sb3_a2c": "SB3 A2C",
    "sb3_dqn": "SB3 DQN",
    "sb3_dqn_nstep5": "SB3 DQN (n=5)",
}
EXPECTED_DQN_N_STEPS = {
    "sb3_dqn": 1,
    "sb3_dqn_nstep5": 5,
}
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
SUCCESS_RE = re.compile(r"\[step\s+(\d+)\] eval_success_rate=(\d+\.\d+)")
TRAIN_SPLIT_RE = re.compile(r"\[DATASET\] train_split=(\S+) train_samples=(\d+)")
EVAL_SPLIT_RE = re.compile(
    r"\[DATASET\] eval_split=(\S+) eval_samples=(\d+) "
    r"eval_pool_sha256=([0-9a-f]{64})"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate provenance-checked hard-4x4 baseline records."
    )
    parser.add_argument(
        "--upi-log-dir",
        default=str(REPO_ROOT / "results/table3_hard_6to8"),
        help="Directory containing m1_persistent_nc_nomask_s*.log files.",
    )
    parser.add_argument(
        "--external-root",
        default=str(REPO_ROOT / "results/neurips2026/external_hard4x4_20k"),
        help="Directory containing sb3 ppo/a2c seed outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/paper_ready/hard4x4_trusted_baselines_20k"),
        help="Directory for paper-ready aggregate outputs.",
    )
    return parser.parse_args()


def _parse_final_success_rate(log_path: Path) -> dict[str, Any]:
    last_step = None
    last_rate = None
    eval_split = None
    eval_samples = None
    eval_pool_sha256 = None
    train_split = None
    for line in log_path.read_text().splitlines():
        train_match = TRAIN_SPLIT_RE.search(line)
        if train_match:
            train_split = train_match.group(1)
        split_match = EVAL_SPLIT_RE.search(line)
        if split_match:
            eval_split = split_match.group(1)
            eval_samples = int(split_match.group(2))
            eval_pool_sha256 = split_match.group(3)
        match = SUCCESS_RE.search(line)
        if match:
            last_step = int(match.group(1))
            last_rate = float(match.group(2))
    if last_rate is None:
        raise RuntimeError(f"No eval_success_rate entries found in {log_path}")
    if (
        eval_split != "test"
        or train_split != "train"
        or eval_samples != EXPECTED_EVAL_SAMPLES
        or eval_pool_sha256 is None
    ):
        raise RuntimeError(
            f"UPI result {log_path} lacks the required held-out "
            f"{EXPECTED_EVAL_SAMPLES}-instance test-pool provenance. Legacy or "
            "overlapping evaluations cannot be compared with test-set baselines."
        )
    if last_step != UPI_STEP:
        raise RuntimeError(
            f"Expected final UPI step {UPI_STEP} in {log_path}, found {last_step}"
        )
    return {
        "success_rate": last_rate,
        "eval_pool_sha256": eval_pool_sha256,
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _t_95_interval(values: list[float]) -> tuple[float, float]:
    mean = _mean(values)
    if len(values) <= 1:
        return mean, mean
    df = len(values) - 1
    t_crit = T_CRIT_95[df] if df in T_CRIT_95 else 1.96
    half_width = t_crit * _std(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def _round_rate(value: float, digits: int = 3) -> float:
    return round(value, digits)


def _format_mean_std(values: list[float], digits: int) -> str:
    return f"{_mean(values):.{digits}f} ± {_std(values):.{digits}f}"


def _format_steps_k(train_steps: int) -> str:
    if train_steps % 1_000 == 0:
        return f"{train_steps // 1_000}k"
    return str(train_steps)


def _format_budget(summary: dict[str, Any]) -> str:
    value = _format_steps_k(int(summary["train_steps"]))
    if summary["budget_unit"] == "outer_updates":
        return f"{value} outer updates"
    if summary["budget_unit"] == "environment_steps":
        return f"{value} env steps"
    return f"{value} {summary['budget_unit']}"


def _load_upi_results(log_dir: Path) -> dict[str, Any]:
    per_seed = []
    pool_hashes = set()
    for seed in range(10):
        log_path = log_dir / f"m1_persistent_nc_nomask_s{seed}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing UPI log: {log_path}")
        parsed = _parse_final_success_rate(log_path)
        pool_hashes.add(parsed["eval_pool_sha256"])
        per_seed.append(
            {
                "method": UPI_METHOD,
                "seed": seed,
                "train_steps": UPI_STEP,
                "budget_unit": "outer_updates",
                "success_rate": parsed["success_rate"],
                "mean_return": None,
                "invalid_action_rate": None,
                "eval_points": None,
                "all_eval_success_zero": None,
                "eval_pool_sha256": parsed["eval_pool_sha256"],
                "source": str(log_path),
            }
        )
    if len(pool_hashes) != 1:
        raise RuntimeError(f"UPI seeds use different evaluation pools: {sorted(pool_hashes)}")
    return {
        "method": UPI_METHOD,
        "display_name": DISPLAY_NAMES[UPI_METHOD],
        "per_seed": per_seed,
        "eval_pool_sha256": next(iter(pool_hashes)),
    }


def _load_external_results(
    method: str,
    algo_dir: Path,
    *,
    expected_dqn_n_steps: int | None = None,
) -> dict[str, Any]:
    per_seed = []
    total_eval_points = 0
    all_eval_success_zero = True
    total_final_eval_successes = 0
    total_final_eval_episodes = 0
    expected_train_steps = None
    pool_hashes = set()

    for seed in range(10):
        summary_path = algo_dir / f"seed{seed}" / "train_summary.json"
        history_path = algo_dir / f"seed{seed}" / "eval_history.jsonl"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing trusted-baseline summary: {summary_path}")
        if not history_path.exists():
            raise FileNotFoundError(f"Missing trusted-baseline history: {history_path}")

        summary = json.loads(summary_path.read_text())
        final_eval = summary["final_eval"]
        if (
            final_eval.get("split") != "test"
            or int(final_eval.get("episodes", 0)) != EXPECTED_EVAL_SAMPLES
            or not final_eval.get("pool_sha256")
            or summary.get("train_split") != "train"
            or summary.get("eval_split") != "test"
        ):
            raise RuntimeError(
                f"External result {summary_path} lacks the required held-out "
                f"{EXPECTED_EVAL_SAMPLES}-instance test-pool provenance."
            )
        pool_hashes.add(final_eval["pool_sha256"])
        if expected_dqn_n_steps is not None:
            actual_dqn_n_steps = int(
                summary.get("dqn_n_steps", summary.get("sb3_hparams", {}).get("n_steps", 1))
            )
            if actual_dqn_n_steps != expected_dqn_n_steps:
                raise RuntimeError(
                    f"Expected DQN n_steps={expected_dqn_n_steps} for {method}, "
                    f"found {actual_dqn_n_steps} in {summary_path}"
                )
        train_steps = int(summary["train_steps"])
        if expected_train_steps is None:
            expected_train_steps = train_steps
        elif train_steps != expected_train_steps:
            raise RuntimeError(
                f"Inconsistent external train steps in {algo_dir}: "
                f"expected {expected_train_steps}, found {train_steps} in {summary_path}"
            )

        eval_points = 0
        seed_all_eval_success_zero = True
        with history_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if (
                    record.get("split") != "test"
                    or int(record.get("episodes", 0)) != EXPECTED_EVAL_SAMPLES
                    or record.get("pool_sha256") != final_eval["pool_sha256"]
                ):
                    raise RuntimeError(
                        f"Evaluation history row in {history_path} does not match "
                        "the declared held-out pool."
                    )
                eval_points += 1
                if record["success_rate"] > 0:
                    seed_all_eval_success_zero = False
                    all_eval_success_zero = False
        total_eval_points += eval_points

        final_successes = round(final_eval["success_rate"] * final_eval["episodes"])
        total_final_eval_successes += final_successes
        total_final_eval_episodes += final_eval["episodes"]

        per_seed.append(
            {
                "method": method,
                "seed": seed,
                "train_steps": train_steps,
                "budget_unit": "environment_steps",
                "success_rate": float(final_eval["success_rate"]),
                "mean_return": float(final_eval["mean_return"]),
                "invalid_action_rate": float(final_eval["invalid_action_rate"]),
                "eval_points": eval_points,
                "all_eval_success_zero": seed_all_eval_success_zero,
                "eval_pool_sha256": final_eval["pool_sha256"],
                "source": str(summary_path),
            }
        )

    if len(pool_hashes) != 1:
        raise RuntimeError(
            f"External {method} seeds use different evaluation pools: {sorted(pool_hashes)}"
        )
    return {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "per_seed": per_seed,
        "train_steps": expected_train_steps,
        "total_eval_points": total_eval_points,
        "all_eval_success_zero": all_eval_success_zero,
        "final_eval_successes": total_final_eval_successes,
        "final_eval_episodes": total_final_eval_episodes,
        "eval_pool_sha256": next(iter(pool_hashes)),
    }


def _has_complete_external_results(algo_dir: Path) -> bool:
    if not algo_dir.exists():
        return False
    for seed in range(10):
        seed_dir = algo_dir / f"seed{seed}"
        if not (seed_dir / "train_summary.json").exists():
            return False
        if not (seed_dir / "eval_history.jsonl").exists():
            return False
    return True


def _summarize_method(method_data: dict[str, Any]) -> dict[str, Any]:
    success_rates = [row["success_rate"] for row in method_data["per_seed"]]
    summary = {
        "method": method_data["method"],
        "display_name": method_data["display_name"],
        "train_steps": method_data["per_seed"][0]["train_steps"],
        "budget_unit": method_data["per_seed"][0]["budget_unit"],
        "num_seeds": len(method_data["per_seed"]),
        "success_rate_mean": _mean(success_rates),
        "success_rate_std": _std(success_rates),
        "success_rate_ci_95": list(_t_95_interval(success_rates)),
        "per_seed_success_rates": success_rates,
        "eval_pool_sha256": method_data["eval_pool_sha256"],
    }

    returns = [row["mean_return"] for row in method_data["per_seed"] if row["mean_return"] is not None]
    invalid_rates = [
        row["invalid_action_rate"]
        for row in method_data["per_seed"]
        if row["invalid_action_rate"] is not None
    ]
    if returns:
        summary["mean_return_mean"] = _mean(returns)
        summary["mean_return_std"] = _std(returns)
    if invalid_rates:
        summary["invalid_action_rate_mean"] = _mean(invalid_rates)
        summary["invalid_action_rate_std"] = _std(invalid_rates)

    if "total_eval_points" in method_data:
        summary["total_eval_points"] = method_data["total_eval_points"]
        summary["all_eval_success_zero"] = method_data["all_eval_success_zero"]
        summary["final_eval_successes"] = method_data["final_eval_successes"]
        summary["final_eval_episodes"] = method_data["final_eval_episodes"]
    return summary


def _write_per_seed_csv(output_dir: Path, all_rows: list[dict[str, Any]]) -> Path:
    output_path = output_dir / "per_seed_metrics.csv"
    fieldnames = [
        "method",
        "seed",
        "train_steps",
        "budget_unit",
        "success_rate",
        "mean_return",
        "invalid_action_rate",
        "eval_points",
        "all_eval_success_zero",
        "eval_pool_sha256",
        "source",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    return output_path


def _write_summary_csv(
    output_dir: Path,
    summaries: dict[str, Any],
    method_order: list[str],
) -> Path:
    output_path = output_dir / "summary.csv"
    fieldnames = [
        "method",
        "display_name",
        "train_steps",
        "budget_unit",
        "num_seeds",
        "success_rate_mean",
        "success_rate_std",
        "success_rate_ci_95_low",
        "success_rate_ci_95_high",
        "mean_return_mean",
        "mean_return_std",
        "invalid_action_rate_mean",
        "invalid_action_rate_std",
        "total_eval_points",
        "all_eval_success_zero",
        "final_eval_successes",
        "final_eval_episodes",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for method in method_order:
            data = summaries[method]
            writer.writerow(
                {
                    "method": method,
                    "display_name": data["display_name"],
                    "train_steps": data["train_steps"],
                    "budget_unit": data["budget_unit"],
                    "num_seeds": data["num_seeds"],
                    "success_rate_mean": data["success_rate_mean"],
                    "success_rate_std": data["success_rate_std"],
                    "success_rate_ci_95_low": data["success_rate_ci_95"][0],
                    "success_rate_ci_95_high": data["success_rate_ci_95"][1],
                    "mean_return_mean": data.get("mean_return_mean"),
                    "mean_return_std": data.get("mean_return_std"),
                    "invalid_action_rate_mean": data.get("invalid_action_rate_mean"),
                    "invalid_action_rate_std": data.get("invalid_action_rate_std"),
                    "total_eval_points": data.get("total_eval_points"),
                    "all_eval_success_zero": data.get("all_eval_success_zero"),
                    "final_eval_successes": data.get("final_eval_successes"),
                    "final_eval_episodes": data.get("final_eval_episodes"),
                }
            )
    return output_path


def _write_summary_json(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _write_table_tex(
    output_dir: Path,
    summaries: dict[str, Any],
    method_order: list[str],
) -> Path:
    output_path = output_dir / "table_hard4x4_trusted_baselines.tex"
    lines = [
        "% Auto-generated by scripts/aggregate_hard4x4_trusted_baselines.py",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Budget & Seeds & Success (\%) & Return & Invalid \\",
        r"\midrule",
    ]
    upi = summaries[UPI_METHOD]
    lines.append(
        f"UPI--TRM & {_format_budget(upi)} & {upi['num_seeds']} & "
        f"{100.0 * upi['success_rate_mean']:.1f} $\\pm$ "
        f"{100.0 * upi['success_rate_std']:.1f} & -- & -- \\\\"
    )
    for method in method_order:
        if method == UPI_METHOD:
            continue
        summary = summaries[method]
        lines.append(
            f"{summary['display_name']} & {_format_budget(summary)} & {summary['num_seeds']} & "
            f"{100.0 * summary['success_rate_mean']:.1f} $\\pm$ "
            f"{100.0 * summary['success_rate_std']:.1f} & "
            f"{summary['mean_return_mean']:.2f} $\\pm$ "
            f"{summary['mean_return_std']:.2f} & "
            f"{summary['invalid_action_rate_mean']:.3f} $\\pm$ "
            f"{summary['invalid_action_rate_std']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))
    return output_path


def _write_summary_md(
    output_dir: Path,
    summaries: dict[str, Any],
    method_order: list[str],
) -> Path:
    output_path = output_dir / "SUMMARY.md"
    upi = summaries[UPI_METHOD]
    external_methods = [method for method in method_order if method != UPI_METHOD]
    external_names = [summaries[method]["display_name"] for method in external_methods]
    table_rows = [
        (
            f"| UPI-TRM | {_format_budget(upi)} | {upi['num_seeds']} | "
            f"{_format_mean_std(upi['per_seed_success_rates'], 3)} | "
            f"[{upi['success_rate_ci_95'][0]:.3f}, {upi['success_rate_ci_95'][1]:.3f}] | -- | -- |"
        )
    ]
    for method in external_methods:
        summary = summaries[method]
        table_rows.append(
            f"| {summary['display_name']} | {_format_budget(summary)} | {summary['num_seeds']} | "
            f"{_format_mean_std(summary['per_seed_success_rates'], 3)} | "
            f"[{summary['success_rate_ci_95'][0]:.3f}, {summary['success_rate_ci_95'][1]:.3f}] | "
            f"{summary['mean_return_mean']:.3f} ± {summary['mean_return_std']:.3f} | "
            f"{summary['invalid_action_rate_mean']:.3f} ± {summary['invalid_action_rate_std']:.3f} |"
        )
    lines = [
        "# Hard 4x4 Baseline Record Summary",
        "",
        (
            "Provenance-checked hard-4x4 aggregate: UPI-TRM plus external "
            f"{'/'.join(name.replace('SB3 ', '') for name in external_names)} "
            "records on one validated ordered evaluation pool."
        ),
        "",
        "## Aggregate Table",
        "",
        "| Method | Budget | Seeds | Success Rate | 95% CI | Mean Return | Invalid Action Rate |",
        "|--------|--------|-------|--------------|--------|-------------|---------------------|",
        *table_rows,
        "",
        "## Interpretation",
        "",
        (
            "These rows compare complete implementation protocols on the same ordered "
            "evaluation pool. They do not use the same training budget: UPI-TRM is "
            "indexed by outer updates, while SB3 baselines are indexed by environment steps."
        ),
        (
            "No between-method gap interval is reported because unequal interaction budgets "
            "make that contrast unsuitable for a sample-efficiency or superiority claim."
        ),
        "",
    ]
    output_path.write_text("\n".join(lines))
    return output_path


def main() -> int:
    args = _parse_args()
    upi_results = _load_upi_results(Path(args.upi_log_dir))
    all_results = {UPI_METHOD: upi_results}
    included_external_methods = []
    for method in REQUIRED_EXTERNAL_METHODS:
        algo_dir = Path(args.external_root) / EXTERNAL_METHOD_DIRS[method]
        all_results[method] = _load_external_results(
            method,
            algo_dir,
            expected_dqn_n_steps=EXPECTED_DQN_N_STEPS.get(method),
        )
        included_external_methods.append(method)
    for method in OPTIONAL_EXTERNAL_METHODS:
        algo_dir = Path(args.external_root) / EXTERNAL_METHOD_DIRS[method]
        if _has_complete_external_results(algo_dir):
            all_results[method] = _load_external_results(
                method,
                algo_dir,
                expected_dqn_n_steps=EXPECTED_DQN_N_STEPS.get(method),
            )
            included_external_methods.append(method)

    expected_pool_hash = all_results[UPI_METHOD]["eval_pool_sha256"]
    for method in included_external_methods:
        actual_pool_hash = all_results[method]["eval_pool_sha256"]
        if actual_pool_hash != expected_pool_hash:
            raise RuntimeError(
                f"Evaluation-pool mismatch: UPI={expected_pool_hash}, "
                f"{method}={actual_pool_hash}."
            )

    method_order = [UPI_METHOD, *included_external_methods]
    summaries = {method: _summarize_method(data) for method, data in all_results.items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for method in method_order:
        all_rows.extend(all_results[method]["per_seed"])

    outputs = {
        "per_seed_csv": str(_write_per_seed_csv(output_dir, all_rows)),
        "summary_csv": str(_write_summary_csv(output_dir, summaries, method_order)),
        "summary_json": str(
            _write_summary_json(
                output_dir,
                {
                    "method_order": method_order,
                    "methods": summaries,
                    "comparison_scope": "complete_protocols_unequal_training_budgets",
                },
            )
        ),
        "summary_md": str(_write_summary_md(output_dir, summaries, method_order)),
        "table_tex": str(_write_table_tex(output_dir, summaries, method_order)),
    }

    print("[hard4x4] generated aggregate artifacts")
    for key, path in outputs.items():
        print(f"[hard4x4] {key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
