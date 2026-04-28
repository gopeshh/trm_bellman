#!/usr/bin/env python3
"""
Postprocess the episodic-z hard-suite rerun.

Outputs:
  - per_seed_eval.jsonl
  - per_seed_diagnostics.jsonl
  - summary.json
  - diagnostics_summary.json
  - cpi_penalty_grid.json

The script parses eval lines from the per-seed training logs and runs the
frozen-batch diagnostic runner on the saved checkpoints at the requested
milestones.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


EVAL_RE = re.compile(
    r"^\[step (?P<step>\d+)\](?: \[update \d+\])? "
    r"eval_success_rate=(?P<success_rate>[-+0-9.eE]+) "
    r"eval_mean_score=(?P<mean_score>[-+0-9.eE]+) "
    r"eval_policy_mode=(?P<eval_policy_mode>\S+) "
    r"\[solved=(?P<solved>\d+)/(?P<episodes>\d+), .*?\]"
    r"(?: eval_mean_return=(?P<mean_return>[-+0-9.eE]+))?"
    r"(?: eval_invalid_action_rate=(?P<invalid_action_rate>[-+0-9.eE]+))?"
    r"$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config-yaml", type=Path, required=True)
    parser.add_argument("--closure-batch", type=Path, required=True)
    parser.add_argument("--directions-npz", type=Path, required=True)
    parser.add_argument("--diag-runner", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    return parser.parse_args()


def parse_numeric(text: str | None) -> float | None:
    if text is None:
        return None
    return float(text)


def parse_eval_history(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    entries: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = EVAL_RE.match(line)
            if not match:
                continue
            entry = {
                "step": int(match.group("step")),
                "success_rate": float(match.group("success_rate")),
                "mean_score": float(match.group("mean_score")),
                "eval_policy_mode": match.group("eval_policy_mode"),
                "solved": int(match.group("solved")),
                "episodes": int(match.group("episodes")),
                "mean_return": parse_numeric(match.group("mean_return")),
                "invalid_action_rate": parse_numeric(match.group("invalid_action_rate")),
                "source_log": str(log_path),
            }
            entries.append(entry)
    if not entries:
        raise RuntimeError(f"No eval lines found in {log_path}")
    return entries


def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "inf":
            return math.inf
        if lowered == "-inf":
            return -math.inf
        if lowered == "nan":
            return math.nan
        return float(value)
    raise TypeError(f"Unsupported scalar value: {value!r}")


def summarize_values(values: Sequence[Any]) -> Dict[str, Any]:
    scalars = [normalize_scalar(value) for value in values]
    finite = [value for value in scalars if math.isfinite(value)]
    pos_inf_count = sum(1 for value in scalars if value == math.inf)
    neg_inf_count = sum(1 for value in scalars if value == -math.inf)
    nan_count = sum(1 for value in scalars if math.isnan(value))

    if finite:
        mean_value: Any = statistics.fmean(finite)
        sample_std: Any = statistics.stdev(finite) if len(finite) > 1 else 0.0
        min_value: Any = min(finite)
        max_value: Any = max(finite)
    elif pos_inf_count > 0 and neg_inf_count == 0 and nan_count == 0:
        mean_value = math.inf
        sample_std = 0.0
        min_value = math.inf
        max_value = math.inf
    elif neg_inf_count > 0 and pos_inf_count == 0 and nan_count == 0:
        mean_value = -math.inf
        sample_std = 0.0
        min_value = -math.inf
        max_value = -math.inf
    else:
        mean_value = math.nan
        sample_std = math.nan
        min_value = math.nan
        max_value = math.nan

    return {
        "count": len(scalars),
        "finite_count": len(finite),
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "nan_count": nan_count,
        "mean": mean_value,
        "sample_std": sample_std,
        "min": min_value,
        "max": max_value,
    }


def run_checkpoint_diagnostics(
    *,
    diag_runner: Path,
    checkpoint: Path,
    config_yaml: Path,
    closure_batch: Path,
    directions_npz: Path,
    output_json: Path,
    device: str,
) -> Dict[str, Any]:
    if not output_json.exists():
        command = [
            str(diag_runner),
            "checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--config-yaml",
            str(config_yaml),
            "--closure-batch",
            str(closure_batch),
            "--directions-npz",
            str(directions_npz),
            "--output-json",
            str(output_json),
            "--device",
            device,
        ]
        subprocess.run(command, check=True)
    return load_json(output_json)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    log_dir = args.log_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    config_yaml = args.config_yaml.resolve()
    closure_batch = args.closure_batch.resolve()
    directions_npz = args.directions_npz.resolve()
    diag_runner = args.diag_runner.resolve()

    diagnostics_root = results_root / "diagnostics"
    per_seed_eval_path = results_root / "per_seed_eval.jsonl"
    per_seed_diag_path = results_root / "per_seed_diagnostics.jsonl"
    summary_path = results_root / "summary.json"
    diagnostics_summary_path = results_root / "diagnostics_summary.json"
    cpi_penalty_grid_path = results_root / "cpi_penalty_grid.json"

    eval_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []

    checkpoint_steps = sorted(int(step) for step in args.checkpoint_steps)
    required_eval_steps = set(checkpoint_steps)

    for seed in args.seeds:
        log_path = log_dir / f"seed{seed}.log"
        seed_eval_rows = parse_eval_history(log_path)
        eval_by_step = {int(row["step"]): row for row in seed_eval_rows}
        missing_eval_steps = sorted(required_eval_steps.difference(eval_by_step))
        if missing_eval_steps:
            raise RuntimeError(
                f"Seed {seed} is missing eval lines for steps {missing_eval_steps} in {log_path}"
            )

        for step in checkpoint_steps:
            eval_entry = dict(eval_by_step[step])
            eval_entry["seed"] = int(seed)
            eval_rows.append(eval_entry)

            checkpoint_path = checkpoint_root / f"seed{seed}" / f"rl_checkpoint_step_{step}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

            diag_path = diagnostics_root / f"seed{seed}" / f"step{step}.json"
            payload = run_checkpoint_diagnostics(
                diag_runner=diag_runner,
                checkpoint=checkpoint_path,
                config_yaml=config_yaml,
                closure_batch=closure_batch,
                directions_npz=directions_npz,
                output_json=diag_path,
                device=args.device,
            )
            diagnostics = payload["diagnostics"]
            diag_entry = {
                "seed": int(seed),
                "step": int(step),
                "checkpoint": str(checkpoint_path),
                "diagnostics_path": str(diag_path),
                "eps_res_n": diagnostics["eps_res_n"],
                "eps_cent": diagnostics["eps_cent"],
                "eps_A_cand": diagnostics["eps_A_cand"],
                "eps_CPI_ref": diagnostics["eps_CPI_ref"],
                "L_V": diagnostics["L_V"],
                "rho_R": diagnostics["rho_R"],
                "L_z_post": diagnostics["L_z_post"],
                "projection_active_rate": diagnostics["projection_active_rate"],
                "eta_old_proxy": diagnostics["eta_old_proxy"],
                "eta_old_proxy_std": diagnostics["eta_old_proxy_std"],
                "alpha_grid": diagnostics["alpha_grid"],
            }
            diag_rows.append(diag_entry)

    jsonl_write(per_seed_eval_path, eval_rows)
    jsonl_write(per_seed_diag_path, diag_rows)

    final_step = max(checkpoint_steps)
    final_eval_rows = [row for row in eval_rows if int(row["step"]) == final_step]

    summary = {
        "experiment": "episodic_z_hard_suite_20k_seed41_50",
        "seeds": [int(seed) for seed in args.seeds],
        "checkpoint_steps": checkpoint_steps,
        "final_step": final_step,
        "artifacts": {
            "per_seed_eval_jsonl": str(per_seed_eval_path),
            "per_seed_diagnostics_jsonl": str(per_seed_diag_path),
        },
        "aggregate_final_eval": {
            "success_rate": summarize_values([row["success_rate"] for row in final_eval_rows]),
            "mean_return": summarize_values([row["mean_return"] for row in final_eval_rows]),
            "invalid_action_rate": summarize_values(
                [row["invalid_action_rate"] for row in final_eval_rows]
            ),
            "mean_score": summarize_values([row["mean_score"] for row in final_eval_rows]),
        },
        "per_seed_final_eval": final_eval_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="ascii")

    scalar_metrics = [
        "eps_res_n",
        "eps_cent",
        "eps_A_cand",
        "eps_CPI_ref",
        "L_V",
        "rho_R",
        "L_z_post",
        "projection_active_rate",
        "eta_old_proxy",
        "eta_old_proxy_std",
    ]
    diagnostics_summary: Dict[str, Any] = {
        "experiment": "episodic_z_hard_suite_20k_seed41_50",
        "seeds": [int(seed) for seed in args.seeds],
        "checkpoint_steps": checkpoint_steps,
        "by_step": {},
    }
    cpi_penalty_grid: Dict[str, Any] = {
        "experiment": "episodic_z_hard_suite_20k_seed41_50",
        "seeds": [int(seed) for seed in args.seeds],
        "checkpoint_steps": checkpoint_steps,
        "by_step": {},
    }

    for step in checkpoint_steps:
        rows_for_step = [row for row in diag_rows if int(row["step"]) == step]
        diagnostics_summary["by_step"][str(step)] = {
            metric: summarize_values([row[metric] for row in rows_for_step])
            for metric in scalar_metrics
        }

        alpha_keys = sorted(
            {
                alpha_key
                for row in rows_for_step
                for alpha_key in row["alpha_grid"].keys()
            },
            key=float,
        )
        cpi_penalty_grid["by_step"][str(step)] = {}
        for alpha_key in alpha_keys:
            alpha_rows = [row["alpha_grid"][alpha_key] for row in rows_for_step]
            cpi_penalty_grid["by_step"][str(step)][alpha_key] = {
                "surrogate_true_gap": summarize_values(
                    [row["surrogate_true_gap"] for row in alpha_rows]
                ),
                "predicted_penalty": summarize_values(
                    [row["predicted_penalty"] for row in alpha_rows]
                ),
                "eta_proxy": summarize_values([row["eta_proxy"] for row in alpha_rows]),
                "lhat_alpha": summarize_values([row["lhat_alpha"] for row in alpha_rows]),
                "expected_adv_mean": summarize_values(
                    [row["expected_adv_mean"] for row in alpha_rows]
                ),
            }

    diagnostics_summary_path.write_text(
        json.dumps(diagnostics_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="ascii",
    )
    cpi_penalty_grid_path.write_text(
        json.dumps(cpi_penalty_grid, indent=2, sort_keys=True, default=str) + "\n",
        encoding="ascii",
    )

    print(json.dumps({
        "summary_json": str(summary_path),
        "diagnostics_summary_json": str(diagnostics_summary_path),
        "cpi_penalty_grid_json": str(cpi_penalty_grid_path),
        "per_seed_eval_jsonl": str(per_seed_eval_path),
        "per_seed_diagnostics_jsonl": str(per_seed_diag_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
