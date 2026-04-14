#!/usr/bin/env python3
"""
Measure value-head Lipschitz estimates on theorem-facing Exp1 refreeze checkpoints.

This script targets the theorem-facing isolation study package:
  - checkpoints/exp1_v4_refreeze/model_a_prime|model_b/seed41..50/model_step_5000.pt
  - artifacts/eval_batches/exp1_v4_refreeze/b0.pt

For each checkpoint, it:
  1. Recomputes z^(n_eval) on the frozen B0 batch using the tested `used_value()`
     path (episodic-z, flattened z / x / y embeddings).
  2. Estimates hat(L_V) by finite differences at multiple perturbation scales.
  3. Repeats the random-direction estimator several times per scale to assess
     sampling stability rather than reporting a single cherry-picked number.

Outputs:
  - results/paper_ready/exp1_value_head_lipschitz/
    - summary.json
    - CLAIMS.md
    - PROVENANCE.md
    - table_exp1_value_head_lipschitz.tex
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.unroll_sensitivity import load_batch, load_model_for_eval
from utils.lipschitz import estimate_Lv


DEFAULT_CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "exp1_v4_refreeze"
DEFAULT_BATCH_PATH = PROJECT_ROOT / "artifacts" / "eval_batches" / "exp1_v4_refreeze" / "b0.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "paper_ready" / "exp1_value_head_lipschitz"
DEFAULT_SEEDS = list(range(41, 51))
DEFAULT_EPS_LIST = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
DEFAULT_N_EVAL = 8
DEFAULT_NUM_TRIALS = 8
DEFAULT_NUM_DIRECTIONS = 16
DEFAULT_CENTRAL_EPS = 1e-4

CONDITIONS: Dict[str, Dict[str, str]] = {
    "model_a_prime": {
        "label": "Model A' (no contraction)",
        "subdir": "model_a_prime",
        "config_yaml": str(
            PROJECT_ROOT / "configs" / "ablations" / "upi_trm_feasibility_no_contraction_no_vhead_norm.yaml"
        ),
    },
    "model_b": {
        "label": "Model B (contraction)",
        "subdir": "model_b",
        "config_yaml": str(
            PROJECT_ROOT / "configs" / "ablations" / "upi_trm_feasibility_contraction_no_vhead_norm.yaml"
        ),
    },
}


@dataclass
class SeedScaleResult:
    condition: str
    condition_label: str
    seed: int
    checkpoint: str
    n_eval: int
    batch_name: str
    batch_size: int
    eps: float
    num_trials: int
    num_directions: int
    trial_maxes: List[float]
    trial_mean: float
    trial_std: float
    trial_min: float
    trial_max: float


def parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_base",
        type=Path,
        default=DEFAULT_CHECKPOINT_BASE,
        help="Base directory containing exp1_v4_refreeze checkpoints",
    )
    parser.add_argument(
        "--batch_path",
        type=Path,
        default=DEFAULT_BATCH_PATH,
        help="Path to frozen B0 theorem-facing batch (.pt)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_ints,
        default=DEFAULT_SEEDS,
        help="Comma-separated seed list",
    )
    parser.add_argument(
        "--eps_list",
        type=parse_csv_floats,
        default=DEFAULT_EPS_LIST,
        help="Comma-separated perturbation scales",
    )
    parser.add_argument(
        "--central_eps",
        type=float,
        default=DEFAULT_CENTRAL_EPS,
        help="Scale highlighted in CLAIMS/table text",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=DEFAULT_N_EVAL,
        help="Evaluator depth used to build theorem-facing z^(n)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Repeated random-direction trials per eps",
    )
    parser.add_argument(
        "--num_directions",
        type=int,
        default=DEFAULT_NUM_DIRECTIONS,
        help="Random perturbation directions per trial",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: cpu, cuda, or auto (default: cpu for portability)",
    )
    return parser.parse_args()


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def checkpoint_path(base: Path, condition_key: str, seed: int) -> Path:
    subdir = CONDITIONS[condition_key]["subdir"]
    return base / subdir / f"seed{seed}" / "model_step_5000.pt"


def format_eps(eps: float) -> str:
    return f"{eps:.0e}"


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def to_batch_tensors(states: Sequence[Any], device: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inputs = torch.stack([state.inputs for state in states]).to(device)
    puzzle_ids = torch.stack([state.puzzle_identifier for state in states]).to(device)
    plans = torch.stack([state.plan for state in states]).to(device)
    x = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}
    return x, plans


def build_value_head_inputs(
    model: torch.nn.Module,
    x: Dict[str, torch.Tensor],
    y: torch.Tensor,
    n_eval: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        _, z_n = model.used_value(x, y, n_eval)
        batch = model._standardize_latent_batch(x, y)
        latent_context = model._build_latent_context_with_plan(batch)
        input_embeddings = latent_context["input_embeddings"]
        plan_embeddings = latent_context["plan_embeddings"]

        # Match TinyRecursiveReasoningModel_ACTV1.used_value() exactly.
        z_vec = z_n.z_H.view(z_n.z_H.shape[0], -1)
        x_embed = input_embeddings.view(input_embeddings.shape[0], -1)
        y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)
        combined_embed = torch.cat([x_embed, y_embed], dim=-1)

    return z_vec, combined_embed


def evaluate_checkpoint(
    condition_key: str,
    seed: int,
    ckpt_path: Path,
    x: Dict[str, torch.Tensor],
    y: torch.Tensor,
    batch_name: str,
    batch_size: int,
    n_eval: int,
    eps_list: Sequence[float],
    num_trials: int,
    num_directions: int,
    device: str,
) -> List[SeedScaleResult]:
    model, _ = load_model_for_eval(
        str(ckpt_path),
        device=device,
        config_yaml_path=CONDITIONS[condition_key]["config_yaml"],
    )
    if model.value_head is None:
        raise RuntimeError(f"{ckpt_path} has no value head")

    z_vec, combined_embed = build_value_head_inputs(model, x, y, n_eval)
    results: List[SeedScaleResult] = []

    for eps in eps_list:
        trial_maxes = [
            float(
                estimate_Lv(
                    model.value_head,
                    z_vec,
                    combined_embed,
                    num_samples=num_directions,
                    eps=eps,
                )
            )
            for _ in range(num_trials)
        ]
        results.append(
            SeedScaleResult(
                condition=condition_key,
                condition_label=CONDITIONS[condition_key]["label"],
                seed=seed,
                checkpoint=str(ckpt_path),
                n_eval=n_eval,
                batch_name=batch_name,
                batch_size=batch_size,
                eps=eps,
                num_trials=num_trials,
                num_directions=num_directions,
                trial_maxes=trial_maxes,
                trial_mean=float(np.mean(trial_maxes)),
                trial_std=float(np.std(trial_maxes)),
                trial_min=float(np.min(trial_maxes)),
                trial_max=float(np.max(trial_maxes)),
            )
        )

    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def aggregate_results(
    records: Sequence[SeedScaleResult],
    central_eps: float,
) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {"by_condition": {}}

    grouped: Dict[Tuple[str, float], List[SeedScaleResult]] = {}
    for record in records:
        grouped.setdefault((record.condition, record.eps), []).append(record)

    for condition_key, condition_info in CONDITIONS.items():
        per_eps: Dict[str, Any] = {}
        mean_by_eps: Dict[float, float] = {}
        for eps in sorted({record.eps for record in records if record.condition == condition_key}):
            subset = grouped[(condition_key, eps)]
            seed_means = [record.trial_mean for record in subset]
            seed_maxes = [record.trial_max for record in subset]
            entry = {
                "condition_label": condition_info["label"],
                "eps": eps,
                "n_seeds": len(subset),
                "seed_mean_of_trial_mean": float(np.mean(seed_means)),
                "seed_std_of_trial_mean": sample_std(seed_means),
                "seed_min_of_trial_mean": float(np.min(seed_means)),
                "seed_max_of_trial_mean": float(np.max(seed_means)),
                "seed_mean_of_trial_max": float(np.mean(seed_maxes)),
                "seed_std_of_trial_max": sample_std(seed_maxes),
            }
            per_eps[str(eps)] = entry
            mean_by_eps[eps] = entry["seed_mean_of_trial_mean"]

        if not mean_by_eps:
            continue

        min_eps = min(mean_by_eps, key=mean_by_eps.get)
        max_eps = max(mean_by_eps, key=mean_by_eps.get)
        min_mean = mean_by_eps[min_eps]
        max_mean = mean_by_eps[max_eps]
        stability_ratio = float(max_mean / min_mean) if min_mean > 0 else math.inf

        aggregated["by_condition"][condition_key] = {
            "condition_label": condition_info["label"],
            "central_eps": central_eps,
            "eps_grid": [float(eps) for eps in sorted(mean_by_eps)],
            "per_eps": per_eps,
            "scale_span": {
                "min_eps": float(min_eps),
                "max_eps": float(max_eps),
                "min_mean": float(min_mean),
                "max_mean": float(max_mean),
                "ratio": stability_ratio,
            },
        }

    return aggregated


def central_entry(aggregated: Dict[str, Any], condition_key: str, central_eps: float) -> Dict[str, Any]:
    return aggregated["by_condition"][condition_key]["per_eps"][str(central_eps)]


def generate_table(aggregated: Dict[str, Any], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Finite-difference $\hat L_V$ estimates for the theorem-facing Exp1 "
            r"refreeze isolation package on the frozen B0 batch at $n_{\mathrm{eval}}{=}8$. "
            r"Each entry is mean$\pm$sample-std over 10 seeds of the per-seed average across "
            r"8 repeated random-direction trials (16 perturbations per trial).}"
        ),
        r"\label{tab:exp1_value_head_lipschitz}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Condition & $\epsilon$ & $\hat L_V$ & Seed range \\",
        r"\midrule",
    ]

    for condition_key in CONDITIONS:
        if condition_key not in aggregated["by_condition"]:
            continue
        condition_label = aggregated["by_condition"][condition_key]["condition_label"]
        for eps in aggregated["by_condition"][condition_key]["eps_grid"]:
            entry = aggregated["by_condition"][condition_key]["per_eps"][str(eps)]
            mean_str = f"{entry['seed_mean_of_trial_mean']:.3f}$\\pm${entry['seed_std_of_trial_mean']:.3f}"
            range_str = f"[{entry['seed_min_of_trial_mean']:.3f}, {entry['seed_max_of_trial_mean']:.3f}]"
            lines.append(
                f"  {condition_label} & {format_eps(eps)} & {mean_str} & {range_str} \\\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    out_path.write_text("\n".join(lines))


def generate_claims(aggregated: Dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Exp1 Value-Head Lipschitz Diagnostic: Claims",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This diagnostic measures finite-difference $\\hat L_V$ on the theorem-facing Exp1",
        "refreeze checkpoints (B0 batch, episodic-z, n_eval=8) across multiple perturbation scales.",
        "",
        "## Central-scale summary",
        "",
    ]

    for condition_key in CONDITIONS:
        if condition_key not in aggregated["by_condition"]:
            continue
        condition = aggregated["by_condition"][condition_key]
        entry = central_entry(aggregated, condition_key, condition["central_eps"])
        lines.append(
            f"- **{condition['condition_label']}** at eps={format_eps(condition['central_eps'])}: "
            f"mean seedwise $\\hat L_V$ = {entry['seed_mean_of_trial_mean']:.3f} "
            f"(sample std {entry['seed_std_of_trial_mean']:.3f}, "
            f"seed range [{entry['seed_min_of_trial_mean']:.3f}, {entry['seed_max_of_trial_mean']:.3f}])."
        )

    lines.extend(["", "## Across-scale stability", ""])
    for condition_key in CONDITIONS:
        if condition_key not in aggregated["by_condition"]:
            continue
        condition = aggregated["by_condition"][condition_key]
        span = condition["scale_span"]
        lines.append(
            f"- **{condition['condition_label']}** spans {span['min_mean']:.3f} to {span['max_mean']:.3f} "
            f"over eps={format_eps(span['min_eps'])}..{format_eps(span['max_eps'])} "
            f"(max/min ratio {span['ratio']:.2f}x)."
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- This is a numerical diagnostic, not a certified global Lipschitz bound.",
            "- It speaks only to the theorem-facing Exp1 B0 isolation setting, not the hard-4x4 anchor.",
            "- The kill criterion for manuscript use is scale fragility by an order of magnitude or worse; this file reports the actual span needed to make that call.",
        ]
    )
    out_path.write_text("\n".join(lines))


def generate_provenance(args: argparse.Namespace, out_path: Path) -> None:
    lines = [
        "# Exp1 Value-Head Lipschitz Diagnostic: Provenance",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        f"- Checkpoint base: `{args.checkpoint_base}`",
        f"- Batch path: `{args.batch_path}`",
        f"- Seeds: `{','.join(map(str, args.seeds))}`",
        f"- n_eval: `{args.n_eval}`",
        f"- eps grid: `{','.join(format_eps(eps) for eps in args.eps_list)}`",
        f"- trials per eps: `{args.num_trials}`",
        f"- directions per trial: `{args.num_directions}`",
        "",
        "## Regeneration",
        "",
        "```bash",
        "cd /home/buiksat/fbsource/fbcode",
        (
            "buck2 run //buiksat_trm:exp1_value_head_lipschitz -- "
            f"--n_eval {args.n_eval} "
            f"--eps_list {','.join(str(eps) for eps in args.eps_list)} "
            f"--num_trials {args.num_trials} "
            f"--num_directions {args.num_directions}"
        ),
        "```",
    ]
    out_path.write_text("\n".join(lines))


def write_summary(
    args: argparse.Namespace,
    records: Sequence[SeedScaleResult],
    aggregated: Dict[str, Any],
    out_path: Path,
) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "args": {
            "checkpoint_base": str(args.checkpoint_base),
            "batch_path": str(args.batch_path),
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "eps_list": args.eps_list,
            "central_eps": args.central_eps,
            "n_eval": args.n_eval,
            "num_trials": args.num_trials,
            "num_directions": args.num_directions,
            "device": args.device,
        },
        "records": [asdict(record) for record in records],
        "aggregated": aggregated,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.central_eps not in args.eps_list:
        raise ValueError("--central_eps must be included in --eps_list")

    print(f"[Device] {args.device}")
    print(f"[Batch] Loading theorem-facing batch from {args.batch_path}")
    states, metadata = load_batch(str(args.batch_path))
    x, y = to_batch_tensors(states, args.device)

    records: List[SeedScaleResult] = []
    for condition_key, condition_info in CONDITIONS.items():
        print(f"\n{'=' * 72}")
        print(f"[Condition] {condition_info['label']}")
        print("=" * 72)
        for seed in args.seeds:
            ckpt_path = checkpoint_path(args.checkpoint_base, condition_key, seed)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            print(f"[Eval] seed={seed} checkpoint={ckpt_path}")
            seed_results = evaluate_checkpoint(
                condition_key=condition_key,
                seed=seed,
                ckpt_path=ckpt_path,
                x=x,
                y=y,
                batch_name=metadata.batch_name,
                batch_size=len(states),
                n_eval=args.n_eval,
                eps_list=args.eps_list,
                num_trials=args.num_trials,
                num_directions=args.num_directions,
                device=args.device,
            )
            records.extend(seed_results)
            for result in seed_results:
                print(
                    "  "
                    f"eps={format_eps(result.eps)} "
                    f"hat_Lv={result.trial_mean:.3f} "
                    f"(trial std {result.trial_std:.3f}, "
                    f"trial range [{result.trial_min:.3f}, {result.trial_max:.3f}])"
                )

    aggregated = aggregate_results(records, args.central_eps)
    generate_table(aggregated, args.out_dir / "table_exp1_value_head_lipschitz.tex")
    generate_claims(aggregated, args.out_dir / "CLAIMS.md")
    generate_provenance(args, args.out_dir / "PROVENANCE.md")
    write_summary(args, records, aggregated, args.out_dir / "summary.json")

    print(f"\n[Done] Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
