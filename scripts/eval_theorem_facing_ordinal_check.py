#!/usr/bin/env python3
"""
Bounded theorem-facing ordinal check for the Exp1 refreeze package.

For each selected seed/condition pair, this script:
1. loads the frozen checkpoint from exp1_v4_refreeze;
2. splits the fixed B0 batch into deterministic update/eval subsets;
3. computes held-out theory proxies on the eval subset;
4. performs one exact-centering policy update from the frozen checkpoint; and
5. evaluates old / exact-mixture / candidate policies on held-out rollouts.

The intended use is the Phase 1 smoke gate from
REVIEWERS_FEEDBACK_v5_CLOSURE_PLAN.md: produce one interpretable per-seed
result before deciding whether to keep the exact-mixture path or fall back.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import torch
import yaml

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import (
    make_sudoku_feasibility_checker,
    sudoku_checker,
    sudoku_constraint_checker,
    sudoku_progress_checker,
)
from rl.sudoku_utils import sudoku_is_solved
from rl.task_config import get_task_config
from rl.upi_trm_trainer import UPITrmTrainer
from scripts.eval.unroll_sensitivity import load_batch, load_model_for_eval


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "exp1_v4_refreeze"
DEFAULT_BATCH_PATH = PROJECT_ROOT / "artifacts" / "eval_batches" / "exp1_v4_refreeze" / "b0.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "validation" / "exp1_v4_refreeze_ordinal_check"
DEFAULT_SEEDS = list(range(41, 51))
DEFAULT_CONDITIONS = ["model_a_prime", "model_b"]

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
class PolicyEvalStats:
    mode: str
    discounted_return_mean: float
    discounted_return_std: float
    final_score_mean: float
    final_score_std: float
    success_rate_mean: float
    success_rate_std: float
    num_rollouts: int


@dataclass
class SeedConditionResult:
    seed: int
    condition: str
    condition_label: str
    checkpoint: str
    config_yaml: str
    unroll_n: int
    mixture_alpha: float
    update_states: int
    eval_states: int
    collection_passes: int
    eval_repeats: int
    theory_metrics_heldout: Dict[str, float]
    policy_update: Dict[str, float]
    old_eval: PolicyEvalStats
    mixture_eval: PolicyEvalStats
    candidate_eval: PolicyEvalStats
    delta_discounted_return_mixture_vs_old: float
    delta_final_score_mixture_vs_old: float
    delta_success_rate_mixture_vs_old: float


class FrozenBatchDataset:
    """Minimal dataset wrapper over saved PuzzleState objects."""

    def __init__(self, states: Sequence[Any]):
        if not states:
            raise ValueError("FrozenBatchDataset requires at least one state")
        self._samples: List[Dict[str, Any]] = []
        max_pid = 0
        for state in states:
            puzzle_id = state.puzzle_identifier.clone()
            if puzzle_id.ndim == 0:
                puzzle_id = puzzle_id.unsqueeze(0)
            max_pid = max(max_pid, int(puzzle_id.max().item()))
            self._samples.append(
                {
                    "inputs": state.inputs.clone(),
                    "puzzle_identifiers": puzzle_id,
                    "initial_plan": state.plan.clone(),
                }
            )
        self.seq_len = int(self._samples[0]["inputs"].numel())
        self.num_identifiers = max_pid + 1

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._samples[idx]
        return {
            "inputs": sample["inputs"].clone(),
            "puzzle_identifiers": sample["puzzle_identifiers"].clone(),
            "initial_plan": sample["initial_plan"].clone(),
        }


class BatchDataset(Protocol):
    seq_len: int

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Dict[str, Any]: ...


def parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_strings(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def checkpoint_path(base: Path, condition_key: str, seed: int) -> Path:
    return base / CONDITIONS[condition_key]["subdir"] / f"seed{seed}" / "model_step_5000.pt"


def make_checker(rl_cfg: RLConfig):
    if getattr(rl_cfg, "use_feasibility_checker", False):
        return make_sudoku_feasibility_checker(
            w_v=float(getattr(rl_cfg, "feasibility_violation_weight", 2.0)),
            w_z=float(getattr(rl_cfg, "feasibility_zerocand_weight", 5.0)),
        )
    if getattr(rl_cfg, "use_progress_checker", False):
        return sudoku_progress_checker
    if getattr(rl_cfg, "use_constraint_checker", False):
        return sudoku_constraint_checker
    return sudoku_checker


def load_theory_exact_rl_config(
    config_yaml: str,
    *,
    unroll_n: int,
    batch_size: int,
) -> RLConfig:
    with open(config_yaml, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = RLConfig(**raw)
    base = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    base.update(
        {
            "inner_unroll_n": int(unroll_n),
            "batch_size": int(batch_size),
            "exact_baseline_summation": True,
            "theory_exact_mixture": True,
            "distill_mixture_policy": False,
            "batch_centered_advantage": False,
            "policy_epsilon": 0.0,
            "use_tqdm": False,
            "track_theory_metrics": False,
        }
    )
    return RLConfig(**base)


def build_env(
    dataset: BatchDataset,
    checker,
    rl_cfg: RLConfig,
    *,
    vocab_size: int,
) -> tuple[PlanEditEnv, PlanEditEnvConfig]:
    env_cfg = PlanEditEnvConfig(
        max_edits=int(rl_cfg.max_edits),
        gamma=float(rl_cfg.gamma),
        reward_shaping=bool(rl_cfg.reward_shaping),
        vocab_size=int(vocab_size),
        solved_threshold=rl_cfg.solved_threshold,
        task_type=getattr(rl_cfg, "task_name", "sudoku"),
        stop_action_mode=getattr(rl_cfg, "stop_action_mode", "noop"),
        stop_action_penalty=float(getattr(rl_cfg, "stop_action_penalty", -0.1)),
        C_max=float(rl_cfg.C_max),
        fail_terminal_reward=float(getattr(rl_cfg, "fail_terminal_reward", 0.0)),
        solve_terminal_reward=float(getattr(rl_cfg, "solve_terminal_reward", 0.0)),
        disable_constraint_masking=bool(getattr(rl_cfg, "disable_constraint_masking", False)),
    )
    task_config = get_task_config(
        "sudoku",
        disable_constraint_masking=bool(getattr(rl_cfg, "disable_constraint_masking", False)),
    )
    env = PlanEditEnv(dataset=dataset, checker=checker, config=env_cfg, task_config=task_config)
    env.set_stop_action_id(stop_id=dataset.seq_len * int(vocab_size))
    return env, env_cfg


def sorted_states(states: Sequence[Any]) -> List[Any]:
    return sorted(states, key=lambda state: state.state_id)


def split_states(states: Sequence[Any], update_fraction: float) -> tuple[List[Any], List[Any]]:
    ordered = sorted_states(states)
    split_idx = max(1, min(len(ordered) - 1, int(round(len(ordered) * update_fraction))))
    return ordered[:split_idx], ordered[split_idx:]


def collect_fixed_batch_episodes(trainer: UPITrmTrainer, num_episodes: int) -> int:
    env = trainer.env
    dataset_size = len(env.dataset)
    if dataset_size == 0:
        raise ValueError("Cannot collect episodes from an empty dataset")

    original_reset = env.reset
    counter = {"value": 0}

    def sequential_reset(idx: Optional[int] = None):
        if idx is None:
            idx = counter["value"] % dataset_size
            counter["value"] += 1
        return original_reset(idx=idx)

    env.reset = sequential_reset  # type: ignore[assignment]
    try:
        for _ in range(num_episodes):
            trainer.collect_episode()
    finally:
        env.reset = original_reset  # type: ignore[assignment]

    return len(trainer.replay)


def compute_heldout_theory_metrics(
    model: TinyRecursiveReasoningModel_ACTV1,
    rl_cfg: RLConfig,
    dataset: BatchDataset,
    checker,
    *,
    vocab_size: int,
    collection_passes: int,
    device: str,
    seed: int,
) -> Dict[str, float]:
    set_global_seed(seed)
    model_copy = copy.deepcopy(model)
    env, _ = build_env(dataset, checker, rl_cfg, vocab_size=vocab_size)
    trainer = UPITrmTrainer(model=model_copy, env=env, rl_cfg=rl_cfg, device=torch.device(device))
    trainer.set_checker_fn(checker)
    collect_fixed_batch_episodes(trainer, num_episodes=collection_passes * len(dataset))
    metrics = trainer._compute_theory_metrics()
    metrics["replay_size"] = float(len(trainer.replay))
    return metrics


def rollout_policy(
    trainer: UPITrmTrainer,
    dataset: BatchDataset,
    checker,
    env_cfg: PlanEditEnvConfig,
    *,
    mode: str,
    stochastic: bool,
    eval_repeats: int,
    device: str,
    seed: int,
) -> PolicyEvalStats:
    if mode not in {"old", "mixture", "candidate"}:
        raise ValueError(f"Unknown policy mode: {mode}")

    if mode in {"mixture", "candidate"}:
        trainer._sync_candidate_backbone_from_model()

    episodic_latent = bool(getattr(trainer.rl_cfg, "episodic_latent", True))
    rollout_returns: List[float] = []
    rollout_scores: List[float] = []
    rollout_success: List[float] = []

    for repeat_idx in range(eval_repeats):
        set_global_seed(seed + 1000 * repeat_idx)
        task_config = get_task_config(
            "sudoku",
            disable_constraint_masking=bool(getattr(trainer.rl_cfg, "disable_constraint_masking", False)),
        )
        eval_env = PlanEditEnv(dataset=dataset, checker=checker, config=env_cfg, task_config=task_config)
        eval_vocab_size = eval_env.vocab_size
        if eval_vocab_size is None:
            raise RuntimeError("Evaluation environment has no vocabulary size.")
        eval_env.set_stop_action_id(stop_id=dataset.seq_len * int(eval_vocab_size))

        for state_idx in range(len(dataset)):
            x, y = eval_env.reset(idx=state_idx)
            done = False
            gamma_pow = 1.0
            discounted_return = 0.0
            z = None

            if not episodic_latent:
                batched = trainer._state_is_batched(x)
                batch_x = trainer._prepare_batch_x(x, batched=batched)
                batch_y = trainer._prepare_plan(y, batched=batched)
                z = trainer.model.init_latent(batch_x, batch_y)

            while not done and eval_env.step_count < min(trainer.rl_cfg.max_edits, env_cfg.max_edits):
                batched = trainer._state_is_batched(x)
                batch_x = trainer._prepare_batch_x(x, batched=batched)
                batch_y = trainer._prepare_plan(y, batched=batched)
                action_mask = eval_env.get_action_mask()
                if action_mask is not None:
                    action_mask = action_mask.to(device)

                with torch.no_grad():
                    if mode == "old":
                        dist, z_new = trainer.policy_model_old.policy_dist(
                            batch_x,
                            batch_y,
                            n=trainer.rl_cfg.inner_unroll_n,
                            action_mask=action_mask,
                            z=z,
                        )
                    elif mode == "candidate":
                        dist, z_new = trainer.policy_model_candidate.policy_dist(
                            batch_x,
                            batch_y,
                            n=trainer.rl_cfg.inner_unroll_n,
                            action_mask=action_mask,
                            z=z,
                        )
                    else:
                        dist, z_new = trainer._mixed_policy_dist(
                            batch_x,
                            batch_y,
                            n=trainer.rl_cfg.inner_unroll_n,
                            action_mask=action_mask,
                            z=z,
                        )

                if stochastic:
                    action = int(dist.sample().item())
                else:
                    action = int(dist.logits.argmax(dim=-1).item())

                if not episodic_latent:
                    z = z_new

                (x, y), reward, done, _ = eval_env.step(action)
                discounted_return += gamma_pow * float(reward)
                gamma_pow *= float(env_cfg.gamma)

            final_score = float(checker(x, y))
            solved = False
            if torch.is_tensor(y) and y.numel() in (16, 81):
                solved = sudoku_is_solved(y)

            rollout_returns.append(discounted_return)
            rollout_scores.append(final_score)
            rollout_success.append(1.0 if solved else 0.0)

    def _std(values: Sequence[float]) -> float:
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    return PolicyEvalStats(
        mode=mode,
        discounted_return_mean=float(statistics.mean(rollout_returns)),
        discounted_return_std=_std(rollout_returns),
        final_score_mean=float(statistics.mean(rollout_scores)),
        final_score_std=_std(rollout_scores),
        success_rate_mean=float(statistics.mean(rollout_success)),
        success_rate_std=_std(rollout_success),
        num_rollouts=len(rollout_returns),
    )


def aggregate_results(results: Sequence[SeedConditionResult]) -> Dict[str, Any]:
    by_condition: Dict[str, List[SeedConditionResult]] = {}
    for result in results:
        by_condition.setdefault(result.condition, []).append(result)

    summary: Dict[str, Any] = {"conditions": {}}
    for condition, items in by_condition.items():
        summary["conditions"][condition] = {
            "condition_label": items[0].condition_label,
            "num_seeds": len(items),
            "heldout_bellman_residual_mean": float(
                statistics.mean(item.theory_metrics_heldout["bellman_residual_mean"] for item in items)
            ),
            "heldout_hat_Lv_mean": float(
                statistics.mean(item.theory_metrics_heldout["hat_Lv"] for item in items)
            ),
            "heldout_hat_Lz_mean": float(
                statistics.mean(item.theory_metrics_heldout["hat_Lz"] for item in items)
            ),
            "mixture_delta_return_mean": float(
                statistics.mean(item.delta_discounted_return_mixture_vs_old for item in items)
            ),
            "mixture_delta_final_score_mean": float(
                statistics.mean(item.delta_final_score_mixture_vs_old for item in items)
            ),
            "mixture_delta_success_mean": float(
                statistics.mean(item.delta_success_rate_mixture_vs_old for item in items)
            ),
        }
    return summary


def write_markdown_summary(results: Sequence[SeedConditionResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Exp1 Refreeze Ordinal Check\n\n")
        f.write("| Condition | Seed | Held-out Bellman Res. | Held-out hat(L_V) | Held-out hat(L_z) | ")
        f.write("Old Return | Mixture Return | Delta Return |\n")
        f.write("|-----------|------|-----------------------|-------------------|-------------------|")
        f.write("-----------------|----------------|--------------|\n")
        for result in results:
            f.write(
                f"| {result.condition_label} | {result.seed} | "
                f"{result.theory_metrics_heldout['bellman_residual_mean']:.4f} | "
                f"{result.theory_metrics_heldout['hat_Lv']:.4f} | "
                f"{result.theory_metrics_heldout['hat_Lz']:.4f} | "
                f"{result.old_eval.discounted_return_mean:.4f} | "
                f"{result.mixture_eval.discounted_return_mean:.4f} | "
                f"{result.delta_discounted_return_mixture_vs_old:.4f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-base", type=Path, default=DEFAULT_CHECKPOINT_BASE)
    parser.add_argument("--batch-path", type=Path, default=DEFAULT_BATCH_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--conditions", type=parse_csv_strings, default=DEFAULT_CONDITIONS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--unroll-n",
        type=int,
        default=8,
        help="Internal evaluator depth used for the held-out ordinal check.",
    )
    parser.add_argument(
        "--update-fraction",
        type=float,
        default=0.5,
        help="Fraction of B0 states used for the update subset; the remainder is held out.",
    )
    parser.add_argument(
        "--collection-passes",
        type=int,
        default=1,
        help="How many sequential passes to collect over the update/eval subsets.",
    )
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=4,
        help="Monte Carlo repeats per held-out start state for policy evaluation.",
    )
    parser.add_argument(
        "--greedy-eval",
        action="store_true",
        help="Use greedy argmax instead of stochastic sampling for rollout evaluation.",
    )
    return parser.parse_args()


def run_single_condition(
    *,
    seed: int,
    condition: str,
    checkpoint_base: Path,
    batch_states: Sequence[Any],
    out_dir: Path,
    device: str,
    unroll_n: int,
    update_fraction: float,
    collection_passes: int,
    eval_repeats: int,
    greedy_eval: bool,
) -> SeedConditionResult:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    ckpt_path = checkpoint_path(checkpoint_base, condition, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    config_yaml = CONDITIONS[condition]["config_yaml"]
    model, model_cfg = load_model_for_eval(
        str(ckpt_path),
        device=device,
        config_yaml_path=config_yaml,
    )

    update_states, eval_states = split_states(batch_states, update_fraction=update_fraction)
    update_dataset = FrozenBatchDataset(update_states)
    eval_dataset = FrozenBatchDataset(eval_states)

    rl_cfg = load_theory_exact_rl_config(
        config_yaml,
        unroll_n=unroll_n,
        batch_size=min(32, max(4, len(update_dataset))),
    )
    checker = make_checker(rl_cfg)

    heldout_metrics = compute_heldout_theory_metrics(
        model=model,
        rl_cfg=rl_cfg,
        dataset=eval_dataset,
        checker=checker,
        vocab_size=int(model_cfg["vocab_size"]),
        collection_passes=collection_passes,
        device=device,
        seed=seed,
    )
    required_metrics = ("bellman_residual_mean", "hat_Lv", "hat_Lz")
    missing_metrics = [name for name in required_metrics if name not in heldout_metrics]
    if missing_metrics:
        available = ", ".join(sorted(heldout_metrics.keys()))
        raise RuntimeError(
            f"Missing required held-out theory metrics {missing_metrics} for {condition} seed {seed}. "
            f"Available metrics: [{available}]"
        )

    set_global_seed(seed)
    update_env, env_cfg = build_env(update_dataset, checker, rl_cfg, vocab_size=int(model_cfg["vocab_size"]))
    trainer = UPITrmTrainer(model=model, env=update_env, rl_cfg=rl_cfg, device=torch.device(device))
    trainer.set_checker_fn(checker)
    collect_fixed_batch_episodes(trainer, num_episodes=collection_passes * len(update_dataset))
    policy_update = trainer.policy_update()

    old_eval = rollout_policy(
        trainer,
        eval_dataset,
        checker,
        env_cfg,
        mode="old",
        stochastic=not greedy_eval,
        eval_repeats=eval_repeats,
        device=device,
        seed=seed,
    )
    mixture_eval = rollout_policy(
        trainer,
        eval_dataset,
        checker,
        env_cfg,
        mode="mixture",
        stochastic=not greedy_eval,
        eval_repeats=eval_repeats,
        device=device,
        seed=seed + 100_000,
    )
    candidate_eval = rollout_policy(
        trainer,
        eval_dataset,
        checker,
        env_cfg,
        mode="candidate",
        stochastic=not greedy_eval,
        eval_repeats=eval_repeats,
        device=device,
        seed=seed + 200_000,
    )

    result = SeedConditionResult(
        seed=seed,
        condition=condition,
        condition_label=CONDITIONS[condition]["label"],
        checkpoint=str(ckpt_path),
        config_yaml=config_yaml,
        unroll_n=unroll_n,
        mixture_alpha=float(rl_cfg.mixture_alpha),
        update_states=len(update_dataset),
        eval_states=len(eval_dataset),
        collection_passes=collection_passes,
        eval_repeats=eval_repeats,
        theory_metrics_heldout=heldout_metrics,
        policy_update={k: float(v) for k, v in policy_update.items()},
        old_eval=old_eval,
        mixture_eval=mixture_eval,
        candidate_eval=candidate_eval,
        delta_discounted_return_mixture_vs_old=(
            mixture_eval.discounted_return_mean - old_eval.discounted_return_mean
        ),
        delta_final_score_mixture_vs_old=(
            mixture_eval.final_score_mean - old_eval.final_score_mean
        ),
        delta_success_rate_mixture_vs_old=(
            mixture_eval.success_rate_mean - old_eval.success_rate_mean
        ),
    )

    seed_out = out_dir / condition / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    with open(seed_out / "result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    batch_states, metadata = load_batch(str(args.batch_path))

    all_results: List[SeedConditionResult] = []
    for condition in args.conditions:
        for seed in args.seeds:
            result = run_single_condition(
                seed=seed,
                condition=condition,
                checkpoint_base=args.checkpoint_base,
                batch_states=batch_states,
                out_dir=args.out_dir,
                device=device,
                unroll_n=args.unroll_n,
                update_fraction=args.update_fraction,
                collection_passes=args.collection_passes,
                eval_repeats=args.eval_repeats,
                greedy_eval=args.greedy_eval,
            )
            all_results.append(result)
            print(
                f"[OrdinalCheck] {result.condition_label} seed {seed}: "
                f"heldout_res={result.theory_metrics_heldout['bellman_residual_mean']:.4f}, "
                f"delta_return={result.delta_discounted_return_mixture_vs_old:.4f}, "
                f"delta_score={result.delta_final_score_mixture_vs_old:.4f}"
            )

    summary = aggregate_results(all_results)
    summary["batch_metadata"] = asdict(metadata)
    summary["device"] = device
    summary["unroll_n"] = args.unroll_n
    summary["update_fraction"] = args.update_fraction
    summary["collection_passes"] = args.collection_passes
    summary["eval_repeats"] = args.eval_repeats
    summary["greedy_eval"] = bool(args.greedy_eval)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_markdown_summary(all_results, args.out_dir / "SUMMARY.md")

    print(f"[OrdinalCheck] Wrote {args.out_dir / 'summary.json'}")
    print(f"[OrdinalCheck] Wrote {args.out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
