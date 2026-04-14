#!/usr/bin/env python3
"""
Batched primary finite-R evaluator for the exp1_v4_refreeze package.

This runner executes only the primary target used by the finite-R plan:
- frozen B0 batch only
- fixed absolute depths (default: 2,8)
- two-model compare outputs matching the existing per-state CSV schema

It exists because the legacy compare path loops state-by-state, which is too
slow when CPU fallback is required.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from rl.envs.plan_edit_env import PlanEditEnv
from scripts.eval.unroll_sensitivity import (
    EvalMetrics,
    check_model_compatibility,
    compute_kl_divergence,
    get_git_sha,
    load_batch,
    load_model_for_eval,
    write_per_state_csv,
    write_summary_csv,
)


class BatchedProjectionStats:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._original_project = None
        self._hooked = False
        self.pre_norms: List[torch.Tensor] = []
        self.post_norms: List[torch.Tensor] = []

    def __enter__(self):
        if hasattr(self.model, "inner") and hasattr(self.model.inner, "_project_to_ball"):
            self._original_project = self.model.inner._project_to_ball

            def instrumented_project(z: torch.Tensor, radius: float) -> torch.Tensor:
                pre = z.norm(p=2, dim=(1, 2)).detach().cpu()
                z_proj = self._original_project(z, radius)
                post = z_proj.norm(p=2, dim=(1, 2)).detach().cpu()
                self.pre_norms.append(pre)
                self.post_norms.append(post)
                return z_proj

            self.model.inner._project_to_ball = instrumented_project
            self._hooked = True
        return self

    def __exit__(self, *args):
        if self._hooked and self._original_project is not None:
            self.model.inner._project_to_ball = self._original_project

    def mean_norms(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.pre_norms:
            zeros = torch.zeros(batch_size, dtype=torch.float32)
            return zeros, zeros
        pre = torch.stack(self.pre_norms, dim=0).mean(dim=0)
        post = torch.stack(self.post_norms, dim=0).mean(dim=0)
        return pre, post


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def chunk_ranges(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(total, start + chunk_size)


def batch_from_states(states, device: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    state_ids = [state.state_id for state in states]
    inputs = torch.stack([state.inputs for state in states]).to(device)
    puzzle_identifiers = torch.stack([state.puzzle_identifier for state in states]).to(device)
    plans = torch.stack([state.plan for state in states]).to(device)
    x = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
    }
    return x, plans, state_ids


def evaluate_model_batched(
    model: torch.nn.Module,
    states,
    n_values: Sequence[int],
    config: Dict[str, object],
    device: str,
    eval_batch_size: int,
) -> List[EvalMetrics]:
    radius = float(config.get("latent_ball_radius", 0.0))
    vocab_size = int(config["vocab_size"])
    num_actions = int(config["num_actions"])
    stop_action_id = num_actions - 1

    all_metrics: List[EvalMetrics] = []

    for start, end in chunk_ranges(len(states), eval_batch_size):
        chunk_states = states[start:end]
        x, y, state_ids = batch_from_states(chunk_states, device)
        batch_size = len(chunk_states)
        action_mask = PlanEditEnv.compute_batch_action_mask(
            x["inputs"],
            vocab_size,
            stop_action_id,
        ).to(device)

        values: Dict[int, torch.Tensor] = {}
        policies: Dict[int, torch.Tensor] = {}
        latents: Dict[int, torch.Tensor] = {}
        pre_norms: Dict[int, torch.Tensor] = {}
        post_norms: Dict[int, torch.Tensor] = {}

        model.eval()
        with torch.inference_mode():
            for n in n_values:
                with BatchedProjectionStats(model) as stats:
                    value, z_n = model.used_value(x, y, n)
                    dist, _ = model.policy_dist(x, y, n, action_mask=action_mask)
                    values[n] = value.reshape(-1).detach().cpu()
                    policies[n] = dist.probs.detach().cpu()
                    latents[n] = z_n.z_H.detach().cpu()
                    pre, post = stats.mean_norms(batch_size)
                    pre_norms[n] = pre
                    post_norms[n] = post

        for idx, state_id in enumerate(state_ids):
            for i, n1 in enumerate(n_values):
                for n2 in n_values[i + 1 :]:
                    p = policies[n1][idx]
                    q = policies[n2][idx]
                    delta_pi = compute_kl_divergence(p, q)
                    delta_z = float(torch.norm((latents[n1][idx] - latents[n2][idx]).reshape(-1), p=2).item())
                    argmax_agree = int(p.argmax().item() == q.argmax().item())
                    z_pre = float(pre_norms[n1][idx].item())
                    z_post = float(post_norms[n1][idx].item())
                    saturated = -1 if radius <= 0 else int(z_pre >= 0.95 * radius)
                    all_metrics.append(
                        EvalMetrics(
                            state_id=state_id,
                            n1=int(n1),
                            n2=int(n2),
                            delta_V=float(torch.abs(values[n1][idx] - values[n2][idx]).item()),
                            delta_pi=delta_pi,
                            delta_z=delta_z,
                            argmax_agree=argmax_agree,
                            z_pre_norm=z_pre,
                            z_post_norm=z_post,
                            saturated=saturated,
                        )
                    )

    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched primary finite-R evaluator")
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--config_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--config_b", required=True)
    parser.add_argument("--batch_b0", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--n_values", default="2,8")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_values = parse_int_list(args.n_values)

    model_a, config_a = load_model_for_eval(
        args.checkpoint_a,
        device,
        config_yaml_path=args.config_a,
        latent_ball_radius_override=args.radius,
    )
    model_b, config_b = load_model_for_eval(
        args.checkpoint_b,
        device,
        config_yaml_path=args.config_b,
        latent_ball_radius_override=args.radius,
    )

    compatible, diffs = check_model_compatibility(config_a, config_b)
    if not compatible:
        raise RuntimeError(f"Incompatible models: {diffs}")

    b0_states, _ = load_batch(args.batch_b0)

    metrics_a = evaluate_model_batched(
        model_a,
        b0_states,
        n_values=n_values,
        config=config_a,
        device=device,
        eval_batch_size=args.eval_batch_size,
    )
    metrics_b = evaluate_model_batched(
        model_b,
        b0_states,
        n_values=n_values,
        config=config_b,
        device=device,
        eval_batch_size=args.eval_batch_size,
    )

    write_per_state_csv(metrics_a, str(out_dir / "model_a_b0_per_state.csv"))
    write_summary_csv(metrics_a, "b0", str(out_dir / "model_a_b0_summary.csv"))
    write_per_state_csv(metrics_b, str(out_dir / "model_b_b0_per_state.csv"))
    write_summary_csv(metrics_b, "b0", str(out_dir / "model_b_b0_summary.csv"))

    run_meta = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "config_a": config_a,
        "config_b": config_b,
        "n_values": n_values,
        "radius": args.radius,
        "seed": args.seed,
        "device": device,
        "eval_batch_size": args.eval_batch_size,
        "git_sha": get_git_sha(),
    }
    with open(out_dir / "finite_r_primary_run_metadata.json", "w") as handle:
        json.dump(run_meta, handle, indent=2, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
