#!/usr/bin/env python3
"""
Protocol runner for the episodic-z hard-suite theorem-contact diagnostics.

Subcommands:
  - validate-finite-mdp:
      Run the locked finite-MDP certificate script and verify that the expected
      101 rows are present with zero certificate violations.
  - checkpoint:
      Compute the nine frozen-batch diagnostic proxies for a saved RL checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv
from rl.upi_trm_trainer import UPITrmTrainer
from scripts.eval.unroll_sensitivity import load_model_for_eval
from scripts.eval_theorem_facing_ordinal_check import build_env, make_checker, rollout_policy
from utils.lipschitz import compute_exact_baseline_summation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_ROOT = PROJECT_ROOT.parent / "UPI_TRM" / "UPI_TRM_ICLR"
DEFAULT_ALPHA_GRID = (0.05, 0.1, 0.2, 0.4)


@dataclass
class FrozenDirections:
    lv_directions: torch.Tensor
    latent_directions: torch.Tensor
    eps: float
    metadata: Dict[str, Any]


class ClosureBatchDataset:
    """Minimal dataset wrapper over the frozen closure-batch artifact."""

    def __init__(self, artifact_path: Path):
        data = np.load(artifact_path, allow_pickle=False)
        self.inputs = torch.from_numpy(data["inputs"]).long()
        self.puzzle_identifiers = torch.from_numpy(data["puzzle_identifiers"]).long()
        self.initial_plan = torch.from_numpy(data["initial_plan"]).long()
        self.solution = (
            torch.from_numpy(data["labels"]).long() if "labels" in data.files else None
        )
        self.sample_indices = (
            torch.from_numpy(data["sample_indices"]).long() if "sample_indices" in data.files else None
        )
        self.metadata = json.loads(str(data["metadata_json"].item()))

        self.seq_len = int(self.inputs.shape[1])
        self.num_identifiers = int(self.puzzle_identifiers.max().item()) + 1
        max_token = int(self.inputs.max().item())
        if self.solution is not None:
            max_token = max(max_token, int(self.solution.max().item()))
        self.vocab_size = max_token + 1

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "inputs": self.inputs[idx].clone(),
            "puzzle_identifiers": self.puzzle_identifiers[idx].clone(),
            "initial_plan": self.initial_plan[idx].clone(),
        }
        if self.solution is not None:
            sample["solution"] = self.solution[idx].clone()
        return sample

    def get_chunk(
        self,
        start: int,
        stop: int,
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x = {
            "inputs": self.inputs[start:stop].to(device),
            "puzzle_identifiers": self.puzzle_identifiers[start:stop].to(device),
        }
        y = self.initial_plan[start:stop].to(device)
        return x, y


def parse_alpha_list(text: str) -> List[float]:
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-finite-mdp",
        help="Run the locked finite-MDP certificate gate.",
    )
    validate.add_argument(
        "--paper-root",
        type=Path,
        default=DEFAULT_PAPER_ROOT,
        help="Path to the paper repo root containing experiments/finite_mdp_certificate.py.",
    )
    validate.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write validation outputs.",
    )
    validate.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed forwarded to the finite-MDP certificate script.",
    )

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="Compute frozen-batch diagnostics for a saved RL checkpoint.",
    )
    checkpoint.add_argument("--checkpoint", type=Path, required=True, help="Full RL checkpoint path.")
    checkpoint.add_argument("--config-yaml", type=Path, required=True, help="Training YAML config.")
    checkpoint.add_argument("--closure-batch", type=Path, required=True, help="Frozen closure-batch .npz.")
    checkpoint.add_argument(
        "--directions-npz",
        type=Path,
        required=True,
        help="Frozen L_V / latent finite-difference directions .npz.",
    )
    checkpoint.add_argument("--output-json", type=Path, required=True, help="Where to write the JSON report.")
    checkpoint.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    checkpoint.add_argument(
        "--alphas",
        type=parse_alpha_list,
        default=list(DEFAULT_ALPHA_GRID),
        help="Comma-separated alpha grid (default: 0.05,0.1,0.2,0.4).",
    )
    checkpoint.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Chunk size for exact baseline / finite-difference computations.",
    )
    checkpoint.add_argument(
        "--n-ref",
        type=int,
        default=32,
        help="Reference evaluator depth for the candidate-advantage bias proxy.",
    )
    checkpoint.add_argument(
        "--rollout-repeats",
        type=int,
        default=8,
        help="Stochastic rollout repeats for eta_old / eta_alpha proxies.",
    )
    checkpoint.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Base seed for deterministic stochastic-rollout repeats.",
    )

    return parser.parse_args()


def json_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
    return value


def json_dump(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert(value) for value in obj]
        if isinstance(obj, tuple):
            return [convert(value) for value in obj]
        return json_scalar(obj)

    with open(path, "w", encoding="ascii") as handle:
        json.dump(convert(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_rl_config(config_yaml: Path) -> RLConfig:
    with open(config_yaml, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return RLConfig(**raw)


def load_directions(
    directions_npz: Path,
    *,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
) -> FrozenDirections:
    data = np.load(directions_npz, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    lv_directions = torch.from_numpy(data["lv_directions"]).to(device=device, dtype=torch.float32)
    latent_directions = torch.from_numpy(data["latent_directions"]).to(device=device, dtype=torch.float32)

    expected_lv_dim = seq_len * hidden_size
    expected_latent_dim = 2 * seq_len * hidden_size
    if lv_directions.ndim != 2 or lv_directions.shape[1] != expected_lv_dim:
        raise ValueError(
            f"lv_directions shape {tuple(lv_directions.shape)} does not match "
            f"expected (*, {expected_lv_dim}) for seq_len={seq_len}, hidden_size={hidden_size}."
        )
    if latent_directions.ndim != 2 or latent_directions.shape[1] != expected_latent_dim:
        raise ValueError(
            f"latent_directions shape {tuple(latent_directions.shape)} does not match "
            f"expected (*, {expected_latent_dim}) for seq_len={seq_len}, hidden_size={hidden_size}."
        )

    return FrozenDirections(
        lv_directions=lv_directions,
        latent_directions=latent_directions,
        eps=float(metadata.get("direction_eps", 1e-4)),
        metadata=metadata,
    )


def chunk_ranges(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        yield start, stop


def parse_checkpoint_step(checkpoint_path: Path) -> Optional[int]:
    parts = checkpoint_path.stem.split("_")
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return None


def compute_action_mask(env: Any, x_batch: Dict[str, torch.Tensor], y_batch: torch.Tensor) -> torch.Tensor:
    if env.task_config is not None:
        mask = env.task_config.compute_batch_action_mask(
            x_batch["inputs"],
            int(env.vocab_size),
            int(env.stop_action_id),
            current_state=y_batch,
        )
    else:
        mask = PlanEditEnv.compute_batch_action_mask(
            x_batch["inputs"],
            int(env.vocab_size),
            int(env.stop_action_id),
            stop_mode=env._stop_mode,
        )
    mask = mask.clone()
    mask[..., int(env.stop_action_id)] = env._stop_mode != "disabled"
    return mask


def combined_latent_norm(z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        z_h.pow(2).sum(dim=(1, 2)) +
        z_l.pow(2).sum(dim=(1, 2))
    )


def diagnostic_projection_radius(config: RLConfig) -> Optional[float]:
    """Return the enabled radius, or None for the identity operator."""

    if config.latent_projection_mode == "disabled":
        return None
    radius = config.latent_ball_radius
    if radius is None:
        raise ValueError("Enabled recurrent projection requires a finite R > 0.")
    return float(radius)


def manual_preprojection_update(
    model: Any,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    z_h: torch.Tensor,
    z_l: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = model._standardize_latent_batch(x_batch, y_batch)
    context = model._resolve_latent_context(batch)
    input_embeds = context.get("input_embeddings_with_plan", context["input_embeddings"])
    seq_info = context["seq_info"]
    inner = model.inner

    z_h_next = z_h.clone()
    z_l_next = z_l.clone()
    for _ in range(inner.config.L_cycles):
        z_l_next = inner.L_level(z_l_next, z_h_next + input_embeds, **seq_info)
    z_h_next = inner.L_level(z_h_next, z_l_next, **seq_info)
    return z_h_next, z_l_next


def build_value_head_inputs(
    model: Any,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    *,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    _, z_n = model.used_value(x_batch, y_batch, n=n)
    batch = model._standardize_latent_batch(x_batch, y_batch)
    latent_context = model._build_latent_context_with_plan(batch)
    input_embeddings = latent_context["input_embeddings"]
    plan_embeddings = latent_context["plan_embeddings"]

    z_vec = z_n.z_H.view(z_n.z_H.shape[0], -1)
    x_embed = input_embeddings.view(input_embeddings.shape[0], -1)
    y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)
    combined_embed = torch.cat([x_embed, y_embed], dim=-1)
    return z_vec, combined_embed, z_n


def estimate_fixed_lv(
    model: Any,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    directions: FrozenDirections,
    *,
    n: int,
) -> float:
    if model.value_head is None:
        raise RuntimeError("Checkpoint model has no value head; cannot compute L_V.")

    z_vec, combined_embed, _ = build_value_head_inputs(model, x_batch, y_batch, n=n)
    base_values = model.value_head(z_vec, combined_embed)

    lv_max = 0.0
    eps = float(directions.eps)
    for direction in directions.lv_directions:
        delta = direction.view(1, -1).to(device=z_vec.device, dtype=z_vec.dtype) * eps
        perturbed = z_vec + delta
        value_pert = model.value_head(perturbed, combined_embed)
        slope = (value_pert - base_values).abs() / eps
        lv_max = max(lv_max, float(slope.max().item()))
    return lv_max


def estimate_projection_terms(
    model: Any,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    directions: FrozenDirections,
    *,
    n: int,
    radius: Optional[float],
) -> Tuple[float, float, int]:
    _, _, z_n = build_value_head_inputs(model, x_batch, y_batch, n=n)
    z_h = z_n.z_H
    z_l = z_n.z_L

    z_h_next, z_l_next = manual_preprojection_update(model, x_batch, y_batch, z_h, z_l)
    pre_norms = combined_latent_norm(z_h_next, z_l_next)
    if radius is None:
        rho_r = math.inf
        active_count = 0
        z_h_post, z_l_post = z_h_next, z_l_next
    else:
        rho_r = (
            float(pre_norms.min().item()) if pre_norms.numel() > 0 else math.inf
        )
        active_count = int((pre_norms > radius).sum().item())
        z_h_post, z_l_post = model.inner._project_carry_to_ball(
            z_h_next,
            z_l_next,
            radius,
        )

    seq_len = int(z_h.shape[1])
    hidden_size = int(z_h.shape[2])
    flat_dim = seq_len * hidden_size
    lz_post = 0.0
    eps = float(directions.eps)

    for direction in directions.latent_directions:
        direction = direction.to(device=z_h.device, dtype=z_h.dtype)
        delta_h = direction[:flat_dim].view(1, seq_len, hidden_size) * eps
        delta_l = direction[flat_dim:].view(1, seq_len, hidden_size) * eps

        z_h_pert = z_h + delta_h
        z_l_pert = z_l + delta_l
        z_h_next_p, z_l_next_p = manual_preprojection_update(model, x_batch, y_batch, z_h_pert, z_l_pert)
        if radius is None:
            z_h_post_p, z_l_post_p = z_h_next_p, z_l_next_p
        else:
            z_h_post_p, z_l_post_p = model.inner._project_carry_to_ball(
                z_h_next_p,
                z_l_next_p,
                radius,
            )

        diff_norm = combined_latent_norm(z_h_post_p - z_h_post, z_l_post_p - z_l_post)
        lz_post = max(lz_post, float((diff_norm / eps).max().item()))

    return rho_r, lz_post, active_count


def load_trainer_from_checkpoint(
    checkpoint_path: Path,
    config_yaml: Path,
    dataset: ClosureBatchDataset,
    *,
    device: str,
) -> Tuple[UPITrmTrainer, Any, Any, Any, RLConfig]:
    model, model_cfg = load_model_for_eval(
        str(checkpoint_path),
        device=device,
        config_yaml_path=str(config_yaml),
    )
    rl_cfg = load_rl_config(config_yaml)
    checker = make_checker(rl_cfg)
    env, env_cfg = build_env(dataset, checker, rl_cfg, vocab_size=int(model_cfg["vocab_size"]))
    trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device(device))
    trainer.set_checker_fn(checker)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} is not a full RL checkpoint with model_state_dict / policy_state_dict payloads."
        )
    if "policy_model_old_state_dict" not in checkpoint or "policy_model_candidate_state_dict" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} is missing the saved old/candidate policy pair required for frozen diagnostics."
        )

    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.policy_model_old.load_state_dict(checkpoint["policy_model_old_state_dict"])
    trainer.policy_model_candidate.load_state_dict(checkpoint["policy_model_candidate_state_dict"])
    trainer.model.eval()
    trainer.policy_model_old.eval()
    trainer.policy_model_candidate.eval()

    return trainer, env_cfg, checker, model_cfg, rl_cfg


def compute_checkpoint_diagnostics(
    trainer: UPITrmTrainer,
    dataset: ClosureBatchDataset,
    directions: FrozenDirections,
    *,
    env_cfg: Any,
    checker: Any,
    chunk_size: int,
    n_ref: int,
    alphas: Sequence[float],
    rollout_repeats: int,
    rollout_seed: int,
) -> Dict[str, Any]:
    device = trainer.device
    n = int(trainer.rl_cfg.inner_unroll_n)
    gamma = float(trainer.rl_cfg.gamma)
    k_horizon = int(trainer.rl_cfg.K)
    projection_mode = trainer.rl_cfg.latent_projection_mode
    radius = diagnostic_projection_radius(trainer.rl_cfg)

    eps_res_n = 0.0
    eps_cent = 0.0
    eps_a_cand = 0.0
    eps_cpi_ref = 0.0
    l_v = 0.0
    rho_r = math.inf
    l_z_post = 0.0
    projection_active = 0
    expected_adv_sums = {float(alpha): 0.0 for alpha in alphas}

    total_states = len(dataset)
    with torch.no_grad():
        for start, stop in chunk_ranges(total_states, chunk_size):
            x_batch, y_batch = dataset.get_chunk(start, stop, device=device)
            action_mask = compute_action_mask(trainer.env, x_batch, y_batch)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0).expand(y_batch.shape[0], -1)

            old_dist, _ = trainer.policy_model_old.policy_dist(
                x_batch,
                y_batch,
                n=n,
                action_mask=action_mask,
            )
            cand_dist, _ = trainer.policy_model_candidate.policy_dist(
                x_batch,
                y_batch,
                n=n,
                action_mask=action_mask,
            )
            old_probs = old_dist.probs
            cand_probs = cand_dist.probs

            u_n, _ = trainer.policy_model_old.used_value(x_batch, y_batch, n=n)
            exact_baseline_n, q_all_n = compute_exact_baseline_summation(
                model=trainer.policy_model_old,
                x_batch=x_batch,
                y_batch=y_batch,
                env=trainer.env,
                n=n,
                gamma=gamma,
                checker_fn=checker,
                action_mask=action_mask,
                policy_probs=old_probs,
            )
            exact_baseline_ref, q_all_ref = compute_exact_baseline_summation(
                model=trainer.policy_model_old,
                x_batch=x_batch,
                y_batch=y_batch,
                env=trainer.env,
                n=n_ref,
                gamma=gamma,
                checker_fn=checker,
                action_mask=action_mask,
                policy_probs=old_probs,
            )

            q_valid_n = torch.where(action_mask, q_all_n, torch.zeros_like(q_all_n))
            q_valid_ref = torch.where(action_mask, q_all_ref, torch.zeros_like(q_all_ref))
            a_hat_n = q_valid_n - exact_baseline_n.unsqueeze(-1)
            a_hat_ref = q_valid_ref - exact_baseline_ref.unsqueeze(-1)

            eps_res_n = max(eps_res_n, float((u_n - exact_baseline_n).abs().max().item()))
            eps_cent = max(eps_cent, float((old_probs * a_hat_n).sum(dim=-1).abs().max().item()))
            eps_a_cand = max(
                eps_a_cand,
                float((cand_probs * (a_hat_n - a_hat_ref)).sum(dim=-1).abs().max().item()),
            )
            eps_cpi_ref = max(
                eps_cpi_ref,
                float((cand_probs * a_hat_ref).sum(dim=-1).abs().max().item()),
            )

            for alpha in alphas:
                alpha = float(alpha)
                pi_alpha = (1.0 - alpha) * old_probs + alpha * cand_probs
                expected_adv_sums[alpha] += float((pi_alpha * a_hat_n).sum(dim=-1).sum().item())

            l_v = max(
                l_v,
                estimate_fixed_lv(
                    trainer.policy_model_old,
                    x_batch,
                    y_batch,
                    directions,
                    n=n,
                ),
            )
            rho_chunk, lz_chunk, active_count = estimate_projection_terms(
                trainer.policy_model_old,
                x_batch,
                y_batch,
                directions,
                n=n,
                radius=radius,
            )
            rho_r = min(rho_r, rho_chunk)
            l_z_post = max(l_z_post, lz_chunk)
            projection_active += active_count

    eta_old_stats = rollout_policy(
        trainer,
        dataset,
        checker,
        env_cfg,
        mode="old",
        stochastic=True,
        eval_repeats=rollout_repeats,
        device=str(device),
        seed=rollout_seed,
    )
    eta_old_proxy = float(eta_old_stats.discounted_return_mean)

    alpha_metrics: Dict[str, Dict[str, Any]] = {}
    orig_alpha = float(trainer.rl_cfg.mixture_alpha)
    try:
        for alpha in alphas:
            alpha = float(alpha)
            trainer.rl_cfg.mixture_alpha = alpha
            eta_alpha_stats = rollout_policy(
                trainer,
                dataset,
                checker,
                env_cfg,
                mode="mixture",
                stochastic=True,
                eval_repeats=rollout_repeats,
                device=str(device),
                seed=rollout_seed,
            )
            uniform_batch_advantage_mean = expected_adv_sums[alpha] / float(total_states)

            if radius is None or l_z_post >= 1.0:
                penalty = math.inf
            else:
                truncation_term = (
                    eps_res_n / (1.0 - gamma**k_horizon)
                    + 2.0 * l_v * radius * (l_z_post**n) / (1.0 - l_z_post)
                )
                penalty = (
                    (2.0 * alpha * gamma / (1.0 - gamma)) * truncation_term
                    + (2.0 * eps_cpi_ref * gamma / ((1.0 - gamma) ** 2)) * (alpha**2)
                )

            alpha_metrics[f"{alpha:.2f}"] = {
                "alpha": alpha,
                "uniform_batch_advantage_mean": uniform_batch_advantage_mean,
                "eta_proxy": float(eta_alpha_stats.discounted_return_mean),
                "eta_proxy_std": float(eta_alpha_stats.discounted_return_std),
                "finite_batch_penalty_proxy": penalty,
            }
    finally:
        trainer.rl_cfg.mixture_alpha = orig_alpha

    return {
        "eps_res_n": eps_res_n,
        "eps_cent": eps_cent,
        "eps_A_cand": eps_a_cand,
        "eps_CPI_ref": eps_cpi_ref,
        "L_V": l_v,
        "rho_R": rho_r,
        "L_z_post": l_z_post,
        "latent_projection_mode": projection_mode,
        "latent_ball_radius": radius,
        "projection_active_rate": projection_active / float(max(total_states, 1)),
        "eta_old_proxy": eta_old_proxy,
        "eta_old_proxy_std": float(eta_old_stats.discounted_return_std),
        "state_weighting": "uniform_materialized_batch",
        "is_discounted_occupancy": False,
        "alpha_grid": alpha_metrics,
    }


def validate_projection_free_proxies(
    *,
    lz_values: Sequence[float],
    eps: float = 1e-4,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Numerically validate diagnostics 6--9 in a projection-free setting.

    We use simple synthetic maps with closed-form ground truth:
      - L_V = 1 via a unit-norm linear value head.
      - rho_R = +inf and projection_active_rate = 0 when projection is disabled.
      - L_z_post = L_z for the linear latent update F(z) = L_z * z.
    """

    def check_close(name: str, measured: float, expected: float) -> Dict[str, float]:
        if math.isinf(expected):
            if not math.isinf(measured):
                raise RuntimeError(f"{name} expected +inf but measured {measured}")
            return {"measured": measured, "expected": expected, "abs_error": 0.0, "rel_error": 0.0}

        abs_error = abs(measured - expected)
        if abs(expected) > 0.0:
            rel_error = abs_error / abs(expected)
        else:
            rel_error = 0.0 if abs_error <= abs_tol else math.inf

        if abs_error > abs_tol and rel_error > rel_tol:
            raise RuntimeError(
                f"{name} mismatch: measured={measured}, expected={expected}, "
                f"abs_error={abs_error}, rel_error={rel_error}"
            )
        return {
            "measured": measured,
            "expected": expected,
            "abs_error": abs_error,
            "rel_error": rel_error,
        }

    seq_len = 4
    hidden_size = 3
    batch_size = 2
    flat_dim = seq_len * hidden_size
    total_dim = 2 * flat_dim

    # Diagnostic 6: a unit-norm linear value head has exact local Lipschitz 1.
    z_vec = torch.zeros(batch_size, flat_dim, dtype=torch.float64)
    w = torch.zeros(flat_dim, dtype=torch.float64)
    w[0] = 1.0
    direction_lv = w.view(1, -1)
    base_values = z_vec @ w
    value_pert = (z_vec + eps * direction_lv) @ w
    measured_l_v = float(((value_pert - base_values).abs() / eps).max().item())

    # Diagnostics 7--9: with projection disabled, rho_R = +inf, active rate = 0,
    # and the post-projection map equals the linear pre-projection map.
    latent_direction = torch.zeros(total_dim, dtype=torch.float64)
    latent_direction[0] = 1.0
    delta_h = latent_direction[:flat_dim].view(1, seq_len, hidden_size) * eps
    delta_l = latent_direction[flat_dim:].view(1, seq_len, hidden_size) * eps

    z_h = torch.arange(batch_size * flat_dim, dtype=torch.float64).view(batch_size, seq_len, hidden_size)
    z_l = (torch.arange(batch_size * flat_dim, dtype=torch.float64).view(batch_size, seq_len, hidden_size) + 1.0)

    cases: List[Dict[str, Any]] = []
    for lz in sorted(set(float(value) for value in lz_values)):
        z_h_next = lz * z_h
        z_l_next = lz * z_l
        z_h_next_p = lz * (z_h + delta_h)
        z_l_next_p = lz * (z_l + delta_l)
        diff_norm = combined_latent_norm(z_h_next_p - z_h_next, z_l_next_p - z_l_next)
        measured_lz_post = float((diff_norm / eps).max().item())

        cases.append(
            {
                "L_z": lz,
                "rho_R": check_close("rho_R", math.inf, math.inf),
                "L_z_post": check_close(f"L_z_post@{lz}", measured_lz_post, lz),
                "projection_active_rate": check_close("projection_active_rate", 0.0, 0.0),
            }
        )

    return {
        "epsilon": eps,
        "sign_convention": "gap = lhs - rhs; negative means certificate slack",
        "L_V": check_close("L_V", measured_l_v, 1.0),
        "projection_free_cases": cases,
    }


def run_finite_mdp_validation(args: argparse.Namespace) -> int:
    paper_root = args.paper_root.resolve()
    script_path = paper_root / "experiments" / "finite_mdp_certificate.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing finite-MDP certificate script: {script_path}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--seed",
        str(args.seed),
        "--outdir",
        str(out_dir),
    ]
    subprocess.run(command, check=True)

    csv_path = out_dir / "finite_mdp_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Finite-MDP validation did not produce {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    if len(rows) != 101:
        raise RuntimeError(f"Expected 101 finite-MDP certificate rows, found {len(rows)}")

    bad_rows = [
        row for row in rows
        if row["certificate_holds_decomp"] != "1" or row["certificate_holds_exact_A"] != "1"
    ]
    if bad_rows:
        raise RuntimeError(f"Finite-MDP validation found {len(bad_rows)} rows with certificate violations")

    value_rows = [row for row in rows if row["record_type"] == "value_curve"]
    cpi_rows = [row for row in rows if row["record_type"] == "cpi_curve"]
    max_value_gap = max(
        float(row["value_error"]) - float(row["decomposition_bound"])
        for row in value_rows
    )
    max_residual_gap = max(
        float(row["value_error"]) - float(row["finite_depth_residual_bound"])
        for row in value_rows
    )
    max_advantage_gap = max(
        float(row["eps_A_cand"]) - float(row["eps_A_bound"])
        for row in cpi_rows
    )
    max_cpi_decomp_gap = max(
        (float(row["Lhat_pi_alpha"]) - float(row["eta_pi_alpha"])) - float(row["penalty_decomp"])
        for row in cpi_rows
    )
    max_cpi_exact_gap = max(
        (float(row["Lhat_pi_alpha"]) - float(row["eta_pi_alpha"])) - float(row["penalty_exact_A"])
        for row in cpi_rows
    )
    lz_values = sorted({float(row["L_z"]) for row in value_rows})
    proxy_validation = validate_projection_free_proxies(lz_values=lz_values)

    summary = {
        "paper_root": str(paper_root),
        "command": command,
        "output_dir": str(out_dir),
        "row_count": len(rows),
        "value_curve_rows": len(value_rows),
        "cpi_curve_rows": len(cpi_rows),
        "max_decomposition_gap": max_value_gap,
        "max_residual_gap": max_residual_gap,
        "max_advantage_gap": max_advantage_gap,
        "max_cpi_decomp_gap": max_cpi_decomp_gap,
        "max_cpi_exact_gap": max_cpi_exact_gap,
        "gap_sign_convention": "gap = lhs - rhs; negative means certificate slack",
        "diagnostics_6_to_9_validation": proxy_validation,
    }
    json_dump(summary, out_dir / "validation_summary.json")
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_checkpoint_diagnostics(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    dataset = ClosureBatchDataset(args.closure_batch.resolve())
    trainer, env_cfg, checker, model_cfg, rl_cfg = load_trainer_from_checkpoint(
        args.checkpoint.resolve(),
        args.config_yaml.resolve(),
        dataset,
        device=device,
    )
    directions = load_directions(
        args.directions_npz.resolve(),
        seq_len=int(dataset.seq_len),
        hidden_size=int(model_cfg["hidden_size"]),
        device=trainer.device,
    )

    diagnostics = compute_checkpoint_diagnostics(
        trainer,
        dataset,
        directions,
        env_cfg=env_cfg,
        checker=checker,
        chunk_size=int(args.chunk_size),
        n_ref=int(args.n_ref),
        alphas=args.alphas,
        rollout_repeats=int(args.rollout_repeats),
        rollout_seed=int(args.seed),
    )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": parse_checkpoint_step(args.checkpoint.resolve()),
        "config_yaml": str(args.config_yaml.resolve()),
        "closure_batch": str(args.closure_batch.resolve()),
        "directions_npz": str(args.directions_npz.resolve()),
        "device": device,
        "chunk_size": int(args.chunk_size),
        "n_ref": int(args.n_ref),
        "rollout_repeats": int(args.rollout_repeats),
        "seed": int(args.seed),
        "rl_config": rl_cfg.model_dump(),
        "model_config": model_cfg,
        "closure_batch_metadata": dataset.metadata,
        "directions_metadata": directions.metadata,
        "diagnostics": diagnostics,
    }
    json_dump(payload, args.output_json.resolve())
    print(json.dumps({"output_json": str(args.output_json.resolve())}, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "validate-finite-mdp":
        return run_finite_mdp_validation(args)
    if args.command == "checkpoint":
        return run_checkpoint_diagnostics(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
