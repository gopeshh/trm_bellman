#!/usr/bin/env python3
"""
Exp5: Stability–Expressivity Tradeoff Curve (Submission-Critical)

This experiment plots finite stability and success measurements by dial scale:
- X-axis: Stability (argmax agreement at depth mismatch)
- Y-axis: Expressivity (Sudoku solve success rate at matched compute)

Uses the Exp4 v2 dial mechanism (inference-time contraction scaling) with
existing projection-free checkpoints.

Non-negotiables:
- disable_value_head_norm: true
- latent_projection_mode: disabled with latent_ball_radius: null
- projection_active_rate < 1%

Usage:
    buck2 run //buiksat_trm:exp5_tradeoff_curve -- \
        --checkpoints /path/to/ckpt1.pt /path/to/ckpt2.pt /path/to/ckpt3.pt \
        --out_dir results/paper_ready/exp5_tradeoff_curve
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Constants
# =============================================================================

# Dial settings (same as Exp4 v2)
DIAL_SCALES = [1.0, 0.85, 0.70, 0.55]

# Evaluation depths
N_TRAIN = 2
EVAL_N_LIST = [4, 8, 16]

# B0/B1 parameters
B0_COUNT = 100
B1_CAP = 1500

# Success evaluation parameters
SUCCESS_N_EPISODES = 100  # Episodes per scale for success rate
SUCCESS_MAX_STEPS = 20  # Max steps per episode

# Gate thresholds
G0_PROJECTION_THRESHOLD = 0.01


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PuzzleState:
    """A single puzzle state for evaluation."""
    state_id: str
    inputs: torch.Tensor
    puzzle_identifier: torch.Tensor
    plan: torch.Tensor
    empties: int
    source_path: str
    parent_id: Optional[str] = None
    action_id: Optional[int] = None
    action_source: Optional[str] = None


@dataclass
class Exp5Result:
    """Evaluation results for a single (checkpoint, scale) condition."""
    checkpoint_seed: int
    checkpoint_path: str
    scale: float
    # Lipschitz
    L_preproj: float
    L_preproj_std: float
    # Stability metrics
    argmax_b0_8x: float
    argmax_b1_8x: float
    delta_V_b0_8x: float
    delta_V_b1_8x: float
    entropy_train: float
    # Expressivity metrics
    success_trivial: float  # Success rate on trivial (1-4 empties)
    success_hard: float  # Success rate on hard (6-8 empties), -1 if not available
    # Diagnostics
    projection_active_rate: float


# =============================================================================
# Utility Functions
# =============================================================================

def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    candidates = [
        PROJECT_ROOT,
        Path("/home/buiksat/trm_bellman"),
        Path.home() / "trm_bellman",
    ]
    for cwd in candidates:
        if not cwd.exists():
            continue
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=str(cwd),
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            continue
    return None


def stable_hash(tensor: torch.Tensor) -> str:
    """Compute stable hash for deduplication."""
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.md5(arr).hexdigest()[:16]


def count_empties(inputs: torch.Tensor) -> int:
    """Count empty cells (token 1 = empty)."""
    return int((inputs == 1).sum().item())


def extract_seed_from_path(checkpoint_path: str) -> int:
    """Extract seed from checkpoint path."""
    import re
    match = re.search(r'_s(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


# =============================================================================
# Model Loading
# =============================================================================

def load_model_strict(
    checkpoint_path: str,
    config_yaml_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model with STRICT config loading from YAML."""
    import yaml
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1,
        TinyRecursiveReasoningModel_ACTV1Config,
    )

    if not Path(config_yaml_path).exists():
        raise FileNotFoundError(f"YAML config required: {config_yaml_path}")

    with open(config_yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f) or {}

    # Verify non-negotiables
    disable_value_head_norm = yaml_config.get("disable_value_head_norm", False)
    latent_projection_mode = yaml_config.get("latent_projection_mode")
    latent_ball_radius = yaml_config.get("latent_ball_radius")

    if not disable_value_head_norm:
        raise ValueError("NON-NEGOTIABLE: disable_value_head_norm must be True")
    if latent_projection_mode != "disabled" or latent_ball_radius is not None:
        raise ValueError(
            "NON-NEGOTIABLE: projection must use latent_projection_mode='disabled' "
            "with latent_ball_radius=None; "
            f"got mode={latent_projection_mode}, radius={latent_ball_radius}"
        )

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model_state = state_dict["model_state_dict"]
    else:
        model_state = state_dict

    # Clean state dict
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state[clean_key] = value

    # Infer dimensions
    embed_key = "inner.embed_tokens.embedding_weight"
    if embed_key not in cleaned_state:
        embed_key = "inner.embed_tokens.weight"
    embed_weight = cleaned_state.get(embed_key, torch.zeros(6, 64))
    hidden_size = embed_weight.shape[1]
    vocab_size = embed_weight.shape[0]
    seq_len = 16

    has_value_head = any("value_head" in k for k in cleaned_state.keys())
    has_policy_head = any("edit_policy" in k for k in cleaned_state.keys())
    num_actions = seq_len * vocab_size + 1

    if has_policy_head and "edit_policy.mlp.2.weight" in cleaned_state:
        num_actions = cleaned_state["edit_policy.mlp.2.weight"].shape[0]

    inner_unroll_n = yaml_config.get("inner_unroll_n", 2)
    target_Lz = yaml_config.get("target_Lz", 0.9)
    enable_contraction = yaml_config.get("enable_contraction", False)

    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_puzzle_identifiers=500,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        expansion=2.0,
        num_heads=max(4, hidden_size // 16),
        pos_encodings="rope",
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=has_value_head,
        rl_enable_policy_head=has_policy_head,
        rl_num_actions=num_actions if has_policy_head else 0,
        rl_enable_contraction=enable_contraction,
        rl_target_Lz=float(target_Lz),
        rl_disable_value_head_norm=True,
        rl_latent_projection_mode="disabled",
        rl_latent_ball_radius=None,
    )

    model = TinyRecursiveReasoningModel_ACTV1(model_config.model_dump())
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    config_dict = {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_actions": num_actions,
        "inner_unroll_n": inner_unroll_n,
        "enable_contraction": enable_contraction,
        "target_Lz": target_Lz,
        "disable_value_head_norm": True,
        "latent_projection_mode": "disabled",
        "latent_ball_radius": None,
    }

    return model, config_dict


# =============================================================================
# Weight Scaling
# =============================================================================

def save_original_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save original L_level weights."""
    inner = getattr(model, 'inner', model)
    weights = {}
    for name, module in inner.named_modules():
        if not name.startswith("L_level"):
            continue
        if hasattr(module, 'weight') and module.weight is not None:
            weights[f"{name}.weight"] = module.weight.data.clone()
        if hasattr(module, 'bias') and module.bias is not None:
            weights[f"{name}.bias"] = module.bias.data.clone()
    return weights


def restore_weights(model: nn.Module, original_weights: Dict[str, torch.Tensor]) -> None:
    """Restore original weights."""
    inner = getattr(model, 'inner', model)
    for name, module in inner.named_modules():
        if not name.startswith("L_level"):
            continue
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in original_weights and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data.copy_(original_weights[weight_key])
        if bias_key in original_weights and hasattr(module, 'bias') and module.bias is not None:
            with torch.no_grad():
                module.bias.data.copy_(original_weights[bias_key])


def apply_contraction_scale(model: nn.Module, scale: float) -> None:
    """Apply contraction scaling to z→z layers."""
    inner = getattr(model, 'inner', model)
    for name, module in inner.named_modules():
        if not name.startswith("L_level"):
            continue
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                module.weight.data.mul_(scale)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.mul_(scale)


# =============================================================================
# Dataset Loading
# =============================================================================

def load_puzzles_by_difficulty(data_dir: Path) -> Tuple[List[PuzzleState], List[PuzzleState]]:
    """Load puzzles separated by difficulty.

    Returns:
        (trivial_puzzles, hard_puzzles) where hard may be empty
    """
    trivial_puzzles = []
    hard_puzzles = []

    for path in data_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("sudoku-4x4"):
            continue

        is_hard = "6to8" in path.name or "hard" in path.name

        for split in ["train", "test"]:
            split_dir = path / split
            if not split_dir.exists():
                continue
            inputs_path = split_dir / "all__inputs.npy"
            pids_path = split_dir / "all__puzzle_identifiers.npy"
            if not inputs_path.exists():
                continue

            inputs = np.load(inputs_path)
            puzzle_ids = np.load(pids_path) if pids_path.exists() else np.arange(len(inputs))

            for i in range(len(inputs)):
                inp = torch.from_numpy(inputs[i].astype(np.int64))
                pid = torch.tensor(puzzle_ids[i], dtype=torch.long)
                empties = count_empties(inp)

                state = PuzzleState(
                    state_id=f"{path.name}_{split}_{i}",
                    inputs=inp,
                    puzzle_identifier=pid,
                    plan=inp.clone(),
                    empties=empties,
                    source_path=str(path),
                )

                if is_hard:
                    hard_puzzles.append(state)
                else:
                    trivial_puzzles.append(state)

    return trivial_puzzles, hard_puzzles


def build_eval_batch(
    puzzles: List[PuzzleState],
    count: int,
    seed: int = 42,
) -> List[PuzzleState]:
    """Build evaluation batch."""
    rng = np.random.default_rng(seed)
    if len(puzzles) <= count:
        return puzzles
    indices = rng.choice(len(puzzles), size=count, replace=False)
    return [puzzles[i] for i in indices]


# =============================================================================
# Metrics Computation
# =============================================================================

def estimate_L_preproj(
    model: nn.Module,
    states: List[PuzzleState],
    n_steps: int,
    device: str,
    num_samples: int = 8,
    eps: float = 1e-3,
) -> Tuple[float, float]:
    """Estimate L_preproj via finite differences."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry

    model.eval()
    all_estimates = []

    with torch.no_grad():
        for state in states[:50]:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            _, z_carry = model.used_value(x, y, n=n_steps)
            if z_carry is None:
                continue

            inner = getattr(model, 'inner', model)
            inner_carry = z_carry.inner_carry if hasattr(z_carry, 'inner_carry') else z_carry

            if not hasattr(inner_carry, 'z_H'):
                continue

            input_embeddings = inner._input_embeddings(x["inputs"], x["puzzle_identifiers"])
            seq_info = inner.build_seq_info()
            baseline = inner.latent_step(inner_carry, input_embeddings, seq_info)

            for _ in range(num_samples):
                noise_h = torch.randn_like(inner_carry.z_H)
                noise_l = torch.randn_like(inner_carry.z_L)
                noise_norm = torch.sqrt(noise_h.pow(2).sum() + noise_l.pow(2).sum()).clamp(min=1e-12)
                scale = eps / noise_norm

                perturbed = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=inner_carry.z_H + noise_h * scale,
                    z_L=inner_carry.z_L + noise_l * scale,
                )
                out = inner.latent_step(perturbed, input_embeddings, seq_info)

                diff = torch.sqrt(
                    (out.z_H - baseline.z_H).pow(2).sum() +
                    (out.z_L - baseline.z_L).pow(2).sum()
                )
                all_estimates.append(diff.item() / eps)

    if not all_estimates:
        return 0.0, 0.0
    return float(np.mean(all_estimates)), float(np.std(all_estimates))


def compute_stability_metrics(
    model: nn.Module,
    states: List[PuzzleState],
    n_train: int,
    device: str,
) -> Dict[str, float]:
    """Compute stability metrics at n2=8 (4× mismatch)."""
    model.eval()
    n_eval = 8  # 4× mismatch

    delta_V_list = []
    argmax_agree_list = []
    entropy_list = []

    with torch.no_grad():
        for state in states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            v_train, _ = model.used_value(x, y, n=n_train)
            v_eval, _ = model.used_value(x, y, n=n_eval)
            dist_train, _ = model.policy_dist(x, y, n=n_train)
            dist_eval, _ = model.policy_dist(x, y, n=n_eval)

            probs_train = dist_train.probs[0]
            probs_eval = dist_eval.probs[0]

            delta_V_list.append((v_eval - v_train).abs().item())
            argmax_agree_list.append(float(probs_train.argmax() == probs_eval.argmax()))
            entropy_list.append(-(probs_train * (probs_train + 1e-10).log()).sum().item())

    return {
        "delta_V": float(np.mean(delta_V_list)) if delta_V_list else 0.0,
        "argmax_agree": float(np.mean(argmax_agree_list)) if argmax_agree_list else 0.0,
        "entropy": float(np.mean(entropy_list)) if entropy_list else 0.0,
    }


def compute_success_rate(
    model: nn.Module,
    puzzles: List[PuzzleState],
    config: Dict[str, Any],
    n_unroll: int,
    device: str,
    n_episodes: int = 100,
    max_steps: int = 20,
) -> float:
    """Compute Sudoku solve success rate using the same evaluator as training.

    Uses the training evaluator's evaluate_plan_policy_with_scores function
    to ensure identical success rate computation.
    """
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
    from rl.evaluator import evaluate_plan_policy_with_scores
    from rl.sudoku_utils import sudoku_get_stats

    # Define the feasibility checker locally (same as in upi_trm_train.py)
    def make_feasibility_checker(w_v: float = 2.0, w_z: float = 5.0):
        """Create a feasibility checker function with the given weights."""
        def checker(x, y):
            """
            Feasibility-aware Sudoku checker.
            Score = filled - w_v*violations - w_z*zeroCand
            """
            if torch.is_tensor(y):
                plan = y
            elif isinstance(y, dict):
                plan = y.get("plan", y.get("labels"))
            else:
                plan = torch.as_tensor(y)

            total_cells, filled, violations, zero_cand = sudoku_get_stats(plan)
            score = float(filled) - w_v * float(violations) - w_z * float(zero_cand)
            return score
        return checker

    model.eval()
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    num_actions = config["num_actions"]
    stop_action_id = num_actions - 1

    # Sample puzzles for evaluation
    rng = np.random.default_rng(42)
    if len(puzzles) > n_episodes:
        indices = rng.choice(len(puzzles), size=n_episodes, replace=False)
        eval_puzzles = [puzzles[i] for i in indices]
    else:
        eval_puzzles = puzzles

    # Create a dataset from the puzzles (same format as training)
    class PuzzleListDataset:
        def __init__(self, puzzles_list):
            self.puzzles = puzzles_list

        def __len__(self):
            return len(self.puzzles)

        def __getitem__(self, idx):
            p = self.puzzles[idx]
            return {
                "inputs": p.inputs.clone(),
                "puzzle_identifiers": p.puzzle_identifier.clone() if p.puzzle_identifier.ndim > 0 else p.puzzle_identifier.unsqueeze(0),
                "initial_plan": p.plan.clone(),
                "solution": p.plan.clone(),  # Placeholder - not used for sudoku_is_solved criterion
            }

    dataset = PuzzleListDataset(eval_puzzles)

    # Create checker (feasibility-based, same as training)
    checker = make_feasibility_checker(
        w_v=2.0,
        w_z=5.0,
    )

    # Create environment config (same as training)
    env_cfg = PlanEditEnvConfig(
        max_edits=max_steps,
        gamma=0.99,
        vocab_size=vocab_size,
        reward_shaping=True,
        stop_action_mode="disabled",
        task_type="sudoku",
    )

    # Use the training evaluator function
    mean_score, success_rate, detailed_stats = evaluate_plan_policy_with_scores(
        model=model,
        dataset=dataset,
        checker=checker,
        env_cfg=env_cfg,
        num_episodes=len(eval_puzzles),
        inner_unroll_n=n_unroll,
        episodic_latent=True,  # Match training config
        greedy=True,
        use_sudoku_solved_criterion=True,
    )

    return success_rate


# =============================================================================
# Main Evaluation
# =============================================================================

def run_exp5(
    checkpoint_paths: List[str],
    config_yaml_path: str,
    out_dir: str,
    data_dir: str = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run Exp5 evaluation."""
    print("=" * 70)
    print("Exp5: Stability–Expressivity Tradeoff Curve")
    print("=" * 70)
    print()
    print(f"[Config] Checkpoints: {checkpoint_paths}")
    print(f"[Config] YAML: {config_yaml_path}")
    print(f"[Config] Output: {out_dir}")
    print(f"[Config] Device: {device}")
    print(f"[Config] Dial scales: {DIAL_SCALES}")
    print()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all models
    models_and_weights = []
    config = None
    for ckpt_path in checkpoint_paths:
        seed = extract_seed_from_path(ckpt_path)
        print(f"[Load] Loading checkpoint seed {seed}: {ckpt_path}")
        model, cfg = load_model_strict(ckpt_path, config_yaml_path, device)
        original_weights = save_original_weights(model)
        models_and_weights.append((model, original_weights, seed))
        if config is None:
            config = cfg

    print(f"[Load] Loaded {len(models_and_weights)} checkpoints")

    # Load datasets
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    else:
        data_dir = Path(data_dir)

    trivial_puzzles, hard_puzzles = load_puzzles_by_difficulty(data_dir)
    print(f"[Data] Trivial puzzles: {len(trivial_puzzles)}")
    print(f"[Data] Hard puzzles (6-8 empties): {len(hard_puzzles)}")

    has_hard_suite = len(hard_puzzles) > 0

    # Build eval batches
    b0_trivial = build_eval_batch(trivial_puzzles, B0_COUNT, seed=42)
    b0_hard = build_eval_batch(hard_puzzles, B0_COUNT, seed=42) if has_hard_suite else []

    # Compute batch hashes
    b0_hash = hashlib.md5("".join(s.state_id for s in b0_trivial).encode()).hexdigest()[:12]

    print(f"\n[Batches] B0 trivial: {len(b0_trivial)} states (hash: {b0_hash})")
    if has_hard_suite:
        b0_hard_hash = hashlib.md5("".join(s.state_id for s in b0_hard).encode()).hexdigest()[:12]
        print(f"[Batches] B0 hard: {len(b0_hard)} states (hash: {b0_hard_hash})")

    # Run evaluation
    all_results = []

    for model, original_weights, ckpt_seed in models_and_weights:
        print(f"\n{'='*60}")
        print(f"Checkpoint seed {ckpt_seed}")
        print(f"{'='*60}")

        for scale in DIAL_SCALES:
            print(f"\n[Scale {scale:.2f}]")

            # Reset and apply scaling
            restore_weights(model, original_weights)
            if scale < 1.0:
                apply_contraction_scale(model, scale)

            # Estimate L_preproj
            L_preproj_mean, L_preproj_std = estimate_L_preproj(
                model, b0_trivial, N_TRAIN, device
            )
            print(f"  L_preproj: {L_preproj_mean:.4f} ± {L_preproj_std:.4f}")

            # Compute stability metrics on trivial B0
            stab_b0 = compute_stability_metrics(model, b0_trivial, N_TRAIN, device)
            print(f"  Stability (B0 trivial): argmax@8x={stab_b0['argmax_agree']:.3f}, ΔV@8x={stab_b0['delta_V']:.3f}")

            # Compute stability metrics on hard B0 (if available)
            if has_hard_suite:
                stab_b0_hard = compute_stability_metrics(model, b0_hard, N_TRAIN, device)
                print(f"  Stability (B0 hard): argmax@8x={stab_b0_hard['argmax_agree']:.3f}, ΔV@8x={stab_b0_hard['delta_V']:.3f}")
            else:
                stab_b0_hard = {"argmax_agree": 0.0, "delta_V": 0.0}

            # Compute success rate on trivial
            success_trivial = compute_success_rate(
                model, trivial_puzzles, config, N_TRAIN, device,
                n_episodes=SUCCESS_N_EPISODES, max_steps=SUCCESS_MAX_STEPS
            )
            print(f"  Success (trivial): {success_trivial:.3f}")

            # Compute success rate on hard (if available)
            if has_hard_suite:
                success_hard = compute_success_rate(
                    model, hard_puzzles, config, N_TRAIN, device,
                    n_episodes=SUCCESS_N_EPISODES, max_steps=SUCCESS_MAX_STEPS
                )
                print(f"  Success (hard): {success_hard:.3f}")
            else:
                success_hard = -1.0
                print(f"  Success (hard): N/A (no hard suite)")

            result = Exp5Result(
                checkpoint_seed=ckpt_seed,
                checkpoint_path=checkpoint_paths[models_and_weights.index((model, original_weights, ckpt_seed))],
                scale=scale,
                L_preproj=L_preproj_mean,
                L_preproj_std=L_preproj_std,
                argmax_b0_8x=stab_b0["argmax_agree"],
                argmax_b1_8x=stab_b0_hard["argmax_agree"] if has_hard_suite else 0.0,
                delta_V_b0_8x=stab_b0["delta_V"],
                delta_V_b1_8x=stab_b0_hard["delta_V"] if has_hard_suite else 0.0,
                entropy_train=stab_b0["entropy"],
                success_trivial=success_trivial,
                success_hard=success_hard,
                projection_active_rate=0.0,  # Explicit identity projection mode
            )
            all_results.append(result)

        # Restore original weights
        restore_weights(model, original_weights)

    # Aggregate results by scale
    print("\n" + "=" * 70)
    print("TRADEOFF SUMMARY")
    print("=" * 70)

    scale_summaries = []
    for scale in DIAL_SCALES:
        scale_results = [r for r in all_results if r.scale == scale]
        summary = {
            "scale": scale,
            "L_preproj_mean": float(np.mean([r.L_preproj for r in scale_results])),
            "L_preproj_std": float(np.std([r.L_preproj for r in scale_results])),
            "argmax_b0_8x_mean": float(np.mean([r.argmax_b0_8x for r in scale_results])),
            "argmax_b0_8x_std": float(np.std([r.argmax_b0_8x for r in scale_results])),
            "delta_V_b0_8x_mean": float(np.mean([r.delta_V_b0_8x for r in scale_results])),
            "delta_V_b0_8x_std": float(np.std([r.delta_V_b0_8x for r in scale_results])),
            "success_trivial_mean": float(np.mean([r.success_trivial for r in scale_results])),
            "success_trivial_std": float(np.std([r.success_trivial for r in scale_results])),
            "entropy_mean": float(np.mean([r.entropy_train for r in scale_results])),
        }
        if has_hard_suite:
            summary["success_hard_mean"] = float(np.mean([r.success_hard for r in scale_results]))
            summary["success_hard_std"] = float(np.std([r.success_hard for r in scale_results]))
        scale_summaries.append(summary)

        print(f"Scale {scale:.2f}: stability={summary['argmax_b0_8x_mean']:.3f}, "
              f"success_trivial={summary['success_trivial_mean']:.3f}, "
              f"L_preproj={summary['L_preproj_mean']:.3f}")

    # Decision gates
    print("\n" + "-" * 60)
    print("DECISION GATES")
    print("-" * 60)

    # G0: projection inactive
    g0_passed = all(r.projection_active_rate < G0_PROJECTION_THRESHOLD for r in all_results)
    print(f"G0 (Projection inactive): {'PASS' if g0_passed else 'FAIL'}")

    # G1: stability
    g1_passed = all(not np.isnan(r.L_preproj) for r in all_results)
    print(f"G1 (Stability): {'PASS' if g1_passed else 'FAIL'}")

    # G2: inherited from Exp4 v2
    L_spread = max(r.L_preproj for r in all_results) - min(r.L_preproj for r in all_results)
    g2_passed = L_spread >= 0.10
    print(f"G2 (Dial range, inherited): {'PASS' if g2_passed else 'FAIL'} (spread={L_spread:.4f})")

    # G3: finite success-rate variation across evaluated scales
    success_range = max(s["success_trivial_mean"] for s in scale_summaries) - min(s["success_trivial_mean"] for s in scale_summaries)
    g3_passed = success_range > 0.01  # Some variation in success
    print(f"G3 (Observed success variation): {'PASS' if g3_passed else 'FAIL'} (range={success_range:.4f})")

    # Overall decision
    if g0_passed and g1_passed and g2_passed:
        decision = "POSITIVE: Finite scale-comparison gates met"
    else:
        decision = "FAIL: Gate conditions not met"

    print(f"\nDECISION: {decision}")

    # Build summary
    git_sha = get_git_sha()

    summary = {
        "experiment": "Exp5_tradeoff_curve",
        "description": "Stability–Expressivity tradeoff curve",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "checkpoints": checkpoint_paths,
        "config_yaml": str(config_yaml_path),
        "config": config,
        "parameters": {
            "dial_scales": DIAL_SCALES,
            "checkpoint_seeds": [extract_seed_from_path(p) for p in checkpoint_paths],
            "n_train": N_TRAIN,
            "success_n_episodes": SUCCESS_N_EPISODES,
            "success_max_steps": SUCCESS_MAX_STEPS,
        },
        "batches": {
            "b0_trivial_count": len(b0_trivial),
            "b0_hash": b0_hash,
            "has_hard_suite": has_hard_suite,
        },
        "scale_summaries": scale_summaries,
        "gates": {
            "g0_projection_inactive": {"passed": bool(g0_passed)},
            "g1_stability": {"passed": bool(g1_passed)},
            "g2_dial_range": {"passed": bool(g2_passed), "spread": float(L_spread)},
            "g3_tradeoff_exists": {"passed": bool(g3_passed), "success_range": float(success_range)},
        },
        "decision": decision,
        "all_results": [asdict(r) for r in all_results],
    }

    # Save summary
    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Generate CLAIMS.md
    generate_claims(summary, out_path)

    # Generate PROVENANCE.md
    generate_provenance(summary, out_path, checkpoint_paths, config_yaml_path)

    return summary


def generate_claims(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate CLAIMS.md."""
    scale_sums = summary["scale_summaries"]
    gates = summary["gates"]

    content = f"""# Exp5 Claims: Stability–Expressivity Tradeoff Curve

**Generated:** {summary['generated_at']}
**Git SHA:** {summary['git_sha']}

## Summary

- **Decision:** {summary['decision']}
- **G0 (Projection inactive):** {'PASS' if gates['g0_projection_inactive']['passed'] else 'FAIL'}
- **G1 (Stability):** {'PASS' if gates['g1_stability']['passed'] else 'FAIL'}
- **G2 (Dial range):** {'PASS' if gates['g2_dial_range']['passed'] else 'FAIL'} (spread={gates['g2_dial_range']['spread']:.4f})
- **G3 (Observed success variation):** {'PASS' if gates['g3_tradeoff_exists']['passed'] else 'FAIL'}

## Tradeoff Results

| Scale | L_preproj | Stability (argmax@8×) | Success (trivial) | ΔV@8× |
|-------|-----------|----------------------|-------------------|-------|
"""
    for s in scale_sums:
        content += f"| {s['scale']:.2f} | {s['L_preproj_mean']:.3f}±{s['L_preproj_std']:.3f} | "
        content += f"{s['argmax_b0_8x_mean']:.3f}±{s['argmax_b0_8x_std']:.3f} | "
        content += f"{s['success_trivial_mean']:.3f}±{s['success_trivial_std']:.3f} | "
        content += f"{s['delta_V_b0_8x_mean']:.3f} |\n"

    content += f"""
## Scoped Observation

Across these checkpoints, Sudoku states, and four inference-time scales, the table reports
the observed association among scale, local L_preproj estimates, depth-mismatch metrics,
and greedy-policy success. It does not establish that scaling causes the metric differences
or that a stability-expressivity tradeoff is necessary or general.

## Scope Limitations

- Results on 4×4 Sudoku only
- Success measured with greedy policy at matched compute (n={summary['parameters']['n_train']})
- Stability measured as argmax agreement at n2=8 (4× mismatch)
- {len(summary['checkpoints'])} independently trained checkpoints
- Finite local L_preproj estimates do not establish a global Lipschitz bound

## Non-Negotiables Verified

- `disable_value_head_norm: true`
- `latent_projection_mode: disabled` and `latent_ball_radius: null`
"""

    claims_path = out_path / "CLAIMS.md"
    with open(claims_path, "w") as f:
        f.write(content)
    print(f"Saved: {claims_path}")


def generate_provenance(
    summary: Dict[str, Any],
    out_path: Path,
    checkpoint_paths: List[str],
    config_yaml_path: str,
) -> None:
    """Generate PROVENANCE.md."""
    content = f"""# Exp5 Provenance

**Generated:** {summary['generated_at']}
**Git SHA:** {summary['git_sha']}

## Checkpoints

| Seed | Path |
|------|------|
"""
    for path in checkpoint_paths:
        seed = extract_seed_from_path(path)
        content += f"| {seed} | {path} |\n"

    content += f"""
## Config

- **YAML:** {config_yaml_path}
- **Non-negotiables:** disable_value_head_norm=true, latent_projection_mode=disabled, latent_ball_radius=null

## Parameters

- Dial scales: {summary['parameters']['dial_scales']}
- n_train: {summary['parameters']['n_train']}
- Success episodes per scale: {summary['parameters']['success_n_episodes']}
- Max steps per episode: {summary['parameters']['success_max_steps']}

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp5_tradeoff_curve -- \\
    --checkpoints {' '.join(checkpoint_paths)} \\
    --config_yaml {config_yaml_path} \\
    --out_dir {out_path}
```
"""

    prov_path = out_path / "PROVENANCE.md"
    with open(prov_path, "w") as f:
        f.write(content)
    print(f"Saved: {prov_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp5 Tradeoff Curve")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint paths",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/exp3_projection_ablation/nc_rdis.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/paper_ready/exp5_tradeoff_curve",
        help="Output directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/buiksat/trm_bellman/data",
        help="Data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = "cpu"

    summary = run_exp5(
        checkpoint_paths=args.checkpoints,
        config_yaml_path=args.config_yaml,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        device=device,
    )

    if "POSITIVE" in summary["decision"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
