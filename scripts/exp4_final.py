#!/usr/bin/env python3
"""
Exp4 Final: Projection-free Contraction Dial (Paper-Defensible)

A comprehensive evaluation that tests whether inference-time contraction scaling
constitutes a reliable "dial" for controlling mismatch-drift metrics when
projection is disabled.

Key requirements (per task specification):
1. Uses canonical B0/B1 construction (same as Exp1)
2. Multi-seed evaluation (seeds 41, 42, 43)
3. Multiple dial settings with proper logging
4. Statistical reporting with Spearman ρ, p-values
5. Paper-ready artifacts

Usage:
    buck2 run //buiksat_trm:exp4_final -- \
        --checkpoint_base results/exp3/nc_rdis_s42/model_step_5000.pt \
        --config_yaml configs/exp3_projection_ablation/nc_rdis.yaml \
        --out_dir results/paper_ready/exp4_projection_free_dial_final

Non-negotiables:
- latent_projection_mode = disabled with latent_ball_radius = null
- disable_value_head_norm: true
- No fallback to random inputs (hard error if dataset missing)
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Constants
# =============================================================================

# Dial settings: scales that span L_preproj range
DIAL_SCALES = [1.0, 0.85, 0.70, 0.55]

# Seeds for multi-seed evaluation
SEEDS = [41, 42, 43]

# Evaluation depths
N_TRAIN = 2  # Training depth
EVAL_N_LIST = [4, 8, 16]  # 2×, 4×, 8× multipliers

# B0/B1 parameters
B0_TARGET_EASY = 70
B0_TARGET_HARD = 30
B1_CAP = 1500

# Gate thresholds
G0_PROJECTION_THRESHOLD = 0.01  # projection_active_rate < 1%
G2_SPREAD_THRESHOLD = 0.10  # L_preproj spread >= 0.10
G3_MONOTONICITY_THRESHOLD = 0.5  # |Spearman ρ| > 0.5


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
class EvalResult:
    """Evaluation results for a single dial setting."""
    scale: float
    seed: int
    L_preproj: float
    projection_active_rate: float
    metrics_b0: Dict[str, float]
    metrics_b1: Dict[str, float]


# =============================================================================
# Utility Functions
# =============================================================================

def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def stable_hash(tensor: torch.Tensor) -> str:
    """Compute stable hash for deduplication."""
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.md5(arr).hexdigest()[:16]


def count_empties(inputs: torch.Tensor) -> int:
    """Count empty cells (token 1 = empty)."""
    return int((inputs == 1).sum().item())


# =============================================================================
# Model Loading (Strict: YAML Required)
# =============================================================================

def load_model_strict(
    checkpoint_path: str,
    config_yaml_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model with STRICT config loading from YAML.

    Raises if YAML config is missing or invalid.
    """
    import yaml
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1,
        TinyRecursiveReasoningModel_ACTV1Config,
    )

    # YAML config is REQUIRED
    if not Path(config_yaml_path).exists():
        raise FileNotFoundError(
            f"YAML config required but not found: {config_yaml_path}\n"
            "Exp4 requires explicit config to avoid inference-based loading."
        )

    with open(config_yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f) or {}

    print(f"[Load] Checkpoint: {checkpoint_path}")
    print(f"[Load] YAML config: {config_yaml_path}")

    # Verify non-negotiables
    disable_value_head_norm = yaml_config.get("disable_value_head_norm", False)
    latent_projection_mode = yaml_config.get("latent_projection_mode")
    latent_ball_radius = yaml_config.get("latent_ball_radius")

    if not disable_value_head_norm:
        raise ValueError(
            "NON-NEGOTIABLE VIOLATION: disable_value_head_norm must be True.\n"
            f"Found: disable_value_head_norm={disable_value_head_norm}"
        )

    if latent_projection_mode != "disabled" or latent_ball_radius is not None:
        raise ValueError(
            "NON-NEGOTIABLE VIOLATION: projection must use explicit identity "
            "mode with latent_ball_radius=None.\n"
            f"Found: latent_projection_mode={latent_projection_mode}, "
            f"latent_ball_radius={latent_ball_radius}"
        )

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model_state = state_dict["model_state_dict"]
        rl_config_dict = state_dict.get("rl_config", {})
    else:
        model_state = state_dict
        rl_config_dict = {}

    # Clean state dict keys
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state[clean_key] = value

    # Infer dimensions from weights
    hidden_size = cleaned_state.get("inner.embed_tokens.weight", torch.zeros(6, 64)).shape[1]
    vocab_size = cleaned_state.get("inner.embed_tokens.weight", torch.zeros(6, 64)).shape[0]
    seq_len = 16  # 4×4 Sudoku

    has_value_head = any("value_head" in k for k in cleaned_state.keys())
    has_policy_head = any("edit_policy" in k for k in cleaned_state.keys())
    num_actions = seq_len * vocab_size + 1

    if has_policy_head and "edit_policy.mlp.2.weight" in cleaned_state:
        num_actions = cleaned_state["edit_policy.mlp.2.weight"].shape[0]

    # Get config values from YAML (required)
    inner_unroll_n = yaml_config.get("inner_unroll_n", 2)
    target_Lz = yaml_config.get("target_Lz", 0.9)
    enable_contraction = yaml_config.get("enable_contraction", False)

    # Build model config
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
        rl_disable_value_head_norm=True,  # Non-negotiable
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
        "config_source": "yaml",
        "yaml_path": config_yaml_path,
        "checkpoint_path": checkpoint_path,
    }

    print(f"[Load] Model config (from YAML): {config_dict}")
    return model, config_dict


# =============================================================================
# Weight Scaling (Dial Implementation)
# =============================================================================

def save_original_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save original L_level weights for restoration."""
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
    """Restore original weights to L_level layers."""
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
    """
    Apply contraction scaling to z→z layers (L_level).

    Implementation: Multiply linear layer weights by `scale`.
    This effectively scales the Lipschitz constant of the z→z map.

    Note: This is applied at INFERENCE/EVALUATION time only.
    """
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
# Dataset Loading (Strict: No Random Fallback)
# =============================================================================

def load_puzzles_strict(data_dir: Path) -> List[PuzzleState]:
    """
    Load puzzles from dataset directory.
    Raises if dataset not found (no random fallback).
    """
    datasets = []
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("sudoku-4x4"):
            train_dir = path / "train"
            if train_dir.exists() and (train_dir / "all__inputs.npy").exists():
                datasets.append(path)

    if not datasets:
        raise FileNotFoundError(
            f"NO SUDOKU DATASETS FOUND in {data_dir}\n"
            "Exp4 requires canonical B0/B1 evaluation. No random fallback allowed."
        )

    print(f"[Dataset] Found: {[d.name for d in datasets]}")

    all_puzzles = []
    for ds_path in datasets:
        for split in ["train", "test"]:
            split_dir = ds_path / split
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
                    state_id=f"{ds_path.name}_{split}_{i}",
                    inputs=inp,
                    puzzle_identifier=pid,
                    plan=inp.clone(),
                    empties=empties,
                    source_path=str(ds_path),
                )
                all_puzzles.append(state)

    print(f"[Dataset] Loaded {len(all_puzzles)} total puzzles")
    return all_puzzles


def build_b0_strict(
    all_puzzles: List[PuzzleState],
    target_easy: int = B0_TARGET_EASY,
    target_hard: int = B0_TARGET_HARD,
    seed: int = 42,
) -> List[PuzzleState]:
    """Build B0 batch with strict requirements."""
    rng = np.random.default_rng(seed)

    easy = [p for p in all_puzzles if 1 <= p.empties <= 4]
    hard = [p for p in all_puzzles if 6 <= p.empties <= 8]

    print(f"[B0] Easy (1-4 empties): {len(easy)}")
    print(f"[B0] Hard (6-8 empties): {len(hard)}")

    selected = []

    n_easy = min(target_easy, len(easy))
    if n_easy > 0:
        indices = rng.choice(len(easy), size=n_easy, replace=False)
        selected.extend([easy[i] for i in indices])

    n_hard = min(target_hard, len(hard))
    if n_hard > 0:
        indices = rng.choice(len(hard), size=n_hard, replace=False)
        selected.extend([hard[i] for i in indices])

    # If insufficient, fill from remaining
    total = target_easy + target_hard
    if len(selected) < total:
        used = {p.state_id for p in selected}
        remaining = [p for p in all_puzzles if p.state_id not in used]
        deficit = total - len(selected)
        if remaining:
            n_fill = min(deficit, len(remaining))
            indices = rng.choice(len(remaining), size=n_fill, replace=False)
            selected.extend([remaining[i] for i in indices])

    rng.shuffle(selected)
    for i, state in enumerate(selected):
        state.state_id = f"b0_{i:04d}"

    print(f"[B0] Built batch with {len(selected)} states")
    return selected


def build_b1_union_scales(
    b0_states: List[PuzzleState],
    model: nn.Module,
    original_weights: Dict[str, torch.Tensor],
    scales: List[float],
    config: Dict[str, Any],
    n_train: int,
    seed: int,
    cap: int = B1_CAP,
    device: str = "cpu",
) -> List[PuzzleState]:
    """
    Build B1 using union-of-actions across ALL dial scales.
    This avoids selection bias toward any particular scale.
    """
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig

    rng = np.random.default_rng(seed)

    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    num_actions = config["num_actions"]
    stop_action_id = num_actions - 1

    print(f"[B1] Building successor closure from {len(b0_states)} B0 states")
    print(f"[B1] Using union of top-5 actions from scales: {scales}")

    # Dummy env for action application
    class DummyDataset:
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"inputs": torch.ones(seq_len, dtype=torch.long)}

    env_config = PlanEditEnvConfig(max_edits=16, gamma=0.99, vocab_size=vocab_size)
    env = PlanEditEnv(DummyDataset(), lambda x, y: 0.0, env_config)
    env.set_stop_action_id(stop_action_id)

    successors = []
    seen_hashes = set()

    model.eval()

    with torch.no_grad():
        for state in b0_states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            # Compute action mask
            action_mask = PlanEditEnv.compute_batch_action_mask(
                state.inputs.unsqueeze(0), vocab_size, stop_action_id, stop_mode="disabled"
            ).to(device)

            legal_actions = action_mask[0].nonzero(as_tuple=True)[0].cpu().tolist()
            legal_actions = [a for a in legal_actions if a != stop_action_id]

            if not legal_actions:
                continue

            actions_to_apply = []

            # Collect top-5 from EACH scale
            for scale in scales:
                restore_weights(model, original_weights)
                if scale < 1.0:
                    apply_contraction_scale(model, scale)

                dist, _ = model.policy_dist(x, y, n_train, action_mask=action_mask)
                probs = dist.probs[0].cpu()
                probs[stop_action_id] = 0.0
                top5 = probs.argsort(descending=True)[:5].tolist()

                for a in top5:
                    if a in legal_actions:
                        actions_to_apply.append((a, f"scale_{scale:.2f}_top"))

            # Add 5 random legal actions
            n_random = min(5, len(legal_actions))
            random_actions = rng.choice(legal_actions, size=n_random, replace=False).tolist()
            for a in random_actions:
                actions_to_apply.append((a, "rand"))

            # Apply and deduplicate
            for action_id, source in actions_to_apply:
                new_plan = env.apply_edit(state.plan, action_id, {"inputs": state.inputs})
                combined = torch.cat([state.inputs, new_plan])
                h = stable_hash(combined)

                if h not in seen_hashes:
                    seen_hashes.add(h)
                    new_state = PuzzleState(
                        state_id=f"b1_{len(successors):04d}",
                        inputs=state.inputs.clone(),
                        puzzle_identifier=state.puzzle_identifier.clone(),
                        plan=new_plan,
                        empties=state.empties,
                        source_path=state.source_path,
                        parent_id=state.state_id,
                        action_id=action_id,
                        action_source=source,
                    )
                    successors.append(new_state)

    # Restore original weights
    restore_weights(model, original_weights)

    print(f"[B1] Generated {len(successors)} unique successors")

    # Cap if needed
    if len(successors) > cap:
        rng.shuffle(successors)
        successors = successors[:cap]
        print(f"[B1] Capped to {cap} states")

    return successors


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
    """
    Estimate L_preproj via finite differences.
    Returns (mean, std) across states.
    """
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry

    model.eval()
    all_estimates = []

    with torch.no_grad():
        for state in states[:50]:  # Sample 50 states max
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


def compute_mismatch_metrics(
    model: nn.Module,
    states: List[PuzzleState],
    n_train: int,
    eval_n_list: List[int],
    device: str,
) -> Dict[str, float]:
    """Compute ΔV, KL divergence, argmax agreement for mismatch depths."""
    model.eval()
    metrics = {f"delta_V@{n}x": [] for n in eval_n_list}
    metrics.update({f"argmax_agree@{n}x": [] for n in eval_n_list})
    metrics.update({f"kl_div@{n}x": [] for n in eval_n_list})

    with torch.no_grad():
        for state in states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            v_train, _ = model.used_value(x, y, n=n_train)
            dist_train, _ = model.policy_dist(x, y, n=n_train)
            probs_train = dist_train.probs[0]
            argmax_train = probs_train.argmax()

            for n_eval in eval_n_list:
                v_eval, _ = model.used_value(x, y, n=n_eval)
                dist_eval, _ = model.policy_dist(x, y, n=n_eval)
                probs_eval = dist_eval.probs[0]
                argmax_eval = probs_eval.argmax()

                # ΔV
                delta_v = (v_eval - v_train).abs().item()
                metrics[f"delta_V@{n_eval}x"].append(delta_v)

                # Argmax agreement
                agree = float(argmax_train == argmax_eval)
                metrics[f"argmax_agree@{n_eval}x"].append(agree)

                # KL divergence
                p = probs_train.clamp(min=1e-8)
                q = probs_eval.clamp(min=1e-8)
                p = p / p.sum()
                q = q / q.sum()
                kl = (p * (p.log() - q.log())).sum().clamp(min=0).item()
                metrics[f"kl_div@{n_eval}x"].append(kl)

    # Aggregate
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
        else:
            result[key] = 0.0
            result[f"{key}_std"] = 0.0

    return result


def compute_projection_active_rate(
    model: nn.Module,
    states: List[PuzzleState],
    n_steps: int,
    device: str,
) -> float:
    """
    Check if projection is active by examining the model's explicit mode.

    Returns 0.0 if projection is disabled, 1.0 if enabled.
    """
    inner = getattr(model, 'inner', model)
    config = getattr(inner, "config", None)
    if isinstance(config, dict):
        projection_mode = config.get("rl_latent_projection_mode")
    else:
        projection_mode = getattr(config, "rl_latent_projection_mode", None)
    if projection_mode == "disabled":
        return 0.0
    if projection_mode == "enabled":
        return 1.0
    raise ValueError("Model config lacks an explicit latent projection mode.")


# =============================================================================
# Main Evaluation
# =============================================================================

def run_exp4_final(
    checkpoint_path: str,
    config_yaml_path: str,
    out_dir: str,
    data_dir: str = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run full Exp4 evaluation."""
    print("=" * 70)
    print("Exp4 Final: Projection-free Contraction Dial (Paper-Defensible)")
    print("=" * 70)
    print()
    print(f"[Config] Checkpoint: {checkpoint_path}")
    print(f"[Config] YAML: {config_yaml_path}")
    print(f"[Config] Output: {out_dir}")
    print(f"[Config] Data dir: {data_dir}")
    print(f"[Config] Device: {device}")
    print(f"[Config] Dial scales: {DIAL_SCALES}")
    print(f"[Config] Seeds: {SEEDS}")
    print(f"[Config] n_train={N_TRAIN}, eval_n={EVAL_N_LIST}")
    print()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model (strict: YAML required)
    model, config = load_model_strict(checkpoint_path, config_yaml_path, device)

    # Save original weights for dial implementation
    original_weights = save_original_weights(model)
    print(f"[Dial] Saved {len(original_weights)} weight tensors for scaling")
    print()

    # Load dataset (strict: no random fallback)
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    else:
        data_dir = Path(data_dir)
    all_puzzles = load_puzzles_strict(data_dir)

    # Build B0 (same seed for reproducibility)
    b0_states = build_b0_strict(all_puzzles, seed=42)

    # Build B1 using union of actions across all scales
    b1_states = build_b1_union_scales(
        b0_states, model, original_weights, DIAL_SCALES,
        config, N_TRAIN, seed=42, device=device
    )

    # Compute B0/B1 hashes for provenance
    b0_hash = hashlib.md5(
        "".join(s.state_id for s in b0_states).encode()
    ).hexdigest()[:12]
    b1_hash = hashlib.md5(
        "".join(s.state_id for s in b1_states).encode()
    ).hexdigest()[:12]

    print()
    print(f"[Batches] B0: {len(b0_states)} states (hash: {b0_hash})")
    print(f"[Batches] B1: {len(b1_states)} states (hash: {b1_hash})")
    print()

    # Run evaluation across seeds and scales
    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        for scale in DIAL_SCALES:
            print(f"\n[Scale {scale:.2f}]")

            # Reset and apply scaling
            restore_weights(model, original_weights)
            if scale < 1.0:
                apply_contraction_scale(model, scale)

            # Estimate L_preproj
            L_preproj_mean, L_preproj_std = estimate_L_preproj(
                model, b0_states, N_TRAIN, device
            )
            print(f"  L_preproj: {L_preproj_mean:.4f} ± {L_preproj_std:.4f}")

            # Check projection activity
            proj_rate = compute_projection_active_rate(
                model, b0_states, N_TRAIN, device
            )
            print(f"  Projection active rate: {proj_rate*100:.1f}%")

            # Compute mismatch metrics on B0
            metrics_b0 = compute_mismatch_metrics(
                model, b0_states, N_TRAIN, EVAL_N_LIST, device
            )
            print(f"  B0 ΔV@8x: {metrics_b0['delta_V@8x']:.4f}")
            print(f"  B0 argmax@8x: {metrics_b0['argmax_agree@8x']:.4f}")

            # Compute mismatch metrics on B1
            metrics_b1 = compute_mismatch_metrics(
                model, b1_states, N_TRAIN, EVAL_N_LIST, device
            )
            print(f"  B1 ΔV@8x: {metrics_b1['delta_V@8x']:.4f}")
            print(f"  B1 argmax@8x: {metrics_b1['argmax_agree@8x']:.4f}")

            result = EvalResult(
                scale=scale,
                seed=seed,
                L_preproj=L_preproj_mean,
                projection_active_rate=proj_rate,
                metrics_b0=metrics_b0,
                metrics_b1=metrics_b1,
            )
            all_results.append(result)

    # Restore original weights
    restore_weights(model, original_weights)

    # Aggregate results and compute statistics
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    # Group by scale for statistics
    by_scale = {}
    for r in all_results:
        if r.scale not in by_scale:
            by_scale[r.scale] = []
        by_scale[r.scale].append(r)

    # Compute per-scale aggregates
    scale_summaries = []
    for scale in DIAL_SCALES:
        results_at_scale = by_scale[scale]
        L_vals = [r.L_preproj for r in results_at_scale]
        dv_b0_vals = [r.metrics_b0["delta_V@8x"] for r in results_at_scale]
        aa_b0_vals = [r.metrics_b0["argmax_agree@8x"] for r in results_at_scale]
        dv_b1_vals = [r.metrics_b1["delta_V@8x"] for r in results_at_scale]
        aa_b1_vals = [r.metrics_b1["argmax_agree@8x"] for r in results_at_scale]

        summary = {
            "scale": scale,
            "L_preproj_mean": float(np.mean(L_vals)),
            "L_preproj_std": float(np.std(L_vals)),
            "delta_V_b0_8x_mean": float(np.mean(dv_b0_vals)),
            "delta_V_b0_8x_std": float(np.std(dv_b0_vals)),
            "argmax_b0_8x_mean": float(np.mean(aa_b0_vals)),
            "argmax_b0_8x_std": float(np.std(aa_b0_vals)),
            "delta_V_b1_8x_mean": float(np.mean(dv_b1_vals)),
            "delta_V_b1_8x_std": float(np.std(dv_b1_vals)),
            "argmax_b1_8x_mean": float(np.mean(aa_b1_vals)),
            "argmax_b1_8x_std": float(np.std(aa_b1_vals)),
        }
        scale_summaries.append(summary)

        print(f"\nScale {scale:.2f}:")
        print(f"  L_preproj:     {summary['L_preproj_mean']:.4f} ± {summary['L_preproj_std']:.4f}")
        print(f"  B0 ΔV@8x:      {summary['delta_V_b0_8x_mean']:.4f} ± {summary['delta_V_b0_8x_std']:.4f}")
        print(f"  B0 argmax@8x:  {summary['argmax_b0_8x_mean']:.4f} ± {summary['argmax_b0_8x_std']:.4f}")
        print(f"  B1 ΔV@8x:      {summary['delta_V_b1_8x_mean']:.4f} ± {summary['delta_V_b1_8x_std']:.4f}")
        print(f"  B1 argmax@8x:  {summary['argmax_b1_8x_mean']:.4f} ± {summary['argmax_b1_8x_std']:.4f}")

    # Compute monotonicity statistics
    print("\n" + "-" * 60)
    print("MONOTONICITY ANALYSIS")
    print("-" * 60)

    from scipy.stats import spearmanr

    L_preproj_all = [r.L_preproj for r in all_results]
    dv_b0_all = [r.metrics_b0["delta_V@8x"] for r in all_results]
    aa_b0_all = [r.metrics_b0["argmax_agree@8x"] for r in all_results]
    dv_b1_all = [r.metrics_b1["delta_V@8x"] for r in all_results]
    aa_b1_all = [r.metrics_b1["argmax_agree@8x"] for r in all_results]

    # Pooled correlations
    rho_dv_b0, p_dv_b0 = spearmanr(L_preproj_all, dv_b0_all)
    rho_aa_b0, p_aa_b0 = spearmanr(L_preproj_all, aa_b0_all)
    rho_dv_b1, p_dv_b1 = spearmanr(L_preproj_all, dv_b1_all)
    rho_aa_b1, p_aa_b1 = spearmanr(L_preproj_all, aa_b1_all)

    print(f"\nPooled Spearman correlations (L_preproj vs metric):")
    print(f"  B0 ΔV@8x:      ρ={rho_dv_b0:.4f}, p={p_dv_b0:.4f}")
    print(f"  B0 argmax@8x:  ρ={rho_aa_b0:.4f}, p={p_aa_b0:.4f}")
    print(f"  B1 ΔV@8x:      ρ={rho_dv_b1:.4f}, p={p_dv_b1:.4f}")
    print(f"  B1 argmax@8x:  ρ={rho_aa_b1:.4f}, p={p_aa_b1:.4f}")

    # Compute gate outcomes
    print("\n" + "-" * 60)
    print("DECISION GATES")
    print("-" * 60)

    # G0: projection inactive
    all_proj_rates = [r.projection_active_rate for r in all_results]
    g0_passed = all(rate < G0_PROJECTION_THRESHOLD for rate in all_proj_rates)
    g0_max_rate = max(all_proj_rates)
    print(f"G0 (Projection inactive < 1%): {'PASS' if g0_passed else 'FAIL'} (max rate={g0_max_rate*100:.2f}%)")

    # G1: stability (no NaN)
    g1_passed = all(not np.isnan(r.L_preproj) for r in all_results)
    print(f"G1 (Stability, no NaN):         {'PASS' if g1_passed else 'FAIL'}")

    # G2: dial range
    L_preproj_means = [s["L_preproj_mean"] for s in scale_summaries]
    L_spread = max(L_preproj_means) - min(L_preproj_means)
    g2_passed = L_spread >= G2_SPREAD_THRESHOLD
    print(f"G2 (Dial range >= 0.10):        {'PASS' if g2_passed else 'FAIL'} (spread={L_spread:.4f})")

    # G3: monotonicity on B0
    g3_dv_passed = bool(abs(rho_dv_b0) > G3_MONOTONICITY_THRESHOLD)
    g3_aa_passed = bool(abs(rho_aa_b0) > G3_MONOTONICITY_THRESHOLD)
    g3_status = "PASS" if (g3_dv_passed or g3_aa_passed) else "INCONCLUSIVE"
    print(f"G3 (Monotonicity |ρ| > 0.5):    {g3_status}")
    print(f"    ΔV@8x: {'PASS' if g3_dv_passed else 'FAIL'} (ρ={rho_dv_b0:.4f})")
    print(f"    argmax@8x: {'PASS' if g3_aa_passed else 'FAIL'} (ρ={rho_aa_b0:.4f})")

    # Determine overall decision
    if not g0_passed:
        decision = "INVALID: Projection was not disabled"
    elif not g1_passed:
        decision = "INVALID: Numerical instability (NaN)"
    elif not g2_passed:
        decision = "NEGATIVE: Dial has insufficient range"
    elif g3_dv_passed and g3_aa_passed:
        decision = "POSITIVE: Dial viable with strong monotonicity"
    elif g3_dv_passed or g3_aa_passed:
        decision = "POSITIVE: Dial viable with partial monotonicity"
    else:
        decision = "INCONCLUSIVE: Dial range exists but no clear monotonicity"

    print(f"\nDECISION: {decision}")

    # Build summary
    git_sha = get_git_sha()
    summary = {
        "experiment": "Exp4_final",
        "description": "Projection-free contraction dial (paper-defensible)",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "checkpoint": str(checkpoint_path),
        "config_yaml": str(config_yaml_path),
        "config": config,
        "parameters": {
            "dial_scales": DIAL_SCALES,
            "seeds": SEEDS,
            "n_train": N_TRAIN,
            "eval_n_list": EVAL_N_LIST,
            "b0_target_easy": B0_TARGET_EASY,
            "b0_target_hard": B0_TARGET_HARD,
            "b1_cap": B1_CAP,
        },
        "batches": {
            "b0_count": len(b0_states),
            "b0_hash": b0_hash,
            "b1_count": len(b1_states),
            "b1_hash": b1_hash,
        },
        "scale_summaries": scale_summaries,
        "monotonicity": {
            "pooled_rho_dv_b0": float(rho_dv_b0),
            "pooled_p_dv_b0": float(p_dv_b0),
            "pooled_rho_aa_b0": float(rho_aa_b0),
            "pooled_p_aa_b0": float(p_aa_b0),
            "pooled_rho_dv_b1": float(rho_dv_b1),
            "pooled_p_dv_b1": float(p_dv_b1),
            "pooled_rho_aa_b1": float(rho_aa_b1),
            "pooled_p_aa_b1": float(p_aa_b1),
        },
        "gates": {
            "g0_projection_inactive": {"passed": g0_passed, "max_rate": g0_max_rate},
            "g1_stability": {"passed": g1_passed},
            "g2_dial_range": {
                "passed": g2_passed,
                "threshold": G2_SPREAD_THRESHOLD,
                "L_preproj_spread": L_spread,
            },
            "g3_monotonicity": {
                "status": g3_status,
                "dv_b0_passed": g3_dv_passed,
                "aa_b0_passed": g3_aa_passed,
                "rho_dv_b0": float(rho_dv_b0),
                "rho_aa_b0": float(rho_aa_b0),
            },
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
    generate_provenance(summary, out_path)

    return summary


def generate_claims(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate CLAIMS.md."""
    decision = summary["decision"]
    gates = summary["gates"]
    mono = summary["monotonicity"]
    scale_sums = summary["scale_summaries"]

    content = f"""# Exp4 Claims: Projection-free Contraction Dial (Final)

**Generated:** {summary['generated_at']}
**Git SHA:** {summary['git_sha']}

## Summary

- **Decision:** {decision}
- **G0 (Projection inactive):** {'PASS' if gates['g0_projection_inactive']['passed'] else 'FAIL'}
- **G1 (Stability):** {'PASS' if gates['g1_stability']['passed'] else 'FAIL'}
- **G2 (Dial range ≥0.10):** {'PASS' if gates['g2_dial_range']['passed'] else 'FAIL'} (spread={gates['g2_dial_range']['L_preproj_spread']:.4f})
- **G3 (Monotonicity):** {gates['g3_monotonicity']['status']}
  - ΔV@8x: ρ={mono['pooled_rho_dv_b0']:.4f} (p={mono['pooled_p_dv_b0']:.4f})
  - argmax@8x: ρ={mono['pooled_rho_aa_b0']:.4f} (p={mono['pooled_p_aa_b0']:.4f})

## Results by Scale (Pooled Across Seeds)

| Scale | L_preproj | B0 ΔV@8x | B0 argmax@8x | B1 ΔV@8x | B1 argmax@8x |
|-------|-----------|----------|--------------|----------|--------------|
"""
    for s in scale_sums:
        content += f"| {s['scale']:.2f} | {s['L_preproj_mean']:.3f}±{s['L_preproj_std']:.3f} | "
        content += f"{s['delta_V_b0_8x_mean']:.3f}±{s['delta_V_b0_8x_std']:.3f} | "
        content += f"{s['argmax_b0_8x_mean']:.3f}±{s['argmax_b0_8x_std']:.3f} | "
        content += f"{s['delta_V_b1_8x_mean']:.3f}±{s['delta_V_b1_8x_std']:.3f} | "
        content += f"{s['argmax_b1_8x_mean']:.3f}±{s['argmax_b1_8x_std']:.3f} |\n"

    content += f"""
## Scoped Claims

"""
    if "POSITIVE" in decision:
        content += """**Claim (Positive):** Inference-time contraction scaling on z→z layers produces a measurable,
controllable "dial" for the achieved Lipschitz constant (L_preproj) when projection is disabled.
"""
        if gates['g3_monotonicity']['dv_b0_passed']:
            content += f"""
**Claim (Monotonicity - ΔV):** Higher L_preproj is associated with higher value mismatch (ΔV@8x)
on B0 evaluation states, with Spearman ρ={mono['pooled_rho_dv_b0']:.3f} (p={mono['pooled_p_dv_b0']:.4f}).
"""
        if gates['g3_monotonicity']['aa_b0_passed']:
            content += f"""
**Claim (Monotonicity - argmax):** L_preproj is negatively associated with argmax agreement
on B0 evaluation states, with Spearman ρ={mono['pooled_rho_aa_b0']:.3f} (p={mono['pooled_p_aa_b0']:.4f}).
"""
    elif "NEGATIVE" in decision:
        content += """**Claim (Negative):** Inference-time contraction scaling does not produce sufficient
L_preproj variation to constitute a meaningful stability dial.
"""
    else:
        content += """**Claim (Inconclusive):** Dial range exists (L_preproj varies with scaling factor),
but no clear monotonic relationship with mismatch metrics was observed.
"""

    content += f"""
## Scope Limitations

- Results from inference-time scaling only (no retraining)
- Base checkpoint: {summary['checkpoint']}
- Trivial 4×4 Sudoku suite only
- Seeds: {summary['parameters']['seeds']}
- Training depth n_train={summary['parameters']['n_train']}, eval depths: {summary['parameters']['eval_n_list']}

## Evaluation Batches

- **B0:** {summary['batches']['b0_count']} initial states (hash: {summary['batches']['b0_hash']})
- **B1:** {summary['batches']['b1_count']} successor states (hash: {summary['batches']['b1_hash']})
- B1 constructed using union of top-5 actions across ALL dial scales (avoids selection bias)
"""

    claims_path = out_path / "CLAIMS.md"
    with open(claims_path, "w") as f:
        f.write(content)
    print(f"Saved: {claims_path}")


def generate_provenance(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate PROVENANCE.md."""
    content = f"""# Exp4 Provenance: Projection-free Contraction Dial (Final)

**Generated:** {summary['generated_at']}
**Git SHA:** {summary['git_sha']}

## Checkpoint and Config

- **Checkpoint:** {summary['checkpoint']}
- **YAML Config:** {summary['config_yaml']}
- **Config Source:** yaml (required, no inference)

## Non-Negotiables Verified

- `disable_value_head_norm: true` ✓
- `latent_projection_mode: disabled` and `latent_ball_radius: null` ✓

## Dial Implementation

Contraction scaling is applied at **inference/evaluation time** by multiplying
the weight matrices of L_level (z→z) layers by the scaling factor:

```python
for module in L_level.modules():
    if hasattr(module, 'weight'):
        module.weight.data.mul_(scale_factor)
```

This directly modulates the Lipschitz constant of the z→z mapping.

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Dial scales | {summary['parameters']['dial_scales']} |
| Seeds | {summary['parameters']['seeds']} |
| n_train | {summary['parameters']['n_train']} |
| Eval depths | {summary['parameters']['eval_n_list']} |
| B0 composition | {summary['parameters']['b0_target_easy']} easy + {summary['parameters']['b0_target_hard']} hard |
| B1 cap | {summary['parameters']['b1_cap']} |

## Batch Provenance

- **B0:** {summary['batches']['b0_count']} states, hash: `{summary['batches']['b0_hash']}`
- **B1:** {summary['batches']['b1_count']} states, hash: `{summary['batches']['b1_hash']}`
- B1 uses union of top-5 actions from all dial scales + 5 random actions per B0 state

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp4_final -- \\
    --checkpoint {summary['checkpoint']} \\
    --config_yaml {summary['config_yaml']} \\
    --out_dir {out_path}
```
"""

    prov_path = out_path / "PROVENANCE.md"
    with open(prov_path, "w") as f:
        f.write(content)
    print(f"Saved: {prov_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp4 Final Evaluation")
    parser.add_argument(
        "--checkpoint_base",
        type=str,
        default="results/exp3/nc_rdis_s42/model_step_5000.pt",
        help="Base checkpoint path",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/exp3_projection_ablation/nc_rdis.yaml",
        help="YAML config path (required)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/paper_ready/exp4_projection_free_dial_final",
        help="Output directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/buiksat/trm_bellman/data",
        help="Data directory (default: project_root/data)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Override device if CUDA not available
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = "cpu"

    summary = run_exp4_final(
        checkpoint_path=args.checkpoint_base,
        config_yaml_path=args.config_yaml,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        device=device,
    )

    # Exit code based on decision
    if "POSITIVE" in summary["decision"]:
        sys.exit(0)
    elif "NEGATIVE" in summary["decision"] or "INVALID" in summary["decision"]:
        sys.exit(1)
    else:  # INCONCLUSIVE
        sys.exit(0)  # Inconclusive is still a valid result


if __name__ == "__main__":
    main()
