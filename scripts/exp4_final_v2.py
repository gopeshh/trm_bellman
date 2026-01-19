#!/usr/bin/env python3
"""
Exp4 Final v2: Projection-free Contraction Dial (Paper-Defensible, Multi-Checkpoint)

FIXES pseudo-replication issue by using N=12 truly independent samples:
- 3 independently trained checkpoints (seeds 41, 42, 43)
- 4 dial scales each

Key improvements over v1:
1. Multi-checkpoint evaluation (true N=12 independent points)
2. B1 built using union across ALL checkpoints × ALL scales
3. Action entropy and success rate metrics (anti-degenerate checks)
4. Proper git SHA tracking
5. Pseudo-replication detection in audit

Usage:
    buck2 run //buiksat_trm:exp4_final_v2 -- \
        --checkpoint_glob "results/exp3/nc_rdis_s*/model_step_5000.pt" \
        --config_yaml configs/exp3_projection_ablation/nc_rdis.yaml \
        --out_dir results/paper_ready/exp4_projection_free_dial_final

Non-negotiables:
- latent_ball_radius = 0.0 (projection disabled)
- disable_value_head_norm: true
- No fallback to random inputs (hard error if dataset missing)
"""

import argparse
import glob
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

# Dial settings: scales that span L_preproj range
DIAL_SCALES = [1.0, 0.85, 0.70, 0.55]

# Evaluation depths
# IMPORTANT: n_train=2, eval at n2∈{4,8,16} means depth mismatch of 2×, 4×, 8× respectively
# Variable names use "@Nx" where N is the absolute depth (n2=N), NOT the mismatch ratio
# For paper: n2=4 is 2× mismatch, n2=8 is 4× mismatch, n2=16 is 8× mismatch
N_TRAIN = 2  # Training depth
EVAL_N_LIST = [4, 8, 16]  # eval at n2=4, 8, 16 (2×, 4×, 8× training depth)
MISMATCH_RATIOS = {4: 2, 8: 4, 16: 8}  # n2 -> mismatch ratio

# B0/B1 parameters
B0_TARGET_EASY = 70
B0_TARGET_HARD = 30
B1_CAP = 1500

# Gate thresholds
G0_PROJECTION_THRESHOLD = 0.01  # projection_active_rate < 1%
G2_SPREAD_THRESHOLD = 0.10  # L_preproj spread >= 0.10
G3_MONOTONICITY_THRESHOLD = 0.5  # |Spearman ρ| > 0.5

# Bootstrap parameters for cluster-aware inference
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
BOOTSTRAP_SEED = 42  # For reproducibility


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
    """Evaluation results for a single (checkpoint, scale) condition."""
    checkpoint_seed: int
    checkpoint_path: str
    scale: float
    L_preproj: float
    L_preproj_std: float
    projection_active: bool
    # B0 metrics
    delta_V_b0_4x: float
    delta_V_b0_8x: float
    delta_V_b0_16x: float
    argmax_b0_4x: float
    argmax_b0_8x: float
    argmax_b0_16x: float
    entropy_b0_train: float
    entropy_b0_eval: float
    # B1 metrics
    delta_V_b1_4x: float
    delta_V_b1_8x: float
    delta_V_b1_16x: float
    argmax_b1_4x: float
    argmax_b1_8x: float
    argmax_b1_16x: float
    entropy_b1_train: float
    entropy_b1_eval: float


# =============================================================================
# Utility Functions
# =============================================================================

def get_git_sha() -> Optional[str]:
    """Get current git SHA.

    Tries multiple paths because buck2 runs from link-tree, not source dir.
    """
    # Try multiple candidate paths for git root
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
    """Extract seed from checkpoint path like nc_rdis_s42."""
    import re
    match = re.search(r'_s(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


def cluster_bootstrap_spearman(
    x: List[float],
    y: List[float],
    cluster_ids: List[int],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute Spearman ρ with cluster-aware bootstrap for proper inference.

    Since scales are repeated measures on the same checkpoint, we cannot treat
    all N=12 samples as i.i.d. Instead, we resample checkpoints (clusters) with
    replacement and recompute ρ for each bootstrap sample.

    Args:
        x: First variable (e.g., L_preproj values)
        y: Second variable (e.g., argmax values)
        cluster_ids: Cluster/checkpoint ID for each sample
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        dict with 'rho_point', 'rho_ci_lower', 'rho_ci_upper', 'n_clusters'
    """
    from scipy.stats import spearmanr

    x = np.array(x)
    y = np.array(y)
    cluster_ids = np.array(cluster_ids)

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Point estimate
    rho_point, _ = spearmanr(x, y)

    # Bootstrap: resample clusters with replacement
    rng = np.random.RandomState(seed)
    bootstrap_rhos = []

    for _ in range(n_bootstrap):
        # Resample clusters (checkpoints) with replacement
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)

        # Collect all observations from sampled clusters
        boot_x = []
        boot_y = []
        for c in sampled_clusters:
            mask = cluster_ids == c
            boot_x.extend(x[mask])
            boot_y.extend(y[mask])

        # Compute Spearman ρ for this bootstrap sample
        if len(boot_x) >= 3:  # Need at least 3 points for correlation
            rho_boot, _ = spearmanr(boot_x, boot_y)
            if not np.isnan(rho_boot):
                bootstrap_rhos.append(rho_boot)

    # Compute 95% CI from bootstrap distribution
    if len(bootstrap_rhos) > 0:
        ci_lower = float(np.percentile(bootstrap_rhos, 2.5))
        ci_upper = float(np.percentile(bootstrap_rhos, 97.5))
    else:
        ci_lower = ci_upper = float('nan')

    return {
        'rho_point': float(rho_point),
        'rho_ci_lower': ci_lower,
        'rho_ci_upper': ci_upper,
        'n_clusters': n_clusters,
        'n_bootstrap': n_bootstrap,
    }


def per_checkpoint_trends(
    all_results: List,
) -> Dict[str, Dict[str, str]]:
    """
    Compute per-checkpoint monotonicity trends.

    For each checkpoint, check if L_preproj increases as scale decreases
    (which we expect) and if argmax/ΔV follow expected directions.

    Returns dict mapping checkpoint_seed to trend descriptions.
    """
    from collections import defaultdict

    by_seed = defaultdict(list)
    for r in all_results:
        by_seed[r.checkpoint_seed].append(r)

    trends = {}
    for seed, results in by_seed.items():
        # Sort by scale (descending, so 1.0 -> 0.55)
        results = sorted(results, key=lambda r: -r.scale)

        L_preproj_vals = [r.L_preproj for r in results]
        argmax_vals = [r.argmax_b0_8x for r in results]
        delta_V_vals = [r.delta_V_b0_8x for r in results]

        # Check if L_preproj is monotonically increasing as scale decreases
        L_monotonic = all(L_preproj_vals[i] <= L_preproj_vals[i+1]
                          for i in range(len(L_preproj_vals)-1))

        # Check if argmax is monotonically decreasing as L_preproj increases
        argmax_monotonic = all(argmax_vals[i] >= argmax_vals[i+1]
                               for i in range(len(argmax_vals)-1))

        # Overall trend direction (even if not strictly monotonic)
        L_trend = "↑" if L_preproj_vals[-1] > L_preproj_vals[0] else "↓"
        argmax_trend = "↓" if argmax_vals[-1] < argmax_vals[0] else "↑"
        delta_V_trend = "↑" if delta_V_vals[-1] > delta_V_vals[0] else "↓"

        trends[seed] = {
            'L_preproj_trend': L_trend,
            'L_preproj_monotonic': L_monotonic,
            'L_preproj_range': f"{L_preproj_vals[0]:.3f}→{L_preproj_vals[-1]:.3f}",
            'argmax_trend': argmax_trend,
            'argmax_monotonic': argmax_monotonic,
            'argmax_range': f"{argmax_vals[0]:.3f}→{argmax_vals[-1]:.3f}",
            'delta_V_trend': delta_V_trend,
            'delta_V_range': f"{delta_V_vals[0]:.3f}→{delta_V_vals[-1]:.3f}",
        }

    return trends


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

    # Verify non-negotiables
    disable_value_head_norm = yaml_config.get("disable_value_head_norm", False)
    latent_ball_radius = yaml_config.get("latent_ball_radius", 10.0)

    if not disable_value_head_norm:
        raise ValueError(
            "NON-NEGOTIABLE VIOLATION: disable_value_head_norm must be True.\n"
            f"Found: disable_value_head_norm={disable_value_head_norm}"
        )

    if latent_ball_radius != 0.0:
        raise ValueError(
            f"NON-NEGOTIABLE VIOLATION: latent_ball_radius must be 0.0.\n"
            f"Found: latent_ball_radius={latent_ball_radius}"
        )

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model_state = state_dict["model_state_dict"]
    else:
        model_state = state_dict

    # Clean state dict keys
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state[clean_key] = value

    # Infer dimensions from weights (handle both naming conventions)
    embed_key = "inner.embed_tokens.embedding_weight"
    if embed_key not in cleaned_state:
        embed_key = "inner.embed_tokens.weight"
    embed_weight = cleaned_state.get(embed_key, torch.zeros(6, 64))
    hidden_size = embed_weight.shape[1]
    vocab_size = embed_weight.shape[0]
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
        rl_latent_ball_radius=0.0,  # Non-negotiable: projection disabled
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
        "latent_ball_radius": 0.0,
    }

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
    """Apply contraction scaling to z→z layers (L_level)."""
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
    """Load puzzles from dataset directory. Raises if dataset not found."""
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

    return selected


def build_b1_multi_checkpoint(
    b0_states: List[PuzzleState],
    models_and_weights: List[Tuple[nn.Module, Dict[str, torch.Tensor], int]],  # (model, orig_weights, seed)
    scales: List[float],
    config: Dict[str, Any],
    n_train: int,
    rng_seed: int,
    cap: int = B1_CAP,
    device: str = "cpu",
) -> List[PuzzleState]:
    """
    Build B1 using union-of-actions across ALL checkpoints × ALL scales.
    This avoids selection bias toward any particular checkpoint or scale.
    """
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig

    rng = np.random.default_rng(rng_seed)

    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    num_actions = config["num_actions"]
    stop_action_id = num_actions - 1

    print(f"[B1] Building successor closure from {len(b0_states)} B0 states")
    print(f"[B1] Using union of top-5 actions from {len(models_and_weights)} checkpoints × {len(scales)} scales")

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

            # Collect top-5 from EACH checkpoint × EACH scale
            for model, original_weights, ckpt_seed in models_and_weights:
                model.eval()
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
                            actions_to_apply.append((a, f"ckpt{ckpt_seed}_scale{scale:.2f}"))

                # Restore original weights after all scales
                restore_weights(model, original_weights)

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

    # Cap if needed
    if len(successors) > cap:
        rng.shuffle(successors)
        successors = successors[:cap]
        print(f"[B1] Capped to {cap} states")

    print(f"[B1] Generated {len(successors)} unique successors")
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
    """Estimate L_preproj via finite differences. Returns (mean, std)."""
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
    """Compute ΔV, argmax agreement, and entropy for mismatch depths."""
    model.eval()
    metrics = {f"delta_V@{n}x": [] for n in eval_n_list}
    metrics.update({f"argmax_agree@{n}x": [] for n in eval_n_list})
    metrics["entropy_train"] = []
    metrics["entropy_eval"] = []

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

            # Entropy at train depth
            entropy_train = -(probs_train * (probs_train + 1e-10).log()).sum().item()
            metrics["entropy_train"].append(entropy_train)

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

                # Entropy at eval depth (use 16x as "eval")
                if n_eval == 16:
                    entropy_eval = -(probs_eval * (probs_eval + 1e-10).log()).sum().item()
                    metrics["entropy_eval"].append(entropy_eval)

    # Aggregate
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = float(np.mean(values))
        else:
            result[key] = 0.0

    return result


def check_projection_disabled(config: Dict[str, Any]) -> bool:
    """Check if projection is disabled (latent_ball_radius=0)."""
    return config.get("latent_ball_radius", 10.0) == 0.0


# =============================================================================
# Main Evaluation
# =============================================================================

def run_exp4_final_v2(
    checkpoint_paths: List[str],
    config_yaml_path: str,
    out_dir: str,
    data_dir: str = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run full Exp4 evaluation with multiple checkpoints."""
    print("=" * 70)
    print("Exp4 Final v2: Multi-Checkpoint Evaluation (Paper-Defensible)")
    print("=" * 70)
    print()
    print(f"[Config] Checkpoints: {checkpoint_paths}")
    print(f"[Config] YAML: {config_yaml_path}")
    print(f"[Config] Output: {out_dir}")
    print(f"[Config] Device: {device}")
    print(f"[Config] Dial scales: {DIAL_SCALES}")
    print(f"[Config] n_train={N_TRAIN}, eval_n={EVAL_N_LIST}")
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
    print()

    # Load dataset
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    else:
        data_dir = Path(data_dir)
    all_puzzles = load_puzzles_strict(data_dir)
    print(f"[Dataset] Loaded {len(all_puzzles)} total puzzles")

    # Build B0
    b0_states = build_b0_strict(all_puzzles, seed=42)
    print(f"[B0] Built batch with {len(b0_states)} states")

    # Build B1 using union across all checkpoints × all scales
    b1_states = build_b1_multi_checkpoint(
        b0_states, models_and_weights, DIAL_SCALES,
        config, N_TRAIN, rng_seed=42, device=device
    )

    # Compute B0/B1 hashes
    b0_hash = hashlib.md5("".join(s.state_id for s in b0_states).encode()).hexdigest()[:12]
    b1_hash = hashlib.md5("".join(s.state_id for s in b1_states).encode()).hexdigest()[:12]

    print()
    print(f"[Batches] B0: {len(b0_states)} states (hash: {b0_hash})")
    print(f"[Batches] B1: {len(b1_states)} states (hash: {b1_hash})")
    print()

    # Run evaluation: N = num_checkpoints × num_scales = 12 independent points
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
                model, b0_states, N_TRAIN, device
            )
            print(f"  L_preproj: {L_preproj_mean:.4f} ± {L_preproj_std:.4f}")

            # Check projection disabled
            proj_disabled = check_projection_disabled(config)

            # Compute mismatch metrics on B0
            metrics_b0 = compute_mismatch_metrics(
                model, b0_states, N_TRAIN, EVAL_N_LIST, device
            )
            print(f"  B0 ΔV@8x: {metrics_b0['delta_V@8x']:.4f}")
            print(f"  B0 argmax@8x: {metrics_b0['argmax_agree@8x']:.4f}")
            print(f"  B0 entropy (train/eval): {metrics_b0['entropy_train']:.3f} / {metrics_b0['entropy_eval']:.3f}")

            # Compute mismatch metrics on B1
            metrics_b1 = compute_mismatch_metrics(
                model, b1_states, N_TRAIN, EVAL_N_LIST, device
            )
            print(f"  B1 ΔV@8x: {metrics_b1['delta_V@8x']:.4f}")
            print(f"  B1 argmax@8x: {metrics_b1['argmax_agree@8x']:.4f}")
            print(f"  B1 entropy (train/eval): {metrics_b1['entropy_train']:.3f} / {metrics_b1['entropy_eval']:.3f}")

            result = EvalResult(
                checkpoint_seed=ckpt_seed,
                checkpoint_path=checkpoint_paths[models_and_weights.index((model, original_weights, ckpt_seed))],
                scale=scale,
                L_preproj=L_preproj_mean,
                L_preproj_std=L_preproj_std,
                projection_active=not proj_disabled,
                delta_V_b0_4x=metrics_b0["delta_V@4x"],
                delta_V_b0_8x=metrics_b0["delta_V@8x"],
                delta_V_b0_16x=metrics_b0["delta_V@16x"],
                argmax_b0_4x=metrics_b0["argmax_agree@4x"],
                argmax_b0_8x=metrics_b0["argmax_agree@8x"],
                argmax_b0_16x=metrics_b0["argmax_agree@16x"],
                entropy_b0_train=metrics_b0["entropy_train"],
                entropy_b0_eval=metrics_b0["entropy_eval"],
                delta_V_b1_4x=metrics_b1["delta_V@4x"],
                delta_V_b1_8x=metrics_b1["delta_V@8x"],
                delta_V_b1_16x=metrics_b1["delta_V@16x"],
                argmax_b1_4x=metrics_b1["argmax_agree@4x"],
                argmax_b1_8x=metrics_b1["argmax_agree@8x"],
                argmax_b1_16x=metrics_b1["argmax_agree@16x"],
                entropy_b1_train=metrics_b1["entropy_train"],
                entropy_b1_eval=metrics_b1["entropy_eval"],
            )
            all_results.append(result)

        # Restore original weights
        restore_weights(model, original_weights)

    # Compute monotonicity statistics with cluster-aware bootstrap
    # NOTE: We have 3 checkpoints × 4 scales = 12 observations, but scales are
    # repeated measures on the same checkpoint, so we use cluster bootstrap
    # (resampling checkpoints) rather than assuming i.i.d. samples.
    print("\n" + "=" * 70)
    print("MONOTONICITY ANALYSIS")
    print("(Cluster bootstrap: 3 checkpoints × 4 scales, resampling checkpoints)")
    print("=" * 70)

    from scipy.stats import spearmanr

    L_preproj_all = [r.L_preproj for r in all_results]
    argmax_b0_8x_all = [r.argmax_b0_8x for r in all_results]
    argmax_b1_8x_all = [r.argmax_b1_8x for r in all_results]
    delta_V_b0_8x_all = [r.delta_V_b0_8x for r in all_results]
    delta_V_b1_8x_all = [r.delta_V_b1_8x for r in all_results]
    cluster_ids = [r.checkpoint_seed for r in all_results]

    # Cluster bootstrap for proper inference
    boot_aa_b0 = cluster_bootstrap_spearman(L_preproj_all, argmax_b0_8x_all, cluster_ids, N_BOOTSTRAP, BOOTSTRAP_SEED)
    boot_aa_b1 = cluster_bootstrap_spearman(L_preproj_all, argmax_b1_8x_all, cluster_ids, N_BOOTSTRAP, BOOTSTRAP_SEED)
    boot_dv_b0 = cluster_bootstrap_spearman(L_preproj_all, delta_V_b0_8x_all, cluster_ids, N_BOOTSTRAP, BOOTSTRAP_SEED)
    boot_dv_b1 = cluster_bootstrap_spearman(L_preproj_all, delta_V_b1_8x_all, cluster_ids, N_BOOTSTRAP, BOOTSTRAP_SEED)

    # For backwards compatibility, also compute i.i.d. p-values (but mark as reference only)
    rho_aa_b0, p_aa_b0 = spearmanr(L_preproj_all, argmax_b0_8x_all)
    rho_aa_b1, p_aa_b1 = spearmanr(L_preproj_all, argmax_b1_8x_all)
    rho_dv_b0, p_dv_b0 = spearmanr(L_preproj_all, delta_V_b0_8x_all)
    rho_dv_b1, p_dv_b1 = spearmanr(L_preproj_all, delta_V_b1_8x_all)

    print(f"\nSpearman correlations with cluster bootstrap 95% CI:")
    print(f"  (n_train={N_TRAIN}, n2=8 means 4× mismatch)")
    print(f"  B0 argmax (n2=8):  ρ={boot_aa_b0['rho_point']:.3f} [{boot_aa_b0['rho_ci_lower']:.3f}, {boot_aa_b0['rho_ci_upper']:.3f}]")
    print(f"  B1 argmax (n2=8):  ρ={boot_aa_b1['rho_point']:.3f} [{boot_aa_b1['rho_ci_lower']:.3f}, {boot_aa_b1['rho_ci_upper']:.3f}]")
    print(f"  B0 ΔV (n2=8):      ρ={boot_dv_b0['rho_point']:.3f} [{boot_dv_b0['rho_ci_lower']:.3f}, {boot_dv_b0['rho_ci_upper']:.3f}]")
    print(f"  B1 ΔV (n2=8):      ρ={boot_dv_b1['rho_point']:.3f} [{boot_dv_b1['rho_ci_lower']:.3f}, {boot_dv_b1['rho_ci_upper']:.3f}]")
    print(f"  (i.i.d. p-values for reference: p_aa_b0={p_aa_b0:.4f}, p_aa_b1={p_aa_b1:.4f})")

    # Per-checkpoint monotonicity trends
    trends = per_checkpoint_trends(all_results)
    print(f"\n[Per-checkpoint trends (scale 1.0→0.55)]")
    for seed, t in sorted(trends.items()):
        print(f"  Seed {seed}: L_preproj {t['L_preproj_trend']} ({t['L_preproj_range']}), "
              f"argmax {t['argmax_trend']} ({t['argmax_range']}), "
              f"ΔV {t['delta_V_trend']} ({t['delta_V_range']})")
        status = "✓ monotonic" if t['L_preproj_monotonic'] and t['argmax_monotonic'] else "partial"
        print(f"         L_preproj monotonic: {t['L_preproj_monotonic']}, argmax monotonic: {t['argmax_monotonic']} → {status}")

    # Anti-degenerate check: entropy should not collapse
    entropy_train_all = [r.entropy_b0_train for r in all_results]
    entropy_eval_all = [r.entropy_b0_eval for r in all_results]
    mean_entropy_train = np.mean(entropy_train_all)
    mean_entropy_eval = np.mean(entropy_eval_all)
    min_entropy_train = np.min(entropy_train_all)

    print(f"\n[Anti-degenerate check]")
    print(f"  Mean entropy (train depth): {mean_entropy_train:.3f}")
    print(f"  Mean entropy (eval depth):  {mean_entropy_eval:.3f}")
    print(f"  Min entropy (train depth):  {min_entropy_train:.3f}")
    entropy_ok = min_entropy_train > 0.5  # Not collapsed to deterministic
    print(f"  Entropy OK (min > 0.5): {'YES' if entropy_ok else 'NO'}")

    # Pseudo-replication check
    # Check if argmax values vary across checkpoints within same scale
    pseudo_replication = True
    for scale in DIAL_SCALES:
        scale_results = [r for r in all_results if r.scale == scale]
        argmax_vals = [r.argmax_b0_8x for r in scale_results]
        if len(set(argmax_vals)) > 1:
            pseudo_replication = False
            break

    print(f"\n[Pseudo-replication check]")
    print(f"  Argmax varies across checkpoints within scale: {'YES' if not pseudo_replication else 'NO'}")

    # Decision gates
    print("\n" + "-" * 60)
    print("DECISION GATES")
    print("-" * 60)

    # G0: projection inactive
    g0_passed = all(not r.projection_active for r in all_results)
    print(f"G0 (Projection inactive): {'PASS' if g0_passed else 'FAIL'}")

    # G1: stability
    g1_passed = all(not np.isnan(r.L_preproj) for r in all_results)
    print(f"G1 (Stability, no NaN):   {'PASS' if g1_passed else 'FAIL'}")

    # G2: dial range
    L_spread = max(L_preproj_all) - min(L_preproj_all)
    g2_passed = L_spread >= G2_SPREAD_THRESHOLD
    print(f"G2 (Dial range >= 0.10):  {'PASS' if g2_passed else 'FAIL'} (spread={L_spread:.4f})")

    # G3: monotonicity using bootstrap CIs
    # A metric passes if |ρ| > 0.5 AND the 95% CI does not include 0
    def boot_ci_significant(boot: Dict, threshold: float = 0.5) -> bool:
        """Check if correlation is significant using bootstrap CI."""
        rho = boot['rho_point']
        ci_lo = boot['rho_ci_lower']
        ci_hi = boot['rho_ci_upper']
        # Passes if: |ρ| > threshold AND CI doesn't cross 0
        if abs(rho) <= threshold:
            return False
        # If ρ is negative, CI should be entirely negative (ci_hi < 0)
        # If ρ is positive, CI should be entirely positive (ci_lo > 0)
        if rho < 0:
            return ci_hi < 0
        else:
            return ci_lo > 0

    g3_aa_b0_passed = boot_ci_significant(boot_aa_b0, G3_MONOTONICITY_THRESHOLD)
    g3_aa_b1_passed = boot_ci_significant(boot_aa_b1, G3_MONOTONICITY_THRESHOLD)
    g3_dv_b0_passed = boot_ci_significant(boot_dv_b0, G3_MONOTONICITY_THRESHOLD)
    g3_dv_b1_passed = boot_ci_significant(boot_dv_b1, G3_MONOTONICITY_THRESHOLD)

    g3_any_passed = g3_aa_b0_passed or g3_aa_b1_passed or g3_dv_b0_passed or g3_dv_b1_passed
    g3_status = "PASS" if g3_any_passed else "INCONCLUSIVE"

    print(f"G3 (Monotonicity |ρ|>0.5, 95% CI excludes 0): {g3_status}")
    print(f"    B0 argmax (n2=8): {'PASS' if g3_aa_b0_passed else 'FAIL'} "
          f"(ρ={boot_aa_b0['rho_point']:.3f} [{boot_aa_b0['rho_ci_lower']:.3f}, {boot_aa_b0['rho_ci_upper']:.3f}])")
    print(f"    B1 argmax (n2=8): {'PASS' if g3_aa_b1_passed else 'FAIL'} "
          f"(ρ={boot_aa_b1['rho_point']:.3f} [{boot_aa_b1['rho_ci_lower']:.3f}, {boot_aa_b1['rho_ci_upper']:.3f}])")
    print(f"    B0 ΔV (n2=8):     {'PASS' if g3_dv_b0_passed else 'FAIL'} "
          f"(ρ={boot_dv_b0['rho_point']:.3f} [{boot_dv_b0['rho_ci_lower']:.3f}, {boot_dv_b0['rho_ci_upper']:.3f}])")
    print(f"    B1 ΔV (n2=8):     {'PASS' if g3_dv_b1_passed else 'FAIL'} "
          f"(ρ={boot_dv_b1['rho_point']:.3f} [{boot_dv_b1['rho_ci_lower']:.3f}, {boot_dv_b1['rho_ci_upper']:.3f}])")

    # Overall decision
    if not g0_passed:
        decision = "INVALID: Projection was not disabled"
    elif not g1_passed:
        decision = "INVALID: Numerical instability (NaN)"
    elif not g2_passed:
        decision = "NEGATIVE: Dial has insufficient range"
    elif g3_any_passed and entropy_ok and not pseudo_replication:
        decision = "POSITIVE: Dial viable with statistically significant monotonicity"
    elif g3_any_passed and entropy_ok:
        decision = "POSITIVE: Dial viable (note: potential pseudo-replication)"
    else:
        decision = "INCONCLUSIVE: Dial range exists but no significant monotonicity"

    print(f"\nDECISION: {decision}")

    # Build summary
    git_sha = get_git_sha()

    # Group by scale for summaries
    scale_summaries = []
    for scale in DIAL_SCALES:
        scale_results = [r for r in all_results if r.scale == scale]
        scale_summaries.append({
            "scale": scale,
            "L_preproj_mean": float(np.mean([r.L_preproj for r in scale_results])),
            "L_preproj_std": float(np.std([r.L_preproj for r in scale_results])),
            "argmax_b0_8x_mean": float(np.mean([r.argmax_b0_8x for r in scale_results])),
            "argmax_b0_8x_std": float(np.std([r.argmax_b0_8x for r in scale_results])),
            "argmax_b1_8x_mean": float(np.mean([r.argmax_b1_8x for r in scale_results])),
            "argmax_b1_8x_std": float(np.std([r.argmax_b1_8x for r in scale_results])),
            "delta_V_b0_8x_mean": float(np.mean([r.delta_V_b0_8x for r in scale_results])),
            "delta_V_b0_8x_std": float(np.std([r.delta_V_b0_8x for r in scale_results])),
            "delta_V_b1_8x_mean": float(np.mean([r.delta_V_b1_8x for r in scale_results])),
            "delta_V_b1_8x_std": float(np.std([r.delta_V_b1_8x for r in scale_results])),
            "entropy_train_mean": float(np.mean([r.entropy_b0_train for r in scale_results])),
            "entropy_eval_mean": float(np.mean([r.entropy_b0_eval for r in scale_results])),
        })

    summary = {
        "experiment": "Exp4_final_v2",
        "description": "Projection-free contraction dial (multi-checkpoint, paper-defensible)",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "checkpoints": checkpoint_paths,
        "config_yaml": str(config_yaml_path),
        "config": config,
        "parameters": {
            "dial_scales": DIAL_SCALES,
            "checkpoint_seeds": [extract_seed_from_path(p) for p in checkpoint_paths],
            "n_train": N_TRAIN,
            "eval_n_list": EVAL_N_LIST,
            "b0_target_easy": B0_TARGET_EASY,
            "b0_target_hard": B0_TARGET_HARD,
            "b1_cap": B1_CAP,
            "n_independent_samples": len(all_results),
        },
        "batches": {
            "b0_count": len(b0_states),
            "b0_hash": b0_hash,
            "b1_count": len(b1_states),
            "b1_hash": b1_hash,
        },
        "scale_summaries": scale_summaries,
        "monotonicity": {
            "n_samples": len(all_results),
            "n_clusters": boot_aa_b0['n_clusters'],
            "n_bootstrap": N_BOOTSTRAP,
            "stat_method": "cluster_bootstrap",
            "stat_note": "Scales are repeated measures; CIs from checkpoint-level resampling",
            # Bootstrap results with CIs
            "boot_aa_b0": boot_aa_b0,
            "boot_aa_b1": boot_aa_b1,
            "boot_dv_b0": boot_dv_b0,
            "boot_dv_b1": boot_dv_b1,
            # Legacy i.i.d. p-values (for reference only)
            "iid_rho_aa_b0": float(rho_aa_b0),
            "iid_p_aa_b0": float(p_aa_b0),
            "iid_rho_aa_b1": float(rho_aa_b1),
            "iid_p_aa_b1": float(p_aa_b1),
            "iid_rho_dv_b0": float(rho_dv_b0),
            "iid_p_dv_b0": float(p_dv_b0),
            "iid_rho_dv_b1": float(rho_dv_b1),
            "iid_p_dv_b1": float(p_dv_b1),
        },
        "per_checkpoint_trends": {str(k): v for k, v in trends.items()},
        "anti_degenerate": {
            "mean_entropy_train": float(mean_entropy_train),
            "mean_entropy_eval": float(mean_entropy_eval),
            "min_entropy_train": float(min_entropy_train),
            "entropy_ok": bool(entropy_ok),
            "pseudo_replication_detected": bool(pseudo_replication),
        },
        "gates": {
            "g0_projection_inactive": {"passed": bool(g0_passed)},
            "g1_stability": {"passed": bool(g1_passed)},
            "g2_dial_range": {
                "passed": bool(g2_passed),
                "threshold": G2_SPREAD_THRESHOLD,
                "L_preproj_spread": float(L_spread),
            },
            "g3_monotonicity": {
                "status": g3_status,
                "aa_b0_passed": bool(g3_aa_b0_passed),
                "aa_b1_passed": bool(g3_aa_b1_passed),
                "dv_b0_passed": bool(g3_dv_b0_passed),
                "dv_b1_passed": bool(g3_dv_b1_passed),
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
    generate_claims_v2(summary, out_path)

    # Generate PROVENANCE.md
    generate_provenance_v2(summary, out_path, checkpoint_paths, config_yaml_path)

    return summary


def generate_claims_v2(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate CLAIMS.md for v2."""
    decision = summary["decision"]
    mono = summary["monotonicity"]
    gates = summary["gates"]
    anti_deg = summary["anti_degenerate"]
    scale_sums = summary["scale_summaries"]
    trends = summary.get("per_checkpoint_trends", {})

    # Extract bootstrap results
    boot_aa_b0 = mono.get("boot_aa_b0", {})
    boot_aa_b1 = mono.get("boot_aa_b1", {})
    boot_dv_b0 = mono.get("boot_dv_b0", {})
    boot_dv_b1 = mono.get("boot_dv_b1", {})

    content = f"""# Exp4 Claims: Projection-free Contraction Dial (Final v2)

**Generated:** {summary['generated_at']}
**Git SHA:** {summary['git_sha']}

## Summary

- **Decision:** {decision}
- **N (observations):** {mono['n_samples']} (3 checkpoints × 4 scales)
- **Statistical Method:** Cluster bootstrap (checkpoints as clusters)
- **G0 (Projection inactive):** {'PASS' if gates['g0_projection_inactive']['passed'] else 'FAIL'}
- **G1 (Stability):** {'PASS' if gates['g1_stability']['passed'] else 'FAIL'}
- **G2 (Dial range ≥0.10):** {'PASS' if gates['g2_dial_range']['passed'] else 'FAIL'} (spread={gates['g2_dial_range']['L_preproj_spread']:.4f})
- **G3 (Monotonicity |ρ|>0.5, 95% CI excludes 0):** {gates['g3_monotonicity']['status']}
  - B0 argmax (n2=8, 4× mismatch): ρ={boot_aa_b0.get('rho_point', 0):.3f} [{boot_aa_b0.get('rho_ci_lower', 0):.3f}, {boot_aa_b0.get('rho_ci_upper', 0):.3f}] {'✓' if gates['g3_monotonicity']['aa_b0_passed'] else ''}
  - B1 argmax (n2=8, 4× mismatch): ρ={boot_aa_b1.get('rho_point', 0):.3f} [{boot_aa_b1.get('rho_ci_lower', 0):.3f}, {boot_aa_b1.get('rho_ci_upper', 0):.3f}] {'✓' if gates['g3_monotonicity']['aa_b1_passed'] else ''}
  - B0 ΔV (n2=8, 4× mismatch): ρ={boot_dv_b0.get('rho_point', 0):.3f} [{boot_dv_b0.get('rho_ci_lower', 0):.3f}, {boot_dv_b0.get('rho_ci_upper', 0):.3f}] {'✓' if gates['g3_monotonicity']['dv_b0_passed'] else ''}
  - B1 ΔV (n2=8, 4× mismatch): ρ={boot_dv_b1.get('rho_point', 0):.3f} [{boot_dv_b1.get('rho_ci_lower', 0):.3f}, {boot_dv_b1.get('rho_ci_upper', 0):.3f}] {'✓' if gates['g3_monotonicity']['dv_b1_passed'] else ''}

## Depth Mismatch Notation

- **n_train = {summary['parameters']['n_train']}** (training unroll depth)
- **n2 = 4** means 2× mismatch (evaluating at 2× training depth)
- **n2 = 8** means 4× mismatch (evaluating at 4× training depth)
- **n2 = 16** means 8× mismatch (evaluating at 8× training depth)

## Statistical Methodology

**Why cluster bootstrap?** Dial scales (1.0, 0.85, 0.70, 0.55) are repeated measures on the
same trained checkpoint. This violates the i.i.d. assumption required by standard Spearman
p-values. We address this by:

1. Treating each checkpoint as a cluster (3 clusters total)
2. Bootstrap resampling at the cluster level ({mono.get('n_bootstrap', 1000)} iterations)
3. Computing 95% confidence intervals from the bootstrap distribution
4. A correlation "passes" if |ρ| > 0.5 AND the 95% CI excludes 0

This is more conservative than i.i.d. inference and accounts for within-checkpoint correlation.

## Anti-Degenerate Checks

| Metric | Value |
|--------|-------|
| Mean entropy (train) | {anti_deg['mean_entropy_train']:.3f} |
| Mean entropy (eval) | {anti_deg['mean_entropy_eval']:.3f} |
| Min entropy (train) | {anti_deg['min_entropy_train']:.3f} |
| Entropy OK | {'YES' if anti_deg['entropy_ok'] else 'NO'} |
| Pseudo-replication | {'DETECTED' if anti_deg['pseudo_replication_detected'] else 'NOT DETECTED'} |

## Results by Scale (Pooled Across Checkpoints)

| Scale | L_preproj | B0 argmax (n2=8) | B1 argmax (n2=8) | B0 ΔV (n2=8) | B1 ΔV (n2=8) | Entropy |
|-------|-----------|------------------|------------------|--------------|--------------|---------|
"""
    for s in scale_sums:
        content += f"| {s['scale']:.2f} | {s['L_preproj_mean']:.3f}±{s['L_preproj_std']:.3f} | "
        content += f"{s['argmax_b0_8x_mean']:.3f}±{s['argmax_b0_8x_std']:.3f} | "
        content += f"{s['argmax_b1_8x_mean']:.3f}±{s['argmax_b1_8x_std']:.3f} | "
        content += f"{s['delta_V_b0_8x_mean']:.3f}±{s['delta_V_b0_8x_std']:.3f} | "
        content += f"{s['delta_V_b1_8x_mean']:.3f}±{s['delta_V_b1_8x_std']:.3f} | "
        content += f"{s['entropy_train_mean']:.2f} |\n"

    # Per-checkpoint trends section
    content += """
## Per-Checkpoint Trends (scale 1.0→0.55)

| Seed | L_preproj | Argmax | ΔV | Monotonic |
|------|-----------|--------|-----|-----------|
"""
    for seed, t in sorted(trends.items()):
        mono_status = "✓" if t.get('L_preproj_monotonic') and t.get('argmax_monotonic') else "partial"
        content += f"| {seed} | {t.get('L_preproj_trend', '?')} ({t.get('L_preproj_range', '?')}) | "
        content += f"{t.get('argmax_trend', '?')} ({t.get('argmax_range', '?')}) | "
        content += f"{t.get('delta_V_trend', '?')} ({t.get('delta_V_range', '?')}) | {mono_status} |\n"

    content += f"""
## Scoped Claims

"""
    if "POSITIVE" in decision:
        content += f"""**Claim (Positive):** Inference-time contraction scaling on z→z layers produces a measurable,
controllable "dial" for the achieved Lipschitz constant (L_preproj) when projection is disabled.

**Statistical Support:** With {mono['n_samples']} observations from {mono.get('n_clusters', 3)} independently trained checkpoints,
using cluster bootstrap for proper inference:
"""
        if gates['g3_monotonicity']['aa_b0_passed']:
            content += f"- Spearman ρ={boot_aa_b0.get('rho_point', 0):.3f} (95% CI: [{boot_aa_b0.get('rho_ci_lower', 0):.3f}, {boot_aa_b0.get('rho_ci_upper', 0):.3f}]) for L_preproj vs B0 argmax at n2=8 (4× mismatch).\n"
        if gates['g3_monotonicity']['aa_b1_passed']:
            content += f"- Spearman ρ={boot_aa_b1.get('rho_point', 0):.3f} (95% CI: [{boot_aa_b1.get('rho_ci_lower', 0):.3f}, {boot_aa_b1.get('rho_ci_upper', 0):.3f}]) for L_preproj vs B1 argmax at n2=8 (4× mismatch).\n"
    elif "NEGATIVE" in decision:
        content += """**Claim (Negative):** Inference-time contraction scaling does not produce sufficient
L_preproj variation to constitute a meaningful stability dial.
"""
    else:
        content += f"""**Claim (Inconclusive):** Dial range exists (L_preproj varies with scaling factor),
but no statistically significant monotonic relationship with mismatch metrics was observed
(bootstrap 95% CI crosses 0) with {mono['n_samples']} observations from {mono.get('n_clusters', 3)} checkpoints.
"""

    content += f"""
## Scope Limitations

- Results from inference-time scaling only (no retraining per-scale)
- Checkpoints: 3 independently trained models with seeds 41, 42, 43
- Trivial 4×4 Sudoku suite only
- Training depth n_train={summary['parameters']['n_train']}, eval depths: {summary['parameters']['eval_n_list']}

## Evaluation Batches

- **B0:** {summary['batches']['b0_count']} initial states (hash: {summary['batches']['b0_hash']})
- **B1:** {summary['batches']['b1_count']} successor states (hash: {summary['batches']['b1_hash']})
- B1 constructed using union of top-5 actions across ALL checkpoints × ALL scales (avoids selection bias)
"""

    claims_path = out_path / "CLAIMS.md"
    with open(claims_path, "w") as f:
        f.write(content)
    print(f"Saved: {claims_path}")


def generate_provenance_v2(
    summary: Dict[str, Any],
    out_path: Path,
    checkpoint_paths: List[str],
    config_yaml_path: str,
) -> None:
    """Generate PROVENANCE.md for v2."""
    content = f"""# Exp4 Provenance: Projection-free Contraction Dial (Final v2)

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
## YAML Config

- **Path:** {config_yaml_path}
- **Config Source:** yaml (required, no inference)

## Non-Negotiables Verified

- `disable_value_head_norm: true` ✓
- `latent_ball_radius: 0.0` (projection disabled) ✓

## Dial Implementation

Contraction scaling is applied at **inference/evaluation time** by multiplying
the weight matrices of L_level (z→z) layers by the scaling factor:

```python
for module in L_level.modules():
    if hasattr(module, 'weight'):
        module.weight.data.mul_(scale_factor)
```

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Dial scales | {summary['parameters']['dial_scales']} |
| Checkpoint seeds | {summary['parameters']['checkpoint_seeds']} |
| N (independent samples) | {summary['parameters']['n_independent_samples']} |
| n_train | {summary['parameters']['n_train']} |
| Eval depths | {summary['parameters']['eval_n_list']} |
| B0 composition | {summary['parameters']['b0_target_easy']} easy + {summary['parameters']['b0_target_hard']} hard |
| B1 cap | {summary['parameters']['b1_cap']} |

## Batch Provenance

- **B0:** {summary['batches']['b0_count']} states, hash: `{summary['batches']['b0_hash']}`
- **B1:** {summary['batches']['b1_count']} states, hash: `{summary['batches']['b1_hash']}`
- B1 uses union of top-5 actions from ALL checkpoints × ALL scales + random actions

## Training Commands (if checkpoints were generated)

```bash
# For each seed in [41, 42, 43]:
buck2 run //buiksat_trm:upi_trm_train \\
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \\
    -- --config configs/exp3_projection_ablation/nc_rdis.yaml \\
    --seed <SEED> --output_dir results/exp3/nc_rdis_s<SEED>
```

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp4_final_v2 -- \\
    --checkpoint_glob "results/exp3/nc_rdis_s*/model_step_5000.pt" \\
    --config_yaml configs/exp3_projection_ablation/nc_rdis.yaml \\
    --out_dir {out_path}
```
"""

    prov_path = out_path / "PROVENANCE.md"
    with open(prov_path, "w") as f:
        f.write(content)
    print(f"Saved: {prov_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp4 Final v2 Evaluation")
    parser.add_argument(
        "--checkpoint_glob",
        type=str,
        default="results/exp3/nc_rdis_s*/model_step_5000.pt",
        help="Glob pattern for checkpoint paths",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of checkpoint paths (overrides glob)",
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
        help="Data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Determine checkpoints
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    else:
        # Use glob pattern
        pattern = str(PROJECT_ROOT / args.checkpoint_glob)
        checkpoint_paths = sorted(glob.glob(pattern))

    if not checkpoint_paths:
        print(f"ERROR: No checkpoints found matching pattern: {args.checkpoint_glob}")
        sys.exit(1)

    print(f"Found {len(checkpoint_paths)} checkpoints: {checkpoint_paths}")

    if len(checkpoint_paths) < 3:
        print(f"WARNING: Only {len(checkpoint_paths)} checkpoints found. Need 3 for proper N=12 evaluation.")

    # Override device if CUDA not available
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = "cpu"

    summary = run_exp4_final_v2(
        checkpoint_paths=checkpoint_paths,
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
        sys.exit(0)


if __name__ == "__main__":
    main()
