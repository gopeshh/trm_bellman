#!/usr/bin/env python3
"""
Phase 5: Centering + Alpha Sensitivity Diagnostic (Theorem Alignment).

This script implements the current Phase 5 experiment tracked in
/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md:

1. Compute advantage baseline two ways:
   - Exact baseline (sum over all ~97 actions) → E_{a~π}[Q̂(s,a)]
   - Approximate baseline (learned value function V(s))

2. Report centering defect:
   ε_cent = sup_s |E_{a~π}[Â(s,a)]|

   For exact: should be ~0 (numerical precision)
   For approximate: should be non-zero

3. Apply single policy update with varying α and measure:
   - KL(π_new || π_old) for each α
   - The finite-sample relationship between KL and α; the local mixture
     expansion is quadratic in α near zero

4. Generate paper-ready artifacts:
   - Centering defect comparison figure
   - One-step KL vs α line plot
   - Appendix table

Output:
    results/paper_ready/phase5_centering_alpha/

Usage:
    python scripts/phase5_centering_alpha.py

    # Or via buck2:
    buck2 run //buiksat_trm:phase5_centering_alpha -- \\
        --checkpoint results/exp5_v2_inputs_eval/nc_rdis_s42/model_step_5000.pt
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
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv
from models.recursive_reasoning.tiny_recursive_reasoning_model_ACTV1 import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "exp5_v2_inputs_eval" / "nc_rdis_s42" / "model_step_5000.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp3_projection_ablation" / "nc_rdis.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_ready" / "phase5_centering_alpha"

# Alpha values for policy update
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.4]

# Batch size for evaluation (small for expensive exact summation)
BATCH_SIZE = 32

# Number of states to evaluate
N_EVAL_STATES = 50  # Smaller for expensive exact baseline computation


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
            pass
    return None


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> Tuple[TinyRecursiveReasoningModel_ACTV1, dict]:
    """Load model from checkpoint using the same pattern as exp4_final_v2."""
    yaml_config = load_yaml_config(config_path)

    # Resolve the recurrent projection as one atomic mode/radius contract.
    disable_value_head_norm = yaml_config.get("disable_value_head_norm", False)
    projection_fields = {
        key: yaml_config[key]
        for key in ("latent_projection_mode", "latent_ball_radius")
        if key in yaml_config
    }
    projection_config = RLConfig(**projection_fields)
    latent_projection_mode = projection_config.latent_projection_mode
    latent_ball_radius = projection_config.latent_ball_radius

    if not disable_value_head_norm:
        print("WARNING: disable_value_head_norm is False in config. Forcing to True.")
        disable_value_head_norm = True

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

    # Infer dimensions from weights
    embed_key = "inner.embed_tokens.embedding_weight"
    if embed_key not in cleaned_state:
        embed_key = "inner.embed_tokens.weight"
    embed_weight = cleaned_state.get(embed_key, torch.zeros(6, 64))
    hidden_size = embed_weight.shape[1]
    vocab_size = embed_weight.shape[0]
    seq_len = 16  # 4×4 Sudoku

    has_value_head = any("value_head" in k for k in cleaned_state.keys())
    has_policy_head = any("edit_policy" in k for k in cleaned_state.keys())
    num_actions = seq_len * vocab_size + 1  # 97 for Sudoku 4x4

    if has_policy_head and "edit_policy.mlp.2.weight" in cleaned_state:
        num_actions = cleaned_state["edit_policy.mlp.2.weight"].shape[0]

    inner_unroll_n = yaml_config.get("inner_unroll_n", 2)
    target_Lz = yaml_config.get("target_Lz", 0.9)
    enable_contraction = yaml_config.get("enable_contraction", False)
    gamma = yaml_config.get("gamma", 0.99)

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
        rl_latent_projection_mode=latent_projection_mode,
        rl_latent_ball_radius=latent_ball_radius,
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
        "latent_projection_mode": latent_projection_mode,
        "latent_ball_radius": latent_ball_radius,
        "gamma": gamma,
    }

    return model, config_dict


def load_dataset(data_path: Path, n_samples: int) -> List[dict]:
    """Load Sudoku puzzles from dataset."""
    dataset_file = data_path / "puzzles.pt"
    if not dataset_file.exists():
        # Try alternative paths
        alt_paths = [
            PROJECT_ROOT / "data" / "sudoku-4x4-trivial" / "puzzles.pt",
            Path("/home/buiksat/trm_bellman/data/sudoku-4x4-trivial/puzzles.pt"),
        ]
        for alt in alt_paths:
            if alt.exists():
                dataset_file = alt
                break
        else:
            print(f"WARNING: Dataset not found at {data_path}. Using random puzzles.")
            return None

    data = torch.load(dataset_file)
    samples = []
    for i in range(min(n_samples, len(data))):
        sample = data[i]
        if isinstance(sample, dict):
            samples.append(sample)
        else:
            # Handle tuple format (inputs, solution)
            samples.append({"inputs": sample[0], "solution": sample[1]})

    return samples


def generate_eval_batch(
    samples: Optional[List[dict]],
    n_states: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[dict, torch.Tensor]:
    """Generate a batch of evaluation states."""
    if samples is None:
        # Generate random puzzles
        inputs_list = []
        plans_list = []
        for _ in range(n_states):
            # Random puzzle with ~4-6 empties
            inputs = torch.randint(2, vocab_size, (16,))
            # Randomly set some cells as "given" (token > 1)
            given_mask = torch.rand(16) > 0.7
            inputs[~given_mask] = 1  # Empty token
            plans_list.append(inputs.clone())
            inputs_list.append(inputs)

        inputs_batch = torch.stack(inputs_list).to(device)
        plans_batch = torch.stack(plans_list).to(device)
    else:
        inputs_list = []
        plans_list = []
        for i in range(min(n_states, len(samples))):
            sample = samples[i]
            inputs = sample["inputs"]
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.flatten()
            else:
                inputs = torch.tensor(inputs).flatten()
            # Plan starts as inputs (empty cells will be edited)
            plan = inputs.clone()
            inputs_list.append(inputs)
            plans_list.append(plan)

        inputs_batch = torch.stack(inputs_list).to(device)
        plans_batch = torch.stack(plans_list).to(device)

    # Create x_batch dict (inputs + puzzle_identifier)
    x_batch = {
        "inputs": inputs_batch,
        "puzzle_identifier": torch.zeros(n_states, dtype=torch.long, device=device),
    }

    return x_batch, plans_batch


# =============================================================================
# Core Diagnostic Functions
# =============================================================================

def apply_edit_batch(
    y_batch: torch.Tensor,
    action: int,
    vocab_size: int,
    stop_action_id: int,
) -> torch.Tensor:
    """Apply an edit action to a batch of plans."""
    B = y_batch.shape[0]
    y_next = y_batch.clone()

    if action == stop_action_id:
        # STOP action: no change
        return y_next

    # Decode action: action = pos * vocab_size + token
    pos = action // vocab_size
    token = action % vocab_size

    if pos < 16:  # Valid position
        y_next[:, pos] = token

    return y_next


def compute_shaped_reward(
    phi_old: torch.Tensor,
    phi_new: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute shaped reward: r = γ * Φ(s') - Φ(s)."""
    return gamma * phi_new - phi_old


def compute_checker_score(
    inputs: torch.Tensor,
    plan: torch.Tensor,
) -> torch.Tensor:
    """Compute simple checker score (number of correct cells)."""
    # For each cell, check if it matches inputs (for given cells) or is filled (for empty cells)
    # This is a simplified scorer for the diagnostic

    # Count filled cells (token > 1)
    filled = (plan > 1).float().sum(dim=-1)
    return filled


def compute_centering_defect_exact(
    model: nn.Module,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    n: int,
    gamma: float,
    vocab_size: int,
    num_actions: int,
    stop_action_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute centering defect for exact baseline via full enumeration.

    For exact baseline:
        baseline = Σ_a π(a|s) × Q̂(s,a)
        Â(s,a) = Q̂(s,a) - baseline
        E_{a~π}[Â(s,a)] = 0 by construction

    Returns:
        centering_defects: [B] - |E_{a~π}[Â(s,a)]| per state (should be ~0)
        q_all: [B, A] - Q values for all actions
        exact_baseline: [B] - Exact baseline per state
    """
    B = y_batch.size(0)

    # Get action mask
    action_mask = PlanEditEnv.compute_batch_action_mask(
        x_batch["inputs"], vocab_size, stop_action_id, stop_mode="disabled"
    ).to(device)

    # Get policy distribution
    with torch.no_grad():
        dist, _ = model.policy_dist(x_batch, y_batch, n=n, action_mask=action_mask)
        probs = dist.probs  # [B, A]

    # Compute Q values for all actions
    q_all = torch.full((B, num_actions), float('-inf'), device=device)

    # Compute phi for current state
    phi_old = compute_checker_score(x_batch["inputs"], y_batch)

    # Get valid action indices
    if action_mask.dim() == 2:
        valid_actions_mask = action_mask.any(dim=0)
    else:
        valid_actions_mask = action_mask
    action_indices = torch.nonzero(valid_actions_mask, as_tuple=True)[0].tolist()

    for a in action_indices:
        with torch.no_grad():
            # Apply action
            y_next = apply_edit_batch(y_batch, a, vocab_size, stop_action_id)

            # Compute new phi
            phi_new = compute_checker_score(x_batch["inputs"], y_next)

            # Compute shaped reward
            rewards = compute_shaped_reward(phi_old, phi_new, gamma)

            # Check if terminal (STOP action - though we disabled it)
            is_stop = (a == stop_action_id)

            if is_stop:
                q_val = rewards
            else:
                # Get V(s')
                v_next, _ = model.used_value(x_batch, y_next, n=n)
                q_val = rewards + gamma * v_next

            # Only update for samples where this action is valid
            if action_mask.dim() == 2:
                valid_mask = action_mask[:, a]
                q_all[valid_mask, a] = q_val[valid_mask]
            else:
                q_all[:, a] = q_val

    # Mask invalid actions
    q_all = q_all.masked_fill(~action_mask, float('-inf'))

    # Compute exact baseline: Σ_a π(a) * Q(s,a)
    q_for_sum = q_all.masked_fill(q_all == float('-inf'), 0.0)
    exact_baseline = (probs * q_for_sum).sum(dim=-1)  # [B]

    # Compute E_{a~π}[Â(s,a)] = Σ_a π(a) * (Q(s,a) - baseline) = 0 by construction
    adv_for_sum = q_for_sum - exact_baseline.unsqueeze(-1)
    expected_adv = (probs * adv_for_sum).sum(dim=-1)  # [B]

    centering_defects = expected_adv.abs()  # Should be ~0 (numerical precision)

    return centering_defects, q_all, exact_baseline


def compute_centering_defect_approx(
    model: nn.Module,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    n: int,
    gamma: float,
    vocab_size: int,
    num_actions: int,
    stop_action_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute centering defect for approximate (learned) baseline.

    For approximate baseline:
        baseline = V(s) (learned value function)
        Â(s,a) = Q̂(s,a) - V(s)
        E_{a~π}[Â(s,a)] ≠ 0 in general

    Returns:
        centering_defects: [B] - |E_{a~π}[Â(s,a)]| per state
        q_all: [B, A] - Q values for all actions
        approx_baseline: [B] - V(s) per state
    """
    B = y_batch.size(0)

    # Get action mask
    action_mask = PlanEditEnv.compute_batch_action_mask(
        x_batch["inputs"], vocab_size, stop_action_id, stop_mode="disabled"
    ).to(device)

    # Get policy distribution and value function
    with torch.no_grad():
        dist, _ = model.policy_dist(x_batch, y_batch, n=n, action_mask=action_mask)
        probs = dist.probs  # [B, A]
        v_s, _ = model.used_value(x_batch, y_batch, n=n)  # [B]

    approx_baseline = v_s

    # Compute Q values (same as exact)
    q_all = torch.full((B, num_actions), float('-inf'), device=device)

    phi_old = compute_checker_score(x_batch["inputs"], y_batch)

    if action_mask.dim() == 2:
        valid_actions_mask = action_mask.any(dim=0)
    else:
        valid_actions_mask = action_mask
    action_indices = torch.nonzero(valid_actions_mask, as_tuple=True)[0].tolist()

    for a in action_indices:
        with torch.no_grad():
            y_next = apply_edit_batch(y_batch, a, vocab_size, stop_action_id)
            phi_new = compute_checker_score(x_batch["inputs"], y_next)
            rewards = compute_shaped_reward(phi_old, phi_new, gamma)

            is_stop = (a == stop_action_id)
            if is_stop:
                q_val = rewards
            else:
                v_next, _ = model.used_value(x_batch, y_next, n=n)
                q_val = rewards + gamma * v_next

            if action_mask.dim() == 2:
                valid_mask = action_mask[:, a]
                q_all[valid_mask, a] = q_val[valid_mask]
            else:
                q_all[:, a] = q_val

    q_all = q_all.masked_fill(~action_mask, float('-inf'))

    # Compute advantages using approximate baseline
    q_for_sum = q_all.masked_fill(q_all == float('-inf'), 0.0)
    adv = q_for_sum - approx_baseline.unsqueeze(-1)  # [B, A]

    # Compute E_{a~π}[Â(s,a)]
    expected_adv = (probs * adv).sum(dim=-1)  # [B]

    centering_defects = expected_adv.abs()

    return centering_defects, q_all, approx_baseline


def compute_kl_vs_alpha(
    model: nn.Module,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    n: int,
    gamma: float,
    vocab_size: int,
    num_actions: int,
    stop_action_id: int,
    alpha_values: List[float],
    device: torch.device,
    use_exact_baseline: bool = True,
) -> Dict[str, List[float]]:
    """
    Compute KL divergence for different α values.

    Uses variance-based approximation:
        KL(π_new || π_old) ≈ (1/2) * α² * E_{s}[Var_{a~π}[Â(s,a)]]

    This is theoretically motivated for small α updates.

    Returns:
        Dict with "alpha" and "kl" lists.
    """
    results = {"alpha": alpha_values, "kl": []}

    # Get action mask
    action_mask = PlanEditEnv.compute_batch_action_mask(
        x_batch["inputs"], vocab_size, stop_action_id, stop_mode="disabled"
    ).to(device)

    # Get policy distribution
    with torch.no_grad():
        dist, _ = model.policy_dist(x_batch, y_batch, n=n, action_mask=action_mask)
        probs = dist.probs  # [B, A]

    # Compute Q values for all actions
    if use_exact_baseline:
        centering_defects, q_all, baseline = compute_centering_defect_exact(
            model, x_batch, y_batch, n, gamma, vocab_size, num_actions,
            stop_action_id, device
        )
    else:
        centering_defects, q_all, baseline = compute_centering_defect_approx(
            model, x_batch, y_batch, n, gamma, vocab_size, num_actions,
            stop_action_id, device
        )

    # Compute advantages for all actions
    q_for_sum = q_all.masked_fill(q_all == float('-inf'), 0.0)
    advantages = q_for_sum - baseline.unsqueeze(-1)  # [B, A]

    # Compute Var_{a~π}[Â(s,a)] = E[Â²] - E[Â]² (but for centered, E[Â]=0)
    # So Var ≈ E_{a~π}[Â²]
    adv_squared = advantages ** 2
    expected_adv_sq = (probs * adv_squared).sum(dim=-1)  # [B]
    mean_var = expected_adv_sq.mean().item()

    # For each alpha, compute KL ≈ (1/2) * α² * Var[Â]
    for alpha in alpha_values:
        kl = 0.5 * (alpha ** 2) * mean_var
        results["kl"].append(kl)

    return results


# =============================================================================
# Paper-Ready Artifact Generation
# =============================================================================

def generate_figures(
    exact_defects: np.ndarray,
    approx_defects: np.ndarray,
    alpha_kl_exact: Dict[str, List[float]],
    alpha_kl_approx: Dict[str, List[float]],
    output_dir: Path,
):
    """Generate paper-ready figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Centering defect comparison
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Box plot comparing defects
    data = [exact_defects, approx_defects]
    labels = ['Exact\n(Statewise)', 'Approximate\n(Learned V)']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax.set_ylabel(r'Centering Defect $|\mathbb{E}_{a \sim \pi}[\hat{A}(s,a)]|$')
    ax.set_title('Centering Defect Comparison')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add text annotations
    ax.text(1, exact_defects.mean(), f'mean={exact_defects.mean():.2e}',
            ha='center', va='bottom', fontsize=9)
    ax.text(2, approx_defects.mean(), f'mean={approx_defects.mean():.2e}',
            ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig_phase5_centering_defect.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'fig_phase5_centering_defect.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Figure 2: KL vs alpha
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    alphas = np.array(alpha_kl_exact['alpha'])
    kl_exact = np.array(alpha_kl_exact['kl'])
    kl_approx = np.array(alpha_kl_approx['kl'])

    ax.plot(alphas, kl_exact, 'g-o', label='Exact Baseline', markersize=8)
    ax.plot(alphas, kl_approx, 'r-s', label='Approximate Baseline', markersize=8)

    # Fit linear trend to show O(α) vs O(α²) scaling
    # Theory predicts KL ∝ α² for small α
    alpha_sq = alphas ** 2

    # Plot α² reference line (scaled to match)
    scale_exact = kl_exact[-1] / (alphas[-1] ** 2) if kl_exact[-1] > 0 else 1
    ax.plot(alphas, scale_exact * alpha_sq, 'g--', alpha=0.5, label=r'$\propto \alpha^2$ (exact)')

    ax.set_xlabel(r'Mixture Parameter $\alpha$')
    ax.set_ylabel(r'$\mathrm{KL}(\pi_{new} \| \pi_{old})$')
    ax.set_title(r'KL Divergence vs $\alpha$ (One-Step Update)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig_phase5_kl_vs_alpha.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'fig_phase5_kl_vs_alpha.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Figures saved to {output_dir}")


def generate_table(
    exact_defects: np.ndarray,
    approx_defects: np.ndarray,
    alpha_kl_exact: Dict[str, List[float]],
    alpha_kl_approx: Dict[str, List[float]],
    output_dir: Path,
):
    """Generate LaTeX table for appendix."""

    table_content = r"""\begin{table}[h]
\centering
\caption{Phase 5: Centering Defect and KL Sensitivity}
\label{tab:phase5}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Exact Baseline} & \textbf{Approximate Baseline} \\
\midrule
Centering Defect (mean) & %.2e & %.2e \\
Centering Defect (max) & %.2e & %.2e \\
\midrule
""" % (exact_defects.mean(), approx_defects.mean(),
       exact_defects.max(), approx_defects.max())

    # Add KL vs alpha rows
    for i, alpha in enumerate(alpha_kl_exact['alpha']):
        kl_e = alpha_kl_exact['kl'][i]
        kl_a = alpha_kl_approx['kl'][i]
        table_content += r"KL @ $\alpha=%.2f$ & %.4f & %.4f \\" % (alpha, kl_e, kl_a)
        table_content += "\n"

    table_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'table_phase5_centering.tex', 'w') as f:
        f.write(table_content)

    print(f"Table saved to {output_dir / 'table_phase5_centering.tex'}")


def generate_claims(
    exact_defects: np.ndarray,
    approx_defects: np.ndarray,
    alpha_kl_exact: Dict[str, List[float]],
    output_dir: Path,
    git_sha: str,
    latent_projection_mode: str,
    latent_ball_radius: Optional[float],
):
    """Generate CLAIMS.md for Phase 5."""

    content = f"""# Phase 5 Claims: Centering + Alpha Sensitivity

**Generated:** {datetime.now().isoformat()}
**Git SHA:** {git_sha}

## Summary

Phase 5 reports finite diagnostics for exact statewise baseline centering:

### Centering Defect

| Baseline | Mean Defect | Max Defect |
|----------|-------------|------------|
| Exact (statewise) | {exact_defects.mean():.2e} | {exact_defects.max():.2e} |
| Approximate (Learned V) | {approx_defects.mean():.2e} | {approx_defects.max():.2e} |

**Interpretation:**
- On the sampled represented states, exact enumeration records ε_cent ≈ {exact_defects.mean():.1e}
- On the same states, the learned-value baseline records ε_cent ≈ {approx_defects.mean():.1e}
- Recorded ratio: {approx_defects.mean() / max(exact_defects.mean(), 1e-10):.1f}×

### KL Scaling with α

| α | KL (Exact) | α² Reference |
|---|------------|--------------|
"""

    for i, alpha in enumerate(alpha_kl_exact['alpha']):
        kl = alpha_kl_exact['kl'][i]
        alpha_sq = alpha ** 2
        content += f"| {alpha:.2f} | {kl:.4f} | {alpha_sq:.4f} |\n"

    content += f"""
**Interpretation:**
- The recorded KL values are compared with an α² reference.
- This finite comparison establishes neither the population CPI bound nor its
  uniform premises.

## Scoped Observations

1. **Exact enumeration on sampled states:** the recorded centering defect is {exact_defects.mean():.2e}.

2. **Learned-value baseline on the same states:** the recorded centering defect is {approx_defects.mean():.2e}.

3. **Finite alpha comparison:** recorded single-step KL values are shown beside α²; no population scaling law is inferred.

## Scope Limitations

- Evaluated on {len(exact_defects)} states from Sudoku 4×4 trivial suite
- Single checkpoint (seed 42)
- KL approximation uses variance-based formula (not exact gradient step)

## Recorded Configuration

- `disable_value_head_norm: true`
- `latent_projection_mode: {latent_projection_mode}`
- `latent_ball_radius: {"null" if latent_ball_radius is None else latent_ball_radius}`
"""

    with open(output_dir / 'CLAIMS.md', 'w') as f:
        f.write(content)

    print(f"CLAIMS.md saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Centering + Alpha Diagnostic")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to config YAML")
    parser.add_argument("--out_dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for artifacts")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for evaluation")
    parser.add_argument("--n_states", type=int, default=N_EVAL_STATES,
                        help="Number of states to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5: Centering + Alpha Sensitivity Diagnostic")
    print("=" * 70)
    print()

    # Get git SHA
    git_sha = get_git_sha() or "unknown"
    print(f"Git SHA: {git_sha}")

    # Check checkpoint exists
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Please train a model first or specify a valid checkpoint path.")
        sys.exit(1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    device = torch.device(args.device)

    print("Loading model and config...")
    model, config_dict = load_model(args.checkpoint, args.config, device)

    vocab_size = config_dict["vocab_size"]
    num_actions = config_dict["num_actions"]
    stop_action_id = vocab_size * 16  # For Sudoku 4x4: 16 * 6 = 96
    n = config_dict["inner_unroll_n"]
    gamma = config_dict["gamma"]

    print(f"  vocab_size: {vocab_size}")
    print(f"  num_actions: {num_actions}")
    print(f"  inner_unroll_n: {n}")
    print(f"  gamma: {gamma}")
    print()

    # Load dataset
    print("Loading dataset...")
    data_path = PROJECT_ROOT / "data" / "sudoku-4x4-trivial"
    samples = load_dataset(data_path, args.n_states)

    print("Generating evaluation batch...")
    x_batch, y_batch = generate_eval_batch(samples, args.n_states, vocab_size, device)

    print(f"  Batch size: {y_batch.shape[0]}")
    print()

    print("=" * 70)
    print("Computing Centering Defects")
    print("=" * 70)
    print()

    # Compute exact centering defect
    print("Computing exact centering defect (this may take a while)...")
    exact_defects, _, _ = compute_centering_defect_exact(
        model, x_batch, y_batch, n, gamma, vocab_size, num_actions,
        stop_action_id, device
    )
    exact_defects_np = exact_defects.cpu().numpy()
    print(f"  Exact defect: mean={exact_defects_np.mean():.2e}, max={exact_defects_np.max():.2e}")

    # Compute approximate centering defect
    print("Computing approximate centering defect...")
    approx_defects, _, _ = compute_centering_defect_approx(
        model, x_batch, y_batch, n, gamma, vocab_size, num_actions,
        stop_action_id, device
    )
    approx_defects_np = approx_defects.cpu().numpy()
    print(f"  Approx defect: mean={approx_defects_np.mean():.2e}, max={approx_defects_np.max():.2e}")

    print()
    print("=" * 70)
    print("Computing KL vs Alpha")
    print("=" * 70)
    print()

    # Run alpha sensitivity with exact baseline
    print("Running alpha sensitivity (exact baseline)...")
    alpha_kl_exact = compute_kl_vs_alpha(
        model, x_batch, y_batch, n, gamma, vocab_size, num_actions,
        stop_action_id, ALPHA_VALUES, device, use_exact_baseline=True
    )
    for alpha, kl in zip(alpha_kl_exact['alpha'], alpha_kl_exact['kl']):
        print(f"  α={alpha:.2f}: KL={kl:.4f}")

    # Run alpha sensitivity with approximate baseline
    print("Running alpha sensitivity (approximate baseline)...")
    alpha_kl_approx = compute_kl_vs_alpha(
        model, x_batch, y_batch, n, gamma, vocab_size, num_actions,
        stop_action_id, ALPHA_VALUES, device, use_exact_baseline=False
    )
    for alpha, kl in zip(alpha_kl_approx['alpha'], alpha_kl_approx['kl']):
        print(f"  α={alpha:.2f}: KL={kl:.4f}")

    print()
    print("=" * 70)
    print("Generating Paper-Ready Artifacts")
    print("=" * 70)
    print()

    # Generate figures
    generate_figures(
        exact_defects_np, approx_defects_np,
        alpha_kl_exact, alpha_kl_approx,
        args.out_dir
    )

    # Generate table
    generate_table(
        exact_defects_np, approx_defects_np,
        alpha_kl_exact, alpha_kl_approx,
        args.out_dir
    )

    # Generate CLAIMS.md
    generate_claims(
        exact_defects_np, approx_defects_np,
        alpha_kl_exact,
        args.out_dir, git_sha,
        config_dict["latent_projection_mode"],
        config_dict["latent_ball_radius"],
    )

    # Save summary.json
    summary = {
        "git_sha": git_sha,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint),
        "n_states": args.n_states,
        "exact_defect_mean": float(exact_defects_np.mean()),
        "exact_defect_max": float(exact_defects_np.max()),
        "approx_defect_mean": float(approx_defects_np.mean()),
        "approx_defect_max": float(approx_defects_np.max()),
        "alpha_kl_exact": alpha_kl_exact,
        "alpha_kl_approx": alpha_kl_approx,
    }

    with open(args.out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {args.out_dir / 'summary.json'}")

    print()
    print("=" * 70)
    print("Phase 5 COMPLETE")
    print("=" * 70)
    print()
    print(f"Artifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
