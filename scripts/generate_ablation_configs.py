#!/usr/bin/env python3
"""
Generate ablation configs for ICML experiments.

Creates configs that systematically remove each theory-exact feature to isolate contributions.
Addresses Bahram's guidance (main.tex:1426-1429):

    "Add ablations that remove (i) contraction enforcement, (ii) exact centering,
     and (iii) conservative mixture updates, to show each is necessary for stability."
"""

import os
import yaml
from pathlib import Path

# Base directory for ablation configs
ABLATIONS_DIR = Path("configs/ablations")
ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Base theory-exact config (all features ON)
BASE_CONFIG = {
    # Shaped rewards with rush-to-fail mitigation
    "reward_shaping": True,
    "fail_terminal_reward": -10.0,
    "solve_terminal_reward": 0.0,

    # Core RL params
    "gamma": 0.99,
    "K": 1,
    "inner_unroll_n": 4,
    "max_edits": 120,

    # Theory-exact features (ALL ON in base)
    "exact_k_step_targets": True,
    "exact_baseline_summation": True,
    "theory_exact_mixture": True,
    "distill_mixture_policy": False,
    "batch_centered_advantage": False,

    # Contraction & Lipschitz
    "enable_contraction": True,
    "target_Lz": 0.9,
    "target_Lv": 1.0,
    "latent_ball_radius": 10.0,

    # Conservative updates
    "mixture_alpha": 0.05,

    # Latent mode
    "episodic_latent": True,

    # STOP action
    "stop_action_mode": "noop",
    "stop_action_penalty": -0.1,

    # Optimization
    "value_lr": 3.0e-4,
    "policy_lr": 3.0e-4,
    "entropy_coef": 0.01,
    "value_grad_clip": 5.0,
    "policy_grad_clip": 1.0,
    "lr_schedule": "cosine",
    "lr_warmup_steps": 500,
    "lr_min_factor": 0.1,

    # Replay & training
    "replay_capacity": 100000,
    "batch_size": 256,
    "num_train_steps": 10000,
    "rollout_episodes_per_step": 4,

    # Logging
    "log_interval": 10,
    "eval_interval": 50,
    "eval_num_episodes": 50,
    "use_tqdm": False,
    "debug_checks": False,
    "track_theory_metrics": True,

    # Task
    "task_name": "sudoku",
    "solved_threshold": 10.0,
}


def create_ablation(name: str, changes: dict, description: str):
    """Create an ablation config with specified changes."""
    config = BASE_CONFIG.copy()
    config.update(changes)

    # Add header comment
    header = f"""# Ablation: {name}
#
# {description}
#
# Changes from base theory-exact config:
"""
    for key, value in changes.items():
        header += f"#   - {key}: {value}\n"
    header += "#\n"

    # Write config
    filepath = ABLATIONS_DIR / f"ablation_{name}.yaml"
    with open(filepath, 'w') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {filepath}")
    return filepath


# ===================================================================
# ABLATION 1: Remove Contraction Enforcement (Assumption 4.2)
# ===================================================================
create_ablation(
    name="no_contraction",
    changes={
        "enable_contraction": False,
        "target_Lz": 1.0,  # No longer enforced, but set to 1.0 to disable
    },
    description=(
        "Disables contraction enforcement (Assumption 4.2).\n"
        "# Tests: Can the agent learn without L_z < 1 guarantee?\n"
        "# Expected: Higher variance, possible instability, worse sample efficiency.\n"
        "# This removes spectral normalization, so latent dynamics can expand."
    )
)

# ===================================================================
# ABLATION 2: Remove Exact Baseline (Theorem 5.9 - KEY CONTRIBUTION)
# ===================================================================
create_ablation(
    name="no_exact_baseline",
    changes={
        "exact_baseline_summation": False,
        "batch_centered_advantage": True,  # Fall back to batch-level centering heuristic
    },
    description=(
        "Disables exact baseline summation (Theorem 5.9 - KEY CONTRIBUTION).\n"
        "# Tests: Impact of O(α·ε_{A,0}) vs O(ε_A/(1-γ)) evaluation-error penalty.\n"
        "# Expected: Weaker improvement guarantee, possibly more gradient variance.\n"
        "# Uses batch-level centering instead (heuristic, not theory-exact)."
    )
)

# ===================================================================
# ABLATION 3: Remove Conservative Mixture (Aggressive α)
# ===================================================================
create_ablation(
    name="no_conservative_mixture",
    changes={
        "mixture_alpha": 1.0,  # Greedy update (no conservative mixing)
    },
    description=(
        "Disables conservative mixture updates (sets α=1.0).\n"
        "# Tests: Can the agent learn with greedy policy updates?\n"
        "# Expected: Possible instability, policy oscillation, worse final performance.\n"
        "# The CPI bound degenerates to standard PI (no conservatism)."
    )
)

# ===================================================================
# ABLATION 4: Remove Forward-Invariant Projection (Assumption 4.1)
# ===================================================================
create_ablation(
    name="no_projection",
    changes={
        "latent_ball_radius": 0.0,  # Disable projection
    },
    description=(
        "Disables forward-invariant projection (Assumption 4.1).\n"
        "# Tests: Can the agent learn without latent ball constraint?\n"
        "# Expected: Latents can escape the contractive region, breaking bounds.\n"
        "# May lead to NaN/Inf issues or divergence."
    )
)

# ===================================================================
# ABLATION 5: Remove Theory-Exact Mixture (Parameter-space interpolation)
# ===================================================================
create_ablation(
    name="no_theory_exact_mixture",
    changes={
        "theory_exact_mixture": False,
    },
    description=(
        "Disables theory-exact CPI mixture (Issue 4).\n"
        "# Tests: Parameter-space interpolation vs policy-space mixture.\n"
        "# Expected: Similar performance but technically NOT CPI-compliant.\n"
        "# Uses param.lerp_(candidate_param, α) instead of π_new = (1-α)π + απ_0."
    )
)

# ===================================================================
# ABLATION 6: ALL theory features disabled (baseline RL)
# ===================================================================
create_ablation(
    name="no_theory_features",
    changes={
        "enable_contraction": False,
        "target_Lz": 1.0,
        "exact_baseline_summation": False,
        "batch_centered_advantage": True,
        "mixture_alpha": 1.0,
        "latent_ball_radius": 0.0,
        "theory_exact_mixture": False,
        "exact_k_step_targets": False,
    },
    description=(
        "Disables ALL theory-exact features (baseline RL).\n"
        "# Tests: How much does the theory framework contribute?\n"
        "# Expected: Worst performance, highest variance, possible divergence.\n"
        "# This is a standard actor-critic baseline without UPI-TRM guarantees."
    )
)

# ===================================================================
# ABLATION 7: Sparse rewards WITHOUT theory features
# ===================================================================
create_ablation(
    name="sparse_no_theory",
    changes={
        "reward_shaping": False,
        "enable_contraction": False,
        "target_Lz": 1.0,
        "exact_baseline_summation": False,
        "batch_centered_advantage": True,
        "mixture_alpha": 1.0,
        "latent_ball_radius": 0.0,
        "theory_exact_mixture": False,
        "exact_k_step_targets": False,
    },
    description=(
        "Sparse rewards + NO theory features (hardest baseline).\n"
        "# Tests: Can standard actor-critic learn Sudoku from terminal rewards?\n"
        "# Expected: Very slow learning or complete failure.\n"
        "# This is the 'default RL' approach without any UPI-TRM innovations."
    )
)

print("\n" + "="*70)
print("Ablation configs generated successfully!")
print("="*70)
print(f"\nGenerated {len(list(ABLATIONS_DIR.glob('*.yaml')))} ablation configs in {ABLATIONS_DIR}/")
print("\nTo run ablations:")
print("  python upi_trm_train.py --config configs/ablations/ablation_<name>.yaml")
print("\nRecommended ablation sweep:")
print("  1. ablation_no_contraction.yaml      - Tests Assumption 4.2")
print("  2. ablation_no_exact_baseline.yaml   - Tests Theorem 5.9 (KEY)")
print("  3. ablation_no_conservative_mixture.yaml - Tests CPI benefit")
print("  4. ablation_no_theory_features.yaml  - Full ablation (baseline)")
