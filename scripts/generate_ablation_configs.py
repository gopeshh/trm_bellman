#!/usr/bin/env python3
"""
Generate ablation configs for diagnostic comparisons.

Creates configs that remove one implementation feature at a time. Results from
these finite comparisons describe only the evaluated runs; they do not establish
that a feature is necessary or sufficient for stability.
"""

import os
import yaml
from pathlib import Path

# Base directory for ablation configs
ABLATIONS_DIR = Path("configs/ablations")
ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Base theory-exact config (all features ON)
BASE_CONFIG = {
    # Shaped rewards with explicit terminal-outcome terms
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
    "latent_projection_mode": "enabled",
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
# ABLATION 1: Remove the contraction-oriented intervention
# ===================================================================
create_ablation(
    name="no_contraction",
    changes={
        "enable_contraction": False,
        "target_Lz": 1.0,  # No longer enforced, but set to 1.0 to disable
    },
    description=(
        "Disables the contraction-oriented architectural intervention.\n"
        "# This finite ablation does not test or establish the paper's\n"
        "# domain-relative contraction premise.\n"
        "# Spectral normalization is removed, so latent dynamics may expand."
    )
)

# ===================================================================
# ABLATION 2: Remove exact statewise baseline centering
# ===================================================================
create_ablation(
    name="no_exact_baseline",
    changes={
        "exact_baseline_summation": False,
        "batch_centered_advantage": True,  # Fall back to batch-level centering heuristic
    },
    description=(
        "Disables exact statewise baseline summation.\n"
        "# Uses batch-level centering instead (heuristic, not theory-exact).\n"
        "# Finite ablation results do not establish either population CPI\n"
        "# certificate or its required estimator-defect premise."
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
        "# This is a configuration contrast only; finite outcomes do not\n"
        "# establish the population CPI premises or a causal stability effect."
    )
)

# ===================================================================
# ABLATION 4: Remove forward-invariant recurrent projection
# ===================================================================
create_ablation(
    name="no_projection",
    changes={
        "latent_projection_mode": "disabled",
        "latent_ball_radius": None,
    },
    description=(
        "Disables forward-invariant recurrent projection.\n"
        "# This finite ablation does not establish or refute any invariant-set\n"
        "# or contraction premise from the paper."
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
        "Replaces exact pointwise probability mixing with parameter interpolation.\n"
        "# Uses param.lerp_(candidate_param, α), which is not the exact mixture\n"
        "# π_new = (1-α)π + απ_0 analyzed by the paper."
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
        "latent_projection_mode": "disabled",
        "latent_ball_radius": None,
        "theory_exact_mixture": False,
        "exact_k_step_targets": False,
    },
    description=(
        "Disables the configured paper-facing mechanisms (baseline RL).\n"
        "# This configuration supplies none of those mechanisms; finite outcomes\n"
        "# establish neither their population premises nor causal effects."
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
        "latent_projection_mode": "disabled",
        "latent_ball_radius": None,
        "theory_exact_mixture": False,
        "exact_k_step_targets": False,
    },
    description=(
        "Sparse rewards with the configured paper-facing mechanisms disabled.\n"
        "# This is a configuration contrast only; no performance ordering or\n"
        "# population-theorem premise is inferred from finite outcomes."
    )
)

print("\n" + "="*70)
print("Ablation configs generated successfully!")
print("="*70)
print(f"\nGenerated {len(list(ABLATIONS_DIR.glob('*.yaml')))} ablation configs in {ABLATIONS_DIR}/")
print("\nTo run ablations:")
print("  python upi_trm_train.py --config configs/ablations/ablation_<name>.yaml")
print("\nRecommended ablation sweep:")
print("  1. ablation_no_contraction.yaml      - Disables contraction intervention")
print("  2. ablation_no_exact_baseline.yaml   - Uses approximate centering")
print("  3. ablation_no_conservative_mixture.yaml - Tests CPI benefit")
print("  4. ablation_no_theory_features.yaml  - Full ablation (baseline)")
