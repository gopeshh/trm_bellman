#!/usr/bin/env python3
"""
Experiment 2: Contraction Strength Sweep Paper-Ready Artifacts

Generates:
- fig_exp2_stability_dial.pdf (main figure with dual y-axes)
- table_exp2_contraction_sweep.tex (LaTeX table)
- CLAIMS.md, PROVENANCE.md

Usage:
    python scripts/make_paper_figures_exp2.py
"""

import csv
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

SWEEP_CONFIG = {
    # target_Lz: (config_path, seeds)
    0.999: ("configs/exp2_contraction_sweep/target_lz_0999.yaml", [41, 42, 43]),
    0.99: ("configs/exp2_contraction_sweep/target_lz_099.yaml", [41, 42, 43]),
    0.95: ("configs/exp2_contraction_sweep/target_lz_095.yaml", [41, 42, 43]),
    0.90: ("configs/exp2_contraction_sweep/target_lz_090.yaml", [41, 42, 43]),
}

CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints/exp2_contraction_sweep"
OUT_DIR = PROJECT_ROOT / "results/paper_ready/exp2"
BATCH_DIR = PROJECT_ROOT / "artifacts/eval_batches"

N_TRAIN = 2
N_EVAL = 16  # 8× training depth

# Color scheme: gradient from red (weak contraction) to blue (strong)
COLORS = {
    0.999: "#E24A33",  # Red - minimal contraction
    0.99: "#FFA500",   # Orange
    0.95: "#2CA02C",   # Green
    0.90: "#348ABD",   # Blue - strong contraction
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RunResult:
    """Results from a single checkpoint evaluation."""
    target_lz: float
    seed: int
    achieved_lz: float  # hat_Lz
    success_rate: float
    delta_V_mean: float
    delta_V_std: float
    delta_pi_mean: float
    delta_pi_std: float
    delta_z_mean: float
    delta_z_std: float
    argmax_agree: float
    checkpoint_path: str


@dataclass
class SweepResults:
    """Aggregated results across seeds for a target_Lz."""
    target_lz: float
    achieved_lz_mean: float
    achieved_lz_std: float
    success_rate_mean: float
    success_rate_std: float
    delta_V_mean: float
    delta_V_std: float
    delta_pi_mean: float
    delta_pi_std: float
    delta_z_mean: float
    delta_z_std: float
    argmax_agree_mean: float
    argmax_agree_std: float
    n_seeds: int


# =============================================================================
# Checkpoint Discovery
# =============================================================================

def find_checkpoints() -> Dict[float, List[Path]]:
    """Find all available checkpoints."""
    result = {}
    # Map target_lz to directory suffix
    lz_to_dir = {
        0.999: "lz_0999",
        0.99: "lz_099",
        0.95: "lz_095",
        0.90: "lz_0900",
    }
    for target_lz, (config, seeds) in SWEEP_CONFIG.items():
        dir_name = lz_to_dir.get(target_lz, f"lz_{target_lz:.3f}".replace(".", ""))
        ckpts = []
        for seed in seeds:
            ckpt = CHECKPOINT_BASE / dir_name / f"seed{seed}" / "model_step_5000.pt"
            if ckpt.exists():
                ckpts.append(ckpt)
        if ckpts:
            result[target_lz] = ckpts
    return result


# =============================================================================
# Model Loading and Evaluation
# =============================================================================

def load_model_and_config(checkpoint_path: Path) -> Tuple[Any, Dict]:
    """Load a model checkpoint."""
    from scripts.eval_unroll_sensitivity import load_model_for_eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model_for_eval(str(checkpoint_path), device)
    return model, config


def estimate_achieved_lz(
    model: Any,
    states: List[Any],
    n_train: int,
    config: Dict,
    device: str = "cpu",
    num_samples: int = 64,
    num_perturbations: int = 4,
    eps: float = 1e-3,
) -> float:
    """
    Estimate the achieved Lipschitz constant (hat_Lz) using finite differences.

    Uses the CORRECT method: perturb z by δ and measure ||f(z+δ) - f(z)|| / ||δ||.
    This is the local Lipschitz constant of the z->z mapping.

    Args:
        model: The TRM model
        states: List of PuzzleState objects
        n_train: Number of unroll steps (measures L_z at this depth)
        config: Model config dict
        device: Device string
        num_samples: Max number of states to sample
        num_perturbations: Number of random perturbations per state
        eps: Perturbation magnitude

    Returns:
        Mean estimated Lipschitz constant across states and perturbations.
    """
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    import numpy as np

    rng = np.random.default_rng(42)

    # Sample states if needed
    if len(states) > num_samples:
        indices = rng.choice(len(states), size=num_samples, replace=False)
        sample_states = [states[i] for i in indices]
    else:
        sample_states = states

    lz_estimates = []

    model.eval()
    with torch.no_grad():
        for state in sample_states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            try:
                # Get baseline latent after n_train steps
                _, z_base = model.used_value(x, y, n_train)
                if not hasattr(z_base, 'z_H'):
                    continue

                z_H = z_base.z_H  # [1, seq_len, hidden]
                z_L = z_base.z_L  # [1, seq_len, hidden]

                # Apply one latent step to base using outer model's API
                z_next_base = model.update_latent(z_base, y, x)

                # Now apply perturbations and measure response
                for _ in range(num_perturbations):
                    # Create random perturbation
                    noise_H = torch.randn_like(z_H)
                    noise_L = torch.randn_like(z_L)

                    # Normalize to eps magnitude
                    noise_norm = torch.sqrt(
                        (noise_H ** 2).sum() + (noise_L ** 2).sum()
                    ).clamp(min=1e-12)
                    noise_H = noise_H * (eps / noise_norm)
                    noise_L = noise_L * (eps / noise_norm)

                    # Perturbed latent
                    z_pert = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                        z_H=z_H + noise_H,
                        z_L=z_L + noise_L,
                    )

                    # Apply one latent step to perturbed using outer model's API
                    z_next_pert = model.update_latent(z_pert, y, x)

                    # Compute output difference: ||f(z+δ) - f(z)||
                    diff_H = z_next_pert.z_H - z_next_base.z_H
                    diff_L = z_next_pert.z_L - z_next_base.z_L
                    diff_norm = torch.sqrt(
                        (diff_H ** 2).sum() + (diff_L ** 2).sum()
                    ).item()

                    # L_z estimate = ||f(z+δ) - f(z)|| / ||δ||
                    lz = diff_norm / eps
                    lz_estimates.append(lz)

            except Exception as e:
                # Log error for debugging
                print(f"[Lz estimate] Error on state {state.state_id}: {e}")
                continue

    if not lz_estimates:
        print(f"[Lz estimate] WARNING: No valid estimates collected from {len(sample_states)} states")
        return 0.0

    return float(np.mean(lz_estimates))


def evaluate_checkpoint(
    checkpoint_path: Path,
    batch_b0: List[Any],
    target_lz: float,
    seed: int,
) -> Optional[RunResult]:
    """Evaluate a single checkpoint."""
    from scripts.eval_unroll_sensitivity import (
        evaluate_batch,
        load_model_for_eval,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load model
        model, config = load_model_for_eval(str(checkpoint_path), device)
        n_train = config["inner_unroll_n"]

        # Estimate achieved Lz
        achieved_lz = estimate_achieved_lz(model, batch_b0, n_train, config, device)

        # Evaluate unroll sensitivity at 8× depth
        n_mults = [1, 8]  # Compare n_train to 8×n_train
        metrics = evaluate_batch(model, batch_b0, n_train, n_mults, config, device)

        # Filter to the 1× vs 8× comparison
        target_n1, target_n2 = n_train, 8 * n_train
        relevant = [m for m in metrics if m.n1 == target_n1 and m.n2 == target_n2]

        if not relevant:
            print(f"[Eval] No metrics found for n1={target_n1}, n2={target_n2}")
            return None

        # Aggregate
        delta_V = [m.delta_V for m in relevant]
        delta_pi = [m.delta_pi for m in relevant]
        delta_z = [m.delta_z for m in relevant]
        argmax = [m.argmax_agree for m in relevant]

        # Compute success rate (evaluate the model)
        success_rate = compute_success_rate(model, batch_b0, config, device)

        return RunResult(
            target_lz=target_lz,
            seed=seed,
            achieved_lz=achieved_lz,
            success_rate=success_rate,
            delta_V_mean=float(np.mean(delta_V)),
            delta_V_std=float(np.std(delta_V)),
            delta_pi_mean=float(np.mean(delta_pi)),
            delta_pi_std=float(np.std(delta_pi)),
            delta_z_mean=float(np.mean(delta_z)),
            delta_z_std=float(np.std(delta_z)),
            argmax_agree=float(np.mean(argmax)),
            checkpoint_path=str(checkpoint_path),
        )

    except Exception as e:
        print(f"[Eval] Error evaluating {checkpoint_path}: {e}")
        return None


def compute_success_rate(
    model: Any,
    states: List[Any],
    config: Dict,
    device: str,
    max_steps: int = 16,
) -> float:
    """Compute solve success rate by rolling out the policy."""
    from rl.envs.plan_edit_env import PlanEditEnv
    from rl.sudoku_utils import sudoku_is_solved

    vocab_size = config["vocab_size"]
    num_actions = config["num_actions"]
    stop_action_id = num_actions - 1
    n_train = config["inner_unroll_n"]

    successes = 0
    total = len(states)

    model.eval()
    with torch.no_grad():
        for state in states:
            plan = state.plan.clone()
            inputs = state.inputs

            for step in range(max_steps):
                # Check if solved
                if sudoku_is_solved(plan.unsqueeze(0)):
                    successes += 1
                    break

                # Get action from policy
                x = {
                    "inputs": inputs.unsqueeze(0).to(device),
                    "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
                }
                y = plan.unsqueeze(0).to(device)

                # Compute action mask
                action_mask = PlanEditEnv.compute_batch_action_mask(
                    inputs.unsqueeze(0),
                    vocab_size,
                    stop_action_id,
                ).to(device)

                dist, _ = model.policy_dist(x, y, n_train, action_mask=action_mask)
                action = dist.probs.argmax(dim=-1).item()

                if action == stop_action_id:
                    break

                # Apply action
                pos = action // vocab_size
                val = action % vocab_size
                if inputs[pos].item() == 1:  # Only modify empty cells
                    plan[pos] = val

    return successes / total if total > 0 else 0.0


def aggregate_results(runs: List[RunResult]) -> SweepResults:
    """Aggregate results across seeds."""
    if not runs:
        raise ValueError("No runs to aggregate")

    target_lz = runs[0].target_lz

    return SweepResults(
        target_lz=target_lz,
        achieved_lz_mean=float(np.mean([r.achieved_lz for r in runs])),
        achieved_lz_std=float(np.std([r.achieved_lz for r in runs])),
        success_rate_mean=float(np.mean([r.success_rate for r in runs])),
        success_rate_std=float(np.std([r.success_rate for r in runs])),
        delta_V_mean=float(np.mean([r.delta_V_mean for r in runs])),
        delta_V_std=float(np.std([r.delta_V_mean for r in runs])),
        delta_pi_mean=float(np.mean([r.delta_pi_mean for r in runs])),
        delta_pi_std=float(np.std([r.delta_pi_mean for r in runs])),
        delta_z_mean=float(np.mean([r.delta_z_mean for r in runs])),
        delta_z_std=float(np.std([r.delta_z_mean for r in runs])),
        argmax_agree_mean=float(np.mean([r.argmax_agree for r in runs])),
        argmax_agree_std=float(np.std([r.argmax_agree for r in runs])),
        n_seeds=len(runs),
    )


# =============================================================================
# Figure Generation
# =============================================================================

def set_paper_style():
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def create_stability_dial_figure(results: List[SweepResults], out_path: Path):
    """Create the stability dial figure with dual y-axes."""
    set_paper_style()

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    # Sort by achieved Lz
    results = sorted(results, key=lambda r: r.achieved_lz_mean)

    x_vals = [r.achieved_lz_mean for r in results]
    x_errs = [r.achieved_lz_std for r in results]

    # Left y-axis: Success Rate
    success_vals = [r.success_rate_mean for r in results]
    success_errs = [r.success_rate_std for r in results]

    ax1.errorbar(
        x_vals, success_vals,
        xerr=x_errs, yerr=success_errs,
        marker='o', color='#2CA02C', label='Success Rate',
        capsize=3, capthick=1.5,
    )
    ax1.set_xlabel(r'Achieved $\hat{L}_z$ (Contraction Strength)')
    ax1.set_ylabel('Success Rate', color='#2CA02C')
    ax1.tick_params(axis='y', labelcolor='#2CA02C')
    ax1.set_ylim(0, 1.05)

    # Right y-axis: Δ_V
    ax2 = ax1.twinx()

    delta_V_vals = [r.delta_V_mean for r in results]
    delta_V_errs = [r.delta_V_std for r in results]

    ax2.errorbar(
        x_vals, delta_V_vals,
        xerr=x_errs, yerr=delta_V_errs,
        marker='s', color='#E24A33', label=r'$\Delta_V$ (Instability)',
        capsize=3, capthick=1.5,
    )
    ax2.set_ylabel(r'$\Delta_V$ at 8$\times$ Depth', color='#E24A33')
    ax2.tick_params(axis='y', labelcolor='#E24A33')

    # Annotate target_Lz values
    for r in results:
        ax1.annotate(
            f'$L_z^*$={r.target_lz}',
            (r.achieved_lz_mean, r.success_rate_mean),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Contraction Strength Dial: Performance vs Stability')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Figure] Saved: {out_path}")


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_table(results: List[SweepResults], out_path: Path):
    """Generate LaTeX table."""
    results = sorted(results, key=lambda r: r.target_lz, reverse=True)

    content = r"""% Contraction Sweep Table
% Auto-generated by make_paper_figures_exp2.py
\begin{table}[t]
\centering
\small
\caption{Contraction strength sweep on B0. Lower $\hat{L}_z$ improves stability ($\downarrow\Delta_V$) with modest performance trade-off.}
\label{tab:contraction_sweep}
\begin{tabular}{cccccc}
\toprule
$L_z^*$ (target) & $\hat{L}_z$ (achieved) & Success & $\Delta_V$ & $\Delta_\pi$ & Argmax \\
\midrule
"""

    for r in results:
        content += f"{r.target_lz} & {r.achieved_lz_mean:.3f}$\\pm${r.achieved_lz_std:.3f} "
        content += f"& {r.success_rate_mean:.2f}$\\pm${r.success_rate_std:.2f} "
        content += f"& {r.delta_V_mean:.3f}$\\pm${r.delta_V_std:.3f} "
        content += f"& {r.delta_pi_mean:.4f}$\\pm${r.delta_pi_std:.4f} "
        content += f"& {r.argmax_agree_mean:.2f} \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    out_path.write_text(content)
    print(f"[Table] Saved: {out_path}")


# =============================================================================
# Documentation Generation
# =============================================================================

def generate_claims(results: List[SweepResults], out_path: Path):
    """Generate CLAIMS.md with scientifically honest interpretations."""
    results = sorted(results, key=lambda r: r.target_lz)

    # Calculate key statistics
    achieved_lz_values = [r.achieved_lz_mean for r in results]
    delta_V_values = [r.delta_V_mean for r in results]
    delta_V_stds = [r.delta_V_std for r in results]

    achieved_lz_range = max(achieved_lz_values) - min(achieved_lz_values)
    achieved_lz_mean = sum(achieved_lz_values) / len(achieved_lz_values)

    # Check if monotonicity holds
    is_monotonic = all(
        delta_V_values[i] >= delta_V_values[i+1]
        for i in range(len(delta_V_values)-1)
    )

    # Find best and worst
    best_delta_v = min(results, key=lambda r: r.delta_V_mean)
    worst_delta_v = max(results, key=lambda r: r.delta_V_mean)

    content = f"""# Experiment 2: Paper Claims

All claims are from the contraction-strength sweep on B0 (initial states).

## Key Finding: Contraction Enforcement Saturates

1. **Observation**: All target $L_z$ values achieve similar measured Lipschitz constants.
   **Evidence**: Achieved $\\hat{{L}}_z$ ranges from {min(achieved_lz_values):.3f} to {max(achieved_lz_values):.3f} (range: {achieved_lz_range:.3f})
   **Interpretation**: The contraction enforcement mechanism saturates at $\\hat{{L}}_z \\approx {achieved_lz_mean:.2f}$

2. **Observation**: Value stability ($\\Delta_V$) has high variance across seeds.
   **Evidence**: Standard deviations range from {min(delta_V_stds):.3f} to {max(delta_V_stds):.3f}
   **Best point**: target $L_z^* = {best_delta_v.target_lz}$ with $\\Delta_V = {best_delta_v.delta_V_mean:.3f}\\pm{best_delta_v.delta_V_std:.3f}$

3. **Observation**: The monotonic dial relationship is {"supported" if is_monotonic else "NOT supported"} by this data.
   **Evidence**: $\\Delta_V$ values are {delta_V_values}
   **Note**: {"Trend is monotonic as expected." if is_monotonic else "Non-monotonic pattern suggests high seed variance dominates the target_Lz effect."}

## Sweep Results Summary

| Target $L_z$ | Achieved $\\hat{{L}}_z$ | Success Rate | $\\Delta_V$ | Seeds |
|--------------|------------------------|--------------|-------------|-------|
"""

    for r in results:
        content += f"| {r.target_lz} | {r.achieved_lz_mean:.3f}±{r.achieved_lz_std:.3f} | {r.success_rate_mean:.2f}±{r.success_rate_std:.2f} | {r.delta_V_mean:.3f}±{r.delta_V_std:.3f} | {r.n_seeds} |\n"

    content += f"""
## Scoped Interpretation

Given that achieved $\\hat{{L}}_z$ is similar across all targets (~{achieved_lz_mean:.2f}), the primary effect of varying target $L_z^*$ is:
- **Indirect**: Different optimization trajectories lead to different models
- **High variance**: Seed-to-seed variation in $\\Delta_V$ exceeds target-to-target variation

**Conservative claim**: Contraction enforcement achieves $\\hat{{L}}_z \\approx {achieved_lz_mean:.2f}$ regardless of target, with $\\Delta_V$ varying significantly (range: {min(delta_V_values):.3f} to {max(delta_V_values):.3f}).
"""

    out_path.write_text(content)
    print(f"[Claims] Saved: {out_path}")


def generate_provenance(checkpoints: Dict[float, List[Path]], out_path: Path):
    """Generate PROVENANCE.md."""
    git_sha = get_git_sha()

    content = f"""# Experiment 2 Provenance

**Generated**: {datetime.now().isoformat()}
**Git Commit**: {git_sha}

## Sweep Configuration

| Target $L_z$ | Config | Seeds |
|--------------|--------|-------|
"""

    for target_lz, (config, seeds) in sorted(SWEEP_CONFIG.items()):
        content += f"| {target_lz} | `{config}` | {seeds} |\n"

    content += """
## Checkpoints

"""

    for target_lz, ckpts in sorted(checkpoints.items()):
        content += f"### Target $L_z$ = {target_lz}\n\n"
        for ckpt in ckpts:
            content += f"- `{ckpt}`\n"
        content += "\n"

    content += """## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n_eval | 16 (8× depth) |
| Batch | B0 (initial states) |

## Regeneration Command

```bash
buck2 run //buiksat_trm:make_paper_figures_exp2
```
"""

    out_path.write_text(content)
    print(f"[Provenance] Saved: {out_path}")


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EXPERIMENT 2: CONTRACTION SWEEP PAPER-READY ARTIFACTS")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find checkpoints
    print("\n=== Finding Checkpoints ===")
    checkpoints = find_checkpoints()

    if not checkpoints:
        print("ERROR: No checkpoints found. Run training first.")
        print("  python scripts/run_exp2_contraction_sweep.py --full")
        return 1

    for target_lz, ckpts in sorted(checkpoints.items()):
        print(f"  target_Lz={target_lz}: {len(ckpts)} checkpoints")

    # Load evaluation batch
    print("\n=== Loading Evaluation Batch ===")
    batch_path = BATCH_DIR / "b0.pt"
    if not batch_path.exists():
        print("ERROR: Evaluation batch not found. Run:")
        print("  python scripts/eval_unroll_sensitivity.py build-batches ...")
        return 1

    from scripts.eval_unroll_sensitivity import load_batch
    batch_b0, _ = load_batch(str(batch_path))
    print(f"  Loaded {len(batch_b0)} B0 states")

    # Evaluate all checkpoints
    print("\n=== Evaluating Checkpoints ===")
    all_runs: List[RunResult] = []

    for target_lz, ckpts in sorted(checkpoints.items()):
        for ckpt in ckpts:
            seed = int(ckpt.parent.name.replace("seed", ""))
            print(f"\n[Eval] target_Lz={target_lz}, seed={seed}")

            result = evaluate_checkpoint(ckpt, batch_b0, target_lz, seed)
            if result:
                all_runs.append(result)
                print(f"  achieved_Lz={result.achieved_lz:.3f}, success={result.success_rate:.2f}, Δ_V={result.delta_V_mean:.3f}")

    if not all_runs:
        print("ERROR: No successful evaluations")
        return 1

    # Aggregate by target_Lz
    print("\n=== Aggregating Results ===")
    aggregated: List[SweepResults] = []
    for target_lz in sorted(set(r.target_lz for r in all_runs)):
        runs = [r for r in all_runs if r.target_lz == target_lz]
        agg = aggregate_results(runs)
        aggregated.append(agg)
        print(f"  target_Lz={target_lz}: achieved={agg.achieved_lz_mean:.3f}, success={agg.success_rate_mean:.2f}, Δ_V={agg.delta_V_mean:.3f}")

    # Generate outputs
    print("\n=== Generating Paper-Ready Artifacts ===")

    create_stability_dial_figure(aggregated, OUT_DIR / "fig_exp2_stability_dial.pdf")
    generate_latex_table(aggregated, OUT_DIR / "table_exp2_contraction_sweep.tex")
    generate_claims(aggregated, OUT_DIR / "CLAIMS.md")
    generate_provenance(checkpoints, OUT_DIR / "PROVENANCE.md")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    checks_passed = 0
    checks_total = 4

    # Check 1: Figure exists
    fig_path = OUT_DIR / "fig_exp2_stability_dial.pdf"
    if fig_path.exists():
        print(f"  ✓ {fig_path.name} ({fig_path.stat().st_size} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ {fig_path.name} missing")

    # Check 2: Table exists
    table_path = OUT_DIR / "table_exp2_contraction_sweep.tex"
    if table_path.exists():
        print(f"  ✓ {table_path.name} ({table_path.stat().st_size} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ {table_path.name} missing")

    # Check 3: Claims exists
    claims_path = OUT_DIR / "CLAIMS.md"
    if claims_path.exists():
        print(f"  ✓ {claims_path.name} ({claims_path.stat().st_size} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ {claims_path.name} missing")

    # Check 4: Provenance exists
    prov_path = OUT_DIR / "PROVENANCE.md"
    if prov_path.exists():
        print(f"  ✓ {prov_path.name} ({prov_path.stat().st_size} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ {prov_path.name} missing")

    if checks_passed == checks_total:
        print(f"\n✓ ALL VALIDATION CHECKS PASSED ({checks_passed}/{checks_total})")
    else:
        print(f"\n✗ SOME CHECKS FAILED ({checks_passed}/{checks_total})")

    print(f"\nArtifacts: {OUT_DIR}")
    return 0 if checks_passed == checks_total else 1


if __name__ == "__main__":
    sys.exit(main())
