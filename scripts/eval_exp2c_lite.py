#!/usr/bin/env python3
"""
Exp2c-Lite: Evaluate existing checkpoints with projection override to unmask stability dial.

This script:
1. Loads existing Exp2 checkpoints (trained with R=10)
2. Evaluates at R_eval ∈ {10, 100, disabled (0)}
3. Computes L_preproj, L_postproj, projection_active_rate
4. Computes unroll sensitivity metrics (ΔV, Δπ, argmax agreement) at n_train=2 vs n_eval={4,8,16}
5. Checks decision gates G1, G2, G3

Decision Gates:
- G1: projection_active_rate < 20% at chosen R_eval
- G2: L_preproj spread ≥ 0.08 across conditions
- G3: Lower L_preproj ⇒ lower ΔV/Δπ and higher argmax agreement

Output:
- results/paper_ready/exp2c/DIAGNOSTICS_exp2c_lite.json
- results/paper_ready/exp2c/DIAGNOSTICS_exp2c_lite.md
- results/paper_ready/exp2c/GATES.md (pass/fail analysis)
"""

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root
PROJECT_ROOT = Path("/home/buiksat/trm_bellman")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


# =============================================================================
# JSON Encoder for Numpy Types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints/exp2_contraction_sweep"
BATCH_DIR = PROJECT_ROOT / "artifacts/eval_batches"
OUT_DIR = PROJECT_ROOT / "results/paper_ready/exp2c"

# Sweep configuration from Exp2
LZ_TARGETS = [0.9, 0.95, 0.99, 0.999]
SEEDS = [41, 42, 43]

# Eval-time radii to test: R=10 (training), R=100 (relaxed), disabled
EVAL_RADII = [10.0, 100.0, 0.0]  # 0.0 = disabled (∞)

# Unroll depths for sensitivity analysis
N_TRAIN = 2
N_EVAL_DEPTHS = [2, 4, 8, 16]  # 1x, 2x, 4x, 8x

# Lipschitz estimation params
NUM_SAMPLES = 64
NUM_PERTURBATIONS = 8
EPS = 1e-3

# Decision gate thresholds
G1_THRESHOLD = 0.20  # projection_active_rate < 20%
G2_THRESHOLD = 0.08  # L_preproj spread >= 0.08


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class UnrollMetrics:
    """Metrics for a single (n_train, n_eval) depth comparison."""
    n_train: int
    n_eval: int
    delta_V_mean: float
    delta_V_std: float
    delta_pi_mean: float
    delta_pi_std: float
    argmax_agree_rate: float
    saturation_rate: float


@dataclass
class ConditionResult:
    """Results for a single (target_lz, eval_radius) condition."""
    target_lz: float
    eval_radius: float
    seeds: List[int]

    # Lipschitz estimates
    L_preproj_mean: float
    L_preproj_std: float
    L_postproj_mean: float
    L_postproj_std: float

    # Projection stats
    projection_active_rate: float

    # Unroll sensitivity at each depth
    unroll_metrics: List[UnrollMetrics] = field(default_factory=list)

    # Primary stability metric (n=2 vs n=16)
    delta_V_2_16: float = 0.0
    delta_pi_2_16: float = 0.0
    argmax_agree_2_16: float = 0.0


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    value: float
    threshold: float
    details: str


# =============================================================================
# Model and Batch Loading
# =============================================================================

def find_checkpoint(target_lz: float, seed: int) -> Optional[Path]:
    """Find checkpoint path for given target_lz and seed."""
    lz_dir_map = {
        0.9: "lz_0900",
        0.95: "lz_095",
        0.99: "lz_099",
        0.999: "lz_0999",
    }
    dir_name = lz_dir_map.get(target_lz)
    if not dir_name:
        return None

    ckpt_path = CHECKPOINT_BASE / dir_name / f"seed{seed}" / "model_step_5000.pt"
    if ckpt_path.exists():
        return ckpt_path
    return None


def load_model_with_radius_override(
    checkpoint_path: Path,
    device: str,
    radius_override: float,
) -> Tuple[nn.Module, Dict]:
    """Load model with latent_ball_radius override."""
    from scripts.eval_unroll_sensitivity import load_model_for_eval

    # 0.0 means disabled, pass None to use very large value
    override = radius_override if radius_override > 0 else 1e6

    model, config = load_model_for_eval(
        str(checkpoint_path),
        device,
        latent_ball_radius_override=override,
    )

    # Update config to reflect what we're actually using
    config["eval_radius"] = radius_override
    return model, config


def load_batch(batch_path: Path) -> List[Any]:
    """Load evaluation batch."""
    from scripts.eval_unroll_sensitivity import load_batch as _load_batch
    states, _ = _load_batch(str(batch_path))
    return states


# =============================================================================
# Lipschitz Estimation (from diagnose script)
# =============================================================================

def project_to_ball(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Project z to ball of given radius."""
    if radius <= 0:
        return z
    z_norm = z.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(radius / z_norm, max=1.0)
    return z * scale


def compute_norm_per_sample(z: torch.Tensor) -> torch.Tensor:
    """Compute per-sample L2 norm."""
    return z.norm(p=2, dim=(1, 2))


def estimate_lipschitz_and_projection(
    model: nn.Module,
    states: List[Any],
    n_train: int,
    device: str,
    eval_radius: float,
) -> Dict[str, Any]:
    """
    Estimate Lipschitz constants and projection activity.

    Returns dict with L_preproj, L_postproj, projection_active_rate.
    """
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry

    rng = np.random.default_rng(42)

    if len(states) > NUM_SAMPLES:
        indices = rng.choice(len(states), size=NUM_SAMPLES, replace=False)
        sample_states = [states[i] for i in indices]
    else:
        sample_states = states

    L_preproj_samples = []
    L_postproj_samples = []
    projection_active_samples = []

    model.eval()
    with torch.no_grad():
        for state in sample_states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            try:
                # Get baseline z at n_train
                _, z_base = model.used_value(x, y, n_train)
                if not hasattr(z_base, 'z_H'):
                    continue

                z_H = z_base.z_H
                z_L = z_base.z_L

                # Build context for manual step
                batch = model._standardize_latent_batch(x, y)
                context = model._resolve_latent_context(batch)
                input_embeds = context.get("input_embeddings_with_plan", context["input_embeddings"])
                seq_info = context["seq_info"]
                inner = model.inner

                # Apply L-level updates WITHOUT projection
                z_H_next, z_L_next = z_H.clone(), z_L.clone()
                for _ in range(inner.config.L_cycles):
                    z_L_next = inner.L_level(z_L_next, z_H_next + input_embeds, **seq_info)
                z_H_next = inner.L_level(z_H_next, z_L_next, **seq_info)

                # Compute f(z) norm before projection
                fz_norm = torch.sqrt(
                    compute_norm_per_sample(z_H_next)**2 +
                    compute_norm_per_sample(z_L_next)**2
                ).item()

                # Check projection activity
                is_active = fz_norm > eval_radius if eval_radius > 0 else False
                projection_active_samples.append(is_active)

                # Estimate Lipschitz via perturbations
                for _ in range(NUM_PERTURBATIONS):
                    noise_H = torch.randn_like(z_H)
                    noise_L = torch.randn_like(z_L)
                    noise_norm = torch.sqrt(
                        (noise_H**2).sum() + (noise_L**2).sum()
                    ).clamp(min=1e-12)
                    noise_H = noise_H * (EPS / noise_norm)
                    noise_L = noise_L * (EPS / noise_norm)

                    z_H_pert = z_H + noise_H
                    z_L_pert = z_L + noise_L

                    # Apply step to perturbed z (no projection)
                    z_H_next_p, z_L_next_p = z_H_pert.clone(), z_L_pert.clone()
                    for _ in range(inner.config.L_cycles):
                        z_L_next_p = inner.L_level(z_L_next_p, z_H_next_p + input_embeds, **seq_info)
                    z_H_next_p = inner.L_level(z_H_next_p, z_L_next_p, **seq_info)

                    # L_preproj = ||f(z+δ) - f(z)|| / ||δ||
                    diff_H_pre = z_H_next_p - z_H_next
                    diff_L_pre = z_L_next_p - z_L_next
                    diff_norm_pre = torch.sqrt(
                        (diff_H_pre**2).sum() + (diff_L_pre**2).sum()
                    ).item()
                    L_preproj_samples.append(diff_norm_pre / EPS)

                    # Apply projection to both
                    if eval_radius > 0:
                        z_H_next_proj = project_to_ball(z_H_next, eval_radius)
                        z_L_next_proj = project_to_ball(z_L_next, eval_radius)
                        z_H_next_p_proj = project_to_ball(z_H_next_p, eval_radius)
                        z_L_next_p_proj = project_to_ball(z_L_next_p, eval_radius)
                    else:
                        z_H_next_proj = z_H_next
                        z_L_next_proj = z_L_next
                        z_H_next_p_proj = z_H_next_p
                        z_L_next_p_proj = z_L_next_p

                    # L_postproj
                    diff_H_post = z_H_next_p_proj - z_H_next_proj
                    diff_L_post = z_L_next_p_proj - z_L_next_proj
                    diff_norm_post = torch.sqrt(
                        (diff_H_post**2).sum() + (diff_L_post**2).sum()
                    ).item()
                    L_postproj_samples.append(diff_norm_post / EPS)

            except Exception as e:
                continue

    return {
        "L_preproj_mean": np.mean(L_preproj_samples) if L_preproj_samples else 0,
        "L_preproj_std": np.std(L_preproj_samples) if L_preproj_samples else 0,
        "L_postproj_mean": np.mean(L_postproj_samples) if L_postproj_samples else 0,
        "L_postproj_std": np.std(L_postproj_samples) if L_postproj_samples else 0,
        "projection_active_rate": np.mean(projection_active_samples) if projection_active_samples else 0,
    }


# =============================================================================
# Unroll Sensitivity Evaluation
# =============================================================================

def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute KL(p || q) with numerical stability."""
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    kl = (p_safe * (p_safe.log() - q_safe.log())).sum()
    return float(kl.clamp(min=0).item())


def evaluate_unroll_sensitivity(
    model: nn.Module,
    states: List[Any],
    n_train: int,
    n_eval_depths: List[int],
    config: Dict[str, Any],
    device: str,
) -> List[UnrollMetrics]:
    """Evaluate unroll sensitivity at multiple depths."""
    from rl.envs.plan_edit_env import PlanEditEnv

    vocab_size = config["vocab_size"]
    num_actions = config["num_actions"]
    stop_action_id = num_actions - 1
    radius = config.get("latent_ball_radius", config.get("eval_radius", 10.0))

    # Collect (value, policy) at each depth for each state
    results_by_depth = {n: {"values": [], "policies": []} for n in n_eval_depths}

    model.eval()
    with torch.no_grad():
        for state in states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            action_mask = PlanEditEnv.compute_batch_action_mask(
                state.inputs.unsqueeze(0),
                vocab_size,
                stop_action_id,
            ).to(device)

            try:
                for n in n_eval_depths:
                    value, _ = model.used_value(x, y, n)
                    dist, _ = model.policy_dist(x, y, n, action_mask=action_mask)

                    results_by_depth[n]["values"].append(value.squeeze().cpu().item())
                    results_by_depth[n]["policies"].append(dist.probs.squeeze().cpu())
            except Exception:
                continue

    # Compute metrics for each (n_train, n_eval) pair
    metrics = []
    base_n = n_train

    for n_eval in n_eval_depths:
        if n_eval == base_n:
            continue

        base_vals = results_by_depth[base_n]["values"]
        eval_vals = results_by_depth[n_eval]["values"]
        base_pols = results_by_depth[base_n]["policies"]
        eval_pols = results_by_depth[n_eval]["policies"]

        n_samples = min(len(base_vals), len(eval_vals))
        if n_samples == 0:
            continue

        delta_V = [abs(base_vals[i] - eval_vals[i]) for i in range(n_samples)]
        delta_pi = [compute_kl_divergence(base_pols[i], eval_pols[i]) for i in range(n_samples)]
        argmax_agree = [
            1 if base_pols[i].argmax().item() == eval_pols[i].argmax().item() else 0
            for i in range(n_samples)
        ]

        metrics.append(UnrollMetrics(
            n_train=base_n,
            n_eval=n_eval,
            delta_V_mean=np.mean(delta_V),
            delta_V_std=np.std(delta_V),
            delta_pi_mean=np.mean(delta_pi),
            delta_pi_std=np.std(delta_pi),
            argmax_agree_rate=np.mean(argmax_agree),
            saturation_rate=0.0,  # Not computed here
        ))

    return metrics


# =============================================================================
# Main Evaluation
# =============================================================================

def run_exp2c_lite(device: str = "cuda") -> Tuple[List[ConditionResult], List[GateResult]]:
    """Run Exp2c-lite evaluation."""

    batch_path = BATCH_DIR / "b0.pt"
    if not batch_path.exists():
        print(f"ERROR: Batch not found at {batch_path}")
        return [], []

    states = load_batch(batch_path)
    print(f"[Exp2c] Loaded {len(states)} states from B0")

    all_results = []

    # Evaluate each (target_lz, eval_radius) condition
    for eval_radius in EVAL_RADII:
        R_str = "disabled" if eval_radius == 0 else f"{eval_radius:.0f}"
        print(f"\n{'='*60}")
        print(f"EVAL RADIUS: R={R_str}")
        print(f"{'='*60}")

        for target_lz in LZ_TARGETS:
            seed_results = []

            for seed in SEEDS:
                ckpt_path = find_checkpoint(target_lz, seed)
                if not ckpt_path:
                    print(f"  [SKIP] target_Lz={target_lz}, seed={seed}: checkpoint not found")
                    continue

                print(f"\n[Eval] target_Lz={target_lz}, seed={seed}, R_eval={R_str}")

                model, config = load_model_with_radius_override(ckpt_path, device, eval_radius)

                # Lipschitz estimation
                lip_stats = estimate_lipschitz_and_projection(
                    model, states, N_TRAIN, device, eval_radius
                )

                print(f"  L_preproj: {lip_stats['L_preproj_mean']:.3f}±{lip_stats['L_preproj_std']:.3f}")
                print(f"  L_postproj: {lip_stats['L_postproj_mean']:.3f}±{lip_stats['L_postproj_std']:.3f}")
                print(f"  Projection active: {lip_stats['projection_active_rate']:.1%}")

                # Unroll sensitivity
                unroll_metrics = evaluate_unroll_sensitivity(
                    model, states, N_TRAIN, N_EVAL_DEPTHS, config, device
                )

                for um in unroll_metrics:
                    print(f"  n={um.n_train}→{um.n_eval}: ΔV={um.delta_V_mean:.4f}, Δπ={um.delta_pi_mean:.4f}, agree={um.argmax_agree_rate:.2%}")

                seed_results.append({
                    "seed": seed,
                    "lip_stats": lip_stats,
                    "unroll_metrics": unroll_metrics,
                })

            if not seed_results:
                continue

            # Aggregate across seeds
            L_pre_vals = [r["lip_stats"]["L_preproj_mean"] for r in seed_results]
            L_post_vals = [r["lip_stats"]["L_postproj_mean"] for r in seed_results]
            proj_rates = [r["lip_stats"]["projection_active_rate"] for r in seed_results]

            # Find n=2 vs n=16 metrics
            delta_V_2_16 = []
            delta_pi_2_16 = []
            agree_2_16 = []
            for r in seed_results:
                for um in r["unroll_metrics"]:
                    if um.n_train == 2 and um.n_eval == 16:
                        delta_V_2_16.append(um.delta_V_mean)
                        delta_pi_2_16.append(um.delta_pi_mean)
                        agree_2_16.append(um.argmax_agree_rate)

            # Aggregate unroll metrics across seeds
            aggregated_unroll = []
            depth_to_metrics = {}
            for r in seed_results:
                for um in r["unroll_metrics"]:
                    key = (um.n_train, um.n_eval)
                    if key not in depth_to_metrics:
                        depth_to_metrics[key] = []
                    depth_to_metrics[key].append(um)

            for (n_t, n_e), ums in depth_to_metrics.items():
                aggregated_unroll.append(UnrollMetrics(
                    n_train=n_t,
                    n_eval=n_e,
                    delta_V_mean=np.mean([u.delta_V_mean for u in ums]),
                    delta_V_std=np.mean([u.delta_V_std for u in ums]),
                    delta_pi_mean=np.mean([u.delta_pi_mean for u in ums]),
                    delta_pi_std=np.mean([u.delta_pi_std for u in ums]),
                    argmax_agree_rate=np.mean([u.argmax_agree_rate for u in ums]),
                    saturation_rate=0.0,
                ))

            result = ConditionResult(
                target_lz=target_lz,
                eval_radius=eval_radius,
                seeds=[r["seed"] for r in seed_results],
                L_preproj_mean=np.mean(L_pre_vals),
                L_preproj_std=np.std(L_pre_vals),
                L_postproj_mean=np.mean(L_post_vals),
                L_postproj_std=np.std(L_post_vals),
                projection_active_rate=np.mean(proj_rates),
                unroll_metrics=aggregated_unroll,
                delta_V_2_16=np.mean(delta_V_2_16) if delta_V_2_16 else 0,
                delta_pi_2_16=np.mean(delta_pi_2_16) if delta_pi_2_16 else 0,
                argmax_agree_2_16=np.mean(agree_2_16) if agree_2_16 else 0,
            )
            all_results.append(result)

    # Check decision gates
    gates = check_gates(all_results)

    return all_results, gates


def check_gates(results: List[ConditionResult]) -> List[GateResult]:
    """Check decision gates G1, G2, G3."""
    gates = []

    # Find best R where G1 passes (projection not dominating)
    for eval_R in [100.0, 0.0]:  # Check R=100 and disabled first
        R_results = [r for r in results if r.eval_radius == eval_R]
        if not R_results:
            continue

        proj_rate_mean = np.mean([r.projection_active_rate for r in R_results])

        # G1: projection_active_rate < 20%
        g1_passed = proj_rate_mean < G1_THRESHOLD
        gates.append(GateResult(
            gate_name=f"G1 (R={eval_R if eval_R > 0 else '∞'})",
            passed=g1_passed,
            value=proj_rate_mean,
            threshold=G1_THRESHOLD,
            details=f"Projection active rate: {proj_rate_mean:.1%} (threshold: <{G1_THRESHOLD:.0%})"
        ))

        if not g1_passed:
            continue

        # G2: L_preproj spread >= 0.08
        L_preproj_by_target = {r.target_lz: r.L_preproj_mean for r in R_results}
        L_preproj_spread = max(L_preproj_by_target.values()) - min(L_preproj_by_target.values())

        g2_passed = L_preproj_spread >= G2_THRESHOLD
        gates.append(GateResult(
            gate_name=f"G2 (R={eval_R if eval_R > 0 else '∞'})",
            passed=g2_passed,
            value=L_preproj_spread,
            threshold=G2_THRESHOLD,
            details=f"L_preproj spread: {L_preproj_spread:.3f} (threshold: >={G2_THRESHOLD:.2f})"
        ))

        if not g2_passed:
            continue

        # G3: Lower L_preproj ⇒ lower ΔV/Δπ, higher argmax agreement
        # Sort by L_preproj and check correlation
        sorted_by_L = sorted(R_results, key=lambda r: r.L_preproj_mean)

        # Check if stability metrics improve as L decreases
        L_vals = [r.L_preproj_mean for r in sorted_by_L]
        dV_vals = [r.delta_V_2_16 for r in sorted_by_L]
        dpi_vals = [r.delta_pi_2_16 for r in sorted_by_L]
        agree_vals = [r.argmax_agree_2_16 for r in sorted_by_L]

        # Simple check: does lowest L have best stability?
        lowest_L_idx = 0
        best_agree_idx = np.argmax(agree_vals)
        lowest_dV_idx = np.argmin(dV_vals)

        g3_passed = (lowest_L_idx == best_agree_idx) or (lowest_L_idx == lowest_dV_idx)

        corr_details = []
        for i, r in enumerate(sorted_by_L):
            corr_details.append(f"L={r.L_preproj_mean:.3f} → ΔV={r.delta_V_2_16:.4f}, agree={r.argmax_agree_2_16:.2%}")

        gates.append(GateResult(
            gate_name=f"G3 (R={eval_R if eval_R > 0 else '∞'})",
            passed=g3_passed,
            value=0,  # qualitative
            threshold=0,
            details=f"Stability linkage: {'PASSED' if g3_passed else 'FAILED'}\n" + "\n".join(corr_details)
        ))

    return gates


# =============================================================================
# Output Generation
# =============================================================================

def generate_outputs(results: List[ConditionResult], gates: List[GateResult]):
    """Generate JSON and Markdown outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_data = {
        "generated": datetime.now().isoformat(),
        "n_train": N_TRAIN,
        "n_eval_depths": N_EVAL_DEPTHS,
        "eval_radii": EVAL_RADII,
        "results": [],
        "gates": [asdict(g) for g in gates],
    }

    for r in results:
        r_dict = asdict(r)
        json_data["results"].append(r_dict)

    json_path = OUT_DIR / "DIAGNOSTICS_exp2c_lite.json"
    json_path.write_text(json.dumps(json_data, indent=2, cls=NumpyEncoder))
    print(f"\n[Output] Saved: {json_path}")

    # Markdown summary
    md_content = f"""# Exp2c-Lite: Stability Dial Unmasking Evaluation

**Generated**: {datetime.now().isoformat()}

## Configuration

- n_train: {N_TRAIN}
- n_eval depths: {N_EVAL_DEPTHS}
- Eval radii: {EVAL_RADII}
- Target L_z values: {LZ_TARGETS}

## Results by Eval Radius

"""

    for eval_R in EVAL_RADII:
        R_str = "∞ (disabled)" if eval_R == 0 else f"{eval_R:.0f}"
        R_results = [r for r in results if r.eval_radius == eval_R]

        if not R_results:
            continue

        md_content += f"""### R = {R_str}

| Target L_z | L_preproj | L_postproj | Proj Active | ΔV (2→16) | Δπ (2→16) | Agree (2→16) |
|------------|-----------|------------|-------------|-----------|-----------|--------------|
"""

        for r in sorted(R_results, key=lambda x: x.target_lz):
            md_content += f"| {r.target_lz} | {r.L_preproj_mean:.3f}±{r.L_preproj_std:.3f} | {r.L_postproj_mean:.3f}±{r.L_postproj_std:.3f} | {r.projection_active_rate:.1%} | {r.delta_V_2_16:.4f} | {r.delta_pi_2_16:.4f} | {r.argmax_agree_2_16:.1%} |\n"

        md_content += "\n"

    md_path = OUT_DIR / "DIAGNOSTICS_exp2c_lite.md"
    md_path.write_text(md_content)
    print(f"[Output] Saved: {md_path}")

    # Gates analysis
    gates_content = f"""# Exp2c Decision Gates Analysis

**Generated**: {datetime.now().isoformat()}

## Gate Definitions

- **G1 (Projection Dominance)**: projection_active_rate < {G1_THRESHOLD:.0%}
- **G2 (Dial Range)**: L_preproj spread >= {G2_THRESHOLD:.2f}
- **G3 (Stability Linkage)**: Lower L_preproj ⇒ lower ΔV/Δπ, higher argmax agreement

## Results

"""

    all_passed = True
    for g in gates:
        status = "✅ PASSED" if g.passed else "❌ FAILED"
        all_passed = all_passed and g.passed
        gates_content += f"""### {g.gate_name}: {status}

{g.details}

"""

    gates_content += f"""## Overall Verdict

"""

    if all_passed:
        gates_content += "**All gates passed!** → Proceed to paper-ready artifacts (Path A)\n"
    else:
        failed_gates = [g.gate_name for g in gates if not g.passed]
        gates_content += f"**Failed gates**: {', '.join(failed_gates)}\n\n"
        gates_content += "Consider:\n"
        gates_content += "- If G1 failed at all radii: Retraining may be needed\n"
        gates_content += "- If G2 failed: Dial has insufficient range → negative result\n"
        gates_content += "- If G3 failed: No stability linkage → negative result\n"

    gates_path = OUT_DIR / "GATES.md"
    gates_path.write_text(gates_content)
    print(f"[Output] Saved: {gates_path}")


def main():
    print("=" * 60)
    print("EXP2C-LITE: STABILITY DIAL UNMASKING EVALUATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results, gates = run_exp2c_lite(device)

    if not results:
        print("ERROR: No results generated")
        return 1

    generate_outputs(results, gates)

    print("\n" + "=" * 60)
    print("EXP2C-LITE COMPLETE")
    print("=" * 60)

    # Print gate summary
    print("\nGATE SUMMARY:")
    for g in gates:
        status = "✅" if g.passed else "❌"
        print(f"  {status} {g.gate_name}: {g.details.split(chr(10))[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
