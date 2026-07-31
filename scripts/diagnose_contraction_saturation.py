#!/usr/bin/env python3
"""
Diagnose contraction saturation in Exp2 checkpoints.

This script computes:
1. L_hat_preproj: Lipschitz constant WITHOUT projection
2. L_hat_postproj: Lipschitz constant WITH projection
3. projection_active_rate: fraction where ||f(z)|| > R
4. Norm distributions: ||z||, ||f(z)||, ||ΠR(f(z))||, (||f(z)||-R)+
5. clamp_activity: how often spectral norm clamping triggers

Output:
- results/paper_ready/exp2/DIAGNOSTICS_saturation.json
- results/paper_ready/exp2/DIAGNOSTICS_saturation.md
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints/exp2_contraction_sweep"
BATCH_DIR = PROJECT_ROOT / "artifacts/eval_batches"
OUT_DIR = PROJECT_ROOT / "results/paper_ready/exp2"

# Sweep configuration from Exp2
LZ_TARGETS = [0.9, 0.95, 0.99, 0.999]
SEEDS = [41, 42, 43]

# Eval-time radii to test (without retraining)
EVAL_RADII = [10.0, 100.0, 1000.0, 0.0]  # 0.0 = disabled

# Lipschitz estimation params
NUM_SAMPLES = 64
NUM_PERTURBATIONS = 8
EPS = 1e-3


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NormStats:
    mean: float
    median: float
    p95: float
    max: float
    min: float


@dataclass
class DiagnosticResult:
    target_lz: float
    seed: int
    eval_radius: float

    # Lipschitz estimates
    L_hat_preproj_mean: float
    L_hat_preproj_std: float
    L_hat_postproj_mean: float
    L_hat_postproj_std: float

    # Projection stats
    projection_active_rate: float
    distance_beyond_radius_mean: float
    distance_beyond_radius_p95: float

    # Norm distributions
    z_norm: NormStats
    fz_preproj_norm: NormStats
    fz_postproj_norm: NormStats

    # Clamp activity (if available)
    clamp_activity_rate: float


# =============================================================================
# Model Loading
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


def load_model(
    checkpoint_path: Path, device: str = "cpu"
) -> Tuple[TinyRecursiveReasoningModel_ACTV1, Dict]:
    """Load model from checkpoint."""
    from scripts.eval_unroll_sensitivity import load_model_for_eval
    model, config = load_model_for_eval(str(checkpoint_path), device)
    return model, config


def load_batch(batch_path: Path) -> List[Any]:
    """Load evaluation batch."""
    from scripts.eval_unroll_sensitivity import load_batch as _load_batch
    states, _ = _load_batch(str(batch_path))
    return states


# =============================================================================
# Projection Utilities
# =============================================================================

def project_to_ball(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Project z to ball of given radius."""
    if radius <= 0:
        return z
    z_norm = z.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(radius / z_norm, max=1.0)
    return z * scale


def compute_norm(z: torch.Tensor) -> float:
    """Compute L2 norm of z."""
    return z.norm(p=2).item()


def compute_norm_per_sample(z: torch.Tensor) -> torch.Tensor:
    """Compute per-sample L2 norm."""
    return z.norm(p=2, dim=(1, 2))


# =============================================================================
# Lipschitz Estimation with Projection Control
# =============================================================================

def apply_latent_step_with_projection_control(
    model: TinyRecursiveReasoningModel_ACTV1,
    z: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    x: Dict[str, torch.Tensor],
    y: torch.Tensor,
    apply_projection: bool,
    projection_radius: float,
) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, torch.Tensor]:
    """
    Apply one latent step with explicit control over projection.

    Returns:
        - z_next: next latent state (with or without projection)
        - fz_preproj_norm: norm of f(z) BEFORE projection
        - fz_postproj_norm: norm of f(z) AFTER projection
    """
    # Build context
    batch = model._standardize_latent_batch(x, y)
    context = model._resolve_latent_context(batch)
    input_embeds = context["input_embeddings_with_plan"]
    seq_info = context["seq_info"]

    # Get the inner model
    inner = model.inner

    # Apply L-level updates (same as latent_step but without projection)
    z_H, z_L = z.z_H, z.z_L
    for _ in range(inner.config.L_cycles):
        z_L = inner.L_level(z_L, z_H + input_embeds, **seq_info)
    z_H = inner.L_level(z_H, z_L, **seq_info)

    # Compute pre-projection norms
    fz_preproj_H_norm = compute_norm_per_sample(z_H)
    fz_preproj_L_norm = compute_norm_per_sample(z_L)
    fz_preproj_norm = torch.sqrt(fz_preproj_H_norm**2 + fz_preproj_L_norm**2)

    # Apply projection if requested
    if apply_projection and projection_radius > 0:
        z_H_proj = project_to_ball(z_H, projection_radius)
        z_L_proj = project_to_ball(z_L, projection_radius)
    else:
        z_H_proj = z_H
        z_L_proj = z_L

    # Compute post-projection norms
    fz_postproj_H_norm = compute_norm_per_sample(z_H_proj)
    fz_postproj_L_norm = compute_norm_per_sample(z_L_proj)
    fz_postproj_norm = torch.sqrt(fz_postproj_H_norm**2 + fz_postproj_L_norm**2)

    z_next = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H_proj, z_L=z_L_proj)

    return z_next, fz_preproj_norm, fz_postproj_norm


def estimate_lipschitz_with_projection_control(
    model: TinyRecursiveReasoningModel_ACTV1,
    states: List[Any],
    n_train: int,
    device: str,
    eval_radius: float,
    num_samples: int = NUM_SAMPLES,
    num_perturbations: int = NUM_PERTURBATIONS,
    eps: float = EPS,
) -> Dict[str, Any]:
    """
    Estimate Lipschitz constants with projection control.

    Returns dict with:
    - L_hat_preproj_samples: list of per-perturbation L estimates without projection
    - L_hat_postproj_samples: list of per-perturbation L estimates with projection
    - projection_active_samples: list of booleans indicating if projection was active
    - z_norms, fz_preproj_norms, fz_postproj_norms: norm samples
    - distance_beyond_radius: (||f(z)||-R)+ samples
    """
    rng = np.random.default_rng(42)

    if len(states) > num_samples:
        indices = rng.choice(len(states), size=num_samples, replace=False)
        sample_states = [states[i] for i in indices]
    else:
        sample_states = states

    L_preproj_samples = []
    L_postproj_samples = []
    projection_active_samples = []
    z_norms = []
    fz_preproj_norms = []
    fz_postproj_norms = []
    distance_beyond_radius = []

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

                # Record z norm
                z_norm = torch.sqrt(
                    compute_norm_per_sample(z_H)**2 +
                    compute_norm_per_sample(z_L)**2
                ).item()
                z_norms.append(z_norm)

                # Apply one step WITHOUT projection to get f(z)
                z_next_no_proj, fz_pre_norm, _ = apply_latent_step_with_projection_control(
                    model, z_base, x, y,
                    apply_projection=False,
                    projection_radius=0,
                )
                fz_preproj_norms.append(fz_pre_norm.item())

                # Apply one step WITH projection
                z_next_with_proj, _, fz_post_norm = apply_latent_step_with_projection_control(
                    model, z_base, x, y,
                    apply_projection=True,
                    projection_radius=eval_radius,
                )
                fz_postproj_norms.append(fz_post_norm.item())

                # Check if projection is active
                is_active = fz_pre_norm.item() > eval_radius if eval_radius > 0 else False
                projection_active_samples.append(is_active)

                # Distance beyond radius
                if eval_radius > 0:
                    dist = max(0, fz_pre_norm.item() - eval_radius)
                else:
                    dist = 0
                distance_beyond_radius.append(dist)

                # Estimate Lipschitz via perturbations
                for _ in range(num_perturbations):
                    # Create perturbation
                    noise_H = torch.randn_like(z_H)
                    noise_L = torch.randn_like(z_L)
                    noise_norm = torch.sqrt(
                        (noise_H**2).sum() + (noise_L**2).sum()
                    ).clamp(min=1e-12)
                    noise_H = noise_H * (eps / noise_norm)
                    noise_L = noise_L * (eps / noise_norm)

                    z_pert = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                        z_H=z_H + noise_H,
                        z_L=z_L + noise_L,
                    )

                    # Apply step WITHOUT projection
                    z_next_pert_no_proj, _, _ = apply_latent_step_with_projection_control(
                        model, z_pert, x, y,
                        apply_projection=False,
                        projection_radius=0,
                    )

                    # L_preproj = ||f(z+δ) - f(z)|| / ||δ||
                    diff_H_pre = z_next_pert_no_proj.z_H - z_next_no_proj.z_H
                    diff_L_pre = z_next_pert_no_proj.z_L - z_next_no_proj.z_L
                    diff_norm_pre = torch.sqrt(
                        (diff_H_pre**2).sum() + (diff_L_pre**2).sum()
                    ).item()
                    L_preproj = diff_norm_pre / eps
                    L_preproj_samples.append(L_preproj)

                    # Apply step WITH projection
                    z_next_pert_with_proj, _, _ = apply_latent_step_with_projection_control(
                        model, z_pert, x, y,
                        apply_projection=True,
                        projection_radius=eval_radius,
                    )

                    # L_postproj = ||ΠR(f(z+δ)) - ΠR(f(z))|| / ||δ||
                    diff_H_post = z_next_pert_with_proj.z_H - z_next_with_proj.z_H
                    diff_L_post = z_next_pert_with_proj.z_L - z_next_with_proj.z_L
                    diff_norm_post = torch.sqrt(
                        (diff_H_post**2).sum() + (diff_L_post**2).sum()
                    ).item()
                    L_postproj = diff_norm_post / eps
                    L_postproj_samples.append(L_postproj)

            except Exception as e:
                print(f"[Diagnostic] Error: {e}")
                continue

    return {
        "L_preproj_samples": L_preproj_samples,
        "L_postproj_samples": L_postproj_samples,
        "projection_active_samples": projection_active_samples,
        "z_norms": z_norms,
        "fz_preproj_norms": fz_preproj_norms,
        "fz_postproj_norms": fz_postproj_norms,
        "distance_beyond_radius": distance_beyond_radius,
    }


def compute_norm_stats(values: List[float]) -> NormStats:
    """Compute summary statistics for norm values."""
    if not values:
        return NormStats(0, 0, 0, 0, 0)
    arr = np.array(values)
    return NormStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        max=float(np.max(arr)),
        min=float(np.min(arr)),
    )


# =============================================================================
# Clamp Activity Detection
# =============================================================================

def estimate_clamp_activity(
    model: nn.Module,
    states: List[Any],
    n_train: int,
    device: str,
    num_samples: int = 32,
) -> float:
    """
    Estimate how often spectral norm clamping is active.

    This checks if the Lipschitz scale factors are < 1.0 (indicating clamping).
    """
    # Check if model has Lipschitz scale parameters
    lipschitz_scales = []
    for name, param in model.named_parameters():
        if "lipschitz_scale" in name:
            lipschitz_scales.append((name, param))

    if not lipschitz_scales:
        return 0.0  # No clamping mechanism

    # Check how many are < 1.0 (clamped)
    clamped_count = 0
    total_count = len(lipschitz_scales)

    for name, scale in lipschitz_scales:
        if scale.item() < 0.999:  # Allow small numerical tolerance
            clamped_count += 1

    return clamped_count / total_count if total_count > 0 else 0.0


# =============================================================================
# Main Diagnostic
# =============================================================================

def run_diagnostics(device: str = "cuda") -> List[DiagnosticResult]:
    """Run full diagnostics on all Exp2 checkpoints."""

    # Load batch
    batch_path = BATCH_DIR / "b0.pt"
    if not batch_path.exists():
        print(f"ERROR: Batch not found at {batch_path}")
        return []

    states = load_batch(batch_path)
    print(f"[Diagnostic] Loaded {len(states)} states from B0")

    results = []

    for target_lz in LZ_TARGETS:
        for seed in SEEDS:
            ckpt_path = find_checkpoint(target_lz, seed)
            if not ckpt_path:
                print(f"[Diagnostic] Missing: target_Lz={target_lz}, seed={seed}")
                continue

            print(f"\n[Diagnostic] target_Lz={target_lz}, seed={seed}")
            print(f"  Checkpoint: {ckpt_path}")

            model, config = load_model(ckpt_path, device)
            n_train = config.get("inner_unroll_n", 4)

            # Check clamp activity
            clamp_rate = estimate_clamp_activity(model, states, n_train, device)
            print(f"  Clamp activity: {clamp_rate:.2%}")

            for eval_radius in EVAL_RADII:
                print(f"  Eval radius: {eval_radius}")

                # Run Lipschitz estimation
                diag = estimate_lipschitz_with_projection_control(
                    model, states, n_train, device, eval_radius
                )

                # Compute statistics
                L_pre_mean = np.mean(diag["L_preproj_samples"]) if diag["L_preproj_samples"] else 0
                L_pre_std = np.std(diag["L_preproj_samples"]) if diag["L_preproj_samples"] else 0
                L_post_mean = np.mean(diag["L_postproj_samples"]) if diag["L_postproj_samples"] else 0
                L_post_std = np.std(diag["L_postproj_samples"]) if diag["L_postproj_samples"] else 0

                proj_rate = np.mean(diag["projection_active_samples"]) if diag["projection_active_samples"] else 0

                dist_beyond = diag["distance_beyond_radius"]
                dist_mean = np.mean(dist_beyond) if dist_beyond else 0
                dist_p95 = np.percentile(dist_beyond, 95) if dist_beyond else 0

                result = DiagnosticResult(
                    target_lz=target_lz,
                    seed=seed,
                    eval_radius=eval_radius,
                    L_hat_preproj_mean=L_pre_mean,
                    L_hat_preproj_std=L_pre_std,
                    L_hat_postproj_mean=L_post_mean,
                    L_hat_postproj_std=L_post_std,
                    projection_active_rate=proj_rate,
                    distance_beyond_radius_mean=dist_mean,
                    distance_beyond_radius_p95=dist_p95,
                    z_norm=compute_norm_stats(diag["z_norms"]),
                    fz_preproj_norm=compute_norm_stats(diag["fz_preproj_norms"]),
                    fz_postproj_norm=compute_norm_stats(diag["fz_postproj_norms"]),
                    clamp_activity_rate=clamp_rate,
                )

                results.append(result)

                print(f"    L_preproj: {L_pre_mean:.3f}±{L_pre_std:.3f}")
                print(f"    L_postproj: {L_post_mean:.3f}±{L_post_std:.3f}")
                print(f"    Projection active: {proj_rate:.1%}")

    return results


def generate_json_output(results: List[DiagnosticResult], out_path: Path):
    """Save results as JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\n[Output] Saved: {out_path}")


def generate_markdown_output(results: List[DiagnosticResult], out_path: Path):
    """Generate human-readable markdown summary."""

    # Group by target_lz and eval_radius
    by_target = {}
    for r in results:
        key = (r.target_lz, r.eval_radius)
        if key not in by_target:
            by_target[key] = []
        by_target[key].append(r)

    content = f"""# Exp2 Contraction Saturation Diagnostics

**Generated**: {datetime.now().isoformat()}

## Summary

This diagnostic investigates WHY achieved_Lz saturates at ~0.23 across all target_Lz values.

## Key Metrics

| target_Lz | R | L_preproj | L_postproj | Proj Active | Clamp Active |
|-----------|---|-----------|------------|-------------|--------------|
"""

    for (target_lz, eval_radius), group in sorted(by_target.items()):
        L_pre_vals = [r.L_hat_preproj_mean for r in group]
        L_post_vals = [r.L_hat_postproj_mean for r in group]
        proj_rates = [r.projection_active_rate for r in group]
        clamp_rates = [r.clamp_activity_rate for r in group]

        L_pre_mean = np.mean(L_pre_vals)
        L_post_mean = np.mean(L_post_vals)
        proj_mean = np.mean(proj_rates)
        clamp_mean = np.mean(clamp_rates)

        R_str = "∞" if eval_radius == 0 else f"{eval_radius:.0f}"
        content += f"| {target_lz} | {R_str} | {L_pre_mean:.3f} | {L_post_mean:.3f} | {proj_mean:.1%} | {clamp_mean:.1%} |\n"

    # Decision gate analysis
    content += """
## Decision Gate Analysis

"""

    # Check if L_preproj varies but L_postproj saturates
    R10_results = [r for r in results if r.eval_radius == 10.0]
    R_disabled = [r for r in results if r.eval_radius == 0.0]

    if R10_results and R_disabled:
        L_pre_by_target = {}
        L_post_by_target = {}
        for r in R10_results:
            if r.target_lz not in L_pre_by_target:
                L_pre_by_target[r.target_lz] = []
                L_post_by_target[r.target_lz] = []
            L_pre_by_target[r.target_lz].append(r.L_hat_preproj_mean)
            L_post_by_target[r.target_lz].append(r.L_hat_postproj_mean)

        L_pre_means = [np.mean(v) for v in L_pre_by_target.values()]
        L_post_means = [np.mean(v) for v in L_post_by_target.values()]

        L_pre_range = max(L_pre_means) - min(L_pre_means)
        L_post_range = max(L_post_means) - min(L_post_means)

        proj_rate_mean = np.mean([r.projection_active_rate for r in R10_results])

        content += f"""### R=10 Analysis

- L_preproj range across targets: {L_pre_range:.3f}
- L_postproj range across targets: {L_post_range:.3f}
- Mean projection active rate: {proj_rate_mean:.1%}

"""

        if L_pre_range > 0.05 and L_post_range < 0.05 and proj_rate_mean > 0.8:
            content += "**DIAGNOSIS: Projection dominates** - L_preproj varies but L_postproj saturates with high projection rate.\n\n"
            content += "**RECOMMENDATION**: Increase R or disable projection for dial experiments.\n"
        elif L_pre_range < 0.05 and L_post_range < 0.05:
            clamp_mean = np.mean([r.clamp_activity_rate for r in R10_results])
            if clamp_mean > 0.5:
                content += "**DIAGNOSIS: Clamp/enforcement saturates** - Both L values saturate with high clamp activity.\n\n"
                content += "**RECOMMENDATION**: Add explicit z→z scaling knob.\n"
            else:
                content += "**DIAGNOSIS: Architecture naturally contracts** - Both L values saturate without external enforcement dominating.\n\n"
                content += "**RECOMMENDATION**: The dial may be ineffective; document as negative result.\n"
        else:
            content += "**DIAGNOSIS: Inconclusive** - Neither projection nor clamping fully explains saturation.\n"

    # Norm distributions
    content += """
## Norm Distributions (R=10)

| target_Lz | ||z|| mean | ||f(z)|| pre | ||f(z)|| post | Beyond R |
|-----------|------------|--------------|---------------|----------|
"""

    for (target_lz, eval_radius), group in sorted(by_target.items()):
        if eval_radius != 10.0:
            continue
        z_norms = [r.z_norm.mean for r in group]
        fz_pre = [r.fz_preproj_norm.mean for r in group]
        fz_post = [r.fz_postproj_norm.mean for r in group]
        beyond = [r.distance_beyond_radius_mean for r in group]

        content += f"| {target_lz} | {np.mean(z_norms):.2f} | {np.mean(fz_pre):.2f} | {np.mean(fz_post):.2f} | {np.mean(beyond):.2f} |\n"

    out_path.write_text(content)
    print(f"[Output] Saved: {out_path}")


def main():
    print("=" * 60)
    print("EXP2 CONTRACTION SATURATION DIAGNOSTICS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run diagnostics
    results = run_diagnostics(device)

    if not results:
        print("ERROR: No diagnostic results generated")
        return 1

    # Generate outputs
    generate_json_output(results, OUT_DIR / "DIAGNOSTICS_saturation.json")
    generate_markdown_output(results, OUT_DIR / "DIAGNOSTICS_saturation.md")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
