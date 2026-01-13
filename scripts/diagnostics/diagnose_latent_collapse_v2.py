#!/usr/bin/env python3
"""
Latent Collapse Diagnostic V2: Isolates z→z contraction from value-head normalization.

This diagnostic creates a 2×2 factorial design:
  Axis 1: z→z contraction (zcon_OFF vs zcon_ON)
  Axis 2: value-head normalization (vhead_OFF vs vhead_ON)

Conditions:
  A: zcon_OFF + vhead_OFF
  B: zcon_ON  + vhead_OFF  (isolates z→z effect)
  C: zcon_OFF + vhead_ON
  D: zcon_ON  + vhead_ON   (matches "contraction ON" behavior)

Key improvements over v1:
  - Measures BOTH pre-projection and post-projection latents
  - Isolates z→z contraction from value head normalization
  - Reports full value stats (mean/std/min/max), not just variance

Usage:
    buck2 run //buiksat_trm:diagnose_latent_collapse_v2 \
        -c fbcode.nvcc_arch=a100 \
        -c fbcode.enable_gpu_sections=true \
        --local-only -- --batch-size 128 --seed 42
"""

import argparse
import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from utils.lipschitz import (
    apply_opnorm_clamp_to_trm,
    enforce_global_contraction,
    apply_spectral_norm_to_value_head,
    enforce_global_contraction_on_value_head,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_4x4_data(data_dir: str, batch_size: int, seed: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Load a batch of 4x4 Sudoku puzzles."""
    train_dir = Path(data_dir) / "train"
    inputs = np.load(train_dir / "all__inputs.npy")
    labels = np.load(train_dir / "all__labels.npy")
    puzzle_ids = np.load(train_dir / "all__puzzle_identifiers.npy")

    print(f"Loaded dataset: {inputs.shape[0]} puzzles, seq_len={inputs.shape[1]}")

    np.random.seed(seed)
    indices = np.random.choice(len(inputs), size=min(batch_size, len(inputs)), replace=False)

    batch = {
        "inputs": torch.tensor(inputs[indices], dtype=torch.long),
        "puzzle_identifiers": torch.tensor(puzzle_ids[indices], dtype=torch.long),
    }
    y = torch.tensor(labels[indices], dtype=torch.long)
    return batch, y


def build_base_model(seed: int = 42) -> TinyRecursiveReasoningModel_ACTV1:
    """
    Build a TRM model with NO contraction applied (base model).
    Contraction will be manually applied as needed.
    """
    torch.manual_seed(seed)

    config = dict(
        batch_size=256,
        seq_len=16,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=500,
        vocab_size=6,
        H_cycles=2,
        L_cycles=2,
        H_layers=1,
        L_layers=2,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        rl_enable_value_head=True,
        rl_enable_policy_head=False,
        rl_num_actions=97,
        # KEY: Disable automatic contraction - we'll apply manually
        rl_enable_contraction=False,
        rl_target_Lz=0.9,
        rl_target_Lv=1.0,
        rl_value_hidden_dim=128,
        # KEY: Disable projection in model - we'll compute pre/post ourselves
        rl_latent_ball_radius=0.0,
        forward_dtype="float32",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TinyRecursiveReasoningModel_ACTV1(config)

    return model.float()


def apply_z_contraction(model: TinyRecursiveReasoningModel_ACTV1, target_Lz: float = 0.9):
    """Apply z→z contraction to the inner model (L_level layers only)."""
    apply_opnorm_clamp_to_trm(
        model.inner,
        per_layer_max=1.0,
        num_power_iters=10,
        restrict_to_reasoning_layers=True,
    )
    enforce_global_contraction(
        model.inner,
        target_Lz,
        restrict_to_reasoning_layers=True,
    )


def apply_vhead_normalization(model: TinyRecursiveReasoningModel_ACTV1, target_Lv: float = 1.0):
    """Apply spectral norm + scaling to value head."""
    if model.value_head is not None:
        apply_spectral_norm_to_value_head(model.value_head)
        enforce_global_contraction_on_value_head(model.value_head, target_Lv)


def project_to_ball(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Project z to ball of given radius (per-sample)."""
    z_norm = z.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(radius / z_norm, max=1.0)
    return z * scale


def unroll_latent_with_pre_post(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    y: torch.Tensor,
    n: int,
    radius: float = 10.0,
) -> Tuple[
    TinyRecursiveReasoningModel_ACTV1InnerCarry,  # z_pre (before projection)
    TinyRecursiveReasoningModel_ACTV1InnerCarry,  # z_post (after projection)
]:
    """
    Run latent unrolling and return both pre-projection and post-projection z.

    The model is built with rl_latent_ball_radius=0 so no internal projection happens.
    We manually compute both versions.
    """
    # Initialize z^(0) - no projection since radius=0 in model
    z = model.init_latent(batch, y)

    # Pre-compute context
    batch_internal = model._standardize_latent_batch(batch, y)
    batch_internal["_latent_context"] = model._build_latent_context_with_plan(batch_internal)

    # Unroll n steps
    for _ in range(n):
        context = model._resolve_latent_context(batch_internal)
        input_embeds = context.get("input_embeddings_with_plan", context["input_embeddings"])
        # This runs without projection since radius=0
        z = model.inner.latent_step(z, input_embeds, context["seq_info"])

    # z is now pre-projection
    z_pre = TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=z.z_H.clone(),
        z_L=z.z_L.clone(),
    )

    # Apply projection manually
    z_post = TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=project_to_ball(z.z_H, radius),
        z_L=project_to_ball(z.z_L, radius),
    )

    return z_pre, z_post


def compute_latent_metrics(z_H: torch.Tensor, z_L: torch.Tensor) -> Dict[str, float]:
    """Compute norm and diversity metrics for latent tensors."""
    z_H = z_H.float()
    z_L = z_L.float()
    B = z_H.shape[0]

    z_H_flat = z_H.view(B, -1)
    z_L_flat = z_L.view(B, -1)

    # Per-sample L2 norms
    norms_H = torch.norm(z_H_flat, dim=1)
    norms_L = torch.norm(z_L_flat, dim=1)

    # Per-dim variance
    per_dim_var_H = z_H_flat.var(dim=0).mean().item()
    per_dim_var_L = z_L_flat.var(dim=0).mean().item()

    # Total variance
    mean_z_H = z_H_flat.mean(dim=0, keepdim=True)
    mean_z_L = z_L_flat.mean(dim=0, keepdim=True)
    total_var_H = ((z_H_flat - mean_z_H) ** 2).sum(dim=1).mean().item()
    total_var_L = ((z_L_flat - mean_z_L) ** 2).sum(dim=1).mean().item()

    # Cosine similarity (sample random pairs)
    num_pairs = min(1000, B * (B - 1) // 2)
    if B > 1 and num_pairs > 0:
        idx1 = torch.randint(0, B, (num_pairs * 2,))
        idx2 = torch.randint(0, B, (num_pairs * 2,))
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask][:num_pairs], idx2[mask][:num_pairs]

        if len(idx1) > 0:
            z_H_norm = F.normalize(z_H_flat, dim=1)
            z_L_norm = F.normalize(z_L_flat, dim=1)
            cos_sim_H = (z_H_norm[idx1] * z_H_norm[idx2]).sum(dim=1).mean().item()
            cos_sim_L = (z_L_norm[idx1] * z_L_norm[idx2]).sum(dim=1).mean().item()
        else:
            cos_sim_H = cos_sim_L = 1.0
    else:
        cos_sim_H = cos_sim_L = 1.0

    return {
        "z_H_norm_mean": norms_H.mean().item(),
        "z_H_norm_std": norms_H.std().item(),
        "z_L_norm_mean": norms_L.mean().item(),
        "z_L_norm_std": norms_L.std().item(),
        "per_dim_var_H": per_dim_var_H,
        "per_dim_var_L": per_dim_var_L,
        "total_var_H": total_var_H,
        "total_var_L": total_var_L,
        "cos_sim_H": cos_sim_H,
        "cos_sim_L": cos_sim_L,
    }


def compute_value_stats(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    y: torch.Tensor,
    z_post: TinyRecursiveReasoningModel_ACTV1InnerCarry,
) -> Dict[str, float]:
    """Compute value head statistics using post-projection z."""
    with torch.no_grad():
        # Get embeddings
        batch_internal = model._standardize_latent_batch(batch, y)
        context = model._build_latent_context_with_plan(batch_internal)
        input_embeddings = context["input_embeddings"]
        plan_embeddings = context["plan_embeddings"]

        # Flatten z and embeddings
        z_vec = z_post.z_H.view(z_post.z_H.shape[0], -1)
        x_embed = input_embeddings.view(input_embeddings.shape[0], -1)
        y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)
        combined_embed = torch.cat([x_embed, y_embed], dim=-1)

        # Compute value
        value = model.value_head(z_vec, combined_embed)

    return {
        "V_mean": value.mean().item(),
        "V_std": value.std().item(),
        "V_min": value.min().item(),
        "V_max": value.max().item(),
        "V_var": value.var().item(),
    }


def run_condition(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    y: torch.Tensor,
    unroll_depths: List[int],
    condition_name: str,
    radius: float = 10.0,
) -> Dict[str, Any]:
    """Run diagnostic for a single condition."""
    model.eval()
    device = next(model.parameters()).device
    batch_on_device = {k: v.to(device) for k, v in batch.items()}
    y_on_device = y.to(device)

    results = {"condition": condition_name, "depths": {}}

    for n in unroll_depths:
        print(f"    n={n}...")

        with torch.no_grad():
            z_pre, z_post = unroll_latent_with_pre_post(
                model, batch_on_device, y_on_device, n, radius
            )

        # Metrics for pre-projection z
        pre_metrics = compute_latent_metrics(z_pre.z_H, z_pre.z_L)
        pre_metrics = {f"pre_{k}": v for k, v in pre_metrics.items()}

        # Metrics for post-projection z
        post_metrics = compute_latent_metrics(z_post.z_H, z_post.z_L)
        post_metrics = {f"post_{k}": v for k, v in post_metrics.items()}

        # Value stats (using post-projection z)
        value_stats = compute_value_stats(model, batch_on_device, y_on_device, z_post)

        results["depths"][n] = {**pre_metrics, **post_metrics, **value_stats}

    return results


def format_report(all_results: Dict[str, Any], seed: int) -> str:
    """Generate human-readable REPORT.md content."""

    report = f"""# Latent Collapse Diagnostic V2 Report

## Experiment Overview

**Goal**: Isolate the effect of z→z contraction from value-head normalization, and measure
pre-projection vs post-projection latents.

**2×2 Factorial Design**:
- **Axis 1**: z→z contraction (zcon_OFF vs zcon_ON with target_Lz=0.9)
- **Axis 2**: value-head normalization (vhead_OFF vs vhead_ON with target_Lv=1.0)

**Conditions**:
| Condition | z→z Contraction | Value Head Norm |
|-----------|-----------------|-----------------|
| A         | OFF             | OFF             |
| B         | ON              | OFF             |
| C         | OFF             | ON              |
| D         | ON              | ON              |

**Setup**:
- Seed: {seed}
- Dataset: 4×4 Sudoku (trivial, 1–4 empties)
- Batch size: {all_results.get('batch_size', 128)} states
- Projection radius: {all_results.get('radius', 10.0)}

---

## Pre-Projection Latent Norms (z_H)

Shows ||z|| BEFORE applying the ball projection. If contraction causes collapse,
we expect lower norms / less diversity here.

| n | Condition | mean ||z|| | std ||z|| | Total Var | Cos Sim |
|---|-----------|-----------|---------|-----------|---------|
"""

    depths = sorted(all_results["A"]["depths"].keys())
    for n in depths:
        for cond in ["A", "B", "C", "D"]:
            d = all_results[cond]["depths"][n]
            report += f"| {n} | {cond} | {d['pre_z_H_norm_mean']:.4f} | {d['pre_z_H_norm_std']:.4f} | {d['pre_total_var_H']:.4f} | {d['pre_cos_sim_H']:.4f} |\n"

    report += """
## Post-Projection Latent Norms (z_H)

Shows ||z|| AFTER applying the ball projection (radius=10.0).

| n | Condition | mean ||z|| | std ||z|| | Total Var | Cos Sim |
|---|-----------|-----------|---------|-----------|---------|
"""

    for n in depths:
        for cond in ["A", "B", "C", "D"]:
            d = all_results[cond]["depths"][n]
            report += f"| {n} | {cond} | {d['post_z_H_norm_mean']:.4f} | {d['post_z_H_norm_std']:.4f} | {d['post_total_var_H']:.4f} | {d['post_cos_sim_H']:.4f} |\n"

    report += """
## Value Head Statistics

Shows V(z) statistics. Comparing B vs A isolates z→z effect on value (same value head).

| n | Condition | V_mean | V_std | V_min | V_max | V_var |
|---|-----------|--------|-------|-------|-------|-------|
"""

    for n in depths:
        for cond in ["A", "B", "C", "D"]:
            d = all_results[cond]["depths"][n]
            report += f"| {n} | {cond} | {d['V_mean']:.4f} | {d['V_std']:.4f} | {d['V_min']:.4f} | {d['V_max']:.4f} | {d['V_var']:.6f} |\n"

    # Analysis at deepest n
    max_n = max(depths)
    A = all_results["A"]["depths"][max_n]
    B = all_results["B"]["depths"][max_n]
    C = all_results["C"]["depths"][max_n]
    D = all_results["D"]["depths"][max_n]

    report += f"""
---

## Analysis (at n={max_n})

### Effect of z→z Contraction (B vs A, holding value head constant = OFF)

| Metric | A (zcon OFF) | B (zcon ON) | B/A Ratio |
|--------|--------------|-------------|-----------|
| pre ||z_H|| mean | {A['pre_z_H_norm_mean']:.4f} | {B['pre_z_H_norm_mean']:.4f} | {B['pre_z_H_norm_mean']/A['pre_z_H_norm_mean']:.3f} |
| pre Total Var | {A['pre_total_var_H']:.4f} | {B['pre_total_var_H']:.4f} | {B['pre_total_var_H']/A['pre_total_var_H']:.3f} |
| pre Cos Sim | {A['pre_cos_sim_H']:.4f} | {B['pre_cos_sim_H']:.4f} | {B['pre_cos_sim_H']-A['pre_cos_sim_H']:+.4f} (diff) |
| V_var | {A['V_var']:.6f} | {B['V_var']:.6f} | {B['V_var']/(A['V_var']+1e-10):.3f} |

### Effect of z→z Contraction (D vs C, holding value head constant = ON)

| Metric | C (zcon OFF) | D (zcon ON) | D/C Ratio |
|--------|--------------|-------------|-----------|
| pre ||z_H|| mean | {C['pre_z_H_norm_mean']:.4f} | {D['pre_z_H_norm_mean']:.4f} | {D['pre_z_H_norm_mean']/C['pre_z_H_norm_mean']:.3f} |
| pre Total Var | {C['pre_total_var_H']:.4f} | {D['pre_total_var_H']:.4f} | {D['pre_total_var_H']/C['pre_total_var_H']:.3f} |
| pre Cos Sim | {C['pre_cos_sim_H']:.4f} | {D['pre_cos_sim_H']:.4f} | {D['pre_cos_sim_H']-C['pre_cos_sim_H']:+.4f} (diff) |
| V_var | {C['V_var']:.6f} | {D['V_var']:.6f} | {D['V_var']/(C['V_var']+1e-10):.3f} |

### Effect of Value Head Norm (C vs A, holding z→z constant = OFF)

| Metric | A (vhead OFF) | C (vhead ON) | C/A Ratio |
|--------|---------------|--------------|-----------|
| V_var | {A['V_var']:.6f} | {C['V_var']:.6f} | {C['V_var']/(A['V_var']+1e-10):.3f} |

### Projection Saturation Check

How much does projection clip the norm? (pre vs post at n={max_n})

| Condition | pre ||z_H|| | post ||z_H|| | Saturation |
|-----------|------------|-------------|------------|
"""

    for cond in ["A", "B", "C", "D"]:
        d = all_results[cond]["depths"][max_n]
        pre_norm = d['pre_z_H_norm_mean']
        post_norm = d['post_z_H_norm_mean']
        saturated = "YES" if pre_norm > 10.0 else "NO"
        report += f"| {cond} | {pre_norm:.4f} | {post_norm:.4f} | {saturated} |\n"

    # Conclusions
    report += """
---

## Conclusions

"""

    # Check for collapse in pre-projection z
    B_pre_var = B['pre_total_var_H']
    A_pre_var = A['pre_total_var_H']
    D_pre_var = D['pre_total_var_H']
    C_pre_var = C['pre_total_var_H']

    # B vs A ratio
    ba_var_ratio = B_pre_var / (A_pre_var + 1e-10)
    dc_var_ratio = D_pre_var / (C_pre_var + 1e-10)

    if ba_var_ratio < 0.5 and dc_var_ratio < 0.5:
        report += """**EVIDENCE OF COLLAPSE IN PRE-PROJECTION Z**

z→z contraction significantly reduces latent variance before projection:
- B/A variance ratio: {:.3f} (< 0.5)
- D/C variance ratio: {:.3f} (< 0.5)

This suggests contraction causes the latent representations to cluster together.
""".format(ba_var_ratio, dc_var_ratio)
    elif ba_var_ratio > 1.5 and dc_var_ratio > 1.5:
        report += """**NO COLLAPSE - CONTRACTION INCREASES DIVERSITY**

z→z contraction actually increases latent variance before projection:
- B/A variance ratio: {:.3f} (> 1.5)
- D/C variance ratio: {:.3f} (> 1.5)

Collapse is NOT the mechanism by which contraction affects learning.
""".format(ba_var_ratio, dc_var_ratio)
    else:
        report += """**MIXED/INCONCLUSIVE RESULTS**

z→z contraction has inconsistent effects on latent variance:
- B/A variance ratio: {:.3f}
- D/C variance ratio: {:.3f}

The effect of contraction on latent diversity is unclear.
""".format(ba_var_ratio, dc_var_ratio)

    # Projection saturation
    if A['pre_z_H_norm_mean'] > 10.0 or B['pre_z_H_norm_mean'] > 10.0:
        report += """
**PROJECTION SATURATION DETECTED**

Pre-projection norms exceed the ball radius (10.0), so projection is active.
This explains why post-projection ||z|| = 10.0 uniformly.
"""
    else:
        report += """
**NO PROJECTION SATURATION**

Pre-projection norms are within the ball radius, so projection has minimal effect.
"""

    report += """
---

*Report generated by scripts/diagnostics/diagnose_latent_collapse_v2.py*
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Latent collapse diagnostic v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data/sudoku-4x4-trivial")
    parser.add_argument("--target-Lz", type=float, default=0.9)
    parser.add_argument("--target-Lv", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--output-dir", type=str, default="results/diagnostics_latent_collapse_v2")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    batch, y = load_4x4_data(args.data_dir, args.batch_size, args.seed)
    print(f"Batch: inputs={batch['inputs'].shape}, y={y.shape}")

    unroll_depths = [1, 2, 4, 8, 16]

    # Build 4 models with same base weights, then apply different treatments
    print("\n=== Building base model ===")
    set_seed(args.seed)
    base_state = build_base_model(args.seed).state_dict()

    all_results = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "target_Lz": args.target_Lz,
        "target_Lv": args.target_Lv,
        "radius": args.radius,
        "unroll_depths": unroll_depths,
    }

    conditions = [
        ("A", False, False, "zcon_OFF + vhead_OFF"),
        ("B", True, False, "zcon_ON + vhead_OFF"),
        ("C", False, True, "zcon_OFF + vhead_ON"),
        ("D", True, True, "zcon_ON + vhead_ON"),
    ]

    for cond_name, zcon_on, vhead_on, desc in conditions:
        print(f"\n=== Condition {cond_name}: {desc} ===")

        # Build fresh model and load base weights
        set_seed(args.seed)
        model = build_base_model(args.seed)
        model.load_state_dict(base_state, strict=True)

        # Apply treatments
        if zcon_on:
            print(f"  Applying z→z contraction (target_Lz={args.target_Lz})")
            apply_z_contraction(model, args.target_Lz)
        if vhead_on:
            print(f"  Applying value head normalization (target_Lv={args.target_Lv})")
            apply_vhead_normalization(model, args.target_Lv)

        model = model.to(device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        print(f"  Running diagnostics...")
        results = run_condition(model, batch, y, unroll_depths, cond_name, args.radius)
        all_results[cond_name] = results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"latent_collapse_v2_seed{args.seed}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    # Generate report
    report = format_report(all_results, args.seed)
    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY (at n=16)")
    print("="*70)

    max_n = max(unroll_depths)
    for cond in ["A", "B", "C", "D"]:
        d = all_results[cond]["depths"][max_n]
        print(f"\n{cond} ({all_results[cond]['condition']}):")
        print(f"  PRE:  ||z_H||={d['pre_z_H_norm_mean']:.4f}, TotalVar={d['pre_total_var_H']:.4f}, CosSim={d['pre_cos_sim_H']:.4f}")
        print(f"  POST: ||z_H||={d['post_z_H_norm_mean']:.4f}, TotalVar={d['post_total_var_H']:.4f}, CosSim={d['post_cos_sim_H']:.4f}")
        print(f"  V: mean={d['V_mean']:.4f}, std={d['V_std']:.4f}, var={d['V_var']:.6f}")

    print("\nSee REPORT.md for full analysis.")


if __name__ == "__main__":
    main()
