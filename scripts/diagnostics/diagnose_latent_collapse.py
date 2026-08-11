#!/usr/bin/env python3
"""
Diagnostic script to test the latent collapse hypothesis under contraction enforcement.

Hypothesis: When we enforce contraction on the inner z→z recursion (opnorm clamp + scaling,
target_Lz<1), the latent fixed point / late-iteration latent may "collapse" (e.g., z shrinks
toward 0 or becomes nearly constant across different (x,y)), which could reduce useful
downstream signal and hurt learning.

Compares two conditions:
  A) Contraction OFF (no clamp/scale on z→z)
  B) Contraction ON (opnorm clamp + scaling; target_Lz=0.9)

Usage:
    buck2 run //buiksat_trm:diagnose_latent_collapse \
        -c fbcode.nvcc_arch=a100 \
        -c fbcode.enable_gpu_sections=true \
        -- --batch-size 128 --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_4x4_data(data_dir: str, batch_size: int, seed: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Load a batch of 4x4 Sudoku puzzles.

    Returns:
        batch: dict with 'inputs' and 'puzzle_identifiers'
        y: [batch_size, 16] solutions (fully filled)
    """
    train_dir = Path(data_dir) / "train"

    # Load numpy arrays
    inputs = np.load(train_dir / "all__inputs.npy")
    labels = np.load(train_dir / "all__labels.npy")
    puzzle_ids = np.load(train_dir / "all__puzzle_identifiers.npy")

    print(f"Loaded dataset: {inputs.shape[0]} puzzles, seq_len={inputs.shape[1]}")

    # Random sample
    np.random.seed(seed)
    indices = np.random.choice(len(inputs), size=min(batch_size, len(inputs)), replace=False)

    x_tensor = torch.tensor(inputs[indices], dtype=torch.long)
    y_tensor = torch.tensor(labels[indices], dtype=torch.long)
    id_tensor = torch.tensor(puzzle_ids[indices], dtype=torch.long)

    # Return as batch dict for model compatibility
    batch = {
        "inputs": x_tensor,
        "puzzle_identifiers": id_tensor,
    }
    return batch, y_tensor


def build_model(enable_contraction: bool, target_Lz: float = 0.9, seed: int = 42) -> TinyRecursiveReasoningModel_ACTV1:
    """
    Build a TRM model with or without contraction enforcement.

    Uses random initialization (no pretrained weights) to ensure fair comparison.
    """
    torch.manual_seed(seed)

    config = dict(
        batch_size=256,
        seq_len=16,
        puzzle_emb_ndim=0,  # Disable puzzle embeddings for simplicity
        puzzle_emb_len=0,   # Must be 0 when puzzle_emb_ndim=0
        num_puzzle_identifiers=500,
        vocab_size=6,
        H_cycles=2,
        L_cycles=2,
        H_layers=1,  # Required by config (ignored)
        L_layers=2,  # Match typical TRM depth
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=16,  # Required by config
        halt_exploration_prob=0.1,  # Required by config
        rl_enable_value_head=True,
        rl_enable_policy_head=False,  # Not needed for this diagnostic
        rl_num_actions=97,
        rl_enable_contraction=enable_contraction,
        rl_target_Lz=target_Lz,
        rl_target_Lv=1.0,  # Value head Lipschitz
        rl_value_hidden_dim=128,
        rl_latent_ball_radius=10.0,  # Forward-invariant projection
        forward_dtype="float32",  # Use float32 instead of bfloat16 for stability
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress contraction warnings
        model = TinyRecursiveReasoningModel_ACTV1(config)

    # Ensure all parameters are float32
    model = model.float()

    return model


def compute_latent_norms(z_H: torch.Tensor, z_L: torch.Tensor) -> Dict[str, float]:
    """
    Compute norm statistics for latent tensors.

    Args:
        z_H: [B, seq_len, hidden_size] H-level latent
        z_L: [B, seq_len, hidden_size] L-level latent

    Returns:
        Dict of norm statistics
    """
    # Convert to float for numerical stability
    z_H = z_H.float()
    z_L = z_L.float()

    # Flatten to [B, -1] for per-sample norms
    z_H_flat = z_H.view(z_H.shape[0], -1)
    z_L_flat = z_L.view(z_L.shape[0], -1)

    # Per-sample L2 norms
    norms_H = torch.norm(z_H_flat, dim=1)
    norms_L = torch.norm(z_L_flat, dim=1)

    # Combined z norms (using z_H as primary, but also report z_L)
    stats = {
        "z_H_norm_mean": norms_H.mean().item(),
        "z_H_norm_std": norms_H.std().item(),
        "z_H_norm_p10": norms_H.quantile(0.1).item(),
        "z_H_norm_p50": norms_H.quantile(0.5).item(),
        "z_H_norm_p90": norms_H.quantile(0.9).item(),
        "z_L_norm_mean": norms_L.mean().item(),
        "z_L_norm_std": norms_L.std().item(),
        "z_L_norm_p10": norms_L.quantile(0.1).item(),
        "z_L_norm_p50": norms_L.quantile(0.5).item(),
        "z_L_norm_p90": norms_L.quantile(0.9).item(),
    }
    return stats


def compute_collapse_metrics(z_H: torch.Tensor, z_L: torch.Tensor, num_pairs: int = 1000) -> Dict[str, float]:
    """
    Compute collapse metrics across the batch.

    Args:
        z_H: [B, seq_len, hidden_size]
        z_L: [B, seq_len, hidden_size]
        num_pairs: Number of random pairs for cosine similarity approximation

    Returns:
        Dict of collapse metrics
    """
    # Convert to float for numerical stability
    z_H = z_H.float()
    z_L = z_L.float()

    B = z_H.shape[0]

    # Flatten to [B, -1]
    z_H_flat = z_H.view(B, -1)
    z_L_flat = z_L.view(B, -1)

    # 1. Mean per-dimension variance: mean_j Var(z[:, j])
    per_dim_var_H = z_H_flat.var(dim=0).mean().item()
    per_dim_var_L = z_L_flat.var(dim=0).mean().item()

    # 2. Total variance: mean_i ||z_i - mean(z)||^2
    mean_z_H = z_H_flat.mean(dim=0, keepdim=True)
    mean_z_L = z_L_flat.mean(dim=0, keepdim=True)
    total_var_H = ((z_H_flat - mean_z_H) ** 2).sum(dim=1).mean().item()
    total_var_L = ((z_L_flat - mean_z_L) ** 2).sum(dim=1).mean().item()

    # 3. Mean pairwise cosine similarity (approximate with random pairs)
    # Higher = more collapse (all samples pointing same direction)
    if B > 1:
        num_pairs = min(num_pairs, B * (B - 1) // 2)
        idx1 = torch.randint(0, B, (num_pairs,))
        idx2 = torch.randint(0, B, (num_pairs,))
        # Avoid self-comparisons
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

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
        "per_dim_var_H": per_dim_var_H,
        "per_dim_var_L": per_dim_var_L,
        "total_var_H": total_var_H,
        "total_var_L": total_var_L,
        "cos_sim_H": cos_sim_H,
        "cos_sim_L": cos_sim_L,
    }


def compute_value_variance(model: TinyRecursiveReasoningModel_ACTV1, batch: Dict[str, torch.Tensor], y: torch.Tensor, n: int) -> float:
    """
    Compute variance of value head output across the batch.
    """
    with torch.no_grad():
        value, _ = model.used_value(batch, y, n)
    return value.var().item()


def run_diagnostic(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    y: torch.Tensor,
    unroll_depths: List[int],
    condition_name: str,
) -> Dict[str, Any]:
    """
    Run diagnostic for a single condition (contraction ON or OFF).
    """
    model.eval()
    device = next(model.parameters()).device

    # Move batch tensors to device
    batch_on_device = {k: v.to(device) for k, v in batch.items()}
    y = y.to(device)

    results = {"condition": condition_name, "depths": {}}

    for n in unroll_depths:
        print(f"  Unroll depth n={n}...")

        with torch.no_grad():
            z_n, z_trajectory = model.unroll_latent(batch_on_device, y, n)

        z_H = z_n.z_H  # [B, seq_len, hidden_size]
        z_L = z_n.z_L

        # Compute metrics
        norm_stats = compute_latent_norms(z_H, z_L)
        collapse_stats = compute_collapse_metrics(z_H, z_L)
        value_var = compute_value_variance(model, batch_on_device, y, n)

        results["depths"][n] = {
            **norm_stats,
            **collapse_stats,
            "value_variance": value_var,
        }

    return results


def format_report(results_A: Dict, results_B: Dict, seed: int) -> str:
    """Generate human-readable REPORT.md content."""

    report = f"""# Latent Collapse Diagnostic Report

## Experiment Overview

**Hypothesis**: When we enforce contraction on the inner z→z recursion (opnorm clamp + scaling,
target_Lz=0.9), the latent representations may "collapse" — i.e., z norms shrink toward 0 and/or
latent vectors become nearly constant across different (x,y) states. This could reduce useful
downstream signal and hurt learning.

**Test Setup**:
- Seed: {seed}
- Dataset: 4×4 Sudoku (trivial, 1–4 empties)
- Batch size: {results_A.get('batch_size', 128)} states
- Conditions:
  - **A (Contraction OFF)**: No opnorm clamp or scaling on z→z path
  - **B (Contraction ON)**: opnorm clamp + scaling with target_Lz=0.9

**Collapse indicators**:
- Lower ||z|| norms → latent shrinks toward zero
- Lower per-dim variance → reduced signal variation
- Lower total variance → samples cluster together
- Higher cosine similarity → samples point in same direction (extreme collapse)
- Lower Var(V) → value function loses discriminative power

---

## Results

### Latent Norm Statistics (z_H)

| n | Condition | mean ||z|| | std ||z|| | p10 | p50 | p90 |
|---|-----------|-----------|---------|-----|-----|-----|
"""

    depths = sorted(results_A["depths"].keys())
    for n in depths:
        A = results_A["depths"][n]
        B = results_B["depths"][n]
        report += f"| {n} | A (OFF) | {A['z_H_norm_mean']:.4f} | {A['z_H_norm_std']:.4f} | {A['z_H_norm_p10']:.4f} | {A['z_H_norm_p50']:.4f} | {A['z_H_norm_p90']:.4f} |\n"
        report += f"| {n} | B (ON)  | {B['z_H_norm_mean']:.4f} | {B['z_H_norm_std']:.4f} | {B['z_H_norm_p10']:.4f} | {B['z_H_norm_p50']:.4f} | {B['z_H_norm_p90']:.4f} |\n"

    report += """
### Latent Norm Statistics (z_L)

| n | Condition | mean ||z|| | std ||z|| | p10 | p50 | p90 |
|---|-----------|-----------|---------|-----|-----|-----|
"""
    for n in depths:
        A = results_A["depths"][n]
        B = results_B["depths"][n]
        report += f"| {n} | A (OFF) | {A['z_L_norm_mean']:.4f} | {A['z_L_norm_std']:.4f} | {A['z_L_norm_p10']:.4f} | {A['z_L_norm_p50']:.4f} | {A['z_L_norm_p90']:.4f} |\n"
        report += f"| {n} | B (ON)  | {B['z_L_norm_mean']:.4f} | {B['z_L_norm_std']:.4f} | {B['z_L_norm_p10']:.4f} | {B['z_L_norm_p50']:.4f} | {B['z_L_norm_p90']:.4f} |\n"

    report += """
### Collapse Metrics (z_H)

| n | Condition | Per-dim Var | Total Var | Cosine Sim |
|---|-----------|------------|-----------|------------|
"""
    for n in depths:
        A = results_A["depths"][n]
        B = results_B["depths"][n]
        report += f"| {n} | A (OFF) | {A['per_dim_var_H']:.6f} | {A['total_var_H']:.4f} | {A['cos_sim_H']:.4f} |\n"
        report += f"| {n} | B (ON)  | {B['per_dim_var_H']:.6f} | {B['total_var_H']:.4f} | {B['cos_sim_H']:.4f} |\n"

    report += """
### Collapse Metrics (z_L)

| n | Condition | Per-dim Var | Total Var | Cosine Sim |
|---|-----------|------------|-----------|------------|
"""
    for n in depths:
        A = results_A["depths"][n]
        B = results_B["depths"][n]
        report += f"| {n} | A (OFF) | {A['per_dim_var_L']:.6f} | {A['total_var_L']:.4f} | {A['cos_sim_L']:.4f} |\n"
        report += f"| {n} | B (ON)  | {B['per_dim_var_L']:.6f} | {B['total_var_L']:.4f} | {B['cos_sim_L']:.4f} |\n"

    report += """
### Value Variance (Downstream Signal)

| n | Condition | Var(V) |
|---|-----------|--------|
"""
    for n in depths:
        A = results_A["depths"][n]
        B = results_B["depths"][n]
        report += f"| {n} | A (OFF) | {A['value_variance']:.6f} |\n"
        report += f"| {n} | B (ON)  | {B['value_variance']:.6f} |\n"

    # Analysis
    # Compare at deepest unroll
    max_n = max(depths)
    A_deep = results_A["depths"][max_n]
    B_deep = results_B["depths"][max_n]

    norm_ratio = B_deep['z_H_norm_mean'] / (A_deep['z_H_norm_mean'] + 1e-8)
    var_ratio = B_deep['total_var_H'] / (A_deep['total_var_H'] + 1e-8)
    cos_diff = B_deep['cos_sim_H'] - A_deep['cos_sim_H']
    value_ratio = B_deep['value_variance'] / (A_deep['value_variance'] + 1e-8)

    report += f"""
---

## Analysis (at n={max_n})

**Norm Comparison**:
- ||z_H|| ratio (B/A): {norm_ratio:.3f}
- If << 1: contraction shrinks latent norms significantly

**Variance Comparison**:
- Total variance ratio (B/A): {var_ratio:.3f}
- If << 1: sampled contracted representations are more clustered

**Cosine Similarity**:
- Δ cos_sim (B - A): {cos_diff:+.4f}
- If >> 0: contraction makes samples more similar (collapse)

**Value Variance**:
- Var(V) ratio (B/A): {value_ratio:.3f}
- If << 1: value function loses discriminative power under contraction

---

## Conclusion

"""

    # Generate conclusion based on metrics
    collapse_evidence = []
    no_collapse_evidence = []

    if norm_ratio < 0.5:
        collapse_evidence.append(f"- z_H norms are {(1-norm_ratio)*100:.0f}% smaller with contraction")
    elif norm_ratio > 0.8:
        no_collapse_evidence.append(f"- z_H norms are comparable (ratio={norm_ratio:.2f})")

    if var_ratio < 0.5:
        collapse_evidence.append(f"- Total variance is {(1-var_ratio)*100:.0f}% lower with contraction")
    elif var_ratio > 0.8:
        no_collapse_evidence.append(f"- Total variance is comparable (ratio={var_ratio:.2f})")

    if cos_diff > 0.1:
        collapse_evidence.append(f"- Cosine similarity increased by {cos_diff:.3f} (samples more aligned)")
    elif abs(cos_diff) < 0.05:
        no_collapse_evidence.append(f"- Cosine similarity unchanged (Δ={cos_diff:+.3f})")

    if value_ratio < 0.5:
        collapse_evidence.append(f"- Value variance is {(1-value_ratio)*100:.0f}% lower (less discriminative)")
    elif value_ratio > 0.8:
        no_collapse_evidence.append(f"- Value variance is comparable (ratio={value_ratio:.2f})")

    if len(collapse_evidence) >= 2:
        report += """**EVIDENCE SUPPORTS COLLAPSE HYPOTHESIS**

The sampled contracted representations are more clustered:
"""
        for e in collapse_evidence:
            report += f"{e}\n"
        if no_collapse_evidence:
            report += "\nHowever, some metrics are stable:\n"
            for e in no_collapse_evidence:
                report += f"{e}\n"
    elif len(no_collapse_evidence) >= 2:
        report += """**NO SIGNIFICANT EVIDENCE OF COLLAPSE**

The sampled diagnostics do not show substantial additional clustering:
"""
        for e in no_collapse_evidence:
            report += f"{e}\n"
        if collapse_evidence:
            report += "\nSome minor collapse indicators:\n"
            for e in collapse_evidence:
                report += f"{e}\n"
    else:
        report += """**INCONCLUSIVE RESULTS**

The evidence is mixed. Some metrics suggest collapse, others do not.

Collapse indicators:
"""
        for e in collapse_evidence:
            report += f"{e}\n"
        report += "\nStable metrics:\n"
        for e in no_collapse_evidence:
            report += f"{e}\n"

    report += """
---

*Report generated by scripts/diagnostics/diagnose_latent_collapse.py*
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose latent collapse under contraction")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data/sudoku-4x4-trivial",
                        help="Path to 4x4 Sudoku dataset")
    parser.add_argument("--target-Lz", type=float, default=0.9, help="Target Lipschitz constant for contraction")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics_latent_collapse",
                        help="Output directory for results")
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    batch, y = load_4x4_data(args.data_dir, args.batch_size, args.seed)
    print(f"Batch: inputs={batch['inputs'].shape}, y={y.shape}")

    # Unroll depths to test
    unroll_depths = [1, 2, 4, 8, 16]

    # Build models
    print("\n=== Condition A: Contraction OFF ===")
    model_A = build_model(enable_contraction=False, seed=args.seed)
    model_A = model_A.to(device)
    print(f"Model A parameters: {sum(p.numel() for p in model_A.parameters()):,}")

    print("\n=== Condition B: Contraction ON (target_Lz={}) ===".format(args.target_Lz))
    model_B = build_model(enable_contraction=True, target_Lz=args.target_Lz, seed=args.seed)
    model_B = model_B.to(device)
    print(f"Model B parameters: {sum(p.numel() for p in model_B.parameters()):,}")

    # Run diagnostics
    print("\n--- Running Condition A diagnostics ---")
    results_A = run_diagnostic(model_A, batch, y, unroll_depths, "A_contraction_OFF")
    results_A["batch_size"] = args.batch_size

    print("\n--- Running Condition B diagnostics ---")
    results_B = run_diagnostic(model_B, batch, y, unroll_depths, "B_contraction_ON")
    results_B["batch_size"] = args.batch_size

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    all_results = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "target_Lz": args.target_Lz,
        "unroll_depths": unroll_depths,
        "condition_A": results_A,
        "condition_B": results_B,
    }
    json_path = output_dir / f"latent_collapse_seed{args.seed}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    # Generate report
    report = format_report(results_A, results_B, args.seed)
    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    max_n = max(unroll_depths)
    A = results_A["depths"][max_n]
    B = results_B["depths"][max_n]
    print(f"\nAt unroll depth n={max_n}:")
    print(f"  ||z_H|| mean: A={A['z_H_norm_mean']:.4f}, B={B['z_H_norm_mean']:.4f} (ratio={B['z_H_norm_mean']/A['z_H_norm_mean']:.3f})")
    print(f"  Total var:    A={A['total_var_H']:.4f}, B={B['total_var_H']:.4f} (ratio={B['total_var_H']/A['total_var_H']:.3f})")
    print(f"  Cos sim:      A={A['cos_sim_H']:.4f}, B={B['cos_sim_H']:.4f} (Δ={B['cos_sim_H']-A['cos_sim_H']:+.4f})")
    print(f"  Var(V):       A={A['value_variance']:.6f}, B={B['value_variance']:.6f} (ratio={B['value_variance']/A['value_variance']:.3f})")
    print("\nSee REPORT.md for full analysis and conclusion.")


if __name__ == "__main__":
    main()
