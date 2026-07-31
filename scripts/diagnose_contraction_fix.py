#!/usr/bin/env python3
"""
Diagnostic script to validate the operator-norm clamp fix for contraction.

Tests the new opnorm_clamp approach vs the old spectral_norm approach.

Variants tested:
  (i) OFF: no clamping, no scaling
  (ii) CLAMP-only: opnorm clamp, no scaling
  (iii) CLAMP+SCALE: opnorm clamp + enforce_global_contraction (restricted)
  (iv) SN-only: spectral_norm (baseline, expected to explode)
  (v) SN+SCALE: spectral_norm + scaling (baseline, expected to explode)

Expected results:
  - OFF: Lz ~ 0.8-1.0 (natural network behavior)
  - CLAMP-only: Lz ~ 0.8-1.0 (per-layer norm ≤ 1)
  - CLAMP+SCALE: Lz ≈ target_Lz (goal!)
  - SN-only: Lz >> 1 (explosion)
  - SN+SCALE: Lz >> 1 (explosion)

Usage:
    buck2 run //buiksat_trm:diagnose_contraction_fix -- \\
        --dataset /home/buiksat/trm_bellman/data/sudoku-4x4-trivial \\
        --batch-size 4 --num-repeats 10 --target-lz 0.9 --eps-list 1e-2,1e-3,1e-4

    # Also test with lower target_Lz
    buck2 run //buiksat_trm:diagnose_contraction_fix -- \\
        --dataset /home/buiksat/trm_bellman/data/sudoku-4x4-trivial \\
        --batch-size 4 --num-repeats 10 --target-lz 0.5 --eps-list 1e-2,1e-3,1e-4
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.lipschitz import (
    apply_spectral_norm_to_trm,
    enforce_global_contraction,
    estimate_local_Lz,
    apply_opnorm_clamp_to_trm,
)


def load_batch(dataset_path: str, batch_size: int = 4) -> Tuple[Dict, torch.Tensor, int, int]:
    """Load a small batch of puzzles from the dataset."""
    split_path = Path(dataset_path) / "train"
    inputs = np.load(split_path / "all__inputs.npy")

    inputs = inputs[:batch_size]
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)

    with open(split_path / "dataset.json") as f:
        meta = json.load(f)

    vocab_size = meta.get("vocab_size", 6)
    seq_len = meta.get("seq_len", 16)

    x_batch = {
        "inputs": inputs_tensor,
        "puzzle_identifiers": torch.arange(batch_size, dtype=torch.long),
    }
    y_batch = inputs_tensor.clone()

    return x_batch, y_batch, vocab_size, seq_len


def count_layers(model) -> Tuple[int, int, Optional[float], int]:
    """Count SN layers, scaled layers, clamped layers, and get sample scale value."""
    sn_count = 0
    scale_count = 0
    sample_scale = None
    clamped_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            sn_count += 1
        if hasattr(module, '_inner_lip_scale'):
            scale_count += 1
            if sample_scale is None:
                sample_scale = module._inner_lip_scale.item()
        # Count clamped layers (those that have been through opnorm_clamp)
        # We don't have a marker, but we can check if the layer is in L_level
        if hasattr(module, 'weight') and 'L_level' in name:
            clamped_count += 1
    return sn_count, scale_count, sample_scale, clamped_count


def get_layer_norms(model) -> Dict[str, float]:
    """Get spectral norms of all linear layers."""
    from utils.lipschitz import _power_iteration
    norms = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() == 2:
            sigma = _power_iteration(module.weight, num_iters=10)
            norms[name] = sigma
    return norms


def build_base_model(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    hidden_size: int = 64,
) -> TinyRecursiveReasoningModel_ACTV1:
    """Build a base TRM model with contraction OFF."""
    num_actions = seq_len * vocab_size + 1

    cfg_dict = dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=batch_size,
        vocab_size=vocab_size,
        H_cycles=2,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=hidden_size,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=False,  # Always OFF - we apply manually
        rl_target_Lz=0.9,
        rl_target_Lv=0.9,
        rl_enable_policy_head=True,
        rl_num_actions=num_actions,
        rl_latent_ball_radius=0.0,
    )

    model = TinyRecursiveReasoningModel_ACTV1(cfg_dict)
    return model


def estimate_Lz_stats(
    model: TinyRecursiveReasoningModel_ACTV1,
    x_batch: Dict,
    y_batch: torch.Tensor,
    num_repeats: int = 10,
    eps: float = 1e-3,
) -> Tuple[float, float]:
    """Estimate Lz mean and std over multiple repeats."""
    model.eval()

    with torch.no_grad():
        batch = model._standardize_latent_batch(x_batch, y_batch)
        z0 = model.init_latent(x_batch, y_batch)
        context = model._resolve_latent_context(batch)

        # Warmup forward passes
        for _ in range(5):
            _ = model.inner.latent_step(z0, context["input_embeddings_with_plan"], context["seq_info"])

        lz_values = []
        for _ in range(num_repeats):
            lz_mean = estimate_local_Lz(
                inner_model=model.inner,
                carry=z0,
                context=context,
                num_samples=4,
                eps=eps,
                return_samples=False,
            )
            lz_values.append(lz_mean)

        lz_array = np.array(lz_values)
        return float(lz_array.mean()), float(lz_array.std())


def run_variant(
    base_model: TinyRecursiveReasoningModel_ACTV1,
    x_batch: Dict,
    y_batch: torch.Tensor,
    variant: str,
    target_lz: float,
    num_repeats: int,
    eps: float,
) -> Dict:
    """Run a single variant and return results."""
    # Deep copy to avoid modifying base model
    model = copy.deepcopy(base_model)

    # Apply transformations based on variant
    if variant == "OFF":
        pass  # No modifications
    elif variant == "CLAMP-only":
        # New approach: opnorm clamp only
        sigma_dict = apply_opnorm_clamp_to_trm(
            model.inner,
            per_layer_max=1.0,
            num_power_iters=10,
            restrict_to_reasoning_layers=True,
        )
    elif variant == "CLAMP+SCALE":
        # New approach: opnorm clamp + global scaling
        apply_opnorm_clamp_to_trm(
            model.inner,
            per_layer_max=1.0,
            num_power_iters=10,
            restrict_to_reasoning_layers=True,
        )
        enforce_global_contraction(model.inner, target_lz, restrict_to_reasoning_layers=True)
    elif variant == "SN-only":
        # Old approach: spectral_norm only
        apply_spectral_norm_to_trm(model.inner)
    elif variant == "SN+SCALE":
        # Old approach: spectral_norm + scaling
        apply_spectral_norm_to_trm(model.inner)
        enforce_global_contraction(model.inner, target_lz)
    elif variant == "SCALE-only":
        # Just scaling, no normalization
        enforce_global_contraction(model.inner, target_lz, restrict_to_reasoning_layers=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Count layers
    sn_count, scale_count, sample_scale, clamped_count = count_layers(model)

    # Get per-layer norms
    layer_norms = get_layer_norms(model)
    l_level_norms = {k: v for k, v in layer_norms.items() if "L_level" in k}
    max_layer_norm = max(l_level_norms.values()) if l_level_norms else 0.0

    # Estimate Lz
    lz_mean, lz_std = estimate_Lz_stats(model, x_batch, y_batch, num_repeats, eps)

    return {
        "variant": variant,
        "eps": eps,
        "target_lz": target_lz if variant in ["CLAMP+SCALE", "SCALE-only", "SN+SCALE"] else "N/A",
        "lz_mean": lz_mean,
        "lz_std": lz_std,
        "sn_count": sn_count,
        "scale_count": scale_count,
        "sample_scale": sample_scale,
        "max_layer_norm": max_layer_norm,
    }


def print_table(results: List[Dict], title: str = ""):
    """Print a formatted table of results."""
    if title:
        print(f"\n{title}")
    print("=" * 120)
    print(f"{'Variant':<14} {'eps':<10} {'target_Lz':<10} {'Lz_mean':<14} {'Lz_std':<12} {'max_σ(W)':<12} {'Notes'}")
    print("-" * 120)
    for r in results:
        target = r["target_lz"] if r["target_lz"] != "N/A" else "N/A"
        target_str = f"{target:.2f}" if isinstance(target, float) else target
        scale_str = f"scale={r['sample_scale']:.4f}" if r['sample_scale'] else "no scale"
        notes = f"SN:{r['sn_count']}, scaled:{r['scale_count']}, {scale_str}"
        status = "EXPLODED" if r["lz_mean"] > 10 else ("OK" if r["lz_mean"] < 1.5 else "HIGH")
        print(f"{r['variant']:<14} {r['eps']:<10.0e} {target_str:<10} {r['lz_mean']:<14.4f} {r['lz_std']:<12.4f} {r['max_layer_norm']:<12.4f} [{status}]")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Validate contraction fix with opnorm clamping")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Sudoku dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--target-lz", type=float, default=0.9)
    parser.add_argument("--eps-list", type=str, default="1e-3", help="Comma-separated eps values")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--skip-sn", action="store_true", help="Skip spectral_norm variants (known to explode)")
    args = parser.parse_args()

    eps_values = [float(e) for e in args.eps_list.split(",")]

    # Define variants
    new_variants = ["OFF", "CLAMP-only", "SCALE-only", "CLAMP+SCALE"]
    old_variants = ["SN-only", "SN+SCALE"]

    if args.skip_sn:
        variants = new_variants
    else:
        variants = new_variants + old_variants

    print("=" * 120)
    print("CONTRACTION FIX VALIDATION (opnorm clamp vs spectral_norm)")
    print("=" * 120)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num repeats: {args.num_repeats}")
    print(f"Target Lz: {args.target_lz}")
    print(f"Eps values: {eps_values}")
    print(f"Variants: {variants}")
    print()

    # Load batch
    x_batch, y_batch, vocab_size, seq_len = load_batch(args.dataset, args.batch_size)
    print(f"Loaded {args.batch_size} puzzles (seq_len={seq_len}, vocab_size={vocab_size})")

    # Build base model
    print("Building base model (contraction=OFF)...")
    base_model = build_base_model(vocab_size, seq_len, args.batch_size, args.hidden_size)

    # Show initial layer norms
    print("\nInitial per-layer spectral norms (L_level only):")
    initial_norms = get_layer_norms(base_model)
    for name, sigma in initial_norms.items():
        if "L_level" in name:
            print(f"  {name}: σ = {sigma:.4f}")
    print()

    # Run ablation for each eps
    all_results = []
    for eps in eps_values:
        print(f"Running variants with eps={eps:.0e}...")
        for variant in variants:
            result = run_variant(
                base_model, x_batch, y_batch,
                variant, args.target_lz, args.num_repeats, eps
            )
            all_results.append(result)
            status = "EXPLODED" if result["lz_mean"] > 10 else "OK"
            print(f"  {variant}: Lz={result['lz_mean']:.4f} [{status}]")

    # Print summary tables grouped by eps
    for eps in eps_values:
        eps_results = [r for r in all_results if r["eps"] == eps]
        print_table(eps_results, f"Results for eps={eps:.0e}, target_lz={args.target_lz}")

    # Interpretation
    print("\n" + "=" * 120)
    print("INTERPRETATION")
    print("=" * 120)

    for eps in eps_values:
        eps_results = [r for r in all_results if r["eps"] == eps]

        off_lz = next((r["lz_mean"] for r in eps_results if r["variant"] == "OFF"), None)
        clamp_lz = next((r["lz_mean"] for r in eps_results if r["variant"] == "CLAMP-only"), None)
        clamp_scale_lz = next((r["lz_mean"] for r in eps_results if r["variant"] == "CLAMP+SCALE"), None)
        sn_lz = next((r["lz_mean"] for r in eps_results if r["variant"] == "SN-only"), None)
        sn_scale_lz = next((r["lz_mean"] for r in eps_results if r["variant"] == "SN+SCALE"), None)

        print(f"\neps={eps:.0e}:")
        if off_lz is not None:
            print(f"  OFF:          Lz={off_lz:.2f}")
        if clamp_lz is not None:
            print(f"  CLAMP-only:   Lz={clamp_lz:.2f} {'<-- OK' if clamp_lz < 2 else '<-- HIGH'}")
        if clamp_scale_lz is not None:
            expected = args.target_lz
            diff = abs(clamp_scale_lz - expected)
            status = "GOOD" if diff < 0.2 else ("OK" if diff < 0.5 else "NEEDS TUNING")
            print(f"  CLAMP+SCALE:  Lz={clamp_scale_lz:.2f} (target={expected:.2f}) [{status}]")
        if sn_lz is not None:
            print(f"  SN-only:      Lz={sn_lz:.2f} {'<-- EXPLODED!' if sn_lz > 10 else ''}")
        if sn_scale_lz is not None:
            print(f"  SN+SCALE:     Lz={sn_scale_lz:.2f} {'<-- EXPLODED!' if sn_scale_lz > 10 else ''}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print("If CLAMP+SCALE shows Lz close to target_Lz and SN variants explode,")
    print("the operator-norm clamp fix is working correctly.")
    print()
    print("Recommendations:")
    print("  1. Use apply_opnorm_clamp_to_trm() instead of apply_spectral_norm_to_trm()")
    print("  2. Use enforce_global_contraction(..., restrict_to_reasoning_layers=True)")
    print("  3. Re-apply opnorm clamp periodically during training if weights drift")


if __name__ == "__main__":
    main()
