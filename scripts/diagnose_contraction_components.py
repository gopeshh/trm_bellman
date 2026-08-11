#!/usr/bin/env python3
"""
Ablation diagnostic comparing sampled Lz behavior across contraction components.

Tests 4 variants:
  (i) OFF: no spectral_norm, no scaling
  (ii) SN-only: spectral_norm but no scaling
  (iii) SCALE-only: scaling but no spectral_norm
  (iv) SN+SCALE: both (current behavior)

Usage:
    buck2 run //buiksat_trm:diagnose_contraction_components -- \
        --dataset /path/to/sudoku-4x4-trivial \
        --batch-size 4 \
        --num-repeats 10 \
        --target-lz 0.9 \
        --eps-list 1e-2,1e-3,1e-4
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


def count_layers(model) -> Tuple[int, int, Optional[float]]:
    """Count SN layers, scaled layers, and get sample scale value."""
    sn_count = 0
    scale_count = 0
    sample_scale = None
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            sn_count += 1
        if hasattr(module, '_inner_lip_scale'):
            scale_count += 1
            if sample_scale is None:
                sample_scale = module._inner_lip_scale.item()
    return sn_count, scale_count, sample_scale


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
        rl_latent_projection_mode="disabled",
        rl_latent_ball_radius=None,
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
    elif variant == "SN-only":
        apply_spectral_norm_to_trm(model.inner)
    elif variant == "SCALE-only":
        enforce_global_contraction(model.inner, target_lz)
    elif variant == "SN+SCALE":
        apply_spectral_norm_to_trm(model.inner)
        enforce_global_contraction(model.inner, target_lz)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Count layers
    sn_count, scale_count, sample_scale = count_layers(model)

    # Estimate Lz
    lz_mean, lz_std = estimate_Lz_stats(model, x_batch, y_batch, num_repeats, eps)

    return {
        "variant": variant,
        "eps": eps,
        "target_lz": target_lz if variant in ["SCALE-only", "SN+SCALE"] else "N/A",
        "lz_mean": lz_mean,
        "lz_std": lz_std,
        "sn_count": sn_count,
        "scale_count": scale_count,
        "sample_scale": sample_scale,
    }


def print_table(results: List[Dict], title: str = ""):
    """Print a formatted table of results."""
    if title:
        print(f"\n{title}")
    print("=" * 100)
    print(f"{'Variant':<12} {'eps':<10} {'target_Lz':<10} {'Lz_mean':<14} {'Lz_std':<12} {'Notes'}")
    print("-" * 100)
    for r in results:
        target = r["target_lz"] if r["target_lz"] != "N/A" else "N/A"
        target_str = f"{target:.2f}" if isinstance(target, float) else target
        scale_str = f"scale={r['sample_scale']:.4f}" if r['sample_scale'] else "no scale"
        notes = f"SN:{r['sn_count']}, scaled:{r['scale_count']}, {scale_str}"
        print(f"{r['variant']:<12} {r['eps']:<10.0e} {target_str:<10} {r['lz_mean']:<14.4f} {r['lz_std']:<12.4f} {notes}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Ablation diagnostic for contraction components")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Sudoku dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--target-lz", type=float, default=0.9)
    parser.add_argument("--eps-list", type=str, default="1e-3", help="Comma-separated eps values")
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    eps_values = [float(e) for e in args.eps_list.split(",")]
    variants = ["OFF", "SN-only", "SCALE-only", "SN+SCALE"]

    print("=" * 100)
    print("CONTRACTION COMPONENT ABLATION DIAGNOSTIC")
    print("=" * 100)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num repeats: {args.num_repeats}")
    print(f"Target Lz: {args.target_lz}")
    print(f"Eps values: {eps_values}")
    print()

    # Load batch
    x_batch, y_batch, vocab_size, seq_len = load_batch(args.dataset, args.batch_size)
    print(f"Loaded {args.batch_size} puzzles (seq_len={seq_len}, vocab_size={vocab_size})")

    # Build base model
    print("Building base model (contraction=OFF)...")
    base_model = build_base_model(vocab_size, seq_len, args.batch_size, args.hidden_size)

    # Check dtypes
    with torch.no_grad():
        batch = base_model._standardize_latent_batch(x_batch, y_batch)
        z0 = base_model.init_latent(x_batch, y_batch)
        print(f"z0.z_H dtype: {z0.z_H.dtype}")
        for name, param in base_model.inner.named_parameters():
            if 'weight' in name:
                print(f"First weight dtype: {param.dtype} ({name})")
                break
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
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    # Find which variants exploded (Lz > 10)
    for eps in eps_values:
        eps_results = [r for r in all_results if r["eps"] == eps]
        off_lz = next(r["lz_mean"] for r in eps_results if r["variant"] == "OFF")
        sn_lz = next(r["lz_mean"] for r in eps_results if r["variant"] == "SN-only")
        scale_lz = next(r["lz_mean"] for r in eps_results if r["variant"] == "SCALE-only")
        both_lz = next(r["lz_mean"] for r in eps_results if r["variant"] == "SN+SCALE")

        print(f"\neps={eps:.0e}:")
        print(f"  OFF:        Lz={off_lz:.2f}")
        print(f"  SN-only:    Lz={sn_lz:.2f} {'<-- EXPLODED!' if sn_lz > 10 else ''}")
        print(f"  SCALE-only: Lz={scale_lz:.2f} {'<-- EXPLODED!' if scale_lz > 10 else ''}")
        print(f"  SN+SCALE:   Lz={both_lz:.2f} {'<-- EXPLODED!' if both_lz > 10 else ''}")

        if sn_lz > 10 and scale_lz <= 10:
            print("  ==> only the sampled spectral_norm condition exceeds 10")
        elif scale_lz > 10 and sn_lz <= 10:
            print("  ==> only the sampled scaling condition exceeds 10")
        elif sn_lz > 10 and scale_lz > 10:
            print("  ==> both sampled component conditions exceed 10")
        elif both_lz > 10 and sn_lz <= 10 and scale_lz <= 10:
            print("  ==> only the sampled combined condition exceeds 10")
        else:
            print("  ==> no explosion detected")


if __name__ == "__main__":
    main()
