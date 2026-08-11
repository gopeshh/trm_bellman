#!/usr/bin/env python3
"""
Diagnostic script comparing finite sampled local-Lz estimates.

Reports estimates from separately initialized contraction-ON and contraction-OFF
models. This is not a controlled causal comparison or a global Lipschitz bound.

Usage:
    buck2 run //buiksat_trm:diagnose_contraction_lipschitz -- \
        --dataset /path/to/sudoku-4x4-trivial \
        --batch-size 4 \
        --num-repeats 10 \
        --target-lz 0.9
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.lipschitz import estimate_local_Lz


def load_batch(
    dataset_path: str, batch_size: int = 4
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int, int]:
    """Load a small batch of puzzles from the dataset."""
    split_path = Path(dataset_path) / "train"
    inputs = np.load(split_path / "all__inputs.npy")

    # Take first batch_size puzzles
    inputs = inputs[:batch_size]
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)

    # Load metadata for vocab_size and seq_len
    with open(split_path / "dataset.json") as f:
        meta = json.load(f)

    vocab_size = meta.get("vocab_size", 6)
    seq_len = meta.get("seq_len", 16)

    # Create batch dict
    x_batch = {
        "inputs": inputs_tensor,
        "puzzle_identifiers": torch.arange(batch_size, dtype=torch.long),
    }

    # Use inputs as initial plan (empty cells filled with 1 = empty token)
    y_batch = inputs_tensor.clone()

    return x_batch, y_batch, vocab_size, seq_len


def count_spectral_norm_layers(model):
    """Count layers with spectral norm applied."""
    sn_count = 0
    scale_count = 0
    sn_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            sn_count += 1
            sn_layers.append(name)
        if hasattr(module, '_inner_lip_scale'):
            scale_count += 1
    return sn_count, scale_count, sn_layers


def build_model(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    enable_contraction: bool,
    target_Lz: float,
    hidden_size: int = 64,
) -> TinyRecursiveReasoningModel_ACTV1:
    """Build a TRM model with given contraction settings."""
    # Determine number of actions (for 4x4 Sudoku)
    num_actions = seq_len * vocab_size + 1  # +1 for STOP action

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
        rl_enable_contraction=enable_contraction,
        rl_target_Lz=target_Lz,
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
    num_samples_per_repeat: int = 4,
) -> Tuple[float, float]:
    """Estimate Lz mean and std over multiple repeats."""
    model.eval()

    with torch.no_grad():
        # Standardize batch and initialize latent
        batch = model._standardize_latent_batch(x_batch, y_batch)
        z0 = model.init_latent(x_batch, y_batch)

        # Get context for latent_step
        context = model._resolve_latent_context(batch)

        # Warmup forward passes to advance spectral-norm power iteration.
        for _ in range(5):
            _ = model.inner.latent_step(z0, context["input_embeddings_with_plan"], context["seq_info"])

        # Debug: Check output norms
        baseline = model.inner.latent_step(z0, context["input_embeddings_with_plan"], context["seq_info"])
        base_norm = torch.sqrt(baseline.z_H.pow(2).sum() + baseline.z_L.pow(2).sum()).item()
        z0_norm = torch.sqrt(z0.z_H.pow(2).sum() + z0.z_L.pow(2).sum()).item()
        print(f"    z0 norm: {z0_norm:.4f}, baseline output norm: {base_norm:.4f}")

        # Debug: Manual Lz check with one perturbation
        eps = 1e-3
        noise_h = torch.randn_like(z0.z_H)
        noise_l = torch.randn_like(z0.z_L)
        noise_norm = torch.sqrt(noise_h.pow(2).sum() + noise_l.pow(2).sum())
        scale = eps / noise_norm
        perturbed = z0.__class__(
            z_H=z0.z_H + noise_h * scale,
            z_L=z0.z_L + noise_l * scale,
        )
        perturbed_out = model.inner.latent_step(perturbed, context["input_embeddings_with_plan"], context["seq_info"])
        diff_norm = torch.sqrt((perturbed_out.z_H - baseline.z_H).pow(2).sum() + (perturbed_out.z_L - baseline.z_L).pow(2).sum()).item()
        manual_lz = diff_norm / eps
        print(f"    Manual Lz check: diff_norm={diff_norm:.6f}, eps={eps}, Lz={manual_lz:.4f}")

        # Collect Lz estimates over repeats
        lz_values = []
        for _ in range(num_repeats):
            lz_mean = estimate_local_Lz(
                inner_model=model.inner,
                carry=z0,
                context=context,
                num_samples=num_samples_per_repeat,
                eps=1e-3,
                return_samples=False,
            )
            lz_values.append(lz_mean)

        lz_array = np.array(lz_values)
        return float(lz_array.mean()), float(lz_array.std())


def main():
    parser = argparse.ArgumentParser(description="Compare sampled local Lz estimates")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Sudoku dataset")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num-repeats", type=int, default=10, help="Number of Lz estimation repeats")
    parser.add_argument("--target-lz", type=float, default=0.9, help="Target Lz for contraction ON")
    parser.add_argument("--hidden-size", type=int, default=64, help="Model hidden size")
    args = parser.parse_args()

    print(f"=== Contraction Lipschitz Diagnostic ===")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num repeats: {args.num_repeats}")
    print(f"Target Lz: {args.target_lz}")
    print()

    # Load batch
    x_batch, y_batch, vocab_size, seq_len = load_batch(args.dataset, args.batch_size)
    print(f"Loaded {args.batch_size} puzzles (seq_len={seq_len}, vocab_size={vocab_size})")
    print()

    results = []

    # Test contraction OFF
    print("Building model with contraction OFF...")
    model_off = build_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=args.batch_size,
        enable_contraction=False,
        target_Lz=args.target_lz,
        hidden_size=args.hidden_size,
    )
    sn_off, scale_off, _ = count_spectral_norm_layers(model_off)
    print(f"  Layers with spectral_norm: {sn_off}, with _inner_lip_scale: {scale_off}")
    lz_mean_off, lz_std_off = estimate_Lz_stats(
        model_off, x_batch, y_batch, args.num_repeats
    )
    results.append({
        "setting": "contraction=OFF",
        "target_Lz": "N/A",
        "Lz_est_mean": lz_mean_off,
        "Lz_est_std": lz_std_off,
        "notes": "No spectral norm, no scaling",
    })
    print(f"  Lz_est: {lz_mean_off:.4f} ± {lz_std_off:.4f}")
    print()

    # Test contraction ON with primary target
    print(f"Building model with contraction ON (target_Lz={args.target_lz})...")
    model_on = build_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=args.batch_size,
        enable_contraction=True,
        target_Lz=args.target_lz,
        hidden_size=args.hidden_size,
    )
    sn_on, scale_on, sn_layers = count_spectral_norm_layers(model_on)
    print(f"  Layers with spectral_norm: {sn_on}, with _inner_lip_scale: {scale_on}")
    print(f"  SN layers: {sn_layers[:3]}...") if len(sn_layers) > 3 else print(f"  SN layers: {sn_layers}")

    # Print scale values
    for name, module in model_on.named_modules():
        if hasattr(module, '_inner_lip_scale'):
            print(f"    {name}: scale={module._inner_lip_scale.item():.6f}")
            break  # Just print first one as sample

    lz_mean_on, lz_std_on = estimate_Lz_stats(
        model_on, x_batch, y_batch, args.num_repeats
    )
    results.append({
        "setting": "contraction=ON",
        "target_Lz": args.target_lz,
        "Lz_est_mean": lz_mean_on,
        "Lz_est_std": lz_std_on,
        "notes": "spectral_norm + scaling",
    })
    print(f"  Lz_est: {lz_mean_on:.4f} ± {lz_std_on:.4f}")
    print()

    # Test contraction ON with stricter target (0.5)
    stricter_target = 0.5
    print(f"Building model with contraction ON (target_Lz={stricter_target})...")
    model_strict = build_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=args.batch_size,
        enable_contraction=True,
        target_Lz=stricter_target,
        hidden_size=args.hidden_size,
    )
    lz_mean_strict, lz_std_strict = estimate_Lz_stats(
        model_strict, x_batch, y_batch, args.num_repeats
    )
    results.append({
        "setting": "contraction=ON",
        "target_Lz": stricter_target,
        "Lz_est_mean": lz_mean_strict,
        "Lz_est_std": lz_std_strict,
        "notes": "spectral_norm + scaling (stricter)",
    })
    print(f"  Lz_est: {lz_mean_strict:.4f} ± {lz_std_strict:.4f}")
    print()

    # Print summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Setting':<20} {'target_Lz':<12} {'Lz_est_mean':<14} {'Lz_est_std':<12} {'Notes'}")
    print("-" * 80)
    for r in results:
        target = r["target_Lz"] if r["target_Lz"] != "N/A" else "N/A"
        target_str = f"{target:.2f}" if isinstance(target, float) else target
        print(f"{r['setting']:<20} {target_str:<12} {r['Lz_est_mean']:<14.4f} {r['Lz_est_std']:<12.4f} {r['notes']}")
    print("=" * 80)
    print()

    # Interpretation
    print("INTERPRETATION:")
    if lz_mean_on < lz_mean_off:
        print(f"  Contraction-ON run has the lower sampled mean: {lz_mean_off:.4f} → {lz_mean_on:.4f}")
    else:
        print(f"  Contraction-ON run does not have the lower sampled mean: {lz_mean_off:.4f} → {lz_mean_on:.4f}")

    if lz_mean_strict < lz_mean_on:
        print(f"  Stricter-target run has the lower sampled mean: {lz_mean_on:.4f} → {lz_mean_strict:.4f}")
    else:
        print(f"  Stricter-target run does not have the lower sampled mean: {lz_mean_on:.4f} → {lz_mean_strict:.4f}")

    # Check if measured Lz roughly tracks target
    ratio = lz_mean_on / args.target_lz if args.target_lz > 0 else float('inf')
    print(f"  Sampled local-Lz mean / target_Lz ratio: {ratio:.2f}")

    if lz_mean_on < 1.0:
        print("  Sampled contraction-ON local-Lz mean is below 1.0.")
    else:
        print("  Sampled contraction-ON local-Lz mean is at least 1.0.")
    print("  Models were initialized separately, so differences do not identify a causal effect.")
    print("  Finite local estimates below 1.0 do not establish a global contraction bound.")


if __name__ == "__main__":
    main()
