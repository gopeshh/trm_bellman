#!/usr/bin/env python3
"""
TEMPORARY diagnostic script comparing sampled inner local-Lz estimates with
and without spectral_norm on the VALUE HEAD.

This script:
1. Builds two models with identical settings (contraction=ON, target_Lz=0.9)
2. Model A: Keeps value_head spectral_norm intact (baseline)
3. Model B: Removes spectral_norm from value_head ONLY (ablation)
4. Measures inner Lz for both and compares

The comparison reports whether the sampled estimates differ by a configured
threshold. It does not prove independence or identify a causal effect.

Usage:
    python scripts/_tmp_diagnose_value_head_sn_effect.py \
        --dataset data/sudoku-4x4-trivial \
        --target-lz 0.9 \
        --eps 1e-3 \
        --seeds 42,123,456

DO NOT COMMIT THIS SCRIPT.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.lipschitz import estimate_local_Lz


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


def build_model(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    target_Lz: float,
    hidden_size: int = 64,
) -> TinyRecursiveReasoningModel_ACTV1:
    """Build a TRM model with contraction enabled."""
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
        rl_enable_contraction=True,  # Enables spectral_norm on value_head
        rl_target_Lz=target_Lz,
        rl_target_Lv=0.9,
        rl_enable_policy_head=True,
        rl_num_actions=num_actions,
        rl_latent_projection_mode="disabled",
        rl_latent_ball_radius=None,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TinyRecursiveReasoningModel_ACTV1(cfg_dict)
    return model


def count_spectral_norm_layers(model) -> Tuple[int, List[str]]:
    """Count and list layers with spectral norm applied."""
    sn_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            sn_layers.append(name)
    return len(sn_layers), sn_layers


def remove_spectral_norm_from_value_head(model) -> List[str]:
    """
    Remove spectral_norm from value_head ONLY.

    This uses torch.nn.utils.remove_spectral_norm() which:
    - Removes the hook
    - Replaces weight_orig with computed weight
    - Removes weight_u, weight_v buffers

    Returns list of layer names that had spectral_norm removed.
    """
    removed = []
    if model.value_head is None:
        return removed

    for name, module in model.value_head.named_modules():
        if hasattr(module, 'weight_orig'):
            # This layer has spectral_norm applied
            nn.utils.remove_spectral_norm(module)
            removed.append(f"value_head.{name}")

    return removed


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


def run_single_seed(
    dataset_path: str,
    batch_size: int,
    target_lz: float,
    eps: float,
    num_repeats: int,
    hidden_size: int,
    seed: int,
) -> Dict:
    """Run the comparison for a single seed."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load batch
    x_batch, y_batch, vocab_size, seq_len = load_batch(dataset_path, batch_size)

    # === Condition A: Baseline (value_head spectral_norm intact) ===
    model_A = build_model(vocab_size, seq_len, batch_size, target_lz, hidden_size)
    sn_count_A, sn_layers_A = count_spectral_norm_layers(model_A)
    lz_mean_A, lz_std_A = estimate_Lz_stats(model_A, x_batch, y_batch, num_repeats, eps)

    # === Condition B: Ablation (remove value_head spectral_norm) ===
    # Reset seed to get same initial weights
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_B = build_model(vocab_size, seq_len, batch_size, target_lz, hidden_size)

    # Remove spectral_norm from value_head ONLY
    removed_layers = remove_spectral_norm_from_value_head(model_B)

    sn_count_B, sn_layers_B = count_spectral_norm_layers(model_B)
    lz_mean_B, lz_std_B = estimate_Lz_stats(model_B, x_batch, y_batch, num_repeats, eps)

    return {
        "seed": seed,
        "condition_A": {
            "name": "baseline (value_head SN intact)",
            "sn_count": sn_count_A,
            "sn_layers": sn_layers_A,
            "lz_mean": lz_mean_A,
            "lz_std": lz_std_A,
        },
        "condition_B": {
            "name": "ablation (value_head SN removed)",
            "sn_count": sn_count_B,
            "sn_layers": sn_layers_B,
            "removed_layers": removed_layers,
            "lz_mean": lz_mean_B,
            "lz_std": lz_std_B,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare sampled inner Lz with value-head spectral norm on and off")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Sudoku dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--target-lz", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 100)
    print("VALUE HEAD SPECTRAL_NORM SAMPLED INNER-Lz COMPARISON")
    print("=" * 100)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num repeats: {args.num_repeats}")
    print(f"Target Lz: {args.target_lz}")
    print(f"Eps: {args.eps}")
    print(f"Seeds: {seeds}")
    print()
    print("Comparison: sampled inner local-Lz with value-head spectral_norm intact vs removed")
    print("Scope: finite seeds, one batch, and sampled perturbations; no causal or independence claim")
    print()

    all_results = []

    for seed in seeds:
        print(f"Running seed {seed}...")
        result = run_single_seed(
            args.dataset,
            args.batch_size,
            args.target_lz,
            args.eps,
            args.num_repeats,
            args.hidden_size,
            seed,
        )
        all_results.append(result)

        A = result["condition_A"]
        B = result["condition_B"]
        print(f"  Condition A (SN intact):  Lz = {A['lz_mean']:.6f} ± {A['lz_std']:.6f}  (SN layers: {A['sn_count']})")
        print(f"  Condition B (SN removed): Lz = {B['lz_mean']:.6f} ± {B['lz_std']:.6f}  (SN layers: {B['sn_count']})")
        print(f"  Removed layers: {B['removed_layers']}")
        diff = abs(A['lz_mean'] - B['lz_mean'])
        print(f"  Difference: {diff:.6f}")
        print()

    # Aggregate statistics
    lz_A_means = [r["condition_A"]["lz_mean"] for r in all_results]
    lz_B_means = [r["condition_B"]["lz_mean"] for r in all_results]

    lz_A_overall_mean = np.mean(lz_A_means)
    lz_A_overall_std = np.std(lz_A_means)
    lz_B_overall_mean = np.mean(lz_B_means)
    lz_B_overall_std = np.std(lz_B_means)

    print("=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)
    print(f"Condition A (value_head SN intact):")
    print(f"  Lz_est_mean ± std: {lz_A_overall_mean:.6f} ± {lz_A_overall_std:.6f}")
    print(f"  Lz / target_Lz ratio: {lz_A_overall_mean / args.target_lz:.4f}")
    print()
    print(f"Condition B (value_head SN removed):")
    print(f"  Lz_est_mean ± std: {lz_B_overall_mean:.6f} ± {lz_B_overall_std:.6f}")
    print(f"  Lz / target_Lz ratio: {lz_B_overall_mean / args.target_lz:.4f}")
    print()

    overall_diff = abs(lz_A_overall_mean - lz_B_overall_mean)
    threshold = 0.01  # Reporting threshold, not a significance test.

    print("=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print(f"Difference: |A - B| = {overall_diff:.6f}")

    if overall_diff < threshold:
        print(f"Difference is below the reporting threshold (< {threshold}).")
        print("No sampled difference above the threshold was detected in these runs.")
        print("This does not establish independence or absence of an effect.")
    else:
        print(f"Difference exceeds the reporting threshold ({threshold}).")
        print("These runs contain a sampled difference that warrants controlled follow-up.")
        print("The diagnostic alone does not identify its cause.")
        print()
        print("Code snippet used to remove spectral_norm from value_head:")
        print("-" * 60)
        print("""
def remove_spectral_norm_from_value_head(model):
    removed = []
    if model.value_head is None:
        return removed

    for name, module in model.value_head.named_modules():
        if hasattr(module, 'weight_orig'):
            nn.utils.remove_spectral_norm(module)
            removed.append(f"value_head.{name}")

    return removed
""")
        print("-" * 60)

    print()
    print("Raw per-seed data:")
    for r in all_results:
        print(f"  Seed {r['seed']}: A={r['condition_A']['lz_mean']:.6f}, B={r['condition_B']['lz_mean']:.6f}, diff={abs(r['condition_A']['lz_mean'] - r['condition_B']['lz_mean']):.6f}")


if __name__ == "__main__":
    main()
