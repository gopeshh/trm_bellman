#!/usr/bin/env python3
"""
Exp4 Range Test: Inference-Time Contraction Scaling

Compares finite local L_preproj estimates and mismatch metrics after applying
different z→z weight-scaling factors at evaluation time. The ≥0.10 spread
threshold is a go/no-go criterion for a larger sweep.

This is a cheap go/no-go test before committing to a full training sweep.

Usage:
    buck2 run //buiksat_trm:exp4_range_test -- \
        --checkpoint results/exp3/nc_rdis_s42/model_step_5000.pt \
        --out_dir results/paper_ready/exp4_projection_free_dial
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def load_model_and_config(
    checkpoint_path: str,
    device: str = "cpu",
    config_yaml_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model from checkpoint."""
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1,
        TinyRecursiveReasoningModel_ACTV1Config,
    )
    from rl.config import RLConfig
    import yaml

    print(f"[Load] Checkpoint: {checkpoint_path}")

    # Load YAML config if provided
    yaml_config = {}
    if config_yaml_path and Path(config_yaml_path).exists():
        with open(config_yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model_state = state_dict["model_state_dict"]
        rl_config_dict = state_dict.get("rl_config", {})
    else:
        model_state = state_dict
        rl_config_dict = {}

    # Clean state dict keys
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state[clean_key] = value

    # Infer model config
    hidden_size = 64
    if "inner.embed_inputs.weight" in cleaned_state:
        hidden_size = cleaned_state["inner.embed_inputs.weight"].shape[1]

    vocab_size = 6
    if "inner.embed_inputs.weight" in cleaned_state:
        vocab_size = cleaned_state["inner.embed_inputs.weight"].shape[0]

    seq_len = 16
    has_value_head = any("value_head" in k for k in cleaned_state.keys())
    has_policy_head = any("edit_policy" in k for k in cleaned_state.keys())
    num_actions = seq_len * vocab_size + 1

    if has_policy_head and "edit_policy.mlp.2.weight" in cleaned_state:
        num_actions = cleaned_state["edit_policy.mlp.2.weight"].shape[0]

    # Get config values with priority: YAML > checkpoint > defaults
    rl_cfg = RLConfig()

    enable_contraction = yaml_config.get(
        "enable_contraction",
        rl_config_dict.get("enable_contraction", False)
    )
    # Force explicit identity projection mode for the range test.
    latent_projection_mode = "disabled"
    latent_ball_radius = None
    target_Lz = yaml_config.get(
        "target_Lz",
        rl_config_dict.get("target_Lz", rl_cfg.target_Lz)
    )
    disable_value_head_norm = yaml_config.get(
        "disable_value_head_norm",
        rl_config_dict.get("disable_value_head_norm", True)
    )
    inner_unroll_n = yaml_config.get(
        "inner_unroll_n",
        rl_config_dict.get("inner_unroll_n", rl_cfg.inner_unroll_n)
    )

    # Build model config
    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_puzzle_identifiers=500,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        expansion=2.0,
        num_heads=max(4, hidden_size // 16),
        pos_encodings="rope",
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=has_value_head,
        rl_enable_policy_head=has_policy_head,
        rl_num_actions=num_actions if has_policy_head else 0,
        rl_enable_contraction=enable_contraction,
        rl_target_Lz=float(target_Lz),
        rl_disable_value_head_norm=bool(disable_value_head_norm),
        rl_latent_projection_mode=latent_projection_mode,
        rl_latent_ball_radius=latent_ball_radius,
    )

    model = TinyRecursiveReasoningModel_ACTV1(model_config.model_dump())
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    config_dict = {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_actions": num_actions,
        "inner_unroll_n": inner_unroll_n,
        "enable_contraction": enable_contraction,
        "target_Lz": target_Lz,
        "disable_value_head_norm": disable_value_head_norm,
        "latent_projection_mode": latent_projection_mode,
        "latent_ball_radius": latent_ball_radius,
    }

    return model, config_dict


def apply_contraction_scaling(
    model: nn.Module,
    scaling_factor: float,
) -> None:
    """
    Apply contraction scaling to z→z layers (L_level) in the model.

    This scales weights in the L_level linear layers. The sampled diagnostics
    below measure how the resulting model differs; they do not establish a
    global Lipschitz constant.
    """
    # Access inner model
    inner = getattr(model, 'inner', model)

    # Find L_level layers and scale their weights directly
    for name, module in inner.named_modules():
        if not name.startswith("L_level"):
            continue
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                module.weight.data.mul_(scaling_factor)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.mul_(scaling_factor)


def reset_contraction_scaling(model: nn.Module, original_weights: Dict[str, torch.Tensor]) -> None:
    """Restore original weights to the model."""
    inner = getattr(model, 'inner', model)

    for name, module in inner.named_modules():
        if not name.startswith("L_level"):
            continue
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in original_weights and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data.copy_(original_weights[weight_key])
        if bias_key in original_weights and hasattr(module, 'bias') and module.bias is not None:
            with torch.no_grad():
                module.bias.data.copy_(original_weights[bias_key])


def save_original_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save original weights from L_level layers."""
    inner = getattr(model, 'inner', model)
    weights = {}

    for name, module in inner.named_modules():
        if not name.startswith("L_level"):
            continue
        if hasattr(module, 'weight') and module.weight is not None:
            weights[f"{name}.weight"] = module.weight.data.clone()
        if hasattr(module, 'bias') and module.bias is not None:
            weights[f"{name}.bias"] = module.bias.data.clone()

    return weights


def estimate_L_preproj(
    model: nn.Module,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    n_steps: int = 2,
    num_samples: int = 8,
    eps: float = 1e-3,
) -> float:
    """
    Estimate L_preproj via finite differences.

    Measures ||f(z+δ) - f(z)|| / ||δ|| where f is the z→z map.
    """
    model.eval()
    norms = []

    with torch.no_grad():
        # Get latent at n_steps
        _, z_carry = model.used_value(x_batch, y_batch, n=n_steps)
        if z_carry is None:
            return 0.0

        # Get inner model and carry
        inner = getattr(model, 'inner', model)
        inner_carry = z_carry.inner_carry if hasattr(z_carry, 'inner_carry') else z_carry

        if not hasattr(inner_carry, 'z_H') or not hasattr(inner_carry, 'z_L'):
            return 0.0

        # Build context for latent_step
        input_embeddings = inner._input_embeddings(x_batch["inputs"], x_batch["puzzle_identifiers"])
        seq_info = inner.build_seq_info()

        # Get baseline output from latent_step
        baseline = inner.latent_step(inner_carry, input_embeddings, seq_info)

        # Perturb and measure Lipschitz
        for _ in range(num_samples):
            # Create random perturbations for both z_H and z_L
            noise_h = torch.randn_like(inner_carry.z_H)
            noise_l = torch.randn_like(inner_carry.z_L)

            # Normalize perturbation to have magnitude eps
            noise_norm = torch.sqrt(
                noise_h.pow(2).sum() + noise_l.pow(2).sum()
            ).clamp(min=1e-12)
            scale = eps / noise_norm

            # Create perturbed carry
            from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry
            perturbed = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=inner_carry.z_H + noise_h * scale,
                z_L=inner_carry.z_L + noise_l * scale,
            )

            # Get perturbed output
            out = inner.latent_step(perturbed, input_embeddings, seq_info)

            # Compute output difference norm
            diff = torch.sqrt(
                (out.z_H - baseline.z_H).pow(2).sum() +
                (out.z_L - baseline.z_L).pow(2).sum()
            )

            # Lipschitz estimate: ||f(z+δ) - f(z)|| / ||δ||
            norms.append(diff.item() / eps)

    if not norms:
        return 0.0

    return float(np.mean(norms))


def compute_mismatch_metrics(
    model: nn.Module,
    x_batch: Dict[str, torch.Tensor],
    y_batch: torch.Tensor,
    n_train: int = 2,
    n_eval_list: List[int] = [4, 8, 16],
) -> Dict[str, float]:
    """Compute depth mismatch metrics (ΔV, argmax agreement)."""
    model.eval()
    metrics = {}

    with torch.no_grad():
        # Get values and policies at training depth
        v_train, _ = model.used_value(x_batch, y_batch, n=n_train)
        dist_train, _ = model.policy_dist(x_batch, y_batch, n=n_train)
        probs_train = dist_train.probs
        argmax_train = probs_train.argmax(dim=-1)

        for n_eval in n_eval_list:
            # Get values and policies at eval depth
            v_eval, _ = model.used_value(x_batch, y_batch, n=n_eval)
            dist_eval, _ = model.policy_dist(x_batch, y_batch, n=n_eval)
            probs_eval = dist_eval.probs
            argmax_eval = probs_eval.argmax(dim=-1)

            # ΔV: absolute difference in value
            delta_v = (v_eval - v_train).abs().mean().item()

            # Argmax agreement
            argmax_agree = (argmax_train == argmax_eval).float().mean().item()

            suffix = f"@{n_eval}x"
            metrics[f"delta_V{suffix}"] = delta_v
            metrics[f"argmax_agree{suffix}"] = argmax_agree

    return metrics


def load_eval_batch(
    device: str = "cpu",
    num_samples: int = 50,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Load a batch of evaluation states."""
    # Try to load from existing B0 batch or generate simple states
    data_dir = PROJECT_ROOT / "data"
    trivial_dir = data_dir / "sudoku-4x4-1to4empties-trivial"

    rng = np.random.default_rng(seed)

    if trivial_dir.exists():
        train_dir = trivial_dir / "train"
        inputs_path = train_dir / "all__inputs.npy"
        if inputs_path.exists():
            all_inputs = np.load(inputs_path)
            indices = rng.choice(len(all_inputs), size=min(num_samples, len(all_inputs)), replace=False)
            inputs = torch.tensor(all_inputs[indices], dtype=torch.long, device=device)
            plans = inputs.clone()  # Initial plan = puzzle state
            puzzle_ids = torch.arange(len(inputs), device=device)

            x_batch = {
                "inputs": inputs,
                "puzzle_identifiers": puzzle_ids,
            }
            return x_batch, plans

    # Fallback: generate random valid-ish inputs
    print("[Warning] No dataset found, using random inputs")
    inputs = torch.randint(2, 6, (num_samples, 16), device=device)
    # Add some empty cells (token 1)
    mask = torch.rand(num_samples, 16) < 0.25
    inputs[mask] = 1
    plans = inputs.clone()
    puzzle_ids = torch.arange(num_samples, device=device)

    return {"inputs": inputs, "puzzle_identifiers": puzzle_ids}, plans


def run_range_test(
    checkpoint_path: str,
    config_yaml_path: Optional[str] = None,
    out_dir: str = "results/paper_ready/exp4_projection_free_dial",
    scaling_factors: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_samples: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the range test."""
    print("=" * 60)
    print("Exp4 Range Test: Inference-Time Contraction Scaling")
    print("=" * 60)
    print()

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model_and_config(checkpoint_path, device, config_yaml_path)
    print(f"Model config: {config}")
    print()

    # Load eval batch
    x_batch, y_batch = load_eval_batch(device, num_samples, seed)
    print(f"Loaded {len(y_batch)} evaluation samples")
    print()

    # Save original weights for restoring between scaling factors
    original_weights = save_original_weights(model)
    print(f"Saved {len(original_weights)} weight tensors for restoration")
    print()

    # Run scaling sweep
    results = []
    print("Scaling sweep:")
    print("-" * 60)
    print(f"{'Scale':>8} {'L_preproj':>10} {'ΔV@8x':>10} {'Argmax@8x':>10}")
    print("-" * 60)

    for scale in scaling_factors:
        # Restore original weights, then apply scaling
        reset_contraction_scaling(model, original_weights)
        if scale < 1.0:
            apply_contraction_scaling(model, scale)

        # Estimate L_preproj
        L_preproj = estimate_L_preproj(model, x_batch, y_batch, n_steps=2)

        # Compute mismatch metrics
        mismatch = compute_mismatch_metrics(model, x_batch, y_batch, n_train=2)

        result = {
            "scaling_factor": scale,
            "L_preproj": L_preproj,
            **mismatch,
        }
        results.append(result)

        print(f"{scale:>8.2f} {L_preproj:>10.4f} {mismatch['delta_V@8x']:>10.4f} {mismatch['argmax_agree@8x']:>10.4f}")

    print("-" * 60)
    print()

    # Compute gate metrics
    L_preproj_values = [r["L_preproj"] for r in results]
    L_preproj_spread = max(L_preproj_values) - min(L_preproj_values)

    # G0 follows from the explicit identity projection mode used at construction.
    g0_passed = (
        config.get("latent_projection_mode") == "disabled"
        and config.get("latent_ball_radius") is None
    )

    # G1: stability (no NaN)
    g1_passed = bool(all(not np.isnan(r["L_preproj"]) for r in results))

    # G2: dial range
    g2_threshold = 0.10
    g2_passed = bool(L_preproj_spread >= g2_threshold)

    # G3: monotonic linkage (Spearman correlation)
    try:
        from scipy.stats import spearmanr
        scales = [r["scaling_factor"] for r in results]
        delta_vs = [r["delta_V@8x"] for r in results]
        rho, p_value = spearmanr(L_preproj_values, delta_vs)
        if np.isnan(rho):
            rho = 0.0
            g3_passed = False
        else:
            g3_passed = bool(abs(rho) > 0.5)
        g3_status = "PASS" if g3_passed else "INCONCLUSIVE"
        rho = float(rho)
        p_value = float(p_value) if p_value is not None and not np.isnan(p_value) else None
    except ImportError:
        # Fallback: simple correlation check
        rho = np.corrcoef(L_preproj_values, [r["delta_V@8x"] for r in results])[0, 1]
        if np.isnan(rho):
            rho = 0.0
        rho = float(rho)
        p_value = None
        g3_passed = bool(abs(rho) > 0.5)
        g3_status = "PASS" if g3_passed else "INCONCLUSIVE"

    # Print gate results
    print("DECISION GATES")
    print("-" * 60)
    print(f"G0 (Projection inactive): {'PASS' if g0_passed else 'FAIL'} (explicit disabled mode)")
    print(f"G1 (Training stability):  {'PASS' if g1_passed else 'FAIL'}")
    print(f"G2 (Dial range ≥0.10):    {'PASS' if g2_passed else 'FAIL'} (spread={L_preproj_spread:.4f})")
    print(f"G3 (Monotonic linkage):   {g3_status} (ρ={rho:.4f})")
    print("-" * 60)
    print()

    # Overall decision
    if not g2_passed:
        decision = "NEGATIVE: Observed dial range is below the configured threshold"
        claim = "For this checkpoint and sampled states, the observed L_preproj spread is below 0.10."
    elif g3_passed:
        decision = "POSITIVE: Sampled range and association criteria met"
        claim = (
            "For this checkpoint and sampled states, the observed L_preproj spread is at least 0.10 "
            "and is monotonically associated with the recorded mismatch metric under the configured criterion."
        )
    else:
        decision = "INCONCLUSIVE: Observed range met threshold without clear monotonicity"
        claim = (
            "For this checkpoint and sampled states, the observed L_preproj spread is at least 0.10, "
            "but the configured monotonic-association criterion was not met."
        )

    print(f"DECISION: {decision}")
    print()

    # Build summary
    summary = {
        "experiment": "Exp4",
        "description": "Projection-free contraction dial range test",
        "generated_at": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "config": config,
        "scaling_factors": scaling_factors,
        "results": results,
        "gates": {
            "g0_projection_inactive": {"passed": g0_passed, "rate": 0.0},
            "g1_stability": {"passed": g1_passed},
            "g2_dial_range": {
                "passed": g2_passed,
                "threshold": g2_threshold,
                "L_preproj_min": min(L_preproj_values),
                "L_preproj_max": max(L_preproj_values),
                "L_preproj_spread": L_preproj_spread,
            },
            "g3_monotonic_linkage": {
                "passed": g3_passed,
                "status": g3_status,
                "spearman_rho": float(rho),
                "p_value": float(p_value) if p_value is not None else None,
            },
        },
        "decision": decision,
        "claim": claim,
        "proceed_to_full_sweep": g2_passed,
        "git_sha": get_git_sha(),
    }

    # Save summary
    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Generate CLAIMS.md
    claims_content = f"""# Exp4 Claims: Projection-free Contraction Dial (Range Test)

**Generated:** {summary['generated_at']}

## Summary

- **Decision:** {decision}
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** {'PASS' if g1_passed else 'FAIL'}
- **G2 (Dial range ≥0.10):** {'PASS' if g2_passed else 'FAIL'} (spread={L_preproj_spread:.4f})
- **G3 (Monotonic linkage):** {g3_status} (ρ={rho:.4f})

## Results by Scaling Factor

| Scale | L_preproj | ΔV@8x | Argmax@8x |
|-------|-----------|-------|-----------|
"""
    for r in results:
        claims_content += f"| {r['scaling_factor']:.2f} | {r['L_preproj']:.4f} | {r['delta_V@8x']:.4f} | {r['argmax_agree@8x']:.4f} |\n"

    claims_content += f"""
## Scoped Claim

{claim}

## Scope Limitations

- Results from inference-time scaling only (no retraining)
- Single checkpoint (nc_rdis_s42) from Exp3
- Trivial 4×4 Sudoku suite
- {num_samples} evaluation samples
- Finite local estimates do not establish causality or a global Lipschitz bound

## Proceed to Full Sweep?

{"YES - G2 met the sampled spread threshold" if g2_passed else "NO - G2 did not meet the sampled spread threshold"}
"""

    claims_path = out_path / "CLAIMS.md"
    with open(claims_path, "w") as f:
        f.write(claims_content)
    print(f"Saved: {claims_path}")

    # Generate PROVENANCE.md
    provenance_content = f"""# Exp4 Provenance: Range Test

**Generated:** {summary['generated_at']}
**Git SHA:** {summary['git_sha']}

## Checkpoint

- **Path:** {checkpoint_path}
- **Config source:** {config.get('config_source', 'inferred')}

## Evaluation Parameters

- **Scaling factors:** {scaling_factors}
- **Num samples:** {num_samples}
- **Seed:** {seed}
- **Device:** {device}

## Gate Thresholds

- **G2 (Dial range):** L_preproj spread ≥ 0.10
- **G3 (Monotonic linkage):** |Spearman ρ| > 0.5

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp4_range_test -- \\
    --checkpoint {checkpoint_path} \\
    --out_dir {out_dir}
```
"""

    provenance_path = out_path / "PROVENANCE.md"
    with open(provenance_path, "w") as f:
        f.write(provenance_content)
    print(f"Saved: {provenance_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp4 Range Test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/exp3/nc_rdis_s42/model_step_5000.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to YAML config used during training",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/paper_ready/exp4_projection_free_dial",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of evaluation samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    summary = run_range_test(
        checkpoint_path=args.checkpoint,
        config_yaml_path=args.config_yaml,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Exit with code based on G2 gate
    if not summary["gates"]["g2_dial_range"]["passed"]:
        print("\nExiting with code 1 (G2 failed - dial has insufficient range)")
        sys.exit(1)


if __name__ == "__main__":
    main()
