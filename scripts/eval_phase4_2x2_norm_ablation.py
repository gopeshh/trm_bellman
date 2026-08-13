#!/usr/bin/env python3
"""
Phase 4: 2x2 Norm Ablation Evaluation Script.

Evaluates all 12 checkpoints (4 conditions × 3 seeds) and generates:
- schema-v2 summary.json with measured per-condition/per-seed metrics
- CLAIMS.md, PROVENANCE.md

Metrics:
- Stability: Var(V), L̂_z (pre-proj), projection_active_rate

This script does not run an environment rollout or load training history. The
schema records the corresponding success, final-loss, and training-history
metrics as unavailable instead of emitting placeholder values.

Usage:
    buck2 run //buiksat_trm:eval_phase4_2x2_norm_ablation -- \
        --out_dir results/paper_ready/phase4_2x2_norm_ablation/v2
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_result_schema import (  # noqa: E402
    PHASE4_SCHEMA_VERSION,
    phase4_metric_availability,
    write_phase4_summary,
)


# =============================================================================
# Configuration
# =============================================================================

CONDITIONS = ["nc_nv", "nc_yv", "yc_nv", "yc_yv"]
CONDITION_LABELS = {
    "nc_nv": "C-OFF, V-OFF",
    "nc_yv": "C-OFF, V-ON",
    "yc_nv": "C-ON, V-OFF",
    "yc_yv": "C-ON, V-ON",
}
SEEDS = [41, 42, 43]
N_TRAIN = 2


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ConditionResult:
    """Results for a single condition-seed pair."""
    condition: str
    seed: int
    checkpoint_path: str
    # Config verification
    enable_contraction: bool
    disable_value_head_norm: bool
    latent_projection_mode: str
    latent_ball_radius: Optional[float]
    # Lipschitz
    L_preproj: float
    L_preproj_std: float
    # Stability metrics
    var_V: float
    projection_active_rate: float
    argmax_agreement_4x: float
    argmax_agreement_8x: float
    delta_V_4x: float
    delta_V_8x: float


@dataclass
class ConditionAggregate:
    """Aggregated results for a condition across seeds."""
    condition: str
    label: str
    enable_contraction: bool
    disable_value_head_norm: bool
    latent_projection_mode: str
    latent_ball_radius: Optional[float]
    n_seeds: int
    # Lipschitz
    L_preproj_mean: float
    L_preproj_std: float
    # Stability
    var_V_mean: float
    var_V_std: float
    projection_active_rate_mean: float
    argmax_4x_mean: float
    argmax_4x_std: float
    argmax_8x_mean: float
    argmax_8x_std: float
    delta_V_4x_mean: float
    delta_V_8x_mean: float


# =============================================================================
# Model Loading
# =============================================================================

def load_checkpoint(ckpt_path: str, config_path: str, device: str = "cuda"):
    """Load model from checkpoint with config.

    Uses rl_checkpoint_step_*.pt which contains both model weights and rl_config.
    Falls back to loading model_step_*.pt and constructing config from YAML.
    """
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    # Load YAML config for ablation settings
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Check for rl_checkpoint which has the full config
    rl_ckpt_path = ckpt_path.replace("model_step_", "rl_checkpoint_step_")

    if os.path.exists(rl_ckpt_path):
        ckpt = torch.load(rl_ckpt_path, map_location=device, weights_only=False)
        saved_rl_config = ckpt.get("rl_config", {})
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_rl_config = {}

    # Use saved rl_config if available, otherwise construct from YAML and defaults
    # Key parameters from saved config
    seq_len = saved_rl_config.get("max_edits", 16)
    vocab_size = 32  # The model was trained with extended vocab for position info
    hidden_size = saved_rl_config.get("hidden_size", 64)
    rl_num_actions = saved_rl_config.get("num_actions", 513)  # Trained with 513 actions

    projection_mode = yaml_config.get("latent_projection_mode", "enabled")
    if projection_mode == "enabled":
        projection_radius = yaml_config.get("latent_ball_radius", 10.0)
        if (
            projection_radius is None
            or projection_radius <= 0.0
            or not np.isfinite(projection_radius)
        ):
            raise ValueError("Enabled projection requires latent_ball_radius > 0")
    elif projection_mode == "disabled":
        projection_radius = yaml_config.get("latent_ball_radius")
        if projection_radius is not None:
            raise ValueError("Disabled projection requires latent_ball_radius=None")
    else:
        raise ValueError("latent_projection_mode must be 'enabled' or 'disabled'")

    # Return the normalized pair so serialized results expose canonical state.
    yaml_config = dict(yaml_config)
    yaml_config["latent_projection_mode"] = projection_mode
    yaml_config["latent_ball_radius"] = projection_radius

    # Build TRM config dict (matching upi_trm_train.py structure)
    trm_cfg_dict = dict(
        batch_size=32,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=500,
        vocab_size=vocab_size,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,  # Only 1 L layer (matching the checkpoint)
        hidden_size=hidden_size,
        expansion=2.0,
        num_heads=max(4, hidden_size // 16),
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=yaml_config.get("enable_contraction", False),
        rl_target_Lz=yaml_config.get("target_Lz", 0.9),
        rl_target_Lv=yaml_config.get("target_Lv", 1.0),
        rl_disable_value_head_norm=yaml_config.get("disable_value_head_norm", True),
        rl_enable_policy_head=True,
        rl_num_actions=rl_num_actions,
        rl_latent_projection_mode=projection_mode,
        rl_latent_ball_radius=projection_radius,
    )

    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(trm_cfg_dict).to(device)

    # Load weights with strict=False to allow some mismatches
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model, yaml_config


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_lipschitz(model, states: List[Dict], n_steps: int = 2, device: str = "cuda") -> Tuple[float, float]:
    """Compute L_preproj using finite differences.

    Uses the correct model API: model.used_value(x, y, n) and inner.latent_step().
    """
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry

    model.eval()
    inner = model.inner
    all_estimates = []
    eps = 0.01

    with torch.no_grad():
        for state in states[:50]:
            x = {
                "inputs": torch.tensor(state["inputs"], dtype=torch.long, device=device).unsqueeze(0),
                "puzzle_identifiers": torch.tensor([0], dtype=torch.long, device=device),
            }
            y = torch.tensor(state.get("plan", state["inputs"]), dtype=torch.long, device=device).unsqueeze(0)

            # Get latent after n_steps
            _, z_carry = model.used_value(x, y, n=n_steps)
            if z_carry is None:
                continue

            # Get inner model and carry (handle different model structures)
            inner = getattr(model, 'inner', model)
            inner_carry = z_carry.inner_carry if hasattr(z_carry, 'inner_carry') else z_carry

            if not hasattr(inner_carry, 'z_H'):
                continue

            # Get input embeddings for latent step
            input_embeddings = inner._input_embeddings(x["inputs"], x["puzzle_identifiers"])
            seq_info = inner.build_seq_info()

            # Baseline step
            baseline = inner.latent_step(inner_carry, input_embeddings, seq_info)

            # Perturbed step
            for _ in range(3):  # Multiple perturbation trials
                noise_h = torch.randn_like(inner_carry.z_H)
                noise_l = torch.randn_like(inner_carry.z_L)
                scale = eps / max(
                    noise_h.norm().item(),
                    noise_l.norm().item(),
                    1e-8,
                )

                perturbed = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=inner_carry.z_H + noise_h * scale,
                    z_L=inner_carry.z_L + noise_l * scale,
                )
                out = inner.latent_step(perturbed, input_embeddings, seq_info)

                diff = torch.sqrt(
                    (out.z_H - baseline.z_H).pow(2).sum() +
                    (out.z_L - baseline.z_L).pow(2).sum()
                )
                all_estimates.append(diff.item() / eps)

    if not all_estimates:
        return 0.0, 0.0
    return float(np.mean(all_estimates)), float(np.std(all_estimates))


def compute_stability_metrics(
    model, states: List[Dict], n_train: int = 2, device: str = "cuda"
) -> Dict[str, float]:
    """Compute stability metrics at depth mismatch."""
    model.eval()
    results = {"var_V": 0.0, "argmax_4x": 0.0, "argmax_8x": 0.0, "delta_V_4x": 0.0, "delta_V_8x": 0.0}

    V_train_all = []
    agreements_4x = []
    agreements_8x = []
    delta_V_4x = []
    delta_V_8x = []

    with torch.no_grad():
        for state in states[:100]:
            y = torch.tensor(state.get("plan", state["inputs"]), dtype=torch.long, device=device).unsqueeze(0)
            x = {
                "inputs": torch.tensor(state["inputs"], dtype=torch.long, device=device).unsqueeze(0),
                "puzzle_identifiers": torch.tensor([0], dtype=torch.long, device=device),
                "plan": y,  # Add plan to batch for _resolve_latent_context
            }

            # Get value and policy at training depth
            V_train, carry_train = model.used_value(x, y, n=n_train)
            if V_train is None:
                continue
            V_train_all.append(V_train.item())

            # Get policy at training depth
            if model.edit_policy is not None:
                # Get embeddings for policy head
                latent_ctx = model._resolve_latent_context(x)
                # Get inner carry (handle different model structures)
                inner_carry_train = carry_train.inner_carry if hasattr(carry_train, 'inner_carry') else carry_train
                z_flat = inner_carry_train.z_L.flatten(start_dim=1) if hasattr(inner_carry_train, 'z_L') else None
                if z_flat is None:
                    continue
                x_embed_flat = latent_ctx["input_embeddings"].flatten(start_dim=1)
                y_embed_flat = model.encode_plan(y, x).flatten(start_dim=1)
                dist_train = model.edit_policy(z_flat, x_embed_flat, y_embed_flat)
                action_train = dist_train.logits.argmax(dim=-1)

                # Compare at different depths
                for n_eval, key_suffix in [(4, "4x"), (8, "8x")]:
                    V_eval, carry_eval = model.used_value(x, y, n=n_eval)
                    if V_eval is not None:
                        inner_carry_eval = carry_eval.inner_carry if hasattr(carry_eval, 'inner_carry') else carry_eval
                        z_flat_eval = inner_carry_eval.z_L.flatten(start_dim=1) if hasattr(inner_carry_eval, 'z_L') else None
                        if z_flat_eval is None:
                            continue
                        dist_eval = model.edit_policy(z_flat_eval, x_embed_flat, y_embed_flat)
                        action_eval = dist_eval.logits.argmax(dim=-1)

                        if key_suffix == "4x":
                            agreements_4x.append((action_train == action_eval).float().item())
                            delta_V_4x.append(abs(V_train.item() - V_eval.item()))
                        else:
                            agreements_8x.append((action_train == action_eval).float().item())
                            delta_V_8x.append(abs(V_train.item() - V_eval.item()))

    if V_train_all:
        results["var_V"] = float(np.var(V_train_all))
    if agreements_4x:
        results["argmax_4x"] = float(np.mean(agreements_4x))
    if agreements_8x:
        results["argmax_8x"] = float(np.mean(agreements_8x))
    if delta_V_4x:
        results["delta_V_4x"] = float(np.mean(delta_V_4x))
    if delta_V_8x:
        results["delta_V_8x"] = float(np.mean(delta_V_8x))

    return results


def compute_projection_rate(model, states: List[Dict], n_steps: int = 2, device: str = "cuda") -> float:
    """Compute next-step activation using production projection instrumentation."""
    model.eval()
    if not hasattr(model, "config"):
        raise ValueError("Model config is required for projection diagnostics")
    projection_mode = model.config.rl_latent_projection_mode
    radius = model.config.rl_latent_ball_radius
    if projection_mode == "disabled":
        if radius is not None:
            raise ValueError("Disabled projection must not have a radius")
        return 0.0
    if (
        projection_mode != "enabled"
        or radius is None
        or radius <= 0.0
        or not np.isfinite(radius)
    ):
        raise ValueError("Enabled projection requires a positive finite radius")

    active_count = 0
    total_count = 0

    with torch.no_grad():
        for state in states[:50]:
            x = {
                "inputs": torch.tensor(state["inputs"], dtype=torch.long, device=device).unsqueeze(0),
                "puzzle_identifiers": torch.tensor([0], dtype=torch.long, device=device),
            }
            y = torch.tensor(state.get("plan", state["inputs"]), dtype=torch.long, device=device).unsqueeze(0)

            _, carry = model.used_value(x, y, n=n_steps)
            if carry is None:
                continue

            inner_carry = carry.inner_carry if hasattr(carry, 'inner_carry') else carry
            if not hasattr(inner_carry, 'z_H'):
                continue

            batch = model._standardize_latent_batch(x, y)
            context = model._resolve_latent_context(batch)
            input_embeddings = (
                context["input_embeddings_with_plan"]
                if "input_embeddings_with_plan" in context
                else context["input_embeddings"]
            )
            _, _, projection_active = model.inner.latent_step_with_projection_info(
                inner_carry,
                input_embeddings,
                context["seq_info"],
            )
            active_count += projection_active.sum().item()
            total_count += projection_active.numel()

    return active_count / total_count if total_count > 0 else 0.0


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_condition(
    condition: str,
    seed: int,
    checkpoint_dir: Path,
    config_dir: Path,
    data_dir: Path,
    device: str = "cuda",
) -> Optional[ConditionResult]:
    """Evaluate a single condition-seed pair."""
    ckpt_path = checkpoint_dir / f"{condition}_s{seed}" / "model_step_5000.pt"
    config_path = config_dir / f"{condition}.yaml"

    if not ckpt_path.exists():
        print(f"  [Skip] Checkpoint not found: {ckpt_path}")
        return None

    print(f"  Evaluating {condition} seed={seed}...")

    # Load model and config
    model, config = load_checkpoint(str(ckpt_path), str(config_path), device)

    # Load the inputs used by the finite checkpoint diagnostics. This is not an
    # environment rollout and therefore does not produce a success metric.
    trivial_puzzles = []

    for dataset_name in ["sudoku-4x4-trivial"]:
        dataset_path = data_dir / dataset_name
        if not dataset_path.exists():
            continue

        for split in ["train", "test"]:
            split_dir = dataset_path / split
            if not split_dir.exists():
                continue
            inputs_path = split_dir / "all__inputs.npy"
            if not inputs_path.exists():
                continue

            inputs = np.load(inputs_path)
            pids_path = split_dir / "all__puzzle_identifiers.npy"
            puzzle_ids = np.load(pids_path) if pids_path.exists() else np.arange(len(inputs))

            for i in range(min(len(inputs), 50)):
                puzzle = {
                    "inputs": inputs[i].astype(np.int64).tolist(),
                    "puzzle_identifier": int(puzzle_ids[i]),
                    "plan": inputs[i].astype(np.int64).tolist(),
                }
                trivial_puzzles.append(puzzle)

    if not trivial_puzzles:
        print(f"  [Skip] No puzzles found in {data_dir}")
        return None

    # Compute metrics using list of puzzle dicts
    L_preproj, L_preproj_std = compute_lipschitz(model, trivial_puzzles, N_TRAIN, device)
    stability = compute_stability_metrics(model, trivial_puzzles, N_TRAIN, device)
    proj_rate = compute_projection_rate(model, trivial_puzzles, N_TRAIN, device)

    return ConditionResult(
        condition=condition,
        seed=seed,
        checkpoint_path=str(ckpt_path),
        enable_contraction=config.get("enable_contraction", False),
        disable_value_head_norm=config.get("disable_value_head_norm", True),
        latent_projection_mode=config["latent_projection_mode"],
        latent_ball_radius=config["latent_ball_radius"],
        L_preproj=L_preproj,
        L_preproj_std=L_preproj_std,
        var_V=stability["var_V"],
        projection_active_rate=proj_rate,
        argmax_agreement_4x=stability["argmax_4x"],
        argmax_agreement_8x=stability["argmax_8x"],
        delta_V_4x=stability["delta_V_4x"],
        delta_V_8x=stability["delta_V_8x"],
    )


def aggregate_results(results: List[ConditionResult]) -> List[ConditionAggregate]:
    """Aggregate results by condition."""
    aggregates = []

    for condition in CONDITIONS:
        cond_results = [r for r in results if r.condition == condition]
        if not cond_results:
            continue

        n = len(cond_results)
        aggregates.append(ConditionAggregate(
            condition=condition,
            label=CONDITION_LABELS[condition],
            enable_contraction=cond_results[0].enable_contraction,
            disable_value_head_norm=cond_results[0].disable_value_head_norm,
            latent_projection_mode=cond_results[0].latent_projection_mode,
            latent_ball_radius=cond_results[0].latent_ball_radius,
            n_seeds=n,
            L_preproj_mean=float(np.mean([r.L_preproj for r in cond_results])),
            L_preproj_std=float(np.std([r.L_preproj for r in cond_results])),
            var_V_mean=float(np.mean([r.var_V for r in cond_results])),
            var_V_std=float(np.std([r.var_V for r in cond_results])),
            projection_active_rate_mean=float(
                np.mean([r.projection_active_rate for r in cond_results])
            ),
            argmax_4x_mean=float(
                np.mean([r.argmax_agreement_4x for r in cond_results])
            ),
            argmax_4x_std=float(
                np.std([r.argmax_agreement_4x for r in cond_results])
            ),
            argmax_8x_mean=float(
                np.mean([r.argmax_agreement_8x for r in cond_results])
            ),
            argmax_8x_std=float(
                np.std([r.argmax_agreement_8x for r in cond_results])
            ),
            delta_V_4x_mean=float(
                np.mean([r.delta_V_4x for r in cond_results])
            ),
            delta_V_8x_mean=float(
                np.mean([r.delta_V_8x for r in cond_results])
            ),
        ))

    return aggregates


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def generate_claims_md(aggregates: List[ConditionAggregate], out_path: Path) -> None:
    """Generate CLAIMS.md."""
    content = """# Phase 4 Claims: 2×2 Norm Ablation

## Recorded finite-run metrics

The table below reports the recorded conditions without inferring a causal
effect, a population-level stability guarantee, or an ordering not established
by the values themselves.

## Evidence

| Condition | z→z | V-head | Var(V) | Argmax@4× |
|-----------|-----|--------|--------|-----------|
"""
    for agg in aggregates:
        c_status = "ON" if agg.enable_contraction else "OFF"
        v_status = "OFF" if agg.disable_value_head_norm else "ON"
        content += f"| {agg.label} | {c_status} | {v_status} | "
        content += f"{agg.var_V_mean:.3f}±{agg.var_V_std:.3f} | "
        content += f"{agg.argmax_4x_mean:.3f}±{agg.argmax_4x_std:.3f} |\n"

    content += """
## Scope

These are finite checkpoint diagnostics on loaded input arrays. They do not
establish a causal effect, a uniform stability premise, or an environment-level
performance result.

## Unavailable metrics

| Field | Status | Reason |
|-------|--------|--------|
"""
    for metric, metadata in phase4_metric_availability().items():
        content += (
            f"| `{metric}` | {metadata['status']} | "
            f"`{metadata['reason']}` |\n"
        )

    with open(out_path / "CLAIMS.md", "w") as f:
        f.write(content)
    print(f"Saved: {out_path / 'CLAIMS.md'}")


def generate_provenance_md(
    config_dir: Path,
    out_path: Path,
    git_sha: str,
) -> None:
    """Generate PROVENANCE.md."""
    content = f"""# Phase 4 Provenance

## Git SHA: {git_sha}
## Generated: {datetime.now().isoformat()}
## Summary schema: {PHASE4_SCHEMA_VERSION}

## Metric availability

The evaluator did not run an environment rollout or load a training log or
training history. New schema-v2 summaries omit the retired numeric fields and
record their availability as follows:

```json
{json.dumps(phase4_metric_availability(), indent=2)}
```

## Config Files

"""
    for condition in CONDITIONS:
        config_path = config_dir / f"{condition}.yaml"
        content += f"### {condition}\n```yaml\n"
        with open(config_path, "r") as f:
            content += f.read()
        content += "```\n\n"

    with open(out_path / "PROVENANCE.md", "w") as f:
        f.write(content)
    print(f"Saved: {out_path / 'PROVENANCE.md'}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Evaluation")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/paper_ready/phase4_2x2_norm_ablation/v2",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="results/phase4_2x2_norm_ablation")
    parser.add_argument("--config_dir", type=str, default="configs/phase4_2x2_norm_ablation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out_dir
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    config_dir = PROJECT_ROOT / args.config_dir
    data_dir = PROJECT_ROOT / args.data_dir

    out_path.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = "cpu"

    print("=" * 60)
    print("Phase 4: 2×2 Norm Ablation Evaluation")
    print("=" * 60)
    print()

    # Evaluate all conditions
    results = []
    for condition in CONDITIONS:
        print(f"\n[{condition}] {CONDITION_LABELS[condition]}")
        for seed in SEEDS:
            result = evaluate_condition(
                condition, seed, checkpoint_dir, config_dir, data_dir, device
            )
            if result:
                results.append(result)

    if not results:
        print("\nERROR: No results collected. Check checkpoints exist.")
        return 1

    # Aggregate
    aggregates = aggregate_results(results)

    # Generate outputs
    git_sha = get_git_sha()

    summary = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "metric_availability": phase4_metric_availability(),
        "experiment": "Phase4_2x2_norm_ablation",
        "description": "Multi-seed 2×2 norm ablation (z→z contraction × value-head norm)",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "conditions": CONDITIONS,
        "seeds": SEEDS,
        "all_results": [asdict(r) for r in results],
        "aggregates": [asdict(a) for a in aggregates],
    }

    write_phase4_summary(summary, out_path / "summary.json")
    print(f"\nSaved: {out_path / 'summary.json'}")

    generate_claims_md(aggregates, out_path)
    generate_provenance_md(config_dir, out_path, git_sha)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
