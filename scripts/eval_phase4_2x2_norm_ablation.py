#!/usr/bin/env python3
"""
Phase 4: 2x2 Norm Ablation Evaluation Script.

Evaluates all 12 checkpoints (4 conditions × 3 seeds) and generates:
- schema-v4 summary.json with measured per-condition/per-seed metrics
- CLAIMS.md, PROVENANCE.md

Metrics:
- Stability: Var(V), L̂_z (pre-proj), projection_active_rate

This script does not run an environment rollout or load training history. The
schema records the corresponding success, final-loss, and training-history
metrics as unavailable instead of emitting placeholder values.

Build the evaluator PAR, freeze its SHA-256 externally, and invoke it only
through ``phase4_runtime_launcher --purpose phase4-evaluator``. Direct PAR or
``buck2 run`` execution fails the required pre-import runtime attestation.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts.phase4_result_schema import (  # noqa: E402
    PHASE4_LIPSCHITZ_PERTURBATION_SCHEME,
    PHASE4_LIPSCHITZ_PERTURBATION_SEED,
    PHASE4_SCHEMA_VERSION,
    phase4_metric_availability,
    write_phase4_summary,
)
from scripts.phase4_diagnostic_inputs import (  # noqa: E402
    load_phase4_diagnostic_states,
)
from utils.run_identity import (  # noqa: E402
    canonical_json_sha256,
    discover_clean_git_source,
)
from scripts.phase4_checkpoint import (  # noqa: E402
    Phase4CheckpointIdentity,
    load_phase4_checkpoint,
    phase4_checkpoint_relpath,
)
from scripts.phase4_source import (  # noqa: E402
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    require_phase4_runtime_attestation,
    resolve_phase4_path,
    resolve_phase4_source_roots,
    verify_phase4_producer_source,
    verify_phase4_runtime_sources,
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
    checkpoint_sha256: str
    model_state_sha256: str
    checkpoint_step: int
    training_run_id: str
    config_sha256: str
    rl_config_sha256: str
    model_config_sha256: str
    dataset_provenance_sha256: str
    producer_git_commit: str
    producer_source_manifest_sha256: str
    training_runtime_artifact_sha256: str
    initialization_kind: str
    checkpoint_schema_version: int
    training_invocation_schema_version: int
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
    argmax_agreement_n4: float
    argmax_agreement_n8: float
    delta_V_n4: float
    delta_V_n8: float
    lipschitz_sample_count: int
    stability_sample_count: int
    projection_sample_count: int


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
    argmax_n4_mean: float
    argmax_n4_std: float
    argmax_n8_mean: float
    argmax_n8_std: float
    delta_V_n4_mean: float
    delta_V_n8_mean: float


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_lipschitz(
    model,
    states: List[Dict],
    n_steps: int = 2,
    device: str = "cuda",
    perturbation_seed: int = PHASE4_LIPSCHITZ_PERTURBATION_SEED,
) -> Tuple[float, float, int]:
    """Compute finite differences of the production pre-projection map."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry

    if not states:
        raise ValueError("L_preproj requires at least one diagnostic state")
    model.eval()
    all_estimates = []
    eps = 0.01
    if isinstance(perturbation_seed, bool) or not isinstance(
        perturbation_seed, int
    ):
        raise ValueError("L_preproj perturbation seed must be an integer")
    perturbation_generator = torch.Generator(device="cpu")
    perturbation_generator.manual_seed(perturbation_seed)

    with torch.no_grad():
        for state in states[:50]:
            x = {
                "inputs": torch.tensor(state["inputs"], dtype=torch.long, device=device).unsqueeze(0),
                "puzzle_identifiers": torch.tensor(
                    [state["puzzle_identifier"]],
                    dtype=torch.long,
                    device=device,
                ),
            }
            y = torch.tensor(state.get("plan", state["inputs"]), dtype=torch.long, device=device).unsqueeze(0)

            # Get latent after n_steps
            _, z_carry = model.used_value(x, y, n=n_steps)
            if z_carry is None:
                raise RuntimeError("L_preproj model did not return a latent carry")

            inner = model.inner
            inner_carry = z_carry.inner_carry if hasattr(z_carry, 'inner_carry') else z_carry
            batch = model._standardize_latent_batch(x, y)
            context = model._resolve_latent_context(batch)
            input_embeddings = context["input_embeddings_with_plan"]
            seq_info = context["seq_info"]

            # Baseline step
            baseline = inner.latent_step_pre_projection(
                inner_carry,
                input_embeddings,
                seq_info,
            )

            # Perturbed step
            for _ in range(3):  # Multiple perturbation trials
                noise_h = torch.randn(
                    inner_carry.z_H.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=perturbation_generator,
                ).to(
                    device=inner_carry.z_H.device,
                    dtype=inner_carry.z_H.dtype,
                )
                noise_l = torch.randn(
                    inner_carry.z_L.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=perturbation_generator,
                ).to(
                    device=inner_carry.z_L.device,
                    dtype=inner_carry.z_L.dtype,
                )
                noise_norm = inner._joint_carry_geometry(noise_h, noise_l)[0]
                if not bool(torch.isfinite(noise_norm).all().item()) or bool(
                    (noise_norm <= 0.0).any().item()
                ):
                    raise RuntimeError("L_preproj sampled an invalid perturbation")
                scale = noise_norm.new_tensor(eps) / noise_norm
                delta_h = noise_h * scale
                delta_l = noise_l * scale

                perturbed = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=inner_carry.z_H + delta_h,
                    z_L=inner_carry.z_L + delta_l,
                )
                out = inner.latent_step_pre_projection(
                    perturbed,
                    input_embeddings,
                    seq_info,
                )
                denominator = inner._joint_carry_geometry(delta_h, delta_l)[0]
                numerator = inner._joint_carry_geometry(
                    out.z_H - baseline.z_H,
                    out.z_L - baseline.z_L,
                )[0]
                quotient = numerator / denominator
                if not bool(torch.isfinite(quotient).all().item()):
                    raise RuntimeError("L_preproj produced a non-finite quotient")
                all_estimates.extend(quotient.reshape(-1).tolist())

    if not all_estimates:
        raise RuntimeError("L_preproj produced no observations")
    return (
        float(np.mean(all_estimates)),
        float(np.std(all_estimates)),
        len(all_estimates),
    )


def compute_stability_metrics(
    model,
    states: List[Dict],
    rl_config: Dict,
    n_train: int = 2,
    device: str = "cuda",
) -> Dict[str, float | int]:
    """Compute value and production-policy stability at depth mismatch."""
    from rl.task_config import get_task_config

    if not states:
        raise ValueError("Stability metrics require at least one diagnostic state")
    if rl_config.get("task_name", "sudoku") != "sudoku":
        raise ValueError("Phase 4 stability requires the Sudoku task")
    if rl_config.get("stop_action_mode") != "disabled":
        raise ValueError("Phase 4 stability requires disabled STOP actions")
    task_config = get_task_config(
        "sudoku",
        disable_constraint_masking=bool(
            rl_config.get("disable_constraint_masking", False)
        ),
    )
    model.eval()

    V_train_all = []
    agreements_n4 = []
    agreements_n8 = []
    delta_V_n4 = []
    delta_V_n8 = []

    with torch.no_grad():
        for state in states[:100]:
            y = torch.tensor(state.get("plan", state["inputs"]), dtype=torch.long, device=device).unsqueeze(0)
            x = {
                "inputs": torch.tensor(state["inputs"], dtype=torch.long, device=device).unsqueeze(0),
                "puzzle_identifiers": torch.tensor(
                    [state["puzzle_identifier"]],
                    dtype=torch.long,
                    device=device,
                ),
            }
            stop_action_id = model.config.rl_num_actions - 1
            action_mask = task_config.compute_batch_action_mask(
                x["inputs"],
                model.config.vocab_size,
                stop_action_id,
                current_state=y,
            ).clone()
            action_mask[:, stop_action_id] = False
            expected_shape = (x["inputs"].shape[0], model.config.rl_num_actions)
            if action_mask.dtype != torch.bool or tuple(action_mask.shape) != expected_shape:
                raise RuntimeError("Phase 4 action mask has the wrong type or shape")
            if bool((~action_mask.any(dim=-1)).any().item()):
                raise RuntimeError("Phase 4 action mask has an empty support row")

            V_train, _ = model.used_value(x, y, n=n_train)
            if V_train is None:
                raise RuntimeError("Phase 4 model did not produce a value")
            dist_train, _ = model.policy_dist(
                x,
                y,
                n=n_train,
                action_mask=action_mask,
            )
            V_4x, _ = model.used_value(x, y, n=4)
            V_8x, _ = model.used_value(x, y, n=8)
            if V_4x is None or V_8x is None:
                raise RuntimeError("Phase 4 model did not produce depth-mismatch values")
            dist_4x, _ = model.policy_dist(x, y, n=4, action_mask=action_mask)
            dist_8x, _ = model.policy_dist(x, y, n=8, action_mask=action_mask)

            value_train = float(V_train.item())
            value_4x = float(V_4x.item())
            value_8x = float(V_8x.item())
            values = (value_train, value_4x, value_8x)
            if not all(np.isfinite(value) for value in values):
                raise RuntimeError("Phase 4 stability produced a non-finite value")
            action_train = dist_train.logits.argmax(dim=-1)
            action_4x = dist_4x.logits.argmax(dim=-1)
            action_8x = dist_8x.logits.argmax(dim=-1)
            V_train_all.append(value_train)
            agreements_n4.append((action_train == action_4x).float().item())
            agreements_n8.append((action_train == action_8x).float().item())
            delta_V_n4.append(abs(value_train - value_4x))
            delta_V_n8.append(abs(value_train - value_8x))

    sample_count = len(V_train_all)
    if sample_count == 0 or any(
        len(values) != sample_count
        for values in (agreements_n4, agreements_n8, delta_V_n4, delta_V_n8)
    ):
        raise RuntimeError("Phase 4 stability produced an incomplete sample set")
    return {
        "var_V": float(np.var(V_train_all)),
        "argmax_n4": float(np.mean(agreements_n4)),
        "argmax_n8": float(np.mean(agreements_n8)),
        "delta_V_n4": float(np.mean(delta_V_n4)),
        "delta_V_n8": float(np.mean(delta_V_n8)),
        "sample_count": sample_count,
    }


def compute_projection_rate(
    model,
    states: List[Dict],
    n_steps: int = 2,
    device: str = "cuda",
) -> Tuple[float, int]:
    """Compute next-step activation using production projection instrumentation."""
    model.eval()
    if not states:
        raise ValueError("Projection rate requires at least one diagnostic state")
    if not hasattr(model, "config"):
        raise ValueError("Model config is required for projection diagnostics")
    projection_mode = model.config.rl_latent_projection_mode
    radius = model.config.rl_latent_ball_radius
    if projection_mode == "disabled":
        if radius is not None:
            raise ValueError("Disabled projection must not have a radius")
        raise ValueError("Phase 4 publication requires enabled projection")
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
                "puzzle_identifiers": torch.tensor(
                    [state["puzzle_identifier"]],
                    dtype=torch.long,
                    device=device,
                ),
            }
            y = torch.tensor(state.get("plan", state["inputs"]), dtype=torch.long, device=device).unsqueeze(0)

            _, carry = model.used_value(x, y, n=n_steps)
            if carry is None:
                raise RuntimeError("Projection diagnostic did not return a latent carry")

            inner_carry = carry.inner_carry if hasattr(carry, 'inner_carry') else carry
            if not hasattr(inner_carry, 'z_H'):
                raise RuntimeError("Projection diagnostic returned an invalid carry")

            batch = model._standardize_latent_batch(x, y)
            context = model._resolve_latent_context(batch)
            input_embeddings = context["input_embeddings_with_plan"]
            _, _, projection_active = model.inner.latent_step_with_projection_info(
                inner_carry,
                input_embeddings,
                context["seq_info"],
            )
            active_count += projection_active.sum().item()
            total_count += projection_active.numel()

    if total_count <= 0:
        raise RuntimeError("Projection rate produced no observations")
    return active_count / total_count, total_count


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_condition(
    condition: str,
    seed: int,
    checkpoint_dir: Path,
    config_dir: Path,
    expected_producer_source: Dict[str, object],
    expected_training_runtime_sha256: str,
    diagnostic_states: List[Dict],
    device: str = "cuda",
) -> Optional[ConditionResult]:
    """Evaluate a single condition-seed pair."""
    checkpoint_relpath = phase4_checkpoint_relpath(condition, seed)
    ckpt_path = checkpoint_dir / checkpoint_relpath
    config_path = config_dir / f"{condition}.yaml"

    if not ckpt_path.exists():
        print(f"  [Skip] Checkpoint not found: {ckpt_path}")
        return None

    print(f"  Evaluating {condition} seed={seed}...")

    loaded = load_phase4_checkpoint(
        ckpt_path,
        config_path,
        condition=condition,
        seed=seed,
        expected_producer_source=expected_producer_source,
        expected_training_runtime_sha256=expected_training_runtime_sha256,
        device=device,
    )
    model = loaded.model
    config = loaded.rl_config
    identity: Phase4CheckpointIdentity = loaded.identity

    # Compute finite diagnostics on the single input population loaded before
    # any checkpoint. Every design cell therefore sees identical ordered states.
    L_preproj, L_preproj_std, lipschitz_sample_count = compute_lipschitz(
        model,
        diagnostic_states,
        N_TRAIN,
        device,
        PHASE4_LIPSCHITZ_PERTURBATION_SEED,
    )
    stability = compute_stability_metrics(
        model,
        diagnostic_states,
        config,
        N_TRAIN,
        device,
    )
    proj_rate, projection_sample_count = compute_projection_rate(
        model,
        diagnostic_states,
        N_TRAIN,
        device,
    )

    return ConditionResult(
        condition=condition,
        seed=seed,
        checkpoint_path=checkpoint_relpath,
        checkpoint_sha256=identity.checkpoint_sha256,
        model_state_sha256=identity.model_state_sha256,
        checkpoint_step=identity.checkpoint_step,
        training_run_id=identity.training_run_id,
        config_sha256=identity.config_sha256,
        rl_config_sha256=identity.rl_config_sha256,
        model_config_sha256=identity.model_config_sha256,
        dataset_provenance_sha256=identity.dataset_provenance_sha256,
        producer_git_commit=identity.producer_git_commit,
        producer_source_manifest_sha256=(
            identity.producer_source_manifest_sha256
        ),
        training_runtime_artifact_sha256=(
            identity.training_runtime_artifact_sha256
        ),
        initialization_kind=identity.initialization_kind,
        checkpoint_schema_version=identity.checkpoint_schema_version,
        training_invocation_schema_version=(
            identity.training_invocation_schema_version
        ),
        enable_contraction=config["enable_contraction"],
        disable_value_head_norm=config["disable_value_head_norm"],
        latent_projection_mode=config["latent_projection_mode"],
        latent_ball_radius=config["latent_ball_radius"],
        L_preproj=L_preproj,
        L_preproj_std=L_preproj_std,
        var_V=stability["var_V"],
        projection_active_rate=proj_rate,
        argmax_agreement_n4=stability["argmax_n4"],
        argmax_agreement_n8=stability["argmax_n8"],
        delta_V_n4=stability["delta_V_n4"],
        delta_V_n8=stability["delta_V_n8"],
        lipschitz_sample_count=lipschitz_sample_count,
        stability_sample_count=int(stability["sample_count"]),
        projection_sample_count=projection_sample_count,
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
            argmax_n4_mean=float(
                np.mean([r.argmax_agreement_n4 for r in cond_results])
            ),
            argmax_n4_std=float(
                np.std([r.argmax_agreement_n4 for r in cond_results])
            ),
            argmax_n8_mean=float(
                np.mean([r.argmax_agreement_n8 for r in cond_results])
            ),
            argmax_n8_std=float(
                np.std([r.argmax_agreement_n8 for r in cond_results])
            ),
            delta_V_n4_mean=float(
                np.mean([r.delta_V_n4 for r in cond_results])
            ),
            delta_V_n8_mean=float(
                np.mean([r.delta_V_n8 for r in cond_results])
            ),
        ))

    return aggregates


def get_evaluator_git_commit(project_root: Path) -> str:
    """Require and return the clean evaluator repository commit."""

    identity = discover_clean_git_source(project_root)
    commit = identity["git_commit"]
    if not isinstance(commit, str):
        raise RuntimeError("Evaluator Git identity did not contain a commit")
    return commit


def generate_claims_md(aggregates: List[ConditionAggregate], out_path: Path) -> None:
    """Generate CLAIMS.md."""
    content = """# Phase 4 Claims: 2×2 Norm Ablation

## Recorded finite-run metrics

The table below reports the recorded conditions without inferring a causal
effect, a population-level stability guarantee, or an ordering not established
by the values themselves.

## Evidence

| Condition | z→z | V-head | Var(V) | Argmax@n=4 |
|-----------|-----|--------|--------|-----------|
"""
    for agg in aggregates:
        c_status = "ON" if agg.enable_contraction else "OFF"
        v_status = "OFF" if agg.disable_value_head_norm else "ON"
        content += f"| {agg.label} | {c_status} | {v_status} | "
        content += f"{agg.var_V_mean:.3f}±{agg.var_V_std:.3f} | "
        content += f"{agg.argmax_n4_mean:.3f}±{agg.argmax_n4_std:.3f} |\n"

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
training history. New schema-v4 summaries omit the retired numeric fields and
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


def main(*, runtime_attestation: Optional[Dict[str, Any]] = None):
    attestation = require_phase4_runtime_attestation(
        runtime_attestation,
        PHASE4_EVALUATOR_SOURCE_PROFILE,
    )
    parser = argparse.ArgumentParser(description="Phase 4 Evaluation")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--fbcode_root", type=str, required=True)
    parser.add_argument("--producer_project_root", type=str, required=True)
    parser.add_argument(
        "--expected_producer_git_commit",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--expected_training_runtime_sha256",
        type=str,
        required=True,
        help="Externally authorized Phase 4 training PAR SHA-256",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/paper_ready/phase4_2x2_norm_ablation/v4",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="results/phase4_2x2_norm_ablation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if re.fullmatch(
        r"[0-9a-f]{64}", args.expected_training_runtime_sha256
    ) is None:
        parser.error("--expected_training_runtime_sha256 must be lowercase SHA-256")

    project_root, _ = resolve_phase4_source_roots(
        args.project_root,
        args.fbcode_root,
    )
    evaluator_git_commit = get_evaluator_git_commit(project_root)
    evaluator_source_manifest_sha256 = verify_phase4_runtime_sources(
        project_root,
        PHASE4_EVALUATOR_SOURCE_PROFILE,
    )
    if (
        evaluator_git_commit != attestation["source_git_commit"]
        or evaluator_source_manifest_sha256
        != attestation["source_manifest_sha256"]
    ):
        raise RuntimeError(
            "Evaluator checkout differs from the pre-import runtime attestation."
        )
    producer_project_root = Path(args.producer_project_root).expanduser().resolve(
        strict=True
    )
    expected_producer_source = verify_phase4_producer_source(
        producer_project_root,
        args.expected_producer_git_commit,
    )
    out_path = resolve_phase4_path(args.out_dir, project_root)
    checkpoint_dir = resolve_phase4_path(
        args.checkpoint_dir,
        producer_project_root,
    )
    config_dir = (
        producer_project_root / "configs" / "phase4_2x2_norm_ablation"
    )
    data_dir = resolve_phase4_path(args.data_dir, project_root)

    out_path.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = "cpu"

    print("=" * 60)
    print("Phase 4: 2×2 Norm Ablation Evaluation")
    print("=" * 60)
    print()

    diagnostic_states, diagnostic_dataset = load_phase4_diagnostic_states(
        data_dir
    )

    # Evaluate all conditions
    results = []
    for condition in CONDITIONS:
        print(f"\n[{condition}] {CONDITION_LABELS[condition]}")
        for seed in SEEDS:
            result = evaluate_condition(
                condition,
                seed,
                checkpoint_dir,
                config_dir,
                expected_producer_source,
                args.expected_training_runtime_sha256,
                diagnostic_states,
                device,
            )
            if result:
                results.append(result)

    if not results:
        print("\nERROR: No results collected. Check checkpoints exist.")
        return 1

    # Aggregate
    aggregates = aggregate_results(results)

    # Generate outputs
    git_sha = get_evaluator_git_commit(project_root)
    if git_sha != evaluator_git_commit:
        raise RuntimeError("Evaluator source identity changed during evaluation.")
    final_source_manifest_sha256 = verify_phase4_runtime_sources(
        project_root,
        PHASE4_EVALUATOR_SOURCE_PROFILE,
    )
    if final_source_manifest_sha256 != evaluator_source_manifest_sha256:
        raise RuntimeError("Evaluator runtime source changed during evaluation.")
    if verify_phase4_producer_source(
        producer_project_root,
        args.expected_producer_git_commit,
    ) != expected_producer_source:
        raise RuntimeError("Producer source identity changed during evaluation.")

    summary = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "metric_availability": phase4_metric_availability(),
        "experiment": "Phase4_2x2_norm_ablation",
        "description": "Multi-seed 2×2 norm ablation (z→z contraction × value-head norm)",
        "generated_at": datetime.now().isoformat(),
        "evaluator_git_commit": git_sha,
        "evaluator_source_manifest_sha256": (
            evaluator_source_manifest_sha256
        ),
        "evaluator_runtime_artifact_sha256": attestation["runtime_sha256"],
        "diagnostic_dataset": diagnostic_dataset,
        "diagnostic_dataset_sha256": canonical_json_sha256(
            diagnostic_dataset
        ),
        "lipschitz_perturbation_seed": (
            PHASE4_LIPSCHITZ_PERTURBATION_SEED
        ),
        "lipschitz_perturbation_scheme": (
            PHASE4_LIPSCHITZ_PERTURBATION_SCHEME
        ),
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
