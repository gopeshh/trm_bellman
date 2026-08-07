#!/usr/bin/env python3
"""
Unroll Sensitivity Evaluation Script (ICML Phase 1 - "Theory Saver")

This script reports finite-sample changes in model predictions at deeper
unroll depths. These diagnostics do not establish a uniform contraction,
residual, or value-error premise from the paper.

Subcommands:
    build-batches: Build evaluation batches B0 (initial states) and B1 (successor closure)
    eval: Evaluate a single checkpoint on batches
    compare: Compare two checkpoints (Model A vs Model B) with compatibility assertions

Usage:
    python scripts/eval_unroll_sensitivity.py build-batches \
        --checkpoint_a path/to/model_a.pt \
        --checkpoint_b path/to/model_b.pt \
        --out_dir artifacts/eval_batches/ \
        --seed 42

    python scripts/eval_unroll_sensitivity.py eval \
        --checkpoint path/to/model.pt \
        --batch_b0 artifacts/eval_batches/b0.pt \
        --batch_b1 artifacts/eval_batches/b1.pt \
        --out_dir results/plot_data/unroll_sensitivity/

    python scripts/eval_unroll_sensitivity.py compare \
        --checkpoint_a path/to/model_a.pt \
        --checkpoint_b path/to/model_b.pt \
        --batch_b0 artifacts/eval_batches/b0.pt \
        --batch_b1 artifacts/eval_batches/b1.pt \
        --table_out results/tables/unroll_sensitivity_summary.md
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PuzzleState:
    """A single puzzle state for evaluation."""
    state_id: str
    inputs: torch.Tensor  # [seq_len]
    puzzle_identifier: torch.Tensor  # scalar or [1]
    plan: torch.Tensor  # [seq_len] (initial plan = copy of inputs)
    empties: int  # number of empty cells
    source_path: str  # which dataset directory
    parent_id: Optional[str] = None  # for B1 states
    action_id: Optional[int] = None  # for B1 states
    action_source: Optional[str] = None  # "A_top", "B_top", "rand"


@dataclass
class BatchMetadata:
    """Metadata for a saved batch."""
    batch_name: str  # "b0" or "b1"
    num_states: int
    empties_distribution: Dict[int, int]  # empties -> count
    seed: int
    source_paths: List[str]
    creation_time: str
    git_sha: Optional[str] = None
    checkpoint_a_path: Optional[str] = None
    checkpoint_b_path: Optional[str] = None
    # For B1
    parent_batch: Optional[str] = None
    cap: Optional[int] = None


@dataclass
class EvalMetrics:
    """Metrics for a single state at a pair of unroll depths."""
    state_id: str
    n1: int
    n2: int
    delta_V: float
    delta_pi: float  # KL divergence
    delta_z: float  # L2 norm
    argmax_agree: int  # 1 if argmax matches, 0 otherwise
    z_pre_norm: float  # norm before projection
    z_post_norm: float  # norm after projection
    saturated: int  # 1 if z_pre_norm >= 0.95 * R


@dataclass
class RunMetadata:
    """Metadata for a run."""
    checkpoint_path: str
    checkpoint_config: Dict[str, Any]
    n_train: int
    n_mults: List[int]
    seed: int
    batch_b0_path: str
    batch_b1_path: Optional[str]
    git_sha: Optional[str]
    creation_time: str


# =============================================================================
# Utility Functions
# =============================================================================

def get_git_sha() -> Optional[str]:
    """Get current git SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def stable_hash(tensor: torch.Tensor) -> str:
    """Compute a stable hash of a tensor for deduplication."""
    # Convert to bytes and hash
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.md5(arr).hexdigest()[:16]


def count_empties(inputs: torch.Tensor) -> int:
    """Count empty cells in a puzzle (token value 1 = empty)."""
    return int((inputs == 1).sum().item())


def compute_kl_divergence(
    p: torch.Tensor,  # [num_actions]
    q: torch.Tensor,  # [num_actions]
    eps: float = 1e-8,
) -> float:
    """Compute KL(p || q) with numerical stability."""
    # Clamp to avoid log(0)
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    # Renormalize after clamping
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    kl = (p_safe * (p_safe.log() - q_safe.log())).sum()
    return float(kl.clamp(min=0).item())  # KL is non-negative


def aggregate_metrics(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, p90, p99, max for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def estimate_hat_Lz_batch(
    model: TinyRecursiveReasoningModel_ACTV1,
    states: List["PuzzleState"],
    n_train: int,
    config: Dict[str, Any],
    device: str = "cpu",
    num_samples: int = 128,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate achieved Lipschitz constant (hat_Lz) of the z->z mapping.

    Uses finite differences: for each state, perturb z at step n and measure
    how the perturbation propagates to step n+1.

    Returns:
        Dict with 'mean', 'p95', 'max' of estimated Lz values.
    """
    rng = np.random.default_rng(seed)

    # Sample states if we have too many
    if len(states) > num_samples:
        indices = rng.choice(len(states), size=num_samples, replace=False)
        sample_states = [states[i] for i in indices]
    else:
        sample_states = states

    lz_estimates = []
    eps = 1e-3  # Perturbation magnitude

    model.eval()
    with torch.no_grad():
        for state in sample_states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            try:
                # Get baseline latent at n_train
                _, z_base = model.used_value(x, y, n_train)
                if not hasattr(z_base, 'z_H'):
                    continue

                # Get latent at n_train + 1
                _, z_next = model.used_value(x, y, n_train + 1)

                # Compute the "step" in latent space
                z_h_base = z_base.z_H.squeeze()
                z_h_next = z_next.z_H.squeeze()

                delta = z_h_next - z_h_base
                delta_norm = float(torch.norm(delta, p=2).item())
                base_norm = float(torch.norm(z_h_base, p=2).item())

                if base_norm > 1e-6:
                    # Lz ~ ||f(z) - z|| / ||z|| (contraction factor)
                    lz = delta_norm / base_norm
                    lz_estimates.append(lz)

            except Exception:
                # Skip states that cause errors
                continue

    if not lz_estimates:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}

    arr = np.array(lz_estimates)
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


# =============================================================================
# Model Loading and Config Extraction
# =============================================================================

def load_model_for_eval(
    checkpoint_path: str,
    device: str = "cpu",
    config_yaml_path: Optional[str] = None,
    latent_ball_radius_override: Optional[float] = None,
    latent_projection_mode_override: Optional[str] = None,
) -> Tuple[TinyRecursiveReasoningModel_ACTV1, Dict[str, Any]]:
    """
    Load a model checkpoint and extract its config.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        config_yaml_path: Optional path to YAML config used during training.
                          If provided, uses exact training config values.
                          If not provided, infers config from checkpoint.
        latent_ball_radius_override: Positive radius override. This enables
                                     Euclidean projection.
        latent_projection_mode_override: Explicit ``enabled`` or ``disabled``
                                         recurrent projection mode.

    Returns:
        (model, config_dict) where config_dict contains key settings.
    """
    from rl.config import RLConfig
    import yaml

    print(f"[Load] Loading checkpoint from {checkpoint_path}")
    if config_yaml_path:
        print(f"[Load] Using YAML config: {config_yaml_path}")
    if latent_projection_mode_override not in (None, "enabled", "disabled"):
        raise ValueError("latent_projection_mode_override must be enabled or disabled")
    if latent_ball_radius_override is not None and latent_ball_radius_override <= 0.0:
        raise ValueError("latent_ball_radius_override must be positive")
    if (
        latent_projection_mode_override == "disabled"
        and latent_ball_radius_override is not None
    ):
        raise ValueError("Disabled projection cannot also specify a radius override")

    # Load YAML config if provided
    yaml_config = {}
    if config_yaml_path and Path(config_yaml_path).exists():
        with open(config_yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        print(f"[Load] Loaded YAML config with keys: {list(yaml_config.keys())[:10]}")

    # Full RL checkpoints contain replay Transition objects. These are trusted
    # local artifacts, not arbitrary downloaded pickle files.
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Handle nested checkpoint structure (RL checkpoints have "model_state_dict")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model_state = state_dict["model_state_dict"]
        rl_config_dict = state_dict.get("rl_config", {})
        persisted_model_config = state_dict.get("model_config")
    else:
        model_state = state_dict
        rl_config_dict = {}
        persisted_model_config = None

    # Clean state dict keys (remove _orig_mod. prefix from torch.compile)
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state[clean_key] = value

    # Check for value head to determine if RL model
    has_value_head = any("value_head" in k for k in cleaned_state.keys())
    has_policy_head = any("edit_policy" in k for k in cleaned_state.keys())

    # === CRITICAL FIX: Detect contraction from checkpoint ===
    # Model B (trained with contraction) has _lip_scale keys from opnorm_clamp
    lip_scale_keys = [k for k in cleaned_state.keys() if "_lip_scale" in k]
    has_contraction_keys = len(lip_scale_keys) > 0
    print(f"[Load] Detected {len(lip_scale_keys)} Lipschitz scale keys -> contraction={'ENABLED' if has_contraction_keys else 'DISABLED'}")

    rl_cfg = RLConfig()

    # Get inner_unroll_n (CRITICAL for correct depth labels)
    if "inner_unroll_n" in yaml_config:
        inner_unroll_n = int(yaml_config["inner_unroll_n"])
    elif "inner_unroll_n" in rl_config_dict:
        inner_unroll_n = int(rl_config_dict["inner_unroll_n"])
    else:
        inner_unroll_n = int(rl_cfg.inner_unroll_n)

    episodic_latent = yaml_config.get("episodic_latent", rl_config_dict.get("episodic_latent", rl_cfg.episodic_latent))
    use_feasibility_checker = yaml_config.get("use_feasibility_checker", rl_config_dict.get("use_feasibility_checker", rl_cfg.use_feasibility_checker))

    if persisted_model_config is not None:
        if hasattr(persisted_model_config, "model_dump"):
            persisted_model_config = persisted_model_config.model_dump()
        elif hasattr(persisted_model_config, "dict"):
            persisted_model_config = persisted_model_config.dict()
        if not isinstance(persisted_model_config, dict):
            raise TypeError("Checkpoint model_config must be a dictionary.")
        construction_config = dict(persisted_model_config)
        # Sparse-embedding scratch buffers are non-persistent. Evaluation uses
        # one state at a time, so they can be rebuilt for batch size one.
        construction_config["batch_size"] = 1
        if latent_projection_mode_override is not None:
            construction_config["rl_latent_projection_mode"] = (
                latent_projection_mode_override
            )
            if latent_projection_mode_override == "disabled":
                construction_config["rl_latent_ball_radius"] = None
        if latent_ball_radius_override is not None:
            construction_config["rl_latent_projection_mode"] = "enabled"
            construction_config["rl_latent_ball_radius"] = float(
                latent_ball_radius_override
            )
            print(
                "[Load] Using latent_ball_radius OVERRIDE: "
                f"{latent_ball_radius_override}"
            )
        model_config = TinyRecursiveReasoningModel_ACTV1Config(**construction_config)
        config_source = "checkpoint_model_config"
    else:
        embedding_key = None
        for candidate in (
            "inner.embed_tokens.embedding_weight",
            "inner.embed_inputs.weight",
        ):
            if candidate in cleaned_state:
                embedding_key = candidate
                break
        if embedding_key is None:
            raise RuntimeError(
                "Cannot infer the legacy TRM architecture: token embedding "
                "weights are missing from the checkpoint."
            )
        if "inner.puzzle_emb.weights" in cleaned_state:
            raise RuntimeError(
                "Legacy checkpoint contains per-puzzle embeddings but no "
                "model_config. Their sequence length and held-out identifier "
                "mapping cannot be reconstructed safely."
            )

        token_weights = cleaned_state[embedding_key]
        vocab_size = int(token_weights.shape[0])
        hidden_size = int(token_weights.shape[1])
        num_actions = 0
        seq_len = 0
        policy_output_key = "edit_policy.mlp.2.weight"
        if has_policy_head and policy_output_key in cleaned_state:
            num_actions = int(cleaned_state[policy_output_key].shape[0])
            flat_edit_actions = num_actions - 1
            if flat_edit_actions <= 0 or flat_edit_actions % vocab_size != 0:
                raise RuntimeError(
                    "Cannot infer sequence length from the legacy policy head: "
                    f"num_actions={num_actions}, vocab_size={vocab_size}."
                )
            seq_len = flat_edit_actions // vocab_size
        elif has_value_head and "value_head.linear1.weight" in cleaned_state:
            input_width = int(cleaned_state["value_head.linear1.weight"].shape[1])
            if input_width % (3 * hidden_size) != 0:
                raise RuntimeError(
                    "Cannot infer sequence length from the legacy value head: "
                    f"input_width={input_width}, hidden_size={hidden_size}."
                )
            seq_len = input_width // (3 * hidden_size)
        else:
            raise RuntimeError(
                "Legacy checkpoint lacks a policy or value head from which to "
                "infer sequence length."
            )

        if "enable_contraction" in yaml_config:
            enable_contraction = bool(yaml_config["enable_contraction"])
        elif "enable_contraction" in rl_config_dict:
            enable_contraction = bool(rl_config_dict["enable_contraction"])
        else:
            enable_contraction = has_contraction_keys
        if "latent_projection_mode" in yaml_config or "latent_ball_radius" in yaml_config:
            latent_projection_mode = yaml_config.get("latent_projection_mode")
            raw_latent_ball_radius = yaml_config.get("latent_ball_radius")
        elif (
            "latent_projection_mode" in rl_config_dict
            or "latent_ball_radius" in rl_config_dict
        ):
            latent_projection_mode = rl_config_dict.get("latent_projection_mode")
            raw_latent_ball_radius = rl_config_dict.get("latent_ball_radius")
        else:
            latent_projection_mode = rl_cfg.latent_projection_mode
            raw_latent_ball_radius = rl_cfg.latent_ball_radius
        if latent_projection_mode_override is not None:
            latent_projection_mode = latent_projection_mode_override
            if latent_projection_mode == "disabled":
                raw_latent_ball_radius = None
        if latent_ball_radius_override is not None:
            latent_projection_mode = "enabled"
            raw_latent_ball_radius = latent_ball_radius_override
        latent_ball_radius = (
            None
            if raw_latent_ball_radius is None
            else float(raw_latent_ball_radius)
        )
        target_lz_value = yaml_config.get(
            "target_Lz",
            rl_config_dict.get("target_Lz", rl_cfg.target_Lz),
        )
        if target_lz_value is None:
            target_lz_value = rl_cfg.target_Lz
        target_lz = float(target_lz_value)
        disable_value_head_norm = bool(
            yaml_config.get(
                "disable_value_head_norm",
                rl_config_dict.get(
                    "disable_value_head_norm",
                    rl_cfg.disable_value_head_norm,
                ),
            )
        )
        projection_config: Dict[str, Any] = {
            "rl_latent_ball_radius": latent_ball_radius
        }
        if latent_projection_mode is not None:
            projection_config["rl_latent_projection_mode"] = latent_projection_mode
        model_config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=1,
            seq_len=seq_len,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_puzzle_identifiers=1,
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
            rl_target_Lz=target_lz,
            rl_disable_value_head_norm=disable_value_head_norm,
            **projection_config,
        )
        config_source = "legacy_inferred"

    hidden_size = int(model_config.hidden_size)
    vocab_size = int(model_config.vocab_size)
    seq_len = int(model_config.seq_len)
    num_actions = int(model_config.rl_num_actions)
    enable_contraction = bool(model_config.rl_enable_contraction)
    target_Lz = float(model_config.rl_target_Lz)
    disable_value_head_norm = bool(model_config.rl_disable_value_head_norm)
    latent_projection_mode = model_config.rl_latent_projection_mode
    latent_ball_radius = model_config.rl_latent_ball_radius

    # Create model (expects dict, not config object)
    model = TinyRecursiveReasoningModel_ACTV1(model_config.model_dump())

    # New checkpoints carry their exact construction config and must load
    # without structural drift. Legacy checkpoints retain the permissive path
    # because older operator-norm wrappers used slightly different buffers.
    result = model.load_state_dict(
        cleaned_state,
        strict=persisted_model_config is not None,
    )
    if result.missing_keys:
        print(f"[Load] Missing keys: {result.missing_keys[:5]}...")
    if result.unexpected_keys:
        print(f"[Load] Unexpected keys: {result.unexpected_keys[:5]}...")

    model.to(device)
    model.eval()

    # Config dict for reporting and comparison
    config_dict = {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_actions": num_actions,
        "inner_unroll_n": inner_unroll_n,
        "enable_contraction": enable_contraction,
        "target_Lz": target_Lz,
        "disable_value_head_norm": disable_value_head_norm,
        "episodic_latent": episodic_latent,
        "latent_projection_mode": latent_projection_mode,
        "latent_ball_radius": latent_ball_radius,
        "use_feasibility_checker": use_feasibility_checker,
        "puzzle_emb_ndim": int(model_config.puzzle_emb_ndim),
        "puzzle_emb_len": int(model_config.puzzle_emb_len),
        "num_puzzle_identifiers": int(model_config.num_puzzle_identifiers),
        "H_cycles": int(model_config.H_cycles),
        "L_cycles": int(model_config.L_cycles),
        "L_layers": int(model_config.L_layers),
        "model_config": model_config.model_dump(),
        "config_source": config_source,
    }

    print(f"[Load] Model loaded. Config: {config_dict}")
    return model, config_dict


def check_model_compatibility(
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Check if two models are compatible for comparison.

    Required to match:
        - num_actions (action space size)
        - episodic_latent
        - use_feasibility_checker

    Returns:
        (is_compatible, list_of_diffs)
    """
    required_keys = ["num_actions", "episodic_latent", "use_feasibility_checker"]
    diffs = []

    for key in required_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            diffs.append(f"{key}: A={val_a}, B={val_b}")

    return len(diffs) == 0, diffs


# =============================================================================
# Dataset Loading
# =============================================================================

def find_sudoku_4x4_datasets(data_dir_override: Optional[str] = None) -> List[str]:
    """Find all sudoku-4x4 dataset directories."""
    if data_dir_override:
        data_dir = Path(data_dir_override)
        # If data_dir_override is a specific dataset, return it directly
        if data_dir.name.startswith("sudoku-4x4"):
            train_dir = data_dir / "train"
            if train_dir.exists() and (train_dir / "all__inputs.npy").exists():
                return [str(data_dir)]
            return []
    else:
        data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        return []

    datasets = []
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("sudoku-4x4"):
            # Check for train/test subdirs with numpy files
            train_dir = path / "train"
            if train_dir.exists() and (train_dir / "all__inputs.npy").exists():
                datasets.append(str(path))

    return sorted(datasets)


def load_puzzles_from_dataset(
    dataset_path: str,
    split: str = "train",
) -> List[PuzzleState]:
    """Load all puzzles from a dataset directory."""
    split_dir = Path(dataset_path) / split

    inputs = np.load(split_dir / "all__inputs.npy")
    puzzle_ids = np.load(split_dir / "all__puzzle_identifiers.npy")

    puzzles = []
    for i in range(len(inputs)):
        inp = torch.from_numpy(inputs[i].astype(np.int64))
        pid = torch.tensor(puzzle_ids[i], dtype=torch.long)
        empties = count_empties(inp)

        state = PuzzleState(
            state_id=f"{Path(dataset_path).name}_{split}_{i}",
            inputs=inp,
            puzzle_identifier=pid,
            plan=inp.clone(),  # Start with clues as initial plan
            empties=empties,
            source_path=dataset_path,
        )
        puzzles.append(state)

    return puzzles


# =============================================================================
# Batch Building (B0 and B1)
# =============================================================================

def build_b0_batch(
    target_easy: int = 70,
    target_hard: int = 30,
    seed: int = 42,
    data_dir_override: Optional[str] = None,
) -> Tuple[List[PuzzleState], BatchMetadata]:
    """
    Build batch B0: 100 initial puzzles with target composition.

    Target: 70 easy (1-4 empties), 30 hard (6-8 empties).
    Degrades gracefully if insufficient puzzles available.
    """
    rng = np.random.default_rng(seed)

    # Find all datasets
    datasets = find_sudoku_4x4_datasets(data_dir_override)
    if not datasets:
        raise RuntimeError(f"No sudoku-4x4 datasets found in {data_dir_override or 'data/'}")

    print(f"[B0] Found datasets: {datasets}")

    # Load all puzzles
    all_puzzles = []
    for ds_path in datasets:
        puzzles = load_puzzles_from_dataset(ds_path, "train")
        all_puzzles.extend(puzzles)
        # Also try test split
        try:
            puzzles_test = load_puzzles_from_dataset(ds_path, "test")
            all_puzzles.extend(puzzles_test)
        except FileNotFoundError:
            pass

    print(f"[B0] Loaded {len(all_puzzles)} total puzzles")

    # Bucket by empties
    easy_puzzles = [p for p in all_puzzles if 1 <= p.empties <= 4]
    hard_puzzles = [p for p in all_puzzles if 6 <= p.empties <= 8]
    medium_puzzles = [p for p in all_puzzles if p.empties == 5]

    print(f"[B0] Easy (1-4 empties): {len(easy_puzzles)}")
    print(f"[B0] Hard (6-8 empties): {len(hard_puzzles)}")
    print(f"[B0] Medium (5 empties): {len(medium_puzzles)}")

    # Sample with graceful degradation
    selected = []

    # Sample easy
    n_easy = min(target_easy, len(easy_puzzles))
    if n_easy > 0:
        indices = rng.choice(len(easy_puzzles), size=n_easy, replace=False)
        selected.extend([easy_puzzles[i] for i in indices])

    # Sample hard
    n_hard = min(target_hard, len(hard_puzzles))
    if n_hard > 0:
        indices = rng.choice(len(hard_puzzles), size=n_hard, replace=False)
        selected.extend([hard_puzzles[i] for i in indices])

    # Fill remainder if needed
    total_needed = target_easy + target_hard
    if len(selected) < total_needed:
        deficit = total_needed - len(selected)
        # Try to fill from medium, then from any bucket
        used_ids = {p.state_id for p in selected}
        remaining = [p for p in all_puzzles if p.state_id not in used_ids]

        if remaining:
            n_fill = min(deficit, len(remaining))
            indices = rng.choice(len(remaining), size=n_fill, replace=False)
            selected.extend([remaining[i] for i in indices])

        print(f"[B0] Warning: Filled {deficit} states from remaining pool")

    # Shuffle and assign stable IDs
    rng.shuffle(selected)
    for i, state in enumerate(selected):
        state.state_id = f"b0_{i:04d}"

    # Build metadata
    empties_dist = {}
    for s in selected:
        empties_dist[s.empties] = empties_dist.get(s.empties, 0) + 1

    metadata = BatchMetadata(
        batch_name="b0",
        num_states=len(selected),
        empties_distribution=empties_dist,
        seed=seed,
        source_paths=datasets,
        creation_time=__import__("datetime").datetime.now().isoformat(),
        git_sha=get_git_sha(),
    )

    print(f"[B0] Built batch with {len(selected)} states")
    print(f"[B0] Empties distribution: {empties_dist}")

    return selected, metadata


def build_b1_batch(
    b0_states: List[PuzzleState],
    model_a: TinyRecursiveReasoningModel_ACTV1,
    model_b: Optional[TinyRecursiveReasoningModel_ACTV1],
    config_a: Dict[str, Any],
    config_b: Optional[Dict[str, Any]],
    n_train: int,
    seed: int = 42,
    cap: int = 1500,
    device: str = "cpu",
) -> Tuple[List[PuzzleState], BatchMetadata]:
    """
    Build batch B1: successor closure from B0.

    For each state in B0:
        - Top-5 actions under Model A at n_train
        - Top-5 actions under Model B at n_train (if available)
        - 5 random legal actions

    Apply actions and deduplicate. Cap at `cap` states.
    """
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig

    rng = np.random.default_rng(seed)

    vocab_size = config_a["vocab_size"]
    seq_len = config_a["seq_len"]
    num_actions = config_a["num_actions"]
    stop_action_id = num_actions - 1  # STOP is last action

    print(f"[B1] Building successor closure from {len(b0_states)} B0 states")
    print(f"[B1] Action space: {num_actions} actions (vocab={vocab_size}, seq={seq_len})")

    # Create environment for action application
    # Use a dummy dataset - we only need apply_edit
    class DummyDataset:
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"inputs": torch.ones(seq_len, dtype=torch.long)}

    env_config = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        vocab_size=vocab_size,
    )
    env = PlanEditEnv(DummyDataset(), lambda x, y: 0.0, env_config)
    env.set_stop_action_id(stop_action_id)

    # Collect all successors
    successors = []
    seen_hashes = set()

    model_a.eval()
    if model_b is not None:
        model_b.eval()

    with torch.no_grad():
        for state in b0_states:
            x = {
                "inputs": state.inputs.unsqueeze(0).to(device),
                "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
            }
            y = state.plan.unsqueeze(0).to(device)

            # Compute action mask (prevent editing clues, mask STOP)
            from rl.envs.plan_edit_env import PlanEditEnv
            action_mask = PlanEditEnv.compute_batch_action_mask(
                state.inputs.unsqueeze(0),
                vocab_size,
                stop_action_id,
                stop_mode="disabled",  # Mask out STOP
            ).to(device)

            legal_actions = action_mask[0].nonzero(as_tuple=True)[0].cpu().tolist()
            # Exclude STOP action
            legal_actions = [a for a in legal_actions if a != stop_action_id]

            if not legal_actions:
                continue

            actions_to_apply = []

            # Top-5 under Model A
            dist_a, _ = model_a.policy_dist(x, y, n_train, action_mask=action_mask)
            probs_a = dist_a.probs[0].cpu()
            # Mask out STOP for ranking
            probs_a[stop_action_id] = 0.0
            top5_a = probs_a.argsort(descending=True)[:5].tolist()
            for a in top5_a:
                if a in legal_actions:
                    actions_to_apply.append((a, "A_top"))

            # Top-5 under Model B (if available)
            if model_b is not None:
                dist_b, _ = model_b.policy_dist(x, y, n_train, action_mask=action_mask)
                probs_b = dist_b.probs[0].cpu()
                probs_b[stop_action_id] = 0.0
                top5_b = probs_b.argsort(descending=True)[:5].tolist()
                for a in top5_b:
                    if a in legal_actions:
                        actions_to_apply.append((a, "B_top"))

            # 5 random legal actions
            n_random = min(5, len(legal_actions))
            random_actions = rng.choice(legal_actions, size=n_random, replace=False).tolist()
            for a in random_actions:
                actions_to_apply.append((a, "rand"))

            # Apply actions and collect successors
            for action_id, source in actions_to_apply:
                new_plan = env.apply_edit(state.plan, action_id, {"inputs": state.inputs})

                # Hash for deduplication (combine inputs + plan)
                combined = torch.cat([state.inputs, new_plan])
                h = stable_hash(combined)

                if h not in seen_hashes:
                    seen_hashes.add(h)

                    new_state = PuzzleState(
                        state_id=f"b1_{len(successors):04d}",
                        inputs=state.inputs.clone(),
                        puzzle_identifier=state.puzzle_identifier.clone(),
                        plan=new_plan,
                        empties=state.empties,  # Same puzzle, different plan
                        source_path=state.source_path,
                        parent_id=state.state_id,
                        action_id=action_id,
                        action_source=source,
                    )
                    successors.append(new_state)

    print(f"[B1] Generated {len(successors)} unique successors")

    # Cap by stable ordering (hash)
    if len(successors) > cap:
        # Sort by hash for deterministic ordering
        successors.sort(key=lambda s: stable_hash(torch.cat([s.inputs, s.plan])))
        successors = successors[:cap]
        print(f"[B1] Capped to {cap} states")

    # Build metadata
    empties_dist = {}
    for s in successors:
        empties_dist[s.empties] = empties_dist.get(s.empties, 0) + 1

    metadata = BatchMetadata(
        batch_name="b1",
        num_states=len(successors),
        empties_distribution=empties_dist,
        seed=seed,
        source_paths=[],  # Derived from B0
        creation_time=__import__("datetime").datetime.now().isoformat(),
        git_sha=get_git_sha(),
        parent_batch="b0",
        cap=cap,
    )

    return successors, metadata


def save_batch(
    states: List[PuzzleState],
    metadata: BatchMetadata,
    out_path: str,
) -> None:
    """Save a batch to disk."""
    # Convert states to tensors
    data = {
        "state_ids": [s.state_id for s in states],
        "inputs": torch.stack([s.inputs for s in states]),
        "puzzle_identifiers": torch.stack([s.puzzle_identifier for s in states]),
        "plans": torch.stack([s.plan for s in states]),
        "empties": torch.tensor([s.empties for s in states]),
        "source_paths": [s.source_path for s in states],
        "parent_ids": [s.parent_id for s in states],
        "action_ids": [s.action_id for s in states],
        "action_sources": [s.action_source for s in states],
        "metadata": asdict(metadata),
    }

    torch.save(data, out_path)
    print(f"[Save] Saved batch to {out_path}")

    # Also save metadata as JSON for easy inspection
    json_path = out_path.replace(".pt", "_metadata.json")
    with open(json_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)


def load_batch(batch_path: str) -> Tuple[List[PuzzleState], BatchMetadata]:
    """Load a batch from disk."""
    data = torch.load(batch_path)

    states = []
    for i in range(len(data["state_ids"])):
        state = PuzzleState(
            state_id=data["state_ids"][i],
            inputs=data["inputs"][i],
            puzzle_identifier=data["puzzle_identifiers"][i],
            plan=data["plans"][i],
            empties=int(data["empties"][i].item()),
            source_path=data["source_paths"][i],
            parent_id=data["parent_ids"][i],
            action_id=data["action_ids"][i],
            action_source=data["action_sources"][i],
        )
        states.append(state)

    metadata = BatchMetadata(**data["metadata"])
    return states, metadata


# =============================================================================
# Projection Stats Instrumentation
# =============================================================================

class ProjectionStatsWrapper:
    """
    Wrapper to capture pre/post projection norms during unroll_latent.

    This is a non-invasive approach that hooks into the model temporarily.
    """

    def __init__(
        self,
        model: TinyRecursiveReasoningModel_ACTV1,
        radius: Optional[float],
    ):
        self.model = model
        self.radius = radius
        self.z_pre_norms: List[float] = []
        self.z_post_norms: List[float] = []
        self._original_project = None
        self._hooked = False

    def __enter__(self):
        if hasattr(self.model, "inner") and hasattr(self.model.inner, "_project_carry_to_ball"):
            self._original_project = self.model.inner._project_carry_to_ball

            def instrumented_project(z_h, z_l, radius):
                z_norm = torch.sqrt(
                    z_h.pow(2).sum(dim=(1, 2))
                    + z_l.pow(2).sum(dim=(1, 2))
                )
                self.z_pre_norms.append(float(z_norm.mean().item()))

                z_h_proj, z_l_proj = self._original_project(z_h, z_l, radius)

                z_proj_norm = torch.sqrt(
                    z_h_proj.pow(2).sum(dim=(1, 2))
                    + z_l_proj.pow(2).sum(dim=(1, 2))
                )
                self.z_post_norms.append(float(z_proj_norm.mean().item()))

                return z_h_proj, z_l_proj

            self.model.inner._project_carry_to_ball = instrumented_project
            self._hooked = True

        return self

    def __exit__(self, *args):
        # Restore original function
        if self._hooked and self._original_project is not None:
            self.model.inner._project_carry_to_ball = self._original_project

    def get_stats(self) -> Tuple[float, float, bool]:
        """
        Get aggregated stats.

        Returns:
            (mean_pre_norm, mean_post_norm, is_saturated)
        """
        if not self.z_pre_norms:
            return 0.0, 0.0, False

        pre = np.mean(self.z_pre_norms)
        post = np.mean(self.z_post_norms)
        saturated = (
            self.radius is not None and pre >= 0.95 * self.radius
        )

        return pre, post, saturated


def joint_latent_delta(
    z_h_a: torch.Tensor,
    z_l_a: torch.Tensor,
    z_h_b: torch.Tensor,
    z_l_b: torch.Tensor,
) -> torch.Tensor:
    """Euclidean product-norm distance between two recurrent carries."""

    delta_h = (z_h_a - z_h_b).reshape(z_h_a.shape[0], -1)
    delta_l = (z_l_a - z_l_b).reshape(z_l_a.shape[0], -1)
    return torch.sqrt(delta_h.square().sum(dim=1) + delta_l.square().sum(dim=1))


# =============================================================================
# Evaluation Logic
# =============================================================================

def evaluate_state_at_depths(
    model: TinyRecursiveReasoningModel_ACTV1,
    state: PuzzleState,
    n_values: List[int],
    config: Dict[str, Any],
    device: str = "cpu",
) -> Dict[Tuple[int, int], EvalMetrics]:
    """
    Evaluate a single state at multiple unroll depths.

    Returns metrics for all pairs (n1, n2) where n1 < n2.
    """
    vocab_size = config["vocab_size"]
    num_actions = config["num_actions"]
    stop_action_id = num_actions - 1
    radius = config.get("latent_ball_radius")

    # Prepare inputs
    x = {
        "inputs": state.inputs.unsqueeze(0).to(device),
        "puzzle_identifiers": state.puzzle_identifier.unsqueeze(0).to(device),
    }
    y = state.plan.unsqueeze(0).to(device)

    # Compute action mask
    from rl.envs.plan_edit_env import PlanEditEnv
    action_mask = PlanEditEnv.compute_batch_action_mask(
        state.inputs.unsqueeze(0),
        vocab_size,
        stop_action_id,
    ).to(device)

    # Collect values, policies, latents at each depth
    values = {}
    policies = {}
    latents = {}
    pre_norms = {}
    post_norms = {}

    model.eval()
    with torch.no_grad():
        for n in n_values:
            # Use projection stats wrapper
            with ProjectionStatsWrapper(model, radius) as psw:
                # Get value
                value, z_n = model.used_value(x, y, n)
                values[n] = value.squeeze().cpu()

                # Get policy distribution
                dist, _ = model.policy_dist(x, y, n, action_mask=action_mask)
                policies[n] = dist.probs.squeeze().cpu()

                latents[n] = (
                    z_n.z_H.detach().cpu(),
                    z_n.z_L.detach().cpu(),
                )

                # Get projection stats
                pre, post, _ = psw.get_stats()
                pre_norms[n] = pre
                post_norms[n] = post

    # Compute metrics for all pairs
    results = {}
    for i, n1 in enumerate(n_values):
        for n2 in n_values[i+1:]:
            # Delta_V
            delta_V = float(torch.abs(values[n1] - values[n2]).item())

            # Delta_pi (KL divergence)
            delta_pi = compute_kl_divergence(policies[n1], policies[n2])

            # Delta_z (L2 norm)
            z_h_1, z_l_1 = latents[n1]
            z_h_2, z_l_2 = latents[n2]
            delta_z = float(
                joint_latent_delta(z_h_1, z_l_1, z_h_2, z_l_2)[0].item()
            )

            # Argmax agreement
            argmax1 = policies[n1].argmax().item()
            argmax2 = policies[n2].argmax().item()
            argmax_agree = 1 if argmax1 == argmax2 else 0

            # Use n1 stats for pre/post norms
            z_pre = pre_norms[n1]
            z_post = post_norms[n1]
            # Disabled projection has no radius; saturation is N/A (use -1).
            if radius is None:
                saturated = -1
            else:
                saturated = 1 if z_pre >= 0.95 * radius else 0

            metrics = EvalMetrics(
                state_id=state.state_id,
                n1=n1,
                n2=n2,
                delta_V=delta_V,
                delta_pi=delta_pi,
                delta_z=delta_z,
                argmax_agree=argmax_agree,
                z_pre_norm=z_pre,
                z_post_norm=z_post,
                saturated=saturated,
            )
            results[(n1, n2)] = metrics

    return results


def evaluate_batch(
    model: TinyRecursiveReasoningModel_ACTV1,
    states: List[PuzzleState],
    n_train: int,
    n_mults: List[int],
    config: Dict[str, Any],
    device: str = "cpu",
) -> List[EvalMetrics]:
    """Evaluate a batch of states."""
    n_values = [n_train * m for m in n_mults]
    print(f"[Eval] Evaluating {len(states)} states at depths {n_values}")

    all_metrics = []
    for i, state in enumerate(states):
        if (i + 1) % 10 == 0:
            print(f"[Eval] Progress: {i+1}/{len(states)}")

        state_metrics = evaluate_state_at_depths(
            model, state, n_values, config, device
        )
        all_metrics.extend(state_metrics.values())

    return all_metrics


def write_per_state_csv(
    metrics: List[EvalMetrics],
    out_path: str,
) -> None:
    """Write per-state metrics to CSV."""
    with open(out_path, "w") as f:
        f.write("state_id,n1,n2,delta_V,delta_pi,delta_z,argmax_agree,z_pre_norm,z_post_norm,saturated\n")
        for m in metrics:
            f.write(f"{m.state_id},{m.n1},{m.n2},{m.delta_V:.6f},{m.delta_pi:.6f},{m.delta_z:.6f},{m.argmax_agree},{m.z_pre_norm:.4f},{m.z_post_norm:.4f},{m.saturated}\n")
    print(f"[CSV] Wrote per-state metrics to {out_path}")


def write_summary_csv(
    metrics: List[EvalMetrics],
    batch_name: str,
    out_path: str,
) -> None:
    """Write summary statistics to CSV."""
    # Group by (n1, n2)
    by_pair = {}
    for m in metrics:
        key = (m.n1, m.n2)
        if key not in by_pair:
            by_pair[key] = []
        by_pair[key].append(m)

    with open(out_path, "w") as f:
        header = "batch,n1,n2"
        for metric in ["delta_V", "delta_pi", "delta_z"]:
            for stat in ["mean", "median", "p90", "p99", "max"]:
                header += f",{metric}_{stat}"
        header += ",argmax_agree_rate,saturation_rate\n"
        f.write(header)

        for (n1, n2), ms in sorted(by_pair.items()):
            row = f"{batch_name},{n1},{n2}"

            for metric in ["delta_V", "delta_pi", "delta_z"]:
                values = [getattr(m, metric) for m in ms]
                agg = aggregate_metrics(values)
                for stat in ["mean", "median", "p90", "p99", "max"]:
                    row += f",{agg[stat]:.6f}"

            # Argmax agreement rate
            argmax_rate = np.mean([m.argmax_agree for m in ms])
            row += f",{argmax_rate:.4f}"

            # Saturation rate. Disabled projection uses -1 as an N/A sentinel.
            valid_sat = [m.saturated for m in ms if m.saturated >= 0]
            sat_rate = np.mean(valid_sat) if valid_sat else -1.0
            row += f",{sat_rate:.4f}"

            f.write(row + "\n")

    print(f"[CSV] Wrote summary to {out_path}")


def write_markdown_table(
    metrics_a: Dict[str, List[EvalMetrics]],
    metrics_b: Dict[str, List[EvalMetrics]],
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    out_path: str,
    n_train: int,
    hat_lz_a: Optional[Dict[str, float]] = None,
    hat_lz_b: Optional[Dict[str, float]] = None,
) -> None:
    """Write comparison markdown table."""
    n_compare = 4 * n_train  # Focus on 1x vs 4x

    with open(out_path, "w") as f:
        f.write("# Unroll Sensitivity Comparison: Model A vs Model B\n\n")
        f.write(f"**Training depth (n_train)**: {n_train}\n")
        f.write(f"**Comparison**: n={n_train} (1×) vs n={n_compare} (4×)\n\n")

        f.write("## Configuration Comparison\n\n")
        f.write("| Setting | Model A | Model B |\n")
        f.write("|---------|---------|--------|\n")
        # Include all key settings
        for key in ["enable_contraction", "target_Lz", "disable_value_head_norm",
                    "episodic_latent", "latent_projection_mode",
                    "latent_ball_radius", "inner_unroll_n", "config_source"]:
            val_a = config_a.get(key, "N/A")
            val_b = config_b.get(key, "N/A")
            f.write(f"| {key} | {val_a} | {val_b} |\n")
        f.write("\n")

        # Sanity checks
        R_a = config_a.get("latent_ball_radius")
        R_b = config_b.get("latent_ball_radius")
        f.write("## Projection Sanity Check\n\n")
        f.write(f"- Model A: mode = {config_a.get('latent_projection_mode')}, R = {R_a}, max Δ_z bound (2R) = {2*R_a if R_a is not None else 'N/A (projection disabled)'}\n")
        f.write(f"- Model B: mode = {config_b.get('latent_projection_mode')}, R = {R_b}, max Δ_z bound (2R) = {2*R_b if R_b is not None else 'N/A (projection disabled)'}\n\n")

        # Hat_Lz section if available
        if hat_lz_a is not None and hat_lz_b is not None:
            f.write("## Achieved Lipschitz Constant (hat_Lz)\n\n")
            f.write("| Statistic | Model A | Model B |\n")
            f.write("|-----------|---------|--------|\n")
            f.write(f"| mean | {hat_lz_a['mean']:.4f} | {hat_lz_b['mean']:.4f} |\n")
            f.write(f"| p95 | {hat_lz_a['p95']:.4f} | {hat_lz_b['p95']:.4f} |\n")
            f.write(f"| max | {hat_lz_a['max']:.4f} | {hat_lz_b['max']:.4f} |\n\n")

        f.write("## Metrics Summary\n\n")

        for batch in ["b0", "b1"]:
            if batch not in metrics_a or batch not in metrics_b:
                continue

            f.write(f"### Batch {batch.upper()}\n\n")
            f.write("| Metric | Statistic | Model A | Model B | Delta |\n")
            f.write("|--------|-----------|---------|---------|-------|\n")

            # Filter to the comparison pair
            ma = [m for m in metrics_a[batch] if m.n1 == n_train and m.n2 == n_compare]
            mb = [m for m in metrics_b[batch] if m.n1 == n_train and m.n2 == n_compare]

            if not ma or not mb:
                f.write("| (no data) | | | | |\n")
                continue

            for metric in ["delta_V", "delta_pi", "delta_z"]:
                vals_a = [getattr(m, metric) for m in ma]
                vals_b = [getattr(m, metric) for m in mb]
                agg_a = aggregate_metrics(vals_a)
                agg_b = aggregate_metrics(vals_b)

                for stat in ["mean", "median", "p99"]:
                    delta = agg_b[stat] - agg_a[stat]
                    sign = "+" if delta > 0 else ""
                    f.write(f"| Δ_{metric.split('_')[1]} | {stat} | {agg_a[stat]:.4f} | {agg_b[stat]:.4f} | {sign}{delta:.4f} |\n")

            # Argmax agreement
            agree_a = np.mean([m.argmax_agree for m in ma])
            agree_b = np.mean([m.argmax_agree for m in mb])
            f.write(f"| argmax_agree | rate | {agree_a:.4f} | {agree_b:.4f} | {agree_b - agree_a:+.4f} |\n")

            # Saturation rate (handle -1 = N/A when projection is disabled).
            valid_sat_a = [m.saturated for m in ma if m.saturated >= 0]
            valid_sat_b = [m.saturated for m in mb if m.saturated >= 0]
            if valid_sat_a and valid_sat_b:
                sat_a = np.mean(valid_sat_a)
                sat_b = np.mean(valid_sat_b)
                f.write(f"| saturation | rate | {sat_a:.4f} | {sat_b:.4f} | {sat_b - sat_a:+.4f} |\n")
            else:
                # Projection-disabled case.
                f.write("| saturation | rate | N/A | N/A | N/A |\n")

            # z_pre and z_post norms
            pre_a = np.mean([m.z_pre_norm for m in ma])
            pre_b = np.mean([m.z_pre_norm for m in mb])
            post_a = np.mean([m.z_post_norm for m in ma])
            post_b = np.mean([m.z_post_norm for m in mb])
            f.write(f"| z_pre_norm | mean | {pre_a:.4f} | {pre_b:.4f} | {pre_b - pre_a:+.4f} |\n")
            f.write(f"| z_post_norm | mean | {post_a:.4f} | {post_b:.4f} | {post_b - post_a:+.4f} |\n")

            f.write("\n")

        f.write("## Interpretation\n\n")
        f.write("- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths\n")
        f.write("- **Higher argmax_agree** means policy decisions are more consistent\n")
        f.write("- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)\n")
        f.write("- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled\n")
        f.write("- Δ_z should respect 2R bound when projection is enabled (post-projection latents)\n")

    print(f"[MD] Wrote summary table to {out_path}")


# =============================================================================
# CLI Subcommands
# =============================================================================

def cmd_build_batches(args):
    """Build evaluation batches B0 and B1."""
    import datetime

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build B0 (always possible)
    print("\n=== Building Batch B0 ===")
    b0_states, b0_meta = build_b0_batch(seed=args.seed, data_dir_override=args.data_dir)
    save_batch(b0_states, b0_meta, str(out_dir / "b0.pt"))

    # Build B1 (requires at least checkpoint_a)
    if not args.checkpoint_a:
        print("\n[B1] Skipped: No checkpoint_a provided")
        return

    print("\n=== Building Batch B1 ===")

    model_a, config_a = load_model_for_eval(
        args.checkpoint_a, device,
        config_yaml_path=getattr(args, 'config_a', None)
    )
    n_train = config_a["inner_unroll_n"]

    model_b = None
    config_b = None
    if args.checkpoint_b:
        model_b, config_b = load_model_for_eval(
            args.checkpoint_b, device,
            config_yaml_path=getattr(args, 'config_b', None)
        )
        b0_meta.checkpoint_b_path = args.checkpoint_b

    b0_meta.checkpoint_a_path = args.checkpoint_a

    b1_states, b1_meta = build_b1_batch(
        b0_states,
        model_a,
        model_b,
        config_a,
        config_b,
        n_train,
        seed=args.seed,
        cap=args.cap_b1,
        device=device,
    )

    b1_meta.checkpoint_a_path = args.checkpoint_a
    b1_meta.checkpoint_b_path = args.checkpoint_b

    save_batch(b1_states, b1_meta, str(out_dir / "b1.pt"))

    print("\n=== Done ===")


def cmd_eval(args):
    """Evaluate a single checkpoint on batches."""
    import datetime

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with config YAML if provided
    model, config = load_model_for_eval(
        args.checkpoint, device,
        config_yaml_path=getattr(args, 'config_yaml', None)
    )
    n_train = config["inner_unroll_n"]
    n_mults = [int(m) for m in args.n_mults.split(",")]

    # Load batches
    b0_states, b0_meta = load_batch(args.batch_b0)
    print(f"[Eval] Loaded B0: {len(b0_states)} states")

    b1_states = []
    if args.batch_b1 and Path(args.batch_b1).exists():
        b1_states, b1_meta = load_batch(args.batch_b1)
        print(f"[Eval] Loaded B1: {len(b1_states)} states")

    # Get checkpoint name for output files
    ckpt_name = Path(args.checkpoint).stem

    # Evaluate B0
    print("\n=== Evaluating B0 ===")
    b0_metrics = evaluate_batch(model, b0_states, n_train, n_mults, config, device)
    write_per_state_csv(b0_metrics, str(out_dir / f"{ckpt_name}_b0_per_state.csv"))
    write_summary_csv(b0_metrics, "b0", str(out_dir / f"{ckpt_name}_b0_summary.csv"))

    # Evaluate B1
    if b1_states:
        print("\n=== Evaluating B1 ===")
        b1_metrics = evaluate_batch(model, b1_states, n_train, n_mults, config, device)
        write_per_state_csv(b1_metrics, str(out_dir / f"{ckpt_name}_b1_per_state.csv"))
        write_summary_csv(b1_metrics, "b1", str(out_dir / f"{ckpt_name}_b1_summary.csv"))

    # Save run metadata
    run_meta = RunMetadata(
        checkpoint_path=args.checkpoint,
        checkpoint_config=config,
        n_train=n_train,
        n_mults=n_mults,
        seed=args.seed,
        batch_b0_path=args.batch_b0,
        batch_b1_path=args.batch_b1,
        git_sha=get_git_sha(),
        creation_time=__import__("datetime").datetime.now().isoformat(),
    )

    with open(out_dir / f"{ckpt_name}_run_metadata.json", "w") as f:
        json.dump(asdict(run_meta), f, indent=2)

    print("\n=== Done ===")


def cmd_compare(args):
    """Compare two checkpoints."""
    import datetime

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models with config YAML paths if provided
    radius_override = getattr(args, 'latent_ball_radius_override', None)
    mode_override = getattr(args, 'latent_projection_mode_override', None)
    model_a, config_a = load_model_for_eval(
        args.checkpoint_a, device,
        config_yaml_path=getattr(args, 'config_a', None),
        latent_ball_radius_override=radius_override,
        latent_projection_mode_override=mode_override,
    )
    model_b, config_b = load_model_for_eval(
        args.checkpoint_b, device,
        config_yaml_path=getattr(args, 'config_b', None),
        latent_ball_radius_override=radius_override,
        latent_projection_mode_override=mode_override,
    )

    # Check compatibility
    compatible, diffs = check_model_compatibility(config_a, config_b)
    if not compatible:
        print("\n[Compare] ERROR: Models are not compatible!")
        print("Differences:")
        for d in diffs:
            print(f"  - {d}")
        print("\nAllowing single-model eval but refusing compare.")
        return 1

    n_train = config_a["inner_unroll_n"]

    # Determine depth values: either from --n_values (absolute) or --n_mults (relative)
    n_values_arg = getattr(args, 'n_values', None)
    if n_values_arg:
        # Absolute depth values
        n_values = [int(n) for n in n_values_arg.split(",")]
        # Convert to multipliers for compatibility with existing code
        n_mults = [n // n_train for n in n_values]
        print(f"[Compare] Using absolute depths: {n_values} (n_train={n_train})")
    else:
        n_mults = [int(m) for m in args.n_mults.split(",")]
        n_values = [n_train * m for m in n_mults]
        print(f"[Compare] Using depth multipliers: {n_mults} -> depths {n_values}")

    # Load batches
    b0_states, _ = load_batch(args.batch_b0)
    b1_states = []
    if args.batch_b1 and Path(args.batch_b1).exists():
        b1_states, _ = load_batch(args.batch_b1)

    # Evaluate both models
    metrics_a = {}
    metrics_b = {}

    print("\n=== Evaluating Model A ===")
    metrics_a["b0"] = evaluate_batch(model_a, b0_states, n_train, n_mults, config_a, device)
    if b1_states:
        metrics_a["b1"] = evaluate_batch(model_a, b1_states, n_train, n_mults, config_a, device)

    print("\n=== Evaluating Model B ===")
    metrics_b["b0"] = evaluate_batch(model_b, b0_states, n_train, n_mults, config_b, device)
    if b1_states:
        metrics_b["b1"] = evaluate_batch(model_b, b1_states, n_train, n_mults, config_b, device)

    # Estimate hat_Lz if requested
    hat_lz_a = None
    hat_lz_b = None
    if getattr(args, 'estimate_lz', False):
        print("\n=== Estimating hat_Lz ===")
        all_states = b0_states + (b1_states if b1_states else [])
        hat_lz_a = estimate_hat_Lz_batch(model_a, all_states, n_train, config_a, device, seed=args.seed)
        hat_lz_b = estimate_hat_Lz_batch(model_b, all_states, n_train, config_b, device, seed=args.seed)
        print(f"[hat_Lz] Model A: mean={hat_lz_a['mean']:.4f}, p95={hat_lz_a['p95']:.4f}, max={hat_lz_a['max']:.4f}")
        print(f"[hat_Lz] Model B: mean={hat_lz_b['mean']:.4f}, p95={hat_lz_b['p95']:.4f}, max={hat_lz_b['max']:.4f}")

    # Write CSVs
    for label, model_metrics in [("model_a", metrics_a), ("model_b", metrics_b)]:
        for batch, ms in model_metrics.items():
            write_per_state_csv(ms, str(out_dir / f"{label}_{batch}_per_state.csv"))
            write_summary_csv(ms, batch, str(out_dir / f"{label}_{batch}_summary.csv"))

    # Write comparison markdown
    write_markdown_table(
        metrics_a, metrics_b,
        config_a, config_b,
        args.table_out,
        n_train,
        hat_lz_a=hat_lz_a,
        hat_lz_b=hat_lz_b,
    )

    # Save run metadata
    run_meta = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "config_a": config_a,
        "config_b": config_b,
        "n_train": n_train,
        "n_mults": n_mults,
        "n_values": n_values,
        "seed": args.seed,
        "git_sha": get_git_sha(),
        "creation_time": __import__("datetime").datetime.now().isoformat(),
        "hat_lz_a": hat_lz_a,
        "hat_lz_b": hat_lz_b,
    }

    with open(out_dir / "compare_run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    print("\n=== Done ===")
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unroll Sensitivity Evaluation (ICML Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # build-batches
    p_build = subparsers.add_parser("build-batches", help="Build evaluation batches")
    p_build.add_argument("--checkpoint_a", type=str, help="Path to Model A checkpoint")
    p_build.add_argument("--checkpoint_b", type=str, help="Path to Model B checkpoint (optional)")
    p_build.add_argument("--config_a", type=str, help="Path to Model A training config YAML")
    p_build.add_argument("--config_b", type=str, help="Path to Model B training config YAML")
    p_build.add_argument("--out_dir", type=str, default="artifacts/eval_batches/")
    p_build.add_argument("--data_dir", type=str, default=None, help="Path to data directory (overrides PROJECT_ROOT/data)")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--cap_b1", type=int, default=1500, help="Max states in B1")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--config_yaml", type=str, help="Path to training config YAML for this checkpoint")
    p_eval.add_argument("--batch_b0", type=str, required=True)
    p_eval.add_argument("--batch_b1", type=str, default=None)
    p_eval.add_argument("--out_dir", type=str, default="results/plot_data/unroll_sensitivity/")
    p_eval.add_argument("--n_mults", type=str, default="1,2,4", help="Comma-separated depth multipliers (relative to n_train)")
    p_eval.add_argument("--n_values", type=str, default=None, help="Comma-separated absolute depth values (overrides n_mults)")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--estimate_lz", action="store_true", help="Estimate hat_Lz on the batch")

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare two checkpoints")
    p_cmp.add_argument("--checkpoint_a", type=str, required=True)
    p_cmp.add_argument("--checkpoint_b", type=str, required=True)
    p_cmp.add_argument("--config_a", type=str, help="Path to Model A training config YAML")
    p_cmp.add_argument("--config_b", type=str, help="Path to Model B training config YAML")
    p_cmp.add_argument("--batch_b0", type=str, required=True)
    p_cmp.add_argument("--batch_b1", type=str, default=None)
    p_cmp.add_argument("--out_dir", type=str, default="results/plot_data/unroll_sensitivity/")
    p_cmp.add_argument("--table_out", type=str, default="results/tables/unroll_sensitivity_summary.md")
    p_cmp.add_argument("--n_mults", type=str, default="1,2,4", help="Comma-separated depth multipliers (relative to n_train)")
    p_cmp.add_argument("--n_values", type=str, default=None, help="Comma-separated absolute depth values (overrides n_mults)")
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--latent_ball_radius_override", type=float, default=None,
                       help="Enable projection at positive R for BOTH models")
    p_cmp.add_argument(
        "--latent_projection_mode_override",
        choices=("enabled", "disabled"),
        default=None,
        help="Explicit recurrent projection mode for BOTH models",
    )
    p_cmp.add_argument("--estimate_lz", action="store_true", help="Estimate hat_Lz on the batch")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "build-batches":
        cmd_build_batches(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "compare":
        return cmd_compare(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
