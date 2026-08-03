
import argparse
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Workaround for A100 (CC 8.0) when PyTorch build lacks sm_80 kernels.
# Disables SDPA backends that may cause hangs due to kernel fallback/JIT.
if hasattr(torch.backends, 'cuda'):
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)

try:
    from tqdm import trange
except ImportError:  # pragma: no cover
    trange = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None
    WANDB_AVAILABLE = False

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.norec_encoder import NoRecursionEncoder, NoRecEncoderConfig
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.upi_trm_trainer import UPITrmTrainer
from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.algos.a2c import A2CTrainer, A2CConfig
from rl.algos.dqn import DQNTrainer, DQNConfig
from rl.sudoku_checkers import (
    dummy_checker,
    sudoku_checker,
    sudoku_constraint_checker,
    sudoku_feasibility_checker,
    sudoku_progress_checker,
)
from rl.training_setup import (
    DummyPuzzleDataset,
    OfflinePuzzleDataset,
    build_dataset_from_paths,
    offset_puzzle_identifiers,
    resolve_checker_from_dataset,
)
from rl.sudoku_utils import sudoku_is_solved, sudoku_get_stats
from utils.seeding import set_global_seed
from utils.dataset_provenance import (
    DatasetProvenanceError,
    assert_matching_dataset_provenance,
    build_dataset_provenance,
    dataset_input_sha256s,
    dataset_pool_sha256,
    dataset_sample_sha256s,
    dataset_source_build_metadata,
    validate_dataset_provenance,
)

__all__ = [
    "BaselineSelection",
    "DummyPuzzleDataset",
    "OfflinePuzzleDataset",
    "build_dataset_from_paths",
    "resolve_checker_from_dataset",
    "dummy_checker",
    "sudoku_checker",
    "sudoku_constraint_checker",
    "sudoku_feasibility_checker",
    "sudoku_progress_checker",
    "select_baseline_from_configs",
    "build_trainer",
    "load_checkpoint",
    "save_checkpoint",
    "resume_from_checkpoint",
]


# =============================================================================
# Baseline Selection Helper (SINGLE SOURCE OF TRUTH)
# =============================================================================

class BaselineSelection:
    """
    Result of baseline selection from CLI and YAML configs.
    
    This is the SINGLE SOURCE OF TRUTH for baseline selection logic.
    main() and tests both use this to ensure consistency.
    """
    def __init__(
        self,
        selected_baseline: Optional[str],
        yaml_algorithm: Optional[str],
        get_yaml_key: callable,
    ):
        self.selected_baseline = selected_baseline  # Effective baseline: CLI > YAML > None
        self.yaml_algorithm = yaml_algorithm  # Algorithm from YAML (for logging)
        self.get_yaml_key = get_yaml_key  # Function to retrieve config values


def select_baseline_from_configs(
    cli_baseline: Optional[str],
    config_paths: Optional[List[str]],
) -> BaselineSelection:
    """
    Determine baseline algorithm from CLI flag and/or YAML configs.
    
    This is the SINGLE SOURCE OF TRUTH for baseline selection logic.
    main() calls this function; tests also call this function.
    
    Args:
        cli_baseline: Value from --baseline CLI arg (None, "ppo", "a2c", "dqn", "ddqn")
        config_paths: List of YAML config file paths from --config args
    
    Returns:
        BaselineSelection with:
        - selected_baseline: The effective baseline ("ppo", "a2c", "dqn", "ddqn", or None for UPI-TRM)
        - yaml_algorithm: The algorithm specified in YAML (for logging, may differ from selected)
        - get_yaml_key: A function get_yaml_key(key, default) to retrieve baseline-specific config values
        
    Override order:
    - CLI --baseline overrides YAML algorithm
    - Later --config files override earlier ones (last wins)
    """
    import yaml
    
    # Extract algorithm from YAML configs (last config wins)
    # Iterate forward - each iteration overwrites, so last config wins
    yaml_algorithm = None
    if config_paths is not None:
        for config_path in config_paths:
            with open(config_path, "r") as f:
                override = yaml.safe_load(f) or {}
            if "algorithm" in override:
                yaml_algorithm = override["algorithm"].lower()
    
    # CLI overrides YAML
    selected_baseline = cli_baseline
    if selected_baseline is None and yaml_algorithm is not None:
        if yaml_algorithm in ("ppo", "a2c", "dqn", "ddqn"):
            selected_baseline = yaml_algorithm
    
    # Create get_yaml_key helper that searches in reverse order (last config wins)
    def get_yaml_key(key: str, default):
        """Get key from YAML configs (last config wins)."""
        if config_paths is not None:
            for config_path in reversed(config_paths):
                with open(config_path, "r") as f:
                    override = yaml.safe_load(f) or {}
                if key in override:
                    return override[key]
        return default
    
    return BaselineSelection(selected_baseline, yaml_algorithm, get_yaml_key)


# =============================================================================
# Trainer Construction Helper (SINGLE SOURCE OF TRUTH for trainer instantiation)
# =============================================================================

def build_trainer(
    model: nn.Module,
    env,  # PlanEditEnv
    rl_cfg: "RLConfig",
    device: torch.device,
    baseline_selection: BaselineSelection,
    cli_baseline: Optional[str] = None,
    verbose: bool = True,
) -> "Union[UPITrmTrainer, PPOTrainer, A2CTrainer, DQNTrainer]":
    """
    Construct the appropriate trainer based on baseline selection.
    
    This is the SINGLE SOURCE OF TRUTH for trainer instantiation.
    main() calls this function; tests also call this function directly.
    
    Args:
        model: The neural network model (TRM or NoRecursionEncoder)
        env: The PlanEditEnv environment
        rl_cfg: RLConfig with training hyperparameters
        device: torch device (cpu/cuda)
        baseline_selection: Result from select_baseline_from_configs()
        cli_baseline: Original CLI --baseline value (for logging only)
        verbose: Whether to print trainer selection logs
    
    Returns:
        The constructed trainer (UPITrmTrainer, PPOTrainer, A2CTrainer, or DQNTrainer)
    """
    from rl.upi_trm_trainer import UPITrmTrainer
    from rl.algos.ppo import PPOTrainer, PPOConfig
    from rl.algos.a2c import A2CTrainer, A2CConfig
    from rl.algos.dqn import DQNTrainer, DQNConfig, compute_epsilon_decay_steps
    
    selected_baseline = baseline_selection.selected_baseline
    yaml_algorithm = baseline_selection.yaml_algorithm
    get_yaml_key = baseline_selection.get_yaml_key
    
    if verbose:
        print("=" * 60)
        print("TRAINER SELECTION")
        print("=" * 60)
    
    trainer = None
    
    if selected_baseline is None:
        # Default: UPI-TRM (theory-aligned algorithm)
        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=device)
        if verbose:
            print(f"[TRAINER] UPI-TRM (K={rl_cfg.K}, inner_n={rl_cfg.inner_unroll_n})")
            print(f"[TRAINER] CLI --baseline: {cli_baseline}, YAML algorithm: {yaml_algorithm}")
    
    elif selected_baseline == "ppo":
        # PPO baseline - wire YAML keys
        # MINIBATCH MAPPING:
        # - ppo_num_minibatches: count of minibatches per epoch (takes precedence)
        # - ppo_minibatch_size: size in samples => num_minibatches = num_steps // size
        ppo_num_steps = get_yaml_key("ppo_num_steps", 128)
        ppo_num_minibatches_explicit = get_yaml_key("ppo_num_minibatches", None)
        ppo_minibatch_size_explicit = get_yaml_key("ppo_minibatch_size", None)
        
        if ppo_num_minibatches_explicit is not None:
            ppo_num_minibatches = ppo_num_minibatches_explicit
            ppo_minibatch_size = ppo_num_steps // max(1, ppo_num_minibatches)
        elif ppo_minibatch_size_explicit is not None:
            if ppo_minibatch_size_explicit > ppo_num_steps:
                if verbose:
                    print(f"[WARNING] ppo_minibatch_size={ppo_minibatch_size_explicit} > ppo_num_steps={ppo_num_steps}")
                    print(f"[WARNING] This is likely wrong. Using num_minibatches=1 (full batch).")
                ppo_num_minibatches = 1
                ppo_minibatch_size = ppo_num_steps
            else:
                ppo_num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size_explicit)
                ppo_minibatch_size = ppo_minibatch_size_explicit
        else:
            ppo_num_minibatches = 4
            ppo_minibatch_size = ppo_num_steps // ppo_num_minibatches
        
        ppo_cfg = PPOConfig(
            clip_eps=get_yaml_key("ppo_clip_eps", 0.2),
            vf_coef=get_yaml_key("vf_coef", 0.5),
            entropy_coef=rl_cfg.entropy_coef,
            max_grad_norm=get_yaml_key("max_grad_norm", 0.5),
            num_steps=ppo_num_steps,
            num_epochs=get_yaml_key("ppo_epochs", 4),
            num_minibatches=ppo_num_minibatches,
            gamma=rl_cfg.gamma,
            gae_lambda=get_yaml_key("gae_lambda", 0.95),
            normalize_advantages=get_yaml_key("normalize_advantages", True),
            clip_vf_loss=get_yaml_key("clip_vf_loss", False),
            policy_lr=rl_cfg.policy_lr,
            value_lr=rl_cfg.value_lr,
            backbone_lr=rl_cfg.backbone_lr,
            inner_unroll_n=rl_cfg.inner_unroll_n,
            log_interval=rl_cfg.log_interval,
            eval_interval=rl_cfg.eval_interval,
            num_train_steps=rl_cfg.num_train_steps,
            eval_num_episodes=rl_cfg.eval_num_episodes,
        )
        trainer = PPOTrainer(model=model, env=env, config=ppo_cfg, device=device)
        if verbose:
            print(f"[TRAINER] PPO baseline selected")
            print(f"[TRAINER] CLI --baseline: {cli_baseline}, YAML algorithm: {yaml_algorithm}")
            print(f"[TRAINER] PPO config: clip_eps={ppo_cfg.clip_eps}, epochs={ppo_cfg.num_epochs}, "
                  f"num_steps={ppo_cfg.num_steps}, num_minibatches={ppo_cfg.num_minibatches}, "
                  f"minibatch_size={ppo_minibatch_size}")
    
    elif selected_baseline == "a2c":
        # A2C baseline - wire YAML keys
        a2c_cfg = A2CConfig(
            vf_coef=get_yaml_key("vf_coef", 0.5),
            entropy_coef=rl_cfg.entropy_coef,
            max_grad_norm=get_yaml_key("max_grad_norm", 0.5),
            num_steps=get_yaml_key("a2c_num_steps", 5),
            gamma=rl_cfg.gamma,
            use_gae=get_yaml_key("use_gae", True),
            gae_lambda=get_yaml_key("gae_lambda", 0.95),
            lr=rl_cfg.policy_lr,
            inner_unroll_n=rl_cfg.inner_unroll_n,
            log_interval=rl_cfg.log_interval,
            eval_interval=rl_cfg.eval_interval,
            num_train_steps=rl_cfg.num_train_steps,
            eval_num_episodes=rl_cfg.eval_num_episodes,
        )
        trainer = A2CTrainer(model=model, env=env, config=a2c_cfg, device=device)
        if verbose:
            print(f"[TRAINER] A2C baseline selected")
            print(f"[TRAINER] CLI --baseline: {cli_baseline}, YAML algorithm: {yaml_algorithm}")
            print(f"[TRAINER] A2C config: num_steps={a2c_cfg.num_steps}, use_gae={a2c_cfg.use_gae}, "
                  f"gae_lambda={a2c_cfg.gae_lambda}")
    
    elif selected_baseline in ("dqn", "ddqn"):
        # DQN/Double DQN baseline - wire YAML keys
        use_double = (selected_baseline == "ddqn") or get_yaml_key("dqn_double_dqn", False)
        train_freq = get_yaml_key("dqn_train_freq", 4)
        exploration_fraction = get_yaml_key("dqn_exploration_fraction", 0.1)
        epsilon_decay_steps = compute_epsilon_decay_steps(
            num_train_steps=rl_cfg.num_train_steps,
            exploration_fraction=exploration_fraction,
            train_freq=train_freq,
        )
        
        dqn_cfg = DQNConfig(
            double_dqn=use_double,
            gamma=rl_cfg.gamma,
            epsilon_start=get_yaml_key("dqn_exploration_initial_eps", 1.0),
            epsilon_end=get_yaml_key("dqn_exploration_final_eps", 0.01),
            epsilon_decay_steps=epsilon_decay_steps,
            buffer_size=get_yaml_key("dqn_buffer_size", 10000),
            batch_size=get_yaml_key("dqn_batch_size", 64),
            min_buffer_size=get_yaml_key("dqn_learning_starts", 500),
            target_update_freq=get_yaml_key("dqn_target_update_interval", 100),
            learning_rate=rl_cfg.value_lr,
            max_grad_norm=get_yaml_key("max_grad_norm", 1.0),
            dqn_n_step=get_yaml_key("dqn_n_step", 1),
            inner_unroll_n=rl_cfg.inner_unroll_n,
            train_freq=train_freq,
            gradient_steps=get_yaml_key("dqn_gradient_steps", 1),
            num_train_steps=rl_cfg.num_train_steps,
            log_interval=rl_cfg.log_interval,
            eval_interval=rl_cfg.eval_interval,
            eval_num_episodes=rl_cfg.eval_num_episodes,
        )
        trainer = DQNTrainer(model=model, env=env, config=dqn_cfg, device=device)
        if verbose:
            algo_name = "Double DQN" if use_double else "DQN"
            print(f"[TRAINER] {algo_name} baseline selected")
            print(f"[TRAINER] CLI --baseline: {cli_baseline}, YAML algorithm: {yaml_algorithm}")
            print(f"[TRAINER] DQN config: buffer={dqn_cfg.buffer_size}, batch={dqn_cfg.batch_size}, "
                  f"target_update={dqn_cfg.target_update_freq}, epsilon_decay={epsilon_decay_steps}, "
                  f"n_step={dqn_cfg.dqn_n_step}")
    
    else:
        raise ValueError(f"Unknown baseline algorithm: {selected_baseline}")
    
    if verbose:
        print("=" * 60)
    
    return trainer


# =============================================================================
# Checkpoint Loading/Saving (aligned with pretrain.py)
# =============================================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load model weights from a pretrained checkpoint.
    
    Handles puzzle embedding resizing if shapes don't match.
    
    Args:
        model: The TRM model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Target device (auto-detected if None)
        strict: If True, raise error on missing/unexpected keys
        
    Returns:
        Dict with loading info (missing_keys, unexpected_keys, etc.)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Checkpoint] Loading from {checkpoint_path} to {device}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle torch.compile wrapper naming
    # Pretrained models may have "_orig_mod." prefix from torch.compile
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove _orig_mod. prefix if present
        clean_key = key.replace("_orig_mod.", "")
        # Also handle model.inner -> inner mapping for wrapped models
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]  # Remove "model." prefix
        cleaned_state_dict[clean_key] = value
    
    # Handle puzzle embedding size mismatch
    puzzle_emb_key = "inner.puzzle_emb.weights"
    if puzzle_emb_key in cleaned_state_dict:
        loaded_emb = cleaned_state_dict[puzzle_emb_key]
        if hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb"):
            expected_shape = model.inner.puzzle_emb.weights.shape
            if loaded_emb.shape != expected_shape:
                print(f"[Checkpoint] Resizing puzzle embeddings: {loaded_emb.shape} -> {expected_shape}")
                # Resize by averaging existing embeddings
                if loaded_emb.shape[1] == expected_shape[1]:
                    # Same embedding dim, different num identifiers
                    if loaded_emb.shape[0] < expected_shape[0]:
                        # Expand by replicating mean
                        mean_emb = loaded_emb.mean(dim=0, keepdim=True)
                        cleaned_state_dict[puzzle_emb_key] = mean_emb.expand(expected_shape).contiguous()
                    else:
                        # Truncate
                        cleaned_state_dict[puzzle_emb_key] = loaded_emb[:expected_shape[0]]
                else:
                    # Different dims - reinitialize
                    print(f"[Checkpoint] Puzzle embedding dim mismatch, reinitializing")
                    del cleaned_state_dict[puzzle_emb_key]
    
    # Skip RL heads if loading from supervised checkpoint (they won't exist)
    # Filter out keys that don't exist in the model
    model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    filtered_state_dict = {}
    skipped_keys = []
    for key, value in cleaned_state_dict.items():
        if key in model_keys or any(key.startswith(mk.rsplit(".", 1)[0]) for mk in model_keys):
            filtered_state_dict[key] = value
        else:
            skipped_keys.append(key)
    
    if skipped_keys:
        print(f"[Checkpoint] Skipped {len(skipped_keys)} keys not in model: {skipped_keys[:5]}...")
    
    # Load with strict=False to allow missing RL heads
    result = model.load_state_dict(filtered_state_dict, strict=strict)
    
    print(f"[Checkpoint] Loaded successfully")
    if result.missing_keys:
        print(f"[Checkpoint] Missing keys (will use random init): {result.missing_keys[:10]}...")
    if result.unexpected_keys:
        print(f"[Checkpoint] Unexpected keys (ignored): {result.unexpected_keys[:10]}...")
    
    return {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys}


def _capture_rng_state() -> Dict[str, Any]:
    """Capture every process-global RNG used by the training paths."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    """Restore a state produced by :func:`_capture_rng_state`."""

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(rng_state["torch_cpu"])
    cuda_rng_state = rng_state.get("torch_cuda")
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)


def _config_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("Checkpoint configuration must be a Pydantic model or dictionary.")


def save_checkpoint(
    model: nn.Module,
    trainer: "UPITrmTrainer",
    step: int,
    checkpoint_dir: str,
    puzzle_emb_optimizer: Optional[torch.optim.Optimizer] = None,
    rl_cfg: Optional["RLConfig"] = None,
    dataset_provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save full training state for resumable RL training.

    Args:
        model: The TRM model
        trainer: UPITrmTrainer instance (for optimizers)
        step: Current training step
        checkpoint_dir: Directory to save checkpoints
        puzzle_emb_optimizer: Optional optimizer for puzzle embeddings
        rl_cfg: Optional RLConfig used for training (saved for reproducibility)
        dataset_provenance: Content-based train/eval dataset identity

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    is_exact_upi_checkpoint = isinstance(trainer, UPITrmTrainer)
    if is_exact_upi_checkpoint and dataset_provenance is None:
        raise RuntimeError(
            "Schema-v3 exact checkpoints require dataset provenance."
        )
    canonical_dataset_provenance: Optional[Dict[str, Any]] = None
    if dataset_provenance is not None:
        if is_exact_upi_checkpoint:
            canonical_dataset_provenance = validate_dataset_provenance(
                dataset_provenance
            )
        else:
            # Baseline checkpoints are weights-only. Preserve their historical
            # optional metadata without presenting it as exact-resume identity.
            canonical_dataset_provenance = dict(dataset_provenance)

    rng_state = _capture_rng_state()
    checkpoint = {
        "checkpoint_schema_version": 3 if is_exact_upi_checkpoint else 2,
        "trainer_kind": type(trainer).__name__,
        "step": step,
        "progress": {
            "env_steps": int(getattr(trainer, "_env_step_count", 0)),
            "optimizer_updates": int(getattr(trainer, "_train_step_count", 0)),
        },
        "model_state_dict": model.state_dict(),
        "rng_state": rng_state,
    }
    if canonical_dataset_provenance is not None:
        checkpoint["dataset_provenance"] = canonical_dataset_provenance
    model_config = getattr(model, "config", None)
    if model_config is not None:
        checkpoint["model_config"] = _config_dict(model_config)

    # Preserve the old/candidate policy pair for post-candidate diagnostics.
    if hasattr(trainer, "policy_model_old") and trainer.policy_model_old is not None:
        checkpoint["policy_model_old_state_dict"] = trainer.policy_model_old.state_dict()
    if hasattr(trainer, "policy_model_candidate") and trainer.policy_model_candidate is not None:
        checkpoint["policy_model_candidate_state_dict"] = trainer.policy_model_candidate.state_dict()
    if hasattr(trainer, "target_model") and trainer.target_model is not None:
        checkpoint["target_model_state_dict"] = trainer.target_model.state_dict()

    # Save RL config for reproducibility and correct eval loading
    effective_rl_cfg = rl_cfg if rl_cfg is not None else getattr(trainer, "rl_cfg", None)
    if effective_rl_cfg is None:
        raise RuntimeError("Exact checkpoint requires the active RL configuration.")
    checkpoint["rl_config"] = _config_dict(effective_rl_cfg)

    # Save optimizer states - different trainers have different optimizer structures
    if hasattr(trainer, 'value_opt') and hasattr(trainer, 'policy_opt'):
        # UPITrmTrainer has separate value and policy optimizers
        checkpoint["value_optimizer_state_dict"] = trainer.value_opt.state_dict()
        checkpoint["policy_optimizer_state_dict"] = trainer.policy_opt.state_dict()
        if getattr(trainer, "old_policy_distill_opt", None) is not None:
            checkpoint["old_policy_distill_optimizer_state_dict"] = (
                trainer.old_policy_distill_opt.state_dict()
            )
    elif hasattr(trainer, 'optimizer'):
        # PPO/A2C have a single combined optimizer
        checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

    if puzzle_emb_optimizer is not None:
        checkpoint["puzzle_emb_optimizer_state_dict"] = puzzle_emb_optimizer.state_dict()

    if getattr(trainer, "value_scheduler", None) is not None:
        checkpoint["value_scheduler_state_dict"] = trainer.value_scheduler.state_dict()
    if getattr(trainer, "policy_scheduler", None) is not None:
        checkpoint["policy_scheduler_state_dict"] = trainer.policy_scheduler.state_dict()

    checkpoint["trainer_state"] = {
        "next_episode_id": int(getattr(trainer, "_next_episode_id", 0)),
        "train_step_count": int(getattr(trainer, "_train_step_count", 0)),
        "env_step_count": int(getattr(trainer, "_env_step_count", 0)),
        "kl_coef": float(getattr(trainer, "_kl_coef", 1.0)),
        "term_stats": dict(getattr(trainer, "term_stats", {})),
        "debug_episode_lengths": list(
            getattr(trainer, "_debug_episode_lengths", [])
        ),
        "debug_episode_returns": list(
            getattr(trainer, "_debug_episode_returns", [])
        ),
        "debug_stop_probs": list(getattr(trainer, "_debug_stop_probs", [])),
        "debug_score_changes": list(
            getattr(trainer, "_debug_score_changes", [])
        ),
        "drift_values": list(getattr(trainer, "_drift_values", [])),
        "plan_changes": list(getattr(trainer, "_plan_changes", [])),
        "value_of_memory": list(getattr(trainer, "_value_of_memory", [])),
    }
    if is_exact_upi_checkpoint:
        checkpoint["trainer_state"].update(
            {
                "collection_state": trainer.collection_checkpoint_state(),
                "environment_state": trainer.env.checkpoint_state(),
            }
        )
    else:
        print(
            "[Checkpoint] Baseline trainer checkpoint is weights-only for future "
            "warm starts; exact resume requires the schema-v3 UPI path."
        )

    # Replay is required for a semantic resume. It can make checkpoints large,
    # but storing only its length caused resumed runs to start from empty data.
    if hasattr(trainer, 'replay'):
        checkpoint["replay_buffer_size"] = len(trainer.replay)
        checkpoint["replay_capacity"] = trainer.replay.storage.maxlen
        checkpoint["replay_transitions"] = list(trainer.replay.storage)
    
    path = os.path.join(checkpoint_dir, f"rl_checkpoint_step_{step}.pt")
    try:
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")

        # Also save just the model weights for easy loading
        model_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"[Checkpoint] Warning: Failed to save checkpoint: {e}")
        print("[Checkpoint] Continuing training without saving...")
        return ""
    finally:
        # Checkpointing must not perturb an uninterrupted stochastic run.
        _restore_rng_state(rng_state)

    return path


def resume_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    trainer: "UPITrmTrainer",
    device: str,
    puzzle_emb_optimizer: Optional[torch.optim.Optimizer] = None,
    expected_dataset_provenance: Optional[Dict[str, Any]] = None,
    allow_legacy_warm_start: bool = False,
) -> int:
    """
    Resume RL training from a saved checkpoint.
    
    Returns:
        Starting step number
    """
    print(f"[Checkpoint] Resuming from {checkpoint_path}")
    # Resume checkpoints contain replay Transition dataclasses, so this is a
    # trusted local artifact rather than a weights-only file.
    checkpoint = torch.load(
        checkpoint_path,
        # Replay transitions are intentionally CPU-resident. Loading a large
        # replay directly onto CUDA can exhaust accelerator memory before the
        # model state is restored.
        map_location="cpu",
        weights_only=False,
    )
    
    schema_version = int(checkpoint.get("checkpoint_schema_version", 0))
    if schema_version < 3:
        legacy_flag_note = (
            " The deprecated allow_legacy_warm_start flag cannot make this an "
            "exact resume."
            if allow_legacy_warm_start
            else ""
        )
        raise RuntimeError(
            "Checkpoint predates schema-v3 live-episode and provenance support; "
            "resume is refused before state mutation. Use the ordinary "
            "--load-checkpoint weights-only flow for an explicit non-resume warm "
            "start."
            + legacy_flag_note
        )
    if schema_version != 3:
        raise RuntimeError(
            f"Unsupported checkpoint schema version {schema_version}; expected 3."
        )
    if checkpoint.get("trainer_kind") != type(trainer).__name__:
        raise RuntimeError(
            "Checkpoint trainer mismatch: "
            f"{checkpoint.get('trainer_kind')!r} != {type(trainer).__name__!r}."
        )

    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        raise RuntimeError("Schema-v3 checkpoint is missing RNG state.")
    cuda_rng_state = rng_state.get("torch_cuda")
    if cuda_rng_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state, but CUDA is unavailable; "
                "exact continuation is impossible."
            )
        if len(cuda_rng_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device topology differs from the saved checkpoint; exact "
                "continuation is impossible."
            )

    checkpoint_provenance = checkpoint.get("dataset_provenance")
    if expected_dataset_provenance is None:
        raise RuntimeError(
            "Schema-v3 exact resume requires current dataset provenance."
        )
    if not isinstance(checkpoint_provenance, dict):
        raise RuntimeError(
            "Checkpoint has no valid dataset provenance; refusing to mix restored "
            "replay with an unverified dataset."
        )
    try:
        assert_matching_dataset_provenance(
            checkpoint_provenance,
            expected_dataset_provenance,
        )
    except DatasetProvenanceError as exc:
        raise RuntimeError(str(exc)) from exc

    saved_rl_config = checkpoint.get("rl_config")
    current_rl_config = _config_dict(trainer.rl_cfg)
    if saved_rl_config != current_rl_config:
        raise RuntimeError(
            "Checkpoint RL configuration mismatch; exact continuation is impossible."
        )
    current_model_config = _config_dict(model.config)
    if checkpoint.get("model_config") != current_model_config:
        raise RuntimeError(
            "Checkpoint model configuration mismatch; exact continuation is impossible."
        )
    progress = checkpoint.get("progress")
    trainer_state = checkpoint.get("trainer_state")
    required_top_level = {
        "model_state_dict",
        "policy_model_old_state_dict",
        "policy_model_candidate_state_dict",
        "target_model_state_dict",
        "value_optimizer_state_dict",
        "policy_optimizer_state_dict",
        "replay_transitions",
        "replay_capacity",
    }
    missing = sorted(required_top_level - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"Schema-v3 checkpoint is missing required fields: {missing}."
        )
    if not isinstance(progress, dict) or not isinstance(trainer_state, dict):
        raise RuntimeError("Schema-v3 checkpoint is missing progress or trainer state.")
    if "environment_state" not in trainer_state or "collection_state" not in trainer_state:
        raise RuntimeError(
            "Schema-v3 checkpoint is missing live environment or collector state."
        )
    if int(checkpoint["replay_capacity"]) != int(trainer.replay.storage.maxlen):
        raise RuntimeError(
            "Checkpoint replay capacity mismatch; exact continuation is impossible."
        )
    saved_train_steps = int(trainer_state.get("train_step_count", -1))
    saved_env_steps = int(trainer_state.get("env_step_count", -1))
    if saved_train_steps < 0 or saved_env_steps < 0:
        raise RuntimeError("Checkpoint contains negative progress counters.")
    if saved_train_steps != int(progress.get("optimizer_updates", -1)):
        raise RuntimeError("Checkpoint optimizer-update counters disagree.")
    if saved_env_steps != int(progress.get("env_steps", -1)):
        raise RuntimeError("Checkpoint environment-step counters disagree.")
    replay_transitions = checkpoint["replay_transitions"]
    if len(replay_transitions) != int(checkpoint.get("replay_buffer_size", -1)):
        raise RuntimeError("Checkpoint replay size does not match its transition list.")
    saved_puzzle_optimizer = "puzzle_emb_optimizer_state_dict" in checkpoint
    if saved_puzzle_optimizer != (puzzle_emb_optimizer is not None):
        raise RuntimeError(
            "Checkpoint puzzle-embedding optimizer presence does not match the run."
        )
    optional_state_pairs = (
        (
            "old_policy_distill_optimizer_state_dict",
            getattr(trainer, "old_policy_distill_opt", None) is not None,
        ),
        (
            "value_scheduler_state_dict",
            getattr(trainer, "value_scheduler", None) is not None,
        ),
        (
            "policy_scheduler_state_dict",
            getattr(trainer, "policy_scheduler", None) is not None,
        ),
    )
    for field, expected_present in optional_state_pairs:
        if (field in checkpoint) != expected_present:
            raise RuntimeError(
                f"Checkpoint field {field!r} does not match the active trainer."
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    trainer.policy_model_old.load_state_dict(checkpoint["policy_model_old_state_dict"])
    trainer.policy_model_candidate.load_state_dict(
        checkpoint["policy_model_candidate_state_dict"]
    )
    trainer.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    trainer.value_opt.load_state_dict(checkpoint["value_optimizer_state_dict"])
    trainer.policy_opt.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    if (
        "old_policy_distill_optimizer_state_dict" in checkpoint
        and getattr(trainer, "old_policy_distill_opt", None) is not None
    ):
        trainer.old_policy_distill_opt.load_state_dict(
            checkpoint["old_policy_distill_optimizer_state_dict"]
        )
    if (
        "value_scheduler_state_dict" in checkpoint
        and getattr(trainer, "value_scheduler", None) is not None
    ):
        trainer.value_scheduler.load_state_dict(checkpoint["value_scheduler_state_dict"])
    if (
        "policy_scheduler_state_dict" in checkpoint
        and getattr(trainer, "policy_scheduler", None) is not None
    ):
        trainer.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
    
    if puzzle_emb_optimizer is not None:
        puzzle_emb_optimizer.load_state_dict(checkpoint["puzzle_emb_optimizer_state_dict"])

    trainer._next_episode_id = int(trainer_state["next_episode_id"])
    trainer._train_step_count = saved_train_steps
    trainer._env_step_count = saved_env_steps
    trainer._kl_coef = float(trainer_state.get("kl_coef", 1.0))
    if trainer_state.get("term_stats") is not None:
        trainer.term_stats = dict(trainer_state["term_stats"])
    trainer._debug_episode_lengths = list(
        trainer_state.get("debug_episode_lengths", [])
    )
    trainer._debug_episode_returns = list(
        trainer_state.get("debug_episode_returns", [])
    )
    trainer._debug_stop_probs = list(trainer_state.get("debug_stop_probs", []))
    trainer._debug_score_changes = list(
        trainer_state.get("debug_score_changes", [])
    )
    trainer._drift_values = list(trainer_state.get("drift_values", []))
    trainer._plan_changes = list(trainer_state.get("plan_changes", []))
    trainer._value_of_memory = list(trainer_state.get("value_of_memory", []))

    trainer.replay.clear()
    for transition in replay_transitions:
        trainer.replay.add(transition)
    trainer.env.load_checkpoint_state(trainer_state["environment_state"])
    trainer.load_collection_checkpoint_state(trainer_state["collection_state"])

    # Restore generators last so model/optimizer/replay reconstruction cannot
    # perturb the next random draw relative to an uninterrupted run.
    _restore_rng_state(rng_state)

    start_step = trainer._train_step_count
    print(
        f"[Checkpoint] Resumed at env_step={trainer._env_step_count}, "
        f"optimizer_update={start_step}"
    )

    return start_step


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveReasoningModel with plan-space RL.")
    parser.add_argument("--dataset-paths", nargs="+", default=None, help="Optional list of supervised dataset directories.")
    parser.add_argument("--train-split", default="train", help="Dataset split used for training.")
    parser.add_argument("--eval-split", default="test", help="Disjoint dataset split used for evaluation.")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Legacy resume remains fail-closed; "
            "use --load-checkpoint for an explicit weights-only warm start."
        ),
    )
    parser.add_argument(
        "--eval-pool-size",
        type=int,
        default=None,
        help="Number of fixed evaluation instances to materialize (default: eval episode count).",
    )
    parser.add_argument("--train-steps", type=int, default=200, help="Number of outer RL steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for TD updates.")
    parser.add_argument("--rollouts-per-step", type=int, default=1, help="Episodes collected before each optimization step.")
    parser.add_argument("--max-edits", type=int, default=8, help="Maximum edits per episode.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in train steps.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval in train steps.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of episodes per evaluation call.")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1729,
        help="Private deterministic seed for evaluation (default: 1729).",
    )
    parser.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bar (disabled by default).")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument("--debug-checks", action="store_true", help="Enable additional debug assertions/prints.")
    # Baseline algorithm selection
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["ppo", "a2c", "dqn", "ddqn", None],
        help="Use baseline algorithm instead of UPI-TRM. Options: ppo, a2c, dqn, ddqn. Default: None (use UPI-TRM).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="trm",
        choices=["trm", "norec-mlp", "norec-transformer"],
        help="Model backbone. Options: trm (default), norec-mlp, norec-transformer. "
             "norec-* use NoRecursionEncoder (simpler baselines without latent recursion).",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help="Optional path to YAML config overriding RLConfig defaults. "
             "Can be specified multiple times to layer configs (later files override earlier).",
    )
    # WandB arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="UPI-TRM-RL",
        help="WandB project name (default: UPI-TRM-RL).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not specified).",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run WandB in offline mode (no cloud sync).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging (disabled by default).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging (deprecated, WandB is now disabled by default).",
    )
    # Checkpoint arguments
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint to load (e.g., from supervised pretraining).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to RL checkpoint to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: checkpoints/<run_name>).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (0 to disable).",
    )
    parser.add_argument(
        "--env-step-budget",
        type=int,
        default=None,
        help="Optional exact environment-step budget; when set, training stops at this env-step count.",
    )
    parser.add_argument(
        "--log-env-interval",
        type=int,
        default=None,
        help="Optional logging interval in environment steps when --env-step-budget is active.",
    )
    parser.add_argument(
        "--eval-env-interval",
        type=int,
        default=None,
        help="Optional evaluation interval in environment steps when --env-step-budget is active.",
    )
    parser.add_argument(
        "--save-env-interval",
        type=int,
        default=None,
        help="Optional checkpoint interval in environment steps when --env-step-budget is active.",
    )
    # Model architecture arguments (to match pretrained model)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden dimension of TRM (default: 64, pretrained models often use 128).",
    )
    parser.add_argument(
        "--h-cycles",
        type=int,
        default=2,
        help="Number of H (outer) cycles in TRM.",
    )
    parser.add_argument(
        "--l-cycles",
        type=int,
        default=2,
        help="Number of L (inner) cycles in TRM.",
    )
    parser.add_argument(
        "--l-layers",
        type=int,
        default=1,
        help="Number of transformer layers in L-level.",
    )
    # Puzzle embedding arguments (NEW - closes the gap with pretrain.py)
    parser.add_argument(
        "--puzzle-emb-ndim",
        type=int,
        default=0,
        help="Dimension of per-puzzle learnable embeddings (0 to disable, typically hidden_size).",
    )
    parser.add_argument(
        "--puzzle-emb-len",
        type=int,
        default=16,
        help="Length of puzzle embedding sequence (default: 16).",
    )
    parser.add_argument(
        "--puzzle-emb-lr",
        type=float,
        default=1e-2,
        help="Learning rate for puzzle embeddings (uses SignSGD, default: 1e-2).",
    )
    parser.add_argument(
        "--puzzle-emb-weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for puzzle embeddings (default: 0.1).",
    )
    # Imitation learning pre-training arguments
    parser.add_argument(
        "--imitation-pretrain",
        action="store_true",
        help="Pre-train policy with imitation learning from oracle before RL.",
    )
    parser.add_argument(
        "--imitation-epochs",
        type=int,
        default=50,
        help="Number of imitation learning epochs (default: 50).",
    )
    return parser.parse_args()


def _write_progress(step_iter: Any, message: str) -> None:
    if step_iter is not None and hasattr(step_iter, "write"):
        step_iter.write(message)
    else:
        print(message)


def _step_prefix(progress_step: int, outer_step: int) -> str:
    prefix = f"[step {progress_step:05d}]"
    if progress_step != outer_step:
        prefix += f" [update {outer_step:05d}]"
    return prefix


def _log_training_metrics(
    *,
    progress_step: int,
    outer_step: int,
    metrics: Dict[str, float],
    selected_baseline: Optional[str],
    trainer: Any,
    step_iter: Any,
    use_wandb: bool,
) -> None:
    loss_value = metrics.get("loss_value", metrics.get("loss_q", 0.0))
    loss_policy = metrics.get("loss_policy", 0.0)
    prefix = _step_prefix(progress_step, outer_step)

    if selected_baseline in ("dqn", "ddqn"):
        epsilon = metrics.get("epsilon", trainer._get_epsilon() if hasattr(trainer, "_get_epsilon") else 0)
        mean_q = metrics.get("mean_q", 0.0)
        msg = (
            f"{prefix} q_loss={loss_value:.6f} "
            f"mean_q={mean_q:.3f} epsilon={epsilon:.3f}"
        )
    else:
        msg = (
            f"{prefix} value_loss={loss_value:.6f} "
            f"policy_loss={loss_policy:.6f}"
        )
    _write_progress(step_iter, msg)

    if "target_mean" in metrics:
        clip_str = f" clip={metrics['value_clip']:.1f}" if "value_clip" in metrics else ""
        target_msg = (
            f"{prefix} VALUE_DEBUG: "
            f"target(mean={metrics['target_mean']:.2f} std={metrics['target_std']:.2f} "
            f"min={metrics['target_min']:.2f} max={metrics['target_max']:.2f}) "
            f"V(s)(mean={metrics['value_mean']:.2f} std={metrics['value_std']:.2f}) "
            f"reward(mean={metrics['reward_mean']:.3f} std={metrics['reward_std']:.3f}){clip_str}"
        )
        _write_progress(step_iter, target_msg)

    if use_wandb:
        wandb_metrics = {
            "train/loss_value": loss_value,
            "train/loss_policy": loss_policy,
            "train/term_stop": metrics.get("term_stop", 0),
            "train/term_solved": metrics.get("term_solved", 0),
            "train/term_budget": metrics.get("term_budget", 0),
            "train/env_steps_total": metrics.get("env_steps_total", 0),
            "train/env_steps_collected": metrics.get("env_steps_collected", 0),
        }
        if selected_baseline in ("dqn", "ddqn"):
            wandb_metrics["train/mean_q"] = metrics.get("mean_q", 0.0)
            wandb_metrics["train/epsilon"] = metrics.get("epsilon", 0.0)
            wandb_metrics["train/buffer_size"] = metrics.get("buffer_size", 0)
        if "value_lr" in metrics:
            wandb_metrics["train/lr_value"] = metrics["value_lr"]
        if "policy_lr" in metrics:
            wandb_metrics["train/lr_policy"] = metrics["policy_lr"]
        if "policy_kl" in metrics:
            wandb_metrics["train/policy_kl"] = metrics["policy_kl"]
        if "kl_coef" in metrics:
            wandb_metrics["train/kl_coef"] = metrics["kl_coef"]
        for key in [
            "hat_Cz",
            "hat_Lz",
            "hat_Lv",
            "unrolling_term",
            "bellman_residual_mean",
            "bellman_residual_max",
            "drift_mean",
            "drift_max",
            "plan_change_mean",
        ]:
            if key in metrics:
                wandb_metrics[f"theory/{key}"] = metrics[key]
        for key in [
            "value_mean",
            "value_std",
            "adv_mean",
            "adv_std",
            "target_mean",
            "target_std",
            "target_min",
            "target_max",
            "reward_mean",
            "reward_std",
        ]:
            if key in metrics:
                wandb_metrics[f"debug/{key}"] = metrics[key]
        wandb.log(wandb_metrics, step=progress_step)


def _run_eval_and_log(
    *,
    progress_step: int,
    outer_step: int,
    trainer: Any,
    env_cfg: Any,
    dataset: Any,
    checker_fn: Any,
    step_iter: Any,
    use_wandb: bool,
) -> None:
    eval_metrics = None
    try:
        eval_metrics = trainer.evaluate_policy_metrics(
            env_cfg=env_cfg,
            dataset=dataset,
            checker=checker_fn,
        )
    except AttributeError:
        pass
    except Exception as e:
        print(f"[WARN] evaluate_policy_metrics error: {type(e).__name__}: {e}")

    prefix = _step_prefix(progress_step, outer_step)
    if eval_metrics is not None:
        eval_policy_mode = eval_metrics.get("eval_policy_mode", "unknown")
        eval_seed = eval_metrics.get("eval_seed")
        solved_count = eval_metrics.get("solved_count", 0)
        total_episodes = eval_metrics.get("total_episodes", 0)
        score_min = eval_metrics.get("score_min", 0.0)
        score_max = eval_metrics.get("score_max", 0.0)
        max_possible = eval_metrics.get("max_possible_score")
        initial_mean = eval_metrics.get("initial_score_mean", 0.0)
        mean_return = eval_metrics.get("mean_return")
        invalid_action_rate = eval_metrics.get("invalid_action_rate")
        filled_mean = eval_metrics.get("final_filled_mean")
        violations_mean = eval_metrics.get("final_violations_mean")
        zero_cand_mean = eval_metrics.get("final_zero_cand_mean")

        eval_msg = (
            f"{prefix} "
            f"eval_success_rate={eval_metrics['success_rate']:.3f} "
            f"eval_mean_score={eval_metrics['mean_score']:.3f} "
            f"eval_policy_mode={eval_policy_mode} "
            f"{f'eval_seed={eval_seed} ' if eval_seed is not None else ''}"
            f"[solved={solved_count}/{total_episodes}, "
            f"score_range={score_min:.2f}-{score_max:.2f}"
            f"{f'/{max_possible:.1f}' if max_possible else ''}, "
            f"initial={initial_mean:.2f}]"
        )
        if mean_return is not None:
            eval_msg += f" eval_mean_return={mean_return:.3f}"
        if invalid_action_rate is not None:
            eval_msg += f" eval_invalid_action_rate={invalid_action_rate:.3f}"
        _write_progress(step_iter, eval_msg)

        if filled_mean is not None:
            progress_msg = (
                f"{prefix} PROGRESS: "
                f"filled={filled_mean:.2f} "
                f"violations={violations_mean:.2f} "
                f"zero_cand={zero_cand_mean:.2f}"
            )
            _write_progress(step_iter, progress_msg)

        if use_wandb:
            wandb_eval = {
                "eval/success_rate": eval_metrics["success_rate"],
                "eval/mean_score": eval_metrics["mean_score"],
                "eval/solved_count": solved_count,
                "eval/score_min": score_min,
                "eval/score_max": score_max,
                "eval/initial_score_mean": initial_mean,
            }
            if eval_seed is not None:
                wandb_eval["eval/seed"] = eval_seed
            if max_possible is not None:
                wandb_eval["eval/max_possible_score"] = max_possible
            if filled_mean is not None:
                wandb_eval["eval/filled_mean"] = filled_mean
                wandb_eval["eval/violations_mean"] = violations_mean
                wandb_eval["eval/zero_cand_mean"] = zero_cand_mean
            wandb.log(wandb_eval, step=progress_step)
    else:
        print(f"{prefix} (eval not available for baseline trainer)")

    debug_stats = None
    if hasattr(trainer, "get_debug_stats"):
        debug_stats = trainer.get_debug_stats()
    if debug_stats:
        debug_msg = (
            f"{prefix} DEBUG: "
            f"ep_len={debug_stats.get('avg_episode_length', 0):.1f} "
            f"ep_ret={debug_stats.get('avg_episode_return', 0):.3f} "
            f"stop_prob={debug_stats.get('avg_stop_prob', 0):.3f} "
            f"score_chg={debug_stats.get('avg_score_change', 0):.4f}"
        )
        _write_progress(step_iter, debug_msg)
        if use_wandb:
            wandb_debug = {}
            for key, value in debug_stats.items():
                wandb_debug[f"debug/{key}"] = value
            wandb.log(wandb_debug, step=progress_step)

    if hasattr(trainer, "clear_debug_stats"):
        trainer.clear_debug_stats()


def main():
    args = parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)

    rl_cfg = RLConfig(
        batch_size=args.batch_size,
        num_train_steps=args.train_steps,
        rollout_episodes_per_step=args.rollouts_per_step,
        max_edits=args.max_edits,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_num_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        use_tqdm=args.tqdm,
        debug_checks=args.debug_checks,
    )

    if args.config is not None:
        import yaml

        # The override config is expected to be a flat dict with keys matching RLConfig
        # fields, e.g. {"gamma": 0.95, "K": 3}. Nested structures (e.g. trainer: rl: ...)
        # are not currently supported.
        #
        # Multiple --config args are layered in order: later files override earlier.
        # Example: --config base.yaml --config override.yaml
        #   -> base.yaml settings are loaded first, then override.yaml on top
        base_dict = rl_cfg.model_dump() if hasattr(rl_cfg, "model_dump") else rl_cfg.dict()
        for config_path in args.config:
            with open(config_path, "r") as f:
                override = yaml.safe_load(f) or {}
            base_dict = {**base_dict, **override}
        rl_cfg = RLConfig(**base_dict)
    
    # === CLI overrides take priority over YAML ===
    # This ensures --train-steps 100 overrides num_train_steps from YAML
    if args.train_steps != 200:  # 200 is the argparse default
        rl_cfg = RLConfig(**{
            **(rl_cfg.model_dump() if hasattr(rl_cfg, "model_dump") else rl_cfg.dict()),
            "num_train_steps": args.train_steps,
        })
    if args.batch_size != 32:  # 32 is the argparse default
        rl_cfg = RLConfig(**{
            **(rl_cfg.model_dump() if hasattr(rl_cfg, "model_dump") else rl_cfg.dict()),
            "batch_size": args.batch_size,
        })
    if args.max_edits != 8:  # 8 is the argparse default
        rl_cfg = RLConfig(**{
            **(rl_cfg.model_dump() if hasattr(rl_cfg, "model_dump") else rl_cfg.dict()),
            "max_edits": args.max_edits,
        })
    
    # === YAML-based baseline selection (uses helper - SINGLE SOURCE OF TRUTH) ===
    # This calls select_baseline_from_configs() which is the canonical implementation.
    # Tests also call this function to ensure consistency with main().
    baseline_selection = select_baseline_from_configs(args.baseline, args.config)
    selected_baseline = baseline_selection.selected_baseline
    yaml_algorithm = baseline_selection.yaml_algorithm
    get_yaml_key = baseline_selection.get_yaml_key  # Function to retrieve baseline-specific config values
    
    if selected_baseline is not None and args.baseline is None:
        # Baseline was auto-selected from YAML
        print(f"[INFO] Baseline algorithm '{selected_baseline}' auto-selected from YAML config")

    if args.dataset_paths and args.train_split == args.eval_split:
        raise RuntimeError(
            "Training and evaluation must use different dataset splits; got "
            f"{args.train_split!r} for both."
        )

    dataset, seq_len, vocab_size, num_identifiers = build_dataset_from_paths(
        dataset_paths=args.dataset_paths,
        pool_size=max(rl_cfg.batch_size, 8),
        split=args.train_split,
    )
    train_identifier_count = num_identifiers
    if args.dataset_paths:
        eval_pool_size = args.eval_pool_size or rl_cfg.eval_num_episodes
        eval_dataset, eval_seq_len, eval_vocab_size, eval_num_identifiers = build_dataset_from_paths(
            dataset_paths=args.dataset_paths,
            pool_size=max(eval_pool_size, rl_cfg.eval_num_episodes),
            split=args.eval_split,
        )
        if (eval_seq_len, eval_vocab_size) != (seq_len, vocab_size):
            raise RuntimeError(
                "Training and evaluation splits have incompatible shapes: "
                f"train={(seq_len, vocab_size)}, eval={(eval_seq_len, eval_vocab_size)}."
            )
        if len(eval_dataset) < rl_cfg.eval_num_episodes:
            raise RuntimeError(
                "Evaluation split is smaller than eval_num_episodes; refusing to "
                f"repeat instances ({len(eval_dataset)} < {rl_cfg.eval_num_episodes})."
            )
        offset_puzzle_identifiers(eval_dataset, train_identifier_count)
        eval_num_identifiers = train_identifier_count + eval_num_identifiers
        overlap = set(dataset_input_sha256s(dataset)).intersection(
            dataset_input_sha256s(eval_dataset)
        )
        if overlap:
            raise RuntimeError(
                "Training and evaluation pools overlap; refusing an in-sample "
                f"evaluation ({len(overlap)} duplicate records)."
            )
        num_identifiers = eval_num_identifiers
    else:
        eval_dataset = dataset

    # === DATASET PROVENANCE LOGGING (for audit/reproducibility) ===
    # These lines are grep-friendly for verifying which dataset was used
    dataset_name = "dummy"
    if args.dataset_paths:
        dataset_name = os.path.basename(args.dataset_paths[0])
        print(f"[DATASET] dataset_paths={args.dataset_paths}")
        print(f"[DATASET] resolved_dataset_name={dataset_name}")
        train_pool_hash = dataset_pool_sha256(dataset, len(dataset))
        eval_pool_hash = dataset_pool_sha256(
            eval_dataset, rl_cfg.eval_num_episodes
        )
        print(
            f"[DATASET] train_split={args.train_split} train_samples={len(dataset)} "
            f"train_pool_sha256={train_pool_hash}"
        )
        print(
            f"[DATASET] eval_split={args.eval_split} "
            f"eval_samples={rl_cfg.eval_num_episodes} "
            f"eval_pool_sha256={eval_pool_hash} "
            f"eval_puzzle_id_offset={train_identifier_count}"
        )
        source_build_metadata = dataset_source_build_metadata(args.dataset_paths)
        if len(source_build_metadata) == 1:
            provenance_builder_name = source_build_metadata[0]["builder_name"]
            provenance_builder_version = source_build_metadata[0][
                "builder_version"
            ]
            provenance_generation_seed = source_build_metadata[0][
                "generation_seed"
            ]
        else:
            provenance_builder_name = "composite_materialized_dataset"
            provenance_builder_version = 1
            provenance_generation_seed = None
        provenance_train_split = args.train_split
        provenance_eval_split = args.eval_split
        provenance_eval_count = rl_cfg.eval_num_episodes
        provenance_metadata = {
            "train_pool_sha256": train_pool_hash,
            "eval_pool_sha256": eval_pool_hash,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "num_identifiers": num_identifiers,
            "eval_puzzle_id_offset": train_identifier_count,
            "dataset_source_names": [
                os.path.basename(path.rstrip(os.sep)) for path in args.dataset_paths
            ],
            "source_build_metadata": source_build_metadata,
            "materialization_seed": 0,
        }
    else:
        print(f"[DATASET] dataset_paths=None (using dummy dataset)")
        print(f"[DATASET] resolved_dataset_name=dummy")
        print(f"[DATASET] num_samples={len(dataset)}")
        provenance_builder_name = "rl.training_setup.DummyPuzzleDataset"
        provenance_builder_version = 1
        provenance_generation_seed = args.seed
        provenance_train_split = "dummy"
        provenance_eval_split = "dummy"
        provenance_eval_count = len(eval_dataset)
        provenance_metadata = {
            "train_pool_sha256": dataset_pool_sha256(dataset, len(dataset)),
            "eval_pool_sha256": dataset_pool_sha256(
                eval_dataset, len(eval_dataset)
            ),
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "num_identifiers": num_identifiers,
            "eval_puzzle_id_offset": 0,
            "dataset_source_names": [],
        }

    checker_resolution = resolve_checker_from_dataset(rl_cfg=rl_cfg, dataset=dataset, seq_len=seq_len)
    checker_fn = checker_resolution.checker_fn
    checker_kind = checker_resolution.checker_kind
    for line in checker_resolution.info_lines:
        print(line)
    is_sudoku_checker = checker_kind in {"solution", "constraint", "progress", "feasibility"}

    env_cfg = PlanEditEnvConfig(
        max_edits=rl_cfg.max_edits,
        gamma=rl_cfg.gamma,
        reward_shaping=rl_cfg.reward_shaping,
        vocab_size=vocab_size,
        solved_threshold=rl_cfg.solved_threshold if is_sudoku_checker else None,
        task_type=getattr(rl_cfg, "task_name", "sudoku"),
        # STOP action behavior from RLConfig
        stop_action_mode=getattr(rl_cfg, "stop_action_mode", "noop"),
        stop_action_penalty=getattr(rl_cfg, "stop_action_penalty", -0.1),
        # Terminal rewards (Paper Remark 2.6: rush-to-fail mitigation)
        fail_terminal_reward=getattr(rl_cfg, "fail_terminal_reward", 0.0),
        solve_terminal_reward=getattr(rl_cfg, "solve_terminal_reward", 0.0),
        disable_constraint_masking=getattr(rl_cfg, "disable_constraint_masking", False),
    )
    
    # Optionally use task-specific configuration
    task_config = None
    try:
        from rl.task_config import get_task_config
        task_name = getattr(rl_cfg, "task_name", "sudoku")
        if is_sudoku_checker:
            task_config = get_task_config(
                "sudoku",
                disable_constraint_masking=getattr(rl_cfg, "disable_constraint_masking", False),
            )
        elif checker_kind == "dummy":
            task_config = get_task_config("dummy")
    except ImportError:
        pass  # task_config module not available
    
    env = PlanEditEnv(dataset=dataset, checker=checker_fn, config=env_cfg, task_config=task_config)
    
    env_step_budget = args.env_step_budget
    if env_step_budget is not None:
        print(
            f"[INFO] Training to env-step budget {env_step_budget} "
            f"(gamma={rl_cfg.gamma:.3f}, K={rl_cfg.K}, inner_unroll_n={rl_cfg.inner_unroll_n})"
        )
    else:
        print(
            f"[INFO] Training for {rl_cfg.num_train_steps} steps "
            f"(gamma={rl_cfg.gamma:.3f}, K={rl_cfg.K}, inner_unroll_n={rl_cfg.inner_unroll_n})"
        )
    print(f"[INFO] Terminal rewards: solve={env_cfg.solve_terminal_reward:.3f}, "
          f"fail={env_cfg.fail_terminal_reward:.3f}")

    num_edit_actions = seq_len * vocab_size
    rl_num_actions = num_edit_actions + 1  # STOP action appended at the end
    env.set_stop_action_id(stop_id=rl_num_actions - 1)

    dataset_provenance = build_dataset_provenance(
        builder_name=provenance_builder_name,
        builder_version=provenance_builder_version,
        generation_seed=provenance_generation_seed,
        train_record_sha256s=dataset_sample_sha256s(dataset),
        eval_record_sha256s=dataset_sample_sha256s(
            eval_dataset,
            count=provenance_eval_count,
        ),
        train_split=provenance_train_split,
        eval_split=provenance_eval_split,
        environment_config=dict(vars(env_cfg)),
        action_mask_config={
            "task_config_class": (
                type(task_config).__name__ if task_config is not None else None
            ),
            "task_config_name": (
                getattr(task_config, "name", None) if task_config is not None else None
            ),
            "disable_constraint_masking": env_cfg.disable_constraint_masking,
            "stop_action_mode": env._stop_mode,
            "stop_action_id": env.stop_action_id,
            "enable_undo": env._enable_undo,
            "undo_action_id": env.undo_action_id,
            "vocab_size": env.vocab_size,
            "num_actions": (
                env.undo_action_id + 1
                if env.undo_action_id is not None
                else rl_num_actions
            ),
            "masked_token_ids": [0, 1],
        },
        metadata=provenance_metadata,
    )

    # === Model Configuration ===
    # Use CLI args for architecture (allows matching pretrained model)
    hidden_size = args.hidden_size
    puzzle_emb_ndim = args.puzzle_emb_ndim
    puzzle_emb_len = args.puzzle_emb_len if puzzle_emb_ndim > 0 else 0
    
    trm_cfg_dict = dict(
        batch_size=rl_cfg.batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=puzzle_emb_ndim,  # NEW: Per-puzzle learnable embeddings
        puzzle_emb_len=puzzle_emb_len,     # NEW: Embedding sequence length
        num_puzzle_identifiers=max(num_identifiers, rl_cfg.batch_size),
        vocab_size=vocab_size,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        H_layers=0,
        L_layers=args.l_layers,
        hidden_size=hidden_size,
        expansion=2.0,
        num_heads=max(4, hidden_size // 16),  # Scale heads with hidden size
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=rl_cfg.enable_contraction,
        rl_target_Lz=rl_cfg.target_Lz,
        rl_target_Lv=rl_cfg.target_Lv,
        rl_disable_value_head_norm=getattr(rl_cfg, "disable_value_head_norm", False),
        rl_enable_policy_head=True,
        rl_num_actions=rl_num_actions,
        # Forward-invariant projection (Assumption 4.1)
        rl_latent_ball_radius=getattr(rl_cfg, "latent_ball_radius", 0.0),
    )

    # === Model Selection ===
    # Support for different backbones: TRM (default) or NoRecursionEncoder (simpler baselines)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == "trm":
        # Standard TRM backbone
        model = TinyRecursiveReasoningModel_ACTV1(trm_cfg_dict)
        print(f"[INFO] Using TRM backbone (hidden={hidden_size}, H={args.h_cycles}, L={args.l_cycles})")
    elif args.backbone in ("norec-mlp", "norec-transformer"):
        # NoRecursionEncoder baseline (no latent recursion)
        encoder_type = "mlp" if args.backbone == "norec-mlp" else "transformer"
        norec_cfg = NoRecEncoderConfig(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dim=hidden_size,
            num_layers=2,
            encoder_type=encoder_type,
            rl_num_actions=rl_num_actions,
        )
        model = NoRecursionEncoder(norec_cfg)
        print(f"[INFO] Using NoRecursionEncoder backbone (type={encoder_type}, hidden={hidden_size})")
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # === Load pretrained checkpoint if provided ===
    if args.load_checkpoint is not None:
        load_checkpoint(model, args.load_checkpoint, device=str(device), strict=False)

    # Debug: verify policy head initialization
    if hasattr(model, 'edit_policy') and model.edit_policy is not None:
        stop_bias = model.edit_policy.mlp[-1].bias[-1].item()
        print(f"[DEBUG] Policy head STOP bias: {stop_bias:.2f} (should be 0.0 for uniform init)")

    # Log puzzle embedding status (only for TRM backbone)
    if args.backbone == "trm":
        if puzzle_emb_ndim > 0:
            print(f"[INFO] Puzzle embeddings ENABLED: dim={puzzle_emb_ndim}, len={puzzle_emb_len}")
            print(f"[INFO] Puzzle embedding LR: {args.puzzle_emb_lr}, weight_decay: {args.puzzle_emb_weight_decay}")
        else:
            print("[INFO] Puzzle embeddings DISABLED (set --puzzle-emb-ndim > 0 to enable)")

    # === Trainer Selection (uses build_trainer - SINGLE SOURCE OF TRUTH) ===
    # This calls build_trainer() which is the canonical implementation.
    # Tests also call this function to verify trainer instantiation.
    trainer = build_trainer(
        model=model,
        env=env,
        rl_cfg=rl_cfg,
        device=device,
        baseline_selection=baseline_selection,
        cli_baseline=args.baseline,
        verbose=True,
    )
    
    # === Setup puzzle embedding optimizer (separate from main optimizer) ===
    puzzle_emb_optimizer = None
    puzzle_emb_optimizer_managed_by_trainer = False
    if puzzle_emb_ndim > 0 and hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb"):
        # Use SignSGD for sparse puzzle embeddings (same as pretrain.py)
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            model.inner.puzzle_emb.buffers(),
            lr=args.puzzle_emb_lr,
            weight_decay=args.puzzle_emb_weight_decay,
            world_size=1,  # Single GPU for now
        )
        print(f"[INFO] Puzzle embedding optimizer: SignSGD (lr={args.puzzle_emb_lr})")
        if hasattr(trainer, "set_puzzle_embedding_optimizer"):
            trainer.set_puzzle_embedding_optimizer(puzzle_emb_optimizer)
            puzzle_emb_optimizer_managed_by_trainer = True
    
    # === Resume from RL checkpoint if provided ===
    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = resume_from_checkpoint(
            args.resume_checkpoint,
            model,
            trainer,
            str(device),
            puzzle_emb_optimizer,
            expected_dataset_provenance=dataset_provenance,
            allow_legacy_warm_start=args.allow_legacy_resume,
        )
    
    # === Setup checkpoint directory ===
    # Note: dataset_name is already set in the provenance logging section above
    checkpoint_dir = args.checkpoint_dir
    checkpointing_enabled = args.save_interval > 0 or (
        env_step_budget is not None
        and args.save_env_interval is not None
        and args.save_env_interval > 0
    )
    if checkpoint_dir is None and checkpointing_enabled:
        checkpoint_dir = os.path.join("checkpoints", f"rl_{dataset_name}_seed{args.seed or 0}")
        print(f"[INFO] Checkpoint directory: {checkpoint_dir}")

    # === Set checker function for exact baseline computation (Theorem 5.9) ===
    # Only applies to UPI-TRM trainer
    if hasattr(trainer, 'set_checker_fn'):
        # When exact_baseline_summation=True, the trainer uses this to compute
        # E_{a ~ π}[Q̂(s,a)] via exact summation over all discrete actions,
        # enabling the O(α·ε_A) bound instead of naive O(ε_A).
        trainer.set_checker_fn(checker_fn)

    # === Validate theory alignment and show warnings (UPI-TRM only) ===
    if selected_baseline is None:
        print("\n" + "="*60)
        print("UPI-TRM THEORY ALIGNMENT CHECK")
        print("="*60)
        validation = rl_cfg.validate_theory_alignment(warn=False)  # Get results without duplicate warnings

        if validation["theory_aligned"]:
            print("✓ Configuration aligns with theoretical guarantees")
        else:
            print("⚠ Configuration has theory gaps:")
            for issue in validation["issues"]:
                print(f"  • {issue}")

        print(f"\nTheory status:")
        print(f"  Forward-invariant projection: {'✓' if validation['forward_invariant'] else '✗'}")
        print(f"  Clamping intervention enabled: {'✓' if rl_cfg.enable_contraction else '✗'}")
        print("  Global contraction certified: ✗ (local proxies are not a certificate)")
        print(f"  Exact baseline (Thm 5.9): {'✓' if validation['exact_baseline'] else '✗'}")
        print(f"  Distillation (not in theory): {'✗ ENABLED' if validation['distillation_used'] else '✓ disabled'}")
        print(f"\nIs theory-exact: {rl_cfg.is_theory_exact()}")
        print("="*60 + "\n")

        print("UPI-TRM theoretical dials:")
        print(f"  L_z target (rl_cfg.target_Lz): {rl_cfg.target_Lz}")
        print(f"  L_V target (rl_cfg.target_Lv): {rl_cfg.target_Lv}")
        print(f"  Inner unroll n (rl_cfg.inner_unroll_n): {rl_cfg.inner_unroll_n}")
        print(f"  K-step horizon K (rl_cfg.K): {rl_cfg.K}")
        print(f"  Mixture alpha (rl_cfg.mixture_alpha): {rl_cfg.mixture_alpha}")
        print(f"  Latent ball radius (rl_cfg.latent_ball_radius): {rl_cfg.latent_ball_radius}")
        print(f"  Episodic latent: {rl_cfg.episodic_latent}")
        print(f"  Exact K-step targets: {rl_cfg.exact_k_step_targets}")
        print(f"  Exact baseline summation: {rl_cfg.exact_baseline_summation}")
        print()
    else:
        # Baseline algorithm info
        print(f"\n[INFO] Using {selected_baseline.upper()} baseline algorithm")
        print(f"[INFO] Backbone: {args.backbone}")
        print(f"[INFO] This is a comparison baseline (no theory-exact features)")
        print()

    # Log additional configuration settings
    stop_mode = getattr(rl_cfg, "stop_action_mode", "noop")
    print(f"[INFO] STOP action mode: {stop_mode}")
    if selected_baseline is None:  # Only show theory-specific info for UPI-TRM
        if getattr(rl_cfg, "exact_baseline_summation", False):
            print("[INFO] Using EXACT baseline summation (Theorem 5.9 O(α·ε_A) bound)")
        if getattr(rl_cfg, "latent_ball_radius", 0.0) > 0:
            print(f"[INFO] Forward-invariant projection enabled (R={rl_cfg.latent_ball_radius})")
        if getattr(rl_cfg, "track_drift_metrics", False):
            print("[INFO] Drift tracking enabled (Lemma 4.4)")
        if getattr(rl_cfg, "compute_value_of_memory", False):
            print("[INFO] Value-of-memory computation enabled (Section 5.4)")

    # === Initialize WandB (disabled by default, enable with --wandb) ===
    use_wandb = WANDB_AVAILABLE and args.wandb and not args.no_wandb
    if use_wandb:
        # Set offline mode if requested
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        # Generate run name if not provided
        # Note: dataset_name is already set in the provenance logging section
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"{dataset_name}-K{rl_cfg.K}-seed{args.seed or 0}"
        
        # Prepare config dict for WandB
        wandb_config = rl_cfg.model_dump() if hasattr(rl_cfg, "model_dump") else rl_cfg.dict()
        wandb_config.update({
            "dataset_paths": args.dataset_paths,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "num_identifiers": num_identifiers,
            "num_actions": rl_num_actions,
            "device": str(device),
            "theory_exact": rl_cfg.is_theory_exact(),
            # Model architecture
            "hidden_size": hidden_size,
            "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles,
            "l_layers": args.l_layers,
            # Puzzle embeddings (NEW)
            "puzzle_emb_ndim": puzzle_emb_ndim,
            "puzzle_emb_len": puzzle_emb_len,
            "puzzle_emb_lr": args.puzzle_emb_lr,
            "puzzle_emb_weight_decay": args.puzzle_emb_weight_decay,
            # Checkpointing
            "load_checkpoint": args.load_checkpoint,
            "resume_checkpoint": args.resume_checkpoint,
        })
        
        # Initialize WandB
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            settings=wandb.Settings(_disable_stats=True),
        )
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_params": num_params}, step=0)
        print(f"[WandB] Initialized: project={args.wandb_project}, run={run_name}")
        print(f"[WandB] Model parameters: {num_params:,}")
    else:
        if args.wandb and not WANDB_AVAILABLE:
            print("[WandB] Requested but not available (install with: pip install wandb)")
        elif not args.wandb:
            print("[WandB] Disabled by default (enable with: --wandb)")

    # === Imitation learning pre-training (optional but recommended) ===
    if args.imitation_pretrain:
        if not hasattr(trainer, "imitation_pretrain"):
            raise ValueError(
                "--imitation-pretrain is only supported for the default UPI-TRM trainer, "
                "not for PPO/A2C/DQN baselines."
            )
        print("\n" + "="*60)
        print("IMITATION LEARNING PRE-TRAINING")
        print("="*60)
        print("Pre-training policy from oracle demonstrations...")
        print("This bootstraps RL by giving the policy a good starting point.")
        print()
        
        imitation_stats = trainer.imitation_pretrain(
            dataset=dataset,
            checker=checker_fn,
            num_epochs=args.imitation_epochs,
            batch_size=rl_cfg.batch_size,
            log_interval=10,
        )
        print()
        
        # Quick evaluation after imitation pre-training
        if args.eval_interval > 0:
            print("Evaluating policy after imitation pre-training...")
            eval_metrics = trainer.evaluate_policy_metrics(
                env_cfg=env_cfg,
                dataset=eval_dataset,
                checker=checker_fn,
            )
            post_imitation_msg = (
                f"[Post-imitation] eval_success_rate={eval_metrics['success_rate']:.3f} "
                f"eval_mean_score={eval_metrics['mean_score']:.3f}"
            )
            if "mean_return" in eval_metrics:
                post_imitation_msg += f" eval_mean_return={eval_metrics['mean_return']:.3f}"
            if "invalid_action_rate" in eval_metrics:
                post_imitation_msg += (
                    f" eval_invalid_action_rate={eval_metrics['invalid_action_rate']:.3f}"
                )
            print(post_imitation_msg)
            print()

    # === Training loop with puzzle embedding updates and checkpointing ===
    last_saved_progress_step: Optional[int] = None

    if env_step_budget is None:
        total_steps = rl_cfg.num_train_steps
        remaining_steps = total_steps - start_step

        if rl_cfg.use_tqdm and trange is not None:
            step_iter = trange(remaining_steps, desc="UPI-TRM RL training", initial=start_step, total=total_steps)
        else:
            step_iter = range(remaining_steps)

        for local_step in step_iter:
            step = start_step + local_step
            outer_step = step + 1
            metrics = trainer.train_step()

            if (
                puzzle_emb_optimizer is not None
                and not puzzle_emb_optimizer_managed_by_trainer
            ):
                puzzle_emb_optimizer.step()
                puzzle_emb_optimizer.zero_grad()

            if outer_step % rl_cfg.log_interval == 0:
                _log_training_metrics(
                    progress_step=outer_step,
                    outer_step=outer_step,
                    metrics=metrics,
                    selected_baseline=selected_baseline,
                    trainer=trainer,
                    step_iter=step_iter,
                    use_wandb=use_wandb,
                )

            if outer_step % rl_cfg.eval_interval == 0:
                _run_eval_and_log(
                    progress_step=outer_step,
                    outer_step=outer_step,
                    trainer=trainer,
                    env_cfg=env_cfg,
                    dataset=eval_dataset,
                    checker_fn=checker_fn,
                    step_iter=step_iter,
                    use_wandb=use_wandb,
                )

            if args.save_interval > 0 and checkpoint_dir is not None and outer_step % args.save_interval == 0:
                save_checkpoint(
                    model,
                    trainer,
                    outer_step,
                    checkpoint_dir,
                    puzzle_emb_optimizer,
                    rl_cfg,
                    dataset_provenance,
                )
                last_saved_progress_step = outer_step

        final_progress_step = total_steps
    else:
        if not isinstance(trainer, UPITrmTrainer):
            raise ValueError(
                "Exact --env-step-budget collection currently requires "
                "UPITrmTrainer; baseline trainers must not fall back to nominal "
                "outer-step accounting."
            )
        restored_env_steps = trainer.get_env_step_count()
        if env_step_budget < restored_env_steps:
            raise ValueError(
                "Requested env-step budget is behind the restored checkpoint "
                f"({env_step_budget} < {restored_env_steps})."
            )

        log_env_interval = args.log_env_interval
        eval_env_interval = args.eval_env_interval
        save_env_interval = args.save_env_interval

        def next_strict_multiple(interval: Optional[int]) -> Optional[int]:
            if interval is None or interval <= 0:
                return None
            return (restored_env_steps // interval + 1) * interval

        next_log_env = next_strict_multiple(log_env_interval)
        next_eval_env = next_strict_multiple(eval_env_interval)
        next_save_env = next_strict_multiple(save_env_interval)
        outer_step = int(getattr(trainer, "_train_step_count", start_step))
        step_iter = None

        while trainer.get_env_step_count() < env_step_budget:
            current_env_step = trainer.get_env_step_count()
            targets = [env_step_budget]
            for target in (next_log_env, next_eval_env, next_save_env):
                if target is not None and target > current_env_step:
                    targets.append(target)
            next_target = min(targets)
            collect_budget = next_target - current_env_step
            if collect_budget <= 0:
                raise RuntimeError("Exact-budget scheduler produced a nonpositive cap.")

            metrics = trainer.train_step(max_env_steps_to_collect=collect_budget)
            new_env_step = trainer.get_env_step_count()
            if new_env_step <= current_env_step:
                raise RuntimeError(
                    "Exact-budget trainer made no environment-step progress."
                )
            if new_env_step > next_target:
                raise RuntimeError(
                    "Exact-budget trainer exceeded its collection cap "
                    f"({new_env_step} > {next_target})."
                )
            outer_step = int(
                metrics.get("train_steps_total", trainer._train_step_count)
            )

            if (
                puzzle_emb_optimizer is not None
                and not puzzle_emb_optimizer_managed_by_trainer
                and metrics.get("optimization_performed", 0.0) > 0.0
            ):
                puzzle_emb_optimizer.step()
                puzzle_emb_optimizer.zero_grad()

            current_env_step = trainer.get_env_step_count()
            progress_step = current_env_step

            if next_log_env is not None and current_env_step >= next_log_env:
                _log_training_metrics(
                    progress_step=progress_step,
                    outer_step=outer_step,
                    metrics=metrics,
                    selected_baseline=selected_baseline,
                    trainer=trainer,
                    step_iter=step_iter,
                    use_wandb=use_wandb,
                )
                while next_log_env is not None and current_env_step >= next_log_env:
                    next_log_env += log_env_interval

            if next_eval_env is not None and current_env_step >= next_eval_env:
                _run_eval_and_log(
                    progress_step=progress_step,
                    outer_step=outer_step,
                    trainer=trainer,
                    env_cfg=env_cfg,
                    dataset=eval_dataset,
                    checker_fn=checker_fn,
                    step_iter=step_iter,
                    use_wandb=use_wandb,
                )
                while next_eval_env is not None and current_env_step >= next_eval_env:
                    next_eval_env += eval_env_interval

            if next_save_env is not None and checkpoint_dir is not None and current_env_step >= next_save_env:
                save_checkpoint(
                    model,
                    trainer,
                    progress_step,
                    checkpoint_dir,
                    puzzle_emb_optimizer,
                    rl_cfg,
                    dataset_provenance,
                )
                last_saved_progress_step = progress_step
                while next_save_env is not None and current_env_step >= next_save_env:
                    next_save_env += save_env_interval

        final_progress_step = trainer.get_env_step_count()

    # === Save final checkpoint ===
    if checkpointing_enabled and checkpoint_dir is not None and last_saved_progress_step != final_progress_step:
        save_checkpoint(
            model,
            trainer,
            final_progress_step,
            checkpoint_dir,
            puzzle_emb_optimizer,
            rl_cfg,
            dataset_provenance,
        )
    
    # === WandB: Finish logging ===
    if use_wandb:
        wandb.finish()
        print("[WandB] Run finished successfully")


if __name__ == "__main__":
    # Configure logging only when running as CLI entrypoint (not on import)
    # Only set up if no handlers already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
