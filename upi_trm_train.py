
import argparse
import copy
import hashlib
import importlib.metadata
import logging
import math
import os
import platform
import random
import tempfile
from pathlib import Path
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
from rl.replay import (
    ReplayBuffer,
    ReplayIntegrityError,
    ReplayLatent,
    Transition,
    validate_transition,
    validate_transition_continuity,
)
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
    dataset_puzzle_identifier_sha256s,
    dataset_pool_sha256,
    dataset_sample_sha256s,
    dataset_source_build_metadata,
    ordered_record_sha256,
    validate_dataset_provenance,
)
from utils.run_identity import (
    RunIdentityError,
    assert_git_files_match_head,
    assert_matching_run_identity,
    build_checkpoint_lineage,
    build_run_identity,
    canonical_json_sha256,
    discover_clean_git_source,
    file_sha256,
    validate_run_identity,
    validate_checkpoint_lineage,
    validate_upi_effective_config,
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


FIXED_BASE_CHECKPOINT_SCHEMA_VERSION = 5
LEGACY_UPI_CHECKPOINT_SCHEMA_VERSION = 4


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
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load model weights from a pretrained checkpoint.
    
    Handles puzzle embedding resizing if shapes don't match.
    
    Args:
        model: The TRM model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Target device (auto-detected if None)
        strict: If True, raise error on missing/unexpected keys
        expected_sha256: Optional content hash bound before model mutation
        
    Returns:
        Dict with loading info (missing_keys, unexpected_keys, etc.)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Checkpoint] Loading from {checkpoint_path} to {device}")
    if expected_sha256 is not None:
        state_dict, loaded_sha256 = _load_checkpoint_payload(
            checkpoint_path,
            expected_sha256=expected_sha256,
        )
        if loaded_sha256 != expected_sha256:
            raise RuntimeError("Initialization checkpoint SHA-256 mismatch.")
    else:
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


def _canonical_device(device: Any) -> str:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{index}"
    return str(resolved)


def _runtime_fingerprint() -> Dict[str, Any]:
    """Capture runtime settings that can change exact continuation semantics."""

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
    cudnn = getattr(torch.backends, "cudnn", None)
    package_versions = {}
    for package_name in ("einops", "omegaconf", "pydantic", "PyYAML"):
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = None
    return {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": str(np.__version__),
        "package_versions": package_versions,
        "torch_version": str(torch.__version__),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": (
            int(cudnn.version())
            if cudnn is not None and cudnn.version() is not None
            else None
        ),
        "cuda_devices": cuda_devices,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(getattr(cudnn, "benchmark", False)),
        "cudnn_deterministic": bool(getattr(cudnn, "deterministic", False)),
        "cuda_matmul_allow_tf32": bool(
            getattr(cuda_matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(getattr(cudnn, "allow_tf32", False)),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None
        ),
    }


def _verify_producer_source_matches_runtime(lookup_root: str | Path) -> None:
    """Reject a clean but unrelated repository passed as the producer root."""

    producer_root = Path(lookup_root).expanduser().resolve()
    runtime_root = Path(__file__).resolve().parent

    def source_inventory(root: Path, *, label: str) -> set[str]:
        inventory = {"upi_trm_train.py", "puzzle_dataset.py"}
        for root_file in tuple(inventory):
            if not (root / root_file).is_file():
                raise RuntimeError(f"{label} is missing source {root_file!r}.")
        for source_directory in (
            "dataset",
            "evaluators",
            "models",
            "rl",
            "utils",
        ):
            directory = root / source_directory
            if not directory.is_dir():
                raise RuntimeError(
                    f"{label} is missing source directory {source_directory!r}."
                )
            inventory.update(
                str(path.relative_to(root))
                for path in directory.rglob("*.py")
                if "__pycache__" not in path.parts
            )
        return inventory

    producer_sources = source_inventory(producer_root, label="Producer repository")
    runtime_sources = source_inventory(runtime_root, label="Runtime source tree")
    if producer_sources != runtime_sources:
        missing_from_producer = sorted(runtime_sources - producer_sources)
        missing_from_runtime = sorted(producer_sources - runtime_sources)
        raise RuntimeError(
            "Producer/runtime source inventories differ: "
            f"missing_from_producer={missing_from_producer[:8]}, "
            f"missing_from_runtime={missing_from_runtime[:8]}."
        )
    behavior_sources = producer_sources
    try:
        assert_git_files_match_head(
            producer_root,
            sorted(behavior_sources),
        )
    except RunIdentityError as exc:
        raise RuntimeError(
            "Producer source inventory does not match the recorded Git commit."
        ) from exc
    for relative_path in sorted(behavior_sources):
        producer_source = producer_root / relative_path
        runtime_source = runtime_root / relative_path
        if not producer_source.is_file() or not runtime_source.is_file():
            raise RuntimeError(
                f"Producer repository is missing behavior source {relative_path!r}."
            )
        try:
            producer_sha256 = file_sha256(producer_source)
            runtime_sha256 = file_sha256(runtime_source)
        except RunIdentityError as exc:
            raise RuntimeError(
                f"Cannot hash behavior source {relative_path!r}."
            ) from exc
        if producer_sha256 != runtime_sha256:
            raise RuntimeError(
                "Producer repository does not match the executing source for "
                f"{relative_path!r}."
            )


def _checkpoint_modules(
    model: nn.Module,
    trainer: "UPITrmTrainer",
) -> Dict[str, nn.Module]:
    modules = {"model": model}
    for checkpoint_name, attribute_name in (
        ("policy_model_old", "policy_model_old"),
        ("policy_model_candidate", "policy_model_candidate"),
        ("target_model", "target_model"),
    ):
        module = getattr(trainer, attribute_name, None)
        if module is not None:
            modules[checkpoint_name] = module
    return modules


def _capture_parameter_gradients(module: nn.Module) -> Dict[str, Any]:
    return {
        name: (
            parameter.grad.detach().cpu().clone()
            if parameter.grad is not None
            else None
        )
        for name, parameter in module.named_parameters()
    }


def _restore_parameter_gradients(
    module: nn.Module,
    saved_gradients: Any,
    *,
    label: str,
) -> None:
    if not isinstance(saved_gradients, dict):
        raise RuntimeError(f"Checkpoint {label} gradients must be a dictionary.")
    parameters = dict(module.named_parameters())
    if set(saved_gradients) != set(parameters):
        raise RuntimeError(f"Checkpoint {label} gradient parameter set mismatch.")
    for name, parameter in parameters.items():
        gradient = saved_gradients[name]
        if gradient is None:
            parameter.grad = None
            continue
        if (
            not torch.is_tensor(gradient)
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
        ):
            raise RuntimeError(
                f"Checkpoint {label} gradient shape or dtype mismatch for {name!r}."
            )
        parameter.grad = gradient.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).clone()


def _strict_checkpoint_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Checkpoint {label} must be a non-negative integer.")
    return value


def _strict_checkpoint_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Checkpoint {label} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"Checkpoint {label} must be a finite number.")
    return converted


def _strict_checkpoint_list(
    value: Any,
    *,
    label: str,
    integers: bool = False,
) -> List[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"Checkpoint {label} must be a list.")
    if integers:
        return [
            _strict_checkpoint_int(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    return [
        _strict_checkpoint_float(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _validate_schema_v5_replay(
    transitions: Any,
    *,
    action_count: int,
    require_behavior_log_prob: bool,
    episodic_latent: bool,
    max_edits: int,
    expected_latent_shape: Tuple[int, ...],
    expected_latent_dtype: torch.dtype,
    seq_len: int,
    vocab_size: int,
    num_puzzle_identifiers: int,
) -> List[Transition]:
    if not isinstance(transitions, list):
        raise RuntimeError("Schema-v5 replay transitions must be a list.")

    def validate_state(state: Any, *, label: str) -> None:
        if not isinstance(state, dict) or not {
            "inputs",
            "puzzle_identifiers",
            "remaining_edits",
        }.issubset(state):
            raise RuntimeError(f"Schema-v5 replay {label} has an invalid state inventory.")
        inputs = state["inputs"]
        puzzle_identifier = state["puzzle_identifiers"]
        if (
            not torch.is_tensor(inputs)
            or tuple(inputs.shape) != (seq_len,)
            or inputs.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(f"Schema-v5 replay {label}.inputs is invalid.")
        if inputs.numel() and (
            int(inputs.min().item()) < 0 or int(inputs.max().item()) >= vocab_size
        ):
            raise RuntimeError(f"Schema-v5 replay {label}.inputs is out of range.")
        if (
            not torch.is_tensor(puzzle_identifier)
            or puzzle_identifier.numel() != 1
            or puzzle_identifier.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(
                f"Schema-v5 replay {label}.puzzle_identifiers is invalid."
            )
        puzzle_id = int(puzzle_identifier.detach().cpu().reshape(()).item())
        if puzzle_id < 0 or puzzle_id >= num_puzzle_identifiers:
            raise RuntimeError(
                f"Schema-v5 replay {label}.puzzle_identifiers is out of range."
            )

    def validate_plan(plan: Any, *, label: str) -> None:
        if (
            not torch.is_tensor(plan)
            or tuple(plan.shape) != (seq_len,)
            or plan.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(f"Schema-v5 replay {label} is invalid.")
        if plan.numel() and (
            int(plan.min().item()) < 0 or int(plan.max().item()) >= vocab_size
        ):
            raise RuntimeError(f"Schema-v5 replay {label} is out of range.")

    previous: Optional[Transition] = None
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Transition):
            raise RuntimeError(
                f"Schema-v5 replay record {index} is not a Transition."
            )
        try:
            validate_transition(transition, require_clock=True)
        except (ReplayIntegrityError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"Schema-v5 replay record {index} is invalid."
            ) from exc
        validate_state(transition.x, label=f"record {index} x")
        validate_state(transition.x_next, label=f"record {index} x_next")
        if set(transition.x) != set(transition.x_next):
            raise RuntimeError(
                f"Schema-v5 replay record {index} changes state-field inventory."
            )
        validate_plan(transition.y, label=f"record {index} y")
        validate_plan(transition.y_next, label=f"record {index} y_next")
        if episodic_latent:
            if transition.latent is not None or transition.next_latent is not None:
                raise RuntimeError(
                    "Schema-v5 episodic replay must not carry transition latents."
                )
        else:
            if transition.latent is None or transition.next_latent is None:
                raise RuntimeError(
                    "Schema-v5 persistent replay requires input and successor latents."
                )
            for latent_name, latent in (
                ("latent", transition.latent),
                ("next_latent", transition.next_latent),
            ):
                if (
                    tuple(latent.z_H.shape) != expected_latent_shape
                    or tuple(latent.z_L.shape) != expected_latent_shape
                    or latent.z_H.dtype != expected_latent_dtype
                    or latent.z_L.dtype != expected_latent_dtype
                ):
                    raise RuntimeError(
                        f"Schema-v5 replay {latent_name} {index} has wrong model shape."
                    )
        clock = int(
            transition.x["remaining_edits"].detach().cpu().reshape(()).item()
        )
        next_clock = int(
            transition.x_next["remaining_edits"]
            .detach()
            .cpu()
            .reshape(())
            .item()
        )
        expected_clock = max_edits - transition.timestep
        if clock != expected_clock or next_clock != expected_clock - 1:
            raise RuntimeError(
                f"Schema-v5 replay clock {index} disagrees with timestep."
            )
        done = bool(transition.done.detach().cpu().reshape(()).item())
        if next_clock == 0 and not done:
            raise RuntimeError(
                f"Schema-v5 replay record {index} is nonterminal at zero budget."
            )
        action = transition.action
        if (
            not torch.is_tensor(action)
            or action.numel() != 1
            or action.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(f"Schema-v5 replay action {index} is not scalar integer.")
        action_value = int(action.detach().cpu().reshape(()).item())
        if action_value < 0 or action_value >= action_count:
            raise RuntimeError(f"Schema-v5 replay action {index} is out of range.")
        reward = transition.reward
        if (
            not torch.is_tensor(reward)
            or reward.numel() != 1
            or not bool(torch.isfinite(reward).all().item())
        ):
            raise RuntimeError(f"Schema-v5 replay reward {index} is invalid.")
        behavior_log_prob = transition.behavior_log_prob
        if require_behavior_log_prob:
            if (
                not torch.is_tensor(behavior_log_prob)
                or behavior_log_prob.numel() != 1
                or not bool(torch.isfinite(behavior_log_prob).all().item())
                or float(behavior_log_prob.detach().cpu().reshape(()).item()) > 0
            ):
                raise RuntimeError(
                    f"Schema-v5 replay behavior log probability {index} is invalid."
                )
        if previous is not None:
            if transition.episode_id == previous.episode_id:
                try:
                    validate_transition_continuity(
                        previous,
                        transition,
                        require_clock=True,
                    )
                except ReplayIntegrityError as exc:
                    raise RuntimeError(
                        f"Schema-v5 replay continuity fails at record {index}."
                    ) from exc
            else:
                previous_done = bool(previous.done.detach().cpu().reshape(()).item())
                if (
                    transition.episode_id != previous.episode_id + 1
                    or transition.timestep != 0
                    or not previous_done
                ):
                    raise RuntimeError(
                        f"Schema-v5 replay episode boundary fails at record {index}."
                    )
        previous = transition
    return transitions


def _validate_schema_v5_active_latent(
    collection_state: Any,
    *,
    episodic_latent: bool,
    expected_latent_shape: Tuple[int, ...],
    expected_latent_dtype: torch.dtype,
) -> None:
    if not isinstance(collection_state, dict):
        raise RuntimeError("Schema-v5 collector state must be a dictionary.")
    active_episode = collection_state.get("active_episode")
    if active_episode is None:
        return
    if not isinstance(active_episode, dict):
        raise RuntimeError("Schema-v5 active episode must be a dictionary.")
    active_latent = active_episode.get("latent")
    if episodic_latent:
        if active_latent is not None:
            raise RuntimeError(
                "Schema-v5 episodic active state must not carry a latent."
            )
        return
    if not isinstance(active_latent, ReplayLatent):
        raise RuntimeError("Schema-v5 persistent active state has no carried latent.")
    for component_name, component in (
        ("z_H", active_latent.z_H),
        ("z_L", active_latent.z_L),
    ):
        if (
            not torch.is_tensor(component)
            or tuple(component.shape) != expected_latent_shape
            or component.dtype != expected_latent_dtype
            or not bool(torch.isfinite(component).all().item())
        ):
            raise RuntimeError(
                f"Schema-v5 active latent component {component_name} is invalid."
            )


def _atomic_torch_save(value: Any, destination: str) -> None:
    """Publish one checkpoint atomically without replacing an existing file."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=str(destination_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination_path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint {destination_path.name!r}."
            ) from exc
        directory_fd = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fixed_base_effective_config(
    *,
    args: argparse.Namespace,
    rl_config: Dict[str, Any],
    model_config: Dict[str, Any],
    execution_device: str,
    train_record_count: int,
    eval_record_count: int,
) -> Dict[str, Any]:
    """Build path-free behavior and schedule identity for one registered run."""

    config_source_sha256s = [file_sha256(path) for path in (args.config or [])]
    return {
        "effective_config_schema_version": 1,
        "algorithm": "upi_trm",
        "training_protocol": "fixed_base_exact",
        "backbone": args.backbone,
        "rl_config": rl_config,
        "model_config": model_config,
        "execution_device": execution_device,
        "runtime_fingerprint_sha256": canonical_json_sha256(
            _runtime_fingerprint()
        ),
        "dataset": {
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "train_record_count": train_record_count,
            "eval_record_count": eval_record_count,
        },
        "budget": {
            "outer_train_steps": rl_config["num_train_steps"],
            "environment_interactions": args.env_step_budget,
        },
        "schedule": {
            "log_outer_interval": rl_config["log_interval"],
            "eval_outer_interval": rl_config["eval_interval"],
            "save_outer_interval": args.save_interval,
            "log_environment_interval": args.log_env_interval,
            "eval_environment_interval": args.eval_env_interval,
            "save_environment_interval": args.save_env_interval,
        },
        "evaluation": {
            "episode_count": rl_config["eval_num_episodes"],
            "seed": rl_config["eval_seed"],
            "pool_size": eval_record_count,
        },
        "puzzle_embedding_optimizer": {
            "learning_rate": args.puzzle_emb_lr,
            "weight_decay": args.puzzle_emb_weight_decay,
        },
        "imitation": {
            "enabled": args.imitation_pretrain,
            "epochs": args.imitation_epochs,
        },
        "external_logging": "disabled",
        "debug_checks": args.debug_checks,
        "config_source_sha256s": config_source_sha256s,
    }


def _validate_run_identity_bindings(
    identity: Any,
    *,
    rl_config: Dict[str, Any],
    model_config: Dict[str, Any],
    dataset_provenance: Dict[str, Any],
    execution_device: str,
    runtime_fingerprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        canonical = validate_run_identity(identity)
    except RunIdentityError as exc:
        raise RuntimeError(f"Invalid schema-v5 run identity: {exc}") from exc
    effective = canonical["effective_config"]
    try:
        effective = validate_upi_effective_config(effective)
    except RunIdentityError as exc:
        raise RuntimeError(f"Invalid schema-v5 effective config: {exc}") from exc
    expected_bindings = {
        "training_protocol": "fixed_base_exact",
        "rl_config": rl_config,
        "model_config": model_config,
        "execution_device": execution_device,
    }
    for field, expected in expected_bindings.items():
        if effective.get(field) != expected:
            raise RuntimeError(
                f"Schema-v5 run identity {field!r} does not match the active run."
            )
    active_runtime_fingerprint = (
        runtime_fingerprint
        if runtime_fingerprint is not None
        else _runtime_fingerprint()
    )
    if effective["runtime_fingerprint_sha256"] != canonical_json_sha256(
        active_runtime_fingerprint
    ):
        raise RuntimeError(
            "Schema-v5 runtime settings changed after run identity construction."
        )
    if canonical["dataset_provenance_sha256"] != canonical_json_sha256(
        dataset_provenance
    ):
        raise RuntimeError(
            "Schema-v5 run identity does not match dataset provenance."
        )
    dataset_identity = effective["dataset"]
    provenance_splits = dataset_provenance["splits"]
    provenance_records = dataset_provenance["ordered_records"]
    train_record_hashes = provenance_records["train"]["record_sha256s"]
    eval_record_hashes = provenance_records["eval"]["record_sha256s"]
    if len(set(train_record_hashes)) != len(train_record_hashes):
        raise RuntimeError("Schema-v5 training provenance contains duplicate records.")
    if len(set(eval_record_hashes)) != len(eval_record_hashes):
        raise RuntimeError("Schema-v5 evaluation provenance contains duplicate records.")
    if set(train_record_hashes).intersection(eval_record_hashes):
        raise RuntimeError("Schema-v5 train/evaluation provenance overlaps.")
    dataset_bindings = {
        "train_split": provenance_splits["train"],
        "eval_split": provenance_splits["eval"],
        "train_record_count": provenance_records["train"]["count"],
        "eval_record_count": provenance_records["eval"]["count"],
    }
    if dataset_identity != dataset_bindings:
        raise RuntimeError(
            "Schema-v5 effective dataset configuration does not match provenance."
        )
    if effective["budget"]["outer_train_steps"] != rl_config["num_train_steps"]:
        raise RuntimeError("Schema-v5 outer-step budget does not match RL config.")
    schedule_bindings = {
        "log_outer_interval": rl_config["log_interval"],
        "eval_outer_interval": rl_config["eval_interval"],
    }
    for field, expected in schedule_bindings.items():
        if effective["schedule"][field] != expected:
            raise RuntimeError(
                f"Schema-v5 schedule field {field!r} does not match RL config."
            )
    evaluation_bindings = {
        "episode_count": rl_config["eval_num_episodes"],
        "seed": rl_config["eval_seed"],
        "pool_size": provenance_records["eval"]["count"],
    }
    if effective["evaluation"] != evaluation_bindings:
        raise RuntimeError(
            "Schema-v5 evaluation configuration does not match RL config and provenance."
        )
    if effective["imitation"]["enabled"]:
        raise RuntimeError(
            "Schema-v5 checkpoints do not support resumable imitation pretraining."
        )
    if effective["debug_checks"] != rl_config["debug_checks"]:
        raise RuntimeError("Schema-v5 debug setting does not match RL config.")
    return canonical


def _validate_schema_v5_progress(
    *,
    checkpoint_step: int,
    environment_steps: int,
    outer_steps: int,
    value_optimizer_steps: int,
    policy_optimizer_steps: int,
    distill_optimizer_steps: int,
    puzzle_optimizer_steps: int,
    effective_config: Dict[str, Any],
) -> None:
    checkpoint_step = _strict_checkpoint_int(checkpoint_step, label="step")
    counters = {
        "environment_steps": environment_steps,
        "outer_steps": outer_steps,
        "value_optimizer_steps": value_optimizer_steps,
        "policy_optimizer_steps": policy_optimizer_steps,
        "distill_optimizer_steps": distill_optimizer_steps,
        "puzzle_optimizer_steps": puzzle_optimizer_steps,
    }
    for name, value in counters.items():
        _strict_checkpoint_int(value, label=name)
    environment_budget = effective_config["budget"]["environment_interactions"]
    outer_budget = effective_config["budget"]["outer_train_steps"]
    if environment_budget is not None:
        if checkpoint_step != environment_steps:
            raise RuntimeError(
                "Environment-budget checkpoint step must equal exact interactions."
            )
        if environment_steps > environment_budget:
            raise RuntimeError("Checkpoint exceeds its registered interaction budget.")
    else:
        if checkpoint_step != outer_steps:
            raise RuntimeError(
                "Outer-budget checkpoint step must equal completed outer steps."
            )
        if outer_steps > outer_budget:
            raise RuntimeError("Checkpoint exceeds its registered outer-step budget.")
    if value_optimizer_steps > outer_steps or policy_optimizer_steps > outer_steps:
        raise RuntimeError("Optimizer-step counters exceed completed outer steps.")
    if distill_optimizer_steps > policy_optimizer_steps:
        raise RuntimeError("Distillation steps exceed policy optimizer steps.")
    if puzzle_optimizer_steps > value_optimizer_steps:
        raise RuntimeError("Puzzle optimizer steps exceed value optimizer steps.")


def _load_schema_v5_resume_metadata(
    checkpoint_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint, checkpoint_sha256 = _load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or checkpoint.get(
        "checkpoint_schema_version"
    ) != FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Confirmatory resume requires a schema-v5 fixed-base checkpoint."
        )
    try:
        identity = validate_run_identity(checkpoint.get("run_identity"))
        validate_checkpoint_lineage(checkpoint.get("checkpoint_lineage"))
    except RunIdentityError as exc:
        raise RuntimeError(f"Checkpoint lineage metadata is invalid: {exc}") from exc
    step = checkpoint.get("step")
    progress = checkpoint.get("progress")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or not isinstance(progress, dict)
    ):
        raise RuntimeError("Schema-v5 resume checkpoint has invalid progress metadata.")
    environment_steps = progress.get("env_steps")
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps < 0
    ):
        raise RuntimeError("Schema-v5 resume checkpoint has invalid interaction count.")
    try:
        parent_link = build_checkpoint_lineage(
            parent_checkpoint_sha256=checkpoint_sha256,
            parent_checkpoint_step=step,
            parent_environment_steps=environment_steps,
        )
    except RunIdentityError as exc:
        raise RuntimeError("Cannot hash schema-v5 resume checkpoint.") from exc
    return identity, parent_link


def _load_checkpoint_payload(
    checkpoint_path: str,
    *,
    expected_sha256: Optional[str] = None,
) -> Tuple[Any, str]:
    """Hash and load one stable open file, detecting replacement or mutation."""

    digest_before = hashlib.sha256()
    try:
        with open(checkpoint_path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest_before.update(block)
            checkpoint_sha256 = digest_before.hexdigest()
            if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
                raise RuntimeError(
                    "Resume checkpoint SHA-256 does not match its registered parent link."
                )
            handle.seek(0)
            checkpoint = torch.load(
                handle,
                map_location="cpu",
                weights_only=False,
            )
            handle.seek(0)
            digest_after = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest_after.update(block)
    except OSError as exc:
        raise RuntimeError("Resume checkpoint cannot be read.") from exc
    if digest_after.hexdigest() != checkpoint_sha256:
        raise RuntimeError("Resume checkpoint changed while it was being loaded.")
    return checkpoint, checkpoint_sha256


def save_checkpoint(
    model: nn.Module,
    trainer: "UPITrmTrainer",
    step: int,
    checkpoint_dir: str,
    puzzle_emb_optimizer: Optional[torch.optim.Optimizer] = None,
    rl_cfg: Optional["RLConfig"] = None,
    dataset_provenance: Optional[Dict[str, Any]] = None,
    run_identity: Optional[Dict[str, Any]] = None,
    checkpoint_lineage: Optional[Dict[str, Any]] = None,
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
        run_identity: Required clean-source identity for fixed-base checkpoints
        checkpoint_lineage: Content-addressed parent checkpoint, or a root marker

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    is_upi_checkpoint = isinstance(trainer, UPITrmTrainer)
    if is_upi_checkpoint and dataset_provenance is None:
        raise RuntimeError(
            "UPI checkpoints require dataset provenance."
        )
    canonical_dataset_provenance: Optional[Dict[str, Any]] = None
    if dataset_provenance is not None:
        if is_upi_checkpoint:
            canonical_dataset_provenance = validate_dataset_provenance(
                dataset_provenance
            )
        else:
            # Baseline checkpoints are weights-only. Preserve their historical
            # optional metadata without presenting it as exact-resume identity.
            canonical_dataset_provenance = dict(dataset_provenance)

    trainer_rl_cfg = getattr(trainer, "rl_cfg", None)
    if is_upi_checkpoint:
        if trainer_rl_cfg is None:
            raise RuntimeError("UPI checkpoint requires the trainer RL configuration.")
        if rl_cfg is not None and _config_dict(rl_cfg) != _config_dict(trainer_rl_cfg):
            raise RuntimeError(
                "Caller RL configuration differs from the active trainer configuration."
            )
        effective_rl_cfg = trainer_rl_cfg
    else:
        effective_rl_cfg = rl_cfg if rl_cfg is not None else trainer_rl_cfg
    if effective_rl_cfg is None:
        raise RuntimeError("Checkpoint requires the active RL configuration.")
    training_protocol = (
        str(getattr(effective_rl_cfg, "training_protocol", "legacy"))
        if is_upi_checkpoint
        else "weights_only"
    )
    if training_protocol == "fixed_base_exact" and bool(
        getattr(trainer, "_train_step_active", False)
    ):
        raise RuntimeError(
            "Schema-v5 checkpoint save requires an idle trainer between train calls."
        )
    model_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
    path = os.path.join(checkpoint_dir, f"rl_checkpoint_step_{step}.pt")
    stale_model_paths = sorted(
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("model_step_") and name.endswith(".pt")
    )
    if training_protocol == "fixed_base_exact" and stale_model_paths:
        raise RuntimeError(
            "Refusing fixed-base checkpoint save because stale single-model "
            f"artifacts exist: {stale_model_paths}"
        )
    intended_outputs = [path]
    if training_protocol != "fixed_base_exact":
        intended_outputs.append(model_path)
    existing_outputs = [
        output for output in intended_outputs if os.path.exists(output)
    ]
    if existing_outputs:
        raise RuntimeError(
            "Refusing to overwrite existing checkpoint artifacts: "
            f"{existing_outputs}"
        )

    try:
        execution_device = _canonical_device(next(model.parameters()).device)
    except StopIteration:
        execution_device = "cpu"

    model_config_object = getattr(model, "config", None)
    if model_config_object is None:
        raise RuntimeError("Checkpoint requires the active model configuration.")
    model_config = _config_dict(model_config_object)
    rl_config = _config_dict(effective_rl_cfg)
    runtime_fingerprint = _runtime_fingerprint()
    canonical_run_identity: Optional[Dict[str, Any]] = None
    if training_protocol == "fixed_base_exact":
        if not is_upi_checkpoint:
            raise RuntimeError(
                "fixed_base_exact checkpoints require UPITrmTrainer."
            )
        if run_identity is None:
            raise RuntimeError(
                "Schema-v5 fixed-base checkpoints require a run identity."
            )
        assert canonical_dataset_provenance is not None
        canonical_run_identity = _validate_run_identity_bindings(
            run_identity,
            rl_config=rl_config,
            model_config=model_config,
            dataset_provenance=canonical_dataset_provenance,
            execution_device=execution_device,
            runtime_fingerprint=runtime_fingerprint,
        )
        if checkpoint_lineage is None:
            raise RuntimeError(
                "Schema-v5 fixed-base checkpoints require checkpoint lineage."
            )
        try:
            canonical_checkpoint_lineage = validate_checkpoint_lineage(
                checkpoint_lineage
            )
        except RunIdentityError as exc:
            raise RuntimeError(f"Invalid schema-v5 checkpoint lineage: {exc}") from exc
        checkpoint_schema_version = FIXED_BASE_CHECKPOINT_SCHEMA_VERSION
    else:
        if run_identity is not None:
            raise RuntimeError(
                "Run identity is reserved for schema-v5 fixed-base checkpoints."
            )
        if checkpoint_lineage is not None:
            raise RuntimeError(
                "Checkpoint lineage is reserved for schema-v5 fixed-base checkpoints."
            )
        canonical_checkpoint_lineage = None
        checkpoint_schema_version = (
            LEGACY_UPI_CHECKPOINT_SCHEMA_VERSION if is_upi_checkpoint else 2
        )

    rng_state = _capture_rng_state()
    if checkpoint_schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        progress = {
            "env_steps": int(getattr(trainer, "_env_step_count", 0)),
            "outer_steps": int(getattr(trainer, "_train_step_count", 0)),
            "value_optimizer_steps": int(
                getattr(trainer, "_value_optimizer_step_count", 0)
            ),
            "policy_optimizer_steps": int(
                getattr(trainer, "_policy_optimizer_step_count", 0)
            ),
            "distill_optimizer_steps": int(
                getattr(trainer, "_distill_optimizer_step_count", 0)
            ),
            "puzzle_optimizer_steps": int(
                getattr(trainer, "_puzzle_optimizer_step_count", 0)
            ),
        }
    else:
        progress = {
            "env_steps": int(getattr(trainer, "_env_step_count", 0)),
            "optimizer_updates": int(getattr(trainer, "_train_step_count", 0)),
        }
    if checkpoint_schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        assert canonical_run_identity is not None
        _validate_schema_v5_progress(
            checkpoint_step=step,
            environment_steps=progress["env_steps"],
            outer_steps=progress["outer_steps"],
            value_optimizer_steps=progress["value_optimizer_steps"],
            policy_optimizer_steps=progress["policy_optimizer_steps"],
            distill_optimizer_steps=progress["distill_optimizer_steps"],
            puzzle_optimizer_steps=progress["puzzle_optimizer_steps"],
            effective_config=canonical_run_identity["effective_config"],
        )
    collection_checkpoint_state = None
    environment_checkpoint_state = None
    replay_snapshot = list(
        getattr(getattr(trainer, "replay", None), "storage", [])
    )
    if is_upi_checkpoint:
        collection_checkpoint_state = trainer.collection_checkpoint_state()
        environment_checkpoint_state = trainer.env.checkpoint_state()
    if checkpoint_schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        model_puzzle_emb_len = int(model_config.get("puzzle_emb_len", 0))
        if model_puzzle_emb_len == 0:
            model_puzzle_emb_len = -(
                -int(model_config.get("puzzle_emb_ndim", 0))
                // int(model_config["hidden_size"])
            )
        expected_latent_shape = (
            1,
            int(model_config["seq_len"]) + model_puzzle_emb_len,
            int(model_config["hidden_size"]),
        )
        _validate_schema_v5_replay(
            replay_snapshot,
            action_count=int(model_config["rl_num_actions"]),
            require_behavior_log_prob=bool(
                rl_config.get("use_importance_sampling", True)
            ),
            episodic_latent=bool(rl_config["episodic_latent"]),
            max_edits=int(rl_config["max_edits"]),
            expected_latent_shape=expected_latent_shape,
            expected_latent_dtype=next(model.parameters()).dtype,
            seq_len=int(model_config["seq_len"]),
            vocab_size=int(model_config["vocab_size"]),
            num_puzzle_identifiers=int(model_config["num_puzzle_identifiers"]),
        )
        _validate_schema_v5_active_latent(
            collection_checkpoint_state,
            episodic_latent=bool(rl_config["episodic_latent"]),
            expected_latent_shape=expected_latent_shape,
            expected_latent_dtype=next(model.parameters()).dtype,
        )
        if replay_snapshot:
            assert isinstance(collection_checkpoint_state, dict)
            active_episode = collection_checkpoint_state.get("active_episode")
            active_timestep = (
                int(active_episode.get("timestep", -1))
                if isinstance(active_episode, dict)
                else None
            )
            expected_next_episode_id = (
                replay_snapshot[-1].episode_id
                if active_timestep is not None and active_timestep > 0
                else replay_snapshot[-1].episode_id + 1
            )
            if int(getattr(trainer, "_next_episode_id", -1)) != (
                expected_next_episode_id
            ):
                raise RuntimeError(
                    "Schema-v5 next episode ID disagrees with replay before save."
                )
    checkpoint = {
        "checkpoint_schema_version": checkpoint_schema_version,
        "training_protocol": training_protocol,
        "execution_device": execution_device,
        "trainer_kind": type(trainer).__name__,
        "step": step,
        "progress": progress,
        "model_state_dict": model.state_dict(),
        "rng_state": rng_state,
    }
    if canonical_run_identity is not None:
        checkpoint["run_identity"] = canonical_run_identity
        checkpoint["checkpoint_lineage"] = canonical_checkpoint_lineage
        modules = _checkpoint_modules(model, trainer)
        checkpoint["runtime_fingerprint"] = runtime_fingerprint
        checkpoint["module_training_modes"] = {
            name: bool(module.training) for name, module in modules.items()
        }
        checkpoint["parameter_gradients"] = {
            name: _capture_parameter_gradients(module)
            for name, module in modules.items()
        }
        checkpoint["checkpoint_phase"] = "idle_between_training_calls"
    if canonical_dataset_provenance is not None:
        checkpoint["dataset_provenance"] = canonical_dataset_provenance
    checkpoint["model_config"] = model_config

    # Preserve the old/candidate policy pair for post-candidate diagnostics.
    if hasattr(trainer, "policy_model_old") and trainer.policy_model_old is not None:
        checkpoint["policy_model_old_state_dict"] = trainer.policy_model_old.state_dict()
    if hasattr(trainer, "policy_model_candidate") and trainer.policy_model_candidate is not None:
        checkpoint["policy_model_candidate_state_dict"] = trainer.policy_model_candidate.state_dict()
    if hasattr(trainer, "target_model") and trainer.target_model is not None:
        checkpoint["target_model_state_dict"] = trainer.target_model.state_dict()

    # Save RL config for reproducibility and correct eval loading.
    checkpoint["rl_config"] = rl_config

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
        "value_optimizer_step_count": int(
            getattr(trainer, "_value_optimizer_step_count", 0)
        ),
        "policy_optimizer_step_count": int(
            getattr(trainer, "_policy_optimizer_step_count", 0)
        ),
        "distill_optimizer_step_count": int(
            getattr(trainer, "_distill_optimizer_step_count", 0)
        ),
        "puzzle_optimizer_step_count": int(
            getattr(trainer, "_puzzle_optimizer_step_count", 0)
        ),
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
        "opnorm_clamp_warned": bool(
            getattr(trainer, "_opnorm_clamp_warned", False)
        ),
    }
    if is_upi_checkpoint:
        checkpoint["trainer_state"].update(
            {
                "collection_state": collection_checkpoint_state,
                "environment_state": environment_checkpoint_state,
            }
        )
    else:
        print(
            "[Checkpoint] Baseline trainer checkpoint is weights-only for future "
            "warm starts; exact resume requires the schema-v4 UPI path."
        )

    # Replay is required for a semantic resume. It can make checkpoints large,
    # but storing only its length caused resumed runs to start from empty data.
    if hasattr(trainer, 'replay'):
        checkpoint["replay_buffer_size"] = len(trainer.replay)
        checkpoint["replay_capacity"] = trainer.replay.storage.maxlen
        checkpoint["replay_transitions"] = replay_snapshot
    
    try:
        _atomic_torch_save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")

        # A single state dict cannot represent an exact old/candidate mixture.
        # Keep the convenient weights-only artifact only for protocols whose
        # deployed policy is a single model.
        if training_protocol != "fixed_base_exact":
            _atomic_torch_save(model.state_dict(), model_path)
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
    expected_run_identity: Optional[Dict[str, Any]] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    allow_legacy_warm_start: bool = False,
) -> int:
    """
    Resume RL training from a saved checkpoint.
    
    Returns:
        Starting step number
    """
    if bool(getattr(trainer, "_train_step_active", False)):
        raise RuntimeError("Cannot resume while a trainer step is active.")
    print(f"[Checkpoint] Resuming from {checkpoint_path}")
    # Resume checkpoints contain replay Transition dataclasses, so this is a
    # trusted local artifact rather than a weights-only file.
    checkpoint, loaded_checkpoint_sha256 = _load_checkpoint_payload(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Resume checkpoint must contain a dictionary payload.")
    
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
    if schema_version not in (3, 4, FIXED_BASE_CHECKPOINT_SCHEMA_VERSION):
        raise RuntimeError(
            "Unsupported checkpoint schema version "
            f"{schema_version}; expected 3, 4, or 5."
        )
    if checkpoint.get("trainer_kind") != type(trainer).__name__:
        raise RuntimeError(
            "Checkpoint trainer mismatch: "
            f"{checkpoint.get('trainer_kind')!r} != {type(trainer).__name__!r}."
        )

    active_protocol = str(getattr(trainer.rl_cfg, "training_protocol", "legacy"))
    validated_checkpoint_lineage: Optional[Dict[str, Any]] = None
    saved_rl_config = checkpoint.get("rl_config")
    if not isinstance(saved_rl_config, dict):
        raise RuntimeError(f"Schema-v{schema_version} checkpoint is missing RL config.")
    saved_rl_config = dict(saved_rl_config)
    if schema_version == 3:
        if active_protocol != "legacy":
            raise RuntimeError(
                "Schema-v3 checkpoints predate fixed-base protocol identity and "
                "cannot resume into fixed_base_exact."
            )
        saved_protocol = "legacy"
        saved_rl_config.setdefault("training_protocol", "legacy")
    else:
        saved_protocol = checkpoint.get("training_protocol")
        nested_protocol = saved_rl_config.get("training_protocol")
        if not isinstance(saved_protocol, str) or not isinstance(nested_protocol, str):
            raise RuntimeError(
                f"Schema-v{schema_version} checkpoint is missing training protocol identity."
            )
        if saved_protocol != nested_protocol:
            raise RuntimeError(
                "Checkpoint top-level and nested training protocols disagree."
            )
        if saved_protocol != active_protocol:
            raise RuntimeError(
                "Checkpoint training protocol does not match the active trainer."
            )

        if schema_version == LEGACY_UPI_CHECKPOINT_SCHEMA_VERSION:
            if active_protocol == "fixed_base_exact":
                raise RuntimeError(
                    "Schema-v4 fixed-base checkpoints lack confirmatory run identity; "
                    "use them only as historical weights, not exact resume inputs."
                )
            if expected_run_identity is not None:
                raise RuntimeError(
                    "Legacy schema-v4 resume must not be assigned a schema-v5 run identity."
                )
        else:
            if active_protocol != "fixed_base_exact":
                raise RuntimeError(
                    "Schema-v5 checkpoints are reserved for fixed_base_exact runs."
                )
            if expected_run_identity is None:
                raise RuntimeError(
                    "Schema-v5 exact resume requires the current run identity."
                )
            if expected_checkpoint_sha256 is None:
                raise RuntimeError(
                    "Schema-v5 exact resume requires the registered checkpoint SHA-256."
                )
            if loaded_checkpoint_sha256 != expected_checkpoint_sha256:
                raise RuntimeError("Schema-v5 checkpoint SHA-256 mismatch.")
            try:
                assert_matching_run_identity(
                    checkpoint.get("run_identity"),
                    expected_run_identity,
                )
            except RunIdentityError as exc:
                raise RuntimeError(
                    f"Run identity mismatch; state was not restored: {exc}"
                ) from exc
            try:
                validated_checkpoint_lineage = validate_checkpoint_lineage(
                    checkpoint.get("checkpoint_lineage")
                )
            except RunIdentityError as exc:
                raise RuntimeError(
                    f"Checkpoint lineage is invalid; state was not restored: {exc}"
                ) from exc

        saved_device = checkpoint.get("execution_device")
        try:
            active_device = _canonical_device(next(model.parameters()).device)
        except StopIteration:
            active_device = "cpu"
        requested_device = _canonical_device(device)
        if requested_device != active_device:
            raise RuntimeError(
                "Requested resume device does not match the active model device."
            )
        if not isinstance(saved_device, str) or saved_device != active_device:
            raise RuntimeError(
                "Checkpoint execution device does not match the active model device."
            )

    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        raise RuntimeError(f"Schema-v{schema_version} checkpoint is missing RNG state.")
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
            f"Schema-v{schema_version} exact resume requires current dataset provenance."
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
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        assert expected_run_identity is not None
        assert isinstance(checkpoint_provenance, dict)
        active_runtime_fingerprint = _runtime_fingerprint()
        _validate_run_identity_bindings(
            checkpoint["run_identity"],
            rl_config=current_rl_config,
            model_config=current_model_config,
            dataset_provenance=checkpoint_provenance,
            execution_device=_canonical_device(device),
            runtime_fingerprint=active_runtime_fingerprint,
        )
        if checkpoint.get("checkpoint_phase") != "idle_between_training_calls":
            raise RuntimeError("Schema-v5 checkpoint has an invalid training phase.")
        if checkpoint.get("runtime_fingerprint") != active_runtime_fingerprint:
            raise RuntimeError(
                "Runtime fingerprint differs from the schema-v5 checkpoint."
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
            f"Schema-v{schema_version} checkpoint is missing required fields: {missing}."
        )
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        schema_v5_required = {
            "run_identity",
            "runtime_fingerprint",
            "module_training_modes",
            "parameter_gradients",
            "checkpoint_phase",
            "checkpoint_lineage",
        }
        missing_v5 = sorted(schema_v5_required - set(checkpoint))
        if missing_v5:
            raise RuntimeError(
                f"Schema-v5 checkpoint is missing required fields: {missing_v5}."
            )
    if not isinstance(progress, dict) or not isinstance(trainer_state, dict):
        raise RuntimeError(
            f"Schema-v{schema_version} checkpoint is missing progress or trainer state."
        )
    if "environment_state" not in trainer_state or "collection_state" not in trainer_state:
        raise RuntimeError(
            f"Schema-v{schema_version} checkpoint is missing live environment or collector state."
        )
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        saved_next_episode_id = _strict_checkpoint_int(
            trainer_state.get("next_episode_id"),
            label="trainer_state.next_episode_id",
        )
        saved_kl_coef = _strict_checkpoint_float(
            trainer_state.get("kl_coef"),
            label="trainer_state.kl_coef",
        )
        if saved_kl_coef < 0:
            raise RuntimeError("Checkpoint trainer_state.kl_coef must be non-negative.")
        raw_term_stats = trainer_state.get("term_stats")
        if not isinstance(raw_term_stats, dict) or set(raw_term_stats) != {
            "stop",
            "solved",
            "budget",
        }:
            raise RuntimeError(
                "Checkpoint trainer_state.term_stats has an invalid inventory."
            )
        saved_term_stats = {
            key: _strict_checkpoint_float(
                raw_term_stats[key],
                label=f"trainer_state.term_stats.{key}",
            )
            for key in ("stop", "solved", "budget")
        }
        if any(value < 0 for value in saved_term_stats.values()):
            raise RuntimeError("Checkpoint termination counts must be non-negative.")
        saved_debug_episode_lengths = _strict_checkpoint_list(
            trainer_state.get("debug_episode_lengths"),
            label="trainer_state.debug_episode_lengths",
            integers=True,
        )
        saved_debug_episode_returns = _strict_checkpoint_list(
            trainer_state.get("debug_episode_returns"),
            label="trainer_state.debug_episode_returns",
        )
        saved_debug_stop_probs = _strict_checkpoint_list(
            trainer_state.get("debug_stop_probs"),
            label="trainer_state.debug_stop_probs",
        )
        saved_debug_score_changes = _strict_checkpoint_list(
            trainer_state.get("debug_score_changes"),
            label="trainer_state.debug_score_changes",
        )
        saved_drift_values = _strict_checkpoint_list(
            trainer_state.get("drift_values"),
            label="trainer_state.drift_values",
        )
        saved_plan_changes = _strict_checkpoint_list(
            trainer_state.get("plan_changes"),
            label="trainer_state.plan_changes",
        )
        saved_value_of_memory = _strict_checkpoint_list(
            trainer_state.get("value_of_memory"),
            label="trainer_state.value_of_memory",
        )
        raw_opnorm_warning = trainer_state.get("opnorm_clamp_warned")
        if not isinstance(raw_opnorm_warning, bool):
            raise RuntimeError(
                "Checkpoint trainer_state.opnorm_clamp_warned must be boolean."
            )
        saved_opnorm_warning = raw_opnorm_warning
        saved_value_optimizer_steps = _strict_checkpoint_int(
            trainer_state.get("value_optimizer_step_count"),
            label="trainer_state.value_optimizer_step_count",
        )
        saved_policy_optimizer_steps = _strict_checkpoint_int(
            trainer_state.get("policy_optimizer_step_count"),
            label="trainer_state.policy_optimizer_step_count",
        )
        saved_distill_optimizer_steps = _strict_checkpoint_int(
            trainer_state.get("distill_optimizer_step_count"),
            label="trainer_state.distill_optimizer_step_count",
        )
        saved_puzzle_optimizer_steps = _strict_checkpoint_int(
            trainer_state.get("puzzle_optimizer_step_count"),
            label="trainer_state.puzzle_optimizer_step_count",
        )
    else:
        saved_next_episode_id = int(trainer_state["next_episode_id"])
        saved_kl_coef = float(trainer_state.get("kl_coef", 1.0))
        saved_term_stats = (
            dict(trainer_state["term_stats"])
            if trainer_state.get("term_stats") is not None
            else None
        )
        saved_debug_episode_lengths = list(
            trainer_state.get("debug_episode_lengths", [])
        )
        saved_debug_episode_returns = list(
            trainer_state.get("debug_episode_returns", [])
        )
        saved_debug_stop_probs = list(trainer_state.get("debug_stop_probs", []))
        saved_debug_score_changes = list(
            trainer_state.get("debug_score_changes", [])
        )
        saved_drift_values = list(trainer_state.get("drift_values", []))
        saved_plan_changes = list(trainer_state.get("plan_changes", []))
        saved_value_of_memory = list(trainer_state.get("value_of_memory", []))
        saved_opnorm_warning = bool(
            trainer_state.get("opnorm_clamp_warned", False)
        )
        saved_value_optimizer_steps = int(
            trainer_state.get("value_optimizer_step_count", 0)
        )
        saved_policy_optimizer_steps = int(
            trainer_state.get("policy_optimizer_step_count", 0)
        )
        saved_distill_optimizer_steps = int(
            trainer_state.get("distill_optimizer_step_count", 0)
        )
        saved_puzzle_optimizer_steps = int(
            trainer_state.get("puzzle_optimizer_step_count", 0)
        )
    if int(checkpoint["replay_capacity"]) != int(trainer.replay.storage.maxlen):
        raise RuntimeError(
            "Checkpoint replay capacity mismatch; exact continuation is impossible."
        )
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        saved_train_steps = _strict_checkpoint_int(
            trainer_state.get("train_step_count"),
            label="trainer_state.train_step_count",
        )
        saved_env_steps = _strict_checkpoint_int(
            trainer_state.get("env_step_count"),
            label="trainer_state.env_step_count",
        )
    else:
        saved_train_steps = int(trainer_state.get("train_step_count", -1))
        saved_env_steps = int(trainer_state.get("env_step_count", -1))
    if saved_train_steps < 0 or saved_env_steps < 0:
        raise RuntimeError("Checkpoint contains negative progress counters.")
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        expected_progress_fields = {
            "env_steps",
            "outer_steps",
            "value_optimizer_steps",
            "policy_optimizer_steps",
            "distill_optimizer_steps",
            "puzzle_optimizer_steps",
        }
        if set(progress) != expected_progress_fields:
            raise RuntimeError("Schema-v5 progress counter inventory is invalid.")
        progress_counters = {
            field: _strict_checkpoint_int(progress.get(field), label=f"progress.{field}")
            for field in sorted(expected_progress_fields)
        }
        expected_counters = {
            "env_steps": saved_env_steps,
            "outer_steps": saved_train_steps,
            "value_optimizer_steps": saved_value_optimizer_steps,
            "policy_optimizer_steps": saved_policy_optimizer_steps,
            "distill_optimizer_steps": saved_distill_optimizer_steps,
            "puzzle_optimizer_steps": saved_puzzle_optimizer_steps,
        }
        if progress_counters != expected_counters:
            raise RuntimeError("Schema-v5 progress counters disagree with trainer state.")
    elif saved_train_steps != int(progress.get("optimizer_updates", -1)):
        raise RuntimeError("Checkpoint optimizer-update counters disagree.")
    if saved_env_steps != int(progress.get("env_steps", -1)):
        raise RuntimeError("Checkpoint environment-step counters disagree.")
    checkpoint_step = (
        _strict_checkpoint_int(checkpoint.get("step"), label="step")
        if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION
        else int(checkpoint.get("step", -1))
    )
    if checkpoint_step < 0:
        raise RuntimeError("Checkpoint step must be non-negative.")
    if validated_checkpoint_lineage is not None and (
        validated_checkpoint_lineage["parent_checkpoint_step"] is not None
    ):
        if validated_checkpoint_lineage["parent_checkpoint_step"] > checkpoint_step:
            raise RuntimeError("Checkpoint parent step exceeds the child step.")
        if (
            validated_checkpoint_lineage["parent_environment_steps"]
            > saved_env_steps
        ):
            raise RuntimeError(
                "Checkpoint parent interaction count exceeds the child count."
            )
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        _validate_schema_v5_progress(
            checkpoint_step=checkpoint_step,
            environment_steps=saved_env_steps,
            outer_steps=saved_train_steps,
            value_optimizer_steps=saved_value_optimizer_steps,
            policy_optimizer_steps=saved_policy_optimizer_steps,
            distill_optimizer_steps=saved_distill_optimizer_steps,
            puzzle_optimizer_steps=saved_puzzle_optimizer_steps,
            effective_config=checkpoint["run_identity"]["effective_config"],
        )
    replay_transitions = checkpoint["replay_transitions"]
    expected_replay_latent_shape: Optional[Tuple[int, ...]] = None
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        puzzle_emb_len = int(current_model_config.get("puzzle_emb_len", 0))
        if puzzle_emb_len == 0:
            puzzle_emb_ndim = int(current_model_config.get("puzzle_emb_ndim", 0))
            hidden_size = int(current_model_config["hidden_size"])
            puzzle_emb_len = -(-puzzle_emb_ndim // hidden_size)
        expected_replay_latent_shape = (
            1,
            int(current_model_config["seq_len"]) + puzzle_emb_len,
            int(current_model_config["hidden_size"]),
        )
        replay_transitions = _validate_schema_v5_replay(
            replay_transitions,
            action_count=int(current_model_config["rl_num_actions"]),
            require_behavior_log_prob=bool(
                current_rl_config.get("use_importance_sampling", True)
            ),
            episodic_latent=bool(current_rl_config["episodic_latent"]),
            max_edits=int(current_rl_config["max_edits"]),
            expected_latent_shape=expected_replay_latent_shape,
            expected_latent_dtype=next(model.parameters()).dtype,
            seq_len=int(current_model_config["seq_len"]),
            vocab_size=int(current_model_config["vocab_size"]),
            num_puzzle_identifiers=int(
                current_model_config["num_puzzle_identifiers"]
            ),
        )
    elif not isinstance(replay_transitions, list):
        raise RuntimeError("Checkpoint replay transitions must be a list.")
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

    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        collection_state_for_latent = trainer_state["collection_state"]
        assert expected_replay_latent_shape is not None
        _validate_schema_v5_active_latent(
            collection_state_for_latent,
            episodic_latent=bool(current_rl_config["episodic_latent"]),
            expected_latent_shape=expected_replay_latent_shape,
            expected_latent_dtype=next(model.parameters()).dtype,
        )

    # Validate nested mutable state on isolated shells before restoring any live
    # module, optimizer, replay, environment, collector, or RNG state.
    environment_probe = copy.copy(trainer.env)
    environment_probe.load_checkpoint_state(trainer_state["environment_state"])
    replay_probe = ReplayBuffer(capacity=int(checkpoint["replay_capacity"]))
    for transition in replay_transitions:
        replay_probe.add(transition)
    trainer_probe = copy.copy(trainer)
    trainer_probe.env = environment_probe
    trainer_probe.replay = replay_probe
    trainer_probe._next_episode_id = saved_next_episode_id
    trainer_probe.load_collection_checkpoint_state(trainer_state["collection_state"])
    if replay_transitions:
        replay_tail_episode_id = replay_transitions[-1].episode_id
        active_episode_state = trainer_state["collection_state"].get(
            "active_episode"
        )
        active_timestep = (
            int(active_episode_state.get("timestep", -1))
            if isinstance(active_episode_state, dict)
            else None
        )
        expected_next_episode_id = (
            replay_tail_episode_id
            if active_timestep is not None and active_timestep > 0
            else replay_tail_episode_id + 1
        )
        if saved_next_episode_id != expected_next_episode_id:
            raise RuntimeError(
                "Checkpoint next episode ID disagrees with the replay tail."
            )

    module_state_fields = {
        "model": "model_state_dict",
        "policy_model_old": "policy_model_old_state_dict",
        "policy_model_candidate": "policy_model_candidate_state_dict",
        "target_model": "target_model_state_dict",
    }
    modules = _checkpoint_modules(model, trainer)
    if set(modules) != set(module_state_fields):
        raise RuntimeError("Checkpoint module inventory is incomplete.")
    saved_modes = checkpoint.get("module_training_modes")
    saved_gradients = checkpoint.get("parameter_gradients")
    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        if not isinstance(saved_modes, dict) or set(saved_modes) != set(modules):
            raise RuntimeError("Schema-v5 module training-mode inventory mismatch.")
        if not all(isinstance(value, bool) for value in saved_modes.values()):
            raise RuntimeError("Schema-v5 module training modes must be boolean.")
        if not isinstance(saved_gradients, dict) or set(saved_gradients) != set(
            modules
        ):
            raise RuntimeError("Schema-v5 parameter-gradient inventory mismatch.")

    # Module and optimizer state dictionaries are parsed on isolated copies.
    # A corrupt late field therefore cannot leave the live run half-restored.
    for name, module in modules.items():
        try:
            module_probe = copy.deepcopy(module)
            module_probe.load_state_dict(
                checkpoint[module_state_fields[name]],
                strict=True,
            )
            if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
                assert isinstance(saved_gradients, dict)
                _restore_parameter_gradients(
                    module_probe,
                    saved_gradients[name],
                    label=name,
                )
            del module_probe
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint {name} state failed preflight; state was not restored."
            ) from exc

    optimizer_state_pairs = (
        (trainer.value_opt, "value_optimizer_state_dict"),
        (trainer.policy_opt, "policy_optimizer_state_dict"),
        (
            getattr(trainer, "old_policy_distill_opt", None),
            "old_policy_distill_optimizer_state_dict",
        ),
        (puzzle_emb_optimizer, "puzzle_emb_optimizer_state_dict"),
    )
    for optimizer, field in optimizer_state_pairs:
        if optimizer is not None:
            try:
                optimizer_probe = copy.deepcopy(optimizer)
                optimizer_probe.load_state_dict(checkpoint[field])
                del optimizer_probe
            except Exception as exc:
                raise RuntimeError(
                    f"Checkpoint {field} failed preflight; state was not restored."
                ) from exc
    scheduler_state_pairs = (
        (getattr(trainer, "value_scheduler", None), "value_scheduler_state_dict"),
        (getattr(trainer, "policy_scheduler", None), "policy_scheduler_state_dict"),
    )
    for scheduler, field in scheduler_state_pairs:
        if scheduler is not None:
            try:
                scheduler_probe = copy.deepcopy(scheduler)
                scheduler_probe.load_state_dict(checkpoint[field])
                del scheduler_probe
            except Exception as exc:
                raise RuntimeError(
                    f"Checkpoint {field} failed preflight; state was not restored."
                ) from exc

    current_rng_state = _capture_rng_state()
    try:
        try:
            _restore_rng_state(rng_state)
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint RNG state failed preflight; state was not restored."
            ) from exc
    finally:
        _restore_rng_state(current_rng_state)

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

    if schema_version == FIXED_BASE_CHECKPOINT_SCHEMA_VERSION:
        assert isinstance(saved_modes, dict)
        assert isinstance(saved_gradients, dict)
        for name, module in modules.items():
            module.train(saved_modes[name])
            _restore_parameter_gradients(
                module,
                saved_gradients[name],
                label=name,
            )

    trainer._next_episode_id = saved_next_episode_id
    trainer._train_step_count = saved_train_steps
    trainer._env_step_count = saved_env_steps
    trainer._value_optimizer_step_count = saved_value_optimizer_steps
    trainer._policy_optimizer_step_count = saved_policy_optimizer_steps
    trainer._distill_optimizer_step_count = saved_distill_optimizer_steps
    trainer._puzzle_optimizer_step_count = saved_puzzle_optimizer_steps
    trainer._kl_coef = saved_kl_coef
    if saved_term_stats is not None:
        trainer.term_stats = saved_term_stats
    trainer._debug_episode_lengths = saved_debug_episode_lengths
    trainer._debug_episode_returns = saved_debug_episode_returns
    trainer._debug_stop_probs = saved_debug_stop_probs
    trainer._debug_score_changes = saved_debug_score_changes
    trainer._drift_values = saved_drift_values
    trainer._plan_changes = saved_plan_changes
    trainer._value_of_memory = saved_value_of_memory
    trainer._opnorm_clamp_warned = saved_opnorm_warning

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
        f"outer_step={start_step}, "
        f"value_optimizer_steps={trainer._value_optimizer_step_count}, "
        f"policy_optimizer_steps={trainer._policy_optimizer_step_count}"
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
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable registered run identifier (required for fixed_base_exact).",
    )
    parser.add_argument(
        "--producer-repo-root",
        type=str,
        default=None,
        help=(
            "Git lookup root used to verify a clean producer commit; the path is "
            "never stored in checkpoint metadata."
        ),
    )
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

    confirmatory_fixed_base = (
        selected_baseline is None
        and rl_cfg.training_protocol == "fixed_base_exact"
    )
    producer_repo_root = args.producer_repo_root or os.getcwd()
    initial_producer_identity: Optional[Dict[str, Any]] = None
    if confirmatory_fixed_base:
        if args.seed is None:
            raise RuntimeError(
                "fixed_base_exact requires an explicit --seed for schema-v5 identity."
            )
        if args.run_id is None:
            raise RuntimeError(
                "fixed_base_exact requires an explicit --run-id for schema-v5 identity."
            )
        if args.imitation_pretrain:
            raise RuntimeError(
                "Schema-v5 fixed-base runs currently reject imitation pretraining "
                "because its completion phase is not resumable."
            )
        if args.wandb:
            raise RuntimeError(
                "Schema-v5 fixed-base runs disable WandB because its process-level "
                "RNG isolation has not been established."
            )
        if args.load_checkpoint is not None and args.resume_checkpoint is not None:
            raise RuntimeError(
                "Use either --load-checkpoint or --resume-checkpoint, not both."
            )
        if not args.dataset_paths:
            raise RuntimeError(
                "Schema-v5 fixed-base runs require materialized disjoint train and "
                "evaluation splits; the dummy dataset is smoke-only."
            )
        try:
            initial_producer_identity = discover_clean_git_source(
                producer_repo_root
            )
            _verify_producer_source_matches_runtime(producer_repo_root)
        except RunIdentityError as exc:
            raise RuntimeError(
                f"fixed_base_exact requires a clean producer Git tree: {exc}"
            ) from exc

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
        train_input_hashes = dataset_input_sha256s(dataset)
        eval_input_hashes = dataset_input_sha256s(eval_dataset)
        overlap = set(train_input_hashes).intersection(eval_input_hashes)
        if overlap:
            raise RuntimeError(
                "Training and evaluation pools overlap; refusing an in-sample "
                f"evaluation ({len(overlap)} duplicate records)."
            )
        if confirmatory_fixed_base:
            if len(set(train_input_hashes)) != len(train_input_hashes):
                raise RuntimeError(
                    "Schema-v5 training pool contains duplicate inputs."
                )
            if len(set(eval_input_hashes)) != len(eval_input_hashes):
                raise RuntimeError(
                    "Schema-v5 evaluation pool contains duplicate inputs."
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
        eval_pool_hash = dataset_pool_sha256(eval_dataset, len(eval_dataset))
        print(
            f"[DATASET] train_split={args.train_split} train_samples={len(dataset)} "
            f"train_pool_sha256={train_pool_hash}"
        )
        print(
            f"[DATASET] eval_split={args.eval_split} "
            f"eval_pool_samples={len(eval_dataset)} "
            f"eval_episodes_per_checkpoint={rl_cfg.eval_num_episodes} "
            f"eval_pool_sha256={eval_pool_hash} "
            f"eval_puzzle_id_offset={train_identifier_count}"
        )
        source_build_metadata = dataset_source_build_metadata(args.dataset_paths)
        if confirmatory_fixed_base:
            incomplete_sources = [
                source["source_name"]
                for source in source_build_metadata
                if source["builder_name"] == "unrecorded"
                or source["builder_version"] == "unrecorded"
                or source["generation_seed"] is None
                or source["build_config_sha256"] is None
            ]
            if incomplete_sources:
                raise RuntimeError(
                    "Schema-v5 dataset sources lack builder/version/seed/config "
                    f"provenance: {incomplete_sources}"
                )
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
        provenance_eval_count = len(eval_dataset)
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
            "train_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                dataset_puzzle_identifier_sha256s(dataset)
            ),
            "eval_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                dataset_puzzle_identifier_sha256s(eval_dataset)
            ),
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
            "train_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                dataset_puzzle_identifier_sha256s(dataset)
            ),
            "eval_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                dataset_puzzle_identifier_sha256s(eval_dataset)
            ),
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

    initialization_kind = "random"
    initialization_artifact_sha256: Optional[str] = None
    checkpoint_lineage = build_checkpoint_lineage(
        parent_checkpoint_sha256=None,
        parent_checkpoint_step=None,
        parent_environment_steps=None,
    )
    if confirmatory_fixed_base and args.resume_checkpoint is not None:
        saved_identity, checkpoint_lineage = _load_schema_v5_resume_metadata(
            args.resume_checkpoint
        )
        initialization_kind = saved_identity["initialization"]["kind"]
        initialization_artifact_sha256 = saved_identity["initialization"][
            "artifact_sha256"
        ]
    elif confirmatory_fixed_base and args.load_checkpoint is not None:
        initialization_kind = "weights_checkpoint"
        try:
            initialization_artifact_sha256 = file_sha256(args.load_checkpoint)
        except RunIdentityError as exc:
            raise RuntimeError(
                f"Cannot bind initialization checkpoint identity: {exc}"
            ) from exc

    # === Load pretrained checkpoint if provided ===
    if args.load_checkpoint is not None:
        load_checkpoint(
            model,
            args.load_checkpoint,
            device=str(device),
            strict=False,
            expected_sha256=(
                initialization_artifact_sha256
                if confirmatory_fixed_base
                else None
            ),
        )

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
    fixed_base_exact = (
        isinstance(trainer, UPITrmTrainer)
        and getattr(rl_cfg, "training_protocol", "legacy") == "fixed_base_exact"
    )
    if fixed_base_exact and puzzle_emb_ndim > 0:
        print(
            "[INFO] Puzzle embeddings are frozen under fixed_base_exact; "
            "no sparse optimizer is attached."
        )
    elif puzzle_emb_ndim > 0 and hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb"):
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

    run_identity: Optional[Dict[str, Any]] = None
    if confirmatory_fixed_base:
        assert args.seed is not None
        assert args.run_id is not None
        effective_config = _fixed_base_effective_config(
            args=args,
            rl_config=_config_dict(rl_cfg),
            model_config=_config_dict(model.config),
            execution_device=_canonical_device(device),
            train_record_count=len(dataset),
            eval_record_count=len(eval_dataset),
        )
        try:
            producer_before_identity = discover_clean_git_source(
                producer_repo_root
            )
            _verify_producer_source_matches_runtime(producer_repo_root)
            producer_after_identity = discover_clean_git_source(
                producer_repo_root
            )
            if (
                producer_before_identity != initial_producer_identity
                or producer_after_identity != initial_producer_identity
            ):
                raise RuntimeError(
                    "Producer Git identity changed while constructing the run."
                )
            run_identity = build_run_identity(
                run_id=args.run_id,
                training_seed=args.seed,
                git_lookup_root=producer_repo_root,
                effective_config=effective_config,
                dataset_provenance=dataset_provenance,
                initialization_kind=initialization_kind,
                initialization_artifact_sha256=initialization_artifact_sha256,
            )
        except RunIdentityError as exc:
            raise RuntimeError(f"Cannot establish schema-v5 run identity: {exc}") from exc
        if run_identity["producer"] != initial_producer_identity:
            raise RuntimeError(
                "Producer Git identity changed while constructing the run."
            )
        print(
            "[IDENTITY] "
            f"run_id={run_identity['run_id']} "
            f"commit={run_identity['producer']['git_commit']} "
            f"config_sha256={run_identity['effective_config_sha256']}"
        )
    
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
            expected_run_identity=run_identity,
            expected_checkpoint_sha256=(
                checkpoint_lineage["parent_checkpoint_sha256"]
                if confirmatory_fixed_base
                else None
            ),
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

    def save_training_checkpoint(progress_step: int) -> str:
        nonlocal checkpoint_lineage
        if checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory is not configured.")
        lineage_for_save: Optional[Dict[str, Any]] = None
        if confirmatory_fixed_base:
            assert initial_producer_identity is not None
            try:
                producer_identity_before_hash = discover_clean_git_source(
                    producer_repo_root
                )
                _verify_producer_source_matches_runtime(producer_repo_root)
                producer_identity_after_hash = discover_clean_git_source(
                    producer_repo_root
                )
            except RunIdentityError as exc:
                raise RuntimeError(
                    f"Cannot revalidate producer source before checkpoint: {exc}"
                ) from exc
            if (
                producer_identity_before_hash != initial_producer_identity
                or producer_identity_after_hash != initial_producer_identity
            ):
                raise RuntimeError(
                    "Producer Git identity changed before checkpoint publication."
                )
            lineage_for_save = checkpoint_lineage
        saved_path = save_checkpoint(
            model,
            trainer,
            progress_step,
            checkpoint_dir,
            puzzle_emb_optimizer,
            rl_cfg,
            dataset_provenance,
            run_identity,
            lineage_for_save,
        )
        if confirmatory_fixed_base:
            try:
                checkpoint_lineage = build_checkpoint_lineage(
                    parent_checkpoint_sha256=file_sha256(saved_path),
                    parent_checkpoint_step=progress_step,
                    parent_environment_steps=trainer.get_env_step_count(),
                )
            except RunIdentityError as exc:
                raise RuntimeError(
                    "Published checkpoint cannot be linked into the run lineage."
                ) from exc
        return saved_path

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
            print("✓ Configuration matches the frozen proposal protocol")
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
        print(
            "\nFixed-base proposal protocol exact: "
            f"{rl_cfg.is_fixed_base_proposal_exact()}"
        )
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
            "fixed_base_proposal_exact": rl_cfg.is_fixed_base_proposal_exact(),
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
                save_training_checkpoint(outer_step)
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
                save_training_checkpoint(progress_step)
                last_saved_progress_step = progress_step
                while next_save_env is not None and current_env_step >= next_save_env:
                    next_save_env += save_env_interval

        final_progress_step = trainer.get_env_step_count()

    # === Save final checkpoint ===
    if checkpointing_enabled and checkpoint_dir is not None and last_saved_progress_step != final_progress_step:
        save_training_checkpoint(final_progress_step)
    
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
