
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # type: ignore
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.norec_encoder import NoRecursionEncoder, NoRecEncoderConfig
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.upi_trm_trainer import UPITrmTrainer
from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.algos.a2c import A2CTrainer, A2CConfig
from utils.seeding import set_global_seed


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


def save_checkpoint(
    model: nn.Module,
    trainer: "UPITrmTrainer",
    step: int,
    checkpoint_dir: str,
    puzzle_emb_optimizer: Optional[torch.optim.Optimizer] = None,
) -> str:
    """
    Save full training state for resumable RL training.
    
    Args:
        model: The TRM model
        trainer: UPITrmTrainer instance (for optimizers)
        step: Current training step
        checkpoint_dir: Directory to save checkpoints
        puzzle_emb_optimizer: Optional optimizer for puzzle embeddings
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    # Save optimizer states - different trainers have different optimizer structures
    if hasattr(trainer, 'value_opt') and hasattr(trainer, 'policy_opt'):
        # UPITrmTrainer has separate value and policy optimizers
        checkpoint["value_optimizer_state_dict"] = trainer.value_opt.state_dict()
        checkpoint["policy_optimizer_state_dict"] = trainer.policy_opt.state_dict()
    elif hasattr(trainer, 'optimizer'):
        # PPO/A2C have a single combined optimizer
        checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

    if puzzle_emb_optimizer is not None:
        checkpoint["puzzle_emb_optimizer_state_dict"] = puzzle_emb_optimizer.state_dict()

    # Save replay buffer size (not contents, too large) if trainer has replay buffer
    if hasattr(trainer, 'replay'):
        checkpoint["replay_buffer_size"] = len(trainer.replay)
    
    path = os.path.join(checkpoint_dir, f"rl_checkpoint_step_{step}.pt")
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path}")
    
    # Also save just the model weights for easy loading
    model_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
    torch.save(model.state_dict(), model_path)
    
    return path


def resume_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    trainer: "UPITrmTrainer",
    device: str,
    puzzle_emb_optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    Resume RL training from a saved checkpoint.
    
    Returns:
        Starting step number
    """
    print(f"[Checkpoint] Resuming from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer.value_opt.load_state_dict(checkpoint["value_optimizer_state_dict"])
    trainer.policy_opt.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    
    if puzzle_emb_optimizer is not None and "puzzle_emb_optimizer_state_dict" in checkpoint:
        puzzle_emb_optimizer.load_state_dict(checkpoint["puzzle_emb_optimizer_state_dict"])
    
    start_step = checkpoint["step"]
    print(f"[Checkpoint] Resumed from step {start_step}")
    
    return start_step


def _to_plan_tensor(value):
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def dummy_checker(x, y) -> float:
    """
    Placeholder checker: reward is negative L1 distance between plan and inputs.
    """
    target = x["inputs"]
    plan = _to_plan_tensor(y)
    target = target.to(torch.float32)
    plan = plan.to(torch.float32)
    return float(-(target - plan).abs().mean().item())


def sudoku_checker(x, y) -> float:
    """
    Returns a scaled score for how many cells match the Sudoku solution.
    Scaled to [0, 10] range to provide meaningful reward signal while keeping values bounded.
    Falls back to dummy_checker if no solution is attached to the sample.
    """

    solution = x.get("solution")
    if solution is None:
        return dummy_checker(x, y)

    plan = _to_plan_tensor(y).to(torch.long)
    solution_tensor = _to_plan_tensor(solution).to(torch.long)
    if plan.shape != solution_tensor.shape:
        solution_tensor = solution_tensor.view_as(plan)
    matches = (plan == solution_tensor).to(torch.float32)
    return float(matches.mean().item() * 10.0)  # Returns 0-10 range for meaningful rewards


class DummyPuzzleDataset:
    """
    DummyPuzzleDataset is only used when no real Sudoku dataset is found.
    It provides a tiny synthetic environment for smoke-testing the RL loop.
    Tiny in-memory dataset suitable for smoke tests of the RL loop.
    """

    def __init__(self, num_instances: int = 32, seq_len: int = 16, vocab_size: int = 32):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_instances

        self.samples: List[dict] = []
        for idx in range(num_instances):
            inputs = torch.randint(low=0, high=vocab_size, size=(seq_len,), dtype=torch.long)
            puzzle_identifier = torch.tensor(idx, dtype=torch.long)
            sample = {
                "inputs": inputs,  # tokenized Sudoku grid encoded like the real dataset
                "puzzle_identifiers": puzzle_identifier,
                "initial_plan": torch.zeros_like(inputs),
                "solution": inputs.clone(),  # dummy solution identical to inputs
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class OfflinePuzzleDataset:
    """
    Wraps a finite list of samples gathered from PuzzleDataset to provide __len__/__getitem__.
    """

    def __init__(self, samples: List[dict], seq_len: int, vocab_size: int, num_identifiers: int):
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_identifiers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_dataset_from_paths(
    dataset_paths: Optional[List[str]],
    pool_size: int,
) -> Tuple[object, int, int, int]:
    """
    Attempt to load a handful of samples from the supervised PuzzleDataset to bootstrap RL.
    Falls back to DummyPuzzleDataset if paths are missing or loading fails.
    """

    if dataset_paths:
        try:
            ds_cfg = PuzzleDatasetConfig(
                seed=0,
                dataset_paths=dataset_paths,
                global_batch_size=pool_size,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=0,
                num_replicas=1,
            )
            iterable = PuzzleDataset(ds_cfg, split="train")
            samples: List[dict] = []
            for _set_name, batch, _ in iterable:
                batch_inputs = batch["inputs"]
                batch_ids = batch["puzzle_identifiers"]
                batch_labels = batch.get("labels")
                batch_size = batch_inputs.shape[0]
                for i in range(batch_size):
                    inputs = batch_inputs[i].clone()
                    puzzle_id = batch_ids[i].clone()
                    solution = batch_labels[i].clone() if batch_labels is not None else None
                    initial_plan = inputs.clone()
                    samples.append(
                        {
                            "inputs": inputs,
                            "puzzle_identifiers": puzzle_id,
                            "initial_plan": initial_plan,
                            **({"solution": solution} if solution is not None else {}),
                        }
                    )
                    if len(samples) >= pool_size:
                        break
                if len(samples) >= pool_size:
                    break
            if samples:
                num_identifiers = int(torch.stack([s["puzzle_identifiers"] for s in samples]).max().item() + 1)
                dataset = OfflinePuzzleDataset(
                    samples=samples,
                    seq_len=samples[0]["inputs"].shape[-1],
                    vocab_size=iterable.metadata.vocab_size,
                    num_identifiers=max(num_identifiers, iterable.metadata.num_puzzle_identifiers),
                )
                return dataset, dataset.seq_len, dataset.vocab_size, dataset.num_identifiers
        except Exception as exc:  # pragma: no cover - best-effort bootstrap
            print(f"[upi_trm_train] Falling back to dummy dataset (reason: {exc})")

    dummy = DummyPuzzleDataset()
    return dummy, dummy.seq_len, dummy.vocab_size, dummy.num_identifiers


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveReasoningModel with plan-space RL.")
    parser.add_argument("--dataset-paths", nargs="+", default=None, help="Optional list of supervised dataset directories.")
    parser.add_argument("--train-steps", type=int, default=200, help="Number of outer RL steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for TD updates.")
    parser.add_argument("--rollouts-per-step", type=int, default=1, help="Episodes collected before each optimization step.")
    parser.add_argument("--max-edits", type=int, default=8, help="Maximum edits per episode.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in train steps.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval in train steps.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of episodes per evaluation call.")
    parser.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bar (disabled by default).")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument("--debug-checks", action="store_true", help="Enable additional debug assertions/prints.")
    # Baseline algorithm selection
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["ppo", "a2c", None],
        help="Use baseline algorithm instead of UPI-TRM. Options: ppo, a2c. Default: None (use UPI-TRM).",
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
        default=None,
        help="Optional path to YAML config overriding RLConfig defaults.",
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
        use_tqdm=args.tqdm,
        debug_checks=args.debug_checks,
    )

    if args.config is not None:
        import yaml

        # The override config is expected to be a flat dict with keys matching RLConfig
        # fields, e.g. {"gamma": 0.95, "K": 3}. Nested structures (e.g. trainer: rl: ...)
        # are not currently supported.
        with open(args.config, "r") as f:
            override = yaml.safe_load(f) or {}
        # Pydantic v2 uses model_dump(), v1 uses dict()
        base_dict = rl_cfg.model_dump() if hasattr(rl_cfg, "model_dump") else rl_cfg.dict()
        rl_cfg = RLConfig(**{**base_dict, **override})
    
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

    dataset, seq_len, vocab_size, num_identifiers = build_dataset_from_paths(
        dataset_paths=args.dataset_paths,
        pool_size=max(rl_cfg.batch_size, 8),
    )

    checker_fn = sudoku_checker
    if len(dataset) == 0:
        checker_fn = dummy_checker
    else:
        sample = dataset[0]
        if not (isinstance(sample, dict) and "solution" in sample):
            checker_fn = dummy_checker
    is_sudoku_checker = checker_fn is sudoku_checker

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
    )
    
    # Optionally use task-specific configuration
    task_config = None
    try:
        from rl.task_config import get_task_config
        task_name = getattr(rl_cfg, "task_name", "sudoku")
        if is_sudoku_checker:
            task_config = get_task_config("sudoku")
        elif checker_fn is dummy_checker:
            task_config = get_task_config("dummy")
    except ImportError:
        pass  # task_config module not available
    
    env = PlanEditEnv(dataset=dataset, checker=checker_fn, config=env_cfg, task_config=task_config)
    
    # Log resolved training parameters
    print(f"[INFO] Training for {rl_cfg.num_train_steps} steps "
          f"(gamma={rl_cfg.gamma:.3f}, K={rl_cfg.K}, inner_unroll_n={rl_cfg.inner_unroll_n})")
    print(f"[INFO] Terminal rewards: solve={env_cfg.solve_terminal_reward:.3f}, "
          f"fail={env_cfg.fail_terminal_reward:.3f}")

    num_edit_actions = seq_len * vocab_size
    rl_num_actions = num_edit_actions + 1  # STOP action appended at the end
    env.set_stop_action_id(stop_id=rl_num_actions - 1)

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

    # === Trainer Selection ===
    # Support for different algorithms: UPI-TRM (default), PPO, or A2C
    if args.baseline is None:
        # Default: UPI-TRM (theory-aligned algorithm)
        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=device)
        print(f"[INFO] Using UPI-TRM algorithm (K={rl_cfg.K}, inner_n={rl_cfg.inner_unroll_n})")
    elif args.baseline == "ppo":
        # PPO baseline
        ppo_cfg = PPOConfig(
            clip_eps=0.2,
            vf_coef=0.5,
            entropy_coef=rl_cfg.entropy_coef,
            max_grad_norm=0.5,
            num_steps=128,  # Standard PPO rollout length
            num_epochs=4,
            num_minibatches=4,
            gamma=rl_cfg.gamma,
            gae_lambda=0.95,
            normalize_advantages=True,
            policy_lr=rl_cfg.policy_lr,
            value_lr=rl_cfg.value_lr,
            inner_unroll_n=rl_cfg.inner_unroll_n,
            log_interval=rl_cfg.log_interval,
            eval_interval=rl_cfg.eval_interval,
            num_train_steps=rl_cfg.num_train_steps,
        )
        trainer = PPOTrainer(model=model, env=env, config=ppo_cfg, device=device)
        print(f"[INFO] Using PPO baseline (clip_eps={ppo_cfg.clip_eps}, epochs={ppo_cfg.num_epochs})")
    elif args.baseline == "a2c":
        # A2C baseline
        a2c_cfg = A2CConfig(
            vf_coef=0.5,
            entropy_coef=rl_cfg.entropy_coef,
            max_grad_norm=0.5,
            num_steps=5,  # Standard A2C uses small rollouts
            gamma=rl_cfg.gamma,
            use_gae=True,
            gae_lambda=0.95,
            lr=rl_cfg.policy_lr,
            inner_unroll_n=rl_cfg.inner_unroll_n,
            log_interval=rl_cfg.log_interval,
            eval_interval=rl_cfg.eval_interval,
            num_train_steps=rl_cfg.num_train_steps,
        )
        trainer = A2CTrainer(model=model, env=env, config=a2c_cfg, device=device)
        print(f"[INFO] Using A2C baseline (num_steps={a2c_cfg.num_steps})")
    else:
        raise ValueError(f"Unknown baseline algorithm: {args.baseline}")
    
    # === Setup puzzle embedding optimizer (separate from main optimizer) ===
    puzzle_emb_optimizer = None
    if puzzle_emb_ndim > 0 and hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb"):
        # Use SignSGD for sparse puzzle embeddings (same as pretrain.py)
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            model.inner.puzzle_emb.buffers(),
            lr=args.puzzle_emb_lr,
            weight_decay=args.puzzle_emb_weight_decay,
            world_size=1,  # Single GPU for now
        )
        print(f"[INFO] Puzzle embedding optimizer: SignSGD (lr={args.puzzle_emb_lr})")
    
    # === Resume from RL checkpoint if provided ===
    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = resume_from_checkpoint(
            args.resume_checkpoint, model, trainer, str(device), puzzle_emb_optimizer
        )
    
    # === Setup checkpoint directory ===
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.save_interval > 0:
        dataset_name = "dummy"
        if args.dataset_paths:
            dataset_name = os.path.basename(args.dataset_paths[0])
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
    if args.baseline is None:
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
        print(f"  Contraction enforced (L_z < 1): {'✓' if validation['contraction_enforced'] else '✗'}")
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
        print(f"\n[INFO] Using {args.baseline.upper()} baseline algorithm")
        print(f"[INFO] Backbone: {args.backbone}")
        print(f"[INFO] This is a comparison baseline (no theory-exact features)")
        print()

    # Log additional configuration settings
    stop_mode = getattr(rl_cfg, "stop_action_mode", "noop")
    print(f"[INFO] STOP action mode: {stop_mode}")
    if args.baseline is None:  # Only show theory-specific info for UPI-TRM
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
        run_name = args.wandb_run_name
        if run_name is None:
            dataset_name = "dummy"
            if args.dataset_paths:
                dataset_name = os.path.basename(args.dataset_paths[0])
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
                dataset=dataset,
                checker=checker_fn,
            )
            print(f"[Post-imitation] eval_success_rate={eval_metrics['success_rate']:.3f} "
                  f"eval_mean_score={eval_metrics['mean_score']:.3f}")
            print()

    # === Training loop with puzzle embedding updates and checkpointing ===
    total_steps = rl_cfg.num_train_steps
    remaining_steps = total_steps - start_step
    
    if rl_cfg.use_tqdm and trange is not None:
        step_iter = trange(remaining_steps, desc="UPI-TRM RL training", initial=start_step, total=total_steps)
    else:
        step_iter = range(remaining_steps)

    for local_step in step_iter:
        step = start_step + local_step
        metrics = trainer.train_step()
        
        # === Step puzzle embedding optimizer ===
        if puzzle_emb_optimizer is not None:
            puzzle_emb_optimizer.step()
            puzzle_emb_optimizer.zero_grad()
        
        if (step + 1) % rl_cfg.log_interval == 0:
            msg = (
                f"[step {step+1:05d}] value_loss={metrics['loss_value']:.6f} "
                f"policy_loss={metrics['loss_policy']:.6f}"
            )
            if hasattr(step_iter, "write"):
                step_iter.write(msg)
            else:
                print(msg)
            
            # Log value target stats for debugging scale issues
            if "target_mean" in metrics:
                clip_str = f" clip={metrics['value_clip']:.1f}" if "value_clip" in metrics else ""
                target_msg = (
                    f"[step {step+1:05d}] VALUE_DEBUG: "
                    f"target(mean={metrics['target_mean']:.2f} std={metrics['target_std']:.2f} "
                    f"min={metrics['target_min']:.2f} max={metrics['target_max']:.2f}) "
                    f"V(s)(mean={metrics['value_mean']:.2f} std={metrics['value_std']:.2f}) "
                    f"reward(mean={metrics['reward_mean']:.3f} std={metrics['reward_std']:.3f}){clip_str}"
                )
                if hasattr(step_iter, "write"):
                    step_iter.write(target_msg)
                else:
                    print(target_msg)
            
            # === WandB: Log training metrics ===
            if use_wandb:
                wandb_metrics = {
                    "train/loss_value": metrics["loss_value"],
                    "train/loss_policy": metrics["loss_policy"],
                    "train/term_stop": metrics.get("term_stop", 0),
                    "train/term_solved": metrics.get("term_solved", 0),
                    "train/term_budget": metrics.get("term_budget", 0),
                }
                # Add learning rates if available
                if "value_lr" in metrics:
                    wandb_metrics["train/lr_value"] = metrics["value_lr"]
                if "policy_lr" in metrics:
                    wandb_metrics["train/lr_policy"] = metrics["policy_lr"]
                # Add KL metrics if using trust region
                if "policy_kl" in metrics:
                    wandb_metrics["train/policy_kl"] = metrics["policy_kl"]
                if "kl_coef" in metrics:
                    wandb_metrics["train/kl_coef"] = metrics["kl_coef"]
                # Add theory metrics if tracked
                for key in ["hat_Cz", "hat_Lz", "hat_Lv", "unrolling_term",
                            "bellman_residual_mean", "bellman_residual_max",
                            "drift_mean", "drift_max", "plan_change_mean"]:
                    if key in metrics:
                        wandb_metrics[f"theory/{key}"] = metrics[key]
                # Add debug metrics if present (value target stats and advantage stats)
                for key in ["value_mean", "value_std", "adv_mean", "adv_std",
                            "target_mean", "target_std", "target_min", "target_max",
                            "reward_mean", "reward_std"]:
                    if key in metrics:
                        wandb_metrics[f"debug/{key}"] = metrics[key]
                wandb.log(wandb_metrics, step=step + 1)

        if (step + 1) % rl_cfg.eval_interval == 0:
            # All trainers (UPI-TRM, PPO, A2C) now have evaluate_policy_metrics
            # Use try/except to handle old cached binaries that may not have the method
            eval_metrics = None
            try:
                eval_metrics = trainer.evaluate_policy_metrics(
                    env_cfg=env_cfg,
                    dataset=dataset,
                    checker=checker_fn,
                )
            except AttributeError:
                pass  # Old cached binary without evaluate_policy_metrics
            if eval_metrics is not None:
                eval_policy_mode = eval_metrics.get("eval_policy_mode", "unknown")
                solved_count = eval_metrics.get("solved_count", 0)
                total_episodes = eval_metrics.get("total_episodes", 0)
                score_min = eval_metrics.get("score_min", 0.0)
                score_max = eval_metrics.get("score_max", 0.0)
                max_possible = eval_metrics.get("max_possible_score")
                initial_mean = eval_metrics.get("initial_score_mean", 0.0)

                eval_msg = (
                    f"[step {step+1:05d}] "
                    f"eval_success_rate={eval_metrics['success_rate']:.3f} "
                    f"eval_mean_score={eval_metrics['mean_score']:.3f} "
                    f"eval_policy_mode={eval_policy_mode} "
                    f"[solved={solved_count}/{total_episodes}, "
                    f"score_range={score_min:.2f}-{score_max:.2f}"
                    f"{f'/{max_possible:.1f}' if max_possible else ''}, "
                    f"initial={initial_mean:.2f}]"
                )
                if hasattr(step_iter, "write"):
                    step_iter.write(eval_msg)
                else:
                    print(eval_msg)

                # === WandB: Log evaluation metrics ===
                if use_wandb:
                    wandb_eval = {
                        "eval/success_rate": eval_metrics["success_rate"],
                        "eval/mean_score": eval_metrics["mean_score"],
                        "eval/solved_count": solved_count,
                        "eval/score_min": score_min,
                        "eval/score_max": score_max,
                        "eval/initial_score_mean": initial_mean,
                    }
                    if max_possible is not None:
                        wandb_eval["eval/max_possible_score"] = max_possible
                    wandb.log(wandb_eval, step=step + 1)
            else:
                # Baselines: just print a message that eval is not available
                print(f"[step {step+1:05d}] (eval not available for baseline trainer)")

            # Print debug stats every eval interval (if trainer supports it)
            debug_stats = None
            if hasattr(trainer, 'get_debug_stats'):
                debug_stats = trainer.get_debug_stats()
            if debug_stats:
                debug_msg = (
                    f"[step {step+1:05d}] DEBUG: "
                    f"ep_len={debug_stats.get('avg_episode_length', 0):.1f} "
                    f"ep_ret={debug_stats.get('avg_episode_return', 0):.3f} "
                    f"stop_prob={debug_stats.get('avg_stop_prob', 0):.3f} "
                    f"score_chg={debug_stats.get('avg_score_change', 0):.4f}"
                )
                if hasattr(step_iter, "write"):
                    step_iter.write(debug_msg)
                else:
                    print(debug_msg)

                # === WandB: Log debug stats ===
                if use_wandb:
                    wandb_debug = {}
                    for key, value in debug_stats.items():
                        wandb_debug[f"debug/{key}"] = value
                    wandb.log(wandb_debug, step=step + 1)

            if hasattr(trainer, 'clear_debug_stats'):
                trainer.clear_debug_stats()
        
        # === Save checkpoint periodically ===
        if args.save_interval > 0 and checkpoint_dir is not None and (step + 1) % args.save_interval == 0:
            save_checkpoint(model, trainer, step + 1, checkpoint_dir, puzzle_emb_optimizer)
    
    # === Save final checkpoint ===
    if args.save_interval > 0 and checkpoint_dir is not None:
        save_checkpoint(model, trainer, total_steps, checkpoint_dir, puzzle_emb_optimizer)
    
    # === WandB: Finish logging ===
    if use_wandb:
        wandb.finish()
        print("[WandB] Run finished successfully")


if __name__ == "__main__":
    main()

