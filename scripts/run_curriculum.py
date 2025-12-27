#!/usr/bin/env python3
"""
Curriculum training script for 4×4 → 9×9 Sudoku transition.

This script implements a two-stage curriculum where the agent first learns
on easier 4×4 puzzles, then transfers to harder 9×9 puzzles.

Usage:
    python scripts/run_curriculum.py \
        --stage1-dataset data/sudoku-4x4-ultra-easy \
        --stage2-dataset data/sudoku-extreme-1k-aug-1000 \
        --stage1-steps 2000 \
        --total-steps 10000 \
        --config configs/rl_sudoku_shaped_theory_exact.yaml

Key considerations for curriculum transfer:
1. Action space changes (4×4: 80 actions vs 9×9: 892 actions)
2. Policy head must be reinitialized for new action space
3. Backbone weights (TRM encoder) can transfer
4. Value head can partially transfer (same latent dim)
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def modify_config_for_stage(
    config: Dict[str, Any],
    dataset_path: str,
    num_steps: int,
    stage: int,
) -> Dict[str, Any]:
    """Modify config for a curriculum stage."""
    config = copy.deepcopy(config)
    config["num_train_steps"] = num_steps

    # Stage-specific adjustments
    if stage == 1:
        # Stage 1: Easier puzzles, faster exploration
        config["entropy_coef"] = config.get("entropy_coef", 0.01) * 2  # More exploration
        config["mixture_alpha"] = config.get("mixture_alpha", 0.05) * 2  # Faster policy updates
    elif stage == 2:
        # Stage 2: Harder puzzles, more conservative
        config["entropy_coef"] = config.get("entropy_coef", 0.01)  # Standard exploration
        config["mixture_alpha"] = config.get("mixture_alpha", 0.05)  # Standard updates

    return config


def get_action_space_size(dataset_path: str, vocab_size: int = 11) -> int:
    """
    Compute action space size for a dataset.

    For Sudoku:
    - 4×4: seq_len = 16, vocab_size = 5 → 16 * 5 + 1 = 81 actions
    - 9×9: seq_len = 81, vocab_size = 11 → 81 * 11 + 1 = 892 actions
    """
    # Infer from dataset name
    if "4x4" in dataset_path.lower():
        seq_len = 16
        vocab_size = 5  # 0-4 for 4×4 Sudoku
    else:
        seq_len = 81
        vocab_size = 11  # 0-10 for 9×9 Sudoku

    return seq_len * vocab_size + 1  # +1 for STOP action


def reinitialize_policy_head(
    model: torch.nn.Module,
    new_action_dim: int,
) -> None:
    """
    Reinitialize the policy head for a new action space size.

    This is necessary when transferring from 4×4 to 9×9 Sudoku.
    """
    if hasattr(model, "edit_policy"):
        policy_head = model.edit_policy
        old_action_dim = policy_head.output_layer.out_features

        if old_action_dim != new_action_dim:
            print(f"Reinitializing policy head: {old_action_dim} → {new_action_dim} actions")

            # Create new output layer with same hidden dim
            hidden_dim = policy_head.output_layer.in_features
            new_output = torch.nn.Linear(hidden_dim, new_action_dim)

            # Initialize with small weights
            torch.nn.init.orthogonal_(new_output.weight, gain=0.01)
            torch.nn.init.zeros_(new_output.bias)

            policy_head.output_layer = new_output


def run_curriculum_training(
    stage1_dataset: str,
    stage2_dataset: str,
    stage1_steps: int,
    total_steps: int,
    config_path: str,
    checkpoint_dir: Optional[str] = None,
    seed: int = 42,
    wandb_project: Optional[str] = None,
) -> None:
    """
    Run two-stage curriculum training.

    Stage 1: Train on 4×4 puzzles (easier)
    Stage 2: Transfer to 9×9 puzzles (harder)

    Args:
        stage1_dataset: Path to 4×4 Sudoku dataset
        stage2_dataset: Path to 9×9 Sudoku dataset
        stage1_steps: Number of training steps for stage 1
        total_steps: Total training steps (stage2 = total - stage1)
        config_path: Path to YAML config file
        checkpoint_dir: Directory to save checkpoints
        seed: Random seed
        wandb_project: WandB project name
    """
    # Import here to avoid circular imports
    try:
        from upi_trm_train import (
            create_model,
            create_env,
            create_trainer,
            load_dataset,
        )
    except ImportError:
        print("Error: Could not import from upi_trm_train.py")
        print("Make sure you're running from the project root.")
        sys.exit(1)

    # Set seed
    torch.manual_seed(seed)

    # Load base config
    base_config = load_config(config_path)

    # ====================
    # STAGE 1: 4×4 Sudoku
    # ====================
    print("=" * 60)
    print("STAGE 1: Training on 4×4 Sudoku")
    print("=" * 60)

    stage1_config = modify_config_for_stage(base_config, stage1_dataset, stage1_steps, stage=1)
    stage1_actions = get_action_space_size(stage1_dataset)

    print(f"Dataset: {stage1_dataset}")
    print(f"Steps: {stage1_steps}")
    print(f"Action space: {stage1_actions}")

    # Create model, env, trainer for stage 1
    dataset1 = load_dataset(stage1_dataset)
    model1 = create_model(stage1_config, stage1_actions)
    env1 = create_env(dataset1, stage1_config)
    trainer1 = create_trainer(model1, env1, stage1_config)

    # Train stage 1
    for step in range(stage1_steps):
        metrics = trainer1.train_step()

        if step % stage1_config.get("log_interval", 100) == 0:
            print(f"Stage 1 - Step {step}: {metrics}")

    # Save stage 1 checkpoint
    if checkpoint_dir:
        stage1_ckpt = os.path.join(checkpoint_dir, "stage1_checkpoint.pt")
        torch.save({
            "model_state_dict": model1.state_dict(),
            "step": stage1_steps,
            "stage": 1,
        }, stage1_ckpt)
        print(f"Saved stage 1 checkpoint: {stage1_ckpt}")

    # ====================
    # STAGE 2: 9×9 Sudoku
    # ====================
    print("\n" + "=" * 60)
    print("STAGE 2: Transferring to 9×9 Sudoku")
    print("=" * 60)

    stage2_steps = total_steps - stage1_steps
    stage2_config = modify_config_for_stage(base_config, stage2_dataset, stage2_steps, stage=2)
    stage2_actions = get_action_space_size(stage2_dataset)

    print(f"Dataset: {stage2_dataset}")
    print(f"Steps: {stage2_steps}")
    print(f"Action space: {stage2_actions}")

    # Create new model for stage 2 action space
    model2 = create_model(stage2_config, stage2_actions)

    # Transfer weights from stage 1 (backbone only)
    print("\nTransferring backbone weights from stage 1...")
    stage1_state = model1.state_dict()
    stage2_state = model2.state_dict()

    transferred = 0
    for name, param in stage1_state.items():
        # Skip policy head (different action space)
        if "edit_policy" in name:
            continue

        # Skip if shapes don't match
        if name in stage2_state and stage2_state[name].shape == param.shape:
            stage2_state[name] = param
            transferred += 1

    model2.load_state_dict(stage2_state)
    print(f"Transferred {transferred} parameter tensors")

    # Create env and trainer for stage 2
    dataset2 = load_dataset(stage2_dataset)
    env2 = create_env(dataset2, stage2_config)
    trainer2 = create_trainer(model2, env2, stage2_config)

    # Train stage 2
    for step in range(stage2_steps):
        metrics = trainer2.train_step()

        if step % stage2_config.get("log_interval", 100) == 0:
            print(f"Stage 2 - Step {step}: {metrics}")

    # Save final checkpoint
    if checkpoint_dir:
        final_ckpt = os.path.join(checkpoint_dir, "curriculum_final.pt")
        torch.save({
            "model_state_dict": model2.state_dict(),
            "step": total_steps,
            "stage": 2,
        }, final_ckpt)
        print(f"Saved final checkpoint: {final_ckpt}")

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for Sudoku")
    parser.add_argument(
        "--stage1-dataset",
        type=str,
        default="data/sudoku-4x4-ultra-easy",
        help="Path to 4×4 Sudoku dataset",
    )
    parser.add_argument(
        "--stage2-dataset",
        type=str,
        default="data/sudoku-extreme-1k-aug-1000",
        help="Path to 9×9 Sudoku dataset",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=2000,
        help="Training steps for stage 1 (4×4)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000,
        help="Total training steps (stage2 = total - stage1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_sudoku_shaped_theory_exact.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.stage1_dataset).exists():
        print(f"Warning: Stage 1 dataset not found: {args.stage1_dataset}")

    if not Path(args.stage2_dataset).exists():
        print(f"Warning: Stage 2 dataset not found: {args.stage2_dataset}")

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    run_curriculum_training(
        stage1_dataset=args.stage1_dataset,
        stage2_dataset=args.stage2_dataset,
        stage1_steps=args.stage1_steps,
        total_steps=args.total_steps,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
