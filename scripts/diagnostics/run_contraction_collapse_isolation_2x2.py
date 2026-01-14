#!/usr/bin/env python3
"""
2x2 Ablation: Isolate cause of contraction collapse.

This script runs 4 training conditions to isolate whether the collapse
observed with enable_contraction=True is caused by:
  - z→z contraction (opnorm clamp + output scaling on L_level layers)
  - value head normalization (spectral norm + output scaling on value head)
  - their interaction

Conditions:
  A: zcon OFF, vhead_norm OFF      (baseline - should match no_contraction behavior)
  B: zcon ON,  vhead_norm OFF      (isolates z→z contraction)
  C: zcon OFF, vhead_norm ON       (isolates value head normalization)
  D: zcon ON,  vhead_norm ON       (matches current enable_contraction=true behavior)

Usage:
  python scripts/diagnostics/run_contraction_collapse_isolation_2x2.py

Output:
  results/verification_2x2/A_zconOFF_vheadOFF_seed42_200.log
  results/verification_2x2/B_zconON_vheadOFF_seed42_200.log
  results/verification_2x2/C_zconOFF_vheadON_seed42_200.log
  results/verification_2x2/D_zconON_vheadON_seed42_200.log
"""

import argparse
import logging
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from upi_trm_train import build_dataset_from_paths, OfflinePuzzleDataset
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.upi_trm_trainer import UPITrmTrainer
from utils.seeding import set_global_seed
from utils.lipschitz import (
    apply_opnorm_clamp_to_trm,
    enforce_global_contraction,
    apply_spectral_norm_to_value_head,
    enforce_global_contraction_on_value_head,
    _linear_module_types,
)


@dataclass
class ConditionConfig:
    """Configuration for one experimental condition."""
    name: str
    label: str
    zcon_on: bool  # z→z contraction
    vhead_norm_on: bool  # value head normalization


CONDITIONS = [
    ConditionConfig("A", "zconOFF_vheadOFF", zcon_on=False, vhead_norm_on=False),
    ConditionConfig("B", "zconON_vheadOFF", zcon_on=True, vhead_norm_on=False),
    ConditionConfig("C", "zconOFF_vheadON", zcon_on=False, vhead_norm_on=True),
    ConditionConfig("D", "zconON_vheadON", zcon_on=True, vhead_norm_on=True),
]


def remove_spectral_norm_from_module(module: nn.Module) -> bool:
    """Remove spectral normalization from a module if present."""
    if hasattr(module, "weight_orig"):
        try:
            nn_utils.remove_spectral_norm(module)
            return True
        except ValueError:
            pass
    return False


def remove_value_head_normalization(value_head: nn.Module) -> int:
    """
    Remove all normalization from value head:
    1. Remove spectral_norm wrappers
    2. Reset _value_lip_scale buffers to 1.0

    Returns number of modules modified.
    """
    count = 0
    for name, module in value_head.named_modules():
        # Remove spectral norm if present
        if remove_spectral_norm_from_module(module):
            count += 1
            print(f"  [vhead] Removed spectral_norm from {name}")

        # Reset value lip scale if present
        if hasattr(module, "_value_lip_scale"):
            with torch.no_grad():
                module._value_lip_scale.fill_(1.0)
            count += 1
            print(f"  [vhead] Reset _value_lip_scale to 1.0 in {name}")

    return count


def remove_z_contraction(inner_model: nn.Module) -> int:
    """
    Remove z→z contraction from inner model:
    1. Reset _inner_lip_scale buffers to 1.0

    Note: We don't "un-clamp" the weights (that's been done already),
    but we do reset the output scaling factors.

    Returns number of modules modified.
    """
    count = 0
    for name, module in inner_model.named_modules():
        if hasattr(module, "_inner_lip_scale"):
            with torch.no_grad():
                module._inner_lip_scale.fill_(1.0)
            count += 1
            print(f"  [zcon] Reset _inner_lip_scale to 1.0 in {name}")
    return count


def apply_z_contraction(inner_model: nn.Module, target_Lz: float = 0.9) -> None:
    """Apply z→z contraction: opnorm clamp + output scaling."""
    print(f"  [zcon] Applying opnorm clamp + enforce_global_contraction (target_Lz={target_Lz})")
    sigma_dict = apply_opnorm_clamp_to_trm(
        inner_model,
        per_layer_max=1.0,
        num_power_iters=10,
        restrict_to_reasoning_layers=True,
    )
    print(f"  [zcon] Clamped {len(sigma_dict)} layers: {list(sigma_dict.keys())[:3]}...")
    enforce_global_contraction(
        inner_model,
        target_Lz,
        restrict_to_reasoning_layers=True,
    )
    print(f"  [zcon] Applied global contraction scaling")


def apply_vhead_normalization(value_head: nn.Module, target_Lv: float = 1.0) -> None:
    """Apply value head normalization: spectral norm + output scaling."""
    print(f"  [vhead] Applying spectral_norm + enforce_global_contraction (target_Lv={target_Lv})")
    apply_spectral_norm_to_value_head(value_head)
    enforce_global_contraction_on_value_head(value_head, target_Lv)
    print(f"  [vhead] Applied value head normalization")


def configure_model_for_condition(
    model: TinyRecursiveReasoningModel_ACTV1,
    condition: ConditionConfig,
    target_Lz: float = 0.9,
    target_Lv: float = 1.0,
) -> None:
    """
    Configure model for a specific experimental condition.

    This function does POST-CONSTRUCTION surgery on the model to
    enable/disable z→z contraction and value head normalization independently.
    """
    print(f"\n[Config] Condition {condition.name}: {condition.label}")
    print(f"  zcon_on={condition.zcon_on}, vhead_norm_on={condition.vhead_norm_on}")

    # First, ensure we're starting from a clean state by removing any existing
    # normalization (the model was built with enable_contraction=False)

    # Handle z→z contraction
    if condition.zcon_on:
        apply_z_contraction(model.inner, target_Lz)
    else:
        # Ensure no z→z contraction scaling is active
        count = remove_z_contraction(model.inner)
        if count == 0:
            print(f"  [zcon] No existing contraction scaling found (clean)")

    # Handle value head normalization
    if condition.vhead_norm_on:
        apply_vhead_normalization(model.value_head, target_Lv)
    else:
        # Remove any value head normalization
        count = remove_value_head_normalization(model.value_head)
        if count == 0:
            print(f"  [vhead] No existing normalization found (clean)")


def create_checker_fn(grid_size: int = 4):
    """Create progress-based checker function."""
    from rl.sudoku_utils import (
        count_sudoku_violations_4x4,
        count_sudoku_violations_9x9,
        sudoku_filled_cells,
    )

    def checker_fn(x_batch: Dict, y: torch.Tensor) -> float:
        """Progress-based checker: returns number of correctly filled cells."""
        if grid_size == 4:
            violations = count_sudoku_violations_4x4(y)
        else:
            violations = count_sudoku_violations_9x9(y)

        filled = sudoku_filled_cells(y)

        # If there are violations, penalize
        if violations > 0:
            return max(0.0, filled - violations)
        return float(filled)

    return checker_fn


def run_training(
    condition: ConditionConfig,
    dataset_path: str,
    config_path: str,
    num_steps: int = 200,
    seed: int = 42,
    output_dir: str = "results/verification_2x2",
) -> str:
    """
    Run a single training experiment for one condition.

    Returns path to log file.
    """
    import yaml

    # Setup logging to file
    log_filename = f"{condition.name}_{condition.label}_seed{seed}_{num_steps}.log"
    log_path = Path(output_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure file logging
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger = logging.getLogger(f"condition_{condition.name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print(f"\n{'='*60}")
    print(f"Running Condition {condition.name}: {condition.label}")
    print(f"{'='*60}")

    # Set seed
    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Override for this diagnostic
    base_config['num_train_steps'] = num_steps
    base_config['enable_contraction'] = False  # We'll apply manually
    base_config['log_interval'] = 10
    base_config['eval_interval'] = 50

    # Create RLConfig
    rl_cfg = RLConfig(**{k: v for k, v in base_config.items() if hasattr(RLConfig, k)})

    # Load dataset using build_dataset_from_paths
    dataset, seq_len, vocab_size, num_identifiers = build_dataset_from_paths(
        dataset_paths=[dataset_path],
        pool_size=500,  # Load 500 puzzles for training
    )
    print(f"Loaded dataset: {len(dataset)} puzzles, seq_len={seq_len}, vocab_size={vocab_size}")

    # Create model with enable_contraction=False
    # We'll apply the contraction/normalization manually after construction

    # Compute rl_num_actions for 4x4 Sudoku
    grid_size = 4
    rl_num_actions = seq_len * vocab_size + 1  # +1 for STOP action

    trm_cfg_dict = dict(
        batch_size=rl_cfg.batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,  # No puzzle embeddings for this diagnostic
        puzzle_emb_len=0,
        num_puzzle_identifiers=max(num_identifiers, rl_cfg.batch_size),
        vocab_size=vocab_size,
        H_cycles=3,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=128,
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
        rl_enable_contraction=False,  # Disable at construction - we'll apply manually
        rl_target_Lz=base_config.get("target_Lz", 0.9),
        rl_target_Lv=base_config.get("target_Lv", 1.0),
        rl_enable_policy_head=True,
        rl_num_actions=rl_num_actions,
        rl_latent_ball_radius=base_config.get("latent_ball_radius", 10.0),
    )

    model = TinyRecursiveReasoningModel_ACTV1(trm_cfg_dict).to(device)

    # Create env config
    env_cfg = PlanEditEnvConfig(
        max_edits=base_config.get("max_edits", 20),
        gamma=base_config.get("gamma", 0.99),
        vocab_size=vocab_size,
        reward_shaping=base_config.get("reward_shaping", True),
        stop_action_mode=base_config.get("stop_action_mode", "disabled"),
        fail_terminal_reward=base_config.get("fail_terminal_reward", -16.0),
        solve_terminal_reward=base_config.get("solve_terminal_reward", 0.0),
        solved_threshold=base_config.get("solved_threshold", 16.0),
    )

    # Create checker
    checker_fn = create_checker_fn(grid_size=4)

    # Create env with dataset and checker
    env = PlanEditEnv(dataset=dataset, checker=checker_fn, config=env_cfg, task_config=None)
    # Set stop action ID (last action in action space)
    env.set_stop_action_id(stop_id=rl_num_actions - 1)

    # Create trainer FIRST, THEN apply condition-specific normalization
    # This ensures all model copies (model, target_model, policy_model_candidate)
    # have the same architecture before we apply normalization
    trainer = UPITrmTrainer(
        model=model,
        env=env,
        rl_cfg=rl_cfg,
        device=torch.device(device),
    )
    # Set checker_fn for exact baseline computation
    trainer._checker_fn = checker_fn

    # Now apply condition-specific normalization to ALL models
    target_Lz = base_config.get("target_Lz", 0.9)
    target_Lv = base_config.get("target_Lv", 1.0)

    for model_instance in [trainer.model, trainer.target_model, trainer.policy_model_candidate]:
        configure_model_for_condition(
            model_instance,
            condition,
            target_Lz=target_Lz,
            target_Lv=target_Lv,
        )

    # Training loop with VALUE_DEBUG logging
    print(f"\n[Training] Starting {num_steps} steps...")

    log_data = []

    for step in range(1, num_steps + 1):
        # Run one training step
        metrics = trainer.train_step()

        # Log VALUE_DEBUG style output every log_interval steps
        if step % rl_cfg.log_interval == 0:
            # Get target and value statistics from the last batch
            # These keys match what value_update() returns
            target_mean = metrics.get("target_mean", float('nan'))
            target_std = metrics.get("target_std", float('nan'))
            target_min = metrics.get("target_min", float('nan'))
            target_max = metrics.get("target_max", float('nan'))
            v_mean = metrics.get("value_mean", float('nan'))
            v_std = metrics.get("value_std", float('nan'))

            log_line = (
                f"Step {step}: "
                f"target(mean={target_mean:.3f}, std={target_std:.3f}, min={target_min:.3f}, max={target_max:.3f}) | "
                f"V(s)(mean={v_mean:.3f}, std={v_std:.3f})"
            )
            print(log_line)
            logger.info(log_line)

            log_data.append({
                "step": step,
                "target_mean": target_mean,
                "target_std": target_std,
                "target_min": target_min,
                "target_max": target_max,
                "v_mean": v_mean,
                "v_std": v_std,
            })

        # Eval at eval_interval (optional, skip if causes issues)
        if step % rl_cfg.eval_interval == 0:
            try:
                eval_metrics = trainer.evaluate_policy_metrics(env_cfg, dataset, checker_fn)
                success_rate = eval_metrics.get("success_rate", 0.0)
                mean_score = eval_metrics.get("mean_score", 0.0)
                eval_line = f"  [Eval] Step {step}: success_rate={success_rate:.3f}, mean_score={mean_score:.3f}"
                print(eval_line)
                logger.info(eval_line)
            except Exception as e:
                logger.info(f"  [Eval] Step {step}: skipped ({e})")

    print(f"\n[Training] Completed {num_steps} steps. Log saved to {log_path}")

    # Write summary to log
    logger.info("\n" + "="*60)
    logger.info(f"Summary for Condition {condition.name}: {condition.label}")
    logger.info("="*60)
    for entry in log_data:
        logger.info(str(entry))

    return str(log_path)


def parse_log_for_metrics(log_path: str) -> List[Dict]:
    """Parse a log file to extract VALUE_DEBUG metrics."""
    metrics = []

    pattern = re.compile(
        r"Step (\d+): target\(mean=([-\d.nan]+), std=([-\d.nan]+), min=([-\d.nan]+), max=([-\d.nan]+)\) \| "
        r"V\(s\)\(mean=([-\d.nan]+), std=([-\d.nan]+)\)"
    )

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                def safe_float(s):
                    try:
                        return float(s)
                    except ValueError:
                        return float('nan')

                metrics.append({
                    "step": int(match.group(1)),
                    "target_mean": safe_float(match.group(2)),
                    "target_std": safe_float(match.group(3)),
                    "target_min": safe_float(match.group(4)),
                    "target_max": safe_float(match.group(5)),
                    "v_mean": safe_float(match.group(6)),
                    "v_std": safe_float(match.group(7)),
                })

    return metrics


def detect_collapse(metrics: List[Dict]) -> Tuple[bool, Optional[int]]:
    """
    Detect collapse based on:
    - V(s) mean <= -19.9 AND V(s) std <= 0.05 for ≥3 consecutive intervals
    - OR target mean <= -19.7 early (by step 20)

    Returns (is_collapsed, collapse_step).
    """
    # Check early target collapse
    for m in metrics:
        if m["step"] <= 20 and m["target_mean"] <= -19.7:
            return True, m["step"]

    # Check V(s) collapse (3 consecutive)
    consecutive = 0
    for m in metrics:
        if m["v_mean"] <= -19.9 and m["v_std"] <= 0.05:
            consecutive += 1
            if consecutive >= 3:
                return True, m["step"]
        else:
            consecutive = 0

    return False, None


def create_results_table(all_metrics: Dict[str, List[Dict]]) -> str:
    """Create markdown comparison table."""
    # Find all unique steps
    all_steps = set()
    for metrics in all_metrics.values():
        for m in metrics:
            all_steps.add(m["step"])
    steps = sorted(all_steps)

    # Header
    header = "| Step |"
    for cond in CONDITIONS:
        header += f" {cond.name} target | {cond.name} V(s) |"
    header += "\n"

    # Separator
    sep = "|------|"
    for _ in CONDITIONS:
        sep += "---------|---------|"
    sep += "\n"

    # Rows
    rows = ""
    for step in steps:
        row = f"| {step} |"
        for cond in CONDITIONS:
            metrics = all_metrics.get(cond.name, [])
            m = next((x for x in metrics if x["step"] == step), None)
            if m:
                row += f" {m['target_mean']:.2f} | {m['v_mean']:.2f} |"
            else:
                row += " - | - |"
        row += "\n"
        rows += row

    return header + sep + rows


def main():
    parser = argparse.ArgumentParser(description="2x2 Contraction Collapse Isolation Experiment")
    parser.add_argument("--steps", type=int, default=200, help="Training steps per condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str,
                       default="data/sudoku-4x4-easy_6to8empties",
                       help="Dataset path")
    parser.add_argument("--config", type=str,
                       default="configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml",
                       help="Base config path")
    parser.add_argument("--output-dir", type=str, default="results/verification_2x2",
                       help="Output directory for logs")
    parser.add_argument("--condition", type=str, default=None,
                       help="Run only specific condition (A, B, C, or D)")
    args = parser.parse_args()

    print("="*60)
    print("2x2 Contraction Collapse Isolation Experiment")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    # Verify paths
    if not Path(args.dataset).exists():
        print(f"\nERROR: Dataset not found at {args.dataset}")
        sys.exit(1)
    if not Path(args.config).exists():
        print(f"\nERROR: Config not found at {args.config}")
        sys.exit(1)

    # Run conditions
    conditions_to_run = CONDITIONS
    if args.condition:
        conditions_to_run = [c for c in CONDITIONS if c.name == args.condition]
        if not conditions_to_run:
            print(f"\nERROR: Unknown condition '{args.condition}'. Use A, B, C, or D.")
            sys.exit(1)

    log_paths = {}
    all_metrics = {}

    for condition in conditions_to_run:
        log_path = run_training(
            condition=condition,
            dataset_path=args.dataset,
            config_path=args.config,
            num_steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        log_paths[condition.name] = log_path

        # Parse metrics from log
        metrics = parse_log_for_metrics(log_path)
        all_metrics[condition.name] = metrics

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nLog files:")
    for name, path in log_paths.items():
        print(f"  {name}: {path}")

    print("\nCollapse detection:")
    collapse_results = {}
    for cond in conditions_to_run:
        metrics = all_metrics.get(cond.name, [])
        is_collapsed, collapse_step = detect_collapse(metrics)
        collapse_results[cond.name] = (is_collapsed, collapse_step)
        status = f"COLLAPSED at step {collapse_step}" if is_collapsed else "STABLE"
        print(f"  {cond.name} ({cond.label}): {status}")

    # Create comparison table
    if len(all_metrics) > 0:
        print("\nComparison Table:")
        table = create_results_table(all_metrics)
        print(table)

        # Write README
        readme_path = Path(args.output_dir) / "README.md"
        with open(readme_path, 'w') as f:
            f.write("# 2x2 Contraction Collapse Isolation Results\n\n")
            f.write("## Experiment Design\n\n")
            f.write("This experiment isolates the cause of training collapse when `enable_contraction=True`.\n\n")
            f.write("### Conditions\n\n")
            f.write("| Condition | z→z Contraction | Value Head Norm | Description |\n")
            f.write("|-----------|-----------------|-----------------|-------------|\n")
            f.write("| A | OFF | OFF | Baseline (should match no_contraction) |\n")
            f.write("| B | ON | OFF | Isolates z→z contraction |\n")
            f.write("| C | OFF | ON | Isolates value head normalization |\n")
            f.write("| D | ON | ON | Full enable_contraction=true |\n\n")

            f.write("### Commands Used\n\n")
            f.write("```bash\n")
            f.write("buck2 clean\n")
            f.write(f"python scripts/diagnostics/run_contraction_collapse_isolation_2x2.py \\\n")
            f.write(f"    --dataset {args.dataset} \\\n")
            f.write(f"    --config {args.config} \\\n")
            f.write(f"    --steps {args.steps} \\\n")
            f.write(f"    --seed {args.seed}\n")
            f.write("```\n\n")

            f.write("## Results\n\n")
            f.write("### Collapse Detection\n\n")
            f.write("| Condition | Status | Collapse Step |\n")
            f.write("|-----------|--------|---------------|\n")
            for cond in CONDITIONS:
                if cond.name in collapse_results:
                    is_collapsed, collapse_step = collapse_results[cond.name]
                    status = "COLLAPSED" if is_collapsed else "STABLE"
                    step_str = str(collapse_step) if collapse_step else "-"
                    f.write(f"| {cond.name} ({cond.label}) | {status} | {step_str} |\n")
            f.write("\n")

            f.write("### Comparison Table\n\n")
            f.write(table)
            f.write("\n")

            f.write("## Conclusion\n\n")
            # Determine cause of collapse
            a_collapsed = collapse_results.get("A", (False, None))[0]
            b_collapsed = collapse_results.get("B", (False, None))[0]
            c_collapsed = collapse_results.get("C", (False, None))[0]
            d_collapsed = collapse_results.get("D", (False, None))[0]

            if c_collapsed and not b_collapsed:
                f.write("**Value head normalization is the culprit.**\n\n")
                f.write("Condition C (vhead_norm ON, zcon OFF) collapses but Condition B (zcon ON, vhead_norm OFF) is stable.\n")
                f.write("This indicates that the spectral normalization + Lv scaling on the value head causes V(s) to saturate.\n")
            elif b_collapsed and not c_collapsed:
                f.write("**z→z contraction is the culprit.**\n\n")
                f.write("Condition B (zcon ON, vhead_norm OFF) collapses but Condition C (vhead_norm ON, zcon OFF) is stable.\n")
                f.write("This indicates that the opnorm clamping + Lz scaling on z→z layers causes the collapse.\n")
            elif d_collapsed and not b_collapsed and not c_collapsed:
                f.write("**Interaction effect between z→z contraction and value head normalization.**\n\n")
                f.write("Only Condition D (both ON) collapses; B and C individually are stable.\n")
                f.write("The collapse requires both mechanisms acting together.\n")
            elif b_collapsed and c_collapsed:
                f.write("**Both mechanisms independently cause collapse.**\n\n")
                f.write("Both Condition B and Condition C collapse independently.\n")
            elif not any([b_collapsed, c_collapsed, d_collapsed]):
                f.write("**No collapse detected in any condition.**\n\n")
                f.write("All conditions remained stable during this short run.\n")
                f.write("Consider running longer (e.g., --steps 500) to detect collapse.\n")
            else:
                f.write("**Inconclusive results.**\n\n")
                f.write("The pattern of collapse does not clearly indicate a single cause.\n")

            f.write("\n## Log Files\n\n")
            for name, path in log_paths.items():
                f.write(f"- {name}: `{path}`\n")

        print(f"\nREADME written to: {readme_path}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
