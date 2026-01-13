#!/usr/bin/env python3
"""
Standalone runner for contraction vs SGD tradeoff experiments.

Tests Brett's hypothesis: contraction enforcement may hurt performance
because it acts like a hard constraint that fights SGD / shrinks function class.

Runs 4 conditions with GPU parallelization:
  GPU0: No contraction (enable_contraction=false)
  GPU1: Weak contraction (target_Lz=0.99)
  GPU2: Standard contraction (target_Lz=0.90)
  GPU3: Scheduled contraction (off for 70%, then on with target_Lz=0.90)

Usage:
    python scripts/run_contraction_sgd_tradeoff.py --seed 42

    # With optional second seed for confirmation
    python scripts/run_contraction_sgd_tradeoff.py --seed 42 --confirm-seed 123
"""

import argparse
import os
import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


# Buck2 must run from fbcode directory
FBCODE_DIR = Path.home() / "fbsource" / "fbcode"

# Base config path (relative to buiksat_trm within fbcode)
BASE_CONFIG = "buiksat_trm/configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml"

# Experiment conditions
CONDITIONS = {
    "1_no_contraction": {
        "config_override": "buiksat_trm/configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml",
        "scheduled": False,
        "gpu": 0,
    },
    "2_weak_contraction": {
        "config_override": "buiksat_trm/configs/experiments/contraction_sgd_tradeoff/2_weak_contraction.yaml",
        "scheduled": False,
        "gpu": 1,
    },
    "3_standard_contraction": {
        "config_override": "buiksat_trm/configs/experiments/contraction_sgd_tradeoff/3_standard_contraction.yaml",
        "scheduled": False,
        "gpu": 2,
    },
    "4_scheduled_contraction": {
        "config_override": "buiksat_trm/configs/experiments/contraction_sgd_tradeoff/4_scheduled_contraction.yaml",
        "scheduled": True,  # Special handling required
        "gpu": 3,
    },
}

# Dataset and output paths (defaults, can be overridden by CLI args)
DEFAULT_DATASET = "buiksat_trm/data/sudoku-4x4-trivial"
DEFAULT_RESULTS_DIR = "buiksat_trm/results/plot_data_contraction_sgd_tradeoff"
DEFAULT_PLOTS_DIR = "buiksat_trm/results/plots_contraction_sgd_tradeoff"


def run_standard_experiment(
    condition_name: str,
    seed: int,
    gpu: int,
    base_config: str,
    override_config: str,
    output_dir: str,
    dataset: str,
) -> subprocess.Popen:
    """Launch a standard (non-scheduled) experiment."""

    run_name = f"{condition_name}_seed{seed}"
    # Use absolute path for output file (FBCODE_DIR / output_dir)
    output_file = str(FBCODE_DIR / output_dir / f"{run_name}.log")

    # Build the command
    cmd = [
        "buck2", "run", "//buiksat_trm:upi_trm_train",
        "-c", "fbcode.nvcc_arch=a100",
        "-c", "fbcode.enable_gpu_sections=true",
        "--",
        "--dataset-paths", dataset,
        "--config", base_config,
        "--config", override_config,
        "--seed", str(seed),
        "--train-steps", "5000",
        "--no-wandb",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[{condition_name}] Starting on GPU {gpu} (seed {seed})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output: {output_file}")

    with open(output_file, "w") as f:
        f.write(f"# {condition_name} seed={seed} gpu={gpu}\n")
        f.write(f"# Command: {' '.join(cmd)}\n\n")

    with open(output_file, "a") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(FBCODE_DIR),
        )

    return proc


def run_scheduled_experiment(
    condition_name: str,
    seed: int,
    gpu: int,
    base_config: str,
    override_config: str,
    output_dir: str,
    dataset: str,
) -> subprocess.Popen:
    """
    Launch a scheduled contraction experiment.

    This runs the experiment in two phases:
    - Phase 1 (steps 0-3499): No contraction
    - Phase 2 (steps 3500-4999): Contraction ON with target_Lz=0.90

    We use a checkpoint-resume approach since the training script
    doesn't support mid-run contraction toggling.
    """

    run_name = f"{condition_name}_seed{seed}"
    # Use absolute paths for output files (FBCODE_DIR / output_dir)
    output_file = str(FBCODE_DIR / output_dir / f"{run_name}.log")
    checkpoint_dir = str(FBCODE_DIR / output_dir / f"ckpt_{run_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[{condition_name}] Starting PHASE 1 on GPU {gpu} (seed {seed}) - no contraction")

    # Phase 1: Train for 3500 steps with no contraction, save checkpoint
    cmd_phase1 = [
        "buck2", "run", "//buiksat_trm:upi_trm_train",
        "-c", "fbcode.nvcc_arch=a100",
        "-c", "fbcode.enable_gpu_sections=true",
        "--",
        "--dataset-paths", dataset,
        "--config", base_config,
        "--config", override_config,  # enable_contraction: false
        "--seed", str(seed),
        "--train-steps", "3500",
        "--checkpoint-dir", checkpoint_dir,
        "--save-interval", "3500",  # Save at end of phase 1
        "--no-wandb",
    ]

    with open(output_file, "w") as f:
        f.write(f"# {condition_name} seed={seed} gpu={gpu}\n")
        f.write(f"# PHASE 1 (steps 0-3499): No contraction\n")
        f.write(f"# Command: {' '.join(cmd_phase1)}\n\n")

    with open(output_file, "a") as f:
        proc1 = subprocess.Popen(
            cmd_phase1,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(FBCODE_DIR),
        )

    # Wait for phase 1 to complete
    proc1.wait()

    print(f"[{condition_name}] Starting PHASE 2 on GPU {gpu} (seed {seed}) - contraction ON")

    # Find the checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, "step_3500.pt")
    if not os.path.exists(checkpoint_file):
        # Try alternative naming
        checkpoints = list(Path(checkpoint_dir).glob("*.pt"))
        if checkpoints:
            checkpoint_file = str(checkpoints[-1])
        else:
            print(f"[{condition_name}] ERROR: No checkpoint found in {checkpoint_dir}")
            return None

    # Phase 2: Resume with contraction ON for remaining 1500 steps
    # Create a temporary config for phase 2 with contraction enabled
    phase2_config_abs = str(FBCODE_DIR / output_dir / f"phase2_{run_name}.yaml")
    phase2_config_rel = f"{output_dir}/phase2_{run_name}.yaml"  # Relative path for buck2
    with open(phase2_config_abs, "w") as f:
        f.write("enable_contraction: true\n")
        f.write("target_Lz: 0.90\n")

    cmd_phase2 = [
        "buck2", "run", "//buiksat_trm:upi_trm_train",
        "-c", "fbcode.nvcc_arch=a100",
        "-c", "fbcode.enable_gpu_sections=true",
        "--",
        "--dataset-paths", dataset,
        "--config", base_config,
        "--config", phase2_config_rel,  # enable_contraction: true
        "--seed", str(seed),
        "--train-steps", "5000",  # Total steps including phase 1
        "--resume-checkpoint", checkpoint_file,
        "--no-wandb",
    ]

    with open(output_file, "a") as f:
        f.write(f"\n# PHASE 2 (steps 3500-4999): Contraction ON (target_Lz=0.90)\n")
        f.write(f"# Command: {' '.join(cmd_phase2)}\n\n")

    with open(output_file, "a") as f:
        proc2 = subprocess.Popen(
            cmd_phase2,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(FBCODE_DIR),
        )

    return proc2


def parse_results(log_file: str) -> Dict:
    """Parse the log file to extract final success rate and learning curve."""
    results = {
        "final_success_rate": None,
        "final_mean_score": None,
        "learning_curve": [],  # List of (step, success_rate) tuples
    }

    if not os.path.exists(log_file):
        return results

    with open(log_file, "r") as f:
        for line in f:
            # Look for evaluation results: "[eval] step=X success_rate=Y mean_score=Z"
            if "eval_success_rate=" in line or "success_rate=" in line:
                try:
                    # Try to extract step number
                    step = None
                    if "step " in line:
                        step_str = line.split("step ")[-1].split()[0].strip("]:")
                        step = int(step_str.replace(",", ""))
                    elif "step=" in line:
                        step_str = line.split("step=")[-1].split()[0]
                        step = int(step_str)

                    # Extract success rate
                    if "eval_success_rate=" in line:
                        sr_str = line.split("eval_success_rate=")[-1].split()[0]
                    else:
                        sr_str = line.split("success_rate=")[-1].split()[0]
                    success_rate = float(sr_str)

                    if step is not None:
                        results["learning_curve"].append((step, success_rate))
                    results["final_success_rate"] = success_rate

                    # Extract mean score if available
                    if "mean_score=" in line:
                        ms_str = line.split("mean_score=")[-1].split()[0]
                        results["final_mean_score"] = float(ms_str)
                    elif "eval_mean_score=" in line:
                        ms_str = line.split("eval_mean_score=")[-1].split()[0]
                        results["final_mean_score"] = float(ms_str)

                except (ValueError, IndexError):
                    continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Run contraction vs SGD tradeoff experiments")
    parser.add_argument("--seed", type=int, default=42, help="Primary seed")
    parser.add_argument("--confirm-seed", type=int, default=None, help="Optional confirmation seed")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated list of conditions to run (default: all)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run experiments sequentially instead of in parallel")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Dataset path (default: {DEFAULT_DATASET})")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: auto-generated based on dataset)")
    args = parser.parse_args()

    # Auto-generate results directory based on dataset if not specified
    if args.results_dir:
        results_dir = args.results_dir
        plots_dir = args.results_dir.replace("plot_data", "plots")
    else:
        # Use dataset name for directory (results go to buiksat_trm/results/)
        dataset_name = os.path.basename(args.dataset)
        results_dir = f"buiksat_trm/results/plot_data_contraction_sgd_{dataset_name}"
        plots_dir = f"buiksat_trm/results/plots_contraction_sgd_{dataset_name}"

    # Create output directories (relative to FBCODE_DIR)
    os.makedirs(FBCODE_DIR / results_dir, exist_ok=True)
    os.makedirs(FBCODE_DIR / plots_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Results dir: {results_dir}")
    print(f"Plots dir: {plots_dir}")

    # Determine which conditions to run
    if args.conditions:
        condition_names = [c.strip() for c in args.conditions.split(",")]
    else:
        condition_names = list(CONDITIONS.keys())

    seeds = [args.seed]
    if args.confirm_seed:
        seeds.append(args.confirm_seed)

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENTS WITH SEED {seed}")
        print(f"{'='*60}\n")

        processes = {}

        for cond_name in condition_names:
            cond = CONDITIONS[cond_name]

            if cond["scheduled"]:
                # Run scheduled experiment (blocking for now due to two-phase design)
                proc = run_scheduled_experiment(
                    condition_name=cond_name,
                    seed=seed,
                    gpu=cond["gpu"],
                    base_config=BASE_CONFIG,
                    override_config=cond["config_override"],
                    output_dir=results_dir,
                    dataset=args.dataset,
                )
                if proc:
                    processes[cond_name] = proc
            else:
                proc = run_standard_experiment(
                    condition_name=cond_name,
                    seed=seed,
                    gpu=cond["gpu"],
                    base_config=BASE_CONFIG,
                    override_config=cond["config_override"],
                    output_dir=results_dir,
                    dataset=args.dataset,
                )
                processes[cond_name] = proc

            if args.sequential:
                if proc:
                    proc.wait()

        # Wait for all processes to complete
        if not args.sequential:
            print("\nWaiting for all experiments to complete...")
            for cond_name, proc in processes.items():
                if proc:
                    ret = proc.wait()
                    status = "DONE" if ret == 0 else f"FAILED (exit {ret})"
                    print(f"  [{cond_name}] {status}")

        # Parse results
        for cond_name in condition_names:
            log_file = str(FBCODE_DIR / results_dir / f"{cond_name}_seed{seed}.log")
            results = parse_results(log_file)
            key = f"{cond_name}_seed{seed}"
            all_results[key] = results
            print(f"  [{cond_name}] final_success_rate={results['final_success_rate']}")

    # Save summary
    summary_file = str(FBCODE_DIR / results_dir / "summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Print final table
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Condition':<30} {'Seed':<8} {'Success Rate':<15} {'Mean Score':<15}")
    print("-" * 68)
    for key, res in all_results.items():
        cond, seed_str = key.rsplit("_seed", 1)
        sr = f"{res['final_success_rate']:.3f}" if res['final_success_rate'] is not None else "N/A"
        ms = f"{res['final_mean_score']:.3f}" if res['final_mean_score'] is not None else "N/A"
        print(f"{cond:<30} {seed_str:<8} {sr:<15} {ms:<15}")

    print("\nDone!")


if __name__ == "__main__":
    main()
