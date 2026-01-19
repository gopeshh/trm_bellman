#!/usr/bin/env python3
"""
Exp5 v2: Retrain nc_rdis checkpoints with proper dataset paths.

This script fixes the critical bug where nc_rdis/Phase4 checkpoints were trained
without --dataset-paths, causing them to use DummyPuzzleDataset (synthetic data)
instead of real Sudoku puzzles.

The fix:
- Explicitly pass --dataset-paths data/sudoku-4x4-trivial
- This ensures models are trained/evaluated on actual unsolved puzzles
- Expected: initial_score ~13.5 (not 16.0) during training evaluation

Usage:
    python scripts/train_nc_rdis_v2.py

    # Dry run
    python scripts/train_nc_rdis_v2.py --dry-run

Runs 3 seeds {41, 42, 43} in parallel on GPUs 0-2.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent

# Configuration
SEEDS = [41, 42, 43]
NUM_GPUS = 4
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp3_projection_ablation" / "nc_rdis.yaml"
CHECKPOINT_BASE = PROJECT_ROOT / "results" / "exp5_v2_inputs_eval"
DATASET_PATH = PROJECT_ROOT / "data" / "sudoku-4x4-trivial"

# Buck2 command template
BUCK2_CMD = [
    "buck2", "run", "//buiksat_trm:upi_trm_train",
    "-c", "fbcode.nvcc_arch=a100",
    "-c", "fbcode.enable_gpu_sections=true",
    "--local-only",
    "--",
]


def run_job(seed: int, gpu_id: int, dry_run: bool = False) -> Optional[subprocess.Popen]:
    """Launch a single training job on a specific GPU."""
    ckpt_dir = CHECKPOINT_BASE / f"nc_rdis_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = PROJECT_ROOT / f"training_exp5_v2_nc_rdis_s{seed}.log"

    cmd = BUCK2_CMD + [
        "--config", str(CONFIG_PATH),
        "--seed", str(seed),
        "--checkpoint-dir", str(ckpt_dir),
        "--dataset-paths", str(DATASET_PATH),  # CRITICAL FIX
        "--save-interval", "1000",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting nc_rdis seed={seed}")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Log: {log_path}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print(f"  [DRY RUN - skipping execution]")
        return None

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT.parent.parent / "fbcode"),  # Run from fbcode
        )

    return proc


def main():
    parser = argparse.ArgumentParser(description="Train nc_rdis checkpoints with proper dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to train")
    args = parser.parse_args()

    print("=" * 70)
    print("Exp5 v2: Retraining nc_rdis with proper dataset paths")
    print("=" * 70)
    print()
    print(f"Seeds: {args.seeds}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {CHECKPOINT_BASE}")
    print()

    # Verify prerequisites
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found: {CONFIG_PATH}")
        sys.exit(1)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    # Launch jobs in parallel (one per GPU)
    processes = []
    for i, seed in enumerate(args.seeds):
        gpu_id = i % NUM_GPUS
        proc = run_job(seed, gpu_id, dry_run=args.dry_run)
        if proc:
            processes.append((seed, proc))
        # Small delay between launches
        time.sleep(2)

    if args.dry_run:
        print("\n[DRY RUN] No processes started.")
        return

    print(f"\nStarted {len(processes)} training jobs.")
    print("Waiting for completion...")

    # Wait for all jobs to complete
    completed = 0
    failed = 0
    for seed, proc in processes:
        ret = proc.wait()
        if ret == 0:
            completed += 1
            print(f"  ✓ Seed {seed} completed successfully")
        else:
            failed += 1
            print(f"  ✗ Seed {seed} FAILED (exit code {ret})")

    print()
    print("=" * 70)
    print(f"Completed: {completed}/{len(processes)}")
    if failed > 0:
        print(f"Failed: {failed}")
        sys.exit(1)

    # Verify checkpoints exist
    print()
    print("Verifying checkpoints...")
    all_exist = True
    for seed in args.seeds:
        ckpt = CHECKPOINT_BASE / f"nc_rdis_s{seed}" / "model_step_5000.pt"
        if ckpt.exists():
            print(f"  ✓ {ckpt}")
        else:
            print(f"  ✗ MISSING: {ckpt}")
            all_exist = False

    if all_exist:
        print()
        print("SUCCESS: All checkpoints created.")
        print()
        print("Next step: Run Exp5 v2 evaluation with:")
        print(f"  python scripts/exp5_tradeoff_curve_v2.py")
    else:
        print()
        print("WARNING: Some checkpoints missing. Check training logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
