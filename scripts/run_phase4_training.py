#!/usr/bin/env python3
"""
Phase 4: Multi-seed 2x2 Norm Ablation Training Orchestrator.

Runs 12 training jobs (4 conditions × 3 seeds) on 4 GPUs.
Uses a job queue to run 4 jobs at a time in parallel.

Conditions:
- nc_nv: No Contraction, No Value-head Norm
- nc_yv: No Contraction, Yes Value-head Norm
- yc_nv: Yes Contraction, No Value-head Norm
- yc_yv: Yes Contraction, Yes Value-head Norm

Seeds: {41, 42, 43}

Usage:
    python scripts/run_phase4_training.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Configuration
CONDITIONS = ["nc_nv", "nc_yv", "yc_nv", "yc_yv"]
SEEDS = [41, 42, 43]
NUM_GPUS = 4

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "phase4_2x2_norm_ablation"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "phase4_2x2_norm_ablation"

# Buck2 command template
BUCK2_CMD = [
    "buck2", "run", "//buiksat_trm:upi_trm_train",
    "-c", "fbcode.nvcc_arch=a100",
    "-c", "fbcode.enable_gpu_sections=true",
    "--local-only",
    "--",
]


def run_job(condition: str, seed: int, gpu_id: int) -> subprocess.Popen:
    """Launch a single training job on a specific GPU."""
    config_path = CONFIG_DIR / f"{condition}.yaml"
    ckpt_dir = CHECKPOINT_DIR / f"{condition}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = PROJECT_ROOT / f"training_phase4_{condition}_s{seed}.log"

    cmd = BUCK2_CMD + [
        "--config", str(config_path),
        "--seed", str(seed),
        "--checkpoint-dir", str(ckpt_dir),
        "--save-interval", "1000",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting {condition} seed={seed}")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Log: {log_path}")

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
    print("=" * 60)
    print("Phase 4: 2x2 Norm Ablation Training")
    print("=" * 60)
    print()
    print(f"Conditions: {CONDITIONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total jobs: {len(CONDITIONS) * len(SEEDS)}")
    print(f"GPUs: {NUM_GPUS}")
    print()

    # Build job queue
    jobs: List[Tuple[str, int]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            jobs.append((condition, seed))

    print(f"Job queue: {len(jobs)} jobs")
    for i, (cond, seed) in enumerate(jobs):
        print(f"  [{i}] {cond} seed={seed}")
    print()

    # Run jobs in batches of NUM_GPUS
    running: List[Tuple[subprocess.Popen, str, int, int]] = []  # (proc, condition, seed, gpu_id)
    job_idx = 0
    completed = 0
    failed = 0

    while job_idx < len(jobs) or running:
        # Start new jobs if GPUs are available
        while len(running) < NUM_GPUS and job_idx < len(jobs):
            condition, seed = jobs[job_idx]
            gpu_id = len(running)  # Assign to next available GPU slot
            proc = run_job(condition, seed, gpu_id)
            running.append((proc, condition, seed, gpu_id))
            job_idx += 1

        # Check for completed jobs
        still_running = []
        for proc, condition, seed, gpu_id in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, condition, seed, gpu_id))
            else:
                if ret == 0:
                    print(f"[GPU {gpu_id}] COMPLETED: {condition} seed={seed}")
                    completed += 1
                else:
                    print(f"[GPU {gpu_id}] FAILED (exit={ret}): {condition} seed={seed}")
                    failed += 1

        running = still_running

        if running:
            time.sleep(30)  # Check every 30 seconds

    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Completed: {completed}/{len(jobs)}")
    print(f"Failed: {failed}/{len(jobs)}")

    if failed > 0:
        print("\nWARNING: Some jobs failed. Check logs for details.")
        return 1

    print("\nAll jobs completed successfully!")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
