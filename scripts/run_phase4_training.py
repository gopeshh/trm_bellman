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
    buck2 run //buiksat_trm:run_phase4_training -- \
        --project-root /absolute/path/to/trm_bellman \
        --fbcode-root /absolute/path/to/fbsource/fbcode
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

from scripts.phase4_result_schema import phase4_run_id
from scripts.phase4_source import resolve_phase4_source_roots

# Configuration
CONDITIONS = ["nc_nv", "nc_yv", "yc_nv", "yc_yv"]
SEEDS = [41, 42, 43]
NUM_GPUS = 4

# Buck2 command template
BUCK2_CMD = [
    "buck2", "run", "//buiksat_trm:upi_trm_train",
    "-c", "fbcode.nvcc_arch=a100",
    "-c", "fbcode.enable_gpu_sections=true",
    "--local-only",
    "--",
]


def run_job(
    condition: str,
    seed: int,
    gpu_id: int,
    *,
    project_root: Path,
    fbcode_root: Path,
) -> subprocess.Popen:
    """Launch a single training job on a specific GPU."""
    config_path = (
        project_root
        / "configs"
        / "phase4_2x2_norm_ablation"
        / f"{condition}.yaml"
    )
    checkpoint_root = project_root / "results" / "phase4_2x2_norm_ablation"
    ckpt_dir = checkpoint_root / f"{condition}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Keep the log beside the ignored checkpoint output. Creating an untracked
    # root-level log before the child starts would invalidate the child's clean
    # producer-worktree preflight.
    log_path = ckpt_dir / "training.log"

    cmd = BUCK2_CMD + [
        "--config", str(config_path),
        "--seed", str(seed),
        "--run-id", phase4_run_id(condition, seed),
        "--checkpoint-dir", str(ckpt_dir),
        "--producer-repo-root", str(project_root),
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
            cwd=str(fbcode_root),
        )

    return proc


def main():
    parser = argparse.ArgumentParser(
        description="Run the registered Phase 4 2x2 training design."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Exact clean trm_bellman checkout used as the producer source",
    )
    parser.add_argument(
        "--fbcode-root",
        required=True,
        help="fbcode cell whose buiksat_trm entry resolves to --project-root",
    )
    args = parser.parse_args()
    project_root, fbcode_root = resolve_phase4_source_roots(
        args.project_root,
        args.fbcode_root,
    )
    checkpoint_root = project_root / "results" / "phase4_2x2_norm_ablation"

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
            occupied_gpu_ids = {entry[3] for entry in running}
            gpu_id = next(
                candidate
                for candidate in range(NUM_GPUS)
                if candidate not in occupied_gpu_ids
            )
            proc = run_job(
                condition,
                seed,
                gpu_id,
                project_root=project_root,
                fbcode_root=fbcode_root,
            )
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
    print(f"Checkpoints saved to: {checkpoint_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
