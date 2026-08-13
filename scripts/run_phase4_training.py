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
        --fbcode-root /absolute/path/to/fbsource/fbcode \
        --phase4-launcher /absolute/path/to/phase4_runtime_launcher \
        --training-runtime /absolute/path/to/upi_trm_train.par \
        --expected-runtime-sha256 <sha256> \
        --expected-producer-git-commit <commit>
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


def _require_absolute_file(path_value: str, label: str) -> Path:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path.")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist.") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file.")
    return resolved


def _require_lower_hex(value: str, length: int, label: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{label} must be {length} lowercase hexadecimal characters."
        )
    return value


def run_job(
    condition: str,
    seed: int,
    gpu_id: int,
    *,
    project_root: Path,
    fbcode_root: Path,
    phase4_launcher: Path,
    training_runtime: Path,
    expected_runtime_sha256: str,
    expected_producer_git_commit: str,
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

    cmd = [
        str(phase4_launcher),
        "--purpose",
        "phase4-training",
        "--runtime-archive",
        str(training_runtime),
        "--expected-runtime-sha256",
        expected_runtime_sha256,
        "--source-project-root",
        str(project_root),
        "--expected-source-git-commit",
        expected_producer_git_commit,
        "--",
        "--config", str(config_path),
        "--seed", str(seed),
        "--run-id", phase4_run_id(condition, seed),
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
    parser.add_argument(
        "--phase4-launcher",
        required=True,
        help="Absolute path to the Phase 4 runtime launcher",
    )
    parser.add_argument(
        "--training-runtime",
        required=True,
        help="Absolute path to the frozen Phase 4 training PAR",
    )
    parser.add_argument(
        "--expected-runtime-sha256",
        required=True,
        help="Externally supplied SHA-256 for the training PAR",
    )
    parser.add_argument(
        "--expected-producer-git-commit",
        required=True,
        help="Externally authorized producer source commit",
    )
    args = parser.parse_args()
    try:
        phase4_launcher = _require_absolute_file(
            args.phase4_launcher,
            "Phase 4 launcher",
        )
        training_runtime = _require_absolute_file(
            args.training_runtime,
            "Phase 4 training runtime",
        )
        expected_runtime_sha256 = _require_lower_hex(
            args.expected_runtime_sha256,
            64,
            "Expected training runtime SHA-256",
        )
        expected_producer_git_commit = _require_lower_hex(
            args.expected_producer_git_commit,
            40,
            "Expected producer Git commit",
        )
    except ValueError as error:
        parser.error(str(error))
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
                phase4_launcher=phase4_launcher,
                training_runtime=training_runtime,
                expected_runtime_sha256=expected_runtime_sha256,
                expected_producer_git_commit=(
                    expected_producer_git_commit
                ),
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
