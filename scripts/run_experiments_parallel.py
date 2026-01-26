#!/usr/bin/env python3
"""
Parallel experiment launcher for multi-GPU training.

Runs experiments across multiple GPUs in parallel, managing job queues
and collecting results. Supports multiple configs and seeds.

Usage:
    # Run all 9x9 experiments with 5 seeds on 4 GPUs
    python scripts/run_experiments_parallel.py \
        --config-dir configs/sudoku9x9 \
        --seeds 0,1,2,3,4 \
        --gpus 0,1,2,3 \
        --output-dir results/sudoku9x9

    # Run specific configs
    python scripts/run_experiments_parallel.py \
        --configs configs/sudoku9x9/upi_trm_9x9.yaml,configs/sudoku9x9/ppo_9x9.yaml \
        --seeds 0,1,2,3,4 \
        --gpus 0,1,2,3

    # Dry run to see what would be executed
    python scripts/run_experiments_parallel.py \
        --config-dir configs/sudoku9x9 \
        --seeds 0,1,2 \
        --gpus 0,1 \
        --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ExperimentJob:
    """Represents a single experiment to run."""
    config_path: str
    seed: int
    gpu_id: int
    output_dir: str
    baseline: Optional[str] = None  # ppo, dqn, or None for upi_trm

    @property
    def name(self) -> str:
        config_name = Path(self.config_path).stem
        return f"{config_name}_seed{self.seed}"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.name)


def detect_baseline_from_config(config_path: str) -> Optional[str]:
    """Detect baseline type from config file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    algorithm = config.get('algorithm', 'upi_trm')
    if algorithm in ['ppo', 'a2c']:
        return 'ppo'
    elif algorithm in ['dqn', 'ddqn']:
        return 'dqn'
    return None


def run_experiment(job: ExperimentJob, dry_run: bool = False) -> Tuple[str, bool, str]:
    """
    Run a single experiment and return (job_name, success, message).
    """
    # Build command
    cmd = [
        sys.executable, "upi_trm_train.py",
        "--config", job.config_path,
        "--seed", str(job.seed),
        "--output-dir", job.run_dir,
    ]

    if job.baseline:
        cmd.extend(["--baseline", job.baseline])

    # Set environment for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    if dry_run:
        return (job.name, True, f"[DRY RUN] Would run on GPU {job.gpu_id}: {' '.join(cmd)}")

    # Create output directory
    os.makedirs(job.run_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(job.run_dir, "train.log")

    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        elapsed = time.time() - start_time
        return (job.name, True, f"Completed in {elapsed/60:.1f} min")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        return (job.name, False, f"Failed after {elapsed/60:.1f} min: {e}")
    except Exception as e:
        return (job.name, False, f"Error: {e}")


def create_job_queue(
    configs: List[str],
    seeds: List[int],
    gpus: List[int],
    output_dir: str
) -> List[ExperimentJob]:
    """Create the queue of experiment jobs."""
    jobs = []

    for config_path in configs:
        baseline = detect_baseline_from_config(config_path)

        for seed in seeds:
            # Round-robin GPU assignment based on job index
            gpu_idx = len(jobs) % len(gpus)
            gpu_id = gpus[gpu_idx]

            job = ExperimentJob(
                config_path=config_path,
                seed=seed,
                gpu_id=gpu_id,
                output_dir=output_dir,
                baseline=baseline
            )
            jobs.append(job)

    return jobs


def run_parallel(
    jobs: List[ExperimentJob],
    num_workers: int,
    dry_run: bool = False
) -> Dict[str, Tuple[bool, str]]:
    """Run jobs in parallel using process pool."""
    results = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(run_experiment, job, dry_run): job
            for job in jobs
        }

        # Collect results as they complete
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                name, success, message = future.result()
                results[name] = (success, message)
                status = "OK" if success else "FAILED"
                print(f"[{status}] {name}: {message}")
            except Exception as e:
                results[job.name] = (False, str(e))
                print(f"[ERROR] {job.name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments in parallel across GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config selection (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing config files (runs all .yaml files)"
    )
    config_group.add_argument(
        "--configs",
        type=str,
        help="Comma-separated list of config files"
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of seeds (default: 0,1,2,3,4)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of GPU IDs (default: 0,1,2,3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum parallel jobs (default: number of GPUs)"
    )

    args = parser.parse_args()

    # Parse seeds and GPUs
    seeds = [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]

    # Collect config files
    if args.config_dir:
        config_dir = Path(args.config_dir)
        configs = sorted([str(p) for p in config_dir.glob("*.yaml")])
        if not configs:
            print(f"Error: No .yaml files found in {args.config_dir}")
            sys.exit(1)
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    # Validate configs exist
    for config in configs:
        if not os.path.exists(config):
            print(f"Error: Config file not found: {config}")
            sys.exit(1)

    # Create job queue
    jobs = create_job_queue(configs, seeds, gpus, args.output_dir)

    # Determine parallelism
    num_workers = args.max_parallel or len(gpus)

    print(f"=== Parallel Experiment Launcher ===")
    print(f"Configs: {len(configs)}")
    print(f"  " + "\n  ".join(configs))
    print(f"Seeds: {seeds}")
    print(f"GPUs: {gpus}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Max parallel: {num_workers}")
    print(f"Output: {args.output_dir}")
    print()

    if args.dry_run:
        print("=== DRY RUN ===")
        for job in jobs:
            print(f"  [{job.gpu_id}] {job.name}")
        print()

    # Run experiments
    start_time = time.time()
    results = run_parallel(jobs, num_workers, args.dry_run)
    elapsed = time.time() - start_time

    # Summary
    print()
    print("=== Summary ===")
    successes = sum(1 for s, _ in results.values() if s)
    failures = len(results) - successes
    print(f"Completed: {successes}/{len(results)}")
    print(f"Failed: {failures}")
    print(f"Total time: {elapsed/60:.1f} min")

    if failures > 0:
        print()
        print("Failed jobs:")
        for name, (success, message) in results.items():
            if not success:
                print(f"  {name}: {message}")

    # Save results manifest
    if not args.dry_run:
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "configs": configs,
            "seeds": seeds,
            "gpus": gpus,
            "results": {
                name: {"success": s, "message": m}
                for name, (s, m) in results.items()
            }
        }
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to: {manifest_path}")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
