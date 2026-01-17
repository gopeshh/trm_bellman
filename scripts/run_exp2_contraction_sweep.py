#!/usr/bin/env python3
"""
Experiment 2: Contraction Strength Sweep

This script orchestrates training runs for the contraction dial experiment.
Sweeps target_Lz in {0.999, 0.99, 0.95, 0.90} with multiple seeds.

Usage:
    # Run full sweep (3 seeds for 0.999, 0.99, 0.95; 1 seed for 0.90)
    python scripts/run_exp2_contraction_sweep.py --full

    # Run single config
    python scripts/run_exp2_contraction_sweep.py --target_lz 0.95 --seed 42

    # Check status
    python scripts/run_exp2_contraction_sweep.py --status
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# Sweep configuration
SWEEP_CONFIG = {
    # target_Lz: (config_file, seeds)
    0.999: ("configs/exp2_contraction_sweep/target_lz_0999.yaml", [41, 42, 43]),
    0.99: ("configs/exp2_contraction_sweep/target_lz_099.yaml", [41, 42, 43]),
    0.95: ("configs/exp2_contraction_sweep/target_lz_095.yaml", [41, 42, 43]),
    0.90: ("configs/exp2_contraction_sweep/target_lz_090.yaml", [41]),
}

CHECKPOINT_BASE = "checkpoints/exp2_contraction_sweep"


def get_checkpoint_path(target_lz: float, seed: int) -> Path:
    """Get checkpoint path for a given target_Lz and seed."""
    lz_str = f"{target_lz:.3f}".replace(".", "")
    return PROJECT_ROOT / CHECKPOINT_BASE / f"lz_{lz_str}" / f"seed{seed}" / "model_step_5000.pt"


def check_status() -> Dict[Tuple[float, int], bool]:
    """Check which runs are complete."""
    status = {}
    for target_lz, (config, seeds) in SWEEP_CONFIG.items():
        for seed in seeds:
            ckpt = get_checkpoint_path(target_lz, seed)
            status[(target_lz, seed)] = ckpt.exists()
    return status


def print_status():
    """Print sweep status."""
    status = check_status()
    print("\n=== Exp2 Contraction Sweep Status ===\n")
    print(f"{'target_Lz':<12} {'Seed':<8} {'Status':<12} {'Path'}")
    print("-" * 80)

    for target_lz, (config, seeds) in sorted(SWEEP_CONFIG.items()):
        for seed in seeds:
            ckpt = get_checkpoint_path(target_lz, seed)
            done = status[(target_lz, seed)]
            status_str = "✓ DONE" if done else "○ PENDING"
            print(f"{target_lz:<12} {seed:<8} {status_str:<12} {ckpt}")

    done_count = sum(status.values())
    total_count = len(status)
    print(f"\n{done_count}/{total_count} runs complete")


def run_training(target_lz: float, seed: int, use_buck: bool = True, dry_run: bool = False):
    """Run a single training job."""
    config_file, _ = SWEEP_CONFIG[target_lz]
    lz_str = f"{target_lz:.3f}".replace(".", "")
    out_dir = PROJECT_ROOT / CHECKPOINT_BASE / f"lz_{lz_str}" / f"seed{seed}"

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_buck:
        cmd = [
            "buck2", "run", "//buiksat_trm:upi_trm_train",
            "-c", "fbcode.nvcc_arch=a100",
            "-c", "fbcode.enable_gpu_sections=true",
            "--local-only",
            "--",
            "--config", str(PROJECT_ROOT / config_file),
            "--seed", str(seed),
            "--checkpoint-dir", str(out_dir),
        ]
    else:
        cmd = [
            "python", str(PROJECT_ROOT / "upi_trm_train.py"),
            "--config", str(PROJECT_ROOT / config_file),
            "--seed", str(seed),
            "--checkpoint-dir", str(out_dir),
        ]

    print(f"\n[Exp2] Running: target_Lz={target_lz}, seed={seed}")
    print(f"[Exp2] Config: {config_file}")
    print(f"[Exp2] Output: {out_dir}")
    print(f"[Exp2] Command: {' '.join(cmd)}")

    if dry_run:
        print("[Exp2] DRY RUN - skipping execution")
        return True

    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT.parent.parent) if use_buck else str(PROJECT_ROOT),
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[Exp2] ERROR: Training failed with exit code {e.returncode}")
        return False


def run_sweep(skip_existing: bool = True, use_buck: bool = True, dry_run: bool = False):
    """Run full sweep."""
    print("\n=== Starting Exp2 Contraction Sweep ===\n")

    status = check_status()

    for target_lz, (config, seeds) in sorted(SWEEP_CONFIG.items()):
        for seed in seeds:
            if skip_existing and status[(target_lz, seed)]:
                print(f"[Exp2] Skipping target_Lz={target_lz}, seed={seed} (already complete)")
                continue

            success = run_training(target_lz, seed, use_buck=use_buck, dry_run=dry_run)
            if not success:
                print(f"[Exp2] Failed at target_Lz={target_lz}, seed={seed}")
                if not dry_run:
                    return False

    print("\n=== Sweep Complete ===")
    print_status()
    return True


def main():
    parser = argparse.ArgumentParser(description="Exp2 Contraction Sweep Runner")
    parser.add_argument("--status", action="store_true", help="Print sweep status")
    parser.add_argument("--full", action="store_true", help="Run full sweep")
    parser.add_argument("--target_lz", type=float, help="Run single target_Lz value")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single run")
    parser.add_argument("--no-buck", action="store_true", help="Use Python directly instead of Buck2")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--force", action="store_true", help="Rerun even if checkpoint exists")

    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    if args.full:
        return 0 if run_sweep(
            skip_existing=not args.force,
            use_buck=not args.no_buck,
            dry_run=args.dry_run
        ) else 1

    if args.target_lz is not None:
        if args.target_lz not in SWEEP_CONFIG:
            print(f"ERROR: target_Lz must be one of {list(SWEEP_CONFIG.keys())}")
            return 1

        success = run_training(
            args.target_lz, args.seed,
            use_buck=not args.no_buck,
            dry_run=args.dry_run
        )
        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
