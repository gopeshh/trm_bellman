#!/usr/bin/env python3
"""
Exp5 v2: Stability-Expressivity Tradeoff Curve with Fixed Dataset.

This is the CORRECT version of Exp5 that uses checkpoints trained with proper
--dataset-paths, ensuring models were trained/evaluated on real unsolved puzzles.

Key fix:
- Uses results/exp5_v2_inputs_eval/nc_rdis_s{41,42,43} checkpoints
- These were trained with --dataset-paths data/sudoku-4x4-trivial
- Training evaluation showed initial=13.52 (not 16.00)

Differences from original Exp5:
- Checkpoints from exp5_v2_inputs_eval/ instead of exp3/
- Success rates are now comparable to baseline (~30-60% expected, not ~5%)
- All other parameters identical (dial scales, mismatch depths, projection disabled)

Usage:
    python scripts/exp5_tradeoff_curve_v2.py

Output:
    results/paper_ready/exp5_tradeoff_curve_v2/
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Constants
# =============================================================================

# Use v2 checkpoints (trained with proper dataset)
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "exp5_v2_inputs_eval"
CHECKPOINTS = [
    CHECKPOINT_DIR / "nc_rdis_s41" / "model_step_5000.pt",
    CHECKPOINT_DIR / "nc_rdis_s42" / "model_step_5000.pt",
    CHECKPOINT_DIR / "nc_rdis_s43" / "model_step_5000.pt",
]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp3_projection_ablation" / "nc_rdis.yaml"

# Dial settings (same as original Exp5)
DIAL_SCALES = [1.0, 0.85, 0.70, 0.55]

# Evaluation depths
N_TRAIN = 2
EVAL_N_LIST = [4, 8, 16]

# B0/B1 parameters
B0_COUNT = 100
B1_CAP = 1500

# Success evaluation parameters
SUCCESS_N_EPISODES = 100
SUCCESS_MAX_STEPS = 20

# Gate thresholds
G0_PROJECTION_THRESHOLD = 0.01

# Output
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_ready" / "exp5_tradeoff_curve_v2"


# =============================================================================
# Utility Functions
# =============================================================================

def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def stable_hash(tensor: torch.Tensor) -> str:
    """Compute stable hash for deduplication."""
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.md5(arr).hexdigest()[:16]


def count_empties(inputs: torch.Tensor) -> int:
    """Count empty cells (token 1 = empty)."""
    return int((inputs == 1).sum().item())


def extract_seed_from_path(checkpoint_path: str) -> int:
    """Extract seed from checkpoint path."""
    import re
    match = re.search(r'_s(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


# =============================================================================
# Placeholder for full implementation
# =============================================================================

def run_exp5_v2():
    """
    Run Exp5 v2 evaluation.

    This is a placeholder. The full implementation should:
    1. Load models from exp5_v2_inputs_eval checkpoints
    2. Apply inference-time contraction scaling
    3. Compute mismatch stability metrics (argmax agreement, ΔV)
    4. Compute success rate on trivial suite
    5. Generate tradeoff curve figure + table
    6. Write paper-ready artifacts

    For now, we verify checkpoints exist and create placeholders.
    """
    print("=" * 70)
    print("Exp5 v2: Stability-Expressivity Tradeoff (Fixed Dataset)")
    print("=" * 70)
    print()

    # Check checkpoints
    print("Checking checkpoints...")
    all_exist = True
    for ckpt in CHECKPOINTS:
        if ckpt.exists():
            print(f"  ✓ {ckpt}")
        else:
            print(f"  ✗ MISSING: {ckpt}")
            all_exist = False

    if not all_exist:
        print()
        print("ERROR: Some checkpoints missing. Run training first:")
        print("  python scripts/train_nc_rdis_v2.py")
        sys.exit(1)

    print()
    print("Checkpoints available. Running evaluation...")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # For now, import and run the original exp5 script with the new checkpoint paths
    # The actual implementation would be a full port, but for efficiency we can
    # modify the original script to accept checkpoint paths as arguments.

    from scripts.exp5_tradeoff_curve import main as exp5_main

    # Build args list
    sys.argv = [
        "exp5_tradeoff_curve_v2",
        "--checkpoints", str(CHECKPOINTS[0]), str(CHECKPOINTS[1]), str(CHECKPOINTS[2]),
        "--out_dir", str(OUTPUT_DIR),
    ]

    # Run the evaluation
    exp5_main()


def main():
    parser = argparse.ArgumentParser(description="Exp5 v2: Tradeoff Curve with Fixed Dataset")
    parser.add_argument("--check-only", action="store_true", help="Only check if checkpoints exist")
    args = parser.parse_args()

    if args.check_only:
        print("Checking checkpoint availability...")
        all_exist = all(ckpt.exists() for ckpt in CHECKPOINTS)
        if all_exist:
            print("All checkpoints available.")
            sys.exit(0)
        else:
            print("Some checkpoints missing.")
            sys.exit(1)

    run_exp5_v2()


if __name__ == "__main__":
    main()
