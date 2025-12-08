#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_sudoku_rl.sh [DATASET_DIR]
# Defaults to the augmented Sudoku-Extreme training split shipped with the repo.

DATASET_DIR=${1:-data/sudoku-extreme-1k-aug-1000}

python upi_trm_train.py \
    --dataset-paths "${DATASET_DIR}" \
    --config configs/rl_sudoku_k1.yaml

