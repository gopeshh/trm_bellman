#!/usr/bin/env bash
# =============================================================================
# Full-Featured UPI-TRM Training Script
# =============================================================================
# This script demonstrates all the new features:
#   - Puzzle embeddings (per-instance learnable vectors)
#   - Checkpoint loading (from pretrained) and saving
#   - WandB logging
#   - Configurable model architecture
#
# Usage:
#   ./scripts/run_sudoku_rl_full.sh                     # 4x4 ultra-easy
#   ./scripts/run_sudoku_rl_full.sh extreme             # 9x9 extreme
#   ./scripts/run_sudoku_rl_full.sh /path/to/checkpoint # Load pretrained
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
MODE=${1:-"4x4"}  # "4x4", "extreme", or path to checkpoint

# Model architecture (adjust to match pretrained if loading)
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
PUZZLE_EMB_NDIM=${PUZZLE_EMB_NDIM:-128}  # Set to 0 to disable
H_CYCLES=${H_CYCLES:-2}
L_CYCLES=${L_CYCLES:-4}
L_LAYERS=${L_LAYERS:-2}

# Training
TRAIN_STEPS=${TRAIN_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-128}
SEED=${SEED:-42}

# =============================================================================
# Select dataset and config based on mode
# =============================================================================
LOAD_CHECKPOINT=""

if [[ "$MODE" == "4x4" ]]; then
    DATASET="data/sudoku-4x4-ultra-easy"
    CONFIG="configs/rl_sudoku_4x4_full_features.yaml"
    RUN_NAME="4x4-ultra-easy-emb${PUZZLE_EMB_NDIM}-seed${SEED}"
    
elif [[ "$MODE" == "extreme" ]]; then
    DATASET="data/sudoku-extreme-1k-aug-1000"
    CONFIG="configs/rl_sudoku_k1.yaml"
    RUN_NAME="extreme-emb${PUZZLE_EMB_NDIM}-seed${SEED}"
    
elif [[ -f "$MODE" ]]; then
    # Assume it's a checkpoint path
    LOAD_CHECKPOINT="$MODE"
    DATASET="data/sudoku-extreme-1k-aug-1000"
    CONFIG="configs/rl_sudoku_k1.yaml"
    RUN_NAME="finetuned-emb${PUZZLE_EMB_NDIM}-seed${SEED}"
    echo "Loading checkpoint from: $LOAD_CHECKPOINT"
    
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [4x4|extreme|/path/to/checkpoint]"
    exit 1
fi

# =============================================================================
# Build command
# =============================================================================
CMD="python upi_trm_train.py"
CMD+=" --dataset-paths ${DATASET}"
CMD+=" --config ${CONFIG}"
CMD+=" --hidden-size ${HIDDEN_SIZE}"
CMD+=" --h-cycles ${H_CYCLES}"
CMD+=" --l-cycles ${L_CYCLES}"
CMD+=" --l-layers ${L_LAYERS}"
CMD+=" --puzzle-emb-ndim ${PUZZLE_EMB_NDIM}"
CMD+=" --puzzle-emb-len 16"
CMD+=" --puzzle-emb-lr 0.01"
CMD+=" --train-steps ${TRAIN_STEPS}"
CMD+=" --batch-size ${BATCH_SIZE}"
CMD+=" --seed ${SEED}"
CMD+=" --checkpoint-dir checkpoints/rl_${RUN_NAME}"
CMD+=" --save-interval 1000"
CMD+=" --wandb-project UPI-TRM-RL"
CMD+=" --wandb-run-name ${RUN_NAME}"

if [[ -n "$LOAD_CHECKPOINT" ]]; then
    CMD+=" --load-checkpoint ${LOAD_CHECKPOINT}"
fi

# =============================================================================
# Print and run
# =============================================================================
echo "=============================================="
echo "UPI-TRM Full-Featured Training"
echo "=============================================="
echo "Dataset: ${DATASET}"
echo "Config: ${CONFIG}"
echo "Hidden size: ${HIDDEN_SIZE}"
echo "Puzzle emb dim: ${PUZZLE_EMB_NDIM}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Seed: ${SEED}"
if [[ -n "$LOAD_CHECKPOINT" ]]; then
    echo "Checkpoint: ${LOAD_CHECKPOINT}"
fi
echo "=============================================="
echo ""
echo "Running: $CMD"
echo ""

exec $CMD

