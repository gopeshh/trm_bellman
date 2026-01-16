#!/bin/bash
# Multi-seed training for ICML Experiment 1 (Unroll Sensitivity)
# Trains Model A' (no contraction) and Model B (contraction) for seeds 41, 42, 43

set -e

# Configuration
SEEDS="41 42 43"
TRAIN_STEPS=5000
SAVE_INTERVAL=1000
DATA_PATH="/home/buiksat/trm_bellman/data/sudoku-4x4-trivial"

# Config paths
CONFIG_A="/home/buiksat/trm_bellman/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml"
CONFIG_B="/home/buiksat/trm_bellman/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml"

# Output base
CKPT_BASE="/home/buiksat/trm_bellman/checkpoints/exp1_v4"

echo "=== ICML Experiment 1: Multi-seed Training ==="
echo "Seeds: $SEEDS"
echo "Steps: $TRAIN_STEPS"
echo ""

# Create checkpoint directories
mkdir -p "$CKPT_BASE"

for SEED in $SEEDS; do
    echo "========================================"
    echo "Seed: $SEED"
    echo "========================================"

    # Train Model A' (no contraction)
    CKPT_DIR_A="$CKPT_BASE/model_a_prime/seed$SEED"
    if [ -f "$CKPT_DIR_A/model_step_${TRAIN_STEPS}.pt" ]; then
        echo "[Model A'] Seed $SEED: Already trained, skipping"
    else
        echo "[Model A'] Seed $SEED: Training..."
        mkdir -p "$CKPT_DIR_A"
        cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES="" buck2 run //buiksat_trm:upi_trm_train -- \
            --seed "$SEED" \
            --train-steps "$TRAIN_STEPS" \
            --config "$CONFIG_A" \
            --checkpoint-dir "$CKPT_DIR_A" \
            --save-interval "$SAVE_INTERVAL" \
            --dataset-paths "$DATA_PATH" \
            2>&1 | tee "$CKPT_DIR_A/training.log"
        echo "[Model A'] Seed $SEED: Done"
    fi

    # Train Model B (contraction)
    CKPT_DIR_B="$CKPT_BASE/model_b/seed$SEED"
    if [ -f "$CKPT_DIR_B/model_step_${TRAIN_STEPS}.pt" ]; then
        echo "[Model B] Seed $SEED: Already trained, skipping"
    else
        echo "[Model B] Seed $SEED: Training..."
        mkdir -p "$CKPT_DIR_B"
        cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES="" buck2 run //buiksat_trm:upi_trm_train -- \
            --seed "$SEED" \
            --train-steps "$TRAIN_STEPS" \
            --config "$CONFIG_B" \
            --checkpoint-dir "$CKPT_DIR_B" \
            --save-interval "$SAVE_INTERVAL" \
            --dataset-paths "$DATA_PATH" \
            2>&1 | tee "$CKPT_DIR_B/training.log"
        echo "[Model B] Seed $SEED: Done"
    fi

    echo ""
done

echo "=== Training Complete ==="
echo "Checkpoints saved to: $CKPT_BASE"
echo ""
echo "Model A' (no contraction):"
ls -la "$CKPT_BASE/model_a_prime/"
echo ""
echo "Model B (contraction):"
ls -la "$CKPT_BASE/model_b/"
