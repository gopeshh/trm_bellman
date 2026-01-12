#!/bin/bash
#
# Run contraction vs SGD tradeoff experiments
#
# Tests Brett's hypothesis: contraction enforcement may hurt performance
# because it acts like a hard constraint that fights SGD.
#
# 4 conditions on 4 GPUs in parallel:
#   GPU0: No contraction
#   GPU1: Weak contraction (target_Lz=0.99)
#   GPU2: Standard contraction (target_Lz=0.90)
#   GPU3: Scheduled contraction (two-phase: off for 3.5k, then on for 1.5k steps)
#
# Usage:
#   ./scripts/run_contraction_sgd_tradeoff.sh [SEED]
#
#   Default seed is 42.

set -e

SEED=${1:-42}
DATASET="data/sudoku-4x4-trivial"
BASE_CONFIG="configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml"
RESULTS_DIR="results/plot_data_contraction_sgd_tradeoff_light"
TOTAL_STEPS=5000
PHASE1_STEPS=3500

# Create output directory
mkdir -p "$RESULTS_DIR"

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo "Seed: $SEED"
echo "Dataset: $DATASET"
echo "Results: $RESULTS_DIR"
echo ""

# Function to run a standard experiment
run_experiment() {
    local name=$1
    local gpu=$2
    local override_config=$3
    local logfile="$RESULTS_DIR/${name}_seed${SEED}.log"

    echo "[${name}] Starting on GPU $gpu..."

    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 \
        -c fbcode.enable_gpu_sections=true \
        -- \
        --dataset-paths "$DATASET" \
        --config "$BASE_CONFIG" \
        --config "$override_config" \
        --seed "$SEED" \
        --train-steps "$TOTAL_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1 &

    echo "[${name}] PID=$! -> $logfile"
}

# Function to run scheduled experiment (two-phase)
run_scheduled_experiment() {
    local name="4_scheduled_contraction"
    local gpu=3
    local logfile="$RESULTS_DIR/${name}_seed${SEED}.log"
    local ckpt_dir="$RESULTS_DIR/ckpt_${name}_seed${SEED}"

    mkdir -p "$ckpt_dir"

    echo "[${name}] Starting PHASE 1 on GPU $gpu (no contraction, 0-$PHASE1_STEPS steps)..."

    # Phase 1: No contraction
    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 \
        -c fbcode.enable_gpu_sections=true \
        -- \
        --dataset-paths "$DATASET" \
        --config "$BASE_CONFIG" \
        --config "configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml" \
        --seed "$SEED" \
        --train-steps "$PHASE1_STEPS" \
        --checkpoint-dir "$ckpt_dir" \
        --save-interval "$PHASE1_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1

    echo "[${name}] Phase 1 complete. Starting PHASE 2 (contraction ON, $PHASE1_STEPS-$TOTAL_STEPS steps)..."

    # Find checkpoint
    CKPT_FILE=$(ls -t "$ckpt_dir"/*.pt 2>/dev/null | head -1)
    if [ -z "$CKPT_FILE" ]; then
        echo "[${name}] ERROR: No checkpoint found in $ckpt_dir"
        return 1
    fi

    # Phase 2: Contraction ON
    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 \
        -c fbcode.enable_gpu_sections=true \
        -- \
        --dataset-paths "$DATASET" \
        --config "$BASE_CONFIG" \
        --config "configs/experiments/contraction_sgd_tradeoff/3_standard_contraction.yaml" \
        --seed "$SEED" \
        --train-steps "$TOTAL_STEPS" \
        --resume-checkpoint "$CKPT_FILE" \
        --no-wandb \
        >> "$logfile" 2>&1 &

    echo "[${name}] PID=$! -> $logfile"
}

echo "=========================================="
echo "LAUNCHING EXPERIMENTS (seed=$SEED)"
echo "=========================================="
echo ""

# Launch conditions 1-3 in parallel
run_experiment "1_no_contraction" 0 \
    "configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml"

run_experiment "2_weak_contraction" 1 \
    "configs/experiments/contraction_sgd_tradeoff/2_weak_contraction.yaml"

run_experiment "3_standard_contraction" 2 \
    "configs/experiments/contraction_sgd_tradeoff/3_standard_contraction.yaml"

# Launch condition 4 (scheduled) - this blocks during phase 1
run_scheduled_experiment

echo ""
echo "=========================================="
echo "Waiting for all experiments to complete..."
echo "=========================================="

wait

echo ""
echo "All experiments complete!"
echo ""

# Parse results
echo "=========================================="
echo "RESULTS SUMMARY (seed=$SEED)"
echo "=========================================="
echo ""

for logfile in "$RESULTS_DIR"/*_seed${SEED}.log; do
    name=$(basename "$logfile" _seed${SEED}.log)

    # Extract final eval metrics
    final_line=$(grep -E "eval_success_rate=|success_rate=" "$logfile" | tail -1 || echo "")

    if [ -n "$final_line" ]; then
        success_rate=$(echo "$final_line" | grep -oP 'success_rate=\K[0-9.]+' || echo "N/A")
        mean_score=$(echo "$final_line" | grep -oP 'mean_score=\K[0-9.]+' || echo "N/A")
        echo "$name: success_rate=$success_rate mean_score=$mean_score"
    else
        echo "$name: No results found"
    fi
done

echo ""
echo "Raw logs in: $RESULTS_DIR/"
echo ""
echo "Next: Run plot script to generate figures"
echo "  python scripts/plot_contraction_sgd_tradeoff.py --results-dir $RESULTS_DIR"
