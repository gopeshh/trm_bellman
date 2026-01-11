#!/bin/bash
# Run feasibility ablation experiments to validate contraction fix
# Usage: ./scripts/run_contraction_fix_rerun.sh

set -e

FBCODE_DIR="$HOME/fbsource/fbcode"
LOG_ROOT="/home/buiksat/trm_bellman/runs/feasibility_contraction_fix_rerun"
DATASET="buiksat_trm/data/sudoku-4x4-trivial"
STEPS=5000
EVAL_INTERVAL=100
EVAL_EPISODES=50

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configs and names
CONFIGS=(
    "buiksat_trm/configs/rl_sudoku_4x4_feasibility.yaml:rl_sudoku_4x4_feasibility"
    "buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z.yaml:upi_trm_feasibility_persistent_z"
    "buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml:upi_trm_feasibility_persistent_z_no_contraction"
)
SEEDS=(42 123 456)

cd "$FBCODE_DIR"

echo "Starting contraction fix rerun experiments at $TIMESTAMP"
echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total jobs: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

# Array to store PIDs
declare -a PIDS=()
GPU=0

for config_pair in "${CONFIGS[@]}"; do
    config="${config_pair%%:*}"
    name="${config_pair##*:}"

    for seed in "${SEEDS[@]}"; do
        LOG_DIR="$LOG_ROOT/$name"
        LOG_FILE="$LOG_DIR/${seed}_${TIMESTAMP}.log"

        echo "Launching: $name seed=$seed on GPU $GPU"
        echo "  Log: $LOG_FILE"

        CUDA_VISIBLE_DEVICES=$GPU buck2 run //buiksat_trm:upi_trm_train \
            -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
            --dataset-paths "$DATASET" \
            --config "$config" \
            --train-steps $STEPS --seed $seed \
            --eval-interval $EVAL_INTERVAL --eval-episodes $EVAL_EPISODES \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        GPU=$(( (GPU + 1) % 4 ))

        # Wait for wave of 4 to complete
        if [ ${#PIDS[@]} -eq 4 ]; then
            echo "Waiting for wave of 4 jobs..."
            wait "${PIDS[@]}"
            echo "Wave complete!"
            PIDS=()
        fi
    done
done

# Wait for remaining jobs
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for final ${#PIDS[@]} jobs..."
    wait "${PIDS[@]}"
    echo "Final wave complete!"
fi

echo "All experiments finished at $(date)"
echo "Logs in: $LOG_ROOT"
