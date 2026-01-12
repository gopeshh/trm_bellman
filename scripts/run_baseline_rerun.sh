#!/bin/bash
# Rerun PPO, A2C, DQN baselines with tuned hyperparameters
#
# Fixes applied to baseline configs:
# - Disabled contraction (baselines shouldn't have UPI-TRM theory features)
# - PPO: Lowered policy_lr 3e-4 -> 1e-4, reduced num_steps 128 -> 64
# - A2C: Increased num_steps 5 -> 32 (was causing high variance!)
# - DQN: Faster exploration decay, earlier learning start
#
# Usage: ./scripts/run_baseline_rerun.sh

set -e

FBCODE_DIR="$HOME/fbsource/fbcode"
LOG_ROOT="/home/buiksat/trm_bellman/runs/baseline_rerun"
DATASET="buiksat_trm/data/sudoku-4x4-trivial"
STEPS=5000
EVAL_INTERVAL=100
EVAL_EPISODES=50

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Baseline configs (algorithm:config_path:name)
BASELINES=(
    "ppo:buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml:ppo"
    "a2c:buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml:a2c"
    "dqn:buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml:dqn"
)
SEEDS=(42 123 456)

cd "$FBCODE_DIR"

mkdir -p "$LOG_ROOT"

echo "========================================"
echo "Baseline Rerun Experiments"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Dataset: $DATASET"
echo "Steps: $STEPS"
echo "Baselines: ${#BASELINES[@]}"
echo "Seeds: ${#SEEDS[@]}"
echo "Total jobs: $(( ${#BASELINES[@]} * ${#SEEDS[@]} ))"
echo "========================================"

# Array to store PIDs
declare -a PIDS=()
GPU=0

for baseline_info in "${BASELINES[@]}"; do
    IFS=':' read -r algo config name <<< "$baseline_info"

    for seed in "${SEEDS[@]}"; do
        LOG_DIR="$LOG_ROOT/$name"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/seed${seed}_${TIMESTAMP}.log"

        echo "[$(date +%H:%M:%S)] Launching: $name seed=$seed on GPU $GPU"
        echo "  Config: $config"
        echo "  Log: $LOG_FILE"

        CUDA_VISIBLE_DEVICES=$GPU buck2 run //buiksat_trm:upi_trm_train \
            -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
            --dataset-paths "$DATASET" \
            --config "$config" \
            --baseline "$algo" \
            --train-steps $STEPS --seed $seed \
            --eval-interval $EVAL_INTERVAL --eval-episodes $EVAL_EPISODES \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        GPU=$(( (GPU + 1) % 4 ))

        # Wait for wave of 4 to complete (if using 4 GPUs)
        if [ ${#PIDS[@]} -eq 4 ]; then
            echo "[$(date +%H:%M:%S)] Waiting for wave of 4 jobs..."
            wait "${PIDS[@]}"
            echo "[$(date +%H:%M:%S)] Wave complete!"
            PIDS=()
        fi
    done
done

# Wait for remaining jobs
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "[$(date +%H:%M:%S)] Waiting for final ${#PIDS[@]} jobs..."
    wait "${PIDS[@]}"
    echo "[$(date +%H:%M:%S)] Final wave complete!"
fi

echo ""
echo "========================================"
echo "All baseline experiments finished!"
echo "Timestamp: $(date)"
echo "Logs in: $LOG_ROOT"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Parse logs: python scripts/parse_feasibility_logs.py $LOG_ROOT"
echo "  2. Plot results: buck2 run //buiksat_trm:plot_feasibility_curves -- --input <csv> --output-dir results/plots_baseline_rerun"
