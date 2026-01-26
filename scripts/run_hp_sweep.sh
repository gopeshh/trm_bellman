#!/bin/bash
# Hyperparameter sweep for baselines on hard 4x4 Sudoku (6-8 empties)
# Uses 4 GPUs in parallel, 20k training steps, seed 42 for initial sweep

set -e

cd ~/fbsource/fbcode

DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
RESULTS_DIR="/home/buiksat/trm_bellman/results/hp_tuning_sweep"
TRAIN_STEPS=20000
SEED=42

mkdir -p "$RESULTS_DIR"

echo "=== Hyperparameter Sweep for Baselines on HARD dataset (6-8 empties) ==="
echo "Training steps: $TRAIN_STEPS"
echo "Dataset: $DATA_PATH"
echo "Results: $RESULTS_DIR"
echo "Seed: $SEED"
echo ""

# Config directory - use fbcode path for buck2
CONFIG_DIR="buiksat_trm/configs/hp_tuning"

run_experiment() {
    local gpu=$1
    local config=$2
    local name=$3
    local logfile="$RESULTS_DIR/${name}_s${SEED}.log"

    echo "[GPU $gpu] Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        -- --config "$config" \
        --seed "$SEED" \
        --dataset-paths "$DATA_PATH" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1

    # Extract final success rate
    local final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+" || echo "N/A")
    echo "[GPU $gpu] Finished $name -> success=$final_success"
}

echo "=== Batch 1/3: PPO variants ==="
run_experiment 0 "$CONFIG_DIR/ppo_high_entropy.yaml" "ppo_high_entropy" &
run_experiment 1 "$CONFIG_DIR/ppo_deep_unroll.yaml" "ppo_deep_unroll" &
run_experiment 2 "$CONFIG_DIR/ppo_high_lr_epochs.yaml" "ppo_high_lr_epochs" &
run_experiment 3 "$CONFIG_DIR/ppo_strong_reward.yaml" "ppo_strong_reward" &
wait
echo "Batch 1 complete"

echo "=== Batch 2/3: A2C variants ==="
run_experiment 0 "$CONFIG_DIR/a2c_high_entropy.yaml" "a2c_high_entropy" &
run_experiment 1 "$CONFIG_DIR/a2c_deep_unroll.yaml" "a2c_deep_unroll" &
run_experiment 2 "$CONFIG_DIR/a2c_high_lr_rollout.yaml" "a2c_high_lr_rollout" &
run_experiment 3 "$CONFIG_DIR/a2c_strong_reward.yaml" "a2c_strong_reward" &
wait
echo "Batch 2 complete"

echo "=== Batch 3/3: DQN variants ==="
run_experiment 0 "$CONFIG_DIR/dqn_slow_explore.yaml" "dqn_slow_explore" &
run_experiment 1 "$CONFIG_DIR/dqn_deep_unroll.yaml" "dqn_deep_unroll" &
run_experiment 2 "$CONFIG_DIR/dqn_large_buffer.yaml" "dqn_large_buffer" &
run_experiment 3 "$CONFIG_DIR/dqn_strong_reward.yaml" "dqn_strong_reward" &
wait
echo "Batch 3 complete"

echo ""
echo "=== All sweep experiments complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""

# Print summary table
echo "=== Sweep Results Summary ==="
echo "Config | Success Rate"
echo "-------|-------------"
for logfile in "$RESULTS_DIR"/*_s${SEED}.log; do
    if [ -f "$logfile" ]; then
        name=$(basename "$logfile" "_s${SEED}.log")
        final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+" || echo "N/A")
        echo "$name | $final_success"
    fi
done
