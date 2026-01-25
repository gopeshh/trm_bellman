#!/bin/bash
# Re-run PPO/A2C/DQN baselines on harder 4x4 Sudoku (6-8 empties)
# After eval bug fix
# Uses 4 GPUs in parallel, 20k training steps

set -e

cd ~/fbsource/fbcode

DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_hard_6to8"
TRAIN_STEPS=20000

mkdir -p "$RESULTS_DIR"

echo "=== Re-running PPO/A2C/DQN baselines on HARDER dataset (6-8 empties) ==="
echo "Training steps: $TRAIN_STEPS"
echo "Dataset: $DATA_PATH"
echo "Results: $RESULTS_DIR"
echo ""

# Backup old logs
echo "Backing up old logs..."
for method in ppo a2c dqn; do
    for seed in 42 123 456; do
        logfile="$RESULTS_DIR/${method}_s${seed}.log"
        if [ -f "$logfile" ]; then
            mv "$logfile" "${logfile}.bak"
        fi
    done
done

# Configs for each method
declare -A CONFIGS
CONFIGS["ppo"]="buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml"
CONFIGS["a2c"]="buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
CONFIGS["dqn"]="buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"

SEEDS=(42 123 456)

run_experiment() {
    local gpu=$1
    local method=$2
    local seed=$3
    local config="${CONFIGS[$method]}"
    local logfile="$RESULTS_DIR/${method}_s${seed}.log"

    echo "[GPU $gpu] Starting $method seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        -- --config "$config" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1

    # Extract final success rate
    local final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
    echo "[GPU $gpu] Finished $method seed=$seed -> success=$final_success"
}

echo "=== Batch 1/3: ppo (3 seeds) + a2c seed 42 ==="
run_experiment 0 "ppo" 42 &
run_experiment 1 "ppo" 123 &
run_experiment 2 "ppo" 456 &
run_experiment 3 "a2c" 42 &
wait
echo "Batch 1 complete"

echo "=== Batch 2/3: a2c (2 seeds) + dqn (2 seeds) ==="
run_experiment 0 "a2c" 123 &
run_experiment 1 "a2c" 456 &
run_experiment 2 "dqn" 42 &
run_experiment 3 "dqn" 123 &
wait
echo "Batch 2 complete"

echo "=== Batch 3/3: dqn seed 456 ==="
run_experiment 0 "dqn" 456 &
wait
echo "Batch 3 complete"

echo ""
echo "=== All baseline experiments complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""

# Print summary
echo "=== Final Results Summary ==="
for method in ppo a2c dqn; do
    echo "$method:"
    for seed in 42 123 456; do
        logfile="$RESULTS_DIR/${method}_s${seed}.log"
        if [ -f "$logfile" ]; then
            final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            echo "  seed $seed: $final_success"
        fi
    done
done
