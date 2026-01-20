#!/bin/bash
# Run Table 3 experiments on harder 4x4 Sudoku (6-8 empties)
# Uses 4 GPUs in parallel, 20k training steps
# Same configs as trivial dataset experiments

set -e

cd ~/fbsource/fbcode

DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_hard_6to8"
TRAIN_STEPS=20000

mkdir -p "$RESULTS_DIR"

echo "=== Running Table 3 experiments on HARDER dataset (6-8 empties) ==="
echo "Training steps: $TRAIN_STEPS"
echo "Dataset: $DATA_PATH"
echo "Results: $RESULTS_DIR"
echo ""

# Configs for each method
declare -A CONFIGS
CONFIGS["persistent_nc"]="buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml"
CONFIGS["episodic_nc"]="buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml"
CONFIGS["episodic_c_clean"]="buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml"
CONFIGS["ppo"]="buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml"
CONFIGS["a2c"]="buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
CONFIGS["dqn"]="buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"

SEEDS=(42 123 456)

# Run experiments in batches of 4 (one per GPU)
# Total: 6 methods × 3 seeds = 18 experiments

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

echo "=== Batch 1/5: persistent_nc (3 seeds) + episodic_nc (1 seed) ==="
run_experiment 0 "persistent_nc" 42 &
run_experiment 1 "persistent_nc" 123 &
run_experiment 2 "persistent_nc" 456 &
run_experiment 3 "episodic_nc" 42 &
wait
echo "Batch 1 complete"

echo "=== Batch 2/5: episodic_nc (2 seeds) + episodic_c_clean (2 seeds) ==="
run_experiment 0 "episodic_nc" 123 &
run_experiment 1 "episodic_nc" 456 &
run_experiment 2 "episodic_c_clean" 42 &
run_experiment 3 "episodic_c_clean" 123 &
wait
echo "Batch 2 complete"

echo "=== Batch 3/5: episodic_c_clean (1 seed) + ppo (3 seeds) ==="
run_experiment 0 "episodic_c_clean" 456 &
run_experiment 1 "ppo" 42 &
run_experiment 2 "ppo" 123 &
run_experiment 3 "ppo" 456 &
wait
echo "Batch 3 complete"

echo "=== Batch 4/5: a2c (3 seeds) + dqn (1 seed) ==="
run_experiment 0 "a2c" 42 &
run_experiment 1 "a2c" 123 &
run_experiment 2 "a2c" 456 &
run_experiment 3 "dqn" 42 &
wait
echo "Batch 4 complete"

echo "=== Batch 5/5: dqn (2 seeds) ==="
run_experiment 0 "dqn" 123 &
run_experiment 1 "dqn" 456 &
wait
echo "Batch 5 complete"

echo ""
echo "=== All experiments complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""

# Print summary
echo "=== Final Results Summary ==="
for method in persistent_nc episodic_nc episodic_c_clean ppo a2c dqn; do
    echo "$method:"
    for seed in 42 123 456; do
        logfile="$RESULTS_DIR/${method}_s${seed}.log"
        if [ -f "$logfile" ]; then
            final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            echo "  seed $seed: $final_success"
        fi
    done
done
