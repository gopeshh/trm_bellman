#!/bin/bash
# Run ALL ablations on 6-8 empties dataset (20k steps)
# 7 configs × 3 seeds = 21 runs, 4 at a time on 4 GPUs

set -e

FBCODE_DIR="$HOME/fbsource/fbcode"
LOG_ROOT="/home/buiksat/trm_bellman/runs/feasibility_6to8empties_all_ablations_20k"
DATASET="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
STEPS=20000
EVAL_INTERVAL=200
EVAL_EPISODES=50

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configs: (algo_key, yaml_path)
declare -a CONFIGS=(
    "upi_trm:buiksat_trm/configs/rl_sudoku_4x4_feasibility.yaml"
    "no_contraction:buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml"
    "ablation_persistent_z:buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z.yaml"
    "persistent_z_no_contraction:buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml"
    "ablation_no_conservative:buiksat_trm/configs/ablations/upi_trm_feasibility_no_conservative.yaml"
    "a2c:buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
    "dqn:buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"
)
SEEDS=(42 123 456)
NUM_GPUS=4

cd "$FBCODE_DIR"

echo "Starting 6-8 empties ALL ablations (20k steps) at $TIMESTAMP"
echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total jobs: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

# Build full job list
declare -a JOBS=()
for config_pair in "${CONFIGS[@]}"; do
    algo="${config_pair%%:*}"
    yaml="${config_pair##*:}"
    for seed in "${SEEDS[@]}"; do
        JOBS+=("$algo:$yaml:$seed")
    done
done

# Run jobs 4 at a time
declare -a PIDS=()
GPU=0
for job in "${JOBS[@]}"; do
    algo="${job%%:*}"
    rest="${job#*:}"
    yaml="${rest%%:*}"
    seed="${rest##*:}"

    LOG_FILE="$LOG_ROOT/$algo/${seed}_${TIMESTAMP}.log"

    echo "Launching: $algo seed=$seed on GPU $GPU"
    echo "  Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" \
        --config "$yaml" \
        --train-steps $STEPS --seed $seed \
        --eval-interval $EVAL_INTERVAL --eval-episodes $EVAL_EPISODES \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    GPU=$(( (GPU + 1) % NUM_GPUS ))

    # Wait for wave of NUM_GPUS to complete
    if [ ${#PIDS[@]} -eq $NUM_GPUS ]; then
        echo "Waiting for wave of $NUM_GPUS jobs..."
        wait "${PIDS[@]}"
        echo "Wave complete!"
        PIDS=()
    fi
done

# Wait for remaining jobs
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for final ${#PIDS[@]} jobs..."
    wait "${PIDS[@]}"
    echo "Final wave complete!"
fi

echo "All experiments finished at $(date)"
echo "Logs in: $LOG_ROOT"
