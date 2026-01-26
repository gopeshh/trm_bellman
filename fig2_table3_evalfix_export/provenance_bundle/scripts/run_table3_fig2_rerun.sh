#!/bin/bash
# Table 3 + Figure 2 RERUN (post-evaluator-fix)
# 18 trainings: 6 methods × 3 seeds
# 4 GPUs in parallel

set -e

RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_baselines_rerun_evalfix_2026_01_22"
DATASET="buiksat_trm/data/sudoku-4x4-trivial"

# Create results directory
mkdir -p "$RESULTS_DIR"
TRAIN_STEPS=5000
SEEDS=(42 123 456)

# Method definitions: key config_path
declare -A METHODS=(
    ["persistent_nc"]="buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml"
    ["episodic_nc"]="buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml"
    ["episodic_c_clean"]="buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml"
    ["ppo"]="buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml"
    ["a2c"]="buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
    ["dqn"]="buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"
)

# Order for batching
METHOD_ORDER=(persistent_nc episodic_nc episodic_c_clean ppo a2c dqn)

cd /home/buiksat/fbsource/fbcode

echo "============================================================"
echo "Table 3 + Figure 2 RERUN (post-evaluator-fix)"
echo "============================================================"
echo "Training steps: $TRAIN_STEPS"
echo "Dataset: $DATASET"
echo "Results: $RESULTS_DIR"
echo "Seeds: ${SEEDS[*]}"
echo "Methods: ${METHOD_ORDER[*]}"
echo "============================================================"
echo ""

# Build all jobs list
declare -a JOBS
for seed in "${SEEDS[@]}"; do
    for method in "${METHOD_ORDER[@]}"; do
        JOBS+=("${method}:${seed}:${METHODS[$method]}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs: $TOTAL_JOBS"
echo ""

# Function to run a single job
run_job() {
    local gpu=$1
    local method=$2
    local seed=$3
    local config=$4
    local log_file="$RESULTS_DIR/${method}_s${seed}.log"

    echo "[GPU $gpu] Starting $method seed=$seed"
    echo "  Config: $config"

    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
        -- \
        --config "$config" \
        --seed "$seed" \
        --dataset-paths "$DATASET" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        2>&1 | tee "$log_file"

    # Extract final success rate
    local final_success=$(grep "eval_success_rate=" "$log_file" | tail -1 | sed 's/.*eval_success_rate=\([0-9.]*\).*/\1/')
    echo "[GPU $gpu] Finished $method seed=$seed -> success=${final_success:-N/A}"
}

# Run jobs in batches of 4
batch_num=0
for ((i=0; i<TOTAL_JOBS; i+=4)); do
    batch_num=$((batch_num + 1))
    batch_end=$((i + 4))
    if [ $batch_end -gt $TOTAL_JOBS ]; then
        batch_end=$TOTAL_JOBS
    fi

    echo ""
    echo "=== Batch $batch_num: Jobs $((i+1)) to $batch_end of $TOTAL_JOBS ==="

    pids=()
    for ((j=i; j<batch_end; j++)); do
        gpu=$((j - i))
        IFS=':' read -r method seed config <<< "${JOBS[$j]}"
        run_job $gpu "$method" "$seed" "$config" &
        pids+=($!)
    done

    # Wait for batch to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "Batch $batch_num complete"
done

echo ""
echo "============================================================"
echo "All experiments complete"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"

# Summary
echo ""
echo "=== Final Success Rates ==="
for method in "${METHOD_ORDER[@]}"; do
    echo ""
    echo "$method:"
    for seed in "${SEEDS[@]}"; do
        log_file="$RESULTS_DIR/${method}_s${seed}.log"
        if [ -f "$log_file" ]; then
            final_success=$(grep "eval_success_rate=" "$log_file" | tail -1 | sed 's/.*eval_success_rate=\([0-9.]*\).*/\1/')
            echo "  seed $seed: ${final_success:-N/A}"
        else
            echo "  seed $seed: MISSING"
        fi
    done
done
