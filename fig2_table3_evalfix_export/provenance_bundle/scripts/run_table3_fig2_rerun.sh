#!/bin/bash
#
# Table 3 + Figure 2 rerun protocol after the baseline audits.
#
# Audit findings addressed here:
# 1. "5k train steps" was not comparable across baselines because PPO/A2C/DQN
#    consume 64 / 32 / 4 env interactions per outer step respectively.
# 2. The provenance plot labeled these outer steps as "Training Steps", which
#    overstated cross-baseline comparability.
# 3. Reviewer hD9M requested an n-step DQN baseline stronger than vanilla DQN.
#
# This script equalizes the baseline interaction budget at PPO's historical
# 320k env interactions while keeping the existing UPI-TRM budget unchanged.

set -euo pipefail

RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_baselines_rerun_envbudget_2026_04_24"
DATASET="buiksat_trm/data/sudoku-4x4-trivial"
ENV_INTERACTION_BUDGET=320000  # PPO historical baseline: 5000 outer steps × 64 env steps

if [ -d "/home/buiksat/fbsource/fbcode" ]; then
    FBSOURCE_ROOT="/home/buiksat/fbsource/fbcode"
elif [ -d "/data/repos/fbsource/fbcode" ]; then
    FBSOURCE_ROOT="/data/repos/fbsource/fbcode"
else
    echo "Could not find fbsource/fbcode checkout." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
else
    GPU_COUNT=1
fi
if [ "${GPU_COUNT:-0}" -lt 1 ]; then
    GPU_COUNT=1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
SEEDS=(42 123 456)

# Order for batching
METHOD_ORDER=(persistent_nc episodic_nc episodic_c_clean ppo a2c dqn dqn_nstep5)

get_train_steps() {
    local method=$1
    case "$method" in
        ppo)
            echo $((ENV_INTERACTION_BUDGET / 64))
            ;;
        a2c)
            echo $((ENV_INTERACTION_BUDGET / 32))
            ;;
        dqn|dqn_nstep5)
            echo $((ENV_INTERACTION_BUDGET / 4))
            ;;
        persistent_nc|episodic_nc|episodic_c_clean)
            echo 5000
            ;;
        *)
            echo "Unknown method for train-step budget: $method" >&2
            return 1
            ;;
    esac
}

get_env_steps_per_outer() {
    local method=$1
    case "$method" in
        ppo)
            echo 64
            ;;
        a2c)
            echo 32
            ;;
        dqn|dqn_nstep5)
            echo 4
            ;;
        persistent_nc|episodic_nc|episodic_c_clean)
            echo 1
            ;;
        *)
            echo "Unknown method for env-step conversion: $method" >&2
            return 1
            ;;
    esac
}

cd "$FBSOURCE_ROOT"

echo "============================================================"
echo "Table 3 + Figure 2 RERUN (interaction-budget aligned)"
echo "============================================================"
echo "Baseline env-interaction budget: $ENV_INTERACTION_BUDGET"
echo "fbsource root: $FBSOURCE_ROOT"
echo "Detected GPUs: $GPU_COUNT"
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
        JOBS+=("${method}:${seed}")
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
    local log_file="$RESULTS_DIR/${method}_s${seed}.log"
    local train_steps
    local env_steps_per_outer
    local effective_env_interactions
    local config_args=()

    train_steps=$(get_train_steps "$method")
    env_steps_per_outer=$(get_env_steps_per_outer "$method")
    effective_env_interactions=$((train_steps * env_steps_per_outer))

    case "$method" in
        persistent_nc)
            config_args=(
                --config "buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml"
            )
            ;;
        episodic_nc)
            config_args=(
                --config "buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml"
            )
            ;;
        episodic_c_clean)
            config_args=(
                --config "buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml"
            )
            ;;
        ppo)
            config_args=(
                --config "buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml"
            )
            ;;
        a2c)
            config_args=(
                --config "buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
            )
            ;;
        dqn)
            config_args=(
                --config "buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"
            )
            ;;
        dqn_nstep5)
            config_args=(
                --config "buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"
                --config "buiksat_trm/configs/baselines/dqn_trm_feasibility_nstep5.yaml"
            )
            ;;
        *)
            echo "Unknown method: $method" >&2
            return 1
            ;;
    esac

    echo "[GPU $gpu] Starting $method seed=$seed"
    echo "  Train steps: $train_steps"
    echo "  Env steps / outer step: $env_steps_per_outer"
    echo "  Effective env interactions: $effective_env_interactions"

    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
        -- \
        "${config_args[@]}" \
        --seed "$seed" \
        --dataset-paths "$DATASET" \
        --train-steps "$train_steps" \
        --no-wandb \
        2>&1 | tee "$log_file"

    # Extract final success rate
    local final_success=$(grep "eval_success_rate=" "$log_file" | tail -1 | sed 's/.*eval_success_rate=\([0-9.]*\).*/\1/')
    echo "[GPU $gpu] Finished $method seed=$seed -> success=${final_success:-N/A}"
}

# Run jobs in GPU-sized batches
batch_num=0
for ((i=0; i<TOTAL_JOBS; i+=GPU_COUNT)); do
    batch_num=$((batch_num + 1))
    batch_end=$((i + GPU_COUNT))
    if [ $batch_end -gt $TOTAL_JOBS ]; then
        batch_end=$TOTAL_JOBS
    fi

    echo ""
    echo "=== Batch $batch_num: Jobs $((i+1)) to $batch_end of $TOTAL_JOBS ==="

    pids=()
    for ((j=i; j<batch_end; j++)); do
        gpu=$((j - i))
        IFS=':' read -r method seed <<< "${JOBS[$j]}"
        run_job $gpu "$method" "$seed" &
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
