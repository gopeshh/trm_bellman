#!/bin/bash
# Run fresh no-mask hard-4x4 architecture-matched baselines across 4 GPUs.
#
# Methods:
#   - TRM + PPO
#   - TRM + A2C
#   - TRM + DQN
#
# Protocol:
#   - dataset: sudoku-4x4-easy_6to8empties
#   - steps: 20000
#   - seeds: 0..9
#   - no-mask via configs/table3_hard_controlled/no_mask_overlay.yaml
#
# Logs are written to a fresh results directory to avoid mixing with the older
# 3-seed masked table3_hard_6to8 runs. Checkpoints are also isolated per
# method/seed pair.

set -euo pipefail

FBCODE_DIR="$HOME/fbsource/fbcode"
WORKTREE="/home/buiksat/trm_bellman"
DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
RESULTS_DIR="$WORKTREE/results/hard4x4_trm_baselines_nomask_10seed"
CHECKPOINT_ROOT="$WORKTREE/checkpoints/hard4x4_trm_baselines_nomask_10seed"
OVERLAY_CONFIG="buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml"
TRAIN_STEPS=20000
SEEDS=(0 1 2 3 4 5 6 7 8 9)
METHODS=(ppo a2c dqn)

declare -A CONFIGS
CONFIGS["ppo"]="buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml"
CONFIGS["a2c"]="buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
CONFIGS["dqn"]="buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"

# Coarse runtime estimates used only to balance fixed per-GPU queues.
declare -A ESTIMATED_HOURS
ESTIMATED_HOURS["ppo"]=10
ESTIMATED_HOURS["a2c"]=6
ESTIMATED_HOURS["dqn"]=2

declare -a GPU_QUEUE_0=()
declare -a GPU_QUEUE_1=()
declare -a GPU_QUEUE_2=()
declare -a GPU_QUEUE_3=()
GPU_LOADS=(0 0 0 0)

MANIFEST="$RESULTS_DIR/job_manifest.tsv"
FAILURES_FILE="$RESULTS_DIR/failed_jobs.txt"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_ROOT"
: > "$FAILURES_FILE"

append_job_to_gpu() {
    local gpu=$1
    local job=$2
    case "$gpu" in
        0) GPU_QUEUE_0+=("$job") ;;
        1) GPU_QUEUE_1+=("$job") ;;
        2) GPU_QUEUE_2+=("$job") ;;
        3) GPU_QUEUE_3+=("$job") ;;
        *) echo "Invalid GPU index: $gpu" >&2; exit 1 ;;
    esac
}

get_queue_jobs() {
    local gpu=$1
    case "$gpu" in
        0) printf "%s\n" "${GPU_QUEUE_0[@]}" ;;
        1) printf "%s\n" "${GPU_QUEUE_1[@]}" ;;
        2) printf "%s\n" "${GPU_QUEUE_2[@]}" ;;
        3) printf "%s\n" "${GPU_QUEUE_3[@]}" ;;
        *) return 1 ;;
    esac
}

extract_final_success() {
    local logfile=$1
    local success=""
    success=$(rg -o "eval_success_rate=[0-9.]+" "$logfile" 2>/dev/null | tail -1 | cut -d= -f2 || true)
    if [ -n "$success" ]; then
        printf "%s" "$success"
    else
        printf "N/A"
    fi
}

is_complete() {
    local logfile=$1
    [ -f "$logfile" ] && rg -q "\[step 20000\] eval_success_rate" "$logfile"
}

assign_jobs() {
    local method
    local seed
    local est_hours
    local min_gpu
    local gpu
    local job

    # Longest jobs first to reduce the final tail.
    for method in ppo a2c dqn; do
        est_hours=${ESTIMATED_HOURS[$method]}
        for seed in "${SEEDS[@]}"; do
            job="${method}:${seed}:${est_hours}"
            min_gpu=0
            for gpu in 1 2 3; do
                if [ "${GPU_LOADS[$gpu]}" -lt "${GPU_LOADS[$min_gpu]}" ]; then
                    min_gpu=$gpu
                fi
            done
            append_job_to_gpu "$min_gpu" "$job"
            GPU_LOADS[$min_gpu]=$(( GPU_LOADS[$min_gpu] + est_hours ))
        done
    done
}

write_manifest() {
    printf "gpu\tmethod\tseed\testimated_hours\tconfig\toverlay\tlogfile\tcheckpoint_dir\n" > "$MANIFEST"
    local gpu
    local job
    local method
    local seed
    local est_hours
    local logfile
    local checkpoint_dir
    for gpu in 0 1 2 3; do
        while IFS= read -r job; do
            [ -n "$job" ] || continue
            IFS=":" read -r method seed est_hours <<< "$job"
            logfile="$RESULTS_DIR/${method}_nomask_s${seed}.log"
            checkpoint_dir="$CHECKPOINT_ROOT/${method}_s${seed}"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$gpu" \
                "$method" \
                "$seed" \
                "$est_hours" \
                "${CONFIGS[$method]}" \
                "$OVERLAY_CONFIG" \
                "$logfile" \
                "$checkpoint_dir" >> "$MANIFEST"
        done < <(get_queue_jobs "$gpu")
    done
}

print_schedule() {
    local gpu
    local job
    echo "============================================================"
    echo "Fresh hard-4x4 TRM baselines (no-mask, 10 seeds)"
    echo "============================================================"
    echo "Training steps: $TRAIN_STEPS"
    echo "Dataset: $DATA_PATH"
    echo "Results: $RESULTS_DIR"
    echo "Checkpoints: $CHECKPOINT_ROOT"
    echo "Overlay: $OVERLAY_CONFIG"
    echo "Seeds: ${SEEDS[*]}"
    echo "Manifest: $MANIFEST"
    echo "============================================================"
    echo ""
    for gpu in 0 1 2 3; do
        echo "[GPU $gpu] Estimated load: ${GPU_LOADS[$gpu]}h"
        while IFS= read -r job; do
            [ -n "$job" ] || continue
            echo "  - $job"
        done < <(get_queue_jobs "$gpu")
        echo ""
    done
}

run_experiment() {
    local gpu=$1
    local method=$2
    local seed=$3
    local config="${CONFIGS[$method]}"
    local logfile="$RESULTS_DIR/${method}_nomask_s${seed}.log"
    local checkpoint_dir="$CHECKPOINT_ROOT/${method}_s${seed}"

    if is_complete "$logfile"; then
        echo "[GPU $gpu] Skipping $method seed=$seed (already complete, success=$(extract_final_success "$logfile"))"
        return 0
    fi

    if [ -f "$logfile" ]; then
        mv "$logfile" "${logfile}.partial.$(date +%Y%m%d_%H%M%S)"
    fi

    mkdir -p "$checkpoint_dir"

    echo "[GPU $gpu] Starting $method seed=$seed"
    echo "  Config: $config"
    echo "  Overlay: $OVERLAY_CONFIG"
    echo "  Log: $logfile"
    echo "  Checkpoint: $checkpoint_dir"

    if CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        --local-only \
        -- --config "$config" \
        --config "$OVERLAY_CONFIG" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --checkpoint-dir "$checkpoint_dir" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] Finished $method seed=$seed -> success=$(extract_final_success "$logfile")"
    else
        echo "${method}:${seed}:${logfile}" >> "$FAILURES_FILE"
        echo "[GPU $gpu] FAILED $method seed=$seed (see $logfile)"
        return 1
    fi
}

run_gpu_queue() {
    local gpu=$1
    shift
    local job
    local method
    local seed
    local est_hours

    for job in "$@"; do
        IFS=":" read -r method seed est_hours <<< "$job"
        if ! run_experiment "$gpu" "$method" "$seed"; then
            :
        fi
    done
}

print_final_summary() {
    local method
    local seed
    local logfile
    echo ""
    echo "============================================================"
    echo "Run complete"
    echo "Results: $RESULTS_DIR"
    echo "============================================================"
    echo ""
    for method in "${METHODS[@]}"; do
        echo "$method:"
        for seed in "${SEEDS[@]}"; do
            logfile="$RESULTS_DIR/${method}_nomask_s${seed}.log"
            if is_complete "$logfile"; then
                echo "  seed $seed: $(extract_final_success "$logfile")"
            elif [ -f "$logfile" ]; then
                echo "  seed $seed: INCOMPLETE"
            else
                echo "  seed $seed: MISSING"
            fi
        done
        echo ""
    done

    if [ -s "$FAILURES_FILE" ]; then
        echo "Failures:"
        cat "$FAILURES_FILE"
    else
        echo "Failures: none"
    fi
}

cd "$FBCODE_DIR"

assign_jobs
write_manifest
print_schedule

run_gpu_queue 0 "${GPU_QUEUE_0[@]}" &
PID0=$!
run_gpu_queue 1 "${GPU_QUEUE_1[@]}" &
PID1=$!
run_gpu_queue 2 "${GPU_QUEUE_2[@]}" &
PID2=$!
run_gpu_queue 3 "${GPU_QUEUE_3[@]}" &
PID3=$!

wait "$PID0" "$PID1" "$PID2" "$PID3"

print_final_summary
