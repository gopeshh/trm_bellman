#!/bin/bash
set -euo pipefail

FBCODE_DIR="${FBCODE_DIR:-/data/repos/fbsource/fbcode}"
REPO_ROOT="${REPO_ROOT:-/home/buiksat/trm_bellman}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/data/sudoku-4x4-easy_6to8empties}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/neurips2026/external_hard4x4}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1000}"
ALGOS_STRING="${ALGOS:-ppo a2c}"
SEEDS_STRING="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

read -r -a ALGOS <<< "$ALGOS_STRING"
read -r -a SEEDS <<< "$SEEDS_STRING"

BUCK_FLAGS=()
if [[ "$DEVICE" == "cuda" ]]; then
    BUCK_FLAGS=(-c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true)
    if [[ "$LOCAL_ONLY" == "1" ]]; then
        BUCK_FLAGS+=(--local-only)
    fi
fi

is_complete_run() {
    local run_dir="$1"
    local run_config="$run_dir/run_config.json"
    local summary_json="$run_dir/train_summary.json"

    [[ -f "$run_config" && -f "$summary_json" && -f "$run_dir/model_final.zip" ]] || return 1
    grep -q "\"train_steps\": $TRAIN_STEPS" "$run_config" || return 1
    grep -q "\"device\": \"$DEVICE\"" "$run_config" || return 1
    return 0
}

has_incompatible_run() {
    local run_dir="$1"
    local run_config="$run_dir/run_config.json"

    [[ -f "$run_config" ]] || return 1
    grep -q "\"train_steps\": $TRAIN_STEPS" "$run_config" || return 0
    grep -q "\"device\": \"$DEVICE\"" "$run_config" || return 0
    return 1
}

archive_incompatible_run() {
    local run_dir="$1"
    local algo="$2"
    local seed="$3"

    if ! has_incompatible_run "$run_dir"; then
        return 0
    fi

    local archive_root="$OUTPUT_ROOT/_archived_runs"
    local archive_dir="$archive_root/${algo}_seed${seed}_$(date +%Y%m%d_%H%M%S)"

    mkdir -p "$archive_root"
    mv "$run_dir" "$archive_dir"
    mkdir -p "$run_dir"
    echo "  archived incompatible existing artifacts to $archive_dir"
}

cd "$FBCODE_DIR"

echo "========================================"
echo "External SB3 Baseline Sweep"
echo "========================================"
echo "Repo root: $REPO_ROOT"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_ROOT"
echo "Device: $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
    echo "GPU: $GPU"
fi
echo "Train steps: $TRAIN_STEPS"
echo "Eval freq: $EVAL_FREQ"
echo "Eval episodes: $EVAL_EPISODES"
echo "Checkpoint freq: $CHECKPOINT_FREQ"
echo "Algos: ${ALGOS[*]}"
echo "Seeds: ${SEEDS[*]}"
if [[ ${#BUCK_FLAGS[@]} -gt 0 ]]; then
    echo "Buck flags: ${BUCK_FLAGS[*]}"
fi
echo "========================================"

for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_dir="$OUTPUT_ROOT/$algo/seed$seed"
        if is_complete_run "$run_dir"; then
            echo "[$(date +%H:%M:%S)] Skipping completed algo=$algo seed=$seed"
            continue
        fi

        archive_incompatible_run "$run_dir" "$algo" "$seed"
        mkdir -p "$run_dir"
        log_file="$run_dir/train.log"

        echo "[$(date +%H:%M:%S)] Launching algo=$algo seed=$seed"
        echo "  log: $log_file"

        if [[ "$DEVICE" == "cuda" ]]; then
            CUDA_VISIBLE_DEVICES="$GPU" buck2 run fbcode//buiksat_trm:run_baseline "${BUCK_FLAGS[@]}" -- \
                --algo "$algo" \
                --env sudoku4x4 \
                --backend sb3 \
                --mode train \
                --seed "$seed" \
                --split train \
                --eval-split test \
                --dataset-dir "$DATASET_DIR" \
                --output-root "$OUTPUT_ROOT" \
                --device "$DEVICE" \
                --train-steps "$TRAIN_STEPS" \
                --eval-freq "$EVAL_FREQ" \
                --eval-episodes "$EVAL_EPISODES" \
                --checkpoint-freq "$CHECKPOINT_FREQ" \
                2>&1 | tee "$log_file"
        else
            buck2 run fbcode//buiksat_trm:run_baseline -- \
                --algo "$algo" \
                --env sudoku4x4 \
                --backend sb3 \
                --mode train \
                --seed "$seed" \
                --split train \
                --eval-split test \
                --dataset-dir "$DATASET_DIR" \
                --output-root "$OUTPUT_ROOT" \
                --device "$DEVICE" \
                --train-steps "$TRAIN_STEPS" \
                --eval-freq "$EVAL_FREQ" \
                --eval-episodes "$EVAL_EPISODES" \
                --checkpoint-freq "$CHECKPOINT_FREQ" \
                2>&1 | tee "$log_file"
        fi
    done
done

echo "========================================"
echo "External baseline sweep complete"
echo "========================================"
