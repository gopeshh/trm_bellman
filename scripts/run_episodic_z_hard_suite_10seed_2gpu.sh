#!/bin/bash
# Launch the theorem-aligned episodic-z hard-suite rerun across 2 GPUs.
#
# Protocol:
#   - dataset: sudoku-4x4-easy_6to8empties
#   - env-step budget: 20000
#   - diagnostics/eval/checkpoints: 5000 / 10000 / 15000 / 20000 env steps
#   - seeds: 41..50
#   - latent mode: episodic-z
#   - config: exact-baseline / theory-exact mixture revision config

set -euo pipefail

FBCODE_DIR="$HOME/fbsource/fbcode"
FBSOURCE_ROOT="$HOME/fbsource"
CODE_ROOT="/home/buiksat/trm_bellman"
PAPER_ROOT="/home/buiksat/UPI_TRM/UPI_TRM_NIPS"

DATA_PATH="$CODE_ROOT/data/sudoku-4x4-easy_6to8empties"
CONFIG_PATH="$CODE_ROOT/configs/revision/upi_trm_feasibility_episodic_z_hard_suite_theory_exact.yaml"

RESULTS_ROOT="$PAPER_ROOT/results/episodic_z_hard_suite_20k_seed41_50"
LOG_DIR="$RESULTS_ROOT/logs"
CHECKPOINT_ROOT="$CODE_ROOT/checkpoints/episodic_z_hard_suite_20k_seed41_50"
MANIFEST="$RESULTS_ROOT/job_manifest.tsv"
FAILURES_FILE="$RESULTS_ROOT/failed_jobs.txt"

ENV_STEP_BUDGET=20000
INTERVAL=5000
SEEDS=(41 42 43 44 45 46 47 48 49 50)

mkdir -p "$LOG_DIR" "$CHECKPOINT_ROOT"
: > "$FAILURES_FILE"

echo "Building upi_trm_train..."
cd "$FBSOURCE_ROOT" || exit 1
RUNNER_OUTPUT=$(buck2 build fbcode//buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    --show-output 2>&1)
if ! echo "$RUNNER_OUTPUT" | grep -q "BUILD SUCCEEDED"; then
    echo "ERROR: buck2 build failed"
    echo "$RUNNER_OUTPUT"
    exit 1
fi
RUNNER_REL=$(echo "$RUNNER_OUTPUT" | awk '/upi_trm_train/ {print $2}' | tail -n1)
if [ -z "$RUNNER_REL" ]; then
    echo "ERROR: could not resolve upi_trm_train output path"
    echo "$RUNNER_OUTPUT"
    exit 1
fi
RUNNER="$FBSOURCE_ROOT/$RUNNER_REL"
echo "Using runner: $RUNNER"
cd "$CODE_ROOT" || exit 1

is_complete() {
    local logfile=$1
    local checkpoint_dir=$2
    [ -f "$logfile" ] && \
        rg -q "\\[step 20000\\] eval_success_rate" "$logfile" && \
        [ -f "$checkpoint_dir/rl_checkpoint_step_20000.pt" ]
}

write_manifest() {
    printf "gpu\tseed\tconfig\tlogfile\tcheckpoint_dir\n" > "$MANIFEST"
    local gpu
    local seed
    for gpu in 0 1; do
        for seed in "${SEEDS[@]}"; do
            if [ $((seed % 2)) -ne $gpu ]; then
                continue
            fi
            printf "%s\t%s\t%s\t%s\t%s\n" \
                "$gpu" \
                "$seed" \
                "$CONFIG_PATH" \
                "$LOG_DIR/seed${seed}.log" \
                "$CHECKPOINT_ROOT/seed${seed}" >> "$MANIFEST"
        done
    done
}

run_seed() {
    local gpu=$1
    local seed=$2
    local logfile="$LOG_DIR/seed${seed}.log"
    local checkpoint_dir="$CHECKPOINT_ROOT/seed${seed}"

    mkdir -p "$checkpoint_dir"

    if is_complete "$logfile" "$checkpoint_dir"; then
        echo "[GPU $gpu] Skipping seed=$seed (already complete)"
        return 0
    fi

    if [ -f "$logfile" ]; then
        mv "$logfile" "${logfile}.partial.$(date +%Y%m%d_%H%M%S)"
    fi

    echo "[GPU $gpu] Starting seed=$seed"
    echo "  Log: $logfile"
    echo "  Checkpoints: $checkpoint_dir"

    if CUDA_VISIBLE_DEVICES=$gpu "$RUNNER" \
        --config "$CONFIG_PATH" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --checkpoint-dir "$checkpoint_dir" \
        --env-step-budget "$ENV_STEP_BUDGET" \
        --log-env-interval "$INTERVAL" \
        --eval-env-interval "$INTERVAL" \
        --save-env-interval "$INTERVAL" \
        --no-wandb \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] Finished seed=$seed"
    else
        echo "seed=${seed}\tgpu=${gpu}\tlog=${logfile}" >> "$FAILURES_FILE"
        echo "[GPU $gpu] FAILED seed=$seed (see $logfile)"
        return 1
    fi
}

run_gpu_queue() {
    local gpu=$1
    shift
    local seed
    for seed in "$@"; do
        run_seed "$gpu" "$seed"
    done
}

write_manifest

GPU0_SEEDS=(41 43 45 47 49)
GPU1_SEEDS=(42 44 46 48 50)

echo "============================================================"
echo "Episodic-z hard-suite rerun"
echo "============================================================"
echo "Config: $CONFIG_PATH"
echo "Dataset: $DATA_PATH"
echo "Env-step budget: $ENV_STEP_BUDGET"
echo "Checkpoint/eval interval: $INTERVAL"
echo "Logs: $LOG_DIR"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Manifest: $MANIFEST"
echo "============================================================"

run_gpu_queue 0 "${GPU0_SEEDS[@]}" &
PID0=$!
run_gpu_queue 1 "${GPU1_SEEDS[@]}" &
PID1=$!

wait "$PID0"
RC0=$?
wait "$PID1"
RC1=$?

if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
    echo "One or more GPU workers failed."
    exit 1
fi

if [ -s "$FAILURES_FILE" ]; then
    echo "Failures recorded in $FAILURES_FILE"
    exit 1
fi

echo "All episodic-z hard-suite jobs completed successfully."
