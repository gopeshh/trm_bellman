#!/bin/bash
# Historical launcher for the retired episodic-z hard-suite protocol.
#
# The referenced fixed_base_exact config is not a registered confirmatory
# assignment. Keep this file for provenance, but fail before creating output
# directories, building binaries, or starting training.
#
# Protocol:
#   - dataset: sudoku-4x4-easy_6to8empties
#   - env-step budget: 20000
#   - diagnostics/eval/checkpoints: 5000 / 10000 / 15000 / 20000 env steps
#   - seeds: 41..50
#   - latent mode: episodic-z
#   - config: exact-baseline / theory-exact mixture revision config

set -euo pipefail

echo \
    "ERROR: this historical launcher is retired; its fixed_base_exact config is not a registered executable assignment." \
    >&2
exit 2

FBCODE_DIR="$HOME/fbsource/fbcode"
FBSOURCE_ROOT="$HOME/fbsource"
CODE_ROOT="/home/buiksat/trm_bellman"
PAPER_ROOT="/home/buiksat/UPI_TRM/UPI_TRM_NIPS"

DATA_PATH="$CODE_ROOT/data/sudoku-4x4-easy_6to8empties"
CONFIG_PATH="$CODE_ROOT/configs/revision/upi_trm_feasibility_episodic_z_hard_suite_theory_exact.yaml"
CLOSURE_BATCH_PATH="$PAPER_ROOT/results/closure_batch_seed1729.npz"
DIRECTIONS_PATH="$PAPER_ROOT/results/lv_directions_seed1729_eps1e-4.npz"

RESULTS_ROOT="$PAPER_ROOT/results/episodic_z_hard_suite_20k_seed41_50"
LOG_DIR="$RESULTS_ROOT/logs"
DIAG_LOG_DIR="$RESULTS_ROOT/diagnostic_logs"
CHECKPOINT_ROOT="$CODE_ROOT/checkpoints/episodic_z_hard_suite_20k_seed41_50"
MANIFEST="$RESULTS_ROOT/job_manifest.tsv"
FAILURES_FILE="$RESULTS_ROOT/failed_jobs.txt"

ENV_STEP_BUDGET=20000
INTERVAL=5000
SEEDS=(41 42 43 44 45 46 47 48 49 50)
DIAG_STEPS=(5000 10000 15000 20000)

mkdir -p "$LOG_DIR" "$DIAG_LOG_DIR" "$CHECKPOINT_ROOT"
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

echo "Building episodic_z_hard_suite_diagnostics..."
DIAG_OUTPUT=$(buck2 build fbcode//buiksat_trm:episodic_z_hard_suite_diagnostics \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    --show-output 2>&1)
if ! echo "$DIAG_OUTPUT" | grep -q "BUILD SUCCEEDED"; then
    echo "ERROR: buck2 build failed for episodic_z_hard_suite_diagnostics"
    echo "$DIAG_OUTPUT"
    exit 1
fi
DIAG_REL=$(echo "$DIAG_OUTPUT" | awk '/episodic_z_hard_suite_diagnostics/ {print $2}' | tail -n1)
if [ -z "$DIAG_REL" ]; then
    echo "ERROR: could not resolve episodic_z_hard_suite_diagnostics output path"
    echo "$DIAG_OUTPUT"
    exit 1
fi
DIAG_RUNNER="$FBSOURCE_ROOT/$DIAG_REL"
echo "Using diagnostic runner: $DIAG_RUNNER"
cd "$CODE_ROOT" || exit 1

is_complete() {
    local logfile=$1
    local checkpoint_dir=$2
    [ -f "$logfile" ] && \
        rg -q "\\[step 20000\\]( \\[update [0-9]+\\])? eval_success_rate" "$logfile" && \
        [ -f "$checkpoint_dir/rl_checkpoint_step_20000.pt" ]
}

diagnostics_complete() {
    local seed=$1
    local step
    for step in "${DIAG_STEPS[@]}"; do
        [ -f "$RESULTS_ROOT/diagnostics/seed${seed}/step${step}.json" ] || return 1
    done
    return 0
}

run_seed_diagnostics() {
    local gpu=$1
    local seed=$2
    local diag_log=$3
    local checkpoint_dir="$CHECKPOINT_ROOT/seed${seed}"
    local step

    : > "$diag_log"
    for step in "${DIAG_STEPS[@]}"; do
        local checkpoint_path="$checkpoint_dir/rl_checkpoint_step_${step}.pt"
        local output_json="$RESULTS_ROOT/diagnostics/seed${seed}/step${step}.json"

        if [ -f "$output_json" ]; then
            echo "[diagnostics] seed=${seed} step=${step} reuse ${output_json}" >> "$diag_log"
            continue
        fi
        if [ ! -f "$checkpoint_path" ]; then
            echo "[diagnostics] missing checkpoint ${checkpoint_path}" >> "$diag_log"
            return 1
        fi

        echo "[diagnostics] seed=${seed} step=${step} checkpoint=${checkpoint_path}" >> "$diag_log"
        if ! CUDA_VISIBLE_DEVICES=$gpu "$DIAG_RUNNER" checkpoint \
            --checkpoint "$checkpoint_path" \
            --config-yaml "$CONFIG_PATH" \
            --closure-batch "$CLOSURE_BATCH_PATH" \
            --directions-npz "$DIRECTIONS_PATH" \
            --output-json "$output_json" \
            --device auto \
            --chunk-size 256 \
            >> "$diag_log" 2>&1; then
            return 1
        fi
    done
    return 0
}

write_manifest() {
    printf "gpu\tseed\tconfig\tlogfile\tcheckpoint_dir\n" > "$MANIFEST"
    local seed
    for seed in "${GPU0_SEEDS[@]}"; do
        printf "%s\t%s\t%s\t%s\t%s\n" \
            "0" \
            "$seed" \
            "$CONFIG_PATH" \
            "$LOG_DIR/seed${seed}.log" \
            "$CHECKPOINT_ROOT/seed${seed}" >> "$MANIFEST"
    done
    for seed in "${GPU1_SEEDS[@]}"; do
        printf "%s\t%s\t%s\t%s\t%s\n" \
            "1" \
            "$seed" \
            "$CONFIG_PATH" \
            "$LOG_DIR/seed${seed}.log" \
            "$CHECKPOINT_ROOT/seed${seed}" >> "$MANIFEST"
    done
}

run_seed() {
    local gpu=$1
    local seed=$2
    local logfile="$LOG_DIR/seed${seed}.log"
    local diag_log="$DIAG_LOG_DIR/seed${seed}.log"
    local checkpoint_dir="$CHECKPOINT_ROOT/seed${seed}"

    mkdir -p "$checkpoint_dir"

    if is_complete "$logfile" "$checkpoint_dir"; then
        echo "[GPU $gpu] Skipping training for seed=$seed (already complete)"
    else
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
            echo "[GPU $gpu] Finished training seed=$seed"
        else
            echo "seed=${seed}\tgpu=${gpu}\tstage=train\tlog=${logfile}" >> "$FAILURES_FILE"
            echo "[GPU $gpu] FAILED training seed=$seed (see $logfile)"
            return 1
        fi
    fi

    if diagnostics_complete "$seed"; then
        echo "[GPU $gpu] Skipping diagnostics for seed=$seed (already complete)"
        return 0
    fi

    if [ -f "$diag_log" ]; then
        mv "$diag_log" "${diag_log}.partial.$(date +%Y%m%d_%H%M%S)"
    fi

    echo "[GPU $gpu] Starting diagnostics seed=$seed"
    if run_seed_diagnostics "$gpu" "$seed" "$diag_log"; then
        echo "[GPU $gpu] Finished diagnostics seed=$seed"
    else
        echo "seed=${seed}\tgpu=${gpu}\tstage=diagnostics\tlog=${diag_log}" >> "$FAILURES_FILE"
        echo "[GPU $gpu] FAILED diagnostics seed=$seed (see $diag_log)"
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

GPU0_SEEDS=(41 43 45 47 49)
GPU1_SEEDS=(42 44 46 48 50)

write_manifest

echo "============================================================"
echo "Episodic-z hard-suite rerun"
echo "============================================================"
echo "Config: $CONFIG_PATH"
echo "Dataset: $DATA_PATH"
echo "Env-step budget: $ENV_STEP_BUDGET"
echo "Checkpoint/eval interval: $INTERVAL"
echo "Logs: $LOG_DIR"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Closure batch: $CLOSURE_BATCH_PATH"
echo "Directions: $DIRECTIONS_PATH"
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

python3 "$CODE_ROOT/scripts/postprocess_episodic_z_hard_suite.py" \
    --results-root "$RESULTS_ROOT" \
    --log-dir "$LOG_DIR" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --config-yaml "$CONFIG_PATH" \
    --closure-batch "$CLOSURE_BATCH_PATH" \
    --directions-npz "$DIRECTIONS_PATH" \
    --diag-runner "$DIAG_RUNNER" \
    --device auto \
    --chunk-size 256 \
    --seeds "${SEEDS[@]}" \
    --checkpoint-steps "${DIAG_STEPS[@]}"

echo "All episodic-z hard-suite jobs completed successfully."
