#!/bin/bash
# Post-audit 9x9 Sudoku TRM+PPO baseline rerun (3 seeds, 50k steps).
#
# Historical launcher retained for source archaeology only. Its original
# report and outputs were invalidated and removed. It is not part of the
# registered protocol-v2 experiment path and must not be used as paper evidence.
#
# Scope:
#   - Method: TRM+PPO only
#   - Seeds:  0, 1, 2 (3 seeds, one per GPU)
#   - Dataset: data/sudoku-9x9 (regenerated with --seed 42, sizes matching
#             the Feb 2026 protocol: 1000 train / 100 val / 100 test)
#   - Steps:  50,000
#   - Config: configs/sudoku9x9/ppo_9x9.yaml (Feb 2026 config; current code
#             already supports it end-to-end post-audit)
#   - Eval:   100 episodes per eval checkpoint, every 500 steps
#
# Results:
#   logfile = results/9x9_trm_baselines_post3c2e959_3seed/ppo_9x9_s{0,1,2}.log
#   checkpoints = checkpoints/9x9_trm_baselines_post3c2e959_3seed/ppo_s{0,1,2}/
#   manifest = results/.../job_manifest.tsv
#
# Runtime estimate (from Feb 2026 wall-clock on the same config):
#   ~37 hours per PPO seed. On 3 GPUs in parallel, wall-clock ~37 hours.
#
# Usage:
#   bash scripts/run_9x9_trm_ppo_post_audit_3seed.sh

set -euo pipefail

WORKTREE="/home/buiksat/trm_bellman"
FBCODE_DIR="/data/users/buiksat/fbsource/fbcode"
DATA_PATH="buiksat_trm/data/sudoku-9x9"
RESULTS_DIR="$WORKTREE/results/9x9_trm_baselines_post3c2e959_3seed"
CHECKPOINT_ROOT="$WORKTREE/checkpoints/9x9_trm_baselines_post3c2e959_3seed"
CONFIG="buiksat_trm/configs/sudoku9x9/ppo_9x9.yaml"
TRAIN_STEPS=50000
SEEDS=(0 1 2)

MANIFEST="$RESULTS_DIR/job_manifest.tsv"
FAILURES_FILE="$RESULTS_DIR/failed_jobs.txt"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_ROOT"
: > "$FAILURES_FILE"

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
    [ -f "$logfile" ] && rg -q "\[step $TRAIN_STEPS\] eval_success_rate" "$logfile"
}

write_manifest() {
    printf "gpu\tmethod\tseed\tconfig\tlogfile\tcheckpoint_dir\n" > "$MANIFEST"
    local seed
    local gpu
    for seed in "${SEEDS[@]}"; do
        gpu=$seed  # seed 0 -> GPU 0, seed 1 -> GPU 1, seed 2 -> GPU 2
        printf "%s\tppo\t%s\t%s\t%s\t%s\n" \
            "$gpu" \
            "$seed" \
            "$CONFIG" \
            "$RESULTS_DIR/ppo_9x9_s${seed}.log" \
            "$CHECKPOINT_ROOT/ppo_s${seed}" \
            >> "$MANIFEST"
    done
}

run_experiment() {
    local gpu=$1
    local seed=$2
    local logfile="$RESULTS_DIR/ppo_9x9_s${seed}.log"
    local checkpoint_dir="$CHECKPOINT_ROOT/ppo_s${seed}"

    if is_complete "$logfile"; then
        echo "[GPU $gpu] Skipping ppo seed=$seed (already complete, success=$(extract_final_success "$logfile"))"
        return 0
    fi

    if [ -f "$logfile" ]; then
        mv "$logfile" "${logfile}.partial.$(date +%Y%m%d_%H%M%S)"
    fi

    mkdir -p "$checkpoint_dir"

    echo "[GPU $gpu] Starting ppo seed=$seed"
    echo "  Config:     $CONFIG"
    echo "  Log:        $logfile"
    echo "  Checkpoint: $checkpoint_dir"

    if CUDA_VISIBLE_DEVICES=$gpu buck2 run fbcode//buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        --local-only \
        -- --config "$CONFIG" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --checkpoint-dir "$checkpoint_dir" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] Finished ppo seed=$seed -> success=$(extract_final_success "$logfile")"
    else
        echo "ppo:${seed}:${logfile}" >> "$FAILURES_FILE"
        echo "[GPU $gpu] FAILED ppo seed=$seed (logged to $FAILURES_FILE)"
        return 1
    fi
}

main() {
    cd "$FBCODE_DIR"

    write_manifest

    echo "============================================================"
    echo "9x9 post-audit TRM+PPO baseline rerun (3 seeds, 50k steps)"
    echo "============================================================"
    echo "  Dataset:     $DATA_PATH"
    echo "  Config:      $CONFIG"
    echo "  Steps:       $TRAIN_STEPS"
    echo "  Seeds:       ${SEEDS[*]}"
    echo "  Results:     $RESULTS_DIR"
    echo "  Checkpoints: $CHECKPOINT_ROOT"
    echo "  Manifest:    $MANIFEST"
    echo "  Expected:    ~37h wall-clock (3 seeds on 3 GPUs in parallel)"
    echo "============================================================"
    echo ""

    local pids=()
    local seed
    for seed in "${SEEDS[@]}"; do
        run_experiment "$seed" "$seed" &
        pids+=($!)
    done

    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=$((fail + 1))
        fi
    done

    echo ""
    echo "============================================================"
    echo "9x9 PPO sweep complete. Failures: $fail / ${#SEEDS[@]}"
    echo "============================================================"
    if [ "$fail" -gt 0 ]; then
        echo "Failed jobs:"
        cat "$FAILURES_FILE"
        return 1
    fi
    return 0
}

main "$@"
