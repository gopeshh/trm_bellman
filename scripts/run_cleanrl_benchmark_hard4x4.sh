#!/bin/bash
# Hard 4x4 baseline benchmark sweep: 3 algos × 10 seeds = 30 runs
# PPO and A2C are CleanRL implementations; DQN is in-house DQNTrainer wrapper.
# DQN n=5 is not included (in-house DQNTrainer lacks n-step support).
# Protocol: 320k env interactions, no-mask, feasibility checker
#
# Queues at most 2 concurrent runs (one per GPU).
# Tracks completion via train_summary.json; skips already-finished runs.
# Logs failures to failed_jobs.txt for post-mortem.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$REPO_ROOT/configs/baselines"
OUT_ROOT="$REPO_ROOT/results/cleanrl_benchmark/hard4x4"
LOG_DIR="$OUT_ROOT/logs"
FAILED_LOG="$OUT_ROOT/failed_jobs.txt"
MANIFEST="$OUT_ROOT/manifest.txt"

# Build the runner fresh so we never use a stale artifact
echo "Building cleanrl_runner..."
cd "$HOME/fbsource" || exit 1
RUNNER_OUTPUT=$(buck2 build fbcode//buiksat_trm_cleanrl:cleanrl_runner --show-output 2>&1)
if ! echo "$RUNNER_OUTPUT" | grep -q "BUILD SUCCEEDED"; then
    echo "ERROR: buck2 build failed"
    echo "$RUNNER_OUTPUT"
    exit 1
fi
RUNNER="$HOME/fbsource/$(echo "$RUNNER_OUTPUT" | grep cleanrl_runner.par | awk '{print $2}')"
echo "Using runner: $RUNNER"
cd "$REPO_ROOT" || exit 1

mkdir -p "$LOG_DIR"
: > "$FAILED_LOG"

declare -A CONFIGS
CONFIGS["cleanrl_ppo"]="$CONFIG_DIR/cleanrl_ppo_trm.yaml"
CONFIGS["cleanrl_a2c"]="$CONFIG_DIR/cleanrl_a2c_trm.yaml"
CONFIGS["inhouse_dqn"]="$CONFIG_DIR/cleanrl_dqn_trm.yaml"

# DQN uses in-house DQNTrainer wrapper (not independent CleanRL).
# DQN n=5 dropped: in-house DQNTrainer lacks n-step support in this checkout.
ALGOS=(cleanrl_ppo cleanrl_a2c inhouse_dqn)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Build job list, skipping already-completed runs
JOBS=()
for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        outdir="$OUT_ROOT/${algo}/seed${seed}"
        summary="$outdir/train_summary.json"
        if [[ -f "$summary" ]]; then
            echo "[SKIP] $algo seed=$seed (train_summary.json exists)"
            continue
        fi
        JOBS+=("${algo}:${seed}")
    done
done

TOTAL=${#JOBS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "All runs already complete. Nothing to do."
    exit 0
fi
echo "Queued $TOTAL runs across 2 GPUs."
echo ""

# Generate manifest
: > "$MANIFEST"
for job in "${JOBS[@]}"; do
    echo "$job" >> "$MANIFEST"
done

# Run one job, return 0 on success, 1 on failure
run_job() {
    local algo="$1"
    local seed="$2"
    local gpu="$3"
    local outdir="$OUT_ROOT/${algo}/seed${seed}"
    local logfile="$LOG_DIR/${algo}_seed${seed}.log"
    local config="${CONFIGS[$algo]}"

    mkdir -p "$outdir"
    echo "[GPU $gpu] Starting $algo seed=$seed"

    CUDA_VISIBLE_DEVICES=$gpu "$RUNNER" \
        --config "$config" \
        --seed "$seed" \
        --output-dir "$outdir" \
        > "$logfile" 2>&1
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "[GPU $gpu] FAILED $algo seed=$seed (exit $rc)"
        echo "$algo:$seed:gpu$gpu:exit$rc" >> "$FAILED_LOG"
        return 1
    fi

    if [[ ! -f "$outdir/train_summary.json" ]]; then
        echo "[GPU $gpu] FAILED $algo seed=$seed (no train_summary.json)"
        echo "$algo:$seed:gpu$gpu:no_summary" >> "$FAILED_LOG"
        return 1
    fi

    echo "[GPU $gpu] DONE $algo seed=$seed"
    return 0
}

# Queue: 2 workers (one per GPU), process jobs sequentially per worker
job_idx=0
completed=0
failed=0

# Split jobs into two queues
GPU0_JOBS=()
GPU1_JOBS=()
for i in "${!JOBS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_JOBS+=("${JOBS[$i]}")
    else
        GPU1_JOBS+=("${JOBS[$i]}")
    fi
done

# Worker function: process a queue sequentially on one GPU
run_queue() {
    local gpu="$1"
    shift
    local jobs=("$@")
    local worker_failed=0
    for job in "${jobs[@]}"; do
        IFS=':' read -r algo seed <<< "$job"
        if ! run_job "$algo" "$seed" "$gpu"; then
            worker_failed=$((worker_failed + 1))
        fi
    done
    return $worker_failed
}

echo "=== Starting benchmark at $(date) ==="
echo "GPU 0: ${#GPU0_JOBS[@]} jobs"
echo "GPU 1: ${#GPU1_JOBS[@]} jobs"
echo ""

# Launch both workers in parallel (one per GPU)
run_queue 0 "${GPU0_JOBS[@]}" &
PID0=$!
run_queue 1 "${GPU1_JOBS[@]}" &
PID1=$!

# Wait for both workers
wait $PID0
RC0=$?
wait $PID1
RC1=$?

echo ""
echo "=== Benchmark finished at $(date) ==="

# Check worker exit codes
if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
    echo "WARNING: Worker exit codes: GPU0=$RC0 GPU1=$RC1"
fi

# Count results against expected total
completed=$(find "$OUT_ROOT" -name "train_summary.json" | wc -l)
failed_count=$(wc -l < "$FAILED_LOG" 2>/dev/null || echo 0)
expected=$TOTAL

echo "Completed: $completed / $expected"
if [[ -s "$FAILED_LOG" ]]; then
    echo "Failed jobs ($failed_count):"
    cat "$FAILED_LOG"
fi
if [[ $completed -lt $expected ]]; then
    missing=$((expected - completed))
    echo "ERROR: $missing runs did not produce train_summary.json"
    exit 1
fi
if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
    echo "ERROR: Worker(s) exited with nonzero status (GPU0=$RC0 GPU1=$RC1)"
    exit 1
fi

echo "All $expected runs succeeded."
