#!/bin/bash
# Post-audit rerun of the appendix baseline-tuning sweep on the current code.
#
# Motivation:
#   The historical masked-control sweep behind the appendix tuning table was
#   launched from commit 519745e before the in-house baseline trainer audit.
#   This rerun keeps the historical protocol fixed (same dataset, seed, step
#   budget, and 11 completed configs) but executes on the current post-audit
#   code SHA to measure how much the audited trainers move the appendix rows.
#
# Scope:
#   - Dataset: data/sudoku-4x4-easy_6to8empties
#   - Seed:    42
#   - Steps:   20,000
#   - Configs: the 11 historically completed configs only
#              (ppo_high_lr_epochs is excluded by design because the historical
#               run never completed at 20k and therefore does not back a table row)
#   - Launch:  four GPU workers pulling from a queue that isolates the single
#              dominating job (`ppo_deep_unroll`) and front-loads the 10
#              shorter configs on the other three workers. That preserves the
#              same ~28h makespan bound while maximizing the chance that only
#              `ppo_deep_unroll` spills if the host disappears around T+24h.
#   - Resume:  incomplete jobs can be resumed from the latest
#              `rl_checkpoint_step_*.pt` in their checkpoint directory.
#
# Results:
#   results/hp_tuning_sweep_post<sha>_11cfg_seed42/
#     - launcher.log        (written by tmux redirection at launch time)
#     - job_manifest.tsv
#     - runtime_estimates.tsv
#     - failed_jobs.txt
#     - *_s42.log
#
# Runtime estimate (from historical trainer logs):
#   total GPU-hours  = 99:30:52
#   average / 4 GPUs = 24:52:43
#   critical path    = 28:12:26 (ppo_deep_unroll)
#   expected wall-clock with 4 GPU workers: ~28-30h
#
# Usage:
#   bash scripts/run_hp_tuning_post_audit_11cfg.sh

set -euo pipefail

WORKTREE="/home/buiksat/trm_bellman"
FBCODE_DIR="/data/users/buiksat/fbsource/fbcode"
DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
TRAIN_STEPS=20000
SEED=42
GPU_WORKERS=(0 1 2 3)

CODE_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
CODE_SHA_SHORT="${CODE_SHA:0:7}"

RESULTS_DIR="$WORKTREE/results/hp_tuning_sweep_post${CODE_SHA_SHORT}_11cfg_seed${SEED}"
CHECKPOINT_ROOT="$WORKTREE/checkpoints/hp_tuning_sweep_post${CODE_SHA_SHORT}_11cfg_seed${SEED}"
MANIFEST="$RESULTS_DIR/job_manifest.tsv"
RUNTIME_FILE="$RESULTS_DIR/runtime_estimates.tsv"
QUEUE_FILE="$RESULTS_DIR/job_queue.tsv"
QUEUE_LOCK="$RESULTS_DIR/job_queue.lock"
FAILURES_FILE="$RESULTS_DIR/failed_jobs.txt"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_ROOT"
: > "$FAILURES_FILE"
: > "$QUEUE_LOCK"

extract_final_success() {
    local logfile=$1
    local success=""
    success=$(tr '\r' '\n' < "$logfile" 2>/dev/null | rg -o "eval_success_rate=[0-9.]+" | tail -1 | cut -d= -f2 || true)
    if [ -n "$success" ]; then
        printf "%s" "$success"
    else
        printf "N/A"
    fi
}

extract_final_score() {
    local logfile=$1
    local score=""
    score=$(tr '\r' '\n' < "$logfile" 2>/dev/null | rg -o "eval_mean_score=-?[0-9.]+" | tail -1 | cut -d= -f2 || true)
    if [ -n "$score" ]; then
        printf "%s" "$score"
    else
        printf "N/A"
    fi
}

is_complete() {
    local logfile=$1
    [ -f "$logfile" ] && tr '\r' '\n' < "$logfile" | rg -q "\[step 20000\] eval_success_rate="
}

latest_resume_checkpoint() {
    local checkpoint_dir=$1
    find "$checkpoint_dir" -maxdepth 1 -type f -name 'rl_checkpoint_step_*.pt' 2>/dev/null | sort -V | tail -1
}

write_job_files() {
    cat > "$MANIFEST" <<EOF
config_name	algorithm	historical_elapsed	config_rel	logfile	checkpoint_dir
ppo_deep_unroll	ppo	28:12:26	buiksat_trm/configs/hp_tuning/ppo_deep_unroll.yaml	$RESULTS_DIR/ppo_deep_unroll_s${SEED}.log	$CHECKPOINT_ROOT/ppo_deep_unroll_s${SEED}
ppo_strong_reward	ppo	17:49:54	buiksat_trm/configs/hp_tuning/ppo_strong_reward.yaml	$RESULTS_DIR/ppo_strong_reward_s${SEED}.log	$CHECKPOINT_ROOT/ppo_strong_reward_s${SEED}
a2c_high_lr_rollout	a2c	14:02:24	buiksat_trm/configs/hp_tuning/a2c_high_lr_rollout.yaml	$RESULTS_DIR/a2c_high_lr_rollout_s${SEED}.log	$CHECKPOINT_ROOT/a2c_high_lr_rollout_s${SEED}
ppo_high_entropy	ppo	11:43:31	buiksat_trm/configs/hp_tuning/ppo_high_entropy.yaml	$RESULTS_DIR/ppo_high_entropy_s${SEED}.log	$CHECKPOINT_ROOT/ppo_high_entropy_s${SEED}
a2c_strong_reward	a2c	4:46:27	buiksat_trm/configs/hp_tuning/a2c_strong_reward.yaml	$RESULTS_DIR/a2c_strong_reward_s${SEED}.log	$CHECKPOINT_ROOT/a2c_strong_reward_s${SEED}
a2c_deep_unroll	a2c	9:52:23	buiksat_trm/configs/hp_tuning/a2c_deep_unroll.yaml	$RESULTS_DIR/a2c_deep_unroll_s${SEED}.log	$CHECKPOINT_ROOT/a2c_deep_unroll_s${SEED}
a2c_high_entropy	a2c	5:57:37	buiksat_trm/configs/hp_tuning/a2c_high_entropy.yaml	$RESULTS_DIR/a2c_high_entropy_s${SEED}.log	$CHECKPOINT_ROOT/a2c_high_entropy_s${SEED}
dqn_deep_unroll	dqn	2:15:30	buiksat_trm/configs/hp_tuning/dqn_deep_unroll.yaml	$RESULTS_DIR/dqn_deep_unroll_s${SEED}.log	$CHECKPOINT_ROOT/dqn_deep_unroll_s${SEED}
dqn_large_buffer	dqn	2:01:52	buiksat_trm/configs/hp_tuning/dqn_large_buffer.yaml	$RESULTS_DIR/dqn_large_buffer_s${SEED}.log	$CHECKPOINT_ROOT/dqn_large_buffer_s${SEED}
dqn_strong_reward	dqn	2:01:27	buiksat_trm/configs/hp_tuning/dqn_strong_reward.yaml	$RESULTS_DIR/dqn_strong_reward_s${SEED}.log	$CHECKPOINT_ROOT/dqn_strong_reward_s${SEED}
dqn_slow_explore	dqn	0:47:21	buiksat_trm/configs/hp_tuning/dqn_slow_explore.yaml	$RESULTS_DIR/dqn_slow_explore_s${SEED}.log	$CHECKPOINT_ROOT/dqn_slow_explore_s${SEED}
EOF

    cat > "$RUNTIME_FILE" <<EOF
source	commit_or_artifact	total_gpu_time	avg_time_per_4_gpus	critical_path	expected_wall_clock
historical_logs	results/hp_tuning_sweep/*_s42.log	99:30:52	24:52:43	28:12:26	~28-30h
EOF

    tail -n +2 "$MANIFEST" > "$QUEUE_FILE"
}

claim_next_job() {
    local line=""
    local tmp=""

    exec 9<>"$QUEUE_LOCK"
    flock -x 9

    if [ ! -s "$QUEUE_FILE" ]; then
        flock -u 9
        exec 9>&-
        return 1
    fi

    IFS= read -r line < "$QUEUE_FILE" || true
    if [ -z "$line" ]; then
        flock -u 9
        exec 9>&-
        return 1
    fi

    tmp=$(mktemp)
    tail -n +2 "$QUEUE_FILE" > "$tmp" || true
    mv "$tmp" "$QUEUE_FILE"

    flock -u 9
    exec 9>&-

    printf "%s\n" "$line"
}

run_experiment() {
    local gpu=$1
    local config_name=$2
    local algorithm=$3
    local historical_elapsed=$4
    local config_rel=$5
    local logfile=$6
    local checkpoint_dir=$7
    local timestamp=""
    local resume_checkpoint=""
    local resume_step=""
    local -a resume_args=()

    if is_complete "$logfile"; then
        echo "[GPU $gpu] Skipping $config_name (already complete, success=$(extract_final_success "$logfile"), score=$(extract_final_score "$logfile"))"
        return 0
    fi

    mkdir -p "$checkpoint_dir"
    resume_checkpoint=$(latest_resume_checkpoint "$checkpoint_dir" || true)
    timestamp=$(date +%Y%m%d_%H%M%S)

    if [ -f "$logfile" ]; then
        mv "$logfile" "${logfile}.partial.${timestamp}"
    fi

    if [ -n "$resume_checkpoint" ]; then
        resume_step=$(basename "$resume_checkpoint" | sed -E 's/^rl_checkpoint_step_([0-9]+)\.pt$/\1/')
        resume_args=(--resume-checkpoint "$resume_checkpoint")
    elif [ -d "$checkpoint_dir" ] && [ -n "$(ls -A "$checkpoint_dir" 2>/dev/null)" ]; then
        mv "$checkpoint_dir" "${checkpoint_dir}.partial.${timestamp}"
        mkdir -p "$checkpoint_dir"
    fi

    if [ -n "$resume_checkpoint" ]; then
        echo "[GPU $gpu] Resuming $config_name from step $resume_step"
        echo "  Resume:     $resume_checkpoint"
    else
        echo "[GPU $gpu] Starting $config_name"
    fi
    echo "  Algorithm:  $algorithm"
    echo "  Config:     $config_rel"
    echo "  Historical: $historical_elapsed"
    echo "  Log:        $logfile"
    echo "  Checkpoint: $checkpoint_dir"

    if CUDA_VISIBLE_DEVICES=$gpu buck2 run fbcode//buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        --local-only \
        -- --config "$config_rel" \
        --seed "$SEED" \
        --dataset-paths "$DATA_PATH" \
        "${resume_args[@]}" \
        --checkpoint-dir "$checkpoint_dir" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] Finished $config_name -> success=$(extract_final_success "$logfile"), score=$(extract_final_score "$logfile")"
        return 0
    fi

    printf "%s\t%s\t%s\t%s\t%s\n" \
        "$gpu" \
        "$config_name" \
        "$config_rel" \
        "$logfile" \
        "$checkpoint_dir" \
        >> "$FAILURES_FILE"
    echo "[GPU $gpu] FAILED $config_name (logged to $FAILURES_FILE)"
    return 1
}

worker_loop() {
    local gpu=$1
    local worker_failed=0
    local line=""
    local config_name=""
    local algorithm=""
    local historical_elapsed=""
    local config_rel=""
    local logfile=""
    local checkpoint_dir=""

    while line=$(claim_next_job); do
        IFS=$'\t' read -r config_name algorithm historical_elapsed config_rel logfile checkpoint_dir <<< "$line"
        if ! run_experiment "$gpu" "$config_name" "$algorithm" "$historical_elapsed" "$config_rel" "$logfile" "$checkpoint_dir"; then
            worker_failed=1
        fi
    done

    return "$worker_failed"
}

main() {
    cd "$FBCODE_DIR"
    write_job_files

    echo "============================================================"
    echo "Post-audit appendix baseline-tuning rerun (11 completed configs)"
    echo "============================================================"
    echo "  Code SHA:    $CODE_SHA"
    echo "  Dataset:     $DATA_PATH"
    echo "  Seed:        $SEED"
    echo "  Steps:       $TRAIN_STEPS"
    echo "  Workers:     ${GPU_WORKERS[*]}"
    echo "  Results:     $RESULTS_DIR"
    echo "  Checkpoints: $CHECKPOINT_ROOT"
    echo "  Manifest:    $MANIFEST"
    echo "  Runtime:     ~28-30h wall-clock on 4 GPUs (historical critical path 28:12:26)"
    echo "============================================================"
    echo ""

    local pids=()
    local gpu=""
    local fail=0

    for gpu in "${GPU_WORKERS[@]}"; do
        worker_loop "$gpu" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=$((fail + 1))
        fi
    done

    echo ""
    echo "============================================================"
    echo "Post-audit appendix sweep complete. Worker failures: $fail / ${#GPU_WORKERS[@]}"
    echo "============================================================"
    if [ "$fail" -gt 0 ]; then
        echo "Failed jobs:"
        cat "$FAILURES_FILE"
        return 1
    fi
    return 0
}

main "$@"
