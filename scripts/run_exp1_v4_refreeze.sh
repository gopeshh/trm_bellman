#!/bin/bash

set -euo pipefail

WORKTREE="${WORKTREE:-/home/buiksat/trm_bellman}"
FBROOT="${FBROOT:-/home/buiksat/fbsource/fbcode}"

SEEDS="${SEEDS:-41 42 43 44 45 46 47 48 49 50}"
BATCH_SEED="${BATCH_SEED:-42}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
RADII="${RADII:-10 100 0}"
N_MULTS="${N_MULTS:-1,2,4,8}"

DATA_PATH="${DATA_PATH:-$WORKTREE/data/sudoku-4x4-trivial}"
DATA_ROOT="${DATA_ROOT:-$WORKTREE/data}"
CONFIG_A="${CONFIG_A:-$WORKTREE/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml}"
CONFIG_B="${CONFIG_B:-$WORKTREE/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml}"

CKPT_BASE="${CKPT_BASE:-$WORKTREE/checkpoints/exp1_v4_refreeze}"
BATCH_DIR="${BATCH_DIR:-$WORKTREE/artifacts/eval_batches/exp1_v4_refreeze}"
RESULTS_BASE="${RESULTS_BASE:-$WORKTREE/results/validation/exp1_v4_refreeze}"
PLOT_OUT_DIR="${PLOT_OUT_DIR:-$WORKTREE/results/plot_data/exp1_v4_refreeze}"

GPU_SEED42_A="${GPU_SEED42_A:-0}"
GPU_SEED42_B="${GPU_SEED42_B:-1}"
GPU_TRAIN_A="${GPU_TRAIN_A:-0}"
GPU_TRAIN_B="${GPU_TRAIN_B:-1}"
GPU_EVAL="${GPU_EVAL:-2}"

BUCK_FLAGS=(-c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only)

usage() {
    cat <<EOF
Usage: $0 <mode>

Modes:
  seed42         Train the seed-${BATCH_SEED} A'/B checkpoint pair in parallel
  build-batches  Freeze replacement B0/B1 from the seed-${BATCH_SEED} pair
  train-all      Train all seeds in \$SEEDS (skips completed checkpoints)
  eval-all       Evaluate all seeds in \$SEEDS on frozen B0/B1
  aggregate      Aggregate eval outputs into plot-data CSV/MD summaries
  all            Run seed42 -> build-batches -> train-all -> eval-all -> aggregate
  print-config   Print the resolved paths and GPU assignments

Important defaults:
  CKPT_BASE=$CKPT_BASE
  BATCH_DIR=$BATCH_DIR
  RESULTS_BASE=$RESULTS_BASE
  PLOT_OUT_DIR=$PLOT_OUT_DIR
EOF
}

ensure_roots() {
    mkdir -p "$CKPT_BASE" "$BATCH_DIR" "$RESULTS_BASE" "$PLOT_OUT_DIR"
}

check_prereqs() {
    if [[ ! -d "$FBROOT" ]]; then
        echo "[Error] fbcode root not found: $FBROOT" >&2
        exit 1
    fi
    if [[ ! -d "$DATA_PATH" ]]; then
        echo "[Error] dataset not found: $DATA_PATH" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_A" || ! -f "$CONFIG_B" ]]; then
        echo "[Error] config file missing" >&2
        exit 1
    fi
}

checkpoint_a_for_seed() {
    local seed="$1"
    echo "$CKPT_BASE/model_a_prime/seed${seed}/model_step_${TRAIN_STEPS}.pt"
}

checkpoint_b_for_seed() {
    local seed="$1"
    echo "$CKPT_BASE/model_b/seed${seed}/model_step_${TRAIN_STEPS}.pt"
}

train_model() {
    local label="$1"
    local seed="$2"
    local config="$3"
    local ckpt_dir="$4"
    local gpu="$5"

    if [[ -f "$ckpt_dir/model_step_${TRAIN_STEPS}.pt" ]]; then
        echo "[$label] seed $seed already complete: $ckpt_dir"
        return 0
    fi

    mkdir -p "$ckpt_dir"
    echo "[$label] seed $seed -> GPU $gpu"

    (
        cd "$FBROOT"
        CUDA_VISIBLE_DEVICES="$gpu" buck2 run //buiksat_trm:upi_trm_train "${BUCK_FLAGS[@]}" -- \
            --seed "$seed" \
            --train-steps "$TRAIN_STEPS" \
            --config "$config" \
            --checkpoint-dir "$ckpt_dir" \
            --save-interval "$SAVE_INTERVAL" \
            --dataset-paths "$DATA_PATH" \
            --no-wandb
    ) 2>&1 | tee "$ckpt_dir/training.log"
}

train_seed_pair_parallel() {
    local seed="$1"
    local gpu_a="$2"
    local gpu_b="$3"
    local ckpt_a_dir="$CKPT_BASE/model_a_prime/seed${seed}"
    local ckpt_b_dir="$CKPT_BASE/model_b/seed${seed}"

    if [[ -f "$ckpt_a_dir/model_step_${TRAIN_STEPS}.pt" && -f "$ckpt_b_dir/model_step_${TRAIN_STEPS}.pt" ]]; then
        echo "[Seed $seed] both checkpoints already complete"
        return 0
    fi

    train_model "Model A'" "$seed" "$CONFIG_A" "$ckpt_a_dir" "$gpu_a" &
    local pid_a=$!
    train_model "Model B" "$seed" "$CONFIG_B" "$ckpt_b_dir" "$gpu_b" &
    local pid_b=$!

    wait "$pid_a"
    wait "$pid_b"
}

build_batches() {
    local ckpt_a
    local ckpt_b
    ckpt_a="$(checkpoint_a_for_seed "$BATCH_SEED")"
    ckpt_b="$(checkpoint_b_for_seed "$BATCH_SEED")"

    if [[ ! -f "$ckpt_a" || ! -f "$ckpt_b" ]]; then
        echo "[Error] seed-${BATCH_SEED} checkpoints are required before building batches" >&2
        echo "        missing: $ckpt_a" >&2
        echo "        missing: $ckpt_b" >&2
        exit 1
    fi

    mkdir -p "$BATCH_DIR"
    echo "[Batches] freezing replacement B0/B1 into $BATCH_DIR"

    (
        cd "$FBROOT"
        CUDA_VISIBLE_DEVICES="$GPU_EVAL" buck2 run //buiksat_trm:eval_unroll_sensitivity "${BUCK_FLAGS[@]}" -- build-batches \
            --checkpoint_a "$ckpt_a" \
            --config_a "$CONFIG_A" \
            --checkpoint_b "$ckpt_b" \
            --config_b "$CONFIG_B" \
            --out_dir "$BATCH_DIR" \
            --data_dir "$DATA_ROOT" \
            --seed "$BATCH_SEED"
    )
}

train_all() {
    for seed in $SEEDS; do
        train_seed_pair_parallel "$seed" "$GPU_TRAIN_A" "$GPU_TRAIN_B"
    done
}

eval_seed() {
    local seed="$1"
    local ckpt_a
    local ckpt_b
    local out_seed

    ckpt_a="$(checkpoint_a_for_seed "$seed")"
    ckpt_b="$(checkpoint_b_for_seed "$seed")"
    out_seed="$RESULTS_BASE/seed${seed}"

    if [[ ! -f "$ckpt_a" || ! -f "$ckpt_b" ]]; then
        echo "[Error] missing checkpoints for seed $seed" >&2
        exit 1
    fi
    if [[ ! -f "$BATCH_DIR/b0.pt" || ! -f "$BATCH_DIR/b1.pt" ]]; then
        echo "[Error] frozen batches are missing under $BATCH_DIR" >&2
        exit 1
    fi

    mkdir -p "$out_seed"
    echo "[Eval] seed $seed -> $out_seed"

    (
        cd "$FBROOT"
        CUDA_VISIBLE_DEVICES="$GPU_EVAL" buck2 run //buiksat_trm:eval_unroll_sensitivity "${BUCK_FLAGS[@]}" -- compare \
            --checkpoint_a "$ckpt_a" \
            --config_a "$CONFIG_A" \
            --checkpoint_b "$ckpt_b" \
            --config_b "$CONFIG_B" \
            --batch_b0 "$BATCH_DIR/b0.pt" \
            --batch_b1 "$BATCH_DIR/b1.pt" \
            --out_dir "$out_seed" \
            --table_out "$out_seed/unroll_sensitivity_summary.md" \
            --n_mults "$N_MULTS" \
            --seed "$seed" \
            --estimate_lz
    )

    for radius in $RADII; do
        local out_r="$out_seed/R${radius}"
        mkdir -p "$out_r"
        echo "[Eval] seed $seed radius $radius -> $out_r"

        (
            cd "$FBROOT"
            CUDA_VISIBLE_DEVICES="$GPU_EVAL" buck2 run //buiksat_trm:eval_unroll_sensitivity "${BUCK_FLAGS[@]}" -- compare \
                --checkpoint_a "$ckpt_a" \
                --config_a "$CONFIG_A" \
                --checkpoint_b "$ckpt_b" \
                --config_b "$CONFIG_B" \
                --batch_b0 "$BATCH_DIR/b0.pt" \
                --batch_b1 "$BATCH_DIR/b1.pt" \
                --out_dir "$out_r" \
                --table_out "$out_r/radius_sweep.md" \
                --n_mults "$N_MULTS" \
                --latent_ball_radius_override "$radius" \
                --seed "$seed"
        )
    done
}

eval_all() {
    for seed in $SEEDS; do
        eval_seed "$seed"
    done
}

aggregate_results() {
    python3 "$WORKTREE/scripts/aggregate_exp1_results.py" \
        --mode both \
        --results_dir "$RESULTS_BASE" \
        --seeds "${SEEDS// /,}" \
        --radii "${RADII// /,}" \
        --out_dir "$PLOT_OUT_DIR" \
        --batch b0
}

print_config() {
    cat <<EOF
WORKTREE=$WORKTREE
FBROOT=$FBROOT
SEEDS=$SEEDS
BATCH_SEED=$BATCH_SEED
TRAIN_STEPS=$TRAIN_STEPS
DATA_PATH=$DATA_PATH
DATA_ROOT=$DATA_ROOT
CONFIG_A=$CONFIG_A
CONFIG_B=$CONFIG_B
CKPT_BASE=$CKPT_BASE
BATCH_DIR=$BATCH_DIR
RESULTS_BASE=$RESULTS_BASE
PLOT_OUT_DIR=$PLOT_OUT_DIR
GPU_SEED42_A=$GPU_SEED42_A
GPU_SEED42_B=$GPU_SEED42_B
GPU_TRAIN_A=$GPU_TRAIN_A
GPU_TRAIN_B=$GPU_TRAIN_B
GPU_EVAL=$GPU_EVAL
EOF
}

main() {
    local mode="${1:-}"

    if [[ -z "$mode" || "$mode" == "help" || "$mode" == "--help" || "$mode" == "-h" ]]; then
        usage
        exit 0
    fi

    check_prereqs
    ensure_roots

    case "$mode" in
        seed42)
            train_seed_pair_parallel "$BATCH_SEED" "$GPU_SEED42_A" "$GPU_SEED42_B"
            ;;
        build-batches)
            build_batches
            ;;
        train-all)
            train_all
            ;;
        eval-all)
            eval_all
            ;;
        aggregate)
            aggregate_results
            ;;
        print-config)
            print_config
            ;;
        all)
            train_seed_pair_parallel "$BATCH_SEED" "$GPU_SEED42_A" "$GPU_SEED42_B"
            build_batches
            train_all
            eval_all
            aggregate_results
            ;;
        *)
            echo "[Error] unknown mode: $mode" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
