#!/bin/bash
# Comprehensive Experiment Runner for UPI-TRM ICML 2026
# Runs experiments on all 4 GPUs in parallel

set -e

EXP_DIR="/home/buiksat/trm_bellman/runs/exp_$(date +%Y%m%d)"
mkdir -p "$EXP_DIR"
echo "=========================================="
echo "UPI-TRM ICML 2026 Experiment Suite"
echo "Experiment Directory: $EXP_DIR"
echo "=========================================="

cd ~/fbsource/fbcode

BUCK_FLAGS="-c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true"
DATA_4x4="/home/buiksat/trm_bellman/data/sudoku-4x4-ultra-easy"
DATA_9x9="/home/buiksat/trm_bellman/data/sudoku-9x9-hard"
CONFIGS="/home/buiksat/trm_bellman/configs"

TRAIN_STEPS=5000
SEEDS="42 123 456"

# Function to run experiment
run_exp() {
    local gpu=$1
    local name=$2
    local config=$3
    local dataset=$4
    local seed=$5
    local extra_args="${6:-}"

    local logfile="$EXP_DIR/${name}_seed${seed}.log"
    echo "[GPU $gpu] Starting: $name (seed=$seed)"

    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train $BUCK_FLAGS -- \
        --dataset-paths "$dataset" \
        --config "$config" \
        --train-steps $TRAIN_STEPS \
        --seed $seed \
        $extra_args \
        > "$logfile" 2>&1

    echo "[GPU $gpu] Completed: $name (seed=$seed)"
}

# ============================================================
# GPU 0: Main UPI-TRM experiments (multiple seeds)
# ============================================================
run_gpu0() {
    echo "=== GPU 0: UPI-TRM Main Experiments ==="

    # UPI-TRM persistent z with constraint checker (3 seeds)
    for seed in $SEEDS; do
        run_exp 0 "upi_trm_persistent_z_constraint" \
            "$CONFIGS/paper_persistent_z_constraint.yaml" \
            "$DATA_4x4" $seed
    done

    # UPI-TRM with progress checker
    run_exp 0 "upi_trm_progress_checker" \
        "$CONFIGS/paper_progress_checker.yaml" \
        "$DATA_4x4" 42

    echo "=== GPU 0: All experiments completed ==="
}

# ============================================================
# GPU 1: DQN experiments (4x4 and 9x9)
# ============================================================
run_gpu1() {
    echo "=== GPU 1: DQN Experiments ==="

    # DQN on 9x9 (3 seeds) - previously achieved 100%
    for seed in $SEEDS; do
        run_exp 1 "dqn_9x9" \
            "$CONFIGS/9x9/dqn_trm.yaml" \
            "$DATA_9x9" $seed \
            "--baseline dqn --backbone trm"
    done

    # DQN on 4x4 for comparison
    run_exp 1 "dqn_4x4" \
        "$CONFIGS/baselines/dqn_trm_sudoku.yaml" \
        "$DATA_4x4" 42 \
        "--baseline dqn --backbone trm"

    echo "=== GPU 1: All experiments completed ==="
}

# ============================================================
# GPU 2: Ablation experiments (Set 1 - Theory features)
# ============================================================
run_gpu2() {
    echo "=== GPU 2: Ablation Experiments (Theory) ==="

    # Key ablation: replace exact statewise centering with an approximation
    for seed in $SEEDS; do
        run_exp 2 "ablation_no_exact_baseline" \
            "$CONFIGS/ablations/ablation_no_exact_baseline.yaml" \
            "$DATA_4x4" $seed
    done

    # Ablation: disable the contraction-oriented intervention
    run_exp 2 "ablation_no_contraction" \
        "$CONFIGS/ablations/ablation_no_contraction.yaml" \
        "$DATA_4x4" 42

    # Ablation: No conservative mixture
    run_exp 2 "ablation_no_conservative_mixture" \
        "$CONFIGS/ablations/ablation_no_conservative_mixture.yaml" \
        "$DATA_4x4" 42

    echo "=== GPU 2: All experiments completed ==="
}

# ============================================================
# GPU 3: Ablation experiments (Set 2 - Other ablations + Baselines)
# ============================================================
run_gpu3() {
    echo "=== GPU 3: Ablation Experiments (Other) + Baselines ==="

    # Ablation: No theory features (all disabled)
    for seed in $SEEDS; do
        run_exp 3 "ablation_no_theory_features" \
            "$CONFIGS/ablations/ablation_no_theory_features.yaml" \
            "$DATA_4x4" $seed
    done

    # Ablation: Sparse rewards + no theory
    run_exp 3 "ablation_sparse_no_theory" \
        "$CONFIGS/ablations/ablation_sparse_no_theory.yaml" \
        "$DATA_4x4" 42

    # PPO baseline for comparison
    run_exp 3 "ppo_trm_4x4" \
        "$CONFIGS/baselines/ppo_trm_sudoku.yaml" \
        "$DATA_4x4" 42 \
        "--baseline ppo --backbone trm"

    # A2C baseline
    run_exp 3 "a2c_trm_4x4" \
        "$CONFIGS/baselines/a2c_trm_sudoku.yaml" \
        "$DATA_4x4" 42 \
        "--baseline a2c --backbone trm"

    echo "=== GPU 3: All experiments completed ==="
}

# ============================================================
# Launch all GPUs in parallel
# ============================================================
echo ""
echo "Launching experiments on all 4 GPUs..."
echo ""

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!
run_gpu2 &
PID2=$!
run_gpu3 &
PID3=$!

echo "PIDs: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
echo ""
echo "Monitoring progress... (Ctrl+C to stop monitoring, experiments continue)"
echo ""

# Wait for all to complete
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $EXP_DIR"
echo "=========================================="

# Summary
echo ""
echo "=== Experiment Summary ==="
for logfile in "$EXP_DIR"/*.log; do
    if [ -f "$logfile" ]; then
        name=$(basename "$logfile" .log)
        success=$(grep -o "eval_success_rate=[0-9.]*" "$logfile" | tail -1 || echo "N/A")
        echo "  $name: $success"
    fi
done
