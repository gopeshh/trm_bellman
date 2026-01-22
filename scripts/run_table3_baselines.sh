#!/bin/bash
# Run all Table 3 baseline experiments
# Uses 4 GPUs in parallel
#
# BASELINE SELECTION:
# - Baselines (PPO/A2C/DQN) are selected via YAML `algorithm:` key
# - upi_trm_train.py auto-detects algorithm from YAML config
# - CLI --baseline flag overrides YAML if needed
# - Trainer selection is logged at startup: "TRAINER SELECTION" block
#
# YAML configs used:
# - baselines/ppo_trm_feasibility.yaml: algorithm: "ppo"
# - baselines/a2c_trm_feasibility.yaml: algorithm: "a2c"
# - baselines/dqn_trm_feasibility.yaml: algorithm: "dqn"

set -e

FBCODE_DIR="$HOME/fbsource/fbcode"
RESULTS_DIR="$HOME/trm_bellman/results/table3_baselines"
DATA_PATH="buiksat_trm/data/sudoku-4x4-trivial"

mkdir -p "$RESULTS_DIR"

cd "$FBCODE_DIR"

run_experiment() {
    local gpu=$1
    local config=$2
    local seed=$3
    local name=$4

    echo "[GPU $gpu] Starting $name seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        -- --config "buiksat_trm/configs/$config" \
        --seed "$seed" --dataset-paths "$DATA_PATH" --no-wandb \
        > "$RESULTS_DIR/${name}_s${seed}.log" 2>&1

    # Extract final success rate
    local success=$(grep "eval_success_rate" "$RESULTS_DIR/${name}_s${seed}.log" | tail -1 | sed 's/.*eval_success_rate=\([0-9.]*\).*/\1/')
    echo "[GPU $gpu] Finished $name seed=$seed -> success=$success"
}

# Batch 1: episodic_no_contraction (3 seeds) + persistent_contraction (1 seed)
echo "=== Batch 1/4 ==="
run_experiment 0 "ablations/upi_trm_feasibility_no_contraction.yaml" 42 "episodic_nc" &
run_experiment 1 "ablations/upi_trm_feasibility_no_contraction.yaml" 123 "episodic_nc" &
run_experiment 2 "ablations/upi_trm_feasibility_no_contraction.yaml" 456 "episodic_nc" &
run_experiment 3 "ablations/upi_trm_feasibility_persistent_z.yaml" 42 "persistent_c" &
wait
echo "Batch 1 complete"

# Batch 2: persistent_contraction (2 seeds) + ppo (2 seeds)
echo "=== Batch 2/4 ==="
run_experiment 0 "ablations/upi_trm_feasibility_persistent_z.yaml" 123 "persistent_c" &
run_experiment 1 "ablations/upi_trm_feasibility_persistent_z.yaml" 456 "persistent_c" &
run_experiment 2 "baselines/ppo_trm_feasibility.yaml" 42 "ppo" &
run_experiment 3 "baselines/ppo_trm_feasibility.yaml" 123 "ppo" &
wait
echo "Batch 2 complete"

# Batch 3: ppo (1 seed) + a2c (3 seeds)
echo "=== Batch 3/4 ==="
run_experiment 0 "baselines/ppo_trm_feasibility.yaml" 456 "ppo" &
run_experiment 1 "baselines/a2c_trm_feasibility.yaml" 42 "a2c" &
run_experiment 2 "baselines/a2c_trm_feasibility.yaml" 123 "a2c" &
run_experiment 3 "baselines/a2c_trm_feasibility.yaml" 456 "a2c" &
wait
echo "Batch 3 complete"

# Batch 4: dqn (3 seeds)
echo "=== Batch 4/4 ==="
run_experiment 0 "baselines/dqn_trm_feasibility.yaml" 42 "dqn" &
run_experiment 1 "baselines/dqn_trm_feasibility.yaml" 123 "dqn" &
run_experiment 2 "baselines/dqn_trm_feasibility.yaml" 456 "dqn" &
wait
echo "Batch 4 complete"

echo "=== All experiments complete ==="

# Extract and summarize results
echo ""
echo "=== RESULTS SUMMARY ==="
for name in episodic_nc persistent_c ppo a2c dqn; do
    echo ""
    echo "$name:"
    for seed in 42 123 456; do
        logfile="$RESULTS_DIR/${name}_s${seed}.log"
        if [ -f "$logfile" ]; then
            success=$(grep "eval_success_rate" "$logfile" | tail -1 | sed 's/.*eval_success_rate=\([0-9.]*\).*/\1/')
            echo "  seed $seed: $success"
        else
            echo "  seed $seed: (no log)"
        fi
    done
done
