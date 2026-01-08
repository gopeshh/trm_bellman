#!/bin/bash
# Run 4x4 Sudoku experiments with the new Feasibility Checker
#
# This script runs all experiments needed to validate the feasibility checker:
# - UPI-TRM with feasibility checker (main config)
# - PPO, A2C, DQN baselines
# - Ablations (no conservative mixture, no contraction)
#
# Usage:
#   ./scripts/run_feasibility_experiments.sh [--seeds "42 123 456"] [--steps 5000]

SEEDS="${SEEDS:-42 123 456}"
STEPS="${STEPS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
DATASET="data/sudoku-4x4-ultra-easy"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Feasibility Checker Experiments"
echo "======================================"
echo "Seeds: $SEEDS"
echo "Steps: $STEPS"
echo "Dataset: $DATASET"
echo "======================================"

# Results directory
RESULTS_DIR="results/feasibility_checker_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run function
run_experiment() {
    local name="$1"
    local config="$2"
    local seed="$3"
    local extra_args="${4:-}"

    local log_file="$RESULTS_DIR/${name}_seed${seed}.log"

    echo "[$(date +%H:%M:%S)] Starting $name (seed $seed)..."

    python upi_trm_train.py \
        --dataset-paths "$DATASET" \
        --config "$config" \
        --train-steps "$STEPS" \
        --seed "$seed" \
        $extra_args \
        2>&1 | tee "$log_file"

    echo "[$(date +%H:%M:%S)] Completed $name (seed $seed)"
}

# Main experiments
for seed in $SEEDS; do
    echo ""
    echo "=== Seed $seed ==="

    # 1. UPI-TRM with feasibility checker
    run_experiment "upi_trm_feasibility" \
        "configs/rl_sudoku_4x4_feasibility.yaml" \
        "$seed"

    # 2. PPO baseline
    run_experiment "ppo_trm_feasibility" \
        "configs/baselines/ppo_trm_feasibility.yaml" \
        "$seed" \
        "--baseline ppo"

    # 3. A2C baseline
    run_experiment "a2c_trm_feasibility" \
        "configs/baselines/a2c_trm_feasibility.yaml" \
        "$seed" \
        "--baseline a2c"

    # 4. DQN baseline
    run_experiment "dqn_trm_feasibility" \
        "configs/baselines/dqn_trm_feasibility.yaml" \
        "$seed" \
        "--baseline dqn"

    # 5. Ablation: No conservative mixture (alpha=1.0)
    run_experiment "ablation_no_conservative" \
        "configs/ablations/upi_trm_feasibility_no_conservative.yaml" \
        "$seed"

    # 6. Ablation: No contraction
    run_experiment "ablation_no_contraction" \
        "configs/ablations/upi_trm_feasibility_no_contraction.yaml" \
        "$seed"
done

echo ""
echo "======================================"
echo "All experiments completed!"
echo "Results saved to: $RESULTS_DIR"
echo "======================================"

# Generate summary CSV
echo "algorithm,seed,success_rate,mean_score,mean_filled,mean_violations,mean_zero_cand" > "$RESULTS_DIR/summary.csv"
echo "Run 'python scripts/aggregate_feasibility_results.py $RESULTS_DIR' to aggregate results"
