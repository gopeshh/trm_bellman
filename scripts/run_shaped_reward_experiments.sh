#!/bin/bash
# Run shaped-reward experiments + ablations for ICML submission
#
# Addresses Bahram's experimental guidance (main.tex:1426-1429):
#   1. Shaped-reward Sudoku (checker + potential shaping + absorbing normalization)
#   2. Ablations removing: (i) contraction, (ii) exact centering, (iii) conservative mixture
#   3. Sparse reward baseline for comparison
#
# Usage:
#   ./scripts/run_shaped_reward_experiments.sh [OPTIONS]
#
# Options:
#   --dataset PATH      Path to Sudoku dataset (default: data/sudoku-extreme-1k-aug-1000)
#   --seeds N           Number of random seeds (default: 3)
#   --steps N           Training steps (default: 10000)
#   --parallel          Run experiments in parallel (uses GNU parallel if available)
#   --dry-run           Print commands without executing

set -e  # Exit on error

# ===================================================================
# CONFIGURATION
# ===================================================================
DATASET="data/sudoku-extreme-1k-aug-1000"
SEEDS=3
TRAIN_STEPS=10000
PARALLEL=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --steps)
            TRAIN_STEPS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset PATH] [--seeds N] [--steps N] [--parallel] [--dry-run]"
            exit 1
            ;;
    esac
done

echo "======================================================================="
echo "UPI-TRM Shaped Reward Experiments (ICML 2026)"
echo "======================================================================="
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Training steps: $TRAIN_STEPS"
echo "Parallel execution: $PARALLEL"
echo "Dry run: $DRY_RUN"
echo ""

# Check dataset exists
if [ ! -d "$DATASET" ] && [ "$DRY_RUN" = false ]; then
    echo "ERROR: Dataset not found: $DATASET"
    echo "Run: python dataset/build_sudoku_dataset.py --output-dir $DATASET"
    exit 1
fi

# ===================================================================
# EXPERIMENT CONFIGURATIONS
# ===================================================================
# Format: "config_name:wandb_group:description"
EXPERIMENTS=(
    # Main paper-facing configurations; runtime settings do not certify premises.
    "configs/rl_sudoku_shaped_theory_exact.yaml:shaped-theory-exact:Paper-facing shaped-reward configuration"
    "configs/rl_sudoku_sparse_theory_exact.yaml:sparse-theory-exact:Paper-facing sparse-reward configuration"

    # Ablations (remove one feature at a time)
    "configs/ablations/ablation_no_contraction.yaml:ablation-no-contraction:No contraction-oriented intervention"
    "configs/ablations/ablation_no_exact_baseline.yaml:ablation-no-exact-baseline:Approximate rather than exact statewise centering"
    "configs/ablations/ablation_no_conservative_mixture.yaml:ablation-no-conservative-mixture:No conservative mixture (α=1.0)"
    "configs/ablations/ablation_no_projection.yaml:ablation-no-projection:No forward-invariant projection"

    # Full ablation
    "configs/ablations/ablation_no_theory_features.yaml:ablation-all-off:All theory features OFF (standard RL)"

    # Hardest baseline
    "configs/ablations/ablation_sparse_no_theory.yaml:sparse-no-theory:Sparse + no theory (hardest)"
)

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
run_experiment() {
    local config=$1
    local group=$2
    local desc=$3
    local seed=$4

    local run_name="${group}_seed${seed}"
    local cmd="python upi_trm_train.py \
        --dataset-paths $DATASET \
        --config $config \
        --train-steps $TRAIN_STEPS \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Shaped-Rewards \
        --wandb-group $group \
        --wandb-run-name $run_name"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $cmd"
    else
        echo ""
        echo "======================================================================="
        echo "Running: $desc (seed $seed)"
        echo "Group: $group"
        echo "======================================================================="
        $cmd
    fi
}

# ===================================================================
# RUN EXPERIMENTS
# ===================================================================
if [ "$PARALLEL" = true ] && command -v parallel &> /dev/null; then
    echo "Running experiments in parallel using GNU parallel..."

    # Generate all experiment commands
    > /tmp/upi_trm_experiments.txt
    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r config group desc <<< "$exp"
        for seed in $(seq 0 $((SEEDS - 1))); do
            echo "$config $group \"$desc\" $seed" >> /tmp/upi_trm_experiments.txt
        done
    done

    # Run in parallel
    if [ "$DRY_RUN" = true ]; then
        cat /tmp/upi_trm_experiments.txt | while read line; do
            run_experiment $line
        done
    else
        export -f run_experiment
        export DATASET TRAIN_STEPS DRY_RUN
        cat /tmp/upi_trm_experiments.txt | parallel --colsep ' ' run_experiment {1} {2} {3} {4}
    fi
else
    echo "Running experiments sequentially..."

    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r config group desc <<< "$exp"

        for seed in $(seq 0 $((SEEDS - 1))); do
            run_experiment "$config" "$group" "$desc" "$seed"
        done
    done
fi

echo ""
echo "======================================================================="
echo "All experiments completed!"
echo "======================================================================="
echo ""
echo "Results logged to W&B project: UPI-TRM-ICML-Shaped-Rewards"
echo ""
echo "Analysis commands:"
echo "  - View all runs: wandb project UPI-TRM-ICML-Shaped-Rewards"
echo "  - Compare groups: Use W&B comparison view"
echo "  - Download data: wandb export UPI-TRM-ICML-Shaped-Rewards"
echo ""
echo "Ablation conditions:"
echo "  1. shaped-theory-exact: all configured paper-facing mechanisms"
echo "  2. ablation-no-contraction: contraction-oriented intervention disabled"
echo "  3. ablation-no-exact-baseline: approximate rather than exact statewise centering"
echo "  4. ablation-no-conservative-mixture: full candidate step"
echo "  5. ablation-all-off: shaped-reward baseline configuration"
echo "  6. sparse-no-theory: sparse-reward baseline configuration"
