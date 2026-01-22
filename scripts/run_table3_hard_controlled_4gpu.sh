#!/bin/bash
# Run Table 3 Hard CONTROLLED 2×2 experiment to deconfound contraction vs projection
#
# 2×2 Design:
#   Factor A: enable_contraction ∈ {false, true}
#   Factor B: latent_ball_radius R ∈ {0.0, 10.0}
#
# Cells:
#   NC-R0:  No contraction, R=0 (projection OFF)
#   NC-R10: No contraction, R=10 (projection ON)
#   C-R0:   Contraction, R=0 (projection OFF)
#   C-R10:  Contraction, R=10 (projection ON)
#
# Critical controls (per CLAUDE.md):
#   - disable_value_head_norm: true for ALL cells
#   - episodic_latent: true for ALL cells
#   - use_feasibility_checker: true
#   - num_train_steps: 20000
#   - max_edits: 16 (T=16)
#
# Uses 4 GPUs in parallel, 3 seeds each → 12 runs total (3 batches)

set -e

cd ~/fbsource/fbcode

DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled"
TRAIN_STEPS=20000

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Table 3 Hard CONTROLLED: 2×2 Contraction × Projection"
echo "============================================================"
echo "Training steps: $TRAIN_STEPS"
echo "Dataset: $DATA_PATH"
echo "Results: $RESULTS_DIR"
echo ""
echo "Design:"
echo "  NC-R0:  enable_contraction=false, latent_ball_radius=0.0"
echo "  NC-R10: enable_contraction=false, latent_ball_radius=10.0"
echo "  C-R0:   enable_contraction=true,  latent_ball_radius=0.0"
echo "  C-R10:  enable_contraction=true,  latent_ball_radius=10.0"
echo ""
echo "Controls: disable_value_head_norm=true, episodic_latent=true, T=16"
echo "============================================================"
echo ""

# Configs for each cell of the 2×2
declare -A CONFIGS
CONFIGS["nc_r0"]="buiksat_trm/configs/table3_hard_controlled/episodic_nc_r0_hard.yaml"
CONFIGS["nc_r10"]="buiksat_trm/configs/table3_hard_controlled/episodic_nc_r10_hard.yaml"
CONFIGS["c_r0"]="buiksat_trm/configs/table3_hard_controlled/episodic_c_r0_hard.yaml"
CONFIGS["c_r10"]="buiksat_trm/configs/table3_hard_controlled/episodic_c_r10_hard.yaml"

SEEDS=(42 123 456)

run_experiment() {
    local gpu=$1
    local cell=$2
    local seed=$3
    local config="${CONFIGS[$cell]}"
    local logfile="$RESULTS_DIR/${cell}_s${seed}.log"

    echo "[GPU $gpu] Starting $cell seed=$seed"
    echo "  Config: $config"

    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        -- --config "$config" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1

    # Extract final success rate
    local final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
    echo "[GPU $gpu] Finished $cell seed=$seed -> success=$final_success"
}

# 4 configs × 3 seeds = 12 runs → 3 batches of 4

echo "=== Batch 1/3: All 4 cells, seed=42 ==="
run_experiment 0 "nc_r0" 42 &
run_experiment 1 "nc_r10" 42 &
run_experiment 2 "c_r0" 42 &
run_experiment 3 "c_r10" 42 &
wait
echo "Batch 1 complete"
echo ""

echo "=== Batch 2/3: All 4 cells, seed=123 ==="
run_experiment 0 "nc_r0" 123 &
run_experiment 1 "nc_r10" 123 &
run_experiment 2 "c_r0" 123 &
run_experiment 3 "c_r10" 123 &
wait
echo "Batch 2 complete"
echo ""

echo "=== Batch 3/3: All 4 cells, seed=456 ==="
run_experiment 0 "nc_r0" 456 &
run_experiment 1 "nc_r10" 456 &
run_experiment 2 "c_r0" 456 &
run_experiment 3 "c_r10" 456 &
wait
echo "Batch 3 complete"
echo ""

echo "============================================================"
echo "All experiments complete"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
echo ""

# Print 2×2 summary table
echo "=== 2×2 Results Summary (Final Success Rate) ==="
echo ""
echo "                    R=0 (proj OFF)    R=10 (proj ON)"
echo "---------------------------------------------------"

# Compute means for each cell
for contraction in "nc" "c"; do
    if [ "$contraction" == "nc" ]; then
        label="No Contraction"
    else
        label="Contraction   "
    fi

    r0_sum=0
    r10_sum=0
    r0_count=0
    r10_count=0

    for seed in 42 123 456; do
        # R=0 cell
        logfile="$RESULTS_DIR/${contraction}_r0_s${seed}.log"
        if [ -f "$logfile" ]; then
            val=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            if [ -n "$val" ]; then
                r0_sum=$(echo "$r0_sum + $val" | bc -l)
                r0_count=$((r0_count + 1))
            fi
        fi

        # R=10 cell
        logfile="$RESULTS_DIR/${contraction}_r10_s${seed}.log"
        if [ -f "$logfile" ]; then
            val=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            if [ -n "$val" ]; then
                r10_sum=$(echo "$r10_sum + $val" | bc -l)
                r10_count=$((r10_count + 1))
            fi
        fi
    done

    # Compute means
    if [ $r0_count -gt 0 ]; then
        r0_mean=$(echo "scale=3; $r0_sum / $r0_count" | bc -l)
    else
        r0_mean="N/A"
    fi

    if [ $r10_count -gt 0 ]; then
        r10_mean=$(echo "scale=3; $r10_sum / $r10_count" | bc -l)
    else
        r10_mean="N/A"
    fi

    printf "%s     %s              %s\n" "$label" "$r0_mean" "$r10_mean"
done

echo "---------------------------------------------------"
echo ""

# Per-seed details
echo "=== Per-Seed Details ==="
for cell in nc_r0 nc_r10 c_r0 c_r10; do
    echo "$cell:"
    for seed in 42 123 456; do
        logfile="$RESULTS_DIR/${cell}_s${seed}.log"
        if [ -f "$logfile" ]; then
            final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            echo "  seed $seed: $final_success"
        else
            echo "  seed $seed: (log not found)"
        fi
    done
done

echo ""
echo "Logs saved to: $RESULTS_DIR/"
