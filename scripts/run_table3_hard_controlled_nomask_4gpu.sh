#!/bin/bash
# Run the no-mask Table 3 hard controlled 2x2 across 4 GPUs.
#
# Cells:
#   nc_r0  = no contraction, projection OFF
#   nc_r10 = no contraction, projection ON
#   c_r0   = contraction, projection OFF
#   c_r10  = contraction, projection ON
#
# Uses the no-mask overlay to recover the pre-constraint-masking regime and
# schedules seeds 0..9 in 10 batches. Existing completed logs are skipped.

set -euo pipefail

cd ~/fbsource/fbcode

DATA_PATH="buiksat_trm/data/sudoku-4x4-easy_6to8empties"
RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask"
OVERLAY_CONFIG="buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml"
TRAIN_STEPS=20000
SEEDS=(0 1 2 3 4 5 6 7 8 9)
CELLS=(nc_r0 nc_r10 c_r0 c_r10)

declare -A CONFIGS
CONFIGS["nc_r0"]="buiksat_trm/configs/table3_hard_controlled/episodic_nc_r0_hard.yaml"
CONFIGS["nc_r10"]="buiksat_trm/configs/table3_hard_controlled/episodic_nc_r10_hard.yaml"
CONFIGS["c_r0"]="buiksat_trm/configs/table3_hard_controlled/episodic_c_r0_hard.yaml"
CONFIGS["c_r10"]="buiksat_trm/configs/table3_hard_controlled/episodic_c_r10_hard.yaml"

mkdir -p "$RESULTS_DIR"

is_complete() {
    local logfile=$1
    [ -f "$logfile" ] && rg -q "\[step 20000\] eval_success_rate" "$logfile"
}

run_experiment() {
    local gpu=$1
    local cell=$2
    local seed=$3
    local config="${CONFIGS[$cell]}"
    local logfile="$RESULTS_DIR/${cell}_s${seed}.log"
    local checkpoint_dir="buiksat_trm/checkpoints/table3_hard_6to8_controlled_nomask_${cell}_s${seed}"

    if is_complete "$logfile"; then
        local final_success
        final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
        echo "[GPU $gpu] Skipping $cell seed=$seed (already complete, success=$final_success)"
        return 0
    fi

    echo "[GPU $gpu] Starting $cell seed=$seed"
    echo "  Config: $config"

    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        --local-only \
        -- --config "$config" \
        --config "$OVERLAY_CONFIG" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --checkpoint-dir "$checkpoint_dir" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1

    local final_success
    final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
    echo "[GPU $gpu] Finished $cell seed=$seed -> success=$final_success"
}

echo "============================================================"
echo "Table 3 Hard CONTROLLED (No Mask): 2x2 Contraction x Projection"
echo "============================================================"
echo "Training steps: $TRAIN_STEPS"
echo "Dataset: $DATA_PATH"
echo "Results: $RESULTS_DIR"
echo "Overlay: $OVERLAY_CONFIG"
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "=== Batch for seed=$seed ==="
    for gpu_idx in "${!CELLS[@]}"; do
        cell="${CELLS[$gpu_idx]}"
        run_experiment "$gpu_idx" "$cell" "$seed" &
    done
    wait
    echo "Batch seed=$seed complete"
    echo ""
done

echo "============================================================"
echo "All no-mask controlled 2x2 experiments complete"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
