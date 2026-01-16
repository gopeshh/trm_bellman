#!/bin/bash
# Run Experiment 1 v4 Evaluation
# Runs unroll sensitivity and radius sweep for all seeds

set -e

# Configuration
SEEDS="41 42 43"
RADII="10 100 0"  # Simplified radius sweep
N_MULTS="1,2,4,8"  # Depth multipliers
BATCH_DIR="/home/buiksat/trm_bellman/artifacts/eval_batches/v3"

# Checkpoint base
CKPT_BASE="/home/buiksat/trm_bellman/checkpoints/exp1_v4"

# Config paths
CONFIG_A="/home/buiksat/trm_bellman/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml"
CONFIG_B="/home/buiksat/trm_bellman/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml"

# Output base
RESULTS_BASE="/home/buiksat/trm_bellman/results/validation/exp1_v4"

echo "=== ICML Experiment 1 v4: Evaluation ==="
echo "Seeds: $SEEDS"
echo "Radii: $RADII"
echo "Depth multipliers: $N_MULTS"
echo ""

# Create output directories
mkdir -p "$RESULTS_BASE"

for SEED in $SEEDS; do
    echo "========================================"
    echo "Seed: $SEED"
    echo "========================================"

    CKPT_A="$CKPT_BASE/model_a_prime/seed$SEED/model_step_5000.pt"
    CKPT_B="$CKPT_BASE/model_b/seed$SEED/model_step_5000.pt"

    if [ ! -f "$CKPT_A" ]; then
        echo "[Skip] Model A' checkpoint not found: $CKPT_A"
        continue
    fi

    if [ ! -f "$CKPT_B" ]; then
        echo "[Skip] Model B checkpoint not found: $CKPT_B"
        continue
    fi

    OUT_SEED="$RESULTS_BASE/seed$SEED"
    mkdir -p "$OUT_SEED"

    # 1. Unroll sensitivity (default R=10)
    echo "[Seed $SEED] Running unroll sensitivity..."
    cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES="" buck2 run //buiksat_trm:eval_unroll_sensitivity -- compare \
        --checkpoint_a "$CKPT_A" \
        --config_a "$CONFIG_A" \
        --checkpoint_b "$CKPT_B" \
        --config_b "$CONFIG_B" \
        --batch_b0 "$BATCH_DIR/b0.pt" \
        --batch_b1 "$BATCH_DIR/b1.pt" \
        --out_dir "$OUT_SEED" \
        --table_out "$OUT_SEED/unroll_sensitivity_summary.md" \
        --n_mults "$N_MULTS" \
        --seed "$SEED" \
        --estimate_lz

    # 2. Radius sweep
    for R in $RADII; do
        echo "[Seed $SEED] Running radius sweep at R=$R..."
        OUT_R="$OUT_SEED/R$R"
        mkdir -p "$OUT_R"

        cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES="" buck2 run //buiksat_trm:eval_unroll_sensitivity -- compare \
            --checkpoint_a "$CKPT_A" \
            --config_a "$CONFIG_A" \
            --checkpoint_b "$CKPT_B" \
            --config_b "$CONFIG_B" \
            --batch_b0 "$BATCH_DIR/b0.pt" \
            --batch_b1 "$BATCH_DIR/b1.pt" \
            --out_dir "$OUT_R" \
            --table_out "$OUT_R/radius_sweep_R${R}.md" \
            --n_mults "$N_MULTS" \
            --latent_ball_radius_override "$R" \
            --seed "$SEED"
    done

    echo ""
done

echo "=== Evaluation Complete ==="
echo "Results saved to: $RESULTS_BASE"
echo ""

# Aggregate across seeds
echo "=== Aggregating Results ==="
python3 /home/buiksat/trm_bellman/scripts/aggregate_exp1_results.py \
    --mode both \
    --results_dir "$RESULTS_BASE" \
    --seeds "${SEEDS// /,}" \
    --radii "${RADII// /,}" \
    --out_dir /home/buiksat/trm_bellman/results/plot_data/exp1_v4 \
    --batch b0

echo ""
echo "=== Done ==="
