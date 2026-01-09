#!/bin/bash
# Fast Wave Launcher (UPI-TRM, A2C, DQN only - no PPO)
#
# Runs 3 experiments in parallel on specified GPUs, avoiding GPU1 where PPO runs.
# Usage:
#   ./scripts/launch_fast_wave.sh <seed> [gpu0] [gpu2] [gpu3]
#   Default GPUs: 0, 2, 3 (avoiding GPU1 for PPO)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FBCODE_DIR="$HOME/fbsource/fbcode"

SEED="${1:-123}"
GPU_UPI="${2:-0}"
GPU_A2C="${3:-2}"
GPU_DQN="${4:-3}"

STEPS=5000
DATASET="buiksat_trm/data/sudoku-4x4-trivial"
LOG_BASE="$REPO_DIR/runs/feasibility"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directories
mkdir -p "$LOG_BASE/upi_trm" "$LOG_BASE/a2c" "$LOG_BASE/dqn"

echo ""
echo "=============================================="
echo "FAST WAVE: seed=$SEED (UPI-TRM, A2C, DQN only)"
echo "=============================================="
echo "GPUs: UPI=$GPU_UPI, A2C=$GPU_A2C, DQN=$GPU_DQN"
echo "Timestamp: $TIMESTAMP"

cd "$FBCODE_DIR"

LOG_UPI="$LOG_BASE/upi_trm/${SEED}_${TIMESTAMP}.log"
LOG_A2C="$LOG_BASE/a2c/${SEED}_${TIMESTAMP}.log"
LOG_DQN="$LOG_BASE/dqn/${SEED}_${TIMESTAMP}.log"

echo "[$(date +%H:%M:%S)] Starting upi_trm (seed $SEED) on GPU $GPU_UPI"
CUDA_VISIBLE_DEVICES=$GPU_UPI buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --dataset-paths "$DATASET" --config buiksat_trm/configs/rl_sudoku_4x4_feasibility.yaml \
    --train-steps "$STEPS" --seed "$SEED" \
    > "$LOG_UPI" 2>&1 &
PID_UPI=$!

echo "[$(date +%H:%M:%S)] Starting a2c (seed $SEED) on GPU $GPU_A2C"
CUDA_VISIBLE_DEVICES=$GPU_A2C buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --baseline a2c --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml \
    --train-steps "$STEPS" --seed "$SEED" \
    > "$LOG_A2C" 2>&1 &
PID_A2C=$!

echo "[$(date +%H:%M:%S)] Starting dqn (seed $SEED) on GPU $GPU_DQN"
CUDA_VISIBLE_DEVICES=$GPU_DQN buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --baseline dqn --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml \
    --train-steps "$STEPS" --seed "$SEED" \
    > "$LOG_DQN" 2>&1 &
PID_DQN=$!

echo "[$(date +%H:%M:%S)] Launched 3 experiments: PIDs $PID_UPI $PID_A2C $PID_DQN"
echo "[$(date +%H:%M:%S)] Logs:"
echo "  UPI: $LOG_UPI"
echo "  A2C: $LOG_A2C"
echo "  DQN: $LOG_DQN"

# Quick parallelism check
echo ""
echo "[$(date +%H:%M:%S)] Parallelism check (waiting 60s)..."
sleep 60

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits

echo ""
echo "=== Training Progress ==="
for log in "$LOG_UPI" "$LOG_A2C" "$LOG_DQN"; do
    if [ -f "$log" ]; then
        algo=$(basename "$(dirname "$log")")
        if grep -q "UPI-TRM RL training\|eval_success_rate" "$log" 2>/dev/null; then
            echo "  $algo: Training started ✓"
        else
            build_line=$(tail -n 1 "$log" 2>/dev/null | head -c 80)
            echo "  $algo: $build_line"
        fi
    fi
done

echo ""
echo "[$(date +%H:%M:%S)] Waiting for fast wave to complete..."
wait $PID_UPI $PID_A2C $PID_DQN
echo "[$(date +%H:%M:%S)] Fast wave seed=$SEED complete!"

# Generate summary
echo ""
echo "=== Final Results for seed=$SEED ==="
for log in "$LOG_UPI" "$LOG_A2C" "$LOG_DQN"; do
    if [ -f "$log" ]; then
        algo=$(basename "$(dirname "$log")")
        final_eval=$(grep "eval_success_rate" "$log" | tail -n 1)
        echo "$algo: $final_eval"
    fi
done
