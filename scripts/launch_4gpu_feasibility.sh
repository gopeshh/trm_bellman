#!/bin/bash
# 4-GPU Parallel Launcher for Feasibility Checker Experiments
#
# Runs experiments in waves of 4 using buck2 run with separate isolation dirs.
# Each experiment runs in a separate background process.
# Logs are saved to runs/feasibility/<algo>/<seed>_<timestamp>.log
#
# Usage:
#   ./scripts/launch_4gpu_feasibility.sh [wave1|wave2|wave3|wave4|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FBCODE_DIR="$HOME/fbsource/fbcode"

STEPS=5000
DATASET="buiksat_trm/data/sudoku-4x4-trivial"
LOG_BASE="$REPO_DIR/runs/feasibility"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directories
mkdir -p "$LOG_BASE/upi_trm" "$LOG_BASE/ppo" "$LOG_BASE/a2c" "$LOG_BASE/dqn" \
         "$LOG_BASE/ablation_no_conservative" "$LOG_BASE/ablation_no_contraction"

# Record commands
COMMANDS_FILE="$LOG_BASE/COMMANDS.md"
echo "# Feasibility Experiment Commands" > "$COMMANDS_FILE"
echo "Generated: $(date)" >> "$COMMANDS_FILE"
echo "" >> "$COMMANDS_FILE"

# Parallelism check function
check_parallelism() {
    local wave_name="$1"
    shift
    local log_files=("$@")

    echo ""
    echo "[$(date +%H:%M:%S)] Parallelism check for $wave_name..."
    sleep 90  # Wait for builds to complete and training to start

    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits

    echo ""
    echo "=== Training Progress ==="
    for log in "${log_files[@]}"; do
        if [ -f "$log" ]; then
            algo=$(basename "$(dirname "$log")")
            # Check for training start or eval metrics
            train_line=$(grep -m1 "UPI-TRM RL training\|eval_success_rate" "$log" 2>/dev/null | head -c 100)
            if [ -n "$train_line" ]; then
                echo "  $algo: Training started ✓"
            else
                build_line=$(tail -1 "$log" 2>/dev/null | head -c 80)
                echo "  $algo: $build_line"
            fi
        else
            echo "  $(basename "$log"): Log not found"
        fi
    done
    echo ""
}

# Wave 1: seed=42 (UPI, PPO, A2C, DQN)
run_wave1() {
    echo ""
    echo "=============================================="
    echo "WAVE 1: seed=42 (UPI-TRM, PPO, A2C, DQN)"
    echo "=============================================="

    cd "$FBCODE_DIR"

    local LOG_UPI="$LOG_BASE/upi_trm/42_${TIMESTAMP}.log"
    local LOG_PPO="$LOG_BASE/ppo/42_${TIMESTAMP}.log"
    local LOG_A2C="$LOG_BASE/a2c/42_${TIMESTAMP}.log"
    local LOG_DQN="$LOG_BASE/dqn/42_${TIMESTAMP}.log"

    # Launch all 4 in parallel with unique isolation dirs
    echo "[$(date +%H:%M:%S)] Starting upi_trm (seed 42) on GPU 0"
    CUDA_VISIBLE_DEVICES=0 buck2 --isolation-dir "gpu0_upi_s42" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/rl_sudoku_4x4_feasibility.yaml \
        --train-steps "$STEPS" --seed 42 \
        > "$LOG_UPI" 2>&1 &
    PID0=$!

    echo "[$(date +%H:%M:%S)] Starting ppo (seed 42) on GPU 1"
    CUDA_VISIBLE_DEVICES=1 buck2 --isolation-dir "gpu1_ppo_s42" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline ppo --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 42 \
        > "$LOG_PPO" 2>&1 &
    PID1=$!

    echo "[$(date +%H:%M:%S)] Starting a2c (seed 42) on GPU 2"
    CUDA_VISIBLE_DEVICES=2 buck2 --isolation-dir "gpu2_a2c_s42" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline a2c --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 42 \
        > "$LOG_A2C" 2>&1 &
    PID2=$!

    echo "[$(date +%H:%M:%S)] Starting dqn (seed 42) on GPU 3"
    CUDA_VISIBLE_DEVICES=3 buck2 --isolation-dir "gpu3_dqn_s42" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline dqn --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 42 \
        > "$LOG_DQN" 2>&1 &
    PID3=$!

    echo "[$(date +%H:%M:%S)] All 4 experiments launched: PIDs $PID0 $PID1 $PID2 $PID3"
    echo "[$(date +%H:%M:%S)] Logs: $LOG_UPI, $LOG_PPO, $LOG_A2C, $LOG_DQN"

    # Check parallelism after launch
    check_parallelism "Wave 1" "$LOG_UPI" "$LOG_PPO" "$LOG_A2C" "$LOG_DQN"

    echo "[$(date +%H:%M:%S)] Waiting for Wave 1 to complete..."
    wait $PID0 $PID1 $PID2 $PID3
    echo "[$(date +%H:%M:%S)] Wave 1 complete!"
}

# Wave 2: seed=123 (UPI, PPO, A2C, DQN)
run_wave2() {
    echo ""
    echo "=============================================="
    echo "WAVE 2: seed=123 (UPI-TRM, PPO, A2C, DQN)"
    echo "=============================================="

    cd "$FBCODE_DIR"

    local LOG_UPI="$LOG_BASE/upi_trm/123_${TIMESTAMP}.log"
    local LOG_PPO="$LOG_BASE/ppo/123_${TIMESTAMP}.log"
    local LOG_A2C="$LOG_BASE/a2c/123_${TIMESTAMP}.log"
    local LOG_DQN="$LOG_BASE/dqn/123_${TIMESTAMP}.log"

    echo "[$(date +%H:%M:%S)] Starting upi_trm (seed 123) on GPU 0"
    CUDA_VISIBLE_DEVICES=0 buck2 --isolation-dir "gpu0_upi_s123" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/rl_sudoku_4x4_feasibility.yaml \
        --train-steps "$STEPS" --seed 123 \
        > "$LOG_UPI" 2>&1 &
    PID0=$!

    echo "[$(date +%H:%M:%S)] Starting ppo (seed 123) on GPU 1"
    CUDA_VISIBLE_DEVICES=1 buck2 --isolation-dir "gpu1_ppo_s123" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline ppo --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 123 \
        > "$LOG_PPO" 2>&1 &
    PID1=$!

    echo "[$(date +%H:%M:%S)] Starting a2c (seed 123) on GPU 2"
    CUDA_VISIBLE_DEVICES=2 buck2 --isolation-dir "gpu2_a2c_s123" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline a2c --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 123 \
        > "$LOG_A2C" 2>&1 &
    PID2=$!

    echo "[$(date +%H:%M:%S)] Starting dqn (seed 123) on GPU 3"
    CUDA_VISIBLE_DEVICES=3 buck2 --isolation-dir "gpu3_dqn_s123" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline dqn --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 123 \
        > "$LOG_DQN" 2>&1 &
    PID3=$!

    echo "[$(date +%H:%M:%S)] All 4 experiments launched: PIDs $PID0 $PID1 $PID2 $PID3"

    check_parallelism "Wave 2" "$LOG_UPI" "$LOG_PPO" "$LOG_A2C" "$LOG_DQN"

    echo "[$(date +%H:%M:%S)] Waiting for Wave 2 to complete..."
    wait $PID0 $PID1 $PID2 $PID3
    echo "[$(date +%H:%M:%S)] Wave 2 complete!"
}

# Wave 3: seed=456 (UPI, PPO, A2C, DQN)
run_wave3() {
    echo ""
    echo "=============================================="
    echo "WAVE 3: seed=456 (UPI-TRM, PPO, A2C, DQN)"
    echo "=============================================="

    cd "$FBCODE_DIR"

    local LOG_UPI="$LOG_BASE/upi_trm/456_${TIMESTAMP}.log"
    local LOG_PPO="$LOG_BASE/ppo/456_${TIMESTAMP}.log"
    local LOG_A2C="$LOG_BASE/a2c/456_${TIMESTAMP}.log"
    local LOG_DQN="$LOG_BASE/dqn/456_${TIMESTAMP}.log"

    echo "[$(date +%H:%M:%S)] Starting upi_trm (seed 456) on GPU 0"
    CUDA_VISIBLE_DEVICES=0 buck2 --isolation-dir "gpu0_upi_s456" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/rl_sudoku_4x4_feasibility.yaml \
        --train-steps "$STEPS" --seed 456 \
        > "$LOG_UPI" 2>&1 &
    PID0=$!

    echo "[$(date +%H:%M:%S)] Starting ppo (seed 456) on GPU 1"
    CUDA_VISIBLE_DEVICES=1 buck2 --isolation-dir "gpu1_ppo_s456" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline ppo --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 456 \
        > "$LOG_PPO" 2>&1 &
    PID1=$!

    echo "[$(date +%H:%M:%S)] Starting a2c (seed 456) on GPU 2"
    CUDA_VISIBLE_DEVICES=2 buck2 --isolation-dir "gpu2_a2c_s456" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline a2c --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 456 \
        > "$LOG_A2C" 2>&1 &
    PID2=$!

    echo "[$(date +%H:%M:%S)] Starting dqn (seed 456) on GPU 3"
    CUDA_VISIBLE_DEVICES=3 buck2 --isolation-dir "gpu3_dqn_s456" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --baseline dqn --dataset-paths "$DATASET" --config buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml \
        --train-steps "$STEPS" --seed 456 \
        > "$LOG_DQN" 2>&1 &
    PID3=$!

    echo "[$(date +%H:%M:%S)] All 4 experiments launched: PIDs $PID0 $PID1 $PID2 $PID3"

    check_parallelism "Wave 3" "$LOG_UPI" "$LOG_PPO" "$LOG_A2C" "$LOG_DQN"

    echo "[$(date +%H:%M:%S)] Waiting for Wave 3 to complete..."
    wait $PID0 $PID1 $PID2 $PID3
    echo "[$(date +%H:%M:%S)] Wave 3 complete!"
}

# Wave 4: Ablations (all 3 seeds)
run_wave4() {
    echo ""
    echo "=============================================="
    echo "WAVE 4a: Ablations batch 1 (no_conservative: 42, 123, 456; no_contraction: 42)"
    echo "=============================================="

    cd "$FBCODE_DIR"

    local LOG_NC42="$LOG_BASE/ablation_no_conservative/42_${TIMESTAMP}.log"
    local LOG_NC123="$LOG_BASE/ablation_no_conservative/123_${TIMESTAMP}.log"
    local LOG_NC456="$LOG_BASE/ablation_no_conservative/456_${TIMESTAMP}.log"
    local LOG_NCTR42="$LOG_BASE/ablation_no_contraction/42_${TIMESTAMP}.log"

    echo "[$(date +%H:%M:%S)] Starting no_conservative (seed 42) on GPU 0"
    CUDA_VISIBLE_DEVICES=0 buck2 --isolation-dir "gpu0_nocons_s42" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_conservative.yaml \
        --train-steps "$STEPS" --seed 42 \
        > "$LOG_NC42" 2>&1 &
    PID0=$!

    echo "[$(date +%H:%M:%S)] Starting no_conservative (seed 123) on GPU 1"
    CUDA_VISIBLE_DEVICES=1 buck2 --isolation-dir "gpu1_nocons_s123" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_conservative.yaml \
        --train-steps "$STEPS" --seed 123 \
        > "$LOG_NC123" 2>&1 &
    PID1=$!

    echo "[$(date +%H:%M:%S)] Starting no_conservative (seed 456) on GPU 2"
    CUDA_VISIBLE_DEVICES=2 buck2 --isolation-dir "gpu2_nocons_s456" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_conservative.yaml \
        --train-steps "$STEPS" --seed 456 \
        > "$LOG_NC456" 2>&1 &
    PID2=$!

    echo "[$(date +%H:%M:%S)] Starting no_contraction (seed 42) on GPU 3"
    CUDA_VISIBLE_DEVICES=3 buck2 --isolation-dir "gpu3_nocontr_s42" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml \
        --train-steps "$STEPS" --seed 42 \
        > "$LOG_NCTR42" 2>&1 &
    PID3=$!

    echo "[$(date +%H:%M:%S)] All 4 experiments launched: PIDs $PID0 $PID1 $PID2 $PID3"

    check_parallelism "Wave 4a" "$LOG_NC42" "$LOG_NC123" "$LOG_NC456" "$LOG_NCTR42"

    echo "[$(date +%H:%M:%S)] Waiting for Wave 4a to complete..."
    wait $PID0 $PID1 $PID2 $PID3
    echo "[$(date +%H:%M:%S)] Wave 4a complete!"

    echo ""
    echo "=============================================="
    echo "WAVE 4b: Ablations batch 2 (no_contraction: 123, 456)"
    echo "=============================================="

    local LOG_NCTR123="$LOG_BASE/ablation_no_contraction/123_${TIMESTAMP}.log"
    local LOG_NCTR456="$LOG_BASE/ablation_no_contraction/456_${TIMESTAMP}.log"

    echo "[$(date +%H:%M:%S)] Starting no_contraction (seed 123) on GPU 0"
    CUDA_VISIBLE_DEVICES=0 buck2 --isolation-dir "gpu0_nocontr_s123" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml \
        --train-steps "$STEPS" --seed 123 \
        > "$LOG_NCTR123" 2>&1 &
    PID0=$!

    echo "[$(date +%H:%M:%S)] Starting no_contraction (seed 456) on GPU 1"
    CUDA_VISIBLE_DEVICES=1 buck2 --isolation-dir "gpu1_nocontr_s456" run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
        --dataset-paths "$DATASET" --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml \
        --train-steps "$STEPS" --seed 456 \
        > "$LOG_NCTR456" 2>&1 &
    PID1=$!

    echo "[$(date +%H:%M:%S)] 2 experiments launched: PIDs $PID0 $PID1"
    echo "[$(date +%H:%M:%S)] Waiting for Wave 4b to complete..."

    wait $PID0 $PID1
    echo "[$(date +%H:%M:%S)] Wave 4b complete!"
}

# Main execution
case "${1:-all}" in
    wave1)
        run_wave1
        ;;
    wave2)
        run_wave2
        ;;
    wave3)
        run_wave3
        ;;
    wave4)
        run_wave4
        ;;
    all)
        echo "=============================================="
        echo "4-GPU Feasibility Checker Experiment Suite"
        echo "=============================================="
        echo "Started: $(date)"
        echo "Timestamp: $TIMESTAMP"
        echo "Steps: $STEPS"
        echo "Dataset: $DATASET"
        echo "Logs: $LOG_BASE/"
        echo "=============================================="

        run_wave1
        run_wave2
        run_wave3
        run_wave4

        echo ""
        echo "=============================================="
        echo "ALL EXPERIMENTS COMPLETED!"
        echo "Finished: $(date)"
        echo "Logs saved to: $LOG_BASE/"
        echo "=============================================="
        ;;
    *)
        echo "Usage: $0 [wave1|wave2|wave3|wave4|all]"
        exit 1
        ;;
esac
