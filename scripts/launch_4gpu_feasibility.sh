#!/bin/bash
# 4-GPU Parallel Launcher for Feasibility Checker Experiments
#
# Runs experiments in waves of 4 using buck2 run.
# Each experiment runs in a separate background process.
# Logs are saved to runs/feasibility/<algo>/<seed>.log
#
# Usage:
#   ./scripts/launch_4gpu_feasibility.sh [wave1|wave2|wave3|wave4|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FBCODE_DIR="$HOME/fbsource/fbcode"

STEPS=5000
DATASET="buiksat_trm/data/sudoku-4x4-trivial"
LOG_BASE="$REPO_DIR/runs/feasibility"

# Create log directories
mkdir -p "$LOG_BASE/upi_trm" "$LOG_BASE/ppo" "$LOG_BASE/a2c" "$LOG_BASE/dqn" \
         "$LOG_BASE/ablation_no_conservative" "$LOG_BASE/ablation_no_contraction"

# Record commands
COMMANDS_FILE="$LOG_BASE/COMMANDS.md"
echo "# Feasibility Experiment Commands" > "$COMMANDS_FILE"
echo "Generated: $(date)" >> "$COMMANDS_FILE"
echo "" >> "$COMMANDS_FILE"

# Function to run a single experiment on a specific GPU
# Returns the PID
run_experiment() {
    local gpu=$1
    local name=$2
    local config=$3
    local seed=$4
    local extra_args="${5:-}"
    local log_dir=$6

    local log_file="$log_dir/${seed}.log"
    local full_config="buiksat_trm/$config"

    echo "# $name seed=$seed on GPU $gpu" >> "$COMMANDS_FILE"
    echo "CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train ... --seed $seed $extra_args" >> "$COMMANDS_FILE"
    echo "" >> "$COMMANDS_FILE"

    echo "[$(date +%H:%M:%S)] Starting $name (seed $seed) on GPU $gpu -> $log_file"

    # Run in a subshell to avoid issues with job control
    (
        cd "$FBCODE_DIR"
        CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
            -c fbcode.nvcc_arch=a100 \
            -c fbcode.enable_gpu_sections=true \
            -- \
            --dataset-paths "$DATASET" \
            --config "$full_config" \
            --train-steps "$STEPS" \
            --seed "$seed" \
            $extra_args \
            > "$log_file" 2>&1
    ) &

    local pid=$!
    echo "  -> PID: $pid"
    echo $pid
}

# Wait for a list of PIDs
wait_for_pids() {
    local wave_name=$1
    shift
    local pids=("$@")

    echo "[$(date +%H:%M:%S)] Waiting for $wave_name (${#pids[@]} jobs)..."

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait $pid 2>/dev/null; then
            echo "[ERROR] Process $pid failed"
            failed=$((failed + 1))
        else
            echo "[OK] Process $pid completed"
        fi
    done

    if [ $failed -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] $wave_name completed successfully!"
    else
        echo "[$(date +%H:%M:%S)] $wave_name completed with $failed failures"
    fi
    return $failed
}

# Wave 1: seed=42 (UPI, PPO, A2C, DQN)
run_wave1() {
    echo ""
    echo "=============================================="
    echo "WAVE 1: seed=42 (UPI-TRM, PPO, A2C, DQN)"
    echo "=============================================="

    local pids=()
    local pid

    pid=$(run_experiment 0 "upi_trm" "configs/rl_sudoku_4x4_feasibility.yaml" 42 "" "$LOG_BASE/upi_trm" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 1 "ppo" "configs/baselines/ppo_trm_feasibility.yaml" 42 "--baseline ppo" "$LOG_BASE/ppo" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 2 "a2c" "configs/baselines/a2c_trm_feasibility.yaml" 42 "--baseline a2c" "$LOG_BASE/a2c" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 3 "dqn" "configs/baselines/dqn_trm_feasibility.yaml" 42 "--baseline dqn" "$LOG_BASE/dqn" | tail -1)
    pids+=($pid)

    wait_for_pids "Wave 1" "${pids[@]}"
}

# Wave 2: seed=123 (UPI, PPO, A2C, DQN)
run_wave2() {
    echo ""
    echo "=============================================="
    echo "WAVE 2: seed=123 (UPI-TRM, PPO, A2C, DQN)"
    echo "=============================================="

    local pids=()
    local pid

    pid=$(run_experiment 0 "upi_trm" "configs/rl_sudoku_4x4_feasibility.yaml" 123 "" "$LOG_BASE/upi_trm" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 1 "ppo" "configs/baselines/ppo_trm_feasibility.yaml" 123 "--baseline ppo" "$LOG_BASE/ppo" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 2 "a2c" "configs/baselines/a2c_trm_feasibility.yaml" 123 "--baseline a2c" "$LOG_BASE/a2c" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 3 "dqn" "configs/baselines/dqn_trm_feasibility.yaml" 123 "--baseline dqn" "$LOG_BASE/dqn" | tail -1)
    pids+=($pid)

    wait_for_pids "Wave 2" "${pids[@]}"
}

# Wave 3: seed=456 (UPI, PPO, A2C, DQN)
run_wave3() {
    echo ""
    echo "=============================================="
    echo "WAVE 3: seed=456 (UPI-TRM, PPO, A2C, DQN)"
    echo "=============================================="

    local pids=()
    local pid

    pid=$(run_experiment 0 "upi_trm" "configs/rl_sudoku_4x4_feasibility.yaml" 456 "" "$LOG_BASE/upi_trm" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 1 "ppo" "configs/baselines/ppo_trm_feasibility.yaml" 456 "--baseline ppo" "$LOG_BASE/ppo" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 2 "a2c" "configs/baselines/a2c_trm_feasibility.yaml" 456 "--baseline a2c" "$LOG_BASE/a2c" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 3 "dqn" "configs/baselines/dqn_trm_feasibility.yaml" 456 "--baseline dqn" "$LOG_BASE/dqn" | tail -1)
    pids+=($pid)

    wait_for_pids "Wave 3" "${pids[@]}"
}

# Wave 4: Ablations (all 3 seeds)
run_wave4() {
    echo ""
    echo "=============================================="
    echo "WAVE 4a: Ablations batch 1"
    echo "=============================================="

    local pids=()
    local pid

    pid=$(run_experiment 0 "ablation_no_conservative" "configs/ablations/upi_trm_feasibility_no_conservative.yaml" 42 "" "$LOG_BASE/ablation_no_conservative" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 1 "ablation_no_conservative" "configs/ablations/upi_trm_feasibility_no_conservative.yaml" 123 "" "$LOG_BASE/ablation_no_conservative" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 2 "ablation_no_conservative" "configs/ablations/upi_trm_feasibility_no_conservative.yaml" 456 "" "$LOG_BASE/ablation_no_conservative" | tail -1)
    pids+=($pid)

    pid=$(run_experiment 3 "ablation_no_contraction" "configs/ablations/upi_trm_feasibility_no_contraction.yaml" 42 "" "$LOG_BASE/ablation_no_contraction" | tail -1)
    pids+=($pid)

    wait_for_pids "Wave 4a" "${pids[@]}"

    echo ""
    echo "=============================================="
    echo "WAVE 4b: Ablations batch 2"
    echo "=============================================="

    local pids2=()

    pid=$(run_experiment 0 "ablation_no_contraction" "configs/ablations/upi_trm_feasibility_no_contraction.yaml" 123 "" "$LOG_BASE/ablation_no_contraction" | tail -1)
    pids2+=($pid)

    pid=$(run_experiment 1 "ablation_no_contraction" "configs/ablations/upi_trm_feasibility_no_contraction.yaml" 456 "" "$LOG_BASE/ablation_no_contraction" | tail -1)
    pids2+=($pid)

    wait_for_pids "Wave 4b" "${pids2[@]}"
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
        echo "Commands recorded in: $COMMANDS_FILE"
        echo "=============================================="
        ;;
    *)
        echo "Usage: $0 [wave1|wave2|wave3|wave4|all]"
        exit 1
        ;;
esac
