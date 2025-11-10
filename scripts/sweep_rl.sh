#!/bin/bash
# Ablation study sweep for TRM RL training
# Systematically varies key hyperparameters and logs results

set -e  # Exit on error

# Configuration
OUTPUT_DIR="runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${OUTPUT_DIR}/sweep_${TIMESTAMP}"
RESULTS_FILE="${SWEEP_DIR}/sweep_results.csv"

# Create output directory
mkdir -p "${SWEEP_DIR}"

echo "============================================"
echo "TRM RL Ablation Sweep"
echo "============================================"
echo "Output directory: ${SWEEP_DIR}"
echo "Results file: ${RESULTS_FILE}"
echo ""

# Initialize CSV header
echo "run_id,K,n_inner,spectral_target_prod,ppo_clip,status,final_loss,best_accuracy,steps_to_solution,runtime_sec" > "${RESULTS_FILE}"

# Hyperparameter ranges
K_VALUES=(1 3 5)
N_INNER_VALUES=(2 4 6 8)
SPECTRAL_TARGET_VALUES=(0.90 0.95 0.99)
PPO_CLIP_VALUES=(0.1 0.2 0.3)

# Base configuration (common across all runs)
BASE_CMD="python train_rl.py \
  arch=trm \
  data_paths=[data/sudoku-extreme-1k-aug-1000] \
  rl.enabled=True \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  epochs=1"

# Counter for run IDs
run_id=0
total_runs=$((${#K_VALUES[@]} * ${#N_INNER_VALUES[@]} * ${#SPECTRAL_TARGET_VALUES[@]} * ${#PPO_CLIP_VALUES[@]}))

echo "Total experiments to run: ${total_runs}"
echo ""

# Nested loops for Cartesian product of hyperparameters
for K in "${K_VALUES[@]}"; do
  for n_inner in "${N_INNER_VALUES[@]}"; do
    for spectral_target in "${SPECTRAL_TARGET_VALUES[@]}"; do
      for ppo_clip in "${PPO_CLIP_VALUES[@]}"; do
        run_id=$((run_id + 1))

        run_name="rl_K${K}_n${n_inner}_sn${spectral_target}_ppo${ppo_clip}"
        run_dir="${SWEEP_DIR}/${run_name}"
        mkdir -p "${run_dir}"

        echo "----------------------------------------"
        echo "Run ${run_id}/${total_runs}: ${run_name}"
        echo "  K=${K}, n_inner=${n_inner}, spectral_target=${spectral_target}, ppo_clip=${ppo_clip}"
        echo "----------------------------------------"

        # Start timer
        start_time=$(date +%s)

        # Run training
        status="success"
        final_loss="N/A"
        best_accuracy="N/A"
        steps_to_solution="N/A"

        if ${BASE_CMD} \
          rl.K=${K} \
          rl.n_inner=${n_inner} \
          rl.spectral.target_prod=${spectral_target} \
          rl.ppo_clip=${ppo_clip} \
          +run_name="${run_name}" \
          +checkpoint_path="${run_dir}" \
          > "${run_dir}/training.log" 2>&1; then

          echo "✓ Training completed successfully"

          # Parse results from log file
          if [ -f "${run_dir}/training.log" ]; then
            # Extract final loss (last occurrence of loss/total)
            final_loss=$(grep -o "loss/total: [0-9.]*" "${run_dir}/training.log" | tail -1 | awk '{print $2}' || echo "N/A")

            # Extract best accuracy (if logged)
            best_accuracy=$(grep -o "accuracy: [0-9.]*" "${run_dir}/training.log" | sort -rn -k2 | head -1 | awk '{print $2}' || echo "N/A")

            # Count total steps (number of step log entries)
            steps_to_solution=$(grep -c "Step [0-9]*]" "${run_dir}/training.log" || echo "N/A")
          fi
        else
          echo "✗ Training failed"
          status="failed"
        fi

        # End timer
        end_time=$(date +%s)
        runtime=$((end_time - start_time))

        # Log results to CSV
        echo "${run_id},${K},${n_inner},${spectral_target},${ppo_clip},${status},${final_loss},${best_accuracy},${steps_to_solution},${runtime}" >> "${RESULTS_FILE}"

        echo "  Status: ${status}"
        echo "  Final loss: ${final_loss}"
        echo "  Best accuracy: ${best_accuracy}"
        echo "  Steps: ${steps_to_solution}"
        echo "  Runtime: ${runtime}s"
        echo ""
      done
    done
  done
done

echo "============================================"
echo "Sweep completed!"
echo "============================================"
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Summary:"
cat "${RESULTS_FILE}" | column -t -s ','
echo ""

# Generate JSON summary
python scripts/parse_sweep_results.py "${SWEEP_DIR}" || echo "Warning: Could not generate JSON summary (parse_sweep_results.py not found or failed)"

echo "To analyze results:"
echo "  cat ${RESULTS_FILE}"
echo "  python scripts/parse_sweep_results.py ${SWEEP_DIR}"
