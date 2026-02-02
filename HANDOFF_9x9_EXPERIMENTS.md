# Handoff: 9x9 Sudoku Experiments for Multi-Machine Execution

**Date:** 2026-01-31
**Last Updated:** 2026-01-31 11:18 AM PST
**Purpose:** Instructions for running remaining 9x9 Sudoku experiments on additional machines and merging results for analysis.

---

## CRITICAL: Dataset Path Required

**Always include `--dataset-paths` flag when running experiments!**

Without this flag, the training will use a dummy 4x4 dataset instead of the actual 9x9 Sudoku puzzles.

```bash
--dataset-paths /path/to/trm_bellman/data/sudoku-9x9
```

---

## Current Status (Machine 1)

**Updated:** 2026-01-31 11:18 AM PST

### Completed Experiments

| Algorithm | Seed | Steps | Success Rate | Mean Score | Log File |
|-----------|------|-------|--------------|------------|----------|
| UPI-TRM | 0 | 50,000 | **8%** (12% peak) | 54.08 | `upi_trm_50k_s0.log` |
| DQN | 0 | 50,000 | 0% | 29.30 | `dqn_50k_s0.log` |
| A2C | 0 | 50,000 | 0% | 32.28 | `a2c_50k_s0.log` |

### In Progress (Machine 1)

| Algorithm | Seed | Progress | Est. Completion | Log File |
|-----------|------|----------|-----------------|----------|
| PPO | 0 | 54% | ~22h remaining | `ppo_50k_s0.log` |
| UPI-TRM | 1 | ~0% (restarted) | ~50h remaining | `upi_trm_50k_s1.log` |

**Note:** UPI-TRM seed 1 was restarted on 2026-01-31 after discovering it was using wrong dataset.

---

## Experiments to Run on Other Machines

### Priority 1: Complete Multi-Seed Coverage

| Algorithm | Seeds Needed | Config File | Priority |
|-----------|--------------|-------------|----------|
| UPI-TRM | 2 | `configs/sudoku9x9/upi_trm_9x9_50k.yaml` | HIGH |
| PPO | 1, 2 | `configs/sudoku9x9/ppo_9x9.yaml` | MEDIUM |
| DQN | 1, 2 | `configs/sudoku9x9/dqn_9x9.yaml` | MEDIUM |
| A2C | 1, 2 | `configs/sudoku9x9/a2c_9x9.yaml` | MEDIUM |

### Priority 2: Ablation Studies (After seeds complete)

| Ablation | Config Change | Seeds | Priority |
|----------|---------------|-------|----------|
| Episodic-z | `episodic_latent: true` | 0, 1, 2 | LOW |
| Contraction ON | `enable_contraction: true` | 0, 1, 2 | LOW |

---

## Execution Commands

### Buck2 (on devservers with fbsource)

```bash
cd ~/fbsource/fbcode

# UPI-TRM seed 2
CUDA_VISIBLE_DEVICES=0 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
  -- --config /path/to/trm_bellman/configs/sudoku9x9/upi_trm_9x9_50k.yaml \
  --seed 2 \
  --dataset-paths /path/to/trm_bellman/data/sudoku-9x9 \
  --checkpoint-dir /path/to/trm_bellman/checkpoints/rl_sudoku-9x9_seed2 \
  > /path/to/trm_bellman/results/9x9_experiments_machine2/upi_trm_50k_s2.log 2>&1 &

# PPO seed 1
CUDA_VISIBLE_DEVICES=1 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
  -- --config /path/to/trm_bellman/configs/sudoku9x9/ppo_9x9.yaml \
  --seed 1 \
  --dataset-paths /path/to/trm_bellman/data/sudoku-9x9 \
  --checkpoint-dir /path/to/trm_bellman/checkpoints/rl_ppo-9x9_seed1 \
  > /path/to/trm_bellman/results/9x9_experiments_machine2/ppo_50k_s1.log 2>&1 &
```

### Python Direct (if Buck2 unavailable)

```bash
cd /path/to/trm_bellman

# UPI-TRM seed 2
CUDA_VISIBLE_DEVICES=0 nohup python upi_trm_train.py \
  --config configs/sudoku9x9/upi_trm_9x9_50k.yaml \
  --seed 2 \
  --dataset-paths data/sudoku-9x9 \
  --checkpoint-dir checkpoints/rl_sudoku-9x9_seed2 \
  > results/9x9_experiments_machine2/upi_trm_50k_s2.log 2>&1 &
```

---

## Log File Naming Convention (CRITICAL)

### Standard Format

```
{algorithm}_50k_s{seed}.log
```

### Examples

| Algorithm | Seed | Filename |
|-----------|------|----------|
| UPI-TRM | 0 | `upi_trm_50k_s0.log` |
| UPI-TRM | 1 | `upi_trm_50k_s1.log` |
| UPI-TRM | 2 | `upi_trm_50k_s2.log` |
| PPO | 0 | `ppo_50k_s0.log` |
| PPO | 1 | `ppo_50k_s1.log` |
| DQN | 0 | `dqn_50k_s0.log` |
| DQN | 1 | `dqn_50k_s1.log` |
| A2C | 0 | `a2c_50k_s0.log` |
| A2C | 1 | `a2c_50k_s1.log` |

---

## Directory Structure for Multi-Machine Results

### On Each Machine

Create a results directory with machine identifier:

```
results/
  9x9_experiments_machine1/     # Machine 1 (primary)
    upi_trm_50k_s0.log
    upi_trm_50k_s1.log
    ppo_50k_s0.log
    dqn_50k_s0.log
    a2c_50k_s0.log
  9x9_experiments_machine2/     # Machine 2
    upi_trm_50k_s2.log
    ppo_50k_s1.log
    ppo_50k_s2.log
    dqn_50k_s1.log
    dqn_50k_s2.log
  9x9_experiments_machine3/     # Machine 3 (if needed)
    a2c_50k_s1.log
    a2c_50k_s2.log
```

### Merged Directory (for Analysis)

After all experiments complete, merge into:

```
results/
  9x9_all_seeds/
    upi_trm_50k_s0.log
    upi_trm_50k_s1.log
    upi_trm_50k_s2.log
    ppo_50k_s0.log
    ppo_50k_s1.log
    ppo_50k_s2.log
    dqn_50k_s0.log
    dqn_50k_s1.log
    dqn_50k_s2.log
    a2c_50k_s0.log
    a2c_50k_s1.log
    a2c_50k_s2.log
```

---

## Merging Log Files for Analysis

### Step 1: Copy Logs from Each Machine

```bash
# From Machine 2
scp machine2:/path/to/trm_bellman/results/9x9_experiments_machine2/*.log \
    results/9x9_all_seeds/

# From Machine 3
scp machine3:/path/to/trm_bellman/results/9x9_experiments_machine3/*.log \
    results/9x9_all_seeds/

# From Machine 1 (local)
cp results/9x9_experiments_seed0/*.log results/9x9_all_seeds/
```

### Step 2: Verify All Files Present

```bash
ls -la results/9x9_all_seeds/*.log | wc -l
# Expected: 12 files (4 algorithms x 3 seeds)
```

### Step 3: Check All Runs Completed

```bash
# Check each log ends with step 50000
for log in results/9x9_all_seeds/*.log; do
    echo -n "$log: "
    grep -o '\[step [0-9]*\]' "$log" | tail -1
done
# Expected: All should show [step 50000]
```

---

## Log File Format Reference

### Key Metrics to Parse

Logs contain eval entries every 500 steps in this format:

```
[step 16500] eval_success_rate=0.020 eval_mean_score=33.120 eval_policy_mode=greedy [solved=1/50, score_range=18.00-81.00/81.0, initial=26.84]
```

### Parsing Script (Python)

```python
import re
import os

def parse_log(log_path):
    """Parse a training log and extract eval metrics."""
    results = []
    pattern = r'\[step (\d+)\] eval_success_rate=([\d.]+) eval_mean_score=([\d.]+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                results.append({
                    'step': int(match.group(1)),
                    'success_rate': float(match.group(2)),
                    'mean_score': float(match.group(3)),
                })
    return results

def parse_all_logs(log_dir):
    """Parse all logs in a directory."""
    all_results = {}
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            # Extract algorithm and seed from filename
            # Format: {algo}_50k_s{seed}.log
            parts = filename.replace('.log', '').split('_')
            algo = parts[0]
            seed = int(parts[-1].replace('s', ''))

            log_path = os.path.join(log_dir, filename)
            results = parse_log(log_path)

            if algo not in all_results:
                all_results[algo] = {}
            all_results[algo][seed] = results

    return all_results

# Usage
results = parse_all_logs('results/9x9_all_seeds/')

# Compute mean ± std across seeds for each algorithm
import numpy as np

for algo, seeds in results.items():
    final_success_rates = []
    final_mean_scores = []

    for seed, data in seeds.items():
        if data:
            final = data[-1]  # Last eval point
            final_success_rates.append(final['success_rate'])
            final_mean_scores.append(final['mean_score'])

    print(f"{algo}:")
    print(f"  Success Rate: {np.mean(final_success_rates)*100:.1f}% ± {np.std(final_success_rates)*100:.1f}%")
    print(f"  Mean Score: {np.mean(final_mean_scores):.2f} ± {np.std(final_mean_scores):.2f}")
```

---

## Checkpoint Files

### Naming Convention

```
checkpoints/rl_sudoku-9x9_seed{seed}/rl_checkpoint_step_50000.pt
```

### Checkpoint Contents

Each checkpoint contains:
- Model weights (policy + value networks)
- Optimizer states
- Training step counter
- Config used

### Checkpoints to Save

Only the final checkpoint is needed for analysis:
- `rl_checkpoint_step_50000.pt`

Intermediate checkpoints (every 10k steps) can be deleted after training completes unless needed for learning curve analysis.

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size` to 32
   - Reduce `rollout_episodes_per_step` to 1

2. **Buck2 build failures**
   - Run `hg goto master --clean && hg checkout -`
   - Retry build

3. **Config file not found**
   - Use absolute paths for config files
   - Verify config exists: `ls -la configs/sudoku9x9/`

4. **Process killed unexpectedly**
   - Check `dmesg | tail -20` for OOM killer
   - Check GPU health: `nvidia-smi`

### Monitoring Progress

```bash
# Check current step
grep -oE '\| [0-9]+/50000' results/9x9_experiments_machine2/upi_trm_50k_s2.log | tail -1

# Check latest eval
grep '\[step.*eval_success_rate' results/9x9_experiments_machine2/upi_trm_50k_s2.log | tail -1

# Estimate completion
# PPO: ~1,100 steps/hour
# A2C: ~2,600 steps/hour
# DQN: ~22,000 steps/hour
# UPI-TRM: ~1,000 steps/hour
```

---

## Expected Results (Based on Seed 0)

| Algorithm | Expected Success | Expected Mean Score | Training Time |
|-----------|------------------|---------------------|---------------|
| UPI-TRM | 5-12% | 50-57 | ~50 hours |
| PPO | 0% | 27-30 | ~45 hours |
| DQN | 0% | 28-30 | ~2.5 hours |
| A2C | 0% | 30-33 | ~19 hours |

**Note:** Variance across seeds expected to be high for UPI-TRM (small eval sample of 50 episodes).

---

## After All Experiments Complete

1. **Merge all log files** into `results/9x9_all_seeds/`
2. **Verify completeness** (12 logs, all at step 50000)
3. **Run analysis script** to compute mean ± std
4. **Generate comparison plots** using existing plotting scripts
5. **Update EXPERIMENT_PLAN_ICML.md** with final results
6. **Commit results** with descriptive message

---

## Contact / Questions

For issues or questions, refer to:
- `CLAUDE.md` - Operational guidelines
- `EXPERIMENT_PLAN_ICML.md` - Full experiment details
- `results/9x9_experiments_seed0/EXPERIMENT_REPORT.md` - Detailed analysis of seed 0
