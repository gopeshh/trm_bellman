# 9x9 Sudoku Experiments - Handoff Document

**Date:** 2026-02-03
**Author:** Claude Code
**Status:** 10/12 experiments complete, 2 running

---

## Quick Summary

**Main Result:** UPI-TRM achieves **4% success rate** on 9x9 Sudoku, while all baselines (DQN, A2C, PPO) achieve **0%**.

---

## Experiment Status

### Completed (10/12)

| Algorithm | Seed | Success | Score | Log File |
|-----------|------|---------|-------|----------|
| UPI-TRM | 0 | **8%** | 54.08 | `upi_trm_50k_s0.log` |
| UPI-TRM | 1 | 0% | 51.94 | `upi_trm_50k_s1.log` |
| UPI-TRM | 2 | **4%** | 53.76 | `upi_trm_50k_s2.log` |
| DQN | 0 | 0% | 29.30 | `dqn_50k_s0.log` |
| DQN | 1 | 0% | 29.90 | `dqn_50k_s1.log` |
| DQN | 2 | 0% | 29.28 | `dqn_50k_s2.log` |
| A2C | 0 | 0% | 32.28 | `a2c_50k_s0.log` |
| A2C | 1 | 0% | 31.98 | `a2c_50k_s1.log` |
| A2C | 2 | 0% | 31.58 | `a2c_50k_s2.log` |
| PPO | 1 | 0% | 29.72 | `ppo_50k_s1.log` |

### Running (2/12)

| Algorithm | Seed | Progress | GPU | ETA |
|-----------|------|----------|-----|-----|
| PPO | 0 | 2% | 0 | ~35h |
| PPO | 2 | 2% | 1 | ~35h |

---

## How to Check Progress

```bash
# Check GPU status
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# Check experiment progress
for f in ~/trm_bellman/results/9x9_experiments_seed0/*50k*.log; do
  grep "eval_success" "$f" | tail -1
done

# Monitor log (updates every 60 min)
cat ~/trm_bellman/results/9x9_monitor.log
```

---

## How to Run Additional Experiments

### Start a new experiment

```bash
cd ~/fbsource/fbcode

# UPI-TRM
CUDA_VISIBLE_DEVICES=0 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only -- \
  --config buiksat_trm/configs/sudoku9x9/upi_trm_9x9_50k.yaml \
  --dataset-paths buiksat_trm/data/sudoku-9x9 --seed <SEED> --no-wandb \
  > ~/trm_bellman/results/9x9_experiments_seed0/upi_trm_50k_s<SEED>.log 2>&1 &

# DQN/A2C/PPO (use respective config)
CUDA_VISIBLE_DEVICES=0 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only -- \
  --config buiksat_trm/configs/sudoku9x9/<ALGO>_9x9.yaml \
  --dataset-paths buiksat_trm/data/sudoku-9x9 --seed <SEED> --no-wandb \
  > ~/trm_bellman/results/9x9_experiments_seed0/<ALGO>_50k_s<SEED>.log 2>&1 &
```

---

## Key Files

| File | Location |
|------|----------|
| Experiment logs | `~/trm_bellman/results/9x9_experiments_seed0/` |
| Configs | `~/fbsource/fbcode/buiksat_trm/configs/sudoku9x9/` |
| Dataset | `~/fbsource/fbcode/buiksat_trm/data/sudoku-9x9/` |
| Report | `~/trm_bellman/results/9x9_experiments_seed0/EXPERIMENT_REPORT.md` |

---

## Git Status

```bash
# Branch
feature/upi-trm-clean

# Recent commits
81ec9d4 Merge remote branch and resolve log conflicts
6dd55a2 update 4 gpu
d3b6af5 Add 9x9 Sudoku experiment results (multi-seed)
```

---

## TODO for Next Session

1. [ ] Wait for PPO s0, s2 to complete (~35h remaining)
2. [ ] Update EXPERIMENT_REPORT.md with final PPO results
3. [ ] Generate summary figures for paper
4. [ ] Commit final results

---

## Important Notes

1. **Use Buck2 for all training** - Python scripts don't have correct dependencies
2. **Always specify `--dataset-paths buiksat_trm/data/sudoku-9x9`** - Otherwise defaults to 4x4
3. **Check CLAUDE.md for config guidelines** - Especially `disable_value_head_norm: true`

---

*Last updated: 2026-02-03 09:54 PST*
