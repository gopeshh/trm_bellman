# Handoff: 9x9 Sudoku Experiments

**Date:** 2026-02-03
**Last Updated:** 2026-02-03 10:00 AM PST
**Status:** 10/12 experiments complete, 2 running

---

## Executive Summary

**UPI-TRM achieves 4% average success rate on 9x9 Sudoku while all baselines (DQN, A2C, PPO) achieve 0%.**

| Algorithm | Seed 0 | Seed 1 | Seed 2 | Avg Success | Avg Score |
|-----------|--------|--------|--------|-------------|-----------|
| **UPI-TRM** | **8%** | 0% | **4%** | **4.0%** | **53.3** |
| DQN | 0% | 0% | 0% | 0% | 29.5 |
| A2C | 0% | 0% | 0% | 0% | 31.9 |
| PPO | running | 0% | running | 0% | 29.7 |

---

## Experiment Status

### Completed (10/12)

| Algorithm | Seed | Success Rate | Mean Score | Log File |
|-----------|------|--------------|------------|----------|
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
| PPO | 0 | ~2% | 0 | ~35h |
| PPO | 2 | ~2% | 1 | ~35h |

---

## Quick Commands

### Check Progress

```bash
# GPU status
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# All experiments status
for f in ~/trm_bellman/results/9x9_experiments_seed0/*50k*.log; do
  name=$(basename "$f" .log)
  last_eval=$(grep "eval_success" "$f" | tail -1)
  step=$(echo "$last_eval" | grep -oP '\[step \K\d+')
  success=$(echo "$last_eval" | grep -oP 'eval_success_rate=\K[0-9.]+')
  printf "%-18s step %5s  success=%s\n" "$name" "$step" "$success"
done
```

### Run New Experiment

```bash
cd ~/fbsource/fbcode

CUDA_VISIBLE_DEVICES=<GPU> nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only -- \
  --config buiksat_trm/configs/sudoku9x9/<CONFIG>.yaml \
  --dataset-paths buiksat_trm/data/sudoku-9x9 --seed <SEED> --no-wandb \
  > ~/trm_bellman/results/9x9_experiments_seed0/<ALGO>_50k_s<SEED>.log 2>&1 &
```

---

## Key Files

| Description | Path |
|-------------|------|
| Experiment logs | `~/trm_bellman/results/9x9_experiments_seed0/` |
| Detailed report | `~/trm_bellman/results/9x9_experiments_seed0/EXPERIMENT_REPORT.md` |
| Configs | `~/fbsource/fbcode/buiksat_trm/configs/sudoku9x9/` |
| Dataset | `~/fbsource/fbcode/buiksat_trm/data/sudoku-9x9/` |

---

## CRITICAL Notes

1. **Always use `--dataset-paths buiksat_trm/data/sudoku-9x9`** - Otherwise defaults to 4x4
2. **Use Buck2 for training** - Direct Python lacks dependencies
3. **Check CLAUDE.md** - For config guidelines (e.g., `disable_value_head_norm: true`)

---

## TODO

1. [ ] Wait for PPO s0, s2 to complete (~35h)
2. [ ] Update reports with final PPO results
3. [ ] Generate summary figures for paper

---

*Last updated: 2026-02-03 10:00 AM PST*
