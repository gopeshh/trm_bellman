# HANDOFF.md - UPI-TRM Project

**Date:** 2026-02-03
**Branch:** `feature/upi-trm-clean`
**Status:** 9x9 experiments 10/12 complete

---

## Executive Summary

**UPI-TRM achieves 4% success rate on 9x9 Sudoku while all baselines (DQN, A2C, PPO) achieve 0%.**

| Algorithm | Seed 0 | Seed 1 | Seed 2 | Avg Success | Avg Score |
|-----------|--------|--------|--------|-------------|-----------|
| **UPI-TRM** | **8%** | 0% | **4%** | **4.0%** | **53.3** |
| DQN | 0% | 0% | 0% | 0% | 29.5 |
| A2C | 0% | 0% | 0% | 0% | 31.9 |
| PPO | running | 0% | running | 0% | 29.7 |

---

## Current Status

### 9x9 Experiments (10/12 complete)

| Algorithm | Seed | Success | Score | Status |
|-----------|------|---------|-------|--------|
| UPI-TRM | 0 | **8%** | 54.08 | ✅ Done |
| UPI-TRM | 1 | 0% | 51.94 | ✅ Done |
| UPI-TRM | 2 | **4%** | 53.76 | ✅ Done |
| DQN | 0 | 0% | 29.30 | ✅ Done |
| DQN | 1 | 0% | 29.90 | ✅ Done |
| DQN | 2 | 0% | 29.28 | ✅ Done |
| A2C | 0 | 0% | 32.28 | ✅ Done |
| A2C | 1 | 0% | 31.98 | ✅ Done |
| A2C | 2 | 0% | 31.58 | ✅ Done |
| PPO | 0 | - | - | 🔄 Running |
| PPO | 1 | 0% | 29.72 | ✅ Done |
| PPO | 2 | - | - | 🔄 Running |

**Logs:** `results/9x9_experiments_seed0/`
**Report:** `results/9x9_experiments_seed0/EXPERIMENT_REPORT.md`

---

## Quick Commands

### Check Experiment Progress

```bash
# GPU status
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# All 9x9 experiments
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

**Available configs:**
- `upi_trm_9x9_50k.yaml` - UPI-TRM
- `dqn_9x9.yaml` - DQN baseline
- `a2c_9x9.yaml` - A2C baseline
- `ppo_9x9.yaml` - PPO baseline

---

## Key Files

| Description | Path |
|-------------|------|
| Experiment logs | `results/9x9_experiments_seed0/*.log` |
| Detailed report | `results/9x9_experiments_seed0/EXPERIMENT_REPORT.md` |
| Configs | `~/fbsource/fbcode/buiksat_trm/configs/sudoku9x9/` |
| Dataset | `~/fbsource/fbcode/buiksat_trm/data/sudoku-9x9/` |
| Project guidelines | `CLAUDE.md` |

---

## Critical Notes

1. **Always use `--dataset-paths buiksat_trm/data/sudoku-9x9`** - Otherwise defaults to 4x4
2. **Use Buck2 for training** - Direct Python lacks dependencies
3. **Config guidelines from CLAUDE.md:**
   - `use_feasibility_checker: true` (required)
   - `disable_value_head_norm: true` (for stability)

---

## Previous Experiments (Completed)

### 4x4 Hard Sudoku (Table 3)
- Location: `results/table3_hard_controlled/`
- Result: UPI-TRM 57% vs baselines 0%

### 4x4 Trivial Sudoku
- Location: `results/table3_baselines/`
- Result: UPI-TRM 90-93% vs baselines 28-31%

---

## TODO

- [ ] Wait for PPO s0, s2 to complete (~35h remaining)
- [ ] Update report with final PPO results
- [ ] Generate summary figures for paper

---

*Last updated: 2026-02-03*
