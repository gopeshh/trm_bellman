# HANDOFF.md - UPI-TRM Project

**Date:** 2026-02-08
**Branch:** `feature/upi-trm-clean`
**Status:** 9x9 experiments complete (all seeds at 50k steps)

---

## Executive Summary

**UPI-TRM achieves 4% success rate on 9x9 Sudoku while all baselines (DQN, A2C, PPO) achieve 0%.**

| Algorithm | Seed 0 | Seed 1 | Seed 2 | Avg Success | Avg Score |
|-----------|--------|--------|--------|-------------|-----------|
| **UPI-TRM** | **8%** | 0% | **4%** | **4.0%** | **53.3** |
| DQN | 0% | 0% | 0% | 0% | 29.5 |
| A2C | 0% | 0% | 0% | 0% | 31.9 |
| PPO | 0% | 0% | 0% | 0% | 29.5 |

---

## Next Session: Continue PPO Seeds 0, 2 to 50k

### Current State

PPO seeds 0 and 2 completed **25k steps** (not full 50k). Seed 1 has full 50k.

| Seed | Current Steps | Target | Status |
|------|---------------|--------|--------|
| PPO s0 | 25,000 | 50,000 | ⚠️ Need 25k more |
| PPO s1 | 50,000 | 50,000 | ✅ Complete |
| PPO s2 | 25,000 | 50,000 | ⚠️ Need 25k more |

**Log files:**
- `results/9x9_experiments_seed0/ppo_25k_s0.log` (25k complete)
- `results/9x9_experiments_seed0/ppo_50k_s1.log` (50k complete)
- `results/9x9_experiments_seed0/ppo_25k_s2.log` (25k complete)

### Commands to Continue PPO to 50k

```bash
cd ~/fbsource/fbcode

# PPO seed 0 - continue from 25k to 50k (use GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only -- \
  --config /home/buiksat/trm_bellman/configs/sudoku9x9/ppo_9x9.yaml \
  --dataset-paths /home/buiksat/trm_bellman/data/sudoku-9x9 \
  --seed 0 --no-wandb \
  --resume-checkpoint /home/buiksat/trm_bellman/checkpoints/rl_sudoku-9x9_seed0_ppo25k/rl_checkpoint_step_25000.pt \
  > /home/buiksat/trm_bellman/results/9x9_experiments_seed0/ppo_50k_s0_continued.log 2>&1 &

# PPO seed 2 - continue from 25k to 50k (use GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only -- \
  --config /home/buiksat/trm_bellman/configs/sudoku9x9/ppo_9x9.yaml \
  --dataset-paths /home/buiksat/trm_bellman/data/sudoku-9x9 \
  --seed 2 --no-wandb \
  --resume-checkpoint /home/buiksat/trm_bellman/checkpoints/rl_sudoku-9x9_seed2_ppo25k/rl_checkpoint_step_25000.pt \
  > /home/buiksat/trm_bellman/results/9x9_experiments_seed0/ppo_50k_s2_continued.log 2>&1 &
```

**Note:** Check if checkpoint paths exist. If not, you may need to re-run from scratch with the full 50k config:

```bash
# Alternative: Run full 50k from scratch
CUDA_VISIBLE_DEVICES=0 nohup buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only -- \
  --config /home/buiksat/trm_bellman/configs/sudoku9x9/ppo_9x9.yaml \
  --dataset-paths /home/buiksat/trm_bellman/data/sudoku-9x9 \
  --seed 0 --no-wandb \
  > /home/buiksat/trm_bellman/results/9x9_experiments_seed0/ppo_50k_s0_full.log 2>&1 &
```

### After PPO Completes

1. **Regenerate plots:**
```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:plot_9x9_success_vs_steps
buck2 run //buiksat_trm:plot_9x9_mean_score_vs_steps
```

2. **Update plotting scripts** to include the new 50k logs (modify glob patterns in `scripts/plot_9x9_*.py`)

3. **Update EXPERIMENT_REPORT.md** with final results

---

## Current Plotting Behavior

The plotting scripts now handle mixed-length data:
- **Solid line:** Mean across all seeds (common steps only, 0-25k for PPO)
- **Dashed line:** Individual seed lines for extended range (25k-50k for PPO s1)

This will automatically show full 50k data once all PPO seeds complete.

---

## Key Files

| Description | Path |
|-------------|------|
| Experiment logs | `results/9x9_experiments_seed0/*.log` |
| Detailed report | `results/9x9_experiments_seed0/EXPERIMENT_REPORT.md` |
| Plotting scripts | `scripts/plot_9x9_*.py` |
| Configs | `configs/sudoku9x9/` |
| Dataset | `data/sudoku-9x9/` |
| Project guidelines | `CLAUDE.md` |

---

## Figures Location

Generated figures are saved to:
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures/fig_9x9_success_vs_steps.pdf`
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures/fig_9x9_mean_score_vs_steps.pdf`

---

## Critical Notes

1. **Always use absolute paths** for configs and data (buck2 sandbox issues)
2. **Use CUDA_VISIBLE_DEVICES** to assign GPUs (0-3 available)
3. **Config guidelines from CLAUDE.md:**
   - `use_feasibility_checker: true` (required)
   - `disable_value_head_norm: true` (for stability)

---

## TODO

- [x] Run PPO s0 to 50k steps
- [x] Run PPO s2 to 50k steps
- [x] Update plotting scripts to include new 50k logs
- [x] Regenerate figures with all 50k data
- [x] Update EXPERIMENT_REPORT.md with final results
- [ ] Commit and push changes

---

*Last updated: 2026-02-08*
