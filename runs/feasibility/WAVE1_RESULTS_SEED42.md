# Wave 1 Results: Feasibility Checker Experiments

**Date:** 2026-01-08
**Seed:** 42
**Dataset:** sudoku-4x4-trivial (1-4 empties, mean 2.46)
**Steps:** 5000

## Summary

| Algorithm | Success Rate | Solved/50 | Mean Score | Peak Success |
|-----------|--------------|-----------|------------|--------------|
| **UPI-TRM** | **42%** | **21/50** | **14.80** | 44% (step 4700) |
| A2C | 26% | 13/50 | 13.44 | 26% |
| DQN | 12% | 6/50 | 10.68 | 20% (step 500) |
| PPO | 16%* | 8/50* | 11.28* | *still running |

*PPO was at step 1200/5000 when summary generated (very slow at ~3.5s/step)

## Key Findings

1. **UPI-TRM outperforms all baselines by a large margin** (42% vs next best 26%)
2. The trivial dataset (1-4 empties) works correctly - all algorithms show non-zero success
3. UPI-TRM shows consistent learning progress: 18% → 32% → 38% → 44% → 42%
4. DQN peaked early (20% at step 500) then regressed - may need hyperparameter tuning
5. A2C showed stable learning with consistent 24-26% success rate

## Training Details

### UPI-TRM (GPU 0)
- Config: `configs/rl_sudoku_4x4_feasibility.yaml`
- Final success rate: 42% (21/50 solved)
- Training time: ~25 minutes
- Score range: 13.00-16.00 (all successful solutions near-optimal)

### A2C (GPU 2)
- Config: `configs/baselines/a2c_trm_feasibility.yaml`
- Final success rate: 26% (13/50 solved)
- Training time: ~20 minutes
- Score range: 6.00-16.00

### DQN (GPU 3)
- Config: `configs/baselines/dqn_trm_feasibility.yaml`
- Final success rate: 12% (6/50 solved)
- Training time: ~14 minutes
- Score range: -3.00-16.00 (some failures with violations)

### PPO (GPU 1) - Incomplete
- Config: `configs/baselines/ppo_trm_feasibility.yaml`
- Current success rate at step 1200: 16% (8/50 solved)
- Estimated total time: ~4.8 hours (3.5s/step)
