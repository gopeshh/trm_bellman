# UPI-TRM Experiment Plan (ICML 2026)

## Current Status (2026-01-07 14:00 PST)

### Major Change: All Configs Now Use Progress Checker

All configs have been updated to use `use_progress_checker: true`:
- Score = filled_cells (range 0-16 for 4×4 Sudoku)
- Initial score = number of clue cells (e.g., ~8.84)
- Solved score = 16 (all cells filled correctly)
- `solved_threshold: 16.0`
- `fail_terminal_reward: -16.0`
- `value_target_clip: 20.0` (CRITICAL FIX - was 10.0)

This provides more informative scoring than the constraint checker (which couldn't distinguish partial from solved).

### Critical Fix: value_target_clip

**Bug**: Default `value_target_clip: 10.0` was too low for progress checker (score range 0-16).
- This caused value targets to be clipped, preventing proper learning.

**Fix Applied**:
1. Changed default in `rl/config.py` from 10.0 to 20.0
2. Added `value_target_clip: 20.0` to all 12 configs:
   - 2 paper configs
   - 3 baseline configs
   - 7 ablation configs

### Configs Updated

**Ablation Configs (7 files)**:
- `ablation_no_exact_baseline.yaml`
- `ablation_no_contraction.yaml`
- `ablation_no_conservative_mixture.yaml`
- `ablation_no_projection.yaml`
- `ablation_no_theory_exact_mixture.yaml`
- `ablation_no_theory_features.yaml`
- `ablation_sparse_no_theory.yaml`

**Baseline Configs (3 files)**:
- `baselines/ppo_trm_sudoku.yaml`
- `baselines/a2c_trm_sudoku.yaml`
- `baselines/dqn_trm_sudoku.yaml`

**Paper Configs (2 files)**:
- `paper_persistent_z_constraint.yaml`
- `paper_episodic_z_constraint.yaml`

### Experiment Plan: Baseline Comparison with Progress Checker

**Goal**: Compare UPI-TRM vs standard RL baselines (PPO, A2C, DQN) on 4×4 Sudoku with the progress checker.

| Experiment | Config | GPU | Seeds | Status | Result |
|------------|--------|-----|-------|--------|--------|
| DQN-TRM | `baselines/dqn_trm_sudoku.yaml` | 1 | 42 | ✅ Done | **50% peak** |
| A2C-TRM | `baselines/a2c_trm_sudoku.yaml` | 0 | 42 | ✅ Done | 0% |
| PPO-TRM | `baselines/ppo_trm_sudoku.yaml` | 0 | 42 | ✅ Done | 0% |
| UPI-TRM (persistent z) | `paper_persistent_z_constraint.yaml` | 0 | 42 | ✅ Done | 0% |
| UPI-TRM (episodic z) | `paper_episodic_z_constraint.yaml` | 1 | 42 | ⏸️ Too slow | - |

### Key Finding: Progress Checker Results

**DQN is the only algorithm that solves puzzles with the progress checker.**

| Algorithm | Peak Success | Mean Score | Notes |
|-----------|-------------|------------|-------|
| **DQN-TRM** | **50%** | 8.78 | Best performer! |
| A2C-TRM | 0% | 9.0 | Never solved |
| PPO-TRM | 0% | 7.08 | Never solved |
| UPI-TRM Persistent z | 0% | 6.98 | Never solved |

### Why DQN Wins with Progress Checker

1. **Off-policy learning**: Reuses experience efficiently
2. **Epsilon-greedy exploration**: More random initially, explores better
3. **Direct Q-value learning**: Simpler objective than policy gradient
4. **Experience replay**: Sample efficiency advantage

### Comparison: Progress Checker vs Constraint Checker

| Checker | UPI-TRM Success | DQN Success | Task Type |
|---------|----------------|-------------|-----------|
| Constraint | **44%** | 0% | Avoid violations (start at max) |
| Progress | 0% | **50%** | Fill correctly (must improve) |

**Insight**: The two checkers favor different algorithms:
- Constraint checker: Rewards maintaining score → UPI-TRM's conservative updates help
- Progress checker: Rewards improvement → DQN's exploration helps

### Commands

```bash
# PPO baseline
cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES=0 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --baseline ppo --backbone trm \
    --dataset-paths /home/buiksat/trm_bellman/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/baselines/ppo_trm_sudoku.yaml \
    --train-steps 5000 --seed 42

# A2C baseline
cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES=0 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --baseline a2c --backbone trm \
    --dataset-paths /home/buiksat/trm_bellman/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/baselines/a2c_trm_sudoku.yaml \
    --train-steps 5000 --seed 42

# DQN baseline
cd ~/fbsource/fbcode && CUDA_VISIBLE_DEVICES=1 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --baseline dqn --backbone trm \
    --dataset-paths /home/buiksat/trm_bellman/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/baselines/dqn_trm_sudoku.yaml \
    --train-steps 5000 --seed 42
```

### Progress Checker Details

The new progress checker (`sudoku_progress_checker`) in `upi_trm_train.py`:

```python
def sudoku_progress_checker(x, y, violation_penalty: float = 2.0) -> float:
    filled_cells = (plan != 1).sum().item()  # Count non-empty cells
    violations = count_sudoku_violations_4x4(plan)

    if violations == 0:
        score = float(filled_cells)
    else:
        score = float(filled_cells - violations * violation_penalty)

    return score
```

**Advantages**:
1. Initial score = clue count (e.g., 12) - NOT 10.0
2. Solved score = 16 (all filled) - distinguishes from partial
3. More informative learning signal

### Previous Results (with Constraint Checker)

| Method | Success Rate | Notes |
|--------|--------------|-------|
| UPI-TRM Persistent z | 44% ± 2% | Peak across 3 seeds |
| PPO-TRM | 0% | All 3 seeds |
| DQN-TRM | 0% | All 3 seeds |
| Ablation: No Theory Features | 100% | After bug fix |
| Ablation: Sparse No Theory | 100% | After bug fix |

**Note**: With the progress checker, results may differ due to more informative scoring.

### Related Files

- `EXPERIMENT_RESULTS.md` - Detailed experiment results
- `CLAUDE.md` - Project instructions including checker documentation
- `upi_trm_train.py` - Contains `sudoku_progress_checker()` function
- `rl/config.py` - Contains `use_progress_checker` flag

### Test Suite Status (2026-01-07)

**All 117 tests pass across 22 test targets.**

Bug fixes applied:
1. **rl/value_targets.py**: Fixed C_max terminal bootstrap - now correctly uses `-C_max` for terminal states per paper Eq. 12
2. **tests/test_theory_metrics.py**: Fixed shape mismatch - changed from `.mean()` pooling to `.view()` flattening to match model's `used_value()` implementation

New test targets added:
- `test_baselines` - Baseline algorithms (PPO, A2C, NoRecursionEncoder)
- `test_undo_and_sequences` - UNDO action and sequence sampling
- `test_sudoku_checkers` - Sudoku checker functions (constraint/progress)
- `test_config_integrity` - Config file validation
- `test_convergence_smoke` - Training convergence smoke test
