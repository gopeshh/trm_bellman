# UPI-TRM Experiment Plan (ICML 2026)

## Current Status (2026-01-07 12:00 PST)

### Major Change: All Configs Now Use Progress Checker

All configs have been updated to use `use_progress_checker: true`:
- Score = filled_cells (range 0-16 for 4×4 Sudoku)
- Initial score = number of clue cells (e.g., 12)
- Solved score = 16 (all cells filled correctly)
- `solved_threshold: 16.0`
- `fail_terminal_reward: -16.0`

This provides more informative scoring than the constraint checker (which couldn't distinguish partial from solved).

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

| Experiment | Config | GPU | Seeds | Status |
|------------|--------|-----|-------|--------|
| PPO-TRM | `baselines/ppo_trm_sudoku.yaml` | 0 | 42 | 🔄 Running |
| A2C-TRM | `baselines/a2c_trm_sudoku.yaml` | 0 | 42 | 🔄 Running |
| DQN-TRM | `baselines/dqn_trm_sudoku.yaml` | 1 | 42 | 🔄 Running |
| UPI-TRM (persistent z) | `paper_persistent_z_constraint.yaml` | 1 | 42 | ⏳ Queued |

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
