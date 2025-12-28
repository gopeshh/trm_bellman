# UPI-TRM 4×4 Sudoku Experiment Tracking

**Date**: December 27, 2024
**Dataset**: sudoku-4x4-ultra-easy (450 puzzles)
**Training Steps**: 5000 (quick experiments for paper validation)
**Seeds**: 42, 123, 456

---

## Experiment Status

| Experiment ID | Method | Config | Seeds | Steps | Status | Success Rate | Mean Score |
|--------------|--------|--------|-------|-------|--------|--------------|------------|
| EXP-01 | Imitation Learning | supervised | 42 | 100 epochs | ✅ COMPLETE | **100%** | 10.0/10.0 |
| EXP-02 | UPI-TRM (baseline) | rl_sudoku_4x4_ultra_easy.yaml | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-03 | UPI-TRM (theory-exact) | rl_sudoku_shaped_theory_exact.yaml | 42,123,456 | 5000 | ⏳ PENDING | - | - |
| EXP-04 | PPO-MLP | baseline ppo | 42,123,456 | 5000 | ⏳ PENDING | - | - |
| EXP-05 | A2C-MLP | baseline a2c | 42,123,456 | 5000 | ⏳ PENDING | - | - |
| EXP-06 | Ablation: No Exact Baseline | ablation_no_exact_baseline.yaml | 42 | 5000 | ⏳ PENDING | - | - |
| EXP-07 | Ablation: No Contraction | ablation_no_contraction.yaml | 42 | 5000 | ⏳ PENDING | - | - |
| EXP-08 | Ablation: No CPI (α=1.0) | ablation_no_conservative_mixture.yaml | 42 | 5000 | ⏳ PENDING | - | - |

---

## Results Summary

### Completed Experiments

#### EXP-01: Imitation Learning (Upper Bound)
- **Method**: Supervised learning with oracle labels
- **Training**: 100 epochs on 450 puzzles
- **Final Accuracy**: 100.00%
- **Evaluation**: 50/50 puzzles solved
- **Conclusion**: Task is solvable; this is the upper bound

### Running/Pending Experiments

Results will be populated as experiments complete...

---

## Commands Used

### EXP-01: Imitation Learning
```bash
buck2 run //buiksat_trm:imitation_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --num-epochs 100 --seed 42
```

### EXP-02: UPI-TRM Baseline
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/rl_sudoku_4x4_ultra_easy.yaml \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-03: UPI-TRM Theory-Exact
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/rl_sudoku_shaped_theory_exact.yaml \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-04: PPO-MLP
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline ppo --backbone norec-mlp \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-05: A2C-MLP
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline a2c --backbone norec-mlp \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --train-steps 5000 --seed 42 --no-wandb
```

---

## Paper Table Draft

**Table 1: 4×4 Sudoku Results (5000 training steps)**

| Method | Success Rate | Mean Score | Notes |
|--------|--------------|------------|-------|
| Imitation Learning (Upper Bound) | 100% | 10.0 | Supervised with oracle |
| UPI-TRM (Theory-Exact) | TBD | TBD | All theory features |
| UPI-TRM (Baseline) | TBD | TBD | Default config |
| PPO-MLP | TBD | TBD | Standard baseline |
| A2C-MLP | TBD | TBD | Simplest baseline |

**Table 2: Ablation Study**

| Ablation | Success Rate | Δ vs Full | Tests |
|----------|--------------|-----------|-------|
| Full UPI-TRM | TBD | - | All features |
| No Exact Baseline | TBD | TBD | Theorem 5.9 |
| No Contraction | TBD | TBD | Assumption 4.2 |
| No CPI (α=1.0) | TBD | TBD | Conservative update |

---

## Notes

- 4×4 Sudoku is a simple task (450 puzzles, 1-7 empty cells)
- Imitation learning achieves 100% immediately
- RL methods are expected to converge with sufficient training
- Key question: How do theory-exact features improve over baselines?
