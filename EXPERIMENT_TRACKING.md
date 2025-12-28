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
| EXP-02 | UPI-TRM (theory-exact, episodic z) | rl_sudoku_shaped_theory_exact.yaml | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-03 | UPI-TRM (baseline) | rl_sudoku_4x4_ultra_easy.yaml | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-04 | PPO-TRM | baseline ppo + trm | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-05 | PPO-MLP | baseline ppo + mlp | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-06 | A2C-MLP | baseline a2c + mlp | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-07 | UPI-TRM (constraint-checker) | rl_sudoku_4x4_constraint_checker.yaml | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-08 | UPI-TRM (persistent z + constraint) | rl_sudoku_4x4_constraint_persistent_z.yaml | 42,123,456 | 5000 | ⏳ RUNNING | - | - |
| EXP-09 | UPI-TRM (persistent z, batch-centered) | rl_sudoku_shaped_theory_exact_persistent_z.yaml | 42,123,456 | 5000 | ⏳ RUNNING | - | - |

---

## Important Theory Notes

### Episodic vs Persistent Latent z

The paper defines two latent modes (Section 5.4, Lemma 4.4):

1. **Episodic z** (`episodic_latent=true`): z is reinitialized from (x, y) at every step
   - Enables `exact_baseline_summation=true` for Theorem 5.9's O(α·ε_A) bound
   - This is the **theory-exact** mode

2. **Persistent z** (`episodic_latent=false`): z is initialized once per episode and carried across steps (RNN-like)
   - The exact baseline summation becomes the "memoryless approximation" from Section 5.4
   - Theorem 5.9 does NOT strictly apply
   - Use `batch_centered_advantage=true` as a heuristic instead
   - Analyzed via Lemma 4.4 (two-timescale bound)

**Key Insight**: EXP-02 (episodic z) tests the main theorem; EXP-09 (persistent z) tests the practical approximation.

### CPI Mixture Policy Modes (Updated Dec 27, 2024)

The implementation supports three Conservative Policy Improvement (CPI) modes. **For strict theoretical correctness, use `theory_exact_mixture=True`**.

**Why Importance Sampling Weights Are NOT Required** (in `theory_exact_mixture=True` mode):

A concern was raised that data collected from the mixture policy `π_mix = (1-α)π_old + α·π_cand` should require importance sampling (IS) weights `ρ = π_cand(a|s) / π_mix(a|s)` when training `π_cand`. This is **not** the case because:

1. **The deployed policy IS the mixture**: We don't train `π_cand` to behave like `π_mix`. We train `π_cand` with standard policy gradient, keep `π_old` fixed, and deploy the explicit mixture.

2. **CPI's guarantee is about the mixture**: The improvement bound `V^{π_new} ≥ V^{π_old} - O(α·ε_A)` applies to the **deployed mixture policy**, not to `π_cand` in isolation.

**Three CPI Modes** (in `rl/upi_trm_trainer.py:policy_update()`):

| Mode | Config | Theory Status | Description |
|------|--------|---------------|-------------|
| **Theory-Exact** | `theory_exact_mixture=True` | ✅ CPI bound applies | `π_old` is **not updated**; behavior policy is explicit mixture |
| **Distillation** | `distill_mixture_policy=True` | ⚠️ Heuristic | Mixture distilled into `π_old` via KL; introduces projection error |
| **Default** | Both `False` | ⚠️ Heuristic | Parameter-space `lerp`; NOT equivalent to probability mixing |

**Recommendation for ICML 2026**: Always use `theory_exact_mixture=True` in theory-aligned configs (EXP-02, EXP-07, etc.).

---

## Critical Finding: Reward Signal Issue

**Root Cause of 0% RL Success Rate Identified (Dec 27, 2024)**

The current `sudoku_checker()` only compares the plan to the **known solution** (cell matching):
- Score = (matching_cells / total_cells) × 10.0
- This does NOT provide intermediate signals for constraint satisfaction
- The shaped reward `r = γ·Φ(s') - Φ(s)` only rewards matching the exact solution

**Solution: Constraint-Based Checker (EXP-07)**

Implemented `sudoku_constraint_checker()` that scores based on **Sudoku constraint violations**:
- Counts row/column/box duplicates (violations)
- Score = 10.0 × (1 - violations / max_violations)
- Provides positive reward for moves that **reduce** violations
- Provides negative reward for moves that **increase** violations
- This is the correct intermediate signal for RL learning!

**Key Implementation Files:**
- `upi_trm_train.py`: Added `count_sudoku_violations_4x4()` and `sudoku_constraint_checker()`
- `rl/config.py`: Added `use_constraint_checker: bool` field
- `configs/rl_sudoku_4x4_constraint_checker.yaml`: New config with constraint checker enabled

---

## Results Summary

### Completed Experiments

#### EXP-01: Imitation Learning (Upper Bound)
- **Method**: Supervised learning with oracle labels
- **Training**: 100 epochs on 450 puzzles
- **Final Accuracy**: 100.00%
- **Evaluation**: 50/50 puzzles solved
- **Conclusion**: Task is solvable; this is the upper bound

### Running Experiments

All experiments are currently running with 5000 training steps.
Results will be populated as experiments complete...

---

## Commands Used

### EXP-01: Imitation Learning
```bash
buck2 run //buiksat_trm:imitation_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --num-epochs 100 --seed 42
```

### EXP-02: UPI-TRM Theory-Exact (MAIN CONTRIBUTION)
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/rl_sudoku_shaped_theory_exact.yaml \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-03: UPI-TRM Baseline (Non-Theory-Exact)
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/rl_sudoku_4x4_ultra_easy.yaml \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-04: PPO-TRM
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline ppo --backbone trm \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-05: PPO-MLP
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline ppo --backbone norec-mlp \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-06: A2C-MLP
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline a2c --backbone norec-mlp \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --train-steps 5000 --seed 42 --no-wandb
```

### EXP-07: UPI-TRM (Constraint-Based Checker) - NEW
```bash
buck2 run //buiksat_trm:upi_trm_train -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/rl_sudoku_4x4_constraint_checker.yaml \
    --train-steps 5000 --seed 42 --no-wandb
```

---

## Theory-Exact Config Key Features

The `rl_sudoku_shaped_theory_exact.yaml` config enables all paper features:

| Feature | Config Key | Value | Paper Reference |
|---------|------------|-------|-----------------|
| Exact baseline summation | `exact_baseline_summation` | `true` | Theorem 5.9 (KEY) |
| Contraction enforcement | `enable_contraction` | `true` | Assumption 4.2 |
| Theory-exact CPI mixture | `theory_exact_mixture` | `true` | Section 6.5 |
| Forward-invariant projection | `latent_ball_radius` | `10.0` | Assumption 4.1 |
| Rush-to-fail mitigation | `fail_terminal_reward` | `-10.0` | Remark 2.6 |
| Inner unroll depth | `inner_unroll_n` | `4` | Dial 'n' |
| Mixture alpha | `mixture_alpha` | `0.05` | Dial 'α' |

---

## Paper Table Draft

**Table 1: 4×4 Sudoku Results (5000 training steps)**

| Method | Success Rate | Mean Score | Notes |
|--------|--------------|------------|-------|
| Imitation Learning (Upper Bound) | 100% | 10.0 | Supervised with oracle |
| UPI-TRM (Theory-Exact) | TBD | TBD | All theory features |
| UPI-TRM (Baseline) | TBD | TBD | No theory features |
| PPO-TRM | TBD | TBD | PPO with TRM backbone |
| PPO-MLP | TBD | TBD | Standard baseline |
| A2C-MLP | TBD | TBD | Simplest baseline |

---

## Notes

- 4×4 Sudoku is a simple task (450 puzzles, 1-7 empty cells)
- Imitation learning achieves 100% immediately
- RL methods are expected to converge with sufficient training
- Key question: How do theory-exact features improve over baselines?
- Theory-exact config should show "Is theory-exact: True" at startup
