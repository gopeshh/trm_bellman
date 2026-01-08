# UPI-TRM 4x4 Sudoku Experiment Results (Feasibility Checker)

**Date**: January 8, 2026
**Project**: UPI-TRM (Unrolled Policy Iteration for Tiny Recursive Models)
**Target**: ICML 2026 Submission (Deadline: January 18)

---

## ✅ Pre-Experiment Sanity Check (PASSED)

Before running experiments, the feasibility checker implementation was verified:

| Check | Status | Details |
|-------|--------|---------|
| Config correctness | ✅ PASS | All 6 configs have correct settings |
| Termination safety | ✅ SAFE | `solved_threshold=null` prevents premature termination |
| Evaluation metric | ✅ CORRECT | Uses `sudoku_is_solved()` (solution-independent) |
| Score formula | ✅ VERIFIED | `filled - 2.0*violations - 5.0*zeroCand` |

**Key invariants verified:**
- Empty grid: `is_solved=False`, score=0
- Partial valid grid: `is_solved=False`
- Solved grid: `is_solved=True`, score=16
- Dead-end state: `zeroCand > 0`, negative score penalty

**Sanity check scripts:** `scripts/sanity_check_standalone.py`, `scripts/sanity_check_feasibility.py`

---

## Experimental Setup

### Task: 4×4 Sudoku with Feasibility-Aware Checker

| Parameter | Value |
|-----------|-------|
| **Checker Type** | Feasibility-aware (`use_feasibility_checker: true`) |
| **Score Formula** | `filled - w_v × violations - w_z × zeroCand` |
| **Weights** | w_v=2.0 (violations), w_z=5.0 (zero-candidates) |
| **Score Range** | Unbounded (typically -20 to 16 for 4×4) |
| **Success Criterion** | `filled == 16 AND violations == 0` (solution-independent) |

### Why Feasibility Checker?

Previous checkers had limitations:
- **Constraint checker**: Cannot distinguish empty vs solved grids (both score 10.0)
- **Progress checker**: Doesn't penalize dead-end states (impossible to complete)

The feasibility checker combines:
1. **Dense progress signal**: Rewards filling cells correctly
2. **Dead-end detection**: Heavily penalizes zero-candidate cells
3. **Solution-independent evaluation**: Success doesn't require ground truth

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `data/sudoku-4x4-ultra-easy` (1-4 empty cells) |
| Action Space | 97 discrete (16 positions × 6 values + STOP) |
| Episode Length | max 16 edits |
| Training Steps | 5000 |
| Seeds | 42, 123, 456 |
| Evaluation | Every 100 steps, 50 episodes |

### Shared Hyperparameters

| Parameter | Value |
|-----------|-------|
| `gamma` | 0.99 |
| `max_edits` | 16 |
| `fail_terminal_reward` | -16.0 |
| `solved_threshold` | null (uses `sudoku_is_solved()`) |
| `value_target_clip` | 50.0 |

---

## Feasibility Checker Implementation

### Score Formula

```
score = filled - w_v × violations - w_z × zeroCand
```

Where:
- `filled`: Number of non-empty cells (0 to 16)
- `violations`: Count of constraint violations (duplicates in row/col/box)
- `zeroCand`: Count of empty cells with 0 legal candidates (dead-end cells)
- `w_v = 2.0`: Violation penalty weight
- `w_z = 5.0`: Zero-candidate penalty weight (strong to discourage dead-ends)

### Score Examples (4×4 Sudoku)

| State | Filled | Violations | ZeroCand | Score |
|-------|--------|------------|----------|-------|
| Empty grid | 0 | 0 | 0 | 0 |
| Partial valid (8 cells) | 8 | 0 | 0 | 8 |
| Partial with 2 violations | 8 | 2 | 0 | 4 |
| Dead-end state | 8 | 2 | 1 | -1 |
| Solved | 16 | 0 | 0 | 16 |

### Success Definition (Solution-Independent)

An episode is **solved** if and only if:
```
filled == N AND violations == 0
```

Where N is the total cells (16 for 4×4, 81 for 9×9).

**This is solution-independent** - we don't compare against any ground truth solution.

---

## Experiment Configurations

### Main Experiments

| Config | Algorithm | Description |
|--------|-----------|-------------|
| `rl_sudoku_4x4_feasibility.yaml` | UPI-TRM | Main config with all theory features |
| `baselines/ppo_trm_feasibility.yaml` | PPO-TRM | PPO baseline |
| `baselines/a2c_trm_feasibility.yaml` | A2C-TRM | A2C baseline |
| `baselines/dqn_trm_feasibility.yaml` | DQN-TRM | DQN baseline |

### Ablation Studies

| Config | Ablation | Key Change |
|--------|----------|------------|
| `ablations/upi_trm_feasibility_no_conservative.yaml` | No Conservative Mixture | `mixture_alpha=1.0` |
| `ablations/upi_trm_feasibility_no_contraction.yaml` | No Contraction | `enable_contraction=false` |

---

## Results

### Summary Table

| Algorithm | Seed | Step | Success Rate | Mean Filled | Mean Violations | Mean ZeroCand |
|-----------|------|------|--------------|-------------|-----------------|---------------|
| *Pending experiments* | | | | | | |

### Learning Curves

*To be populated after experiments complete.*

---

## Running Experiments

### Quick Start

```bash
# Run all experiments with the experiment script
./scripts/run_feasibility_experiments.sh --seeds "42 123 456" --steps 5000
```

### Manual Commands

```bash
# UPI-TRM main
python upi_trm_train.py \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/rl_sudoku_4x4_feasibility.yaml \
    --train-steps 5000 --seed 42

# PPO baseline
python upi_trm_train.py \
    --baseline ppo \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/baselines/ppo_trm_feasibility.yaml \
    --train-steps 5000 --seed 42

# DQN baseline
python upi_trm_train.py \
    --baseline dqn \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/baselines/dqn_trm_feasibility.yaml \
    --train-steps 5000 --seed 42

# A2C baseline
python upi_trm_train.py \
    --baseline a2c \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/baselines/a2c_trm_feasibility.yaml \
    --train-steps 5000 --seed 42

# Ablation: No conservative mixture
python upi_trm_train.py \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/ablations/upi_trm_feasibility_no_conservative.yaml \
    --train-steps 5000 --seed 42

# Ablation: No contraction
python upi_trm_train.py \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/ablations/upi_trm_feasibility_no_contraction.yaml \
    --train-steps 5000 --seed 42
```

---

## Implementation Files

| File | Description |
|------|-------------|
| `rl/sudoku_utils.py` | Shared utilities: violations, filled cells, zero-candidate detection, `sudoku_is_solved()` |
| `upi_trm_train.py:sudoku_feasibility_checker()` | The feasibility checker function |
| `rl/config.py:use_feasibility_checker` | Config flag to enable feasibility checker |
| `rl/evaluator.py` | Updated to use `sudoku_is_solved()` for success |
| `rl/algos/dqn.py` | Updated to use `sudoku_is_solved()` for success |

---

## Expected Improvements Over Previous Checkers

The feasibility checker should provide:
1. **Better credit assignment**: Dead-end penalties help the agent learn to avoid violations early
2. **Clearer progress signal**: Score increases monotonically for valid placements
3. **Solution-independent evaluation**: Success doesn't depend on having ground truth
4. **Distinguishes partial vs solved**: Unlike constraint checker which scores both as 10.0

---

## Analysis (To Be Completed)

### Questions to Answer

1. Does UPI-TRM outperform baselines with the feasibility checker?
2. Do the ablations validate the importance of theory features?
3. Is the feasibility checker better than constraint/progress checkers?

### Metrics to Track

- **Success Rate**: % of puzzles solved (filled=16, violations=0)
- **Mean Filled**: Average cells filled at end of episode
- **Mean Violations**: Average violations at end of episode
- **Mean ZeroCand**: Average zero-candidate cells (dead-ends)
- **Score**: The feasibility score (filled - 2*violations - 5*zeroCand)

---

*Report created: January 8, 2026 - Ready for feasibility checker experiments*
