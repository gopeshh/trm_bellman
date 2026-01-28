# Constraint-Aware Action Masking Implementation Report

**Date:** January 28, 2026
**Author:** Claude Code
**Status:** Experiments launched

---

## Overview

This report documents the implementation of constraint-aware action masking for Sudoku puzzles in the UPI-TRM framework. The improvement significantly reduces the effective action space by masking actions that would violate Sudoku constraints.

---

## Problem Statement

### Previous Limitation
The original action masking only prevented:
1. Overwriting given cells (clues)
2. Placing PAD (0) or empty (1) tokens

This left the agent free to make obvious Sudoku violations:
- Placing duplicate digits in the same row
- Placing duplicate digits in the same column
- Placing duplicate digits in the same 3×3 box

### Impact
With 729 possible actions (81 positions × 9 digits), the agent wasted exploration on invalid moves. This slowed learning significantly.

---

## Implementation Details

### 1. Enhanced Action Masking (`rl/task_config.py`)

**File:** `rl/task_config.py`
**Method:** `SudokuTaskConfig.compute_action_mask()`

**Changes:**
- Added `current_state` parameter to check constraints against the current board (not just initial puzzle)
- Implemented row constraint checking
- Implemented column constraint checking
- Implemented box constraint checking (2×2 for 4×4, 3×3 for 9×9)

**Algorithm:**
```python
def compute_action_mask(self, inputs, vocab_size, stop_action_id, current_state=None):
    # 1. Mask given cells (unchanged)
    # 2. Mask PAD/empty tokens (unchanged)

    # 3. NEW: Constraint-aware masking
    for each position:
        row, col, box = get_position_info(pos)
        used_digits = set()

        # Check row for existing digits
        for c in range(grid_size):
            if state[row, c] > 1:
                used_digits.add(state[row, c])

        # Check column for existing digits
        for r in range(grid_size):
            if state[r, col] > 1:
                used_digits.add(state[r, col])

        # Check box for existing digits
        for each cell in box:
            if cell_value > 1:
                used_digits.add(cell_value)

        # Mask actions that place used digits
        for digit in used_digits:
            mask[pos * vocab_size + digit] = False
```

### 2. Dynamic Mask Updates (`rl/envs/plan_edit_env.py`)

**File:** `rl/envs/plan_edit_env.py`
**Method:** `_compute_action_mask()` and `step()`

**Changes:**
1. Updated `_compute_action_mask()` to pass current state `self.y` to the task config
2. Modified `step()` to recompute mask after every action (not just when UNDO is enabled)

**Before:**
```python
# Only recomputed when UNDO enabled
if self._enable_undo and not done:
    self._compute_action_mask()
```

**After:**
```python
# Always recompute for constraint-aware masking
if not done:
    self._compute_action_mask()
```

### 3. Configuration Updates

**Files:**
- `configs/sudoku9x9/upi_trm_9x9.yaml`
- `configs/sudoku9x9/ppo_9x9.yaml`
- `configs/sudoku9x9/dqn_9x9.yaml`

**Change:** Increased training steps from 25,000 to 50,000
```yaml
num_train_steps: 50000  # Previously 25000
```

---

## Test Suite

**File:** `tests/test_constraint_aware_masking.py`

**Tests implemented:**

| Test | Description |
|------|-------------|
| `test_given_cells_masked_4x4` | Verify clue cells are protected |
| `test_pad_and_empty_tokens_masked` | Verify PAD/empty tokens are blocked |
| `test_row_constraint_masking` | Verify same digit blocked in row |
| `test_column_constraint_masking` | Verify same digit blocked in column |
| `test_box_constraint_masking_4x4` | Verify same digit blocked in 2×2 box |
| `test_constraint_masking_9x9` | Verify constraints work on 9×9 grid |
| `test_current_state_updates_mask` | Verify mask updates with current state |
| `test_stop_action_always_valid` | Verify STOP action remains valid |
| `test_no_nan_in_mask` | Verify no NaN values in mask |
| `test_mask_is_deterministic` | Verify consistent mask computation |
| `test_valid_actions_exist_for_empty_cells` | Verify unconstrained cells have actions |

---

## Expected Impact

### Action Space Reduction

**Before (no constraint masking):**
- 729 possible actions per step (81 positions × 9 digits)
- Most actions invalid but agent must learn this

**After (constraint-aware masking):**
- Typically 50-150 valid actions per step
- ~80% reduction in exploration space
- Agent can only select constraint-valid moves

### Learning Efficiency

| Aspect | Expected Improvement |
|--------|---------------------|
| Exploration efficiency | 4-5× faster (fewer invalid actions) |
| Credit assignment | Clearer (all actions are valid) |
| Sample efficiency | 2-3× better (no wasted samples on violations) |
| Final performance | Higher success rate |

---

## Experiment Setup

### Launched Experiments

| Algorithm | GPU | Seed | Log File |
|-----------|-----|------|----------|
| UPI-TRM | 0 | 0 | `results/sudoku9x9_constraint_aware/upi_trm_seed0.log` |
| PPO | 1 | 0 | `results/sudoku9x9_constraint_aware/ppo_seed0.log` |
| DQN | 2 | 0 | `results/sudoku9x9_constraint_aware/dqn_seed0.log` |

### Configuration Summary

**UPI-TRM:**
```yaml
algorithm: "upi_trm"
num_train_steps: 50000
batch_size: 64
inner_unroll_n: 2
episodic_latent: false        # Persistent z
enable_contraction: false
disable_value_head_norm: true
latent_ball_radius: 10.0
use_feasibility_checker: true
```

**PPO:**
```yaml
algorithm: "ppo"
num_train_steps: 50000
batch_size: 256
ppo_clip_eps: 0.2
ppo_epochs: 4
use_gae: true
gae_lambda: 0.95
```

**DQN:**
```yaml
algorithm: "dqn"
num_train_steps: 50000
batch_size: 256
dqn_double_dqn: true
dqn_buffer_size: 50000
```

---

## Comparison with Previous Experiments

### Previous Results (Without Constraint Masking)

| Algorithm | Mean Score | Initial | Change | Success Rate |
|-----------|------------|---------|--------|--------------|
| DQN | 25.57 | 26.84 | -1.27 | 0% |
| PPO | 26.02 | 26.84 | -0.82 | 0% |
| UPI-TRM | 27.64 | 26.64 | +1.00 | 0% |

### Expected Results (With Constraint Masking)

| Algorithm | Expected Change |
|-----------|-----------------|
| DQN | Positive learning expected (was negative) |
| PPO | Improved learning expected |
| UPI-TRM | Further improvement, possible >0% success |

---

## Files Modified

| File | Changes |
|------|---------|
| `rl/task_config.py` | Added constraint-aware masking to `SudokuTaskConfig.compute_action_mask()` |
| `rl/envs/plan_edit_env.py` | Updated mask computation to use current state, recompute after each step |
| `configs/sudoku9x9/upi_trm_9x9.yaml` | Increased training steps to 50k |
| `configs/sudoku9x9/ppo_9x9.yaml` | Increased training steps to 50k |
| `configs/sudoku9x9/dqn_9x9.yaml` | Increased training steps to 50k |

## Files Added

| File | Purpose |
|------|---------|
| `tests/test_constraint_aware_masking.py` | Comprehensive tests for constraint masking |
| `BUCK` (modified) | Added test target for new tests |

---

## Next Steps

1. **Monitor experiments** - Wait for 50k step training to complete
2. **Analyze results** - Compare success rates and scores with previous experiments
3. **Additional experiments if needed:**
   - Curriculum learning (start with easier puzzles)
   - Deeper unrolling (`inner_unroll_n: 4`)
   - Larger batch sizes for UPI-TRM
4. **Update EXPERIMENT_REPORT.md** with new results

---

## Appendix: Code Snippets

### Key Constraint Masking Logic

```python
# === 3. Constraint-aware masking: mask digits in same row/col/box ===
for pos in range(num_positions):
    if self.is_given_cell(int(flat_inputs[pos].item())):
        continue

    row = pos // grid_size
    col = pos % grid_size
    box_row = (row // box_size) * box_size
    box_col = (col // box_size) * box_size

    used_digits = set()

    # Check row
    for c in range(grid_size):
        cell_pos = row * grid_size + c
        val = int(state[cell_pos].item())
        if val > 1:
            used_digits.add(val)

    # Check column
    for r in range(grid_size):
        cell_pos = r * grid_size + col
        val = int(state[cell_pos].item())
        if val > 1:
            used_digits.add(val)

    # Check box
    for br in range(box_size):
        for bc in range(box_size):
            cell_pos = (box_row + br) * grid_size + (box_col + bc)
            val = int(state[cell_pos].item())
            if val > 1:
                used_digits.add(val)

    # Mask out actions that place used digits at this position
    for digit_token in used_digits:
        action_idx = pos * vocab_size + digit_token
        if action_idx < num_actions - 1:
            mask[action_idx] = False
```

---

**Report generated:** January 28, 2026
**Implementation status:** Complete
**Experiments status:** Running
