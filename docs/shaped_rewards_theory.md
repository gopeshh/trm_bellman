# Shaped Reward Implementation: Theory Alignment

This document explains how the UPI-TRM codebase implements potential-based reward shaping to match the ICML 2026 paper's formal setting.

## Paper Specification (Section 2.3)

The paper defines shaped rewards using potential-based shaping (Ng et al. 1999):

**Equation 4:**
```
r(s, a, s') = r_0(s, a, s') + γ·Φ(s') - Φ(s)
```

Where:
- `Φ(s) = c(x, y)` is the potential function (checker score)
- `r_0` is the base reward (nonzero only on terminal transitions)
- `γ` is the discount factor

**Absorbing State Convention (Appendix B.3):**
- The paper defines an explicit absorbing state `s_abs` with `Φ(s_abs) = C_max`
- Absorbing self-loop reward: `r(s_abs, a, s_abs) = (γ - 1)·C_max`
- This implies `V^π(s_abs) = -C_max` for all policies

**Rush-to-Fail Mitigation (Remark 2.6):**
- Terminal reward for failing: `r_term_fail ≤ -γ·C_max`
- This condition ensures failing from any state yields non-positive total reward
- Prevents "rush to fail" where agent terminates early to avoid negative shaping

## Code Implementation

### Sudoku Checker (C_max = 10.0)

File: `upi_trm_train.py:212-228`

```python
def sudoku_checker(x, y) -> float:
    """
    Returns scaled score for cell matches.
    Range: [0, 10] where 10 = perfect match (100% cells correct).
    """
    plan = _to_plan_tensor(y).to(torch.long)
    solution_tensor = _to_plan_tensor(solution).to(torch.long)
    matches = (plan == solution_tensor).to(torch.float32)
    return float(matches.mean().item() * 10.0)  # C_max = 10.0
```

**Key property:** `c(x, y_solved) = 10.0` for any solved Sudoku puzzle.

### Reward Computation

File: `rl/envs/plan_edit_env.py:375-443`

The `compute_transition_reward()` method implements the paper's equation:

```python
def compute_transition_reward(
    self,
    phi_old: float,      # c(x, y_old) = Φ(s)
    phi_new: float,      # c(x, y_new) = Φ(s')
    is_stop_action: bool,
    is_terminal: bool,
    is_solved: bool,
) -> float:
    gamma = self.config.gamma

    # Terminal reward r_0 (Paper Remark 2.6)
    r_0 = 0.0
    if is_terminal:
        if is_solved:
            r_0 = self.config.solve_terminal_reward  # default: 0.0
        else:
            r_0 = self.config.fail_terminal_reward    # set to -C_max for theory

    if self.config.reward_shaping:
        # Paper Eq. 4: r = r_0 + γ·Φ(s') - Φ(s)
        r = r_0 + gamma * phi_new - phi_old + stop_penalty
    else:
        # Sparse: only terminal states get checker score
        if is_terminal:
            r = phi_new + r_0 + stop_penalty
        else:
            r = stop_penalty

    return r
```

### Configuration Flags

File: `rl/config.py:74-90`

```python
class RLConfig(BaseModel):
    # Reward shaping toggle
    reward_shaping: bool = True  # Enable potential-based shaping

    # Terminal rewards (Paper Remark 2.6: Rush-to-fail mitigation)
    # fail_terminal_reward: Applied when episode ends WITHOUT solving.
    #   Set to -C_max (e.g., -10.0 for Sudoku) to prevent "rush to fail".
    #   Default 0.0 does NOT fully prevent rush-to-fail.
    fail_terminal_reward: float = 0.0

    # solve_terminal_reward: Applied when episode ends WITH solving.
    #   Default 0.0 relies purely on shaping.
    solve_terminal_reward: float = 0.0
```

## Theory Alignment Checklist

| Paper Requirement | Code Implementation | Status |
|-------------------|---------------------|--------|
| Potential Φ(s) = c(x, y) | `sudoku_checker(x, y)` returns [0, 10] | ✅ |
| C_max = 10.0 | Checker returns 10.0 for solved puzzles | ✅ |
| Shaped reward: r = r_0 + γΦ' - Φ | `compute_transition_reward()` | ✅ |
| Terminal r_0 for solve | `solve_terminal_reward` (default: 0.0) | ✅ |
| Terminal r_0 for fail | `fail_terminal_reward` (default: 0.0) | ⚠️ |
| Rush-to-fail mitigation | **Needs:** `fail_terminal_reward = -10.0` | ❌ |

## How to Enable Theory-Exact Shaped Rewards

### Option 1: YAML Config (Recommended)

Create `configs/rl_sudoku_shaped_theory_exact.yaml`:

```yaml
# Theory-exact shaped rewards for Sudoku (C_max = 10.0)
reward_shaping: true
fail_terminal_reward: -10.0   # -γ·C_max ≈ -9.9 for γ=0.99
solve_terminal_reward: 0.0    # Rely on shaping alone

# Other theory-exact features
exact_k_step_targets: true
exact_baseline_summation: true  # KEY for Theorem 5.9
theory_exact_mixture: true
enable_contraction: true
target_Lz: 0.9
latent_ball_radius: 10.0
```

### Option 2: CLI Override

```bash
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1.yaml \
    --fail-terminal-reward -10.0 \
    --solve-terminal-reward 0.0 \
    --reward-shaping true \
    --seed 42
```

### Option 3: Programmatic (for experiments)

```python
from rl.config import RLConfig

config = RLConfig(
    reward_shaping=True,
    fail_terminal_reward=-10.0,  # -C_max for theory alignment
    solve_terminal_reward=0.0,
    gamma=0.99,
    # ... other params
)
```

## Absorbing State vs. Direct Termination

**Paper's Explicit Absorbing State (Appendix B.3):**
```
s_abs with Φ(s_abs) = C_max
Self-loop: r(s_abs, a, s_abs) = (γ - 1)·C_max = -0.01·10 = -0.1
Fixed value: V^π(s_abs) = -C_max = -10.0
```

**Code's Direct Termination:**
```python
# Episode halts at final plan (x, y_final)
# No explicit s_abs node in the state space
# Terminal reward r_0 applied directly
```

**Why this is equivalent:**

For shaped rewards with terminal bonus:
```
Paper:   r_term = r_0 + γ·Φ(s_abs) - Φ(s_final)
              = r_0 + γ·C_max - c(x, y_final)

Code:    r_term = r_0 + γ·phi_new - phi_old
              where phi_old = c(x, y_final), phi_new = c(x, y_final)
              = r_0  (since plan doesn't change on terminal transition)
```

The key difference: the paper's formulation explicitly adds `γ·C_max` on the terminal transition, while the code relies on the terminal reward `r_0` to capture this.

**For rush-to-fail mitigation:** Setting `fail_terminal_reward = -γ·C_max ≈ -9.9` makes the code equivalent to the paper's absorbing-state convention.

## STOP Action Interaction with Shaping

File: `rl/envs/plan_edit_env.py:66-73`

When `stop_action_mode="noop"` with `reward_shaping=True`, STOP yields:
```
r_stop = stop_action_penalty + (γ·Φ(y) - Φ(y))
       = stop_action_penalty + (γ - 1)·Φ(y)
```

Since γ < 1, the term `(γ - 1)·Φ(y)` is **negative** when Φ(y) > 0.

**Implication:** STOP is strongly penalized near high-scoring states. This is **intentional** and prevents the agent from stopping when progress is possible.

## Verification Tests

To verify shaped reward implementation:

```bash
# Run theory-exact test suite
pytest tests/test_shaped_rewards.py -v

# Check reward computation
pytest tests/test_rl_k_step_targets.py::test_shaped_reward_calculation -v

# Verify rush-to-fail mitigation
pytest tests/test_plan_edit_env.py::test_fail_terminal_reward -v
```

## Experimental Validation

From Bahram's guidance (main.tex:1426-1429):

> First, make the experiments faithfully match the paper's formal setting. Concretely,
> include a shaped-reward Sudoku setting (checker score + potential shaping + absorbing
> normalization) alongside the sparse 0/1 terminal reward version.

**Recommended experiment:**

```bash
# Baseline: shaped rewards WITHOUT rush-to-fail mitigation
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1.yaml \
    --reward-shaping true \
    --fail-terminal-reward 0.0 \
    --seed 42

# Theory-exact: shaped rewards WITH rush-to-fail mitigation
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1_theory_exact.yaml \
    --reward-shaping true \
    --fail-terminal-reward -10.0 \
    --seed 42

# Sparse rewards (for comparison)
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1.yaml \
    --reward-shaping false \
    --seed 42
```

**What to measure:**
1. Success rate (% of puzzles solved)
2. Mean checker score at episode end
3. Early termination rate (% episodes ending before max_edits)
4. Mean episode length

**Expected results:**
- Shaped + rush-to-fail mitigation should have lowest early termination
- Sparse rewards should learn slower (harder credit assignment)
- Shaped without mitigation may show "rush to fail" behavior

## Summary

The current implementation **already supports** theory-exact shaped rewards. To enable full alignment with the paper:

1. Set `fail_terminal_reward = -10.0` (or `-gamma * C_max`)
2. Keep `reward_shaping = True`
3. Verify with `config.validate_theory_alignment()`

The code is production-ready for the ICML experiments once this parameter is set correctly in the config files.
