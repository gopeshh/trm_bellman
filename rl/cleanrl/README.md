# CleanRL Baseline Integration

## Strategy

Dual-implementation approach: the repo has two baseline paths.

1. **In-house baselines** (`rl/algos/{ppo,a2c,dqn}.py`): TRM-shared
   custom trainers using the same backbone and environment as UPI-TRM.
   Used for the paper's main Table 3 comparison.

2. **CleanRL baselines** (`rl/cleanrl/`): independent reimplementation
   following upstream CleanRL conventions with TRM model adapters.
   Validates that baseline results are not artifacts of shared training
   infrastructure. Files and upstream diffs documented in per-file
   docstrings.

Both paths share the same evaluator (`rl/evaluator.py`), checker, dataset
pool construction (`max(batch_size, 8)` via `build_dataset_from_paths`),
and TRM model factory defaults (via `RLConfig`).

## Files

| File | Role |
|------|------|
| `trm_adapter.py` | GymPlanEditEnv, TRMActorCritic, TRMQNetwork, build_sudoku_bundle |
| `ppo_trm.py` | PPO training loop (CartPole + Sudoku) |
| `a2c_trm.py` | A2C as thin PPO wrapper (epochs=1, no clip) |
| `dqn_trm.py` | DQN with Double DQN and n-step returns |
| `cleanrl_runner.py` | Dispatch entrypoint (routes by `algo` field) |

## Smoke configurations

The baseline configuration files remain available for development tests. The
historical pass/fail tables and learned outcomes were removed with the invalid
experiment corpus. Run fresh tests against the exact current commit before
using these wrappers.

The CleanRL Sudoku wrappers are not registered protocol-v2 methods and their
outputs are not paper evidence.
