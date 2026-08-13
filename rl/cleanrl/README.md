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

## CartPole Smoke Gates

Standard: 3 seeds on CartPole-v1, >195 mean greedy eval return.
Budgets: PPO 100k, A2C 100k, DQN/DQN-n5 500k env steps.

## PPO CartPole Gate — PASSED

Config: `configs/baselines/cleanrl_ppo_cartpole.yaml`

Fixes applied before pass (see git history):
1. Truncation bootstrap: `next_done` tracks `terminated` only; truncated
   episodes get `gamma * V(final_obs)` added to reward.
2. Separate actor/critic MLPs in `CartPoleActorCritic` (CartPole smoke
   class only — does not affect `TRMActorCritic`).
3. Canonical orthogonal init: hidden std=sqrt(2), actor head std=0.01,
   critic head std=1.0.

Results (2026-04-26, `results/cleanrl_cartpole_gate_v2/`):

| Seed | Final mean return | Peak | Curve shape |
|------|-------------------|------|-------------|
| 0    | 440.15            | 500  | Rises to 500 at 30k, stays 420–500 |
| 1    | 487.85            | 500  | Rises to 500 at 10k, stays 478–500 |
| 2    | 414.80            | 500  | Rises to 500 at 15k, stays 375–440 |

Aggregate: **447.60 ± 37.09** — 3/3 above 400, no collapse-recovery.

Pre-fix comparison (before truncation/arch/init fixes):
- Seed 0: 292.55 (collapsed from 462), Seed 1: 478.20, Seed 2: 442.85
- Aggregate: 404.53 ± 98.58 — 2/3 above 400, unstable curves.

## A2C CartPole Gate — PASSED

Config: `configs/baselines/cleanrl_a2c_cartpole.yaml`

Key differences from PPO: no clipping, no minibatches/epochs, single
full-batch gradient step per rollout, RMSprop optimizer, shorter rollout
(num_steps=5) with more envs (num_envs=16), gae_lambda=1.0.

Same truncation bootstrap and CartPoleActorCritic (separate nets +
orthogonal init) as PPO.

Results (2026-04-26, `results/cleanrl_a2c_cartpole_gate_v1/`):

| Seed | Final mean return | Peak  | Curve shape |
|------|-------------------|-------|-------------|
| 0    | 426.40            | 491.4 | Rises to 440 at 10k, stays 410–490 |
| 1    | 449.95            | 500   | Rises to 500 at 40k, stays 420–500 |
| 2    | 417.90            | 500   | Rises to 500 at 20k, stays 418–495 |

Aggregate: **431.42 ± 16.60** — 3/3 above 400, stable curves.

Note: the original task spec requires >200 within 200k steps. This gate
passed at 100k (stricter budget), which subsumes the 200k requirement.

## DQN (n=1) CartPole Gate — PASSED

Config: `configs/baselines/cleanrl_dqn_cartpole.yaml`

Off-policy with replay buffer, epsilon-greedy exploration (linear decay
over 50% of training), hard target network updates every 500 steps.
Canonical CleanRL hyperparameters: learning_starts=10000, train_freq=10,
buffer_size=10000, batch_size=128.

Results (2026-04-26, `results/cleanrl_dqn_cartpole_gate_500k/`):

| Seed | Final mean return | Peak | Convergence step |
|------|-------------------|------|------------------|
| 0    | 500.00            | 500  | 275k (stable 275k–500k, one dip at 450k) |
| 1    | 500.00            | 500  | 475k (slow climb with dip at 225–300k) |
| 2    | 500.00            | 500  | 350k (dip at 200–300k, then stable) |

Aggregate: **500.00 ± 0.00** — 3/3 at maximum, all converged.

## DQN (n=5) CartPole Gate — PASSED

Config: `configs/baselines/cleanrl_dqn_n5_cartpole.yaml`

Same as DQN n=1 but with 5-step return accumulation in the replay buffer.
Bootstrap discount is gamma^5. Uses `NStepBuffer` to accumulate rewards
before pushing to the main replay.

Results (2026-04-26, `results/cleanrl_dqn_n5_cartpole_gate_500k/`):

| Seed | Final mean return | Peak | Convergence step |
|------|-------------------|------|------------------|
| 0    | 500.00            | 500  | 300k (unstable 100–275k, locked at 300k) |
| 1    | 500.00            | 500  | 225k (stable 225k–500k, one dip at 350k) |
| 2    | 500.00            | 500  | 225k (stable 225k–500k) |

Aggregate: **500.00 ± 0.00** — 3/3 at maximum, all converged.

## Hard 4x4 Sudoku Benchmark — PENDING

Protocol: 320,000 env interactions, 10 seeds per algorithm, no-mask
variant (matches `run_hard4x4_trm_baselines_nomask_10seed_4gpu.sh`).

Dataset pool: `max(batch_size, 8)` = 256 puzzles (same as in-house).

| Algorithm | Config | Implementation | Evaluator |
|-----------|--------|----------------|-----------|
| PPO | `cleanrl_ppo_trm.yaml` | CleanRL training loop | `evaluate_plan_policy_with_scores` |
| A2C | `cleanrl_a2c_trm.yaml` | CleanRL (thin PPO wrapper) | `evaluate_plan_policy_with_scores` |
| DQN n=1 | `cleanrl_dqn_trm.yaml` | In-house `DQNTrainer` wrapper | `trainer.evaluate_policy_metrics` (greedy Q) |

Note: DQN uses the in-house `DQNTrainer` because the CleanRL DQN training
loop has an unresolved observation-encoding divergence on Sudoku. The DQN
row should be labeled "in-house DQN via benchmark harness," not CleanRL.

DQN n=5 is dropped from the Sudoku benchmark. The in-house `DQNTrainer`
supports n-step targets, but the benchmark wrapper currently configures only
one-step targets and rejects `n_step > 1`. The CartPole DQN n=5 smoke gate
passed through the CleanRL implementation.
