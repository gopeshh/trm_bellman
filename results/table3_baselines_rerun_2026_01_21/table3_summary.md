# Table 3: Baseline Comparison on TRIVIAL 4×4 Sudoku

**Dataset:** `data/sudoku-4x4-trivial` (1-4 empties)
**Evaluation:** Greedy (argmax) + action masking, 50 episodes
**Training steps:** 5000
**Seeds:** 42, 123, 456

## Final Success Rates (Step 5000)

| Method    | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|-----------|---------|----------|----------|------------|
| PPO       | 20% (10/50) | 20% (10/50) | 8% (4/50) | **16.0% ± 6.9%** |
| A2C       | 36% (18/50) | 32% (16/50) | 38% (19/50) | **35.3% ± 3.1%** |
| DQN       | N/A* | N/A* | N/A* | **N/A*** |
| UPI-TRM   | 92% (46/50) | 96% (48/50) | 92% (46/50) | **93.3% ± 2.3%** |

\* DQN evaluation unavailable due to tensor dimensionality bug in `evaluate_policy_metrics()`

## Mean Scores (Step 5000)

| Method    | Seed 42 | Seed 123 | Seed 456 | Mean |
|-----------|---------|----------|----------|------|
| PPO       | 11.68 | 11.74 | 9.90 | 11.11 |
| A2C       | 14.78 | 14.44 | 14.58 | 14.60 |
| DQN       | N/A | N/A | N/A | N/A |
| UPI-TRM   | 15.68 | 15.96 | 15.88 | 15.84 |

## Key Observations

1. **UPI-TRM dramatically outperforms all baselines** with 93.3% mean success rate vs 35.3% for A2C (best baseline)
2. **PPO struggles significantly** on this task (16% mean) despite feasibility reward shaping
3. **A2C shows moderate performance** (35.3% mean), benefiting from faster updates (num_steps=32)
4. **DQN training completed** but evaluation failed due to a tensor shape bug in the checker functions

## Configuration Notes

- All methods used feasibility checker: `use_feasibility_checker: true`
- Contraction disabled for all: `enable_contraction: false`
- UPI-TRM used episodic latent mode: `episodic_latent: true`
- Max episode length: `max_edits: 16`
- Inner unroll depth: `inner_unroll_n: 2`

## Files

- PPO logs: `ppo_s42.log`, `ppo_s123.log`, `ppo_s456.log`
- A2C logs: `a2c_s42.log`, `a2c_s123.log`, `a2c_s456.log`
- DQN logs: `dqn_s42.log`, `dqn_s123.log`, `dqn_s456.log`
- UPI-TRM logs: `upitrm_s42.log`, `upitrm_s123.log`, `upitrm_s456.log`
