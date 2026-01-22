# Table 3: Baseline Comparison on TRIVIAL 4×4 Sudoku

**Dataset:** `data/sudoku-4x4-trivial` (1-4 empties)
**Evaluation:** Greedy (argmax) + action masking, 50 episodes
**Training steps:** 5000
**Seeds:** 42, 123, 456

## Final Success Rates (Step 5000)

| Method    | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|-----------|---------|----------|----------|------------|
| Random    | 52% (26/50) | 52% (26/50) | 52% (26/50) | **52.0% ± 0.0%** |
| PPO       | 20% (10/50) | 20% (10/50) | 8% (4/50) | **16.0% ± 6.9%** |
| A2C       | 36% (18/50) | 32% (16/50) | 38% (19/50) | **35.3% ± 3.1%** |
| DQN       | 42% (21/50) | 42% (21/50) | 42% (21/50) | **42.0% ± 0.0%** |
| UPI-TRM   | 92% (46/50) | 96% (48/50) | 92% (46/50) | **93.3% ± 2.3%** |

## Mean Scores (Step 5000)

| Method    | Seed 42 | Seed 123 | Seed 456 | Mean |
|-----------|---------|----------|----------|------|
| PPO       | 11.68 | 11.74 | 9.90 | 11.11 |
| A2C       | 14.78 | 14.44 | 14.58 | 14.60 |
| DQN       | 12.58 | 12.84 | 13.30 | 12.91 |
| UPI-TRM   | 15.68 | 15.96 | 15.88 | 15.84 |

## Random Baseline Definition

The **Random (action mask)** baseline is computed as follows:

1. **Same environment:** Uses `PlanEditEnv` with identical configuration to baselines
2. **Same action masking:** Only samples from valid actions (excludes clue cells)
3. **Same STOP behavior:** `stop_action_mode="disabled"` (STOP is masked out)
4. **Same max_edits:** 16 steps per episode
5. **Same success criterion:** `sudoku_is_solved()` - all cells filled with no violations
6. **Same episodes/seeds:** 50 episodes × 3 seeds (42, 123, 456)

**Action selection:** Uniform random sampling over the set of valid edit actions at each step.
Valid actions are those that:
- Edit non-clue cells only (respects clue mask from inputs)
- Use valid digit values (1-4 for 4×4 Sudoku)

**Reproducibility:** Run `scripts/eval_random_policy.py` to regenerate:
```bash
python scripts/eval_random_policy.py \
    --dataset-path data/sudoku-4x4-ultra-easy \
    --seeds 42 123 456 \
    --num-episodes 50 \
    --output results/plot_data/random_baseline.csv
```

## Key Observations

1. **UPI-TRM dramatically outperforms all baselines** with 93.3% mean success rate
2. **Random baseline achieves 52%** with action masking (trivial puzzles have 1-4 empties)
3. **DQN underperforms random** with 42.0% - learned policy is worse than random!
4. **A2C underperforms random** with 35.3% - also worse than random
5. **PPO is worst** at 16.0% - significantly below random baseline

Note: The high random success rate on trivial puzzles (1-4 empties) with action masking
makes this a weak benchmark for distinguishing baseline algorithms. UPI-TRM's 93.3%
remains impressive as it approaches optimal performance.

## Configuration Notes

- All methods used feasibility checker: `use_feasibility_checker: true`
- Checker weights: `w_v=2.0` (violations), `w_z=5.0` (zero-candidates)
- Contraction disabled for all: `enable_contraction: false`
- UPI-TRM used episodic latent mode: `episodic_latent: true`
- Max episode length: `max_edits: 16`
- Inner unroll depth: `inner_unroll_n: 2`
- Discount factor: `gamma: 0.99`

## Files

- PPO logs: `ppo_s42.log`, `ppo_s123.log`, `ppo_s456.log`
- A2C logs: `a2c_s42.log`, `a2c_s123.log`, `a2c_s456.log`
- DQN logs: `dqn_s42_fixed.log`, `dqn_s123_fixed.log`, `dqn_s456_fixed.log`
- UPI-TRM logs: `upitrm_s42.log`, `upitrm_s123.log`, `upitrm_s456.log`
- Random baseline: `../plot_data/random_baseline.csv`, `../plot_data/random_baseline.json`
- Figure: `figure2_learning_curves.pdf`
