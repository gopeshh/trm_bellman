# Table 3: Baseline Comparison on 4×4 Sudoku (TRIVIAL)

**Dataset:** `data/sudoku-4x4-trivial`
**Evaluation:** Greedy (argmax) + action masking, 50 episodes per seed
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
| Random    | 13.72 | 13.72 | 13.72 | 13.72 |
| PPO       | 11.68 | 11.74 | 9.90 | 11.11 |
| A2C       | 14.78 | 14.44 | 14.58 | 14.60 |
| DQN       | 14.92 | 14.92 | 14.92 | 14.92 |
| UPI-TRM   | 15.68 | 15.96 | 15.88 | 15.84 |

## Random Baseline Definition

The **Random (action mask)** baseline is computed using `scripts/eval_random_policy.py`:

- **Environment:** `PlanEditEnv` with identical configuration to baseline trainers
- **Action masking:** Samples uniformly from valid actions only (excludes clue cells)
- **STOP behavior:** `stop_action_mode="disabled"` (STOP is masked out, matching baselines)
- **Episode length:** `max_edits=16`
- **Success criterion:** `sudoku_is_solved()` — all cells filled with no constraint violations
- **Evaluation:** 50 episodes × 3 seeds (42, 123, 456)
- **Checker:** Feasibility checker with `w_v=2.0`, `w_z=5.0`

On `trivial` puzzles (1-4 empty cells), random action selection achieves **52% success rate** because:
- With few empty cells and action masking, random guessing can solve some puzzles
- Mean violations are low (~1.14 per episode) due to the simple puzzle structure
- This provides a meaningful baseline showing that even on easy puzzles, learned methods should beat random

**Reproducibility:**
```bash
python scripts/eval_random_policy.py \
    --dataset-path data/sudoku-4x4-trivial \
    --seeds 42 123 456 \
    --num-episodes 50 \
    --output results/plot_data/random_baseline.csv
```

## Key Observations

1. **UPI-TRM dramatically outperforms all baselines** with 93.3% mean success rate
2. **Random baseline achieves 52%** — setting the bar for trivial 4×4 puzzles
3. **DQN achieves 42%** — fails to beat random baseline
4. **A2C achieves 35.3%** — also fails to beat random baseline
5. **PPO achieves 16.0%** — worst among learned methods

**Critical finding:** PPO, A2C, and DQN all perform **worse than random** on this task, while UPI-TRM achieves 93.3% — demonstrating the value of the universal planning invariant approach.

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
