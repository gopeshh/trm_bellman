# Experiments Documentation for ICML 2026 Paper Section 6

**Document Purpose**: This file documents the experimental setup, hyperparameters, and methodology for reproducing the results in Section 6 of the ICML 2026 paper on UPI-TRM.

**Last Updated**: December 27, 2024

---

## 1. Experimental Overview

### 1.1 Research Questions

The experiments address the following questions:

1. **Q1 (Main Result)**: Does UPI-TRM outperform standard RL baselines (PPO, A2C) on constraint satisfaction tasks?
2. **Q2 (Architecture)**: Does the recursive latent structure (TRM) provide benefits over non-recursive encoders?
3. **Q3 (Theory Components)**: Which theoretical components contribute most to performance?
   - Exact baseline summation (Theorem 5.9)
   - Contraction enforcement (Assumption 4.2)
   - Conservative policy iteration (CPI mixture)
   - Forward-invariant projection (Assumption 4.1)

### 1.2 Task: 9×9 Sudoku

**Dataset**: `data/sudoku-extreme-1k-aug-1000`
- 1,000 extreme-difficulty Sudoku puzzles
- Each puzzle augmented 1,000× via symmetry transformations
- Total: 1M training examples

**Task Formulation**:
- **State**: (puzzle x, candidate solution y) where x ∈ {0,...,9}^81, y ∈ {0,...,9}^81
- **Actions**: 892 discrete actions = 81 positions × 11 values + 1 STOP action
- **Reward**: Shaped reward with potential-based shaping (Equation 4)
- **Episode**: Maximum T=20 edits per episode
- **Success**: Puzzle is solved when checker(x, y) = 10.0 (maximum score)

**Checker Function**:
```python
def sudoku_checker(x, y) -> float:
    """Returns score in [0, 10] based on constraint satisfaction."""
    # Score = 10 - (row_violations + col_violations + box_violations)
    # Maximum score 10.0 = fully solved puzzle
```

---

## 2. Methods Compared

### 2.1 UPI-TRM (Our Method)

**Architecture**:
- TRM backbone with ~7M parameters
- Inner recursion: n=4 latent unrolling steps
- Outer loop: K=5 bootstrapped value targets
- EditPolicyHead for discrete action selection
- LatentValueHead for state value estimation

**Theory-Exact Features** (all enabled):
| Feature | Config Key | Value | Paper Reference |
|---------|------------|-------|-----------------|
| Exact baseline summation | `exact_baseline_summation` | `true` | Theorem 5.9 |
| Exact K-step targets | `exact_k_step_targets` | `true` | Section 5 |
| Contraction enforcement | `enable_contraction` | `true` | Assumption 4.2 |
| Forward-invariant projection | `latent_ball_radius` | `10.0` | Assumption 4.1 |
| Conservative mixture | `mixture_alpha` | `0.05` | Section 6.5 |
| Theory-exact mixture | `theory_exact_mixture` | `true` | CPI formulation |

**Reward Shaping**:
- `reward_shaping: true`
- `fail_terminal_reward: -10.0` (satisfies Remark 2.6: ≤ -γC_max)
- `solve_terminal_reward: 0.0`

### 2.2 PPO Baseline

**Architecture**: NoRecursionEncoder (MLP variant)
- 2-layer MLP encoder (no latent recursion)
- Same policy/value head structure as UPI-TRM
- ~2M parameters

**Algorithm**: Proximal Policy Optimization
- Clipped surrogate objective (ε=0.2)
- 4 epochs per update
- 4 minibatches per epoch
- GAE advantage estimation (λ=0.95)

### 2.3 A2C Baseline

**Architecture**: NoRecursionEncoder (MLP variant)
- Same as PPO baseline

**Algorithm**: Advantage Actor-Critic
- Single gradient step per rollout
- Standard advantage estimation
- Entropy regularization

### 2.4 Ablation Variants

| Ablation | What's Disabled | Tests |
|----------|-----------------|-------|
| No Contraction | `enable_contraction: false` | Assumption 4.2 |
| No Exact Baseline | `exact_baseline_summation: false` | Theorem 5.9 (KEY) |
| No CPI Mixture | `mixture_alpha: 1.0` | Conservative update |
| No Projection | `latent_ball_radius: 0` | Assumption 4.1 |
| No Shaped Rewards | `reward_shaping: false` | Equation 4 |
| High Entropy | `entropy_coef: 0.1` | Exploration |
| Low Alpha | `mixture_alpha: 0.01` | Conservative update |

---

## 3. Hyperparameters

### 3.1 Common Hyperparameters (All Methods)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Training steps | 20,000 | ~4 hours on A100 |
| Batch size | 32 | Transitions per update |
| Discount (γ) | 0.99 | Standard value |
| Max edits (T) | 20 | Episode horizon |
| Eval interval | 50 steps | Evaluation frequency |
| Eval episodes | 50 | Episodes per evaluation |
| Seeds | {42, 123, 456} | 3 seeds for statistics |

### 3.2 UPI-TRM Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| K (bootstrap horizon) | 5 | K-step returns |
| n (inner unroll) | 4 | Latent recursion steps |
| Policy LR | 1e-4 | Adam optimizer |
| Value LR | 3e-4 | Adam optimizer |
| Entropy coef | 0.01 | Exploration bonus |
| Mixture α | 0.05 | CPI update rate |
| Target L_z | 0.9 | Lipschitz target |
| Latent ball R | 10.0 | Projection radius |
| C_max | 10.0 | Max checker score |

### 3.3 PPO Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| Clip ε | 0.2 | PPO clipping |
| VF coef | 0.5 | Value loss weight |
| Entropy coef | 0.01 | Same as UPI-TRM |
| PPO epochs | 4 | Updates per rollout |
| Minibatches | 4 | Per epoch |
| GAE λ | 0.95 | Advantage estimation |
| Rollout length | 128 | Steps before update |

### 3.4 A2C Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| Entropy coef | 0.01 | Same as others |
| VF coef | 0.5 | Value loss weight |
| Rollout length | 5 | Steps before update |

---

## 4. Experimental Protocol

### 4.1 Training Procedure

1. **Initialization**: Random weight initialization (no pretraining)
2. **Data sampling**: Uniform sampling from augmented dataset
3. **Episode structure**:
   - Sample puzzle x from dataset
   - Initialize y = empty grid (all zeros)
   - Run up to T=20 edit steps
   - Episode ends on STOP action or T steps
4. **Evaluation**: Every 50 steps, run 50 greedy episodes

### 4.2 Metrics Tracked

**Primary Metrics** (for paper figures):
- `eval/success_rate`: Fraction of puzzles solved (checker=10.0)
- `eval/mean_score`: Average final checker score [0, 10]

**Secondary Metrics** (for analysis):
- `train/value_loss`: TD error for value function
- `train/policy_loss`: Policy gradient loss
- `theory/hat_Lz`: Empirical Lipschitz constant of latent map
- `debug/avg_episode_length`: Average steps per episode
- `debug/avg_stop_prob`: STOP action probability

### 4.3 Evaluation Protocol

- **Greedy evaluation**: argmax policy (no sampling)
- **50 episodes** per evaluation point
- **Same puzzle set** across all methods for fair comparison
- **Report**: mean ± std across 3 seeds

---

## 5. Experiment Commands

### 5.1 Main Experiments

```bash
# WandB Project: UPI-TRM-ICML-Full-Experiments
# Dataset: data/sudoku-extreme-1k-aug-1000

# UPI-TRM (Our Method) - 3 seeds
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --config configs/rl_sudoku_shaped_theory_exact.yaml \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name upi-trm-seed$seed
done

# PPO + MLP Baseline - 3 seeds
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --baseline ppo \
        --backbone norec-mlp \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name ppo-mlp-seed$seed
done

# A2C + MLP Baseline - 3 seeds
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --baseline a2c \
        --backbone norec-mlp \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name a2c-mlp-seed$seed
done

# PPO + TRM Backbone (isolates algorithm vs architecture)
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --baseline ppo \
        --backbone trm \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name ppo-trm-seed$seed
done
```

### 5.2 Ablation Experiments

```bash
# Ablation: No Exact Baseline (KEY - tests Theorem 5.9)
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --config configs/ablations/ablation_no_exact_baseline.yaml \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name ablation-no-exact-baseline-seed$seed
done

# Ablation: No Contraction (tests Assumption 4.2)
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --config configs/ablations/ablation_no_contraction.yaml \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name ablation-no-contraction-seed$seed
done

# Ablation: No CPI Mixture (α=1.0)
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --config configs/ablations/ablation_no_conservative_mixture.yaml \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name ablation-no-cpi-seed$seed
done

# Ablation: No Shaped Rewards (Sparse)
for seed in 42 123 456; do
    buck2 run //buiksat_trm:upi_trm_train -- \
        --dataset-paths data/sudoku-extreme-1k-aug-1000 \
        --config configs/rl_sudoku_sparse_theory_exact.yaml \
        --train-steps 20000 \
        --seed $seed \
        --wandb-project UPI-TRM-ICML-Full-Experiments \
        --wandb-run-name sparse-theory-exact-seed$seed
done
```

---

## 6. Expected Results

### 6.1 Hypothesized Outcomes

Based on theory and preliminary experiments:

| Method | Expected Success Rate | Notes |
|--------|----------------------|-------|
| UPI-TRM (full) | 40-60% | Best performance |
| PPO + TRM | 20-35% | Algorithm matters |
| PPO + MLP | 10-25% | Baseline |
| A2C + MLP | 5-20% | Simplest baseline |
| UPI-TRM (no exact baseline) | 25-40% | Theorem 5.9 critical |
| UPI-TRM (no contraction) | 30-45% | Stability affected |
| UPI-TRM (sparse) | 15-30% | Credit assignment harder |

### 6.2 Key Comparisons

1. **UPI-TRM vs PPO+MLP**: Shows overall method improvement
2. **UPI-TRM vs PPO+TRM**: Isolates algorithm contribution (vs architecture)
3. **Full vs No Exact Baseline**: Tests Theorem 5.9 (O(α·ε_A) bound)
4. **Full vs No Contraction**: Tests Assumption 4.2 stability
5. **Shaped vs Sparse**: Tests reward shaping contribution

---

## 7. Figure Generation

### 7.1 Learning Curves (Figure 1)

```bash
# After experiments complete, aggregate data
python scripts/aggregate_runs.py \
    --project UPI-TRM-ICML-Full-Experiments \
    --output artifacts/summary.parquet

# Generate main learning curve figure
python scripts/plot_learning_curves.py \
    --input artifacts/summary.parquet \
    --output paper/figures/learning_curves_main.pdf \
    --methods upi-trm,ppo-mlp,a2c-mlp,ppo-trm

# Generate ablation learning curves
python scripts/plot_learning_curves.py \
    --input artifacts/summary.parquet \
    --output paper/figures/learning_curves_ablations.pdf \
    --methods upi-trm,ablation-no-exact-baseline,ablation-no-contraction,sparse-theory-exact
```

### 7.2 Ablation Bar Chart (Figure 2)

```bash
python scripts/plot_ablations.py \
    --input artifacts/summary.parquet \
    --output paper/figures/ablation_bar_chart.pdf
```

### 7.3 Results Table (Table 1)

```bash
python scripts/make_tables.py \
    --input artifacts/summary.parquet \
    --output paper/tables/main_results.tex
```

---

## 8. Compute Resources

### 8.1 Hardware

- **GPU**: NVIDIA A100 (40GB)
- **CPU**: AMD EPYC (128 cores)
- **Memory**: 512GB RAM

### 8.2 Time Estimates

| Experiment Set | Runs | Time per Run | Total Time |
|---------------|------|--------------|------------|
| Main (4 methods × 3 seeds) | 12 | ~4 hours | ~48 hours |
| Ablations (4 configs × 3 seeds) | 12 | ~4 hours | ~48 hours |
| **Total** | 24 | - | **~96 hours** |

With parallel execution on 4 GPUs: **~24 hours total**

---

## 9. Reproducibility Checklist

- [ ] Random seeds set (42, 123, 456)
- [ ] Same dataset for all methods
- [ ] Same evaluation protocol (50 episodes, greedy)
- [ ] WandB logging enabled for all runs
- [ ] Checkpoints saved every 1000 steps
- [ ] Config files versioned in git
- [ ] Code commit hash recorded

---

## 10. Paper Text Draft (Section 6)

### 6.1 Experimental Setup

> We evaluate UPI-TRM on 9×9 Sudoku puzzles from the extreme-difficulty dataset (1,000 puzzles, 1M augmented examples). The task is formulated as a meta-MDP where the agent iteratively edits a candidate solution until it satisfies all constraints. We compare against PPO and A2C baselines with both MLP and TRM backbones.

### 6.2 Main Results

> Table 1 shows that UPI-TRM achieves [X]% success rate compared to [Y]% for PPO+MLP and [Z]% for A2C+MLP. The improvement demonstrates the effectiveness of our theory-exact training algorithm.

### 6.3 Ablation Study

> Figure 2 shows the ablation results. Removing exact baseline summation (Theorem 5.9) causes a [X]% drop in performance, confirming the importance of the O(α·ε_A) bound. Disabling contraction enforcement reduces stability, as predicted by Assumption 4.2.

---

## Appendix: Config File Locations

| Config | Path |
|--------|------|
| UPI-TRM Full | `configs/rl_sudoku_shaped_theory_exact.yaml` |
| Sparse Baseline | `configs/rl_sudoku_sparse_theory_exact.yaml` |
| No Contraction | `configs/ablations/ablation_no_contraction.yaml` |
| No Exact Baseline | `configs/ablations/ablation_no_exact_baseline.yaml` |
| No CPI | `configs/ablations/ablation_no_conservative_mixture.yaml` |
| No Projection | `configs/ablations/ablation_no_projection.yaml` |
| High Entropy | `configs/ablations/ablation_high_entropy.yaml` |
| Low Alpha | `configs/ablations/ablation_low_alpha.yaml` |
| K=1 | `configs/ablations/ablation_k1.yaml` |
