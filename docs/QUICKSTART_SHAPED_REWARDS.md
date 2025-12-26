# Shaped Reward Experiments - Quick Start Guide

## ✅ Implementation Status: COMPLETE

All infrastructure for shaped-reward Sudoku experiments is ready to run.

## What's Been Implemented

### 1. Theory-Exact Configs

| Config | Description | Key Features |
|--------|-------------|--------------|
| `rl_sudoku_shaped_theory_exact.yaml` | Main contribution (all features ON) | Shaped rewards + exact baseline + CPI + contraction |
| `rl_sudoku_sparse_theory_exact.yaml` | Sparse rewards baseline | Terminal-only rewards (harder credit assignment) |

### 2. Ablation Configs (7 total)

| Config | Tests | Expected Impact |
|--------|-------|-----------------|
| `ablation_no_contraction.yaml` | Assumption 4.2 | Higher variance, instability |
| `ablation_no_exact_baseline.yaml` | **Theorem 5.9 (KEY)** | Weaker improvement, more variance |
| `ablation_no_conservative_mixture.yaml` | CPI benefit (α=1.0 greedy) | Policy oscillation, instability |
| `ablation_no_projection.yaml` | Assumption 4.1 | Latents escape, possible divergence |
| `ablation_no_theory_exact_mixture.yaml` | Policy-space vs param-space | Similar but not CPI-compliant |
| `ablation_no_theory_features.yaml` | Full ablation (all OFF) | Baseline RL, worst performance |
| `ablation_sparse_no_theory.yaml` | Hardest baseline | Sparse + no theory features |

## Quick Start

### 1. Verify Installation

```bash
# Check dataset
ls -la data/sudoku-extreme-1k-aug-1000/

# Generate ablation configs (if not already done)
python scripts/generate_ablation_configs.py

# Verify all configs are valid
python scripts/verify_configs.py
```

### 2. Run Single Experiment (Test)

```bash
# Shaped theory-exact (main contribution)
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_shaped_theory_exact.yaml \
    --train-steps 5000 \
    --seed 42

# Sparse baseline (comparison)
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_sparse_theory_exact.yaml \
    --train-steps 5000 \
    --seed 42

# Ablation: no exact baseline (tests Theorem 5.9 - KEY CONTRIBUTION)
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/ablations/ablation_no_exact_baseline.yaml \
    --train-steps 5000 \
    --seed 42
```

### 3. Run Full Experiment Suite

```bash
# All configs × 3 seeds = 24 total runs
./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 10000

# Run in parallel (faster, requires GNU parallel)
./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 10000 --parallel

# Dry run (test commands without executing)
./scripts/run_shaped_reward_experiments.sh --dry-run
```

### 4. Monitor Results

Results are logged to W&B:
- **Project:** `UPI-TRM-ICML-Shaped-Rewards`
- **Groups:** `shaped-theory-exact`, `sparse-theory-exact`, `ablation-no-contraction`, etc.

Key metrics:
- `eval_success_rate` - % puzzles solved (main metric)
- `eval_mean_score` - Average checker score
- `eval_mean_episode_length` - Episode length
- `theory/hat_Lz`, `theory/hat_Cz`, `theory/unrolling_term` - Theory metrics

## Expected Results (Based on Theory)

```
Shaped Theory-Exact (Best)
├─ Success rate: ~70-90% (depends on puzzle difficulty)
├─ Stable training (low variance)
└─ Fast convergence

Ablation: No Exact Baseline
├─ Success rate: ~60-80% (worse than shaped)
├─ Higher variance
└─ Slower convergence
└─ **Tests Theorem 5.9 - KEY CONTRIBUTION**

Ablation: No Contraction
├─ Success rate: ~50-70%
├─ Training instability (high variance)
└─ Possible divergence

Ablation: All Features OFF
├─ Success rate: ~30-50% (baseline RL)
├─ Highest variance
└─ Slowest convergence

Sparse (No Shaping)
├─ Success rate: ~40-60%
├─ Harder credit assignment
└─ Much slower than shaped

Sparse + No Theory (Hardest)
├─ Success rate: ~10-30% (may fail completely)
├─ Standard actor-critic baseline
└─ Demonstrates UPI-TRM necessity
```

## Files Created

### Documentation
1. `docs/shaped_rewards_theory.md` - Theory-to-code alignment
2. `docs/shaped_rewards_implementation_complete.md` - Implementation summary

### Configs
3. `configs/rl_sudoku_shaped_theory_exact.yaml` - Main shaped config
4. `configs/rl_sudoku_sparse_theory_exact.yaml` - Sparse baseline
5. `configs/ablations/ablation_*.yaml` - 7 ablation configs

### Scripts
6. `scripts/generate_ablation_configs.py` - Auto-generate ablations
7. `scripts/run_shaped_reward_experiments.sh` - Full experiment runner
8. `scripts/verify_configs.py` - Config validation

## Validation Checklist

- [x] Shaped-reward environment implemented (already in codebase)
- [x] Theory-exact config created (fail_terminal_reward = -10.0)
- [x] Sparse baseline config created
- [x] Ablation configs generated (7 configs)
- [x] Experiment runner script created
- [x] Config validation script created
- [x] Documentation written

Ready to run experiments! ✅

## Next Steps

1. **Pilot run** (quick test):
   ```bash
   ./scripts/run_shaped_reward_experiments.sh --seeds 1 --steps 5000
   ```

2. **Full run** (publication quality):
   ```bash
   ./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 20000
   ```

3. **Analyze results** in W&B and generate figures for paper

4. **Write results section** addressing Bahram's guidance

## Questions?

See detailed documentation:
- Theory alignment: `docs/shaped_rewards_theory.md`
- Implementation details: `docs/shaped_rewards_implementation_complete.md`
- Main README: `README.md`
