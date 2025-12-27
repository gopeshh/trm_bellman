# Baseline Implementation Summary

**Date**: December 27, 2024
**Status**: Complete

## Overview

This document summarizes the implementation of PPO/A2C baseline algorithms for comparison with UPI-TRM in the ICML 2026 paper.

## Components Implemented

### 1. Baseline Trainers (`rl/algos/`)

#### PPO Trainer (`rl/algos/ppo.py`)
- **PPOConfig**: Configuration dataclass with:
  - `clip_eps`: Clipping epsilon (default: 0.2)
  - `vf_coef`: Value function coefficient (default: 0.5)
  - `entropy_coef`: Entropy bonus coefficient
  - `num_epochs`: PPO epochs per update (default: 4)
  - `num_minibatches`: Minibatches per epoch (default: 4)

- **PPOTrainer**: Full PPO implementation with:
  - Rollout collection with GAE
  - Clipped surrogate objective
  - Value function clipping
  - Minibatch updates

#### A2C Trainer (`rl/algos/a2c.py`)
- **A2CConfig**: Simpler configuration for A2C
- **A2CTrainer**: Advantage Actor-Critic with:
  - Single gradient step per rollout
  - Standard advantage estimation
  - Entropy regularization

### 2. NoRecursionEncoder (`models/norec_encoder.py`)

A baseline encoder that replaces TRM's latent recursion with a direct encoding:

- **NoRecEncoderConfig**: Configuration for the encoder
- **MLPEncoder**: Simple MLP over concatenated (x, y) embeddings
- **TransformerEncoder**: Small Transformer encoder variant
- **NoRecursionEncoder**: Wrapper matching TRM's interface:
  - `encode(x, y)` → latent representation
  - `used_value(x, y, n)` → value estimate (n ignored)
  - `policy_dist(x, y, n, action_mask)` → policy distribution (n ignored)
  - Dummy methods for TRM compatibility: `init_latent`, `unroll_latent`, `eval_latent`, `continue_latent`

### 3. Training Script Integration (`upi_trm_train.py`)

Added CLI arguments:
```bash
--baseline {ppo,a2c}     # Algorithm selection (default: None = UPI-TRM)
--backbone {trm,norec-mlp,norec-transformer}  # Model architecture
```

Logic flow:
1. Parse `--backbone` to select model architecture
2. Parse `--baseline` to select trainer type
3. Create appropriate model (TRM or NoRecursionEncoder)
4. Create appropriate trainer (UPITrmTrainer, PPOTrainer, or A2CTrainer)
5. Run training loop

### 4. Baseline Configs (`configs/baselines/`)

Pre-configured YAML files:
- `ppo_trm_sudoku.yaml` - PPO with TRM backbone
- `a2c_trm_sudoku.yaml` - A2C with TRM backbone
- `ppo_norec_sudoku.yaml` - PPO with MLP encoder
- `a2c_norec_sudoku.yaml` - A2C with MLP encoder

### 5. Tests (`tests/test_baselines_unittest.py`)

12 unit tests covering:
- NoRecEncoderConfig serialization
- MLPEncoder forward pass
- TransformerEncoder forward pass
- NoRecursionEncoder interface compatibility
- Action masking
- Dummy method returns

## Usage Examples

### Quick Smoke Test
```bash
# PPO with MLP (fastest)
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline ppo \
    --backbone norec-mlp \
    --train-steps 10 \
    --max-edits 4

# A2C with Transformer
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline a2c \
    --backbone norec-transformer \
    --train-steps 10 \
    --max-edits 4
```

### Full Experiments
```bash
# PPO baseline on 9x9 Sudoku
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline ppo \
    --backbone norec-mlp \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --train-steps 20000 \
    --seed 42 \
    --wandb-project UPI-TRM-ICML-Baselines

# Compare PPO with TRM backbone (isolate algorithm contribution)
buck2 run //buiksat_trm:upi_trm_train -- \
    --baseline ppo \
    --backbone trm \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --train-steps 20000 \
    --seed 42
```

## Experimental Comparison Matrix

| Method | Backbone | Algorithm | Purpose |
|--------|----------|-----------|---------|
| UPI-TRM | TRM | UPI-TRM | Our method (theory-exact) |
| PPO-TRM | TRM | PPO | Test algorithm contribution |
| A2C-TRM | TRM | A2C | Simpler algorithm baseline |
| PPO-NoRec | MLP | PPO | Test recursion contribution |
| A2C-NoRec | MLP | A2C | Simplest baseline |

## Key Differences from UPI-TRM

| Feature | UPI-TRM | PPO/A2C Baselines |
|---------|---------|-------------------|
| Inner recursion | z^(0) → z^(n) | None (single forward pass) |
| Exact baseline summation | Yes (over all actions) | No (batch mean) |
| Conservative policy iteration | Yes (mixture updates) | No (direct updates) |
| K-step bootstrapping | Configurable K | Standard γ^K |
| Contraction enforcement | Spectral normalization | None |
| Theory guarantees | O(α·ε_A) bound | None |

## Files Modified/Created

### New Files
- `rl/algos/__init__.py`
- `rl/algos/ppo.py`
- `rl/algos/a2c.py`
- `models/norec_encoder.py`
- `configs/baselines/ppo_trm_sudoku.yaml`
- `configs/baselines/a2c_trm_sudoku.yaml`
- `configs/baselines/ppo_norec_sudoku.yaml`
- `configs/baselines/a2c_norec_sudoku.yaml`
- `tests/test_baselines_unittest.py`
- `docs/BASELINE_IMPLEMENTATION.md` (this file)

### Modified Files
- `upi_trm_train.py` - Added baseline/backbone CLI args and selection logic
- `CLAUDE.md` - Updated with baseline usage instructions

## Test Results

```
Tests finished: Pass 12. Fail 0. Fatal 0. Skip 0.
```

All 12 baseline tests passing after Buck2 cache clean and rebuild.

## Next Steps

1. **Run comparative experiments** on 9x9 Sudoku
2. **Generate learning curves** comparing UPI-TRM vs baselines
3. **Include results** in ICML 2026 paper Section 6
