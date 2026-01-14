# Verification Results

These logs were produced **after** the multi-config merge fix (commit `b9968dc`) and after running `buck2 clean` to ensure the fresh binary was used.

## Purpose

Verify that:
1. The fixed binary correctly loads multiple `--config` files
2. Configs merge correctly (K=1, exact_baseline_summation=True when requested)
3. Value function does NOT collapse to -20 early

## Test Conditions

- **Dataset**: `buiksat_trm/data/sudoku-4x4-easy_6to8empties`
- **Base config**: `configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml`
- **Condition config**: `configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml`
- **Steps**: 500
- **Seed**: 42

## Verified Settings (from log)

| Setting | Expected | Actual | Status |
|---------|----------|--------|--------|
| K | 1 | 1 | ✓ |
| exact_k_step_targets | True | True | ✓ |
| exact_baseline_summation | True | True | ✓ |
| theory_exact_mixture | True | True | ✓ |

## Value Function Stability (NOT collapsed)

| Step | Target Mean | V(s) Mean | Status |
|------|-------------|-----------|--------|
| 10 | -1.41 | -0.94 | ✓ Stable |
| 20 | -1.07 | -1.32 | ✓ Stable |
| 30 | -1.50 | -1.63 | ✓ Stable |
| 40 | -1.50 | -1.62 | ✓ Stable |
| 50 | -1.41 | -1.84 | ✓ Stable |
| 100 | -2.30 | -2.17 | ✓ Stable |

Compare to **INVALID** (broken) runs:
- Step 10: target = -14.64, V(s) = 0.15
- Step 100: target = -10.86, V(s) = -11.19
- Step 500+: V(s) = -20.0 (SATURATED)

## Command Used

```bash
# After buck2 clean
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    -- \
    --dataset-paths buiksat_trm/data/sudoku-4x4-easy_6to8empties \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml \
    --train-steps 500 --seed 42 --no-wandb
```

## Files

- `verification_1_no_contraction_seed42_500steps.log` - No-contraction training log
- `verification_3_standard_contraction_seed42_500steps.log` - Standard contraction training log (complete 500 steps)

---

## Contraction Comparison Test

A paired verification was run with `3_standard_contraction.yaml` to compare the effect of enabling contraction (`enable_contraction=true`, `target_Lz=0.90`).

### Contraction Config Settings (from log)

| Setting | Expected | Actual | Status |
|---------|----------|--------|--------|
| K | 1 | 1 | ✓ |
| enable_contraction | True | True | ✓ |
| target_Lz | 0.90 | 0.90 | ✓ |
| exact_baseline_summation | True | True | ✓ |
| Is theory-exact | True | True | ✓ |

### Critical Finding: Contraction Causes Value Collapse

| Step | No-Contraction Target | No-Contraction V(s) | Contraction Target | Contraction V(s) |
|------|----------------------|---------------------|-------------------|------------------|
| 10 | -1.41 | -0.94 | **-19.84** | **-12.94** |
| 20 | -1.07 | -1.32 | **-19.91** | **-17.76** |
| 30 | -1.50 | -1.63 | **-19.91** | **-19.70** |
| 50 | -1.41 | -1.84 | -19.9x | **-20.00** |
| 100 | -2.30 | -2.17 | -19.9x | **-20.00** |
| 250 | - | - | -19.89 | **-20.00** (eval: 0% solve) |
| 300 | - | - | -19.90 | **-20.00** (eval: 0% solve) |
| 500 | - | - | -19.84 | **-20.00** (eval: 0% solve, 0/50) |

**Key Observations**:
1. With contraction, targets saturate to -20 by step 10
2. V(s) = **-20.00 with std=0.00** from ~step 40 onwards (completely flat)
3. Evaluation shows **0% solve rate** at both step 250 and 300
4. Without contraction, values remain stable in -1 to -2 range with meaningful learning

### Hypothesis

The contraction modification (spectral normalization with target_Lz=0.9) significantly alters network behavior. The warning in the log states:

> "rl_enable_contraction=True: Applying operator-norm clamping and contraction scaling to z->z path layers. This is REQUIRED for Assumption 3.2 (L_z < 1) but will modify network behavior. Pretrained weights from vanilla TRM may require fine-tuning."

Without pretrained weights or fine-tuning, the contraction-enabled network may start in a poor initialization state, causing immediate pessimistic collapse.

### Contraction Command Used

```bash
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    -- \
    --dataset-paths buiksat_trm/data/sudoku-4x4-easy_6to8empties \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/3_standard_contraction.yaml \
    --train-steps 500 --seed 42 --no-wandb
```

---

## Conclusions

1. **Multi-config merge fix is verified working** - Both configs correctly load K=1, exact_baseline_summation=True, etc.

2. **No-contraction condition is stable** - Value function remains in -1 to -2 range, does not collapse to -20.

3. **Contraction causes immediate collapse** - With `enable_contraction=true`, targets saturate to -20 within 10 steps. This requires further investigation (pretraining, different target_Lz, etc.).
