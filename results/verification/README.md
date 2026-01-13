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

- `verification_1_no_contraction_seed42_500steps.log` - Full training log

## Conclusion

The multi-config merge fix is **verified working**. The value function remains stable and does not collapse to -20.
