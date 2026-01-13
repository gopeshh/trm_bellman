# INVALID RESULTS

**These logs are INVALID and should NOT be used for analysis or publication.**

## Reason

These experiments were run with a **stale cached buck2 binary** that did not include the multi-config merge fix (commit `b9968dc`). The argparse `--config` argument was using `type=str` instead of `action='append'`, causing only the last config file to be loaded.

## Expected vs Actual Settings

| Setting | Expected (from base config) | Actual (due to bug) |
|---------|----------------------------|---------------------|
| K | 1 | 5 |
| exact_baseline_summation | True | False |
| theory_exact_mixture | True | False |
| exact_k_step_targets | True | False |

## Symptoms

- Value function collapsed to -20.0 (saturated) within ~100 steps
- No learning signal after collapse
- All conditions showed 0% success rate

Example from logs:
```
Step 10:  V(s) = 0.15,  target = -14.64
Step 50:  V(s) = -2.50, target = -12.93
Step 100: V(s) = -11.19
Step 500+: V(s) = -20.0 (SATURATED)
```

## Fix

The bug was fixed in commit `b9968dc`:
- Changed `--config` argument from `type=str` to `action='append'`
- Updated config loading to merge multiple YAML files in order

## Verified Working Command

After running `buck2 clean`, the following command correctly loads both configs:

```bash
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    -- \
    --dataset-paths buiksat_trm/data/sudoku-4x4-easy_6to8empties \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml \
    --train-steps 500 --seed 42 --no-wandb
```

With the fix, value targets are sane (~-1.5) and do not collapse to -20.

## Valid Results Location

See `results/verification/` for results produced with the fixed binary.
