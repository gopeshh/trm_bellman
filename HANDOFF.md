# Session Handoff: Contraction vs SGD Tradeoff Experiments

**Date**: 2026-01-13
**Branch**: `feature/upi-trm-clean`
**Last Commit**: `b9968dc` - Fix multi-config loading bug and add diagnostic tools

---

## Summary

This session investigated why previous contraction vs SGD experiments failed on the 6-8 empties dataset. We discovered and fixed a critical bug in config loading, and started a verification test.

---

## Key Bug Fixed

### Problem: Multi-Config Loading Bug

When passing multiple `--config` arguments to `upi_trm_train.py`:
```bash
--config base.yaml --config override.yaml
```

**Before fix**: Only `override.yaml` was loaded (argparse `type=str` kept only last value)
**After fix**: Both configs are merged in order (using `action='append'`)

### Impact

The previous experiments ran with **wrong settings**:
| Setting | Expected | Actual (Bug) |
|---------|----------|--------------|
| K | 1 | 5 |
| exact_baseline_summation | True | False |
| theory_exact_mixture | True | False |
| Exact baseline (Thm 5.9) | ✓ | ✗ |

This caused the value function to collapse to -20.0 (saturated) with no learning.

---

## Files Modified

| File | Change |
|------|--------|
| `upi_trm_train.py` | Fixed `--config` to use `action='append'`, updated config loading to merge multiple configs |
| `scripts/run_contraction_sgd_tradeoff.py` | Fixed paths for buck2 (FBCODE_DIR), absolute paths for output files |
| `scripts/plot_value_collapse_diagnostic.py` | NEW - generates diagnostic plots/CSV for value function analysis |

---

## Current Status

### Quick 500-Step Verification Test

A 500-step test is running to verify the fix works:
```bash
buck2 run //buiksat_trm:upi_trm_train \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml \
    --config buiksat_trm/configs/experiments/contraction_sgd_tradeoff/1_no_contraction.yaml \
    --train-steps 500 --seed 42
```

**Log file**: `/tmp/quick_test_500.log`

**Verified settings loaded correctly**:
- K=1 ✓
- Exact baseline (Thm 5.9): ✓
- Exact K-step targets: True ✓
- Exact baseline summation: True ✓

### Performance Note

`exact_baseline_summation=True` is slow because it computes Q(s,a) for all 97 actions per sample. Estimated time:
- 500 steps: ~15-30 minutes
- 5000 steps: ~50+ hours

---

## Previous Experiment Results (Failed - Wrong Config)

Results from `results/plot_data_contraction_sgd_sudoku-4x4-easy_6to8empties/`:

| Condition | Success Rate | Mean Score | Notes |
|-----------|--------------|------------|-------|
| 1_no_contraction | 0.0% | 5.838 | Value collapsed to -20.0 |
| 2_weak_contraction | 0.0% | 5.775 | Value collapsed to -20.0 |
| 3_standard_contraction | 0.0% | 5.850 | Value collapsed to -20.0 |
| 4_scheduled_contraction | N/A | N/A | Never ran (session crashed) |

These results are **invalid** due to the config loading bug.

---

## Next Steps

1. **Check verification test results**:
   ```bash
   tail -50 /tmp/quick_test_500.log
   grep "eval_success" /tmp/quick_test_500.log
   ```

2. **If verification passes, run full experiments**:
   ```bash
   python3 scripts/run_contraction_sgd_tradeoff.py --seed 42
   ```
   Note: This will take 50+ hours with exact_baseline_summation=True

3. **Alternative: Faster experiments without exact baseline**:
   Create a config with `exact_baseline_summation: false` for faster iteration (loses Theorem 5.9 bound but much faster)

4. **Analyze results**:
   ```bash
   python3 scripts/plot_value_collapse_diagnostic.py \
       --log-dir results/plot_data_contraction_sgd_sudoku-4x4-trivial \
       --output results/plots_contraction_sgd_sudoku-4x4-trivial/diagnostic.png
   ```

---

## Key Diagnostic: Value Function Collapse

The original failure showed classic value function collapse:

```
Step 10:  V(s) = 0.15,  target = -14.64
Step 50:  V(s) = -2.50, target = -12.93
Step 100: V(s) = -11.19
Step 500+: V(s) = -20.0 (SATURATED - no learning signal)
```

With the fix, V(s) should NOT saturate immediately because:
1. K=1 (stable single-step TD)
2. Exact baseline removes variance from advantage estimates
3. Proper theory-exact settings

---

## Commands Reference

### Run experiments (from any directory)
```bash
python3 scripts/run_contraction_sgd_tradeoff.py --seed 42
```

### Check running experiments
```bash
ps aux | grep upi_trm | grep -v grep
nvidia-smi --query-gpu=index,utilization.gpu --format=csv
```

### Parse results
```bash
python3 -c "
from scripts.run_contraction_sgd_tradeoff import parse_results
import json
result = parse_results('path/to/log.log')
print(json.dumps(result, indent=2))
"
```

---

## Commit History (This Session)

```
b9968dc Fix multi-config loading bug and add diagnostic tools
```

Previous commits for context:
```
39f8e06 f
02a8c00 Add contraction vs SGD tradeoff experiment (trivial 4x4, inconclusive)
afcacda Fix 6-8 empties plots to match trivial plot paper style
```

---

## Files to Check

- `/tmp/quick_test_500.log` - Current verification test output
- `results/plot_data_contraction_sgd_sudoku-4x4-easy_6to8empties/` - Failed experiment logs (for reference)
- `results/plots_contraction_sgd_sudoku-4x4-easy_6to8empties/value_collapse_diagnostic.csv` - Diagnostic data showing collapse
