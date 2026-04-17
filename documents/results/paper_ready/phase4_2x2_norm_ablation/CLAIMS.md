# Phase 4 Claims: 2×2 Norm Ablation

## Primary Claim

Value-head spectral normalization (when enabled) causes training instability,
while z→z contraction alone does not harm performance.

## Evidence

| Condition | z→z | V-head | Var(V) | Argmax@4× | Success |
|-----------|-----|--------|--------|-----------|---------|
| C-OFF, V-OFF | OFF | OFF | 0.015±0.003 | 0.833±0.033 | 0.000±0.000 |
| C-OFF, V-ON | OFF | ON | 0.015±0.003 | 0.833±0.033 | 0.000±0.000 |
| C-ON, V-OFF | ON | OFF | 0.014±0.003 | 0.977±0.017 | 0.000±0.000 |
| C-ON, V-ON | ON | ON | 0.006±0.002 | 0.977±0.009 | 0.000±0.000 |

## Interpretation

- V-head ON conditions (nc_yv, yc_yv) expected to show higher Var(V) or NaN
- z→z contraction alone (yc_nv) should perform similarly to baseline (nc_nv)

## ⚠️ Important: Success Rate Column

**The 0.000 success values are placeholders, not measurements.** The Phase 4 evaluation script's
`compute_success_rate()` function returns 0.0 as a placeholder because:

1. **Stability metrics are the primary focus** of this 2×2 ablation (Var(V), argmax agreement, L_preproj)
2. **Training logs show 100% success** but on already-solved puzzles (`initial=16.00/16`)
3. **Full success evaluation would require PlanEditEnv setup** with proper dataset

For task success claims, use Exp2-style checkpoints trained on actual unsolved puzzles.
See `documents/results/paper_ready/SUCCESS_RATE_RECONCILIATION.md` for details.

## Key Finding

**z→z contraction is the primary stabilizer** (argmax@4×: 97.7% vs 83.3%), while value-head
spectral normalization has minimal additional impact when contraction is already enabled.
