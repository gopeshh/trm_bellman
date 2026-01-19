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
