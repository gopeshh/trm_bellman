# Exp3 Claims: Training-Time Projection Ablation

**Generated:** 2026-01-18T13:29:30.622926

## Summary

- **G1 (Stability):** 6/6 conditions passed

## Results by Condition

| Condition | Contraction | R | Success | NaN | G1 |
|-----------|-------------|---|---------|-----|-----|
| nc_r10 | False | 10.0 | 0.940 | NO | PASS |
| nc_r100 | False | 100.0 | 0.960 | NO | PASS |
| nc_rdis | False | 0.0 | 0.960 | NO | PASS |
| c_r10 | True | 10.0 | 0.980 | NO | PASS |
| c_r100 | True | 100.0 | 0.960 | NO | PASS |
| c_rdis | True | 0.0 | 0.960 | NO | PASS |

## Scoped Claims

**Claim 2 (Positive):** Training without projection (R=disabled) is stable for conditions: nc_rdis, c_rdis

## Scope Limitations

- Results on trivial 4×4 Sudoku only
- Single seed (42) per condition
- 5000 training steps
