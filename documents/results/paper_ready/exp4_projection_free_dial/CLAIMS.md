# Exp4 Claims: Projection-free Contraction Dial (Range Test)

**Generated:** 2026-01-18T14:48:54.936113

## Summary

- **Decision:** INCONCLUSIVE: Dial has range but no clear monotonicity
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range ≥0.10):** PASS (spread=0.4288)
- **G3 (Monotonic linkage):** INCONCLUSIVE (ρ=0.2000)

## Results by Scaling Factor

| Scale | L_preproj | ΔV@8x | Argmax@8x |
|-------|-----------|-------|-----------|
| 1.00 | 0.9483 | 0.5663 | 0.6600 |
| 0.90 | 0.7192 | 0.4918 | 0.6600 |
| 0.80 | 0.5876 | 0.4174 | 0.7000 |
| 0.70 | 0.5195 | 0.4389 | 0.7400 |
| 0.60 | 0.5212 | 0.6385 | 0.6900 |

## Scoped Claim

Contraction scaling produces L_preproj spread ≥0.10 but does not show clear monotonic relationship with stability metrics.

## Scope Limitations

- Results from inference-time scaling only (no retraining)
- Single checkpoint (nc_rdis_s42) from Exp3
- Trivial 4×4 Sudoku suite
- 100 evaluation samples

## Proceed to Full Sweep?

YES - G2 passed, dial range is sufficient
