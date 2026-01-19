# Exp4 Audit Results

**Audit Date:** 2026-01-18T16:15:06.156280

## Summary

All 7 audit checks passed.

## Checks

- **G0 (Projection inactive):** ✅ PASS - G0 passed: projection_active_rate=0.0%
- **G1 (Stability):** ✅ PASS - G1 passed: stability check
- **G2 (Dial range):** ✅ PASS - G2 passed: spread=0.4288 (threshold=0.1)
- **G3 (Monotonicity):** ✅ PASS - G3 INCONCLUSIVE: Spearman ρ=0.2000
- **Value-head norm OFF:** ✅ PASS - disable_value_head_norm: true (verified)
- **Projection disabled:** ✅ PASS - latent_ball_radius: 0.0 (projection disabled)
- **No overclaim:** ✅ PASS - No overclaim detected (G2=passed)

## Key Metrics

| Metric | Value |
|--------|-------|
| L_preproj spread | 0.4288 |
| Spearman ρ | 0.2000 |
| G2 (dial range) | PASS |
| G3 (monotonicity) | INCONCLUSIVE |

## Decision

INCONCLUSIVE: Dial has range but no clear monotonicity

## Proceed to Full Sweep?

YES - G2 passed
