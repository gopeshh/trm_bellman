# Exp4 Final Audit Results

**Audit Date:** 2026-01-18T17:32:25.554091

## Summary

All 9 audit checks passed.

## Checks

- **G0 (Projection inactive):** PASS - G0 passed: projection_active_rate=0.0%
- **G1 (Stability):** PASS - G1 passed: stability check
- **G2 (Dial range):** PASS - G2 passed: spread=0.4125 (threshold=0.1)
- **G3 (Monotonicity):** PASS - G3 PASS: argmax ρ=-0.7773
- **Value-head norm OFF:** PASS - disable_value_head_norm: true (verified)
- **Projection disabled:** PASS - latent_ball_radius: 0.0 (projection disabled)
- **Multi-seed evaluation:** PASS - Multi-seed evaluation verified (3 seeds)
- **Dial scales:** PASS - Dial scales verified (4 scales)
- **No overclaim:** PASS - No overclaim detected (decision=POSITIVE)

## Key Metrics

| Metric | Value |
|--------|-------|
| L_preproj spread | 0.4125 |
| Spearman ρ (argmax) | -0.7773 |
| G2 (dial range) | PASS |
| G3 (monotonicity) | PASS |

## Decision

POSITIVE: Dial viable with partial monotonicity
