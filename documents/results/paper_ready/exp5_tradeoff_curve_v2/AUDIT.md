# Exp5 Audit Report

**Audit Date:** 2026-01-19T16:42:06.502978
**Summary File:** Exp5_tradeoff_curve
**Generated At:** 2026-01-19T16:40:12.981900
**Git SHA:** b4973924040d

## Audit Result

**Status:** ✅ PASSED
**Checks Passed:** 9/9
**Warnings:** 0

## Check Details

| # | Check | Status | Result |
|---|-------|--------|--------|
| 1. Value-head norm OFF | ✓ | PASS: disable_value_head_norm=true |
| 2. Projection disabled | ✓ | PASS: latent_ball_radius=0.0 (projection disabled) |
| 3. Projection inactive | ✓ | PASS: max projection_active_rate=0.0000 (<1%) |
| 4. Git SHA captured | ✓ | PASS: git_sha=b4973924040d |
| 5. Multiple checkpoints | ✓ | PASS: 3 checkpoints with seeds [41, 42, 43] |
| 6. Dial scale coverage | ✓ | PASS: 4 scales: [1.0, 0.85, 0.7, 0.55] |
| 7. Decision gates | ✓ | PASS: G0=True, G1=True, G2=True, G3=False |
| 8. Scale summaries complete | ✓ | PASS: 4 scale summaries with required fields |
| 9. Success variation | ✓ | PASS: Success rate varies (range=0.0033) |


## Decision Gates

| Gate | Status | Details |
|------|--------|---------|
| G0 (Projection inactive) | PASS | latent_ball_radius=0 |
| G1 (Stability) | PASS | No NaN values |
| G2 (Dial range) | PASS | spread=0.6532 |
| G3 (Tradeoff exists) | FAIL | success_range=0.0033 |

## Non-Negotiables Verified

- `disable_value_head_norm: true` (value-head spectral norm OFF)
- `latent_ball_radius: 0.0` (projection disabled)

