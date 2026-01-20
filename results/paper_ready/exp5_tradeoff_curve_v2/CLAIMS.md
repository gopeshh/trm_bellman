# Exp5 Claims: Stability–Expressivity Tradeoff Curve

**Generated:** 2026-01-19T16:40:12.981900
**Git SHA:** b4973924040d

## Summary

- **Decision:** POSITIVE: Tradeoff curve generated successfully
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range):** PASS (spread=0.6532)
- **G3 (Tradeoff exists):** FAIL

## Tradeoff Results

| Scale | L_preproj | Stability (argmax@8×) | Success (trivial) | ΔV@8× |
|-------|-----------|----------------------|-------------------|-------|
| 1.00 | 0.801±0.238 | 0.700±0.085 | 0.067±0.017 | 0.528 |
| 0.85 | 0.562±0.136 | 0.730±0.057 | 0.070±0.014 | 0.439 |
| 0.70 | 0.489±0.073 | 0.770±0.098 | 0.067±0.017 | 0.458 |
| 0.55 | 0.524±0.031 | 0.730±0.107 | 0.067±0.012 | 0.680 |

## Scoped Claim

**Claim:** Inference-time contraction scaling provides a controllable tradeoff between
stability (policy consistency under depth mismatch) and expressivity (task success rate).

- Higher contraction (lower scale) → higher stability, potentially lower success
- Lower contraction (higher scale) → lower stability, potentially higher success

## Scope Limitations

- Results on 4×4 Sudoku only
- Success measured with greedy policy at matched compute (n=2)
- Stability measured as argmax agreement at n2=8 (4× mismatch)
- 3 independently trained checkpoints

## Non-Negotiables Verified

- `disable_value_head_norm: true`
- `latent_ball_radius: 0.0` (projection disabled)
