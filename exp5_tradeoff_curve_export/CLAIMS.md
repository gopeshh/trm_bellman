# Exp5 Claims: Stability–Expressivity Tradeoff Curve

**Generated:** 2026-01-19T00:33:32.384726
**Git SHA:** b3082724f6c3

## Summary

- **Decision:** POSITIVE: Tradeoff curve generated successfully
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range):** PASS (spread=0.6560)
- **G3 (Tradeoff exists):** PASS

## Tradeoff Results

| Scale | L_preproj | Stability (argmax@8×) | Success (trivial) | ΔV@8× |
|-------|-----------|----------------------|-------------------|-------|
| 1.00 | 0.457±0.308 | 0.903±0.101 | 0.047±0.017 | 0.266 |
| 0.85 | 0.377±0.164 | 0.887±0.083 | 0.050±0.022 | 0.185 |
| 0.70 | 0.388±0.079 | 0.873±0.086 | 0.053±0.019 | 0.496 |
| 0.55 | 0.488±0.033 | 0.817±0.079 | 0.057±0.017 | 1.343 |

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
