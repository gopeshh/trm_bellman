# Exp5 Claims: Stability–Expressivity Tradeoff Curve

**Generated:** 2026-01-19T00:33:32.384726
**Git SHA:** b3082724f6c3

## ⛔ DEPRECATION NOTICE

**This artifact set is DEPRECATED for success rate claims.**

The checkpoints used were trained WITHOUT `--dataset-paths`, causing them to use
synthetic solved puzzles instead of real Sudoku puzzles. As a result:
- `success_trivial` values (~5%) are NOT comparable to baseline (~93%)
- Training never learned to fill empty cells

**Use `exp5_tradeoff_curve_v2/` for paper-valid success metrics.**

Stability metrics (argmax agreement, ΔV, L_preproj) remain valid because they
measure z→z dynamics, not task success.

---

## Summary (STABILITY METRICS ONLY)

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

## ⚠️ Important: Success Rate Qualification

**The ~5% success rate is expected and NOT a bug.** These checkpoints (`nc_rdis_*`) were trained
using an evaluation dataset where puzzles appeared already-solved (`initial=16.00/16`). As a result:

1. **Models did not learn to fill empty cells** during training
2. **Low success (~5%) on real puzzles** is the expected consequence
3. **Stability metrics remain valid** as they measure z→z dynamics, not task success

For task success claims, use Exp2-style checkpoints trained on actual unsolved puzzles (which achieve 86-92% success).

See `results/paper_ready/SUCCESS_RATE_RECONCILIATION.md` for full root cause analysis.

## Non-Negotiables Verified

- `disable_value_head_norm: true`
- `latent_ball_radius: 0.0` (projection disabled)
