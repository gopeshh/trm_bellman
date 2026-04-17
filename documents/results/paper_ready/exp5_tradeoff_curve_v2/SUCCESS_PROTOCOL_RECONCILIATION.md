# SUCCESS_PROTOCOL_RECONCILIATION.md

**Date:** 2026-01-19
**Git SHA:** b828635

## Executive Summary

The observed discrepancy between training success (~88%) and Exp5 success (~6-7%) is **NOT a bug**. It reflects fundamentally different evaluation protocols designed for different purposes.

## Discrepancy Summary

| Metric | Training Evaluation | Exp5 Evaluation |
|--------|---------------------|-----------------|
| Success Rate | ~88% | ~6-7% |
| Unroll Depth (n_eval) | 4 (matched to n_train=2 × 2) | 2 (matched compute) |
| Projection (R) | 10.0 | 0.0 (disabled) |
| Purpose | Training quality monitoring | Stability-expressivity tradeoff |

## Root Cause Analysis

### Training Evaluation Protocol
- **Setting:** n_eval=4, R=10 (projection enabled)
- **Purpose:** Monitor whether the model is learning to solve puzzles
- **Result:** 88-98% success rate indicates the model learns well
- **Log evidence:** `eval_success_rate=0.880...initial=13.52` in training logs

### Exp5 Evaluation Protocol
- **Setting:** n_eval=2 (matched compute), R=0 (projection disabled)
- **Purpose:** Measure success at matched compute budget for tradeoff curve
- **Result:** 6-7% success rate at matched depth
- **Interpretation:** Model's policy at n=2 alone doesn't solve puzzles

## Why the Difference?

1. **Unroll Depth Impact:**
   - n=2 only gives the model 2 reasoning iterations
   - n=4 gives 4 iterations, allowing more refinement
   - Trivial puzzles (1-4 empties) typically need 2-4 correct edits

2. **Projection Impact:**
   - R=10 constrains latent norms, providing implicit regularization
   - R=0 allows unbounded latents, which may lead to instability

3. **Matched vs. Extended Compute:**
   - Exp5 measures "what can the model do with MATCHED compute?"
   - Training eval measures "what can the model do with SUFFICIENT compute?"

## Checkpoints Verified

| Path | Status | Training Success |
|------|--------|------------------|
| `results/exp5_v2_inputs_eval/nc_rdis_s41/model_step_5000.pt` | ✓ | 88% |
| `results/exp5_v2_inputs_eval/nc_rdis_s42/model_step_5000.pt` | ✓ | 98% |
| `results/exp5_v2_inputs_eval/nc_rdis_s43/model_step_5000.pt` | ✓ | 92% |

All checkpoints were trained with:
- Dataset: `sudoku-4x4-trivial` (explicit --dataset-paths)
- Initial score: ~13.52/16 (unsolved puzzles, NOT solved)
- Config: `configs/exp3_projection_ablation/nc_rdis.yaml`

## Conclusion

The Exp5 results are **valid and informative**, but measure a different quantity than training success:

- **Training success** = "Can the model solve puzzles given sufficient compute?" (Answer: YES, 88-98%)
- **Exp5 success** = "Can the model solve puzzles at matched compute with R=0?" (Answer: NO, 6-7%)

This confirms that the Exp5 tradeoff curve shows a valid (if flat) relationship at the matched-compute setting. The claim "no tradeoff observed" (G3 FAIL) is accurate: varying the contraction scale from 1.0 to 0.55 at n=2, R=0 does not significantly change success rate.

## Recommendations

1. **Update CLAIMS.md:** Clarify that success is measured at matched compute (n=2, R=0), not extended compute
2. **Update paper text:** Ensure readers understand that 6-7% success at matched compute does not contradict ~90% success at extended compute
3. **Add note about compute tradeoff:** Higher n_eval → higher success, but this is a separate dimension from the stability dial

## Regeneration Commands

```bash
# Training (achieved 88-98% success)
buck2 run //buiksat_trm:upi_trm_train -- \
    --config configs/exp3_projection_ablation/nc_rdis.yaml \
    --seed 41 \
    --dataset-paths data/sudoku-4x4-trivial

# Exp5 Evaluation (shows 6-7% success at matched compute)
buck2 run //buiksat_trm:exp5_tradeoff_curve -- \
    --checkpoints results/exp5_v2_inputs_eval/nc_rdis_s41/model_step_5000.pt \
    --config_yaml configs/exp3_projection_ablation/nc_rdis.yaml
```
