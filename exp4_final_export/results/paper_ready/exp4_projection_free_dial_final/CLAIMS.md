# Exp4 Claims: Projection-free Contraction Dial (Final)

**Generated:** 2026-01-18T17:30:11.650029
**Git SHA:** None

## Summary

- **Decision:** POSITIVE: Dial viable with partial monotonicity
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range ≥0.10):** PASS (spread=0.4125)
- **G3 (Monotonicity):** PASS
  - ΔV@8x: ρ=0.0000 (p=1.0000)
  - argmax@8x: ρ=-0.7773 (p=0.0029)

## Results by Scale (Pooled Across Seeds)

| Scale | L_preproj | B0 ΔV@8x | B0 argmax@8x | B1 ΔV@8x | B1 argmax@8x |
|-------|-----------|----------|--------------|----------|--------------|
| 1.00 | 0.923±0.006 | 0.608±0.000 | 0.700±0.000 | 0.797±0.000 | 0.638±0.000 |
| 0.85 | 0.621±0.002 | 0.448±0.000 | 0.660±0.000 | 0.579±0.000 | 0.674±0.000 |
| 0.70 | 0.510±0.001 | 0.503±0.000 | 0.720±0.000 | 0.475±0.000 | 0.701±0.000 |
| 0.55 | 0.542±0.001 | 0.787±0.000 | 0.710±0.000 | 0.707±0.000 | 0.664±0.000 |

## Scoped Claims

**Claim (Positive):** Inference-time contraction scaling on z→z layers produces a measurable,
controllable "dial" for the achieved Lipschitz constant (L_preproj) when projection is disabled.

**Claim (Monotonicity - argmax):** L_preproj is negatively associated with argmax agreement
on B0 evaluation states, with Spearman ρ=-0.777 (p=0.0029).

## Scope Limitations

- Results from inference-time scaling only (no retraining)
- Base checkpoint: /home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt
- Trivial 4×4 Sudoku suite only
- Seeds: [41, 42, 43]
- Training depth n_train=2, eval depths: [4, 8, 16]

## Evaluation Batches

- **B0:** 100 initial states (hash: 9ceab78310f3)
- **B1:** 789 successor states (hash: 4a3bd6f6fd5b)
- B1 constructed using union of top-5 actions across ALL dial scales (avoids selection bias)
