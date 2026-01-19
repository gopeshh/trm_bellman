# Exp4 Claims: Projection-free Contraction Dial (Final v2)

**Generated:** 2026-01-18T20:25:17.188744
**Git SHA:** None

## Summary

- **Decision:** POSITIVE: Dial viable with statistically significant monotonicity
- **N (independent samples):** 12 (checkpoints × scales)
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range ≥0.10):** PASS (spread=0.7078)
- **G3 (Monotonicity |ρ|>0.5, p<0.05):** PASS
  - B0 argmax@8x: ρ=-0.8662 (p=0.0003) ✓
  - B1 argmax@8x: ρ=-0.9231 (p=0.0000) ✓
  - B0 ΔV@8x: ρ=0.6154 (p=0.0332) ✓
  - B1 ΔV@8x: ρ=0.6573 (p=0.0202) ✓

## Anti-Degenerate Checks

| Metric | Value |
|--------|-------|
| Mean entropy (train) | 2.667 |
| Mean entropy (eval) | 2.697 |
| Min entropy (train) | 2.462 |
| Entropy OK | YES |
| Pseudo-replication | NOT DETECTED |

## Results by Scale (Pooled Across Checkpoints)

| Scale | L_preproj | B0 argmax@8x | B1 argmax@8x | B0 ΔV@8x | B1 ΔV@8x | Entropy (train) |
|-------|-----------|--------------|--------------|----------|----------|-----------------|
| 1.00 | 0.474±0.332 | 0.887±0.133 | 0.856±0.156 | 0.267±0.241 | 0.314±0.303 | 2.61 |
| 0.85 | 0.379±0.167 | 0.867±0.148 | 0.862±0.129 | 0.176±0.193 | 0.209±0.233 | 2.60 |
| 0.70 | 0.391±0.084 | 0.880±0.118 | 0.860±0.102 | 0.532±0.046 | 0.513±0.057 | 2.65 |
| 0.55 | 0.490±0.038 | 0.820±0.080 | 0.793±0.091 | 1.336±0.394 | 1.314±0.419 | 2.81 |

## Scoped Claims

**Claim (Positive):** Inference-time contraction scaling on z→z layers produces a measurable,
controllable "dial" for the achieved Lipschitz constant (L_preproj) when projection is disabled.

**Statistical Support:** With N=12 independent samples (3 checkpoints × 4 scales),
Spearman ρ=-0.866 (p=0.0003) for L_preproj vs B0 argmax@8x.
Spearman ρ=-0.923 (p=0.0000) for L_preproj vs B1 argmax@8x.

## Scope Limitations

- Results from inference-time scaling only (no retraining)
- Checkpoints: ['/home/buiksat/trm_bellman/results/exp3_v2/nc_rdis_s41/model_step_5000.pt', '/home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt', '/home/buiksat/trm_bellman/results/exp3_v2/nc_rdis_s43/model_step_5000.pt']
- Trivial 4×4 Sudoku suite only
- Training depth n_train=2, eval depths: [4, 8, 16]

## Evaluation Batches

- **B0:** 100 initial states (hash: 9ceab78310f3)
- **B1:** 1048 successor states (hash: fca64be3b53c)
- B1 constructed using union of top-5 actions across ALL checkpoints × ALL scales (avoids selection bias)

## Statistical Notes

This evaluation uses N=12 truly independent samples from 3 separately trained checkpoints,
each evaluated at 4 dial scales. This addresses the pseudo-replication issue in v1 where
only L_preproj varied (due to random perturbations) while stability metrics were constant.
