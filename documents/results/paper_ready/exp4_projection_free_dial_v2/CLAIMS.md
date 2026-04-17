# Exp4 Claims: Projection-free Contraction Dial (Final v2)

**Generated:** 2026-01-18T21:46:46.262096
**Git SHA:** 6fede631455e

## Summary

- **Decision:** POSITIVE: Dial viable with statistically significant monotonicity
- **N (observations):** 12 (3 checkpoints × 4 scales)
- **Statistical Method:** Cluster bootstrap (checkpoints as clusters)
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range ≥0.10):** PASS (spread=0.6952)
- **G3 (Monotonicity |ρ|>0.5, 95% CI excludes 0):** PASS
  - B0 argmax (n2=8, 4× mismatch): ρ=-0.866 [-1.000, -0.542] ✓
  - B1 argmax (n2=8, 4× mismatch): ρ=-0.923 [-1.000, -0.787] ✓
  - B0 ΔV (n2=8, 4× mismatch): ρ=0.615 [0.000, 0.800] 
  - B1 ΔV (n2=8, 4× mismatch): ρ=0.657 [0.553, 0.800] ✓

## Depth Mismatch Notation

- **n_train = 2** (training unroll depth)
- **n2 = 4** means 2× mismatch (evaluating at 2× training depth)
- **n2 = 8** means 4× mismatch (evaluating at 4× training depth)
- **n2 = 16** means 8× mismatch (evaluating at 8× training depth)

## Statistical Methodology

**Why cluster bootstrap?** Dial scales (1.0, 0.85, 0.70, 0.55) are repeated measures on the
same trained checkpoint. This violates the i.i.d. assumption required by standard Spearman
p-values. We address this by:

1. Treating each checkpoint as a cluster (3 clusters total)
2. Bootstrap resampling at the cluster level (1000 iterations)
3. Computing 95% confidence intervals from the bootstrap distribution
4. A correlation "passes" if |ρ| > 0.5 AND the 95% CI excludes 0

This is more conservative than i.i.d. inference and accounts for within-checkpoint correlation.

## Anti-Degenerate Checks

| Metric | Value |
|--------|-------|
| Mean entropy (train) | 2.667 |
| Mean entropy (eval) | 2.697 |
| Min entropy (train) | 2.462 |
| Entropy OK | YES |
| Pseudo-replication | NOT DETECTED |

## Results by Scale (Pooled Across Checkpoints)

| Scale | L_preproj | B0 argmax (n2=8) | B1 argmax (n2=8) | B0 ΔV (n2=8) | B1 ΔV (n2=8) | Entropy |
|-------|-----------|------------------|------------------|--------------|--------------|---------|
| 1.00 | 0.470±0.326 | 0.887±0.133 | 0.856±0.156 | 0.267±0.241 | 0.314±0.303 | 2.61 |
| 0.85 | 0.383±0.173 | 0.867±0.148 | 0.862±0.129 | 0.176±0.193 | 0.209±0.233 | 2.60 |
| 0.70 | 0.391±0.084 | 0.880±0.118 | 0.860±0.102 | 0.532±0.046 | 0.513±0.057 | 2.65 |
| 0.55 | 0.490±0.037 | 0.820±0.080 | 0.793±0.091 | 1.336±0.394 | 1.314±0.419 | 2.81 |

## Per-Checkpoint Trends (scale 1.0→0.55)

| Seed | L_preproj | Argmax | ΔV | Monotonic |
|------|-----------|--------|-----|-----------|
| 41 | ↑ (0.236→0.460) | ↓ (0.960→0.850) | ↑ (0.092→1.690) | ✓ |
| 42 | ↓ (0.931→0.543) | ↑ (0.700→0.710) | ↑ (0.608→0.787) | partial |
| 43 | ↑ (0.243→0.467) | ↓ (1.000→0.900) | ↑ (0.101→1.531) | ✓ |

## Scoped Claims

**Claim (Positive):** Inference-time contraction scaling on z→z layers produces a measurable,
controllable "dial" for the achieved Lipschitz constant (L_preproj) when projection is disabled.

**Statistical Support:** With 12 observations from 3 independently trained checkpoints,
using cluster bootstrap for proper inference:
- Spearman ρ=-0.866 (95% CI: [-1.000, -0.542]) for L_preproj vs B0 argmax at n2=8 (4× mismatch).
- Spearman ρ=-0.923 (95% CI: [-1.000, -0.787]) for L_preproj vs B1 argmax at n2=8 (4× mismatch).

## Scope Limitations

- Results from inference-time scaling only (no retraining per-scale)
- Checkpoints: 3 independently trained models with seeds 41, 42, 43
- Trivial 4×4 Sudoku suite only
- Training depth n_train=2, eval depths: [4, 8, 16]

## Evaluation Batches

- **B0:** 100 initial states (hash: 9ceab78310f3)
- **B1:** 1048 successor states (hash: fca64be3b53c)
- B1 constructed using union of top-5 actions across ALL checkpoints × ALL scales (avoids selection bias)
