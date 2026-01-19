# Exp4 v2 Audit Report

**Audit Date:** 2026-01-18T21:47:07.700495
**Summary File:** Exp4_final_v2
**Generated At:** 2026-01-18T21:46:46.262096
**Git SHA:** 6fede631455e

## Audit Result

**Status:** ✅ PASSED
**Checks Passed:** 11/11
**Warnings:** 0

## Check Details

| # | Check | Status | Result |
|---|-------|--------|--------|
| 1. Multi-seed independence | ✓ | PASS: 3 checkpoints with seeds [41, 42, 43] |
| 2. Dial scale coverage | ✓ | PASS: 4 scales: [1.0, 0.85, 0.7, 0.55] |
| 3. Non-negotiable compliance | ✓ | PASS: latent_ball_radius=0.0, disable_value_head_norm=True |
| 4. Sample count (N >= 12) | ✓ | PASS: N=12 (3 checkpoints × 4 scales) |
| 5. Pseudo-replication check | ✓ | PASS: Metrics vary across checkpoints (4/4 scales) |
| 6. Anti-degenerate (entropy) | ✓ | PASS: mean_entropy=2.667, min_entropy=2.462 |
| 7. Batch provenance | ✓ | PASS: B0=100 (hash:9ceab78310f3), B1=1048 (hash:fca64be3b53c) |
| 8. Git SHA tracking | ✓ | PASS: git_sha=6fede631455e |
| 9. Decision gates | ✓ | PASS: G0=True, G1=True, G2=True, G3=PASS |
| 10. Statistical method | ✓ | PASS: cluster_bootstrap (1000 iterations, 3 clusters) |
| 11. Statistical reporting | ✓ | PASS: ρ=-0.866 [-1.000, -0.542] (cluster bootstrap, 1000 iter, 3 clusters) |


## Statistical Methodology

- **Method:** cluster_bootstrap
- **N Observations:** 12
- **N Clusters (checkpoints):** 3
- **Bootstrap Iterations:** 1000
- **Rationale:** Scales are repeated measures on the same checkpoint; cluster bootstrap accounts for within-checkpoint correlation

## Decision Gates

| Gate | Status | Details |
|------|--------|---------|
| G0 (Projection inactive) | PASS | latent_ball_radius=0 |
| G1 (Stability) | PASS | No NaN values |
| G2 (Dial range) | PASS | spread=0.6952 (threshold ≥0.10) |
| G3 (Monotonicity) | PASS | |ρ|>0.5, 95% CI excludes 0 |

## Depth Mismatch Notation

Metrics labeled "(n2=8)" refer to evaluation at depth n2=8, which is **4× the training depth** (n_train=2).
This is NOT "8× mismatch" - the mismatch multiplier is n2/n_train = 8/2 = 4.

## Non-Negotiables Verified

- `disable_value_head_norm: true` (value-head spectral norm OFF)
- `latent_ball_radius: 0.0` (projection disabled)
