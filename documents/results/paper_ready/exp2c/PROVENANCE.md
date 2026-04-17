# Exp2c-Lite Provenance

**Generated**: 2026-01-17
**Script**: `scripts/eval_exp2c_lite.py`

## Purpose

Test whether the contraction "dial" works when projection is not dominating by evaluating existing Exp2 checkpoints at different projection radii.

## Checkpoints Evaluated

All 12 checkpoints from Exp2 contraction sweep (4 target_Lz × 3 seeds):
- target_Lz ∈ {0.9, 0.95, 0.99, 0.999}
- seeds ∈ {41, 42, 43}
- Base path: `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/`

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n_eval depths | [4, 8, 16] |
| Eval radii | [10, 100, disabled] |
| Batch | B0 (100 initial states) |

## Decision Gates

| Gate | Threshold | Result |
|------|-----------|--------|
| G1 (Projection Dominance) | <20% at R≥100 | ✅ PASSED |
| G2 (Dial Range) | L_preproj spread ≥0.08 | ❌ FAILED (0.064) |
| G3 (Stability Linkage) | Correlation with stability | INCONCLUSIVE |

## Path Decision

**Path B (Negative Result)**: The spectral-norm dial has insufficient range to provide controllable contraction.

## Key Finding

Projection at R=10 provides substantial stability benefit:
- With projection: ΔV = 0.5-2.2, argmax agree = 97-99%
- Without projection: ΔV = 7.7-14.8, argmax agree = 88-91%

## Regeneration Command

```bash
buck2 run //buiksat_trm:eval_exp2c_lite
```

## Output Files

- `DIAGNOSTICS_exp2c_lite.json`: Full results with all metrics
- `DIAGNOSTICS_exp2c_lite.md`: Summary table by eval radius
- `GATES.md`: Decision gate analysis and recommendations
