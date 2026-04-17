# Exp1 Lipschitz/Projection Diagnostic: Provenance

**Generated**: 2026-01-18T11:10:21.197736

## Checkpoints

| Condition | Seed | Checkpoint |
|-----------|------|------------|
| No Contraction | 41 | `checkpoints/exp1_v4/model_a_prime_seed41.pt` |
| No Contraction | 42 | `checkpoints/exp1_v4/model_a_prime_seed42.pt` |
| No Contraction | 43 | `checkpoints/exp1_v4/model_a_prime_seed43.pt` |
| Contraction | 41 | `checkpoints/exp1_v4/model_b_seed41.pt` |
| Contraction | 42 | `checkpoints/exp1_v4/model_b_seed42.pt` |
| Contraction | 43 | `checkpoints/exp1_v4/model_b_seed43.pt` |

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Eval radii | [10.0, 100.0] |
| n_train | 2 |
| Batch | B0 |
| Samples per checkpoint | 50 |
| Perturbations per sample | 5 |

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp1_lipschitz_diag
```
