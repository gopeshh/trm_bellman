# Experiment 2 Provenance

**Generated**: 2026-01-17T15:21:07.643874
**Git Commit**: 8c6994a44bbe7204d1c883b9e93a3c4b1f2c3cd9

## Sweep Configuration

| Target $L_z$ | Config | Seeds |
|--------------|--------|-------|
| 0.9 | `configs/exp2_contraction_sweep/target_lz_090.yaml` | [41, 42, 43] |
| 0.95 | `configs/exp2_contraction_sweep/target_lz_095.yaml` | [41, 42, 43] |
| 0.99 | `configs/exp2_contraction_sweep/target_lz_099.yaml` | [41, 42, 43] |
| 0.999 | `configs/exp2_contraction_sweep/target_lz_0999.yaml` | [41, 42, 43] |

## Checkpoints

### Target $L_z$ = 0.9

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0900/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0900/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0900/seed43/model_step_5000.pt`

### Target $L_z$ = 0.95

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed43/model_step_5000.pt`

### Target $L_z$ = 0.99

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_099/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_099/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_099/seed43/model_step_5000.pt`

### Target $L_z$ = 0.999

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0999/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0999/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0999/seed43/model_step_5000.pt`

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n_eval | 16 (8× depth) |
| Batch | B0 (initial states) |

## Regeneration Command

```bash
buck2 run //buiksat_trm:make_paper_figures_exp2
```
