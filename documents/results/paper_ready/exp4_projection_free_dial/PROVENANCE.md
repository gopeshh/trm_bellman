# Exp4 Provenance: Range Test

**Generated:** 2026-01-18T14:48:54.936113
**Git SHA:** None

## Checkpoint

- **Path:** /home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt
- **Config source:** inferred

## Evaluation Parameters

- **Scaling factors:** [1.0, 0.9, 0.8, 0.7, 0.6]
- **Num samples:** 100
- **Seed:** 42
- **Device:** cuda

## Gate Thresholds

- **G2 (Dial range):** L_preproj spread ≥ 0.10
- **G3 (Monotonic linkage):** |Spearman ρ| > 0.5

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp4_range_test -- \
    --checkpoint /home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt \
    --out_dir /home/buiksat/trm_bellman/results/paper_ready/exp4_projection_free_dial
```
