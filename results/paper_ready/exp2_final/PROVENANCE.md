# Exp2 Final: Provenance

**Generated**: 2026-01-18T09:56:16.469398

## Commit References

| Experiment | Commit | Description |
|------------|--------|-------------|
| Exp2 | a460efd | Initial contraction sweep |
| Exp2b | 0d32097 | Projection dominance diagnostics |
| Exp2c | db3de70 | Dial unmasking evaluation |

## Checkpoint Paths

All checkpoints from Exp2 contraction sweep:

```
checkpoints/exp2_contraction_sweep/
├── lz_0900/seed{41,42,43}/model_step_5000.pt
├── lz_095/seed{41,42,43}/model_step_5000.pt
├── lz_099/seed{41,42,43}/model_step_5000.pt
└── lz_0999/seed{41,42,43}/model_step_5000.pt
```

## Source Data Files

| File | Description |
|------|-------------|
| `results/paper_ready/exp2c/DIAGNOSTICS_exp2c_lite.json` | Exp2c evaluation results (primary source) |

## Regeneration Commands

```bash
# Generate figures and tables
buck2 run //buiksat_trm:make_paper_figures_exp2_final

# Run audit
buck2 run //buiksat_trm:audit_exp2_final_paper_ready
```

## Configuration

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n_eval | 16 (deepest mismatch) |
| Eval radii | [10.0, 100.0, disabled (0.0)] |
| Batch | B0 (initial states) |
| Seeds | [41, 42, 43] per target $L_z$ |
| target $L_z$ values | [0.9, 0.95, 0.99, 0.999] |

## Key Configuration Notes

- `disable_value_head_norm: true` - Value-head spectral norm OFF (prevents collapse)
- `use_feasibility_checker: true` - Standard checker
- `episodic_latent: true` - Reset latent at episode boundaries
