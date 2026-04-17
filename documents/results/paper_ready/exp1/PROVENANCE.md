# Experiment 1 Provenance

**Generated**: 2026-04-12T10:12:20.914275
**Commit**: 521bfc1

## Protocol Note

- This bundle uses the refrozen replacement protocol under `exp1_v4_refreeze`.
- Historical `exp1_v4` checkpoints and frozen batches were not recoverable on this machine.
- Replacement `B0/B1` preserve the intended composition semantics, but are not byte-identical to the historical artifacts.
## Input CSVs

| Purpose | Path |
|---------|------|
| Unroll B0 | `results/tables/unroll_sensitivity_b0_mismatch.csv` |
| Unroll B1 | `results/tables/unroll_sensitivity_b1_mismatch.csv` |
| Radius B0 | `results/tables/radius_sweep_b0_aggregated.csv` |
| Radius B1 | `results/tables/radius_sweep_b1_aggregated.csv` |

## Refreeze Sources

- Results root: `/home/buiksat/trm_bellman/results/validation/exp1_v4_refreeze`
- Checkpoints: `/home/buiksat/trm_bellman/checkpoints/exp1_v4_refreeze`
- Frozen batches: `/home/buiksat/trm_bellman/artifacts/eval_batches/exp1_v4_refreeze`

## Checkpoints

| Seed | No Contraction | Contraction |
|------|----------------|-------------|
| 41 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed41/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed41/model_step_5000.pt` |
| 42 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed42/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed42/model_step_5000.pt` |
| 43 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed43/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed43/model_step_5000.pt` |
| 44 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed44/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed44/model_step_5000.pt` |
| 45 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed45/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed45/model_step_5000.pt` |
| 46 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed46/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed46/model_step_5000.pt` |
| 47 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed47/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed47/model_step_5000.pt` |
| 48 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed48/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed48/model_step_5000.pt` |
| 49 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed49/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed49/model_step_5000.pt` |
| 50 | `checkpoints/exp1_v4_refreeze/model_a_prime/seed50/model_step_5000.pt` | `checkpoints/exp1_v4_refreeze/model_b/seed50/model_step_5000.pt` |

## YAML Configs

- No Contraction: `configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml`
- Contraction: `configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml`

## Key Parameters

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n2 (eval depths) | [4, 8, 16] |
| main unroll comparison | 2→8 |
| Radii | [0.0, 10.0, 100.0] |
| Seeds | [41, 42, 43, 44, 45, 46, 47, 48, 49, 50] |
| disable_value_head_norm | true |
| target_Lz (contraction) | 0.9 |
| projection at R=10 | active 100% on B0/B1 |

## Regeneration Commands

```bash
# Refresh paper-facing CSVs from the refrozen eval outputs
python scripts/postprocess_exp1_v4_1.py --results_dir /home/buiksat/trm_bellman/results/validation/exp1_v4_refreeze --out_dir /home/buiksat/trm_bellman/results/tables --seeds 41,42,43,44,45,46,47,48,49,50 --radii 10,100,0 --n_train 2 --radius_n2 8

# Generate all paper-ready artifacts
python scripts/make_paper_figures_exp1_final.py

# Full audit
python scripts/audit_exp1_paper_ready.py
```

## Output Artifacts

### Main Paper
- `fig_exp1_unroll_sensitivity_main.pdf` (B0, 1×3)
- `fig_exp1_radius_sweep_main.pdf` (B0, 1×3)
- `table_exp1_unroll_sensitivity.tex`
- `table_exp1_radius_sweep_main.tex`

### Appendix
- `fig_exp1_unroll_sensitivity_appendix.pdf` (B1, 1×3)
- `fig_exp1_radius_sweep_appendix.pdf` (B1, 1×3)
- `table_exp1_radius_sweep_appendix.tex`
