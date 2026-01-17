# Experiment 1 Provenance

**Generated**: 2026-01-17T00:07:10.295172
**Git Commit**: 35b3d8ad2db7cb74572b1c8b76ebad7cfd75698b
**Python Version**: 3.12.12+meta
**Torch Version**: not available

## Input CSVs

| Purpose | Path |
|---------|------|
| Unroll B0 | `results/tables/unroll_sensitivity_b0_mismatch.csv` |
| Unroll B1 | `results/tables/unroll_sensitivity_b1_mismatch.csv` |
| Radius B0 | `results/tables/radius_sweep_b0_aggregated.csv` |
| Radius B1 | `results/tables/radius_sweep_b1_aggregated.csv` |

## Checkpoints

| Seed | No Contraction | Contraction |
|------|----------------|-------------|
| 41 | `checkpoints/exp1_v4/model_a_prime_seed41.pt` | `checkpoints/exp1_v4/model_b_seed41.pt` |
| 42 | `checkpoints/exp1_v4/model_a_prime_seed42.pt` | `checkpoints/exp1_v4/model_b_seed42.pt` |
| 43 | `checkpoints/exp1_v4/model_a_prime_seed43.pt` | `checkpoints/exp1_v4/model_b_seed43.pt` |

## YAML Configs

- No Contraction: `configs/ablations/upi_trm_feasibility_no_contraction.yaml`
- Contraction: `configs/ablations/upi_trm_feasibility_contraction.yaml`

## Key Parameters

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n2 (eval depths) | [4, 8, 16] |
| Radii | [0.0, 10.0, 100.0] |
| Seeds | [41, 42, 43] |
| disable_value_head_norm | true |
| target_Lz (contraction) | 0.9 |
| latent_ball_radius | 10.0 |

## Regeneration Commands

```bash
# One-command regeneration
cd /data/repos/fbsource/fbcode
buck2 run //buiksat_trm:make_paper_figures_exp1_final

# Or directly with Python (from trm_bellman root)
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

### Documentation
- `CLAIMS.md` - Scoped claims with evidence
- `PROVENANCE.md` - This file
- `AUDIT.md` - Validation report
