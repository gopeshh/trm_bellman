# Experiment 1 Provenance

**Generated**: 2026-01-16T17:07:13.477838

## Checkpoints

| Seed | Model A (No Contraction) | Model B (Contraction) |
|------|--------------------------|----------------------|
| 41 | checkpoints/exp1_v4/model_a_prime_seed41.pt | checkpoints/exp1_v4/model_b_seed41.pt |
| 42 | checkpoints/exp1_v4/model_a_prime_seed42.pt | checkpoints/exp1_v4/model_b_seed42.pt |
| 43 | checkpoints/exp1_v4/model_a_prime_seed43.pt | checkpoints/exp1_v4/model_b_seed43.pt |

## YAML Configs

- Model A (No Contraction): `configs/ablations/upi_trm_feasibility_no_contraction.yaml`
- Model B (Contraction): `configs/ablations/upi_trm_feasibility_contraction.yaml`

## Key Config Differences

| Setting | No Contraction | Contraction |
|---------|----------------|-------------|
| enable_contraction | false | true |
| target_Lz | N/A | 0.9 |
| disable_value_head_norm | true | true |
| latent_ball_radius | 10.0 | 10.0 |
| inner_unroll_n | 2 | 2 |

## Batch Artifacts

- B0: `buiksat_trm/results/validation/exp1_v4/seed42/b0.pt` (100 initial states)
- B1: `buiksat_trm/results/validation/exp1_v4/seed42/b1.pt` (successor closure)

## Regeneration Commands

```bash
# Evaluation (already done, per-state CSVs exist)
buck2 run //buiksat_trm:eval_unroll_sensitivity -- compare \
    --checkpoint_a checkpoints/exp1_v4/model_a_prime_seed42.pt \
    --checkpoint_b checkpoints/exp1_v4/model_b_seed42.pt \
    --batch_b0 artifacts/eval_batches/b0.pt \
    --batch_b1 artifacts/eval_batches/b1.pt \
    --n_mults 1,2,4,8

# Paper figures
buck2 run //buiksat_trm:make_paper_figures_exp1 -- \
    --results_dir results/validation/exp1_v4 \
    --out_dir results/paper_ready/exp1
```
