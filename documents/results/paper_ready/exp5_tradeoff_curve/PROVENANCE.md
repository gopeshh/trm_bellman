# Exp5 Provenance

**Generated:** 2026-01-19T00:33:32.384726
**Git SHA:** b3082724f6c3

## Checkpoints

| Seed | Path |
|------|------|
| 41 | /home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3_v2/nc_rdis_s41/model_step_5000.pt |
| 42 | /home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3/nc_rdis_s42/model_step_5000.pt |
| 43 | /home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3_v2/nc_rdis_s43/model_step_5000.pt |

## Config

- **YAML:** /home/buiksat/fbsource/fbcode/buiksat_trm/configs/exp3_projection_ablation/nc_rdis.yaml
- **Non-negotiables:** disable_value_head_norm=true, latent_ball_radius=0.0

## Parameters

- Dial scales: [1.0, 0.85, 0.7, 0.55]
- n_train: 2
- Success episodes per scale: 100
- Max steps per episode: 20

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp5_tradeoff_curve -- \
    --checkpoints /home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3_v2/nc_rdis_s41/model_step_5000.pt /home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3/nc_rdis_s42/model_step_5000.pt /home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3_v2/nc_rdis_s43/model_step_5000.pt \
    --config_yaml /home/buiksat/fbsource/fbcode/buiksat_trm/configs/exp3_projection_ablation/nc_rdis.yaml \
    --out_dir /home/buiksat/fbsource/fbcode/buiksat_trm/results/paper_ready/exp5_tradeoff_curve
```
