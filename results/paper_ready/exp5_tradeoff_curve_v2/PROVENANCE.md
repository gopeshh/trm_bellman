# Exp5 Provenance

**Generated:** 2026-01-19T15:26:08.389973
**Git SHA:** 857adcd13b77

## Checkpoints

| Seed | Path |
|------|------|
| 41 | /data/repos/fbsource/fbcode/buiksat_trm/results/exp5_v2_inputs_eval/nc_rdis_s41/model_step_5000.pt |
| 42 | /data/repos/fbsource/fbcode/buiksat_trm/results/exp5_v2_inputs_eval/nc_rdis_s42/model_step_5000.pt |
| 43 | /data/repos/fbsource/fbcode/buiksat_trm/results/exp5_v2_inputs_eval/nc_rdis_s43/model_step_5000.pt |

## Config

- **YAML:** /data/repos/fbsource/fbcode/buiksat_trm/configs/exp3_projection_ablation/nc_rdis.yaml
- **Non-negotiables:** disable_value_head_norm=true, latent_ball_radius=0.0

## Parameters

- Dial scales: [1.0, 0.85, 0.7, 0.55]
- n_train: 2
- Success episodes per scale: 100
- Max steps per episode: 20

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp5_tradeoff_curve -- \
    --checkpoints /data/repos/fbsource/fbcode/buiksat_trm/results/exp5_v2_inputs_eval/nc_rdis_s41/model_step_5000.pt /data/repos/fbsource/fbcode/buiksat_trm/results/exp5_v2_inputs_eval/nc_rdis_s42/model_step_5000.pt /data/repos/fbsource/fbcode/buiksat_trm/results/exp5_v2_inputs_eval/nc_rdis_s43/model_step_5000.pt \
    --config_yaml /data/repos/fbsource/fbcode/buiksat_trm/configs/exp3_projection_ablation/nc_rdis.yaml \
    --out_dir /data/repos/fbsource/fbcode/buiksat_trm/results/paper_ready/exp5_tradeoff_curve_v2
```
