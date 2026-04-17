# Unroll Sensitivity Comparison: Model A vs Model B

**Training depth (n_train)**: 2
**Comparison**: n=2 (1×) vs n=8 (4×)

## Configuration Comparison

| Setting | Model A | Model B |
|---------|---------|--------|
| enable_contraction | False | True |
| target_Lz | 0.9 | 0.9 |
| disable_value_head_norm | True | True |
| episodic_latent | True | True |
| latent_ball_radius | 100.0 | 100.0 |
| inner_unroll_n | 2 | 2 |
| config_source | yaml | yaml |

## Projection Sanity Check

- Model A: R = 100.0, max Δ_z bound (2R) = 200.0
- Model B: R = 100.0, max Δ_z bound (2R) = 200.0

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.7575 | 1.5000 | +0.7425 |
| Δ_V | median | 0.5975 | 0.7370 | +0.1396 |
| Δ_V | p99 | 3.5030 | 5.9114 | +2.4084 |
| Δ_pi | mean | 0.0105 | 0.0289 | +0.0184 |
| Δ_pi | median | 0.0055 | 0.0142 | +0.0087 |
| Δ_pi | p99 | 0.0972 | 0.0997 | +0.0025 |
| Δ_z | mean | 40.5250 | 23.7363 | -16.7886 |
| Δ_z | median | 41.4135 | 25.4827 | -15.9308 |
| Δ_z | p99 | 46.9338 | 27.5916 | -19.3422 |
| argmax_agree | rate | 0.4100 | 0.6000 | +0.1900 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 32.1724 | 32.1723 | -0.0000 |
| z_post_norm | mean | 32.1724 | 32.1723 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.7510 | 1.5926 | +0.8415 |
| Δ_V | median | 0.5051 | 0.8644 | +0.3593 |
| Δ_V | p99 | 4.3155 | 6.3896 | +2.0740 |
| Δ_pi | mean | 0.0121 | 0.0323 | +0.0201 |
| Δ_pi | median | 0.0061 | 0.0171 | +0.0110 |
| Δ_pi | p99 | 0.0786 | 0.1055 | +0.0268 |
| Δ_z | mean | 39.3695 | 23.7018 | -15.6677 |
| Δ_z | median | 41.7762 | 25.3402 | -16.4360 |
| Δ_z | p99 | 49.0022 | 28.0924 | -20.9098 |
| argmax_agree | rate | 0.4688 | 0.5925 | +0.1237 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 32.1724 | 32.1723 | -0.0001 |
| z_post_norm | mean | 32.1724 | 32.1723 | -0.0001 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
