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
| Δ_V | mean | 1.6216 | 1.2961 | -0.3255 |
| Δ_V | median | 1.1013 | 1.0529 | -0.0484 |
| Δ_V | p99 | 6.6516 | 4.4087 | -2.2429 |
| Δ_pi | mean | 0.0137 | 0.0156 | +0.0019 |
| Δ_pi | median | 0.0067 | 0.0069 | +0.0002 |
| Δ_pi | p99 | 0.1265 | 0.0934 | -0.0331 |
| Δ_z | mean | 40.7134 | 24.0918 | -16.6215 |
| Δ_z | median | 41.1292 | 23.5446 | -17.5846 |
| Δ_z | p99 | 45.8355 | 30.3238 | -15.5118 |
| argmax_agree | rate | 0.5300 | 0.5200 | -0.0100 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 31.8308 | 31.8307 | -0.0000 |
| z_post_norm | mean | 31.8308 | 31.8307 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.6116 | 1.1589 | -0.4527 |
| Δ_V | median | 1.0063 | 0.9265 | -0.0798 |
| Δ_V | p99 | 7.3867 | 4.7285 | -2.6582 |
| Δ_pi | mean | 0.0139 | 0.0218 | +0.0079 |
| Δ_pi | median | 0.0070 | 0.0068 | -0.0002 |
| Δ_pi | p99 | 0.1071 | 0.3109 | +0.2038 |
| Δ_z | mean | 39.9165 | 24.0233 | -15.8932 |
| Δ_z | median | 41.4568 | 23.5483 | -17.9085 |
| Δ_z | p99 | 46.7880 | 30.4972 | -16.2907 |
| argmax_agree | rate | 0.4947 | 0.5100 | +0.0153 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 31.8308 | 31.8307 | -0.0000 |
| z_post_norm | mean | 31.8308 | 31.8307 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
