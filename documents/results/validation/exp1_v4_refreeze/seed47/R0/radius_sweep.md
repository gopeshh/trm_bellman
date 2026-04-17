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
| latent_ball_radius | 0.0 | 0.0 |
| inner_unroll_n | 2 | 2 |
| config_source | yaml | yaml |

## Projection Sanity Check

- Model A: R = 0.0, max Δ_z bound (2R) = N/A (projection disabled)
- Model B: R = 0.0, max Δ_z bound (2R) = N/A (projection disabled)

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.9784 | 1.2089 | -0.7694 |
| Δ_V | median | 0.7182 | 1.3688 | +0.6505 |
| Δ_V | p99 | 11.3935 | 3.1726 | -8.2209 |
| Δ_pi | mean | 0.0355 | 0.0278 | -0.0077 |
| Δ_pi | median | 0.0112 | 0.0117 | +0.0005 |
| Δ_pi | p99 | 0.2776 | 0.1118 | -0.1658 |
| Δ_z | mean | 30.3197 | 24.5961 | -5.7236 |
| Δ_z | median | 30.1061 | 26.0823 | -4.0238 |
| Δ_z | p99 | 48.1996 | 28.4696 | -19.7300 |
| argmax_agree | rate | 0.5800 | 0.6300 | +0.0500 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.9707 | 1.1827 | -0.7880 |
| Δ_V | median | 0.4719 | 1.0750 | +0.6031 |
| Δ_V | p99 | 10.7856 | 2.9441 | -7.8414 |
| Δ_pi | mean | 0.0311 | 0.0294 | -0.0016 |
| Δ_pi | median | 0.0099 | 0.0091 | -0.0008 |
| Δ_pi | p99 | 0.2893 | 0.1178 | -0.1715 |
| Δ_z | mean | 29.7993 | 24.5281 | -5.2712 |
| Δ_z | median | 30.1632 | 25.9676 | -4.1956 |
| Δ_z | p99 | 48.1789 | 29.0259 | -19.1530 |
| argmax_agree | rate | 0.5548 | 0.5948 | +0.0400 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
