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
| latent_ball_radius | 30.0 | 30.0 |
| inner_unroll_n | 2 | 2 |
| config_source | yaml | yaml |

## Projection Sanity Check

- Model A: R = 30.0, max Δ_z bound (2R) = 60.0
- Model B: R = 30.0, max Δ_z bound (2R) = 60.0

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.9110 | 0.1741 | -1.7369 |
| Δ_V | median | 1.0600 | 0.1574 | -0.9026 |
| Δ_V | p99 | 9.1436 | 0.7410 | -8.4026 |
| Δ_pi | mean | 0.7079 | 0.0281 | -0.6797 |
| Δ_pi | median | 0.4012 | 0.0173 | -0.3838 |
| Δ_pi | p99 | 2.9795 | 0.1828 | -2.7968 |
| Δ_z | mean | 35.4848 | 16.5424 | -18.9424 |
| Δ_z | median | 35.5464 | 16.5034 | -19.0431 |
| Δ_z | p99 | 42.4082 | 20.4916 | -21.9167 |
| argmax_agree | rate | 0.4700 | 0.9300 | +0.4600 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 30.0000 | 30.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.8921 | 1.1331 | -0.7590 |
| Δ_V | median | 1.1844 | 0.1469 | -1.0375 |
| Δ_V | p99 | 10.8454 | 11.0167 | +0.1714 |
| Δ_pi | mean | 0.6714 | 0.0354 | -0.6359 |
| Δ_pi | median | 0.4509 | 0.0189 | -0.4320 |
| Δ_pi | p99 | 3.2651 | 0.2387 | -3.0264 |
| Δ_z | mean | 34.6124 | 16.7647 | -17.8477 |
| Δ_z | median | 36.3379 | 16.8028 | -19.5351 |
| Δ_z | p99 | 42.8915 | 20.8264 | -22.0650 |
| argmax_agree | rate | 0.5193 | 0.9130 | +0.3938 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 30.0000 | 30.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
