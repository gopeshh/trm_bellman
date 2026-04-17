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
| Δ_V | mean | 0.9995 | 0.3682 | -0.6313 |
| Δ_V | median | 0.6053 | 0.3387 | -0.2667 |
| Δ_V | p99 | 5.2803 | 0.8712 | -4.4090 |
| Δ_pi | mean | 0.3551 | 0.0197 | -0.3354 |
| Δ_pi | median | 0.1797 | 0.0184 | -0.1612 |
| Δ_pi | p99 | 1.6413 | 0.0666 | -1.5747 |
| Δ_z | mean | 38.1719 | 16.0751 | -22.0969 |
| Δ_z | median | 38.7719 | 15.8499 | -22.9220 |
| Δ_z | p99 | 44.9570 | 19.0004 | -25.9567 |
| argmax_agree | rate | 0.6000 | 0.9400 | +0.3400 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.0112 | 2.4121 | +1.4010 |
| Δ_V | median | 0.6256 | 0.4577 | -0.1680 |
| Δ_V | p99 | 6.0239 | 23.5964 | +17.5725 |
| Δ_pi | mean | 0.3638 | 0.0389 | -0.3249 |
| Δ_pi | median | 0.2093 | 0.0204 | -0.1890 |
| Δ_pi | p99 | 1.9369 | 0.4169 | -1.5200 |
| Δ_z | mean | 37.0842 | 17.8546 | -19.2296 |
| Δ_z | median | 37.7517 | 15.9441 | -21.8077 |
| Δ_z | p99 | 44.0270 | 36.3980 | -7.6290 |
| argmax_agree | rate | 0.5938 | 0.8783 | +0.2845 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
