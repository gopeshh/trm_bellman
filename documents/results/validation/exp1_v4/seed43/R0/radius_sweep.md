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
| Δ_V | mean | 0.2496 | 0.1490 | -0.1006 |
| Δ_V | median | 0.1576 | 0.1226 | -0.0350 |
| Δ_V | p99 | 1.8426 | 0.6036 | -1.2390 |
| Δ_pi | mean | 0.1643 | 0.0141 | -0.1502 |
| Δ_pi | median | 0.0617 | 0.0069 | -0.0548 |
| Δ_pi | p99 | 1.0023 | 0.0626 | -0.9397 |
| Δ_z | mean | 21.5782 | 15.3612 | -6.2170 |
| Δ_z | median | 21.4437 | 14.9351 | -6.5087 |
| Δ_z | p99 | 30.0380 | 20.6154 | -9.4226 |
| argmax_agree | rate | 0.8400 | 0.9500 | +0.1100 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.3444 | 1.8029 | +1.4585 |
| Δ_V | median | 0.1707 | 0.1292 | -0.0415 |
| Δ_V | p99 | 2.9849 | 18.6167 | +15.6319 |
| Δ_pi | mean | 0.1711 | 0.0285 | -0.1426 |
| Δ_pi | median | 0.0802 | 0.0117 | -0.0685 |
| Δ_pi | p99 | 1.0733 | 0.3650 | -0.7083 |
| Δ_z | mean | 23.4834 | 16.4668 | -7.0166 |
| Δ_z | median | 23.0259 | 15.0201 | -8.0058 |
| Δ_z | p99 | 34.9401 | 30.3925 | -4.5476 |
| argmax_agree | rate | 0.7714 | 0.9118 | +0.1404 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
