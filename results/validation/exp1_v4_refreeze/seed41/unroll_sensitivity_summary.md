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
| latent_ball_radius | 10.0 | 10.0 |
| inner_unroll_n | 2 | 2 |
| config_source | yaml | yaml |

## Projection Sanity Check

- Model A: R = 10.0, max Δ_z bound (2R) = 20.0
- Model B: R = 10.0, max Δ_z bound (2R) = 20.0

## Achieved Lipschitz Constant (hat_Lz)

| Statistic | Model A | Model B |
|-----------|---------|--------|
| mean | 0.2851 | 0.1068 |
| p95 | 0.4447 | 0.1124 |
| max | 0.5871 | 0.1166 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2585 | 0.0394 | -0.2191 |
| Δ_V | median | 0.1817 | 0.0380 | -0.1436 |
| Δ_V | p99 | 1.3311 | 0.1056 | -1.2256 |
| Δ_pi | mean | 0.0003 | 0.0000 | -0.0003 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0025 | 0.0001 | -0.0024 |
| Δ_z | mean | 4.3266 | 1.5444 | -2.7822 |
| Δ_z | median | 4.3240 | 1.5724 | -2.7516 |
| Δ_z | p99 | 7.6888 | 1.6542 | -6.0346 |
| argmax_agree | rate | 0.8900 | 0.9500 | +0.0600 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.8046 | 31.8045 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2698 | 0.0449 | -0.2249 |
| Δ_V | median | 0.1171 | 0.0381 | -0.0790 |
| Δ_V | p99 | 2.0356 | 0.1785 | -1.8571 |
| Δ_pi | mean | 0.0004 | 0.0000 | -0.0003 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0043 | 0.0001 | -0.0042 |
| Δ_z | mean | 4.5963 | 1.6420 | -2.9543 |
| Δ_z | median | 4.7054 | 1.6754 | -3.0300 |
| Δ_z | p99 | 9.2331 | 1.8284 | -7.4048 |
| argmax_agree | rate | 0.8787 | 0.9847 | +0.1060 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.8046 | 31.8045 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
