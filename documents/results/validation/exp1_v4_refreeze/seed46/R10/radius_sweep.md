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

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.0988 | 0.0779 | -0.0209 |
| Δ_V | median | 0.0772 | 0.0751 | -0.0021 |
| Δ_V | p99 | 0.4182 | 0.1756 | -0.2426 |
| Δ_pi | mean | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0000 |
| Δ_pi | p99 | 0.0005 | 0.0001 | -0.0004 |
| Δ_z | mean | 4.0435 | 1.6609 | -2.3826 |
| Δ_z | median | 4.0752 | 1.6595 | -2.4157 |
| Δ_z | p99 | 6.6280 | 1.8097 | -4.8182 |
| argmax_agree | rate | 0.9400 | 0.9600 | +0.0200 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.5233 | 31.5232 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1644 | 0.0843 | -0.0801 |
| Δ_V | median | 0.0795 | 0.0801 | +0.0006 |
| Δ_V | p99 | 1.6546 | 0.2474 | -1.4073 |
| Δ_pi | mean | 0.0002 | 0.0000 | -0.0002 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0030 | 0.0003 | -0.0027 |
| Δ_z | mean | 4.3264 | 1.7581 | -2.5683 |
| Δ_z | median | 4.3819 | 1.7670 | -2.6149 |
| Δ_z | p99 | 8.4041 | 2.3487 | -6.0554 |
| argmax_agree | rate | 0.8893 | 0.9717 | +0.0824 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.5233 | 31.5232 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
