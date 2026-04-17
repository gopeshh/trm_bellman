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
| Δ_V | mean | 1.9853 | 0.2026 | -1.7827 |
| Δ_V | median | 1.3208 | 0.1859 | -1.1349 |
| Δ_V | p99 | 13.4380 | 0.8022 | -12.6358 |
| Δ_pi | mean | 0.7947 | 0.0407 | -0.7540 |
| Δ_pi | median | 0.3846 | 0.0262 | -0.3584 |
| Δ_pi | p99 | 3.4794 | 0.2316 | -3.2478 |
| Δ_z | mean | 39.2862 | 20.1030 | -19.1832 |
| Δ_z | median | 40.1900 | 20.1178 | -20.0723 |
| Δ_z | p99 | 47.5846 | 24.1578 | -23.4268 |
| argmax_agree | rate | 0.5200 | 0.9100 | +0.3900 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 2.1501 | 1.3296 | -0.8205 |
| Δ_V | median | 1.4156 | 0.1773 | -1.2383 |
| Δ_V | p99 | 11.8459 | 12.7720 | +0.9261 |
| Δ_pi | mean | 0.8318 | 0.0511 | -0.7807 |
| Δ_pi | median | 0.5757 | 0.0278 | -0.5478 |
| Δ_pi | p99 | 3.7835 | 0.2972 | -3.4863 |
| Δ_z | mean | 38.0927 | 20.3445 | -17.7482 |
| Δ_z | median | 40.3027 | 20.4002 | -19.9025 |
| Δ_z | p99 | 46.4712 | 24.4261 | -22.0451 |
| argmax_agree | rate | 0.4882 | 0.8981 | +0.4099 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
