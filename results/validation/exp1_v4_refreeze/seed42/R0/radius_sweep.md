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
| Δ_V | mean | 1.8563 | 1.3996 | -0.4567 |
| Δ_V | median | 1.3265 | 1.3594 | +0.0329 |
| Δ_V | p99 | 9.1826 | 3.7677 | -5.4149 |
| Δ_pi | mean | 0.0566 | 0.0474 | -0.0092 |
| Δ_pi | median | 0.0277 | 0.0302 | +0.0024 |
| Δ_pi | p99 | 0.2416 | 0.1502 | -0.0913 |
| Δ_z | mean | 33.7332 | 23.4992 | -10.2340 |
| Δ_z | median | 34.4211 | 24.7515 | -9.6697 |
| Δ_z | p99 | 42.9835 | 27.1147 | -15.8688 |
| argmax_agree | rate | 0.3800 | 0.5800 | +0.2000 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.8328 | 1.2666 | -0.5662 |
| Δ_V | median | 1.3550 | 1.1766 | -0.1784 |
| Δ_V | p99 | 10.7641 | 3.8086 | -6.9554 |
| Δ_pi | mean | 0.0448 | 0.0474 | +0.0026 |
| Δ_pi | median | 0.0264 | 0.0299 | +0.0035 |
| Δ_pi | p99 | 0.2563 | 0.1567 | -0.0996 |
| Δ_z | mean | 32.9726 | 23.3431 | -9.6295 |
| Δ_z | median | 34.4848 | 24.5936 | -9.8912 |
| Δ_z | p99 | 44.9167 | 27.8731 | -17.0436 |
| argmax_agree | rate | 0.3640 | 0.5807 | +0.2167 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
