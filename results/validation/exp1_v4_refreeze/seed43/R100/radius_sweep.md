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
| Δ_V | mean | 1.5091 | 0.4854 | -1.0237 |
| Δ_V | median | 0.7127 | 0.4062 | -0.3065 |
| Δ_V | p99 | 10.7767 | 1.6734 | -9.1033 |
| Δ_pi | mean | 0.0316 | 0.0141 | -0.0175 |
| Δ_pi | median | 0.0145 | 0.0038 | -0.0107 |
| Δ_pi | p99 | 0.3054 | 0.1528 | -0.1526 |
| Δ_z | mean | 35.1931 | 18.4684 | -16.7247 |
| Δ_z | median | 34.5754 | 18.1272 | -16.4482 |
| Δ_z | p99 | 50.3226 | 23.3224 | -27.0002 |
| argmax_agree | rate | 0.5500 | 0.7700 | +0.2200 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 31.9551 | 31.9551 | -0.0001 |
| z_post_norm | mean | 31.9551 | 31.9551 | -0.0001 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.5186 | 0.5291 | -0.9895 |
| Δ_V | median | 0.6479 | 0.4636 | -0.1843 |
| Δ_V | p99 | 11.2540 | 1.6760 | -9.5780 |
| Δ_pi | mean | 0.0413 | 0.0272 | -0.0141 |
| Δ_pi | median | 0.0183 | 0.0044 | -0.0139 |
| Δ_pi | p99 | 0.3977 | 0.2703 | -0.1274 |
| Δ_z | mean | 34.2300 | 19.0442 | -15.1858 |
| Δ_z | median | 34.7837 | 18.9876 | -15.7961 |
| Δ_z | p99 | 50.8871 | 24.6356 | -26.2515 |
| argmax_agree | rate | 0.5289 | 0.7420 | +0.2132 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 31.9551 | 31.9551 | -0.0001 |
| z_post_norm | mean | 31.9551 | 31.9551 | -0.0001 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
