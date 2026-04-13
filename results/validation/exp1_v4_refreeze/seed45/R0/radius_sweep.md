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
| Δ_V | mean | 1.8309 | 1.1242 | -0.7067 |
| Δ_V | median | 0.6148 | 1.0076 | +0.3929 |
| Δ_V | p99 | 9.5689 | 2.8062 | -6.7627 |
| Δ_pi | mean | 0.0205 | 0.1254 | +0.1049 |
| Δ_pi | median | 0.0042 | 0.0362 | +0.0320 |
| Δ_pi | p99 | 0.1562 | 0.5215 | +0.3652 |
| Δ_z | mean | 34.6438 | 23.9081 | -10.7358 |
| Δ_z | median | 34.9320 | 25.1413 | -9.7907 |
| Δ_z | p99 | 49.1424 | 31.4943 | -17.6481 |
| argmax_agree | rate | 0.4900 | 0.5600 | +0.0700 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.5245 | 1.0753 | -0.4492 |
| Δ_V | median | 0.6020 | 0.9555 | +0.3535 |
| Δ_V | p99 | 10.1955 | 2.9498 | -7.2456 |
| Δ_pi | mean | 0.0161 | 0.1335 | +0.1174 |
| Δ_pi | median | 0.0047 | 0.0390 | +0.0343 |
| Δ_pi | p99 | 0.1966 | 0.6043 | +0.4077 |
| Δ_z | mean | 33.1907 | 23.6849 | -9.5058 |
| Δ_z | median | 34.0904 | 25.0397 | -9.0507 |
| Δ_z | p99 | 49.0234 | 32.3842 | -16.6391 |
| argmax_agree | rate | 0.4971 | 0.5948 | +0.0978 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
