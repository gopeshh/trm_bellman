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
| Δ_V | mean | 0.1966 | 0.1114 | -0.0852 |
| Δ_V | median | 0.0956 | 0.0740 | -0.0216 |
| Δ_V | p99 | 1.3505 | 0.6081 | -0.7425 |
| Δ_pi | mean | 0.0001 | 0.0001 | -0.0001 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0008 | 0.0002 | -0.0006 |
| Δ_z | mean | 5.9828 | 1.9638 | -4.0190 |
| Δ_z | median | 6.0865 | 1.7624 | -4.3241 |
| Δ_z | p99 | 8.1327 | 3.9044 | -4.2283 |
| argmax_agree | rate | 0.8400 | 0.9300 | +0.0900 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.8308 | 31.8307 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2368 | 0.1033 | -0.1335 |
| Δ_V | median | 0.1039 | 0.0694 | -0.0345 |
| Δ_V | p99 | 1.9737 | 0.4982 | -1.4754 |
| Δ_pi | mean | 0.0002 | 0.0001 | -0.0001 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0010 | 0.0004 | -0.0007 |
| Δ_z | mean | 6.0198 | 1.9808 | -4.0390 |
| Δ_z | median | 6.2439 | 1.8267 | -4.4173 |
| Δ_z | p99 | 9.0845 | 3.7677 | -5.3169 |
| argmax_agree | rate | 0.8893 | 0.9211 | +0.0318 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.8308 | 31.8307 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
