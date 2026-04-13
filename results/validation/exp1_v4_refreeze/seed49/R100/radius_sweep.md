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
| Δ_V | mean | 4.6204 | 0.8011 | -3.8193 |
| Δ_V | median | 1.8156 | 0.7276 | -1.0880 |
| Δ_V | p99 | 15.4255 | 2.2495 | -13.1761 |
| Δ_pi | mean | 0.0697 | 0.0096 | -0.0601 |
| Δ_pi | median | 0.0209 | 0.0036 | -0.0173 |
| Δ_pi | p99 | 0.3762 | 0.0802 | -0.2960 |
| Δ_z | mean | 36.2143 | 21.1629 | -15.0514 |
| Δ_z | median | 36.9289 | 21.8797 | -15.0492 |
| Δ_z | p99 | 48.8454 | 22.9978 | -25.8476 |
| argmax_agree | rate | 0.4400 | 0.7300 | +0.2900 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 30.5872 | 30.5871 | -0.0001 |
| z_post_norm | mean | 30.5872 | 30.5871 | -0.0001 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 4.8477 | 0.7948 | -4.0529 |
| Δ_V | median | 1.5017 | 0.7404 | -0.7613 |
| Δ_V | p99 | 17.1729 | 2.5942 | -14.5787 |
| Δ_pi | mean | 0.0807 | 0.0129 | -0.0678 |
| Δ_pi | median | 0.0177 | 0.0040 | -0.0137 |
| Δ_pi | p99 | 0.3935 | 0.1358 | -0.2577 |
| Δ_z | mean | 36.1558 | 21.5194 | -14.6363 |
| Δ_z | median | 38.1253 | 21.8753 | -16.2500 |
| Δ_z | p99 | 49.6409 | 24.4780 | -25.1629 |
| argmax_agree | rate | 0.3840 | 0.7621 | +0.3781 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 30.5872 | 30.5871 | -0.0001 |
| z_post_norm | mean | 30.5872 | 30.5871 | -0.0001 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
