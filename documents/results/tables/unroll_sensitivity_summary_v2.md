# Unroll Sensitivity Comparison: Model A vs Model B

**Training depth (n_train)**: 2
**Comparison**: n=2 (1×) vs n=8 (4×)

## Configuration Comparison

| Setting | Model A | Model B |
|---------|---------|--------|
| enable_contraction | False | True |
| target_Lz | 0.9 | 0.9 |
| disable_value_head_norm | False | True |
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
| Δ_V | mean | 0.2070 | 0.0288 | -0.1782 |
| Δ_V | median | 0.1128 | 0.0198 | -0.0930 |
| Δ_V | p99 | 0.8747 | 0.1394 | -0.7353 |
| Δ_pi | mean | 0.0045 | 0.0004 | -0.0041 |
| Δ_pi | median | 0.0016 | 0.0003 | -0.0014 |
| Δ_pi | p99 | 0.0402 | 0.0025 | -0.0377 |
| Δ_z | mean | 4.7473 | 1.5107 | -3.2367 |
| Δ_z | median | 4.6650 | 1.5266 | -3.1383 |
| Δ_z | p99 | 7.4457 | 1.7855 | -5.6601 |
| argmax_agree | rate | 0.9700 | 0.9900 | +0.0200 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2413 | 0.0999 | -0.1414 |
| Δ_V | median | 0.1496 | 0.0229 | -0.1267 |
| Δ_V | p99 | 1.3690 | 0.7261 | -0.6429 |
| Δ_pi | mean | 0.0056 | 0.0005 | -0.0051 |
| Δ_pi | median | 0.0019 | 0.0003 | -0.0016 |
| Δ_pi | p99 | 0.0522 | 0.0031 | -0.0492 |
| Δ_z | mean | 4.7516 | 1.5901 | -3.1616 |
| Δ_z | median | 4.7887 | 1.6166 | -3.1721 |
| Δ_z | p99 | 8.3733 | 1.8908 | -6.4825 |
| argmax_agree | rate | 0.9568 | 0.9930 | +0.0363 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
