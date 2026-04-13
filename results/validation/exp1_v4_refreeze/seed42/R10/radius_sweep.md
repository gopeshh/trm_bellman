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
| Δ_V | mean | 0.0887 | 0.0604 | -0.0283 |
| Δ_V | median | 0.0669 | 0.0555 | -0.0114 |
| Δ_V | p99 | 0.2840 | 0.1603 | -0.1237 |
| Δ_pi | mean | 0.0001 | 0.0001 | +0.0001 |
| Δ_pi | median | 0.0000 | 0.0001 | +0.0001 |
| Δ_pi | p99 | 0.0003 | 0.0004 | +0.0001 |
| Δ_z | mean | 3.2919 | 1.5553 | -1.7366 |
| Δ_z | median | 3.1818 | 1.5810 | -1.6007 |
| Δ_z | p99 | 4.8394 | 1.8024 | -3.0370 |
| argmax_agree | rate | 0.9300 | 0.9300 | +0.0000 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2788 | 0.0564 | -0.2224 |
| Δ_V | median | 0.0950 | 0.0476 | -0.0473 |
| Δ_V | p99 | 3.0750 | 0.1720 | -2.9030 |
| Δ_pi | mean | 0.0003 | 0.0001 | -0.0002 |
| Δ_pi | median | 0.0000 | 0.0001 | +0.0001 |
| Δ_pi | p99 | 0.0046 | 0.0004 | -0.0042 |
| Δ_z | mean | 3.8099 | 1.5948 | -2.2150 |
| Δ_z | median | 3.6784 | 1.6451 | -2.0334 |
| Δ_z | p99 | 9.7514 | 1.9446 | -7.8068 |
| argmax_agree | rate | 0.9140 | 0.9458 | +0.0318 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
