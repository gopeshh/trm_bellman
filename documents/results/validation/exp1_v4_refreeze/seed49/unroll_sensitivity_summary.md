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
| mean | 0.2241 | 0.1129 |
| p95 | 0.3649 | 0.1216 |
| max | 0.6943 | 0.1290 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2383 | 0.0539 | -0.1845 |
| Δ_V | median | 0.0444 | 0.0440 | -0.0004 |
| Δ_V | p99 | 2.0728 | 0.1557 | -1.9171 |
| Δ_pi | mean | 0.0002 | 0.0000 | -0.0002 |
| Δ_pi | median | 0.0000 | 0.0000 | -0.0000 |
| Δ_pi | p99 | 0.0025 | 0.0001 | -0.0024 |
| Δ_z | mean | 2.9853 | 1.6526 | -1.3327 |
| Δ_z | median | 2.8468 | 1.6695 | -1.1773 |
| Δ_z | p99 | 5.6847 | 1.8905 | -3.7942 |
| argmax_agree | rate | 0.9500 | 0.9900 | +0.0400 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 30.5872 | 30.5871 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.3393 | 0.0579 | -0.2815 |
| Δ_V | median | 0.0603 | 0.0481 | -0.0122 |
| Δ_V | p99 | 2.4053 | 0.2041 | -2.2012 |
| Δ_pi | mean | 0.0005 | 0.0000 | -0.0004 |
| Δ_pi | median | 0.0000 | 0.0000 | -0.0000 |
| Δ_pi | p99 | 0.0072 | 0.0003 | -0.0069 |
| Δ_z | mean | 3.5624 | 1.7166 | -1.8458 |
| Δ_z | median | 3.3128 | 1.7345 | -1.5783 |
| Δ_z | p99 | 8.9456 | 1.9950 | -6.9506 |
| argmax_agree | rate | 0.9234 | 0.9776 | +0.0542 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 30.5872 | 30.5871 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
