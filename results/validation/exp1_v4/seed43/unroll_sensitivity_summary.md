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
| mean | 0.1232 | 0.0868 |
| p95 | 0.1813 | 0.1100 |
| max | 0.2317 | 0.1397 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.0334 | 0.0300 | -0.0034 |
| Δ_V | median | 0.0168 | 0.0135 | -0.0033 |
| Δ_V | p99 | 0.1711 | 0.2062 | +0.0352 |
| Δ_pi | mean | 0.0007 | 0.0001 | -0.0005 |
| Δ_pi | median | 0.0002 | 0.0001 | -0.0001 |
| Δ_pi | p99 | 0.0056 | 0.0007 | -0.0049 |
| Δ_z | mean | 1.7323 | 1.2948 | -0.4375 |
| Δ_z | median | 1.4780 | 1.2604 | -0.2176 |
| Δ_z | p99 | 4.4328 | 1.7485 | -2.6843 |
| argmax_agree | rate | 0.9900 | 1.0000 | +0.0100 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.9551 | 31.9551 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.0418 | 0.1055 | +0.0636 |
| Δ_V | median | 0.0201 | 0.0130 | -0.0072 |
| Δ_V | p99 | 0.2433 | 1.1585 | +0.9152 |
| Δ_pi | mean | 0.0009 | 0.0002 | -0.0006 |
| Δ_pi | median | 0.0003 | 0.0001 | -0.0002 |
| Δ_pi | p99 | 0.0083 | 0.0030 | -0.0053 |
| Δ_z | mean | 1.9051 | 1.3609 | -0.5443 |
| Δ_z | median | 1.8326 | 1.2875 | -0.5450 |
| Δ_z | p99 | 4.4030 | 2.2062 | -2.1968 |
| argmax_agree | rate | 0.9801 | 0.9950 | +0.0149 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.9551 | 31.9551 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
