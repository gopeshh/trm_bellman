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
| mean | 0.3835 | 0.0978 |
| p95 | 0.5186 | 0.1176 |
| max | 0.5481 | 0.1229 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1964 | 0.0243 | -0.1720 |
| Δ_V | median | 0.1163 | 0.0156 | -0.1007 |
| Δ_V | p99 | 0.7879 | 0.0959 | -0.6920 |
| Δ_pi | mean | 0.0115 | 0.0003 | -0.0112 |
| Δ_pi | median | 0.0049 | 0.0002 | -0.0047 |
| Δ_pi | p99 | 0.0736 | 0.0019 | -0.0717 |
| Δ_z | mean | 5.4273 | 1.3623 | -4.0649 |
| Δ_z | median | 5.3074 | 1.3660 | -3.9414 |
| Δ_z | p99 | 8.1181 | 1.7564 | -6.3616 |
| argmax_agree | rate | 0.9400 | 0.9800 | +0.0400 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2143 | 0.0695 | -0.1449 |
| Δ_V | median | 0.1312 | 0.0144 | -0.1167 |
| Δ_V | p99 | 1.1739 | 0.6914 | -0.4826 |
| Δ_pi | mean | 0.0105 | 0.0003 | -0.0102 |
| Δ_pi | median | 0.0053 | 0.0002 | -0.0050 |
| Δ_pi | p99 | 0.0710 | 0.0022 | -0.0688 |
| Δ_z | mean | 5.1651 | 1.3870 | -3.7781 |
| Δ_z | median | 5.2479 | 1.4229 | -3.8251 |
| Δ_z | p99 | 8.9202 | 1.8403 | -7.0799 |
| argmax_agree | rate | 0.9267 | 0.9901 | +0.0634 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 33.1913 | 33.1912 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
