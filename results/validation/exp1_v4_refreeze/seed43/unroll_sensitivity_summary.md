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
| mean | 0.2896 | 0.0943 |
| p95 | 0.4406 | 0.1086 |
| max | 0.5906 | 0.1110 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1351 | 0.0309 | -0.1042 |
| Δ_V | median | 0.0628 | 0.0232 | -0.0396 |
| Δ_V | p99 | 1.4170 | 0.1290 | -1.2880 |
| Δ_pi | mean | 0.0003 | 0.0000 | -0.0003 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0034 | 0.0001 | -0.0032 |
| Δ_z | mean | 3.9740 | 1.3615 | -2.6124 |
| Δ_z | median | 3.5112 | 1.3904 | -2.1208 |
| Δ_z | p99 | 9.2270 | 1.5758 | -7.6512 |
| argmax_agree | rate | 0.9100 | 0.9800 | +0.0700 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.9551 | 31.9551 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1566 | 0.0378 | -0.1188 |
| Δ_V | median | 0.0754 | 0.0299 | -0.0455 |
| Δ_V | p99 | 1.1540 | 0.1501 | -1.0039 |
| Δ_pi | mean | 0.0004 | 0.0000 | -0.0003 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0001 |
| Δ_pi | p99 | 0.0032 | 0.0002 | -0.0030 |
| Δ_z | mean | 4.4559 | 1.4386 | -3.0173 |
| Δ_z | median | 4.4001 | 1.4639 | -2.9362 |
| Δ_z | p99 | 8.7026 | 1.7966 | -6.9060 |
| argmax_agree | rate | 0.8728 | 0.9647 | +0.0919 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.9551 | 31.9551 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
