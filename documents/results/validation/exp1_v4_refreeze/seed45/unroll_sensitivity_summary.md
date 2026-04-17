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
| mean | 0.2949 | 0.1172 |
| p95 | 0.4321 | 0.1389 |
| max | 0.5669 | 0.1419 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2390 | 0.0741 | -0.1648 |
| Δ_V | median | 0.0885 | 0.0625 | -0.0259 |
| Δ_V | p99 | 1.4602 | 0.2349 | -1.2253 |
| Δ_pi | mean | 0.0004 | 0.0004 | -0.0000 |
| Δ_pi | median | 0.0001 | 0.0002 | +0.0001 |
| Δ_pi | p99 | 0.0033 | 0.0018 | -0.0014 |
| Δ_z | mean | 4.7433 | 1.8661 | -2.8772 |
| Δ_z | median | 4.9497 | 1.9044 | -3.0453 |
| Δ_z | p99 | 8.0593 | 2.4158 | -5.6435 |
| argmax_agree | rate | 0.9300 | 0.9200 | -0.0100 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 32.4105 | 32.4104 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2616 | 0.0724 | -0.1892 |
| Δ_V | median | 0.0958 | 0.0611 | -0.0347 |
| Δ_V | p99 | 2.2750 | 0.2276 | -2.0474 |
| Δ_pi | mean | 0.0004 | 0.0004 | -0.0000 |
| Δ_pi | median | 0.0001 | 0.0003 | +0.0002 |
| Δ_pi | p99 | 0.0043 | 0.0025 | -0.0017 |
| Δ_z | mean | 5.1460 | 1.8923 | -3.2536 |
| Δ_z | median | 5.2095 | 1.9688 | -3.2406 |
| Δ_z | p99 | 9.3553 | 2.6298 | -6.7255 |
| argmax_agree | rate | 0.8528 | 0.9093 | +0.0565 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 32.4105 | 32.4104 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
