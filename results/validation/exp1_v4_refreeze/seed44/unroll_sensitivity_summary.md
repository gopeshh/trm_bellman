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
| mean | 0.2936 | 0.1203 |
| p95 | 0.4120 | 0.1292 |
| max | 0.4389 | 0.1360 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1183 | 0.0393 | -0.0790 |
| Δ_V | median | 0.0555 | 0.0341 | -0.0213 |
| Δ_V | p99 | 1.2896 | 0.1312 | -1.1584 |
| Δ_pi | mean | 0.0002 | 0.0000 | -0.0001 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0000 |
| Δ_pi | p99 | 0.0017 | 0.0001 | -0.0017 |
| Δ_z | mean | 4.2977 | 1.7736 | -2.5240 |
| Δ_z | median | 4.4081 | 1.7838 | -2.6244 |
| Δ_z | p99 | 7.4011 | 1.9967 | -5.4044 |
| argmax_agree | rate | 0.8800 | 0.9900 | +0.1100 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 30.6861 | 30.6860 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1174 | 0.0469 | -0.0705 |
| Δ_V | median | 0.0567 | 0.0420 | -0.0147 |
| Δ_V | p99 | 1.0108 | 0.1755 | -0.8354 |
| Δ_pi | mean | 0.0002 | 0.0000 | -0.0002 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0000 |
| Δ_pi | p99 | 0.0019 | 0.0001 | -0.0019 |
| Δ_z | mean | 4.5501 | 1.8869 | -2.6632 |
| Δ_z | median | 4.6280 | 1.9086 | -2.7193 |
| Δ_z | p99 | 8.6254 | 2.1717 | -6.4536 |
| argmax_agree | rate | 0.9199 | 0.9812 | +0.0612 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 30.6861 | 30.6860 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
