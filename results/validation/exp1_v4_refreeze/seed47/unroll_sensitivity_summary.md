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
| mean | 0.2254 | 0.1129 |
| p95 | 0.3931 | 0.1260 |
| max | 0.5218 | 0.1277 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1137 | 0.0565 | -0.0572 |
| Δ_V | median | 0.0596 | 0.0570 | -0.0026 |
| Δ_V | p99 | 0.7970 | 0.1456 | -0.6514 |
| Δ_pi | mean | 0.0001 | 0.0001 | -0.0001 |
| Δ_pi | median | 0.0001 | 0.0000 | -0.0000 |
| Δ_pi | p99 | 0.0007 | 0.0002 | -0.0006 |
| Δ_z | mean | 3.6366 | 1.7158 | -1.9209 |
| Δ_z | median | 3.6872 | 1.7484 | -1.9388 |
| Δ_z | p99 | 6.3317 | 1.9564 | -4.3754 |
| argmax_agree | rate | 0.9100 | 0.9400 | +0.0300 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 32.2039 | 32.2038 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1764 | 0.0595 | -0.1169 |
| Δ_V | median | 0.0711 | 0.0608 | -0.0103 |
| Δ_V | p99 | 1.4119 | 0.1490 | -1.2629 |
| Δ_pi | mean | 0.0002 | 0.0001 | -0.0002 |
| Δ_pi | median | 0.0001 | 0.0001 | -0.0000 |
| Δ_pi | p99 | 0.0028 | 0.0002 | -0.0026 |
| Δ_z | mean | 3.9057 | 1.7749 | -2.1308 |
| Δ_z | median | 3.9392 | 1.8413 | -2.0979 |
| Δ_z | p99 | 8.5126 | 2.1876 | -6.3251 |
| argmax_agree | rate | 0.9223 | 0.9611 | +0.0389 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 32.2039 | 32.2038 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
