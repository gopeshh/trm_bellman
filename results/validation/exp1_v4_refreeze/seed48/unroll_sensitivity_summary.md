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
| mean | 0.3016 | 0.1051 |
| p95 | 0.4316 | 0.1195 |
| max | 0.4952 | 0.1214 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1707 | 0.0992 | -0.0715 |
| Δ_V | median | 0.0920 | 0.1019 | +0.0100 |
| Δ_V | p99 | 1.0988 | 0.3139 | -0.7849 |
| Δ_pi | mean | 0.0003 | 0.0001 | -0.0002 |
| Δ_pi | median | 0.0001 | 0.0001 | -0.0001 |
| Δ_pi | p99 | 0.0023 | 0.0002 | -0.0021 |
| Δ_z | mean | 5.3338 | 1.6324 | -3.7014 |
| Δ_z | median | 4.9160 | 1.6742 | -3.2419 |
| Δ_z | p99 | 10.2065 | 1.8617 | -8.3447 |
| argmax_agree | rate | 0.8400 | 0.9600 | +0.1200 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 32.1724 | 32.1723 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.1746 | 0.0984 | -0.0762 |
| Δ_V | median | 0.0967 | 0.0922 | -0.0045 |
| Δ_V | p99 | 1.2662 | 0.3075 | -0.9586 |
| Δ_pi | mean | 0.0002 | 0.0001 | -0.0002 |
| Δ_pi | median | 0.0001 | 0.0001 | -0.0000 |
| Δ_pi | p99 | 0.0022 | 0.0003 | -0.0019 |
| Δ_z | mean | 5.3996 | 1.6658 | -3.7338 |
| Δ_z | median | 5.5354 | 1.7328 | -3.8027 |
| Δ_z | p99 | 9.7991 | 1.9642 | -7.8350 |
| argmax_agree | rate | 0.8575 | 0.9411 | +0.0836 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 32.1724 | 32.1723 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | -0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
