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
| mean | 0.2803 | 0.0969 |
| p95 | 0.3913 | 0.1653 |
| max | 0.4689 | 0.1763 |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2264 | 0.0597 | -0.1667 |
| Δ_V | median | 0.1376 | 0.0366 | -0.1010 |
| Δ_V | p99 | 1.0046 | 0.3766 | -0.6280 |
| Δ_pi | mean | 0.0065 | 0.0001 | -0.0064 |
| Δ_pi | median | 0.0025 | 0.0001 | -0.0024 |
| Δ_pi | p99 | 0.0397 | 0.0006 | -0.0390 |
| Δ_z | mean | 5.0679 | 1.3186 | -3.7492 |
| Δ_z | median | 5.0146 | 1.2855 | -3.7291 |
| Δ_z | p99 | 8.0879 | 1.6605 | -6.4274 |
| argmax_agree | rate | 0.9400 | 0.9900 | +0.0500 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.8046 | 31.8045 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.2012 | 0.1838 | -0.0174 |
| Δ_V | median | 0.1213 | 0.0351 | -0.0863 |
| Δ_V | p99 | 1.1092 | 1.7935 | +0.6843 |
| Δ_pi | mean | 0.0071 | 0.0004 | -0.0068 |
| Δ_pi | median | 0.0034 | 0.0001 | -0.0032 |
| Δ_pi | p99 | 0.0479 | 0.0050 | -0.0429 |
| Δ_z | mean | 4.9796 | 1.4739 | -3.5057 |
| Δ_z | median | 4.8560 | 1.3217 | -3.5342 |
| Δ_z | p99 | 9.0081 | 2.9913 | -6.0168 |
| argmax_agree | rate | 0.9466 | 0.9851 | +0.0385 |
| saturation | rate | 1.0000 | 1.0000 | +0.0000 |
| z_pre_norm | mean | 31.8046 | 31.8045 | -0.0001 |
| z_post_norm | mean | 10.0000 | 10.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
