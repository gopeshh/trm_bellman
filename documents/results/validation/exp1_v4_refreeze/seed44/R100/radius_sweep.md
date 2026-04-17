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
| latent_ball_radius | 100.0 | 100.0 |
| inner_unroll_n | 2 | 2 |
| config_source | yaml | yaml |

## Projection Sanity Check

- Model A: R = 100.0, max Δ_z bound (2R) = 200.0
- Model B: R = 100.0, max Δ_z bound (2R) = 200.0

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.5968 | 0.9919 | +0.3951 |
| Δ_V | median | 0.5265 | 0.8882 | +0.3616 |
| Δ_V | p99 | 1.5942 | 2.4767 | +0.8826 |
| Δ_pi | mean | 0.0040 | 0.0113 | +0.0072 |
| Δ_pi | median | 0.0033 | 0.0100 | +0.0067 |
| Δ_pi | p99 | 0.0167 | 0.0362 | +0.0196 |
| Δ_z | mean | 32.5550 | 22.3958 | -10.1592 |
| Δ_z | median | 32.9335 | 22.4730 | -10.4605 |
| Δ_z | p99 | 40.8379 | 23.9687 | -16.8692 |
| argmax_agree | rate | 0.5200 | 0.7300 | +0.2100 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 30.6861 | 30.6860 | -0.0001 |
| z_post_norm | mean | 30.6861 | 30.6860 | -0.0001 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 0.4971 | 0.9471 | +0.4500 |
| Δ_V | median | 0.4140 | 0.8294 | +0.4153 |
| Δ_V | p99 | 1.5502 | 2.5839 | +1.0338 |
| Δ_pi | mean | 0.0041 | 0.0146 | +0.0106 |
| Δ_pi | median | 0.0030 | 0.0098 | +0.0067 |
| Δ_pi | p99 | 0.0176 | 0.0763 | +0.0588 |
| Δ_z | mean | 32.3100 | 22.7638 | -9.5462 |
| Δ_z | median | 32.8795 | 22.8134 | -10.0661 |
| Δ_z | p99 | 40.4523 | 24.4647 | -15.9877 |
| argmax_agree | rate | 0.5453 | 0.7409 | +0.1955 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 30.6861 | 30.6860 | -0.0001 |
| z_post_norm | mean | 30.6861 | 30.6860 | -0.0001 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
