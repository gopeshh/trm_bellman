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
| Δ_V | mean | 1.0323 | 1.2559 | +0.2236 |
| Δ_V | median | 0.7439 | 1.2227 | +0.4787 |
| Δ_V | p99 | 5.9815 | 2.6631 | -3.3184 |
| Δ_pi | mean | 0.0158 | 0.0189 | +0.0031 |
| Δ_pi | median | 0.0099 | 0.0071 | -0.0028 |
| Δ_pi | p99 | 0.0696 | 0.1610 | +0.0914 |
| Δ_z | mean | 33.2703 | 23.0624 | -10.2079 |
| Δ_z | median | 35.9503 | 22.9753 | -12.9750 |
| Δ_z | p99 | 44.3965 | 25.8309 | -18.5656 |
| argmax_agree | rate | 0.6000 | 0.7000 | +0.1000 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 31.5233 | 31.5232 | -0.0001 |
| z_post_norm | mean | 31.5233 | 31.5232 | -0.0001 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 1.0516 | 1.2417 | +0.1902 |
| Δ_V | median | 0.5567 | 1.2169 | +0.6602 |
| Δ_V | p99 | 5.9744 | 2.9615 | -3.0130 |
| Δ_pi | mean | 0.0219 | 0.0242 | +0.0022 |
| Δ_pi | median | 0.0124 | 0.0077 | -0.0048 |
| Δ_pi | p99 | 0.1139 | 0.2004 | +0.0865 |
| Δ_z | mean | 32.1049 | 23.2231 | -8.8818 |
| Δ_z | median | 35.7945 | 23.5565 | -12.2380 |
| Δ_z | p99 | 45.9576 | 26.4340 | -19.5236 |
| argmax_agree | rate | 0.5878 | 0.6879 | +0.1001 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |
| z_pre_norm | mean | 31.5233 | 31.5232 | -0.0001 |
| z_post_norm | mean | 31.5233 | 31.5232 | -0.0001 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
