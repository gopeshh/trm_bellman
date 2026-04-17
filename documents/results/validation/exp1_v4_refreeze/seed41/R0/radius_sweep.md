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
| latent_ball_radius | 0.0 | 0.0 |
| inner_unroll_n | 2 | 2 |
| config_source | yaml | yaml |

## Projection Sanity Check

- Model A: R = 0.0, max Δ_z bound (2R) = N/A (projection disabled)
- Model B: R = 0.0, max Δ_z bound (2R) = N/A (projection disabled)

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 6.9482 | 0.9088 | -6.0395 |
| Δ_V | median | 6.8749 | 0.9505 | -5.9244 |
| Δ_V | p99 | 16.1613 | 1.7800 | -14.3813 |
| Δ_pi | mean | 0.1063 | 0.0173 | -0.0890 |
| Δ_pi | median | 0.0596 | 0.0130 | -0.0466 |
| Δ_pi | p99 | 0.4231 | 0.0781 | -0.3450 |
| Δ_z | mean | 43.8924 | 20.8074 | -23.0850 |
| Δ_z | median | 45.6997 | 21.0417 | -24.6580 |
| Δ_z | p99 | 54.3620 | 22.1641 | -32.1979 |
| argmax_agree | rate | 0.2300 | 0.6500 | +0.4200 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 6.6896 | 0.8557 | -5.8339 |
| Δ_V | median | 6.4211 | 0.9131 | -5.5080 |
| Δ_V | p99 | 17.0467 | 1.9999 | -15.0468 |
| Δ_pi | mean | 0.1085 | 0.0227 | -0.0858 |
| Δ_pi | median | 0.0600 | 0.0149 | -0.0451 |
| Δ_pi | p99 | 0.4889 | 0.1294 | -0.3596 |
| Δ_z | mean | 42.7951 | 21.1792 | -21.6159 |
| Δ_z | median | 45.1692 | 21.2614 | -23.9077 |
| Δ_z | p99 | 54.6538 | 23.1299 | -31.5239 |
| argmax_agree | rate | 0.2968 | 0.6608 | +0.3640 |
| saturation | rate | N/A | N/A | N/A |
| z_pre_norm | mean | 0.0000 | 0.0000 | +0.0000 |
| z_post_norm | mean | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π, Δ_z** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- **saturation rate** measures how often ||z_pre|| >= 0.95 * R (projection needed)
- **z_pre_norm, z_post_norm** should be nonzero when projection is enabled
- Δ_z should respect 2R bound when projection is enabled (post-projection latents)
