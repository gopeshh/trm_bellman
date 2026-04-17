# Unroll Sensitivity Comparison: Model A vs Model B

**Comparison depth**: n_train (4) vs 4×n_train (16)

## Configuration Comparison

| Setting | Model A | Model B |
|---------|---------|--------|
| enable_contraction | True | True |
| target_Lz | 0.9 | 0.9 |
| disable_value_head_norm | False | False |
| episodic_latent | True | True |

## Metrics Summary

### Batch B0

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 4.0834 | 0.0823 | -4.0011 |
| Δ_V | median | 2.9829 | 0.0544 | -2.9285 |
| Δ_V | p99 | 13.9917 | 0.4932 | -13.4985 |
| Δ_pi | mean | 0.2478 | 0.0058 | -0.2420 |
| Δ_pi | median | 0.1094 | 0.0019 | -0.1075 |
| Δ_pi | p99 | 1.2937 | 0.0515 | -1.2422 |
| Δ_z | mean | 34.9098 | 7.8133 | -27.0965 |
| Δ_z | median | 35.0722 | 7.1957 | -27.8765 |
| Δ_z | p99 | 45.3141 | 13.4335 | -31.8806 |
| argmax_agree | rate | 0.7300 | 0.9500 | +0.2200 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |

### Batch B1

| Metric | Statistic | Model A | Model B | Delta |
|--------|-----------|---------|---------|-------|
| Δ_V | mean | 5.0060 | 0.7772 | -4.2289 |
| Δ_V | median | 3.4663 | 0.0671 | -3.3991 |
| Δ_V | p99 | 19.7176 | 7.9980 | -11.7196 |
| Δ_pi | mean | 0.3505 | 0.0110 | -0.3395 |
| Δ_pi | median | 0.1264 | 0.0033 | -0.1231 |
| Δ_pi | p99 | 2.8752 | 0.1196 | -2.7556 |
| Δ_z | mean | 35.0829 | 8.6671 | -26.4158 |
| Δ_z | median | 36.2218 | 8.3242 | -27.8977 |
| Δ_z | p99 | 48.4198 | 15.4559 | -32.9640 |
| argmax_agree | rate | 0.7029 | 0.9582 | +0.2552 |
| saturation | rate | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **Lower Δ_V, Δ_π** indicates more stable predictions across unroll depths
- **Higher argmax_agree** means policy decisions are more consistent
- If Model B has contraction enabled, expect lower sensitivity metrics
