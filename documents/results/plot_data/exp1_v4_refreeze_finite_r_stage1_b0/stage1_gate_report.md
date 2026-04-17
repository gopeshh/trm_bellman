# STAGE1 Gate Report

- Overall: `PASS`

| Gate | Status | Details |
|---|---|---|
| data_integrity | PASS | All expected fixed-pair per-state CSVs were found. |
| schema_sanity | PASS | All analyzed CSVs expose the preferred `saturated` schema. |
| primary_endpoint_separation | PASS | No Contraction: R=10 vs R=100 Delta_V 0.1658 < 2.2751, Delta_z 4.2616 < 36.1060, Argmax 0.9020 > 0.4730; Contraction: R=10 vs R=100 Delta_V 0.0643 < 1.0972, Delta_z 1.6726 < 22.5728, Argmax 0.9550 > 0.6470 |
| transition_localization | PASS | R=10 z_pre_norm width p95-p05 = 2.6040 (p05=30.5872, p95=33.1912) |
| inactive_plateau_check | PASS | No Contraction off vs R=100 diffs (Delta_V=0.000000, Delta_z=0.000000, Argmax=0.000000); Contraction off vs R=100 diffs (Delta_V=0.000000, Delta_z=0.000000, Argmax=0.000000) |
| intermediate_regime_exists | PASS | No Contraction intermediate hits: R=20, R=30; Contraction intermediate hits: R=20, R=30, R=32, R=34 |
| directional_coherence | PASS | No Contraction saturation means over finite R: 1.000, 1.000, 1.000, 1.000, 1.000, 0.200, 0.000, 0.000; Contraction saturation means over finite R: 1.000, 1.000, 1.000, 1.000, 1.000, 0.200, 0.000, 0.000 |

## Transition Window

```json
{
  "source_radius": 10.0,
  "n": 2000,
  "min": 30.5871,
  "p05": 30.5872,
  "p50": 31.9551,
  "p95": 33.1912,
  "max": 33.1913,
  "mean": 31.8364749
}
```

## Notes

- Statistics are computed over seed-level means, not pooled state rows.
- proj_disp_proxy = z_pre_norm - z_post_norm is a scalar proxy based on norms, not the true displacement norm.
- Rows with saturated < 0 are treated as mechanism-metric N/A.
