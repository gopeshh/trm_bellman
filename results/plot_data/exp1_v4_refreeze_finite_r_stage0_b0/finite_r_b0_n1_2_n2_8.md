# stage0 Fixed-Pair Finite-R Summary

- Batch: `b0`
- Fixed pair: `n1=2`, `n2=8`
- Radii: `proj. off, R=10, R=100`

- Statistics are computed over seed-level means, not pooled state rows.
- proj_disp_proxy = z_pre_norm - z_post_norm is a scalar proxy based on norms, not the true displacement norm.
- Rows with saturated < 0 are treated as mechanism-metric N/A.
- matplotlib is unavailable in the current Python environment; figure generation was skipped.

## Delta_V

| Radius | No Contraction | Contraction |
|---|---|---|
| proj. off | 2.2751±1.9854 (n=10) | 1.0972±0.3055 (n=10) |
| R=10 | 0.1658±0.0637 (n=10) | 0.0643±0.0264 (n=10) |
| R=100 | 2.2751±1.9854 (n=10) | 1.0972±0.3055 (n=10) |

## Delta_z

| Radius | No Contraction | Contraction |
|---|---|---|
| proj. off | 36.1060±4.2704 (n=10) | 22.5728±1.9037 (n=10) |
| R=10 | 4.2616±0.9092 (n=10) | 1.6726±0.1711 (n=10) |
| R=100 | 36.1060±4.2704 (n=10) | 22.5728±1.9037 (n=10) |

## Argmax Agree

| Radius | No Contraction | Contraction |
|---|---|---|
| proj. off | 0.4730±0.1116 (n=10) | 0.6470±0.0833 (n=10) |
| R=10 | 0.9020±0.0391 (n=10) | 0.9550±0.0255 (n=10) |
| R=100 | 0.4730±0.1116 (n=10) | 0.6470±0.0833 (n=10) |

## Saturated Rate

| Radius | No Contraction | Contraction |
|---|---|---|
| proj. off | N/A | N/A |
| R=10 | 1.0000±0.0000 (n=10) | 1.0000±0.0000 (n=10) |
| R=100 | 0.0000±0.0000 (n=10) | 0.0000±0.0000 (n=10) |

## proj_disp_proxy

| Radius | No Contraction | Contraction |
|---|---|---|
| proj. off | N/A | N/A |
| R=10 | 21.8365±0.7753 (n=10) | 21.8364±0.7753 (n=10) |
| R=100 | 0.0000±0.0000 (n=10) | 0.0000±0.0000 (n=10) |

