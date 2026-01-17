# Experiment 1: Unroll Sensitivity (Mismatch Drift)

**Comparison**: n_train=2 vs n₂=16 (8× depth)

**Configuration**: Both models use value-head spectral norm OFF. Only difference is `enable_contraction`.

**Seeds**: 41, 42, 43

| Batch | Condition | Δ_V (mean±std) | Δ_π (mean±std) | Δ_z (mean±std) | Argmax Agree [95% CI] |
|-------|-----------|----------------|----------------|----------------|----------------------|
| B0 (initial) | No Contraction | 0.156±0.225 | 0.0063±0.0124 | 4.16±2.04 | 0.960 [0.931, 0.977] |
|  | Contraction | 0.038±0.055 | 0.0002±0.0002 | 1.33±0.23 | 0.990 [0.971, 0.997] |
| B1 (successors) | No Contraction | 0.156±0.237 | 0.0064±0.0140 | 4.09±1.99 | 0.953 [0.944, 0.961] |
|  | Contraction | 0.120±0.314 | 0.0003±0.0006 | 1.41±0.38 | 0.990 [0.985, 0.993] |

## Key Finding

- **Value stability improvement (B0)**: 4.1× (Δ_V: 0.156 → 0.038)
- **Action consistency (B0)**: 96.0% → 99.0%
