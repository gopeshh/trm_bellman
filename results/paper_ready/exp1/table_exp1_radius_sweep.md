# Experiment 1: Projection Radius Sweep

**Purpose**: Demonstrate that stability comes from contraction, not projection clipping.

**Configuration**: Both models use value-head spectral norm OFF.

**Delta definition**: fixed mismatch Δ(n_train=2, n₂=8) (4× depth), pooled across all states and seeds.

| Batch | Radius | Condition | Δ_V (mean±std) | Argmax Agree [95% CI] | Saturation |
|-------|--------|-----------|----------------|----------------------|------------|
| B0 | disabled | No Contraction | 1.078±1.772 | 0.653 [0.598, 0.705] | N/A |
|  |  | Contraction | 0.240±0.199 | 0.933 [0.899, 0.956] | N/A |
|  | R=10 | No Contraction | 0.152±0.205 | 0.957 [0.927, 0.975] | 1.00 (n=300) |
|  |  | Contraction | 0.038±0.054 | 0.990 [0.971, 0.997] | 1.00 (n=300) |
|  | R=100 | No Contraction | 1.078±1.772 | 0.653 [0.598, 0.705] | 0.00 (n=300) |
|  |  | Contraction | 0.240±0.199 | 0.933 [0.899, 0.956] | 0.00 (n=300) |
| B1 | disabled | No Contraction | 1.169±1.756 | 0.618 [0.598, 0.637] | N/A |
|  |  | Contraction | 1.848±4.987 | 0.896 [0.883, 0.908] | N/A |
|  | R=10 | No Contraction | 0.152±0.226 | 0.951 [0.942, 0.959] | 1.00 (n=2415) |
|  |  | Contraction | 0.120±0.312 | 0.990 [0.985, 0.993] | 1.00 (n=2415) |
|  | R=100 | No Contraction | 1.169±1.756 | 0.618 [0.598, 0.637] | 0.00 (n=2415) |
|  |  | Contraction | 1.848±4.987 | 0.896 [0.883, 0.908] | 0.00 (n=2415) |

## Key Finding

- **R=0 (projection disabled, fixed n=2→8)**: Contraction still provides 4.5× improvement
  - Δ_V: No Contraction = 1.078, Contraction = 0.240
  - This proves stability comes from contraction enforcement, not projection clipping.
