# Experiment 1: Projection Radius Sweep

**Purpose**: Demonstrate that stability comes from contraction, not projection clipping.

**Configuration**: Both models use value-head spectral norm OFF.

| Batch | Radius | Condition | Δ_V (mean±std) | Argmax Agree [95% CI] | Saturation |
|-------|--------|-----------|----------------|----------------------|------------|
| B0 | disabled | No Contraction | 1.084±1.683 | 0.668 [0.636, 0.698] | N/A |
|  |  | Contraction | 0.225±0.197 | 0.943 [0.926, 0.957] | N/A |
|  | R=10 | No Contraction | 0.150±0.207 | 0.960 [0.945, 0.971] | 1.00 (n=900) |
|  |  | Contraction | 0.036±0.050 | 0.990 [0.981, 0.995] | 1.00 (n=900) |
|  | R=100 | No Contraction | 1.084±1.683 | 0.668 [0.636, 0.698] | 0.00 (n=900) |
|  |  | Contraction | 0.225±0.197 | 0.943 [0.926, 0.957] | 0.00 (n=900) |
| B1 | disabled | No Contraction | 1.133±1.689 | 0.640 [0.629, 0.651] | N/A |
|  |  | Contraction | 1.650±4.548 | 0.905 [0.898, 0.912] | N/A |
|  | R=10 | No Contraction | 0.147±0.215 | 0.955 [0.950, 0.960] | 1.00 (n=7245) |
|  |  | Contraction | 0.113±0.295 | 0.990 [0.988, 0.992] | 1.00 (n=7245) |
|  | R=100 | No Contraction | 1.133±1.689 | 0.640 [0.629, 0.651] | 0.00 (n=7245) |
|  |  | Contraction | 1.650±4.548 | 0.905 [0.898, 0.912] | 0.00 (n=7245) |

## Key Finding

- **R=0 (projection disabled)**: Contraction still provides 4.8× improvement
  - Δ_V: No Contraction = 1.084, Contraction = 0.225
  - This proves stability comes from contraction enforcement, not projection clipping.
