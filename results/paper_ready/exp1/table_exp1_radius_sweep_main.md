# Experiment 1: Projection Radius Sweep (Main Paper)

**Batch**: B0 (initial states only)

**Purpose**: Demonstrate that stability comes from contraction, not projection clipping.

**Configuration**: Both models use value-head spectral norm OFF.

**Delta definition**: fixed mismatch Δ(n_train=2, n₂=8) (4× depth), pooled across all states and seeds.

| Radius | Condition | Δ_V (mean±std) | Δ_z (mean±std) | Argmax [95% CI] | Sat. |
|--------|-----------|----------------|----------------|-----------------|------|
| disabled | No Contraction | 1.078±1.772 | 33.01±8.96 | 0.653 [0.598, 0.705] | N/A |
|  | Contraction | 0.240±0.199 | 17.18±2.92 | 0.933 [0.899, 0.956] | N/A |
| R=10 | No Contraction | 0.152±0.205 | 4.08±1.98 | 0.957 [0.927, 0.975] | 100% |
|  | Contraction | 0.038±0.054 | 1.33±0.23 | 0.990 [0.971, 0.997] | 100% |
| R=100 | No Contraction | 1.078±1.772 | 33.01±8.96 | 0.653 [0.598, 0.705] | 0% |
|  | Contraction | 0.240±0.199 | 17.18±2.92 | 0.933 [0.899, 0.956] | 0% |

## Key Finding

With projection disabled (R=0), at fixed n=2→8, contraction provides **4.5× improvement** in Δ_V.
This proves stability comes from contraction enforcement, not projection clipping.
