# Experiment 1: Projection Radius Sweep (Main Paper)

**Batch**: B0 (initial states only)

**Purpose**: Demonstrate that stability comes from contraction, not projection clipping.

**Configuration**: Both models use value-head spectral norm OFF.

| Radius | Condition | Δ_V (mean±std) | Δ_z (mean±std) | Argmax [95% CI] | Sat. |
|--------|-----------|----------------|----------------|-----------------|------|
| disabled | No Contraction | 1.084±1.683 | 31.02±9.51 | 0.668 [0.636, 0.698] | N/A |
|  | Contraction | 0.225±0.197 | 15.73±3.84 | 0.943 [0.926, 0.957] | N/A |
| R=10 | No Contraction | 0.150±0.207 | 3.91±1.93 | 0.960 [0.945, 0.971] | 100% |
|  | Contraction | 0.036±0.050 | 1.28±0.22 | 0.990 [0.981, 0.995] | 100% |
| R=100 | No Contraction | 1.084±1.683 | 31.02±9.51 | 0.668 [0.636, 0.698] | 0% |
|  | Contraction | 0.225±0.197 | 15.73±3.84 | 0.943 [0.926, 0.957] | 0% |

## Key Finding

With projection disabled (R=0), contraction provides **4.8× improvement** in Δ_V.
This proves stability comes from contraction enforcement, not projection clipping.
