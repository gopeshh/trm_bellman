# Experiment 1: Projection Radius Sweep (Appendix)

**Batch**: B1 (one-step successor closure)

**Note**: On B1 with R=0, contraction improves Δ_z and argmax agreement, but Δ_V shows higher variance and does not improve. This suggests that without projection, value estimates on successor states can exhibit increased instability even under contraction.

**Configuration**: Both models use value-head spectral norm OFF.

| Radius | Condition | Δ_V (mean±std) | Δ_z (mean±std) | Argmax [95% CI] | Sat. |
|--------|-----------|----------------|----------------|-----------------|------|
| disabled | No Contraction | 1.133±1.689 | 30.92±9.04 | 0.640 [0.629, 0.651] | N/A |
|  | Contraction | 1.650±4.548 | 16.62±5.28 | 0.905 [0.898, 0.912] | N/A |
| R=10 | No Contraction | 0.147±0.215 | 3.86±1.89 | 0.955 [0.950, 0.960] | 100% |
|  | Contraction | 0.113±0.295 | 1.36±0.36 | 0.990 [0.988, 0.992] | 100% |
| R=100 | No Contraction | 1.133±1.689 | 30.92±9.04 | 0.640 [0.629, 0.651] | 0% |
|  | Contraction | 1.650±4.548 | 16.62±5.28 | 0.905 [0.898, 0.912] | 0% |
