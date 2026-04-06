# Experiment 1: Projection Radius Sweep (Appendix)

**Batch**: B1 (one-step successor closure)

**Note**: On B1 with R=0, contraction improves Δ_z and argmax agreement, but Δ_V shows higher variance and does not improve. This suggests that without projection, value estimates on successor states can exhibit increased instability even under contraction.

**Configuration**: Both models use value-head spectral norm OFF.

**Delta definition**: fixed mismatch Δ(n_train=2, n₂=8) (4× depth), pooled across all states and seeds.

| Radius | Condition | Δ_V (mean±std) | Δ_z (mean±std) | Argmax [95% CI] | Sat. |
|--------|-----------|----------------|----------------|-----------------|------|
| disabled | No Contraction | 1.169±1.756 | 32.89±8.63 | 0.618 [0.598, 0.637] | N/A |
|  | Contraction | 1.848±4.987 | 18.22±4.85 | 0.896 [0.883, 0.908] | N/A |
| R=10 | No Contraction | 0.152±0.226 | 4.02±1.95 | 0.951 [0.942, 0.959] | 100% |
|  | Contraction | 0.120±0.312 | 1.41±0.38 | 0.990 [0.985, 0.993] | 100% |
| R=100 | No Contraction | 1.169±1.756 | 32.89±8.63 | 0.618 [0.598, 0.637] | 0% |
|  | Contraction | 1.848±4.987 | 18.22±4.85 | 0.896 [0.883, 0.908] | 0% |
