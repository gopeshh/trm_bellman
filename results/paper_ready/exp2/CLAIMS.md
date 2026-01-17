# Experiment 2: Paper Claims

All claims are from the contraction-strength sweep on B0 (initial states).

## Key Finding: Contraction Enforcement Saturates

1. **Observation**: All target $L_z$ values achieve similar measured Lipschitz constants.
   **Evidence**: Achieved $\hat{L}_z$ ranges from 0.217 to 0.242 (range: 0.024)
   **Interpretation**: The contraction enforcement mechanism saturates at $\hat{L}_z \approx 0.23$

2. **Observation**: Value stability ($\Delta_V$) has high variance across seeds.
   **Evidence**: Standard deviations range from 0.048 to 0.336
   **Best point**: target $L_z^* = 0.99$ with $\Delta_V = 0.067\pm0.057$

3. **Observation**: The monotonic dial relationship is NOT supported by this data.
   **Evidence**: $\Delta_V$ values are [0.3821410973866781, 0.14034074236949287, 0.06712594767411549, 0.14458184003829955]
   **Note**: Non-monotonic pattern suggests high seed variance dominates the target_Lz effect.

## Sweep Results Summary

| Target $L_z$ | Achieved $\hat{L}_z$ | Success Rate | $\Delta_V$ | Seeds |
|--------------|------------------------|--------------|-------------|-------|
| 0.9 | 0.242±0.003 | 0.20±0.04 | 0.382±0.336 | 3 |
| 0.95 | 0.217±0.005 | 0.22±0.02 | 0.140±0.093 | 3 |
| 0.99 | 0.234±0.002 | 0.21±0.02 | 0.067±0.057 | 3 |
| 0.999 | 0.230±0.010 | 0.21±0.03 | 0.145±0.048 | 3 |

## Scoped Interpretation

Given that achieved $\hat{L}_z$ is similar across all targets (~0.23), the primary effect of varying target $L_z^*$ is:
- **Indirect**: Different optimization trajectories lead to different models
- **High variance**: Seed-to-seed variation in $\Delta_V$ exceeds target-to-target variation

**Conservative claim**: Contraction enforcement achieves $\hat{L}_z \approx 0.23$ regardless of target, with $\Delta_V$ varying significantly (range: 0.067 to 0.382).
