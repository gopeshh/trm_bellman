# Experiment 2: Paper Claims

All claims are from the contraction-strength sweep on B0 (initial states).

## Key Finding: Contraction Dial Does Not Work (Projection Dominates)

### Root Cause Analysis (Exp2b Diagnostics)

The diagnostics reveal WHY achieved $\hat{L}_z$ saturates at ~0.23:

| Metric | Finding |
|--------|---------|
| $\hat{L}_{pre-proj}$ | Varies: 0.69-0.77 across target $L_z^*$ values |
| $\hat{L}_{post-proj}$ at R=10 | Saturates: 0.22-0.24 regardless of target |
| Projection Active Rate | **100%** at R=10 |
| $\|f(z)\|$ before projection | ~45 (vs radius R=10) |

**Diagnosis: Projection to ball of radius R=10 dominates and masks the underlying network contraction.**

The spectral-norm-clamped networks achieve $\hat{L}_{pre-proj} \in [0.69, 0.77]$, which varies with target $L_z^*$. However, latent norm outputs ($\|f(z)\| \approx 45$) exceed the ball radius R=10 by ~35×, causing 100% projection activity. This projection compresses all outputs to R=10, artificially lowering the measured Lipschitz constant to ~0.23.

### Implication

The "contraction dial" is **not controllable** with current hyperparameters because:
1. Projection dominates at R=10 (100% active)
2. Post-projection $\hat{L}_z$ saturates at ~0.23 regardless of spectral norm settings

### Original Observations

1. **Observation**: All target $L_z$ values achieve similar measured Lipschitz constants.
   **Evidence**: Achieved $\hat{L}_z$ ranges from 0.217 to 0.242 (range: 0.024)
   **Root cause**: Projection to R=10 dominates (see diagnostics above)

2. **Observation**: Value stability ($\Delta_V$) has high variance across seeds.
   **Evidence**: Standard deviations range from 0.048 to 0.336
   **Best point**: target $L_z^* = 0.99$ with $\Delta_V = 0.067\pm0.057$

3. **Observation**: The monotonic dial relationship is NOT supported by this data.
   **Evidence**: $\Delta_V$ values are [0.382, 0.140, 0.067, 0.145]
   **Note**: Non-monotonic pattern suggests high seed variance dominates.

## Sweep Results Summary

| Target $L_z$ | Achieved $\hat{L}_z$ | Success Rate | $\Delta_V$ | Seeds |
|--------------|------------------------|--------------|-------------|-------|
| 0.9 | 0.242±0.003 | 0.20±0.04 | 0.382±0.336 | 3 |
| 0.95 | 0.217±0.005 | 0.22±0.02 | 0.140±0.093 | 3 |
| 0.99 | 0.234±0.002 | 0.21±0.02 | 0.067±0.057 | 3 |
| 0.999 | 0.230±0.010 | 0.21±0.03 | 0.145±0.048 | 3 |

## Recommendations for Future Work

1. **Increase R or disable projection** to allow the dial to function
2. **Evaluate at R=∞** to measure true network contraction ($\hat{L}_{pre-proj}$)
3. **Consider alternative architectures** where latent norms stay within a reasonable ball

## Conservative Paper Claim

**The contraction enforcement mechanism achieves $\hat{L}_z \approx 0.23$ at R=10 regardless of target $L_z^*$, due to latent-space projection dominating the measured Lipschitz constant. The underlying spectral-norm-clamped networks achieve $\hat{L}_{pre-proj} \in [0.69, 0.77]$, indicating that the dial can work if projection is relaxed.**
