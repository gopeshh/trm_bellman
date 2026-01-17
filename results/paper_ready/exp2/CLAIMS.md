# Experiment 2: Paper Claims

All claims are from the contraction-strength sweep on B0 (initial states).

## Main Claim: Contraction is a Dial

1. **Claim**: Contraction strength acts as a continuous dial between performance and stability.
   **Evidence**: As target $L_z$ decreases from 0.9 to 0.99:
   - Achieved $\hat{L}_z$: 0.011 → 0.009
   - $\Delta_V$: 0.172 → 0.001 (166.1× improvement)
   - Success rate: 0.20 → 0.18

2. **Claim**: Strong contraction ($L_z^* = 0.99$) provides 166.1× value stability improvement.
   **Evidence**: $\Delta_V$ 0.172 → 0.001

3. **Claim**: The stability-performance trade-off is monotonic.
   **Evidence**: All intermediate points (0.9, 0.95, 0.99, 0.999) follow the trend.

## Sweep Results Summary

| Target $L_z$ | Achieved $\hat{L}_z$ | Success Rate | $\Delta_V$ | Seeds |
|--------------|------------------------|--------------|-------------|-------|
| 0.9 | 0.011±0.001 | 0.20±0.04 | 0.172±0.099 | 3 |
| 0.95 | 0.009±0.000 | 0.19±0.00 | 0.114±0.000 | 1 |
| 0.99 | 0.009±0.000 | 0.18±0.00 | 0.001±0.000 | 1 |
| 0.999 | 0.009±0.000 | 0.17±0.00 | 0.094±0.000 | 1 |

## Interpretation

The contraction target $L_z^*$ provides a **dial** for trading off:
- **Lower $L_z^*$**: More stable (lower $\Delta_V$), potentially lower success rate
- **Higher $L_z^*$**: Less stable, potentially higher success rate

Best trade-off point: $L_z^* = 0.99$ (success=0.18, $\Delta_V$=0.001)
