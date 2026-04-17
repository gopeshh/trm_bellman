# Experiment 1: Paper Claims

All claims are scoped to the refrozen 10-seed Exp1 protocol.
B0 remains the main-text anchor; B1 is appendix-only supporting context.

## Unroll Sensitivity (B0 - Main Result)

1. **Claim**: At fixed n=2→8 mismatch on B0, contraction reduces Δ_V by 2.6×.
   **Evidence**: Δ_V 0.166±0.287 → 0.064±0.060

2. **Observation**: Δ_π is near zero for both models at this comparison and is not a primary discriminator.
   **Evidence**: Δ_π 0.0002 → 0.0001 (2.4× ratio on small absolute values)

3. **Claim**: Latent drift is reduced by 2.5× on B0.
   **Evidence**: Δ_z 4.26 → 1.67

4. **Claim**: Action agreement improves from 90.2% to 95.5% on B0.

## Radius Sweep (B0 - Isolation Result)

5. **Claim**: On B0 (initial states), with R=0 (projection disabled), at fixed 4× mismatch (n=2→8), contraction provides 2.1× value stability improvement.
   **Evidence**: Δ_V (pooled over all states and seeds) 2.275 → 1.097
   **Interpretation**: Contraction contributes a projection-independent stability benefit.

6. **Observation**: Projection remains the dominant stabilizer on top of contraction.
   **Evidence**: On B0, moving from R=0 to R=10 reduces Δ_V 2.275 → 0.166 for No Contraction (13.7×) and 1.097 → 0.064 for Contraction (17.1×).
   **Saturation**: R=10 is active 100% of the time (100% / 100%); R=100 never fires (0% / 0%).

## B1 Observation (Appendix-Only Supporting Context)

⚠️ **Warning**: B1 remains appendix-only and should not replace the B0 main-text anchor.
   Under the refrozen 10-seed protocol, B1 is directionally consistent with B0 even at R=0: Δ_V 2.230 → 1.064, Δ_z 35.28 → 22.70, argmax 47.2% → 64.7%.
   Absolute variability is still higher than on B0, so the main claim remains scoped to B0 initial states.

## One-Sentence Summary

On initial states (B0), contraction enforcement provides a consistent projection-independent stability benefit independent of projection radius (2.1× improvement at fixed n=2→8 even at R=0); projection then provides an additional order-of-magnitude reduction on top of that effect; B1 follows the same qualitative direction but remains supporting appendix evidence rather than the main-text claim anchor.
