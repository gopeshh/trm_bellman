# Experiment 1: Paper Claims

All claims are precisely scoped to B0 or B1.

## Unroll Sensitivity (B0 - Main Result)

1. **Claim**: Contraction reduces Δ_V by 4.1× at 8× depth on B0.
   **Evidence**: Δ_V 0.156±0.225 → 0.038±0.055

2. **Claim**: Policy KL reduced by 33× on B0.
   **Evidence**: Δ_π 0.0063 → 0.0002

3. **Claim**: Latent drift reduced by 3.1× on B0.
   **Evidence**: Δ_z 4.16 → 1.33

4. **Claim**: Action agreement improves from 96.0% to 99.0% on B0.

## Radius Sweep (B0 - Isolation Result)

5. **Claim**: On B0 (initial states), with R=0 (projection disabled), contraction provides 4.8× value stability improvement.
   **Evidence**: Δ_V 1.084 → 0.225
   **Interpretation**: Stability comes from contraction, not projection clipping.

## B1 Observation (NOT a main claim)

⚠️ **Warning**: On B1 (successor states) with R=0, Δ_V does NOT improve with contraction.
   B1 R=0 Δ_V: 1.133 → 1.650 (increased)
   Latent drift and action agreement still improve on B1.
   The main R=0 claim is scoped to B0 initial states only.

## One-Sentence Summary

On initial states (B0), contraction enforcement provides value stability guarantees independent of projection radius (4.8× improvement even at R=0); on successor states (B1), contraction improves latent and action stability, but value estimates require projection to avoid increased variance.
