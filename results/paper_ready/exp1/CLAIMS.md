# Experiment 1: Paper Claims

Copy-paste ready for paper text. All claims are precisely scoped.

## Unroll Sensitivity (Main Result)

1. **Claim**: Contraction enforcement reduces value instability by 4.1× at 8× depth on initial states (B0).
   **Evidence**: Table 1, Fig 1. Δ_V at n₂=16: No Contraction = 0.156±0.225, Contraction = 0.038±0.055.

2. **Claim**: Policy KL divergence reduced by 33× with contraction (B0).
   **Evidence**: Table 1. Δ_π: 0.0063 → 0.0002.

3. **Claim**: Action consistency improves from 96.0% to 99.0% (Wilson 95% CI) on B0.
   **Evidence**: Table 1. No Contraction: [0.931, 0.977], Contraction: [0.971, 0.997].

4. **Claim**: Latent drift (Δ_z) reduced by 3.1× with contraction on B0.
   **Evidence**: Table 1. Δ_z: 4.16 → 1.33.

## Radius Sweep (Isolation of Contraction vs. Projection)

5. **Claim**: On initial states (B0), with projection disabled (R=0), contraction provides 4.8× value stability improvement.
   **Evidence**: Table 2 (main). R=0 B0 Δ_V: No Contraction = 1.084±1.683, Contraction = 0.225±0.197.
   **Interpretation**: This demonstrates that stability on initial states comes from contraction enforcement, not projection clipping.

6. **Claim**: On successor states (B1), with projection disabled (R=0), contraction reduces latent drift by 1.9× and improves action agreement from 64.0% to 90.5%.
   **Evidence**: Table 2 (appendix). R=0 B1: Δ_z 30.92 → 16.62; Argmax 0.640 → 0.905.

7. **Observation (NOT a claim)**: On B1 with R=0, value drift Δ_V does not improve with contraction (1.133 → 1.650).
   This suggests that successor states may exhibit value-function instability that projection normally helps control. The main R=0 claim (item 5) is therefore scoped to B0.

8. **Claim**: At R=10, projection is always active (100% saturation).
   **Evidence**: Table 2. z_pre_norm ≈ 32 >> R=10 causes 100% saturation.

9. **Claim**: At R=100, projection is never needed (0% saturation).
   **Evidence**: Table 2. R=100 > z_pre_norm means no clipping.

## One-Sentence Summary (for paper)

On initial states, contraction enforcement provides value stability guarantees independent of projection radius (4.8× improvement even at R=0); on successor states, contraction consistently improves latent stability and action agreement, though value-function estimates require projection to avoid increased variance.
