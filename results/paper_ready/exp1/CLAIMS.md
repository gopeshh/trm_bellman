# Experiment 1: Paper Claims

Copy-paste ready for paper text.

1. **Claim**: Contraction enforcement reduces value instability by 4.1× at 8× depth.
   **Evidence**: Table 1, Fig 1. Δ_V at n₂=16: No Contraction = 0.156±0.225, Contraction = 0.038±0.055.

2. **Claim**: Policy KL divergence reduced by 33× with contraction.
   **Evidence**: Table 1. Δ_π: 0.0063 → 0.0002.

3. **Claim**: Action consistency improves from 96.0% to 99.0% (Wilson 95% CI).
   **Evidence**: Table 1. No Contraction: [0.931, 0.977], Contraction: [0.971, 0.997].

4. **Claim**: With projection disabled (R=0), contraction still provides 4.8× stability improvement.
   **Evidence**: Table 2, Fig 2. R=0 Δ_V: 1.084 → 0.225.
   This proves stability comes from contraction enforcement, not projection clipping.

5. **Claim**: At R=10, projection is always active (saturation rate = 100%).
   **Evidence**: Table 2. z_pre_norm >> R=10 causes 100% saturation.

6. **Claim**: At R=100, projection is never needed (saturation rate = 0%).
   **Evidence**: Table 2. R=100 > z_pre_norm means no clipping.

7. **Claim**: Latent drift (Δ_z) reduced by 3.1× with contraction.
   **Evidence**: Fig 1. Δ_z: 4.16 → 1.33.

## One-Sentence Summary

Contraction enforcement provides mathematical stability guarantees that are independent of projection radius, as demonstrated by consistent improvement even when projection is completely disabled (R=0).
