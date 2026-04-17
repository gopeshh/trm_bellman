# Exp2 Final: Paper Claims

**Generated**: 2026-04-16T09:50:06.356173
**Status**: Path B (Negative Result for Dial + Positive Result for Projection)

## Claim 1: Dial Failure (Negative Result)

**Statement**: In this TRM setup, targeting spectral-norm-based contraction does NOT provide a reliable 'stability dial'.

**Evidence**:
- $\hat{L}_{preproj}$ varies only weakly across target $L_z$ values
- Spread: 0.064 (threshold for "dial works": $\geq$ 0.08)
- Ordering is non-monotonic: 0.9 → 0.95 → 0.99 → 0.999 yields $\hat{L}_{preproj}$ = 0.473 → 0.409 → 0.448 → 0.440
- At R=10: projection dominates (100% active), $\hat{L}_{postproj}$ saturates at ~0.23

**Scope**:
- Evaluated on B0 (initial states)
- target $L_z \in$ {0.9, 0.95, 0.99, 0.999}
- 3 seeds per condition

## Claim 2: Projection Stabilizes (Positive Result)

**Statement**: Latent-ball projection is a strong stabilizer in this architecture.

**Evidence**:
- $\Delta V$ (n=2 → n=16): improves from 7.7–10.5 (R=disabled) to 0.5–2.2 (R=10)
  - Mean improvement: 9.5 → 1.3 (~7× reduction)
- Argmax agreement: improves from 88–91% to 97–100%
  - Mean improvement: +9.4 percentage points

**Scope**:
- Evaluated on B0 (initial states)
- Comparison: R=10 (projection ON, 100% active) vs R=disabled (projection OFF, 0% active)
- Mismatch protocol: n_train=2, n_eval=16

## Explicit Non-Claims

1. **NO monotonicity claim**: The dial does NOT produce monotonic $\hat{L}_z$ vs target $L_z$
2. **NO dial control claim**: The spectral-norm mechanism does NOT provide controllable contraction
3. **NO projection-vs-R monotonicity claim**: We did not sweep R values; only compared R=10 vs R=disabled

## Audit Requirements

This CLAIMS.md must be verified against summary.json by the audit script.
