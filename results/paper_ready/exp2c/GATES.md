# Exp2c Decision Gates Analysis

**Generated**: 2026-01-17T22:48:30.361552

## Gate Definitions

- **G1 (Projection Dominance)**: projection_active_rate < 20%
- **G2 (Dial Range)**: L_preproj spread >= 0.08
- **G3 (Stability Linkage)**: Lower L_preproj ⇒ lower ΔV/Δπ, higher argmax agreement

## Results

### G1 (R=100.0): ✅ PASSED

Projection active rate: 0.0% (threshold: <20%)

### G2 (R=100.0): ❌ FAILED

L_preproj spread: 0.064 (threshold: >=0.08)

### G1 (R=∞): ✅ PASSED

Projection active rate: 0.0% (threshold: <20%)

### G2 (R=∞): ❌ FAILED

L_preproj spread: 0.064 (threshold: >=0.08)

## Overall Verdict

**Failed gates**: G2 (R=100.0), G2 (R=∞)

### Path: B (Negative Result)

Since G2 failed, the spectral-norm "dial" has insufficient range to control contraction. The experiment proceeds on Path B (negative result).

### G3 Analysis (Informational - since G2 failed)

Even though G2 failed, we can examine the stability linkage:

| Target $L_z^*$ | $\hat{L}_{preproj}$ | ΔV (2→16) | Δπ (2→16) | Argmax Agree |
|----------------|---------------------|-----------|-----------|--------------|
| 0.9 | 0.473 | 10.5 | 0.036 | 90.7% |
| 0.95 | 0.409 | 10.0 | 0.057 | 89.3% |
| 0.99 | 0.448 | 7.7 | 0.044 | 88.0% |
| 0.999 | 0.440 | 10.0 | 0.040 | 89.7% |

**Correlation analysis** (at R=disabled):
- ΔV vs L_preproj: No clear monotonic relationship
- Δπ vs L_preproj: Weak positive trend
- Argmax agree vs L_preproj: Slight positive trend

**G3 verdict**: INCONCLUSIVE - insufficient dial range prevents meaningful correlation analysis.

## Key Unexpected Finding

**Projection at R=10 provides substantial stability benefit:**

| Metric | R=10 (100% proj) | R=disabled (0% proj) |
|--------|------------------|----------------------|
| ΔV range | 0.5 - 2.2 | 7.7 - 14.8 |
| Argmax agree | 97-99% | 88-91% |
| $\hat{L}_{postproj}$ | ~0.23 | ~0.44 |

This suggests **latent-ball projection is the primary stabilization mechanism**, not spectral-norm contraction.

## Recommendations

1. **Do not claim dial control**: The spectral-norm mechanism does not provide controllable contraction
2. **Can claim projection stabilization**: Strong evidence that R=10 projection improves unroll stability
3. **Future work**: Investigate alternative contraction mechanisms (e.g., regularization loss, different architectures)

