# Exp2 Contraction Saturation Diagnostics

**Generated**: 2026-01-17T16:52:58.016632

## Summary

This diagnostic investigates WHY achieved_Lz saturates at ~0.23 across all target_Lz values.

## Key Metrics

| target_Lz | R | L_preproj | L_postproj | Proj Active | Clamp Active |
|-----------|---|-----------|------------|-------------|--------------|
| 0.9 | ∞ | 0.773 | 0.773 | 0.0% | 0.0% |
| 0.9 | 10 | 0.773 | 0.242 | 100.0% | 0.0% |
| 0.9 | 100 | 0.774 | 0.774 | 0.0% | 0.0% |
| 0.9 | 1000 | 0.774 | 0.774 | 0.0% | 0.0% |
| 0.95 | ∞ | 0.696 | 0.696 | 0.0% | 0.0% |
| 0.95 | 10 | 0.696 | 0.218 | 100.0% | 0.0% |
| 0.95 | 100 | 0.696 | 0.696 | 0.0% | 0.0% |
| 0.95 | 1000 | 0.696 | 0.696 | 0.0% | 0.0% |
| 0.99 | ∞ | 0.748 | 0.748 | 0.0% | 0.0% |
| 0.99 | 10 | 0.748 | 0.234 | 100.0% | 0.0% |
| 0.99 | 100 | 0.748 | 0.748 | 0.0% | 0.0% |
| 0.99 | 1000 | 0.748 | 0.748 | 0.0% | 0.0% |
| 0.999 | ∞ | 0.737 | 0.737 | 0.0% | 0.0% |
| 0.999 | 10 | 0.737 | 0.230 | 100.0% | 0.0% |
| 0.999 | 100 | 0.737 | 0.737 | 0.0% | 0.0% |
| 0.999 | 1000 | 0.738 | 0.738 | 0.0% | 0.0% |

## Decision Gate Analysis

### R=10 Analysis

- L_preproj range across targets: 0.077
- L_postproj range across targets: 0.024
- Mean projection active rate: 100.0%

**DIAGNOSIS: Projection dominates** - L_preproj varies but L_postproj saturates with high projection rate.

**RECOMMENDATION**: Increase R or disable projection for dial experiments.

## Norm Distributions (R=10)

| target_Lz | ||z|| mean | ||f(z)|| pre | ||f(z)|| post | Beyond R |
|-----------|------------|--------------|---------------|----------|
| 0.9 | 14.14 | 45.25 | 14.14 | 35.25 |
| 0.95 | 14.14 | 45.25 | 14.14 | 35.25 |
| 0.99 | 14.14 | 45.25 | 14.14 | 35.25 |
| 0.999 | 14.14 | 45.25 | 14.14 | 35.25 |
