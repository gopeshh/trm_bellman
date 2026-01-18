# Exp1 Lipschitz/Projection Diagnostic: Claims

**Generated**: 2026-01-18T11:10:21.197562

## Purpose

Cross-check diagnostic to verify Exp1 checkpoints exhibit the same projection dominance behavior observed in Exp2. This strengthens reviewer defensibility.

## Findings

### Projection Dominance at R=10

- **No Contraction**: projection_active_rate = 100%
- **Contraction**: projection_active_rate = 100%

### Projection Inactive at R=100

- **No Contraction**: projection_active_rate = 0%
- **Contraction**: projection_active_rate = 0%

## Consistency with Exp2

This diagnostic confirms:
1. At R=10, projection is highly active (~100%) for both Exp1 conditions, consistent with Exp2
2. At R=100, projection is mostly inactive, consistent with Exp2c-lite findings
3. The projection dominance effect is architecture-wide, not specific to the Exp2 contraction sweep

## Scoped Claim

> Exp1 checkpoints exhibit the same projection dominance pattern as Exp2: at R=10, projection is active ~100% of samples, masking underlying Lipschitz differences. This is consistent across both "No Contraction" and "Contraction" conditions.
