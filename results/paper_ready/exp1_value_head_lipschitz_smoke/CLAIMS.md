# Exp1 Value-Head Lipschitz Diagnostic: Claims

Generated: 2026-04-14T02:00:08.020174+00:00

This diagnostic measures finite-difference $\hat L_V$ on the theorem-facing Exp1
refreeze checkpoints (B0 batch, episodic-z, n_eval=8) across multiple perturbation scales.

## Central-scale summary

- **Model A' (no contraction)** at eps=1e-04: mean seedwise $\hat L_V$ = 0.082 (sample std 0.015, seed range [0.072, 0.093]).
- **Model B (contraction)** at eps=1e-04: mean seedwise $\hat L_V$ = 0.079 (sample std 0.017, seed range [0.067, 0.091]).

## Across-scale stability

- **Model A' (no contraction)** spans 0.075 to 0.082 over eps=1e-03..1e-04 (max/min ratio 1.10x).
- **Model B (contraction)** spans 0.079 to 0.084 over eps=1e-04..1e-03 (max/min ratio 1.06x).

## Scope

- This is a numerical diagnostic, not a certified global Lipschitz bound.
- It speaks only to the theorem-facing Exp1 B0 isolation setting, not the hard-4x4 anchor.
- The kill criterion for manuscript use is scale fragility by an order of magnitude or worse; this file reports the actual span needed to make that call.