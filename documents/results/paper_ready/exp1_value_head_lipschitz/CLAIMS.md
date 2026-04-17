# Exp1 Value-Head Lipschitz Diagnostic: Claims

Generated: 2026-04-14T02:01:36.978274+00:00

This diagnostic measures finite-difference $\hat L_V$ on the theorem-facing Exp1
refreeze checkpoints (B0 batch, episodic-z, n_eval=8) across multiple perturbation scales.

## Central-scale summary

- **Model A' (no contraction)** at eps=1e-04: mean seedwise $\hat L_V$ = 0.095 (sample std 0.009, seed range [0.082, 0.106]).
- **Model B (contraction)** at eps=1e-04: mean seedwise $\hat L_V$ = 0.090 (sample std 0.025, seed range [0.056, 0.135]).

## Across-scale stability

- **Model A' (no contraction)** spans 0.095 to 0.192 over eps=1e-04..1e-05 (max/min ratio 2.01x).
- **Model B (contraction)** spans 0.085 to 0.200 over eps=1e-03..1e-05 (max/min ratio 2.35x).

## Scope

- This is a numerical diagnostic, not a certified global Lipschitz bound.
- It speaks only to the theorem-facing Exp1 B0 isolation setting, not the hard-4x4 anchor.
- The kill criterion for manuscript use is scale fragility by an order of magnitude or worse; this file reports the actual span needed to make that call.