# UPI-TRM Theory Notes

**Date**: December 27, 2024
**Author**: Internal analysis of CPI implementation

---

## CPI Mixture Policy and Importance Sampling Weights

### The Question

A concern was raised that the `policy_update()` method in `rl/upi_trm_trainer.py` computes the policy gradient using data collected by a mixture policy:

```
π_mix = (1-α)π_old + α·π_cand
```

But calculates the loss as if the data were on-policy for `π_cand`. Shouldn't this require importance sampling (IS) weights?

```
ρ = π_cand(a|s) / π_mix(a|s)
```

### The Answer: IS Weights Are NOT Required (in theory_exact_mixture=True mode)

This is **not** a bug. The CPI framework intentionally does not use IS weights because:

1. **The deployed policy IS the mixture**: In true CPI, we don't try to make `π_cand` behave like `π_mix`. Instead:
   - We train `π_cand` with standard policy gradient on data from `π_mix`
   - We keep `π_old` fixed (not updated)
   - We deploy the explicit mixture `π_mix = (1-α)π_old + α·π_cand` as the actual behavior policy

2. **CPI's guarantee is about the deployed mixture**: The improvement bound:
   ```
   V^{π_new} ≥ V^{π_old} - O(α·ε_A)
   ```
   applies to the **deployed mixture policy `π_mix`**, not to `π_cand` in isolation.

3. **The "bias" is intentional**: When we update `π_cand` using data from `π_mix`, we're improving `π_cand` relative to the current state distribution. CPI theory accounts for this—the small mixture coefficient α limits the distribution shift.

4. **IS weights would be counterproductive**: If we applied IS weights to correct for the `π_mix` vs `π_cand` mismatch, we'd be training `π_cand` to be on-policy for itself, which defeats the purpose of the conservative mixture.

### Three CPI Modes in Implementation

The implementation (`rl/upi_trm_trainer.py:policy_update()`) supports three modes:

| Mode | Config | Theory Status | Behavior |
|------|--------|---------------|----------|
| **Theory-Exact** | `theory_exact_mixture=True` | CPI bound applies | `π_old` is **not updated**; behavior policy is explicit mixture from `_mixed_policy_dist()` |
| **Distillation** | `distill_mixture_policy=True` | Heuristic | Mixture is distilled into `π_old` via KL minimization; introduces projection error |
| **Default** | Both `False` | Heuristic | Parameter-space interpolation `param.lerp_(candidate, α)`; NOT equivalent to probability mixing |

### Code Locations

- **Mode selection**: `rl/upi_trm_trainer.py:1085-1110`
- **Explicit mixture**: `rl/upi_trm_trainer.py:222-284` (`_mixed_policy_dist()`)
- **Parameter interpolation**: `rl/upi_trm_trainer.py:286-327` (`_sync_policy_old_towards_candidate()`)
- **Config options**: `rl/config.py:theory_exact_mixture`, `rl/config.py:distill_mixture_policy`

### Recommendation

**For ICML 2026 experiments, always use `theory_exact_mixture=True` in theory-aligned configs.**

The default parameter interpolation mode (`param.lerp_(candidate, α)`) is faster but does not satisfy CPI guarantees because:
- Parameter interpolation ≠ probability interpolation
- The mixture policy is not explicitly deployed
- `π_old` is updated directly (violating CPI's "keep old policy fixed" principle)

### References

- Kakade & Langford (2002): "Approximately Optimal Approximate Reinforcement Learning"
- ICML 2026 paper Section 6.5: Conservative Policy Improvement
- Paper Theorem 5.9: O(α·ε_A) bound requires exact baseline summation

---

## Related Theory Notes

### Episodic vs Persistent Latent z (Section 5.4)

- **Episodic z** (`episodic_latent=true`): z reinitialized from (x, y) at every step
  - Enables `exact_baseline_summation=true` for Theorem 5.9
  - This is the **theory-exact** mode

- **Persistent z** (`episodic_latent=false`): z initialized once per episode, carried across steps
  - `exact_baseline_summation` becomes "memoryless approximation"
  - Theorem 5.9 does NOT strictly apply
  - Use `batch_centered_advantage=true` as heuristic
  - Analyzed via Lemma 4.4 (two-timescale bound)

### Exact Baseline Summation (Theorem 5.9)

The `exact_baseline_summation=True` flag computes:
```
b(s) = E_{a~π}[Q̂(s,a)] = Σ_a π(a|s) · Q̂(s,a)
```

This is exact summation over ALL discrete actions, ensuring `E_{a~π}[Â(s,a)] = 0` per state. This enables the tight O(α·ε_A) bound instead of O(ε_A/(1-γ)).

Without exact summation, batch-level mean subtraction (`batch_centered_advantage=true`) only provides an approximate baseline.

### Key Config Flags for Theory Alignment

| Flag | Default | Theory Requirement |
|------|---------|-------------------|
| `theory_exact_mixture` | `false` | MUST be `true` for CPI guarantee |
| `exact_baseline_summation` | `false` | MUST be `true` for Theorem 5.9 |
| `episodic_latent` | `true` | MUST be `true` for `exact_baseline_summation` to apply correctly |
| `enable_contraction` | `false` | SHOULD be `true` for Assumption 4.2 |
| `latent_ball_radius` | `0.0` | SHOULD be `> 0` for Assumption 4.1 |

Use `RLConfig.validate_theory_alignment()` to check theory consistency.
