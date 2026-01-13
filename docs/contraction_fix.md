# Contraction Fix: Operator-Norm Clamping

This document describes the fix for the Lipschitz constant explosion issue when using `spectral_norm` for contraction enforcement.

For the latest empirical results/ablations that use this contraction implementation, see:
- `EXPERIMENT_RESULTS_4x4_FEASIBILITY.md`

## Problem: spectral_norm Causes Lz Explosion

The original implementation used PyTorch's `torch.nn.utils.spectral_norm` to enforce per-layer Lipschitz bounds. However, diagnostics revealed that spectral_norm is **numerically unstable** in the TRM architecture:

| Variant | Measured Lz | Status |
|---------|-------------|--------|
| OFF (no contraction) | ~1.0 | OK |
| SN-only (spectral_norm) | 7000 - 600000+ | **EXPLODED** |
| SN+SCALE (spectral_norm + scaling) | 4000 - 65000+ | **EXPLODED** |

The explosion was observed using the `diagnose_contraction_components.py` script with various epsilon values for finite-difference Lipschitz estimation.

## Root Cause

PyTorch's `spectral_norm` works by:
1. Reparameterizing the weight as `W = σ · W_normalized`
2. Using hooks to update `σ` and `W_normalized` during forward passes
3. Estimating `σ` via power iteration

In the TRM architecture with `L_cycles > 1` inner loops and multiple attention/MLP layers, this reparameterization interacts poorly with:
- The residual connections in the reasoning blocks
- The `rms_norm` normalization applied after each layer
- The repeated application of `latent_step()` in the inner loop

## Solution: Operator-Norm Clamping

We replaced `spectral_norm` with a simpler, more robust approach:

### 1. Power Iteration for Spectral Norm Estimation
```python
def _power_iteration(weight: torch.Tensor, num_iters: int = 10) -> float:
    """Estimate spectral norm via power iteration in float32."""
```

### 2. Direct Weight Rescaling
```python
def clamp_linear_operator_norm(module, max_norm: float, num_power_iters: int = 10):
    """If ||W|| > max_norm, rescale W in-place: W <- W * (max_norm / ||W||)"""
```

### 3. Selective Application to z→z Path Layers
```python
def apply_opnorm_clamp_to_trm(inner_model, per_layer_max=1.0, restrict_to_reasoning_layers=True):
    """Only clamp layers in L_level (reasoning stack), not embeddings or output heads."""
```

## Results After Fix

### Final Validated Results (January 2026)

**With target_Lz = 0.9 (eps=1e-3):**

| Variant | Measured Lz | max σ(W) | Status |
|---------|-------------|----------|--------|
| OFF (no contraction) | 1.39 | 3.81 | OK |
| CLAMP-only | **0.70** | 1.06 | **Contractive** |
| SCALE-only | 1.27 | 3.82 | OK |
| CLAMP+SCALE | **0.70** | 1.03 | **Contractive** |
| SN-only | 50257 | 3.65 | **EXPLODED** |
| SN+SCALE | 54055 | 3.81 | **EXPLODED** |

**With target_Lz = 0.5 (eps=1e-3):**

| Variant | Measured Lz | max σ(W) | Status |
|---------|-------------|----------|--------|
| OFF (no contraction) | 0.74 | 3.74 | OK |
| CLAMP-only | **0.65** | 1.04 | **Contractive** |
| SCALE-only | 0.64 | 3.69 | OK |
| CLAMP+SCALE | **0.66** | 1.05 | **Contractive** |

**Key observations:**
- **Measured Lz stays O(1)**: No explosions with CLAMP variants
- **Lz < 1 achieved**: Both CLAMP-only and CLAMP+SCALE give Lz ~0.65-0.70, satisfying Assumption 4.2
- **Per-layer max σ(W) ≤ 1.1**: Confirms the clamping is working correctly
- **SN variants explode**: Lz > 50,000 with spectral_norm, confirming the original issue

The fix achieves:
- **Stable Lz < 1**: Contractive behavior as required by Assumption 4.2
- **No explosions**: Lz stays O(1) regardless of epsilon
- **Selective application**: Only z→z path layers are affected

## Diagnostic Commands

### About the Diagnostic Scripts

**`diagnose_contraction_components.py`**: Tests spectral_norm variants only
- Variants: OFF, SN-only, SCALE-only, SN+SCALE
- Purpose: Isolate whether spectral_norm or scaling causes the explosion
- Result: **spectral_norm is the culprit** (SN-only and SN+SCALE explode)

**`diagnose_contraction_fix.py`**: Tests operator-norm clamp vs spectral_norm
- Variants: OFF, CLAMP-only, SCALE-only, CLAMP+SCALE, SN-only, SN+SCALE
- Purpose: Validate that opnorm clamp fixes the issue
- Result: **CLAMP variants are stable** (Lz < 1), SN variants explode

### Validate the Fix
```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:diagnose_contraction_fix -- \
    --dataset /home/buiksat/trm_bellman/data/sudoku-4x4-trivial \
    --batch-size 4 --num-repeats 10 --target-lz 0.9 --eps-list 1e-2,1e-3,1e-4
```

### Compare Old vs New (includes exploding SN variants)
```bash
buck2 run //buiksat_trm:diagnose_contraction_fix -- \
    --dataset /home/buiksat/trm_bellman/data/sudoku-4x4-trivial \
    --batch-size 4 --num-repeats 10 --target-lz 0.9 --eps-list 1e-3
```

### Skip Exploding Variants
```bash
buck2 run //buiksat_trm:diagnose_contraction_fix -- \
    --dataset /home/buiksat/trm_bellman/data/sudoku-4x4-trivial \
    --batch-size 4 --num-repeats 10 --target-lz 0.5 --eps-list 1e-3 --skip-sn
```

## Code Changes

### Modified Files
1. **utils/lipschitz.py**: Added `_power_iteration()`, `clamp_linear_operator_norm()`, `apply_opnorm_clamp_to_trm()`, `apply_opnorm_clamp_periodically()`
2. **models/recursive_reasoning/trm.py**: Updated contraction enforcement to use opnorm clamp instead of spectral_norm
3. **rl/config.py**: Added `opnorm_clamp_interval` config field
4. **rl/upi_trm_trainer.py**: Added periodic opnorm clamping in `train_step()`
5. **scripts/diagnose_contraction_fix.py**: New diagnostic script to validate the fix

### Key API Changes
```python
# OLD (numerically unstable)
apply_spectral_norm_to_trm(self.inner)
enforce_global_contraction(self.inner, target_Lz)

# NEW (stable)
apply_opnorm_clamp_to_trm(self.inner, per_layer_max=1.0, restrict_to_reasoning_layers=True)
enforce_global_contraction(self.inner, target_Lz, restrict_to_reasoning_layers=True)
```

### Layers Affected
Only layers in the `L_level` (reasoning stack) are clamped:
- `L_level.layers.0.self_attn.qkv_proj`
- `L_level.layers.0.self_attn.o_proj`
- `L_level.layers.0.mlp.gate_up_proj`
- `L_level.layers.0.mlp.down_proj`
- `L_level.layers.1.self_attn.qkv_proj`
- `L_level.layers.1.self_attn.o_proj`
- `L_level.layers.1.mlp.gate_up_proj`
- `L_level.layers.1.mlp.down_proj`

Not affected (intentionally):
- `embed_tokens` (embedding layer)
- `lm_head` (output head)
- `q_head` (output head)

## Usage

### Enable Contraction in Config
```yaml
rl_enable_contraction: true
rl_target_Lz: 0.9  # or lower for stronger contraction
opnorm_clamp_interval: 100  # Re-clamp every 100 steps (0 = disabled)
opnorm_clamp_max_norm: 1.0  # Per-layer max operator norm (default 1.0)
opnorm_clamp_num_power_iters: 10  # Power iterations for spectral norm estimation
opnorm_log_max_sigma: false  # Set true to log max σ(W) when clamp fires
```

### Periodic Re-clamping (Automatic)
Periodic re-clamping is now **automatic** when `enable_contraction=True` and `opnorm_clamp_interval > 0`. The `UPITrmTrainer.train_step()` method handles this internally every N steps.

## Recommendations

1. **Use opnorm clamp**: The new approach is numerically stable and achieves Lz < 1
2. **Restrict to reasoning layers**: Only clamp z→z path, not embeddings or outputs
3. **Enable periodic re-clamping**: Set `opnorm_clamp_interval=100` to maintain contraction during training
4. **Monitor Lz**: Track theory metrics to verify contraction is maintained

## Known Limitations

### Periodic Re-Clamping (January 2026)

**Current behavior**: Operator-norm clamping is now applied both at model initialization AND periodically during training. The `UPITrmTrainer.train_step()` method re-applies operator-norm clamping every N steps (default N=100) when `enable_contraction=True`.

**Configuration**:
```yaml
enable_contraction: true
opnorm_clamp_interval: 100  # Re-clamp every 100 steps (0 = disabled)
opnorm_clamp_max_norm: 1.0  # Per-layer max operator norm
opnorm_clamp_num_power_iters: 10  # Power iterations for spectral norm estimation
opnorm_log_max_sigma: false  # Set true to log max σ(W) at each clamp
```

**Implementation** (in `rl/upi_trm_trainer.py`):
```python
# Every N steps, re-apply operator-norm clamping
if enable_contraction and opnorm_clamp_interval > 0 and step % opnorm_clamp_interval == 0:
    with torch.no_grad():
        sigma_dict = apply_opnorm_clamp_periodically(
            model.inner,
            per_layer_max=opnorm_clamp_max_norm,
            num_power_iters=opnorm_clamp_num_power_iters,
        )
    # Optional: log max σ(W) over L_level layers
    if opnorm_log_max_sigma and sigma_dict:
        max_sigma = max(sigma_dict.values())
        logger.info(f"[step {step}] opnorm clamp applied (max_sigma={max_sigma:.2f})")
```

This ensures contraction is maintained throughout training, not just at initialization.

### Training Sanity Check (January 2026)

A 400-step training run with `enable_contraction=true` and `opnorm_clamp_interval=100` showed:
- **Training does not crash or diverge**
- eval_success_rate = 26% (on 4x4 trivial Sudoku)
- Value and policy losses are stable (no NaN or explosion)
- STOP action correctly disabled (stop_prob=0.000)

Periodic re-clamping helps maintain the per-layer norm constraint throughout training. For strict verification, log/monitor layer norms or Lz proxies during long runs.

**Note**: The trainer clamps `self.model.inner` (the main model used for value updates). The `policy_model_candidate` is synced from `self.model` at each train_step, so clamped weights propagate to policy updates.

## References

- Original issue diagnosed in: `scripts/diagnose_contraction_components.py`
- Fix validated in: `scripts/diagnose_contraction_fix.py`
- Paper Assumption 4.2: Contraction requirement L_z < 1
