# Contraction Fix: Operator-Norm Clamping

This document describes the fix for the Lipschitz constant explosion issue when using `spectral_norm` for contraction enforcement.

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

| Variant | Measured Lz | Status |
|---------|-------------|--------|
| OFF (no contraction) | ~1.0 | OK |
| CLAMP-only (opnorm clamp) | ~0.6 | **OK (contractive!)** |
| CLAMP+SCALE (opnorm clamp + scaling) | ~0.6 | **OK (contractive!)** |

The fix achieves:
- **Stable Lz < 1**: Contractive behavior as required by Assumption 4.2
- **No explosions**: Lz stays O(1) regardless of epsilon
- **Selective application**: Only z→z path layers are affected

## Diagnostic Commands

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
3. **scripts/diagnose_contraction_fix.py**: New diagnostic script to validate the fix

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
```

### During Training (Optional Periodic Re-clamping)
If weights drift during training, re-apply clamping:
```python
from utils.lipschitz import apply_opnorm_clamp_periodically

# Every N training steps
if step % 100 == 0:
    apply_opnorm_clamp_periodically(model.inner, per_layer_max=1.0)
```

## Recommendations

1. **Use opnorm clamp**: The new approach is numerically stable and achieves Lz < 1
2. **Restrict to reasoning layers**: Only clamp z→z path, not embeddings or outputs
3. **Re-clamp periodically**: If weights drift during long training runs
4. **Monitor Lz**: Track theory metrics to verify contraction is maintained

## References

- Original issue diagnosed in: `scripts/diagnose_contraction_components.py`
- Fix validated in: `scripts/diagnose_contraction_fix.py`
- Paper Assumption 4.2: Contraction requirement L_z < 1
