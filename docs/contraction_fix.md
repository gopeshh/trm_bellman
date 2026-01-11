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

## Known Limitations

### Init-Only Clamping (No Periodic Re-Clamp)

**Current behavior**: Operator-norm clamping is applied **only at model initialization** (in `TinyRecursiveReasoningModel_ACTV1.__init__`). There is no automatic periodic re-clamping during RL training.

**Implication**: During training, SGD updates may cause per-layer spectral norms to drift above 1.0, potentially violating the contraction guarantee over time.

**Mitigation**: A helper function `apply_opnorm_clamp_periodically()` exists in `utils/lipschitz.py` but is NOT currently called in the trainer. To enforce contraction throughout training, you would need to add periodic re-clamping to the training loop (e.g., every N steps).

**For strict theory alignment**, consider:
```python
# In the training loop (e.g., UPITrmTrainer.train_step)
if step % 100 == 0:
    apply_opnorm_clamp_periodically(model.inner, per_layer_max=1.0)
```

### Training Sanity Check (January 2026)

A 400-step training run with `enable_contraction=true` (opnorm clamp at init) showed:
- **Training does not crash or diverge**
- eval_success_rate = 26% (on 4x4 trivial Sudoku)
- Value and policy losses are stable (no NaN or explosion)
- STOP action correctly disabled (stop_prob=0.000)

This confirms the fix enables training without immediate issues, but does not prove contraction is maintained throughout long training runs without periodic re-clamping.

## References

- Original issue diagnosed in: `scripts/diagnose_contraction_components.py`
- Fix validated in: `scripts/diagnose_contraction_fix.py`
- Paper Assumption 4.2: Contraction requirement L_z < 1
