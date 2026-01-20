# Value-Head Normalization: Implementation Details

**Date:** 2026-01-19
**Purpose:** Clarify the distinction between z→z contraction and value-head normalization

## Terminology

There are two separate normalization mechanisms in UPI-TRM:

### 1. z→z Contraction (`enable_contraction: true`)

**What it does:** Enforces Lipschitz bound on the z→z latent dynamics

**Implementation location:** `models/recursive_reasoning/trm.py:484-497`

```python
# Apply to L_level layers (reasoning layers) only
apply_opnorm_clamp_to_trm(
    self.inner,
    per_layer_max=1.0,
    num_power_iters=10,
    restrict_to_reasoning_layers=True,
)
enforce_global_contraction(
    self.inner,
    self.config.rl_target_Lz,
    restrict_to_reasoning_layers=True,
)
```

**Key points:**
- Uses **opnorm_clamp** (NOT `torch.nn.utils.spectral_norm`)
- Rescales weight matrices to achieve target L_z
- Only affects `L_level` layers (reasoning loop)
- **Stable and safe for experiments**

### 2. Value-Head Normalization (`disable_value_head_norm: false`)

**What it does:** Applies PyTorch spectral norm + contraction scaling to value head

**Implementation location:** `models/recursive_reasoning/trm.py:500-502`

```python
if self.value_head is not None and not self.config.rl_disable_value_head_norm:
    apply_spectral_norm_to_value_head(self.value_head)
    enforce_global_contraction_on_value_head(self.value_head, self.config.rl_target_Lv)
```

**Key points:**
- Uses **`torch.nn.utils.spectral_norm`** (PyTorch's built-in)
- Applies to all linear layers in value head
- **Known to cause collapse** in certain configurations
- **MUST be disabled** for stability experiments: `disable_value_head_norm: true`

## Config Flags

| Flag | Value | Effect |
|------|-------|--------|
| `enable_contraction` | `true` | z→z opnorm clamping + scaling → stable |
| `enable_contraction` | `false` | No z→z enforcement |
| `disable_value_head_norm` | `true` | **Use this** - no value head spectral norm |
| `disable_value_head_norm` | `false` | Value head spectral norm ON → may cause collapse |

## The 2×2 Ablation Finding (Phase 4)

| Condition | z→z | V-head | Argmax@8× | Conclusion |
|-----------|-----|--------|-----------|------------|
| nc_nv | OFF | OFF | 77.7% | Baseline |
| nc_yv | OFF | ON | 77.7% | V-head norm alone has no effect |
| **yc_nv** | **ON** | OFF | **97.7%** | **z→z contraction is primary stabilizer** |
| yc_yv | ON | ON | 97.0% | Adding V-head norm doesn't improve |

**Key finding:** z→z contraction alone provides ~20% improvement in argmax agreement.
Value-head normalization has minimal additional impact.

## Implementation Files

| File | What it contains |
|------|------------------|
| `rl/config.py` | `rl_enable_contraction`, `rl_disable_value_head_norm` definitions |
| `models/recursive_reasoning/trm.py` | Actual enforcement logic (lines 474-502) |
| `utils/lipschitz.py` | `apply_opnorm_clamp_to_trm()`, `apply_spectral_norm_to_value_head()` |

## Correct Experiment Config

For all "stability dial" experiments:

```yaml
# Required for stability experiments
disable_value_head_norm: true   # CRITICAL - prevents collapse

# Optional - can be varied
enable_contraction: true        # For contraction enforcement
target_Lz: 0.9                  # Contraction strength
```

## Common Mistakes to Avoid

1. **Calling z→z contraction "spectral norm"** - It uses opnorm_clamp, not spectral_norm
2. **Enabling value-head norm for stability tests** - Causes collapse mode
3. **Confusing `target_Lz` with value head** - target_Lz only affects z→z, not value head
