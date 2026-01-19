# Unroll Sensitivity Experiment Validation Report

**Date**: 2026-01-16
**Validator**: Claude Code (automated audit)
**Experiment**: ICML Phase 1 - Unroll Sensitivity / "Theory Saver"

---

## VERDICT: **FAIL** (with caveats - results may be directionally correct)

The experiment has multiple critical issues that undermine confidence in the reported numbers:

1. **Config mislabeling**: The summary table incorrectly reports both models as having identical configs
2. **Depth mislabeling**: Reports "4× training depth" but actual comparison is 8× training depth
3. **Projection disabled during eval**: `rl_latent_ball_radius` not set in eval model config
4. **rl_config not saved in checkpoints**: Makes config verification impossible from artifacts alone

However, the comparison **between models is still valid** because:
- Model B checkpoint has `_lip_scale` keys proving contraction was enabled during training
- Both models were trained with same inner_unroll_n=2
- The relative improvement (Model B >> Model A on stability metrics) is reproducible

---

## 0. Artifact Checklist

| Artifact | Exists | Size | Notes |
|----------|--------|------|-------|
| `checkpoints/model_a/model_step_5000.pt` | YES | 6.7 MB | No Lipschitz keys |
| `checkpoints/model_a/rl_checkpoint_step_5000.pt` | YES | 20 MB | |
| `checkpoints/model_b/model_step_5000.pt` | YES | 6.7 MB | HAS `_lip_scale` keys (4) |
| `checkpoints/model_b/rl_checkpoint_step_5000.pt` | YES | 20 MB | |
| `artifacts/eval_batches/b0.pt` | YES | 32 KB | 100 states |
| `artifacts/eval_batches/b1.pt` | YES | 223 KB | 717 states |
| `artifacts/eval_batches/b0_metadata.json` | YES | 389 B | |
| `artifacts/eval_batches/b1_metadata.json` | YES | 458 B | |
| `results/tables/unroll_sensitivity_summary.md` | YES | 1.9 KB | Original report |

**Timestamps**:
- Model A checkpoints: 2026-01-16 00:21
- Model B checkpoints: 2026-01-16 00:53
- Batches: 2026-01-16 01:06-01:07

---

## 1. Config Verification

### YAML Configs (Ground Truth)

| Setting | Model A (`no_contraction.yaml`) | Model B (`contraction_no_vhead_norm.yaml`) |
|---------|--------------------------------|-------------------------------------------|
| `enable_contraction` | **false** | **true** |
| `disable_value_head_norm` | (default=false) | **true** |
| `target_Lz` | 0.9 (ignored) | **0.9** |
| `inner_unroll_n` | **2** | **2** |
| `latent_ball_radius` | (default=10.0) | **10.0** |
| `use_feasibility_checker` | true | true |
| `episodic_latent` | true | true |

### Checkpoint Evidence

**Model A** (no contraction):
- State dict keys: NO `_lip_scale` keys present
- Confirms contraction was DISABLED during training

**Model B** (contraction enabled):
- State dict keys: HAS 4 `_lip_scale` keys:
  - `inner.L_level.layers.0.self_attn.qkv_proj._inner_lip_scale`
  - `inner.L_level.layers.0.self_attn.o_proj._inner_lip_scale`
  - `inner.L_level.layers.0.mlp.gate_up_proj._inner_lip_scale`
  - `inner.L_level.layers.0.mlp.down_proj._inner_lip_scale`
- Confirms contraction was ENABLED during training

### BUG: rl_config Not Saved

**Location**: `upi_trm_train.py:155-158`

```python
checkpoint = {
    "step": step,
    "model_state_dict": model.state_dict(),
    # rl_config NOT saved here!
}
```

**Impact**: Eval script falls back to RLConfig defaults, causing mislabeled config in summary table.

**Evidence**: Eval script output shows:
```
[Load] Model loaded. Config: {'enable_contraction': True, ...}  # WRONG for Model A
```

---

## 2. Reproducibility Check

### Re-run Results

| Metric (B0, 4→16) | Original | Re-run | Match |
|-------------------|----------|--------|-------|
| Δ_V mean (A) | 4.0834 | 4.0834 | YES |
| Δ_V mean (B) | 0.0823 | 0.0823 | YES |
| Δ_π mean (A) | 0.2478 | 0.2478 | YES |
| Δ_π mean (B) | 0.0058 | 0.0058 | YES |
| Δ_z mean (A) | 34.9098 | 34.9098 | YES |
| Δ_z mean (B) | 7.8133 | 7.8133 | YES |
| argmax_agree (A) | 0.7300 | 0.7300 | YES |
| argmax_agree (B) | 0.9500 | 0.9500 | YES |

**Verdict**: PASS - Results are reproducible with same seed.

---

## 3. Batch Validation

### B0 (Initial States)

| Check | Result |
|-------|--------|
| Total states | 100 (target: 100) PASS |
| Unique state_ids | 100 PASS |
| Empties distribution | {1: 27, 2: 23, 3: 26, 4: 24} |
| Target composition | 70 easy / 30 hard |
| Actual composition | 100 easy / 0 hard (degraded - no hard puzzles in dataset) |
| Source | `data/sudoku-4x4-trivial` |

**Note**: Dataset only contains trivial puzzles (1-4 empties). No 6-8 empty puzzles available.

### B1 (Successor Closure)

| Check | Result |
|-------|--------|
| Total states | 717 (cap: 1500) PASS |
| Parent batch | b0 PASS |
| Checkpoints recorded | Both A and B PASS |
| Empties distribution | {1: 108, 2: 163, 3: 226, 4: 220} |

---

## 4. Metric Correctness

### 4.A KL Divergence Sanity

| Check | Result |
|-------|--------|
| All KL >= 0 | PASS (sampled 20 states) |
| No NaN/Inf | PASS |
| Probabilities sum to 1 | Not directly verified (masked before computation) |

### 4.B Δ_V Sanity

| Check | Result |
|-------|--------|
| No NaN/Inf | PASS |
| Finite values | PASS (max ~20 for Model A) |

### 4.C Argmax Agreement

| Check | Result |
|-------|--------|
| Self-comparison (n==n) | NOT TESTED (no same-n pairs in output) |
| Model B (8→16) | 100% agreement (expected for contractive model) |

### 4.D Projection/Latent Sanity

**CRITICAL ISSUE**: Δ_z violates triangle inequality for projected latents.

| Model | R (config) | Δ_z mean | Δ_z max | 2R bound | Violated? |
|-------|------------|----------|---------|----------|-----------|
| A | 10.0 | 34.9 | 45.9 | 20.0 | **YES** |
| B | 10.0 | 7.8 | 13.6 | 20.0 | NO |

**Root Cause**: Projection is DISABLED during eval.

**Location**: `scripts/eval_unroll_sensitivity.py:247-273`

The model config is created WITHOUT setting `rl_latent_ball_radius`:

```python
model_config = TinyRecursiveReasoningModel_ACTV1Config(
    # ... other params ...
    # rl_latent_ball_radius NOT SET -> defaults to 0.0 (disabled)
)
```

**Evidence**: All `z_pre_norm` and `z_post_norm` values are 0.0 in per-state CSV because:
1. `rl_latent_ball_radius = 0.0` (model config default)
2. Projection only runs when `R > 0.0`
3. `_project_to_ball` never called → instrumentation never fires

### 4.E Depth Mislabeling

**Reported**: "n_train (4) vs 4×n_train (16)"
**Actual**: Models were trained with `inner_unroll_n = 2`

| Eval Depth | Reported Multiplier | Actual Multiplier |
|------------|---------------------|-------------------|
| 4 | 1× | 2× |
| 8 | 2× | 4× |
| 16 | 4× | **8×** |

**Root Cause**: `load_model_for_eval` uses RLConfig default (n=4) instead of actual training config (n=2).

---

## 5. Reported vs Actual Numbers

All headline numbers in the original summary table are **reproducible** from the re-run.

| Metric | Reported | Verified | Match |
|--------|----------|----------|-------|
| B0 Δ_V mean A | 4.0834 | 4.0834 | YES |
| B0 Δ_V mean B | 0.0823 | 0.0823 | YES |
| B0 argmax A | 73.0% | 73.0% | YES |
| B0 argmax B | 95.0% | 95.0% | YES |
| B1 Δ_V mean A | 5.0060 | 5.0060 | YES |
| B1 Δ_V mean B | 0.7772 | 0.7772 | YES |

**However**, the CONFIG comparison table in the summary is WRONG:
- Shows `enable_contraction: True` for both models
- Model A was actually trained with `enable_contraction: false`

---

## 6. Summary of Issues

### Critical (FAIL-worthy)

1. **Config table mislabeled**: Both models shown as having same config
2. **Depth multiplier mislabeled**: "4×" is actually 8× training depth
3. **Projection disabled during eval**: Δ_z computed on unbounded latents

### Moderate (WARNING)

4. **rl_config not saved in checkpoints**: Prevents proper config verification
5. **Saturation stats broken**: z_pre/z_post always 0.0

### Minor

6. **No hard puzzles in B0**: Dataset limitation, not experiment bug
7. **No n==n self-comparison**: Cannot verify argmax_agree(n,n) == 1.0

---

## 7. Recommended Fixes

### Immediate (before publication)

1. **Fix eval script model config** (`scripts/eval_unroll_sensitivity.py`):
   ```python
   model_config = TinyRecursiveReasoningModel_ACTV1Config(
       ...
       rl_latent_ball_radius=config.get("latent_ball_radius", 10.0),  # ADD THIS
   )
   ```

2. **Save rl_config in checkpoints** (`upi_trm_train.py`):
   ```python
   checkpoint = {
       "step": step,
       "model_state_dict": model.state_dict(),
       "rl_config": rl_cfg.model_dump(),  # ADD THIS
   }
   ```

3. **Re-run evaluation** with fixed script and update summary table.

4. **Correct depth labels** in summary:
   - Change "n_train (4)" to "n_train (2)"
   - Or pass `--n_mults 1 2 4` with correct n_train override

### Before ICML Submission

5. **Add n==n sanity check** to metric output
6. **Include Δ_z interpretation** noting it's pre-projection
7. **Re-train with `latent_ball_radius: 10.0` explicitly set for Model A** to ensure apples-to-apples comparison

---

## 8. Final Assessment

| Criterion | Status |
|-----------|--------|
| Artifacts exist | PASS |
| Models differ as intended | PASS (Lipschitz keys present in B only) |
| Reproducible | PASS |
| Config table accurate | **FAIL** |
| Depth labels accurate | **FAIL** |
| Δ_z metric valid | **FAIL** (computed on unbounded latents) |
| Comparison valid | PASS (both models evaluated same way) |

**Overall**: The relative comparison between Model A and Model B is **valid** - Model B (contraction) does show dramatically lower sensitivity than Model A. However, the absolute metrics and labels are incorrect due to eval script bugs.

**Recommendation**: Fix the three critical bugs, re-run, and update the summary before publication.

---

*Generated by automated validation script. Verify all code references manually before making changes.*
