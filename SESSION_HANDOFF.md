# Session Handoff Note

**Date**: December 27, 2024
**Previous Session Status**: Documentation updates complete, experiments running

---

## What Was Done This Session

### 1. CPI Theory Clarification (Gemini's Concern)
Gemini raised a concern about missing importance sampling (IS) weights in the CPI mixture policy update. After analysis:

**Conclusion**: IS weights are **NOT required** in `theory_exact_mixture=True` mode because:
- CPI deploys the explicit mixture `π_mix = (1-α)π_old + α·π_cand`
- The improvement guarantee `V^{π_new} ≥ V^{π_old} - O(α·ε_A)` applies to the deployed mixture
- We're NOT training π_cand to match π_mix - we deploy the mixture directly

**Documentation Updated**:
- `CLAUDE.md` - Added CPI Mixture Policy Modes section
- `EXPERIMENT_TRACKING.md` - Added theory notes on CPI modes
- `docs/THEORY_NOTES.md` - Created new comprehensive theory document
- `DOCUMENTATION_INDEX.md` - Added reference to THEORY_NOTES.md

### 2. Experiments Running
Many background experiments from previous sessions are still running (50+ buck2 processes). Check status with:
```bash
pgrep -f "buck2" | wc -l
```

---

## Current Experiment Status

| Experiment | Method | Status |
|------------|--------|--------|
| EXP-01 | Imitation Learning | ✅ COMPLETE (100% success) |
| EXP-02 | UPI-TRM (theory-exact, episodic z) | ⏳ Background |
| EXP-03 | UPI-TRM (baseline) | ⏳ Background |
| EXP-04 | PPO-TRM | ⏳ Background |
| EXP-05 | PPO-MLP | ⏳ Background |
| EXP-06 | A2C-MLP | ⏳ Background |
| EXP-07 | UPI-TRM (constraint-checker) | ⏳ Background |
| EXP-08 | UPI-TRM (persistent z + constraint) | ⏳ Background |
| EXP-09 | UPI-TRM (persistent z, batch-centered) | ⏳ Background |

See `EXPERIMENT_TRACKING.md` for full details.

---

## What to Do Next

### Option A: Wait for Experiments to Complete
1. Check if experiments finished: `pgrep -f "buck2" | wc -l`
2. If 0 processes, collect results and update `EXPERIMENT_TRACKING.md`

### Option B: Start New Experiment Batch
If starting fresh experiments, use these commands:

```bash
cd ~/fbsource/fbcode

# Main contribution (theory-exact with all features)
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --config /home/buiksat/trm_bellman/configs/rl_sudoku_shaped_theory_exact.yaml \
    --train-steps 5000 --seed 42 --no-wandb

# Baselines
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
    --baseline ppo --backbone norec-mlp \
    --dataset-paths /home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy \
    --train-steps 5000 --seed 42 --no-wandb
```

### Option C: Run Ablation Studies
See `configs/ablations/` for ablation configs. The key ablation is:
- `ablation_no_exact_baseline.yaml` - Tests Theorem 5.9 (KEY contribution)

---

## Key Files to Read

1. **`DOCUMENTATION_INDEX.md`** - Navigation hub
2. **`EXPERIMENT_TRACKING.md`** - Experiment status and theory notes
3. **`docs/THEORY_NOTES.md`** - CPI analysis, IS weights explanation
4. **`CLAUDE.md`** - Complete codebase guide

---

## Important Theory Points

### CPI Modes (use `theory_exact_mixture=True` for ICML 2026)

| Mode | Config | Status |
|------|--------|--------|
| Theory-Exact | `theory_exact_mixture=True` | ✅ CPI bound applies |
| Distillation | `distill_mixture_policy=True` | ⚠️ Heuristic |
| Default | Both `False` | ⚠️ Parameter-space lerp |

### Latent Modes

| Mode | Config | Theorem |
|------|--------|---------|
| Episodic z | `episodic_latent=True` | Theorem 5.9 applies |
| Persistent z | `episodic_latent=False` | Lemma 4.4 (two-timescale) |

---

## Dataset Location
```
/home/buiksat/fbsource/fbcode/buiksat_trm/data/sudoku-4x4-ultra-easy
```

---

**This file can be deleted once the next session has read it.**
