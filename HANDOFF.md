# HANDOFF.md - Session Summary for UPI-TRM Project

**Date:** 2026-01-18
**Branch:** `feature/upi-trm-clean`
**Latest Commit:** `03123db` - Exp1 Lipschitz diagnostic + experiment integration finalization

---

## Project Overview

**UPI-TRM (Unified Policy Iteration with Thinking Recursive Model)** is a research project exploring contraction-based stability for recursive latent reasoning in neural networks. The work targets ICML submission.

### Key Hypothesis
Spectral-norm contraction on the z→z update map should provide stable value function learning with bounded error propagation.

---

## Current State: Experiments Finalized

### Exp1: No Contraction vs Contraction (COMPLETE ✅)
- **Location:** `results/paper_ready/exp1/`
- **Status:** 4/4 audit checks pass
- **Finding:** Contraction reduces ΔV by 6-10×, but projection at R=10 is 100% active

### Exp2: Contraction Sweep (COMPLETE ✅)
- **Location:** `results/paper_ready/exp2_final/`
- **Status:** 7/7 audit checks pass
- **Critical Finding:** **Dial does NOT control L_z**
  - Spectral norm targets {0.999, 0.99, 0.95} produce L_preproj spread of only 0.064
  - Required threshold for "dial works" claim: ≥0.08
  - **Path B (Negative Result):** The dial fails architecturally

### Exp1 Lipschitz Diagnostic (NEW ✅)
- **Location:** `results/paper_ready/exp1_lipschitz_diag/`
- **Status:** 5/5 audit checks pass
- **Purpose:** Cross-check confirming Exp1-Exp2 consistency
- **Finding:** 100% projection at R=10, 0% at R=100 (consistent with Exp2)

---

## Key Technical Findings

### 1. Projection Dominance
At training radius R=10, projection is **always active** (100%), masking underlying contraction differences. This is the primary stabilizer, not spectral-norm contraction.

### 2. Dial Failure
The spectral-norm "dial" on target_Lz does not produce proportional changes in observed L_z:
- L_preproj varies only ~0.064 across dial settings
- Architecture prevents meaningful L_z control through this mechanism

### 3. Positive Finding
R=10 projection provides significant stability benefit (6-10× ΔV improvement). This is a valid result but different from the original hypothesis.

---

## Paper Assets Location

Figures and tables copied to paper directory:
```
/home/buiksat/UPI_TRM/UPI_TRM_ICML/
├── figures/
│   ├── fig_exp2_dial_does_not_control_Lz.pdf
│   └── fig_exp2_projection_is_primary_stabilizer.pdf
└── tables/
    ├── table_exp2_dial_does_not_control_Lz.tex
    └── table_exp2_projection_effect.tex
```

---

## Important Configuration Rules

### From CLAUDE.md (Must Follow)
1. **Checker:** Use `use_feasibility_checker: true` only
2. **Value-head normalization:** `disable_value_head_norm: true` (prevents collapse)
3. **One variable at a time:** Don't mix episodic_latent with contraction changes
4. **Measure projection saturation:** Log pre/post `||z||` norms

### Config Template for Stability Experiments
```yaml
use_feasibility_checker: true
enable_contraction: true
target_Lz: 0.9
disable_value_head_norm: true   # CRITICAL
episodic_latent: true
latent_ball_radius: 10.0
```

---

## Checkpoints Location

```
checkpoints/
├── exp1_v4/
│   ├── model_a_prime/seed{41,42,43}/model_step_5000.pt  # No contraction
│   └── model_b/seed{41,42,43}/model_step_5000.pt        # Contraction
└── exp2_contraction_sweep/
    └── lz{0999,099,095}/seed{41,42,43}/model_step_5000.pt
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/exp1_lipschitz_diag.py` | Exp1 L_preproj/L_postproj diagnostic |
| `scripts/audit_exp1_lipschitz_diag.py` | Audit for Exp1 diagnostic |
| `scripts/make_paper_figures_exp2_final.py` | Generate Exp2 paper figures |
| `scripts/audit_exp2_final_paper_ready.py` | Audit for Exp2 final bundle |
| `scripts/eval_unroll_sensitivity.py` | Core evaluation infrastructure |

---

## What's Next (Potential)

1. **Baselines:** May need comparison with vanilla TRM (see EXPERIMENT_PLAN_ICML.md)
2. **Paper Writing:** Integrate negative dial result into narrative
3. **Additional Diagnostics:** If reviewers request more evidence

---

## Commands Reference

### Run Evaluation
```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:exp1_lipschitz_diag \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only
```

### Run Audits
```bash
buck2 run //buiksat_trm:audit_exp1_paper_ready ...
buck2 run //buiksat_trm:audit_exp2_final_paper_ready ...
buck2 run //buiksat_trm:audit_exp1_lipschitz_diag ...
```

### Training
```bash
buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --config configs/ablations/upi_trm_feasibility_no_contraction.yaml --seed 42
```

---

## Files to Read First

1. `CLAUDE.md` - Operational guidance, non-negotiables
2. `EXPERIMENT_PLAN_ICML.md` - Full experiment plan with deprecation notes
3. `results/paper_ready/exp2_final/CLAIMS.md` - Current paper claims
4. `results/paper_ready/exp1_lipschitz_diag/CLAIMS.md` - Diagnostic claims

---

## Session Log

| Commit | Description |
|--------|-------------|
| `03123db` | Exp1 Lipschitz diagnostic + experiment integration finalization |
| `b804e2b` | Exp2 final: dial failure + projection dominance (paper-ready, audited) |
| `db3de70` | Exp2c: Unmask stability dial - projection provides stability |
| `0d32097` | Exp2b: Diagnose contraction saturation - projection dominates |
| `a460efd` | Exp2: Multi-seed sweep with honest dial claims |

---

*Generated: 2026-01-18*
