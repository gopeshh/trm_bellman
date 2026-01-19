# HANDOFF.md - Session Summary for UPI-TRM Project

**Date:** 2026-01-19
**Branch:** `feature/upi-trm-clean`
**Latest Commit:** `3dd04c6` - Exp5: Stability–Expressivity Tradeoff Curve (9/9 audit PASS)

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
- **Critical Finding:** **Dial does NOT control L_z** (with projection active)
  - Spectral norm targets {0.999, 0.99, 0.95} produce L_preproj spread of only 0.064
  - Required threshold for "dial works" claim: ≥0.08
  - **Path B (Negative Result):** The dial fails architecturally when projection is active

### Exp3: Training-Time Projection Ablation (COMPLETE ✅)
- **Location:** `results/paper_ready/exp3_projection_ablation/`
- **Status:** 6/6 G1 pass (audited)
- **Purpose:** Train models WITHOUT projection to enable Exp4
- **Finding:** Training is stable with projection disabled (R=0)
- **Conditions:** 6 conditions (±contraction × {R=10, R=100, R=disabled})

### Exp4: Projection-Free Contraction Dial (COMPLETE ✅)
- **Location:** `results/paper_ready/exp4_projection_free_dial_v2/`
- **Status:** 11/11 audit checks pass
- **Critical Finding:** **Dial WORKS when projection is disabled**
  - L_preproj spread = 0.695 (well above 0.10 threshold)
  - Spearman ρ = -0.866 (B0) and -0.923 (B1) for L_preproj vs argmax agreement
  - 95% CIs from cluster bootstrap exclude 0
  - **Path A (Positive Result):** Inference-time contraction dial is viable

### Exp1 Lipschitz Diagnostic (COMPLETE ✅)
- **Location:** `results/paper_ready/exp1_lipschitz_diag/`
- **Status:** 5/5 audit checks pass
- **Purpose:** Cross-check confirming Exp1-Exp2 consistency
- **Finding:** 100% projection at R=10, 0% at R=100 (consistent with Exp2)

### Exp5: Stability–Expressivity Tradeoff Curve (COMPLETE ✅)
- **Location:** `results/paper_ready/exp5_tradeoff_curve/`
- **Status:** 9/9 audit checks pass
- **Purpose:** Submission-critical deliverable showing stability-expressivity tradeoff
- **Finding:**
  - Scale dial creates tradeoff: higher stability (argmax@8×) ↔ lower expressivity (success rate)
  - G0-G3 gates all pass
  - L_preproj spread = 0.656 (dial works)
  - Success rate range = 0.01 (tradeoff exists but shallow)

---

## Key Technical Findings

### 1. Projection Dominance (Exp2)
At training radius R=10, projection is **always active** (100%), masking underlying contraction differences. This is the primary stabilizer when projection is on.

### 2. Dial Failure WITH Projection (Exp2)
The spectral-norm "dial" on target_Lz does not produce proportional changes in observed L_z when projection is active:
- L_preproj varies only ~0.064 across dial settings
- Architecture prevents meaningful L_z control through this mechanism

### 3. Dial SUCCESS WITHOUT Projection (Exp4) ⭐
When projection is disabled (R=0):
- L_preproj varies 0.695 across scale factors {1.0, 0.85, 0.70, 0.55}
- Strong negative correlation between L_preproj and argmax agreement
- The dial provides meaningful control over contraction strength

### 4. Training Stability Without Projection (Exp3)
Models can be trained stably without projection (R=0), enabling the Exp4 dial experiment.

---

## Paper Assets Location

Figures and tables copied to paper directory:
```
/home/buiksat/UPI_TRM/UPI_TRM_ICML/
├── figures/
│   ├── fig_exp2_dial_does_not_control_Lz.pdf
│   ├── fig_exp2_projection_is_primary_stabilizer.pdf
│   ├── fig_exp4_dial_scatter.pdf           # L_preproj vs argmax scatter
│   ├── fig_exp4_scale_comparison.pdf       # bar chart by scale
│   ├── fig_exp5_tradeoff_curve.pdf         # NEW: stability vs expressivity tradeoff
│   └── fig_exp5_L_vs_metrics.pdf           # NEW: L_preproj vs metrics dual-axis
└── tables/
    ├── table_exp2_dial_does_not_control_Lz.tex
    ├── table_exp2_projection_effect.tex
    ├── table_exp4_projection_free_dial_v2.tex  # Exp4 dial results
    └── table_exp5_tradeoff_curve.tex       # NEW: Exp5 scale results
```

**Note:** `fig_exp4_projection_free_dial_v2.pdf` is stale/duplicate - use `fig_exp4_dial_scatter.pdf` instead.

---

## Exp4 Figures Summary

| Figure | Content | Use |
|--------|---------|-----|
| `fig_exp4_dial_scatter.pdf` | L_preproj vs Argmax scatter (B0, B1 panels), colored by seed, shaped by scale, with trend line | Main figure |
| `fig_exp4_scale_comparison.pdf` | Bar chart of L_preproj and Argmax by scale (aggregated means ± std) | Supplementary |
| `fig_exp4_projection_free_dial_v2.pdf` | Older version, superseded | DELETE |

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
├── exp2_contraction_sweep/
│   └── lz{0999,099,095}/seed{41,42,43}/model_step_5000.pt
└── results/exp3_v2/                                      # Exp3 no-projection models
    └── nc_rdis_s{41,42,43}/model_step_5000.pt           # Used for Exp4
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/exp5_tradeoff_curve.py` | Exp5 stability-expressivity tradeoff evaluation |
| `scripts/generate_exp5_figures.py` | Generate Exp5 tradeoff curve and dual-axis plots |
| `scripts/audit_exp5_tradeoff_curve.py` | Audit for Exp5 (9 checks) |
| `scripts/exp4_final_v2.py` | Exp4 multi-checkpoint evaluation with cluster bootstrap |
| `scripts/generate_exp4_figures.py` | Generate Exp4 scatter plot and bar chart |
| `scripts/audit_exp4_final_v2.py` | Audit for Exp4 (11 checks) |
| `scripts/exp1_lipschitz_diag.py` | Exp1 L_preproj/L_postproj diagnostic |
| `scripts/make_paper_figures_exp2_final.py` | Generate Exp2 paper figures |
| `scripts/eval_unroll_sensitivity.py` | Core evaluation infrastructure |

---

## Statistical Methodology (Exp4)

**Problem:** Dial scales are repeated measures on same checkpoint (not i.i.d.)

**Solution:** Cluster bootstrap
- Checkpoints treated as clusters (N=3 clusters)
- Resample checkpoints with replacement (1000 iterations)
- Report 95% CI instead of p-values
- Pass criterion: |ρ| > 0.5 AND 95% CI excludes 0

---

## What's Next (Potential)

1. **Paper Writing:** Integrate Exp5 tradeoff curve as submission-critical deliverable
2. **Clean up paper repo:** Delete stale `fig_exp4_projection_free_dial_v2.pdf`
3. **Success Rate Investigation:** Exp5 shows tradeoff exists but success variation is shallow (0.01 range)

---

## Commands Reference

### Run Exp5 Evaluation
```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:exp5_tradeoff_curve \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
  -- --checkpoints /path/to/ckpt1.pt /path/to/ckpt2.pt /path/to/ckpt3.pt \
     --out_dir results/paper_ready/exp5_tradeoff_curve
```

### Generate Exp5 Figures
```bash
buck2 run //buiksat_trm:generate_exp5_figures \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --summary_json results/paper_ready/exp5_tradeoff_curve/summary.json
```

### Run Exp4 Evaluation
```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:exp4_final_v2 \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
  -- --checkpoints /path/to/ckpt1.pt /path/to/ckpt2.pt --out_dir /path/to/output
```

### Generate Exp4 Figures
```bash
buck2 run //buiksat_trm:generate_exp4_figures \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --summary_json results/paper_ready/exp4_projection_free_dial_v2/summary.json
```

### Run Audits
```bash
buck2 run //buiksat_trm:audit_exp4_final_v2 ...
buck2 run //buiksat_trm:audit_exp2_final_paper_ready ...
```

---

## Files to Read First

1. `CLAUDE.md` - Operational guidance, non-negotiables
2. `EXPERIMENT_PLAN_ICML.md` - Full experiment plan with status
3. `results/paper_ready/exp5_tradeoff_curve/CLAIMS.md` - Exp5 tradeoff curve claims
4. `results/paper_ready/exp4_projection_free_dial_v2/CLAIMS.md` - Exp4 positive claims
5. `results/paper_ready/exp2_final/CLAIMS.md` - Exp2 negative claims

---

## Session Log

| Commit | Description |
|--------|-------------|
| `3dd04c6` | Exp5: Stability–Expressivity Tradeoff Curve (9/9 audit PASS) |
| `b308272` | Fix Exp4 LaTeX table: use bootstrap rho/CI instead of missing legacy keys |
| `6b7e8bb` | Exp4 v2: Paper-defensible dial with cluster bootstrap (11/11 audit PASS) |
| `6fede63` | Exp4 v2: Multi-checkpoint dial (N=12, 10/10 audit PASS) |
| `1f13a54` | Exp4 Final: Projection-free contraction dial (paper-defensible, audited) |
| `a133ded` | Exp3: Add training logs for projection ablation (6 conditions) |
| `34245fc` | Exp3: Training-time projection ablation (6/6 G1 pass, audited) |
| `03123db` | Exp1 Lipschitz diagnostic + experiment integration finalization |
| `b804e2b` | Exp2 final: dial failure + projection dominance (paper-ready, audited) |

---

## Bug Fixed This Session

**Issue:** LaTeX table `table_exp4_projection_free_dial_v2.tex` showed ρ=0.000 instead of ρ=-0.866

**Cause:** `generate_exp4_figures.py` was looking for old keys (`mono["rho_aa_b0"]`) that didn't exist in bootstrap format (`mono["boot_aa_b0"]["rho_point"]`)

**Fix:** Updated script to read bootstrap format with 95% CIs, falls back to legacy p-value format for backwards compatibility.

---

*Generated: 2026-01-19*
