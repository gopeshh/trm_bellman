# HANDOFF.md - Session Summary for UPI-TRM Project

**Date:** 2026-01-19
**Branch:** `feature/upi-trm-clean`
**Latest Commit:** `637711d` - Phase 4: 2x2 Norm Ablation (11/11 audit PASS)

---

## Project Overview

**UPI-TRM (Unified Policy Iteration with Thinking Recursive Model)** is a research project exploring contraction-based stability for recursive latent reasoning in neural networks. The work targets ICML submission.

### Key Hypothesis
Spectral-norm contraction on the z→z update map should provide stable value function learning with bounded error propagation.

---

## Current State: Experiments Finalized

### Phase 4: 2×2 Norm Ablation (COMPLETE ✅) - NEW
- **Location:** `results/paper_ready/phase4_2x2_norm_ablation/`
- **Status:** 11/11 audit checks pass
- **Design:** 2×2 factorial (z→z contraction × value-head norm) × 3 seeds = 12 runs
- **Critical Finding:** **z→z contraction is the primary stabilizer**
  - Contraction ON: ~97% argmax agreement, L_preproj ~0.46
  - Contraction OFF: ~78-83% argmax agreement, L_preproj ~0.58
  - Value-head norm effect minimal in this training regime
  - No NaNs in any condition

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

## Phase 4 Summary Table

| Condition | z→z | V-head | L̂_z | Argmax@4× | Argmax@8× |
|-----------|-----|--------|-----|-----------|-----------|
| nc_nv | OFF | OFF | 0.585 | 83.3% | 77.7% |
| nc_yv | OFF | ON | 0.589 | 83.3% | 77.7% |
| yc_nv | ON | OFF | 0.465 | **97.7%** | **97.7%** |
| yc_yv | ON | ON | 0.460 | **97.7%** | 97.0% |

**Conclusion:** z→z contraction is the dominant factor for stability.

---

## Key Technical Findings

### 1. z→z Contraction is Primary Stabilizer (Phase 4) ⭐ NEW
The 2×2 ablation definitively shows that z→z contraction provides ~97% argmax agreement vs ~78-83% without, regardless of value-head normalization setting.

### 2. Projection Dominance (Exp2)
At training radius R=10, projection is **always active** (100%), masking underlying contraction differences. This is the primary stabilizer when projection is on.

### 3. Dial Failure WITH Projection (Exp2)
The spectral-norm "dial" on target_Lz does not produce proportional changes in observed L_z when projection is active:
- L_preproj varies only ~0.064 across dial settings
- Architecture prevents meaningful L_z control through this mechanism

### 4. Dial SUCCESS WITHOUT Projection (Exp4) ⭐
When projection is disabled (R=0):
- L_preproj varies 0.695 across scale factors {1.0, 0.85, 0.70, 0.55}
- Strong negative correlation between L_preproj and argmax agreement
- The dial provides meaningful control over contraction strength

### 5. Training Stability Without Projection (Exp3)
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
│   ├── fig_exp5_tradeoff_curve.pdf         # stability vs expressivity tradeoff
│   ├── fig_exp5_L_vs_metrics.pdf           # L_preproj vs metrics dual-axis
│   ├── fig_phase4_2x2_norm_ablation.pdf    # NEW: Phase 4 main figure
│   └── fig_phase4_bar_comparison.pdf       # NEW: Phase 4 bar chart
└── tables/
    ├── table_exp2_dial_does_not_control_Lz.tex
    ├── table_exp2_projection_effect.tex
    ├── table_exp4_projection_free_dial_v2.tex
    ├── table_exp5_tradeoff_curve.tex
    └── table_phase4_2x2_norm_ablation.tex  # NEW: Phase 4 results
```

**Note:** `fig_exp4_projection_free_dial_v2.pdf` is stale/duplicate - use `fig_exp4_dial_scatter.pdf` instead.

---

## Phase 4 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_phase4_training.py` | GPU-parallelized training (4 GPUs, 12 jobs) |
| `scripts/eval_phase4_2x2_norm_ablation.py` | Evaluation: stability metrics |
| `scripts/make_paper_figures_phase4.py` | Generate Phase 4 figures and table |
| `scripts/audit_phase4_paper_ready.py` | Audit script (11 checks) |

### BUCK Targets
- `//buiksat_trm:eval_phase4_2x2_norm_ablation`
- `//buiksat_trm:make_paper_figures_phase4`
- `//buiksat_trm:audit_phase4_paper_ready`

---

## Key Scripts (All Experiments)

| Script | Purpose |
|--------|---------|
| `scripts/run_phase4_training.py` | Phase 4 training with GPU parallelization |
| `scripts/eval_phase4_2x2_norm_ablation.py` | Phase 4 evaluation |
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
├── results/exp3_v2/                                      # Exp3 no-projection models
│   └── nc_rdis_s{41,42,43}/model_step_5000.pt           # Used for Exp4
└── /home/buiksat/fbsource/fbcode/buiksat_trm/results/phase4_2x2_norm_ablation/
    └── {nc_nv,nc_yv,yc_nv,yc_yv}_s{41,42,43}/model_step_5000.pt  # Phase 4
```

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

1. **Paper Writing:** Integrate Phase 4 ablation and Exp5 tradeoff curve as submission-critical deliverables
2. **Push Commit:** `git push` (blocked by network restrictions in this session)
3. **Export to Paper Repo:** Copy Phase 4 figures/tables to paper directory
4. **Success Rate Investigation:** Exp5 shows tradeoff exists but success variation is shallow (0.01 range)

---

## Commands Reference

### Run Phase 4 Evaluation
```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:eval_phase4_2x2_norm_ablation \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
  -- --checkpoint_dir /path/to/checkpoints \
     --config_dir /path/to/configs/phase4_2x2_norm_ablation \
     --data_dir /path/to/data \
     --out_dir results/paper_ready/phase4_2x2_norm_ablation
```

### Generate Phase 4 Figures
```bash
buck2 run //buiksat_trm:make_paper_figures_phase4 \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --summary_json results/paper_ready/phase4_2x2_norm_ablation/summary.json
```

### Run Phase 4 Audit
```bash
buck2 run //buiksat_trm:audit_phase4_paper_ready \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --results_dir results/paper_ready/phase4_2x2_norm_ablation \
     --config_dir configs/phase4_2x2_norm_ablation
```

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
3. `results/paper_ready/phase4_2x2_norm_ablation/CLAIMS.md` - Phase 4 ablation claims (NEW)
4. `results/paper_ready/exp5_tradeoff_curve/CLAIMS.md` - Exp5 tradeoff curve claims
5. `results/paper_ready/exp4_projection_free_dial_v2/CLAIMS.md` - Exp4 positive claims
6. `results/paper_ready/exp2_final/CLAIMS.md` - Exp2 negative claims

---

## Session Log

| Commit | Description |
|--------|-------------|
| `637711d` | Phase 4: 2x2 Norm Ablation (11/11 audit PASS) - NEW |
| `3ce240f` | Merge remote changes, resolve conflict in EXPERIMENT_PLAN_ICML.md |
| `231f019` | Exp5: Stability–Expressivity Tradeoff Curve (9/9 audit PASS) |
| `b308272` | Fix Exp4 LaTeX table: use bootstrap rho/CI instead of missing legacy keys |
| `6b7e8bb` | Exp4 v2: Paper-defensible dial with cluster bootstrap (11/11 audit PASS) |
| `6fede63` | Exp4 v2: Multi-checkpoint dial (N=12, 10/10 audit PASS) |
| `1f13a54` | Exp4 Final: Projection-free contraction dial (paper-defensible, audited) |
| `a133ded` | Exp3: Add training logs for projection ablation (6 conditions) |
| `34245fc` | Exp3: Training-time projection ablation (6/6 G1 pass, audited) |
| `03123db` | Exp1 Lipschitz diagnostic + experiment integration finalization |
| `b804e2b` | Exp2 final: dial failure + projection dominance (paper-ready, audited) |

---

## This Session Summary

**Completed Phase 4: 2×2 Norm Ablation**
- Created 4 config files: `nc_nv.yaml`, `nc_yv.yaml`, `yc_nv.yaml`, `yc_yv.yaml`
- Ran 12 training jobs (4 conditions × 3 seeds) on 4 GPUs
- Evaluated all checkpoints for stability metrics
- Generated figures and LaTeX table
- Passed 11/11 audit checks
- Key finding: **z→z contraction is the primary stabilizer** (~97% vs ~78% argmax agreement)

**Pending:** `git push` (blocked by network restrictions - push manually)

---

*Generated: 2026-01-19*
