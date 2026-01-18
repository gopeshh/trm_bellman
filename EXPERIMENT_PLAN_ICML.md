# Experiment Execution Plan: ICML Submission (Stability Dial)

**Goal:** Validate the "stability dial" narrative: contraction may reduce peak success rate, but makes the model mathematically consistent (lower truncation/unroll error) when evaluated at deeper inner unrolling than the training depth.

**Timeline:** Immediate (Jan 15 – Jan 20)

**Canonical scope:** 4×4 Sudoku with feasibility checker (`use_feasibility_checker: true`).

**Core deliverables (submission-critical):**
- A single figure: Unroll Sensitivity vs. Unroll Depth comparing No-Contraction vs Contraction (V-head norm OFF).
- A single figure: Stability–Expressivity tradeoff curve (success vs stability) across a contraction-strength sweep.
- A short appendix table: Projection saturation / radius ablation (to rule out "clipping artifacts").

---

## Phase 0 (Gating): Instrumentation & Eval Harness

### Objective

Make every run interpretable and defensible by logging the quantities your theory and discussion summary say matter (unrolling bias, Lipschitz proxy, residual proxies, projection saturation).

### Add these metrics to your eval script (per checkpoint, per batch)

**Unroll sensitivity metrics** (computed for pairs of unroll depths):

1. **Value sensitivity**
   ```
   Δ_V(n₁, n₂) = E_{s~B}[|U_{n₁}(s) - U_{n₂}(s)|]
   ```
   Log: mean, median, p90, p99, max.

2. **Policy sensitivity**
   ```
   Δ_π(n₁, n₂) = E_{s~B}[KL(π_{n₁}(·|s) || π_{n₂}(·|s))]
   ```
   Log: mean/median/p90/p99/max.

3. **Argmax agreement**
   - % states where `argmax_a π_{n₁}(a|s) = argmax_a π_{n₂}(a|s)`

4. **Latent sensitivity**
   ```
   Δ_z(n₁, n₂) = E_{s~B}[||z^{(n₁)}(s) - z^{(n₂)}(s)||_2]
   ```

**Contractivity / stability proxies** (must be logged, not assumed):

- **Empirical Lipschitz proxy `L̂_z`** via finite differences on replay/eval batch (as in the paper's diagnostic)
- **Projection saturation rate**
  - pre-projection norm stats: `||z||_pre` mean/p95/max
  - saturation fraction: `P(||z||_pre ≥ 0.95R)`
  - post-projection norm stats: `||z||_post` (should not be pinned to R if projection isn't dominating)
- **K-step residual proxy** (held-out rollouts)
  - empirical `||U_n - T^π_K U_n||` (or TD-error summaries) as a sanity check aligning with the bound's "architectural residual" term

**Behavioral sanity metrics:**
- success rate, mean checker score, mean violations, mean episode length, STOP probability/frequency

### Standardize outputs

Write everything to:
- `results/plot_data/unroll_sensitivity/*.csv`
- `results/plot_data/contraction_sweep/*.csv`
- `results/plot_data/projection_radius/*.csv`

and generate:
- `results/tables/*.md` (ready to paste into appendix)

---

## Phase 1 (Critical): "Theory Saver" — Unroll Sensitivity

### Objective

Demonstrate that contraction truly behaves like a stability dial by showing predictions change less when evaluated at deeper unroll depths than trained.

This is the experiment that converts your "pivot" into evidence. Without it, the theory-to-results link is weak.

### Select checkpoints (two models, matched variant)

**Important:** Use the same TRM variant for A and B (episodic-z vs persistent-z). Otherwise you will be measuring architecture differences, not contraction effects.

- **Model A (Baseline):** best-performing No Contraction checkpoint.
  - Example family: `configs/ablations/upi_trm_feasibility_no_contraction.yaml`

- **Model B (Constrained):** contraction ON but value-head norm OFF (to avoid the known instability confound).
  - Required toggles:
    - `enable_contraction: true`
    - `target_Lz < 1.0`
    - `disable_value_head_norm: true` (create the ablation YAML if missing)

(Your own paper/summary already isolates value-head norm as the blow-up driver; don't reintroduce that confound.)

### Build evaluation batches (static + closure-aware)

You need more than initial states. Your theory's norms are implicitly over a closure (states visited + one-step successors); measure something close to that.

**Batch B0: Initial states** (static, reproducible)
- 100 puzzles total, stored as puzzle IDs + seed.
- Recommended mix:
  - 40 from trivial (1–4 empties)
  - 30 from ultra-easy (still 1–4 empties)
  - 30 from the harder 6–8 empties suite (use the same split you used for the appendix hard eval)

**Batch B1: One-step successor closure** (static, model-aware but reproducible)

For each state s in B0:
- Sample a fixed set of actions and generate successor states `s' = edit(s, a)`:
  - 5 actions = top-5 under Model A at `n = n_train`
  - 5 actions = top-5 under Model B at `n = n_train`
  - 5 actions = random legal actions (fixed RNG seed)
- Add those successors to the batch.
- Deduplicate and cap to a fixed size (e.g., 1,000–2,000 states) to keep eval cheap.

This approximates the "advantage-evaluation closure" idea in practice and avoids the "you only tested easy initial states" reviewer attack.

### Unroll depths to evaluate

Let:
- `n_train` = checkpoint's training `inner_unroll_n` (e.g., 16)

Evaluate at:
- `n ∈ {n_train, 2·n_train, 4·n_train}`
- Optional "deep": `n_deep = 64` (only if runtime allows), used as a proxy for "near fixed point"

### Metrics (computed on both batches B0 and B1)

For each model and each batch:
- `Δ_V(n_train, 2·n_train)`, `Δ_V(n_train, 4·n_train)`
- `Δ_π(n_train, 2·n_train)`, `Δ_π(n_train, 4·n_train)`
- argmax agreement for the same pairs
- `Δ_z(n_train, 2·n_train)`, `Δ_z(n_train, 4·n_train)`
- `L̂_z`, projection saturation rate, TD/residual proxy summaries

### Execution

1. Build and freeze B0 (puzzle list + seed).
2. Build and freeze B1 from B0 using the action sampling protocol above.
3. For each model (A, B), run forward passes at each n on B0 and B1.
4. Compute and save all metrics + summary tables.

### Deliverables (paper-ready)

- **Figure:** `Δ_V` and `Δ_π` vs. depth multiplier (1×, 2×, 4×) with error bars across states.
- **Table (appendix):** `L̂_z`, saturation rate, and p99 `Δ_V` for A vs B, split by B0 vs B1.

### Expected outcome / narrative

- **Model A:** larger `Δ_V`, `Δ_π` as depth increases (more truncation sensitivity).
- **Model B:** materially smaller `Δ_V`, `Δ_π` and smaller tail risk (p99/max), consistent with "stability dial".

### Failure interpretation (don't spin it)

If Model B does not reduce unroll sensitivity:
- You do not have evidence that contraction is a meaningful stability control in this setting.
- You must either:
  - reframe contraction as a monitoring diagnostic (track `L̂_z`, don't enforce it), or
  - show your unconstrained model already learns `L̂_z < 1` in the regimes that matter (so enforcement is unnecessary).

---

## Phase 2 (High): Projection Radius / "Always Clipping" Ablation

### Objective

Rule out the biggest confound in your own diagnostics: projection saturation. If `||z||_pre ≫ R` almost always, then your "stability" may be an artifact of constant clipping rather than true contractive dynamics.

### Conditions

Pick one setting (either best A/B checkpoints or short retrains) and sweep:
- `latent_ball_radius ∈ {10, 30, 100, inf}`

where `inf` disables projection (no clipping).

Run for:
- No-contraction baseline
- Mild contraction baseline (e.g., `target_Lz = 0.99`, V-head norm OFF)

### Measurements

For each R:
- saturation rate + pre/post `||z||` stats
- `Δ_V`, `Δ_π`, `Δ_z` at (1× vs 4×) unroll
- success rate (secondary; stability is the main point here)

### Deliverable

**Appendix table:** (R, saturation %, p99 ΔV, success) for each condition.

---

## Phase 3 (High): "Sweet Spot" Sweep (Contraction Strength)

### Objective

Turn contraction from a binary "kills learning" into a dial curve: stability improves smoothly as contraction strengthens, but success drops with some knee.

### Variables / conditions

Train on Trivial 4×4 (1–4 empties), with:
- `enable_contraction: true`
- `disable_value_head_norm: true`

Sweep contraction targets (include at least one "almost no-op"):
- `target_Lz ∈ {0.999, 0.99, 0.95, 0.90}`

### Seed discipline

- **Minimum:** 1 seed to identify gross trends.
- **Prefer:** 3 seeds for the 2–3 most informative points (e.g., 0.999, 0.99, 0.95).

### Measurements (per checkpoint)

- success rate
- unroll sensitivity metrics from Phase 1 on B0 and B1
- achieved `L̂_z` (do not trust targets)
- saturation rate + `||z||` norms

### Desired plot (submission-critical)

- **x-axis:** achieved `L̂_z` (not `target_Lz`)
- **y-axis (left):** success rate
- **y-axis (right):** `Δ_V(n_train, 4·n_train)` (or `Δ_π`)
- annotate points with saturation % (or facet by R if you ran Phase 2)

**This plot is the "stability dial" in one picture.**

---

## Phase 4 (Medium): Multi-seed Replication of the 2×2 Norm Ablation

### Objective

Turn the "value-head norm caused collapse" claim into something reviewers believe beyond a one-off diagnostic.

### Design

Replicate across multiple seeds:
- `z→z` contraction: OFF / ON
- value-head norm: OFF / ON

Report:
- stability metrics (target clamp-hit rate, `Var(V)`, `L̂_z`)
- success rate on trivial + (optionally) hard suite

### Deliverable

**Appendix table/figure** summarizing mean±std across seeds.

---

## Phase 5 (Optional, High Value): Centering + α Sensitivity (Theorem-Alignment Diagnostic)

### Objective

Your theory claims conservative mixture updates plus statewise centering reduces evaluation-error penalty to `O(α)`. You currently do approximate centering; reviewers can dismiss this as "nice math, not implemented."

### Minimal viable test (no full retrain required)

On a small batch of states:

1. Compute advantage baseline two ways:
   - **Exact baseline** (sum over all 97 actions) on a small batch
   - Your current approximate/batch-centered baseline

2. Report:
   - centering defect: `ε_cent = sup_s |E_{a~π}[Â(s,a)]|`

3. Apply a single policy update step with varying α and measure:
   - KL shift vs α
   - whether update behavior tracks "linear in α" in the exact-centered case

### Deliverable

**Tiny appendix figure:** centering defect comparison + a "one-step KL vs α" line plot.

---

## Phase 6 (Low / Contingency): Narrative Sanity Checks (Baselines)

### 1) Random baseline verification (hard suite)

- Run random baseline for 10k episodes on the hard 6–8 empties suite.
- Save CSV with confidence intervals.

### 2) Add a "Greedy checker-improvement" baseline (recommended)

A deterministic baseline:
- take the legal edit that maximizes immediate checker improvement (or shaped reward) per step.

This is cheap and will expose whether the environment + shaping are pathological.

### 3) Baseline tuning (only if cheap)

If PPO/A2C < random, try:
- higher entropy coefficient (PPO/A2C)
- STOP penalty adjustments / exploration schedule sanity

But do not burn cycles here if Phase 1–3 aren't complete.

---

## Summary Checklist

### Must complete (submission-critical)

- [ ] **Phase 0:** Log `Δ_V`, `Δ_π`, `Δ_z`, `L̂_z`, saturation rate, residual proxy.
- [ ] **Phase 1:** Unroll sensitivity A vs B on B0 and B1 (depths 1×/2×/4×).
- [ ] **Phase 3:** Contraction sweep; plot success vs stability using achieved `L̂_z`.
- [ ] **Phase 2:** Projection radius sweep (at least 2–3 R values) to rule out clipping artifacts.

### Strong additions if time remains

- [ ] **Phase 4:** Multi-seed 2×2 norm ablation replication.
- [ ] **Phase 5:** Exact-centering α diagnostic.
- [ ] **Phase 6:** Random + greedy baseline sanity.

---

## Exp2b: Diagnosing and Calibrating the Contraction Dial

**Date Added:** 2026-01-17
**Context:** Exp2 (commit a460efd) shows achieved_Lz saturates at ~0.23 regardless of target_Lz, with high seed variance dominating. The "dial" claim is not supported by current data.

### Goal

Either:
- **(A)** Demonstrate a controllable contraction dial where achieved_Lz spans a meaningful range (e.g., 0.4→0.95) and stability metrics improve as achieved_Lz decreases, OR
- **(B)** Prove the current enforcement cannot realize distinct achieved_Lz (projection/architecture saturation) and document as a negative/limitations result.

### Hard Constraint

**`disable_value_head_norm: true`** everywhere. No exceptions.

---

### Step 1: Diagnose WHY achieved_Lz saturates

**Instrumentation required** (in `scripts/diagnose_contraction_saturation.py`):

1. **L_hat_preproj**: max ||f(z+δ)-f(z)||/||δ|| WITHOUT projection (ΠR disabled)
2. **L_hat_postproj**: max ||ΠR(f(z+δ)) - ΠR(f(z))||/||δ|| WITH projection
3. **projection_active_rate**: fraction of samples where ||f(z)||₂ > R
4. **Norm distributions**: mean/median/p95 of ||z||, ||f(z)||, ||ΠR(f(z))||, (||f(z)||-R)⁺
5. **clamp_activity**: how often operator-norm clamping triggers

**Run on:**
- Existing Exp2 checkpoints: target_Lz ∈ {0.9, 0.95, 0.99, 0.999}
- All 3 seeds per target
- Eval-time R ∈ {10, 100, 1000, disabled} (without retraining)

**Output:**
- `results/paper_ready/exp2/DIAGNOSTICS_saturation.json`
- `results/paper_ready/exp2/DIAGNOSTICS_saturation.md`

**Decision Gate:**

| Condition | Root Cause | Action |
|-----------|------------|--------|
| L_hat_preproj varies, L_hat_postproj saturates, projection_active_rate > 80% | Projection dominates | Increase R or disable projection for dial |
| Both L_hat saturate, clamp_activity high | Enforcement/clamp saturates | Add explicit z→z scaling knob |
| Neither explains | Unknown | Document and stop |

---

### Step 2: Attempt ONE clean fix (if dial is broken)

**Case A: Projection dominates**
- Train/eval with rl_latent_ball_radius = 1000 or R=0 (disabled)
- Keep z→z contraction enforcement for stability
- Verify L_hat_preproj varies with target

**Case B: Clamp/schedule saturates**
- Add explicit `contraction_lambda` knob that scales z→z pathway output
- Verify knob changes L_hat_preproj measurably on fixed batch BEFORE training

**Validation before sweep:**
- Dry-run (forward-only) must show ≥3 separated achieved_Lz bands (e.g., ~0.4, ~0.6, ~0.8, ~0.95)
- Must NOT rely on projection to create separation

---

### Step 3: Exp2b sweep (only if achieved_Lz can be varied)

**Training config:**
- Same architecture as Exp1/Exp2 (episodic_latent fixed)
- use_feasibility_checker: true
- **disable_value_head_norm: true**
- 4 dial settings yielding distinct achieved_Lz
- 3 seeds minimum (match Exp1 seeds: 41, 42, 43)

**Evaluation:**
- achieved_Lz (both preproj and postproj, plus projection_active_rate)
- success rate
- Stability under depth mismatch (Exp1 protocol):
  - n_train=2, evaluate n2 ∈ {4, 8, 16}
  - B0 (initial states) primary
  - Metrics: ΔV, Δπ, argmax agreement

**Output artifacts:**
- `results/paper_ready/exp2b/fig_exp2b_dial_vs_stability.pdf`
- `results/paper_ready/exp2b/table_exp2b_dial_vs_stability.tex`
- `results/paper_ready/exp2b/CLAIMS.md` (scoped claims only)
- `results/paper_ready/exp2b/PROVENANCE.md`
- `results/paper_ready/exp2b/AUDIT.md`

**Audit checks:**
- [ ] achieved_Lz spans ≥0.25 range across dial settings
- [ ] value-head norm OFF confirmed in all configs
- [ ] projection_active_rate reported (flag if >80%)
- [ ] Claims match table values (no unsupported monotonicity)

---

### Step 4: If Exp2b fails → finalize negative result

If distinct achieved_Lz cannot be produced without destabilizing training:

1. Keep Exp2 (a460efd) as negative/limitations result
2. Update `results/paper_ready/exp2/CLAIMS.md`:
   - "Target-Lz is not a reliable dial; achieved_Lz saturates; variance dominates"
3. Add diagnostics (DIAGNOSTICS_saturation.*) explaining "why"
4. Ensure audit passes and CLAIMS.md does NOT use "monotonic" language

---

### Deliverables Checklist

- [ ] EXPERIMENT_PLAN_ICML.md updated with Exp2b plan
- [ ] DIAGNOSTICS_saturation.json/.md created
- [ ] Either:
  - (A) exp2b paper-ready bundle + audit pass, OR
  - (B) exp2 negative-result bundle with diagnostics + scoped claims + audit pass
- [ ] Clean commit: "Exp2b: calibrate contraction dial (or document saturation)"

---

## Exp2c: Unmask Stability Dial by Disabling Projection Dominance

### Context

Exp2b diagnostics revealed:
- L_preproj varies (0.69-0.77) across target_Lz values
- L_postproj saturates (~0.22-0.24) at R=10 with projection_active_rate=100%
- Root cause: Projection to R=10 dominates and masks the underlying network contraction

### Hypothesis

When projection is mostly inactive (projection_active_rate ≪ 1), achieved Lipschitz (measured on the actual update map used at eval) should:
1. Vary with contraction setting (target_Lz)
2. Predict unroll-sensitivity metrics (ΔV, Δπ, argmax agreement) under depth mismatch

### Decision Gates

**Gate G1 (Projection Dominance Check):**
- For chosen R_eval, projection_active_rate < 20% on B0 (preferably < 5%)
- If not, increase R_eval until it holds (30→100→300) or switch to R=disabled

**Gate G2 (Dial Range Check):**
- Achieved L_preproj must have non-trivial spread across sweep points
- Threshold: ≥ 0.08 absolute range across conditions averaged over seeds
- If not, widen target_Lz range (e.g., include 0.6/0.7) or adjust enforcement scaling

**Gate G3 (Stability Linkage):**
- Stability metrics under mismatch (ΔV at n_train=2 vs n_eval=16 on B0) must correlate with achieved L_preproj in expected direction:
  - Lower L_preproj ⇒ lower ΔV/Δπ and higher argmax agreement
- If correlation is absent, treat Exp2c as negative result (no "dial works" claim)

### Step 1: Exp2c-lite (Evaluation-Only, No Retraining)

Use existing Exp2 checkpoints and re-evaluate under radii that make projection inactive.

**Evaluation Protocol:**
1. For each checkpoint (lz_{0900,095,099,0999}/seed{41,42,43}):
   - Evaluate at R_eval ∈ {10, 100, disabled}
   - Compute and log:
     - L_preproj: finite-diff Lipschitz of f_θ without Π_R
     - L_postproj: finite-diff Lipschitz of Π_R∘f_θ under R_eval
     - projection_active_rate: fraction of updates where Π_R changed z
     - mean pre-projection and post-projection latent norms

2. Compute dial target metrics under depth mismatch:
   - Use same unroll-sensitivity protocol as Exp1 (for comparability)
   - n_train=2; evaluate at n2 ∈ {4, 8, 16}
   - Batch: B0 (primary for strong claims)
   - Metrics: ΔV, Δπ, argmax agreement

3. CRITICAL: Keep disable_value_head_norm: true (load models with value-head norm OFF)

**Output:** Internal report answering:
- At what R_eval does projection_active_rate drop <20%?
- Does achieved L_preproj vary meaningfully across target_Lz?
- Do stability metrics vs depth mismatch track achieved L_preproj?

**Decision:**
- If G1-G3 pass → proceed to Step 3 (paper-ready packaging)
- If G2 or G3 fail → proceed to Step 2 (retraining) or document negative result

### Step 2: Exp2c-full (Retraining, Only If Needed)

Train contraction sweep with projection not dominating during BOTH training and eval:

**Training Configuration:**
- rl_latent_ball_radius: 100 (or higher if projection_active_rate still high)
- disable_value_head_norm: true
- use_feasibility_checker: true
- Sweep: target_Lz ∈ {OFF (no contraction), 0.9, 0.7, 0.6}
- Seeds: 3 per condition (match Exp1 discipline)
- Training budget: identical across conditions

### Step 3: Paper-Ready Artifact Generation

Create `results/paper_ready/exp2c/` with:

**Figure: Stability Dial (Unmasked)**
- Panel A: achieved L_preproj vs target_Lz (error bars over seeds)
- Panel B: ΔV at deepest mismatch (n2=16 vs n_train=2) on B0
- Panel C: Δπ at deepest mismatch on B0
- Panel D (optional): argmax agreement at deepest mismatch

**Table:** Per target_Lz (and OFF if included):
- achieved L_preproj
- projection_active_rate
- ΔV (deep mismatch)
- Δπ (deep mismatch)
- R_eval used (explicit)

**CLAIMS.md:**
- If G1-G3 pass: "Dial works ONLY when projection is mostly inactive; projection masks dial at R=10"
- If not: Negative result; no monotonicity claim; explain failure mode

**PROVENANCE.md:**
- Commands, commit hash, checkpoints, exact R_eval, seeds

**AUDIT.md + audit script:**
- Check projection_active_rate thresholds reported
- Claims match tables
- "Dial works" claim conditioned on projection inactive
- Value-head norm OFF in all runs
- No "Model A/B" labels

### Step 4: Buck Targets

Add targets:
- `//buiksat_trm:eval_exp2c_lite` - run evaluation-only analysis
- `//buiksat_trm:make_paper_figures_exp2c` - generate paper artifacts
- `//buiksat_trm:audit_exp2c_paper_ready` - run audit

### Deliverables Checklist

- [ ] EXPERIMENT_PLAN_ICML.md updated with Exp2c plan
- [ ] Exp2c-lite evaluation completed with gate analysis
- [ ] Paper-ready artifacts in results/paper_ready/exp2c/
- [ ] Audit passes (projection thresholds, claims match, value-head norm OFF)
- [ ] Clean commit: "Exp2c: Unmask stability dial (projection inactive) + paper-ready artifacts"


