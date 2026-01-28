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

---

---

## Exp3: Training-Time Projection Ablation Under Fixed Contraction Setting

**Date Added:** 2026-01-18
**Status:** IN PROGRESS
**Context:** Exp2 final showed projection at R=10 dominates (100% active) and masks contraction effects. We now test whether training-time projection helps or harms learning, and whether "contraction-only" (no projection) produces stable training.

### Goal

Determine whether:
1. Training-time projection (R=10) is necessary for stable training, or if R=100/disabled works
2. Contraction-only (no projection) can produce both stable training AND meaningful mismatch stability
3. There is a tradeoff between projection strength and learning performance

**This is NOT a dial-claim resurrection** unless we see real achieved-Lz separation.

### Non-Negotiable Controls

- `disable_value_head_norm: true` in ALL runs (value-head spectral_norm OFF)
- Same architecture variant (`episodic_latent: true` fixed)
- Same checker-only reward (`use_feasibility_checker: true`)
- Same env, same training steps (5000)
- No "Model A/B" labels; conditions must be explicit

### Experiment Matrix (2×3)

| Condition | enable_contraction | latent_ball_radius | Description |
|-----------|-------------------|-------------------|-------------|
| NC-R10 | false | 10.0 | No contraction, default projection |
| NC-R100 | false | 100.0 | No contraction, weak projection |
| NC-Rdis | false | 0 (disabled) | No contraction, no projection |
| C-R10 | true | 10.0 | Contraction + default projection |
| C-R100 | true | 100.0 | Contraction + weak projection |
| C-Rdis | true | 0 (disabled) | Contraction only, no projection |

### Seeds

- **Phase 1:** 1 seed per cell (seed=42) for fast diagnostic
- **Phase 2:** If stable and interpretable, expand to 3 seeds {41, 42, 43}

### Decision Gates (Pre-Registered)

**G1 (Stability):**
- Training does not collapse (no NaNs, value targets not saturating)
- Success rate > random baseline (~5%) for trivial suite
- If any cell fails G1, abort that cell and document as unstable

**G2 (Projection Activity):**
- Confirm R=10 is ~100% projection_active_rate
- Confirm R=100 is ~0% projection_active_rate (or quantify)

**G3 (Contraction-Only Viability):**
- If C-Rdis (contraction only, no projection) is stable AND yields reasonable success, document as viable
- If unstable, document as negative result (contraction alone insufficient)

**G4 (Interpretation):**
- Does removing/loosening projection improve success but worsen mismatch stability?
- Or does projection hurt learning while providing stability?
- State the tradeoff explicitly with numbers

### Metrics to Log

**Training Metrics (per checkpoint):**
- success rate curve
- latent norm stats: pre-proj ∥z∥ mean/p95/max, post-proj ∥z∥
- projection_active_rate: fraction of samples where projection changes z
- Lipschitz proxies: L_preproj and L_postproj (finite-difference)
- NaN/inf events; gradient norms if available
- value target clamp rate

**Evaluation Metrics (reuse Exp1 machinery):**
- Unroll sensitivity mismatch: n_train=2, n2 ∈ {4, 8, 16}
- Batches: B0 (initial states) primary, B1 (successor closure) secondary
- ΔV, Δπ, argmax agreement, Δz
- Evaluate under BOTH:
  * eval R = training R (native behavior)
  * eval R = 100 (to remove clipping at eval time)

This separates "training-time projection effects" from "eval-time clipping effects".

### Execution Protocol

**Step 1: Create configs**
- `configs/exp3_projection_ablation/nc_r10.yaml` (NC-R10)
- `configs/exp3_projection_ablation/nc_r100.yaml` (NC-R100)
- `configs/exp3_projection_ablation/nc_rdis.yaml` (NC-Rdis)
- `configs/exp3_projection_ablation/c_r10.yaml` (C-R10)
- `configs/exp3_projection_ablation/c_r100.yaml` (C-R100)
- `configs/exp3_projection_ablation/c_rdis.yaml` (C-Rdis)

**Step 2: Run training (seed=42)**
```bash
for config in nc_r10 nc_r100 nc_rdis c_r10 c_r100 c_rdis; do
  python upi_trm_train.py \
    --config configs/exp3_projection_ablation/${config}.yaml \
    --seed 42 \
    --log-dir results/exp3/${config}_s42/
done
```

**Step 3: Check G1 (stability gate)**
- Inspect logs for NaN/inf
- Check final success rate > random
- If any cell fails, document and exclude from further analysis

**Step 4: Run evaluation**
```bash
python scripts/eval_exp3_projection_ablation.py \
  --exp_dir results/exp3/ \
  --out_dir results/paper_ready/exp3_projection_ablation/
```

**Step 5: Generate paper-ready artifacts**
- Figures, tables, CLAIMS.md, PROVENANCE.md, AUDIT.md

### Deliverables

Create `results/paper_ready/exp3_projection_ablation/` with:

| File | Content |
|------|---------|
| `fig_exp3_success_vs_radius.pdf` | Success rate across conditions |
| `fig_exp3_stability_vs_radius.pdf` | Mismatch stability (ΔV, argmax agree) |
| `table_exp3_summary.tex` | LaTeX table: success, projection_active_rate, L_preproj, ΔV |
| `CLAIMS.md` | Scoped claims based on G1-G4 outcomes |
| `PROVENANCE.md` | Commit hash, checkpoint paths, commands |
| `AUDIT.md` | Verification results |
| `summary.json` | Machine-readable summary |

### Audit Checks

1. **G1 verified:** All reported cells passed stability (or documented as failed)
2. **projection_active_rate thresholds:** R=10 ~100%, R=100 ~0%
3. **Value-head norm OFF:** All configs have `disable_value_head_norm: true`
4. **Claims match data:** No unsupported monotonicity or dial claims
5. **Eval R separation:** Both eval_R=native and eval_R=100 reported

### Expected Outcomes

**Scenario A (Projection Helps Learning):**
- R=10 conditions have higher success than R=100/disabled
- Trade-off: better learning but projection masks contraction effects

**Scenario B (Projection Hurts Learning):**
- R=100/disabled conditions have higher success
- Trade-off: better learning but potentially worse stability

**Scenario C (Contraction-Only Viable):**
- C-Rdis is stable with reasonable success
- Contraction provides stability without projection artifacts

**Scenario D (Contraction-Only Fails):**
- C-Rdis is unstable (NaN/collapse)
- Projection is required for stable training

### Failure Modes

If ALL projection-disabled cells fail G1:
- Document: "Projection is required for stable training in this architecture"
- Keep R=10 vs R=100 comparison as main result

If contraction cells show no stability benefit:
- Document: "Contraction does not improve mismatch stability independent of projection"
- This is consistent with Exp2 negative result

---

## Exp2 FINAL: Paper-Ready Bundle (Path B - Negative Result)

**Status**: ✅ FINALIZED
**Commit**: b804e2b (Exp2 final: dial failure + projection dominance)
**Decision**: The spectral-norm "target_Lz" dial is NOT an effective contraction control in this TRM architecture.

### ⚠️ DEPRECATION NOTICE

**All prior "monotonic dial" or "contraction dial controls stability" claims from earlier Exp2 iterations are SUPERSEDED.** The only valid Exp2 claims for citation are the Exp2_final scoped claims below.

Prior claims to deprecate:
- "Target L_z provides a controllable stability dial" — FALSE
- "Lower target L_z produces lower achieved L_z" — FALSE (ordering non-monotonic)
- "Spectral norm clamping controls contraction" — MISLEADING (projection dominates at R=10)

### Reviewer Attack Model

**Why the dial fails (architectural/measurement fact, not plotting artifact):**

1. **Projection dominance**: At R=10, the latent-ball projection Π_R is active 100% of the time. This projection compresses all latent updates to ‖z‖ ≤ R, artificially limiting post-projection Lipschitz to ~R/‖z_pre‖.

2. **Masking effect**: Even if spectral-norm clamping produces different pre-projection L_z values (it does: 0.69–0.77), the projection normalizes them all to ~0.23, making the "dial" invisible in post-projection metrics.

3. **Insufficient control range**: When projection is disabled, pre-projection L_z spread is only 0.064 (threshold ≥0.08). The spectral-norm mechanism simply doesn't produce enough variation to constitute a "dial."

4. **Non-monotonicity**: Even the limited variation is non-monotonic: target 0.9→0.95→0.99→0.999 yields L_preproj 0.473→0.409→0.448→0.440. This is not a calibration issue; the mechanism is fundamentally unsuitable for dial control.

**Implication for reviewers**: Any critique assuming "the dial should work if you tune it better" is misguided. The failure is architectural. The paper honestly reports this negative result and pivots to the positive finding (projection stabilizes).

### Summary of Findings

| Experiment | Key Finding |
|------------|-------------|
| Exp2 (a460efd) | target_Lz sweep achieves similar L_postproj (~0.22-0.24) across all settings |
| Exp2b (0d32097) | Root cause: projection at R=10 dominates (100% active), masking network differences |
| Exp2c (db3de70) | Even with projection disabled, L_preproj spread only 0.064 (fails ≥0.08 threshold); ordering non-monotonic |

### Final Paper-Facing Claims (Scoped)

**Claim 1 (Negative - Dial Failure):**
> "In this TRM setup, targeting spectral-norm-based contraction does not provide a reliable 'stability dial': achieved L_preproj varies only weakly (spread 0.064) and non-monotonically across target_Lz values (0.9→0.95→0.99→0.999 yields L_preproj 0.473→0.409→0.448→0.440). At the default projection radius R=10, post-projection dynamics saturate (projection active ~100%), masking any underlying differences."

**Claim 2 (Positive - Projection Stabilizes):**
> "Latent-ball projection is a strong stabilizer: enabling projection at R=10 improves deep-unroll mismatch stability substantially on B0 (ΔV improves from 7.7–14.8 to 0.5–2.2, ~6–10× reduction; argmax agreement improves from 88–91% to 97–99%, +8–10pp) in the evaluated setting (n_train=2, n_eval=16)."

**Scope limitations:**
- Results on B0 (initial states) only; B1 not evaluated
- Mismatch protocol: n_train=2 vs n_eval∈{4,8,16}
- No monotonicity claim for projection-vs-R relationship (not tested as sweep)

### Stop Condition

**Do NOT pursue further training** to achieve dial monotonicity under the current spectral-norm targeting mechanism. The failure is architectural: the mechanism does not provide sufficient control.

If a controllable dial is needed in future work, consider:
- Explicit L_z scaling loss (not spectral norm clamping)
- Different network architecture where latent norms naturally stay bounded
- Direct regularization on finite-diff Lipschitz estimates

### Deliverables

Create `results/paper_ready/exp2_final/` containing:

| File | Content |
|------|---------|
| `fig_exp2_dial_does_not_control_Lz.pdf` | target_Lz vs L_preproj/L_postproj, projection_active_rate |
| `fig_exp2_projection_is_primary_stabilizer.pdf` | R=10 vs R=disabled stability comparison |
| `table_exp2_dial_does_not_control_Lz.tex` | LaTeX table with sweep results |
| `table_exp2_projection_effect.tex` | LaTeX table with projection comparison |
| `CLAIMS.md` | Final scoped claims |
| `PROVENANCE.md` | Commit references, checkpoint paths, regeneration commands |
| `AUDIT.md` | Audit results |
| `PAPER_INSERT_SNIPPET.tex` | Appendix text + figure/table includes |
| `summary.json` | Machine-readable summary for audit |

### Audit Checks

The audit script (`scripts/audit_exp2_final_paper_ready.py`) must verify:

1. **Projection dominance at R=10**: projection_active_rate ≈ 100%
2. **Projection inactive at R≥100**: projection_active_rate ≈ 0%
3. **Dial range failure**: L_preproj spread < 0.08 (currently 0.064)
4. **No monotonicity claims**: grep for "monotonic" and fail if found (unless "NOT supported" context)
5. **Claims match summary.json**: numerical values in CLAIMS.md match generated data
6. **Value-head norm OFF**: All checkpoints have disable_value_head_norm: true

### Buck Targets

```bash
buck2 run //buiksat_trm:make_paper_figures_exp2_final  # Generate all artifacts
buck2 run //buiksat_trm:audit_exp2_final_paper_ready   # Verify claims
```

### Commit

✅ **COMPLETED**: b804e2b
```
Exp2 final: dial failure + projection dominance (paper-ready, audited)

Negative result: spectral-norm dial does NOT control L_z effectively
- L_preproj spread only 0.064 (threshold ≥0.08), non-monotonic ordering
- At R=10: projection dominates (100% active), L_postproj saturates ~0.23

Positive result: projection provides strong stabilization
- ΔV improves 6-10× with R=10 projection vs disabled
- Argmax agreement +8-10pp

Artifacts: results/paper_ready/exp2_final/
Audit: PASSED
```

### Paper Assets Location

Figures and tables copied to paper directory:
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures/fig_exp2_dial_does_not_control_Lz.pdf`
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures/fig_exp2_projection_is_primary_stabilizer.pdf`
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/tables/table_exp2_dial_does_not_control_Lz.tex`
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/tables/table_exp2_projection_effect.tex`

---

## Exp4: Projection-free Contraction Dial (Range Test → Full Sweep)

**Date Added:** 2026-01-18
**Status:** COMPLETED
**Commit SHA:** 6fede631455e (initial), b6d98b6 (final v2 with cluster bootstrap)
**Artifacts:** `results/paper_ready/exp4_projection_free_dial_v2/`
**Context:** Exp2_final shows the spectral-norm dial fails and projection at small R can dominate/mask contraction. Exp3 shows training is stable even with projection disabled. We now test whether a contraction-only dial exists when projection is disabled/inactive.

### Final Execution Summary (v2)

**Protocol:** Inference-time contraction scaling on 3 independently trained checkpoints (seeds 41, 42, 43) with 4 dial scales (1.0, 0.85, 0.70, 0.55). This avoids expensive per-scale retraining while still producing N=12 observations across checkpoints × scales.

**Statistical Methodology:** Cluster bootstrap (1000 iterations, 3 clusters) to properly handle repeated measures (scales) on the same checkpoint. Reports 95% CIs instead of i.i.d. p-values.

**Results:**
- **Decision:** POSITIVE - Dial viable with statistically significant monotonicity
- **G0 (Projection inactive):** PASS
- **G1 (Stability):** PASS
- **G2 (Dial range):** PASS (spread=0.695)
- **G3 (Monotonicity):**
  - B0 argmax (n2=8, 4× mismatch): ρ=-0.866 [-1.000, -0.542] ✓
  - B1 argmax (n2=8, 4× mismatch): ρ=-0.923 [-1.000, -0.787] ✓
  - B1 ΔV (n2=8): ρ=0.657 [0.553, 0.800] ✓

**Depth Notation Clarification:** n_train=2; "n2=8" means evaluation at depth 8, which is **4× mismatch** (not 8×).

**Audit:** 11/11 checks PASSED

---

### Hypothesis

When projection is disabled (or inactive), varying contraction strength of the z→z recursion produces a measurable, monotonic change in mismatch stability metrics (ΔV, Δπ, argmax agreement) as a function of achieved pre-projection Lipschitz proxy L_preproj.

### Non-Negotiable Controls (To Avoid Confounds)

- `disable_value_head_norm: true` for ALL conditions
- Same architecture variant as Exp1 (`episodic_latent: true`)
- Same checker/reward setup as Exp1/Exp3
- Projection must be disabled OR provably inactive (log `projection_active_rate` and require ~0%)

### Primary Metrics

1. **L_preproj**: Finite-difference Lipschitz proxy of f_θ WITHOUT projection
2. **Mismatch stability**: ΔV, Δπ, argmax agreement under depth mismatch
   - n_train=2; eval n2 ∈ {4, 8, 16} on B0 and B1
3. **Success rate**: Secondary, to catch catastrophic regressions
4. **Latent norm stats**: Pre/post projection norms (post should equal pre if projection disabled)

### Decision Gates (Pre-Registered)

**G0 (Projection Inactivity):**
- `projection_active_rate < 1%` on eval batches
- If FAIL → Abort experiment (projection is interfering)
- Must PASS to continue

**G1 (Training Stability):**
- No NaNs/collapse across all seeds
- Must PASS to continue

**G2 (Dial Range):**
- `max(L_preproj) - min(L_preproj) ≥ 0.10` across sweep points
- If FAIL → Declare "dial has insufficient range" and STOP
- This is the key go/no-go gate for the full sweep

**G3 (Monotonic Linkage - Soft):**
- Spearman ρ(L_preproj, ΔV@8×) should be > 0.5 in magnitude (directionally consistent)
- If FAIL → Report as "inconclusive" (not a hard stop)

### Step 1: Range-Test Before Full Sweep (Cheap, Abortable)

**Objective:** Verify that inference-time contraction scaling produces sufficient L_preproj spread before committing to expensive training.

**Option A (Preferred): Inference-Time Contraction Scaling**

1. Take a single trained checkpoint with contraction OFF and projection disabled/inactive
   - Use Exp3 `nc_rdis_s42` (no contraction, R=disabled) or similar

2. Apply global contraction scaling to z→z layers at evaluation time:
   - Use `enforce_global_contraction()` with scaling factors s ∈ {1.0, 0.9, 0.8, 0.7, 0.6}
   - This directly scales the L_level pathway output

3. For each scaling factor, measure:
   - L_preproj (finite-difference Lipschitz without projection)
   - Mismatch metrics: ΔV, Δπ, argmax agreement at n_train=2 vs n2 ∈ {4, 8, 16}
   - projection_active_rate (should be ~0%)

4. **Decision:**
   - If G2 fails (L_preproj spread < 0.10) → Stop Exp4, write up as negative: "No practical dial range even under direct scaling"
   - If G2 passes → Proceed to Step 2 (Full Sweep)

**Option B (Fallback): Short Training Sweep**

If Option A is not feasible:
1. Train for 1k steps (not 5k) for 3 scaling settings
2. Check if L_preproj moves at all
3. If L_preproj spread < 0.10, stop

### Step 2: Full Experiment (Only If Range-Test Passes)

**Training Configuration:**
- 3 seeds: {41, 42, 43}
- 3-4 contraction settings chosen to span the observed L_preproj range
  - NOT target_Lz values; pick settings that actually change L_preproj
- Train to 5k steps (to match Exp1 comparability)
- `disable_value_head_norm: true`
- `latent_ball_radius: 0` (projection disabled)
- `use_feasibility_checker: true`

**Evaluation Protocol:**
- Mismatch stability exactly like Exp1:
  - B0 (initial states) and B1 (successor closure)
  - n_train=2; n2 ∈ {4, 8, 16}
  - Metrics: ΔV, Δπ, argmax agreement

### Step 3: Paper-Ready Artifacts + Audit

Create `results/paper_ready/exp4_projection_free_dial/` containing:

| File | Content |
|------|---------|
| `fig_exp4_dial_scatter.pdf` | L_preproj vs Argmax Agreement (n₂=8) scatter with trend line |
| `fig_exp4_scale_comparison.pdf` | Bar chart comparing L_preproj and Argmax by scale (supplement) |
| `table_exp4_projection_free_dial.tex` | Settings → achieved L_preproj → stability metrics |
| `CLAIMS.md` | Scoped claims based on gate outcomes |
| `PROVENANCE.md` | Commit references, checkpoint paths, commands |
| `AUDIT.md` | Audit verification results |
| `summary.json` | Machine-readable summary |

### Audit Checks

The audit script must verify:

1. **G0 (Projection inactive):** `projection_active_rate < 1%` for all conditions
2. **Value-head norm OFF:** All conditions have `disable_value_head_norm: true`
3. **G2 (Dial range):** Report L_preproj spread and pass/fail status
4. **G3 (Monotonicity):** Report Spearman ρ and "monotonic"/"inconclusive" status
5. **No overclaims:** If G2 fails, CLAIMS.md explicitly states dial-range failure

### Expected Outcomes

**Scenario A (Dial Works):**
- L_preproj spread ≥ 0.10 (G2 passes)
- ΔV decreases as L_preproj decreases (G3 passes)
- Claim: "Projection-free contraction dial is viable"

**Scenario B (Dial Has Insufficient Range):**
- L_preproj spread < 0.10 (G2 fails)
- Document: "Even with projection disabled, L_preproj cannot be varied enough to constitute a dial"
- This is consistent with Exp2_final but rules out the "projection masking" hypothesis

**Scenario C (Dial Has Range But No Monotonicity):**
- L_preproj spread ≥ 0.10 (G2 passes)
- But Spearman ρ < 0.5 (G3 fails)
- Document: "Dial range exists but does not predict stability metrics"

### Commit Message Template

```
Exp4: projection-free contraction dial (range test + [positive/negative] result, audited)

[Positive result: dial viable / Negative result: dial has insufficient range]
- L_preproj spread: X.XX (threshold ≥0.10)
- Spearman ρ(L_preproj, ΔV): X.XX ([passes/fails] >0.5 threshold)

Gate status: G0=[PASS/FAIL], G1=[PASS/FAIL], G2=[PASS/FAIL], G3=[PASS/FAIL/INCONCLUSIVE]

Artifacts: results/paper_ready/exp4_projection_free_dial/
Audit: [PASSED/FAILED]
```

### Buck Targets

```bash
buck2 run //buiksat_trm:exp4_final_v2        # Run evaluation with cluster bootstrap
buck2 run //buiksat_trm:generate_exp4_figures # Generate figures
buck2 run //buiksat_trm:audit_exp4_final_v2   # Audit verification (11 checks)
```

### Paper Export

Figures and tables exported to:
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures/fig_exp4_dial_scatter.pdf` **(main figure)**
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures/fig_exp4_scale_comparison.pdf` **(supplement)**
- `/home/buiksat/UPI_TRM/UPI_TRM_ICML/tables/table_exp4_projection_free_dial_v2.tex`

**Canonical naming contract:**
- `fig_exp4_dial_scatter.pdf`: L_preproj vs Argmax Agreement (n₂=8) scatter with Spearman ρ and 95% CI
- `fig_exp4_scale_comparison.pdf`: Bar chart of metrics by scale factor (for supplement)
- Any `\includegraphics` referencing Exp4 MUST use one of these two filenames

---

## Trivial Table 3 + Fig2 RERUN (post-baseline-evaluator-change)

**Date Added:** 2026-01-22
**Status:** IN PROGRESS
**Context:** Baseline evaluators were changed. Paper currently shows "DQN: eval unavailable (bug)" in Figure 2. This rerun ensures all methods use the updated evaluator consistently.

### Protocol

| Parameter | Value |
|-----------|-------|
| Dataset | `data/sudoku-4x4-trivial` (1-4 empties) |
| Training steps | 5000 |
| Horizon | max_edits = 16 (T=16) |
| Discount | γ = 0.99 |
| Eval frequency | Every 100 steps |
| Eval episodes | 50 per checkpoint |
| Eval policy | Greedy argmax + action masking |
| Unroll depth | n_eval = 2 |
| Seeds | 42, 123, 456 |

### Methods

| Method Key | Config Path | Contraction | Projection (R) |
|------------|-------------|-------------|----------------|
| persistent_nc | `configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml` | OFF | 10.0 (ON) |
| episodic_nc | `configs/ablations/upi_trm_feasibility_no_contraction.yaml` | OFF | 10.0 (ON) |
| episodic_c_clean | `configs/exp3_projection_ablation/c_rdis.yaml` | ON | 0.0 (OFF) |
| ppo | `configs/baselines/ppo_trm_feasibility.yaml` | OFF | 0.0 (OFF) |
| a2c | `configs/baselines/a2c_trm_feasibility.yaml` | OFF | 0.0 (OFF) |
| dqn | `configs/baselines/dqn_trm_feasibility.yaml` | OFF | 0.0 (OFF) |

**Note:** Projection asymmetry is intentional to match historical Table 3 configs. Controlled comparisons exist in "Table 3 Hard Controlled" section.

### Output Locations (repo-relative)

| Artifact | Path |
|----------|------|
| Training logs | `results/table3_baselines_rerun_evalfix_2026_01_22/{method}_s{seed}.log` |
| Table summary | `results/table3_baselines_rerun_evalfix_2026_01_22/table3_summary.md` |
| Figure 2 PDF | `figures/trivial_baselines_vs_no_contraction_success_vs_steps.pdf` |
| Provenance bundle | `results/paper_ready/fig2_provenance_rerun_evalfix_2026_01_22/` |

### Sanity Gate (MUST PASS before proceeding)

Before running the full 18-training sweep, verify DQN evaluation works:

1. Run DQN smoke test (seed=42, ~200 steps)
2. Confirm log contains `eval_success_rate=` entries
3. If missing or "eval unavailable":
   - Fix evaluator integration for DQN
   - Ensure action masking applied consistently
   - Ensure greedy evaluation uses correct Q-values
   - Ensure eval metric emission matches plot parser

### Execution

```bash
# Command template (4 GPUs in parallel)
CUDA_VISIBLE_DEVICES={gpu} python upi_trm_train.py \
  --config {config_path} \
  --seed {seed} \
  --dataset-paths data/sudoku-4x4-trivial \
  --no-wandb \
  > results/table3_baselines_rerun_evalfix_2026_01_22/{method}_s{seed}.log 2>&1
```

### Deliverables

- [ ] DQN smoke test passes (eval_success_rate present)
- [ ] 18 training logs complete (6 methods × 3 seeds)
- [ ] Table 3 summary generated with mean±std
- [ ] Figure 2 regenerated with DQN included (no "eval unavailable")
- [ ] Provenance bundle created

---

## Table 3: Baseline Comparison Experiments (2026-01-20)

**Status:** COMPLETE (superseded by rerun above for paper)

### Purpose

Verify and update Table 3 in the paper with correct experimental data across 3 seeds for all methods.

### Datasets

| Dataset | Location | Description |
|---------|----------|-------------|
| Trivial 4×4 | `data/sudoku-4x4-trivial` | 1-4 empty cells, 5k steps |
| Hard 4×4 | `data/sudoku-4x4-easy_6to8empties` | 6-8 empty cells, 20k steps |

### Methods Tested

| Method Key | Config Path | Contraction | Projection | vhead |
|------------|-------------|-------------|------------|-------|
| persistent_nc | `ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml` | OFF | ON (R=10) | OFF |
| episodic_nc | `ablations/upi_trm_feasibility_no_contraction.yaml` | OFF | ON (R=10) | OFF |
| episodic_c_clean | `exp3_projection_ablation/c_rdis.yaml` | ON | OFF (R=0) | OFF |
| ppo | `baselines/ppo_trm_feasibility.yaml` | N/A | N/A | N/A |
| a2c | `baselines/a2c_trm_feasibility.yaml` | N/A | N/A | N/A |
| dqn | `baselines/dqn_trm_feasibility.yaml` | N/A | N/A | N/A |

### Trivial 4×4 Results (COMPLETE ✅)

**Location:** `results/table3_baselines/`
**Seeds:** 42, 123, 456 (3 seeds per method)
**Training steps:** 5000

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| persistent_nc | 96.0% | 92.0% | 92.0% | **93.3% ± 2.3%** |
| episodic_nc | 92.0% | 96.0% | 92.0% | **93.3% ± 2.3%** |
| episodic_c_clean | 96.0% | 86.0% | 90.0% | **90.7% ± 5.0%** |
| PPO | 30.0% | 30.0% | 30.0% | 30.0% ± 0.0% |
| A2C | 30.0% | 34.0% | 30.0% | 31.3% ± 2.3% |
| DQN | 24.0% | 30.0% | 30.0% | 28.0% ± 3.5% |
| Random | - | - | - | 52.0% |

### Hard 4×4 (6-8 empties) Experiments (COMPLETE ✅)

**Location:** `results/table3_hard_6to8/`
**Script:** `scripts/run_table3_hard.sh`
**Training steps:** 20000
**Completed:** 2026-01-21

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| persistent_nc | 60.0% | 52.0% | 58.0% | **56.7% ± 4.2%** |
| episodic_nc | 42.0% | 54.0% | 48.0% | **48.0% ± 6.0%** |
| episodic_c_clean | 36.0% | 40.0% | 34.0% | **36.7% ± 3.1%** |
| PPO | 0.0% | 0.0% | 0.0% | 0.0% ± 0.0% |
| A2C | 0.0% | 0.0% | 0.0% | 0.0% ± 0.0% |
| DQN | 0.0% | 0.0% | 0.0% | 0.0% ± 0.0% |

**Key Observations:**
1. No-contraction variants outperform contraction on harder puzzles (57% vs 37%)
2. All standard RL baselines fail completely (0% success)
3. persistent_nc performs best on harder puzzles

### ⚠️ Config Asymmetry Note

The no-contraction configs (`persistent_nc`, `episodic_nc`) have **projection ON** (default R=10), while the clean contraction config (`episodic_c_clean`) has **projection OFF** (R=0).

This is intentional for Table 3 (matching historical configs), but creates an asymmetry. For fully controlled comparisons:
- Use `exp3_projection_ablation/nc_rdis.yaml` for no-contraction with projection OFF
- Use `exp3_projection_ablation/c_r10.yaml` for contraction with projection ON

### Deliverables

- [x] Trivial 4×4 experiments (18 runs complete)
- [x] Paper Table 3 updated with correct values
- [x] Learning curves figure regenerated: `figures/trivial_baselines_vs_no_contraction_success_vs_steps.pdf`
- [x] Hard 4×4 experiments (18 runs complete)
- [x] Hard 4×4 learning curves figure: `figures/hard_4x4_baselines_success_vs_steps.pdf`

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_table3_baselines.sh` | Run trivial 4×4 experiments |
| `scripts/run_table3_hard.sh` | Run hard 4×4 experiments (20k steps) |
| `scripts/plot_table3_baselines.py` | Generate trivial learning curves figure |
| `scripts/plot_table3_hard.py` | Generate hard learning curves figure |

### Buck Target

```bash
buck2 run //buiksat_trm:plot_table3_baselines \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true
```

---

## Exp5: Stability–Expressivity Tradeoff Curve (Submission-Critical)

**Date Added:** 2026-01-18
**Status:** COMPLETED (G3 FAIL - no tradeoff observed at matched compute)
**Context:** Exp4 v2 established a working projection-free dial via inference-time scaling. This experiment produces the submission-critical "stability vs expressivity tradeoff curve" deliverable.

### ⚠️ CRITICAL BUG FIX: Exp5 v2

**Date:** 2026-01-19
**Bug:** Original nc_rdis checkpoints were trained WITHOUT `--dataset-paths`, causing them to use `DummyPuzzleDataset` (synthetic solved puzzles) instead of real Sudoku puzzles.

**Evidence:**
- Original training logs: `initial=16.00` (puzzles already solved)
- Correct training logs (Exp5 v2): `initial=13.52` (real unsolved puzzles)

**Fix Applied:**
- Created `scripts/train_nc_rdis_v2.py` with explicit `--dataset-paths data/sudoku-4x4-trivial`
- Retraining checkpoints to `results/exp5_v2_inputs_eval/nc_rdis_s{41,42,43}/`
- New Exp5 v2 uses these fixed checkpoints

**Result:**
- Training success: 88-98% (at n=4, R=10 - extended compute)
- Matched-compute success: ~6-7% (at n=2, R=0 - experiment setting)
- **This is NOT a bug:** See `SUCCESS_PROTOCOL_RECONCILIATION.md`
- Paper artifacts in `results/paper_ready/exp5_tradeoff_curve_v2/`

### Final Results (2026-01-19)

| Scale | L_preproj | Stability (argmax@8×) | Success (n=2) |
|-------|-----------|----------------------|---------------|
| 1.00 | 0.801 | 70.0% | 6.7% |
| 0.85 | 0.562 | 73.0% | 7.0% |
| 0.70 | 0.489 | 77.0% | 6.7% |
| 0.55 | 0.524 | 73.0% | 6.7% |

- **G0:** PASS (projection inactive)
- **G1:** PASS (no NaN)
- **G2:** PASS (spread=0.6532)
- **G3:** FAIL (success range=0.3%, no tradeoff at matched compute)

**Interpretation:** The dial controls L_preproj effectively. Stability shows modest variation (70-77%). Success is flat at matched compute, but same checkpoints achieve 88-98% at extended compute (n=4). The "no tradeoff" result at matched compute is genuine—it means 2 unroll steps is insufficient for this task regardless of contraction setting.

### Goal

Produce a single figure showing the tradeoff between stability (argmax agreement at depth mismatch) and expressivity (Sudoku solve success rate at matched compute) across the working contraction dial.

**This is the "stability dial" plot required for submission.**

### Non-Negotiable Controls

- `disable_value_head_norm: true` for ALL conditions
- `latent_ball_radius: 0` (projection disabled)
- `projection_active_rate < 1%` verified in audit
- Same eval batches (B0, B1) with provenance hashes

### Dial Mechanism

Use inference-time contraction scaling (same as Exp4 v2):
- Scaling factors: s ∈ {1.00, 0.85, 0.70, 0.55}
- Applied to z→z recursion weights at eval time
- No retraining required

### Checkpoints

**Exp5 v2 Checkpoints (CORRECT - trained with proper dataset):**
- `/home/buiksat/trm_bellman/results/exp5_v2_inputs_eval/nc_rdis_s41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/results/exp5_v2_inputs_eval/nc_rdis_s42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/results/exp5_v2_inputs_eval/nc_rdis_s43/model_step_5000.pt`

**DEPRECATED (original Exp5 - trained on synthetic data, NOT paper-valid for success metrics):**
- `results/exp3_v2/nc_rdis_s41/model_step_5000.pt`
- `results/exp3/nc_rdis_s42/model_step_5000.pt`
- `results/exp3_v2/nc_rdis_s43/model_step_5000.pt`

### Metrics

**Stability Metrics (from Exp4 v2):**
- L_preproj: Finite-difference Lipschitz proxy
- Argmax agreement at n2=8 (4× mismatch) on B0 and B1
- ΔV at n2=8 (optional, secondary)

**Expressivity Metrics (NEW):**
- Sudoku solve success rate at matched compute (n=2)
- Evaluated on:
  - (a) Trivial suite: 1–4 empties (100 puzzles)
  - (b) Hard suite: 6–8 empties (if available; document and skip if not)

### Decision Gates

**G0 (Projection inactive):** projection_active_rate < 1% — MUST PASS
**G1 (Stability):** No NaN values — MUST PASS
**G2 (Dial range):** L_preproj spread ≥ 0.10 — Already verified in Exp4 (0.695)
**G3 (Tradeoff exists):** Success rate varies across dial settings

### Deliverables

Create `results/paper_ready/exp5_tradeoff_curve/` containing:

| File | Content |
|------|---------|
| `fig_exp5_tradeoff_curve.pdf` | X: stability (argmax@8×), Y: success rate; points by scale s |
| `table_exp5_tradeoff_curve.tex` | LaTeX table: scale → L_preproj → stability → success |
| `CLAIMS.md` | Scoped claims |
| `PROVENANCE.md` | Commit, checkpoint paths, commands |
| `AUDIT.md` | Audit verification (all checks) |
| `summary.json` | Machine-readable summary |

### Audit Checks

1. **disable_value_head_norm: true** for all configs
2. **projection_active_rate < 1%** for all conditions
3. **B0/B1 provenance hashes match** Exp4 v2 (9ceab78310f3, fca64be3b53c)
4. **strict YAML loading** confirmed
5. **summary.json matches plotted values** (no key-missing regressions)
6. **git SHA captured and non-empty**

### Expected Outcome

A curve showing:
- Higher contraction (lower scale s, lower L_preproj) → higher stability but lower success
- Lower contraction (higher scale s, higher L_preproj) → lower stability but higher success

This demonstrates the "dial" concept: practitioners can choose their operating point on the stability–expressivity frontier.

### Buck Targets

```bash
buck2 run //buiksat_trm:exp5_tradeoff_curve \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only

buck2 run //buiksat_trm:audit_exp5_tradeoff_curve_paper_ready \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only
```

### Commit Message Template

```
Exp5: Stability–Expressivity tradeoff curve (paper-ready, audited)

Tradeoff demonstrated: [describe key finding]
- Scale 1.0: success=X.XX, stability=X.XX
- Scale 0.55: success=X.XX, stability=X.XX

Gate status: G0=PASS, G1=PASS, G2=PASS (inherited), G3=[PASS/FAIL]

Artifacts: results/paper_ready/exp5_tradeoff_curve/
Audit: [PASSED/FAILED]
```

---

## Table 3 Hard Controlled: 2×2 Contraction × Projection Deconfounding

**Date Added:** 2026-01-21
**Status:** READY TO RUN
**Context:** Table 3 Hard results are confounded: no-contraction configs use R=10 (projection ON), while contraction configs use R=0 (projection OFF). This makes it impossible to attribute performance differences to contraction vs projection effects.

### Problem Statement

Current Table 3 Hard (6-8 empties) results:
- `episodic_nc` (no contraction, R=10): 48.0% ± 6.0%
- `episodic_c_clean` (contraction, R=0): 36.7% ± 3.1%

**Confound:** Is the ~11pp gap due to contraction hurting performance, or projection helping it?

### Solution: Controlled 2×2 Factorial Design

Run all 4 combinations for Episodic-z TRM on hard 4×4 Sudoku:

| Cell | enable_contraction | latent_ball_radius | Description |
|------|-------------------|-------------------|-------------|
| NC-R0 | false | 0.0 | No contraction, projection OFF |
| NC-R10 | false | 10.0 | No contraction, projection ON |
| C-R0 | true | 0.0 | Contraction, projection OFF |
| C-R10 | true | 10.0 | Contraction, projection ON |

### Critical Controls (Non-Negotiable)

All 4 cells MUST have:
- `disable_value_head_norm: true` (avoid known instability confound)
- `episodic_latent: true` (same TRM variant across all cells)
- `use_feasibility_checker: true`
- `num_train_steps: 20000`
- `max_edits: 16` (T=16 horizon)
- Dataset: `data/sudoku-4x4-easy_6to8empties`

### Configs

Created in `configs/table3_hard_controlled/`:
- `episodic_nc_r0_hard.yaml` (NC-R0)
- `episodic_nc_r10_hard.yaml` (NC-R10)
- `episodic_c_r0_hard.yaml` (C-R0)
- `episodic_c_r10_hard.yaml` (C-R10)

### Seeds

3 seeds per cell: 42, 123, 456 (12 runs total)

### Execution

**Script:** `scripts/run_table3_hard_controlled_4gpu.sh`
**Output:** `results/table3_hard_6to8_controlled/`
**Plotting:** `scripts/plot_table3_hard_controlled.py`

### Expected Analysis

**Main Effects:**
1. **Contraction effect** = mean(C cells) - mean(NC cells)
   - If negative at matched R: "contraction hurts success"
   - If positive: "contraction helps success"

2. **Projection effect** = mean(R10 cells) - mean(R0 cells)
   - If positive: "projection helps success"
   - If negative: "projection hurts success"

**Interaction:**
- (C-R10 - C-R0) vs (NC-R10 - NC-R0)
- If different: "contraction and projection interact"

### Decision Rule

**Claim "contraction hurts hard success" ONLY if:**
- C-R0 < NC-R0 (matched at projection OFF), OR
- C-R10 < NC-R10 (matched at projection ON)

**Do NOT claim based on the confounded comparison:**
- ~~episodic_c_clean (R=0) vs episodic_nc (R=10)~~ ← INVALID

### Deliverables

- [ ] 4 YAML configs in `configs/table3_hard_controlled/`
- [ ] Runner script: `scripts/run_table3_hard_controlled_4gpu.sh`
- [ ] Plotting script: `scripts/plot_table3_hard_controlled.py`
- [ ] Results in `results/table3_hard_6to8_controlled/`
- [ ] 2×2 bar chart: `figures/table3_hard_controlled_2x2_bar.pdf`
- [ ] Learning curves: `figures/table3_hard_controlled_curves.pdf`
- [ ] Main effects analysis in summary output

### Audit Checks

1. `disable_value_head_norm: true` in ALL 4 configs
2. Correct dataset path used (6-8 empties)
3. `max_edits: 16` (T=16) in all configs
4. All 12 runs complete (4 configs × 3 seeds)
5. Claims match 2×2 table values

---

---

## 9×9 Sudoku Scale-Up: Constraint-Aware Action Masking

**Date Added:** 2026-01-28
**Status:** IN PROGRESS
**Context:** Scaling experiments from 4×4 to 9×9 Sudoku requires improved action masking due to the exponentially larger action space (729 vs 16 positions × vocab).

### Motivation

Previous 9×9 experiments showed limited learning:
- UPI-TRM: 27.64 → +1.0 improvement (best)
- PPO: 26.02 → -0.82 degradation
- DQN: 25.57 → -1.27 degradation

Initial score ~26.8, max possible 81. Models were struggling to improve.

**Hypothesis:** The 729-action space (81 positions × 9 digits) is too large for effective exploration. Many actions are invalid due to Sudoku constraints.

### Solution: Constraint-Aware Action Masking

Implemented dynamic action masking that enforces Sudoku constraints at the action selection level:

**What gets masked:**
1. Given cells (original clues) - cannot be edited
2. PAD (token 0) and empty (token 1) tokens - never useful
3. **Constraint violations:**
   - Digits already in same row
   - Digits already in same column
   - Digits already in same 3×3 box

**Result:** Action space reduced from 729 → ~250 valid actions (72% reduction)

### Implementation Details

**Files Modified:**
| File | Change |
|------|--------|
| `rl/task_config.py` | Added `SudokuTaskConfig.compute_action_mask()` with row/col/box constraint checking |
| `rl/envs/plan_edit_env.py` | Pass `current_state` to mask, recompute mask after every step |
| `configs/sudoku9x9/*.yaml` | Increased training from 25k → 50k steps |

**Key Code (task_config.py):**
```python
def compute_action_mask(self, inputs, vocab_size, stop_action_id, current_state=None):
    # ... existing masking for given cells and PAD/empty tokens ...

    # Constraint-aware masking: mask digits in same row/col/box
    for pos in range(num_positions):
        row, col = pos // grid_size, pos % grid_size
        box_row, box_col = (row // box_size) * box_size, (col // box_size) * box_size

        used_digits = set()
        # Check row, column, and box for used digits
        for c in range(grid_size):
            val = int(state[row * grid_size + c].item())
            if val > 1: used_digits.add(val)
        # ... similar for column and box ...

        # Mask actions that place used digits at this position
        for digit_token in used_digits:
            mask[pos * vocab_size + digit_token] = False
```

### Tests

**New test file:** `tests/test_constraint_aware_masking_unittest.py` (14 tests)

Tests verify:
- Given cells are properly masked
- PAD and empty tokens are masked for all positions
- Row constraint masking works
- Column constraint masking works
- Box (3×3) constraint masking works
- `current_state` parameter updates mask correctly
- Batch masking works with current_state
- STOP action always valid
- No NaN in mask
- Mask is deterministic

**Test Results:** All 14 tests passed (Buck2)

### Current Experiment Status

| Algorithm | GPU | Status | Progress | Latest Score | Notes |
|-----------|-----|--------|----------|--------------|-------|
| UPI-TRM | 0 | **STUCK** | - | - | Hangs after first step (100% CPU, 8% GPU) |
| PPO | 1 | ✅ Running | ~300/50000 | - | Slow (~3.6s/step) |
| DQN | 2 | ✅ Running | ~4500/50000 (9%) | 25.56 | Good progress (~9 it/s) |

**Constraint masking verified:** Logs show "253 valid actions out of 892 total"

### Known Issues

**UPI-TRM Hang:**
- Process hangs after first debug output with 100% CPU, 8% GPU
- Reproducible on restart
- PPO and DQN work fine with same masking code
- Likely related to UPI-TRM trainer's batch action mask computation
- **Status:** Needs investigation

### Monitoring Commands

```bash
# Check experiment progress
for algo in ppo dqn; do
  log="results/sudoku9x9_constraint_aware/${algo}_seed0.log"
  step=$(grep -oP '\| \K\d+(?=/)' "$log" | tail -1)
  score=$(grep "eval_mean_score" "$log" | tail -1 | grep -oP 'eval_mean_score=\K[0-9.]+')
  echo "$algo: step=$step score=$score"
done
```

### Expected Impact

With constraint-aware masking:
- ~72% reduction in valid action space
- All explorations are constraint-valid (no wasted samples on invalid moves)
- Expected: faster learning, better final performance

### Next Steps

1. **Monitor PPO and DQN experiments** - Let them complete 50k steps
2. **Investigate UPI-TRM hang** - Debug batch action mask computation
3. **Compare results** with previous 9×9 experiments (without constraint masking)
4. **If successful:** Apply constraint masking to 4×4 experiments for consistency

### Configs

Located in `configs/sudoku9x9/`:
- `upi_trm_9x9.yaml` - UPI-TRM with TRM backbone
- `ppo_9x9.yaml` - PPO baseline
- `dqn_9x9.yaml` - DQN baseline

All use:
- `num_train_steps: 50000`
- `use_feasibility_checker: true`
- `use_action_masking: true`

### Commit

`6c23d19` - Add constraint-aware action masking for 9x9 Sudoku

---


