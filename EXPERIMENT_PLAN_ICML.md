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

