# Experiment Execution Plan: ICML Submission (Stability Dial)

**Goal:** Validate the “stability dial” narrative: contraction may reduce peak success rate, but makes the model **mathematically consistent** under deeper inner unrolling than it was trained with.

**Timeline:** Immediate (Jan 15 – Jan 20)

**Canonical scope:** 4×4 Sudoku with **feasibility checker** (`use_feasibility_checker: true`).

---

## Phase 1 (Critical): “Theory Saver” — Unroll Sensitivity

### Objective

Demonstrate that contraction acts like a stability dial by measuring how much the model’s **value** and **policy** predictions change when evaluated at deeper recursion depth than the training depth.

### Select checkpoints (two models)

- **Model A (Baseline):** best-performing unconstrained model (**No Contraction**).
  - Typical config family: `configs/ablations/upi_trm_feasibility_no_contraction.yaml`
- **Model B (Constrained):** “stable but poorer” model with:
  - **z→z contraction ON** (`enable_contraction: true`, `target_Lz < 1.0`)
  - **Value-head norm OFF** (`disable_value_head_norm: true`)

Notes:
- `enable_contraction` / `target_Lz` are used throughout `configs/`.
- `disable_value_head_norm` is supported by `rl/config.py` but is not currently present in the repo YAMLs; add a small ablation YAML for Model B before running Phase 1.

### Static evaluation batch

Create a static evaluation batch of **100** 4×4 Sudoku initial states (mix of trivial + harder):

- 50 from `data/sudoku-4x4-trivial`
- 50 from `data/sudoku-4x4-ultra-easy`

Save the exact puzzle indices (and seed) so the batch is identical across reruns.

### Define metric: Unroll Sensitivity

Let:
- `n_train` = the model’s training depth (`inner_unroll_n`)
- `n_limit` = deeper evaluation depth (recommend **16** or **32**)

Compute on the fixed batch:

- **Value consistency**
  - \(\Delta_V = \mathbb{E}[|U_{n_{train}}(s) - U_{n_{limit}}(s)|]\)
  - Also log p50/p90/p99 and max.

- **Policy consistency**
  - \(\Delta_\pi = \mathbb{E}[\mathrm{KL}(\pi_{n_{train}}(\cdot|s)\,\|\,\pi_{n_{limit}}(\cdot|s))]\)
  - Also log **argmax agreement** (% states where greedy action matches).

### Execution

1. Build the static batch (100 states).
2. For **each** model (A, B):
   - run inference on the batch at `n_train` and at `n_limit`
   - compute and log \(\Delta_V\), \(\Delta_\pi\), argmax agreement
3. Save results to `results/plot_data/unroll_sensitivity/*.csv` and a small summary table in Markdown.

### Expected outcome / narrative

- **Model A:** higher \(\Delta_V\) and \(\Delta_\pi\) (predictions drift or explode at deep unroll).
- **Model B:** lower \(\Delta_V\) and \(\Delta_\pi\) (predictions are consistent; contract towards a fixed point).
- **Narrative:** While Model A fits the training distribution better, Model B respects the fixed-point property under deeper unrolling, supporting the truncation-bias/stability argument.

---

## Phase 2 (High): “Sweet Spot” Sweep (Contraction Strength)

### Objective

Turn “contraction vs performance” from a binary into a tunable curve. Find a setting that improves stability without destroying success.

### Variables / conditions

Train on **Trivial 4×4 (1–4 empties)** with:
- `enable_contraction: true`
- `disable_value_head_norm: true`
- V-head norm OFF, so contraction dial is primarily `target_Lz`

Run 3 training runs (at minimum 1 seed each; ideally 3 seeds):

- **Run 1:** `target_Lz = 0.99` (very weak contraction)
- **Run 2:** `target_Lz = 0.95` (medium contraction)
- **Run 3:** `target_Lz = 0.90` (strong contraction)

Train for standard duration (**5k–10k steps**). Start with 5k for quick signal.

### Measurements

For each run:

- **Success rate** (existing evaluator metric)
- **Unroll sensitivity** (\(\Delta_V\), \(\Delta_\pi\)) using the Phase 1 batch + script

### Desired plot

- **X-axis:** `target_Lz` (0.90 → 0.99)
- **Left Y-axis:** Success Rate (expressivity cost)
- **Right Y-axis:** Unroll Sensitivity (stability gain)

---

## Phase 3 (Low / Contingency): Narrative Sanity Checks

### 1) Random baseline verification

Confirm the “~0% success” claim on a harder 4×4 suite with enough episodes for tight bounds:

- Run `scripts/eval_random_baseline.py` for **10k episodes** on `data/sudoku-4x4-ultra-easy`.
- Use multiple seeds if time permits and save CSV output.

### 2) Baseline tuning (time permitting)

If PPO/A2C are worse than random on checker-only tasks, increase exploration:

- Increase `entropy_coef` significantly (e.g., `0.05` or `0.1`) in `configs/baselines/*`.
- Goal: prevent early collapse to “do nothing” under sparse rewards.

---

## Summary checklist

- [ ] Identify best checkpoint for **Model A** (No Contraction).
- [ ] Train or identify checkpoint for **Model B** (Contraction ON, V-head norm OFF).
- [ ] Implement + run **Unroll Sensitivity** evaluation at `n_train` vs `n_limit`.
- [ ] Launch 3-run sweep over `target_Lz` on trivial 4×4.
- [ ] Verify random baseline with 10k episodes on harder 4×4.


