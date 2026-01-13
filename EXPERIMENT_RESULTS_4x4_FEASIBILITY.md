# UPI-TRM 4×4 Sudoku Experiment Log (Feasibility Checker)

**Last updated:** 2026-01-12

This is a concise, living summary of the 4×4 Sudoku feasibility-checker experiments used to guide paper figures and next experiments. It supersedes older long-form notes.

---

## Setup (common)

- **Environment:** plan-editing MDP with a deterministic feasibility checker.
- **Success:** episode ends with a valid Sudoku grid (0 violations) within the horizon.
- **Training budgets:**
  - **Trivial suite (1–4 empties):** 5k gradient steps.
  - **Harder suite (6–8 empties):** 20k gradient steps.

---

## Datasets

- **`sudoku-4x4-trivial`**: 1–4 empties (mean ≈ 2.46). Used for the paper’s “trivial” plots.
- **`sudoku-4x4-ultra-easy`**: actually 6–8 empties (mean ≈ 7.04). Used for the “6–8 empties” suite.

> Note: earlier 0% results were traced to mistakenly using the 6–8-empties dataset when we believed we were running 1–4 empties.

---

## Random baselines (masked actions only)

Random policy samples **uniformly among valid/masked actions**.

- **Trivial (1–4 empties):** ~**52%** success.
  - Interpretation: success is a coarse metric on the trivial suite; use score/violations for finer granularity.
- **6–8 empties:** **0%** success (mean over 3 seeds × 100 episodes each = 300 total episodes).
  - See: `results/plot_data_6to8empties_all_ablations_20k/random_baseline_6to8empties.csv`
  - Random typically fills all cells but ends with ~**9.63** violations on average.

---

## Results: Trivial 1–4 empties (5k steps)

**Headline:** removing contraction is critical for performance; the strongest variant is persistent-z without contraction.

- **Best configuration:** **UPI-TRM (Persistent-z, no contraction)** = **93.3% ± 2.3%** final success (3 seeds).
- **Baselines (PPO/A2C/DQN):** non-zero learning but far below UPI-TRM under the same budget.
- **Random is high (~52%)**, so we treat success as coarse and also look at score/violations/dead-end proxies.

Primary plot artifact used for paper-style comparisons:
- `results/plots_contraction_fix_rerun_ALL/feasibility_success_vs_steps.{png,pdf}`

---

## Results: 6–8 empties (20k steps, 7 configs × 3 seeds)

**Headline:** contraction-enforced variants fail completely (0%), while no-contraction variants learn substantially.

Final success rates at step 20k (from `results/plot_data_6to8empties_all_ablations_20k/summary.csv`):

| Config | Seed 42 | Seed 123 | Seed 456 | Mean |
|---|---:|---:|---:|---:|
| **UPI-TRM (Persistent-z, no contraction)** (`persistent_z_no_contraction`) | 0.60 | 0.52 | 0.58 | **0.567** |
| **UPI-TRM (Episodic-z, no contraction)** (`no_contraction`) | 0.42 | 0.54 | 0.48 | **0.480** |
| UPI-TRM (no conservative) (`ablation_no_conservative`) | 0.02 | 0.34 | 0.00 | **0.120** |
| UPI-TRM (Episodic-z, contraction) (`upi_trm`) | 0.00 | 0.00 | 0.00 | **0.000** |
| UPI-TRM (Persistent-z, contraction) (`ablation_persistent_z`) | 0.00 | 0.00 | 0.00 | **0.000** |
| A2C (`a2c`) | 0.00 | 0.00 | 0.00 | **0.000** |
| DQN (`dqn`) | 0.00 | 0.00 | 0.00 | **0.000** |

**Key findings**
- **Contraction kills learning** on 6–8 empties: both episodic-z and persistent-z contraction variants are **0%**.
- **Persistent-z + no contraction is best** (mean 56.7%), suggesting that **carrying latent state helps** when the latent dynamics are not over-regularized.
- **Baselines (A2C/DQN) collapse to 0%**, and random is also 0% (confirming difficulty vs the trivial suite).

Paper-style plot (legend naming matches the trivial plot conventions, uses `S=` seed count, and uses Random (0%)):
- `results/plots_6to8empties_all_ablations_20k/feasibility_success_vs_steps.{png,pdf}`
- Generator script: `scripts/plot_6to8empties_paper_style.py`

---

## Mechanism hypotheses (why contraction blocks learning here)

Working hypothesis for the checker-only edit setting (especially 6–8 empties):

- **Over-regularized latent dynamics:** enforcing a strict contraction (**`L_z < 1`**) can collapse the evaluator into a near-smoothing map that cannot represent the high-curvature, discrete constraint-propagation needed for multi-step satisfaction.
- **Weak/flat credit assignment:** with checker-only feedback and many coordinated edits required, contraction can make value/advantage estimates too homogeneous (small gradients), pushing policies toward conservative/no-op behavior.
- **Interaction with multi-step difficulty:** 6–8 empties requires ~6–8 correct placements; a single wrong move can poison downstream constraints, so expressivity and exploration matter more than stability guarantees.

These are testable: track **`\\hat L_z`**, advantage variance/entropy collapse, and “safe/no-op” edit frequency under contraction vs no-contraction.

---

## Baseline improvement plan (PPO/A2C/DQN)

We are actively trying to improve PPO/A2C/DQN so they are not as bad as random (or do not collapse to 0% in the harder suites). Concrete levers:

- **Exploration:** entropy schedules, action-noise/temperature, and longer warmup.
- **Curriculum:** train on 1–4 empties then continue on 6–8 empties.
- **Credit assignment:** reward normalization and shaping checks; longer horizons / bootstrapping stability; eval protocol parity.
- **Operators:** if the environment supports it, include repair actions (clear/undo) and verify masking is consistent.

---

## Next step: 9×9 Sudoku

Next experiments will scale the same evaluation protocol to **9×9 Sudoku**, re-testing:

- contraction vs no-contraction trade-offs,
- episodic-z vs persistent-z,
- whether baseline tuning closes the gap under matched budgets.

---

## Artifact index

- **6–8 empties (20k) data:** `results/plot_data_6to8empties_all_ablations_20k/`
- **6–8 empties (20k) plots:** `results/plots_6to8empties_all_ablations_20k/`
- **6–8 empties random baseline:** `results/plot_data_6to8empties_all_ablations_20k/random_baseline_6to8empties.csv`
- **Trivial suite paper-style plots:** `results/plots_contraction_fix_rerun_ALL/`
