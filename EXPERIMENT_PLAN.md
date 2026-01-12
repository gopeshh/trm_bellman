# UPI-TRM Experiment Plan (Living)

**Last updated:** 2026-01-12

This file is the forward-looking plan only. For current/validated results and artifact pointers, see:
- `EXPERIMENT_RESULTS_4x4_FEASIBILITY.md`

---

## Current priorities

### 1) Make baselines competitive (PPO/A2C/DQN)

On checker-only plan-editing, PPO/A2C/DQN can underperform badly (and can even collapse to ~0% on harder suites). The goal is to get baselines reliably above random under matched budgets.

Concrete levers to test (in priority order):
- **Exploration:** stronger entropy bonuses / schedules, action-noise/temperature, longer warmup.
- **Curriculum:** 1–4 empties → 6–8 empties.
- **Credit assignment:** reward normalization, shaping audits, horizon/bootstrapping stability.
- **Action space / operators:** confirm masking parity; add repair actions (clear/undo) if supported.

Deliverables:
- tuned baseline configs (documented hyperparams)
- updated plots where tuned baselines replace old baselines under the same protocol

### 2) Re-test contraction as a dial (not a universal win)

Empirically, contraction enforcement can block learning on 6–8 empties. Hypothesis: it over-regularizes the evaluator and flattens advantages/credit assignment in discrete multi-step satisfaction.

Tests:
- sweep contraction strength and unroll depth
- log \(\hat L_z\), advantage variance/entropy collapse, and edit/no-op rates

### 3) Scale to 9×9 Sudoku

Run the same suite and ablations on 9×9 under matched budgets:
- episodic-z vs persistent-z
- contraction vs no contraction
- tuned baselines vs UPI-TRM variants

---

## Artifact pointers

- 6–8 empties (20k): `results/plot_data_6to8empties_all_ablations_20k/` and `results/plots_6to8empties_all_ablations_20k/`
- Trivial suite paper-style plots: `results/plots_contraction_fix_rerun_ALL/`
