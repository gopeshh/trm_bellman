# CLAUDE.md

Operational guidance for Claude Code.
LaTeX paper repository:
`/home/buiksat/UPI_TRM/UPI_TRM_NIPS`

Main experiment plan for the NeurIPS 2026 resubmission:
`/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md`

---

## Non-negotiables

### 1. Checker: feasibility only
```yaml
use_feasibility_checker: true
```
Other checkers (`use_constraint_checker`, `use_progress_checker`) are debug-only. **Never use for paper results.**

### 2. Value-head normalization: OFF for stability experiments

| Flag | Value | Effect |
|------|-------|--------|
| `disable_value_head_norm: true` | **USE THIS** | Value-head spectral norm OFF → stable training |
| `disable_value_head_norm: false` | AVOID | Value-head spectral norm ON → causes collapse |

The 2×2 ablation proved value-head normalization causes target saturation. **Always set `disable_value_head_norm: true` for contraction/stability experiments.**

z→z contraction (`enable_contraction: true`, `target_Lz`) is separate and safe.

### 3. One variable at a time
Do not mix `episodic_latent` changes with contraction changes in the same ablation.

### 4. Measure projection saturation
Log pre/post `||z||` norms. If always clipped to `latent_ball_radius`, your "stability" may be clipping artifacts.

---

## Required metrics contract

Enable `track_theory_metrics: true`. Logged in `rl/upi_trm_trainer.py:_compute_theory_metrics()`:

| Metric | Status | Location |
|--------|--------|----------|
| `hat_Lz` | ✅ Implemented | `utils/lipschitz.py:estimate_local_Lz()` |
| `hat_Cz`, `hat_Lv`, `unrolling_term` | ✅ Implemented | `rl/upi_trm_trainer.py` |
| `Δ_V`, `Δ_π`, `Δ_z` (unroll sensitivity) | ❌ TODO | See `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md` |
| Saturation rate | ❌ TODO | See `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md` |

---

## Eval batch requirements

**Do not evaluate only initial states.** Per `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md`:
- **B0:** Fixed initial puzzles (trivial + hard mix)
- **B1:** One-step successor closure from B0

Both batches TODO; see the external NIPS plan.

---

## Experiment Sequence

Follow `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md` for the current multi-month experiment sequence.
Do not maintain a duplicate phase plan in this repo.

---

## Config template (stability experiments)

```yaml
use_feasibility_checker: true
enable_contraction: true
target_Lz: 0.9
disable_value_head_norm: true   # CRITICAL
episodic_latent: true           # or false, but consistent within ablation
latent_ball_radius: 10.0
```

---

## Commands

### Buck2 (devservers)
```bash
cd ~/fbsource/fbcode

buck2 test //buiksat_trm:test_... \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only

buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --config configs/ablations/upi_trm_feasibility_no_contraction.yaml --seed 42

buck2 clean
```

### Python (local)
```bash
python upi_trm_train.py --train-steps 100 --batch-size 16 --seed 0

pytest tests/ -v --tb=short
```

---

## Repo map

| What | Where |
|------|-------|
| RLConfig (all flags) | `rl/config.py` |
| Trainer + theory metrics | `rl/upi_trm_trainer.py` |
| Contraction utilities | `utils/lipschitz.py` |
| TRM model + enforcement | `models/recursive_reasoning/trm.py` |
| Plan-edit env | `rl/envs/plan_edit_env.py` |
| Configs | `configs/ablations/`, `configs/baselines/` |

---

## Change-management rules

1. **No silent behavior changes.** Update YAMLs and the external NIPS plan if defaults change in a way that affects planned experiments.
2. **Test new metrics.** At least one unit test (no NaNs, deterministic).
3. **Reproducible outputs.** Scripts must accept seed + explicit paths.
4. **Baselines are contingency.** Don't fix until Phases 1–3 done.

---

Project background and architecture: see `README.md`.
