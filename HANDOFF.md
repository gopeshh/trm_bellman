# HANDOFF.md - UPI-TRM Project

**Date:** 2026-04-05
**Branch:** `feature/upi-trm-clean`
**Primary sources of truth:** `CLAUDE.md`, `EXPERIMENT_PLAN_ICML.md`, `README.md`

---

## Current State

This branch focuses on repository cleanup and behavior-preserving refactors around the UPI-TRM training stack. Historical experiment outputs remain under `results/` and the various `*_export/` directories, but the current operational guidance is:

- Follow `EXPERIMENT_PLAN_ICML.md` for paper-facing experiment sequencing.
- Treat `CLAUDE.md` as the guardrail document for stability experiments.
- Treat `README.md` as the current repo layout and entrypoint guide.

## Refactor Notes

- Pretrain Hydra configs live under `configs/pretrain/`.
- The repo-root `config` symlink remains for compatibility with older pretrain workflows.
- Canonical script locations are now:
  - `scripts/eval/unroll_sensitivity.py`
  - `scripts/provenance/create_provenance_bundle.sh`
- Legacy paths remain as thin wrappers:
  - `scripts/eval_unroll_sensitivity.py`
  - `scripts/create_provenance_bundle.sh`
- `rl/training_setup.py` now owns dummy/supervised dataset bootstrap and checker resolution for `upi_trm_train.py`.

## Experiment Guardrails

- Use `use_feasibility_checker: true` for paper results.
- Use `disable_value_head_norm: true` for contraction and stability experiments.
- Keep `track_theory_metrics: true` enabled when evaluating theory-facing claims.
- Change one variable at a time across ablations; do not mix episodic-latent changes with contraction sweeps.
- Measure projection saturation (`||z||` pre/post projection) so stability claims do not collapse into clipping artifacts.

## Common Commands

```bash
# Local smoke checks
python upi_trm_train.py --train-steps 100 --batch-size 16 --seed 0
pytest tests/ -v --tb=short

# Config integrity
python scripts/verify_configs.py

# Buck2 (Meta devservers)
cd ~/fbsource/fbcode
buck2 test //buiksat_trm:test_... \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only

buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
  -- --config configs/ablations/upi_trm_feasibility_no_contraction.yaml --seed 42
```

## Directories to Know

- `configs/` – experiment YAMLs grouped by workflow.
- `entrypoints/` – canonical Python entrypoints for moved training scripts.
- `rl/` – training logic, environment, checkers, and bootstrap helpers.
- `scripts/` – evaluation, plotting, provenance, and audit tooling.
- `results/` – historical experiment outputs and plot data.
- `artifacts/` – evaluation batches and generated auxiliary artifacts.

## Audit Expectations

When changing code that affects experiments:

- Run `python3 -m py_compile` on touched Python modules.
- Run `bash -n` on touched shell scripts.
- Run `python scripts/verify_configs.py` after config/layout changes.
- Run pytest or Buck2 tests when the environment has the required Python dependencies available.

If local test dependencies are missing, record that explicitly in the handoff or review summary.
