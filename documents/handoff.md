# Handoff

Date: 2026-04-26
Repo: `/home/buiksat/trm_bellman`
Branch: `feature/upi-trm-clean`

## Current State

- Canonical paper source tree: `/home/buiksat/UPI_TRM/UPI_TRM_NIPS`
- Canonical experiment plan mirror: `/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS/NIPS_PLAN.md`
- Main capability anchor is still no-mask hard `4x4`; the archived paper table remains:
  - UPI-TRM `0.574 ± 0.122`
  - TRM+PPO `0.320 ± 0.153`
  - TRM+A2C `0.000`
  - TRM+DQN `0.000`

## Baseline Audit / Code Changes

- PPO baseline optimizer wiring is fixed:
  - `edit_policy.*` at `policy_lr`
  - `value_head.*` at `value_lr`
  - shared backbone params at `backbone_lr` if set, else `policy_lr`
- DQN target-network updates now use env-step units instead of outer trainer steps.
- n-step DQN is implemented via `dqn_n_step` with `1` preserving vanilla behavior.
- New overlay config: `configs/baselines/dqn_trm_feasibility_nstep5.yaml`
- Provenance rerun scripts now equalize baseline env interactions at `320000` and add:
  - interaction-based learning-curve plotting
  - legacy outer-step plotting for backward compatibility
- `BUCK` target `upi_trm_train` now sets `keep_gpu_sections = True` so Buck preserves GPU sections for the training binary.

## Validation

- `buck2 test //buiksat_trm:test_rl_algos_mock --local-only` passed
- `buck2 test //buiksat_trm:test_baselines --local-only` passed
- New unit coverage includes:
  - PPO 3-param-group split
  - explicit `backbone_lr`
  - DQN target update in env-step units
  - n-step DQN `n=1` equivalence
  - analytic `n=3` target check
  - early-terminal truncation
  - replay-wrap regression test

## Hard 4x4 Equalized-Budget Rerun

- Result root: `/home/buiksat/trm_bellman/results/hard4x4_trm_baselines_nomask_envbudget_10seed_20260424`
- This rerun is still in progress and is intentionally not committed.
- Status snapshot at `2026-04-26 10:15 PDT`:
  - `34/40` jobs complete
  - `4` jobs running
  - `2` jobs queued
  - `0` failures
- Completed-seed snapshot:
  - TRM+PPO: `9/10`, `0.000 ± 0.000`, mean return `-17.177`
  - TRM+A2C: `8/10`, `0.000 ± 0.000`, mean return `-17.764`
  - TRM+DQN: `9/10`, `0.000 ± 0.000`, mean return `-19.178`
  - TRM+DQN `(n=5)`: `8/10`, `0.010 ± 0.011`, mean return `-18.778`
- Important runtime caveat:
  - this host exposes only `2` real GPUs
  - two workers are on GPU and two are effectively CPU-bound
  - the sweep is healthy but slower than intended
- Do not treat the partial hard-suite numbers as paper-ready until all `10` seeds finish.

## Next Priorities

1. Let the equalized hard-suite rerun finish, then aggregate final `10`-seed numbers side-by-side against the archived table.
2. Run the masked toy-suite confirmation for PPO, A2C, DQN, and DQN `(n=5)`.
3. Decide whether the new baseline numbers materially change the paper story before touching manuscript text.
