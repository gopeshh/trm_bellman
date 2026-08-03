# Persistent checkpoint diagnostics implementation report

Date: 2026-08-03

Branch: `iclr-confirmatory-repair`

Parent commit: `e61b1731b4a4ec73860dfba51017653f9193d3b7`

## Scope completed

The repository now has a registered, CPU-executable diagnostic path for one
schema-v4 `fixed_base_exact` persistent checkpoint. The runner rejects
episodic checkpoints, incomplete policy pairs, mismatched recurrent maps,
unshaped-reward checkpoints, UNDO-enabled environments, incomplete clocks,
ambiguous dataset manifests, and implicit held-out cycling.

The retained state is the full augmented occurrence `(x, y, z, h)`. State
collection preserves the carried input latent before the recurrent unroll,
the post-unroll successor carry, remaining edit budget, environment snapshot,
action mask, terminal flag, and source occurrence identity. Identical plans
with different latents or clocks remain different hashed states.

The runner computes and retains raw rows for:

- exact one-step `U_n - T U_n` and `U_m - T U_m` residuals using
  `PlanEditEnv.step` as the transition oracle;
- shared-path Monte Carlo K-step operator estimates, standard errors, and
  residuals at `n` and every registered finite-reference depth `m`;
- exact action summation and production-baseline parity;
- reconstructed statewise centering after the configured exact recentering
  operation;
- finite-depth candidate advantage discrepancy, which is not the theorem's
  uniform candidate-policy bias;
- the exact probability mixture versus the production `_mixed_policy_dist`
  callback on every retained state, including KL, total variation, and support
  mismatch;
- recurrent path length, per-depth increments, local ratios, value-head drift,
  initial-latent sensitivity, carried-latent continuity, clock strata,
  projection activation, and latent norms;
- current/candidate/evaluator recurrent-map identities before and after the
  diagnostic run.

Every aggregate is labeled `finite_batch`. The artifact does not call a local
Lipschitz proxy a global contraction certificate.

## Input and artifact provenance

Training provenance now binds the complete materialized held-out pool, not
only `eval_num_episodes`. It records ordered puzzle-identifier hashes for both
splits. The loader reconstructs the exact ordered pools, applies the saved
identifier offset, rejects overlap and duplicates, and compares the environment
and action-mask configurations before evaluation.

The deterministic artifact bundle contains strict JSON/JSONL, `MANIFEST.json`,
and `SHA256SUMS`. Publication uses a staged directory and atomic
`RENAME_NOREPLACE`. The bundle rejects duplicate JSON keys, nonfinite JSON
numbers, identity-bearing paths, an existing destination, and an unsupported
execution device.

Because a verified clean producer commit is unavailable, the artifact keeps
the commit status as `not verifiable from supplied evidence`. It separately
hashes the loaded diagnostic entry points and an over-approximation of every
repository-local behavior source under `dataset`, `evaluators`, `models`,
`rl`, `scripts`, and `utils`, plus `BUCK`, `puzzle_dataset.py`, and
`upi_trm_train.py`.

## Registered protocol

`configs/iclr_confirmatory/persistent_diagnostics.json` fixes:

- first 128 held-out records;
- finite-reference offsets 1, 2, and 4 beyond `n`;
- 256 Monte Carlo repeats per state;
- Monte Carlo seed 26080321;
- at most 256 clock-balanced retained states;
- chunk size 16;
- initial-latent perturbation L2 norm 0.01 with seed 26080321.

No outcome was inspected to choose these values.

## Validation

Authoritative final-state gates:

- focused diagnostic runtime plus CLI typecheck: 16 passed, 0 failed;
- 31 non-logging runtime targets: 311 passed, 0 failed;
- isolated logging target from a fresh Buck daemon: 14 passed, 0 failed;
- total runtime cases: 325 passed, 0 failed;
- packaged CLI build: passed;
- `git diff --check`: passed;
- Python bytecode compilation for every touched Python file: passed.

The package-wide `rl-type-checking` target still fails on nine pre-existing
errors in `rl/algos/dqn.py`, `rl/cleanrl/trm_adapter.py`, and
`rl/envs/plan_edit_env.py`. The final typecheck has no error in
`rl/persistent_diagnostics.py`. The failed baseline-debt output is retained in
`reports/PERSISTENT_DIAGNOSTICS_RL_TYPECHECK_BASELINE_DEBT.log`.

The earlier logs are deliberately retained:

- `PERSISTENT_DIAGNOSTICS_INITIAL_TEST.log`: invalidated by a concurrent Buck
  daemon restart;
- `PERSISTENT_DIAGNOSTICS_TEST_RERUN.log`: failed because the first Buck target
  omitted the `puzzle_dataset` dependency;
- `PERSISTENT_DIAGNOSTICS_TEST_AFTER_DEPS.log`: obsolete five-test pass before
  the suite expanded to 15 runtime cases;
- `PERSISTENT_DIAGNOSTICS_TYPECHECKS.log`: superseded typecheck that exposed
  three new narrowing errors, all fixed before the final gates.

These logs are development chronology, not successful validation evidence and
must not enter an anonymous supplement without separate path/identity review.

## Evidence boundary

There is no learned checkpoint in this repository and no registered hard
held-out dataset artifact has been materialized. Therefore:

- historical persistent augmented-state residuals: not verifiable from supplied evidence
- historical centering diagnostics: not verifiable from supplied evidence
- historical deployment-gap diagnostics: not verifiable from supplied evidence
- the bridge experiment: not verifiable from supplied evidence
- the interaction-matched PPO comparison: not verifiable from supplied evidence

The runner reconstructs centering from retained states; it does not recover the
saved training-time estimator. Replay validation is structural and does not
semantically re-execute historical replay transitions. Schema 4 does not bind
the producer commit, training seed, run ID, or complete live resume state.
Those remain blockers for confirmatory training until schema 5 or an equivalent
external immutable run manifest is implemented.

No diagnostic value, learned result, checkpoint property, or experiment
outcome was fabricated or inferred from an unexecuted run.
