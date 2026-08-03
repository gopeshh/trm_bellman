# ICLR repair handoff

Date: 2026-08-03

Branch: `iclr-confirmatory-repair`

This branch is an interrupted implementation checkpoint. It is not a completed
confirmatory experiment and must not be cited as one.

## Implemented in the working diff

- Clock-complete persistent replay with carried input/successor latents,
  remaining-edit clocks, transition and segment validation, and exact terminal
  masking.
- Persistent exact baseline enumeration using the common post-unroll successor
  carry.
- Exact K-step sampling that rejects incomplete nonterminal segments and uses
  the frozen current evaluator rather than the EMA bootstrap in exact mode.
- Live mid-episode collection checkpoints and exact UPI environment-interaction
  budgets. A collector cap no longer creates a false terminal or optimizer
  boundary.
- Schema-v3 fail-closed resume with full replay, live environment/collector
  state, process RNGs, dataset provenance, environment configuration, and
  action-mask configuration.
- Canonical dataset provenance with builder/version, recorded generation seed,
  ordered record hashes and digest, train/evaluation split identity, and nested
  mismatch reporting before mutable training state is restored.
- Held-out CleanRL evaluation, hard overlap/no-cycle checks, PPO truncation
  bootstrap/GAE repair, valid-action DQN exploration, fail-closed masks, and
  exact PPO interaction-budget divisibility.
- Local seeded 4x4, curriculum Sudoku, canonical Sudoku, and maze generation.
- Pure finite-batch augmented-state diagnostic summaries. These outputs are
  explicitly not uniform certificates.
- Regression tests for the paths above.

## Validation completed

- The pre-repair branch baseline ran all 30 declared Python unit-test targets:
  263 tests passed with zero failures. See `reports/PRE_REPAIR_TEST_REPORT.md`.
- `python3 -m py_compile` passes for every changed and newly added Python file.
- `git diff --check` passes.
- The provenance sub-slice also passed `fbpython -m py_compile` and scoped diff
  checks before integration.

## Validation interrupted

The combined focused Buck batch did not finish before the machine handoff.
The first attempt exposed a remote materialization failure for the empty
`dataset/__init__.py`; the file now contains a package docstring. The retry was
terminated after several minutes with no test result so the branch could be
committed before host loss. Its retained log is
`reports/FOCUSED_REPAIR_TEST_ATTEMPT.log`.

Run this first on the next machine:

```bash
cd /data/repos/fbsource/fbcode
buck2 test \
  //buiksat_trm:test_result_provenance \
  //buiksat_trm:test_upi_trm_logging_smoke \
  //buiksat_trm:test_upi_trm_trainer_smoke \
  //buiksat_trm:test_plan_edit_env \
  //buiksat_trm:test_rl_k_step_value_update_trainer \
  //buiksat_trm:test_optimized_exact_baseline \
  //buiksat_trm:test_augmented_replay_diagnostics \
  //buiksat_trm:test_cleanrl_regressions \
  //buiksat_trm:test_dataset_builders \
  //buiksat_trm:test_rl_algos_mock \
  //buiksat_trm:test_config_integrity
```

Then rerun all declared Python unit-test targets and the six package type-check
targets recorded in `reports/PRE_REPAIR_TEST_REPORT.md`.

## Open correctness review

An adversarial review cleared exact probability-space deployment, stochastic
mixture evaluation, persistent carry order, exact centering, and shared
old/candidate recurrent snapshots. It raised one unresolved training-protocol
issue: in a multi-update `theory_exact_mixture` run, replay is collected from
the explicit old/candidate mixture while advantages are centered under the old
component, and the value step can change policy-affecting recurrent parameters
before the next frozen snapshot. The fixed-snapshot theorem remains a
conditional post-update statement, but `is_theory_exact()` overstates the
end-to-end training path. Do not call a multi-step run theorem aligned until
one of these designs is implemented and tested:

1. freeze the recurrent actor map and train only a policy-independent value
   head for one fixed-base proposal, collecting under that same base policy; or
2. represent and promote the exact mixture as the next current policy, then
   evaluate and center under that same policy.

`mixture_alpha` is now constrained to `[0, 1]`.

## Experiments not run

No repaired learned-model result was executed. Persistent checkpoint
diagnostics, the one-factor bridge, matched UPI-TRM/PPO runs, the projection
cross-design, and the second domain remain missing experiments. The registered
Sudoku matrix needs at least 245 GPU-hours on the available single-GPU setup.
Do not infer an outcome from smoke tests or from the historical 57.4% record.
