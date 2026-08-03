# ICLR repair handoff

Date: 2026-08-03

Branch: `iclr-confirmatory-repair`

This branch contains the repaired implementation baseline. It is not a
completed confirmatory experiment and must not be cited as one.

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
- Schema-v4 fail-closed resume with training-protocol and execution-device
  identity, full replay, live environment/collector
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
- A deterministic persistent-checkpoint diagnostic CLI with clock-complete
  state collection, exact one-step and Monte Carlo K-step residuals at `n` and
  registered finite-reference depths, production exact-baseline parity,
  reconstructed centering, production-mixture callback comparison, depth and
  carry diagnostics, immutable held-out materialization, and source manifests.
- Explicit `fixed_base_exact` training: collection from one frozen base,
  value-head-only critic fitting, candidate-head proposal training, exact
  probability-space mixture evaluation, and no recursive promotion.
- Regression tests for the paths above.

## Validation completed

- The pre-repair branch baseline ran all 30 declared Python unit-test targets:
  263 tests passed with zero failures. See `reports/PRE_REPAIR_TEST_REPORT.md`.
- `python3 -m py_compile` passes for every changed and newly added Python file.
- `git diff --check` passes.
- The provenance sub-slice also passed `fbpython -m py_compile` and scoped diff
  checks before integration.

## Post-repair validation

The final focused Buck batch passed 121 tests. The complete post-handoff runtime
gate was partitioned to avoid a reproducible TPX parallel-teardown artifact in
the logging target: 30 targets passed 296 tests together, and the isolated
logging/checkpoint target passed 14 tests. Total: 310 passing cases, zero code
failures. The six known package type-check targets also passed.

Failed and interrupted attempts are retained rather than counted as passes.
The exact chronology, commands, session identifiers, and SHA-256 log manifest
are in `reports/POST_HANDOFF_EXECUTION_REPORT.md` and
`reports/POST_HANDOFF_TEST_LOG_SHA256SUMS.txt`.

The subsequent persistent-diagnostics slice expanded the runtime suite to 32
targets. Its final exact-state gate passed 311 cases across the 31 non-logging
targets and 14 cases in the isolated logging target: 325 total, zero failures.
The focused diagnostic target plus packaged CLI typecheck passed 16 cases.
The full `rl-type-checking` target still reports nine unchanged baseline errors
outside the new files; it reports no error in the persistent diagnostic code.
Evidence and chronology are in
`reports/PERSISTENT_CHECKPOINT_DIAGNOSTICS_REPORT.md` and
`reports/PERSISTENT_DIAGNOSTICS_LOG_SHA256SUMS.txt`.

## Correctness review outcome

The unresolved training-protocol issue was closed by implementing the first
design above as an explicit opt-in. `theory_exact_mixture=True` alone no longer
labels legacy multi-update training as exact. `fixed_base_exact` freezes the
base policy and recurrent map, collects from that base, trains only the value
head and one candidate policy head, and evaluates the explicit mixture without
promoting it. Schema-v4 resume fails closed on protocol or device mismatch.

Two adversarial re-reviews cleared collection identity, recurrent-map
immutability, exact-mixture evaluation, optimizer ownership, pre-mutation
checkpoint validation, and device restoration. They found a stale
`model_step_*.pt` hazard; fixed-base saves now reject any output directory that
contains such a single-model artifact, with a regression test.

## Experiments not run

No repaired learned-model result was executed. The persistent checkpoint
diagnostic pipeline is implemented, but no learned checkpoint exists, so its
historical outputs are not verifiable from supplied evidence. The one-factor
bridge, matched UPI-TRM/PPO runs, projection cross-design, and second domain
remain missing experiments. The registered Sudoku matrix needs at least 245
GPU-hours on the available single-GPU setup. Do not infer an outcome from
smoke tests or from the historical 57.4% record.
