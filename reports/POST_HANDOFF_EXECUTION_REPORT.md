# Post-handoff execution report

Date: 2026-08-03, America/Los_Angeles

## Repository baseline

- Repository: `/home/buiksat/trm_bellman`
- Branch: `iclr-confirmatory-repair`
- Starting commit: `111e3a96655d1b3414ba15801a3ebfa393e8e394`
- Buck cell: `/data/repos/fbsource/fbcode/buiksat_trm`
- The Buck cell is a symlink to the repository. Remote execution cannot
  materialize newly added files through that external symlink, so authoritative
  post-repair runs use `--local-only`.

## Correctness repair

The former `theory_exact_mixture` switch selected a probability-space mixture
but did not define an end-to-end fixed-base training protocol. The repaired
implementation adds the explicit opt-in protocol `fixed_base_exact`:

- collect replay from the frozen base policy;
- freeze the base edit policy and recurrent transition map;
- train only the policy-independent value head in the critic;
- train the candidate edit head as one proposal;
- deploy and evaluate the exact probability-space base/candidate mixture;
- do not recursively promote the mixture or synchronize candidate parameters
  into the base;
- reject scheduled operator-norm clamping and trainable sparse puzzle
  embeddings because both mutate policy-relevant fixed state;
- retain `legacy` as the default so old runs are not reclassified.

Schema-v4 checkpoints record the protocol and execution device. Resume checks
the top-level, nested, and active protocol; dataset provenance; device; model
and optimizer identity; replay; environment; and collector state before any
live state is mutated. Schema-v3 checkpoints remain legacy-only. Fixed-base
saves reject output directories containing any `model_step_*.pt` artifact,
because one state dict cannot represent the deployed mixture.

Regression coverage includes frozen optimizer ownership, base-only collection,
behavior log-probability identity, persistent latent initialization, two-update
base immutability, exact-mixture evaluation, schema-v4 resume equivalence,
schema-v3 rejection, nested-state preflight, device mismatch, and stale
single-model artifacts.

## Validation chronology

1. The first focused remote run passed 71 tests and had two build failures.
   Both failures were missing files during remote materialization of the
   external Buck-cell symlink, not test failures. Evidence:
   `reports/POST_HANDOFF_FOCUSED_RERUN.log`.
2. An initial local command from the repository root failed because that path
   is not a Buck project. The failed attempt is retained in
   `reports/POST_HANDOFF_AUGMENTED_LOCAL_ATTEMPT.log`.
3. The corrected local augmented-state target passed 10 tests. Eleven focused
   targets then passed 121 tests. Evidence:
   `reports/POST_HANDOFF_AUGMENTED_LOCAL_RERUN.log` and
   `reports/POST_HANDOFF_FOCUSED_LOCAL_RERUN.log`.
4. Before the fixed-base edits, all 31 declared Python unittest targets passed
   301 tests, and the six known type-check targets passed. Evidence:
   `reports/POST_REPAIR_FULL_UNITTESTS.log` and
   `reports/POST_REPAIR_SIX_TYPECHECKS.log`.
5. After the edits, one uncapped all-target run passed 308 tests. One logging
   case printed `Ran 1 test ... OK`, then TPX reported that the process never
   completed during parallel teardown. This run is retained as failed and is
   not counted as a clean suite result:
   `reports/POST_FIXED_BASE_FULL_UNITTESTS.log`.
6. A four-worker retry produced no progress event for more than 20 minutes and
   was deliberately interrupted with exit code 130. It is retained as an
   incomplete attempt in `reports/POST_FIXED_BASE_FULL_UNITTESTS_RERUN.log`.
7. The authoritative partitioned runtime gate is green. The 30 non-logging
   targets passed 296 tests in one invocation. The logging/checkpoint target
   passed all 14 tests alone from a fresh daemon. Total: 310 passing cases,
   zero test failures, zero timeouts, zero build failures. Evidence:
   `reports/POST_FIXED_BASE_UNITTESTS_EXCLUDING_LOGGING.log` and
   `reports/POST_FIXED_BASE_LOGGING_FINAL.log`.
8. The six known package type-check targets passed. Evidence:
   `reports/POST_FIXED_BASE_SIX_TYPECHECKS.log`.
9. `git diff --check` passed.

All retained log hashes are in
`reports/POST_HANDOFF_TEST_LOG_SHA256SUMS.txt`.

## Evidence state and next dependency

No `.pt`, `.pth`, or `.ckpt` learned-model checkpoint is present. The existing
episodic diagnostic script resets the latent and omits remaining budget, so it
cannot produce persistent augmented-state evidence. Historical persistent
checkpoint diagnostics are not verifiable from supplied evidence.

`configs/iclr_confirmatory/` and the registered 1,024/256/512 unique hard
train/validation/test split do not exist. The only local corpus is a 450/50
trivial dataset with incomplete builder provenance. Therefore the bridge and
interaction-matched PPO results are not verifiable from supplied evidence.

No confirmatory training was launched in this slice. The next dependency chain
is:

1. extend schema 4 with immutable producer/run/seed identity or bind an
   equivalent external run manifest before confirmatory training;
2. materialize unique, pairwise-disjoint hard splits and immutable manifests;
3. add the corrected persistent reference, bridge, and matched-PPO configs;
4. lock the registry and hashes before inspecting confirmatory outcomes;
5. run debug-only smoke tests on seed 9001;
6. produce a valid persistent checkpoint and run the registered diagnostics;
7. run the one-factor bridge and matched PPO
   comparison in that order.

## Persistent diagnostics addendum

The schema-v4 persistent diagnostic runner is now implemented and tested. It
uses `(x, y, z, h)` occurrences, the production transition and mixture-policy
callbacks, exact one-step and Monte Carlo K-step backups, registered
finite-reference depths, and deterministic artifact/source manifests. The
final runtime gate is 325 passing cases across 32 targets. Full details and log
hashes are in `reports/PERSISTENT_CHECKPOINT_DIAGNOSTICS_REPORT.md` and
`reports/PERSISTENT_DIAGNOSTICS_LOG_SHA256SUMS.txt`.

No learned checkpoint or hard held-out corpus was found. Consequently,
historical persistent diagnostic values, the bridge result, and the matched
PPO result remain not verifiable from supplied evidence.

## Confirmatory dataset addendum

After schema-v5 source identity was committed, a dedicated builder generated
the registered 1,024/256/512 hard Sudoku train/validation/test corpus. It
enforces 6 to 8 empty cells, one valid completion, unique input and record
hashes, pairwise split disjointness, shared provenance hash functions,
runtime-to-producer source equality, and atomic no-replace publication. A full
detached-checkout rebuild was byte-identical. See
`reports/CONFIRMATORY_DATASET_REPORT.md`.

The earlier statement that the hard held-out corpus was absent is retained
above as chronology. The corpus now exists, but no learned checkpoint or
confirmatory result exists. The next blockers are exact-budget PPO,
per-instance evaluation artifacts, complete cell configs, and a locked
registry.
