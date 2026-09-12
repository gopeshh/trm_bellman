# Experiment 1B final sealed-opener review

Date: 2026-09-11

## Snapshot and verdict

- Standalone: `/home/buiksat/trm_bellman`, branch `full-implementation`, HEAD `0f8d0297c13831c4b6cf1921a5b87c91a5726507`
- fbsource package: `/data/repos/fbsource/fbcode/buiksat_trm`
- Verified handoff SHA-256: `05089ff15d30e7da442914c2e157c2f3bfc76ea84fbebb311977970bdf7e0a7a`
- Verified prior report SHA-256: `8bae85b404c6d6689730101f4af7d881093f90d42a44f4c5d5638763f93e48f3`
- Changed test SHA-256 in both trees: `99bab34a314f3e8d4532ecb8719d3894f1fe4eb687b7f06fd9ba704e552d1450`
- Production backend SHA-256 in both trees: `64acd4fabf8a6fa08e760be6ebf782238a887769312cc596dc35d8ffb4d9660b`

**PASS.** The prior Medium finding is closed. `Exp1bTorchSealedOpenerTest` makes the real production opener load-bearing and executes production path authentication, payload validation, census and RL gates, four-module restoration/digest checks, and freezing. No launch-invalidating finding survived.

One low, non-blocking test-matrix omission is recorded below because the review request explicitly asks for any asserted invariant that is not independently mutation-sensitive. Current production behavior is correct, and this omission does not block the experiment run.

## Surviving low finding

### Low: the payload-side model-state identity is not independently mutated

Production checks both halves of the folded identity at `policy_improvement_full_backend.py:6442-6444`:

1. the recomputed fold must equal `payload["model_state_sha256"]`;
2. the recomputed fold must equal the externally supplied `expected_model_state_sha256`.

The test matrix mutates only the external argument at `tests/test_policy_improvement_exp1b_unittest.py:5400-5404`. It does not mutate the payload's own `model_state_sha256` while leaving the external expected value genuine.

Reproduction: replace only the payload-half comparison in memory so `_exp1b_checkpoint_field(payload, "model_state_sha256")` returns the recomputed fold. All four new opener tests remain green. The existing five-region identity mutation removes both comparisons, so the external-argument case kills that coarse mutation without proving each half independently.

Failure mode: a future regression deleting only the payload self-consistency half would not be caught by this class. The external descriptor would still bind the actual module fold, so this is not a current launch blocker.

Concrete fix: add one case to `test_each_payload_invariant_is_enforced_by_the_production_opener` that changes `payload["model_state_sha256"]` to another valid digest while leaving `expected_model_state_sha256` unchanged, and require the production folded-identity refusal.

## Falsification result

The required falsification reproduced exactly inside the final test PAR:

```text
Exp1bTorchSealedOpenerTest
  run=4 failures=0 errors=29 opener_calls=29  -> RED

Exp1bTorchFourModuleRoundTripTest
  run=4 failures=0 errors=0 opener_calls=0   -> GREEN
```

With `open_exp1b_sealed_evaluation_session` replaced by a raising counter, every production-opener call in the new class errors. The older serializer class remains green, as expected. The new class cannot pass without the production opener.

## Disposition of the prior Medium finding

**Closed.** The new class performs the following sequence through the real code:

1. seals four distinct modules through `seal_exp1b_training_checkpoint`;
2. writes the serialized bytes to an external temporary checkpoint;
3. supplies exact path, SHA-256, size, model-state, config, budget, run, seed, and census identities;
4. calls `open_exp1b_sealed_evaluation_session` at `tests/test_policy_improvement_exp1b_unittest.py:5290-5295`;
5. reaches the real module load/digest/freeze loop before the injected final session recorder.

The positive test verifies tensor-for-tensor restoration of each named module, cross-module inequality, `eval()` mode, `requires_grad=False`, and carried session identities at `tests/test_policy_improvement_exp1b_unittest.py:5302-5337`.

## Stub-boundary audit

The injected boundaries are honest and do not replace production validation:

- `_StubOpenerTrainingModule` replaces dataset loading, manifest validation, checker/baseline selection, RL-config construction, and trainer construction at `tests/test_policy_improvement_exp1b_unittest.py:5491-5519`.
- `_injected_opener_boundaries` replaces the model constructor, environment, trainer class/type boundary, and final session constructor at `:5542-5587`.
- The final session replacement is a recorder only. Production module restoration and freezing happen before it.

No stub replaces canonical-path validation, `authenticated_checkpoint_bytes`, sanctioned deserialization, `_exp1b_checkpoint_field`, payload identity checks, module inventory/folding, census recomputation, registered RL comparisons, `load_state_dict`, `state_dict_sha256`, `eval()`, or `requires_grad_(False)`.

The withdrawn coverage statement is now accurate. These production implementations remain unexecuted:

- real dataset loading and manifest validation at `policy_improvement_full_backend.py:6462-6479`;
- real model, environment, and trainer construction at `:6540-6589`;
- the real `Exp1bSealedEvaluationSession.__init__` body.

Everything from canonical-path validation through the freeze loop, except those disclosed construction boundaries, executes as production code.

## Invariant and mutation coverage

The 26 enumerated negative cases reach their intended production refusal, plus a separate non-mapping payload case. Coverage includes:

- canonical path, checkpoint digest, and size;
- strict mapping, schema name/version, run/seed/position/applied-seed;
- budget, effective config, evaluation-access flag, external model-state identity;
- module inventory and model/rl config digests;
- all six registered RL settings;
- census ordering and empty census;
- swapped and missing module states.

Restore and freeze spot checks are load-bearing:

- a runtime no-op `load_state_dict` causes the real opener to refuse at `policy_improvement_full_backend.py:6611-6613` because the restored module digest differs;
- a no-op `eval()` causes four positive-test failures;
- a no-op `Parameter.requires_grad_` causes four positive-test failures.

Five in-memory production-opener mutations, one per region, were independently killed: schema check, folded/external identity check, census comparison, exact-mixture gate, and restored-module digest comparison. The backend function was restored after each mutation. Both trees remain byte-identical at backend hash `64acd4fa...`.

## Test results

### Buck, real Torch

Commands used `@fbcode//mode/dev-nosan`, no `--local-only`, no extra `-c`, and no rebase. Every `buck2 log what-ran` query used an explicit trace ID.

| Target | Result | Review trace |
| --- | --- | --- |
| `test_policy_improvement_exp1b` | **Pass 309, Fail 0, Skip 3** | `09964b52-153b-49df-a1a9-352f4822b61e` |
| `Exp1bTorchSealedOpenerTest` | **Pass 4, Fail 0, Skip 0** | `42e10b6d-370d-466a-98da-f5578791920e` |
| `test_policy_improvement_exp1_diagnostics` | **Pass 46, Fail 0, Skip 0** | `beeb7cf2-c34b-427c-990d-5c49ad73197c` |
| `test_policy_improvement_base_policy_restore` | **Pass 22, Fail 0, Skip 0** | `86c6197c-e808-4f5f-812d-a373952b97a0` |

Trace-specific `what-ran` checks found no `nvcc`, Caffe2, CUB/RadixSortPairs, or `.cu` compile action.

### Standalone and baseline

All standalone Python runs used `PYTHONDONTWRITEBYTECODE=1`, `python3 -B`, and an external `TMPDIR`.

| Check | Result |
| --- | --- |
| Experiment 1B plus diagnostics | **358 OK, 15 skipped** in 108.822 s |
| Five focused publication/recovery classes | **72 OK** in 75.509 s |
| Phase 4 launcher | **26 OK** in 6.620 s |
| Whole working tree | **680 run, 1 failure, 39 errors, 18 skipped** |
| External `git archive HEAD` baseline | **322 run, 1 failure, 39 errors, 3 skipped** |
| Failure identity comparison | **NEW=[]; FIXED=[]** across the same 40 failure/error identities |

All 22 reviewed paths are byte-identical between standalone and fbsource. Static compilation, BUCK parsing, `git diff --check`, cached-diff check, and changed-file whitespace checks pass. No production Python changed.

## Accepted items

- The BUCK comment still says five Torch classes and eleven methods. The actual inventory is six classes and fifteen methods. This is explicitly accepted and is not a finding.
- The three unconditional placeholder tests are not funded before the run. Their non-implementation is an owner decision and is not a finding.

## Coverage boundary remaining until a real run

The following are not exercised by the hermetic opener tests:

1. the registered evaluation dataset loader and split-manifest validator;
2. real `TinyRecursiveReasoningModel_ACTV1`, `PlanEditEnv`, and `UPITrmTrainer` construction from the sealed config;
3. the real `Exp1bSealedEvaluationSession` constructor and its dataset sample identities;
4. the three owner-deferred scientific checks: dataset-backed full restore, frozen-base versus deployed-mixture behavior in a live record, and trainer-estimator centering at the registered `1e-6` tolerance.

These are the declared owner-artifact/owner-funding boundary. The test now accurately covers the production validation and restore/freeze code before that boundary.

## Remaining external launch prerequisites

1. Owner-signed Experiment 1B admission with distinct producer/evaluator authorizations bound to final artifact hashes.
2. Admitted Experiment 0 checkpoint, signed amendment, and identity/source/procedure package.
3. Registered 1,024-record train dataset plus validation-bridge population and split manifests.
4. Distinct clean producer and evaluator roots and a fresh owner-controlled evidence generation on a filesystem with the required durability semantics.
5. `RUN_UPITRM_FULL_EXPERIMENTS=1`, with validation access only after all eight checkpoints are sealed.
6. Owner decision on target-lag tolerance and acceptance of the three deferred placeholder gaps.

## Repository integrity

- Standalone status remains 7 modified tracked files and 15 untracked files; the staged diff is empty; HEAD is unchanged.
- The 22 reviewed paths are byte-identical between standalone and fbsource. Test hash is `99bab34a...`; BUCK is `5d26d16f...`; production backend is `64acd4fa...`.
- No project file was edited, staged, committed, synced, or rebased during this review.
