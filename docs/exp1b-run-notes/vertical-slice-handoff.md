# UPI-TRM Experiment 1B vertical slice — final snapshot

Date: 2026-09-11 (updated: Low finding closed)

Standalone source of truth: `/home/buiksat/trm_bellman`, branch `full-implementation`, base
commit `0f8d0297c13831c4b6cf1921a5b87c91a5726507`. fbsource package:
`/data/repos/fbsource/fbcode/buiksat_trm` at HG revision
`c595211465c40c30b895d2c1847ac1ea3bc0190d`. Nothing staged, committed, pushed, `hg add`ed, or
rebased in either tree. All 22 paths byte-identical between the two trees.

This round answers review `8bae85b404c6d6689730101f4af7d881093f90d42a44f4c5d5638763f93e48f3`
(REJECT, Medium): the round-trip test exercised the production serializer but bypassed the
production authenticated validator and restore loop. It then closes the one Low, non-blocking
item from the follow-up PASS review
`e3a28c94bb860d0cbd88883f74a7bec6653cfd648dc38026796bec6facb1f0a6` (§2a).

---

## 0. What was wrong and what fixes it

**The finding was correct.** `Exp1bTorchFourModuleRoundTripTest._restore` built its own
`torch.nn.Sequential` and called `load_state_dict` itself. The test tree contained no call site
for `open_exp1b_sealed_evaluation_session`, so production authentication and payload validation
at `policy_improvement_full_backend.py:6364-6459` and the four-module load/digest/freeze loop at
`:6592-6619` never ran. Two falsifications proved it: the opener replaced by a raising counter
left all four tests green with `opener_calls=0`, and `schema_version = 999` after serialization
also left them green.

**I also withdraw the item-8 claim from the previous handoff.** It said the payload-validation
half of the opener was covered. It was not, and that claim is not repeated here.

**What fixes it:** a new class, `Exp1bTorchSealedOpenerTest`
(`tests/test_policy_improvement_exp1b_unittest.py:5136-5420`), four tests, no owner artifacts,
which calls the real `open_exp1b_sealed_evaluation_session`. §2 states exactly what it covers
and what it does not.

**Route taken: injected boundaries, no production change.** This was your preferred route and
it reaches everything required, so no extract-method refactor was performed.
`policy_improvement_full_backend.py` is unchanged at `64acd4fa…`.

---

## 1. What changed this round

One path.

| path | before | after |
| --- | --- | --- |
| `tests/test_policy_improvement_exp1b_unittest.py` | `d998b3b3…` | **`947310231ed8240659a03c746d6ed21ff6f133416d08215fec5964cdf7857cf9`** |

`BUCK` is unchanged at `5d26d16f…`. No production Python changed.

Two edits inside that file:

1. `_StubSealRlConfig.dict()` now returns the registered RL settings
   (`_REGISTERED_RL_SETTINGS`) rather than `{"lr", "gamma"}`, because the opener reads them back
   out of the sealed payload and gates on them. The existing round-trip tests recompute
   everything, so they are unaffected.
2. The new `Exp1bTorchSealedOpenerTest` and its stub boundaries.

---

## 2. Coverage statement — precise, not overstated

`Exp1bTorchSealedOpenerTest._drive` seals four distinct modules through the production
serializer, writes the bytes to an external temporary file, and calls the real opener with the
exact expected SHA-256, size, model-state digest, effective-config digest, interaction count,
run identity, and census ordering.

### Executed as production code, and each mutation-checked

Every item below is a real production check reached by the real opener, with one mutation in
`test_each_payload_invariant_is_enforced_by_the_production_opener` asserting
`Exp1bSealedEvaluationError`:

| invariant | production site |
| --- | --- |
| canonical checkpoint path | `:6364-6367` |
| exact byte / size / SHA binding (`authenticated_checkpoint_bytes`) | `:6371-6386` |
| strict mapping | `:6388-6391` (own test, needs a whole non-mapping payload) |
| schema name and schema version | `:6393-6400` |
| run id, seed, seed position, applied seed | `:6402-6410` |
| budget-final interaction count | `:6411-6417` |
| effective config digest | `:6418-6424` |
| `resolved_evaluation_data is False` | `:6425-6431` |
| module inventory | `:6432-6437` |
| folded identity vs the payload's own `model_state_sha256` | `:6441-6442` |
| folded identity vs the external expected model-state digest | `:6443` |
| model-config and rl-config digests | `:6447-6459` |
| census ordering recomputation, and empty census | `:6481-6497` |
| registered gamma, mixture alpha, unroll depth, exact mixture, persistent latent, no epsilon-greedy | `:6500-6539` |
| four-module load + per-module digest comparison | `:6592-6612` |
| missing `<name>_state_dict` | `:6597-6601` |
| freeze loop: `eval()` and `requires_grad_(False)` | `:6613-6619` |

The positive test additionally asserts that each restored module is tensor-for-tensor its own
sealed state and **not** any of the other three, that all four end up frozen and out of train
mode, and that the identities the opener carries into the session match the folded digest.

### Not executed — genuinely owner-artifact-bound

Stated plainly so this is not overclaimed again:

- **Dataset-backed loading and manifest validation**, `:6462-6479`.
  `training_module.build_dataset_from_paths` and `_validate_materialized_split_manifest` are
  supplied by a stub. This needs the registered evaluation split and its manifest.
- **Real model, environment, and trainer construction**, `:6540-6589`.
  `TinyRecursiveReasoningModel_ACTV1`, `PlanEditEnv`, `UPITrmTrainer`, and
  `training_module.build_trainer` are injected. Real construction needs the dataset.
- **`Exp1bSealedEvaluationSession.__init__`**, replaced by a recorder, because it calls
  `dataset_sample_sha256s(evaluation_dataset)` on a real dataset.

Every invariant on your required list is covered. None was dropped.

---

## 2a. Low finding from the PASS review — closed

Production checks **both halves** of the folded identity at
`policy_improvement_full_backend.py:6441-6444`: the recomputed fold must equal the payload's own
`model_state_sha256`, **and** it must equal the externally supplied
`expected_model_state_sha256`. The matrix mutated only the external argument, so a regression
deleting just the payload self-consistency half would have gone uncaught. The five-region
identity mutation removed both comparisons at once, which the external case kills without
proving either half independently.

**One case added**, nothing else changed:

```python
(
    "payload self-consistent model-state identity",
    field("model_state_sha256", _digest("another-folded-identity")),
    {},
),
```

It rewrites the payload's own folded digest while leaving `expected_model_state_sha256`
genuine, and requires the production refusal.

**Self-verified the same way as the opener.** Only the payload-half comparison was removed in
the fbsource production copy — `folded != _exp1b_checkpoint_field(payload,
"model_state_sha256", kind=str)` replaced by `False`, leaving `folded !=
expected_model_state_sha256` intact:

```text
Tests finished: Pass 3. Fail 1.
FAIL: test_each_payload_invariant_is_enforced_by_the_production_opener
      (invariant='payload self-consistent model-state identity')
backend restored: 64acd4fabf8a6fa08e760be6ebf782238a887769312cc596dc35d8ffb4d9660b
```

Exactly one subtest fails, and it is the new one. The "external model-state identity" case
still passes under that mutation, which is what shows the two halves are now proven
independently rather than jointly.

---

## 3. Falsification — the required proof

Your falsification, run inside the test PAR against the final file:

```text
probe: Exp1bTorchSealedOpenerTest         run=4 failures=0 errors=29 opener_calls=29  -> RED
probe: Exp1bTorchFourModuleRoundTripTest  run=4 failures=0 errors=0  opener_calls=0   -> GREEN
```

With `open_exp1b_sealed_evaluation_session` replaced by a counter that raises, the new class
**fails with 29 errors and 29 opener calls**. It cannot pass without the production opener.

The round-trip class stays green with `opener_calls=0`, which is correct and expected: it
covers the serializer half only. The two classes together are what close the finding.

### Production-opener mutation matrix

Each defect introduced into `policy_improvement_full_backend.py` in the fbsource copy, the new
class run under Buck, then the file restored and its SHA-256 compared. All five killed; the
file restored to `64acd4fabf8a6fa08e760be6ebf782238a887769312cc596dc35d8ffb4d9660b`.

| re-introduced defect in the opener | result |
| --- | --- |
| payload: drop the schema name/version check | **Pass 2, Fail 1** |
| identity: drop the folded / expected model-state comparison | **Pass 2, Fail 1** |
| census: drop the ordering comparison | **Pass 2, Fail 1** |
| restore loop: drop the per-module digest comparison | **Pass 2, Fail 1** |
| rl gates: drop the exact-mixture requirement | **Pass 2, Fail 1** |

One mutation per region of the opener — payload, identity, census, RL gates, restore loop —
rather than one per invariant, given the deadline. Each region is sensitive.

---

## 4. Test results

### Buck, real Torch

`@fbcode//mode/dev-nosan`, no `--local-only`, no extra `-c`, no rebase. Every `buck2 log`
command used `--trace-id`.

| target | result | trace | CUDA matches |
| --- | --- | --- | --- |
| `test_policy_improvement_exp1b` | **Pass 309, Fail 0, Skip 3** | `c0e141f7-2b4a-444c-b253-f539f86ce29d` | 0 |
| `test_policy_improvement_exp1_diagnostics` | **Pass 46, Fail 0, Skip 0** | `22959e5c-d178-457d-afc8-2be9f3bc147c` | 0 |
| `test_policy_improvement_base_policy_restore` | **Pass 22, Fail 0, Skip 0** | `5204259b-17d5-4998-973a-8c69d5838c46` | 0 |
| `Exp1bTorchSealedOpenerTest` alone | **Pass 4, Fail 0, Skip 0** | — | 0 |

309 is 305 plus the four new tests; the Low-finding case is a new subTest inside an existing
test method, so the method count is unchanged. Zero `nvcc | RadixSortPairs | caffe2 | .cu` across all
three by trace id. Exit 64 on the exp1b target is tpx's skip convention; `Fail 0`, `Fatal 0`,
`Infra Failure 0`, `Build failure 0`.

### Standalone, no Torch

```text
exp1b (module alone, after the Low fix)          312 tests  OK (skipped=15)
exp1b + diagnostics                              358 tests  OK (skipped=15)
five focused publication/recovery classes         72 tests  OK
phase4 launcher (signing-disabled Git home)       26 tests  OK
```

Skips rose from 11 to 15: the new class is Torch-gated and this checkout has no Torch. All four
run under Buck.

### Whole tree versus `git archive HEAD`

Both trees under the same signing-disabled Git home (`commit.gpgsign=false`,
`tag.gpgsign=false`, `GIT_CONFIG_NOSYSTEM=1`):

```text
HEAD    : Ran 322 tests in 257.5s | FAILED (failures=1, errors=39, skipped=3)
WORKING : Ran 680 tests in 357.4s | FAILED (failures=1, errors=39, skipped=18)
HEAD failing: 40 | working failing: 40
NEW    : []
FIXED  : []
```

### Static

`py_compile` clean on the changed test module; `ast.parse` clean on `BUCK` in both trees;
`git diff --check` exit 0; zero trailing-whitespace lines in the changed file.

---

## 5. Torch-gated inventory

| class | methods | Buck result |
| --- | ---: | --- |
| `Exp1bTorchRepeatabilityTest` | 3 | 3 pass |
| `Exp1bTorchSealedCheckpointTest` | 2 | 1 pass, 1 skip |
| `Exp1bTorchFourModuleRoundTripTest` | 4 | 4 pass |
| **`Exp1bTorchSealedOpenerTest`** | **4** | **4 pass** |
| `Exp1bTorchBaseOperatorTest` | 1 | 1 skip |
| `Exp1bTorchCenteringParityTest` | 1 | 1 skip |

**6 classes, 15 methods, 12 executed, 3 skipped.** The `BUCK` comment says "five classes,
eleven methods"; it is now one class and four methods stale. It is a comment, not a
dependency, and correcting it would change `BUCK` in both trees for no functional gain this
close to the deadline — flagged here instead. The four dependency lines it annotates are
unchanged and correct.

---

## 6. The three placeholders — unchanged, not funded

Not implemented, per your decision. They call `skipTest` on line one and will not self-activate
when artifacts arrive.

1. **`Exp1bTorchSealedCheckpointTest.test_restore_reproduces_every_module_digest`** — its
   pre-dataset half is now genuinely covered by `Exp1bTorchSealedOpenerTest`. What remains
   unique to it is the dataset-backed loading, manifest validation, and real
   model/environment/trainer construction listed in §2.
2. **`Exp1bTorchBaseOperatorTest.test_base_and_deployed_differ_when_the_candidate_has_moved`** —
   the frozen-base versus deployed-mixture seam at `:5700-5760`. The review notes this could be
   exercised with policy/environment/record stubs at lower cost than a full registered session;
   my earlier cost estimate overstated the minimum. Highest scientific value of the three.
3. **`Exp1bTorchCenteringParityTest.test_a_mutated_trainer_estimator_is_caught`** — needs a real
   trainer and representative data to establish the registered 1e-6 tolerance on real float32
   heads. Heaviest. A narrow hermetic mutation test would be cheaper but would not establish
   the tolerance.

---

## 7. Final snapshot: all 22 paths

Identical in both trees, verified by `cmp`.

| SHA-256 | path |
| --- | --- |
| `2539a349cbbf10d6fdc045f6a34905457b4e9cf92e098b08174050e5bc0709a0` | `configs/policy_improvement_exp1b/protocol.json` |
| `0b08942b7136cbed246bd347e1420cb56d0dd7cd2933425244d4e64f5622a95b` | `configs/policy_improvement_exp1b/registry.json` |
| `5b160aa85ffa672a3639da699201fee60d5bb4e281276b901a62caf10ea5f54b` | `configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json` |
| `05d173c3a2f1d139112e4b7249c0c9056e781b8fb8a382d8dcec38214ebf97ee` | `scripts/policy_improvement_exp1_diagnostics.py` |
| `c6fd0132c47a0acbcf7af36df4a3eb8bc6f82599784034f0e33fa78f44885f77` | `scripts/policy_improvement_exp1b_aggregate.py` |
| `d3fe77ccdb8b3ec37b3903c0471d6032c70117fd9b6dc03cbfe9b2e7a879e2e0` | `scripts/policy_improvement_exp1b_auditor.py` |
| `ebfd3535406ffdc88ba5d06585de13d55d70d22543588aa602a8e50c0990532b` | `scripts/policy_improvement_exp1b_bootstrap.py` |
| `4e90a961d0e99eacc9cbad034b936afefff1c3c7a55cd54f915e6c0e2b4dc50c` | `scripts/policy_improvement_exp1b_bridge.py` |
| `1ee3646aa7ceaac28eba73c7bb5dc333e26139c5655718c408a257bac5077648` | `scripts/policy_improvement_exp1b_evidence.py` |
| `52aa93a574532e17ffec1d393c82a68e20f47f4c99d222e6ee5615d239346be8` | `scripts/policy_improvement_exp1b_runtime.py` |
| `8da70117e217f0ee79a9270412c446187a3978f9c5478cfc8e03832f16a49e56` | `scripts/policy_improvement_exp1b_schema.py` |
| `6754b0c861fc3c38846f416a3b44af0f3e422376be17058ad550846c6e8fd355` | `scripts/policy_improvement_exp1b_session.py` |
| `c6a7a6f0e0f8ce12144e6754ea5001cf7489fba4ca44e7a2a1bf4545c3baf302` | `scripts/policy_improvement_exp1b_theory_backend.py` |
| `39e196e3a978eaa5dc90c9b00751296bde1628dd51da4f9180ac6ce4abeda364` | `tests/test_policy_improvement_exp1_diagnostics_unittest.py` |
| **`947310231ed8240659a03c746d6ed21ff6f133416d08215fec5964cdf7857cf9`** | `tests/test_policy_improvement_exp1b_unittest.py` *(changed)* |
| `5d26d16f294d755cb18e19089c845b26e793a1fc255492339d86d8ee08e3b917` | `BUCK` |
| `059f89f059abd5c4653be844cbe14f87b383253cfa0ad001313c6e397b408c35` | `phase4_runtime_launcher.py` |
| `e634d0b6a6518a33b295c57448c14c3fa900e01255648675ddb93deabe24601b` | `phase4_runtime_profile.py` |
| `64acd4fabf8a6fa08e760be6ebf782238a887769312cc596dc35d8ffb4d9660b` | `policy_improvement_full_backend.py` |
| `9b2d1f7643c6b07334d077f489cfdec1d3f63b160d283beac9d1bfe294309059` | `policy_improvement_full_entrypoint.py` |
| `83e726876d173d6a19addfc1e82a1828d4218b210514b37a6fadec63313e336c` | `policy_improvement_theory_bridge_entrypoint.py` |
| `a061afd98b83a1e16b6bdd33cb749d89a4b4ac34e49ae85dfdb087218b38ec7f` | `scripts/policy_improvement_base_policy_restore.py` |

Standalone working tree: 22 entries in `git status --porcelain -uall` — 7 tracked
modifications, 15 untracked. HEAD unchanged.

### fbsource edit log, for reconciliation

| timestamp | file | change |
| --- | --- | --- |
| 2026-09-10 13:23:02 -0700 | `BUCK` | four test-dependency additions → `c305f63e…` |
| 2026-09-10 16:37:10 -0700 | `BUCK` | corrected comments → `5d26d16f…` *(unchanged since)* |
| 2026-09-10 16:37:10 -0700 | `tests/…exp1b_unittest.py` | round-trip class → `d998b3b3…` |
| 2026-09-11 10:06:13 -0700 | `tests/…exp1b_unittest.py` | sealed-opener class → `99bab34a…` |
| **2026-09-11 11:03:11 -0700** | `tests/…exp1b_unittest.py` | **payload-half folded-identity case → `947310231…`** |

`policy_improvement_full_backend.py` was temporarily mutated five times during §3 and twice
more during §2a, and restored each time; final hash `64acd4fa…` matches the reviewed value. No other fbsource file touched.
`hg status` reports zero entries outside `fbcode/buiksat_trm/` and zero `M`/`A`/`R`/`!`
anywhere.

---

## 8. Limitations and residual risks

**The three placeholders in §6** remain the largest coverage gap and do not self-activate.

**The opener test injects four boundaries** — dataset/manifest, model constructor, environment,
trainer, plus the session constructor. §2 lists exactly which production code that leaves
unexecuted. Everything between the canonical-path check and the freeze loop is production.

**The `BUCK` comment is one class and four methods stale** (§5). Deliberately not corrected this
round.

**No `.buckconfig` in the standalone checkout**, so the change is validated under Buck only in
fbsource.

**Buck cache accounting.** On a warm daemon `Cached actions: 0` is not a cache miss; judge build
shape with `what-ran --trace-id` and always pass the trace id — the bare command reported a
different, earlier invocation during the PAR round.

**Carried forward, unchanged.** Advisory locking protects cooperating writers only; the
`nlink == 2` window is recoverable rather than eliminated; two Stage A processes on the same
unpublished seed still race; `evaluator_attestation` is bound to the route but not to a durable
artifact; the standalone auditor's `admission` remains optional by design and reports the
omission; retirement reads the whole checkpoint into memory. Eight pre-existing whole-tree
failures in the v1/v2 and shared suites are present at HEAD.

---

## 9. What remains before a real experiment run

No code or packaging blocker found. Outstanding items are owner artifacts or owner decisions:

1. The owner-signed Experiment 1B admission with distinct producer and evaluator authorizations
   bound to final artifact hashes.
2. The admitted Experiment 0 checkpoint, signed amendment, and identity/source/procedure
   package.
3. The registered 1,024-record train dataset and the validation-bridge population and split
   manifests. These also unblock the dataset-backed half of the opener listed in §2.
4. Distinct clean producer and evaluator roots, an owner-controlled fresh evidence generation,
   and a filesystem with the required durability semantics.
5. `RUN_UPITRM_FULL_EXPERIMENTS=1`, and validation-bridge authorization only after all eight
   checkpoints are sealed.
6. Whether target lag stays record-only or receives a registered refusal tolerance.
7. Whether to fund the three placeholders in §6. Number 3 there is the only thing that will
   establish whether the 1e-6 centering tolerance holds on real float32 heads; if it does not,
   every seed refuses.
