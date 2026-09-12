# Experiment 1B fbsource port-fidelity review

Date: 2026-09-10

## Snapshot and verdict

- Source of truth: `/home/buiksat/trm_bellman`, branch `full-implementation`, HEAD `0f8d0297c13831c4b6cf1921a5b87c91a5726507`
- Destination: `/data/repos/fbsource/fbcode/buiksat_trm`
- Verified port handoff SHA-256: `819ed97f6128ba73ddf4a539861136ef284dd33fec870b120ca087d5f1cf42a4`
- Verified prior PASS report SHA-256: `b4bec8b73064c103a977c52552612ab31c30558c866b5ce9bfda36a95c8512a8`; the report was not modified
- Destination BUCK SHA-256 at review start: `c305f63ea5d5d4b0633261091968609acfd72c849db10d1faee8767e4bafeb46`
- Destination BUCK SHA-256 at review end: `c305f63ea5d5d4b0633261091968609acfd72c849db10d1faee8767e4bafeb46`

**REJECT.** The port is byte-faithful, the four BUCK dependency corrections are valid, the reported Buck test counts reproduce, and the auditor PAR works end to end. One high-severity test-evidence gap remains: none of the four newly running Torch tests performs the claimed four-module `torch.save` serialization or restore round trip. The suite proves Torch/import closure and a four-name constant, while the production serialization and restoration paths remain unexecuted.

The BUCK file did not change during this review. All BUCK evidence below is scoped to digest `c305f63ea5d5d4b0633261091968609acfd72c849db10d1faee8767e4bafeb46`.

## Confirmed finding

### High: the reported real four-module Torch round trip did not run

Evidence:

- The newly unskipped checkpoint test at `fbcode/buiksat_trm/tests/test_policy_improvement_exp1b_unittest.py:4917-4923` imports `policy_improvement_full_backend` and compares `EXP1B_CHECKPOINT_MODULES` with `("model", "policy_model_old", "policy_model_candidate", "target_model")`. That is its entire body.
- The restore-named test at `fbcode/buiksat_trm/tests/test_policy_improvement_exp1b_unittest.py:4925-4929` immediately calls `self.skipTest(...)`.
- A bounded scan of the complete Experiment 1B test module found no call to `seal_exp1b_training_checkpoint`, `torch.save`, or `torch.load`. The other opener references are AST/source-shape checks.
- The production serializer that should be exercised is `seal_exp1b_training_checkpoint` at `fbcode/buiksat_trm/policy_improvement_full_backend.py:5089-5183`; its `torch.save` occurs at `:5170-5172`. The production authenticated load/restore path is at `:6321-6644`. Neither is called by the four passing Torch tests.
- The BUCK comment at `fbcode/buiksat_trm/BUCK:2151-2155` therefore overstates current coverage when it says the dependency enables a four-module round trip. There are four Torch-gated classes containing seven test methods, not seven classes.

Reproduction:

```text
buck2 test @fbcode//mode/dev-nosan \
  fbcode//buiksat_trm:test_policy_improvement_exp1b -- Exp1bTorch

Pass 4, Fail 0, Skip 3
```

The four passes are three RNG/seed-order tests and the tuple assertion above. The three skips include the only restore-named test. Importing the backend executes no serialization call.

Failure mode: a defect in the checkpoint payload, module state dictionaries, folded module digest, safe deserialization, or four-module restore can remain undetected while the Buck suite reports 301 passes and the handoff claims the round trip ran. This was the largest Torch gap in the standalone PASS report, and the port did not close it.

Concrete fix: add a hermetic real-Torch regression that constructs four distinct modules, invokes the production `seal_exp1b_training_checkpoint` path, deserializes the authenticated bytes through the sanctioned loader, and verifies the four state dictionaries, individual state digests, and folded digest after restoration into fresh modules. Keep the owner-dataset/trained-checkpoint end-to-end test separately gated. The current dependency closure already reaches `policy_improvement_checkpoint_allowlist` transitively through `policy_improvement_full_backend -> upi_trm_train_lib`, so no additional dependency omission is established here.

## Port fidelity, path by path

The 21 non-BUCK files are byte-identical. BUCK differs only by the four reviewed test dependency additions.

| Path | Result |
| --- | --- |
| `configs/policy_improvement_exp1b/protocol.json` | byte-identical |
| `configs/policy_improvement_exp1b/registry.json` | byte-identical |
| `configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json` | byte-identical |
| `scripts/policy_improvement_exp1_diagnostics.py` | byte-identical |
| `scripts/policy_improvement_exp1b_aggregate.py` | byte-identical |
| `scripts/policy_improvement_exp1b_auditor.py` | byte-identical |
| `scripts/policy_improvement_exp1b_bootstrap.py` | byte-identical |
| `scripts/policy_improvement_exp1b_bridge.py` | byte-identical |
| `scripts/policy_improvement_exp1b_evidence.py` | byte-identical |
| `scripts/policy_improvement_exp1b_runtime.py` | byte-identical |
| `scripts/policy_improvement_exp1b_schema.py` | byte-identical |
| `scripts/policy_improvement_exp1b_session.py` | byte-identical |
| `scripts/policy_improvement_exp1b_theory_backend.py` | byte-identical |
| `tests/test_policy_improvement_exp1_diagnostics_unittest.py` | byte-identical |
| `tests/test_policy_improvement_exp1b_unittest.py` | byte-identical |
| `phase4_runtime_launcher.py` | byte-identical |
| `phase4_runtime_profile.py` | byte-identical |
| `policy_improvement_full_backend.py` | byte-identical |
| `policy_improvement_full_entrypoint.py` | byte-identical |
| `policy_improvement_theory_bridge_entrypoint.py` | byte-identical |
| `scripts/policy_improvement_base_policy_restore.py` | byte-identical |
| `BUCK` | expected divergence; exact four-dependency delta below |

No reviewed path is missing. The narrow destination config/script/test inventories contain no extra Experiment 1B port path that is absent from the standalone source inventory.

## BUCK dependency delta

The complete destination-versus-standalone BUCK delta at the reviewed digest contains four additions, all under test-target `deps` lists. No `srcs`, resources, test arguments, production library, binary, or Python assertion changed.

| Addition | Location | Verified effect |
| --- | --- | --- |
| `:policy_improvement_theory_bridge_v2_lib` | `BUCK:2109`, diagnostics test target | Satisfies the module-scope oracle import at `tests/test_policy_improvement_exp1_diagnostics_unittest.py:45`. It makes the existing 46 assertions importable; it does not change their oracle or expected values. |
| `fbsource//third-party/pypi/torch:torch` | `BUCK:2156`, Experiment 1B test target | Makes the test module's direct Torch use and `find_spec("torch")` gate honest in the hermetic environment. It intentionally changes four tests from skipped to executed. No production target changes. |
| `:policy_improvement_base_policy_restore` | `BUCK:2163`, Experiment 1B test target | Satisfies direct imports at `tests/test_policy_improvement_exp1b_unittest.py:5541,5563`. The test source and assertions remain byte-identical. |
| `:policy_improvement_full_backend` | `BUCK:2167`, Experiment 1B test target | Makes the direct backend import at `tests/test_policy_improvement_exp1b_unittest.py:4918` importable and brings its declared dependency closure. The file remains a resource at `BUCK:2133` for source-reading tests; resource presence alone was not an import edge. |

These changes affect only hermetic test availability. They do not alter the reviewed Python or experiment runtime. They do change which existing test code runs, deliberately and visibly: four Torch tests now execute, two base-buffer tests no longer error on import, the backend inventory test imports successfully, and all diagnostics tests reach their existing v2 oracle.

## Capability-gated/self-skip sweep

No other current test self-skips because its Buck target omits a capability.

- Experiment 1B has four Torch-gated classes containing seven methods at `tests/test_policy_improvement_exp1b_unittest.py:4813-4819,4913,5249,5949`. The target now declares Torch directly at `BUCK:2156`; the outer guards no longer trigger.
- Diagnostics has no capability skip guard. Its module-scope theory-bridge oracle import is now declared at `BUCK:2109`, and all 46 tests execute.
- `DatasetSymmetryTest` has the package's NumPy availability guard at `tests/test_policy_improvement_v1_unittest.py:22-25,3440`. `test_policy_improvement_v1` declares NumPy at `BUCK:892`; a focused Buck run passed 3 with 0 skips.
- The checkpoint-allowlist tests guard a version-specific Torch safe-globals API at `tests/test_policy_improvement_checkpoint_allowlist_unittest.py:298-305`. Their target declares NumPy and Torch at `BUCK:1015-1016`; both focused safe-global tests passed with 0 skips. Its historical-checkpoint test at `:516-520` is intentionally gated on machine-local retained evidence, not a dependency.
- The launcher-identity guard at `tests/test_policy_improvement_launcher_identity_unittest.py:109-126` is a build-mode/artifact-format guard, not a missing dependency. Under `dev-nosan`, that target currently encounters `BadZipFile` before reaching the guard because `_archive()` runs first. This is pre-existing, outside the Experiment 1B/diagnostics targets, and does not explain any of their skips.

The three remaining Experiment 1B skips are:

1. `Exp1bTorchSealedCheckpointTest.test_restore_reproduces_every_module_digest`, `tests/test_policy_improvement_exp1b_unittest.py:4925-4929`.
2. `Exp1bTorchBaseOperatorTest.test_base_and_deployed_differ_when_the_candidate_has_moved`, `:5253-5257`.
3. `Exp1bTorchCenteringParityTest.test_a_mutated_trainer_estimator_is_caught`, `:5953-5957`.

They are not packaging skips. Each method unconditionally records an unresolved owner-artifact/restored-checkpoint prerequisite. That reason is genuine for a real end-to-end run, but the tests do not automatically become runnable when artifacts appear; the test harness still needs an artifact-aware implementation. Finding 1 is the narrower case that should not require owner artifacts: a hermetic four-module serialization/deserialization test can and should run now.

## Auditor PAR

The auditor packaging finding is closed by execution.

- `policy_improvement_exp1b_auditor_bin` includes `configs/policy_improvement_v2/populations.json` in `BUCK:2062-2079`.
- The built link tree contains that file, and it is byte-identical to the fbsource repository copy. Both hash to `4812189354b49d3182553dc9bd7f2cc98c6aa44e6aead79b4b95f16ecf2977e2`.
- The auditor link tree contains no Torch package.
- The packaged binary audited a publication-valid synthetic result with exit `0`. Its checks included `census_member_ordering_rederived_from_registered_population` alongside the other seven registered checks.
- With only `census_ordering_sha256` changed, the same binary exited `1` with `Published census_ordering_sha256 is not the ordering the registered validation_bridge population produces.`

The positive fixture uses self-consistent synthetic study/evidence documents and no scientific owner result. It exercises the real packaged binary and packaged population registry, which is the port property under review.

## Buck test results

All commands used `@fbcode//mode/dev-nosan`, no `--local-only`, no extra `-c` settings, and no rebase.

| Target or filter | Result |
| --- | --- |
| `fbcode//buiksat_trm:test_policy_improvement_exp1b` | **Pass 301, Fail 0, Skip 3**, total 304. Buck trace `7c9662c4-740d-42de-92b6-a93bed50e095`; Tpx run `10414574336020316`. |
| Same target, `-- Exp1bTorch` | **Pass 4, Fail 0, Skip 3**. Buck trace `bc9cf084-a362-4601-9748-2a52b665aadc`; Tpx run `2533275185742734`. |
| `fbcode//buiksat_trm:test_policy_improvement_exp1_diagnostics` | **Pass 46, Fail 0, Skip 0**. Buck trace `807d3afa-1380-401c-b4e2-90878ba148cf`; Tpx run `27584547764072084`. |

No command output or inspected `what-ran` log showed `nvcc`, Caffe2, CUB/RadixSortPairs, or `.cu` compilation. The 53.6-second Torch-target build used 45 local and 48 remote actions. The only compile identity was standard Python PAR scaffolding; the remaining heavy work was linking, interface generation, and materialization of prebuilt artifacts.

## Cache-count discrepancy

I cannot conclusively explain the `Cached actions: 0` versus `195,791` discrepancy from the available logs, and assign no cause.

The invocations are not directly comparable: the earlier 195,791 count came from a cold-daemon build of `test_policy_improvement_base_policy_restore` that analyzed 125,360 targets, while the later Experiment 1B build analyzed 3 targets after that dependency graph had already been materialized. The later log reports Local 45, Remote 48, Cached 0, Other 91; a subsequent no-op auditor build analyzed 0 targets and ran 0 actions rather than recording cache hits. These facts may explain why the counters have different scale, but the logs do not establish Buck's accounting reason. The build-shape evidence is independent and clear: no Torch/CUDA source compilation occurred.

## Limitations and repository integrity

- The production full, theory-bridge, and launcher PAR builds running concurrently were not used as evidence in this report. Any later BUCK change belongs to their separate handoff. This review's initial and final BUCK digests are identical.
- The three owner-artifact tests remain unimplemented placeholders and unexecuted. No registered train dataset, admitted checkpoint, validation record, or scientific result was opened.
- The auditor result fixture is synthetic, not a real experiment result.
- The fbsource package remains untracked in Sapling; the reviewed BUCK path reports `?`, not staged/added/modified tracked state.
- No file was edited, staged, committed, synced, or rebased by this review. The standalone PASS report remained unchanged.

## Remaining work before a real experiment run

1. Add and run the missing hermetic four-module Torch serialization/deserialization regression described in Finding 1. Do not count the tuple assertion as round-trip coverage.
2. Carry the four test-dependency additions back through a normal review against the standalone source of truth, then resync so BUCK no longer diverges between the two trees.
3. Complete and verify the concurrent `policy_improvement_full`, `policy_improvement_theory_bridge`, and launcher PAR builds. Inspect their dependency closures and `what-ran` logs for the same resource-versus-dependency failure class.
4. Run the shared `test_policy_improvement_base_policy_restore` and relevant v2 suites against the ported shared source if the concurrent build task has not already done so.
5. Supply the owner-signed Experiment 1B admission, admitted Experiment 0 checkpoint and identity package, and registered 1,024-record train dataset/manifests.
6. Replace or supplement the three unconditional owner-artifact skips with artifact-aware tests, then run the real restore, moved-candidate base/deployed distinction, and trainer-estimator centering mutation checks.
7. Use distinct authorized producer/evaluator roots, a fresh owner-controlled evidence generation, the `RUN_UPITRM_FULL_EXPERIMENTS=1` gate, and validation-bridge authorization only after all eight checkpoints are sealed.
