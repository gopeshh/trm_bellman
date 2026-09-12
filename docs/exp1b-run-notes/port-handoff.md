# UPI-TRM Experiment 1B — fbsource port and first real-Torch run

Date: 2026-09-10

Source of truth: `/home/buiksat/trm_bellman`, HEAD `0f8d0297c13831c4b6cf1921a5b87c91a5726507`,
branch `full-implementation`. Handoff `148b1a92…`, review `b4bec8b7…` (PASS), both hashes
verified before starting.

Destination: `/data/repos/fbsource/fbcode/buiksat_trm`, at HG revision
`c595211465c40c30b895d2c1847ac1ea3bc0190d` (`[analytics-agent-embed] Harden conversation
list parsing`).

**Result: both Buck test targets are green. 301 pass / 0 fail / 3 skip on
`test_policy_improvement_exp1b`, 46/46 on `test_policy_improvement_exp1_diagnostics`.** Four
of the seven previously-skipped Torch tests now execute and pass; the remaining three skip on
an inner guard that needs a trained checkpoint. The real-Torch run exposed **four genuine
packaging defects**, all in `BUCK`, all fixed in fbsource only. None was a defect in the
reviewed Python.

---

## 1. What was copied

All 22 reviewed paths, relative layout preserved. Verified byte-identical after the copy.

Before the copy the destination held:

- **13 paths absent** — the three `configs/policy_improvement_exp1b/` documents and the ten
  new `scripts/`/`tests/` Experiment 1B modules.
- **2 paths already present and byte-identical** —
  `scripts/policy_improvement_exp1_diagnostics.py` and
  `tests/test_policy_improvement_exp1_diagnostics_unittest.py`. These are Experiment 1A
  diagnostics files that had already been synced; the copy was a no-op for them.
- **7 paths present and differing** — the seven tracked modifications.

For each of those seven I compared the fbsource copy against the standalone checkout's
**HEAD** version rather than its working-tree version. All seven were byte-identical to
standalone HEAD, so **fbsource had not diverged at all**: the copy applied exactly the
reviewed `git diff HEAD` and nothing else. A snapshot of the nine pre-existing files was
taken to `/tmp/exp1b-fbsource-backup/` before overwriting.

Post-copy verification: 21 of 22 byte-identical to the reviewed tree. The one divergence is
`BUCK`, which carries the four dependency fixes in §4.

---

## 2. BUCK merge decision

**No merge was needed. A straight copy was safe, and the target-body diff confirms it.**

Parsed both files with `ast` and compared every `name=`-bearing rule call:

```text
local targets: 136        fbsource targets: 123
present in fbsource and absent locally: []
added by the port: 13
shared targets: 123       with a changed body: 2
```

The 13 additions are the Experiment 1B and diagnostics targets, including
`test_policy_improvement_exp1b`, `test_policy_improvement_exp1_diagnostics`, and
`policy_improvement_exp1b_auditor_bin`.

The two shared targets with changed bodies are `policy_improvement_full` and
`policy_improvement_theory_bridge`, and both diffs are **pure additions** from the reviewed
change: the two `configs/policy_improvement_exp1b/` resource globs on each, plus
`:policy_improvement_exp1b_runtime` on the full PAR and
`:policy_improvement_checkpoint_allowlist`, `:policy_improvement_exp1b_auditor`,
`:policy_improvement_exp1b_runtime`, `:policy_improvement_exp1b_theory_backend` on the
theory-bridge PAR. Nothing was removed or rewritten on the fbsource side, so overwriting
loses nothing.

---

## 3. Every buck2 command, in order

All from `/data/repos/fbsource/fbcode`. No `--local-only`, no extra `-c`, no rebase, no
`buck2 clean`. Output redirected to files; tails read.

| # | Command | Result |
| --- | --- | --- |
| 1 | `build @fbcode//mode/dev-nosan …:test_policy_improvement_exp1b` | exit 0, **3.4s** |
| 2 | `test  …:test_policy_improvement_exp1b` | exit 32 — **Pass 295, Fail 2, Skip 7** |
| 3 | `build …:test_policy_improvement_exp1b` (after fixes A+B) | exit 0, **53.6s** |
| 4 | `test  …:test_policy_improvement_exp1b` | exit 32 — **Pass 300, Fail 1, Skip 3** |
| 5 | `test  …:test_policy_improvement_exp1b` (after fix C) | exit 64 — **Pass 301, Fail 0, Skip 3** |
| 6 | `test  …:test_policy_improvement_exp1_diagnostics` | exit 32 — **Pass 0, Fatal 46** |
| 7 | `test  …:test_policy_improvement_exp1_diagnostics` (after fix D) | exit 0 — **Pass 46, Fail 0** |
| 8 | `run   …:policy_improvement_exp1b_auditor_bin -- --help` | exit 0 |
| 9 | `run   …auditor_bin -- <genuine result> --parent-populations <PAR copy>` | exit 0, full check list |
| 10 | `run   …auditor_bin -- <substituted ordering> --parent-populations <PAR copy>` | exit 1, census refusal |
| 11 | final `build` + both `test` targets | build 0.1s fully cached; **301/0/3** and **46/0/0** |

Exit 64 on command 5 and 11 is tpx's convention for a run containing skips. `Fail 0`,
`Fatal 0`, `Infra Failure 0`, `Build failure 0` on both.

### Cache statistics and the torch-build guard

```text
build #1   Duration 3.4s    Local 15   Remote  0   Cached 0   Other 43
           materialized 355KiB, downloaded 1.7GiB (RE 1.4GiB + HTTP 262MiB)
           HG c595211465c4, local changes: true

build #3   Duration 53.6s   Local 45   Remote 48   Cached 0   Other 91
           materialized 299MiB, uploaded 6.1MiB

build #11  Duration 0.1s    Local  0   Remote  0   Cached 0   Other  2
```

`buck2 log summary` reported `Cached actions: 0` on every build, which on its face looks
like a 0% cache-hit rate. **It is not a torch source build, and I verified that directly
rather than inferring it from the duration.** `buck2 log what-ran` on the 53.6s build:

```text
   0  matches for nvcc | RadixSortPairs | caffe2 | *.cu
  39  cxx_link_link_group
  30  link_groups_undefined_syms
  28  link_groups_global_syms
  10  generate_shared_library_interface
   3  par
   2  cxx_compile      (static_extension_info.cpp, embedded_main.cpp)
```

The only two compiles are the standard Python-binary scaffolding. Torch arrived as prebuilt
artifacts — that is what the 299MiB materialization and the RE download are. I did not stop,
because the condition the guard exists for (a local caffe2 source build) demonstrably did not
occur. If you want the literal cache-hit metric to read differently, that is a buck2 summary
accounting question, not a build-shape question.

---

## 4. Defects the real-Torch path exposed

Four, all of the same class: **the standalone checkout puts the whole repository on
`sys.path`, so a missing Buck dependency edge is invisible there and only appears in a
hermetic runfiles tree.** All four are `BUCK` dependency omissions. None is a defect in the
reviewed Python, and none required weakening or skipping a test.

**A. `test_policy_improvement_exp1b` had no Torch dependency.** The seven Torch classes gate
on `importlib.util.find_spec("torch")`, so the suite reported OK while silently never running
any of them. The sibling `test_policy_improvement_base_policy_restore` target declares
`fbsource//third-party/pypi/torch:torch` explicitly; this one did not. **This is the most
consequential of the four**: without it, porting to Buck would have produced a green run that
proved nothing about Torch.

**B. `Exp1bBaseArtifactBufferTest` could not import
`scripts.policy_improvement_base_policy_restore`.** Two tests errored with
`ModuleNotFoundError`. Fixed by adding `:policy_improvement_base_policy_restore`.

**C. `Exp1bTorchSealedCheckpointTest` could not import `policy_improvement_full_backend`.**
Surfaced only after fix A let the test run at all. The backend file *was* declared in
`resources`, which places it in the runfiles tree but does not make it importable and does not
bring its own imports along — the actual error was one level deeper,
`No module named 'policy_improvement_non_smoke_checkpoint'`. Fixed by adding
`:policy_improvement_full_backend`, which carries that transitively.

**D. `test_policy_improvement_exp1_diagnostics` could not import
`scripts.policy_improvement_theory_bridge_v2`.** All 46 tests died at import
(`Fatal 46`). The suite's oracle half compares the finite-reference bound against the v2
theory bridge's own arithmetic. Fixed by adding `:policy_improvement_theory_bridge_v2_lib`.
Note this target was already broken in fbsource before the port — both its source files were
already present and byte-identical, so it had evidently never been run under Buck.

### The exact BUCK delta, for the review round on the standalone checkout

`/home/buiksat/trm_bellman/BUCK` is `9fefc08a…` (reviewed, untouched).
`/data/repos/fbsource/fbcode/buiksat_trm/BUCK` is `c305f63e…`. The whole difference:

```diff
@@ target test_policy_improvement_exp1_diagnostics, deps
         ":policy_improvement_exp1_diagnostics",
+        ":policy_improvement_theory_bridge_v2_lib",

@@ target test_policy_improvement_exp1b, deps
+        "fbsource//third-party/pypi/torch:torch",
         ":confirmatory_runtime_launcher_lib",
         ":phase4_runtime_launcher_lib",
         ":phase4_runtime_profile",
+        ":policy_improvement_base_policy_restore",
+        ":policy_improvement_full_backend",
         ":policy_improvement_exp1_diagnostics",
```

(each with an explanatory comment in the file). **I did not apply this to the standalone
checkout.** Per your constraint it stays byte-identical to the reviewed artifact and this
change should go through a normal fix round.

---

## 5. The seven Torch tests, by name

Class-level runs, `--regex <ClassName>`, after all four fixes.

| # | Test | Result |
| --- | --- | --- |
| 1 | `Exp1bTorchRepeatabilityTest.test_two_registered_seeds_produce_different_draws` | **PASS** |
| 2 | `Exp1bTorchRepeatabilityTest.test_one_registered_seed_reproduces_exactly` | **PASS** |
| 3 | `Exp1bTorchRepeatabilityTest.test_the_session_seeds_before_the_model_is_constructed` | **PASS** |
| 4 | `Exp1bTorchSealedCheckpointTest.test_the_sealed_checkpoint_carries_all_four_modules` | **PASS** (failed before fix C) |
| 5 | `Exp1bTorchSealedCheckpointTest.test_restore_reproduces_every_module_digest` | **SKIP** |
| 6 | `Exp1bTorchBaseOperatorTest.test_base_and_deployed_differ_when_the_candidate_has_moved` | **SKIP** |
| 7 | `Exp1bTorchCenteringParityTest.test_a_mutated_trainer_estimator_is_caught` | **SKIP** |

Per-class: `Exp1bTorchRepeatabilityTest` Pass 3 / Skip 0; `Exp1bTorchSealedCheckpointTest`
Pass 1 / Skip 1; `Exp1bTorchBaseOperatorTest` Pass 0 / Skip 1;
`Exp1bTorchCenteringParityTest` Pass 0 / Skip 1. A `--regex Exp1bTorch` run gives
**Pass 4, Fail 0, Skip 3**.

The three remaining skips are **not** the module-level "Torch is not installed" guard, which
is now satisfied. They are inner `skipUnless` guards with their own reasons, printed verbatim
by the run:

```text
test_restore_reproduces_every_module_digest
  "Requires the registered 4x4 dataset and a trained budget-final checkpoint;
   see the handoff's unresolved-risk section."
test_base_and_deployed_differ_when_the_candidate_has_moved
  "Requires a restored four-module checkpoint; see the handoff's unresolved-Torch
   section for the exact fbsource command."
test_a_mutated_trainer_estimator_is_caught
  "Requires a restored four-module checkpoint; see the handoff's unresolved-Torch section."
```

These need owner artifacts (external prerequisites 2 and 3), not code. They cannot be
unblocked by anything in this port.

**What test 4 now proves that nothing proved before:** a real `torch.save` of the four
Experiment 1B modules produces a checkpoint whose sealed inventory contains all four, under
real Torch serialization rather than synthetic bytes. That was the largest standing gap in
eight rounds of review.

---

## 6. Auditor PAR resource — the gap the standalone checkout could not close

Codex's finding 7 was closed only by a source-level test reading the `BUCK` target text.
Three checks now close it by execution.

**`--help` through the PAR** (exit 0) shows `--parent-populations PARENT_POPULATIONS` in the
required-argument group. That proves the argument exists, not that the resource resolves, so
I did not stop there.

**The resource is materialized.** The binary is an inplace PAR whose link tree is
`…/__policy_improvement_exp1b_auditor_bin__/policy_improvement_exp1b_auditor_bin#link-tree/`.
It contains:

```text
configs/policy_improvement_v2/populations.json          ← the one under test
configs/policy_improvement_exp1b/protocol.json
configs/policy_improvement_exp1b/registry.json
configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json
```

`cmp` against the repository document: identical.
SHA-256 `4812189354b49d3182553dc9bd7f2cc98c6aa44e6aead79b4b95f16ecf2977e2`.

**It works end to end from inside the PAR.** I published a genuine result with the reviewed
fixture (read-only against the standalone checkout, external `TMPDIR`) and ran the packaged
binary against it, passing the **PAR's own materialized copy** to `--parent-populations`:

```text
exit 0
exp1b-audit: document_sha256=82312cb78ba8e778e65fa22c2afffbafa2c7ea5c07dde210e364e145c517f783
exp1b-audit: mean_signed_gap=85.9999999999999
exp1b-audit: interval=[79.49999999999993, 91.99999999999991]
exp1b-audit: checks=document_canonical_digest,registered_identity_and_access_flags,
             census_member_ordering_rederived_from_registered_population,
             role_scoped_admission_binding,study_document_bindings,
             per_seed_bound_algebra,secondary_diagnostics,frozen_bootstrap_interval
```

And the negative case, the same invocation with only `census_ordering_sha256` replaced:

```text
exit 1
exp1b-audit: REFUSED: Published census_ordering_sha256 is not the ordering the
             registered validation_bridge population produces.
```

The census rederivation is in the executed check list and it fires on a substitution. **No
BUCK resource or profile fix was needed for the auditor** — the reviewed declarations were
correct.

---

## 7. Repository integrity

**Standalone checkout unmodified.** `git status --porcelain -uall` shows exactly 22 entries,
and all 22 files hash to the values recorded in the reviewed handoff. HEAD unchanged. The
publish-a-result probe ran with `PYTHONDONTWRITEBYTECODE=1` and an external `TMPDIR` and wrote
nothing into the tree.

**fbsource.** `hg status` shows 252 entries, every one an untracked `? fbcode/buiksat_trm/…`
path. Zero `M`, `A`, `R`, or `!` entries anywhere in the repository, including under
`nest/libs/analytics-agent-embed`, which `hg status` reports clean — that work appears to be
committed into `c595211465c4` already. Nothing outside `fbcode/buiksat_trm` was read for
modification or written.

Not done, per your constraints: no rebase, no commit, no amend, no `hg add`, no diff
submission. `buiksat_trm` remains entirely untracked.

---

## 8. What remains before a real experiment run

**Code and packaging.** One item, and it is bookkeeping: the four-line `BUCK` dependency
delta in §4 needs a normal fix round against the standalone checkout so it goes through
review, then a re-sync. Until then the two trees' `BUCK` files differ and fbsource is the
only one that builds correctly.

**Two things this port did not cover.** You asked for three specific commands, so I ran
exactly those. Not built or tested here:

- `policy_improvement_full`, `policy_improvement_theory_bridge`, and the launcher PARs. Both
  of the first two gained real dependency edges in this change, and the same
  resource-versus-dependency mistake that produced defects B, C, and D could be present in
  either. **I would build all three before trusting the launch path.**
- `test_policy_improvement_base_policy_restore` and the other v2 suites that share
  `scripts/policy_improvement_base_policy_restore.py`. You verified that target passes 22/22,
  but that was before this port replaced the file. It should be re-run.

**Owner artifacts, unchanged and still the gate.** The three remaining Torch skips, and any
real run, need: the owner-signed Experiment 1B admission with distinct full and theory-bridge
authorizations bound to final artifact hashes; the admitted Experiment 0 checkpoint with its
signed amendment and identity package; and the registered 1,024-record train dataset with its
dataset/split/ordering and validation-bridge population manifests. Then distinct producer and
evaluator checkout roots, an owner-controlled evidence root, `RUN_UPITRM_FULL_EXPERIMENTS=1`,
and validation-bridge authorization only after all eight checkpoints are sealed.

**One open scientific question**, carried since round 5 and still unanswered by this run: the
trainer-estimator centering tolerance is 1e-6, taken from the frozen v2 amendment. The test
that would exercise it against real float32 heads is skip #7, which needs a restored
four-module checkpoint. If that tolerance does not hold in practice, every seed refuses —
correct behaviour, but an owner decision, and it will surface at the worst possible moment
during the first real run rather than now.
