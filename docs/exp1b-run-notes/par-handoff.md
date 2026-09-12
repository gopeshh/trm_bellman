# UPI-TRM Experiment 1B — PAR builds and the stale-target re-run

Date: 2026-09-10

Follows the port handoff `819ed97f6128ba73ddf4a539861136ef284dd33fec870b120ca087d5f1cf42a4`
(hash verified). Repository `/data/repos/fbsource/fbcode/buiksat_trm` at HG revision
`c595211465c40c30b895d2c1847ac1ea3bc0190d`.

**Result: all three PARs build and run, every declared module resolves inside every runfiles
tree, and `test_policy_improvement_base_policy_restore` is 22/22 against the ported source.
No new defect. No BUCK edit was needed this round — `BUCK` is unchanged at
`c305f63ea5d5d4b0633261091968609acfd72c849db10d1faee8767e4bafeb46`, exactly as the port left
it.**

---

## 1. Every command, in order

All from `/data/repos/fbsource/fbcode`, `@fbcode//mode/dev-nosan`, no `--local-only`, no
extra `-c`, no rebase. Output redirected; tails read.

| # | Command | Result |
| --- | --- | --- |
| 1 | `build …:policy_improvement_full` | exit 0, **1:15.4s** |
| 2 | `run   …:policy_improvement_full -- --help` | exit 1 — entrypoint preflight refusal (expected, §3) |
| 3 | in-PAR import probe, `policy_improvement_full` | **29/29 modules**, torch 2.15.0a0+fb, `cuda_available=True` |
| 4 | in-PAR resource probe, `policy_improvement_full` | all three bundled exp1b documents validate and cross-bind |
| 5 | `build …:policy_improvement_theory_bridge` | exit 0, **1:26.8s**, 4.1GiB materialized |
| 6 | `run   …:policy_improvement_theory_bridge -- --help` | exit 1 — same preflight refusal |
| 7 | in-PAR import probe, `policy_improvement_theory_bridge` | **39/39 modules**, torch present, CUDA available |
| 8 | in-PAR resource probe, `policy_improvement_theory_bridge` | all three documents validate and cross-bind |
| 9 | `build …:phase4_runtime_launcher` | exit 0, **4.3s** |
| 10 | `run   …:phase4_runtime_launcher -- --help` | **exit 0**, full usage, 13 purposes incl. both exp1b ones |
| 11 | `run   …:phase4_runtime_launcher --purpose policy-improvement-exp1b-training …` | exit 2, authorization refusal (§4) |
| 12 | `run   …:phase4_runtime_launcher --purpose policy-improvement-exp1b-bridge …` | exit 2, different authorization refusal |
| 13 | in-PAR import probe, `phase4_runtime_launcher` | **3/3 modules**; torch absent by design |
| 14 | in-PAR profile probe, `phase4_runtime_launcher` | exp1b audit profile carries the population registry |
| 15 | `build …:test_policy_improvement_base_policy_restore` | exit 0, 0.3s, fully warm no-op |
| 16 | `test  …:test_policy_improvement_base_policy_restore` | exit 0 — **Pass 22, Fail 0, Skip 0** |

The launcher PAR target name resolved from `BUCK` is **`phase4_runtime_launcher`**
(`python_binary`, `main_module = "phase4_runtime_launcher"`, `imports_monitor = False`).

### Build statistics, by explicit trace id

| target | duration | local | remote | cached | other |
| --- | --- | --- | --- | --- | --- |
| `policy_improvement_full` | 1:15.4s | 42 | 58 | 0 | 82 |
| `policy_improvement_theory_bridge` | 1:26.8s | 7 | 93 | 0 | 89 |
| `phase4_runtime_launcher` | 4.3s | 2 | 0 | 0 | 41 |
| `test_policy_improvement_base_policy_restore` | 0.3s | 0 | 0 | 0 | 0 |

### CUDA guard

Zero matches for `nvcc | RadixSortPairs | caffe2 | .cu` across **all twelve** invocations of
this round, swept with `buck2 log what-ran --recent 0..11`. The only `cxx_compile` actions in
any build are the three PAR scaffolding sources — `embedded_main.cpp`,
`static_extension_info.cpp`, `static_extension_utils.cpp`. Nothing was waited out; there was
nothing to wait out.

---

## 2. How I proved the imports actually resolve

`-- --help` is not sufficient for two of the three PARs, and it is worth being precise about
why, because it is the difference between "it links" and "its imports resolve".

Both `policy_improvement_full_entrypoint.py` and
`policy_improvement_theory_bridge_entrypoint.py` call `preflight_runtime(...)` **at module
scope, on line 25**, before importing anything else. The behaviour modules are reached later
and dynamically, off the launcher-owned marker. So `--help` executes about twenty lines,
proves `runtime_archive_preflight` imports, and stops. Every module the slice added to these
two PARs — `:policy_improvement_exp1b_runtime`, `:policy_improvement_exp1b_theory_backend`,
`:policy_improvement_exp1b_auditor`, `:policy_improvement_checkpoint_allowlist` — is
downstream of that line and untouched by `--help`. That is exactly where the
resource-versus-dependency mistake would hide.

**What I did instead.** For each PAR I enumerated the complete module closure its `BUCK`
target declares (walking `deps` transitively and collecting every `srcs` entry), then imported
every one of them *inside that PAR's own runfiles tree*: the PAR's bundled native interpreter,
its `LD_LIBRARY_PATH` including the CUDA/cuDNN/TensorRT paths, its
`__python_generated_allocator_preload`, and its link tree on `PYTHONPATH`.

`_bootstrap.sh` cannot be reused directly for this — it hardcodes
`export FB_PAR_MAIN_MODULE="<entrypoint>"`, overwriting any value passed in, so every
invocation lands back on the authenticating entrypoint. The probe runner reproduces the
bootstrap's exports verbatim and sets `FB_PAR_MAIN_MODULE` last, then execs the same
`runtime/bin/*native-main*` binary against the same `__run_lpar_main__.py`. Probe scripts live
in `/tmp/exp1b-parprobe/`; nothing was written into the repository or into `buck-out`.

Results:

```text
policy_improvement_full           probe: 29/29 modules imported
                                  probe: torch 2.15.0a0+fb cuda_available=True
policy_improvement_theory_bridge  probe: 39/39 modules imported
                                  probe: torch 2.15.0a0+fb cuda_available=True
phase4_runtime_launcher           probe: 3/3 modules imported
                                  probe: torch unavailable: ModuleNotFoundError
```

The launcher having no torch is correct, not a gap: its `BUCK` comment states that the
authorization boundary pins the exact executable closure and nothing may run inside it that
the pin does not name. It carries `confirmatory_runtime_launcher`, `phase4_runtime_launcher`,
and `phase4_runtime_profile`, and that is all it should carry.

**I went one step further than imports.** Both runtime PARs gained
`configs/policy_improvement_exp1b/*.json` resource globs in this slice, and an importable
module plus a missing resource is still a broken PAR. A second probe read the three documents
from each PAR's own `FB_PAR_RUNTIME_FILES` and ran the real validators against them:

```text
bundled protocol.json                       sha256=2539a349…0709a0
bundled registry.json                       sha256=0b08942b…22a95b
bundled amendments/reduced_study_exp1b.json sha256=5b160aa8…0ea5f54b
OK  all three exp1b documents validate and cross-bind in the PAR
protocol_sha256=67a149f6c63e858eedaa5dba6599a8030592415c2dd3f52cf90256c88e536d77
```

Identical in both PARs, and the three file digests match the reviewed values. So the code and
the documents work together inside the runfiles tree, which is what the resource globs were
added for.

---

## 3. The `--help` exit-1 on both runtime PARs is correct behaviour

```text
RuntimeError: Authenticated execution requires the verified packaged-runtime launcher.
  runtime_archive_preflight.py:360, from policy_improvement_full_entrypoint.py:25
```

Identical for the theory bridge. This is the entrypoint's own gate refusing an invocation that
did not come through the authenticated launcher — the four-gate contract the review series
spent eight rounds hardening. A PAR that printed usage here would be the defect. I did not
attempt to satisfy the gate: doing so needs the owner-signed admission and a pinned runtime
authorization, which are external prerequisites.

---

## 4. The launcher, driven past `--help`

`--help` exits 0 and lists 13 purposes, including both the slice added:

```text
policy-improvement-exp1b-training
policy-improvement-exp1b-bridge
```

Invoked with each of those and a deliberately absent archive, the launcher reaches its own
authorization validation and refuses with a purpose-specific message:

```text
--purpose policy-improvement-exp1b-training  exit 2
  Phase 4 runtime rejected: Policy-improvement smoke and evidence consumers require
  one complete externally digested runtime authorization.

--purpose policy-improvement-exp1b-bridge    exit 2
  Phase 4 runtime rejected: Policy consumers require a separate complete
  producer-source authorization; other roles cannot accept one.
```

Two different refusals for two different roles means the purpose is registered, routed, and
distinguished inside the PAR — not merely present in an argparse choice list.

The launcher's live `POLICY_IMPROVEMENT_EXP1B_AUDIT_PROFILE_PATHS` (9 paths) contains both
`configs/policy_improvement_v2/populations.json` and
`scripts/policy_improvement_exp1b_auditor.py`, so the profile edit from the previous round is
present in this PAR too, not only in the auditor binary.

---

## 5. `test_policy_improvement_base_policy_restore` re-run

**Pass 22. Fail 0. Skip 0.** Exit 0. Same count as your earlier run, now against the ported
file.

The build before it was a 0.3s no-op with zero actions, which is suspicious on its face for a
target whose shared source had just been replaced — a stale artifact would produce a
meaningless green. It is not stale: the library was already rebuilt earlier in this same
daemon, when the exp1b test target picked up `:policy_improvement_base_policy_restore` as part
of the port's fix B. I confirmed the runfiles copy directly:

```text
<test_policy_improvement_base_policy_restore#link-tree>/scripts/policy_improvement_base_policy_restore.py
  a061afd98b83a1e16b6bdd33cb749d89a4b4ac34e49ae85dfdb087218b38ec7f
repository copy and reviewed handoff value
  a061afd98b83a1e16b6bdd33cb749d89a4b4ac34e49ae85dfdb087218b38ec7f
```

The exp1b test PAR's copy hashes the same.

---

## 6. New defects: none. BUCK edits: none.

No dependency or packaging defect surfaced this round. The four found during the port were the
whole of it, and the two shared targets whose bodies changed —
`policy_improvement_full` and `policy_improvement_theory_bridge` — turn out to have had
complete and correct dependency lists already. That was the specific risk this task existed to
check, and it did not materialize.

**No `BUCK` edit was made.** For Codex's reconciliation: `BUCK` has been unmodified since
2026-09-10 13:23:02 -0700, the timestamp of the last port-round edit, and its SHA-256 is
`c305f63ea5d5d4b0633261091968609acfd72c849db10d1faee8767e4bafeb46`. The only difference from
the reviewed `/home/buiksat/trm_bellman/BUCK` (`9fefc08a…`) remains the four dependency
additions documented in §4 of the port handoff.

---

## 7. The cache metric: what I found

You asked me to say why `Cached actions: 0` disagrees with your cold-daemon 195,791, or say I
couldn't. I have a mechanism and evidence for it, but not proof, so here is both.

**Two separate things were going on.**

**(a) A real reporting trap, now avoided.** `buck2 log summary` with no arguments does not
reliably describe the invocation you just ran. After building
`policy_improvement_theory_bridge` I read a summary reporting `Targets Analyzed: 0` and zero
actions — while the build's own log showed 883MiB downloaded over 87 seconds. The bare command
had shown me a *different, earlier* invocation. Every number in §1 of this handoff was
re-collected with `--trace-id <id>`, the id scraped from each build's own output. **The
port handoff's cache figures were read with the bare command and may be misattributed.** Its
CUDA conclusion is unaffected: I have now swept the last twelve invocations by trace id and by
`--recent`, and every one is zero.

**(b) The zero itself is real.** On these builds `buck2 log what-ran --skip-cache-hits`
removes exactly zero rows (148 → 148 on the theory bridge). If cache hits were being executed
and merely miscounted in the summary, that flag would drop them. It does not. These builds
genuinely had no action-cache hits.

**The mechanism I believe accounts for it, stated as inference.** My builds run against a
daemon that has already analyzed and built this graph in-process. buck2 memoizes unchanged
nodes in its DICE graph, so those actions are not re-executed *and not looked up in the action
cache at all* — they are not actions in that build. `Cached actions` counts cache lookups that
hit, so it reads 0. The clearest instance is
`test_policy_improvement_base_policy_restore`: 0.3s, `Targets Analyzed: 0`, every action
counter 0 — yet the runfiles tree demonstrably contains the freshly ported file, so the work
had been done, just in an earlier build in the same daemon. Your 195,791 came from a cold
daemon, which has no memoized graph and must resolve every action, overwhelmingly via cache
hits.

That explains all three observations coherently. **It is not verified.** I have not read
buck2's source, and I deliberately did not run the control that would settle it — a build
under a fresh `--isolation-dir`, which is a cold daemon and therefore exactly the situation
where a caffe2 source build could start. You told me to stop on that, not to provoke it. If
you want it settled, the safe version is to run the control on a target with no torch in its
closure, such as `fbcode//buiksat_trm:test_policy_improvement_exp1_diagnostics`, and see
whether `Cached actions` becomes nonzero.

**Practical upshot:** on a warm daemon, `Cached actions` is not a useful signal and a low value
is not evidence of a cache miss. Check build *shape* with `buck2 log what-ran --trace-id`, and
always pass the trace id.

---

## 8. Deliberately not done

- **No cold-daemon control run.** Reason above. It is the one experiment that would convert §7
  from inference to fact, and it is the one experiment your constraints rule out on a
  torch-bearing target.
- **No attempt to satisfy the runtime PARs' preflight gate.** It needs owner artifacts.
- **No `BUCK` change.** Nothing required one.
- **No Python source change anywhere.** Nothing exposed a defect in the reviewed Python. Had
  it, I would have stopped and reported rather than edited.
- **The other PARs in the package were not built** — `policy_improvement_audit`,
  `policy_improvement_analysis`, `policy_dataset_builder`, `upi_trm_train`,
  `confirmatory_runtime_launcher`, and the rest. None is part of the slice's changed surface
  and you did not ask for them. `policy_improvement_exp1b_auditor_bin` was covered in the
  previous round.
- **Nothing outside `/data/repos/fbsource/fbcode/buiksat_trm` was written.** `hg status`
  reports zero non-`buiksat_trm` entries and zero `M`/`A`/`R`/`!` entries repository-wide.
  `/home/buiksat/trm_bellman` is unchanged: 22 paths in `git status --porcelain -uall`, HEAD
  still `0f8d029`, and a spot-check of six reviewed files matches the handoff hashes.

---

## 9. What remains before a real experiment run

**Nothing in code or packaging that I can find.** Both gaps I flagged after the port are now
closed: the three untouched PARs are built and exercised, and the stale target is re-run
green. The Buck side of the slice is, as far as dependency-light and Torch-available checks
can establish, complete.

**Still outstanding, all owner artifacts or owner decisions:**

1. The owner-signed Experiment 1B admission with distinct full and theory-bridge
   authorizations, bound to final source, launcher, runtime, and profile hashes. Without it
   neither runtime PAR can get past line 25 of its entrypoint, and the launcher refuses both
   exp1b purposes — as demonstrated in §4.
2. The admitted Experiment 0 checkpoint with its signed base-policy amendment, identity
   document, and source/procedure manifests, plus owner acceptance of the actor-only
   lagged-head provenance.
3. The registered 1,024-record train dataset and the exact dataset, split, ordering, and
   validation-bridge population manifests.
4. Distinct clean producer and evaluator checkout roots at the admitted commit, and an
   owner-controlled fresh evidence generation on a filesystem with the required `flock`,
   hard-link, rename, and directory-`fsync` semantics.
5. `RUN_UPITRM_FULL_EXPERIMENTS=1`, and validation-bridge authorization only after all eight
   checkpoints are sealed.
6. The remaining owner decision on whether target lag stays record-only or receives a
   registered refusal tolerance.

**Two process items, not blockers but worth doing before the run:**

- The four-line `BUCK` dependency delta lives only in fbsource. It should go through a review
  round against the standalone checkout and be re-synced, so the two trees agree and the
  reviewed artifact is the one that actually builds.
- Three Torch tests still skip on inner guards needing a trained checkpoint
  (`test_restore_reproduces_every_module_digest`,
  `test_base_and_deployed_differ_when_the_candidate_has_moved`,
  `test_a_mutated_trainer_estimator_is_caught`). The last of those is the only thing that will
  exercise the 1e-6 trainer-estimator centering tolerance against real float32 heads. If that
  tolerance does not hold, every seed refuses — correct behaviour, owner decision, and it will
  surface during the first real run rather than now.
