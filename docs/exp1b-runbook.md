# Experiment 1B runbook

Written 2026-09-13 after three attempts on a machine that was retired at 17:00 UTC. Stage A
completed twice and Stage B never ran. Everything here is measured on this repository, not
inferred.

## The one thing that decides whether the next attempt works

**Stage B must be proven by execution before Stage A is spent.**

Stage A's eighth seed writes `provenance.json`, which freezes the producer attestation:

```text
runtime_sha256                the policy_improvement_full PAR
runtime_authorization_sha256  the minted authorization
launcher_sha256               the launcher PAR
source_git_commit             the admitted commit
admission_sha256              the signed admission
```

`attestation_matches_authorization` (`scripts/policy_improvement_exp1b_schema.py:1866`) compares
**all five**. Any change to any of the five runtime PARs forces a re-mint, which changes the
authorization digest, which the frozen provenance no longer matches. There is no variant that
preserves the seals: keeping an old PAR fails the minter's
`assert_phase4_archive_matches_profile` against the fixed checkout.

So a Stage B defect found after Stage A costs a **full Stage A re-run**. That happened twice
here, at 12.4 h and 7.8 h. Both Stage B defects were one-line mistakes.

Corollary: **do not start Stage A until a Stage B bridge has actually served a seed.** If you
cannot arrange that, budget for one full Stage A re-run per Stage B defect and assume there is at
least one.

## Ordered procedure

1. **Build the launcher under `@fbcode//mode/opt`.** Not dev-nosan.
   `_launcher_executable_manifest` requires `fbmake.build_mode == "opt"` and refuses a dev
   launcher before the pinned digests are consulted. Every other target uses
   `@fbcode//mode/dev-nosan`.
2. **Build the five runtime PARs** under `@fbcode//mode/dev-nosan`: `upi_trm_train`,
   `policy_improvement_full`, `policy_improvement_theory_bridge`, `policy_improvement_audit`,
   `policy_improvement_analysis`. All six artifacts must be ZIPs; the minter and the launcher
   both open them with `ZipFile`.
3. **Mint the runtime authorization** against a clean checkout at the exact commit.
   `--training-runtime` is **`upi_trm_train`**, not `run_phase4_training` — the latter is the
   Phase 4 2x2 orchestrator whose own docstring takes `--training-runtime upi_trm_train.par` as
   an argument.
4. **Sign the base-policy amendment.** Fourteen artifact fields, not the ten the protocol lists.
   `initialization_kind` is enum-constrained to `{train_only_pretrained, random_base_stress}`;
   the producer's own value is the accepted one. There is no `training_data_sha256` field —
   it carries `training_split_ordered_record_sha256` and `training_dataset_manifest_sha256`,
   which the producer emits under exactly those names.
5. **Sign the execution admission** with distinct full and theory-bridge runtime digests.
6. **Create the evaluator checkout** at the admitted commit, at a different path.
   `phase4_runtime_launcher.py:1540` refuses `producer_source_root == source_root` for the bridge
   purpose. It also needs the dataset copied in: `data/` is gitignored, and
   `scripts/policy_improvement_exp1b_runtime.py:1029` resolves Stage B's `dataset_root` under the
   producer root.
7. **Run Stage A**, one process per GPU. See timings below.
8. **Run Stage B**, one bridge per seed position. The eighth publishes.
9. **Audit** with `--parent-populations`.

## Timings, measured

| arrangement | per seed | 8 seeds |
| --- | --- | --- |
| 1 process, GPU otherwise idle | 1 h 45 m – 1 h 57 m | — |
| **1 process per GPU, 2 GPUs** | **~1 h 50 m** | **~7 h 20 m** |
| 3 processes on one GPU | 9 h 17 m | — |
| 4 processes on one GPU | 12 h 23 m | — |

**Do not stack processes on a GPU.** Three-wide was 1.6x *worse* than running the same three
sequentially. Aggregate throughput is capped near 3.8 train-steps/min however the work is
arranged, so concurrency beyond one process per card buys nothing and costs a lot.

**CPU% is not a progress metric.** CUDA spin-waits, so a process blocked on a contended GPU still
shows ~98% of a core. Measure progress by `train_step` in the log, against a solo baseline.

Stage B is unmeasured. It reaches the route in ~80 s; the census traversal is ~24,800 unbatched
model calls per seed (128 members x 97 actions x 2 depths).

## The four defects, all found at runtime

| # | where | what | cost |
| --- | --- | --- | --- |
| 1 | `run_exp1b_training` | `self._module`; the attribute is `self._training_module` | 1 h 50 m |
| 2 | `bridge_main` | `evaluation_population["split_manifest_sha256"]`; that key is on `training_population` | 12.4 h |
| 3 | opener | `build_dataset_from_paths(pool_size=None)`; declared `int` | 7.8 h |
| 4 | — | (launcher build mode and the training-runtime mix-up, found before any GPU spend) | — |

All three code defects shared one shape: **the only test covering the path substituted the object
under test.** A replacement class, a hand-passed keyword, a stubbed training module. Each looked
covered and executed nothing.

Three static sweeps now guard the classes, and all three are clean on this tree:

- missing `self.<attr>` reads (`test_the_sealed_backend_reads_only_attributes_it_assigns`)
- missing registered-document keys (`test_bridge_main_only_reads_keys_the_protocol_registers`)
- `None` passed where an `int` is declared (`Exp1bStageBArgumentTypeSweepTest`)

They are necessary, not sufficient. They would have caught defects 1, 2 and 3 respectively, and
they cannot catch a logic error.

## Operational traps

- **Never commit while a run is in flight.** Stage A and Stage B both pin
  `--expected-source-git-commit` against a *clean* checkout. Commit artifacts from a linked
  `git worktree`, which leaves the primary checkout's HEAD and working tree untouched. Verify
  before and after.
- **`setsid nohup ... &` + `disown`** for long runs. The harness's background wrapper killed a
  2 h run at 90 s, and a foreground `timeout` killed only the shell — its launcher grandchild
  survived and silently ran a second copy of the same seed.
- **Owner-signed artifacts must live outside the checkout** (`_outside_producer_root`). So must
  the base-policy artifact, which the producer writes inside `data/`.
- **Stage A must not be given `--producer-source-project-root`.** Only audit, analysis,
  theory-bridge and exp1b-bridge take one.
- **Runtime args need a literal `--` separator** before the launcher's `REMAINDER`.

## State at handover

Tree `090a95a` on `full-implementation`. All fixes and tests committed; whole target green at
333 passed, 0 failed, 4 skipped.

Run 3's eight sealed checkpoints exist at
`/home/buiksat/upi-trm-owner-20260911/evidence3/gen-20260913/` and are **unusable**: their
provenance pins the pre-fix full PAR. Their run manifests are committed as the record of what was
produced. The base policy, the regenerated corpus, and all signed documents are committed or
recorded under `artifacts/exp1b-20260913/`.

**No Experiment 1B result was produced. No paired signed gap, no interval, no auditor verdict.**
