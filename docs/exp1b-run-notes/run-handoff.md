# UPI-TRM Experiment 1B — run preparation

Date: 2026-09-11. Repo `/home/buiksat/trm_bellman` at `dc4fd9e` plus one uncommitted `BUCK` edit.

## Status: blocked on one commit, and the step order in the request is inverted

I did not get to steps 3 or 4. Two findings changed the plan, both verified in source.

### Finding A — five more targets need standalone packaging

Both `validate_runtime_archive` (`confirmatory_runtime_launcher.py:589`) **and** the runtime
authorization minter (`scripts/policy_improvement_runtime_authorization.py:481`) open a runtime
artifact with `ZipFile` and refuse anything else. Only `phase4_runtime_launcher` was converted
to standalone in commit `6212b16`. Current state:

```text
phase4_runtime_launcher            standalone   ✓
run_phase4_training                inplace      ✗
policy_improvement_full            inplace      ✗
policy_improvement_theory_bridge   inplace      ✗
policy_improvement_audit           inplace      ✗
policy_improvement_analysis        inplace      ✗
```

All five are required: the minter takes `--training-runtime --full-runtime --theory-runtime
--audit-runtime --analysis-runtime` and hashes each as a ZIP.

**`BUCK` is edited and mirrored to fbsource, uncommitted**, adding `package_style =
"standalone"` to those five. Packaging attribute only — no source, resource, or dependency
change, no compile-mode effect. `BUCK` is now
`fc566772bc4e1f1317a4094786e3cc2b0d990bbaaa0263748ff2ec15a47ccc1a`; `git status` shows only
`M BUCK`. **This needs your commit before anything else can proceed**, because both the minter
and the launcher require a clean checkout at the expected commit.

### Finding B — the authorization must be minted before the amendment, not after

Your steps 1 and 2 are the other way round. `validate_base_policy_amendment`
(`scripts/policy_improvement_v2_schema.py:1236-1288`) requires the amendment to carry
`runtime_authorization_sha256` as one of its seventeen exact fields. So the base-policy
amendment cannot be signed until the authorization exists.

Corrected order:

1. Commit the `BUCK` change.
2. Build the six PARs; collect their real SHA-256 values.
3. Mint the runtime authorization at the new HEAD, binding launcher + five runtimes, with
   **distinct** producer and evaluator `runtime_sha256` (`policy_improvement_full` vs
   `policy_improvement_theory_bridge`, which are genuinely distinct artifacts — required by
   `policy_improvement_exp1b_auditor.py:593`).
4. Sign the base-policy amendment, now able to cite the authorization digest.
5. Sign the exp1b admission, citing the base-policy amendment digest and both role
   authorizations.
6. Run Stage A seed 0, measure, then decide.
7. Stage B, publish, audit.

Neither the amendment nor the admission needs committing — both are read by path, and the
minter writes to an `--owner-directory`. Only the `BUCK` change does.

## The base-policy artifact being registered

Produced 2026-09-11, `data/base-policy-exp0-20260911/`, 51 min 42 s on cuda, final imitation
accuracy 98.71% (best 98.84%; the saved checkpoint is the epoch-40 final-budget one).

| identity field | value |
| --- | --- |
| `initialization_kind` | `train_only_pretrained_base_policy` *(per your decision; producer wrote `train_only_pretrained`)* |
| `architecture_sha256` | `588866a66a831e0a8a60711df1dc316cc5ea733e2b8674c147ea832d88691f3e` |
| `model_state_sha256` | `0d18f790d95e4c10269e73a333dd6de9e90dac1f558278cea5709b77190e77c4` |
| `producer_git_commit` | `dc4fd9ecbe42f2b94d82d027c1c8141a6e82f3a2` |
| `producer_source_manifest_sha256` | `9ba92a2c179c91b1a0300a5a3a0c38c39111b17d3414d98581e79ef624918097` |
| `training_data_sha256` | `73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446` *(assembled per your decision)* |
| `training_procedure_sha256` | `2fc00268c380f2a471b64f8315b380ae787cdd18986bfd61d3e295ba126440b3` |
| `checkpoint_sha256` | `607c3065aa429522c90b89763d215be49f3ee58bffd493ed9280c09f4d377fe7` |
| `shared_across_persistent_and_episodic` | `true` |
| `not_selected_by_validation_or_test` | `true` |

`checkpoint_size_bytes` 6,669,709. Fields 9 and 10 are true as observed facts: the producer
recorded `validation_data_opened: false` and `test_data_opened: false`, and the frozen procedure
sets `rl_environment_interactions: 0`, `early_stop_accuracy: null`,
`checkpoint_selection: final_budget_only`.

**Recorded as instructed:** `scripts/policy_improvement_base_policy.py` does not emit
`training_data_sha256` (`grep -c` = 0). The value above is the train split ordered-record
digest, exactly what the consumer maps at
`scripts/policy_improvement_exp1b_runtime.py:580`. Registering it in the amendment is correct;
**fixing the producer to emit the key remains a real gap for a later round.** Second
reconciliation item: the producer's `initialization_kind` string differs from the exp1b
provenance form used here.

## Wall-clock: the projection you asked for

I cannot measure seed 0 yet. What is known:

- Per seed: 10,000 terminal environment interactions of Stage A training, then a Stage B
  traversal of the 128-record validation_bridge census at two depths (n=2, m=8).
- The base policy did roughly 41k imitation forward/backward passes in 62 min on the same GPU,
  but an environment interaction is heavier than an imitation batch step — it is a model rollout
  plus an environment transition, not a batched update.
- Eight seeds run sequentially; the exact-once claim lease is per seed, and nothing in the
  design parallelises them.

Honest range: **anywhere from ~2 hours to well over 10**, and I will not narrow it by guessing.
This is exactly why measuring seed 0 first is right. When we get there I will report seed 0's
wall clock and a projection before starting seeds 1–7, and stop if it implies more than about
three hours total.

## CUDA guard

No genuine compile action in any command this session. The `cquery` package-style probes ran
analysis only. Every earlier build was checked with `what-ran --trace-id --skip-cache-hits`.

## What remains

1. **Commit `BUCK`** (`fc566772…`). Everything else is blocked on it.
2. Build six PARs; confirm all are ZIPs and CUDA-clean.
3. Mint authorization → sign base-policy amendment → sign exp1b admission.
4. Stage A seed 0, measure, report projection.
5. Seeds 1–7, Stage B, publish, audit with `--parent-populations`.

Deferred and unchanged: the producer `training_data_sha256` gap; the
`initialization_kind` string mismatch; no hermetic regression for the committed producer source
manifest; the three placeholder Torch tests.


---

# LIVE LOG — appended as each step completes

Repo at **`73c8780d70d9063fd88b5e8545515998c8d58a9f`**, clean.

## Step 1 DONE — six PARs built

Trace `9573c9df-9dfb-4479-b3ec-8e0c552994be`, 2:59.4, 51 local / 1356 remote / 1901 cached.
All six confirmed ZIP.

| target | sha256 |
| --- | --- |
| `phase4_runtime_launcher` | `43656ae2f49b8e5f133f4f51009eafbdeb2c48b7b2b2797e17b5f671a6b7ca1d` |
| `run_phase4_training` | `c7df89e51c0881fd4757f7bf09809e28797f96da2c9a6e457c573e871526e30c` |
| `policy_improvement_full` | `0d27c3a9b5db1fb384bcfdcdf3959153b020967756d8bd61f5f29875eb6a0376` |
| `policy_improvement_theory_bridge` | `57959fa8b5bb384236b51e5395d7082296a71361d7f38d82629e66db57f2ace2` |
| `policy_improvement_audit` | `51c9a56a5e4795b89b8b7c20f192037f9b0dbd08cdc0b4b16a2fba287d40cd2f` |
| `policy_improvement_analysis` | `d58bfcfa85be397d4eec6be00354405323de18012741924ac68b54093728ed96` |

Producer runtime = `policy_improvement_full` `0d27c3a9…`; evaluator runtime =
`policy_improvement_theory_bridge` `57959fa8…`. Distinct, as
`policy_improvement_exp1b_auditor.py:593` requires.

**CUDA guard:** 47 rows matched the pattern with cache hits excluded, and I checked rather than
aborted — **all 47 are `py_compile` actions on Torch's Python files** (`caffe2` appears as a
path component), 45 remote and 2 local. Zero `nvcc`, zero `RadixSortPairs`, zero `.cu`. No CUDA
compilation.

Paths recorded in `/tmp/par_hashes.json`.


## Step 1 REDONE — one file was unsynced

The first authorization attempt refused, **verbatim**:

```text
RuntimeAuthorizationGenerationError: Launcher PAR source profile differs from the clean checkout.
```

Real cause, not an obstacle: `configs/iclr_confirmatory/producer_source_manifest.json` — the
file regenerated and committed as `dc4fd9e` — was **never synced to fbsource**, so the launcher
PAR was built from the stale copy. Exactly one of the 171 profile paths across all five runtime
profiles differed; the other 170 matched. Synced and rebuilt.

Rebuild trace `6e67b155-0c7b-4dde-b3d7-c0c9815f46ac`, 0 non-`py_compile` CUDA rows with cache
hits excluded. Final PAR hashes:

| target | sha256 |
| --- | --- |
| `phase4_runtime_launcher` | `39b674a93076078d92ff313ca3321fd50ada9bdce422ecc90ace46f4f50f014e` |
| `run_phase4_training` | `c7df89e51c0881fd4757f7bf09809e28797f96da2c9a6e457c573e871526e30c` |
| `policy_improvement_full` | `024a45772dd4c97ee69efa701079e2e7ccc4ea4abff9a6a883d636e8ec9533d9` |
| `policy_improvement_theory_bridge` | `fb59ee48ed36a1c2cd20283d2bcbc906128b972ed59b50e2dd8da53cfc1c8458` |
| `policy_improvement_audit` | `c83911153739fb259c1388f35ff4b2408b0d0c7200846e6f86541b99c39b53a1` |
| `policy_improvement_analysis` | `d66e9c01f37e5940f796c0c95d39e89532754a88653d6b63f58d71262e67d9a1` |

## Step 2 BLOCKED — the launcher PAR internals are pinned to constants

Second attempt, refused **verbatim**:

```text
RuntimeAuthorizationGenerationError: Launcher PAR native runtime support differs.
```

Source-profile authentication now passes. The failure is one layer deeper, at
`_validate_launcher_native_support` (`scripts/policy_improvement_runtime_authorization.py:630-649`),
which hashes two members of the launcher PAR:

```text
runtime/bin/phase4_runtime_launcher#native-main#platform-runtime#python#py_version_3_12
runtime/lib/__python_generated_allocator_preload
```

and compares their folded digest against the hard-coded
`_LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256 = "974e9a79…"` at line 161. It is one of a family of
constants at lines 150-165 pinning launcher PAR internals: `_LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256`,
`_LAUNCHER_ARCHIVE_PREFIX_SIZE = 8215`, `_LAUNCHER_ARCHIVE_PREFIX_SHA256`,
`_LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256`, `_LAUNCHER_STARTUP_LOADER_NORMALIZED_SHA256`.

**These are build-environment artifact bytes**, recorded whenever the launcher PAR was last
built. Our launcher is built today on a different toolchain and with `package_style =
"standalone"` newly added, so it cannot match. Same structural class as the dataset
`manifest_sha256` finding: source pinning artifact bytes that are not reproducible from source.

Updating those constants is a production source change to a reviewed file. I am not authorized
to make it, and unlike the dataset case there is no additive path — there is one launcher and
the authorization minter pins it. **This needs an owner decision.**

## RESUME POINT for a fresh session

State: repo `73c8780d70d9063fd88b5e8545515998c8d58a9f`, clean, nothing to commit. fbsource
`buiksat_trm` now byte-identical to the standalone repo for all five runtime profiles (171
paths) — the `producer_source_manifest.json` sync is done but is a **working-tree change in
fbsource only**, not committed anywhere.

Done: six ZIP PARs built (hashes above, paths in `/tmp/par_hashes.json`). Owner directory
created at `/home/buiksat/upi-trm-owner-20260911` (0700, empty).

Not done: authorization, base-policy amendment, admission, all eight seeds, Stage B, audit.
**No evidence generation was created and no seed was claimed** — nothing is in a half-claimed
state, and there is nothing to recover.

The base-policy artifact and its ten identity fields are in the section above and are
unaffected.

---

# SESSION 2 — 2026-09-11, second implementer

## Step 2a DONE — launcher pins re-measured. The blocker was the build mode, not the pins.

Coordinator authorized re-pinning all five constants in
`scripts/policy_improvement_runtime_authorization.py`. Measured against a freshly built
launcher, **only two of the five were actually stale.** The real blocker was different.

### The build mode is the blocker

`_launcher_executable_manifest` (line ~710) hard-requires `fbmake["build_mode"] == "opt"`.
The launcher built in session 1 under `@fbcode//mode/dev-nosan` reports
`build_mode: "dev-nosan"`, so it is refused **regardless of what the five pins say**.
`build_mode` is not one of the five constants and was not touched. The launcher must be
built with `@fbcode//mode/opt`, which is also what the 2026-08-24 rebind comment in the
file documents. Deviation from the standing dev-nosan rule, launcher target only; the five
runtime PARs stay dev-nosan (their validators do not pin build mode, and an opt-config
Torch build would be a genuine source build).

### Measured values, both launchers vs. checked-in

| constant | checked-in (2026-08-24) | dev-nosan build | **opt build** | changed? |
| --- | --- | --- | --- | --- |
| `_LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256` | `2bca9c98…` | `e5199614…` | `e5199614…` | **YES** |
| `_LAUNCHER_ARCHIVE_PREFIX_SIZE` | 8215 | 8215 | 8215 | no |
| `_LAUNCHER_ARCHIVE_PREFIX_SHA256` | `87e71b36…` | `87e71b36…` | `87e71b36…` | no |
| `_LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256` | `974e9a79…` | `d38981a9…` | `abb8f988…` | **YES** |
| `_LAUNCHER_STARTUP_LOADER_NORMALIZED_SHA256` | `a0e47cb9…` | `a0e47cb9…` | `a0e47cb9…` | no |

Pinned-support fold is mode-independent (dev == opt): pure upstream fbcode source drift.
Native-support fold is mode-dependent: the two native members are relinked per mode.
Member **inventories** did not move — 44 pinned support members, 2 native members, same set.

### Edit made (UNCOMMITTED, coordinator commits)

`scripts/policy_improvement_runtime_authorization.py` only, +25 / -3:
- `_LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256` → `e51996140e698c24a7d66801b415a0f81a526634f6c168f0976f380000b5265e`
- `_LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256` → `abb8f9887e32087cf11c4d9c188a96bd41ccab678f3e8262708c9dd807d3b04a`
- Header comment: 2026-09-11 rebind record. Nothing else changed. Synced to fbsource.

### Tests referencing the constants

- `tests/test_policy_improvement_launcher_identity_unittest.py:165` — **dynamic**. Builds the
  real launcher as a Buck resource and compares measured values against the constants. No
  update needed; it is the regression that proves the new pins.
- `tests/test_policy_improvement_runtime_authorization_unittest.py:615` — **dynamic**. Inside a
  `mock.patch.object` block that replaces all five constants with digests computed from a
  synthetic archive. No update needed.

`buck2 test @fbcode//mode/opt :test_policy_improvement_launcher_identity
:test_policy_improvement_runtime_authorization` → **23 pass, 0 fail, 0 skip.**

### New latent defect found (reported, not fixed)

`LauncherIdentityTest._require_reviewed_configuration` (line 109) intends to `skipTest` on a
non-opt launcher, but calls `authorization._launcher_executable_manifest(archive)` first,
which raises before the skip. Under `@fbcode//mode/dev-nosan` the target reports **11 errors**
("Launcher PAR does not execute the authenticated launcher module") instead of skipping.
Pre-existing: the raise happens before any pinned digest is read, so the old constants fail
identically. Not fixed — outside the authorized edit.

### Launcher PAR now in play

```
opt   f99099ee418647ee970f84c2c6ba6d373cd143676d37498d0124e80b48c289bd  809958 B
      /data/repos/fbsource/buck-out/v2/art/fbcode/bc711a88eedb76ac/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par
```
Absolute, canonical, non-symlink, regular file — passes `_authenticate_artifact`'s path rules.
Supersedes the dev-nosan `39b674a9…`; both recorded in `/tmp/par_hashes.json`.
CUDA guard on the opt build (trace `edcdd488-adb4-4263-8b92-a1cf101940e1`): 4 rows with cache
hits excluded, **0 matching `nvcc|caffe2|cub|.cu`**.

### The commit will not change any PAR hash

`scripts/policy_improvement_runtime_authorization.py` is in **no** runtime source profile
(checked all nine `*_PROFILE_PATHS` at runtime) and is **not** in
`configs/iclr_confirmatory/producer_source_manifest.json` (`grep -c` = 0). So committing the
re-pin changes only the commit the minter binds via `--expected-git-commit`. The five
dev-nosan runtime PAR hashes and the opt launcher hash all stay valid.

## BLOCKED — waiting on the coordinator's commit

The minter requires a clean checkout at `--expected-git-commit`. The tree is dirty with the
re-pin. Next implementer: get the post-commit HEAD from the coordinator, then mint.

## Step 3 BLOCKED — the wrong PAR was bound as --training-runtime

Minted at `f2af25aabd5b1850bb00debee35793674d79a42e`. Refused **verbatim**:

```text
confirmatory_runtime_launcher.ConfirmatoryRuntimeError: Runtime archive is missing a readable source manifest.
```

Raised from `_training_archive_validator` → `validate_confirmatory_archive_sources` →
`_load_embedded_manifest`, on the artifact passed as `--training-runtime`.

**Real defect, not an obstacle.** Session 1 bound `run_phase4_training.par`. That target is the
Phase 4 2x2-norm-ablation **orchestrator**; its own docstring (`scripts/run_phase4_training.py:22`)
reads `--training-runtime /absolute/path/to/upi_trm_train.par`. It takes the training runtime as
an argument; it is not the training runtime. It carries zero `configs/` members.

The Phase 4 confirmatory training runtime is **`upi_trm_train`**: it globs
`configs/iclr_confirmatory/*` (so it embeds the producer source manifest) and its BUCK comments
already discuss producer source authentication. Session 1 added `package_style = "standalone"`
to five targets and missed this one, so it still built as a 1,154-byte inplace bootstrap PAR.

### BUCK edit made (UNCOMMITTED, coordinator commits)

`package_style = "standalone"` on `upi_trm_train`, plus a comment matching the other five and
naming the `:run_phase4_training` confusion. **Contained:** `BUCK` is in neither
`configs/iclr_confirmatory/producer_source_manifest.json` (`"BUCK" in sources` → False) nor any
of the nine `*_PROFILE_PATHS`, so no digest anywhere moves.

### Verified before asking for the commit

Rebuilt `upi_trm_train` under `@fbcode//mode/dev-nosan`:

```text
sha256 45488e959fc713d56efb074dfc8e9fc424d17099ad42746ced3128ff8b6f6716
size   1,462,770,118 B   members 31,193   ZIP: yes
validate_archive_layout                : OK
validate_confirmatory_archive_sources  : OK
embedded manifest == committed checkout bytes : True
```

Path: `/data/repos/fbsource/buck-out/v2/art/fbcode/40e399fcf13b0098/buiksat_trm/__upi_trm_train__/upi_trm_train.par`

**The bound set is now launcher + upi_trm_train + full + theory_bridge + audit + analysis.**
`run_phase4_training` drops out of the authorization entirely.

CUDA guard, all three builds this session, `what-ran --trace-id --skip-cache-hits`:
minter `88a497e2…` 1 row / 0 CUDA; `upi_trm_train` inplace `cb56c2c8…` 121 rows / 0 CUDA;
`upi_trm_train` standalone `17a75670…` 9 rows / 0 CUDA.

## Step 4 FINDING — one decided value is rejected by the schema; the other field does not exist

Read before signing, not after.

`validate_base_policy_amendment` (`scripts/policy_improvement_v2_schema.py:1236-1352`) does not
take the producer's ten-field identity table. Its `base_policy_artifact` is **14 exact fields**,
and two of the coordinator's instructions do not land where expected:

1. **`initialization_kind` is choice-constrained** to
   `{"train_only_pretrained", "random_base_stress"}` (line 1328).
   `train_only_pretrained_base_policy` **is rejected.** The producer's own value,
   `train_only_pretrained`, is the accepted one.
2. **There is no `training_data_sha256` field** in the amendment. It carries
   `training_split_ordered_record_sha256` and `training_dataset_manifest_sha256` — the exact
   names the producer already emits.

Both suffixed/renamed forms are **hardcoded literals in the runtime**, not fields anyone signs:
`"initialization_kind": "train_only_pretrained_base_policy"` at
`scripts/policy_improvement_exp1b_runtime.py:577`, and the `training_data_sha256` key at line 580
populated from `getattr(base_policy, "training_split_ordered_record_sha256")`. The exp1b
admission has no `initialization_kind` field either.

**So the "producer omits training_data_sha256" gap does not touch either signed document.**
`73110263…` is used, under its real producer name, as a producer output. Nothing is assembled.

### Amendment artifact block — every value is a direct producer output

| # | amendment field | value | source |
| --- | --- | --- | --- |
| 1 | `status` | `available` | amendment literal (producer says `complete`) |
| 2 | `initialization_kind` | `train_only_pretrained` | producer |
| 3 | `architecture_sha256` | `588866a66a831e0a8a60711df1dc316cc5ea733e2b8674c147ea832d88691f3e` | producer |
| 4 | `model_state_sha256` | `0d18f790d95e4c10269e73a333dd6de9e90dac1f558278cea5709b77190e77c4` | producer |
| 5 | `producer_git_commit` | `dc4fd9ecbe42f2b94d82d027c1c8141a6e82f3a2` | producer |
| 6 | `producer_source_manifest_sha256` | `9ba92a2c179c91b1a0300a5a3a0c38c39111b17d3414d98581e79ef624918097` | producer |
| 7 | `training_dataset_manifest_sha256` | `55bf615ebeaf0f13bad1917dc74fd33f7c9736b7316a0ff15572641cb6470b2a` | producer |
| 8 | `training_split` | `train` | producer |
| 9 | `training_split_ordered_record_sha256` | `73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446` | producer |
| 10 | `training_procedure_sha256` | `2fc00268c380f2a471b64f8315b380ae787cdd18986bfd61d3e295ba126440b3` | producer |
| 11 | `checkpoint_sha256` | `607c3065aa429522c90b89763d215be49f3ee58bffd493ed9280c09f4d377fe7` | producer |
| 12 | `checkpoint_size_bytes` | `6669709` | producer |
| 13 | `shared_across_persistent_and_episodic` | `true` | producer |
| 14 | `not_selected_by_validation_or_test` | `true` | producer |

Amendment top level also needs `protocol_sha256`, `population_registry_sha256`,
`source_registry_sha256`, `prior_amendment_history_sha256`, `runtime_authorization_sha256`
(from the mint), `validation_data_inspected: false`, `test_data_opened: false`.

### Run shape confirmed — nothing else needs committing

`scripts/policy_improvement_exp1b_runtime.py` takes `--reduced-study-amendment`,
`--execution-admission`, `--base-policy-amendment` **by path** (lines 319-330 Stage A,
856-868 Stage B). Stage B takes `--parent-populations`. Confirms the session-1 claim.

## BLOCKED — waiting on the coordinator's BUCK commit


---

## SIGNED DOCUMENTS — all three, at commit ef24fa10554490189caea7f63646400b6e26a2ad

Owner directory `/home/buiksat/upi-trm-owner-20260911` (0700). All files 0600.

### 1. Runtime authorization — `exp1b_runtime_authorization.json`

```text
runtime_authorization_sha256  ec598b225d435df0f196c6307c675c5043b06888a061e540ff5e64ba99b43d01
authorization_id              upi-trm-exp1b-runtime-authorization-20260911
created_at_utc                2026-09-11T21:45:13Z
producer_git_commit           ef24fa10554490189caea7f63646400b6e26a2ad
launcher_sha256               f99099ee418647ee970f84c2c6ba6d373cd143676d37498d0124e80b48c289bd
```

Six roles. Producer/evaluator runtimes are distinct, as `policy_improvement_exp1b_auditor.py:593`
requires:

| role | runtime PAR | runtime_sha256 |
| --- | --- | --- |
| `policy-improvement-training` | `upi_trm_train` | `45488e959fc713d56efb074dfc8e9fc424d17099ad42746ced3128ff8b6f6716` |
| `policy-improvement-evaluation` | `upi_trm_train` | `45488e95…` (same by design, minter maps both) |
| `policy-improvement-audit` | `policy_improvement_audit` | `c83911153739fb259c1388f35ff4b2408b0d0c7200846e6f86541b99c39b53a1` |
| `policy-improvement-analysis` | `policy_improvement_analysis` | `d66e9c01f37e5940f796c0c95d39e89532754a88653d6b63f58d71262e67d9a1` |
| **`policy-improvement-full`** | `policy_improvement_full` | **`024a45772dd4c97ee69efa701079e2e7ccc4ea4abff9a6a883d636e8ec9533d9`** |
| **`policy-improvement-theory-bridge`** | `policy_improvement_theory_bridge` | **`fb59ee48ed36a1c2cd20283d2bcbc906128b972ed59b50e2dd8da53cfc1c8458`** |

### 2. Base-policy amendment — `base_policy_amendment_exp0_20260911.json`

```text
canonical sha256  f1325aeedee3818cd8ed9c7dea7f17993f918f5972fe0d2eaa76c03762b2a7eb
amendment_id      base-policy-exp0-20260911
```

Bound: protocol `9df68e4ec8c732948dda9d8497fd1764d29cc0e04f9924a94e5f1bf4375a196a`,
population registry `43340c09531197d4de6e19acf3dc6b4e7ffe37c19dca762a212e28e9996afc39`,
source registry `5770404b81749807c8bf0acbbc97b364073b19a01aee760e07f1a046da4e4397`,
prior amendment history `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
(= `sha256(b"[]")`; the v2 protocol's `amendments` list is genuinely empty),
runtime authorization `ec598b22…`. `validation_data_inspected: false`,
`test_data_opened: false`.

`initialization_kind` is **`train_only_pretrained`**, the measured producer value. The coordinator's
earlier `train_only_pretrained_base_policy` is rejected twice over: by the schema enum at
`policy_improvement_v2_schema.py:1328` and by `BASE_POLICY_INITIALIZATION_KIND` at
`policy_improvement_base_policy_restore.py:39`. Correction accepted by the coordinator.

**All 13 non-`status` artifact values were asserted byte-equal to `base_policy_identity.json`
before signing.** Nothing was assembled, derived, or renamed. `status` is `available` because the
amendment schema requires that literal; the producer writes `complete`.

### 3. Execution admission — `exp1b_execution_admission_20260911.json`

```text
admission_sha256  5f08bf736ea02d15ace0f30e945bae93200031e99ede186d4a003120164114fe
amendment_id      exp1b-execution-admission-20260911
```

Bound: exp1b protocol `add935bac37aa23578cd0586aa8bbed769172b65380fae6d6ccb7af52371a063`,
exp1b registry `a4441a1758dbc61baed4d87263e593019ae415f4ecab586c549feef643457116`,
prior amendment `0f79b1b8ff1098ce4d48cfa196f2a69220f5dfe01f7ac66437c2cd3c93a747a2`,
base-policy amendment `f1325aee…`. `execution_allowed: true`, gate
`RUN_UPITRM_FULL_EXPERIMENTS=1`, slots `["base_policy_artifact", "runtime_authorization"]`.

Distinct per-role authorizations, both validated through `validate_exp1b_admission`:
`policy-improvement-full` → `024a4577…`, `policy-improvement-theory-bridge` → `fb59ee48…`.

## CORRECTION to the Experiment 0 record — do not propagate 98.71%

The earlier handoff prose says "final imitation accuracy 98.71% (best 98.84%)". The artifact
disagrees: `base_policy_identity.json` records
`final_imitation_accuracy = 0.988388360380526`, i.e. **98.84%**. The prose figure is wrong.
`98.71%` must not reach the paper. This touches no signed field.

Field 10 (`not_selected_by_validation_or_test: true`) is truthful regardless of which epoch was
saved: the producer recorded `validation_data_opened: false` and `test_data_opened: false` as
observed facts of the run, and `checkpoint_selection` is `final_budget_only`.

---

## RUN SETUP (done) and SEED 0 (in flight)

Owner-side layout, all owner-only:

```text
/home/buiksat/upi-trm-owner-20260911/
  exp1b_runtime_authorization.json            0600
  base_policy_amendment_exp0_20260911.json    0600
  exp1b_execution_admission_20260911.json     0600
  base-policy-artifact/base_policy.pt         0400  sha256 607c3065... (byte-identical copy)
  evidence/                                   0700
    exp1b-20260911/{payloads,checkpoints}/    0700
```

**The base-policy artifact had to be staged outside the checkout.** `_outside_producer_root`
(`scripts/policy_improvement_exp1b_runtime.py:152`) refuses `--base-policy-artifact` and
`--base-policy-amendment` if they resolve under the producer root, and the producer copy lives at
`data/base-policy-exp0-20260911/`. Copied, not moved; digest verified identical
(`607c3065aa429522c90b89763d215be49f3ee58bffd493ed9280c09f4d377fe7`).

### Registry row for seed position 0

`exp1b-fixed-base-exact-persistent-seed2081976412`, seed `2081976412`,
`method_id fixed_base_exact_persistent`, n=2, K=1, alpha=0.1, 10,000 terminal environment
interactions, train → validation_bridge. Matches the coordinator's spec exactly; protocol also
confirms `reference_depth_m 8`, `gamma 0.99`.

### Two launcher-invocation facts, learned the hard way

1. Runtime args need a literal `--` separator, or argparse rejects them before REMAINDER.
2. **`policy-improvement-exp1b-training` must NOT be given `--producer-source-project-root` /
   `--expected-producer-git-commit`.** Refusal, verbatim:
   `Phase 4 runtime rejected: Policy consumers require a separate complete producer-source authorization; other roles cannot accept one.`
   My input error, not a defect: `separate_producer_consumer`
   (`phase4_runtime_launcher.py:1494`) covers audit, analysis, theory-bridge and
   **exp1b-bridge** — not exp1b-training.
3. **Stage B will need a second, distinct clean checkout** at `ef24fa10…`:
   `phase4_runtime_launcher.py:1540` refuses `producer_source_root == source_root` for the
   bridge purpose. Not yet created.

### Working Stage A command

```bash
RUN_UPITRM_FULL_EXPERIMENTS=1 <opt launcher .par> \
  --purpose policy-improvement-exp1b-training \
  --runtime-archive <policy_improvement_full.par> \
  --expected-runtime-sha256 024a45772dd4c97ee69efa701079e2e7ccc4ea4abff9a6a883d636e8ec9533d9 \
  --source-project-root /home/buiksat/trm_bellman \
  --expected-source-git-commit ef24fa10554490189caea7f63646400b6e26a2ad \
  --runtime-authorization $OWNER/exp1b_runtime_authorization.json \
  --expected-runtime-authorization-sha256 ec598b225d435df0f196c6307c675c5043b06888a061e540ff5e64ba99b43d01 \
  -- \
  --base-policy-artifact $OWNER/base-policy-artifact/base_policy.pt \
  --base-policy-amendment $OWNER/base_policy_amendment_exp0_20260911.json \
  --execution-admission $OWNER/exp1b_execution_admission_20260911.json \
  --evidence-root $OWNER/evidence --evidence-generation exp1b-20260911 \
  --row-id <run_id> --seed <seed>
```

### Seed 0 status

Started 2026-09-11 ~21:52 UTC. Passed every authentication gate; generation lock held at
`evidence/exp1b-20260911/.lock`. At 23:34 UTC: **102 minutes elapsed, still running**, pid 607593
at 98.4% CPU, state R, GPU 0 at 26% / 580 MiB. Genuinely computing, not hung. No checkpoint or
payload written yet.

**Projection already exceeds the three-hour budget by a wide margin.** Eight seeds at >=102 min
each is >=13.6 h of Stage A alone, before Stage B. Reporting to the coordinator.

---

# SEED 0 CHECKPOINT — measured, and it FAILED on a production defect

## Wall clock

**6,594 s = 1 h 49 min 54 s.** Launcher exit code **1**. The wrapper shell exited 0, which is why
the task notification said "exit code 0"; the launcher did not.

Stage A reached and consumed the full 10,000-interaction budget (debug trace ran to
`train_step 350`), then crashed on the **last line of the training loop**, at the sealing step.
Nothing was sealed. No checkpoint, no payload, no result.

## Projection for eight seeds

Seed 0 spent 6,594 s before the crash, and the crash is at the end of the budget, so a
*successful* seed is >= that. Eight seeds:

```text
8 x 6,594 s = 52,752 s = 14 h 39 min   Stage A alone, before Stage B
```

**Roughly 14.7 hours against a ~3-hour budget.** Stopping, per the standing rule. Seeds 1-7 not
started.

## The defect, verbatim

```text
Traceback (most recent call last):
  ...
  File "/proc/self/fd/7/scripts/policy_improvement_exp1b_runtime.py", line 510, in main
  File "/proc/self/fd/7/policy_improvement_full_backend.py", line 5086, in run_exp1b_training
AttributeError: 'SealedFullRunBackend' object has no attribute '_module'. Did you mean: '__module__'?
```

`SealedFullRunBackend.__init__` assigns `self._training_module` (line 4950).
`self._module` is `TorchLearnedRunEngine`'s attribute (line 668), not this class's. The sibling
method `prepare_exp1b_training` already passes `training_module=self._training_module`
correctly at line 5049. **One wrong attribute name, on line 5086, the final statement of
`run_exp1b_training`.**

### Why no test caught it

Every test that touches `run_exp1b_training` **substitutes its own class**
(`tests/test_policy_improvement_exp1b_unittest.py:4466` and `:8290` define replacement methods);
`:3833-3834` only asserts the method name exists and is not a stub. Nothing ever executed the
real `SealedFullRunBackend.run_exp1b_training` body. That is exactly how a one-token typo reached
a 1h50m production path.

## Fix prepared (UNCOMMITTED, coordinator commits)

1. `policy_improvement_full_backend.py:5086` — `self._module` -> `self._training_module`, with a
   comment recording what it cost.
2. `tests/test_policy_improvement_exp1b_unittest.py` — new
   `test_the_sealed_backend_reads_only_attributes_it_assigns` in `Exp1bProductionBackendTest`.
   Torch-free AST check: every `self.<attr>` **read** in `SealedFullRunBackend` must be an
   attribute the class assigns or a method it defines. `FullRunBackend` is a `Protocol` with no
   instance state, so the closure is exact. Catches this bug and every future one of its shape.

**Coverage proven the required way, not assumed.** Reverted the fix, ran the new test alone:
`Pass 0. Fail 1.` Restored the fix, ran the whole target: **`Pass 310. Fail 0. Skip 3.`**
(309 -> 310 is the new test; the 3 skips are the known placeholder Torch tests.)

## The re-sign cascade this triggers

`policy_improvement_full_backend.py` is in the full, theory-bridge, audit, and analysis source
profiles (not the launcher's, not the producer manifest). So committing the fix forces:

1. Rebuild `policy_improvement_full`, `policy_improvement_theory_bridge`,
   `policy_improvement_audit`, `policy_improvement_analysis` -> **four new runtime digests**.
2. **Re-mint** the runtime authorization at the new commit -> new
   `runtime_authorization_sha256`.
3. **Re-sign the base-policy amendment** (it binds `runtime_authorization_sha256`) -> new
   amendment digest.
4. **Re-sign the admission** (it binds the amendment digest, both runtime digests, and
   `source_git_commit`).

The launcher PAR and `upi_trm_train` are unaffected and keep their digests. The three documents
minted today are dead the moment the fix lands; they are recorded above for the audit trail, not
for reuse.

## State left behind: clean, nothing half-claimed

```text
evidence/exp1b-20260911/.lock          0 B
evidence/exp1b-20260911/payloads/      empty
evidence/exp1b-20260911/checkpoints/   empty
```

No `schedule_state.json`, no `provenance.json`, no claim recorded. The seed-0 lease was never
taken. Re-running seed 0 after the fix starts clean; the generation does not need resetting.

---

# DECISIONS 1-3 and 6 DONE. Blocked on the commit before seed 0.

## 1. Typo fixed

`policy_improvement_full_backend.py:5086` — `self._module` -> `self._training_module`, with a
comment recording what it cost.

## 2. A test that executes the REAL production body

New class `Exp1bRealSealedBackendTrainingLoopTest` in
`tests/test_policy_improvement_exp1b_unittest.py`. It constructs the **real**
`SealedFullRunBackend` through the **real** `SealedRuntimeIdentity.from_mapping` validator (exact
nine-field inventory, authorized role, real digests, the profile==manifest==selected equality) and
calls the **real** `run_exp1b_training`. Five cases:

| test | what the real body must do |
| --- | --- |
| `..._seals_with_this_backends_training_module` | seal with `self._training_module`, identity-checked |
| `..._stops_exactly_on_the_registered_budget` | reach exactly 10,000 and shrink the request each call: `[10000, 7000, 4000, 1000]` |
| `..._refuses_a_trainer_that_is_not_fresh` | raise on a nonzero starting step count |
| `..._refuses_a_trainer_that_stalls` | raise on a step that consumes nothing |
| `..._refuses_a_run_that_resolved_evaluation_data` | raise, **and seal nothing** |

Hermetic: no owner artifact, no dataset, no checkpoint, no Torch tensor. Only two collaborators
are stand-ins — a counting trainer and a recorder swapped in for the module-level
`seal_exp1b_training_checkpoint`. Patching the callee does not weaken it: the `AttributeError`
fires while the *argument* is evaluated, before any call.

Also kept: the Torch-free AST check `test_the_sealed_backend_reads_only_attributes_it_assigns`.

## 3. Falsification — RED, as required

Broke line 5086 back to `self._module`, reran:

```text
Tests finished: Pass 3. Fail 3.
  x test_the_sealed_backend_reads_only_attributes_it_assigns
  x test_the_real_body_stops_exactly_on_the_registered_budget
  x test_the_real_body_seals_with_this_backends_training_module
```

The three that stayed green are the three that raise *before* reaching line 5086 — correct, not a
gap. Restored the fix: **`Pass 315. Fail 0. Skip 3.`** (309 baseline + 1 AST + 5 real-body; the
3 skips are the known placeholder Torch tests.)

## 4. Sweep — no second AttributeError, but two more coverage holes of the same shape

Ran the same missing-attribute analysis over every class in
`policy_improvement_full_backend.py`, `policy_improvement_exp1b_runtime.py`,
`policy_improvement_exp1b_evidence.py`, `policy_improvement_exp1b_session.py`.
**Zero classes read an attribute they never assign.** No second typo of this exact kind is
waiting.

Two production surfaces still have only substituted-class coverage:

1. **`SealedFullRunBackend.prepare_exp1b_training`** — `grep '\.prepare_exp1b_training('` over the
   test file returns **zero** real invocations. Coverage is an existence/not-a-stub assertion
   (`:3832`), a signature AST check (`:4687`), and a substituted method (`:4488`). It runs
   immediately before `run_exp1b_training` in the real Stage A path. Lower risk than 5086 was: it
   runs *before* the budget, so a defect fails in seconds rather than after 1h50m.
2. **`Exp1bSealedEvaluationSession`** — never constructed in the test file; production constructs
   it at `policy_improvement_full_backend.py:6628`; tests replace the whole class with a recorder
   at `:5818`. Nine methods including `endpoint_values`, `action_values`, `action_mask`,
   `base_probabilities` — the Stage B numerical surface. **This is the one to worry about**: it
   runs only after all eight Stage A seeds are sealed.

Reported, not fixed, per instruction.

## 6. PARALLELISM: nothing forbids it. The machinery is built for it.

Checked all three places the coordinator named.

**Protocol:** silent. `configs/policy_improvement_exp1b/protocol.json` contains no
`sequential`, `concurrent`, `parallel`, or `serial` language. The only `order` matches are
`draw_order` and `ordered_record_sha256`, which are dataset record ordering.

**Evidence machinery: explicitly concurrent by design.**

- `claim_lease(seed_position)` is **per slot** (`.claim-{N}.lock`), not per generation. Its
  docstring: *"Separate from `exclusive`, which is the generation-wide mutex held only across a
  state transition. This lease spans the whole evaluation, which is precisely the window the
  generation lock is not held for."*
- `publish_checkpoint` docstring: *"The whole call runs under the generation mutex, so **two
  Stage A processes racing on the same seed** cannot interleave; the loser fails on `O_EXCL`."*
- Finalization: *"`O_EXCL` on the provenance makes it exactly-once **regardless of which run
  observes the last manifest**"*; `_finalize_when_complete` is a no-op until all eight positions
  report `complete`, and *"Losing the race is not an error."*
- Already tested: `test_concurrent_writers_are_serialized_and_one_loses` (`:3284`), *"Two OS
  processes racing on the same seed: exactly one payload survives."*

**Generation lock is NOT held across a run.** Stage A takes `exclusive()` only inside
`publish_checkpoint` and the reap/classify reads. The 1h49m of training holds nothing.

**Hardware:** 44 cores, 2x A100. Seed 0 was **CPU-bound** — 98.4% of one core, 26% of one GPU,
592 MiB. Eight concurrent processes need ~8 cores and ~4.7 GB spread over two GPUs. Ample.

**Projection if run 8-wide: ~1.9 h instead of 14.7 h.** Caveat, stated as a caveat: this assumes
near-linear scaling from a CPU-bound single-core workload with 44 cores available. Measured, not
assumed, once seeds 1-7 launch.

## 4. RESUME: a successful seed 0 DOES count. No 1.8 h is thrown away.

`publish_checkpoint` durably writes `checkpoints/seed-0/{checkpoint.pt,run_manifest.json}` with
`O_EXCL` + fsync, and `checkpoint_publication_state(0)` reads it back. The finalizer fires only
when all eight positions are `complete`. Seeds 1-7 continue from a sealed seed 0 with no special
handling. The extra 1.8 h the coordinator offered is not needed.

## BLOCKED: the commit, then the full re-sign cascade

Uncommitted: `policy_improvement_full_backend.py` (+9/-1) and
`tests/test_policy_improvement_exp1b_unittest.py` (+225). Both synced to fbsource.

After the commit, in order: rebuild full/theory/audit/analysis -> re-mint authorization ->
re-sign base-policy amendment -> re-sign admission -> run seed 0 -> report at the seal.

---

# RUN 2 at 5a0ad34e119a37e44cc49edf00a93264cf63d3b1 (2026-09-12)

Coordinator granted standing commit authority. Rebuild -> re-mint -> re-sign cascade done exactly
as predicted: launcher and `upi_trm_train` digests unchanged, four backend-dependent PARs moved.

| target | sha256 | moved? |
| --- | --- | --- |
| `phase4_runtime_launcher` (opt) | `f99099ee418647ee970f84c2c6ba6d373cd143676d37498d0124e80b48c289bd` | no |
| `upi_trm_train` | `45488e959fc713d56efb074dfc8e9fc424d17099ad42746ced3128ff8b6f6716` | no |
| `policy_improvement_full` | `ab1648fa773fdb842df4dae2ae1feb75ea29b001e033028547dd3f4e98e24009` | **yes** |
| `policy_improvement_theory_bridge` | `c2ab9aebf98bd5df65dd77bf6953f761cf96254f2f217d266175d9e37ee1415c` | **yes** |
| `policy_improvement_audit` | `60405d6b2566d5d490183f2f427f7c368223a677798a3429f5fcdf9ca787acda` | **yes** |
| `policy_improvement_analysis` | `5566687b34432a8826a06f25c33d0d2d7d296ac225b046bd62ada032f51fdfab` | **yes** |

CUDA guard, trace `775aabe8-8aaa-4acd-b190-4a0e3150c76e`: 5 rows with cache hits excluded, 0 CUDA.

## Live signed documents (supersede the 2026-09-11 set)

```text
runtime authorization   20c898ee3fd05597bc753314d500c72985f74629f0e2a608e259e4db09a34168
                        exp1b_runtime_authorization_20260912.json
base-policy amendment   2e76355e89c0e6a17b9a739a0fbf33c30dd294baa605cca241c38c170d649df5
                        base_policy_amendment_exp0_20260912.json
execution admission     c48d67ffa80bdea2716a5d4ae7c0bfa6a45554f1853d59cb666fda483a9bd7a1
                        exp1b_execution_admission_20260912.json
```

All 13 non-`status` artifact values re-asserted byte-equal to `base_policy_identity.json`.
Producer/evaluator runtimes distinct: `ab1648fa…` vs `c2ab9aeb…`.

## Stage B evaluator checkout

`/home/buiksat/trm_bellman_evaluator`, clean clone at `5a0ad34e…`, `git status` 0 entries.

**It also needed the dataset.** `scripts/policy_improvement_exp1b_runtime.py:1029` resolves
Stage B's `dataset_root` as `producer_root / registered_training["dataset_root"]`, and `data/` is
gitignored so the clone had none. Copied `data/policy-improvement-v1-owner` (1.1 MB) into the
clone; `diff -r` identical, clone still 0 dirty entries because `.gitignore` covers `data/*`.

## Fresh evidence generation

`$OWNER/evidence2/exp1b-20260912/{payloads,checkpoints}`, all 0700. The 2026-09-11 generation is
abandoned: its documents are superseded and it holds nothing but an empty lock.

## SEED 0 SEALED — 2026-09-12, WALL 7012 s (1 h 56 m 52 s), EXIT 0

The fix works end to end. `checkpoints/seed-0/`: `checkpoint.pt` 26,674,187 B
(`2046db70131f645fd7c103ca7badba4c73366e441f1b2d7f617163483f3628af`) + `run_manifest.json`.

Manifest facts worth keeping:

```text
environment_interactions            10000        exactly the registered budget
resolved_evaluation_data            false
restored_base_model_state_sha256    0d18f790...  identical to the admitted base policy
initialization_artifact_sha256      607c3065...  the admitted checkpoint
initialization_kind                 train_only_pretrained_base_policy   (runtime literal)
model_state_sha256                  a0f90fc7c608bde75678ec67bda9d91e77972fcb074b5771f53d52e04df09db4
train_split_ordered_record_sha256   73110263...
exact_centering_batch_count         350
```

### Launch mechanics, learned twice

Use `setsid nohup ... &` + `disown`. The harness's own background wrapper killed the first
attempt (exit 144), and a foreground `timeout 300` killed only the shell — its launcher
grandchild survived as an orphan and ran seed 0 a second time concurrently. Caught it by
noticing GPU 0 at 97% / 1179 MiB instead of 26% / 592 MiB, and killed the orphan tree. Nothing
was corrupted: neither process had reached publish, and `O_EXCL` would have caught it anyway.

Per-process cost, measured solo: **26% of one A100, 592 MiB, ~1 core, ~5 train steps/min steady
state, 350 steps total.**

## PARALLELISM MEASURED — it holds. 7-wide, no contention.

Seeds 1-7 launched 2026-09-12 18:27 UTC, `setsid nohup`, 4 on GPU 0 and 3 on GPU 1.

Measured 30 min in, per-process CPU:

```text
97.4  97.4  97.4  97.3  97.3  97.3  97.2     (7 processes)
```

Solo seed 0 ran at 98.4%. **Seven concurrent processes each keep ~97% of a core** — near-linear,
as projected. 44 cores, so no CPU contention. Both GPUs report 100% utilization with 2,353 MiB
and 1,766 MiB, but that metric is kernel *residency*, not throughput: the gating resource here is
a single Python core per process, and every process still has one.

Expected completion of all seven ~20:25 UTC, i.e. **~2 h wall for seeds 1-7 instead of ~13.6 h
sequential.**

### CORRECTION — the 97%-CPU reading was CUDA spin-wait, not throughput

My "near-linear, ~1.9 h" projection above was **wrong**, and the error is worth recording. All
seven processes sit at ~97.8% CPU, but CUDA busy-waits on synchronization by default, so a
process blocked on a contended GPU still burns a full core. CPU% is not a progress metric here.

Measured against the solo baseline instead (**solo reached `train_step 50` at 31 min**):

```text
GPU 1, 3 processes (seeds 5,6,7):  train_step 50 at 81 min   -> 2.6x slowdown
GPU 0, 4 processes (seeds 1,2,3,4): not yet at 50 at 81 min  -> worse than 2.6x
```

Note this is *not* raw GPU throughput saturating: solo used only 26% of a card, so three
processes nominally demand 78%. The cost is CUDA context time-slicing across processes with many
tiny kernels (hidden_size 64, batch 32). CUDA MPS would likely recover most of it; not worth
reconfiguring mid-run.

**Revised projection: seeds 1-7 complete in ~5-7 h wall, versus 13.6 h sequential.** Still worth
doing — roughly 2x, not the 7x I claimed. Left running: 7-wide only loses to sequential if the
slowdown exceeds 7x, and it is 2.6x.

### Both concurrency groups measured (`train_step 50`, solo baseline 31 min)

```text
GPU 1, 3 processes   seeds 5,6,7    81 min      2.6x
GPU 0, 4 processes   seeds 1,2,3,4  105-110 min 3.4x
```

Solo steady state after warmup: 300 steps in 86 min (3.49 steps/min).
Scaling that, expected completion from the 18:27 launch:

```text
seeds 5,6,7   ~305 min  -> ~23:30 UTC 2026-09-12
seeds 1,2,3,4 ~410 min  -> ~01:15 UTC 2026-09-13
```

So all eight seeds sealed by roughly 01:20 UTC. Sequential would have finished ~08:00 UTC.
**Net saving ~6.7 h, not the ~11.7 h I projected.** Recorded as measured.

### FINAL parallelism measurement — concurrency buys ~1.2x, not 7x

Precise interval, seed 7 on GPU 1 (3-process group):

```text
step 50  at 19:48:05
step 100 at 21:07:11     50 steps in 79 min = 0.63 steps/min
solo baseline            50 steps in ~14 min = 3.57 steps/min
                         -> 5.6x per-process slowdown
```

Aggregate: 7 x 0.63 = 4.4 steps/min against solo's 3.57. **Concurrency buys roughly 1.2x, not
the 7x I projected.** The bottleneck behaves like a single shared serial resource, not GPU
capacity (solo used 26% of one card; 44 physical cores, 1 thread/core, so CPU is not it).

**Decision: keep running 7-wide.** Restructuring was modelled and rejected. Killing the 4-process
GPU 0 group to give GPU 1 a clear run would forfeit ~2.7 h x 4 of partial work and yield about
the *same* aggregate throughput (3 procs x 1.43 steps/min = 4.3/min vs the current ~4.4/min),
because the cap is the shared resource, not the arrangement. Falling back to sequential now would
be strictly worse: 7 x 117 min from here = 10:40 UTC.

Projected: all eight sealed **~04:40-06:00 UTC 2026-09-13**. Sequential from the 18:27 launch
would have been ~08:00 UTC. Real saving ~2-3 h.

Recorded as measured, including that my earlier 1.9 h projection was wrong by a factor of ~3.

### Both groups, final rates

```text
GPU 1, 3 procs  seed7   50->100 in 79 min   0.63 steps/min   5.6x slower than solo
GPU 0, 4 procs  seed1   50->100 in 105 min  0.48 steps/min   7.4x slower than solo
solo                                        3.57 steps/min
```

Aggregate 4(0.48) + 3(0.63) = **3.81 steps/min against solo's 3.57 — parallelism buys ~7%.**
The machine delivers a fixed ~3.8 steps/min no matter how the work is arranged. Nothing to
reclaim by restructuring; nothing lost by having tried it.

Revised completion: GPU 1 group ~04:35 UTC, GPU 0 group ~06:40 UTC 2026-09-13.

---

# DEADLINE CONSTRAINTS (coordinator, 2026-09-12 ~16:50 PDT / 23:50 UTC)

Machine dies **Sunday 2026-09-13 10:00 PDT = 17:00 UTC**. Nothing here is durable; no reachable
remote.

## Stage B is validated and it is NOT multi-hour

Smoke-ran Stage B twice against the real launcher, evaluator checkout, and signed documents.
Both reached `open_authenticated_exp1b_route` in **81-83 s** and failed on exactly one thing:

```text
Exp1bBridgeError: substitute provenance is unavailable.
```

That is `provenance.json`, which `finalize_exp1b_evidence` writes only after all eight Stage A
seals. **Expected at 1/8.** Everything upstream of it already passes: theory-bridge PAR
authentication, the source profile at `5a0ad34e`, the distinct evaluator checkout at
`/home/buiksat/trm_bellman_evaluator`, the admission, the runtime authorization, the role
attestation, and all four parent documents under the producer root. The only unmeasured part of
Stage B is the census evaluation itself.

## HEAD moved off the admitted commit — caught and restored

The coordinator committed `3dd15aa` ("Add a recovery document and preserve the run notes") onto
`full-implementation`, which moved the primary checkout's HEAD off `5a0ad34e`. **Stage B would
have refused**, and re-minting was not an option: the eight Stage A run manifests embed admission
`c48d67ff…`, so re-signing mid-run would orphan every seal.

Fixed with `git checkout --detach 5a0ad34e`. HEAD is back on the admitted commit with a clean
tree, and `3dd15aa` is preserved on `full-implementation`. Re-ran the Stage B smoke afterwards:
still reaches the provenance gate in 81 s, so the restored state authenticates.

**Do not commit to `full-implementation` while the run is live.** The primary checkout must stay
detached at `5a0ad34e` until the audit finishes.

## Artifact commits: linked worktree, never the pinned checkout

`/home/buiksat/trm_bellman_artifacts` is a `git worktree` on branch `exp1b-artifacts` based at
`5a0ad34e`. Committing there leaves the primary checkout's HEAD and working tree untouched --
verified before and after every commit. Driver script `/tmp/commit_artifacts.sh`.

First commit **`c6fe3ae`**: three signed documents, base-policy producer identity, six PAR
digests, seed-0 run manifest.

**Caveat worth stating plainly: a local commit is not durability.** There is no reachable remote,
so these commits die with the box exactly like the loose files. What they buy is provenance and a
single coherent object to copy off, not survival.

---

# Branch layout changed by the coordinator (2026-09-13 ~00:15 UTC)

`exp1b-artifacts` deleted; commit `c6fe3ae` cherry-picked onto `full-implementation` as
`83fcc9e`; the linked worktree `/home/buiksat/trm_bellman_artifacts` now tracks
`full-implementation`. Verified: primary checkout still **detached at `5a0ad34e`, 0 dirty**.
Commit artifacts from the worktree to `full-implementation` from here on.

Durability premise updated by the coordinator: `origin/full-implementation` is reachable and
already carries `3dd15aa`; they will push `83fcc9e`. So commits there do leave the box. My
earlier "a local commit is not durability" caveat no longer applies.

# Stage B expected wall clock (estimate, from a call count)

Per seed position the bridge does a 128-member census traversal. `action_values` loops all 97
actions (`disable_constraint_masking: true`, so nearly all allowed) and per allowed action does
one env step plus one `_value_at` forward unrolled to the depth, at **both** n=2 and m=8:

```text
128 members x 97 actions x 2 depths  = ~24,800 unbatched model calls per seed
                                     = ~198,000 across eight
```

Unbatched and tiny (hidden_size 64, seq_len 16, H=2/L=2), so latency-bound at a few ms each --
*not* the 0.70 s/interaction Stage A figure, which is dominated by batched value/policy updates.

**Estimate: 3-15 min per bridge, 25 min - 2 h for all eight.** A bridge would have to exceed
~70 min to threaten 17:00 UTC given a ~06:50 last seal. Real number lands within minutes of the
eighth seal; the driver runs position 0 first.

Fallback if bridge 0 is slow: `claim_lease` is per-slot, so all eight bridges can run
concurrently. Expect the same ~1.2x ceiling measured on Stage A, but it is available.

## Why Stage B could not be measured before the eighth seal

Two independent blocks, both recorded rather than worked around:

1. The bridge refuses without `provenance.json`, which `finalize_exp1b_evidence` writes only
   after all eight seals. The real path is unreachable until then.
2. Torch cannot be imported outside the PAR sandbox:
   `ModuleNotFoundError: No module named 'pyjk'` from `torch/_utils_internal.py:47` when running
   the test link tree under plain `python3.12`. `pyjk` is a native extension the PAR's static
   extension finder provides. Timing a forward pass standalone would need a BUCK change, and the
   primary checkout must stay clean and pinned at `5a0ad34e`.

So the 3-15 min/bridge figure is a call count times a plausible per-call latency, not a
measurement, and is labelled as such. Decision threshold: **>70 min per bridge threatens 17:00
UTC.** If bridge 0 exceeds it, run the remaining seven concurrently (per-slot `claim_lease`).

## Results preservation, standing (coordinator, 2026-09-13 ~00:45 UTC)

Commit every result artifact to `full-implementation` from the artifacts worktree **the moment it
exists**, not batched: each run manifest as it seals, then the published result document, its
sidecar, the five diagnostic blocks, the paired signed gap g with its 95% interval, and the
auditor verdict. Small JSON only; checkpoints stay out of git, their digests go in.

Neither agent can push: `GIT_CONFIG_COUNT=3` rewrites SSH GitHub URLs to HTTPS with no
credentials. That is a deliberate control. **Do not work around it, do not unset it.** Commit
locally and stop. `/home/buiksat/push-results.sh` is the owner's one-command push;
a 10-minute cron writes `/home/buiksat/.upi-trm-watchdog/UNPUSHED-RESULTS.txt` when the branch is
ahead, so a local commit becomes a visible prompt within ten minutes.

On audit completion write `/home/buiksat/.upi-trm-watchdog/EXPERIMENT-COMPLETE` containing the
result digest and the verdict. That stops the watchdog and the stall cron.

## SEEDS 5, 6, 7 SEALED — 2026-09-13 03:47 UTC

```text
seed 5  pos 5  EXIT=0  WALL=33466 s  (9 h 17 m)
seed 6  pos 6  EXIT=0  WALL=33462 s  (9 h 17 m)
seed 7  pos 7  EXIT=0  WALL=33323 s  (9 h 15 m)
```

4/8 sealed (positions 0, 5, 6, 7). Manifests committed as `b519884`. GPU 1 now free; seeds 1-4
continue on GPU 0 at step 250/350.

### Honest verdict on the concurrency experiment: it was a wash, arguably a loss

Solo seed 0 took 1.95 h. The 3-wide GPU 1 group took **9.3 h each**, a 4.8x per-process
slowdown:

```text
3 seeds sequential   3 x 1.95 h = 5.85 h
3 seeds 3-wide                  = 9.30 h      1.6x WORSE
```

Running three concurrently on one card was **worse than running them one at a time.** The only
real gain came from using both GPUs at all, not from stacking processes on each. A better plan
would have been 1 process per GPU, 2 at a time: 8 seeds / 2 GPUs x 1.95 h = 7.8 h, against the
~11-12 h this run will take.

Recorded as measured. My 1.9 h projection was wrong by ~6x, and the intermediate 1.2x-aggregate
estimate was still optimistic. The lesson for the next run: on this workload **do not stack more
than one process per GPU**; CPU% is not a progress metric when CUDA spin-waits.

---

# ALL 8 STAGE A SEEDS SEALED — then Stage B hit a blocking defect

```text
pos 0  EXIT=0  WALL=7012 s   (1 h 57 m, solo)
pos 1  EXIT=0  WALL=44588 s  (12 h 23 m)
pos 2  EXIT=0  WALL=44774 s
pos 3  EXIT=0  WALL=44783 s
pos 4  EXIT=0  WALL=44725 s
pos 5  EXIT=0  WALL=33466 s  (9 h 17 m)
pos 6  EXIT=0  WALL=33462 s
pos 7  EXIT=0  WALL=33323 s
```

`provenance.json` written, `access_state: sealed_octet_complete`, 8 sealed checkpoints, all eight
registered seed ids present. Manifests committed: `83fcc9e`, `b519884`, `bba9554`.

## Stage B defect — `KeyError: 'split_manifest_sha256'`

```text
File "scripts/policy_improvement_exp1b_runtime.py", line 1034, in bridge_main
KeyError: 'split_manifest_sha256'
```

`bridge_main` reads `evaluation_population["split_manifest_sha256"]`. The registered exp1b
protocol's `evaluation_population` holds exactly
`['binding_sha256','count','ordered_record_sha256','population_id','selection_use','split']`.
Only `training_population` carries `split_manifest_sha256`.

**Third instance of substituted coverage.** `tests/...exp1b_unittest.py:3811` passes
`evaluation_split_manifest_sha256` directly into the backend inputs, so the extraction in
`bridge_main` that fails is never executed by any test.

### The value exists and the fix is one line

`configs/policy_improvement_v2/protocol.json#/dataset/splits/validation/manifest_sha256` =
`a4b1bffb92f9c7c1ebe7baf9dcaaed83191f9c6249bec0887ed8ff7fb6b4a937`, and
`sha256sum <regen corpus>/manifests/validation.json` is **byte-identical** to it. The route
already authenticates the parent protocol against the exp1b protocol's declared
`parent.protocol_sha256`, so reading it there is sound.

### Why the fix cannot be applied to this evidence

`provenance.json` freezes the producer attestation:

```text
runtime_sha256                ab1648fa...   (policy_improvement_full PAR)
runtime_authorization_sha256  20c898ee...
launcher_sha256               f99099ee...
source_git_commit             5a0ad34e...
admission_sha256              c48d67ff...
```

`attestation_matches_authorization` (`policy_improvement_exp1b_schema.py:1866`) compares **all
five**. `scripts/policy_improvement_exp1b_runtime.py` is in **both** the full and theory source
profiles, so any edit changes the full PAR digest, forcing a re-mint, which changes the
authorization digest, which the frozen provenance no longer matches. Keeping the old full PAR
instead fails the minter's `assert_phase4_archive_matches_profile` against the fixed checkout.
**There is no variant that preserves the eight seals.**

This is the integrity design working as intended, and its corollary is that a Stage B defect
discovered after Stage A costs a full Stage A re-run.

### Sweep for more of the same class

Every literal key the Stage B path reads out of the registered documents:

```text
OK   runtime.py:1032  evaluation_population['split']
MISS runtime.py:1034  evaluation_population['split_manifest_sha256']
OK   runtime.py:1030  registered_training['dataset_root']
OK   bridge.py:834    registered_training['ordered_record_sha256']
```

Exactly one miss. That closes this defect class, not others.
`Exp1bSealedEvaluationSession` (nine methods, the whole Stage B numerical surface) still has zero
real coverage and has never executed.

### Timing against the 17:00 UTC cutoff (assessed 07:25 UTC)

```text
fix + rebuild + re-mint + re-sign        0.5 h
Stage A re-run, 1 proc/GPU, 4 rounds     7.8 h   (floor; stacking measured worse)
Stage B (unmeasured)                     1.0 h
audit + publish                          0.25 h
final handoff                            0.5 h
                                        ~10 h -> ~17:25   MISSES
```

Recommendation given to the coordinator: **do not re-run.** A second defect in a never-executed
surface would consume the remaining window and produce nothing. Spend it instead on the fix plus
real-body tests for `bridge_main`'s extraction and `Exp1bSealedEvaluationSession`, committed to
`full-implementation`, so the next machine succeeds on the first pass.

---

# RUN 3 at dcaf19ef6d8b3f376af5933d86e4363965a2fafb (2026-09-13 07:16 UTC)

Stage B fix committed as `dcaf19e` on `full-implementation`. Implementer took the re-run decision
without waiting: prep finished in 25 min rather than 30, moving Stage A start to 07:16 instead of
~07:55, which turned the projection from "misses 17:00" into "fits with ~45 min". Killing it
costs nothing (GPUs otherwise idle; handoff work needs no GPU).

## Both static sweeps clean across every Stage B module

- missing attribute reads: **0** classes
- missing registered-document keys: **0**

covering `exp1b_theory_backend`, `exp1b_bridge`, `exp1b_aggregate`, `exp1b_auditor`,
`exp1b_evidence`, `exp1b_runtime`, `exp1b_session`, `theory_bridge_entrypoint`,
`full_backend`. The two classes that cost 12.4 h are cleared. **Not** proof Stage B works:
`Exp1bSealedEvaluationSession` (9 methods) still has no real coverage and has never executed.

## Live identifiers (supersede everything above)

```text
commit                  dcaf19ef6d8b3f376af5933d86e4363965a2fafb
runtime authorization   c5d077e0e0b17e2bdb0f444669b7fb8f2a5ce2891dec9161d34a38d51d8da78f
                        exp1b_runtime_authorization_20260913.json
base-policy amendment   47b21e8ba6c826838c5251dd382e87f31962f29093c7e67476e8f8c0cbf43881
                        base_policy_amendment_exp0_20260913.json
execution admission     92e0ed7933d72f5c3d9e53d3f2b75dd65ab98641f175f91158ece75f40d0f96b
                        exp1b_execution_admission_20260913.json
full runtime            09cf7a1f5a1ffe4a72e4163f68e0dfef4d79ddba5287cff79fb0798b952c5216
theory runtime          c5b9a67a11b68179c9a0ddd168dddf3e9f64502e75a764bf452a3643f4551ebf
launcher (opt)          f99099ee418647ee970f84c2c6ba6d373cd143676d37498d0124e80b48c289bd
upi_trm_train           45488e959fc713d56efb074dfc8e9fc424d17099ad42746ced3128ff8b6f6716
evidence generation     $OWNER/evidence3/gen-20260913
evaluator checkout      /home/buiksat/trm_bellman_evaluator @ dcaf19e, clean
```

Scripts: `/tmp/run_seed2.sh`, `/tmp/run_bridge2.sh`, `/tmp/queue.sh` (one process per GPU, 4
rounds), `/tmp/driver2.sh` (fires all eight bridges on the eighth seal).
Logs: `/tmp/queue.log`, `/tmp/n_seed<N>.log`, `/tmp/driver2.log`, `/tmp/n_bridge<N>.log`.

## Schedule

```text
round 0  seeds 0,1  ~09:13      round 2  seeds 4,5  ~13:07
round 1  seeds 2,3  ~11:10      round 3  seeds 6,7  ~15:04
Stage B x8 ~16:00   audit ~16:15   deadline 17:00
```

Round 0 confirmed at 22% / 27% GPU -- the solo profile, no contention.

## Run 2's evidence is dead but recorded

The eight run manifests from the 2026-09-12 run are committed (`83fcc9e`, `b519884`, `bba9554`)
as the record of what the first attempt produced. Its checkpoints remain in
`$OWNER/evidence2/exp1b-20260912/` and cannot be used: the fix changed the full PAR digest that
their provenance froze.

## Stage B numerical surface now has real coverage — and no defect in it

Written while Stage A round 1 ran, on the coordinator's direction. Committed `f8b600f`.

`Exp1bRealSealedEvaluationSessionTest`, nine tests, constructs the **real**
`Exp1bSealedEvaluationSession` (production builds it at
`policy_improvement_full_backend.py:6628`; every prior test replaced it with a recorder).
Stubs only at the dataset/environment and model/trainer boundaries.

Pinned behaviour: depth honoured; `r + gamma * U_q(s')` per allowed action; terminal action takes
the reward with no bootstrap; Bellman operator reads `policy_model_old` not the deployed mixture;
persistent branch evaluates the candidate at `n=0` on the base call's output latent (the K=1
contract); census digest mismatch, unregistered state, unregistered depth and repeated state id
all refuse; record derived once and shared across q=2 and q=8.

**Falsified.** Three mutations of the production entry point, each red:

```text
drop the gamma discount                 -> 2 red
base_model = policy_model_candidate     -> 3 red
remove the census digest comparison     -> 1 red
```

Applied to the **fbsource copy only** and reverted immediately. Verified after each: fbsource
identical to the primary checkout, primary clean at `dcaf19e`, and
`policy_improvement_full.par` still `09cf7a1f`. The in-flight Stage A rounds were never at risk.

**No defect found.** Whole target: 327 passed, 0 failed, 3 skipped.

### Correction to the coordinator's timing assumption

"A fix landed before 15:04 still makes this run" is **false**. Seeds 0 and 1 sealed at 09:01 and
09:06; their run manifests freeze `admission_sha256` and the full PAR digest, so any production
change forces a re-mint and re-sign that invalidates them. Every commit also moves HEAD, and
rounds 2 and 3 pin `--expected-source-git-commit dcaf19e` against a *clean* checkout, so a dirty
or moved primary makes them refuse. **No production change is possible until the audit ends.**
Test-only work is free because tests are in no runtime profile and the artifact commits go to the
linked worktree, which never moves the primary HEAD.

### Run 3 progress

```text
round 0  seeds 0,1  sealed 09:01 / 09:06  exit 0  ~105 min each   commit 0b97640
round 1  seeds 2,3  running
```

---

# RUN 3 RESULT: all 8 Stage A seeds sealed; Stage B blocked by a FOURTH defect

## Stage A completed cleanly

```text
pos 0  EXIT=0  WALL=6663 s    pos 4  EXIT=0
pos 1  EXIT=0  WALL=6300 s    pos 5  EXIT=0
pos 2  EXIT=0                 pos 6  EXIT=0  WALL=6475 s
pos 3  EXIT=0                 pos 7  EXIT=0  WALL=6314 s
```

All eight sealed by 14:32 UTC, ~105-110 min each at one process per GPU. Manifests committed:
`0b97640`, `13aebbe`, `1ee0fcd`, `cdd06ac`. Provenance written; `access_state`
`sealed_octet_complete`.

**One process per GPU was the right arrangement** and is the single most useful operational
finding of this whole exercise: 105 min/seed against 9.3-12.4 h/seed when stacked 3-4 per card.

## Defect 4 — `prepare_exp1b_bridge`, bridge 0, 100 s in

```text
Exp1bTheoryBackendError: Stage B sealed restore refused: [upi_trm_train] Failed to load
the requested dataset split='validation' (ValidationError: 1 validation error for
PuzzleDatasetConfig / global_batch_size / Input should be a valid integer
[type=int_type, input_value=None, input_type=NoneType])
```

`open_exp1b_sealed_evaluation_session` called `build_dataset_from_paths(pool_size=None)`.
`pool_size` is declared `int` and reaches `PuzzleDatasetConfig.global_batch_size`.

**The crash was the lucky outcome.** `pool_size` is a *truncation bound*
(`rl/training_setup.py`: `if len(samples) >= pool_size: break`). The census size (128) would have
loaded the first 128 of the 256 registered validation records and misaligned every higher census
`record_index` — a wrong number rather than a refusal.

### Fixed: `8a55c8d`, corrected by `f4fe002`

`_registered_split_record_count` derives the bound from the split manifest, checking its bytes
against the digest the admission already pins and requiring `generated_count ==
len(record_sha256s)`. 256 for validation, 1024 for train. Absent manifest, digest mismatch,
malformed JSON, missing record list and count disagreement each refuse.

Falsified: `pool_size=None` restored -> red; agreement check dropped -> red.

### Why it escaped, and an error of mine

`Exp1bTorchSealedOpenerTest` *does* drive the real opener, but its fixture passed a bare digest
for `evaluation_split_manifest_sha256` and stubbed the training module, so no manifest had to
exist and nothing observed what `pool_size` the loader received. Fixture now materializes a real
manifest (`f4fe002`) rather than the production check being relaxed.

**I committed `8a55c8d` claiming the suite was green when it was 330/2.** I read the focused run
and not the full-suite line. Both failures were mine and are fixed. Final: **332 passed, 0
failed, 4 skipped.**

**Judgement error worth recording:** I identified `prepare_exp1b_bridge`'s happy path as the last
uncovered link and chose not to test it because it needed a real checkpoint. The nine tests I
wrote instead found nothing — `Exp1bSealedEvaluationSession` is sound. The defect was one layer
up, in the surface I skipped.

## This run is also unsalvageable

Provenance pins the authorization digest; fixing the theory backend forces a re-mint that
invalidates all eight seals. Stage A re-run is 7 h; 2 h 10 m remain. **Four Stage B defects have
now each cost a full Stage A run**, because the integrity design freezes the runtime set at
finalization. That is the structural lesson: on this architecture, Stage B must be proven by
execution *before* Stage A is spent.
