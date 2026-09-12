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
