# Experiment 1A readiness report

Date: 2026-09-08 (America/Los_Angeles)

Task root: `/home/buiksat/exp1a-readiness-20260908.2hI20P`

Paper checkout: `/home/buiksat/UPI_TRM`

Implementation checkout: `/home/buiksat/trm_bellman`

## Decision

| Axis | Finding |
| --- | --- |
| Ready for narrowly scoped setup | **Conditional.** Source inspection identifies a viable train-only producer/restoration foundation and a reusable endpoint evaluator, but the reduced study has no explicit registration and the required production-path tests did not run in this environment. |
| Blocked on artifact/provenance | **Yes.** The reported checkpoint, identity document, producer logs, source/procedure manifests, and base-policy amendment were not available at any documented or configured local path. The historical producer also used one field name for two different identities. |
| Blocked on implementation/runtime | **Yes.** The existing full-run constructor opens the evaluation split before training; the validation bridge requires Stage 1 selection, omits the required base artifact argument, and accepts `validation_select`; the bridge computes only the reference-depth residual and recordwise `B_nm`, not the selected two-residual, separate-maxima, signed-`G_j` endpoint. Current launcher authentication could not be tested. |
| Awaiting scientific authorization | **Yes.** This report is not permission to train, calibrate, open validation records, or evaluate outcomes. |

Initialization recommendation: **Require an explicitly approved provenance clarification.** Do not regenerate merely because the old identity field was mislabeled. First supply the immutable artifact package and clarify that the historical checkpoint contains a repeatedly interpolated deployed policy head, an untrained recurrent map, and an untrained value head. The reported 98.71% is the best epoch's candidate-head training accuracy, not the final serialized policy's accuracy. If the owner intended a fully supervised evaluator or requires measured competence of the serialized deployed policy, clarification is insufficient and a separately approved regeneration would be necessary. Otherwise the artifact may still be scientifically acceptable as an actor-only train-set warm start after all byte, manifest, procedure, and source bindings pass.

## 1. Repository state

### Starting and ending state

| Repository | Canonical branch | Supplied remote HEAD | Starting local HEAD / branch | Ending local HEAD / branch | Worktree |
| --- | --- | --- | --- | --- | --- |
| Paper | `main` | `3588f8c97bcdcc6cd9a16277da00752c8a3a6314` | same / `main` | same / `main` | clean at start and end |
| Implementation | `full-implementation` | `9c773a89095fdcb6bca9b5efb3e0ca5a9884ebc0` | same / `full-implementation` | same / `full-implementation` | clean at start and end |

Both local revisions exactly match the owner-supplied canonical revisions. Fresh remote resolution was attempted at both the start and end. Both attempts exited 128 because this host could not resolve `github.com`; remote identity is therefore not independently reverified. See `logs/02_initial_remote_heads.log` and `logs/19_final_remote_heads.log`.

The paper README still names `iclr-evidence-aligned-revision` as canonical (`README.md:3-6`). That is stale documentation relative to the owner's explicit instruction that `main` is canonical. `main.tex` remained the scientific source of truth.

No repository file was modified. End-state `status --porcelain` was empty in both repositories. All four requested source/config/documentation `git diff --check` and staged checks exited 0. Their exact invocations and output are in `logs/18_final_local_state_and_diff_check.log`. Starting state is in `logs/01_initial_local_state.log`.

Status and diff inspection was restricted to repository metadata and the supplied source/configuration/documentation pathspecs. It did not inspect protected dataset or result payloads.

## 2. Actual local evidence state

`UPI_TRM_EVIDENCE_ROOT` is unset (`logs/03_evidence_root_env.log`). The registered dataset path `/home/buiksat/trm_bellman/data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1` does not exist (`logs/04_registered_dataset_root_stat.log`). Registration records a location and hashes; it is not evidence that bytes are locally available.

The paper handoff names two historical Stage 0 engineering roots. Both are absent now, including the named build manifest, execution manifest, and failed authorization location (`logs/17_documented_evidence_paths.log`). No unrelated directory was searched.

| Item | Classification | Evidence |
| --- | --- | --- |
| Experiment 0 checkpoint | Evidence unavailable | No configured/documented path resolved; exact reported SHA was absent from authorized source text. |
| Experiment 0 identity document | Evidence unavailable | No path resolved. |
| Producer command/logs | Execution reported only | Producer source and historical commit exist. No corresponding local command log was found through configured/documented paths. |
| Producer source manifest | Evidence unavailable | Reported commit exists locally, but the execution-time manifest is absent. |
| Procedure manifest | Code supports it | Current and historical producer freeze the procedure in source. No artifact-side manifest was available. |
| Base-policy amendment | Evidence unavailable | The committed v2 protocol still says `base_policy_artifact.status=unavailable` and `stage1_execution_allowed=false`; the amendments directory has only `theory_bridge_v2.json`. |
| Current-source runtime authorization | Evidence unavailable | No configured evidence root or documented current-HEAD authorization resolved. |
| Launcher/package identity | Code supports verification | The Buck target and real-artifact test exist. No mapped `fbcode//buiksat_trm` package exists in the available fbsource checkout, so no current result was produced. |
| Train-only throughput/calibration | Code supports it; evidence unavailable | The authenticated train-only path exists, but no current or historical result artifact is locally available. The handoff explicitly reports that the old calibration was not run. |
| Fresh learned checkpoints or prior bridge access | Evidence unavailable | No configured evidence location was available. Absence of a marker is not evidence that validation was historically unopened. |

No validation, bridge, test, checkpoint, or scientific metric payload was opened or hashed.

## 3. Initialization admission

### Identity table

| Identity | Reported | Observed | Method | Status / evidence location |
| --- | --- | --- | --- | --- |
| Producer revision | `c99440f9ff3d2befc0778080d29962884ea44150` | Commit exists locally; subject is `Add the train-only Sudoku base-policy producer` | `git show` on the explicitly authorized historical revision | **Verified as source availability**, not execution provenance. `/home/buiksat/trm_bellman` Git object database. |
| Checkpoint SHA-256 | `49ca43d8d92a7716fe1e4626a47d012f7dd51d07f6602fe6ad0430dacc8be414` | No artifact path or bytes | Configured/documented-path inventory only | **Unverified.** |
| Checkpoint size | Not supplied | No artifact path or bytes | None possible | **Unverified.** Required before deserialization. |
| Model-state SHA-256 | `0d18f790d95e4c10269e73a333dd6de9e90dac1f558278cea5709b77190e77c4` | No payload | None possible | **Unverified.** |
| Reported architecture/model-config SHA-256 | `35ff3fc852a85c650691ee55cdaf68c6a42f66d1d505bd16eb4424c76f1e79ac` | Current registered architecture-block digest is `588866a66a831e0a8a60711df1dc316cc5ea733e2b8674c147ea832d88691f3e` | Canonical JSON SHA-256 of `protocol.json["architecture"]` | **Clarification required.** The reported value cannot be the registered architecture-block digest. It may be the historical concrete model-config digest, but payload bytes are absent. See `logs/05_registered_digests.log`. |
| Procedure digest | Not supplied by owner | `2fc00268c380f2a471b64f8315b380ae787cdd18986bfd61d3e295ba126440b3` for the source-frozen procedure | Canonical JSON SHA-256 of `TRAINING_PROCEDURE` | **Code-derived only.** Must match artifact and amendment. `logs/05_registered_digests.log`. |
| Dataset identities | Procedure says all 1,024 train records | Registered train count and hashes exist; local dataset and artifact bindings absent | Protocol metadata only | **Registration verified; execution binding unverified.** |
| Procedure contents | Seed 1904261137; 40 epochs; Adam; LR .003; batch 64; one pass over 1,024 puzzles; zero RL; no early stop; final budget; no held-out | Current producer source matches these fields | Source inspection | **Code supports it.** Actual execution remains unverified. `scripts/policy_improvement_base_policy.py:48-94`. |
| Owner approval | Reported as scientifically accepted | No signed/frozen owner approval located | Configured/documented-path inventory | **Unverified and still required.** |

### What the reported producer actually trains

At historical commit `c99440f...`, the producer creates the ordinary legacy `RLConfig`, builds `model`, constructs `UPITrmTrainer`, calls `imitation_pretrain`, and serializes `model.state_dict()`. The relevant trainer source has not changed between that commit and current HEAD.

The executed parameter path is:

`RLConfig` -> `policy_model_candidate.edit_policy` parameters -> imitation Adam -> gradient clipping at inherited `policy_grad_clip=1.0` -> per-minibatch parameter interpolation of `policy_model_old.edit_policy` toward candidate with inherited `mixture_alpha=0.1` -> serialization of `model`, which aliases `policy_model_old` in legacy mode.

Specific consequences:

- `_freeze_policy_backbone` leaves only the candidate `edit_policy` trainable (`rl/upi_trm_trainer.py:833-870`). The recurrent map and value head receive no imitation optimizer update.
- The producer does not override `mixture_alpha` or `policy_grad_clip`; historical defaults are 0.1 and 1.0 (`rl/config.py` at the historical revision, lines 163-168 and 305-306).
- Every imitation minibatch clips the candidate policy-head gradient and then calls `_sync_policy_old_towards_candidate` (`rl/upi_trm_trainer.py:3290-3369`). That synchronization is parameter interpolation, not an exact probability mixture (`rl/upi_trm_trainer.py:764-800`).
- The serialized object is `model.state_dict()` (`scripts/policy_improvement_base_policy.py:466-491`), so it contains the interpolated deployed head, not the final candidate head.
- `imitation_pretrain` returns final-epoch average loss but the maximum candidate minibatch training accuracy seen over all epochs (`rl/upi_trm_trainer.py:3514-3548`). `final_imitation_accuracy=98.71%` therefore does not measure the serialized policy head, recurrent map, value head, Sudoku solve rate, or held-out performance.

This does not by itself invalidate actor-only supervised initialization. It does invalidate descriptions implying that the whole evaluator was supervised, that candidate weights were serialized, or that 98.71% authenticated competence of the serialized base.

### Historical identity ambiguity

At `c99440f...`, the producer called the canonical digest of the full concrete `model_config` `architecture_sha256` in both checkpoint and identity document. Current source deliberately separates:

1. protocol registered architecture block: digest `588866...`;
2. concrete model configuration: historical/checkpoint field `architecture_sha256`, plus current identity field `model_config_sha256`;
3. serialized model state: reported `0d18f7...`;
4. checkpoint file: reported `49ca43...`;
5. procedure and producer source: digest `2fc002...` and commit `c99440f...`.

The correction in current source does not retroactively make the old identity document a current-format identity, and the old artifact must not be relabeled as produced by current HEAD.

### Loader scope

Current restoration is correctly ordered and fail-closed where it applies:

- Before deserialization, `authenticate_base_policy_artifact` validates the amendment schema/status/kind, canonical non-symlink regular path, size, stable inode metadata, and whole-file SHA-256 (`scripts/policy_improvement_base_policy_restore.py:101-190`).
- Restricted `torch.load(..., weights_only=True)` then checks payload schema/kind, model-state/procedure/producer identities, embedded model configuration and its digest, exact live model configuration, and the state-dict digest (`:193-275`).
- `apply_base_policy_state` performs strict state loading and rehashing (`:278-305`).
- The full backend applies this state before `build_trainer`, which creates old/candidate/target snapshots and optimizers (`policy_improvement_full_backend.py:1283-1319`).

The loader does not independently prove the producer command, source-manifest contents, dataset bytes/order used during production, procedure execution, lack of held-out access, optimizer membership, or meaning of the health statistic. Those remain admission checks against the identity document, manifests, and producer logs.

### Recommendation and unresolved conditions

**Require an explicitly approved provenance clarification.** Required inputs:

1. exact canonical checkpoint and identity-document paths;
2. checkpoint size and read-only stable-file status;
3. base-policy amendment binding the protocol architecture digest separately from the concrete model-config digest;
4. producer source manifest bound to `c99440f...`;
5. procedure and dataset manifests, including ordered 1,024-record train identity;
6. producer command/log correspondence and explicit no-validation/no-test evidence;
7. owner acceptance of actor-only, lagged-interpolation initialization and the corrected meaning of 98.71%.

With those inputs, perform opaque byte/size authentication first, then restricted deserialization with the current restoration functions, then independent provenance checks. No retraining or checkpoint rewrite is presently justified by the evidence available here.

## 4. Mathematical object to implementation and validation map

| Object/invariant | Current implementation | Validation evidence | Readiness |
| --- | --- | --- | --- |
| Frozen base policy | Fixed-base trainer creates a separate `policy_model_old`; exact mode freezes it. Base restoration happens before trainer construction. | Restoration tests were inspected but blocked by missing Torch. Source supports ordering. | Conditional |
| Shared frozen recurrent map | Candidate backbone is synchronized from the frozen source; fixed-base value optimizer accepts only `value_head.*`; candidate optimizer accepts policy-head parameters. | Target/value ownership test was blocked by missing Torch. | Conditional |
| Stored augmented state `(x,y,z,h)` | Persistent collection initializes latent once, stores pre-unroll latent, and carries post-unroll latent only for nonterminal successors. Remaining clock is part of replay. | Listed persistent carry/clock tests were blocked by missing Torch. Synthetic bridge rejects a backend that changes carry with `m`; passed. | Partial |
| `U_2` / `U_8` as endpoint-only functions | Bridge calls depth `n` and `m` endpoints while asserting trajectory, action probabilities, carried successor latent, and recurrent transition are independent of `m`. | Synthetic persistent-`F_m` rejection passed. Production trajectory path not exercised. | Partial |
| Direct self-bootstrap residual at `U_8` | For `K=1`, current bridge computes current-policy expectation of depth-8 action values before absolute residual (`scripts/policy_improvement_theory_bridge_v2.py:1017-1023`). | Synthetic bridge tests pass. | Supported |
| Direct self-bootstrap residual at `U_2` | Not computed or represented. | No test. | Missing |
| Separate-maxima aggregation | Current row is `D_nm + residual_m/(1-gamma)` and summaries take maxima of those recordwise sums (`:1099-1103`, `:1182-1185`). Required quantity is `max D_nm + max residual_m/(1-gamma)`. | No required regression test. | Incorrect for Experiment 1A |
| Signed `G_j` and equal-seed aggregate | `_summary` rejects negative metrics and result schema has no `R_2`, `B_j`, signed `G_j`, paired-seed mean, or percentile bootstrap. | No coverage. | Missing |
| Target lag | Existing bridge keeps target-network residual/lag concepts distinct in its registered schema, but Experiment 1A needs these only as secondary diagnostics alongside both direct residuals. | Source inspection only. | Adapter needed |
| Centering and trainer parity | Bridge separately checks constructed roundoff, trainer centering defect, and tensor parity. It preserves the trainer estimator rather than substituting another estimator. | Two listed synthetic tests passed. | Supported synthetically |
| Exact mixture | Bridge reconstructs `(1-alpha)pi + alpha*pi_candidate` and rejects exact methods when deployed probabilities differ. | Additional inspected synthetic rejection test passed. | Supported synthetically |
| Terminal folding/bootstrap mask | Backend/trainer source uses a common terminal mask and no terminal bootstrap, including budget exhaustion. | Requested clock/budget test was blocked by missing Torch. | Source-supported, unexecuted |
| Finite-population scope | Existing bridge labels output `finite_population_diagnostic_only` and `uniform_certificate=false`. | Source inspection. | Supported |

## 5. Verified blockers

| Severity | Type | Repository/SHA and location | Consequence | Minimal correction |
| --- | --- | --- | --- | --- |
| Critical | Missing evidence | Implementation `9c773a...`; configured evidence root and dataset location | Checkpoint bytes, size, state/config/procedure/dataset/source bindings, logs, current runtime authorization, and throughput evidence cannot be authenticated. | Owner supplies exact canonical paths under an approved evidence root. Do not search or infer them. |
| Critical | Protocol incompatibility | `configs/policy_improvement_v2/protocol.json`; `registry.json` | Protocol remains Stage-0-only with unavailable base. Existing n=2/K=1 Stage 1 rows are three pilot seeds, 10k/20k/40k/80k selection runs on `validation_select`, not eight independent 10k reduced-study runs on `validation_bridge`. | Add an explicit reduced-study registration/amendment and owner approval. Preserve existing Stage 1-3 gates. |
| Critical | Data-access defect | `policy_improvement_full_backend.py:1016-1270` | `_build_session` materializes train plus the evaluation split before model/trainer construction. Truncation to 10k does not make it train-only. Calling it would violate this task's no-held-out boundary. | Add a registered train-only session path that loads all 1,024 train records and no evaluation split, then separately authorize the frozen-checkpoint diagnostic consumer. |
| Critical | Bridge loader defect | `scripts/policy_improvement_theory_backend_v2.py:1263-1344` | Validation backend calls `load_registered_full_run(... require_stage1_selection=True)` without required `base_policy_artifact`; it also requires a `validation_select` selected row. It cannot load the reduced study. | Add reduced-study-specific authenticated loading and base-artifact plumbing. Do not loosen global selection gates. |
| Critical | Endpoint mismatch | `scripts/policy_improvement_theory_bridge_v2.py:1017-1103,1182-1185`; schema v2 | Only depth-8 residual is computed. `B_nm` is a recordwise sum, summaries are nonnegative, and no signed `G_j` exists. This cannot answer the selected primary question. | Add both self-bootstrap residuals, separate population maxima, signed `G_j`, and seed-level aggregation/bootstrap under a new reduced-study schema. |
| High | Extra unauthorized condition | `scripts/policy_improvement_theory_bridge_v2.py:1050-1062`; `scripts/policy_improvement_theory_schema_v2.py:26` | Any current exact-method `validation_bridge` route automatically runs paired returns at alpha 0.05, 0.1, 0.2. The selected study authorizes alpha 0.1 only. | Register an Experiment 1A diagnostic mode with only authorized conditions and separately frozen MC settings. |
| High | Provenance ambiguity | Historical producer `c99440f...`; current `scripts/policy_improvement_base_policy.py:390-395,495-503` | Historical `architecture_sha256` named the concrete model config, while current amendment expects the protocol architecture block. Blindly copying the reported hash into an amendment will fail or misstate identity. | Preserve both digests under distinct fields and explicitly document historical semantics. |
| High | Health-statistic ambiguity | `rl/upi_trm_trainer.py:3290-3369,3448-3548` | Reported 98.71% is best candidate training accuracy, while checkpoint serializes interpolated old policy. It cannot establish serialized-policy competence. | Owner clarifies accepted scope; if serialized competence is required, approve a separate train-only measurement or regeneration protocol before outcomes. |
| High | Test/runtime environment | Python 3.12.14+meta lacks `torch` and `pydantic`; fbsource lacks mapped `fbcode//buiksat_trm` | Production restoration, persistent mechanics, target ownership, clock/terminal test, and real launcher identity were not executed. | Run the same tests in the existing compatible fbcode build environment after mapping the exact inspected source. Do not install or repin during this audit. |
| Medium | RNG incompatibility | `scripts/policy_improvement_theory_bridge_v2.py:443-466` | Return/Bellman RNG includes whole checkpoint SHA. A common base-return bank is not automatically shared across independently trained checkpoints, even when base actor and recurrent kernel match. | Freeze a new evaluation RNG contract keyed to the exact invariants intended for sharing, independent of training seed/checkpoint and comparison depth, or do not reuse the bank. |
| Medium | Documentation drift | Paper `README.md:3-6` | README names the historical feature branch as canonical; owner says `main`. | Documentation correction may be included later, outside this read-only task. |

## 6. Fresh validation

Environment: host `devvm3231.hil0.facebook.com`, `/usr/local/bin/python3`, Python `3.12.14+meta`, `/usr/local/bin/buck2`. Environment details are in `logs/20_environment.log`. All Python commands used `-B` with `PYTHONDONTWRITEBYTECODE=1`, task-local temporary/cache/Matplotlib directories, and no environment changes.

| Command | Outcome | Exit | Log | Scope/limitation |
| --- | --- | ---: | --- | --- |
| Requested producer/config/restoration unittest selection | 9 tests passed; procedure freeze test import failed on missing `pydantic`; restoration module import failed on missing `torch` | 1 | `logs/10_base_policy_tests.log` | Synthetic contracts only; did not authenticate Experiment 0. |
| Requested three Algorithm 2 boundary tests | All three failed at module import because `torch` is absent | 1 | `logs/11_algorithm2_tests.log` | No claim for persistent carry, target ownership, or budget-terminal behavior from current execution. |
| Requested five bridge tests | 5 passed, no skips | 0 | `logs/12_theory_bridge_tests.log` | Synthetic checkpoint bytes and backend only. Covers centering/parity, miscentering rejection, endpoint-only persistent carry rejection, population ordering, and pre-backend held-out refusal. |
| Additional exact-mixture rejection test | 1 passed, no skips | 0 | `logs/16_exact_mixture_test.log` | Synthetic backend only; proves the interface rejects mismatched deployed probabilities. |
| `buck2 root` from implementation checkout | Not a Buck project | 1 | `logs/13_buck_root_probe.log` | Standalone checkout cannot run Buck target. |
| Mapped fbsource package checks | Available fbsource resolves to `/data/repos/fbsource`; `fbcode/buiksat_trm` is absent | 2 | `logs/14_fbsource_mapping_check.log`, `logs/15_fbsource_mapping_check_resolved.log` | The real-artifact launcher identity test was **blocked and not run**. No platform/build-mode skip is counted as a pass. |

Exact Python test commands are recorded as shell traces at the start of each log. The requested Buck command was not launched because its package does not exist in the available Buck project:

```bash
buck2 test --local-only @fbcode//mode/opt \
  fbcode//buiksat_trm:test_policy_improvement_launcher_identity
```

Coverage gaps requiring regression tests in the next task:

- real restoration occurs before all snapshots and optimizers, with common actor/recurrent identities and frozen-base ownership;
- production persistent trajectory keeps pre-unroll stored latent and depth-2 post-unroll successor carry independent of comparison depth;
- terminal reward folding and bootstrap masking for STOP, solved, and budget exhaustion;
- direct residuals at both q=2 and q=8 use current-policy action expectation and the same deployed transition;
- aggregation uses two separate maxima and preserves negative `G_j`;
- eight seed units are weighted equally and bootstrapped as paired units with exactly 10,000 resamples;
- reduced training never resolves or materializes validation/test data;
- reduced diagnostics do not invoke extra alpha conditions;
- production bridge receives and verifies the admitted base artifact.

## 7. Resource and protocol readiness

No authenticated train-only throughput result is available locally. Historical handoff text reports an older A100 host and build attempt, but explicitly says throughput calibration was not run. It is **execution reported**, not a current comparable measurement. No speculative runtime is provided.

Current code has a genuinely train-only throughput constructor: `_build_throughput_session` loads only the train split, sets evaluation episodes/interval to zero, records effective configuration and runtime/dataset identities, and reports interactions, recurrent-map applications, setup/training elapsed time, compute accounting, device, utilization, and memory (`policy_improvement_full_backend.py:692-1010`). Automatic caps are 256 and 1,024 interactions for every registered method. The optional 4,096 tier requires separate authorization (`scripts/policy_improvement_throughput.py:45-48,568-724`). It excludes evaluation rollouts, diagnostics, checkpoint writing time, audit time, and return estimation. It also initializes randomly rather than using the Experiment 0 artifact, so it is useful for mechanics but not a complete Experiment 1A cost estimate.

Separate resource quantities still needed are:

- 10,000 checker-reward training interactions for `fixed_base_exact_persistent`, including recurrent-map applications and checkpoint cost;
- exact action enumeration at q=2 and q=8 over 128 initial augmented states and K=1 successors;
- independently seeded current-policy Monte Carlo returns and their frozen rollout count/horizon;
- initialization authentication/restoration overhead;
- diagnostic packaging, aggregation, bootstrap, and audit overhead.

### PROPOSED — NOT EXECUTED

After current-HEAD package mapping and runtime authorization exist, the smallest existing bounded train-only calibration is the automatic tier. The parser itself fixes caps at 256 and 1,024 and executes all registered methods; no flag can narrow it to one method. The exact launcher form is:

```bash
/ABS/PATH/phase4_runtime_launcher.par \
  --purpose policy-improvement-throughput \
  --runtime-archive /ABS/PATH/policy_improvement_full.par \
  --expected-runtime-sha256 <FULL_RUNTIME_SHA256> \
  --source-project-root /home/buiksat/trm_bellman \
  --expected-source-git-commit 9c773a89095fdcb6bca9b5efb3e0ca5a9884ebc0 \
  --runtime-authorization /ABS/PATH/runtime_authorization_v3.json \
  --expected-runtime-authorization-sha256 <AUTHORIZATION_SHA256> \
  --dataset-root /home/buiksat/trm_bellman/data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1
```

Proposed cap: **1,024 automatic interactions per registered method**, preceded by the built-in 256 sample. Do not add `--authorize-4096-tier`. This command is not currently runnable because the dataset, current runtime PAR, launcher PAR, and runtime authorization are unavailable. It is not a substitute for a reduced-study-specific calibration once the new initialization/restoration path exists.

Common base-return-bank reuse is not ready. Identical base policy, recurrent transition, initialization law, horizon, rewards, terminal masks, and evaluation semantics are necessary. They are not sufficient under the current seed contract because `_seed` includes the whole checkpoint SHA. Actor/kernel identity must not be conflated with whole-checkpoint identity. The owner must either approve a new depth- and training-checkpoint-independent evaluation RNG contract or decline bank reuse.

Unresolved owner decisions, all pre-outcome:

1. initialization admission and the actor-only/lagged-head clarification;
2. explicit Experiment 1A registration and evidence namespace;
3. eight seeds versus the allowed pre-outcome three-seed fallback;
4. exact run seeds or complete derivation and analysis RNG;
5. Monte Carlo return seed namespace, rollout count, horizon, and calibration definitions;
6. common-return-bank policy;
7. resource fallback and operational time/storage/accelerator budgets;
8. retry, hardware-failure, and partial-run rules;
9. final scientific execution and validation-access authorization.

## 8. Smallest next implementation task

Implement one authenticated **Experiment 1A vertical slice**, without changing the existing Stage 1 selection program:

1. Add a new reduced-study protocol/amendment/registry namespace that binds the admitted base artifact, `fixed_base_exact_persistent`, n=2, K=1, m=8, gamma=.99, alpha=.1, projection radius 10, contraction disabled, all 1,024 train records, exactly 10,000 interactions, budget-final checkpoint, eight frozen seeds or the explicitly chosen three-seed fallback, and `validation_bridge` only.
2. Extract or add a train-only production session builder that restores the admitted base before `build_trainer` and never resolves an evaluation split. Keep current selection/authentication gates unchanged for existing rows.
3. Add a reduced-study bridge adapter that authenticates the budget-final checkpoint and base artifact without `require_stage1_selection`, then opens only the registered `validation_bridge` population after training is sealed and access is authorized.
4. Extend a new result schema/aggregator to compute q=2 and q=8 direct self-bootstrap residuals under the same deployed depth-2 kernel, `max D`, `max residual_2`, `max residual_8`, signed `G_j`, equal-seed mean, and the 10,000-resample paired-seed percentile interval. Keep target lag, centering/parity, mixture identity, deployment mismatch, and persistent-state checks as separate fields. Do not invoke the old three-alpha return diagnostic.

Likely existing files to change:

- `scripts/policy_improvement_full_runtime.py`
- `policy_improvement_full_backend.py`
- `scripts/policy_improvement_theory_backend_v2.py`
- `scripts/policy_improvement_theory_bridge_v2.py`
- `scripts/policy_improvement_theory_schema_v2.py`, or preferably a new Experiment 1A schema module
- `phase4_runtime_launcher.py`
- `phase4_runtime_profile.py`
- `BUCK`
- focused unit tests under `tests/`

Proposed new paths, not existing files:

- `configs/policy_improvement_exp1a/protocol.json`
- `configs/policy_improvement_exp1a/registry.json`
- `configs/policy_improvement_exp1a/amendments/base_policy.json`
- `scripts/policy_improvement_exp1a_schema.py`
- `scripts/policy_improvement_exp1a_bridge.py`
- `tests/test_policy_improvement_exp1a_unittest.py`

The task depends on owner decisions 1 through 5 above. It must include the missing regression tests listed in Section 6. Passing synthetic interfaces is not enough; at least one production-backend fixture must prove restore-before-derivation and train/validation access separation.

## 9. Access and execution statement

Read:

- the requested paper source, README, and handoff;
- the requested implementation source/configuration/registration files and narrowly necessary dependencies/tests;
- Git metadata and the explicitly authorized historical producer commit;
- registration metadata, not registered record contents;
- filenames/stat metadata for explicitly documented old evidence paths, which were absent;
- task-local logs created by this audit.

Deliberately not read:

- validation, `validation_bridge`, or test record contents;
- scientific metric/result payloads or existing bridge results;
- any checkpoint payload, because no checkpoint path was available;
- unrelated home/evidence directories;
- any identity metadata inseparable from a scientific result payload.

During the readiness audit, no scientific training, Experiment 0 retraining, calibration, diagnostic rollout, Monte Carlo return evaluation, held-out opening, result reconstruction, source/manuscript/configuration edit, branch operation, commit, push, PR, permission change, or `TEST_OPEN.json` creation occurred. The audit created only the external task directory and its logs/report. This repository copy was added afterward at the owner's request.
