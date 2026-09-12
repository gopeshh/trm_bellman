# UPI-TRM Experiment 0 — train-only base policy

Date: 2026-09-11

Standalone repo `/home/buiksat/trm_bellman` at **`dc4fd9ecbe42f2b94d82d027c1c8141a6e82f3a2`**,
clean before and after. fbsource package at HG `c595211465c4`.

## Status

**Artifact produced. Nine of the ten required identity fields are present in the producer's
output; the tenth, `training_data_sha256`, is absent under that name.** Its truthful value
exists in the document under a different key and the sanctioned consumer maps it. I did not
write the field into anything. Details in §4 — this is the one thing needing your decision.

## 1. How it was driven

The producer requires a clean checkout at the exact commit and refuses otherwise, so it ran
against the main repo at `dc4fd9e` with no local edits, through the Buck-built PAR:

```
buck2 build @fbcode//mode/dev-nosan fbcode//buiksat_trm:policy_improvement_base_policy
policy_improvement_base_policy.par \
  --project-root        /home/buiksat/trm_bellman \
  --expected-git-commit dc4fd9ecbe42f2b94d82d027c1c8141a6e82f3a2 \
  --dataset-root        .../policy-improvement-hard-4x4-v1-regen-20260911 \
  --output-root         /home/buiksat/trm_bellman/data/base-policy-exp0-20260911
```

Documented procedure followed in order: `--print-procedure`, then `--health-check`, then the
real run.

**One refusal, reported verbatim as required.** The first attempt passed the short commit:

```text
Phase4RuntimeProfileError: Expected producer commit must be 40 lowercase hex characters.
```

That is my input error, not a defect — the full 40-character SHA works. No other refusal
occurred; in particular `authorize_phase4_training_source` now passes, confirming the
`dc4fd9e` manifest fix resolved the blocker.

## 2. Training configuration actually used

Frozen in the producer and covered by `training_procedure_sha256`
`2fc00268c380f2a471b64f8315b380ae787cdd18986bfd61d3e295ba126440b3`:

| setting | value |
| --- | --- |
| kind | `oracle_imitation_pretraining` |
| objective | `cross_entropy_to_oracle_cell_fill_action` |
| seed | `1904261137` (distinct from every registered protocol seed) |
| imitation epochs | 40 |
| demonstration repeats | 1 (oracle is deterministic; repeats add no information) |
| demonstration episodes | 1024 → **7,148 demonstrations collected** |
| batch size / optimizer / lr | 64 / adam / 0.003 |
| early stop | `null` — disabled |
| **RL environment interactions** | **0** — no policy-improvement objective, so no method is favoured |
| stop action mode | `terminal` |
| training split / count | `train` / 1024 |
| checkpoint selection | `final_budget_only` |
| validation_data_used / test_data_used | `false` / `false` |
| shared across persistent and episodic | `true` |

Neutral shared config, asserted identical across all four registered method configs
(`fixed_base_exact_persistent`, `fixed_base_exact_episodic`,
`legacy_parameter_interpolation`, `matched_ppo`): `gamma 0.99`, `inner_unroll_n 2`,
`max_edits 16`, `C_max 16.0`, `target_Lv 1.0`, `target_Lz 0.9`, `latent_ball_radius 10.0`,
`latent_projection_mode enabled`, `enable_contraction false`, `reward_shaping true`,
`solve_terminal_reward 1.0`, `fail_terminal_reward -16.0`, `solved_threshold null`,
`disable_constraint_masking true`, `batch_size 32`, `task_name sudoku`.

Model: TRM backbone, `hidden_size 64`, `H_cycles 2`, `L_cycles 2`, `L_layers 1`,
`puzzle_emb_ndim 0`, `seq_len 16`, `vocab_size 6`, `rl_num_actions 97`,
`num_puzzle_identifiers 1792`, `forward_dtype float32`, rope. Device: **cuda**.

Learning curve: epoch 1 37.31% → epoch 6 94.29% → epoch 14 97.24% → epoch 30 98.36% →
**epoch 40 98.71%**, best 98.84%. The saved artifact is the **epoch-40 final-budget**
checkpoint, not the best-accuracy one, per `checkpoint_selection: final_budget_only`. Competent,
and selected by budget rather than by any observed score.

## 3. The ten required identity fields

From `data/base-policy-exp0-20260911/base_policy_identity.json`, in the protocol's declared
order.

| # | field | value |
| --- | --- | --- |
| 1 | `initialization_kind` | `train_only_pretrained` |
| 2 | `architecture_sha256` | `588866a66a831e0a8a60711df1dc316cc5ea733e2b8674c147ea832d88691f3e` |
| 3 | `model_state_sha256` | `0d18f790d95e4c10269e73a333dd6de9e90dac1f558278cea5709b77190e77c4` |
| 4 | `producer_git_commit` | `dc4fd9ecbe42f2b94d82d027c1c8141a6e82f3a2` |
| 5 | `producer_source_manifest_sha256` | `9ba92a2c179c91b1a0300a5a3a0c38c39111b17d3414d98581e79ef624918097` |
| 6 | `training_data_sha256` | **absent under this name — see §4** |
| 7 | `training_procedure_sha256` | `2fc00268c380f2a471b64f8315b380ae787cdd18986bfd61d3e295ba126440b3` |
| 8 | `checkpoint_sha256` | `607c3065aa429522c90b89763d215be49f3ee58bffd493ed9280c09f4d377fe7` |
| 9 | `shared_across_persistent_and_episodic` | `true` |
| 10 | `not_selected_by_validation_or_test` | `true` |

Supporting values the document also carries: `checkpoint_size_bytes` 6,669,709,
`model_config_sha256` `35ff3fc852a85c650691ee55cdaf68c6a42f66d1d505bd16eb4424c76f1e79ac`,
`initialization_sha256` `fd912050eacfadfc09c7ac0c0da13ea8a5a565504ccbf1e940889c55c969431c`,
`training_dataset_manifest_sha256` `55bf615e…` (the regenerated corpus),
`train_manifest_sha256` `05146037…`, `training_split_ordered_record_sha256` `73110263…`.

Note `architecture_sha256` (`588866a6…`) and `model_config_sha256` (`35ff3fc8…`) are distinct
digests; the health check reported only the latter, under the name `architecture_sha256`, which
is a cosmetic inconsistency in the health output and not in the artifact.

### Fields 9 and 10 are true, not merely asserted

The producer records `validation_data_opened: false` and `test_data_opened: false` as observed
facts of the run, and the frozen procedure sets `rl_environment_interactions: 0`,
`early_stop_accuracy: null`, and `checkpoint_selection: final_budget_only`. Nothing was selected
by any observed score, and no validation or test content was resolved, opened, hashed, or
materialized. Field 9 holds because the artifact is a single shared initialization with no
method-specific objective in its training.

## 4. The one gap: `training_data_sha256`

`scripts/policy_improvement_base_policy.py` **never emits that key** — `grep -c` returns 0. The
identity document instead carries three data digests under other names:
`training_dataset_manifest_sha256`, `train_manifest_sha256`, and
`training_split_ordered_record_sha256`.

The sanctioned consumer resolves the ambiguity. `scripts/policy_improvement_exp1b_runtime.py:580`
builds its provenance descriptor with:

```python
"training_data_sha256": str(
    getattr(base_policy, "training_split_ordered_record_sha256", "")
),
```

So the intended value is the **train split ordered-record digest**,
`73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446` — which is exactly the value
registered in `configs/policy_improvement_v2/protocol.json#/dataset/splits/train` and verified
identical in the regenerated corpus.

**I did not write that field into the identity document or anywhere else.** It is a
producer-output naming gap, and the owner's signed base-policy amendment is where the ten
fields get assembled. Two options for you:

1. **Assemble it in the amendment** using `73110263…`, citing the consumer mapping above. No
   code change; the value is truthful and independently verifiable.
2. **Fix the producer** to emit `training_data_sha256` directly, then re-run. That is a
   production source change plus a 52-minute re-run, and it would change nothing about the
   artifact bytes — `checkpoint_sha256` and `model_state_sha256` would be unchanged since the
   identity document is written after training.

A second, smaller naming mismatch to reconcile in the amendment: the producer writes
`initialization_kind: "train_only_pretrained"` while the exp1b provenance descriptor uses
`"train_only_pretrained_base_policy"`. The exp1b schema validates it only as a nonempty
identifier, so either string passes validation, but they should be made consistent when the
amendment is signed.

## 5. Where the artifact landed

```text
data/base-policy-exp0-20260911/
  base_policy.pt              6,669,709 bytes  mode 0400
  base_policy_identity.json       3,278 bytes  mode 0400
```

Both written read-only by the producer. **`.gitignore` covers it**: `git check-ignore -v`
reports `.gitignore:54  data/*` for `base_policy.pt`, and `*.pt`, `*.pth`, `*.ckpt`, and
`checkpoints/` are also present. `git status --porcelain -uall` is **0 entries** — the tree is
clean at `dc4fd9e` and nothing needs committing from this step.

New paths this step: `data/base-policy-exp0-20260911/` and its two files. Nothing else changed
in either repository; both trees remain byte-identical across the 22 reviewed paths.

## 6. Wall-clock cost

| step | cost |
| --- | --- |
| `--print-procedure` | instant |
| `--health-check` (1 epoch, disposable, saves nothing) | 117 s, predicted budget 2995 s |
| **full 40-epoch production run** | **3102 s = 51 min 42 s** |

Prediction was good: 2995 s predicted from one epoch, 3102 s actual, 3.6% over. The PAR build
was already warm from the previous session.

## 7. CUDA guard

Applied the judgement from last time rather than aborting on a raw count. The
`policy_improvement_base_policy` build (trace `12051d5d-853d-42c8-b23c-8b32abeb8087`) showed
17,340 `nvcc|caffe2|cub|.cu` matches across 194,001 `what-ran` rows. With cache hits excluded:
**290 rows, zero of them CUDA**, and the five executed `cxx_compile` actions are PAR
scaffolding. All 193,711 were cache hits; Torch came down prebuilt. The manifest-regeneration
run (trace `dea3dcee-…`) showed 0 CUDA rows with cache hits excluded. No genuine compile action
occurred and nothing was waited out.

## 8. What remains before the Experiment 1B run

1. **Decide the `training_data_sha256` question** in §4. Nothing else blocks on it.
2. **Owner-signed base-policy amendment** carrying the ten identity fields, which flips
   `configs/policy_improvement_v2/protocol.json#/base_policy_artifact` from
   `status: unavailable` / `stage1_execution_allowed: false`. Note this is a v2 protocol edit —
   the same file we deliberately did not touch — so it needs the same care, and it is the
   protocol's own documented transition rather than a re-registration.
3. **Owner-signed Experiment 1B admission** with distinct full and theory-bridge
   authorizations bound to final launcher/runtime/profile hashes, plus the base-policy
   checkpoint and model-state digests above.
4. **Distinct clean producer and evaluator checkout roots** at the admitted commit, and an
   owner-controlled fresh evidence generation on a filesystem with the required `flock`,
   hard-link, rename, and directory-`fsync` semantics.
5. `RUN_UPITRM_FULL_EXPERIMENTS=1`, and validation-bridge authorization only after all eight
   checkpoints are sealed.
6. Standing decisions, unchanged: whether target lag stays record-only or gets a registered
   refusal tolerance, and whether to fund the three placeholder Torch tests. The third of those
   is still the only thing that will establish whether the 1e-6 trainer-estimator centering
   tolerance holds on real float32 heads.

Also still open from earlier rounds, deliberately deferred: no hermetic regression asserts the
committed producer source manifest matches the checkout, because a BUCK resources glob would
duplicate the 83-path list whose drift caused that bug.
