# Policy-improvement v2 fresh evidence namespace

This directory registers `policy-improvement-v2-20260818`. It is a new
protocol, not an amendment that changes the meaning of v1 evidence. A v2
protocol, row, result, authorization, audit, or analysis document has a v2
schema name and cannot be accepted as v1.

`populations.json` freezes all pre-outcome record assignments used before the
test transaction. Stage 0 contains eight records selected from the 1,024-record
training split by the lexicographically smallest SHA-256 scores under
`upi-trm-policy-improvement-v2-stage0-smoke:`.

`train_minus_stage0_smoke` registers the deterministic complement of that
selection: the remaining 1,016 train records, in ascending original train
index order, with explicit indices, record hashes, input hashes, ordered
digests, and a binding digest. Stage 0 evaluates the eight frozen records and
trains on their registered complement. Both halves partition the train split
exactly, and reassembling them by index reproduces the registered
`split_ordered_record_sha256.train`.

This supersedes the earlier ambiguous statement that Stage 0 "evaluates the
registered training population", which read as though Stage 0 both trained and
evaluated on the same 1,024 records. It did, and that violated the schema-v5
train/evaluation disjointness invariant enforced by `upi_trm_train`. Evaluation
is unchanged: the same eight records with the same binding digest. Only the
training pool is now stated explicitly. The general disjointness checks are
unchanged and Stage 0 has no exemption from them.

`train_minus_stage0_smoke` is bound to Stage 0 alone. The protocol registers it
under `training_populations`, only Stage 0 registry rows carry it in
`training_population`, and every other phase must leave that field unset.

The validation split is sorted
under `upi-trm-policy-improvement-v2-validation-partition:`. Its first 128
records are `validation_select`; its remaining 128 records are
`validation_bridge`. Original indices, record hashes, input hashes, ordering
scores, ordered digests, and population binding digests are explicit. The two
validation populations are disjoint and exhaustive.

Stage 0 is systems-only. Its four rows train on `train_minus_stage0_smoke` and
evaluate on the disjoint `stage0_smoke` population, both inside the train
split. Stage 0 cannot affect a method, `n`, `K`, alpha, base policy, or
stopping decision. A Stage 0 solve rate is not paper evidence.

The immutable base registry contains 139 rows:

- 4 Stage 0 rows;
- 24 exact-method Stage 1 screen rows;
- 9 alpha-selection templates;
- 6 separate baseline-readiness templates;
- 32 Stage 2 confirmatory templates;
- 64 Stage 3 ablation templates.

The Stage 1 screen includes only `fixed_base_exact_persistent` and
`fixed_base_exact_episodic`. All 24 method, `n`, `K`, and seed rows reach
10,000 interactions at alpha 0.1. Within each latent mode, the registered
interaction-matched endpoint on `validation_select` selects one `(n,K)` pair.
Ties use smaller `n`, then smaller `K`. Only those two configurations may
continue to 20,000, 40,000, and 80,000 interactions. The 80,000-interaction
choice prefers persistent on a tie, then smaller `n`, then smaller `K`.

Legacy interpolation and matched PPO are not duplicated across inapplicable
`K` or alpha factors. After the exact configuration is selected, the six
baseline-readiness templates materialize one row per baseline and pilot seed.
They cannot enter exact-method or alpha selection.

`base_policy_artifact` is deliberately unavailable. Stage 1 must fail closed
until the protocol binds one authenticated artifact or the user explicitly
chooses the random-base stress-test interpretation. The practical
interpretation uses one competent train-only artifact shared across persistent
and episodic treatments. The random-base interpretation tests one proposal
from a random policy and is not a practical safe-policy-improvement study.

The pre-outcome `theory_bridge_v2.json` amendment registers the read-only
bridge schema, primary reference depth 8, final-checkpoint exploratory depth
16, exact `K=1` masked summation, common-random-number `K=5` Monte Carlo,
predictive current-policy returns, fixed analyses, and the multi-fidelity
screen. Scientific bridge work uses only `validation_bridge` and cannot alter
selection. A Stage 0 bridge smoke uses only the Stage 0 training records.

V2 reuses the exact immutable `policy-improvement-hard-4x4-v1` dataset
registration. The dataset name, owner-relative root, corpus manifest,
producer identities, split manifests, and ordered split identities match the
v1 protocol byte-for-byte. V2 changes the experiment namespace and population
assignment only. Registering these existing hashes does not deserialize test
arrays or create a test-open transaction.

`protocol.json`, `populations.json`, and `registry.json` are canonical JSON.
Regeneration must reproduce each file byte-for-byte. No Stage 1-3 row is
executable while the base-policy artifact is unavailable, and Stage 2-3 also
require a separately authenticated `TEST_OPEN` transaction.
