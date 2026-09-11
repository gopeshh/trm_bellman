# Why Experiment 1B trains on `policy-improvement-hard-4x4-v1-regen-20260911`

Date: 2026-09-11

## Short version

The corpus Experiment 1B trains on is a **regeneration** of the registered
`policy-improvement-hard-4x4-v1`. The data is identical — every content digest was verified
equal after publication. Only the record of *which executable built it* differs, and it differs
because the original corpus no longer exists anywhere and the manifest digest cannot be
reproduced from source. Nothing about the science changed.

## What happened

`configs/policy_improvement_v2/protocol.json#/dataset` registers a corpus at
`data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1`, built on
2026-08 at commit `7317d7011c31ac622c0723d22cf9f4bb0571bdf1`. That directory does not exist on
any machine we have. It was regenerated on 2026-09-11 from the registered generation seeds
(train 1024/26081401, validation 256/26081402, test 512/26081403) through the sanctioned
`phase4_runtime_launcher --purpose policy-dataset-builder` path, with `build_dataset` and
`verify_dataset` both running under the runtime attestation.

## Why it is a new entry rather than a repair of the old one

`MANIFEST.json` embeds `producer_source` — the git commit, the launcher PAR's SHA-256, the
runtime PAR's SHA-256, and the source-profile manifest digest — so the top-level
`manifest_sha256` is a function of *which executable produced the corpus*, not only of the data.
Two of those four fields are hashes of built PAR artifacts. PAR bytes depend on the buck2
version, the platform toolchain, and the packaging style at build time, so they are not
derivable from the source tree and cannot be reproduced a year later on a different machine.

Of the four fields, two do reproduce exactly at the original commit: `git_commit`, and
`source_manifest_sha256` = `5833f7cb1884651015d66901a31ed78cbf620826ee5035bed4192e5e5590f6b1`.
The two PAR hashes do not, and nothing retains the original artifacts.

The registered `policy-improvement-hard-4x4-v1` entry in the v1 and v2 protocols is therefore
left **exactly as it is**, as the historical record of the original build. The regenerated
corpus is registered separately, under its own name and root, recording the producer that
actually built it. Forging the original producer identity would have been an unbound-artifact
substitution — precisely what this slice's authentication chain exists to reject.

## What was verified identical

All six content digests, checked against the published corpus on disk, not against a scratch
computation:

| split | manifest sha256 | ordered_record sha256 |
| --- | --- | --- |
| train (1024) | `05146037857b1adb42520e80a0c2ab250053a517196c8b8ac95e002aa40c6f74` | `73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446` |
| validation (256) | `a4b1bffb92f9c7c1ebe7baf9dcaaed83191f9c6249bec0887ed8ff7fb6b4a937` | `9257ae46c71fa24afd8c0284af6087c5662faffd724b395037c9016ac4110380` |
| test (512) | `bf14acfc94e580bb3678102729f88432fda599610b2ca2578949f8c4cbc775bd` | `94647c77419ef6ec150aa1dde6d07de1a3f9112c2fb768168404a520138ef6ac` |

Every value is character-for-character the one registered in
`configs/policy_improvement_v2/protocol.json#/dataset/splits`.

Additionally, all **128 `validation_bridge` record digests** in
`configs/policy_improvement_v2/populations.json` match the published validation split at their
recorded indices. The population `binding_sha256`
(`2a3fc56085873990d174398dce4118fcd1fe0217d09ce1e392bb9d522ccfb644`) and `ordered_record_sha256`
(`ba724676b172248219918afb7073a07f08adc9b74f57c3395bfda609f0a78a42`) are unchanged and were not
touched.

Counts, generation seeds, the symmetry-canonicalization scheme, and the cross-split overlap
rejection are all unchanged.

## What differs

Only the corpus envelope:

| field | original registration | regenerated entry |
| --- | --- | --- |
| dataset name | `policy-improvement-hard-4x4-v1` | `policy-improvement-hard-4x4-v1-regen-20260911` |
| `manifest_sha256` | `2572bb79faeec976dc83cb75b8520e59691a7c9dc3f8fe252554fc29bfe90ccd` | `55bf615ebeaf0f13bad1917dc74fd33f7c9736b7316a0ff15572641cb6470b2a` |
| `producer_source.git_commit` | `7317d7011c31ac622c0723d22cf9f4bb0571bdf1` | `6212b166cc74f51858e838d3c9ded7fbf402b1b9` |
| `producer_source.launcher_sha256` | `8e73d60512934705f8a295fb67845f5a90b8c6f3006e1ea6ebcb373881c76d6a` | `43656ae2f49b8e5f133f4f51009eafbdeb2c48b7b2b2797e17b5f671a6b7ca1d` |
| `producer_source.runtime_sha256` | `8ded72fa7dc774caad11e62ad10e62604f3393e483109a7c1c2d51d2bbd5ebe8` | `ded2ae94841a75f0ca410fce9c2aabbaa8f40431321494f5c17b0151063d968e` |
| `producer_source.source_manifest_sha256` | `5833f7cb1884651015d66901a31ed78cbf620826ee5035bed4192e5e5590f6b1` | `3ca5a880c982e7336211a803722171d762e2c19df0e62e24cb9af4cbd87efa1a` |

The producer commit `6212b166` is one commit after `d43b2c7`; the only change between them is
building the launcher and dataset-builder as standalone ZIP PARs, which the launcher requires
in order to authenticate a runtime archive at all.

## Why the builder code differing from 2026-08 does not matter

`dataset/build_policy_improvement_4x4.py` changed between `7317d701` and the regeneration
commit — 95 insertions, 28 deletions. Every hunk is inside `verify_dataset`, adding
`verify_content_splits` so a caller can authenticate all three split manifests without opening
the test split's files. `_generate_split`, the canonicalization scheme,
`materialized_split_manifest`, `ordered_sha256`, and `build_dataset`'s manifest assembly are
untouched. That is why the content reproduces exactly.

## What this changed in Experiment 1B

`configs/policy_improvement_exp1b/protocol.json#/training_population` now cites the regenerated
entry: `dataset_name`, `dataset_root`, and `dataset_manifest_sha256`. Its `count`,
`split_manifest_sha256`, `ordered_record_sha256`, `population_id`, `split`, and
`resolves_evaluation_data` are unchanged, because the train split content is unchanged. Its
`source` still cites `configs/policy_improvement_v2/protocol.json#/dataset/splits/train`, which
remains true: that is where the split identity is registered, and the regenerated corpus
reproduces it exactly.

`DATASET_NAME` and `DATASET_ROOT` in `scripts/policy_improvement_exp1b_schema.py` moved with it,
since the validator pins the protocol against those constants.

The exp1b protocol, registry, and amendment digests were re-derived as a consequence. Nothing
in `configs/policy_improvement_v1/`, `configs/policy_improvement_v2/`, or
`scripts/policy_improvement_v2_schema.py` was modified.
