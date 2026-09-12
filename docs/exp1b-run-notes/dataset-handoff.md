# UPI-TRM — dataset reproduction attempt

Date: 2026-09-11

Standalone repo `/home/buiksat/trm_bellman` at `d43b2c7`, clean before and after.
fbsource package `/data/repos/fbsource/fbcode/buiksat_trm` at HG `c595211465c4`.

## Verdict

**The dataset content reproduces bit-exactly. The dataset MANIFEST digest cannot be
reproduced on this machine.** All six content digests — three split manifests and three
ordered-record digests — match the protocol exactly. `MANIFEST.json`'s
`2572bb79…` embeds the producer attestation, and two of its four fields are build-artifact
hashes that are not derivable from source. **No dataset was published and Task 2 was not
started**, since Task 2 is gated on exact equality.

---

## 1. How the builder was driven, and what the attestation actually requires

`policy_dataset_builder_entrypoint.py` calls `preflight_runtime(attestation_required=True,
allowed_phase4_roles={"policy-dataset-builder"})`. The only sanctioned driver is
`phase4_runtime_launcher` with `--purpose policy-dataset-builder`. Good news on
authorization: unlike the policy-improvement roles, this purpose needs **no externally minted
runtime authorization** — the launcher derives `launcher_sha256` locally from
`_launcher_artifact_sha256()` (`phase4_runtime_launcher.py:1717-1720`). So the attestation is
in principle satisfiable here without an owner artifact.

**But the launcher cannot accept the PARs this build shape produces.**
`validate_runtime_archive` (`confirmatory_runtime_launcher.py:548-593`) opens the runtime
archive with `ZipFile` and refuses anything else: *"Runtime artifact is not a valid ZIP-based
PAR."* Under `@fbcode//mode/dev-nosan` both PARs are inplace bootstrap scripts, not zips:

```text
policy_dataset_builder.par   first bytes: #!    zipfile.is_zipfile -> False
phase4_runtime_launcher.par  first bytes: #!    zipfile.is_zipfile -> False
```

Producing a standalone zip PAR needs a different package style, which is outside the build
shape you fixed (`no extra -c`). **I did not bypass, stub, or disable the attestation, and I
did not publish a dataset outside it.**

What I did instead, to answer the reproducibility question without publishing anything: a
read-only diagnostic that calls the builder's **pure generation functions** (`_generate_split`,
`materialized_split_manifest`, `ordered_sha256`) in a scratch directory under `/tmp`, run
inside the `policy_dataset_builder` PAR's own runfiles tree because the standalone checkout has
no numpy. It writes no dataset, touches neither repository, and does not call
`build_dataset` or `verify_dataset` — the two functions the attestation guards.

---

## 2. Digest-by-digest results

### Content — every one matches

| target | protocol value | produced | verdict |
| --- | --- | --- | --- |
| train split manifest | `05146037857b1adb42520e80a0c2ab250053a517196c8b8ac95e002aa40c6f74` | identical | **MATCH** |
| train ordered_record | `73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446` | identical | **MATCH** |
| validation split manifest | `a4b1bffb92f9c7c1ebe7baf9dcaaed83191f9c6249bec0887ed8ff7fb6b4a937` | identical | **MATCH** |
| validation ordered_record | `9257ae46c71fa24afd8c0284af6087c5662faffd724b395037c9016ac4110380` | identical | **MATCH** |
| test split manifest | `bf14acfc94e580bb3678102729f88432fda599610b2ca2578949f8c4cbc775bd` | identical | **MATCH** |
| test ordered_record | `94647c77419ef6ec150aa1dde6d07de1a3f9112c2fb768168404a520138ef6ac` | identical | **MATCH** |

Counts and seeds as registered: train 1024/26081401, validation 256/26081402, test
512/26081403, generated in that order with cross-split symmetry rejection.

**Independent cross-check.** The committed `configs/policy_improvement_v2/populations.json`
`validation_bridge` population already carries binding `2a3fc56085…ccfb644` and ordered_record
`ba724676…f0a78a42`, both matching the exp1b protocol. All **128 of its record digests match
the reproduced validation split at their recorded indices**. The regenerated data is
definitively the registered corpus, not merely a plausible one.

### Dataset manifest — blocked

| target | protocol value | verdict |
| --- | --- | --- |
| dataset `manifest_sha256` | `2572bb79faeec976dc83cb75b8520e59691a7c9dc3f8fe252554fc29bfe90ccd` | **NOT PRODUCED** |

`MANIFEST.json` embeds `producer_source` verbatim
(`dataset/build_policy_improvement_4x4.py:757-767`), so its digest is a function of the
producer attestation, not of the data alone.

### The four producer-attestation fields

| field | recorded | reproducible here? |
| --- | --- | --- |
| `git_commit` | `7317d7011c31ac622c0723d22cf9f4bb0571bdf1` | **yes** — the commit exists; a detached worktree supplies it |
| `source_manifest_sha256` | `5833f7cb1884651015d66901a31ed78cbf620826ee5035bed4192e5e5590f6b1` | **yes — reproduces exactly at `7317d701`** |
| `launcher_sha256` | `8e73d60512934705f8a295fb67845f5a90b8c6f3006e1ea6ebcb373881c76d6a` | **no** — dev-nosan artifact is `04a3726b718425f567242b56ee0a09704d714f6f6cd891ab1c8070d8afadff31` |
| `runtime_sha256` | `8ded72fa7dc774caad11e62ad10e62604f3393e483109a7c1c2d51d2bbd5ebe8` | **no** — dev-nosan artifact is `4214ea426b3916e83d289a9cffbc6c571272767a9c01d4679b3dd57608715902` |

Computing `source_manifest_sha256` correctly took two attempts and the correction matters:
it is **not** the digest of `configs/iclr_confirmatory/producer_source_manifest.json`, and not
`canonical_json_bytes` of a regenerated manifest. It is
`_manifest_sha256(POLICY_DATASET_BUILDER_SOURCE_PROFILE, {path: sha256(file)})` over the 16
profile paths (`phase4_runtime_profile.py:869`). Under that definition the recorded value
reproduces at the producer commit to the character. At HEAD it is
`3ca5a880c982e7336211a803722171d762e2c19df0e62e24cb9af4cbd87efa1a`.

---

## 3. Diagnosis

**The builder-code difference between `7317d701` and HEAD is real but irrelevant to
generation.** `git diff` shows 95 insertions / 28 deletions in
`dataset/build_policy_improvement_4x4.py`, and every hunk is inside `verify_dataset` — it adds
`verify_content_splits` so a caller can authenticate all three split manifests without opening
the test split's files. `_generate_split`, the canonicalization scheme,
`materialized_split_manifest`, `ordered_sha256`, and `build_dataset`'s manifest assembly are
untouched. That is why the content reproduces at HEAD. It was the right first thing to check
and it is cleared.

**The blocker is the manifest schema, not a code defect.** `MANIFEST.json` binds the corpus to
`producer_source`, and two of those four fields are **hashes of built PAR artifacts**. PAR
artifact bytes depend on the buck2 version, platform toolchain, and packaging style at build
time. They are not a function of the source tree, so no checkout of `7317d701` can regenerate
them a year later on a different toolchain. The design intent is clear and defensible — the
manifest asserts *which executable produced this corpus* — but the consequence is that the
registered dataset manifest digest is **not reproducible from source by construction**.

Two independent obstacles, either of which alone is fatal to exact equality here:

1. The recorded `launcher_sha256` / `runtime_sha256` describe **ZIP-based standalone PARs**.
   This build shape produces inplace PARs, which the launcher refuses outright. So the
   sanctioned publish path cannot even be exercised, let alone matched.
2. Even with zip PARs, their hashes would have to match byte-for-byte artifacts built at
   `7317d701` in the original environment. Nothing retains those artifacts — I searched both
   trees and found no archived `.par`/`.xar`.

I did **not** edit the protocol, and did not publish a dataset carrying a HEAD attestation. A
corpus with correct content but manifest digest ≠ `2572bb79…` would be rejected by Stage A
anyway, since the exp1b protocol pins `dataset_manifest_sha256`; writing one would have created
a misleading artifact under the registered root.

---

## 4. Task 2 — not started

Correctly gated: Task 1 did not achieve exact digest equality, so
`scripts/policy_improvement_base_policy.py` was not run and no checkpoint exists. None of the
ten `base_policy_artifact` identity fields can be reported. The protocol's
`base_policy_artifact.status` remains `unavailable` with `stage1_execution_allowed: false`,
unchanged.

Your selection is recorded and unambiguous for whoever picks this up: one authenticated
competent **train-only** artifact shared across persistent and episodic, per the protocol's own
stated preference in `#/base_policy_interpretations`; not the random-base stress regime. It
must be train-only and not selected by validation or test.

---

## 5. New filesystem paths

**None in either repository.** Both are byte-for-byte as I found them.

- `/home/buiksat/trm_bellman`: `git status --porcelain -uall` = 0 entries, HEAD still `d43b2c7`.
  The only directory under `data/` is the pre-existing `data/iclr-confirmatory-sudoku4x4-v1`,
  which I did not touch and did not substitute.
  `data/policy-improvement-v1-owner/` was **not** created.
- fbsource: `hg status` shows zero entries outside `fbcode/buiksat_trm/`; nothing added.
- A detached git worktree at `/tmp/upi-wt7317` was created to read `7317d701` and has been
  removed; `git worktree list` shows only the main tree.
- Scratch outside both repos, safe to delete: `/tmp/upi-dsdiag/` (regenerated splits, ~3 MB),
  `/tmp/exp1b-parprobe/*.py` (probes), `/tmp/vb_records.json`.

**On `.gitignore`:** line 54 is `data/*`, so had the dataset been written to
`data/policy-improvement-v1-owner/…` it would already be ignored. Checkpoints are covered too
(`*.pt`, `*.ckpt`, `checkpoints/`). No `.gitignore` change is needed when this does run.

---

## 6. Buck commands

`@fbcode//mode/dev-nosan`, no `--local-only`, no extra `-c`, no rebase. `--trace-id` on every
log query.

| command | result | trace | CUDA matches |
| --- | --- | --- | --- |
| `build …:policy_dataset_builder` | exit 0, 2.9s, 27 local / 0 remote | `5fef4ce0-f218-47a2-8950-3d31d4b28b20` | 0 |
| `build …:phase4_runtime_launcher` | exit 0 (warm) | — | 0 |

No `nvcc`, Caffe2, CUB, or `.cu` compilation. Nothing was waited out.

---

## 7. What remains before the experiment can start

The dataset is recoverable; its registered provenance envelope is not. This needs an owner
decision, and it is a small one:

**Option A — supply the original corpus.** If the `7317d701`-era
`data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1` still exists anywhere
(backup, another host, the original build machine), copying it in resolves everything with no
protocol change. Its content will match what I reproduced; I have verified that independently.

**Option B — re-register the manifest digest under a fresh producer identity.** Rebuild the
corpus here through the sanctioned launcher and update
`configs/policy_improvement_v2/protocol.json#/dataset/manifest_sha256` and
`#/dataset/producer_source` to the new build's values. The six content digests stay exactly as
registered — that is the scientifically meaningful part and it is unchanged — while the
provenance envelope names the machine that actually produced the bytes. **This is an owner
edit to the protocol; I did not make it, per your instruction that the recorded digests are the
specification.** It also requires resolving obstacle (1): a ZIP-based standalone PAR, i.e. a
package style outside the build shape you fixed.

**Either way, still outstanding afterwards:**

1. The Experiment 0 base policy (Task 2) and its ten identity fields.
2. The owner-signed Experiment 1B admission with distinct producer and evaluator
   authorizations bound to final artifact hashes.
3. Distinct clean producer and evaluator roots and an owner-controlled fresh evidence
   generation.
4. `RUN_UPITRM_FULL_EXPERIMENTS=1`, and validation-bridge authorization only after all eight
   checkpoints are sealed.

**One thing worth flagging for the longer term.** The manifest schema binds corpus identity to
PAR artifact hashes, which makes the registered dataset unreproducible from source by design.
If reproducibility from source is a property you want the protocol to have, the fix is to
separate the two: keep a content-only corpus digest as the acceptance criterion and record the
producer artifact hashes alongside it as provenance rather than folding them into the same
digest. The six content digests already give you exactly that — they matched today, a year
and many commits later, on a different machine.

---

# ADDENDUM — 2026-09-11, acting on the delegated-authority decision

Your reasoning on recording the true producer is right and I am not pushing back on it. Two of
the four decisions have a false premise, though, and I stopped rather than spend the deadline
on choices you would reverse. Everything below is verified, not predicted.

## Status by decision

| decision | status |
| --- | --- |
| 1. standalone zip PAR | **blocked** — needs a BUCK attribute change, which then breaks decision 2 |
| 2. publish via launcher | **verified working up to the archive format**; one packaging step away |
| 3. re-verify published corpus | not reached |
| 4. surgical protocol edit | **not executable as scoped** — it is a document-graph re-registration |
| 5. base policy | blocked behind 4 |

Both repositories are exactly as I found them. `git status --porcelain -uall` = 0 at
`d43b2c7`; no `data/policy-improvement-v1-owner/`; both `BUCK` files still
`5d26d16f294d755cb18e19089c845b26e793a1fc255492339d86d8ee08e3b917`.

## Decision 1 — package style is not a `-c` knob here

`-c` has no effect. `buck2 cquery --output-attribute` shows the target resolving to
`package_style: "inplace"`, `par_style: "live"` under every candidate key:

```text
-c fbcode.par_style=xar                -> "package_style":"inplace","par_style":"live"
-c python.par_style=xar                -> "package_style":"inplace","par_style":"live"
-c python.package_style=standalone     -> "package_style":"inplace","par_style":"live"
-c fbcode.package_style=standalone     -> "package_style":"inplace","par_style":"live"
```

A build with `-c python.package_style=standalone` (trace
`bd614bba-2c50-46c0-9002-eeb3615f63d3`, 4.5s, 0 local / 0 remote / 4404 other actions, **0
CUDA matches**) produced byte-identical artifacts to the plain dev-nosan build: both PARs
still 1226/1234-byte bootstrap scripts, `zipfile.is_zipfile` false, same SHA-256.

So the style is a **target attribute**. The authorized change expressed correctly is
`package_style = "standalone"` on the two `python_binary` targets in `BUCK`.

**And that is what breaks decision 2.** `authorize_phase4_source_profile`
(`phase4_runtime_profile.py:838-841`) requires the source checkout to be at the expected
commit with `git status --porcelain=v1 --untracked-files=all` **empty**. Editing `BUCK` in the
standalone repo makes it dirty, so the launcher refuses. `BUCK` is also one of the 22 reviewed
paths that must stay byte-identical between trees.

I considered and rejected the workarounds. Editing `BUCK` only in fbsource and reverting after
the build would record a `runtime_sha256` for a PAR built from a `BUCK` state present in
neither tree — unbound-artifact provenance, the exact thing this slice exists to reject.
Building from a detached worktree works mechanically but has the same smell.

**The clean route needs one thing from you: commit the two-target `package_style` change.**
Then the checkout is clean at a real commit, the PAR is built from committed source, and every
recorded hash is honest.

## Decision 2 — the sanctioned path works; only the format blocks it

I drove the real launcher end to end:

```text
phase4_runtime_launcher.par --purpose policy-dataset-builder   --runtime-archive .../policy_dataset_builder.par   --expected-runtime-sha256 4214ea426b3916e83d289a9cffbc6c571272767a9c01d4679b3dd57608715902   --source-project-root /home/buiksat/trm_bellman   --expected-source-git-commit d43b2c7f7e2c569e5c80f9a3b9f83ac2ab69827b   build --owner-root <abs> --output-root <abs>

exit 2
Phase 4 runtime rejected: Runtime artifact is not a valid ZIP-based PAR.
```

This is the good outcome. It accepted the purpose, the argument grammar, both absolute paths,
the source project root, the **clean-checkout authorization at d43b2c7**, and the source
profile. The only rejection is the archive format. No externally minted authorization was
needed, exactly as established. One packaging step and this publishes.

## Decision 4 — not a surgical edit; it re-registers the document graph

Two findings, both hard.

**(a) The dataset block is pinned in production code.**
`scripts/policy_improvement_v2_schema.py:56` defines `IMMUTABLE_DATASET_V1`, a hard-coded copy
of the entire dataset block including `manifest_sha256` *and* all four `producer_source`
fields. Line 555 enforces it:

```python
if dataset != IMMUTABLE_DATASET_V1:
    raise PolicyImprovementV2SchemaError(
        "V2 must reuse the exact immutable v1 dataset registration."
    )
```

Editing only the two protocol JSONs makes the v2 protocol **fail its own validator**, which
gates study authentication, the launcher's protocol digest, and the whole exp1b admission
chain. Updating the constant is a production source change, which the standing constraint
forbids.

**(b) The v2 protocol digest is the anchor of the registered document graph.**
Its canonical digest is `9df68e4ec8c732948dda9d8497fd1764d29cc0e04f9924a94e5f1bf4375a196a`.
Changing any byte of that file changes the digest, which is pinned in four further documents:

```text
configs/policy_improvement_v2/amendments/theory_bridge_v2.json
configs/policy_improvement_v2/registry.json
configs/policy_improvement_exp1b/protocol.json          (parent.protocol_sha256 — exact match)
configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json
```

Each of those has its own digest pinned downstream: the exp1b registry pins the exp1b
protocol, the amendment pins protocol and registry, and the exp1b slice's tests assert all
three (`67a149f6…`, `54f37fc8…`, `466a5c07…`). So the edit cascades into the artifact set that
just completed ten review rounds and a PASS.

### Every reference to `2572bb79…`, as requested

Identical in both trees. Three of the five are outside your enumerated list.

| # | location | in your list? | kind |
| --- | --- | --- | --- |
| 1 | `configs/policy_improvement_v2/protocol.json` `#/dataset/manifest_sha256/value` | **yes** | config |
| 2 | `configs/policy_improvement_exp1b/protocol.json:114` `#/training_population/dataset_manifest_sha256` | **yes** | config |
| 3 | `configs/policy_improvement_v1/protocol.json:42` `#/dataset/manifest_sha256/value` | **no** | config, separate registered protocol |
| 4 | `scripts/policy_improvement_v2_schema.py:62` `IMMUTABLE_DATASET_V1` | **no** | **production source** |
| 5 | `tests/test_policy_improvement_base_policy_unittest.py:227` | **no** | test literal |

Plus `tests/test_policy_improvement_v2_unittest.py:211`, which asserts the protocol equals the
constant and passes automatically once 1 and 4 agree.

### The honest change-set

If you want this done, it is:

1. `configs/policy_improvement_v1/protocol.json` — dataset manifest + producer_source
2. `configs/policy_improvement_v2/protocol.json` — same
3. `scripts/policy_improvement_v2_schema.py` — `IMMUTABLE_DATASET_V1` **(production source)**
4. `tests/test_policy_improvement_base_policy_unittest.py:227` — literal
5. Re-derive the v2 protocol digest and update the four documents that pin it
6. Re-derive the exp1b protocol/registry/amendment digests and the three literals the exp1b
   tests assert
7. `configs/policy_improvement_exp1b/protocol.json` — `dataset_manifest_sha256`
8. Re-run the exp1b Buck target (309) and the v1/v2 suites

No content digest is touched anywhere in that list: the three split manifests, three
ordered_record values, the population binding, the 128 record digests, counts, and seeds all
stay exactly as registered. Only the producer envelope and the digests that transitively wrap
it change.

## What I need from you

Two authorizations, and I can finish in one pass:

1. **Commit the two-target `package_style = "standalone"` BUCK change** (or tell me to carry it
   uncommitted in fbsource only and accept the provenance caveat). This unblocks 1, 2, 3.
2. **Authorize the change-set above**, specifically the production-source edit to
   `IMMUTABLE_DATASET_V1` and the re-derivation of the exp1b document digests. This unblocks 4
   and 5.

If you would rather not re-register the graph so close to the deadline, there is a narrower
alternative worth considering: leave every registered digest alone and add the regenerated
corpus as a **second** registered dataset entry under its own name, leaving
`policy-improvement-hard-4x4-v1` as the historical record. That avoids touching the exp1b chain
entirely, at the cost of a protocol addition rather than an edit. I have not done this either —
it is your call.

## Provenance note, ready to insert once authorized

Drafted for `#/dataset` so the record is legible in six months:

> `regeneration_note`: The corpus was regenerated on 2026-09-11 at commit
> `d43b2c7f7e2c569e5c80f9a3b9f83ac2ab69827b` from the registered generation seeds, because the
> original corpus at `7317d7011c31ac622c0723d22cf9f4bb0571bdf1` no longer exists on any
> available machine. All six content digests — the three split manifests and the three
> ordered_record values — and all 128 `validation_bridge` record digests were verified
> identical to the original registration. Only `manifest_sha256` and `producer_source` changed,
> because `MANIFEST.json` embeds the producer attestation and two of its four fields are hashes
> of built PAR artifacts, which are not reproducible from source. The scientific content is
> unchanged and proven so.

## New paths

None. Nothing was created in either repository. Scratch outside both, safe to delete:
`/tmp/upi-dsdiag/`, `/tmp/exp1b-parprobe/*.py`, `/tmp/vb_records.json`, `/tmp/*.log`.

`.gitignore` confirmed: line 54 `data/*` covers the dataset root; `*.pt`, `*.pth`, `*.ckpt`,
and `checkpoints/` cover checkpoint outputs. No `.gitignore` change will be needed.

---

# ADDENDUM 2 — 2026-09-11, additive registration executed

Authorization 1 used, Authorization 2 correctly denied and not used. **The corpus is published
and content-verified. The exp1b repoint is done. The base policy is blocked by a pre-existing
defect I introduced two rounds ago.**

## Which case — answered before editing

**Case 2.** There is no mechanism. `apply_exp1b_admission`
(`scripts/policy_improvement_exp1b_schema.py:1342-1387`) resolves exactly two blocked slots,
`base_policy_artifact` and `runtime_authorization`, plus `execution_gate` and `amendments`. It
never touches `training_population`, and it hard-binds to the protocol revision
(`exp1b_document_sha256(protocol) != admission.protocol_sha256` → refuse). An admission cannot
repoint the dataset.

One thing beyond your stated bound, flagged rather than assumed: `_training_population`
validates `dataset_name` and `dataset_root` against **module constants**
`DATASET_NAME`/`DATASET_ROOT` at `scripts/policy_improvement_exp1b_schema.py:164-165`. Repointing
requires editing those. That file is exp1b-slice code we wrote, not v1/v2 and not pre-existing
production, so I read it as inside "our own new slice" and proceeded.

## Authorization 1 — standalone PARs

`-c` cannot do it; `package_style` is a target attribute. Added
`package_style = "standalone"` to the two `python_binary` targets and committed as
**`6212b166cc74f51858e838d3c9ded7fbf402b1b9`** (`BUCK` only; mirrored to fbsource).

Build trace `133a816b-9450-4b08-adb2-c7bff55cfb82`: 18.1s, 12 local / 0 remote /
**1842 cached**, **0 CUDA matches**. Result — both artifacts are now real ZIPs:

```text
policy_dataset_builder.par    zip=True  16,921,689 B  ded2ae94841a75f0ca410fce9c2aabbaa8f40431321494f5c17b0151063d968e
phase4_runtime_launcher.par   zip=True     809,871 B  43656ae2f49b8e5f133f4f51009eafbdeb2c48b7b2b2797e17b5f671a6b7ca1d
```

## Sanctioned publish — succeeded

```text
phase4_runtime_launcher.par --purpose policy-dataset-builder \
  --runtime-archive <builder.par> --expected-runtime-sha256 ded2ae94… \
  --source-project-root /home/buiksat/trm_bellman \
  --expected-source-git-commit 6212b166cc74f51858e838d3c9ded7fbf402b1b9 \
  build --owner-root .../policy-improvement-v1-owner \
        --output-root .../policy-improvement-hard-4x4-v1-regen-20260911
exit 0
```

`build_dataset` ran under the attestation and internally called `verify_dataset` on the staged
corpus before publishing. A separate `verify` invocation through the same launcher also
returned exit 0.

New registration: `manifest_sha256` =
**`55bf615ebeaf0f13bad1917dc74fd33f7c9736b7316a0ff15572641cb6470b2a`**, producer_source =
git `6212b166…`, launcher `43656ae2…`, runtime `ded2ae94…`, source_manifest `3ca5a880…`.

## Post-publish verification — from the corpus on disk

| digest | registered value | published corpus | verdict |
| --- | --- | --- | --- |
| train split manifest | `05146037…c6f74` | identical | **MATCH** |
| train ordered_record | `73110263…ad446` | identical | **MATCH** |
| validation split manifest | `a4b1bffb…b4a937` | identical | **MATCH** |
| validation ordered_record | `9257ae46…110380` | identical | **MATCH** |
| test split manifest | `bf14acfc…c775bd` | identical | **MATCH** |
| test ordered_record | `94647c77…8ef6ac` | identical | **MATCH** |
| 128 validation_bridge record digests | `populations.json` | all 128 match at their indices | **MATCH** |

Read straight off `manifests/*.json` and `validation/records.json` in the published tree, plus
the launcher's own `verify` output. Not from a scratch diagnostic. No content digest moved.

## exp1b documents re-derived, and why

Only these. Nothing in v1, v2, or pre-existing production source.

| file | change |
| --- | --- |
| `scripts/policy_improvement_exp1b_schema.py` | `DATASET_NAME`, `DATASET_ROOT` → regen entry (validator pins the protocol against them) |
| `configs/policy_improvement_exp1b/protocol.json` | `training_population.dataset_name`, `.dataset_root`, `.dataset_manifest_sha256` |
| `configs/policy_improvement_exp1b/registry.json` | `protocol_sha256` → new protocol digest |
| `configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json` | `protocol_sha256`, `registry_sha256` |
| `configs/policy_improvement_exp1b/REGENERATION.md` | **new** — the provenance note |

Digest re-derivation:

| document | old | new |
| --- | --- | --- |
| exp1b protocol | `67a149f6…e536d77` | `add935bac37aa23578cd0586aa8bbed769172b65380fae6d6ccb7af52371a063` |
| exp1b registry | `54f37fc8…80782f63` | `a4441a1758dbc61baed4d87263e593019ae415f4ecab586c549feef643457116` |
| exp1b amendment | `466a5c07…9ca21da4` | `0f79b1b8ff1098ce4d48cfa196f2a69220f5dfe01f7ac66437c2cd3c93a747a2` |

No test literals needed changing — the exp1b tests compute these digests dynamically. A repo
sweep confirms zero residual references to the old three. `training_population.count`,
`split_manifest_sha256`, `ordered_record_sha256`, `population_id`, `split`, and
`resolves_evaluation_data` are untouched; `source` still cites the v2 train split, which
remains true.

**exp1b suite after the repoint: 312 tests, OK (15 skipped).**

## Base policy — blocked by a pre-existing defect, and it is mine

`scripts/policy_improvement_base_policy.py:286` calls `authorize_phase4_training_source`, which
proves the checkout is clean at the exact commit *and* that the committed producer source
manifest matches the tree. It fails:

```text
Phase4RuntimeProfileError: Producer source manifest differs from the checkout bytes.
```

**This is not caused by anything in this task.** It reproduces identically from a clean
worktree at `d43b2c7`, before my BUCK commit, and `BUCK` is not in the training profile. The
cause is exactly one stale entry out of 83:

```text
stale entries: 1
    phase4_runtime_profile.py
```

I changed `phase4_runtime_profile.py` in review round 8 — adding
`configs/policy_improvement_v2/populations.json` to the exp1b audit profile — and never
regenerated `configs/iclr_confirmatory/producer_source_manifest.json`. Nothing in the exp1b
suite exercises `authorize_phase4_training_source`, so ten review rounds and 309 Buck tests did
not catch it. It breaks the Experiment 0 producer and any other training-source-authorized
path.

**Fix, one command plus your commit:**

```bash
buck2 run @fbcode//mode/dev-nosan fbcode//buiksat_trm:generate_producer_source_manifest \
  -- --root /home/buiksat/trm_bellman
# then commit configs/iclr_confirmatory/producer_source_manifest.json
```

I did not run it: it rewrites a committed file, and the artifact must then be produced from a
clean commit, which needs your commit authority. All ten base-policy identity fields remain
unreported.

## New paths

| path | ignored? |
| --- | --- |
| `data/policy-improvement-v1-owner/` (new owner root, 0700) | yes — `.gitignore:54 data/*` |
| `data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1-regen-20260911/` (1.1 MB: `MANIFEST.json`, `build_config.json`, `identifiers.json`, `manifests/{train,validation,test}.json`, `train/`, `validation/`, `test/`) | yes |
| `configs/policy_improvement_exp1b/REGENERATION.md` | **no — tracked, intended for commit** |

`git check-ignore -v` confirms `data/*` covers both the corpus and a future
`data/base-policy-exp0-20260911/` checkpoint root. `*.pt`, `*.pth`, `*.ckpt`, `checkpoints/`
also present.

Working tree at `6212b16`: 4 modified tracked files and 1 untracked, all listed above, all
yours to commit. Both trees byte-identical across the 22 reviewed paths plus `REGENERATION.md`.

## Buck commands, all with --trace-id

| command | trace | result | CUDA |
| --- | --- | --- | --- |
| build 2 targets (standalone) | `133a816b-…` | 18.1s, 12 local, 1842 cached | 0 |
| build `policy_improvement_base_policy` | `12051d5d-…` | 3:20, 133 local, 62 remote, **193,711 cached** | see below |

That last build showed 17,340 `nvcc|caffe2|cub|.cu` matches and I checked before continuing:
`what-ran --skip-cache-hits` leaves **290 rows, of which zero are CUDA**, and the five executed
`cxx_compile` actions are PAR scaffolding. All 193,711 were cache hits — Torch came down
prebuilt. The guard was not tripped and nothing was waited out.

Incidentally this settles the earlier cache-count puzzle: a build that genuinely pulls the
Torch closure for the first time in a session reports ~193k cached actions, matching your
cold-daemon 195,791. The zeros in previous rounds were warm-daemon DICE memoization, where the
actions are not looked up at all.

## What remains

1. **Commit the exp1b repoint** (4 modified + `REGENERATION.md`).
2. **Regenerate and commit `configs/iclr_confirmatory/producer_source_manifest.json`** — the
   one-line fix above. Unblocks the base policy.
3. Re-run the exp1b Buck target after 1 and 2, and re-sync.
4. Then the base policy, and its ten identity fields.
5. Then the owner-signed Experiment 1B admission, distinct producer/evaluator roots, a fresh
   evidence generation, and `RUN_UPITRM_FULL_EXPERIMENTS=1`.

---

# ADDENDUM 3 — producer source manifest regenerated

`buck2 run @fbcode//mode/dev-nosan fbcode//buiksat_trm:generate_producer_source_manifest -- --root /home/buiksat/trm_bellman`
— exit 0, trace `dea3dcee-ea65-4156-891d-de931f75f73f`, **0 CUDA rows** with cache hits
excluded.

Exactly the predicted one-line change, nothing else:

```diff
 "phase4_runtime_profile.py":
-  "946986e7c3d4dd71dcaa10bdefbfb593da861350bc0ab6bd0651d827da46829c"
+  "e634d0b6a6518a33b295c57448c14c3fa900e01255648675ddb93deabe24601b"
```

`configs/iclr_confirmatory/producer_source_manifest.json`:
`a1ef6ed0ac5fb945c1d8ea28ee3f49b1aa2ea323febbe81373f7f18395496600` →
**`9ba92a2c179c91b1a0300a5a3a0c38c39111b17d3414d98581e79ef624918097`**.
1 file changed, 1 insertion, 1 deletion. Uncommitted, awaiting your commit. The new value is
the current `phase4_runtime_profile.py`, matching the 22-path table above.

## Regression: needs infrastructure, not a handful of lines

The check itself is two lines — compare the committed manifest against
`build_producer_source_manifest(root)`. The cost is making it real in a hermetic runfiles tree:
the manifest covers **83 source files**, and a Buck test can only hash files its target
declares as resources. Without them the test either no-ops (the capability-skip antipattern the
review series spent a round eliminating) or fails under Buck for the wrong reason.

Worse, the 83-path list is not an exported constant — it is derived inside
`build_producer_source_manifest`, so a BUCK `resources` block would be a hand-maintained second
copy of the very list whose drift caused this bug.

Recommendation: skip it. The cheap variants do not catch this failure — asserting the manifest's
key set matches the profile catches added/removed entries, not the content drift that actually
happened. If you want it later, the honest shape is a resources glob covering the profile plus
~15 lines of test, and it should live next to the tool rather than in the exp1b suite.

## Cache-count question: closed by observation

No longer a hypothesis. The `policy_improvement_base_policy` build (trace `12051d5d-…`) was the
first this session to pull the Torch closure and reported **193,711 cached actions**, matching
the 195,791 from the cold-daemon probe. Earlier builds reported `Cached actions: 0` because a
warm daemon serves unchanged DICE nodes without ever performing an action-cache lookup — those
actions are not actions in that build. `what-ran --skip-cache-hits` removing zero rows on those
builds is consistent with the same explanation. **Confirmed by observation; treat
`Cached actions` as meaningless on a warm daemon and judge build shape with
`what-ran --trace-id --skip-cache-hits` instead.**
