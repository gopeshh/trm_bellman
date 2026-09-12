# Experiment 1B — recovery on a new machine

Written 2026-09-12. The machine this ran on (`devvm3231.hil0.facebook.com`) is
released 2026-09-13 10:00. Nothing under `/home/buiksat` survives that: not the
evidence tree, not the corpus, not the base policy, not the `/tmp` handoffs.
This file plus the git history is what you get. It is written so someone with
no context can rebuild everything.

ICLR 2027: abstract 2026-09-18, full paper 2026-09-25.

---

## 1. What state the work is in

Code is complete and reviewed. Eleven findings closed across ten review rounds
plus a reviewer PASS. The experiment was mid-run when the machine expired.

| Item | State |
| --- | --- |
| Experiment 1B slice, 22 paths | committed, `d43b2c7` |
| Regenerated corpus registration | committed, `ebb3321` |
| Base policy artifact | produced, **NOT in git** (6.4 MB, gitignored) |
| Runtime authorization, amendment, admission | signed, **NOT in git** (owner dir) |
| seed-0 | sealed |
| seeds 1-7 | were still training |
| Published result, audit | never reached |

**Nothing that follows requires a decision.** Every choice was made and the
reasoning is recorded here or in the referenced commit messages.

## 2. Commit map

```
5a0ad34  Seal with this backend's training module, and test the real body
ef24fa1  Package the Phase 4 training runtime as a standalone ZIP PAR
f2af25a  Rebind two launcher identity pins to the 2026-09-11 launcher
73c8780  Package the remaining authenticated runtimes as standalone ZIP PARs
dc4fd9e  Refresh the stale producer source manifest entry
ebb3321  Register the regenerated hard-4x4 corpus as its own dataset entry
6212b16  Build the launcher and dataset-builder as standalone ZIP PARs
d43b2c7  Add the Experiment 1B paired-residual vertical slice
```

Read the commit messages. They carry the reasoning, not just the change.

## 3. Rebuild order on a new machine

Each step is reproducible. Timings are measured on 2x A100 40GB.

### 3.1 Corpus — ~minutes

Regenerate, do not hunt for the original; it no longer exists anywhere.
Deterministic from the seeds recorded in
`configs/policy_improvement_v2/protocol.json#/dataset`:

| Split | Count | Seed | Target ordered_record |
| --- | --- | --- | --- |
| train | 1024 | 26081401 | `73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446` |
| validation | 256 | 26081402 | `9257ae46c71fa24afd8c0284af6087c5662faffd724b395037c9016ac4110380` |
| test | 512 | 26081403 | `94647c77419ef6ec150aa1dde6d07de1a3f9112c2fb768168404a520138ef6ac` |

Publish through the launcher with `--purpose policy-dataset-builder`. That
purpose needs no minted authorization; the launcher derives `launcher_sha256`
locally.

**`MANIFEST.json` will NOT reproduce, and that is expected.** It embeds the
producer attestation, two fields of which are hashes of built PAR artifacts.
The corpus is registered as its own entry,
`policy-improvement-hard-4x4-v1-regen-20260911`, precisely so the historical
`policy-improvement-hard-4x4-v1` registration in the v1/v2 protocols stays
untouched. See `configs/policy_improvement_exp1b/REGENERATION.md` and `ebb3321`.

**Do not edit `IMMUTABLE_DATASET_V1` in `scripts/policy_improvement_v2_schema.py`.**
It exists to refuse exactly that rewrite, and it was right to. A new regenerated
corpus gets a new name and the exp1b chain repoints to it.

### 3.2 Base policy — ~61 minutes

`scripts/policy_improvement_base_policy.py`, train-only, on the regenerated
train split. Competent train-only artifact shared across persistent and
episodic; this is the protocol's own `preferred_practical_study`, not the
random-base stress regime.

Expect `final_imitation_accuracy` ≈ `0.9884`, `checkpoint_selection:
final_budget_only`, `validation_data_opened: false`, `test_data_opened: false`.
CUDA training is not bit-deterministic, so the checkpoint digest will differ
between runs. Treat the procedure digest and the accuracy as the stable
signals, never the checkpoint hash.

Two known producer gaps, both handled in the amendment rather than by changing
the producer:

- it never emits `training_data_sha256`. The truthful value is the train split
  ordered-record digest above; the sanctioned consumer performs exactly that
  mapping at `scripts/policy_improvement_exp1b_runtime.py:580`.
- it writes `initialization_kind: train_only_pretrained`. That is the only value
  the schema accepts. Use it.

### 3.3 Authorization and signing — ~minutes

Order matters and is not obvious: **build the PARs, then mint the
authorization, then sign the base-policy amendment, then sign the admission.**
The amendment cannot be signed before the authorization exists.

`scripts/policy_improvement_runtime_authorization.py` needs no signing key. It
binds hashes. `auditor.py:593` requires producer and evaluator
`runtime_sha256` to differ — the full runtime and theory-bridge PARs are
distinct, which satisfies it. It does **not** require distinct human identities
or distinct checkouts, despite what earlier review reports claimed.

The amendment takes 14 fields, not the 10 listed in the protocol; those ten are
a required subset.

### 3.4 Run — see timings below

Eight registered seeds, 10,000 terminal environment interactions each, n=2,
K=1, m=8, gamma=0.99, alpha=0.1, `RUN_UPITRM_FULL_EXPERIMENTS=1`.

### 3.5 Audit

Packaged auditor PAR with `--parent-populations`. The flag is mandatory; an
audit without it cannot perform the census-ordering check.

## 4. Measured timings, and the scheduling finding

| Configuration | Rate | 350 steps |
| --- | --- | --- |
| One seed alone | 0.31 min/step | **110 min** |
| 3 seeds sharing one A100 | 1.54 min/step | ~9 h |
| 4 seeds sharing one A100 | 2.10 min/step | ~12 h |

**Running seeds concurrently does not help.** Each process pins one core at 99%
with 44 cores free, so it looks CPU-bound, but the actual contention is the GPU
time-slicing between processes. Utilisation reads 100% because a kernel is
resident, not because throughput is saturated. Seven concurrent seeds took
~13 h; sequential would have taken ~12.8 h.

**On a new machine, run the seeds sequentially, or one per physical GPU at
most.** Budget ~110 min per seed: about 15 hours for all eight, or ~7.5 hours
two-wide on two cards.

## 5. Build rules that are not optional

- `@fbcode//mode/dev-nosan` for CUDA. Plain `mode/dev` uses ASAN, which breaks
  CUDA init with a misleading out-of-memory error.
- **`@fbcode//mode/opt` for the launcher specifically.**
  `_launcher_executable_manifest` requires `fbmake.build_mode == "opt"` and
  refuses a dev-mode launcher before the identity pins are consulted.
- Never `--local-only`. It bypasses the remote cache and forces a local
  PyTorch/CUDA source build, which has crashed devservers.
- No extra `-c` flags. Each one forks the config hash away from CI's cache.
- `fbsource//third-party/pypi/torch:torch` is an alias to `fbcode//caffe2:torch`,
  a source build. Only cache hits avoid a 30-60 min compile.
- Judge CUDA compilation by `buck2 log what-ran --trace-id <id> --skip-cache-hits`,
  never a raw grep count. A clean build showed 17,340 grep matches and zero real
  CUDA actions.
- **Always pass `--trace-id`.** Bare `buck2 log summary` has reported a
  different, earlier invocation.
- `Cached actions: 0` on a warm daemon is normal, not a cache miss. Buck
  memoizes unchanged DICE nodes and never consults the action cache for them.
- Six targets need `package_style = "standalone"`: the launcher, dataset
  builder, training runtime, full runtime, theory bridge, audit and analysis
  runtimes. `validate_runtime_archive` and the authorization minter both open a
  runtime with `ZipFile` and refuse an inplace bootstrap PAR.
- The launcher identity pins in `scripts/policy_improvement_runtime_authorization.py`
  record build-environment bytes. Rebuilding the launcher on a different
  toolchain will move two of the five. Re-measure them; do not assume the guard
  is broken. The member inventories are the real guard and they should not move.

## 6. The defect class that cost the most time

Three separate times, a test passed while the production path had never
executed:

1. A constant-tuple assertion reported as a `torch.save` round trip.
2. A test restoring through its own helper instead of `open_exp1b_sealed_evaluation_session`.
3. `SealedFullRunBackend.run_exp1b_training` reaching for `self._module` instead
   of `self._training_module` — a one-token typo that survived ten review rounds,
   a reviewer PASS, and 309 Buck tests, because every test substituted its own
   class. It raised on the final line after the full 10,000-interaction budget,
   so it cost 110 minutes of A100 time and sealed nothing.

**The acceptance test that catches all three:** replace the production entry
point with a counter that raises when called, and require the test suite to go
RED. A suite that stays green has not proven what it claims. This is now applied
to the opener and the backend; apply it to anything new.

## 7. What is lost with the machine, and what it costs to rebuild

| Artifact | In git? | Rebuild cost |
| --- | --- | --- |
| All source, configs, tests | yes | — |
| Regenerated corpus | no | minutes, deterministic |
| Base policy checkpoint | no | ~61 min |
| Runtime authorization, amendment, admission | no | minutes |
| seed-0 sealed checkpoint | no | 110 min |
| seeds 1-7 partial progress | no | **total loss** |
| `/tmp/upi-trm-*.md` working handoffs | see `docs/exp1b-run-notes/` | preserved |

Worst case is a full re-run: roughly 1 hour base policy + 15 hours sequential
seeds + audit. That fits before the 2026-09-25 paper deadline but not
comfortably before the 09-18 abstract, so start the seeds first and write the
abstract against seed-0 plus whatever has sealed.

## 8. Still outstanding, independent of the machine

- Three placeholder tests call `skipTest` unconditionally and will **not**
  self-activate when artifacts appear: dataset-backed full restore, frozen-base
  versus deployed-mixture in a live record, and trainer-estimator centering at
  the registered 1e-6 tolerance. They need artifact-aware implementations.
  Costs were assessed: the first two are cheaper than they look and need no
  owner artifact if the dataset boundary is stubbed; the third is the expensive
  one and genuinely needs a real trainer and representative data.
- A BUCK comment undercounts the Torch-gated classes: it says five classes and
  eleven methods, the real inventory is six and fifteen. Comment only.
- No regression compares `configs/iclr_confirmatory/producer_source_manifest.json`
  against the checkout bytes. A stale entry there silently blocked base-policy
  production for a day. A real fix needs a resources glob plus about fifteen
  lines living beside the tool.
