# Adversarial review of UPI-TRM paper and implementation

Review date: 2026-08-12
Review mode: frozen-source audit followed by separately authorized repairs
Verdict at the frozen anchors: **FAIL**

## 1. Review metadata and frozen anchors

### Paper

- Repository: `/home/buiksat/UPI_TRM`
- Branch: `iclr-evidence-aligned-revision`
- Frozen branch head: `b00933db4b87a9eee1624f277394035f5b967073`
- Parent: `b21c720496d9864ef8665a8cfdafcafd95a29315`
- Mathematical source anchor: `2107125a50df1471393bc9a5e387c3005d7db112`
- Canonical source: `UPI_TRM_ICLR/main.tex`

The canonical paper source and all transitive paper dependencies are identical
between `2107125` and `b00933d`. The intervening commits remove obsolete review
prompts and update `handoff.md`.

### Implementation

- Repository: `/home/buiksat/trm_bellman`
- Branch: `full-implementation`
- Frozen head: `e4edcb2107c0f3e7ac0e691bd9dc828c5c6f38a0`
- Parent: `d9ccad73fb58998ccaed609b5e957d29b7878da6`
- Commit subject: `Harden confirmatory unpack and archive validation`

Both local repositories were on the requested branches and clean before the
review began. Remotes were the corresponding GitHub repositories. No
credentials appeared in their URLs.

The user later expanded the task from read-only review to repair and commit.
The findings and verdict below remain anchored to `b00933d` and `e4edcb2`.
Later repair commits do not retroactively change this verdict.

## 2. Verdict

# FAIL

Two P1 findings survived source tracing, counterexample construction, and an
independent default-reject pass:

1. The main finite-reference theorem identifies an arbitrary block-operator
   fixed point with the ordinary discounted policy value without a premise
   that makes the ordinary return exist.
2. PAR validation reasons about `ZipInfo.filename` but does not reject a ZIP
   member whose raw central-directory name differs from that effective name.
   The manifest can therefore authenticate bytes under a different member
   identity from the one used by the bootstrap importer.

Four nonblocking findings also survived: new schema-5 writes could use
historical digest-less identities, Phase 4 published placeholder values as
measurements, live CleanRL scripts named Buck targets that did not exist, and
one runtime-gate test did not establish its own replay-size precondition. Four
cited arXiv `@misc` entries also lacked a field rendered by the tracked
bibliography style.

This verdict does not assess learned-model quality, training convergence,
optimization dynamics, SGD, TD, BPTT, distillation, or recursively promoted
policy sequences. No experiment or learned payload was run.

## 3. Methodology and independent-pass disclosure

The review used separate blind passes for:

1. theorem premises, proofs, constants, measurability, and boundary cases;
2. terminal MDP, STOP, solved states, edit budgets, replay, and absorbing tails;
3. recurrent state ownership, projection, frozen maps, target networks, and
   retention;
4. exact mixing, exact centering, signed defects, CPI, and deployment error;
5. checkpoint identity, source provenance, ZIP/PAR semantics, schemas, Buck
   ownership, and tests;
6. citations, documentation, stale claims, and full-tree reachability.

Independent agents produced candidate lists. Separate skeptic passes then
traced each candidate through current callers and attempted to refute it. The
prior Claude report was read only after blind candidate generation. Findings
were retained only when source evidence and a reachable mismatch remained.

Safe deterministic work was limited to source builds, static parsing, unit
tests, arithmetic checks, archive metadata checks, and path normalization. No
checkpoint, retained result, benchmark, rollout, training job, or learned-model
evaluation was loaded or run.

## 4. Coverage ledger and exclusions

The paper review read `main.tex` completely, the bibliography, TikZ figure,
Makefile, ICLR style and bibliography style, local algorithm styles, local
header style, and `handoff.md`. Static source checks found 218 unique labels,
no missing references, 26 cited keys all present, and 44 unique bibliography
keys.

The implementation review enumerated 3,123 tracked entries totaling
460,572,173 bytes: 3,071 regular non-executable files, 51 executable files,
and one tracked symlink. It read the complete current files, or all
theorem/parity-relevant ranges and reachable callers, for the trainer, model,
environment, replay, value targets, persistent diagnostics, source and run
identity, launcher, packaged entrypoint, confirmatory configs, manifest,
CleanRL adapters, Phase 4 reporting, tests, Buck ownership, README, handoffs,
and parity reports.

Archived empirical files, retained outputs, binary figures, data records, and
learned payloads were classified by path and type but not interpreted.

## 5. Exact paper build and page inspection

The frozen paper commit was exported with `git archive` to a temporary
directory. The canonical command was run from `UPI_TRM_ICLR`:

```bash
make clean
make pdf
```

The build used an isolated Fedora TeX Live 2020 toolchain assembled outside
both repositories. Result:

- exit code: 0;
- pages: 38;
- bytes: 599,759;
- SHA-256: `f27bae2019e10d1f27c5762a81a3aa47d0bca7dbb9645e72aeb7121ea54acfc0`;
- final missing inputs: none;
- final undefined references or citations: none;
- final overfull or underfull boxes: none;
- final LaTeX errors or fatal errors: none.

Every page was rendered and inspected. No clipping, equation overflow,
malformed algorithm, detached proof, broken reference, or table overflow was
found. The PDF hash differs from the maintainer-recorded build because the TeX
and font-map environment differs; source content and page count are stable.

## 6. Theorem, quantifier, constant, and boundary audit

The following derivations were independently checked and cleared:

- measurable Bellman contraction with modulus `gamma^K`;
- the finite-reference residual divisor `1-gamma^K`;
- direct latent endpoint inequality before both path-length relaxations;
- fixed-point measurability from joint measurability, measurable
  initialization, invariance, and uniform contraction;
- saturated-annulus projection modulus `(R/rho_R)L_z^pre(R)`;
- propagated target lag before the ordinary sup-lag fallback;
- exact-mixture occupancy denominator `1-gamma+gamma alpha`;
- signed-defect identity and the span/TV CPI penalty;
- the nonredundant safe-step threshold;
- one-step and K-step policy-overlap factors;
- finite-horizon occupancy timing;
- deployment reward-mismatch timing;
- direct persistent augmented-state residual semantics.

Boundary checks covered `gamma=0`, `alpha` at 0 and 1, interior `alpha`, all
safe-step branches, `Delta_g=0`, `n=0`, `L_z=0`, `H=0`, `h=0`, `K>h`, every
terminal cause, mixed and all-terminal batches, both latent modes, projection
disabled, legacy radius zero, projection boundary inputs, and deployment
discrepancy at 0 and 1.

The one exception is finding `UPITRM-GPT-001`: block-reward boundedness alone
does not identify the block fixed point with the ordinary discounted return.

## 7. Code-to-paper parity matrix

| Contract | Frozen implementation result |
|---|---|
| Edit clock is in state | Match |
| Store terminal transition once, then stop | Match |
| No action or persistent successor latent after terminal | Match |
| STOP, solved, and budget exhaustion share terminal masking | Match |
| Folded absorbing tail counted once | Match |
| Exact K-step segment or earlier terminal | Match |
| Frozen target evaluator supplies bootstrap | Match |
| Retention updates target value head only | Match |
| One Euclidean product-space projection or identity mode | Match |
| Current and candidate policies share one frozen recurrence | Match |
| Persistent replay stores pre-unroll and nonterminal post-unroll carry | Match |
| Exact pointwise probability mixture | Match |
| Exact statewise centering over represented action support | Match |
| No recursive promotion in fixed-base theorem path | Match |
| Complete PAR authenticated before behavior imports | Match, subject to `UPITRM-GPT-002` |
| Runtime digest binds current evidence objects | Match, subject to `UPITRM-GPT-003` at the public writer |

## 8. Verified findings

### UPITRM-GPT-001 - P1 Blocking - high confidence

**Location:** paper `main.tex:415-438`, theorem
`thm:finite_reference_depth`; proof at `main.tex:2984-2995`.

**Claim:** The theorem assumes only that the K-step expected reward is bounded
and measurable, proves that the block operator has a unique fixed point, and
names that fixed point `V^pi`. It never establishes that the ordinary
discounted return exists or satisfies the block Bellman identity.

**Counterexample:** Let the deterministic countable chain move from `s_i` to
`s_(i+1)`, choose `gamma` in `(0,1)`, and set the one-step reward at `s_i` to
`(-1)^i gamma^(-i)`. For `K=2`, the two-step discounted reward is identically
zero, so the block operator has bounded fixed point zero. The ordinary return
from any state is an alternating sum whose terms do not converge to zero, so
the return diverges. The theorem's abstract contraction statement is valid;
the identification with ordinary policy value is not.

**Impact:** The central certificate is stated as a bound to the ordinary policy
value without premises sufficient to define that value.

**Smallest repair:** require uniformly bounded one-step rewards and measurable
fixed-policy conditional reward, then identify the uniformly convergent
discounted series with the block fixed point in the proof.

**Provenance:** pre-existing at the frozen paper anchor. The independent skeptic
reproduced the counterexample.

### UPITRM-GPT-002 - P1 Blocking - high confidence

**Location:** `confirmatory_runtime_launcher.py:360-425`,
`_validate_archive_sources`; `utils/source_identity.py:151-198`,
`assert_runtime_archive_sources_match_manifest`.

**Claim:** Both validators classify and hash members by `ZipInfo.filename` but
do not reject `orig_filename != filename`. CPython can replace the effective
name using Unicode Path extra field `0x7075` while retaining the raw central
name in `orig_filename`. The PAR bootstrap importer indexes the archive's raw
member identity. Validation can therefore authenticate bytes under one name
while execution resolves another member identity.

**Reachability:** The launcher validates all archive members before sealing and
execution, but this semantic alias passes before inventory construction. The
source-identity helper repeats the same assumption. A matching outer digest
authenticates the ambiguous artifact as a whole; it does not prove that the
embedded manifest names the behavior bytes the importer executes.

**Impact:** Confirmatory source provenance is not a semantic binding between
manifest paths and imported paths.

**Smallest repair:** reject every member, including nonbehavior metadata, when
`info.orig_filename != info.filename` before duplicate, path, inventory, or
digest checks. Add a synthetic `0x7075` regression for both validators.

**Provenance:** pre-existing at `e4edcb2`. An independent ZIP/PAR semantics
pass retained it.

### UPITRM-GPT-003 - P2 Major - high confidence

**Location:** `upi_trm_train.py:2143-2172`, `save_checkpoint`; existing reachability
fixture at `tests/test_upi_trm_logging_smoke_unittest.py:846-944,1495-1557`.

**Claim:** Public `save_checkpoint` accepts run identities whose effective
configuration uses historical schemas 1 through 3, skips runtime-artifact
binding for those schemas, then writes a new checkpoint labeled schema 5.

**Impact:** Current code can mint a digest-less schema-5 artifact that is
indistinguishable from an intended historical artifact. Registered CLI runs
are safe because they construct schema 4 under a preverified digest.

**Smallest repair:** require effective-config schema 4 at the schema-5 write
boundary. Keep historical schemas readable in resume and diagnostic paths.

**Provenance:** pre-existing. Independent skeptic retained P2 because the
registered CLI is guarded but the public writer and test path are reachable.

### UPITRM-GPT-004 - P2 Major - high confidence

**Location:** `scripts/eval_phase4_2x2_norm_ablation.py:80-117,421-435,506-566,696-708`.

**Claim:** `compute_success_rate` returns `0.0` without an environment rollout.
The evaluator also writes `final_loss=0.0` and `has_nan=False` without loading
a training log or history. It aggregates these values, while the audit and
figure generator accept and publish them as measured results.

**Impact:** A valid zero and an unavailable measurement are serialized
identically. Paper-facing tables and plots can state empirical outcomes that
were never measured.

**Smallest repair:** remove the retired numeric fields from new records, add a
strict versioned schema with exact unavailable-reason metadata, and make audit
and figure consumers reject schema-less or legacy summaries.

**Provenance:** pre-existing. Independent evaluator and consumer tracing
retained it. No learned payload was opened.

### UPITRM-GPT-005 - P2 Major - high confidence

**Location:** `scripts/run_cleanrl_benchmark_hard4x4.sh:20-29` and
`tests/smoke_test_cartpole.sh:19-67`; no corresponding target in `BUCK`.

**Claim:** The maintained scripts invoke `fbcode//buiksat_trm_cleanrl` targets
that do not exist. The intended dispatcher exists at
`rl/cleanrl/cleanrl_runner.py` but has no executable Buck target.

**Impact:** Both documented entrypoints fail before reaching their configs.
The adjacent README also incorrectly says `DQNTrainer` lacks n-step targets;
the trainer supports them, while only the wrapper fails to expose them.

**Smallest repair:** add one `cleanrl_runner` binary in the existing package,
route both scripts through it, test dispatcher routing, and correct the wrapper
wording. Do not add three redundant algorithm-specific binaries.

**Provenance:** pre-existing. Buck target enumeration and a separate dispatcher
pass retained it.

### UPITRM-GPT-006 - P3 Minor - high confidence

**Location:** `tests/test_upi_trm_trainer_smoke_unittest.py:251-274`,
`test_persistent_exact_baseline_policy_update_runs`.

**Claim:** The test collects one stochastic episode and calls `policy_update`
without ensuring replay size reaches batch size 2. An immediate terminal action
leaves one transition; `policy_update` correctly returns its no-op metrics,
and the test then indexes an update-only centering metric.

**Observed result:** the frozen 13-target gate failed this test with
`KeyError: exact_centering_defect_max`.

**Smallest repair:** collect until `len(replay) >= batch_size` before asserting
update-only metrics.

### UPITRM-GPT-007 - P3 Minor - high confidence

**Location:** `UPI_TRM_ICLR/trm_rl.bib`, entries
`Wang2025OneShotRLVR`, `Su2025RewardBridge`, `DeepSeekR1`, and
`Eshwar2025MonotoneCPI`.

**Claim:** These `@misc` entries contain DOI/eprint metadata, but the tracked
ICLR bibliography style does not render those fields for `@misc`. The rendered
paper gives no locator for the four cited 2025 manuscripts.

**Smallest repair:** add their canonical arXiv URLs, which the tracked style
does render.

## 9. Rejected candidates and cleared risks

- Unpack-directory path replacement: rejected. The launcher passes an open
  directory descriptor and the child verifies the same object.
- Root aliases and canonical ZIP collisions: rejected for the already covered
  names. The new raw/effective identity finding is distinct.
- Forged attestation environment strings: rejected. Sealed descriptors,
  identity checks, rehashing, and module origin are also required.
- Launcher self-authentication: rejected under the declared external trust
  root.
- Runtime digest inside `run_matrix.json`: rejected because it would create a
  self-hash cycle.
- Historical schema readability itself: rejected. Only minting a new schema-5
  artifact from a historical identity is defective.
- Fixed target confused with self-bootstrap: rejected. Paper and code keep
  the operators separate.
- Terminal tensor payloads as successor states: rejected. Terminal rows are
  selected away before successor evaluation, and persistent successor latent
  is forbidden.
- Exact mixture implemented as logits or parameters: rejected. The theorem
  path mixes normalized probability vectors.
- Finite diagnostics promoted to uniform premises: rejected for maintained
  current paths.
- Missing test ownership: rejected. Pytest discovery and Buck twins cover the
  maintained tests except for the new script-entrypoint gap reported above.
- Stale theorem numbering: rejected after resolving shared theorem counters.
- Historical NeurIPS documents and pruned YAML launchers: rejected as outside
  the maintained current path.

## 10. Validation commands and exact results

### Frozen paper

The exact build and 38-page inspection are recorded in section 5.

### Frozen implementation

The prescribed 13-target runtime gate ran from
`/data/users/buiksat/fbsource` with Buck
`083174567c29730448fbd8361efbe47175c41c47`.

Result:

```text
Pass 287
Fail 1
Fatal 1
Timeout 0
Infrastructure 0
Build failure 0
```

The failure was `test_persistent_exact_baseline_policy_update_runs` as described
in `UPITRM-GPT-006`. The fatal result was reported for the config-integrity
target because TPX did not find its result file even though the underlying
unittest printed `Ran 1 test` and `OK`.

Other frozen-tree checks:

- producer manifest regeneration: 79 entries, exact match;
- tracked JSON parse: 625 files, no error;
- tracked shell syntax: 46 files, no error;
- tracked Python AST parse: no error;
- paper references and bibliography keys: no missing target;
- `git diff --check`: clean before authorized repair edits.

The six synchronization type targets, two launcher type targets, real PAR
integration, and full YAML parse were not completed at the frozen anchor before
the user authorized repairs. Maintainer-recorded results were not substituted
as independent evidence. The P1 findings already require `FAIL`, not
`INCOMPLETE`.

## 11. Prior-report reconciliation

| ID | Frozen-source assessment |
|---|---|
| `UPITRM-REV-001` split-root source identity | Prior repair present, but incomplete because raw/effective ZIP identity remained unbound. |
| `UPITRM-REV-002` rowwise Exp1 projection | Cleared. Production joint projection is used. |
| `UPITRM-REV-003` `float(None)` | Cleared. Disabled projection follows an identity branch. |
| `UPITRM-REV-004` rush-to-fail theorem/citation | Cleared. Maintained reward wording no longer makes the false claim. |
| `UPITRM-REV-005` stale numeric references | Cleared in maintained paths. |
| `UPITRM-REV-006` unowned tests | Cleared for the prior touched tests. |
| `UPITRM-REV-007` finite diagnostics as guarantees | Cleared in maintained generators. |
| `UPITRM-UPD-001` pre-import provenance | Cleared, subject to raw/effective ZIP identity. |
| `UPITRM-UPD-002` stale trainer section | Cleared. |
| `UPITRM-UPD-003` theory-exact ablation wording | Cleared. |
| `UPITRM-UPD-004` unpack pathname race | Cleared by descriptor binding. |
| `UPITRM-UPD-005` root aliases/collisions | Cleared for canonical path aliases; does not cover `UPITRM-GPT-002`. |
| `UPITRM-UPD-006` stale validation docs | Cleared at the frozen documented anchor. |

## 12. Claim, citation, and stale-reference audit

The maintained paper and implementation correctly distinguish value and
latent fixed points, fixed-target and self-bootstrap operators, exact
probability mixing and interpolation, invariance and strict contraction,
fixed-snapshot mechanics and optimization convergence, finite diagnostics and
uniform premises, episodic and augmented persistent state, and formal
absorbers and terminal serialization payloads.

No unsupported learned-performance claim exists in the canonical paper. The
four missing rendered arXiv locators are recorded as `UPITRM-GPT-007`. No other
citation mismatch survived.

## 13. Recommended repairs

1. Add the one-step bounded-reward premise and policy-value identification.
2. Reject all ZIP raw/effective member-name mismatches before classification.
3. Require effective-config schema 4 for new schema-5 checkpoint writes.
4. Replace Phase 4 placeholders with strict unavailable metadata and
   fail-closed consumers.
5. Add the unified CleanRL Buck runner and repair both scripts.
6. Establish replay batch size in the stochastic policy-update test.
7. Add rendered URL fields to the four arXiv bibliography entries.

Run the full runtime, type, launcher, manifest, JSON/YAML, shell, paper-build,
and page-inspection gates after repair. Do not run experiments.

## 14. Worktree attestation

At review start, both repositories produced empty `git status --short` output.
The exact frozen commits were inspected without checkout, reset, stash, clean,
or rebase. Temporary build and render artifacts were created only under
`/tmp` or as ignored paper outputs.

After the review verdict, the user explicitly authorized source repairs and
commits. That authorization superseded the original report-only write limit.
The final post-repair commit and worktree state are recorded in the repository
handoffs, not used as evidence for this frozen-anchor verdict.

## 15. Post-review repair record

All seven findings were repaired after the user expanded the task. The paper
source repair is commit `5253692fea5e77cfde3a130c50351183dc0268e3`; its
handoff head is `071037929ed0a16eaf1d8b2f1169786a866eb8a5`. The implementation
source repair is commit `f86bddb607adcd24eba65fd5869af58f91742a52`.

Post-repair validation produced:

- exact paper build: exit 0, 38 pages, 601,498 bytes, SHA-256
  `387ccc5613914f5e2de44f2439de409f984932e8ad8f0250f4f636fae37ab6d6`;
- complete runtime gate: 308 passed, 0 failed, timed out, fatal,
  infrastructure-failed, or build-failed;
- type targets: 6 synchronization, 2 launcher, and 6 Phase 4/CleanRL targets
  built successfully;
- training PAR SHA-256
  `bd10dda2afc422bd07751a02e1ed39ef164adf0d6a4241220db2b94f95e5cb67`;
- real launcher help path: exit 0; direct PAR, wrong digest, and uppercase
  digest failed closed with exits 1, 2, and 2; no private unpack directory
  remained;
- exact 79-entry manifest regeneration, 625 JSON files, 96 YAML files, 243
  Python files, and 46 shell scripts all passed their static checks.

These results verify the repaired source mechanics. They do not change the
historical `FAIL` verdict at the frozen pre-repair anchors, and they do not
replace the independent post-repair review requested by the new prompt.

## Machine-readable appendix

```json
{
  "review_date": "2026-08-12",
  "verdict": "FAIL",
  "anchors": {
    "paper_commit": "b00933db4b87a9eee1624f277394035f5b967073",
    "paper_theory_commit": "2107125a50df1471393bc9a5e387c3005d7db112",
    "implementation_commit": "e4edcb2107c0f3e7ac0e691bd9dc828c5c6f38a0",
    "implementation_parent": "d9ccad73fb58998ccaed609b5e957d29b7878da6"
  },
  "findings": {
    "P0": 0,
    "P1": 2,
    "P2": 3,
    "P3": 2
  },
  "paper_build": {
    "exit_code": 0,
    "pages": 38,
    "bytes": 599759,
    "sha256": "f27bae2019e10d1f27c5762a81a3aa47d0bca7dbb9645e72aeb7121ea54acfc0",
    "all_pages_inspected": true
  },
  "runtime_gate": {
    "passed": 287,
    "failed": 1,
    "fatal": 1,
    "timeout": 0,
    "infrastructure": 0,
    "build_failure": 0
  },
  "experiments_run": false,
  "learned_payloads_loaded": false,
  "repairs_authorized_after_review": true,
  "post_review_repairs": {
    "paper_source_commit": "5253692fea5e77cfde3a130c50351183dc0268e3",
    "paper_handoff_head": "071037929ed0a16eaf1d8b2f1169786a866eb8a5",
    "implementation_source_commit": "f86bddb607adcd24eba65fd5869af58f91742a52",
    "runtime_gate_passed": 308,
    "runtime_gate_failed": 0
  }
}
```
