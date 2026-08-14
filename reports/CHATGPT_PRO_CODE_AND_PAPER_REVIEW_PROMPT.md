# ChatGPT Pro prompt: adversarial review of UPI-TRM code and paper

You are the lead reviewer for a read-only, correctness-critical audit of the
UPI-TRM implementation and its canonical ICLR theory paper. Inspect both
repositories directly. Do not rely on summaries, prior verdicts, or remembered
commit IDs.

Your only permitted repository write is the final report:

`/home/buiksat/trm_bellman/reports/CHATGPT_PRO_CODE_AND_PAPER_REVIEW_REPORT.md`

Do not edit source, tests, configs, paper files, retained artifacts, or existing
reports. Do not commit, push, post comments, open a pull request, or contact
anyone.

## Resolve and freeze the review anchors

Prefer these local repositories:

- implementation: `/home/buiksat/trm_bellman`, branch `full-implementation`;
- canonical paper: `/home/buiksat/UPI_TRM`, branch
  `iclr-evidence-aligned-revision`.

The intended frozen source anchors for this review are:

- implementation behavior-source commit
  `de013fd3fcaae8bc80d124c0c868c7c8611aeede`, parent
  `12fa350951c31e8635ee51747f6de25295c07333`, with previous behavior anchor
  `980f6ede14717e87ad68ceb32acc111bdd7fca1b`;
- paper branch head `5ad61a16281f3b509d0fc2e1756911ae86ae0c1c`;
- canonical mathematical paper source commit
  `5253692fea5e77cfde3a130c50351183dc0268e3`.

Verify that the named branches contain those commits. Review implementation
behavior at the exact source commit even if the branch has moved. Resolve the
current implementation branch tip and prove that every commit after the source
anchor changes only `README.md`, `reports/**`, or other handoff documentation.
If any later commit changes behavior-bearing source, tests, configs, manifests,
or build ownership, stop and request a new explicit source anchor. For the
paper, prove whether the
canonical paper source and its transitive dependencies differ between the
mathematical source commit and the frozen paper branch head. Do not substitute
a newer behavior-source revision without explicit user authorization.

For each target branch, record:

- absolute repository path;
- full commit SHA and parent SHA;
- current checked-out branch and whether it matches the target branch;
- exact `git status --short` output;
- remotes, with credentials redacted;
- recent history sufficient to locate the implementation synchronization
  commit and any later documentation-only commits.

Resolve and record the two target branch tips at review time, but use the exact
source commits above as the review anchors. The implementation prompt may be in
a later documentation-only commit because a commit cannot name its own SHA. Do not
substitute a remembered or newer behavior-source SHA. If a local checkout is on
another branch, inspect the frozen commit with `git show` or `git archive`.
Do not checkout, reset, stash, clean, rebase, fetch into, or otherwise alter the
user's repository or working-tree metadata.

If a local path is unavailable, clone the named feature branch into a directory
created with `mktemp -d` outside any repository:

```bash
git clone --single-branch --branch full-implementation \
  https://github.com/gopeshh/trm_bellman.git <temporary-implementation-path>
git clone --single-branch --branch iclr-evidence-aligned-revision \
  https://github.com/buiksat/UPI_TRM.git <temporary-paper-path>
```

Record the resolved clone SHAs and use them unchanged. If the canonical local
report path does not exist because a clone was required, write the report only
to `reports/CHATGPT_PRO_CODE_AND_PAPER_REVIEW_REPORT.md` in the temporary
implementation clone and return the complete report in the response. State the
path substitution prominently.

Treat the paper as the semantic oracle for paper-parity claims. Treat
implementation tests, handoffs, and prior reviews as hypotheses and evidence,
not as authority.

## Post-fix review focus

This review follows repairs for `UPITRM-UPD-001` through `UPITRM-UPD-006`,
`UPITRM-GPT-001` through `UPITRM-GPT-007`, and `UPITRM-POST-001` through
`UPITRM-POST-005`.
Do not assume those repairs are correct merely because tests or prior reviewers
accepted them. Reconstruct and challenge each contract from current source:

1. Confirmatory execution must authenticate the complete packaged runtime
   before any behavior-bearing model or RL module is imported.
2. Validation and execution must be bound to one immutable runtime object, with
   no post-binding path redirection, same-inode mutation, stale bytecode,
   duplicate archive member, unsafe member path, or shared extraction-cache
   gap.
3. The launcher must copy verified bytes into a sealed anonymous file, execute
   that exact descriptor as a supervised child, bind the private unpack
   directory to an inherited directory descriptor, and avoid a shared
   extraction cache.
4. The packaged entry point must independently verify descriptor identity,
   seals, whole-artifact SHA-256, module origin, and the inherited private
   unpack descriptor before behavior-bearing project imports.
5. The verified runtime SHA-256 must bind current effective configuration,
   evidence identity, confirmatory lock, checkpoint save, and resume. Historical
   schemas must remain readable only under their exact historical contracts,
   and the digest must remain outside PAR inputs to avoid a self-hash cycle.
6. The trainer comment must no longer contain the stale numeric paper mapping.
7. The ablation generator must explicitly emit legacy, non-theorem-facing
   configurations rather than calling its feature-on baseline theory-exact.
8. Archive validation must reject root aliases, noncanonical member names,
   canonical path aliases, and file/directory prefix collisions before the PAR
   bootstrap sees them.
9. Current validation documents must identify the complete 30-target affected
   type gate and must not retain stale test, manifest, artifact, or source
   profile counts.
10. Treat the launcher executable, operating system, host namespace, other
    same-UID processes, and initial launcher environment as the external root
    of trust. Verify descriptor binding after acquisition. Do not demand or
    credit isolation from a compromised host, and do not describe best-effort
    pathname cleanup as a security boundary.
11. The finite-reference theorem must state a premise sufficient to define the
    ordinary discounted fixed-policy value, then prove that this value is the
    unique fixed point of the `K`-step Bellman operator. Actively retry the
    alternating-reward counterexample from the prior review.
12. Both archive-validation layers must reject every member when
    `ZipInfo.orig_filename != ZipInfo.filename`, including nonbehavior members
    rewritten by Info-ZIP Unicode Path field `0x7075`. Trace raw central names,
    effective CPython names, NUL truncation, duplicate aliases, and the actual
    Buck PAR bootstrap importer.
13. The schema-v5 checkpoint writer must accept only effective-config schema 4
    for new writes. Historical schemas 1 through 3 may remain readable only
    under their exact historical contracts and must not mint new schema-v5
    evidence.
14. Phase 4 schema-v4 output must omit the retired numeric success, final-loss,
    and training-history fields while retaining exact unavailability reasons.
    Publication requires the exact four-condition by three-seed design, fixed
    toggles and projection settings, complete finite metric samples, and
    aggregates recomputed from all 12 runs. Audit and figure consumers must
    reject legacy, partial, inconsistent, nonfinite, or placeholder summaries
    before creating output.
15. CleanRL entry scripts must resolve one owned `cleanrl_runner` Buck target.
    Verify PPO, A2C, CartPole DQN, and Sudoku DQN routing, rejection of unknown
    algorithms, and the explicit Sudoku wrapper limitation for `n_step > 1`.
16. The stochastic persistent-policy smoke test must establish
    `len(replay) >= batch_size` before asserting update-only metrics.
17. The four repaired arXiv bibliography entries must render a usable external
    locator under the tracked bibliography style.
18. Phase 4 `L_preproj` must use the exact plan-conditioned production
    pre-projection recurrence. Its denominator must be the actual joint
    `(z_H,z_L)` perturbation norm, not a component scale, and zero or missing
    directional samples must fail closed.
19. Phase 4 policy stability must use the production `policy_dist` path with
    `z_H`, the exact action mask, and absolute unroll depths 2, 4, and 8. It
    must not call the raw policy head on `z_L` or admit masked actions.
20. Every Phase 4 result must be bound to a strict complete checkpoint and its
    checkpoint bytes, model bytes, saved model/RL config, condition, seed, run
    identity, producer source, dataset, and diagnostic input bytes. Audit and
    figure consumers must independently revalidate those identities rather
    than trusting summary strings.
21. Phase 4 evaluator, audit, and figure PARs must authenticate all selected
    runtime Python source bytes against one explicit clean Git checkout. Reject
    stale PAR bytes, missing or extra selected sources, selected bytecode,
    duplicate or raw/effective archive names, worktree bytes that differ from
    HEAD, ignored additions, hidden deletions, and `skip-worktree` or
    `assume-unchanged` index flags.
22. Phase 4 training, evaluator, audit, and figure execution must pass through
    the standard-library external launcher. It must authenticate the complete
    role-specific PAR before behavior-bearing imports, copy it to a sealed memfd, and
    execute that descriptor with a descriptor-bound private unpack directory.
    A source verifier called from ordinary `main()` after NumPy, Torch, or
    behavior-bearing project imports is not sufficient.
23. Audit and figure consumers must independently resolve the authorized
    training-source commit from Git, verify the exact producer manifest from
    that clean tree, and compare both with every checkpoint. Format-valid
    producer strings embedded in a checkpoint are not evidence.
24. All 12 Phase 4 records must bind one externally authorized training-PAR
    SHA-256. Mixed training runtimes, summary-only runtime strings, and a
    checkpoint that omits or contradicts the runtime identity must fail.
25. Strict Phase 4 checkpoint validation must cover the complete schema-4
    resume state, not only model weights. Replay entries must be executed
    through the registered `PlanEditEnv`, including mask, successor state,
    reward, terminal flag, terminal reason, clock, and sequence validation.
26. Evaluator, audit, and figure source profiles must include the full imported
    closure, including every selected `dataset/*.py` file. Imported behavior
    may not sit outside the authenticated profile.
27. Figure output must be staged privately and published only after final
    source, checkpoint, runtime, and diagnostic-input identity checks. The
    evaluator and audit must finish their required identity checks before their
    direct output writes.
28. Launcher-owned child options must reject both split and `--name=value`
    overrides. Registered dummy Phase 4 data must be deterministic, supported
    by the exact action mask, and reproducible from the bound seed and config.

Review the complete implementation change from `e4edcb2` through `de013fd`,
with special attention to `980f6ede..de013fd`, but do not limit the audit to
those diffs. The resulting source tree and all reachable callers remain the
source of truth.

## Non-negotiable scope

The paper is conditional on:

- one fixed MDP;
- one fixed current/candidate policy pair;
- one fixed parameter snapshot;
- one shared frozen recurrent map whenever both policies are represented in the
  same persistent-latent augmented MDP.

Do not broaden the verdict to SGD, TD convergence, BPTT, distillation,
optimization dynamics, learned-model performance, or a sequence of recursively
promoted policies.

Do not add, request, design, launch, rerun, reinterpret, or validate experiments.
Do not use benchmark results, retained outputs, regression losses, finite-batch
maxima, sampled diagnostics, or local Lipschitz estimates to establish a
uniform mathematical premise.

You may run unit tests, static checks, type checks, source builds, and minimal
deterministic reproducers that do not train or evaluate a learned model. Do not
load untrusted archived pickle, Torch checkpoint, or retained learned-model
payloads.

## Required direct inspection

Read the canonical paper completely at the frozen paper commit:

- `UPI_TRM_ICLR/main.tex`;
- every transitive `\input`, `\include`, bibliography, macro, style, and
  figure-source dependency;
- `UPI_TRM_ICLR/trm_rl.bib`;
- the ICLR style and every local style or macro file;
- README, Makefile, `latexmk` configuration, scripts, and CI files that define
  the canonical build;
- the tracked `UPI_TRM_ICLR/main.pdf` at the frozen paper head;
- the independently rebuilt exact-source `main.pdf`.

Hash and inspect every page of the tracked PDF so a repository-only review can
read the rendered artifact. Do not treat that committed binary as independent
build evidence. Export the frozen paper commit to a temporary directory, build
there through the repository's canonical path, record the exact command and
exit code, inspect every rebuilt page, and compare the rebuilt artifact with
the tracked PDF.

Read behavior-bearing implementation source, tests, configs, manifests, and
Buck ownership directly at the frozen source commit. After proving that later
commits are documentation-only, read `README.md` and `reports/**` at the
resolved documentation tip and diff them against their source-commit versions.
At minimum, read these files completely at the applicable anchor:

- `README.md`;
- `.gitignore`;
- `BUCK`;
- `reports/PAPER_PARITY_REPORT.md`;
- `reports/ICLR_REPAIR_HANDOFF.md`;
- canonical ICLR configs and the run registry;
- `configs/phase4_2x2_norm_ablation/*.yaml`;
- the producer source manifest and its generator;
- `models/recursive_reasoning/trm.py`;
- `rl/config.py`;
- `rl/envs/plan_edit_env.py`;
- `rl/replay.py`;
- `rl/value_targets.py`;
- `rl/upi_trm_trainer.py`;
- `rl/persistent_diagnostic_checkpoint.py`;
- `rl/persistent_diagnostics.py`;
- `rl/cleanrl/trm_adapter.py`;
- `rl/cleanrl/cleanrl_runner.py`;
- `upi_trm_train.py`;
- `utils/lipschitz.py`;
- `utils/run_identity.py`;
- `utils/source_identity.py`;
- `confirmatory_runtime_launcher.py`;
- `scripts/phase4_checkpoint.py`;
- `scripts/phase4_diagnostic_inputs.py`;
- `scripts/phase4_result_schema.py`;
- `scripts/phase4_source.py`;
- `scripts/eval_phase4_2x2_norm_ablation.py`;
- `scripts/audit_phase4_paper_ready.py`;
- `scripts/make_paper_figures_phase4.py`;
- `scripts/run_phase4_training.py`;
- `scripts/run_cleanrl_benchmark_hard4x4.sh`;
- `tests/smoke_test_cartpole.sh`;
- `tests/test_phase4_reporting_unittest.py`;
- `tests/test_run_identity_unittest.py`;
- `tests/test_theory_exact_components_unittest.py`;
- `tests/test_upi_trm_logging_smoke_unittest.py`;
- the Phase 4, CleanRL dispatcher, archive identity, source identity,
  checkpoint logging, and trainer smoke tests;
- every importing caller, affected test, owning Buck target, launcher,
  evaluator, diagnostic, and documentation claim reached from those files.

Do not read `reports/CLAUDE_MULTI_AGENT_REPOSITORY_REVIEW_REPORT.md` or
`reports/CHATGPT_PRO_CODE_AND_PAPER_REVIEW_REPORT.md` until the blind review
passes below are complete. Read them afterward and reconcile every finding,
repair claim, and rejected candidate independently.

Enumerate every tracked file in both repositories. Fully read every
behavior-bearing source, test, config, build, CI, manifest, instruction, and
current documentation file. Classify archived results, data, logs, images, and
binary or generated artifacts by path, type, size, provenance role, and whether
runtime code trusts them. Do not interpret their empirical contents.

Inspect the complete relevant commit diffs and both complete committed trees.
Do not reduce this to `HEAD^..HEAD` if the latest commit contains only prompt or
documentation maintenance.

## Blind adversarial passes

Before reading the prior Claude report, run independent passes with separate
contexts:

1. paper assumptions, theorem statements, proofs, constants, measurability,
   and domains;
2. terminal MDP, replay, edit clock, STOP, solved states, budget exhaustion,
   and absorber accounting;
3. recurrent latent ownership, shared frozen map, projection, target networks,
   fixed-K targets, and retention updates;
4. exact probability mixing, exact centering, CPI bounds, signed defects, and
   deployment perturbation;
5. configuration, checkpoint identity, source provenance, build ownership,
   sealed-runtime launch safety, schema compatibility, and test coverage;
6. Phase 4 metric semantics, strict checkpoint and input identity, runtime
   source binding, publication consumers, citations, stale references,
   unsupported claims, and source/PDF consistency.

Use independent agents if ChatGPT Pro exposes them. Otherwise perform genuinely
separate passes and disclose that limitation. Do not claim an agent or model
participated unless it returned usable work.

After blind candidate generation, read the prior Claude and GPT Pro reports.
Independently reverify every `UPITRM-REV-*`, `UPITRM-UPD-*`, `UPITRM-GPT-*`,
and `UPITRM-POST-*` item and every material rejected candidate against the
frozen anchors. Do not inherit either verdict.

For every candidate defect:

1. normalize it to one precise claim;
2. confirm the current file, symbol, and line;
3. trace reachable callers, configuration prerequisites, and guards;
4. compare it with an exact paper clause or another named source of truth;
5. actively try to refute it;
6. assign an independent skeptic with `default real=false`;
7. use a minimal deterministic reproducer when safe;
8. re-grade severity after verification;
9. retain it only if it survives.

Model agreement is not proof. Source evidence, reachable behavior, and a
demonstrated mismatch are required.

## Paper assumptions and proofs to audit explicitly

Re-derive all implementation-facing results and verify every domain,
measurability, boundedness, and invariance premise. A strong explicit
assumption is not itself a defect.

### Ordinary policy value and block fixed point

Verify the finite-reference theorem assumes a uniformly bounded measurable
one-step reward under the fixed policy, or another premise that is genuinely
sufficient for the ordinary discounted return to exist. Check that the proof
constructs

```text
V^pi(s) = sum_{t>=0} gamma^t E[r_t | s_0=s]
```

by uniform convergence, establishes bounded measurability, and proves the
`K`-step Bellman identity before invoking uniqueness of the block-operator
fixed point. Do not accept boundedness of the grouped `K`-step reward alone.
Retry the deterministic chain with
`r_i=(-1)^i gamma^(-i)` and `K=2`: its grouped reward cancels while the ordinary
discounted series diverges. Explain exactly which repaired premise excludes it.

### Bounded value heads

For every invocation on a declared nonabsorbing state domain
`D^circ`, require a finite `V_max` such that

```text
|V_psi(z,x(s),y(s))| <= V_max
|V_barpsi(z,x(s),y(s))| <= V_max
```

for every represented `s in D^circ` and every latent `z` in the declared
`Z_inv`, or in an explicitly declared smaller latent set used by that result.
The absorber must be handled by its separate finite scalar boundary. There is
no absorbing latent. Confirm that projection-aware and persistent fixed-point
uses match these quantifiers and that no Bellman theorem weakens an explicit
`B_b(C)` premise.

### Fixed-point measurability

Where recurrent fixed-point measurability is derived, verify all of these
conditions are explicit:

- `Z_inv` is a closed subset of the normed latent space with its Borel sigma
  algebra;
- `T(s,z)=T_s(z)` is jointly measurable on `D^circ x Z_inv`;
- `z^(0):D^circ -> Z_inv` is measurable;
- invariance and uniform contraction hold on those same domains.

Verify the proof that every iterate
`z^(j+1)(s)=T(s,z^(j)(s))` is measurable by induction, Banach contraction gives
pointwise convergence, and the pointwise limit is measurable. One valid final
step is, for every closed `F subset Z_inv`,

```text
d(z*(s),F) = lim_j d(z^(j)(s),F),
{s:z*(s) in F} = {s:d(z*(s),F)=0}.
```

Metric contraction clauses alone must still be described as insufficient for
measurability.

### Direct latent endpoint discrepancy

Verify both recurrent infinite-horizon and finite-horizon specializations
include the direct endpoint inequality before retaining both path-length
forms:

```text
||U_n-U_m||_infty
<= L_V sup_s ||z^(n)(s)-z^(m)(s)||
<= L_V sup_s sum_{j=n}^{m-1} ||z^(j+1)(s)-z^(j)(s)||
<= L_V sum_{j=n}^{m-1} sup_s ||z^(j+1)(s)-z^(j)(s)||.
```

The first step must follow directly from head Lipschitzness. The next steps use
the triangle inequality and sup-sum interchange. Do not replace or loosen the
established geometric contractive constants.

### Signed occupancy-averaged defect and safe mixture step

Require exact statewise centering under the current policy, one common
`pi`/`pi_cand`-invariant domain `C_pair` for every `alpha in [0,1]`, and one
fixed estimator and candidate-defect object independent of `alpha`. Define

```text
b(s) = E_{a~pi_cand(.|s)}[Ahat(s,a)-A^pi(s,a)],
overline_beta_d >= E_{s~d_pi} b(s),
ghat = E_{s~d_pi,a~pi_cand(.|s)} Ahat(s,a),
M_d = ghat-overline_beta_d.
```

Here `overline_beta_d` must be finite, one-sided, established, and independent
of `alpha`. Verify exact centering gives

```text
ghat = E_{d_pi}[g(s)+b(s)],
E_{d_pi} g = ghat-E_{d_pi} b >= M_d.
```

Then verify, simultaneously for every `alpha in [0,1]`,

```text
eta(pi_alpha)-eta(pi)
>= alpha/(1-gamma)
   [M_d-gamma alpha Delta_g/(1-gamma+gamma alpha)].
```

Verify these specializations without changing the common domain or estimator:

1. exact signed defect:
   `overline_beta_d=E_{d_pi}b(s)`;
2. occupancy-averaged absolute defect:
   `overline_beta_d=E_{d_pi}|b(s)|`;
3. uniform specialization:
   `overline_beta_d=epsilon_{A,cand}`.

For `gamma>0` and `0<M_d<gamma Delta_g`, the certified threshold must be

```text
alpha <= M_d(1-gamma)/(gamma(Delta_g-M_d)),
```

with no redundant outer minimum with one. Check every branch:

- `alpha=0` is equality;
- `alpha=1` requires `M_d>=gamma Delta_g`;
- `gamma=0` and `M_d>=0` certify every `alpha`;
- `M_d<0` certifies no positive step;
- `gamma>0`, `M_d=0`, and `Delta_g>0` certify no positive step;
- `M_d=0` and `Delta_g=0` certify every step with zero lower bound;
- `gamma>0`, `M_d>0`, and `M_d>=gamma Delta_g` certify every step;
- `M_d=gamma Delta_g` belongs to the all-step branch.

### Target-network propagated-lag certificate

On one measurable invariant domain, verify the paper defines

```text
R_targ,q^P
  := epsilon_targ,q
     + gamma^K ||P_pi^K(bar V-U_q)||_infty
```

and only then gives the fallback

```text
R_targ,q^P
<= epsilon_targ,q+gamma^K||bar V-U_q||_infty
 =: R_targ,q^sup.
```

Verify the derivations

```text
||U_q-V^pi||_infty <= R_targ,q^P/(1-gamma^K),

||U_n-V^pi||_infty
<= ||U_n-U_m||_infty+R_targ,m^P/(1-gamma^K)
```

for finite `0<=n<m`. Under exact statewise centering and the stated one-step
policy-overlap assumptions, verify

```text
epsilon_{A,cand}
<= 2 gamma tau_infty R_targ,q^P/(1-gamma^K).
```

Verify composition with the CPI theorem gives

```text
eta(pi_alpha)
>= Lhat_pi(pi_alpha)
   - 2 alpha gamma tau_infty R_targ,q^P
     /((1-gamma)(1-gamma^K))
   - 2 epsilon_CPI gamma alpha^2
     /((1-gamma)(1-gamma+gamma alpha)).
```

For the finite-reference form, the linear penalty must use

```text
||U_n-U_m||_infty+R_targ,m^P/(1-gamma^K).
```

Verify the direct persistent augmented-MDP analogue replaces `P_pi` with
`P_barpi`, `U_q` with `U_tilde_q`, `T_K^pi` with `Tbar_K^barpi`, and uses the
declared augmented invariant domain. Never identify the target-network
population operator with the self-bootstrap Bellman operator. The paper must
state that empirical regression loss, finite-batch maxima, and `L2` error
establish neither required sup norm.

## Constants and semantic distinctions

Confirm these independently. Change none unless a complete derivation proves a
genuine mathematical error:

- Bellman contraction:
  `||T_K^pi V-T_K^pi W||_infty <= gamma^K||V-W||_infty`;
- finite-reference certificate:
  `||U_n-V^pi||_infty <= ||U_n-U_m||_infty +
  ||U_m-T_K^pi U_m||_infty/(1-gamma^K)`;
- exact-mixture occupancy:
  `TV(d_{pi_alpha},d_pi) <= gamma alpha/(1-gamma+gamma alpha)`;
- countable-space form:
  `||d_{pi_alpha}-d_pi||_1 <=
  2 gamma alpha/(1-gamma+gamma alpha)`;
- signed surrogate identity:
  `Lhat_pi(pi_alpha)-L_pi(pi_alpha)=Xi_alpha/(1-gamma)`;
- span CPI penalty:
  `gamma alpha^2 Delta_g/((1-gamma)(1-gamma+gamma alpha))`;
- scalar CPI penalty:
  `2 epsilon_CPI gamma alpha^2/((1-gamma)(1-gamma+gamma alpha))`;
- policy-overlap factors:
  `2 gamma delta_V tau(s)` and `2 gamma^K delta_V tau(s)`;
- deployment perturbation:
  `Delta_r delta_dep/((1-gamma)(1-gamma+gamma delta_dep))`;
- projection modulus: `(R/rho_R)L_z^pre(R)`;
- `L_z^0=1`; if `L_z=0`, then `L_z^n=0` for integer `n>=1`;
- occupancy before decision `t` uses `1-(1-alpha)^t`;
- reward mismatch after decision `t` uses
  `1-(1-delta_dep)^(t+1)`;
- total variation is one half of countable-space `l1`;
- the span-versus-TV inequality has no extra factor of two;
- the common absorbing tail is counted exactly once;
- a value-function fixed point is not a recurrent latent fixed point;
- exact mixing is pointwise probability mixing, not parameter, logit, or
  hidden-state interpolation, clipping, trust-region optimization, or
  distillation;
- the direct persistent residual certificate requires no recurrent latent
  fixed point, contraction, projection, Lipschitz head, or slow-drift
  assumption.

## Code-to-paper parity matrix

Establish or refute each item with exact paper and implementation anchors:

- Algorithm 1 and Algorithm 2 use the same terminal control flow as the formal
  MDP;
- Algorithm 2 types `x in X`, `y_0 in Y`, `h_0 in N_{>=1}`, persistent-only
  `z_0 in Z`, `gamma in [0,1)`, `n in N_0`, `K in N_{>=1}`, `alpha in [0,1]`,
  and target retention `tau in [0,1]`; it selects either `R>0` with Euclidean
  projection or an explicit identity mode, and returns the fitted value head,
  candidate policy, exact mixture, and updated target value head;
- every terminal transition is stored once and collection stops immediately;
- no action, nonabsorbing successor, or persistent latent follows termination;
- the edit clock is part of the state and partial blocks respect `K>h`;
- terminal STOP, solved terminal, budget exhaustion, and every other terminal
  cause share the correct control flow;
- the absorbing boundary and folded tail are counted once;
- persistent replay stores the pre-unroll carry and only nonterminal successors
  retain the post-unroll carry;
- current and candidate policies use one shared frozen recurrent map;
- the fixed-K target bootstraps from the frozen target evaluator;
- target retention is
  `bar_psi_next=tau*bar_psi+(1-tau)*psi_next` and updates only the target value
  head;
- projection is one Euclidean product-norm projection with `R>0`, or an
  explicit identity mode with no radius;
- exact statewise advantage centering uses the full represented action support;
- deployment is the exact pointwise policy mixture;
- the fixed-base path does not recursively promote mixtures or mutate the
  recurrent map;
- checkpoint and source identity bind the bytes and objects that executed;
- confirmatory execution verifies the complete PAR before behavior imports,
  launches a sealed immutable descriptor, and cannot reuse a shared unpack
  cache;
- the entry point verifies the same sealed runtime object that supplied its
  module before importing model or RL code;
- the verified runtime digest is present in every current evidence identity,
  effective configuration, lock, save, and resume binding without changing the
  historical schema contracts;
- the canonical config is registered but not treated as authorized evidence;
- diagnostics and documentation do not claim that finite observations prove
  uniform mathematical premises.

Keep executable parity separate from unproved theorem assumptions.

## Phase 4 publication pipeline

Treat Phase 4 as an optional publication pipeline, not as evidence for the
paper's conditional theorem. Review its numerical semantics and provenance as
strictly as any other paper-facing output.

For `L_preproj`, trace the exact production latent context, including plan
embeddings and puzzle identifiers, into the unprojected recurrent update.
Prove that the diagnostic calls the same map before radial projection. Check
that each perturbation is normalized in the joint Euclidean product space and
that the quotient divides by the measured post-scaling joint norm. Reproduce
the equal-component `sqrt(2)` case and a projection-saturation counterexample.
Require the recorded sample count to equal the number of accepted finite
directional quotients and reject an empty or partial sample set.

For `argmax_agreement_n4` and `argmax_agreement_n8`, which the publication
output labels at absolute `n=4` and `n=8`, trace each compared depth from the
common initial carry. Prove that the diagnostic calls `model.policy_dist`,
supplies the exact environment action mask, and uses the production `z_H`
policy latent. Use a synthetic deterministic head where `z_H != z_L` and the
largest raw logit is a masked action. A renamed alternative latent-head
diagnostic is not a repair for a field published as executable-policy
stability.

For each of the 12 Phase 4 records, require strict loading of the complete
checkpoint state and exact validation of the saved model config, RL config,
condition, seed, run ID, step counters, source identity, dataset identity,
random-initialization kind, execution device, exact replay capacity and
transition types, replay/collector state, value and policy optimizer states,
old/candidate/target model states, auxiliary model state, and RNG state required
by the current checkpoint contract. Verify that checkpoint SHA-256, canonical
model-state SHA-256, canonical config identities, and diagnostic-array hashes
are recomputed from bytes. Reject missing or unexpected behavior keys, swapped
conditions or seeds, nonexistent paths, path escape, edited identity fields,
inconsistent metric aggregates, and payloads reconstructed from fallback
architecture values. Confirm audit and figure consumers revalidate the same
identity objects before writing output. Do not claim that they can detect an
arbitrarily edited metric summary whose values and aggregates remain internally
schema-consistent without recomputing the diagnostics.

For Phase 4 source provenance, enumerate the exact source profile for each of
the evaluator, audit, and figure tools. Compare the clean checkout inventory
and bytes with Git `HEAD`, then compare those bytes with the actual PAR member
inventory. Actively test stale runtime bytes, an ignored added `.py`, a hidden
tracked deletion, `skip-worktree`, `assume-unchanged`, selected `.pyc/.pyo`, a
missing or extra selected source, a duplicate member, and a raw/effective ZIP
name mismatch. All tests must use temporary repositories or archives. Do not
alter either reviewed worktree's Git index flags.

Trace the pre-import boundary for all four Phase 4 roles. Start at the external
`phase4_runtime_launcher`, through complete archive validation and sealed-memfd
copy, then follow training into `upi_trm_train` and consumer roles into
`phase4_runtime_entrypoint`. Prove that role, whole-artifact digest, seals,
module origin, and private unpack directory are checked before NumPy, Torch,
`dataset`, `models`, `rl`, `utils`, or any Phase 4 behavior module is imported.
Direct execution, role mismatch, wrong digest, stale bytes, protected argument
overrides, and startup failure must fail closed and clean up the exact private
directory object.

Independently authorize the training producer. Resolve the claimed commit in a
clean explicit source checkout, reconstruct or verify its tracked producer
manifest, and compare the resulting commit and manifest digest with every
checkpoint. Confirm that all 12 records bind the same externally authorized
training-PAR SHA-256. A checkpoint hash only binds embedded strings; it does
not establish that the producer claims are true.

For checkpoint replay, reconstruct the exact registered environment and run
every retained transition through `PlanEditEnv`. Challenge action support,
serialized state, reward, clock, terminal flag and reason, post-terminal
records, collector state, replay capacity/order, and the nested resume-state
inventory. Use fresh fixtures for each negative case so one mutation cannot
mask another. Confirm that the figure writer stages output and repeats all
external identity checks immediately before atomic publication. Confirm that
the evaluator and audit complete their required checks before direct writes.

## Boundary cases

Check at least:

- `gamma=0`;
- `alpha=0`, interior `alpha`, and `alpha=1`;
- `tau=0`, fractional `tau`, and `tau=1`;
- `M_d<0`, `M_d=0`, `M_d>0`, and `M_d=gamma Delta_g`;
- `Delta_g=0`;
- `n=0`;
- `L_z=0` at `n=0` and `n>=1`;
- `H=0`;
- `h=0`;
- `K>h`;
- terminal STOP;
- solved terminal;
- budget exhaustion;
- mixed terminal/nonterminal and all-terminal batches;
- episodic and persistent latent modes;
- projection disabled;
- legacy radius zero;
- zero, inside, boundary, outside, tiny, and huge projection inputs;
- supported floating-point dtypes;
- corrupt checkpoints;
- missing and unexpected checkpoint keys;
- swapped Phase 4 condition or seed;
- nonexistent and escaping checkpoint paths;
- `z_H != z_L` with a masked maximum-logit action;
- equal and unequal joint latent perturbation component norms;
- zero, partial, nonfinite, and complete Phase 4 metric samples;
- stale Phase 4 PAR source bytes and hidden Git index/worktree state;
- direct Phase 4 PAR execution, wrong digest, and runtime-role mismatch;
- fabricated or unavailable producer commits and manifests;
- mixed training-runtime digests across the 12 records;
- impossible replay actions, rewards, successors, terminal causes, or clocks;
- imported `dataset/*.py` omitted from a runtime source profile;
- publication failure before the final identity check;
- `delta_dep=0` and `delta_dep=1`.

For each, cite the proof, code path, existing test, or temporary reproducer and
state any remaining gap.

## Exact-source paper validation

Build the frozen paper commit from a clean temporary export through its
canonical command, `make clean && make pdf` from `UPI_TRM_ICLR`. Record:

- exact command and workdir;
- exit code;
- page count, byte count, and PDF SHA-256;
- all final warnings, undefined references, missing citations, overfull and
  underfull boxes, errors, and fatal errors;
- confirmation that every page was inspected for clipping, equation overflow,
  malformed algorithms, detached proofs, broken references, table overflow,
  and source/PDF disagreement.

Run full-source searches for stale theorem, equation, proposition, section,
appendix, citation, projection, target-network, exact-mixture, and
finite-diagnostic claims. Confirm every changed or questioned citation exists
in `trm_rl.bib` and supports the attribution.

## Exact implementation validation

Confirm `/data/users/buiksat/fbsource/fbcode/buiksat_trm` resolves to the frozen
implementation repository before using Buck. Do not create or mutate that link
silently. Run the parity runtime gate from `/data/users/buiksat/fbsource`:

```bash
buck2 test --local-only @fbcode//mode/opt \
  fbcode//buiksat_trm:test_config_integrity \
  fbcode//buiksat_trm:test_plan_edit_env \
  fbcode//buiksat_trm:test_theory_exact_components \
  fbcode//buiksat_trm:test_run_identity \
  fbcode//buiksat_trm:test_augmented_replay_diagnostics \
  fbcode//buiksat_trm:test_algorithm2_boundary_contract \
  fbcode//buiksat_trm:test_rl_k_step_targets \
  fbcode//buiksat_trm:test_rl_k_step_value_update_trainer \
  fbcode//buiksat_trm:test_upi_trm_trainer_smoke \
  fbcode//buiksat_trm:test_persistent_checkpoint_diagnostics \
  fbcode//buiksat_trm:test_upi_trm_logging_smoke \
  fbcode//buiksat_trm:test_cleanrl_regressions \
  fbcode//buiksat_trm:test_unroll_sensitivity \
  fbcode//buiksat_trm:test_phase4_reporting \
  fbcode//buiksat_trm:test_phase4_runtime_launcher
```

Run the complete 30-target affected type gate from the same workdir:

```bash
buck2 build --local-only @fbcode//mode/opt \
  fbcode//buiksat_trm:models-type-checking \
  fbcode//buiksat_trm:eval_unroll_sensitivity_lib-type-checking \
  fbcode//buiksat_trm:script_eval_unroll_sensitivity_lib-type-checking \
  fbcode//buiksat_trm:utils-type-checking \
  fbcode//buiksat_trm:test_persistent_checkpoint_diagnostics-library-type-checking \
  fbcode//buiksat_trm:test_augmented_replay_diagnostics-library-type-checking \
  fbcode//buiksat_trm:confirmatory_runtime_launcher_lib-type-checking \
  fbcode//buiksat_trm:confirmatory_runtime_launcher-library-type-checking \
  fbcode//buiksat_trm:runtime_archive_preflight-type-checking \
  fbcode//buiksat_trm:phase4_runtime_profile-type-checking \
  fbcode//buiksat_trm:phase4_runtime_launcher_lib-type-checking \
  fbcode//buiksat_trm:phase4_runtime_launcher-library-type-checking \
  fbcode//buiksat_trm:phase4_runtime_entrypoint-type-checking \
  fbcode//buiksat_trm:test_phase4_runtime_launcher-library-type-checking \
  fbcode//buiksat_trm:audit_phase4_paper_ready-library-type-checking \
  fbcode//buiksat_trm:cleanrl_runner-library-type-checking \
  fbcode//buiksat_trm:eval_phase4_2x2_norm_ablation-library-type-checking \
  fbcode//buiksat_trm:make_paper_figures_phase4-library-type-checking \
  fbcode//buiksat_trm:phase4_result_schema-type-checking \
  fbcode//buiksat_trm:test_phase4_reporting-library-type-checking \
  fbcode//buiksat_trm:phase4_checkpoint-type-checking \
  fbcode//buiksat_trm:phase4_diagnostic_inputs-type-checking \
  fbcode//buiksat_trm:phase4_source-type-checking \
  fbcode//buiksat_trm:run_phase4_training-library-type-checking \
  fbcode//buiksat_trm:upi_trm_train-library-type-checking \
  fbcode//buiksat_trm:upi_trm_train_lib-type-checking \
  fbcode//buiksat_trm:test_run_identity-library-type-checking \
  fbcode//buiksat_trm:test_theory_exact_components-library-type-checking \
  fbcode//buiksat_trm:test_upi_trm_logging_smoke-library-type-checking \
  fbcode//buiksat_trm:test_upi_trm_trainer_smoke-library-type-checking
```

Do not treat failures from an optional repository-wide type target as part of
this required gate; if such a target is run, report its complete output and
distinguish pre-existing errors from errors on changed lines.

Build and exercise the real packaged-runtime boundary without starting
training:

```bash
buck2 build --local-only @fbcode//mode/opt --show-output \
  fbcode//buiksat_trm:upi_trm_train \
  fbcode//buiksat_trm:confirmatory_runtime_launcher \
  fbcode//buiksat_trm:cleanrl_runner

sha256sum /absolute/path/to/upi_trm_train.par

/absolute/path/to/confirmatory_runtime_launcher.par \
  --runtime-archive /absolute/path/to/upi_trm_train.par \
  --expected-runtime-sha256 <actual-lowercase-sha256> -- \
  --confirmatory --help

/absolute/path/to/cleanrl_runner.par --help
```

Build the Phase 4 launcher and three publication tools without running an evaluator or
loading a checkpoint:

```bash
buck2 build --local-only @fbcode//mode/opt --show-output \
  fbcode//buiksat_trm:phase4_runtime_launcher \
  fbcode//buiksat_trm:eval_phase4_2x2_norm_ablation \
  fbcode//buiksat_trm:audit_phase4_paper_ready \
  fbcode//buiksat_trm:make_paper_figures_phase4
```

Hash each artifact. Exercise training, evaluator, audit, and figure `--help`
only through `phase4_runtime_launcher`, using the correct role, artifact digest,
clean checkout, and source commit. Record each artifact SHA-256, canonical
profile digest, selected member count, exact exit, and private-directory cleanup.
A Buck success is not source-identity evidence by itself. Run wrong-digest,
role-mismatch, direct-PAR, and protected-child-argument negative cases. Do not
execute an evaluator, load a checkpoint, create a Phase 4 summary, or run
training as part of this audit.

Record the training, launcher, and CleanRL artifact paths, their SHA-256 values,
exit codes, stderr, and the count of `upi_trm_confirmatory_unpack.*`
directories before and after. The CleanRL help command must dispatch no
training. Also require all of these negative cases to fail before training:

- direct `upi_trm_train.par --confirmatory --help` execution;
- a wrong, uppercase, or malformed expected SHA-256;
- an unsealed or unavailable descriptor;
- source/config inventory tampering, extra or missing behavior sources,
  duplicate members, root aliases, noncanonical or colliding paths, and
  behavior bytecode;
- child startup failure, with private-unpack cleanup.

Inspect environment sanitization for loader, Python-path, shell-function, and
PAR override hooks. Do not call a unit-test mock a real launcher integration.

Record exact commands, resolved repository link, Buck version, exit codes,
pass/fail/timeout/infrastructure/build counts, and warnings. If the Buck
checkout or exact anchor mapping is unavailable, do not silently test another
revision. Record the blocker and use `INCOMPLETE` unless equivalent hermetic
current-anchor evidence is produced.

Also:

- run `git diff --check` for the relevant committed ranges and tracked
  worktree changes;
- regenerate the producer manifest in memory and compare it byte-for-byte or
  structurally with the committed manifest without writing it;
- require the documented source-entry count to match current source rather
  than blindly expecting an old count;
- syntax-check every tracked shell script;
- capture state before and after executing the retired launcher, require its
  documented fail-closed exit, and prove it produced no side effects;
- parse every tracked JSON and YAML file;
- compile every tracked Python source without importing or executing it;
- inspect Buck ownership of every synchronization-touched test;
- inspect the complete final report and verify every cited current line.

The current handoffs record the following maintainer results at the frozen
source anchors. Treat them only as claims to reproduce or refute:

- parity runtime gate: 366 passed, with zero failures, timeouts, fatal errors,
  infrastructure failures, or build failures;
- affected type gate: 30 of 30 targets built successfully;
- producer manifest: 80 entries;
- training, launcher, and CleanRL artifact SHA-256 values:
  `cafabc3314730f09f6251144a2d2d06c329d7f93f5431a59f252f71f8b6e708e`,
  `faf969e3353b0e78b89042472f37eb900913cd0e09f3196715c793cc7d8c4c54`,
  and `c2a088a72d3e65a4b0b977ae73e58188ee3b80940802fd2804497d470b978f0b`;
- Phase 4 launcher SHA-256:
  `a83574644a024b337f0d384814b56cdb4600628a0819635136e7c91828002de9`;
- Phase 4 evaluator, audit, and figure artifact SHA-256 values:
  `82b65fc5a9f8714140ddc8769882b07921eb4a17b71319036232c713768def19`,
  `e0943bf4a0db909c8f7b783d7dc3d92fe4db2e3eb96f4dfe79f5aee218cc2495`,
  and `625d1dccff6dd611cfc7341d7134c7f57fd9825292f02bcb86e0285845ad4057`;
- corresponding Phase 4 source-profile digests:
  `a2ddc393a9d1849a41ab194b9e3bab75b4945dcdbbc1c3db94c91b4225dbb06a`,
  `24ad45bf18566dc48b6df15e9bc31ea4cca42f3c6056156a0cba57491d9c15c0`,
  and `7024d357f9653ca8ab39b8d186b7b045474703c73f2faf073bedd6177efcb4e0`;
- static gates: 46 shell, 625 JSON, 96 YAML, and 251 Python files;
- canonical paper: 38 pages, 544,004 bytes, SHA-256
  `de4fde697f7c835ef2827884569a5a44a12cca473758f91b4fd7dea6d298961d`.

The maintainer also records successful authenticated `--help` paths for all
four Phase 4 roles, with wrong-digest, role-mismatch, and direct-PAR cases
failing closed. Reproduce those checks and the exact source-profile comparison.

Do not copy historical pass counts as current evidence. The repository-wide
package diagnostic is optional because it includes known unrelated type debt.
If run, report every failure without relabeling it.

## Finding standard

Use these severities:

- `P0 Critical`: catastrophic security or data loss, or unavoidable
  repository-wide unsoundness;
- `P1 Blocking`: a reachable defect that invalidates executable parity or a
  central theorem;
- `P2 Major`: substantial correctness, provenance, reliability, numerical, or
  coverage defect outside the always-executed core;
- `P3 Minor`: a localized real defect, stale claim, or missing defense.

For every surviving finding include:

- stable ID;
- severity and confidence;
- exact file, current line, and symbol;
- affected invariant or theorem;
- reachable trigger;
- source evidence and full control-flow trace;
- paper or contract mismatch;
- derivation, counterexample, or deterministic reproducer;
- impact;
- smallest valid repair;
- whether it is pre-existing or introduced by the reviewed change;
- independent skeptic outcome.

List rejected candidates separately with concise refutations. Do not report
style preferences, strong but explicit assumptions, or speculative risks as
defects.

## Required report

Write one self-contained Markdown report to the permitted report path. It must
contain:

1. review metadata and frozen commit anchors;
2. verdict;
3. methodology and independent-pass disclosure;
4. repository coverage ledger and exclusions;
5. exact paper build and page-inspection results;
6. theorem, quantifier, constant, and boundary audit;
7. code-to-paper parity matrix;
8. verified findings;
9. rejected candidates and cleared risks;
10. validation commands and exact results;
11. prior-report reconciliation for every `UPITRM-REV-*`, `UPITRM-UPD-*`,
    `UPITRM-GPT-*`, and `UPITRM-POST-*` item;
12. claim, citation, and stale-reference audit;
13. recommended repairs in priority order;
14. initial and final worktree attestation for both repositories;
15. a machine-readable JSON appendix.

Choose exactly one verdict:

- `PASS`;
- `PASS WITH NONBLOCKING FINDINGS`;
- `FAIL`;
- `INCOMPLETE`.

`PASS` requires complete required coverage and validation with no surviving
finding. `PASS WITH NONBLOCKING FINDINGS` permits only verified P2/P3 issues
that do not invalidate executable parity or a central theorem. Use `FAIL` for
a verified P0/P1 or another defect that invalidates claimed parity. Use
`INCOMPLETE` when required direct access, coverage, exact-source build, safe
validation, or adversarial verification is unavailable.

State explicitly what the verdict does not certify. Include initial and final
`git status --short` for both repositories and confirm that the report is the
only review-created delta.

The review is incomplete until both frozen commits have been inspected
directly, the paper has been rebuilt from its exact source and every page
inspected, every behavior-bearing implementation dependency has been reviewed,
every candidate has passed default-reject verification, required safe checks
have run or been classified, and the report has been written.
