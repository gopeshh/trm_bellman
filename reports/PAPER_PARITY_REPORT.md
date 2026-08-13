# UPI-TRM paper parity report

Date: 2026-08-13

- Paper source anchor: `/home/buiksat/UPI_TRM`, commit
  `5253692fea5e77cfde3a130c50351183dc0268e3`
- Paper handoff head: `a76eb29f2712042742fea738cdb859d354fafce2`
- Paper branch at inspection: `iclr-evidence-aligned-revision`
- Implementation source anchor: branch `full-implementation`, commit
  `980f6ede14717e87ad68ceb32acc111bdd7fca1b`
- Implementation parent: `86ec7363103b3d6a6a36fb25094ec1991cefd48e`
- Previous behavior anchor: `f86bddb607adcd24eba65fd5869af58f91742a52`

## Result

The current implementation has verified source-level mappings for the repaired
Algorithm 1 and Algorithm 2 control flow and object ownership described below.
The green gates apply to the canonical `fixed_base_exact` profile and the
parity targets listed under Validation status.

This report covers executable mechanics. It does not establish the population
sup-norm, occupancy, overlap, invariance, boundedness, contraction, or
measurability premises used by the paper's conditional results. It also makes
no claim about experiment results, learned-model performance, SGD, TD
convergence, BPTT, distillation, or training dynamics.

## Algorithm 1 executable mapping

| Paper contract | Implementation mapping | Source-level status |
| --- | --- | --- |
| The state carries the edit clock. | `PlanEditEnv` stores `remaining_edits`; replay validation requires it in both transition endpoints for exact backups. | Implemented. |
| Collection uses the fixed current policy at one snapshot. | `fixed_base_exact` routes collection through `policy_model_old`, not through the deployed mixture. The fixed base and recurrent map are not promoted after the candidate update. | Implemented for the explicit fixed-base protocol. |
| Sample one action, step once, and retain the terminal transition. | `_collect_episode_impl` samples once, calls `env.step`, and appends that transition to replay before testing `done`. | Implemented. |
| Every terminal cause ends collection immediately. | After storing a `done` transition, the collector finishes the active episode and breaks. Solved states, terminal STOP, and budget exhaustion are environment terminal causes. | Implemented. |
| No action or persistent latent follows a terminal transition. | The collector breaks before another policy call. A terminal replay record always has `next_latent=None`. | Implemented. |
| The absorber has a separate finite scalar boundary. | Training paths evaluate successor values only for nonterminal rows, and the folded return uses the configured `C_max` boundary. Serialized terminal `x_next` and `y_next` tensors are bookkeeping, not a nonabsorbing successor or an absorbing latent. | Semantically implemented without a literal absorber tensor. |
| The current-policy advantage estimate is exactly statewise centered. | With `exact_baseline_summation: true`, the discrete-action baseline is summed under the current policy at each represented state. | The arithmetic is implemented; its theorem-facing population accuracy remains an assumption. |

## Algorithm 2 executable mapping

| Paper contract | Implementation mapping | Source-level status |
| --- | --- | --- |
| Typed scalar inputs. | `RLConfig` enforces `gamma` in `[0,1)`, `inner_unroll_n >= 0`, `K >= 1`, and `mixture_alpha` and `target_ema_tau` in `[0,1]`. `max_edits >= 1` supplies the initial clock budget. | Implemented at configuration boundaries. |
| Projection is either Euclidean projection with `R > 0` or the identity. | `latent_projection_mode: enabled` requires a finite positive `latent_ball_radius`; `disabled` requires no radius and applies no projection. Radius zero is migration input, not the disabled-mode convention. | Implemented. |
| Persistent latent state is carried only across nonterminal augmented states. | Persistent replay records store the input latent and a nonterminal `next_latent`; terminal records reject a successor latent. Episodic mode carries neither. | Implemented. |
| The fixed `K`-step return respects terminal and clock boundaries. | Exact segment validation requires contiguous episode IDs, timesteps, and decreasing clocks. The folded target stops at the first terminal and applies the common absorbing boundary once. | Implemented. |
| Fit the value head without mutating the shared recurrent map. | In `fixed_base_exact`, the value optimizer owns the value head, while collection and policy evaluation retain the frozen base and recurrent map. | Implemented for the explicit fixed-base protocol. |
| Build a candidate and deploy the exact policy mixture. | `_probability_mixture_dist` returns `(1-alpha) * p_current + alpha * p_candidate` pointwise. The exact configuration uses zero exploration mixture and disables distillation. | Implemented. |
| Update the target value head with retention coefficient `tau`. | In `fixed_base_exact`, `_soft_update_target` applies `bar_psi <- tau * bar_psi + (1-tau) * psi` only to `value_head`; schema-v5 guards require every recurrent-map tensor to remain bitwise shared. | Implemented. |
| Expose the updated objects. | The trainer retains the fitted value head, candidate head, exact-mixture callback, and updated target evaluator as distinct objects. | Implemented. |

Exact probability mixing here is not parameter interpolation, logit
interpolation, hidden-state interpolation, clipping, trust-region optimization,
or distillation.

## Target-network distinction

The exact fixed-`K` fitted target uses `self.target_model` as the frozen target
evaluator. It does not use the value head currently being optimized. The
target update then applies the retention convention

`bar_psi_next = tau * bar_psi + (1 - tau) * psi_next`.

This is the paper's target-network population backup, not the self-bootstrap
operator `T_K^pi U_q`. A certificate based on this backup still needs the
target residual and propagated target lag on the declared invariant domain.
An empirical regression loss, a finite-batch maximum, or an `L2` error does
not establish either required sup norm.

## Registered exact configuration and reference fixture

The registered exact parity config is
`configs/iclr_confirmatory/c2_upi.yaml`. The matrix status is
`registered_not_authorized`, so it cannot currently start a confirmatory run.
The config selects terminal STOP, unclipped
value targets, explicit enabled projection with radius `10.0`, exact fixed-`K`
targets, exact statewise baseline summation, exact probability mixing, one
fixed base, zero exploration mixture, no distillation, no scheduled
operator-norm clamp, and no asserted recurrent contraction.

The episodic config at
`configs/revision/upi_trm_feasibility_episodic_z_hard_suite_theory_exact.yaml`
records the corresponding settings as a historical reference fixture. It is
not registered in the confirmatory run matrix and the main runner does not
accept it as a `fixed_base_exact` launch. It is therefore not an executable
entry point, and this report does not define a replacement experiment
protocol for it.

The matched `c2_ppo.yaml` control also uses terminal STOP so the two C2 cells
retain one common edit MDP. The historical `Bt` bridge toggle was removed from
the executable run matrix: after target-network synchronization it no longer
changed the bootstrap evaluator, so retaining it as a one-factor contrast
would have been false. Its old config remains as historical source only.

Confirmatory execution also has a pre-import artifact boundary. The trusted
stdlib-only launcher authenticates an absolute standalone PAR against an
externally frozen SHA-256 and validates the embedded source/config manifest
before it copies the verified bytes into a sealed anonymous file and executes
the training entry point as a supervised child through that immutable
descriptor. Every launch binds its private PAR unpack directory to an inherited
directory descriptor rather than a shared extraction cache. The entry point
independently verifies both descriptors, rehashes the runtime bytes, and
validates the unpack-directory identity before importing model or RL modules.
Both archive-validation layers reject a member when its raw ZIP name differs
from its effective CPython `ZipInfo.filename`, including Unicode Path extra
field rewrites on nonbehavior members.
Source-tree runs and direct unattested PAR runs cannot enter
`--confirmatory` mode. Effective
configuration schema 4, evidence-identity schema 2, and lock schema 4 bind the
verified PAR digest without embedding that self-referential digest in the PAR.
New checkpoint-schema-5 writes require effective-config schema 4. Historical
effective-config schemas remain readable under their original contracts but
cannot create new schema-5 evidence.
The trusted launcher process, operating system, host namespace, other same-UID
processes, and initial environment remain the external trust root. The launcher
is not a sandbox against that root. It strips loader, Python-path, and PAR
override hooks before it starts the training artifact.

## Conditional theorem premises not certified by execution

An invocation of the paper's certificates must separately establish, on the
same declared domain and fixed snapshot:

- one fixed MDP, one fixed current/candidate policy pair, and one fixed
  parameter snapshot;
- one common invariant domain for every exact mixture weight used by a result;
- one shared frozen recurrent map when current and candidate policies are
  represented in the same persistent-latent augmented MDP;
- uniform boundedness of both value heads on every represented nonabsorbing
  state and every latent in the declared invariant set, plus the separate
  finite absorber boundary;
- the population target residual and propagated target-network lag in the
  required sup norms;
- any policy-overlap factor, signed occupancy-averaged candidate defect bound,
  and safe-step margin used by a CPI specialization;
- recurrent invariance, contraction, head Lipschitzness, and fixed-point
  measurability when invoking a specialization that requires them; and
- any deployment discrepancy when evaluation does not use the exact
  probability mixture.

Exact summation implements pointwise centering for represented discrete-action
states. It does not by itself prove that the fitted estimator has the required
uniform error or that a sampled diagnostic covers the theorem's domain.

## Protocol boundary

`fixed_base_exact` implements one frozen-base proposal at one snapshot. It is
not a theorem about a recursive sequence of promoted policies or about
optimization convergence. Schema-v5 checkpoints bind this protocol and its
object roles so historical mutable checkpoints cannot be reinterpreted as
theorem-facing runs.

Phase 4 paper-facing summaries use strict schema version 3. They mark success,
final loss, and training-history NaN status unavailable instead of serializing
placeholder measurements. `L_preproj` uses the exact plan-conditioned
pre-projection map and actual joint perturbation norm. Policy stability calls
the production masked `policy_dist` path on `z_H` at absolute depths 2, 4, and
8. Publication requires the exact four-condition by three-seed design,
registered sample counts, fixed toggles and projection settings, and aggregates
recomputed from all 12 measured records.

Each record binds a strict full checkpoint, complete model/config/run identity,
checkpoint and model-state SHA-256 values, and the exact ordered diagnostic
input population. The summary also records a canonical evaluator-source
manifest digest. Evaluator, audit, and figure PARs compare their runtime source
members with the explicit checkout, its complete Git `HEAD` inventory, safe
index flags, and HEAD-identical worktree bytes. Audit and figure consumers
reopen all 12 checkpoints and revalidate source, config, and input identities
before consuming metrics. This reporting contract does not turn the finite
diagnostics into a uniform theorem certificate.

## Validation status

Validation was run from `/data/users/buiksat/fbsource` with this repository
linked as `fbcode/buiksat_trm`.

The final current-source parity runtime gate was:

```text
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
  fbcode//buiksat_trm:test_phase4_reporting
```

Result: 338 passed, 0 failed, 0 timed out, 0 fatal, 0 infra failures, and 0
build failures.

The complete changed-surface type gate passed 24/24 targets:

```text
buck2 build --local-only @fbcode//mode/opt \
  fbcode//buiksat_trm:models-type-checking \
  fbcode//buiksat_trm:eval_unroll_sensitivity_lib-type-checking \
  fbcode//buiksat_trm:script_eval_unroll_sensitivity_lib-type-checking \
  fbcode//buiksat_trm:utils-type-checking \
  fbcode//buiksat_trm:test_persistent_checkpoint_diagnostics-library-type-checking \
  fbcode//buiksat_trm:test_augmented_replay_diagnostics-library-type-checking \
  fbcode//buiksat_trm:confirmatory_runtime_launcher_lib-type-checking \
  fbcode//buiksat_trm:confirmatory_runtime_launcher-library-type-checking \
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

A historical pre-repair repository-wide diagnostic command,
`buck2 test --local-only @fbcode//mode/opt 'fbcode//buiksat_trm:'`, reported
554 passes and 26 failures before the final type-only cleanup. All 26 failures
were generated Python type-check targets; runtime tests had no failures. Six
failures in synchronization-touched targets were then repaired and cleared by
the changed-surface type gate above. Other pre-existing generated type-check
failures remain across aggregate libraries, tests, runners, diagnostics, and
experiment scripts outside the cleared targets, so the all-target package
command is not claimed green.

The checked producer manifest equals a fresh mechanical regeneration and
contains 79 source entries. The final training PAR SHA-256 is
`1a01d06695200a48b160b10d81fa7160750c3499a133b6cfcdda9b110a5ba577`.
The launcher and CleanRL dispatcher SHA-256 values are
`46cb7201226f39718ea83135e397ec6618b9ab344d7ed963a8c26ddf2d4d83ae` and
`54cb744038a121ef5715c0d52cc87e6e9c7c19616a3a836bb3b96e3f3b8cc9ad`.
The real launcher help path exited 0 with zero private unpack directories before
and after. Wrong-digest, uppercase-digest, malformed-digest, and direct-PAR
confirmatory invocations exited 2, 2, 2, and 1, respectively. The packaged
CleanRL dispatcher also built and its help path exited 0 without training.

The first post-repair Phase 4 PAR inspection rejected stale cached
`phase4_source.py` and `run_identity.py` members. After a clean-daemon rebuild,
the evaluator, audit, and figure PARs matched their exact committed source
profiles. Their artifact SHA-256 values are, respectively,
`e03022090542be8773069b7ba5ccf3e630b220b3f1ac943881394c3a81fc7dc0`,
`50ccb5891320e9a1cd4455795a9da0896d54386850653278e233016ec492875f`, and
`6bbedeebe4ce9b5d77581f5f3c5144868baf2b4e68ebe3410a530c3b6df33744`.
Their canonical runtime source-profile digests are
`99cdbf794b79e70e21cab59eeb4e0143042875a8a233f919075e3d3a24d38500`,
`43211cb68189983870d8877424e840ce20364adec2dbda8dfbd5b1102c28c335`, and
`246f9396498572487056f843d9699f2779adaea1440248bde7f8b4f899b61b08`.

All 46 tracked shell scripts passed `bash -n`; 625 tracked JSON files and 96
tracked YAML files parsed successfully; and 246 tracked Python files passed
source compilation. The retired launcher exited 2 without changing the
worktree. The canonical paper built 38 pages, all pages were inspected, and
the PDF SHA-256 was
`2b5a930136b7c81d2f3e8cc59ea7aa27ab837c913cf7cd2d07200b26a940a534`.
`git diff --check` passes. The complete source diff was inspected before
commit.

## Experiment status

No experiment was added, requested, rerun, reinterpreted, or used to justify
this parity assessment. This report contains no empirical claim.
