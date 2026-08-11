# UPI-TRM paper parity report

Date: 2026-08-06

- Paper anchor: `/home/buiksat/UPI_TRM`, commit `2107125a50df1471393bc9a5e387c3005d7db112`
- Paper branch at inspection: `iclr-evidence-aligned-revision`
- Implementation anchor: branch `full-implementation`
- Implementation scope: the current working tree, including uncommitted repairs

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
descriptor. Every launch uses a fresh private PAR unpack directory rather than
a shared extraction cache, and the launcher removes it when the child exits.
The entry point independently verifies the descriptor seals, rehashes its
bytes, and validates the private unpack directory before importing model or RL
modules. Source-tree runs and direct unattested PAR runs cannot enter
`--confirmatory` mode. Effective
configuration schema 4, evidence-identity schema 2, and lock schema 4 bind the
verified PAR digest without embedding that self-referential digest in the PAR.
The trusted launcher process, host, and initial environment remain the
external trust root; the launcher strips loader, Python-path, and PAR override
hooks before it starts the training artifact.

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
  fbcode//buiksat_trm:test_unroll_sensitivity
```

Result: 243 passed, 0 failed, 0 timed out, 0 infra failures, and 0 build
failures. Buck emitted one `slow_snapshot` soft warning during the run.

After restarting Buck to invalidate its symlink cache, the final type gate for
the synchronization-touched model, evaluator, utility, and checkpoint targets
passed 6/6:

```text
buck2 test --local-only @fbcode//mode/opt \
  fbcode//buiksat_trm:models-type-checking \
  fbcode//buiksat_trm:eval_unroll_sensitivity_lib-type-checking \
  fbcode//buiksat_trm:script_eval_unroll_sensitivity_lib-type-checking \
  fbcode//buiksat_trm:utils-type-checking \
  fbcode//buiksat_trm:test_persistent_checkpoint_diagnostics-library-type-checking \
  fbcode//buiksat_trm:test_augmented_replay_diagnostics-library-type-checking
```

A repository-wide diagnostic command,
`buck2 test --local-only @fbcode//mode/opt 'fbcode//buiksat_trm:'`, reported
554 passes and 26 failures before the final type-only cleanup. All 26 failures
were generated Python type-check targets; runtime tests had no failures. Six
failures in synchronization-touched targets were then repaired and cleared by
the final 6/6 type gate above. Other pre-existing generated type-check failures
remain across aggregate libraries, tests, runners, diagnostics, and experiment
scripts outside the six cleared targets, so the all-target package command is
not claimed green.

The checked producer manifest equals a fresh mechanical regeneration and
contains 78 source entries. `git diff --check` passes. The complete diff,
including both new files, was inspected after the final repairs.

## Experiment status

No experiment was added, requested, rerun, reinterpreted, or used to justify
this parity assessment. This report contains no empirical claim.
