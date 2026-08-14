# ICLR repair handoff

Date: 2026-08-13

Implementation branch: `full-implementation`

Implementation source anchor: `de013fd3fcaae8bc80d124c0c868c7c8611aeede`

Implementation parent: `12fa350951c31e8635ee51747f6de25295c07333`

Previous behavior anchor: `980f6ede14717e87ad68ceb32acc111bdd7fca1b`

Paper source anchor: UPI-TRM commit
`5253692fea5e77cfde3a130c50351183dc0268e3`

Paper handoff head: `5ad61a16281f3b509d0fc2e1756911ae86ae0c1c`

This branch contains the repaired implementation baseline. It is not a
completed confirmatory experiment and must not be cited as one. The detailed
Algorithm 1/2 mapping and the boundary between executable parity and
conditional theorem premises are in `reports/PAPER_PARITY_REPORT.md`.

## Implemented at the current validated source state

- Clock-complete persistent replay with carried input/successor latents,
  remaining-edit clocks, transition and segment validation, and terminal
  masking. Terminal records retain no successor latent and no record may follow
  a terminal transition.
- Collection stops immediately after solved, terminal STOP, budget exhaustion,
  or another environment terminal cause. A collector cap pauses a live episode
  without fabricating a terminal or optimizer boundary.
- Persistent exact baseline enumeration uses the recorded augmented state and
  the common post-unroll latent on nonterminal successors.
- Exact K-step sampling rejects incomplete nonterminal segments and bootstraps
  from the frozen target evaluator. This target-network population backup is
  distinct from self-bootstrapping with the value head currently being fitted.
- Explicit `fixed_base_exact` training collects from one frozen base, fits only
  the policy-independent value head, optimizes one candidate policy head, and
  evaluates the exact probability-space mixture without recursively promoting
  it.
- The current exact protocol requires terminal STOP, an unclipped value target,
  exact statewise baseline summation, zero exploration mixture, no
  distillation, and no scheduled recurrent-map mutation.
- Recurrent projection has two explicit modes: Euclidean projection at a finite
  positive radius, or a projection-disabled identity operator with no radius.
  Radius zero is only a migration input for historical payloads.
- Schema-v5 checkpoints bind training protocol, run and source identity, model
  construction, target state, replay and live collector state, schedulers,
  optimizer state, counters, dataset provenance, and process RNG state.
- Persistent-checkpoint diagnostics report finite-batch quantities only. They
  are not uniform residual, bounded-head, contraction, overlap, or safe-step
  certificates.
- Confirmatory execution verifies and seals the complete PAR before behavior
  imports, passes the private unpack root through an inherited directory
  descriptor, rejects noncanonical or colliding archive paths, and rejects any
  raw/effective ZIP member-name mismatch. The host namespace and other same-UID
  processes remain part of the external trust root.
- The schema-v5 writer accepts only effective-config schema 4 for new
  checkpoints. Historical schemas remain readable but cannot mint new
  schema-v5 evidence.
- Phase 4 computes `L_preproj` from the exact plan-conditioned production
  pre-projection recurrence and divides by the measured joint latent
  perturbation norm. It fails closed unless every required directional sample
  is present and finite.
- Phase 4 policy stability calls the production `policy_dist` path with `z_H`
  and the exact action mask at depths 2, 4, and 8.
- Phase 4 schema-v4 summaries require the exact four-condition by three-seed
  design, strict complete checkpoint loads, checkpoint/model/config/run/source/
  data identities, one independently authorized producer source, one training
  runtime across all 12 records, and recomputed aggregates.
- The standard-library Phase 4 launcher authenticates and seals the training,
  evaluator, audit, or figure PAR before behavior-bearing imports. Direct role
  execution, a role mismatch, a stale artifact, or hidden source state fails
  closed.
- Checkpoint audit executes every retained replay transition through the exact
  registered `PlanEditEnv` and compares the action mask, successor state,
  reward, terminal flag, and terminal reason. Figure output is staged and
  published only after the final identity checks pass.
- CleanRL scripts route through one owned dispatcher target, and dispatcher
  tests cover every supported backend and fail-closed branch.
- Current-source parity runtime and affected type-check gates are green; exact
  commands and counts are recorded below.

## Registered parity configuration and historical fixture

The registered UPI parity configuration is
`configs/iclr_confirmatory/c2_upi.yaml`. The run matrix remains
`registered_not_authorized`, so this identifies the bound configuration and
does not authorize or start a confirmatory run.

The revision config at
`configs/revision/upi_trm_feasibility_episodic_z_hard_suite_theory_exact.yaml`
is a non-runnable historical reference fixture. Both files state the projection
mode explicitly, use `value_target_clip: null`, and make STOP terminal, but only
the C2 file is registered. These settings align the implemented target and
terminal mechanics with Algorithms 1 and 2. They do not establish the paper's
conditional uniform assumptions.

The matched C2 PPO control uses the same terminal STOP MDP. The former `Bt`
bridge cell is no longer executable because its historical toggle does not
change the synchronized target-network backup; the run matrix records that
fact instead of advertising a no-op contrast.

## Schema-v5 evidence boundary

Checkpoint schema 4 remains part of the historical implementation chronology,
but it is not sufficient for a new theorem-facing run. `fixed_base_exact`
requires checkpoint schema 5, an explicit run ID and seed, clean source blobs
equal to the recorded producer commit, effective-config schema 4 with runtime
identity, initialization and parent-checkpoint lineage, actual optimizer-step
counters, and atomic no-overwrite publication. Restore validates the complete
payload on shadow objects before mutating live state.

The schema-v5 design is recorded in
`reports/CHECKPOINT_SCHEMA_V5_REPORT.md`. Historical execution reports describe
earlier revisions and are not validation of the current validated source state.

## Current validation status

The final 15-target parity runtime command recorded in
`reports/PAPER_PARITY_REPORT.md` passed 366/366. The final affected type gate
built all 30 synchronization, launcher, Phase 4, source-identity, training,
and reporting targets. The producer manifest matches a fresh 80-source
regeneration, and `git diff --check` passes.

The final training PAR SHA-256 is
`cafabc3314730f09f6251144a2d2d06c329d7f93f5431a59f252f71f8b6e708e`.
The launcher and CleanRL dispatcher SHA-256 values are
`faf969e3353b0e78b89042472f37eb900913cd0e09f3196715c793cc7d8c4c54` and
`c2a088a72d3e65a4b0b977ae73e58188ee3b80940802fd2804497d470b978f0b`.
The real launcher help path exited 0 with no private unpack directory left
behind. Wrong-digest, uppercase-digest, malformed-digest, and direct-PAR
confirmatory invocations failed closed with exit codes 2, 2, 2, and 1.

The Phase 4 launcher SHA-256 is
`a83574644a024b337f0d384814b56cdb4600628a0819635136e7c91828002de9`.
The evaluator, audit, and figure PARs passed exact committed-source profile
verification. Their artifact SHA-256 values are
`82b65fc5a9f8714140ddc8769882b07921eb4a17b71319036232c713768def19`,
`e0943bf4a0db909c8f7b783d7dc3d92fe4db2e3eb96f4dfe79f5aee218cc2495`, and
`625d1dccff6dd611cfc7341d7134c7f57fd9825292f02bcb86e0285845ad4057`.
The corresponding source-profile digests are
`a2ddc393a9d1849a41ab194b9e3bab75b4945dcdbbc1c3db94c91b4225dbb06a`,
`24ad45bf18566dc48b6df15e9bc31ea4cca42f3c6056156a0cba57491d9c15c0`, and
`7024d357f9653ca8ab39b8d186b7b045474703c73f2faf073bedd6177efcb4e0`.
Authenticated help paths passed for all four Phase 4 roles. Wrong-digest,
role-mismatch, and direct-PAR cases failed closed.

All 46 tracked shell scripts passed `bash -n`; 625 tracked JSON files and 96
tracked YAML files parsed successfully, and all 251 tracked Python sources
compiled. The retired launcher exited 2 without changing the worktree.

The canonical paper rebuilt successfully at the frozen mathematical source:
38 pages, 544,004 bytes, SHA-256
`de4fde697f7c835ef2827884569a5a44a12cca473758f91b4fd7dea6d298961d`.
All 38 rendered pages were inspected and the final log contained no matched
warning, missing-reference, missing-citation, or fatal-error condition.

The historical repository-wide package diagnostic passed 554 targets and
failed 26 generated type-check targets before the final type cleanup. It had no
runtime, timeout, infra, or build failures. The targeted current-source gates
above do not convert that historical diagnostic into an all-package green
claim.

## Theory boundary

Executable parity covers control flow, frozen-object ownership, target-network
use, exact discrete-action centering arithmetic, and pointwise probability
mixing. It does not certify the invariant domains, sup-norm Bellman or target
residuals, bounded value head, recurrent Lipschitz constants, policy overlap,
signed occupancy-averaged defect bound, or common policy-pair closure required
by the paper.

Empirical regression loss, finite-batch maxima, and local finite-difference
diagnostics establish none of those uniform premises. The implementation also
does not turn the fixed-snapshot paper into a claim about SGD, TD convergence,
BPTT, distillation, learned-model performance, or training dynamics.

## Experiments not run

No experiment was added, requested, rerun, reinterpreted, or used to support
this handoff. Runtime smoke tests establish executable mechanics only. They do
not establish a learned-model result or any conditional theorem premise.

## Confirmatory dataset

The registered hard 4 by 4 Sudoku corpus is materialized at
`data/iclr-confirmatory-sudoku4x4-v1`. It contains 1,024/256/512 unique,
pairwise-disjoint train/validation/test records with 6 to 8 empty cells and one
valid completion. Dataset provenance is recorded in
`reports/CONFIRMATORY_DATASET_REPORT.md`.

This closes the raw data-materialization gate only. No training result follows
from it.
