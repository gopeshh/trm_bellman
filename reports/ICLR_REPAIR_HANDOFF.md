# ICLR repair handoff

Date: 2026-08-12

Implementation branch: `full-implementation`

Implementation source anchor: `f86bddb607adcd24eba65fd5869af58f91742a52`

Implementation parent: `e4edcb2107c0f3e7ac0e691bd9dc828c5c6f38a0`

Paper source anchor: UPI-TRM commit
`5253692fea5e77cfde3a130c50351183dc0268e3`

Paper handoff head: `071037929ed0a16eaf1d8b2f1169786a866eb8a5`

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
- Phase 4 schema-v2 summaries omit unmeasured success, loss, and NaN-history
  fields, require the exact four-condition by three-seed design, and recompute
  aggregates from all 12 runs before audit or figure output.
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

The final 14-target parity runtime command recorded in
`reports/PAPER_PARITY_REPORT.md` passed 308/308. The final six-target type gate
for synchronization-touched model, evaluator, utility, and checkpoint code
passed 6/6. The launcher library and binary type targets passed 2/2. Six
repair-specific Phase 4 and CleanRL type targets passed 6/6. The producer
manifest matches a fresh 79-source regeneration, and `git diff --check` passes.

The final training PAR SHA-256 is
`bd10dda2afc422bd07751a02e1ed39ef164adf0d6a4241220db2b94f95e5cb67`.
The real launcher help path exited 0 with no private unpack directory left
behind. Wrong-digest, uppercase-digest, and direct-PAR confirmatory invocations
failed closed with exit codes 2, 2, and 1.

All 46 tracked shell scripts passed `bash -n`; 625 tracked JSON files and 96
tracked YAML files parsed successfully. The retired launcher exited 2 without
changing the worktree.

The historical repository-wide package diagnostic passed 554 targets and
failed 26 generated type-check targets before the final type cleanup. It had no
runtime, timeout, infra, or build failures. Six synchronization-touched type
targets were fixed and cleared afterward. Other pre-existing generated
type-check debt outside the six cleared targets remains, so there is no green
all-package claim.

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
