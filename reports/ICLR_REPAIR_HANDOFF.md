# ICLR repair handoff

Date: 2026-08-13

Implementation branch: `full-implementation`

Implementation source anchor: `980f6ede14717e87ad68ceb32acc111bdd7fca1b`

Implementation parent: `86ec7363103b3d6a6a36fb25094ec1991cefd48e`

Previous behavior anchor: `f86bddb607adcd24eba65fd5869af58f91742a52`

Paper source anchor: UPI-TRM commit
`5253692fea5e77cfde3a130c50351183dc0268e3`

Paper handoff head: `a76eb29f2712042742fea738cdb859d354fafce2`

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
- Phase 4 schema-v3 summaries require the exact four-condition by three-seed
  design, strict complete checkpoint loads, checkpoint/model/config/run/source/
  data identities, and recomputed aggregates. Evaluator, audit, and figure
  PARs bind their behavior-source bytes to a clean explicit Git checkout and
  reject stale or hidden source state before consuming publication evidence.
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
`reports/PAPER_PARITY_REPORT.md` passed 338/338. The final affected type gate
passed all 24 synchronization, launcher, Phase 4, source-identity, training,
and reporting targets. The producer manifest matches a fresh 79-source
regeneration, and `git diff --check` passes.

The final training PAR SHA-256 is
`1a01d06695200a48b160b10d81fa7160750c3499a133b6cfcdda9b110a5ba577`.
The launcher and CleanRL dispatcher SHA-256 values are
`46cb7201226f39718ea83135e397ec6618b9ab344d7ed963a8c26ddf2d4d83ae` and
`54cb744038a121ef5715c0d52cc87e6e9c7c19616a3a836bb3b96e3f3b8cc9ad`.
The real launcher help path exited 0 with no private unpack directory left
behind. Wrong-digest, uppercase-digest, malformed-digest, and direct-PAR
confirmatory invocations failed closed with exit codes 2, 2, 2, and 1.

Cold rebuilt Phase 4 evaluator, audit, and figure PARs passed exact
committed-source profile verification. Their artifact SHA-256 values are
`e03022090542be8773069b7ba5ccf3e630b220b3f1ac943881394c3a81fc7dc0`,
`50ccb5891320e9a1cd4455795a9da0896d54386850653278e233016ec492875f`, and
`6bbedeebe4ce9b5d77581f5f3c5144868baf2b4e68ebe3410a530c3b6df33744`.
The corresponding source-profile digests are
`99cdbf794b79e70e21cab59eeb4e0143042875a8a233f919075e3d3a24d38500`,
`43211cb68189983870d8877424e840ce20364adec2dbda8dfbd5b1102c28c335`, and
`246f9396498572487056f843d9699f2779adaea1440248bde7f8b4f899b61b08`.
The first post-repair inspection detected stale cached runtime bytes; the
verifier rejected them, and all three artifacts were rebuilt from a cold Buck
daemon before the successful checks above.

All 46 tracked shell scripts passed `bash -n`; 625 tracked JSON files and 96
tracked YAML files parsed successfully, and all 246 tracked Python sources
compiled. The retired launcher exited 2 without changing the worktree.

The canonical paper rebuilt successfully at the frozen mathematical source:
38 pages, 544,004 bytes, SHA-256
`2b5a930136b7c81d2f3e8cc59ea7aa27ab837c913cf7cd2d07200b26a940a534`.
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
