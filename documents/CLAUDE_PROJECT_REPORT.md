# Claude Project Report

Last updated: 2026-04-16
Workspace repo: `/home/buiksat/trm_bellman`
Canonical manuscript repo: `/home/buiksat/UPI_TRM/UPI_TRM_NIPS`
Canonical manuscript file: `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/main.tex`

## Purpose

This document is a high-signal orientation note for future Claude/Codex sessions working around the UPI-TRM paper and the `trm_bellman` codebase.

If sources disagree, use this trust order:

1. `main.tex` in the external paper repo.
2. Locked status notes in `documents/CLAUDE.md` and `documents/handoff.md`.
3. Curated paper-facing summaries under `documents/results/paper_ready/`.
4. Planning docs under `documents/UPI_TRM_NIPS/`.
5. Root-level export folders and old logs in `trm_bellman` as provenance only.

The paper repo is canonical for manuscript text. This repo is canonical for code, configs, scripts, raw artifacts, and Markdown mirrors.

## 1. What The Paper Is About

The paper is titled `Policy Improvement under Compute-Truncated Evaluation in Tiny Recursive Models`.

Its core question is:

- What happens to policy-improvement guarantees when the value estimate is produced by truncating the model's own recursive evaluator after only `n` inner refinement steps?

The setting is Tiny Recursive Models (TRMs) trained on checker-feedback tasks such as Sudoku:

- The external state is an instance-plan pair `(x, y)`, where `x` is the puzzle and `y` is the current candidate plan or solution.
- The agent chooses edit actions on `y`.
- A checker provides reward or potential-shaped reward.
- Inside each decision, TRM runs an internal latent refinement loop for `n` steps.
- The value estimate `U_n(s)` depends on that finite internal computation budget.

The main contribution is not "TRM plus generic RL." The paper's contribution is to separate:

- the external MDP over `(instance, plan)` pairs, and
- the internal evaluator computation inside one state evaluation.

That separation lets the paper express evaluator-truncation error using TRM-specific quantities such as `L_z`, `n`, and `R`, then plug that error into a conservative policy-improvement bound.

## 2. Core Formal Objects

Main objects in the paper:

- State: `s = (x, y)`, an instance and current plan.
- Action: one edit to the current plan.
- Transition: apply the edit to get the next plan.
- Reward: checker-derived reward, often with potential-based shaping.
- Inner evaluator update: `z^{k+1} = (Pi_R o f_theta)(z^k, y, x)`.
- Unrolled value: `U_n(s) = V_psi(z^(n)(s), x)`.
- Ideal latent value: `U_*(s) = V_psi(z^*(s), x)` where `z^*` is the latent fixed point.
- CPI update: `pi_new = (1 - alpha) pi + alpha pi_cand`.

Two latent modes matter throughout the paper:

- `episodic-z`: reinitialize latent from the current `(x, y)` every decision step. This is the clean theory-matched regime because the MDP state remains `(x, y)`.
- `persistent-z`: carry latent across edits. This matches the original TRM architecture more closely but requires a slower-drift argument on an augmented state `(x, y, z)`.

The paper repeatedly stresses that increasing evaluator depth `n` is:

- extra internal computation inside one state evaluation,
- not longer environment rollout,
- not tree search,
- not VIN-style planning over external states.

## 3. Theory Chain

The main theorem chain in `main.tex` is:

1. Assumption `Forward-invariant contraction`.
2. Banach-style fixed-point result for the latent evaluator.
3. Value-error decomposition.
4. Advantage-error bridge.
5. CPI bound with centered evaluation error.
6. Projection-aware specialization.

The important equations and meanings are:

- Under the contraction assumption on `z -> (Pi_R o f_theta)(z, y, x)`, the latent fixed point `z^*(x, y)` exists and the finite-unrolling bias decays geometrically like `L_z^n`.
- Proposition `Value-error decomposition` gives:
  `||U_n - V^pi|| <= eps_res^* / (1 - gamma^K) + L_V * (L_z^n / (1 - L_z)) * C_z`.
- The first term is an architectural Bellman-residual term.
- The second term is the evaluator-truncation term.
- Corollary `epsA_dials` lifts this into an advantage-error term.
- Theorem `CPI with centered evaluation error` inserts that advantage error into a CPI lower bound with:
  - a linear `alpha * epsilon_A` penalty, and
  - the standard quadratic CPI term in `alpha^2`.
- Corollary `End-to-end parameter dependence` makes the roles of `L_z`, `n`, `K`, and `alpha` explicit.

Interpretation:

- Smaller `L_z` helps.
- Larger inner depth `n` helps through geometric decay `L_z^n`.
- Larger `K` tightens the residual coefficient.
- Smaller conservative-step size `alpha` is safer.

Projection `R` enters in two different ways in the projection-active regime:

- It can shrink the effective contraction modulus through a geometric factor `R / rho_R`.
- It also enters the architectural residual bound through a baseline-relative term involving `L_V(R) * R`.

This is why the paper now treats projection as more than a numerical trick. In the projection-active regime it changes both the effective evaluator dynamics and the bound structure.

## 4. Scope Of The Theory

The paper is explicit about what it does not prove:

- It is a fixed-parameter-snapshot analysis, not an end-to-end SGD convergence theorem.
- It does not prove TD, BPTT, or optimizer convergence.
- It is not offered as a calibrated predictor of success rates.
- The clean theory target is episodic-`z`, not persistent-`z`.
- The persistent-`z` result in the appendix is conditional and stated on the augmented state `(x, y, z)`, not directly on plan-space states `(x, y)`.

For persistent-`z`, the appendix introduces a slow-drift assumption:

- the fixed point `z^*(x, y)` must move smoothly as the plan changes,
- plan edits must have bounded size in a learned plan embedding,
- the latent must track the moving fixed point closely enough.

The paper openly says this is a design objective for discrete tasks like Sudoku, not a verified property of the benchmark.

## 5. Main Paper-Facing Empirical Story

The current manuscript centers three empirical claims.

### A. Hard no-mask 4x4 is the capability anchor

Paper-facing anchor:

- Task: hard `4x4` Sudoku, `6-8` empties, no-mask protocol.
- Budget: `20k` training steps, horizon `T = 16`.
- UPI-TRM: `57.4% +/- 12.2%` success over `10` seeds.
- Architecture-matched baselines with the same TRM backbone:
  - `TRM+PPO`: `0.0%` over `3` seeds.
  - `TRM+A2C`: `0.0%` over `3` seeds.
  - `TRM+DQN`: `0.0%` over `3` seeds.
- External trusted baselines from SB3:
  - PPO: `0.0%` over `10` seeds.
  - A2C: `0.0%` over `10` seeds.
  - DQN: `0.0%` over `10` seeds.
  - DQN (`n=5`): `0.0%` over `10` seeds.

Meaning:

- The paper now has both architecture-matched and independent-codebase baseline coverage.
- The hard result is genuinely nontrivial within this domain.
- The result uses `persistent-z`, so the paper treats it as empirical evidence, not direct validation of the clean episodic-`z` theorem.

### B. Projection is the dominant stabilizer on hard 4x4

Controlled no-mask hard `4x4` `2x2` factorial:

- `no contraction, proj. off`: `35.0 +/- 8.9`
- `no contraction, R=10`: `48.2 +/- 7.1`
- `contraction, proj. off`: `37.4 +/- 7.8`
- `contraction, R=10`: `50.2 +/- 10.5`

Interpretation locked in both the paper and status docs:

- Projection is the dominant main effect: about `+13` percentage points on average.
- Contraction adds only about `+2.2` percentage points on average.
- The contraction effect is statistically inconclusive at `10` seeds on this hard suite.
- There is no credible variance-reduction claim for contraction under active projection.

This is one of the most important narrative corrections relative to older drafts. The paper should not sell contraction as the main source of hard-suite performance.

### C. Contraction helps most under evaluation-time depth mismatch

This is the closest paper experiment to the theory.

Protocol:

- Easier `4x4` Sudoku suite with `1-4` empties.
- Train at `n_train = 2`.
- Evaluate at deeper unrolls `n_eval in {4, 8, 16}`.
- Use episodic-`z`.

Main B0 result at fixed `2 -> 8` mismatch:

- `Delta_V`: `0.166 +/- 0.287 -> 0.064 +/- 0.060`
- `Delta_z`: `4.26 -> 1.67`
- Argmax agreement: `90.2% -> 95.5%`
- `Delta_pi` remains near zero and is not the main discriminator.

Important isolation result:

- Even with projection disabled (`proj. off`, `R=0`) on B0, contraction still improves value stability at fixed mismatch by about `2.1x` (`2.275 -> 1.097`).

B1 successor-state results are directionally consistent but noisier and remain appendix-only.

## 6. Supporting Results And Non-Headline Results

These matter for context, but they are not the headline story:

- Toy `1-4` empties feasibility:
  - UPI-TRM reaches roughly `90-93%`.
  - Masked random is already `52%`.
  - Therefore the toy suite is only a sanity check, not persuasive capability evidence.
- `9x9` Sudoku:
  - current paper only has a `3`-seed supporting sweep at about `4.0%` success.
  - This is explicitly supporting evidence, not a central claim.
- Exp2 contraction-target sweep:
  - negative result for the "spectral target as a stable dial" story under active projection.
  - projection dominates and post-projection `hat L_z` stays in a narrow band.
- Finite-`R` sweep:
  - useful bounded-contact diagnostic for both truncation and residual channels.
  - still does not identify end-to-end `R` dependence of `eps_res^*`.
- Exp5 tradeoff curve:
  - exploratory only.
  - no strong stability-expressivity tradeoff is supported in the final paper framing.

## 7. What Not To Overclaim

Future Claude sessions should not make these mistakes:

- Do not claim the paper proves end-to-end optimizer convergence.
- Do not claim persistent-`z` hard-suite success validates the main theorem directly.
- Do not claim contraction is a generic performance booster.
- Do not claim contraction reduces hard-suite variance under projection.
- Do not claim masked hard `4x4` is the main capability anchor.
- Do not mix historical pre-`8884572` hard-suite logs with current paper-facing runs.
- Do not treat `9x9` as settled evidence.

Additional repo-specific caution:

- Some intermediate Markdown claim files are stronger than the current paper text.
- Example: some `exp4_projection_free_dial*` notes argue for a positive monotonic dial story, while the current manuscript only relies on weaker projection-free feasibility/range contact and does not make that a central headline claim.
- If a result doc conflicts with `main.tex`, the manuscript wins.

Another planning/status caveat:

- `documents/UPI_TRM_NIPS/PHASE0_EXPERIMENT_QUEUE.md` is still useful for frozen seed sets and protocol definitions.
- Its launch-status tables reflect an earlier planning state in places.
- In particular, the paper-facing trusted external baseline package is now present in `documents/results/paper_ready/hard4x4_trusted_baselines_20k/SUMMARY.md` and in `main.tex`, even though parts of the planning queue still describe trusted external baselines as pending or optional.

## 8. Codebase Map

The clean implementation path in `trm_bellman` is:

- `upi_trm_train.py`
  - main CLI.
  - loads YAML configs and CLI overrides.
  - selects between UPI-TRM and baseline trainers.
  - builds datasets, checker, environment, trainer, checkpoint I/O.

- `rl/config.py`
  - central hyperparameter and theory-alignment config.
  - includes important toggles such as:
    - `episodic_latent`
    - `latent_ball_radius`
    - `enable_contraction`
    - `disable_value_head_norm`
    - `exact_baseline_summation`
    - `theory_exact_mixture`
    - `disable_constraint_masking`

- `rl/envs/plan_edit_env.py`
  - the plan-edit MDP implementation.
  - state is `(x, y)`.
  - actions are edits plus STOP and optional UNDO.
  - reward is checker-based with optional potential shaping.
  - contains Sudoku-specific action masking and constraint tracking.

- `models/recursive_reasoning/trm.py`
  - the TRM model with RL-specific additions.
  - exposes:
    - `init_latent`
    - `update_latent`
    - `unroll_latent`
    - `continue_latent`
    - `used_value`
    - `policy_dist`
  - applies latent-ball projection and optional contraction enforcement.
  - flattens latent/context representations for value and policy heads so Sudoku position information is preserved.

- `models/value_head.py`
  - bounded scalar value head using a scaled `tanh`.

- `rl/upi_trm_trainer.py`
  - main UPI-TRM trainer.
  - handles:
    - replay collection
    - `K`-step bootstrapped value targets
    - policy updates
    - CPI-style mixture behavior
    - theory/debug metrics
    - optional imitation pretraining helpers

- `rl/algos/ppo.py`, `rl/algos/a2c.py`, `rl/algos/dqn.py`
  - in-repo architecture-matched baselines.

- `rl/training_setup.py`
  - dataset loading, offline dataset creation, checker resolution.

- `evaluators/rl_plan_evaluator.py` and `rl/evaluator.py`
  - evaluation helpers.

- `scripts/eval/unroll_sensitivity.py`
  - canonical exp1-style depth-mismatch evaluation entrypoint.

- `external_baselines/sudoku4x4_env.py`
  - trusted external baseline environment wrapper path.

- `tests/`
  - broad unit/smoke coverage for configs, latent unroll, K-step targets, theory-exact pieces, environment behavior, masking, logging, and baselines.

## 9. Important Implementation Caveats

There are several code-level caveats that matter for interpreting results.

### A. Contraction and value-head normalization are separate

In code and in the final paper story:

- `enable_contraction: true` controls `z -> z` contraction enforcement.
- `disable_value_head_norm: true` disables value-head spectral normalization.

This distinction matters because:

- contraction on the latent dynamics is part of the theory story,
- but value-head normalization was found to be a confounding source of instability/collapse in some diagnostics.

For stability studies, the paper explicitly disables value-head spectral normalization.

### B. Projection has two roles

Projection is not only a theorem device.

In this repo it affects:

- the latent evaluator dynamics through active clipping,
- the practical stability regime,
- the paper's projection-active theory specialization.

This is why "projection on vs projection off" is a serious protocol split, not a tiny implementation detail.

### C. Masking protocol changed historically

Commit `8884572` introduced constraint-aware Sudoku masking across Sudoku sizes.

Consequences:

- historical hard-suite runs before that change are not protocol-compatible with current ones,
- masked hard `4x4` is now ceiling-saturated and demoted to a control,
- fresh no-mask reruns are the paper-facing capability anchor.

### D. Some dataset names are legacy and misleading

According to the phase-0 queue:

- `data/sudoku-4x4-trivial` is the true `1-4` empties suite.
- `data/sudoku-4x4-easy_6to8empties` is the hard `6-8` empties paper suite.
- `data/sudoku-4x4-ultra-easy` is also a `6-8` empties split kept as a legacy compatibility path, despite the name.

## 10. Artifact And Directory Map

Most important directories:

- `configs/`
  - experiment configs grouped by purpose.
  - most important paper-related subdirs:
    - `ablations/`
    - `baselines/`
    - `pilots/`
    - `sudoku9x9/`
    - `exp2_contraction_sweep/`
    - `exp3_projection_ablation/`
    - `phase4_2x2_norm_ablation/`
    - `table3_hard_controlled/`

- `data/`
  - local Sudoku datasets.

- `checkpoints/`
  - trained model checkpoints.
  - includes current no-mask hard-suite and controlled-`2x2` checkpoint families.

- `results/`
  - raw logs, plot data, tables, validation outputs, and paper-facing result bundles.
  - important subpaths:
    - `results/table3_hard_6to8/`
    - `results/table3_hard_6to8_controlled_nomask/`
    - `results/validation/`
    - `results/neurips2026/`
    - `results/paper_ready/`

- `artifacts/`
  - fixed evaluation batches and related experiment assets.

- `documents/`
  - organized Markdown mirror and orientation docs.
  - this is the right place for repo-local narrative/context docs.

## 11. Trusted Vs Stale Artifact Zones

Inside `trm_bellman`, the most useful curated Markdown lives under:

- `documents/results/paper_ready/hard4x4_trusted_baselines_20k/`
- `documents/results/paper_ready/exp1/`
- `documents/results/paper_ready/exp2_final/`
- `documents/results/paper_ready/exp5_tradeoff_curve_v2/`

Use the newer/reconciled variants when both old and new exist:

- prefer `hard4x4_trusted_baselines_20k` over `hard4x4_trusted_baselines`
- prefer `exp2_final` over `exp2`
- prefer `exp5_tradeoff_curve_v2` over `exp5_tradeoff_curve`

Be careful with these zones:

- root-level export dirs such as
  - `exp3_projection_ablation_export/`
  - `exp4_final_export/`
  - `exp5_tradeoff_curve_v2_export/`
  - `paper_ready_update/`
  - `phase4_export/`
  - `table3_hard_controlled_2x2_export/`
- they are useful provenance, but not automatically canonical.

Also note:

- the repo contains a top-level `UPI_TRM_NIPS/` directory stub, but the canonical manuscript repo is the external path `/home/buiksat/UPI_TRM/UPI_TRM_NIPS`.
- the Markdown mirror for paper-planning docs inside this repo lives under `documents/UPI_TRM_NIPS/`.

## 12. Current Locked Status

As of the 2026-04 locked notes:

- Main capability anchor: no-mask hard `4x4`, not masked hard `4x4`.
- `M1` no-mask hard `4x4` is complete:
  - UPI: `0.574` mean, `0.122` std, seeds `0..9`
  - in-house A2C: `0.000` over seeds `0..3`
- Controlled no-mask hard `4x4` `2x2` is complete:
  - `nc_r0 = 0.350`
  - `nc_r10 = 0.482`
  - `c_r0 = 0.374`
  - `c_r10 = 0.502`
- Locked interpretation:
  - projection is the primary stabilizer on hard `4x4`
  - contraction adds only a small average lift
  - contraction does not reduce variance under projection
- Another-domain expansion was dropped before the deadline.
- Sudoku remains the main focus.

## 13. Current Empirical Blocker

The next blocker is still the `exp1_v4` package:

- `PF2`: missing fixed eval batches for `B0` and `B1`
- `PF3`: missing historical checkpoints for the old `exp1_v4` runs

Implication:

- if those artifacts cannot be restored, the depth-mismatch package needs to be fully refrozen and rerun under a replacement protocol before using it as a fresh paper-facing artifact package.

This blocker matters because:

- the hard-suite capability story is already locked,
- the controlled hard-suite `2x2` is already locked,
- the remaining place where contraction earns a clean main-text role is the depth-mismatch package.

## 14. Recommended Working Rules For Future Claude Sessions

When working on this project:

1. For manuscript edits, work in `/home/buiksat/UPI_TRM/UPI_TRM_NIPS`, not in `trm_bellman`.
2. For code, configs, scripts, raw results, and diagnostic artifacts, work in `/home/buiksat/trm_bellman`.
3. When summarizing the paper, cite the no-mask hard `4x4` table, the controlled hard `2x2`, and the episodic depth-mismatch study in that order.
4. When summarizing contraction, describe it as a depth-mismatch robustness factor, not as the main hard-suite performance driver.
5. When summarizing projection, treat it as both:
   - a practical stabilizer, and
   - a quantity that enters the projection-active theory.
6. If a Markdown claim doc conflicts with `main.tex`, trust `main.tex`.
7. If a status/planning doc conflicts with a later paper-ready summary, use the later paper-ready summary for current empirical status and keep the planning doc only for protocol details.
8. Be explicit about protocol splits:
   - masked vs no-mask
   - episodic-`z` vs persistent-`z`
   - projection on vs projection off
   - current-code vs historical pre-`8884572`
   - hard `6-8` empties vs toy `1-4` empties

## 15. Short Version

The shortest correct read of the whole project is:

- UPI-TRM treats TRM plan editing as RL on a plan-space MDP while preserving an internal recursive evaluator.
- The paper's theory says evaluator truncation error is structured, not generic approximation noise.
- The paper's strongest empirical result is the no-mask hard `4x4` capability gap.
- The controlled hard-suite `2x2` says projection is the dominant stabilizer.
- Contraction's clearest evidence is depth-mismatch robustness on the easier episodic-`z` protocol.
- The manuscript is external; this repo is the code-and-artifact side.
