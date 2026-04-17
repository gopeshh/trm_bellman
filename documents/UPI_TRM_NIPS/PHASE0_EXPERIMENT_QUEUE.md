# Phase 0 Experiment Queue (T0.1)

This document locks the empirical claims, seed sets, run queue, and launch templates for the NeurIPS 2026 resubmission. It is the executable companion to `NIPS_PLAN.md`.

If a claim or run is not listed here, it is not paper-critical.

## Locked Claims

### 1. Main empirical claim: hard 4x4 best-vs-best

On hard 4x4 Sudoku (`6-8` empties), the current constraint-aware masking protocol is no longer a viable capability anchor on its own. The paper-facing capability anchor should now be the no-mask hard-4x4 protocol, with masked hard-4x4 retained as an ablation/control.

Current anchor:
- Historical internal 3-seed results in `/home/buiksat/trm_bellman/results/table3_hard_6to8/run_all.log` show `persistent_nc` at `0.567` mean success over seeds `42,123,456`, but these runs predate commit `8884572` and used the old fill-only Sudoku action mask. Treat them as provenance only.
- Fresh current-code H1 reruns on `S_hard10 = 0,1,2,3,4,5,6,7,8,9` under the post-`8884572` constraint-aware mask finish at `0.996` mean success at step `20000` with per-seed finals `0.980 / 1.000 / 1.000 / 0.980 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000`.
- A current-code in-house A2C smoke gate on the same masked protocol reaches `1.000` success at step `20000` on seed `0`.
- Full `M1` current-code no-mask sweep is complete: `persistent_nc` reaches `0.574` mean success (`std 0.122`) over `S_hard10`, with per-seed finals `0.560 / 0.500 / 0.680 / 0.500 / 0.640 / 0.700 / 0.740 / 0.420 / 0.380 / 0.620`.
- Current-code in-house no-mask A2C seeds `0-3` all finish at `0.000`, which is enough to lock the in-house no-mask capability gap. Backfilling `4-9` is optional symmetry/optics work, not a paper blocker.
- Interpretation: masked hard-4x4 is ceiling-saturated and should be presented as the "environment gives away constraints" control. No-mask hard-4x4 is now the main capability anchor.

### 2. Stability claim: contraction reduces depth-mismatch sensitivity

The paper may claim that contraction is a stability dial under unroll-depth mismatch if and only if the claim is scoped to the existing `exp1_v4` protocol and expanded to 10 seeds.

Current anchor:
- Existing 3-seed `exp1_v4` package already supports the direction of the claim.
- From `/home/buiksat/trm_bellman/documents/results/paper_ready/exp1/CLAIMS.md` and the associated aggregation artifacts:
  - B0: contraction reduces `Δ_V` by `4.1x` at `8x` depth.
  - B0, `R=0`: contraction improves value stability materially, but the exact multiplier is still under audit and must be recomputed from the authoritative aggregate before it appears in paper text.

### 3. Controlled interpretation claim: projection vs contraction on hard 4x4

The paper should claim that projection is the dominant performance stabilizer on hard 4x4, while contraction is mainly a robustness/stability dial. It should not claim contraction is a generic return booster.

Current anchor:
- Fresh current-code no-mask controlled 2x2 over `S_hard10` in `/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask/` is complete:
  - `nc_r0 = 0.350` (`std 0.089`)
  - `nc_r10 = 0.482` (`std 0.071`)
  - `c_r0 = 0.374` (`std 0.078`)
  - `c_r10 = 0.502` (`std 0.105`)
- Paired contrasts over the 10 seeds:
  - projection effect: `+0.132` without contraction, `+0.128` with contraction
  - contraction effect: `+0.024` at `R=0`, `+0.020` at `R=10`
  - interaction: `-0.004` (effectively zero)
- Interpretation: projection is the dominant performance stabilizer on hard 4x4. Contraction adds only a small average lift on this task and does not reduce variance under projection (`std 0.105` for `c_r10` vs `0.071` for `nc_r10`).
- Historical 3-seed controlled 2x2 logs in `/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled/` remain provenance only and should not be mixed into the current aggregate.

### 4. 9x9 is supporting only

9x9 stays supporting evidence unless fresh seeds under the current `50k` constraint-aware configs land in time.

Current anchor:
- Historical `25k` runs in `/home/buiksat/trm_bellman/results/sudoku9x9_real/` include five attempted seeds (`0-4`): three completed (`1,2,3`) and two truncated (`0,4`) near steps `24600-24700`.
- Over the three completed seeds, UPI mean score improves from `26.64` initial to about `27.64`.
- Those runs predate the current `50k` constraint-aware configs and should not be treated as final paper evidence.

## Explicit Non-Claims

- Do not use the internal `0%` PPO/A2C/DQN reruns as the sole baseline-credibility story.
- Do not claim masked hard-4x4 shows a meaningful UPI-vs-baseline capability gap under constraint-aware masking.
- Do not claim contraction is a generic performance knob.
- Do not claim contraction reduces variance on hard 4x4; the current no-mask 10-seed table does not support that.
- Do not headline the `1-4` empties toy suite.
- Do not claim quantitative theory prediction via `L_V` unless new evidence lands.
- Do not let 9x9 carry the main empirical narrative.

## Frozen Seed Sets

| Name | Seed set | Reason |
|------|----------|--------|
| `S_hard10` | `0,1,2,3,4,5,6,7,8,9` | Fresh current-code hard-4x4 seed set after constraint-aware masking landed. |
| `S_hard_hist3` | `42,123,456` | Historical pre-constraint-masking provenance only. Do not mix into current paper aggregates. |
| `S_exp110` | `41,42,43,44,45,46,47,48,49,50` | Extends the existing `exp1_v4` seeds `41,42,43` to 10. |
| `S_9x9_10` | `0,1,2,3,4,5,6,7,8,9` | Extends the existing 9x9 seed convention. |

## Frozen Hyperparameters

### Hard 4x4 main comparison and controlled 2x2

- Dataset target: `buiksat_trm/data/sudoku-4x4-easy_6to8empties` (restored on this machine via the validated recipe in Preflight `PF1`)
- Checker: feasibility only
- `w_v = 2.0`, `w_z = 5.0`
- `gamma = 0.99`
- `inner_unroll_n = 2`
- `max_edits = 16`
- `train_steps = 20000` via CLI override
- `eval_num_episodes = 50`
- Reported metric: final greedy success rate at step `20000`
- Paper-facing no-mask hard-4x4 capability and controlled 2x2 reruns use `disable_constraint_masking = true`; masked hard-4x4 is retained only as a control/ablation.
- Main best-4x4 UPI variant:
  - `episodic_latent = false`
  - `enable_contraction = false`
- Controlled 2x2 hard package:
  - `episodic_latent = true`
  - `disable_value_head_norm = true`
  - only `enable_contraction` and `latent_ball_radius` vary across the four cells
  - current paper-facing rerun uses `buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml`

### Exp1_v4 unroll-sensitivity package

- Training dataset: `buiksat_trm/data/sudoku-4x4-trivial`
- Train configs:
  - `buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml`
  - `buiksat_trm/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml`
- `train_steps = 5000`
- `disable_value_head_norm = true`
- `inner_unroll_n = 2`
- Fixed eval batches if restored:
  - `/home/buiksat/trm_bellman/artifacts/eval_batches/v3/b0.pt`
  - `/home/buiksat/trm_bellman/artifacts/eval_batches/v3/b1.pt`
- Depth multipliers: `1,2,4,8`
- Radius sweep: `10,100,0`

### 9x9 supporting package

- Use only the current constraint-aware configs:
  - `buiksat_trm/configs/sudoku9x9/upi_trm_9x9.yaml`
  - `buiksat_trm/configs/sudoku9x9/ppo_9x9.yaml`
  - `buiksat_trm/configs/sudoku9x9/a2c_9x9.yaml`
  - `buiksat_trm/configs/sudoku9x9/dqn_9x9.yaml`
- `train_steps = 50000`
- These runs are optional until the hard-4x4 baseline story is settled.

## Common Evaluation Contract

- Hard 4x4 main table and controlled 2x2 use the trainer’s built-in greedy eval and read the final `eval_success_rate` at step `20000`.
- Exp1_v4 reuses the fixed `B0`/`B1` batches and reports aggregated `Δ_V`, `Δ_π`, `Δ_z`, argmax agreement, saturation, and `hat_Lz`.
- If the original `b0.pt` / `b1.pt` cannot be restored byte-for-byte, treat `S_exp110` as a fresh protocol and rerun all 10 seeds under the replacement batches before using the aggregate in paper text.
- 9x9 reports final greedy `eval_mean_score` and success rate at step `50000`.

## Preflight Status

These are the artifact gates that determine what can launch. `PF1` is now cleared on this machine; `PF2`, `PF3`, and `T0.2` remain the real blockers. There is also a protocol split between historical hard-4x4 logs and current-code runs after `8884572`.

### PF1. Hard-4x4 datasets were missing; cleared on this machine via validated regeneration

- Restored on disk:
  - `/home/buiksat/trm_bellman/data/sudoku-4x4-trivial`
  - `/home/buiksat/trm_bellman/data/sudoku-4x4-ultra-easy`
  - `/home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties`
- Frozen regeneration entrypoint:
  - `/home/buiksat/trm_bellman/scripts/restore_4x4_datasets.sh`
- Validation status:
  - `buck2 run //buiksat_trm:inspect_4x4_dataset` confirms `sudoku-4x4-trivial` is the true `1-4` empties split.
  - The same inspection confirms both `sudoku-4x4-ultra-easy` and `sudoku-4x4-easy_6to8empties` are `6-8` empties splits, with `ultra-easy` kept only as a legacy compatibility path.
  - Validation is distribution-level, not archival content-hash recovery: the restore path verifies directory layout, split sizes, and empties-range semantics, but does not prove byte-identical puzzle identity against the historical corpora.
  - Interpretation consequence: use the regenerated datasets as the frozen launch point for new hard-4x4 runs on this machine, but do not describe the resulting `10`-seed package as trained on a byte-identical recovered corpus unless historical hashes or artifacts are later recovered.
- Minimal restore command on a fresh checkout:

```bash
cd /home/buiksat/trm_bellman
bash scripts/restore_4x4_datasets.sh
```

- Equivalent Buck recipe baked into that helper:
  - `//buiksat_trm:build_4x4_trivial` -> `buiksat_trm/data/sudoku-4x4-trivial`
  - `//buiksat_trm:build_4x4_sudoku` -> `buiksat_trm/data/sudoku-4x4-ultra-easy`
  - `//buiksat_trm:build_4x4_sudoku` -> `buiksat_trm/data/sudoku-4x4-easy_6to8empties`
- Launch consequence:
  - `H1` and `C1-C4` are no longer blocked on missing datasets on this machine.
  - On any new machine or fresh fbcode checkout, rerun the helper before starting hard-4x4 jobs.

### PF1b. Historical hard-4x4 logs are not protocol-compatible with current code

- Commit `8884572` added constraint-aware Sudoku action masking across Sudoku sizes, not just 9x9.
- Historical hard-4x4 and controlled 2x2 logs predate that change and use the old fill-only mask:
  - `/home/buiksat/trm_bellman/results/table3_hard_6to8/persistent_nc_s42.log` shows `32` valid actions out of `97`.
- Fresh current-code H1 reruns use the new mask:
  - `/home/buiksat/trm_bellman/results/table3_hard_6to8/persistent_nc_s0.log` shows `12` valid actions out of `97`.
  - Seeds `0,1,2,3` finish at `0.980 / 1.000 / 1.000 / 0.980` at step `20000`.
- Launch consequence:
  - Do not top up `H1` or `C1-C4` by mixing new runs with historical `42/123/456`.
  - Treat the historical hard-4x4 packages as provenance only.
  - Use fresh current-code seed sweeps for all paper-facing hard-4x4 and controlled-2x2 aggregates.

### PF2. `exp1_v4` fixed eval batches are missing

- Missing on disk:
  - `/home/buiksat/trm_bellman/artifacts/eval_batches/v3/b0.pt`
  - `/home/buiksat/trm_bellman/artifacts/eval_batches/v3/b1.pt`
- `b0_metadata.json` shows that `B0` mixed states from `sudoku-4x4-easy_6to8empties`, `sudoku-4x4-trivial`, and `sudoku-4x4-ultra-easy`.
- `b1_metadata.json` shows that `B1` was derived from seed-42 checkpoints, so regeneration is not independent of the missing model artifacts.
- Required action before `I3`: restore the original `.pt` files. If that is impossible, freeze replacement batches and rerun the full `S_exp110` package end-to-end.

### PF3. Historical `exp1_v4` checkpoints are missing

- Missing on disk:
  - `/home/buiksat/trm_bellman/checkpoints/exp1_v4/`
  - `/home/buiksat/trm_bellman/checkpoints/model_a_nc_no_vhead_norm/seed42/model_step_5000.pt`
  - `/home/buiksat/trm_bellman/checkpoints/model_b/model_step_5000.pt`
- Existing results under `/home/buiksat/trm_bellman/results/validation/exp1_v4/seed41-43/` prove those evals ran historically, but they do not make the checkpoints reusable today.
- Required action before `I1`, `I2`, or `I3`: either recover the historical checkpoints for `41,42,43`, or promote `41-50` to a full retrain + re-eval queue under a fresh frozen protocol.

### PF4. New NeurIPS output roots do not exist yet

Create them before the first trusted-baseline or 9x9 launch:

```bash
mkdir -p \
  /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4/ppo \
  /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4/a2c \
  /home/buiksat/trm_bellman/results/neurips2026/sudoku9x9
```

## Required Run Queue

| ID | Supports | Current state | New work now | Config / runner | Dataset | Seeds to run now | Steps | Expected wall-clock | Output location | Owner |
|----|----------|---------------|--------------|-----------------|---------|------------------|-------|---------------------|-----------------|-------|
| `H1` | Masked hard-4x4 protocol sanity check | `10/10` current-code UPI seeds complete at `0.996` mean; single-seed in-house A2C gate reaches `1.000` on the same mask | Archive as completed. Do not spend more masked hard-4x4 UPI-only compute; use `M1` or `O1` if a capability anchor is still needed | `buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml` via `//buiksat_trm:upi_trm_train` with `--train-steps 20000` | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` | none | `20000` | Observed `~2.5 h/seed` on this machine | `/home/buiksat/trm_bellman/results/table3_hard_6to8/persistent_nc_s{seed}.log` | Done |
| `M1` | Capability claim under hard 4x4 without environment-provided constraint information | `UPI 10/10` no-mask seeds complete at `0.574` mean (`std 0.122`); in-house `A2C 4/4` no-mask seeds complete at `0.000` mean | Lock this as the main hard-4x4 capability table. Optional only: backfill in-house `A2C` seeds `4-9` for matched seed counts / reviewer optics | `buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction_no_mask.yaml` and `buiksat_trm/configs/baselines/a2c_trm_feasibility_no_mask.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` | none (optional `A2C 4-9`) | `20000` | Observed `~2.5 h/seed` for UPI and `~3.25 h/seed` for A2C on this machine | `/home/buiksat/trm_bellman/results/table3_hard_6to8/m1_{persistent_nc,a2c}_nomask_s{seed}.log` | Done |
| `H2` | Trusted external baseline coverage for hard 4x4 | `0/10` trusted external PPO seeds; blocked on `T0.2`; `M1` already confirmed a stable no-mask hard-4x4 capability gap worth defending if reviewer-trust coverage is needed | Pursue only if you want stronger reviewer-trust coverage beyond the locked in-house no-mask package, or if you need a trusted external comparison before deciding whether 9x9 is worth promoting | `CleanRL` or `SB3` recurrent PPO wrapper, to be created in `T0.2`; target parity with internal PPO budget and reward | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` or a harder successor anchor | none until `T0.2` exists | `20000` | Provisional `~10-16 h/seed`; reserve this compute only if external-baseline trust becomes the next priority | `/home/buiksat/trm_bellman/results/neurips2026/external_hard4x4/ppo/seed{seed}/` | Blocked on `T0.2` |
| `H3` | Trusted external baseline coverage for hard 4x4 | `0/10` trusted external A2C seeds; blocked on `T0.2`; `M1` already confirmed a stable no-mask hard-4x4 capability gap worth defending if reviewer-trust coverage is needed | Pursue only if you want stronger reviewer-trust coverage beyond the locked in-house no-mask package, or if you need a trusted external comparison before deciding whether 9x9 is worth promoting | `CleanRL` or `SB3` recurrent A2C wrapper, to be created in `T0.2`; target parity with internal A2C budget and reward | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` or a harder successor anchor | none until `T0.2` exists | `20000` | Provisional `~3-6 h/seed`; reserve this compute only if external-baseline trust becomes the next priority | `/home/buiksat/trm_bellman/results/neurips2026/external_hard4x4/a2c/seed{seed}/` | Blocked on `T0.2` |
| `C1` | Controlled 2x2 hard claim | `10/10` current-code no-mask seeds complete at `0.350` mean (`std 0.089`) | Locked as part of the main-text controlled 2x2 table | `buiksat_trm/configs/table3_hard_controlled/episodic_nc_r0_hard.yaml` + `buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` | none | `20000` | Observed `~4-6 h/seed` | `/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask/nc_r0_s{seed}.log` | Done |
| `C2` | Controlled 2x2 hard claim | `10/10` current-code no-mask seeds complete at `0.482` mean (`std 0.071`) | Locked as part of the main-text controlled 2x2 table | `buiksat_trm/configs/table3_hard_controlled/episodic_nc_r10_hard.yaml` + `buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` | none | `20000` | Observed `~4-6 h/seed` | `/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask/nc_r10_s{seed}.log` | Done |
| `C3` | Controlled 2x2 hard claim | `10/10` current-code no-mask seeds complete at `0.374` mean (`std 0.078`) | Locked as part of the main-text controlled 2x2 table | `buiksat_trm/configs/table3_hard_controlled/episodic_c_r0_hard.yaml` + `buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` | none | `20000` | Observed `~4-6 h/seed` | `/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask/c_r0_s{seed}.log` | Done |
| `C4` | Controlled 2x2 hard claim | `10/10` current-code no-mask seeds complete at `0.502` mean (`std 0.105`) | Locked as part of the main-text controlled 2x2 table | `buiksat_trm/configs/table3_hard_controlled/episodic_c_r10_hard.yaml` + `buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-easy_6to8empties` | none | `20000` | Observed `~4-6 h/seed` | `/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask/c_r10_s{seed}.log` | Done |
| `I1` | Depth-mismatch stability claim | Seeds `41-43` have eval outputs; Model A' checkpoints are missing | Recover the historical Model A' checkpoints, or retrain the full `S_exp110` set under a fresh frozen protocol | `buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-trivial` | `44-50` if artifacts restore cleanly; otherwise `41-50` | `5000` | Budget `~1-2 h/model/seed`; low critical-path cost once the protocol is frozen | `/home/buiksat/trm_bellman/checkpoints/exp1_v4/model_a_prime/seed{seed}/` | Coauthor |
| `I2` | Depth-mismatch stability claim | Seeds `41-43` have eval outputs; Model B checkpoints are missing | Recover the historical Model B checkpoints, or retrain the full `S_exp110` set under a fresh frozen protocol | `buiksat_trm/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml` via `//buiksat_trm:upi_trm_train` | `buiksat_trm/data/sudoku-4x4-trivial` | `44-50` if artifacts restore cleanly; otherwise `41-50` | `5000` | Budget `~1-2 h/model/seed` | `/home/buiksat/trm_bellman/checkpoints/exp1_v4/model_b/seed{seed}/` | Coauthor |
| `I3` | Depth-mismatch stability claim | Seeds `41-43` have eval outputs; `b0.pt` and `b1.pt` are missing | Restore `B0` and `B1`; if impossible, regenerate batches and rerun all 10 seeds end-to-end before aggregation | `//buiksat_trm:eval_unroll_sensitivity` compare + radius sweep at `R in {10,100,0}` | restored fixed `B0` / `B1` batches under `/home/buiksat/trm_bellman/artifacts/eval_batches/v3/` | `44-50` only if the original batches are restored; otherwise `41-50` | eval only | Budget `~30-45 min/seed` once batches and checkpoints exist | `/home/buiksat/trm_bellman/results/validation/exp1_v4/seed{seed}/` and `/home/buiksat/trm_bellman/results/plot_data/exp1_v4/` | Coauthor |
| `P1` | Main figures/tables from existing pipelines | partial | Refresh plots and tables after H1, C1-C4, I1-I3 land | `//buiksat_trm:plot_table3_hard`, `//buiksat_trm:plot_table3_hard_controlled`, `//buiksat_trm:make_paper_figures_exp1_final` | n/a | n/a | n/a | `<1 h` total once data exists | existing figure/table locations | You |

## Compute-Contingent Queue

These runs are valuable, but not on the critical path before the hard-4x4 baseline story is credible.

| ID | Purpose | Why contingent | What to do if compute opens up |
|----|---------|----------------|--------------------------------|
| `O1` | 9x9 capability fallback | Historical package is stale: five `25k` attempts exist, but only seeds `1,2,3` completed; current configs are `50k` with constraint-aware masking. `M1` already succeeded, so 9x9 is now supporting evidence rather than a rescue path for the main capability claim | Rerun fresh 9x9 under the current configs, starting with UPI + one trusted external baseline, only if you want stronger supporting evidence beyond the locked hard-4x4 no-mask package. |
| `O2` | Closer baseline: n-step DQN | Useful only after T2.1 shows whether PPO/A2C already settle the reviewer trust problem | Implement after T0.2 only if the hard-4x4 story still looks vulnerable. Cap implementation effort at one working day. |

## Buck Launch Templates

All Python entrypoints should be run via Buck from `/home/buiksat/fbsource/fbcode`.

Path policy in the commands below:
- Inputs inside `buiksat_trm/...` are relative to `/home/buiksat/fbsource/fbcode`.
- Outputs under `/home/buiksat/trm_bellman/...` stay absolute so logs, checkpoints, and artifacts land in the working repo.
- `PF1` is already cleared on this machine via `/home/buiksat/trm_bellman/scripts/restore_4x4_datasets.sh`; rerun that helper on a fresh machine before hard-4x4 launches.
- On this machine, if the default remote `buck2 run` path fails with a `//buiksat_trm:utils` packaging error mentioning `utils/__init__.py`, retry with `--local-only`. Seeds `0-3` of the fresh current-code H1 run were launched successfully this way.
- Do not run the `exp1_v4` eval template until Preflight `PF2` and `PF3` are cleared.

### Hard 4x4 no-mask UPI / controlled-2x2 training

```bash
cd /home/buiksat/fbsource/fbcode
WORKTREE=/home/buiksat/trm_bellman

CUDA_VISIBLE_DEVICES=$GPU buck2 run //buiksat_trm:upi_trm_train \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true -- \
  --config buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction_no_mask.yaml \
  --seed $SEED \
  --dataset-paths buiksat_trm/data/sudoku-4x4-easy_6to8empties \
  --train-steps 20000 \
  --no-wandb \
  > ${WORKTREE}/results/table3_hard_6to8/m1_persistent_nc_nomask_s${SEED}.log 2>&1
```

For the controlled 2x2, replace the base config path with one of:
- `buiksat_trm/configs/table3_hard_controlled/episodic_nc_r0_hard.yaml`
- `buiksat_trm/configs/table3_hard_controlled/episodic_nc_r10_hard.yaml`
- `buiksat_trm/configs/table3_hard_controlled/episodic_c_r0_hard.yaml`
- `buiksat_trm/configs/table3_hard_controlled/episodic_c_r10_hard.yaml`

and add:
- `--config buiksat_trm/configs/table3_hard_controlled/no_mask_overlay.yaml`
- output logs under `${WORKTREE}/results/table3_hard_6to8_controlled_nomask/`

Batch launcher used for the completed `10`-seed no-mask 2x2:

```bash
bash /home/buiksat/trm_bellman/scripts/run_table3_hard_controlled_nomask_4gpu.sh
```

### Exp1_v4 training

```bash
cd /home/buiksat/fbsource/fbcode
WORKTREE=/home/buiksat/trm_bellman

CUDA_VISIBLE_DEVICES="" buck2 run //buiksat_trm:upi_trm_train -- \
  --seed $SEED \
  --train-steps 5000 \
  --config buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml \
  --checkpoint-dir ${WORKTREE}/checkpoints/exp1_v4/model_a_prime/seed${SEED} \
  --save-interval 1000 \
  --dataset-paths buiksat_trm/data/sudoku-4x4-trivial
```

Swap the config and checkpoint dir for Model B:
- config: `buiksat_trm/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml`
- checkpoint dir: `${WORKTREE}/checkpoints/exp1_v4/model_b/seed${SEED}`

This mirrors the historical `scripts/run_exp1_multiseed.sh` launcher, which did not pass `--no-wandb`.

### Exp1_v4 evaluation

```bash
cd /home/buiksat/fbsource/fbcode
WORKTREE=/home/buiksat/trm_bellman

buck2 run //buiksat_trm:eval_unroll_sensitivity -- compare \
  --checkpoint_a ${WORKTREE}/checkpoints/exp1_v4/model_a_prime/seed${SEED}/model_step_5000.pt \
  --config_a buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction_no_vhead_norm.yaml \
  --checkpoint_b ${WORKTREE}/checkpoints/exp1_v4/model_b/seed${SEED}/model_step_5000.pt \
  --config_b buiksat_trm/configs/ablations/upi_trm_feasibility_contraction_no_vhead_norm.yaml \
  --batch_b0 ${WORKTREE}/artifacts/eval_batches/v3/b0.pt \
  --batch_b1 ${WORKTREE}/artifacts/eval_batches/v3/b1.pt \
  --out_dir ${WORKTREE}/results/validation/exp1_v4/seed${SEED} \
  --table_out ${WORKTREE}/results/validation/exp1_v4/seed${SEED}/unroll_sensitivity_summary.md \
  --n_mults 1,2,4,8 \
  --seed ${SEED} \
  --estimate_lz
```

Repeat the same command for the radius sweep with:
- `--latent_ball_radius_override 10`
- `--latent_ball_radius_override 100`
- `--latent_ball_radius_override 0`

### 9x9 parallel launcher

```bash
cd /home/buiksat/fbsource/fbcode
WORKTREE=/home/buiksat/trm_bellman

buck2 run //buiksat_trm:run_experiments_parallel -- \
  --configs \
  buiksat_trm/configs/sudoku9x9/upi_trm_9x9.yaml,buiksat_trm/configs/sudoku9x9/ppo_9x9.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1,2,3 \
  --output-dir ${WORKTREE}/results/neurips2026/sudoku9x9
```

### Trusted external baselines

No Buck command exists yet. This is the T0.2 blocker.

What must exist before launch:
- an env wrapper exposing the Sudoku task to the external codebase
- a run entrypoint for PPO
- a run entrypoint for A2C
- output directories under `/home/buiksat/trm_bellman/results/neurips2026/external_hard4x4/`

## Decision Gates

### Gate G1: baseline credibility

- If trusted external PPO/A2C stay near `0%`, the current main claim survives.
- If trusted external PPO/A2C get nontrivial success but remain clearly below UPI, the claim narrows to a quantified gap rather than "baselines fail completely."
- If a trusted external baseline matches or beats UPI on hard 4x4, the capability claim is dead. The paper must pivot to a stability/mechanism story.

### Gate G1': masking-ablation pivot

- Resolved masked-protocol observation: fresh current-code hard-4x4 `H1` finishes at `0.996` mean over `S_hard10`, and the in-house current-code A2C smoke gate reaches `1.000` on seed `0`. The masked hard-4x4 capability story is therefore ceiling-limited.
- Resolved no-mask capability package: current-code `M1` finishes at `0.574` mean (`std 0.122`) over `S_hard10`, while in-house no-mask A2C stays at `0.000` over seeds `0-3`.
- Action: lock no-mask hard-4x4 as the main capability anchor and present masked-vs-no-mask as the paper's masking-ablation table.
- Optional follow-up only: backfill in-house A2C seeds `4-9` or trusted external baselines if matched-seed reviewer optics becomes necessary.
- Do not spend additional masked hard-4x4 baseline compute unless it directly informs a no-mask or 9x9 package.

### Gate G2: contraction language

- Resolved 10-seed controlled 2x2 no-mask package: `nc_r0 = 0.350`, `nc_r10 = 0.482`, `c_r0 = 0.374`, `c_r10 = 0.502`.
- Projection is the dominant main effect: `+0.132` in the `nc` row and `+0.128` in the `c` row.
- Contraction adds only a small average lift: `+0.024` at `R=0`, `+0.020` at `R=10`, with interaction `-0.004`.
- Keep the narrative: projection drives most hard-4x4 performance stability. Contraction is at most a modest secondary dial and should be justified primarily through `exp1_v4`, not through hard-4x4 return gains.

### Gate G3: exp1_v4 scope

- If the 10-seed `exp1_v4` package preserves the B0 effect, keep the B0 contraction claim in the main paper.
- If the original `B0` / `B1` artifacts cannot be restored, do not mix historical seeds `41-43` with fresh batches; rerun `41-50` under the replacement protocol before making the aggregate claim.
- If the B0 effect weakens materially, move the radius-sweep claim to appendix/supporting and let the controlled hard 2x2 carry the main stability story.

### Gate G4: 9x9 demotion

- If fresh 50k 9x9 runs do not land in time, or if they remain noisy, keep 9x9 in appendix/supporting only.
- Do not mix old `25k` and new `50k` 9x9 results in one table.

## Immediate Next Blockers

The next blockers after this document are `PF2`, `PF3`, and then either `T0.2` or `O1` depending on how much additional baseline credibility you still need beyond the locked in-house no-mask package.

For internal reruns:
- `PF1` is cleared on this machine via `/home/buiksat/trm_bellman/scripts/restore_4x4_datasets.sh`.
- `H1` is complete as a masked-protocol sanity check: fresh current-code `persistent_nc` finishes at `0.996` mean over `S_hard10`.
- A current-code in-house A2C smoke gate reaches `1.000` at step `20000` on the same masked hard-4x4 protocol, so masked hard-4x4 capability framing is retired.
- `M1` is complete: current-code no-mask `persistent_nc` finishes at `0.574` mean (`std 0.122`) over `S_hard10`, while in-house `A2C` is `0.000` over seeds `0-3`.
- `C1-C4` are complete as fresh current-code no-mask reruns: `nc_r0 = 0.350`, `nc_r10 = 0.482`, `c_r0 = 0.374`, `c_r10 = 0.502`.
- The controlled 2x2 resolves the hard-4x4 mechanism story: projection is the primary stabilizer; contraction adds only a small average lift and does not reduce variance under projection.
- `PF2` and `PF3` are now the primary blockers, because `exp1_v4` is the remaining place where contraction can earn a main-text empirical role.
- Optional only: backfill in-house A2C no-mask seeds `4-9` for matched seed counts or reviewer optics; this is not a paper blocker.
- If `PF2` / `PF3` stall, writing can proceed with the locked capability and controlled-2x2 package while treating the contraction stability story as pending.

For trusted external baselines:
- The no-mask hard-4x4 capability anchor is already established via `M1`; external baselines are now optional reviewer-trust work, not a prerequisite for establishing the gap.
- There is currently no `CleanRL` or `Stable-Baselines3` integration anywhere in `/home/buiksat/trm_bellman`.
- The existing baseline shell scripts only rerun the in-house PPO/A2C/DQN modes through `upi_trm_train.py`, which is not enough to repair reviewer trust.
- `T0.2` acceptance remains:

```bash
python run_baseline.py --algo ppo --env sudoku4x4
```

must complete one episode without error once the external baseline wrapper exists.
