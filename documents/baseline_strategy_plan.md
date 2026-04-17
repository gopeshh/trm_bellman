# Baseline Strategy Plan: CleanRL as Diff Target, Not Replacement

**Status:** Draft — to execute after the current no-mask 10-seed sweep completes.
**Context:** UPI-TRM paper baselines review (see prior review rounds in this repo; 11 runtime fixes and 2 test fixes landed in `3c2e959` and follow-ups).

---

## Decision

Keep the current three-tier baseline setup. **Do not** replace `rl/algos/{ppo,a2c,dqn}.py` with CleanRL.

| Tier | Purpose | Location |
|---|---|---|
| 1. TRM-shared custom baselines | Algorithmic ablation: same backbone + same env, varies only the RL update rule. This is what the paper's main comparison table needs. | `rl/algos/{ppo,a2c,dqn}.py` |
| 2. Trusted external baseline (SB3) | Evidence that off-the-shelf PPO/A2C/DQN (MLP backbone) cannot solve the task. | `run_baseline.py` via Stable-Baselines3 |
| 3. CleanRL | Diff target for correctness review + optional third reference point in the results figure. | External — vendored under `external_baselines/cleanrl/` only if we decide to include it in the paper. |

Rationale (verbatim from review):

- Swapping in CleanRL confounds **algorithm** (CPI mixture + exact baseline vs. clipped surrogate) with **architecture** (TRM recursive latent vs. MLP). That destroys the "same backbone, only the update changes" ablation, which is the strongest evidence the theoretical contribution does the work.
- Porting `PlanEditEnv` into a Gym wrapper to feed CleanRL means either stripping the TRM batch dict interface (losing apples-to-apples) or forking CleanRL until it's unrecognizable. Both paths cost a week and still require the custom baselines.
- CleanRL's `ppo.py` / `dqn.py` have been benchmarked across 34+ games and thousands of W&B runs. They're the right **reference implementation** to diff against for catching any remaining bugs in our custom baselines.

---

## Execution Plan

### Phase 0 — Prerequisites

- [ ] Wait for the in-flight `hard4x4_trm_baselines_nomask_10seed` sweep to finish.
- [ ] Freeze results files under `results/hard4x4_trm_baselines_nomask_10seed/`.
- [ ] Tag the commit the sweep ran against (`git tag paper/v1-baselines <sha>`).

### Phase 1 — CleanRL diff review (correctness insurance)

Goal: confirm our custom PPO/A2C/DQN updates don't silently diverge from the community reference beyond the documented TRM-specific deviations.

1. [ ] Clone CleanRL at the commit pinned in `cleanrl-0.md`:
   ```bash
   git clone https://github.com/vwxyzjn/cleanrl.git external_baselines/cleanrl
   cd external_baselines/cleanrl && git checkout v1.0.0  # or latest stable tag
   ```
2. [ ] Line-by-line diff (~1 hour each). For each of the three pairs, produce a short `documents/baseline_diff_{algo}.md` summarizing:
   - [ ] `rl/algos/ppo.py::PPOTrainer.update` vs `external_baselines/cleanrl/cleanrl/ppo.py` lines ~260–360. Check:
     - GAE direction and `last_value` bootstrap semantics
     - Advantage normalization scope (per-minibatch vs. per-batch)
     - Clip coefficient placement (`torch.min(surr1, surr2)` direction)
     - Value-loss clipping semantics
     - Entropy sign and coefficient
     - Learning rate schedule (CleanRL uses linear anneal by default)
   - [ ] `rl/algos/a2c.py::A2CTrainer.update` vs `cleanrl/ppo.py` with `update_epochs=1, clip_coef=inf` (A2C = degenerate PPO).
   - [ ] `rl/algos/dqn.py::DQNTrainer.train_batch` vs `external_baselines/cleanrl/cleanrl/dqn.py` lines ~160–220. Check:
     - Target-network sync schedule (`target_network_frequency` in train steps vs. env steps)
     - Double-DQN action selection
     - Replay buffer layout (`ReplayBuffer` from stable_baselines3 vs. our `deque`)
     - Loss function (Huber vs. MSE — CleanRL uses `F.mse_loss`; we use `F.mse_loss`; ok)
     - `start_e` / `end_e` / `exploration_fraction` semantics (env-step units in both, confirmed fixed on our side in `compute_epsilon_decay_steps`)
3. [ ] For each intentional deviation, add a short code comment in our file citing `cleanrl/{algo}.py` line numbers and the reason (TRM backbone, plan-edit action masking, reward shaping, etc.).
4. [ ] If the diff surfaces a real bug: write a regression test in `tests/test_rl_algos_mock.py` first, then patch. Do **not** land silent fixes.

**Exit criterion:** `documents/baseline_diff_{ppo,a2c,dqn}.md` exist, each ending with either "no divergence" or an enumerated list of commented-deviations. No outstanding TODOs.

### Phase 2 — Sanity calibration on CartPole (optional but cheap)

Confirms the custom PPO/A2C/DQN matchers behave like CleanRL on a task that isn't `PlanEditEnv`.

1. [ ] Write a thin Gym→`PlanEditEnv`-shaped adapter for `CartPole-v1` that wraps observations as the state dict `{inputs, puzzle_identifiers}` expected by our trainers. (Puzzle embedding disabled.)
2. [ ] Run `rl/algos/ppo.py` on adapted CartPole for 50k steps, seeds 0–4. Expect reward ≈ 500.
3. [ ] Run `external_baselines/cleanrl/cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000` for seeds 0–4.
4. [ ] Compare mean ± std; custom should be within ~10% of CleanRL's reference curve.
5. [ ] Archive both runs under `documents/baseline_calibration_cartpole/`.

**Exit criterion:** Plot showing both implementations reach the CartPole optimum at similar wall-clock and sample efficiency. If custom is substantially slower or noisier, investigate before paper submission.

**Skip this phase if compute is tight** — Phase 1 (diff review) catches most issues. Phase 2 is confirmation, not proof.

### Phase 3 — (Optional) Third external baseline in the results figure

Only if SB3 alone isn't persuasive to reviewers.

1. [ ] Wrap `PlanEditEnv` as a Gym `Env` subclass (discrete action space = flattened `pos * vocab + tok` ∪ STOP, observation = flat inputs tensor). Call it `GymSudoku4x4Env` — the shape already exists in `external_baselines/sudoku4x4_env.py`; verify it's CleanRL-compatible.
2. [ ] Run CleanRL PPO on it: 20 000 env steps × 10 seeds, matching the sweep budget. Save under `results/hard4x4_cleanrl_ppo_10seed/`.
3. [ ] Add the CleanRL PPO curve to the main results figure as a third bar alongside TRM+PPO (ours) and SB3 PPO.

**Exit criterion:** Figure shows three PPO curves (TRM-shared, SB3-MLP, CleanRL-MLP) all below UPI-TRM, making the story robust to "your baseline is weak" pushback.

Decision gate: only start Phase 3 if a reviewer or co-author flags SB3 alone as insufficient. Otherwise skip.

### Phase 4 — Paper copy

1. [ ] Add a subsection to the paper's Experiments section explicitly naming all three baseline tiers and their purpose:
   - TRM+{PPO,A2C,DQN} as algorithmic ablation (same backbone)
   - SB3 as trusted external reference
   - (If Phase 3 ran) CleanRL as independent external reference
2. [ ] Cite CleanRL's JMLR 2022 paper (Huang et al.) and SB3 (Raffin et al., 2021) in the bibliography even if not used for plots.
3. [ ] In the baselines appendix, point to `documents/baseline_diff_{ppo,a2c,dqn}.md` for readers who want implementation-level detail.

---

## Non-goals

- Replacing `rl/algos/` with CleanRL. Kills the ablation.
- Porting `PlanEditEnv` to Gym just to run CleanRL. High cost, low payoff unless Phase 3 triggers.
- Using LeanRL (CUDAGraphs fork) — relevant only if sweep wall-clock becomes a blocker; current 20k-step sweeps are fast enough.

---

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 1 diff uncovers a real bug after the sweep is already done | Medium | Rerun only the affected algorithm's 10 seeds, not the full sweep. Keep raw logs so the unaffected algorithms don't need to rerun. |
| Reviewer asks "why not CleanRL as primary baseline?" | High | Phase 4 paper copy explicitly addresses this in the baselines subsection. |
| Phase 3 CleanRL run shows CleanRL PPO beats our TRM+PPO | Low but possible | This would be interesting — means our custom PPO implementation has residual issues vs CleanRL. Address by either fixing and rerunning, or reporting both results honestly. |
| CleanRL pins old `gym` instead of `gymnasium`, conflicts with repo env | Medium | Vendor CleanRL in its own venv under `external_baselines/cleanrl/`. Do not add to the main `requirements.txt`. |

---

## Checklist summary

**Must do before paper submission:**
- [ ] Phase 0: sweep freeze + tag
- [ ] Phase 1: three diff documents + intentional-deviation comments
- [ ] Phase 4.1 + 4.2: paper copy naming baselines + citations

**Should do if compute allows:**
- [ ] Phase 2: CartPole calibration

**Do only if reviewer pushback on baseline strength:**
- [ ] Phase 3: CleanRL PPO on Sudoku4x4 as third curve

---

## References

- CleanRL repo: https://github.com/vwxyzjn/cleanrl
- CleanRL paper: Huang et al., *CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms*, JMLR 2022. https://jmlr.org/papers/v23/21-1342.html
- Current custom baselines: `rl/algos/{ppo,a2c,dqn}.py`
- SB3 external baseline wrapper: `run_baseline.py`, `external_baselines/sudoku4x4_env.py`
- Prior review rounds (runtime fixes): commit `3c2e959` "Fix two rounds of baseline trainer bugs"
- Test regression closure: `tests/test_baseline_selection.py:150` (`MockBaseModel.inner`)
