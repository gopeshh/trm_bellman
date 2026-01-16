# TRM-RL Implementation Plan

> NOTE: This document is a historical implementation plan and may be stale.
> For the current experiment results and next steps, see:
> - `EXPERIMENT_RESULTS_4x4_FEASIBILITY.md`
> - `EXPERIMENT_PLAN_ICML.md`

**Goal:** Extend the official TinyRecursiveModels codebase into a plan-space RL framework implementing UPI–TRM (Unrolled Policy Iteration for TRMs) and producing ICML 2026 submission-ready experiments (Sudoku plan-editing + ablations + diagnostics + figures/tables).

**Constraints honored:**
- Deadline: implementation complete by Jan 05, 2026 (paper deadline Jan 18, 2026)
- Compute: 2× A100
- Backward compatibility: original TRM supervised pipeline remains unchanged; RL code lives in new modules with minimal optional hooks.

**Repo facts we'll leverage:**
- Repo already contains Hydra-style experiment launching, dataset builders (dataset/build_sudoku_dataset.py, etc.), and a supervised pretrain.py pipeline with documented commands and runtimes.
- Key architecture code lives under models/recursive_reasoning/trm.py (and related files like models/layers.py), with training driven by pretrain.py.

---

## Phase Map and Dependencies
- Phase 0 → 1 → 2 → 3 → 4 → 5 → 6
- You can do Phase 3 (baselines) partly in parallel with Phase 2 (UPI core) once the env + logging exist.

Critical path for Jan 05: 0 → 1 → 2 → 4 → 5 → 6 (with minimal baselines from Phase 3).

---

## Phase 0: Environment & Infrastructure Setup

### Tasks

- **Task 0.1:** Create a reproducible RL dev environment
  - Objective: Add an RL-friendly environment while staying compatible with TRM's existing requirements.
  - Plan:
    - Create environment.yml (conda) or requirements_rl.txt (pip) that extends existing requirements.txt (do not replace).
    - Mirror TRM's documented baseline setup: Python 3.10, CUDA 12.x, torch install, adam-atan2, optional W&B.
    - Additions for RL + analysis:
      - gymnasium (or lightweight custom env; gym optional)
      - scipy, pandas
      - matplotlib
      - pytest, pytest-xdist
      - hydra-core (likely already), omegaconf
      - rich (nice CLI logs)
      - tensorboard (optional; keep W&B as first-class)
- **Task 0.2:** Add a clean RL package boundary
  - Objective: Keep RL code isolated to preserve original TRM flows.
  - Plan: Create top-level Python package:
    - trm_rl/ (new)
    - Avoid modifying existing pretrain.py/datasets unless needed for lightweight hooks.
- **Task 0.3:** Add project-wide reproducibility utilities
  - Objective: Uniform seeding + deterministic toggles across env/model/training.
  - Plan: Add trm_rl/utils/reproducibility.py with:
    - seed_everything(seed: int, deterministic: bool) -> None
    - per-worker seed handling for dataloaders / vectorized envs
    - store seed in every checkpoint + run metadata
- **Task 0.4:** Logging/experiment tracking scaffold
  - Objective: Standardize metrics, checkpoints, config dumps.
  - Plan: Add trm_rl/utils/logging.py for:
    - W&B (primary), local JSONL fallback
    - log directory structure: runs/{exp_name}/{run_id}/
    - save full Hydra config to runs/.../config.yaml
    - append Git commit hash (if available)
    - Use W&B exactly like TRM suggests (login optional).

### Files Modified/Created

- **Created:**
  - trm_rl/__init__.py
  - trm_rl/utils/reproducibility.py
  - trm_rl/utils/logging.py
  - trm_rl/utils/paths.py
  - requirements_rl.txt (or) environment.yml
  - pytest.ini
  - tests/test_imports.py
- **Modified (optional):**
  - README.md (add a new "TRM-RL" section without changing existing instructions)

### Checkpoint Criteria

- ✅ python pretrain.py ... still runs exactly as documented in TRM README.
- ✅ python -c "import trm_rl" works
- ✅ pytest -q passes
- ✅ A dummy RL run can create runs/.../config.yaml and log at least one scalar

---

## Phase 1: Plan-Space MDP Environments (Sudoku)

This phase implements the plan-space MDP:
- State: $s = (x, y)$ where $x$ is fixed puzzle instance, $y$ current plan grid
- Action: plan edit (cell, value) (plus optional UNDO)
- Reward: checker + shaping (and a sparse variant)

### Tasks

- **Task 1.1:** Define a canonical Sudoku representation
  - Objective: Standardize state tensors for both TRM and baselines.
  - Decisions (implementation defaults):
    - Grid: int8 tensor of shape (9, 9) values in {0..9} where 0=empty
    - Mask for fixed clues: bool(9,9) (clue cells not editable)
  - Success criterion: deterministic serialization to/from JSON for logging.
- **Task 1.2:** Implement Sudoku checker + shaped reward
  - Objective: Implement $c(x,y) \in [0, C_{max}]$ and shaped reward:
  - From your draft:

$$r(s,a,s') = r_0(s,a,s') + \gamma \Phi(s') - \Phi(s) \quad \text{with } \Phi(x,y)=c(x,y)$$

  and absorbing normalization.
  - Checker score options (support both):
    1. Constraint score: $c = C_{max} - \text{violations}$ (no solution needed)
    2. Exact score: if ground-truth solution exists, optionally track correctness (diagnostic only)
  - Base reward $r_0$:
    - $r_0=+1$ on transition that solves and enters absorbing
    - else 0
  - Absorbing-state convention: Implement the training-time equivalent of your $V(s_{abs})=-C_{max}$ shaping convention by:
    - using terminal bootstrap value $V_{terminal} = -C_{max}$ when done=True
    - and (for K-step rollouts) padding remaining steps as absorbing self-loops in the return computation
- **Task 1.3:** Define action space + masking
  - Default action set: $a = (i, j, d)$ with $i,j \in [0..8]$, $d \in [1..9]$ → 729 actions
  - Mask invalid actions:
    - cannot change clue cell
    - optionally: cannot set same value as already present
    - optionally: allow overwriting non-clue cells (recommended for plan editing)
  - Provide ActionIndexer:
    - encode(i,j,d)->int
    - decode(a_id)->(i,j,d)
    - valid_action_mask(x,y)->Bool[729]
- **Task 1.4:** Episode termination and horizon
  - Implement:
    - done=True if solved OR if $t = T_{max}$
    - default $T_{max}$ sweepable (e.g., 81, 160, 256)
  - Provide info with:
    - is_solved, violations, c_score, steps, terminated_by_timeout
- **Task 1.5:** Dataset loader integration
  - Objective: Reuse TRM's dataset-building pipeline for Sudoku instances when possible. TRM already provides dataset/build_sudoku_dataset.py to generate Sudoku-Extreme data.
  - Implement a small loader that reads puzzles from data/... and yields (x, clue_mask).
- **Task 1.6:** Add vectorized environment runner
  - Objective: Fast experience collection.
  - Implement a simple parallelization:
    - either gymnasium.vector.SyncVectorEnv
    - or custom multiprocessing with shared-memory tensors
  - Must support deterministic seeding per env.

### Files Modified/Created

- **Created:**
  - trm_rl/envs/__init__.py
  - trm_rl/envs/sudoku.py
  - trm_rl/envs/sudoku_checker.py
  - trm_rl/envs/action_space.py
  - trm_rl/envs/vector_env.py
  - trm_rl/data/sudoku_instances.py
  - tests/test_sudoku_env.py
  - tests/test_reward_shaping.py
  - tests/test_action_masking.py

### Checkpoint Criteria

- ✅ pytest -q tests/test_sudoku_env.py passes:
  - stepping is deterministic under fixed seed
  - invalid actions are masked
  - shaped reward matches $\gamma \cdot c_{next} - c_{cur}$ + terminal reward when solved
- ✅ You can run python -m trm_rl.envs.sudoku to play a random agent for 10 episodes and print mean score.
- ✅ $T_{max}$, shaped vs sparse reward, overwrite behavior are all configurable.

---

## Phase 2: Core Algorithm Implementation (UPI–TRM)

This phase implements your algorithmic template:

#### Core Equations from Your Draft (for implementation reference)

- Inner unroll (with optional projection):
$$z^{t+1} = (\Pi_R \circ f_\theta)(z^t, y, x)$$
- Unrolled value: $U_n(s) = V_\psi(z^{(n)}(s), x)$
- K-step target: $G^{(K)} = \sum_{k=0}^{K-1} \gamma^k r_k + \gamma^K \bar{V}(s_K)$
- Value loss: $\mathcal{L}_{val} = \mathbb{E}[(U_n(s_0) - \text{stopgrad}(G^{(K)}))^2]$
- Centered advantage (exact baseline):
$$\hat{A}(s,a) = \hat{Q}(s,a) - \mathbb{E}_{b \sim \pi(\cdot|s)}[\hat{Q}(s,b)]$$
- Mixture update: $\pi_{new} = (1-\alpha) \pi + \alpha \pi_0$ and distill $\pi_{new} \to \pi_\phi$

#### UPI–TRM Pseudocode (Implementation-Oriented)

```
Initialize policy π_φ, value (f_θ, V_ψ), target value V_ψ̄, replay buffer B
for iteration = 1..N:
  # Collect data
  for env step:
    s = (x, y)
    z_n = Unroll(f_θ, x, y, n, Π_R)
    a ~ π_φ(·|x, y, z_n) with action-mask
    s', r, done = env.step(a)
    store (s, a, r, s', done) into B

  # Sample K-step sequences from B
  batch = sample_sequences(B, K)

  # Value update
  compute targets G^(K) using V_ψ̄ and absorbing padding
  minimize (U_n(s_0) - stopgrad(G^(K)))² w.r.t (θ, ψ)

  # Candidate policy update
  compute Q̂ and centered Â with exact baseline under π_φ
  take gradient step to produce candidate policy π_0

  # Conservative mixture + distillation
  π_new = (1-α)π_φ + α π_0   (distribution-level mixture)
  update π_φ by minimizing KL(π_new || π_φ) on batch states

  # Target update
  ψ̄ ← τ ψ̄ + (1-τ) ψ
```

### Tasks

- **Task 2.1:** Implement TRM-to-RL adapter for latent unrolling
  - Objective: Get $z^{(n)}$ and support reset-latent / persistent-latent modes.
  - Integration points:
    - Prefer importing from models/recursive_reasoning/trm.py (official TRM implementation).
  - Approach (backward-compatible):
    - Create trm_rl/models/trm_adapter.py that wraps the existing TRM model.
    - If TRM code doesn't expose latents, add an optional hook (Phase 2.1b) to expose them without changing default behavior.
  - Method signatures (new):

```python
class TRMLatentAdapter(nn.Module):
    def __init__(self, trm_model: nn.Module, z_dim: int, projection_radius: float | None):
        ...

    def reset_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ...

    def unroll(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n: int,
        z0: torch.Tensor | None = None,
        detach_warmup: bool = False,
    ) -> torch.Tensor:
        """Return z^(n). If z0 is None -> reset-latent behavior."""
```

  - Projection $\Pi_R$:

```python
def project_ball(z, R):  # Euclidean projection
    return z if ||z|| <= R else R * z / ||z||
```

- **Task 2.2:** Implement policy head $\pi_\phi$ for plan edits
  - Output: logits over 729 actions (masked)
  - Method signature:

```python
class EditPolicyHead(nn.Module):
    def forward(self, x, y, z_n, action_mask) -> torch.distributions.Categorical:
        ...
```

  - Masking: set logits of invalid actions to -1e9 before softmax.

- **Task 2.3:** Implement value head $V_\psi$
  - Output: scalar value per state
  - Signature:

```python
class LatentValueHead(nn.Module):
    def forward(self, z_n, x) -> torch.Tensor:  # shape [B]
        ...
```

- **Task 2.4:** Implement replay buffer with K-step sampling
  - Requirements:
    - store (x, y, a, r, done, next_y) (x fixed per episode, can store once per episode)
    - allow sampling contiguous K-step windows
    - handle done padding to absorbing return convention
  - Signature:

```python
class SequenceReplayBuffer:
    def add_step(...): ...
    def sample_sequences(self, batch_size: int, K: int) -> Batch:
        ...
```

- **Task 2.5:** Implement centered advantage computation
  - Exact baseline requirement:
$$B_V(s) = \sum_a \pi(a|s) \hat{Q}(s,a) \quad \text{over valid actions}$$
  - Compute $\hat{Q}(s,a)$:
    - start with one-step:
$$\hat{Q}(s,a) = r + \gamma V_{target}(s')$$
    - then optionally implement K-step $\hat{Q}^{(K)}$ later.
  - Signature:

```python
def compute_centered_advantages(
    policy_dist, q_values  # both over action dimension
) -> torch.Tensor:
    """Return A_hat with E_{a~pi}[A_hat]=0 exactly."""
```

- **Task 2.6:** Implement candidate policy update $\pi_0$
  - Implement as:
    - a copy of policy params → apply 1–M gradient steps using A_hat
    - keep it simple: 1 step per iteration initially
  - You'll need a small helper:

```python
def make_candidate_policy(policy: nn.Module, grads, step_size: float) -> nn.Module:
    ...
```

- **Task 2.7:** Implement mixture policy $\pi_{new}$ and distillation
  - Mixture distribution:
$$p_{new} = (1-\alpha) p_{old} + \alpha p_{cand}$$
  - Distill by minimizing $KL(p_{new} \| p_\phi)$ over a minibatch of states
  - Must support action masks.
- **Task 2.8:** Implement value training with K-step bootstrapped regression
  - Target:
$$G^{(K)} = \sum \gamma^k r_k + \gamma^K V_{target}(s_K) \quad \text{with absorbing padding}$$
  - Use target network for stability:
$$\bar{\psi} \leftarrow \tau \bar{\psi} + (1-\tau) \psi$$
- **Task 2.9:** Add contraction diagnostics ($L_z$ estimates)
  - Log:
    - $\hat{L}_z$ via finite differences on replay batches
    - $\hat{C}_z$ = first-step latent delta norm
    - optional: centering_defect should be 0 by construction for exact baseline

### Files Modified/Created

- **Created:**
  - trm_rl/models/__init__.py
  - trm_rl/models/trm_adapter.py
  - trm_rl/models/policy_head.py
  - trm_rl/models/value_head.py
  - trm_rl/buffers/replay.py
  - trm_rl/algos/__init__.py
  - trm_rl/algos/upi_trm.py
  - trm_rl/algos/advantages.py
  - trm_rl/algos/distill.py
  - trm_rl/algos/targets.py
  - trm_rl/train.py (entrypoint)
  - tests/test_advantage_centering.py
  - tests/test_replay_kstep.py
  - tests/test_mixture_policy.py
- **Modified (only if needed):**
  - models/recursive_reasoning/trm.py (optional hook: return latents / expose unroll function)

### Checkpoint Criteria

- ✅ Unit tests pass:
  - exact centering: $|\sum_a \pi(a|s) \cdot \hat{A}(s,a)| < 10^{-6}$
  - mixture policy sums to 1 and respects masking
  - K-step targets match absorbing padding spec
- ✅ python -m trm_rl.train exp=debug_sudoku runs end-to-end for ~5000 env steps without NaNs
- ✅ On a tiny 4×4 or toy Sudoku subset, the agent can overfit (solve rate improves above random)

---

## Phase 3: Baselines & Comparators

Your draft experiments compare:
1. UPI–TRM (ours)
2. Supervised TRM (deep supervision, optional test-time editing)
3. PPO / Actor-Critic over plan edits with standard value net (no contraction control)
4. No-recursion baseline (direct $(x,y) \to (\pi,V)$)

### Tasks

- **Task 3.1:** PPO baseline (plan-editing)
  - Implement minimal PPO with:
    - clipped surrogate
    - entropy bonus
    - value loss
  - Use same env, same action space, same mask.
  - File: trm_rl/algos/ppo.py
- **Task 3.2:** Vanilla actor-critic baseline
  - Simpler than PPO for fast experiments.
  - File: trm_rl/algos/a2c.py
- **Task 3.3:** No-recursion baseline network
  - Replace TRM unroll with a direct encoder:
    - e.g., small Transformer/MLP over grid tokens
  - File: trm_rl/models/norec_encoder.py
- **Task 3.4:** Supervised TRM baseline integration
  - Use TRM's existing pretrain.py recipe for Sudoku.
  - Add an evaluation script that:
    - loads pretrained TRM weights
    - runs a fixed "edit loop" (if applicable) or uses TRM prediction directly
  - File: trm_rl/baselines/supervised_trm_eval.py

### Files Modified/Created

- **Created:**
  - trm_rl/algos/ppo.py
  - trm_rl/algos/a2c.py
  - trm_rl/models/norec_encoder.py
  - trm_rl/baselines/supervised_trm_eval.py
  - trm_rl/baselines/common.py
  - tests/test_ppo_shapes.py

### Checkpoint Criteria

- ✅ PPO/A2C can run 1h without divergence
- ✅ Baselines log the same metric keys as UPI–TRM
- ✅ Supervised baseline evaluation script can reproduce TRM README-level behavior (sanity), e.g., Sudoku-Extreme expected accuracies/runtimes in the supervised setting.

---

## Phase 4: Experiment Harness + Config System

You want experiments "submission-ready" with clear configs and reproducible commands.

### Tasks

- **Task 4.1:** Hydra config tree for RL
  - Match TRM's style (arch=trm, overrides on CLI).
  - Create:
    - trm_rl/config/config.yaml (root)
    - trm_rl/config/exp/*.yaml (each experiment)
    - trm_rl/config/agent/*.yaml (upi_trm, ppo, a2c)
    - trm_rl/config/env/*.yaml (sudoku_shaped, sudoku_sparse, undo_variant)
    - trm_rl/config/model/*.yaml (trm_adapter, norec)
    - trm_rl/config/optim/*.yaml
- **Task 4.2:** Unified metrics schema
  - Define canonical keys logged every eval:
    - `eval/solved_rate`
    - `eval/mean_steps_to_solve`
    - `eval/return_mean`
    - `train/td_residual_mean`
    - `train/Lz_hat_p95`
    - `train/centering_defect` (should be $\approx 0$ for exact baseline)
    - `train/neg_improvement_rate` (see below)
  - Add trm_rl/utils/metrics.py
- **Task 4.3:** Evaluation protocol
  - Fixed evaluation set (held-out puzzles)
  - Report:
    - solved rate within T_max
    - steps-to-solve distribution
    - mean shaped return
  - Add trm_rl/eval/eval_runner.py
- **Task 4.4:** "Update stability" metrics
  - Implement:
    - neg_improvement_rate: fraction of eval checkpoints where mean return decreased vs previous checkpoint
    - policy_kl_to_prev (if available)
    - correlation diagnostics: corr(predicted_penalty, observed_drop) (Phase 5)

### Files Modified/Created

- **Created:**
  - trm_rl/config/config.yaml
  - trm_rl/config/exp/exp_main_sudoku.yaml
  - trm_rl/config/exp/ablations_dials.yaml
  - trm_rl/config/exp/curriculum.yaml
  - trm_rl/config/exp/undo.yaml
  - trm_rl/config/exp/theory_unrolling_decay.yaml
  - trm_rl/config/exp/theory_centering_alpha.yaml
  - trm_rl/config/exp/theory_residual_transfer.yaml
  - trm_rl/config/exp/persistent_vs_reset.yaml
  - trm_rl/eval/eval_runner.py
  - trm_rl/utils/metrics.py

### Checkpoint Criteria

- ✅ Every experiment runs via a single command (examples below)
- ✅ Each run produces:
  - runs/.../config.yaml
  - runs/.../checkpoints/ckpt_latest.pt
  - runs/.../metrics.jsonl
- ✅ Evaluations are deterministic under fixed seed

---

## Phase 5: Implement the Paper Experiments

This section maps each experiment in your \section{Experiments} draft to what to run, what code is needed, hyperparameters, metrics, and expected baselines.

Note: where your draft says "Sudoku reward is 1 on success and 0 otherwise", you also requested a shaped-reward setting aligned with the theory. So every experiment below should be runnable in both:
(A) shaped reward and (B) sparse terminal reward (appendix).

---

### Experiment 1: Main Comparison on 9×9 Sudoku (Shaped + Sparse)

**Objective:** Compare UPI–TRM vs baselines on plan-editing Sudoku.

#### Algorithms

- UPI–TRM: as in Phase 2 pseudocode (mixture + distillation).
- PPO (baseline): standard PPO actor-critic, no contraction enforcement.
- No-recursion baseline: direct encoder $\to (\pi,V)$.
- Supervised TRM: run pretrain.py per TRM README and evaluate as a non-RL baseline.

#### Required modules

- trm_rl/algos/upi_trm.py, ppo.py, a2c.py
- trm_rl/envs/sudoku.py
- trm_rl/eval/eval_runner.py

#### Hyperparameters

- Environment:
- $\gamma=0.99$
- $T_{max} \in \{81, 160\}$ (use 160 for main)
- reward mode: shaped (main) + sparse (appendix)
- UPI–TRM core:
- $n \in \{5, 10\}$ (default 10)
- $K \in \{3, 5\}$ (default 5)
- $\alpha \in \{0.01, 0.05\}$ (default 0.05)
- contraction dial $\lambda \in \{0.8, 0.9\}$ (controls effective $L_z$; default 0.9)
- target EMA: $\tau=0.995$
- Optim:
- lr_actor=3e-4, lr_value=1e-4 (start)
- batch size: 256–1024 (based on VRAM)
- replay size: 1e6 transitions (or 1e5 sequences)

#### Baseline configs

- PPO:
- clip=0.2, ent=0.01, vf_coef=0.5, lr=3e-4
- No-rec:
- comparable params to TRM policy+value heads
- Supervised TRM:
- Use README recipe for Sudoku-Extreme for the "best supervised" reference (not directly comparable compute, but useful anchor).

#### Metrics

- solved rate (%)
- steps-to-solve (median, mean, histogram)
- mean return ± 95% CI
- stability: neg-improvement rate

#### Command

```bash
# UPI–TRM main (shaped)
python -m trm_rl.train exp=exp_main_sudoku agent=upi_trm env=sudoku_shaped seed=0

# PPO baseline
python -m trm_rl.train exp=exp_main_sudoku agent=ppo env=sudoku_shaped seed=0

# Sparse variant (appendix)
python -m trm_rl.train exp=exp_main_sudoku agent=upi_trm env=sudoku_sparse seed=0
```

#### Expected runtime (2×A100)
- 1 run (UPI–TRM): ~2–8 hours depending on env throughput + unroll n
- PPO: ~1–6 hours
- Supervised TRM reference: TRM repo reports < 20 hours for Sudoku-Extreme on 1×L40S for their supervised pretrain recipe.
(On A100 it may be faster, but treat this as a rough anchor.)

---

### Experiment 2: Four-Dial Ablation ($n$, $\lambda$, $K$, $\alpha$)

Your draft sweep ranges:
- $n \in \{1,2,5,10,20\}$
- $\lambda \in \{0.7,0.8,0.9,0.95,1.0\}$ ($\lambda$ controls contraction / empirical $L_z$)
- $K \in \{1,3,5,10\}$
- $\alpha \in \{0.01,0.05,0.1,0.2\}$

#### Implementation requirements

- Configurable dials in Hydra configs
- Logger must record:
- $\hat{L}_z$ distribution
- td_residual proxy
- eval curves

#### Sweep strategy under 2×A100

- Do 1 dial at a time, others fixed:
- default: $n=10, \lambda=0.9, K=5, \alpha=0.05$
- Use:
- 2 seeds for ablations (3 seeds if time allows)
- shorter training horizon for ablations (e.g., 50% steps of main)

#### Commands

```bash
# n sweep
python -m trm_rl.train exp=ablations_dials sweep=n_values="[1,2,5,10,20]" seed=0

# lambda sweep
python -m trm_rl.train exp=ablations_dials sweep=lambda_values="[0.7,0.8,0.9,0.95,1.0]" seed=0
```

#### Metrics

- solved rate vs dial
- stability vs dial:
- neg-improvement rate
- TD residual curves
- $\hat{L}_z$ staying $< 1$

#### Expected runtime

- Per sweep run: 1–4 hours
- Total (all sweeps × 2 seeds): plan for 2–4 days of wall clock with 2 GPUs if parallelized.

---

### Experiment 3: Curriculum $4 \times 4 \to 9 \times 9$ vs Scratch

**Objective:** Validate curriculum improves sample efficiency under fixed compute.

#### Implementation

- Provide trm_rl/data/curriculum.py:
- stage configs: sudoku_4x4, then finetune sudoku_9x9
- stage transitions by steps budget, not epochs
- Reuse same agent/algorithm.

#### Hyperparameters

- total environment steps fixed (e.g., 50M)
- curriculum split: 20% ($4 \times 4$) + 80% ($9 \times 9$)

#### Commands

```bash
python -m trm_rl.train exp=curriculum agent=upi_trm seed=0
python -m trm_rl.train exp=exp_main_sudoku agent=upi_trm seed=0  # scratch
```

#### Metrics

- return/solved curves vs env steps
- compute-normalized performance (solved rate at fixed wall-clock)

#### Runtime

- Similar to main run; curriculum adds minimal overhead.

---

### Experiment 4: Action-Space Variant with UNDO

**Objective:** Test if adding UNDO reduces instability / improves exploration.

#### Implementation

- Extend action space:
- Add action id UNDO that reverts last non-clue edit (stack-based)
- Env state must maintain a small history (stack of edits).
- Keep history in env only; not part of $y$ tensor.

#### Configs

- env=sudoku_shaped_undo vs env=sudoku_shaped

#### Metrics

- solved rate
- steps-to-solve
- number of undos used
- stability

#### Runtime

- Slight overhead; negligible.

---

### Experiment 5: Diagnostics Suite (Theory-Mapped Logging)

Your draft diagnostics:
1. TD residual curves
2. empirical $\hat{L}_z$ over training
3. centering defect $\varepsilon_{cent}$
4. correlation between predicted penalty and observed instability

#### Implementation details

- TD residual proxy: empirical $|U_n(s) - (r + \gamma U_n(s'))|$ or K-step version
- $\hat{L}_z$: finite difference on latent update map (sample pairs $z, z'$)
- $\varepsilon_{cent}$: should be $\approx 0$ if exact baseline is used; log to verify
- penalty term proxy:
- estimate $\alpha \cdot \varepsilon_{A_0}$ using batch estimate
$$\hat{\varepsilon}_{A_0} = \max_s |\mathbb{E}_{a \sim \pi_0} [\hat{A} - A_{ref}]|$$
(in practice, use $|\mathbb{E}_{a \sim \pi_0} [\hat{A}]|$ as a proxy since true $A_{ref}$ unknown)
- correlate with observed negative improvement events

#### Command

```bash
python -m trm_rl.train exp=exp_main_sudoku diagnostics=on seed=0
```


---

### Experiment 6: Theory-Targeted Experiments (Appendix)

These are ideal for an ICML appendix because they validate specific claims.

#### 6.1 Unrolling Bias Decay vs n (Fixed Policy $\pi$)

**Goal:** Show $\|U_n - V^{\pi}_{MC}\|$ decreases with $n$; relate to $\hat{L}_z$.

**Implementation:**
- Freeze policy $\pi$ (trained or random).
- Estimate Monte Carlo $V^{\pi}_{MC}$ with long rollouts (e.g., 512 steps or until termination).
- Compare $U_n(s)$ for $n \in \{1,2,5,10,20\}$.
- Plot error vs $n$ with multiple $\lambda$ values.

**Command:**

```bash
python -m trm_rl.train exp=theory_unrolling_decay agent=fixed_policy
```

#### 6.2 Centering and $\alpha$ Experiment

**Goal:** Compare:
- (a) exact centering (sum over actions)
- (b) approximate baseline (sampled)
- (c) standard V-baseline (GAE style)

and show the $\alpha$-dependent stability behavior.

**Implementation:**
- Add baseline mode flag:
- adv_baseline=exact|sampled|gae
- Track:
- stability (drop rate)
- empirical $\hat{\varepsilon}_{cent}$

**Command:**

```bash
python -m trm_rl.train exp=theory_centering_alpha adv_baseline=exact seed=0
python -m trm_rl.train exp=theory_centering_alpha adv_baseline=gae seed=0
```

#### 6.3 Residual Transfer Experiment

**Goal:** Show correlation between:
- empirical residual $\|U_n - T_K^{\pi} U_n\|$ (proxy)
- training stability / performance improvements

#### 6.4 Persistent vs Reset Latent (Equal Compute)

**Goal:** Compare reset vs persistent latent with a strict "equal compute" budget:
- reset-latent: recompute $z$ from scratch each edit
- persistent-latent: carry $z$ forward but limit inner updates

#### Metrics

- solved rate at fixed wall-clock
- drift proxy $\hat{C}_{drift}(n) = \|z_t^{(n)} - z_{init}(x,y_t)\|$

---

## Phase 6: Reproducibility, Checkpointing, and Run Commands

### Tasks

- **Task 6.1:** Deterministic seed handling
  - Seed:
    - Python, NumPy, torch CPU/GPU
    - env RNG
    - dataloader workers
  - Store in:
    - config dump
    - checkpoint metadata
- **Task 6.2:** Checkpoint format
  - Save:
    - policy_state_dict
    - value_state_dict (including TRM adapter weights)
    - target_value_state_dict
    - optimizer states
    - replay buffer cursor (optional)
    - global step, env step, seed, config hash
- **Task 6.3:** "One command per experiment" docs
  - Add trm_rl/README.md documenting:
    - how to generate Sudoku data (reuse TRM dataset builder)
    - how to run each experiment
    - how to regenerate figures/tables

### Files Modified/Created

- trm_rl/README.md
- trm_rl/utils/checkpointing.py
- trm_rl/scripts/run_all.sh (optional launcher)

### Checkpoint Criteria

- ✅ Rerunning the same command with same seed reproduces learning curves up to expected nondeterminism (log determinism flags).
- ✅ Checkpoint reload resumes training identically.
- ✅ Runs are self-contained in runs/....

---

## Phase 7: ICML Deliverables (Figures, Tables, Stats)

### Tasks

- **Task 7.1:** Log aggregation
  - Parse runs/*/metrics.jsonl and/or download from W&B.
  - Output a tidy dataframe artifacts/summary.parquet.
- **Task 7.2:** Figure generation scripts (ICML-ready)
  - Generate PDF figures sized for:
    - 1-column width
    - 2-column width
  - Standard plots:
    - learning curves (mean ± CI)
    - ablation plots (dial vs solved rate)
    - stability plots (neg-improvement rate)
    - diagnostics plots ($\hat{L}_z$, TD residual)
- **Task 7.3:** Table generation + significance
  - For main table: UPI–TRM vs baselines
  - Use:
    - paired bootstrap over seeds (recommended)
    - or Welch t-test if needed
  - Correct for multiple comparisons (Holm–Bonferroni)
  - Emit:
    - paper/tables/main_results.tex
    - paper/tables/ablations.tex
- **Task 7.4:** Appendix experiment plots
  - unrolling bias decay
  - centering vs $\alpha$
  - persistent vs reset

### Files Modified/Created

- **Created:**
  - trm_rl/scripts/aggregate_runs.py
  - trm_rl/scripts/plot_learning_curves.py
  - trm_rl/scripts/plot_ablations.py
  - trm_rl/scripts/plot_diagnostics.py
  - trm_rl/scripts/make_tables.py
  - trm_rl/scripts/stats_tests.py
  - paper/figures/.gitkeep
  - paper/tables/.gitkeep

### Checkpoint Criteria

- ✅ Running:

```bash
python trm_rl/scripts/aggregate_runs.py --runs runs/
python trm_rl/scripts/plot_learning_curves.py --in artifacts/summary.parquet --out paper/figures/
python trm_rl/scripts/make_tables.py --in artifacts/summary.parquet --out paper/tables/
```

produces ICML-includable PDFs + LaTeX tables.

---

## Phase 8: Execution Plan Under 2×A100 (Practical Scheduling)

**Minimal "ICML submission set" (finish by Jan 05):**

1. Main comparison (shaped): UPI–TRM + PPO + No-rec (2–3 seeds)
2. Dial ablations: $n$ + $\alpha$ sweeps (2 seeds)
3. Two key diagnostics plots: $\hat{L}_z$ and TD residual over time
4. Figure+table scripts complete

**Recommended GPU allocation:**

- GPU0: run training for UPI–TRM / PPO
- GPU1: run ablation jobs in parallel
- Use torchrun only if your implementation truly benefits; otherwise run independent jobs.

---

## Appendix: Troubleshooting Common Issues

1) NaNs / divergence in UPI–TRM
- Symptoms: value loss explodes, logits become NaN
- Fixes:
- reduce `lr_actor` first
- clamp rewards / normalize returns
- ensure masked logits are large negative (e.g., `-1e9`) not `-inf`
- add gradient clipping (1.0)
- increase target EMA smoothing ($\tau$ closer to 1)

2) Centering defect not $\approx 0$
- Likely cause: baseline computed with sampling or mask mismatch.
- Fix: ensure baseline sums only over valid actions with properly normalized $\pi$.

3) Very slow throughput with large $n$
- Fixes:
- cache embeddings of $x$ (clues are fixed per episode)
- reduce $n$ during early training, ramp up later (curriculum on compute)
- batch latent unroll across envs on GPU (avoid per-env Python loops)

4) Reward shaping behaves oddly
- Fix: verify $r = \gamma \cdot c_{next} - c_{cur}$ on nonterminal transitions (unit test).
- Ensure terminal handling matches your absorbing convention (bootstrap $V_{terminal}=-C_{max}$).

5) Action masking bugs (agent edits clues)
- Fix: unit test that every clue cell has all actions masked.
- Log fraction of invalid actions sampled (should be ~0).

---

## Quick Reference: Existing TRM Supervised Commands (Anchor)

TRM's README provides Sudoku-Extreme dataset generation and training commands (useful for supervised baseline and for reusing the dataset builder).


