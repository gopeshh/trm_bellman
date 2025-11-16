# Unrolled Policy Iteration in Plan Space for Tiny Recursive Models (UPI–TRM)

This document describes the RL extension of Tiny Recursive Models (TRMs) implemented in this repo, based on the ICML 2026 submission:

> *Unrolled Policy Iteration in Plan Space for Tiny Recursive Models:  
> Contraction-Controlled Latent Evaluation and Conservative Improvement*

The goal is to:

- Interpret TRM’s outer loop as a **plan-space meta–MDP**.
- Add a **latent value head** and **edit policy head** on top of the TRM backbone.
- Train them via **unrolled generalized policy iteration** (UPI–TRM), with explicit stability dials.

## 1. Plan–Space Meta–MDP

We reinterpret TRM’s outer edit loop as a discounted MDP over instance–plan pairs.

- **State space**
  - Meta–state: \(s = (x, y)\)
    - \(x \in \mathcal{X}\): puzzle instance (e.g., Sudoku grid, ARC input).
    - \(y \in \mathcal{Y}\): current *plan* or candidate solution (grid / sequence / structure).
- **Actions**
  - Edit actions \(a \in \mathcal{A}(y)\): local modifications to the plan, e.g.,
    - grid: “set cell (i, j) to digit d”;
    - sequence: “replace token at position t with token v”.
  - A special **stop** action \(a_{\text{stop}}\) that terminates the episode:
    - No further edits; terminal reward emitted once.
- **Transition dynamics**
  - Given state \(s = (x, y)\) and action \(a\):
    - If \(a = a_{\text{stop}}\): we stay at the same plan and transition to an absorbing terminal state.
    - Otherwise: apply a deterministic edit
      \[
      y' = \mathrm{edit}(y, a; x), \quad s' = (x, y').
      \]
  - The environment object `PlanEditEnv` wraps this logic.
- **Rewards and shaping**
  - We assume a **checker** \(c(x, y) \in \mathbb{R}\) that measures constraint satisfaction or correctness (e.g., number of satisfied Sudoku constraints, 0/1 correctness, negative loss).
  - Base reward:
    - Intermediate steps: `0.0`
    - Terminal step (stop or solved): `r_term = checker(x, y_final)` (possibly normalized / clipped).
  - Potential-based shaping (Ng et al., 1999):
    - Potential \( \Phi(x, y) := c(x, y) \).
    - Shaped reward:
      \[
      r(s, a, s') = r_0(s, a, s') + \gamma \Phi(s') - \Phi(s),
      \]
      where \(r_0\) carries the terminal score.
  - This preserves optimal policies while giving dense feedback.
- **Discount and horizon**
  - Discount factor \( \gamma \in (0, 1) \).
  - Max edit budget `max_edits` (config) to cap episode length.

The helper class `PlanEditEnv` in `rl/envs/plan_edit_env.py` implements this meta–MDP:

- `reset()` → returns initial `(x, y)` from the dataset.
- `step(action)` → applies edits, returns `(next_state, reward, done, info)`.

## 2. Architectural Changes for UPI–TRM

We keep the original TRM backbone but expose inner components and add RL heads.

### 2.1 TRM backbone (unchanged behavior, new APIs)

We introduce explicit APIs for the *inner* latent recursion:

```python
class TRM(nn.Module):
    def init_latent(self, x, y):
        """Initialize latent z^(0) from (x, y)."""

    def update_latent(self, z, y, x):
        """One application of f_θ(z, y, x) → z^(t+1)."""

    def unroll_latent(self, x, y, n: int):
        """Run inner recursion n steps, return z^(n) and optionally all zs."""
        ...
```

The existing supervised forward path is left intact (for pretraining and original experiments) and built on top of these primitives.

### 2.2 Latent value head and “used value” \(U_n\)

We add a value head \(V_\psi(z, x)\) and define
\[
U_n(s) = V_\psi(z^{(n)}(s), x),
\]
where \(z^{(n)}(s)\) is the latent after \(n\) inner steps from \((x, y)\).

Implementation:

- `models/value_head.py`
- `LatentValueHead(z_dim, x_embed_dim, hidden_dim)` → small MLP on `[z, x_embed]`
- TRM exposes:

```python
def used_value(self, x, y, n: int):
    z_n, _ = self.unroll_latent(x, y, n)
    x_embed = self.embed_input(x)
    return self.value_head(z_n, x_embed)  # [B]
```

This is the value function used by the RL algorithm (UPI–TRM).

### 2.3 Edit policy head (plan–space policy)

On top of \(z^{(n)}\), TRM also outputs a plan-editing policy:

- Policy \( \pi_\phi(a \mid s) = \pi_\phi(a \mid y, z^{(n)}, x) \)
- Implemented via `EditPolicyHead`
- Input: concatenation of latent \(z^{(n)}\), embedded input \(x\), and embedded plan \(y\)
- Output: logits over a discrete action space (location + symbol + special STOP action)

TRM exposes:

```python
def policy_dist(self, x, y, n: int, action_mask=None) -> Categorical:
    z_n, _ = self.unroll_latent(x, y, n)
    x_embed = self.embed_input(x)
    y_embed = self.embed_plan(y)
    return self.edit_policy(z_n, x_embed, y_embed, action_mask=action_mask)
```

### 2.4 Contraction via spectral normalization

To control the latent recursion, we enforce a global Lipschitz bound on the inner map \(f_\theta\) in \(z\):
\[
\lVert f_\theta(z, y, x) - f_\theta(z', y, x) \rVert \le L_z \lVert z - z' \rVert, \quad L_z < 1.
\]

Implementation:

- Apply spectral normalization to all linear layers on the \(z \rightarrow z\) path in \(f_\theta\).
- Track a proxy \(L_z\) from the product of spectral bounds.
- Similarly, bound the Lipschitz constant \(L_V\) of the value head with respect to \(z\) (spectral norm on the \(z \rightarrow\) value path).

These constraints support the theory in the paper (existence of fixed-point latent \(z^\star\), geometric decay of finite-unrolling bias, etc.).

## 3. UPI–TRM Training Loop

The RL training follows the “unrolled generalized policy iteration” pattern.

1. **Data collection in plan space**
   - Sample a puzzle \(x\) from the dataset.
   - Initialize a plan \(y\) (trivial or heuristic) → state \(s = (x, y)\).
   - Unroll TRM inner recursion for \(n\) steps to obtain \(z^{(n)}\) and compute:
     - policy \(\pi_\phi(a \mid s)\) via `EditPolicyHead`;
     - used value \(U_n(s)\) via `LatentValueHead`.
   - Sample edit action \(a\) (or STOP), apply `PlanEditEnv.step`, get \((s', r, done)\).
   - Repeat until STOP or `max_edits` reached.
   - Store transitions / \(K\)-step segments in a replay buffer.
2. **Value update (K-step bootstrapped targets)**
   - For each sampled starting state \(s_0\), roll out \(K\) steps under current policy \(\pi\):
     \[
     G^{(K)}(s_0) = \sum_{k=0}^{K-1} \gamma^k r_k + \gamma^K U_n(s_K; \bar\psi),
     \]
     where \(\bar\psi\) are target-network parameters (EMA copy of value head).
   - Minimize bootstrapped MSE:
     \[
     \mathcal{L}_{\text{val}} = \mathbb{E}_{s_0 \sim \mu}\left[ U_n(s_0; \psi) - \operatorname{sg}(G^{(K)}(s_0)) \right]^2.
     \]
   - This is the practical counterpart of the \(K\)-step operator analysis in Section 5 of the paper.
3. **Policy update (conservative improvement)**
   - Compute one-step TD advantages using \(U_n\):
     \[
     \widehat{A}_{U_n}(s, a) = r + \gamma U_n(s') - U_n(s).
     \]
   - Optimize a standard policy-gradient loss with entropy regularization:
     \[
     \mathcal{L}_{\text{policy}} = -\mathbb{E}\left[\log \pi_\phi(a \mid s)\,\widehat{A}_{U_n}(s, a)\right] - \beta\, H(\pi_\phi(\cdot \mid s)).
     \]
   - To align with conservative policy improvement guarantees (CPI / TRPO), maintain old policy \(\pi\) and an improved candidate \(\pi_0\), then deploy a mixture policy:
     \[
     \pi_{\text{new}} = (1 - \alpha)\pi + \alpha \pi_0,
     \]
     with mixture coefficient \(\alpha\) chosen small enough that the lower bound in Theorem 5.5 (paper) is non-vacuous.
4. **Target network & EMA**
   - Maintain a target copy of the value head parameters \(\bar\psi\).
   - Update after each step:
     \[
     \bar\psi \leftarrow \tau \bar\psi + (1 - \tau)\psi,
     \]
     where \(\tau \approx 0.99\text{–}0.999\) is configured via `rl.target_ema_tau`.
5. **Optional policy distillation**
   - To avoid deploying an explicit mixture indefinitely, periodically distill \(\pi_{\text{new}}\) back into a single network by minimizing
     \[
     \mathcal{L}_{\text{distill}} = \mathbb{E}_{s \sim \text{buffer}}\big[\mathrm{KL}(\pi_{\text{new}}(\cdot \mid s)\,\|\,\pi_\phi(\cdot \mid s))\big].
     \]

In practice, `rl/upi_trm_trainer.py` implements this loop using mini-batches + replay.

## 4. Configs and Flags

The RL extension is controlled via an `RLConfig` (or similar) nested in the main experiment config.

- **Meta–MDP / environment**
  - `rl.gamma`: float – discount.
  - `rl.max_edits`: int – max plan edits per episode.
  - `rl.task_type`: str – optional (e.g., `"sudoku"`, `"arc"`).
- **Unrolling / evaluation**
  - `rl.inner_unroll_n`: int – number of inner latent steps (\(n\)).
  - `rl.K`: int – \(K\)-step horizon for bootstrapped targets.
  - `rl.target_ema_tau`: float – EMA coefficient for target value head.
- **Contraction / Lipschitz dials**
  - `rl.enable_contraction`: bool – enable spectral norm on \(f_\theta\) and value head.
  - `rl.target_Lz`: float – target upper bound proxy for latent Lipschitz constant.
  - `rl.target_Lv`: float – target upper bound for value head Lipschitz constant.
- **Policy update / conservative improvement**
  - `rl.mixture_alpha`: float – CPI mixture coefficient between old and new policy.
  - `rl.trust_region_kl`: float – optional KL trust-region radius (TRPO-style variant).
  - `rl.entropy_coef`: float – entropy regularization weight.
- **Optimization and replay**
  - `rl.value_lr`: float – learning rate for value + latent evaluator.
  - `rl.policy_lr`: float – learning rate for policy parameters.
  - `rl.replay_capacity`: int – replay buffer capacity.
  - `rl.batch_size`: int – mini-batch size.

YAML config examples (e.g., `config/rl/upi_trm_sudoku.yaml`) should set these appropriately for each benchmark.

## 5. Mapping to the ICML Paper

This implementation corresponds directly to the theory sections:

- Section 2 (Meta–MDP over plans) → `PlanEditEnv`, state \((x, y)\), action space, potential-based reward shaping.
- Section 3–4 (Latent evaluator, contraction) → `TRM.unroll_latent`, `TRM.used_value`, `LatentValueHead`; spectral normalization utilities for inner map and value head.
- Section 5 (Error bounds) → determines how we choose `inner_unroll_n`, `K`, and spectral norm targets (\(L_z\), \(L_V\)), and how we interpret empirical TD errors.
- Section 6 (Conservative policy improvement) → CPI-style mixture updates with `mixture_alpha`; optional KL trust region; advantage estimates based on \(U_n\).
- Algorithm 1 (UPI–TRM) → implemented by `rl/upi_trm_trainer.py` and the training script (e.g., `upi_trm_train.py`).

This doc is meant as the bridge between the theoretical ICML write-up and the concrete implementation for reviewers and future contributors.

