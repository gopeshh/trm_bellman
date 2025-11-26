import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from evaluators.rl_plan_evaluator import evaluate_plan_policy, evaluate_plan_policy_with_scores
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from utils.lipschitz import (
    estimate_local_Lz,
    estimate_Cz,
    estimate_Lv,
    compute_unrolling_term_proxy,
)


@dataclass
class Transition:
    x: Any
    y: Any
    action: torch.Tensor
    reward: torch.Tensor
    x_next: Any
    y_next: Any
    done: torch.Tensor
    episode_id: int
    timestep: int


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def __len__(self) -> int:
        return len(self.storage)

    def sample_batch(self, batch_size: int) -> List[Transition]:
        assert len(self.storage) >= batch_size, "Not enough transitions in replay buffer"
        indices = torch.randint(low=0, high=len(self.storage), size=(batch_size,)).tolist()
        return [self.storage[i] for i in indices]


def compute_k_step_bootstrapped_target(
    rewards_K: torch.Tensor,
    dones_K: torch.Tensor,
    steps_taken: torch.Tensor,
    v_K: torch.Tensor,
    gamma: float,
    K: int,
    exact_k_step_targets: bool = False,
) -> torch.Tensor:
    """
    Compute K-step bootstrapped targets with proper terminal masking.
    """
    batch_size = rewards_K.shape[0]

    # Mask rewards beyond steps_taken for each sample (defensive against improper padding)
    step_indices = torch.arange(K, device=rewards_K.device).unsqueeze(0)  # [1, K]
    valid_mask = step_indices < steps_taken.unsqueeze(1)  # [batch_size, K]
    rewards_K = rewards_K * valid_mask

    gammas = rewards_K.new_tensor([gamma**k for k in range(K)])
    reward_returns = (rewards_K * gammas).sum(dim=1)

    if exact_k_step_targets:
        bootstrap_factor = gamma**K
    else:
        bootstrap_factor = gamma ** steps_taken.float()

    final_idx = (steps_taken - 1).clamp(min=0)
    batch_indices = torch.arange(batch_size, device=rewards_K.device)
    done_final = dones_K[batch_indices, final_idx]
    # Do not bootstrap from terminal segments: if the segment hits done early,
    # the true return is purely the discounted reward sum (V(s_terminal) = 0).
    not_done_final = (~done_final).to(v_K.dtype)
    v_K = v_K * not_done_final

    return reward_returns + bootstrap_factor * v_K


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE) for a batch of independent transitions.

    This implements a single-step approximation of GAE(λ) from Schulman et al. (2016).
    For independent transitions (not full trajectories), we compute:
        A = δ * (1 + γλ(1-done))
    where δ = r + γV(s') - V(s) is the TD error.

    This approximation assumes the next advantage is approximately equal to the current
    TD error, which provides a λ-weighted blend between TD(0) and a 2-step lookahead.
    - When λ=0: Returns pure TD error (δ)
    - When λ=1: Returns δ * (1 + γ) for non-terminal states (approximates 2-step return)

    For proper multi-step GAE with full trajectory information, use compute_gae_trajectory.

    Args:
        rewards: [B] rewards for each transition
        values: [B] V(s) for each starting state
        next_values: [B] V(s') for each next state
        dones: [B] boolean done flags
        gamma: discount factor
        gae_lambda: GAE λ parameter (0 = TD, 1 = Monte Carlo-like)

    Returns:
        advantages: [B] GAE-approximated advantages
    """
    mask = (~dones).float()
    td_error = rewards + gamma * next_values * mask - values
    
    # For single transitions, we approximate the GAE recursion:
    # A_t = δ_t + γλ * A_{t+1}
    # By assuming A_{t+1} ≈ δ_t (the TD error is a reasonable proxy),
    # we get: A_t ≈ δ_t * (1 + γλ) for non-terminal transitions.
    # This provides meaningful λ-blending without requiring full trajectories.
    gae_factor = 1.0 + gamma * gae_lambda * mask
    return td_error * gae_factor


def compute_gae_trajectory(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute GAE advantages for a full trajectory.

    Args:
        rewards: [T] rewards for each timestep
        values: [T] V(s) for each state in trajectory
        dones: [T] boolean done flags
        gamma: discount factor
        gae_lambda: GAE λ parameter
        last_value: V(s_T) for the final state after trajectory

    Returns:
        advantages: [T] GAE advantages
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0

    # Append last_value for bootstrapping
    values_extended = torch.cat([values, torch.tensor([last_value], device=values.device, dtype=values.dtype)])

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values_extended[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    return advantages


def compute_empirical_bellman_residual(
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> Dict[str, float]:
    """
    Compute empirical Bellman residual statistics for monitoring.

    The Bellman residual is |V(s) - (r + γV(s'))|.

    Args:
        values: [B] V(s) predictions
        rewards: [B] rewards
        next_values: [B] V(s') predictions
        dones: [B] done flags
        gamma: discount factor

    Returns:
        Dictionary with residual statistics (mean, max, std)
    """
    mask = (~dones).float()
    td_target = rewards + gamma * next_values * mask
    residual = (values - td_target).abs()

    return {
        "bellman_residual_mean": float(residual.mean().item()),
        "bellman_residual_max": float(residual.max().item()),
        "bellman_residual_std": float(residual.std(unbiased=False).item()),
    }


class UPITrmTrainer:
    """
    Unrolled Policy Iteration trainer for TinyRecursiveReasoningModel_ACTV1 in plan space.
    """

    def __init__(
        self,
        model: TinyRecursiveReasoningModel_ACTV1,
        env: PlanEditEnv,
        rl_cfg: RLConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.env = env
        self.env_config: PlanEditEnvConfig = env.config
        self.rl_cfg = rl_cfg
        assert math.isclose(
            self.env_config.gamma,
            self.rl_cfg.gamma,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ), f"Env gamma ({self.env_config.gamma}) and RL gamma ({self.rl_cfg.gamma}) must match."
        self.device = device
        self.debug_checks = bool(getattr(self.rl_cfg, "debug_checks", False))

        self.replay = ReplayBuffer(capacity=rl_cfg.replay_capacity)
        self.term_stats = {"stop": 0.0, "solved": 0.0, "budget": 0.0}

        self.target_model = TinyRecursiveReasoningModel_ACTV1(self._config_to_dict(self.model.config)).to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad_(False)
        self._hard_update_target()

        # Policy models:
        # - policy_model_old: deployed policy (data collection)
        # - policy_model_candidate: receives policy-gradient updates
        self.policy_model_old = self.model
        self.policy_model_candidate = TinyRecursiveReasoningModel_ACTV1(
            self._config_to_dict(self.model.config)
        ).to(device)
        self.policy_model_candidate.load_state_dict(self.model.state_dict())

        value_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "edit_policy" in name:
                continue
            value_params.append(param)

        self._freeze_policy_backbone()
        self._sync_candidate_backbone_from_model()

        policy_params: List[nn.Parameter] = []
        for name, param in self.policy_model_candidate.named_parameters():
            if not param.requires_grad:
                continue
            policy_params.append(param)

        self._value_params = value_params
        self._policy_params = policy_params

        old_policy_params: List[nn.Parameter] = []
        for name, param in self.policy_model_old.named_parameters():
            if not param.requires_grad:
                continue
            if not name.startswith("edit_policy"):
                continue
            old_policy_params.append(param)
        self._old_policy_params = old_policy_params

        self.value_opt = torch.optim.Adam(value_params, lr=rl_cfg.value_lr)
        self.policy_opt = torch.optim.Adam(policy_params, lr=rl_cfg.policy_lr)
        self.old_policy_distill_opt: Optional[torch.optim.Optimizer] = None
        if old_policy_params:
            self.old_policy_distill_opt = torch.optim.Adam(old_policy_params, lr=rl_cfg.policy_lr)
        self._next_episode_id: int = 0

        # Adaptive KL coefficient for trust-region updates (PPO-style)
        # This coefficient is adjusted based on the KL divergence from old to new policy
        self._kl_coef: float = 1.0
        self._kl_target: float = getattr(self.rl_cfg, "trust_region_kl", 0.01)

    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        if hasattr(config, "model_dump"):
            return config.model_dump()
        return config.dict()

    def _hard_update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self) -> None:
        tau = self.rl_cfg.target_ema_tau
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target_model.parameters()):
                p_targ.data.mul_(tau).add_(p.data, alpha=1 - tau)

    def _mixed_policy_dist(self, x_batch, y_batch, n: int):
        """
        Return the behavior policy distribution used for data collection.
        Mathematically, this is intended to correspond to a mixture policy
        pi_beh = (1 - alpha) * pi_old + alpha * pi_candidate,
        optionally mixed with a small uniform component (policy_epsilon).
        Note: The deployed policy_model_old is updated toward the candidate policy
        via parameter interpolation in `_sync_policy_old_towards_candidate`, so
        its action distribution is only an approximation of this mixture unless
        `distill_mixture_policy=True`, which triggers an explicit KL distillation.
        """

        alpha = self.rl_cfg.mixture_alpha
        dist_old = self.policy_model_old.policy_dist(x_batch, y_batch, n=n)
        dist_new = self.policy_model_candidate.policy_dist(x_batch, y_batch, n=n)

        probs_old = dist_old.probs
        probs_new = dist_new.probs
        probs_mix = (1.0 - alpha) * probs_old + alpha * probs_new

        eps = getattr(self.rl_cfg, "policy_epsilon", 0.0)
        if eps > 0.0:
            num_actions = probs_mix.shape[-1]
            uniform = torch.full_like(probs_mix, 1.0 / num_actions)
            probs_mix = (1.0 - eps) * probs_mix + eps * uniform

        return torch.distributions.Categorical(probs=probs_mix)

    def _sync_policy_old_towards_candidate(self) -> None:
        """
        Interpolate the deployed policy's edit_policy head toward the candidate's head.
        This function operates in parameter space, not directly in distribution space,
        so policy_model_old's action distribution will only approximate the ideal
        mixture pi_new = (1 - alpha) * pi_old + alpha * pi_candidate.
        
        Note: For a more exact treatment, use distill_mixture_policy=True to distill
        the distributional mixture _mixed_policy_dist into policy_model_old using
        a KL loss, rather than raw weight interpolation.
        """

        alpha = self.rl_cfg.mixture_alpha
        if alpha <= 0.0:
            return

        non_edit_snapshot: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            params_old = dict(self.policy_model_old.named_parameters())
            params_new = dict(self.policy_model_candidate.named_parameters())
            if self.debug_checks:
                for name, p_old in params_old.items():
                    if name.startswith("edit_policy"):
                        continue
                    non_edit_snapshot[name] = p_old.data.clone()
            for name, p_new in params_new.items():
                if not name.startswith("edit_policy"):
                    continue
                p_old = params_old.get(name)
                if p_old is None:
                    continue
                p_old.data.mul_(1.0 - alpha).add_(p_new.data, alpha=alpha)
        if self.debug_checks and non_edit_snapshot:
            with torch.no_grad():
                for name, before in non_edit_snapshot.items():
                    current = dict(self.policy_model_old.named_parameters()).get(name)
                    if current is None:
                        continue
                    if not torch.allclose(current.data, before, atol=1e-7, rtol=1e-5):
                        raise AssertionError(
                            f"_sync_policy_old_towards_candidate unexpectedly modified non-edit parameter `{name}`"
                        )

    def _update_kl_coef(self, kl_div: float) -> None:
        """
        Adaptively update the KL penalty coefficient based on observed KL divergence.
        
        This implements an adaptive scheme similar to PPO's adaptive KL penalty:
        - If KL > 2 * target: increase coef by 1.5x (policy is changing too much)
        - If KL < target / 2: decrease coef by 0.5x (policy is changing too little, can be more aggressive)
        - Otherwise: keep coef stable
        
        The coefficient is clamped to [0.01, 100] to prevent extreme values.
        """
        target = self._kl_target
        
        if kl_div > 2.0 * target:
            # KL too high - increase penalty to constrain policy changes
            self._kl_coef = min(self._kl_coef * 1.5, 100.0)
        elif kl_div < target / 2.0:
            # KL too low - decrease penalty to allow more exploration
            self._kl_coef = max(self._kl_coef / 1.5, 0.01)
        # Otherwise keep coefficient stable

    def _freeze_policy_backbone(self) -> None:
        """
        Only allow the edit-policy head parameters to receive policy-gradient updates.
        """

        # Candidate policy (actor)
        if self.policy_model_candidate.edit_policy is not None:
            for name, param in self.policy_model_candidate.named_parameters():
                trainable = name.startswith("edit_policy")
                param.requires_grad_(trainable)

        # Old policy model (deployed policy, aliased with self.model)
        if self.policy_model_old.edit_policy is not None:
            for name, param in self.policy_model_old.named_parameters():
                trainable = name.startswith("edit_policy")
                if self.policy_model_old is self.model:
                    # Preserve critic training while still documenting which params participate in policy gradients.
                    if trainable:
                        param.requires_grad_(True)
                else:
                    param.requires_grad_(param.requires_grad and trainable)

    def _sync_candidate_backbone_from_model(self) -> None:
        """
        Copy all non-policy-head parameters from the critic / deployed model (self.model)
        into the candidate policy model (self.policy_model_candidate), without touching
        the candidate's edit_policy head weights or their requires_grad flags.
        This keeps the candidate policy operating on the same latent representation
        as the critic, while still allowing its edit_policy head to be optimized
        separately by policy gradients.
        """
        if self.policy_model_candidate is None:
            return

        with torch.no_grad():
            src_params = dict(self.model.named_parameters())
            for name, param in self.policy_model_candidate.named_parameters():
                # Only sync non-policy parameters, i.e., everything except edit_policy.*
                if name.startswith("edit_policy"):
                    continue
                src_param = src_params.get(name)
                if src_param is None:
                    continue
                param.data.copy_(src_param.data)

    def _maybe_run_value_debug_checks(self, x_batch: Dict[str, torch.Tensor], y_batch: torch.Tensor) -> None:
        if not self.debug_checks or self.model.value_head is None:
            return
        self._debug_assert_plan_affects_value(x_batch, y_batch)
        self._debug_log_local_contraction(x_batch, y_batch)

    def _debug_assert_plan_affects_value(
        self, x_batch: Dict[str, torch.Tensor], y_batch: torch.Tensor
    ) -> None:
        if y_batch.dim() == 0 or y_batch.shape[0] < 2:
            return
        perm = torch.randperm(y_batch.shape[0], device=y_batch.device)
        y_shuffled = y_batch[perm]
        if torch.equal(y_shuffled, y_batch):
            return
        with torch.no_grad():
            v_orig = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
            v_perm = self.model.used_value(x_batch, y_shuffled, n=self.rl_cfg.inner_unroll_n)
        if torch.allclose(v_orig, v_perm, atol=1e-5, rtol=1e-4):
            raise AssertionError("used_value outputs are invariant to plan changes in debug mode.")

    def _debug_log_local_contraction(self, x_batch: Dict[str, torch.Tensor], y_batch: torch.Tensor) -> None:
        if y_batch.numel() == 0:
            return
        x_single = {k: v[:1].detach().clone() for k, v in x_batch.items()}
        y_single = y_batch[:1].detach().clone()
        try:
            with torch.no_grad():
                carry = self.model.eval_latent(x_single, y_single, n=self.rl_cfg.inner_unroll_n)
                latent_batch = self.model._standardize_latent_batch(x_single, y_single)
                context = self.model._build_latent_context_with_plan(latent_batch)
                est, samples = estimate_local_Lz(
                    self.model.inner, carry, context, num_samples=2, return_samples=True
                )
            if samples.numel() > 0:
                mean_Lz = float(samples.mean().item())
                median_Lz = float(samples.median().item())
                max_Lz = float(samples.max().item())
            else:
                mean_Lz = median_Lz = max_Lz = 0.0
            print(
                "[debug] est_local_Lz={:.4f} mean_Lz={:.4f} median_Lz={:.4f} max_Lz={:.4f} "
                "target_Lz={:.3f} target_Lv={:.3f}".format(
                    est,
                    mean_Lz,
                    median_Lz,
                    max_Lz,
                    float(self.model.config.rl_target_Lz),
                    float(self.model.config.rl_target_Lv),
                )
            )
        except Exception as exc:
            print(f"[debug] local Lipschitz estimate failed: {exc}")

    def collect_episode(self) -> None:
        """
        Run a single episode in the plan-space env using the current policy_dist,
        store transitions in replay buffer.
        """

        self.model.eval()
        self.policy_model_candidate.eval()
        episode_id = self._next_episode_id
        t = 0
        x, y = self.env.reset()
        done = False
        edit_budget = min(self.rl_cfg.max_edits, self.env_config.max_edits)
        last_info: Optional[Dict[str, Any]] = None

        while not done and self.env.step_count < edit_budget:
            batched = self._state_is_batched(x)
            batch_x = self._prepare_batch_x(x, batched=batched)
            batch_y = self._prepare_plan(y, batched=batched)

            dist = self._mixed_policy_dist(batch_x, batch_y, n=self.rl_cfg.inner_unroll_n)
            action = dist.sample().squeeze()  # Ensure scalar (0-D) tensor for single-state sampling

            (x_next, y_next), reward, done, info = self.env.step(action.item())
            last_info = info

            transition = Transition(
                x=self._clone_state(x),
                y=self._clone_state(y),
                action=action.detach().cpu(),
                reward=torch.as_tensor(reward, dtype=torch.float32).view(1),
                x_next=self._clone_state(x_next),
                y_next=self._clone_state(y_next),
                done=torch.tensor([done], dtype=torch.bool),
                episode_id=episode_id,
                timestep=t,
            )
            self.replay.add(transition)

            x, y = x_next, y_next
            t += 1

        self._next_episode_id += 1
        if last_info is not None:
            reason = last_info.get("done_reason")
            if reason in self.term_stats:
                self.term_stats[reason] += 1

    def _prepare_batch_x(self, x: Dict[str, torch.Tensor], batched: bool) -> Dict[str, torch.Tensor]:
        """Delegate to shared batch_utils.prepare_batch_x."""
        return prepare_batch_x(x, self.device, batched)

    def _prepare_plan(self, y: Any, batched: bool) -> torch.Tensor:
        """Delegate to shared batch_utils.prepare_plan."""
        return prepare_plan(y, self.device, batched)

    def _stack_batch(
        self, transitions: List[Transition]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stack a list of Transition objects into batch tensors.
        Assumes x and x_next are dicts with at least "inputs" and "puzzle_identifiers".
        """

        inputs = torch.stack([t.x["inputs"] for t in transitions], dim=0).to(self.device)
        puzzle_ids = torch.stack([t.x["puzzle_identifiers"] for t in transitions], dim=0).to(self.device)
        x_batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

        inputs_next = torch.stack([t.x_next["inputs"] for t in transitions], dim=0).to(self.device)
        puzzle_ids_next = torch.stack([t.x_next["puzzle_identifiers"] for t in transitions], dim=0).to(self.device)
        x_next_batch = {"inputs": inputs_next, "puzzle_identifiers": puzzle_ids_next}

        y_batch = torch.stack([self._plan_tensor(t.y) for t in transitions], dim=0).to(self.device)
        y_next_batch = torch.stack([self._plan_tensor(t.y_next) for t in transitions], dim=0).to(self.device)

        actions = torch.stack([t.action for t in transitions], dim=0).to(self.device)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        rewards = torch.stack([t.reward for t in transitions], dim=0).to(self.device).squeeze(-1)
        dones = torch.stack([t.done for t in transitions], dim=0).to(self.device).squeeze(-1)

        return x_batch, y_batch, x_next_batch, y_next_batch, actions, rewards, dones

    def _plan_tensor(self, plan: Any) -> torch.Tensor:
        if isinstance(plan, dict):
            plan_tensor = plan.get("inputs")
            if plan_tensor is None:
                raise KeyError("Dictionary plan must contain an `inputs` tensor.")
        else:
            plan_tensor = plan
        if not torch.is_tensor(plan_tensor):
            plan_tensor = torch.as_tensor(plan_tensor)
        return plan_tensor

    def _clone_state(self, state: Any) -> Any:
        if isinstance(state, dict):
            return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        if torch.is_tensor(state):
            return state.clone()
        return state

    def _state_is_batched(self, x: Dict[str, torch.Tensor]) -> bool:
        """Delegate to shared batch_utils.state_is_batched."""
        return state_is_batched(x)

    def _sample_k_step_batch(
        self, batch_size: int
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a batch of K-step segments from replay, collecting up to K rewards per start state.
        
        Returns:
            x_batch: Starting states (inputs and puzzle_identifiers)
            y_batch: Starting plans
            xK_batch: End states after steps_taken steps
            yK_batch: End plans after steps_taken steps
            rewards_K: [batch_size, K] rewards (zero-padded if fewer than K steps)
            dones_K: [batch_size, K] done flags (zero-padded if fewer than K steps)
            steps_taken: [batch_size] actual number of steps collected per sample (1 to K)
        """

        assert self.rl_cfg.K >= 1, "RLConfig.K must be >= 1."
        storage = self.replay.storage
        assert len(storage) >= batch_size, "Not enough transitions in replay buffer"

        indices = torch.randint(low=0, high=len(storage), size=(batch_size,)).tolist()
        horizon = self.rl_cfg.K

        start_inputs: List[torch.Tensor] = []
        start_puzzle_ids: List[torch.Tensor] = []
        start_plans: List[torch.Tensor] = []

        end_inputs: List[torch.Tensor] = []
        end_puzzle_ids: List[torch.Tensor] = []
        end_plans: List[torch.Tensor] = []

        rewards_K = torch.zeros(batch_size, horizon, dtype=torch.float32, device=self.device)
        dones_K = torch.zeros(batch_size, horizon, dtype=torch.bool, device=self.device)
        steps_taken = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for batch_idx, start_idx in enumerate(indices):
            transition = storage[start_idx]
            start_inputs.append(transition.x["inputs"])
            start_puzzle_ids.append(transition.x["puzzle_identifiers"])
            start_plans.append(self._plan_tensor(transition.y))

            last_transition = transition
            current_idx = start_idx
            steps = 0
            while steps < horizon and current_idx < len(storage):
                current = storage[current_idx]
                if current.episode_id != transition.episode_id:
                    break

                last_transition = current
                reward_value = float(current.reward.view(-1)[0].item())
                done_value = bool(current.done.view(-1)[0].item())

                rewards_K[batch_idx, steps] = reward_value
                dones_K[batch_idx, steps] = done_value

                steps += 1
                if done_value:
                    break
                current_idx += 1

            end_inputs.append(last_transition.x_next["inputs"])
            end_puzzle_ids.append(last_transition.x_next["puzzle_identifiers"])
            end_plans.append(self._plan_tensor(last_transition.y_next))
            steps_taken[batch_idx] = steps

        x_batch = {
            "inputs": torch.stack(start_inputs, dim=0).to(self.device),
            "puzzle_identifiers": torch.stack(start_puzzle_ids, dim=0).to(self.device),
        }
        y_batch = torch.stack(start_plans, dim=0).to(self.device)

        xK_batch = {
            "inputs": torch.stack(end_inputs, dim=0).to(self.device),
            "puzzle_identifiers": torch.stack(end_puzzle_ids, dim=0).to(self.device),
        }
        yK_batch = torch.stack(end_plans, dim=0).to(self.device)

        return x_batch, y_batch, xK_batch, yK_batch, rewards_K, dones_K, steps_taken

    def value_update(self) -> float:
        """Perform one value-function update using either 1-step or K-step bootstrapped targets."""

        if len(self.replay) < self.rl_cfg.batch_size:
            return 0.0

        debug_batch: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = None
        if self.rl_cfg.K == 1:
            transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
            x_batch, y_batch, x_next_batch, y_next_batch, _, rewards, dones = self._stack_batch(transitions)
            debug_batch = (x_batch, y_batch)

            self.model.train()
            self.value_opt.zero_grad()

            with torch.no_grad():
                v_next = self.target_model.used_value(x_next_batch, y_next_batch, n=self.rl_cfg.inner_unroll_n)
                mask = (~dones).float()
                v_next = v_next * mask

            v_s = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
            td_target = rewards + self.rl_cfg.gamma * v_next
            value_clip = getattr(self.rl_cfg, "value_target_clip", None)
            if value_clip is not None and value_clip > 0:
                td_target = td_target.clamp(-value_clip, value_clip)
                v_s = v_s.clamp(-value_clip, value_clip)
            loss_val = F.mse_loss(v_s, td_target.detach())
        else:
            (
                x_batch,
                y_batch,
                xK_batch,
                yK_batch,
                rewards_K,
                dones_K,
                steps_taken,
            ) = self._sample_k_step_batch(self.rl_cfg.batch_size)
            debug_batch = (x_batch, y_batch)

            self.model.train()
            self.value_opt.zero_grad()

            with torch.no_grad():
                gamma = self.rl_cfg.gamma
                K = self.rl_cfg.K

                v_K = self.target_model.used_value(xK_batch, yK_batch, n=self.rl_cfg.inner_unroll_n)
                G_K = compute_k_step_bootstrapped_target(
                    rewards_K=rewards_K,
                    dones_K=dones_K,
                    steps_taken=steps_taken,
                    v_K=v_K,
                    gamma=gamma,
                    K=K,
                    exact_k_step_targets=bool(getattr(self.rl_cfg, "exact_k_step_targets", False)),
                )

            v_s = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
            value_clip = getattr(self.rl_cfg, "value_target_clip", None)
            if value_clip is not None and value_clip > 0:
                G_K = G_K.clamp(-value_clip, value_clip)
                v_s = v_s.clamp(-value_clip, value_clip)
            loss_val = F.mse_loss(v_s, G_K.detach())

        loss_val.backward()
        value_grad_clip = getattr(self.rl_cfg, "value_grad_clip", None)
        if value_grad_clip is not None and value_grad_clip > 0 and self._value_params:
            nn_utils.clip_grad_norm_(self._value_params, value_grad_clip)
        self.value_opt.step()
        self._soft_update_target()

        if debug_batch is not None:
            self._maybe_run_value_debug_checks(*debug_batch)

        return float(loss_val.item())

    def policy_update(self) -> Dict[str, float]:
        """
        Perform one policy-gradient step with optional:
        - GAE advantage estimation (use_gae=True)
        - Centered advantage estimator (centered_advantage=True)
        - KL trust-region constraint (enable_kl_trust_region=True)
        
        When enable_kl_trust_region=True, this implements an adaptive KL penalty
        similar to PPO's adaptive penalty scheme:
        - Adds a penalty term: kl_coef * KL(old_policy || new_policy) to the loss
        - Adaptively adjusts kl_coef based on observed KL vs target (trust_region_kl):
          * KL > 2*target: increase coef by 1.5x
          * KL < target/2: decrease coef by 1/1.5x
        - Early stops (skips gradient step) if KL > 1.5*target
        
        Returns dict with:
        - loss_policy: policy loss value
        - policy_kl: KL divergence (if enable_kl_trust_region=True)
        - kl_coef: current adaptive KL coefficient (if enable_kl_trust_region=True)
        - kl_early_stop: 1.0 if gradient step was skipped due to high KL (optional)
        """

        if len(self.replay) < self.rl_cfg.batch_size:
            return {"loss_policy": 0.0}

        # Ensure the candidate policy uses the current critic backbone for its features.
        self._sync_candidate_backbone_from_model()

        transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
        x_batch, y_batch, x_next_batch, y_next_batch, actions, rewards, dones = self._stack_batch(transitions)

        self.model.train()
        self.policy_model_candidate.train()
        self.policy_opt.zero_grad()

        with torch.no_grad():
            v_s = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
            v_next = self.model.used_value(x_next_batch, y_next_batch, n=self.rl_cfg.inner_unroll_n)
            mask = (~dones).float()
            v_next_masked = v_next * mask
            gamma = self.rl_cfg.gamma

            # Compute advantages (GAE or 1-step TD)
            if getattr(self.rl_cfg, "use_gae", False):
                gae_lambda = getattr(self.rl_cfg, "gae_lambda", 0.95)
                adv = compute_gae(
                    rewards=rewards,
                    values=v_s,
                    next_values=v_next,
                    dones=dones,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
            else:
                td_target = rewards + gamma * v_next_masked
                adv = td_target - v_s

            if getattr(self.rl_cfg, "centered_advantage", False):
                # Optional: center the estimator so that E[Â] ≈ 0 under the sampled batch,
                # matching the CPI analysis assumption up to sampling noise.
                adv = adv - adv.mean()

            adv_clip = getattr(self.rl_cfg, "advantage_clip", None)
            if adv_clip is not None and adv_clip > 0:
                adv = adv.clamp(-adv_clip, adv_clip)

        # Get old policy distribution for KL computation
        kl_div = None
        enable_kl_trust_region = getattr(self.rl_cfg, "enable_kl_trust_region", False)
        if enable_kl_trust_region:
            with torch.no_grad():
                old_dist = self.policy_model_old.policy_dist(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
                old_log_probs = old_dist.logits.log_softmax(dim=-1)
                old_probs = old_log_probs.exp()

        dist = self.policy_model_candidate.policy_dist(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        loss_policy = -(log_prob * adv.detach()).mean() - self.rl_cfg.entropy_coef * entropy

        # Add adaptive KL penalty if trust-region is enabled (PPO/TRPO-style)
        if enable_kl_trust_region:
            new_log_probs = dist.logits.log_softmax(dim=-1)
            # KL(old || new) = sum(old_probs * (log_old - log_new))
            kl_div = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()
            
            # Add adaptive KL penalty to policy loss
            loss_policy = loss_policy + self._kl_coef * kl_div
            
            # Early stopping: if KL exceeds 1.5x target, skip gradient step
            # This provides a hard constraint on policy change
            kl_val = kl_div.item()
            if kl_val > 1.5 * self._kl_target:
                # Skip this gradient step entirely - KL too large
                self._update_kl_coef(kl_val)
                result = {
                    "loss_policy": float(loss_policy.item()),
                    "policy_kl": kl_val,
                    "kl_coef": self._kl_coef,
                    "kl_early_stop": 1.0,
                }
                return result

        loss_policy.backward()
        policy_grad_clip = getattr(self.rl_cfg, "policy_grad_clip", None)
        if policy_grad_clip is not None and policy_grad_clip > 0 and self._policy_params:
            nn_utils.clip_grad_norm_(self._policy_params, policy_grad_clip)
        self.policy_opt.step()
        
        # Update adaptive KL coefficient after successful step
        if enable_kl_trust_region and kl_div is not None:
            self._update_kl_coef(kl_div.item())
        
        distill_enabled = getattr(self.rl_cfg, "distill_mixture_policy", False)
        if distill_enabled and self.old_policy_distill_opt is not None:
            num_distill = min(len(self.replay), self.rl_cfg.batch_size)
            if num_distill > 0:
                transitions = self.replay.sample_batch(num_distill)
                x_d, y_d, _, _, _, _, _ = self._stack_batch(transitions)
                with torch.no_grad():
                    mixed_dist = self._mixed_policy_dist(x_d, y_d, n=self.rl_cfg.inner_unroll_n)
                    target_probs = mixed_dist.probs.detach()

                self.policy_model_old.train()
                self.old_policy_distill_opt.zero_grad()
                old_dist = self.policy_model_old.policy_dist(x_d, y_d, n=self.rl_cfg.inner_unroll_n)
                log_probs_old = old_dist.logits.log_softmax(dim=-1)
                kl = (
                    target_probs
                    * (target_probs.clamp_min(1e-8).log() - log_probs_old)
                ).sum(dim=-1).mean()
                kl.backward()

                policy_grad_clip = getattr(self.rl_cfg, "policy_grad_clip", None)
                if policy_grad_clip is not None and policy_grad_clip > 0 and self._old_policy_params:
                    nn_utils.clip_grad_norm_(self._old_policy_params, policy_grad_clip)
                self.old_policy_distill_opt.step()
        
        if not distill_enabled:
            # Use parameter-space interpolation only when we are not distilling the mixture.
            self._sync_policy_old_towards_candidate()

        result = {"loss_policy": float(loss_policy.item())}
        if kl_div is not None:
            result["policy_kl"] = float(kl_div.item())
            result["kl_coef"] = self._kl_coef
        return result

    def train_step(self) -> Dict[str, float]:
        """
        One outer training step: collect data, then run value and policy updates.
        """

        for _ in range(self.rl_cfg.rollout_episodes_per_step):
            self.collect_episode()

        loss_val = self.value_update()
        policy_result = self.policy_update()
        
        # Handle both old (float) and new (dict) return types for backward compatibility
        if isinstance(policy_result, dict):
            loss_policy = policy_result.get("loss_policy", 0.0)
        else:
            loss_policy = policy_result
            policy_result = {}

        debug_metrics: Dict[str, float] = {}
        if self.rl_cfg.debug_checks and len(self.replay) >= self.rl_cfg.batch_size:
            with torch.no_grad():
                transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
                x_b, y_b, x_next_b, y_next_b, _, rewards_b, dones_b = self._stack_batch(transitions)
                v_s = self.model.used_value(x_b, y_b, n=self.rl_cfg.inner_unroll_n)
                v_next = self.model.used_value(x_next_b, y_next_b, n=self.rl_cfg.inner_unroll_n)
                mask = (~dones_b).float()
                td_target = rewards_b + self.rl_cfg.gamma * v_next * mask
                adv = td_target - v_s
                debug_metrics["value_mean"] = float(v_s.mean().item())
                debug_metrics["value_std"] = float(v_s.std(unbiased=False).item())
                debug_metrics["adv_mean"] = float(adv.mean().item())
                debug_metrics["adv_std"] = float(adv.std(unbiased=False).item())
                debug_metrics["adv_abs_mean"] = float(adv.abs().mean().item())
                debug_metrics["adv_abs_max"] = float(adv.abs().max().item())

        # Compute theory metrics (C_z, L_z, L_v, unrolling term) if enabled
        theory_metrics: Dict[str, float] = {}
        if getattr(self.rl_cfg, "track_theory_metrics", False) and len(self.replay) >= self.rl_cfg.batch_size:
            theory_metrics = self._compute_theory_metrics()

        metrics = {
            "loss_value": loss_val,
            "loss_policy": loss_policy,
            "term_stop": float(self.term_stats["stop"]),
            "term_solved": float(self.term_stats["solved"]),
            "term_budget": float(self.term_stats["budget"]),
        }
        metrics.update(debug_metrics)
        metrics.update(theory_metrics)
        
        # Include policy KL metrics if present (from KL trust region)
        for key in ("policy_kl", "kl_coef", "kl_early_stop"):
            if key in policy_result:
                metrics[key] = policy_result[key]
        
        # Reset termination stats for the next logging window
        self.term_stats = {"stop": 0.0, "solved": 0.0, "budget": 0.0}
        
        return metrics

    def _compute_theory_metrics(self) -> Dict[str, float]:
        """
        Compute theory-related metrics from the paper:
        - hat_Cz: Estimated C_z = max ||z^(1) - z^(0)||
        - hat_Lz: Estimated local Lipschitz constant of inner map
        - hat_Lv: Estimated Lipschitz constant of value head w.r.t. z
        - bellman_residual_*: Empirical Bellman residual statistics
        - unrolling_term: L_V * L_z^n * C_z / (1 - L_z)
        """
        metrics: Dict[str, float] = {}
        
        try:
            with torch.no_grad():
                transitions = self.replay.sample_batch(min(self.rl_cfg.batch_size, len(self.replay)))
                x_b, y_b, x_next_b, y_next_b, _, rewards_b, dones_b = self._stack_batch(transitions)
                
                # Estimate C_z
                hat_Cz = estimate_Cz(self.model, x_b, y_b)
                metrics["hat_Cz"] = hat_Cz
                
                # Estimate local L_z
                z_n = self.model.eval_latent(x_b, y_b, n=self.rl_cfg.inner_unroll_n)
                batch = self.model._standardize_latent_batch(x_b, y_b)
                context = self.model._build_latent_context_with_plan(batch)
                hat_Lz = estimate_local_Lz(self.model.inner, z_n, context, num_samples=4)
                metrics["hat_Lz"] = hat_Lz
                
                # Estimate L_v if value head exists
                if self.model.value_head is not None:
                    z_vec = z_n.z_H.mean(dim=1)
                    input_embeddings = context["input_embeddings"]
                    plan_embeddings = context["plan_embeddings"]
                    x_embed = self.model._pool_embedding(input_embeddings)
                    y_embed = self.model._pool_embedding(plan_embeddings)
                    combined_embed = torch.cat([x_embed, y_embed], dim=-1)
                    hat_Lv = estimate_Lv(self.model.value_head, z_vec, combined_embed, num_samples=4)
                    metrics["hat_Lv"] = hat_Lv
                    
                    # Compute unrolling term proxy
                    n = self.rl_cfg.inner_unroll_n
                    unrolling_term = compute_unrolling_term_proxy(hat_Lv, hat_Lz, hat_Cz, n)
                    metrics["unrolling_term"] = unrolling_term
                
                # Compute Bellman residual
                v_s = self.model.used_value(x_b, y_b, n=self.rl_cfg.inner_unroll_n)
                v_next = self.model.used_value(x_next_b, y_next_b, n=self.rl_cfg.inner_unroll_n)
                residual_metrics = compute_empirical_bellman_residual(
                    v_s, rewards_b, v_next, dones_b, self.rl_cfg.gamma
                )
                metrics.update(residual_metrics)
                
        except Exception as e:
            if self.debug_checks:
                print(f"[warning] Theory metrics computation failed: {e}")
        
        return metrics

    def evaluate_policy_metrics(self, env_cfg: PlanEditEnvConfig, dataset: Any, checker: Any) -> Dict[str, float]:
        """
        Evaluate the deployed policy, returning both strict success rate and mean checker score.
        """

        mean_score, success_rate = evaluate_plan_policy_with_scores(
            model=self.policy_model_old,
            dataset=dataset,
            checker=checker,
            env_cfg=env_cfg,
            num_episodes=self.rl_cfg.eval_num_episodes,
            inner_unroll_n=self.rl_cfg.inner_unroll_n,
        )

        return {"mean_score": mean_score, "success_rate": success_rate}

    def evaluate_policy_success_rate(self, env_cfg: PlanEditEnvConfig, dataset: Any, checker: Any) -> float:
        """
        Backwards-compatible alias returning only the strict success rate.
        """

        metrics = self.evaluate_policy_metrics(env_cfg=env_cfg, dataset=dataset, checker=checker)
        return metrics["success_rate"]

