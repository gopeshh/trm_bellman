import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)

# Numerical stability constant: minimum log probability to prevent -inf
# exp(-20) ≈ 2e-9, effectively zero probability
LOG_PROB_MIN = -20.0
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils


def _clip_and_recenter_advantages(
    advantages: torch.Tensor,
    policy_probs: torch.Tensor,
    action_mask: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    """Clip per-action advantages without losing statewise centering."""

    clipped = torch.where(
        action_mask,
        advantages.clamp(-clip_value, clip_value),
        torch.zeros_like(advantages),
    )
    clipped_mean = (policy_probs * clipped).sum(dim=-1, keepdim=True)
    return torch.where(
        action_mask,
        clipped - clipped_mean,
        torch.zeros_like(clipped),
    )


def _mean_categorical_kl(
    reference_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    candidate_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(reference || candidate) without multiplying zero by infinity.

    A categorical action mask represents excluded actions with zero reference
    probability and ``-inf`` candidate log probability. Mask both log tensors
    before subtraction so excluded actions contribute exactly zero and have no
    gradient, instead of forming ``0 * inf``.
    """

    reference_support = reference_probs > 0
    safe_reference_log_probs = reference_log_probs.masked_fill(
        ~reference_support,
        0.0,
    )
    safe_candidate_log_probs = candidate_log_probs.masked_fill(
        ~reference_support,
        0.0,
    )
    per_action_kl = reference_probs * (
        safe_reference_log_probs - safe_candidate_log_probs
    )
    mean_kl = per_action_kl.sum(dim=-1).mean()
    if torch.isnan(mean_kl):
        raise FloatingPointError("Categorical KL produced NaN on reference support")
    return mean_kl


# Lazy import to break circular dependency with evaluators
if TYPE_CHECKING:
    from rl.evaluator import evaluate_plan_policy, evaluate_plan_policy_with_scores

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan, normalize_puzzle_id
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.replay import ReplayBuffer, ReplayLatent, Transition, validate_transition
from rl.value_targets import (
    compute_k_step_bootstrapped_target,
    compute_gae,
    compute_empirical_bellman_residual,
)
from utils.lipschitz import (
    estimate_local_Lz,
    estimate_Cz,
    estimate_Lv,
    compute_unrolling_term_proxy,
    # Theory-exact components (Sections 4.2, 5.4 of paper)
    estimate_Cdrift,
    estimate_plan_change,
    compute_value_of_memory_residual,
    compute_exact_baseline_summation,
    compute_exact_advantage,
    # Periodic operator-norm clamping for contraction enforcement
    apply_opnorm_clamp_periodically,
)
from utils.compute_accounting import (
    COMPUTE_SNAPSHOT_SCHEMA_VERSION,
    MODEL_COUNTER_DEFINITIONS,
    OPTIMIZER_STEP_FIELDS,
    add_model_counters,
    aggregate_model_compute,
    capture_model_compute_state,
    current_cuda_memory_peaks,
    process_peak_rss_bytes,
    restore_model_compute_state,
    subtract_model_counters,
    validate_compute_snapshot,
    validate_model_compute_state,
    validate_model_counters,
    zero_model_counters,
)


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
        self._fixed_base_exact = (
            getattr(self.rl_cfg, "training_protocol", "legacy")
            == "fixed_base_exact"
        )
        self._capture_preinterpolation_pair = bool(
            getattr(self.rl_cfg, "capture_preinterpolation_policy_pair", False)
        )
        self._evaluation_policy_mode = str(
            getattr(self.rl_cfg, "evaluation_policy_mode", "configured")
        )
        if self._evaluation_policy_mode == "preinterpolation_exact_mixture":
            if not self._capture_preinterpolation_pair:
                raise ValueError(
                    "preinterpolation_exact_mixture evaluation requires "
                    "capture_preinterpolation_policy_pair=True."
                )
            if self._fixed_base_exact or self.rl_cfg.theory_exact_mixture:
                raise ValueError(
                    "preinterpolation_exact_mixture is a legacy parameter-"
                    "interpolation bridge evaluator, not a fixed-base training mode."
                )
            if self.rl_cfg.policy_epsilon != 0.0:
                raise ValueError(
                    "preinterpolation_exact_mixture requires policy_epsilon=0."
                )
        if self._capture_preinterpolation_pair and (
            self._fixed_base_exact
            or self.rl_cfg.theory_exact_mixture
            or self.rl_cfg.distill_mixture_policy
        ):
            raise ValueError(
                "Pre-interpolation pair capture requires legacy parameter "
                "interpolation training."
            )
        if self._fixed_base_exact and not self.rl_cfg.is_fixed_base_proposal_exact():
            raise ValueError(
                "training_protocol='fixed_base_exact' requires exact K-step "
                "targets, exact baseline summation, exact mixture deployment, "
                "zero policy epsilon, no distillation, and no scheduled "
                "operator-norm clamping."
            )
        assert math.isclose(
            self.env_config.gamma,
            self.rl_cfg.gamma,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ), f"Env gamma ({self.env_config.gamma}) and RL gamma ({self.rl_cfg.gamma}) must match."
        assert 0.0 < self.rl_cfg.gamma < 1.0, "RLConfig.gamma must be in (0,1) for theory to hold."
        self.device = device
        self.debug_checks = bool(getattr(self.rl_cfg, "debug_checks", False))

        self.replay = ReplayBuffer(capacity=rl_cfg.replay_capacity)
        self.term_stats = {"stop": 0.0, "solved": 0.0, "budget": 0.0}
        
        # Debug tracking for RL behavior analysis
        self._debug_episode_lengths: List[int] = []
        self._debug_episode_returns: List[float] = []
        self._debug_stop_probs: List[float] = []
        self._debug_score_changes: List[float] = []

        # Profiling flag (set True to see per-episode timing breakdown)
        self._profile_rollout: bool = False
        
        # === NEW: Theory-exact tracking (Sections 4.2, 5.4 of paper) ===
        # Track drift for persistent latents (Lemma 4.4)
        self._drift_values: List[float] = []
        # Track plan changes Δy (Assumption 4.3)
        self._plan_changes: List[float] = []
        # Track value of memory (Remark 5.5)
        self._value_of_memory: List[float] = []
        # Store checker function reference for exact baseline computation
        self._checker_fn = None  # Set by caller if using exact_baseline_summation
        # Track if we've warned about opnorm clamp failures (to print only one warning)
        self._opnorm_clamp_warned: bool = False
        self._train_step_active: bool = False

        self.target_model = TinyRecursiveReasoningModel_ACTV1(self._config_to_dict(self.model.config)).to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad_(False)
        self._hard_update_target()

        # Policy models:
        # - policy_model_old: deployed policy (data collection)
        # - policy_model_candidate: receives policy-gradient updates
        if getattr(self.rl_cfg, "theory_exact_mixture", False):
            # The critic is updated before each candidate step. Keep the base
            # actor in a separate frozen module so that those value updates do
            # not silently change the policy or recurrent transition map.
            self.policy_model_old = TinyRecursiveReasoningModel_ACTV1(
                self._config_to_dict(self.model.config)
            ).to(device)
            self.policy_model_old.load_state_dict(self.model.state_dict())
        else:
            self.policy_model_old = self.model
        self.policy_model_candidate = TinyRecursiveReasoningModel_ACTV1(
            self._config_to_dict(self.model.config)
        ).to(device)
        self.policy_model_candidate.load_state_dict(self.model.state_dict())

        if self._fixed_base_exact:
            if self.model.value_head is None:
                raise ValueError("fixed_base_exact requires an enabled value head.")
            for name, param in self.model.named_parameters():
                param.requires_grad_(name.startswith("value_head."))
            for buffer in self.model.buffers():
                if buffer.requires_grad:
                    buffer.requires_grad_(False)

        value_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "edit_policy" in name:
                continue
            if self._fixed_base_exact and not name.startswith("value_head."):
                continue
            value_params.append(param)

        self._freeze_policy_backbone()
        self._sync_candidate_backbone_from_model()

        self.preinterpolation_policy_base: Optional[
            TinyRecursiveReasoningModel_ACTV1
        ] = None
        self.preinterpolation_policy_candidate: Optional[
            TinyRecursiveReasoningModel_ACTV1
        ] = None
        self._preinterpolation_pair_generation = 0
        if self._capture_preinterpolation_pair:
            self.preinterpolation_policy_base = copy.deepcopy(
                self.policy_model_old
            ).to(device)
            self.preinterpolation_policy_candidate = copy.deepcopy(
                self.policy_model_candidate
            ).to(device)
            for snapshot in (
                self.preinterpolation_policy_base,
                self.preinterpolation_policy_candidate,
            ):
                snapshot.eval()
                for parameter in snapshot.parameters():
                    parameter.requires_grad_(False)

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
        self._train_step_count: int = 0  # Completed outer rollout/update boundaries
        self._env_step_count: int = 0
        self._value_optimizer_step_count: int = 0
        self._policy_optimizer_step_count: int = 0
        self._distill_optimizer_step_count: int = 0
        self._puzzle_optimizer_step_count: int = 0
        self._training_wall_time_seconds = 0.0
        self._evaluation_wall_time_seconds = 0.0
        self._training_model_work = zero_model_counters()
        self._evaluation_model_work = zero_model_counters()
        self._peak_process_rss_bytes = process_peak_rss_bytes()
        cuda_allocated, cuda_reserved = current_cuda_memory_peaks(self.device)
        self._peak_cuda_allocated_bytes = cuda_allocated
        self._peak_cuda_reserved_bytes = cuda_reserved
        # Exact-budget collection may pause in the middle of an episode.  Keep
        # all transition-relevant collector state until a real environment
        # terminal is observed; a logging/checkpoint boundary is not terminal.
        self._active_episode: Optional[Dict[str, Any]] = None
        self._completed_episodes_since_update: int = 0
        self.puzzle_emb_optimizer: Optional[torch.optim.Optimizer] = None

        # Learning rate schedulers
        self.value_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.policy_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._setup_lr_schedulers()

        # Adaptive KL coefficient for trust-region updates (PPO-style)
        # This coefficient is adjusted based on the KL divergence from old to new policy
        self._kl_coef: float = 1.0
        self._kl_target: float = getattr(self.rl_cfg, "trust_region_kl", 0.01)

    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        if hasattr(config, "model_dump"):
            return config.model_dump()
        return config.dict()

    def _setup_lr_schedulers(self) -> None:
        """
        Set up learning rate schedulers based on rl_cfg.lr_schedule.
        Supports: "constant", "cosine", "linear"
        """
        schedule_type = getattr(self.rl_cfg, "lr_schedule", "constant")
        warmup_steps = getattr(self.rl_cfg, "lr_warmup_steps", 500)
        min_factor = getattr(self.rl_cfg, "lr_min_factor", 0.1)
        total_steps = self.rl_cfg.num_train_steps

        if schedule_type == "constant":
            return  # No scheduler needed

        def make_lr_lambda(warmup: int, total: int, min_lr_factor: float, schedule: str):
            def lr_lambda(step: int) -> float:
                # Warmup phase
                if step < warmup:
                    return (step + 1) / max(warmup, 1)
                
                # Decay phase
                progress = (step - warmup) / max(total - warmup, 1)
                progress = min(progress, 1.0)
                
                if schedule == "cosine":
                    # Cosine annealing from 1.0 to min_lr_factor
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    return min_lr_factor + (1 - min_lr_factor) * cosine_decay
                elif schedule == "linear":
                    # Linear decay from 1.0 to min_lr_factor
                    return 1.0 - progress * (1 - min_lr_factor)
                else:
                    return 1.0
            return lr_lambda

        lr_lambda = make_lr_lambda(warmup_steps, total_steps, min_factor, schedule_type)
        
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.value_opt, lr_lambda=lr_lambda
        )
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.policy_opt, lr_lambda=lr_lambda
        )

    def _step_lr_schedulers(
        self,
        *,
        value_optimizer_stepped: bool,
        policy_optimizer_stepped: bool,
    ) -> None:
        """Advance each scheduler only with its corresponding optimizer."""
        if value_optimizer_stepped and self.value_scheduler is not None:
            self.value_scheduler.step()
        if policy_optimizer_stepped and self.policy_scheduler is not None:
            self.policy_scheduler.step()

    def get_current_lr(self) -> Dict[str, float]:
        """Get current learning rates for logging."""
        return {
            "value_lr": self.value_opt.param_groups[0]["lr"],
            "policy_lr": self.policy_opt.param_groups[0]["lr"],
        }

    def _hard_update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self) -> None:
        tau = self.rl_cfg.target_ema_tau
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target_model.parameters()):
                p_targ.data.mul_(tau).add_(p.data, alpha=1 - tau)
            parameter_names = {name for name, _ in self.model.named_parameters()}
            source_state = self.model.state_dict()
            target_state = self.target_model.state_dict()
            for name, source_value in source_state.items():
                if name in parameter_names or name not in target_state:
                    continue
                # Persistent buffers, including sparse puzzle embeddings, are
                # part of the evaluator state. Copy them exactly.
                target_state[name].copy_(source_value)

    def _maybe_apply_scheduled_opnorm_clamp(
        self,
        train_step_number: Optional[int],
    ) -> None:
        """Apply scheduled clamping inside the evaluator-update phase."""

        if train_step_number is None:
            return
        clamp_interval = getattr(self.rl_cfg, "opnorm_clamp_interval", 0)
        if (
            self._fixed_base_exact
            and getattr(self.rl_cfg, "enable_contraction", False)
            and clamp_interval > 0
        ):
            raise RuntimeError(
                "Scheduled operator-norm clamping would mutate the fixed base map."
            )
        if (
            not getattr(self.rl_cfg, "enable_contraction", False)
            or clamp_interval <= 0
            or train_step_number % clamp_interval != 0
        ):
            return

        try:
            max_norm = getattr(self.rl_cfg, "opnorm_clamp_max_norm", 1.0)
            num_iters = getattr(self.rl_cfg, "opnorm_clamp_num_power_iters", 10)
            log_sigma = getattr(self.rl_cfg, "opnorm_log_max_sigma", False)
            with torch.no_grad():
                sigma_dict = apply_opnorm_clamp_periodically(
                    self.model.inner,
                    per_layer_max=max_norm,
                    num_power_iters=num_iters,
                    restrict_to_reasoning_layers=True,
                )
            if log_sigma and sigma_dict:
                max_sigma = max(sigma_dict.values())
                logger.info(
                    "[step %05d] opnorm clamp applied (max_sigma=%.2f)",
                    train_step_number,
                    max_sigma,
                )
            else:
                logger.info(
                    "[step %05d] opnorm clamp applied",
                    train_step_number,
                )
        except Exception as exc:
            if not self._opnorm_clamp_warned:
                logger.warning(
                    "Periodic opnorm clamp failed (step %d): %s",
                    train_step_number,
                    exc,
                )
                self._opnorm_clamp_warned = True

    def _probability_mixture_dist(
        self,
        base_model: TinyRecursiveReasoningModel_ACTV1,
        candidate_model: TinyRecursiveReasoningModel_ACTV1,
        x_batch,
        y_batch,
        n: int,
        action_mask: Optional[torch.Tensor] = None,
        z=None,
    ):
        """
        Return the exact deployed proposal distribution.

        This corresponds to pi_alpha = (1-alpha) * pi_old + alpha *
        pi_candidate, optionally mixed with a small uniform component. Fixed-base
        training data are collected separately from pi_old by
        ``_collection_policy_dist``.
        This callback always returns the probability-space mixture. Legacy
        parameter-interpolation and distillation paths are separate deployment
        mechanisms and must not be identified with this distribution.
        
        IMPORTANT INVARIANT: This function assumes that policy_model_old and
        policy_model_candidate have **identical backbone weights** (all non-edit_policy
        parameters). This ensures both policies compute embeddings and latent states
        using the same representation, which is required for proper CPI mixture semantics.
        The invariant is maintained by calling _sync_candidate_backbone_from_model()
        at the start of each train_step() before any episode collection.
        
        Args:
            action_mask: Optional [B, action_dim] or [action_dim] boolean mask.
                         True = valid action, False = invalid (will be masked out).
            z: Optional latent state for persistent mode. If None, uses episodic mode.
            
        Returns:
            Tuple of (distribution, z_n) where z_n is the updated latent state.
        """

        alpha = self.rl_cfg.mixture_alpha
        dist_old, z_old = base_model.policy_dist(
            x_batch, y_batch, n=n, action_mask=action_mask, z=z
        )
        
        if z is not None:
            # Persistent mode: evaluate candidate at the same final latent state z_old
            # to ensure both policies produce distributions at the same state for proper
            # mixture semantics. Use n=0 to avoid running additional latent steps.
            dist_new, _ = candidate_model.policy_dist(
                x_batch, y_batch, n=0, action_mask=action_mask, z=z_old
            )
            # Use z_old as the updated latent (follows old policy's trajectory which
            # dominates the mixture with weight 1-alpha)
            z_new = z_old
        else:
            # Episodic mode: both models initialize fresh (no shared latent trajectory)
            dist_new, z_new = candidate_model.policy_dist(
                x_batch, y_batch, n=n, action_mask=action_mask, z=z
            )

        probs_old = dist_old.probs
        probs_new = dist_new.probs
        probs_mix = (1.0 - alpha) * probs_old + alpha * probs_new

        eps = getattr(self.rl_cfg, "policy_epsilon", 0.0)
        if eps > 0.0:
            num_actions = probs_mix.shape[-1]
            # For epsilon-greedy, only consider valid actions for uniform exploration
            if action_mask is not None:
                # Expand mask to batch size if needed
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0).expand(probs_mix.shape[0], -1)
                # Uniform over valid actions only
                valid_count = action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
                uniform = action_mask.float() / valid_count
            else:
                uniform = torch.full_like(probs_mix, 1.0 / num_actions)
            probs_mix = (1.0 - eps) * probs_mix + eps * uniform

        # Return both distribution and updated z (use candidate's z for persistent mode)
        return torch.distributions.Categorical(probs=probs_mix), z_new

    def _mixed_policy_dist(
        self,
        x_batch,
        y_batch,
        n: int,
        action_mask: Optional[torch.Tensor] = None,
        z=None,
    ):
        return UPITrmTrainer._probability_mixture_dist(
            self,
            self.policy_model_old,
            self.policy_model_candidate,
            x_batch,
            y_batch,
            n=n,
            action_mask=action_mask,
            z=z,
        )

    def _preinterpolation_mixed_policy_dist(
        self,
        x_batch,
        y_batch,
        n: int,
        action_mask: Optional[torch.Tensor] = None,
        z=None,
    ):
        if (
            self.preinterpolation_policy_base is None
            or self.preinterpolation_policy_candidate is None
        ):
            raise RuntimeError("No pre-interpolation policy pair was captured.")
        return UPITrmTrainer._probability_mixture_dist(
            self,
            self.preinterpolation_policy_base,
            self.preinterpolation_policy_candidate,
            x_batch,
            y_batch,
            n=n,
            action_mask=action_mask,
            z=z,
        )

    def _capture_current_preinterpolation_pair(self) -> None:
        if not self._capture_preinterpolation_pair:
            return
        if (
            self.preinterpolation_policy_base is None
            or self.preinterpolation_policy_candidate is None
        ):
            raise RuntimeError("Pre-interpolation policy snapshots are unavailable.")
        self.preinterpolation_policy_base.load_state_dict(
            self.policy_model_old.state_dict()
        )
        self.preinterpolation_policy_candidate.load_state_dict(
            self.policy_model_candidate.state_dict()
        )
        self.preinterpolation_policy_base.eval()
        self.preinterpolation_policy_candidate.eval()
        self._preinterpolation_pair_generation += 1

    def _collection_policy_dist(
        self,
        x_batch,
        y_batch,
        n: int,
        action_mask: Optional[torch.Tensor] = None,
        z=None,
    ):
        """Return the policy used to generate replay transitions."""

        if self._fixed_base_exact:
            return self.policy_model_old.policy_dist(
                x_batch,
                y_batch,
                n=n,
                action_mask=action_mask,
                z=z,
            )
        return self._mixed_policy_dist(
            x_batch,
            y_batch,
            n=n,
            action_mask=action_mask,
            z=z,
        )

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

        if self._fixed_base_exact:
            raise RuntimeError("The fixed base policy cannot be interpolated or promoted.")

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
        Configure requires_grad for policy-gradient updates.
        
        - policy_model_candidate: Freeze backbone (non-edit_policy), only edit_policy trains.
        - policy_model_old (when aliased to self.model): Do NOT freeze backbone because
          self.model also serves as the value function (critic) which needs trainable
          backbone params. Only ensure edit_policy is trainable.
        
        NOTE: value_params are collected from self.model BEFORE this method is called,
        but that's intentional - we preserve requires_grad for value function params.
        """

        # Candidate policy (actor): freeze everything except edit_policy head
        if self.policy_model_candidate.edit_policy is not None:
            for name, param in self.policy_model_candidate.named_parameters():
                is_policy_head = name.startswith("edit_policy")
                param.requires_grad_(is_policy_head)

        # Old/deployed policy model
        if self.policy_model_old.edit_policy is not None:
            exact_fixed_base = bool(
                getattr(self.rl_cfg, "theory_exact_mixture", False)
            )
            for name, param in self.policy_model_old.named_parameters():
                is_policy_head = name.startswith("edit_policy")
                if exact_fixed_base:
                    param.requires_grad_(False)
                elif self.policy_model_old is self.model:
                    # CRITICAL: self.model is shared between value function and deployed policy.
                    # Do NOT freeze backbone params - they're needed for value function training.
                    # Only ensure edit_policy head is trainable for policy distillation.
                    if is_policy_head:
                        param.requires_grad_(True)
                    # Non-edit_policy params intentionally LEFT UNCHANGED (remain trainable for value fn)
                else:
                    # Separate policy model: freeze backbone, only edit_policy trainable
                    param.requires_grad_(param.requires_grad and is_policy_head)

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

        source_model = (
            self.policy_model_old
            if getattr(self.rl_cfg, "theory_exact_mixture", False)
            else self.model
        )
        self._copy_nonpolicy_state(source_model, self.policy_model_candidate)

    def _copy_nonpolicy_state(
        self,
        source_model: TinyRecursiveReasoningModel_ACTV1,
        target_model: TinyRecursiveReasoningModel_ACTV1,
    ) -> None:
        """Copy recurrent/evaluator state while preserving the target policy head."""

        with torch.no_grad():
            source_state = source_model.state_dict()
            target_state = target_model.state_dict()
            for name, source_value in source_state.items():
                if name.startswith("edit_policy") or name not in target_state:
                    continue
                target_state[name].copy_(source_value)

    def _sync_exact_policy_snapshot_from_model(self) -> None:
        """Synchronize evaluator state for the selected training protocol."""

        if not getattr(self.rl_cfg, "theory_exact_mixture", False):
            return
        if self._fixed_base_exact:
            self._copy_value_head_state(self.model, self.policy_model_old)
            self._copy_value_head_state(self.model, self.policy_model_candidate)
            return
        self._copy_nonpolicy_state(self.model, self.policy_model_old)
        self._copy_nonpolicy_state(self.model, self.policy_model_candidate)

    def _copy_value_head_state(
        self,
        source_model: TinyRecursiveReasoningModel_ACTV1,
        target_model: TinyRecursiveReasoningModel_ACTV1,
    ) -> None:
        """Copy the policy-independent value head without changing actor state."""

        if source_model.value_head is None or target_model.value_head is None:
            raise RuntimeError("Fixed-base value-head synchronization requires value heads.")
        target_model.value_head.load_state_dict(source_model.value_head.state_dict())

    def _sync_candidate_policy_from_old(self) -> None:
        """Start the next candidate search from the newly deployed policy head."""

        if (
            self.policy_model_old.edit_policy is None
            or self.policy_model_candidate.edit_policy is None
        ):
            return
        self.policy_model_candidate.edit_policy.load_state_dict(
            self.policy_model_old.edit_policy.state_dict()
        )

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
            v_orig, _ = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
            v_perm, _ = self.model.used_value(x_batch, y_shuffled, n=self.rl_cfg.inner_unroll_n)
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
                local_lz_result = estimate_local_Lz(
                    self.model.inner, carry, context, num_samples=2, return_samples=True
                )
                if not isinstance(local_lz_result, tuple):
                    raise RuntimeError(
                        "estimate_local_Lz(return_samples=True) did not return samples."
                    )
                est, samples = local_lz_result
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

    def _start_episode(self) -> None:
        """Initialize one live episode without incrementing completion counters."""

        if self._active_episode is not None:
            raise RuntimeError("Cannot start a second episode while one is active.")
        x, y = self.env.reset()
        episodic_latent = bool(getattr(self.rl_cfg, "episodic_latent", True))
        latent = None
        if not episodic_latent:
            batched = self._state_is_batched(x)
            batch_x = self._prepare_batch_x(x, batched=batched)
            batch_y = self._prepare_plan(y, batched=batched)
            with torch.no_grad():
                latent_model = (
                    self.policy_model_old if self._fixed_base_exact else self.model
                )
                latent = latent_model.init_latent(batch_x, batch_y)

        self._active_episode = {
            "episode_id": self._next_episode_id,
            "timestep": 0,
            "latent": latent,
            "episode_rewards": [],
            "episode_actions": [],
            "initial_score": None,
            "last_info": None,
            "time_prep": 0.0,
            "time_policy": 0.0,
            "time_env_step": 0.0,
        }

        if self.debug_checks and self._next_episode_id == 0:
            mask = self.env.get_action_mask()
            if mask is not None:
                valid_count = mask.sum().item()
                print(
                    f"[DEBUG] Action mask: {valid_count} valid actions out of "
                    f"{len(mask)} total"
                )
            mode = "episodic" if episodic_latent else "persistent"
            print(f"[DEBUG] Latent mode: {mode}")

    def collection_checkpoint_state(self) -> Dict[str, Any]:
        """Serialize collector-only state for exact mid-episode continuation."""

        self._validate_active_episode_consistency()
        active_state = None
        if self._active_episode is not None:
            active_state = {
                key: value
                for key, value in self._active_episode.items()
                if key != "latent"
            }
            active_state = {
                **active_state,
                "episode_rewards": list(active_state["episode_rewards"]),
                "episode_actions": list(active_state["episode_actions"]),
                "last_info": (
                    dict(active_state["last_info"])
                    if active_state["last_info"] is not None
                    else None
                ),
                "latent": self._clone_latent(self._active_episode["latent"]),
            }
        return {
            "schema_version": 1,
            "completed_episodes_since_update": int(
                self._completed_episodes_since_update
            ),
            "active_episode": active_state,
        }

    @staticmethod
    def _checkpoint_values_equal(left: Any, right: Any) -> bool:
        if torch.is_tensor(left) and torch.is_tensor(right):
            return bool(torch.equal(left.cpu(), right.cpu()))
        if isinstance(left, dict) and isinstance(right, dict):
            return set(left) == set(right) and all(
                UPITrmTrainer._checkpoint_values_equal(left[key], right[key])
                for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                UPITrmTrainer._checkpoint_values_equal(a, b)
                for a, b in zip(left, right)
            )
        return bool(left == right)

    def _validate_active_episode_consistency(self) -> None:
        """Check the replay/environment/latent boundary at a collector pause."""

        active = self._active_episode
        if active is None:
            if self.env.x is not None and not self.env.done:
                raise RuntimeError(
                    "Live nonterminal environment has no UPI active-episode state."
                )
            return
        if self.env.x is None or self.env.y is None or self.env.done:
            raise RuntimeError(
                "UPI active episode requires a live nonterminal environment."
            )
        episode_id = int(active["episode_id"])
        timestep = int(active["timestep"])
        if episode_id != self._next_episode_id or timestep != self.env.step_count:
            raise RuntimeError(
                "UPI active episode disagrees with its episode ID or edit clock."
            )
        if len(active["episode_rewards"]) != timestep or len(
            active["episode_actions"]
        ) != timestep:
            raise RuntimeError("UPI active-episode history length is inconsistent.")
        if timestep == 0:
            return
        if not self.replay.storage:
            raise RuntimeError("UPI active episode has no preceding replay transition.")
        boundary = self.replay.storage[-1]
        if (
            boundary.episode_id != episode_id
            or boundary.timestep != timestep - 1
            or bool(boundary.done.reshape(-1)[0].item())
        ):
            raise RuntimeError("UPI active replay boundary is not contiguous.")
        if not self._checkpoint_values_equal(boundary.x_next, self.env.x) or not self._checkpoint_values_equal(
            boundary.y_next, self.env.y
        ):
            raise RuntimeError(
                "UPI active replay successor does not match the live environment."
            )
        if not bool(getattr(self.rl_cfg, "episodic_latent", True)):
            latent = active.get("latent")
            if latent is None or boundary.next_latent is None:
                raise RuntimeError("Persistent UPI replay boundary is missing a latent.")
            if not torch.equal(boundary.next_latent.z_H, latent.z_H.detach().cpu()):
                raise RuntimeError("Persistent UPI z_H carry is discontinuous.")
            if not torch.equal(boundary.next_latent.z_L, latent.z_L.detach().cpu()):
                raise RuntimeError("Persistent UPI z_L carry is discontinuous.")

    def load_collection_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore collector state after the environment state has been loaded."""

        if not isinstance(state, dict) or int(state.get("schema_version", 0)) != 1:
            raise RuntimeError("Unsupported UPI collector checkpoint schema.")
        completed = int(state.get("completed_episodes_since_update", -1))
        rollout_target = int(self.rl_cfg.rollout_episodes_per_step)
        if completed < 0 or completed >= rollout_target:
            raise RuntimeError(
                "Invalid completed-episode count in UPI collector checkpoint: "
                f"{completed}."
            )
        self._completed_episodes_since_update = completed

        active_state = state.get("active_episode")
        if active_state is None:
            self._active_episode = None
            self._validate_active_episode_consistency()
            return
        if not isinstance(active_state, dict):
            raise RuntimeError("UPI active-episode checkpoint must be a dictionary.")
        if self.env.x is None or self.env.y is None or self.env.done:
            raise RuntimeError(
                "UPI active-episode checkpoint requires a live nonterminal environment."
            )
        episode_id = int(active_state.get("episode_id", -1))
        timestep = int(active_state.get("timestep", -1))
        if episode_id != self._next_episode_id:
            raise RuntimeError(
                "UPI active episode ID does not match next_episode_id "
                f"({episode_id} != {self._next_episode_id})."
            )
        if timestep != self.env.step_count:
            raise RuntimeError(
                "UPI active timestep does not match environment clock "
                f"({timestep} != {self.env.step_count})."
            )
        rewards = list(active_state.get("episode_rewards", []))
        actions = list(active_state.get("episode_actions", []))
        if len(rewards) != timestep or len(actions) != timestep:
            raise RuntimeError(
                "UPI active-episode history length does not match its timestep."
            )

        saved_latent = active_state.get("latent")
        episodic_latent = bool(getattr(self.rl_cfg, "episodic_latent", True))
        if episodic_latent and saved_latent is not None:
            raise RuntimeError("Episodic-latent checkpoint contains a carried latent.")
        if not episodic_latent and not isinstance(saved_latent, ReplayLatent):
            raise RuntimeError("Persistent-latent checkpoint is missing its latent.")
        latent = None
        if saved_latent is not None:
            latent = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=saved_latent.z_H.to(self.device),
                z_L=saved_latent.z_L.to(self.device),
            )

        self._active_episode = {
            "episode_id": episode_id,
            "timestep": timestep,
            "latent": latent,
            "episode_rewards": rewards,
            "episode_actions": actions,
            "initial_score": active_state.get("initial_score"),
            "last_info": (
                dict(active_state["last_info"])
                if active_state.get("last_info") is not None
                else None
            ),
            "time_prep": float(active_state.get("time_prep", 0.0)),
            "time_policy": float(active_state.get("time_policy", 0.0)),
            "time_env_step": float(active_state.get("time_env_step", 0.0)),
        }
        self._validate_active_episode_consistency()

    def _finish_active_episode(self) -> None:
        """Record statistics after, and only after, a real environment terminal."""

        active = self._active_episode
        if active is None or not self.env.done:
            raise RuntimeError("Cannot finish an episode before environment termination.")
        episode_id = int(active["episode_id"])
        timestep = int(active["timestep"])
        last_info = active["last_info"]

        if self.debug_checks and episode_id < 5:
            reason = last_info.get("done_reason", "unknown") if last_info else "unknown"
            print(
                f"[DEBUG] Episode {episode_id} finished: {timestep} steps, "
                f"done=True, reason={reason}",
                flush=True,
            )

        if self._profile_rollout and episode_id < 3:
            prep = float(active["time_prep"])
            policy = float(active["time_policy"])
            env_step = float(active["time_env_step"])
            total = prep + policy + env_step
            if total > 0.0 and timestep > 0:
                print(f"[PROFILE] Episode {episode_id} ({timestep} steps):")
                print(f"  prep:      {prep*1000:7.1f}ms ({100*prep/total:5.1f}%)")
                print(
                    f"  policy:    {policy*1000:7.1f}ms "
                    f"({100*policy/total:5.1f}%)"
                )
                print(
                    f"  env.step:  {env_step*1000:7.1f}ms "
                    f"({100*env_step/total:5.1f}%)"
                )
                print(f"  total:     {total*1000:7.1f}ms ({total/timestep*1000:.2f}ms/step)")

        self._debug_episode_lengths.append(timestep)
        self._debug_episode_returns.append(sum(active["episode_rewards"]))
        if last_info is not None:
            reason = last_info.get("done_reason")
            if reason in self.term_stats:
                self.term_stats[reason] += 1
            initial_score = active["initial_score"]
            final_score = last_info.get("phi_new", initial_score)
            if (
                isinstance(initial_score, (int, float))
                and not isinstance(initial_score, bool)
                and isinstance(final_score, (int, float))
                and not isinstance(final_score, bool)
            ):
                self._debug_score_changes.append(
                    float(final_score) - float(initial_score)
                )

        self._next_episode_id += 1
        self._active_episode = None

    def collect_episode(self, max_env_steps: Optional[int] = None) -> int:
        if self._train_step_active:
            return self._collect_episode_impl(max_env_steps)
        before, _, _ = aggregate_model_compute(self._model_roles())
        started = time.perf_counter()
        try:
            return self._collect_episode_impl(max_env_steps)
        finally:
            self._training_wall_time_seconds += time.perf_counter() - started
            after, _, _ = aggregate_model_compute(self._model_roles())
            self._training_model_work = add_model_counters(
                self._training_model_work,
                subtract_model_counters(after, before),
            )

    def _collect_episode_impl(self, max_env_steps: Optional[int] = None) -> int:
        """Continue one episode, pausing without termination at an exact cap.

        ``max_env_steps`` is a collector budget, not an MDP horizon.  When the
        cap is reached, the plan, edit clock, episode id, and persistent latent
        remain live for the next call.  The return value counts interactions
        collected by this call, not the episode's cumulative length.
        """

        if max_env_steps is not None and max_env_steps < 0:
            raise ValueError("max_env_steps must be non-negative or None.")
        if max_env_steps == 0:
            return 0

        self.model.eval()
        self.policy_model_old.eval()
        self.policy_model_candidate.eval()
        if self._active_episode is None:
            self._start_episode()

        steps_collected = 0
        episodic_latent = bool(getattr(self.rl_cfg, "episodic_latent", True))
        stop_action_id = self.env.stop_action_id
        profile_enabled = bool(self._profile_rollout)

        while max_env_steps is None or steps_collected < max_env_steps:
            active = self._active_episode
            if active is None:
                break
            if self.env.done:
                raise RuntimeError("Active UPI episode has an already-terminal environment.")
            if self.env.step_count >= self.env_config.max_edits:
                raise RuntimeError(
                    "UPI collector reached max_edits without an environment terminal."
                )

            x = self.env.x
            y = self.env.y
            timestep = int(active["timestep"])
            start_prep = time.perf_counter() if profile_enabled else 0.0
            batched = self._state_is_batched(x)
            batch_x = self._prepare_batch_x(x, batched=batched)
            batch_y = self._prepare_plan(y, batched=batched)
            action_mask = self.env.get_action_mask()
            if action_mask is not None:
                action_mask = action_mask.to(self.device)
            start_policy = time.perf_counter() if profile_enabled else 0.0
            if profile_enabled:
                active["time_prep"] += start_policy - start_prep

            if self.debug_checks and active["episode_id"] < 5 and (
                timestep < 5 or timestep % 20 == 0
            ):
                print(
                    f"[DEBUG] Episode {active['episode_id']}, Step {timestep}: "
                    "calling _collection_policy_dist",
                    flush=True,
                )
            latent_before = active["latent"]
            with torch.no_grad():
                dist, latent_after = self._collection_policy_dist(
                    batch_x,
                    batch_y,
                    n=self.rl_cfg.inner_unroll_n,
                    action_mask=action_mask,
                    z=latent_before,
                )
            end_policy = time.perf_counter() if profile_enabled else 0.0
            if profile_enabled:
                active["time_policy"] += end_policy - start_policy

            if not episodic_latent:
                active["latent"] = latent_after
            action = dist.sample().squeeze()
            behavior_log_prob = dist.log_prob(action).detach().cpu().reshape(())

            if timestep == 0 and stop_action_id is not None:
                probs = dist.probs
                stop_prob = (
                    probs[0, stop_action_id].item()
                    if probs.dim() > 1
                    else probs[stop_action_id].item()
                )
                self._debug_stop_probs.append(stop_prob)
                if self.debug_checks and active["episode_id"] == 0:
                    flat_probs = probs[0] if probs.dim() > 1 else probs
                    edit_probs = flat_probs[:-1].sum().item()
                    print(
                        f"[DEBUG] Step 0 probs: STOP={stop_prob:.6f}, "
                        f"edits={edit_probs:.6f}"
                    )
                    print(
                        f"[DEBUG] Top 5 action probs: "
                        f"{flat_probs.topk(min(5, flat_probs.numel()))}"
                    )

            start_env = time.perf_counter() if profile_enabled else 0.0
            (x_next, y_next), reward, done, info = self.env.step(action.item())
            if profile_enabled:
                active["time_env_step"] += time.perf_counter() - start_env

            if self.debug_checks and active["episode_id"] < 5 and (
                timestep < 5 or timestep % 20 == 0
            ):
                print(
                    f"[DEBUG] Episode {active['episode_id']}, Step {timestep} "
                    f"complete: action={action.item()}, reward={reward:.4f}, "
                    f"done={done}",
                    flush=True,
                )

            self._env_step_count += 1
            steps_collected += 1
            active["last_info"] = dict(info)
            if timestep == 0 and info.get("phi_old") is not None:
                active["initial_score"] = info["phi_old"]
            active["episode_rewards"].append(reward)
            active["episode_actions"].append(action.item())

            if getattr(self.rl_cfg, "track_plan_change", False):
                y_tensor = self._prepare_plan(y, batched=False)
                y_next_tensor = self._prepare_plan(y_next, batched=False)
                self._plan_changes.append(
                    estimate_plan_change(y_tensor, y_next_tensor)
                )
            if (
                not episodic_latent
                and getattr(self.rl_cfg, "track_drift_metrics", False)
                and active["latent"] is not None
            ):
                batch_x_next = self._prepare_batch_x(
                    x_next, batched=self._state_is_batched(x_next)
                )
                batch_y_next = self._prepare_plan(
                    y_next, batched=self._state_is_batched(x_next)
                )
                self._drift_values.append(
                    estimate_Cdrift(
                        self.model,
                        batch_x_next,
                        batch_y_next,
                        active["latent"],
                        n=self.rl_cfg.inner_unroll_n,
                    )
                )

            self.replay.add(
                Transition(
                    x=self._clone_state(x),
                    y=self._clone_state(y),
                    action=action.detach().cpu(),
                    reward=torch.as_tensor(reward, dtype=torch.float32).view(1),
                    x_next=self._clone_state(x_next),
                    y_next=self._clone_state(y_next),
                    done=torch.tensor([done], dtype=torch.bool),
                    episode_id=int(active["episode_id"]),
                    timestep=timestep,
                    latent=self._clone_latent(latent_before),
                    next_latent=(
                        self._clone_latent(latent_after)
                        if not episodic_latent
                        else None
                    ),
                    behavior_log_prob=behavior_log_prob,
                )
            )
            active["timestep"] = timestep + 1

            if done:
                self._finish_active_episode()
                break

        return steps_collected

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

        x_batch = self._stack_state_dicts([t.x for t in transitions])
        x_next_batch = self._stack_state_dicts([t.x_next for t in transitions])

        y_batch = torch.stack([self._plan_tensor(t.y) for t in transitions], dim=0).to(self.device)
        y_next_batch = torch.stack([self._plan_tensor(t.y_next) for t in transitions], dim=0).to(self.device)

        actions = torch.stack([t.action for t in transitions], dim=0).to(self.device)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        rewards = torch.stack([t.reward for t in transitions], dim=0).to(self.device).squeeze(-1)
        dones = torch.stack([t.done for t in transitions], dim=0).to(self.device).squeeze(-1)

        return x_batch, y_batch, x_next_batch, y_next_batch, actions, rewards, dones

    def _stack_state_dicts(
        self,
        states: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Stack every tensor field common to a replay-state batch."""

        if not states:
            raise ValueError("Cannot stack an empty state batch.")
        required = {"inputs", "puzzle_identifiers"}
        missing = required - set.intersection(*(set(state) for state in states))
        if missing:
            raise KeyError(f"Replay states are missing required fields: {sorted(missing)}")

        common_keys = set.intersection(*(set(state) for state in states))
        batch: Dict[str, torch.Tensor] = {}
        for key in sorted(common_keys):
            values = [state[key] for state in states]
            if not all(torch.is_tensor(value) for value in values):
                continue
            if key == "puzzle_identifiers":
                stacked = torch.stack(
                    [normalize_puzzle_id(value) for value in values], dim=0
                )
                if stacked.dim() > 1:
                    stacked = stacked.squeeze(-1)
            elif key == "remaining_edits":
                stacked = torch.stack([value.reshape(()) for value in values], dim=0)
            else:
                stacked = torch.stack(values, dim=0)
            batch[key] = stacked.to(self.device)
        return batch

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

    def _clone_latent(
        self,
        latent: Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry],
    ) -> Optional[ReplayLatent]:
        if latent is None:
            return None
        return ReplayLatent(
            z_H=latent.z_H.detach().cpu().clone(),
            z_L=latent.z_L.detach().cpu().clone(),
        )

    def _stack_latents(
        self,
        transitions: List[Transition],
        attribute: str,
    ) -> Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        latents = [getattr(transition, attribute) for transition in transitions]
        return self._stack_optional_latent_list(latents, attribute)

    def _stack_optional_latent_list(
        self,
        latents: List[Optional[ReplayLatent]],
        label: str,
    ) -> Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        if not latents or all(latent is None for latent in latents):
            return None
        concrete_latents: List[ReplayLatent] = []
        for latent in latents:
            if latent is None:
                raise RuntimeError(
                    f"Replay batch mixes transitions with and without `{label}` latent state."
                )
            concrete_latents.append(latent)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.cat([latent.z_H for latent in concrete_latents], dim=0).to(self.device),
            z_L=torch.cat([latent.z_L for latent in concrete_latents], dim=0).to(self.device),
        )

    def _compute_training_action_mask(
        self,
        x_batch: Dict[str, torch.Tensor],
        y_batch: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.env.vocab_size is None or self.env.stop_action_id is None:
            return None
        if self.env.task_config is not None:
            mask = self.env.task_config.compute_batch_action_mask(
                x_batch["inputs"],
                self.env.vocab_size,
                self.env.stop_action_id,
                current_state=y_batch,
            )
        else:
            mask = PlanEditEnv.compute_batch_action_mask(
                x_batch["inputs"],
                self.env.vocab_size,
                self.env.stop_action_id,
                stop_mode=self.env._stop_mode,
            )

        # TaskConfig predates STOP modes and marks STOP valid unconditionally.
        # Enforce the environment contract after task-specific masking.
        mask = mask.clone()
        mask[..., self.env.stop_action_id] = self.env._stop_mode != "disabled"
        return mask

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
        Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry],
        Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry],
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

        horizon = self.rl_cfg.K
        candidate_indices = list(range(len(storage)))
        exact_k_step_targets = bool(
            getattr(self.rl_cfg, "exact_k_step_targets", False)
        )
        if exact_k_step_targets:
            candidate_indices = [
                start_idx
                for start_idx in candidate_indices
                if self.replay.has_complete_segment(
                    start_idx,
                    horizon,
                    require_clock=True,
                )
            ]
            if not candidate_indices:
                raise RuntimeError(
                    "No replay segment contains K transitions or an earlier terminal state."
                )
        sampled_offsets = torch.randint(
            low=0,
            high=len(candidate_indices),
            size=(batch_size,),
        ).tolist()
        indices = [candidate_indices[offset] for offset in sampled_offsets]

        start_states: List[Dict[str, torch.Tensor]] = []
        start_plans: List[torch.Tensor] = []
        start_latents: List[Optional[ReplayLatent]] = []

        end_states: List[Dict[str, torch.Tensor]] = []
        end_plans: List[torch.Tensor] = []
        end_latents: List[Optional[ReplayLatent]] = []

        rewards_K = torch.zeros(batch_size, horizon, dtype=torch.float32, device=self.device)
        dones_K = torch.zeros(batch_size, horizon, dtype=torch.bool, device=self.device)
        steps_taken = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for batch_idx, start_idx in enumerate(indices):
            transition = storage[start_idx]
            start_states.append(transition.x)
            start_plans.append(self._plan_tensor(transition.y))
            start_latents.append(transition.latent)

            if exact_k_step_targets:
                segment = self.replay.contiguous_segment(
                    start_idx,
                    horizon,
                    require_complete=True,
                    require_clock=True,
                )
            else:
                segment = []
                current_idx = start_idx
                while len(segment) < horizon and current_idx < len(storage):
                    current = storage[current_idx]
                    if current.episode_id != transition.episode_id:
                        break
                    segment.append(current)
                    if bool(current.done.view(-1)[0].item()):
                        break
                    current_idx += 1

            last_transition = segment[0]
            for steps, current in enumerate(segment, start=1):
                last_transition = current
                reward_value = float(current.reward.view(-1)[0].item())
                done_value = bool(current.done.view(-1)[0].item())

                rewards_K[batch_idx, steps - 1] = reward_value
                dones_K[batch_idx, steps - 1] = done_value

            end_states.append(last_transition.x_next)
            end_plans.append(self._plan_tensor(last_transition.y_next))
            end_latents.append(last_transition.next_latent)
            steps_taken[batch_idx] = len(segment)

        x_batch = self._stack_state_dicts(start_states)
        y_batch = torch.stack(start_plans, dim=0).to(self.device)

        xK_batch = self._stack_state_dicts(end_states)
        yK_batch = torch.stack(end_plans, dim=0).to(self.device)

        z_batch = self._stack_optional_latent_list(start_latents, "latent")
        zK_batch = self._stack_optional_latent_list(end_latents, "next_latent")
        return (
            x_batch,
            y_batch,
            xK_batch,
            yK_batch,
            rewards_K,
            dones_K,
            steps_taken,
            z_batch,
            zK_batch,
        )

    def _has_complete_k_step_segment(self, start_idx: int, horizon: int) -> bool:
        """Whether a replay start reaches K steps or a terminal transition."""

        return self.replay.has_complete_segment(
            start_idx,
            horizon,
            require_clock=bool(getattr(self.rl_cfg, "exact_k_step_targets", False)),
        )

    def value_update(
        self,
        scheduled_train_step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform one value-function update using either 1-step or K-step bootstrapped targets.
        
        Returns dict with:
        - loss_value: The MSE loss value
        - target_mean/std/min/max: Value target statistics (before clipping)
        - value_mean/std: V(s) statistics
        - reward_mean/std: Reward statistics for the batch
        """

        if len(self.replay) < self.rl_cfg.batch_size:
            return {"loss_value": 0.0, "value_optimizer_step": 0.0}

        debug_batch: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = None
        target_stats: Dict[str, float] = {}
        exact_k_step_targets = bool(
            getattr(self.rl_cfg, "exact_k_step_targets", False)
        )
        bootstrap_model = self.model if exact_k_step_targets else self.target_model
        
        if self.rl_cfg.K == 1:
            transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
            if exact_k_step_targets:
                for transition in transitions:
                    validate_transition(transition, require_clock=True)
            x_batch, y_batch, x_next_batch, y_next_batch, _, rewards, dones = self._stack_batch(transitions)
            z_batch = self._stack_latents(transitions, "latent")
            z_next_batch = self._stack_latents(transitions, "next_latent")
            debug_batch = (x_batch, y_batch)

            self.value_opt.zero_grad()

            bootstrap_model.eval()
            with torch.no_grad():
                v_next, _ = bootstrap_model.used_value(
                    x_next_batch,
                    y_next_batch,
                    n=self.rl_cfg.inner_unroll_n,
                    z=z_next_batch,
                )
                mask = (~dones).float()
                v_next = v_next * mask

            self.model.train()
            v_s, _ = self.model.used_value(
                x_batch,
                y_batch,
                n=self.rl_cfg.inner_unroll_n,
                z=z_batch,
            )
            td_target = rewards + self.rl_cfg.gamma * v_next
            
            # Apply clipping BEFORE capturing stats (so stats show what's actually used)
            value_clip = getattr(self.rl_cfg, "value_target_clip", None)
            if value_clip is not None and value_clip > 0:
                td_target = td_target.clamp(-value_clip, value_clip)
                v_s = v_s.clamp(-value_clip, value_clip)
            
            # Capture stats AFTER clipping for debugging
            target_stats = {
                "target_mean": float(td_target.mean().item()),
                "target_std": float(td_target.std(unbiased=False).item()),
                "target_min": float(td_target.min().item()),
                "target_max": float(td_target.max().item()),
                "value_mean": float(v_s.mean().item()),
                "value_std": float(v_s.std(unbiased=False).item()),
                "reward_mean": float(rewards.mean().item()),
                "reward_std": float(rewards.std(unbiased=False).item()),
                "value_clip": float(value_clip) if value_clip is not None else 0.0,
            }
            
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
                z_batch,
                zK_batch,
            ) = self._sample_k_step_batch(self.rl_cfg.batch_size)
            debug_batch = (x_batch, y_batch)

            self.value_opt.zero_grad()

            bootstrap_model.eval()
            with torch.no_grad():
                gamma = self.rl_cfg.gamma
                K = self.rl_cfg.K

                v_K, _ = bootstrap_model.used_value(
                    xK_batch,
                    yK_batch,
                    n=self.rl_cfg.inner_unroll_n,
                    z=zK_batch,
                )
                G_K = compute_k_step_bootstrapped_target(
                    rewards_K=rewards_K,
                    dones_K=dones_K,
                    steps_taken=steps_taken,
                    v_K=v_K,
                    gamma=gamma,
                    K=K,
                    exact_k_step_targets=bool(getattr(self.rl_cfg, "exact_k_step_targets", False)),
                    C_max=getattr(self.rl_cfg, "C_max", None),  # Paper Eq. 12: V(s_abs) = -C_max
                )

            self.model.train()
            v_s, _ = self.model.used_value(
                x_batch,
                y_batch,
                n=self.rl_cfg.inner_unroll_n,
                z=z_batch,
            )
            
            # Apply clipping BEFORE capturing stats (so stats show what's actually used)
            value_clip = getattr(self.rl_cfg, "value_target_clip", None)
            if value_clip is not None and value_clip > 0:
                G_K = G_K.clamp(-value_clip, value_clip)
                v_s = v_s.clamp(-value_clip, value_clip)
            
            # Capture stats AFTER clipping for debugging
            target_stats = {
                "target_mean": float(G_K.mean().item()),
                "target_std": float(G_K.std(unbiased=False).item()),
                "target_min": float(G_K.min().item()),
                "target_max": float(G_K.max().item()),
                "value_mean": float(v_s.mean().item()),
                "value_std": float(v_s.std(unbiased=False).item()),
                "reward_mean": float(rewards_K.sum(dim=1).mean().item()),
                "reward_std": float(rewards_K.sum(dim=1).std(unbiased=False).item()),
                "value_clip": float(value_clip) if value_clip is not None else 0.0,
            }
            
            loss_val = F.mse_loss(v_s, G_K.detach())

        loss_val.backward()
        value_grad_clip = getattr(self.rl_cfg, "value_grad_clip", None)
        if value_grad_clip is not None and value_grad_clip > 0 and self._value_params:
            nn_utils.clip_grad_norm_(self._value_params, value_grad_clip)
        self.value_opt.step()
        self._value_optimizer_step_count += 1
        if self.puzzle_emb_optimizer is not None:
            # CastedSparseEmbedding stores the active IDs in mutable buffers.
            # Step before any later forward can overwrite those IDs.
            self.puzzle_emb_optimizer.step()
            self._puzzle_optimizer_step_count += 1
            self.puzzle_emb_optimizer.zero_grad()
        # Legacy training may clamp and resnapshot the actor map. The fixed-base
        # protocol rejects scheduled clamping and synchronizes only the value head.
        self._maybe_apply_scheduled_opnorm_clamp(scheduled_train_step)
        self._soft_update_target()
        self._sync_exact_policy_snapshot_from_model()

        if debug_batch is not None:
            self._maybe_run_value_debug_checks(*debug_batch)

        return {
            "loss_value": float(loss_val.item()),
            "value_optimizer_step": 1.0,
            **target_stats,
        }

    def policy_update(self) -> Dict[str, float]:
        """
        Perform one policy-gradient step with optional:
        - GAE advantage estimation (use_gae=True)
        - Batch-level advantage centering (batch_centered_advantage=True)
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
            return {"loss_policy": 0.0, "policy_optimizer_step": 0.0}

        # Ensure the candidate policy uses the current critic backbone for its features.
        self._sync_candidate_backbone_from_model()

        transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
        x_batch, y_batch, x_next_batch, y_next_batch, actions, rewards, dones = self._stack_batch(transitions)
        z_batch = self._stack_latents(transitions, "latent")
        z_next_batch = self._stack_latents(transitions, "next_latent")
        action_mask = self._compute_training_action_mask(x_batch, y_batch)

        self.model.train()
        self.policy_model_candidate.train()
        self.policy_opt.zero_grad()

        with torch.no_grad():
            gamma = self.rl_cfg.gamma
            old_dist, policy_successor_latent = self.policy_model_old.policy_dist(
                x_batch,
                y_batch,
                n=self.rl_cfg.inner_unroll_n,
                action_mask=action_mask,
                z=z_batch,
            )
            old_probs = old_dist.probs
            
            # === Exact baseline computation (Theorem 5.9) ===
            # When exact_baseline_summation=True, compute E_{a ~ π}[Q̂(s,a)] via exact
            # summation over ALL discrete actions. This enables the O(α·ε_A) bound
            # instead of naive O(ε_A) - the key theoretical contribution.
            #
            # FAIL-FAST: If exact_baseline_summation=True but prerequisites are missing,
            # raise an error instead of silently falling back. This prevents accidental
            # misinterpretation of experimental runs as "theory-compatible".
            use_exact_baseline = getattr(self.rl_cfg, "exact_baseline_summation", False)
            persistent_latent = not getattr(self.rl_cfg, "episodic_latent", True)
            
            if use_exact_baseline:
                # Validate prerequisites for exact baseline
                if self._checker_fn is None:
                    raise RuntimeError(
                        "exact_baseline_summation=True requires a checker_fn to be set "
                        "via trainer.set_checker_fn(). Got checker_fn=None. Either set the "
                        "checker function or disable exact_baseline_summation in RLConfig."
                    )
                if self.env is None:
                    raise RuntimeError(
                        "exact_baseline_summation=True requires an env instance to be set. "
                        "Got env=None. This indicates a configuration error in UPITrmTrainer."
                    )
                
                # Exact advantage: Ahat(s,a) = Qhat(s,a) - E_{b~pi}[Qhat(s,b)].
                # Persistent mode conditions on the complete recorded augmented
                # state. The same post-unroll carry is passed to every enumerated
                # edit successor because recurrence precedes action application.
                
                # Ensure action_mask is computed to enable the O(A_valid) loop optimization
                if action_mask is None:
                    vocab_size = self.env.vocab_size
                    stop_action_id = self.env.stop_action_id
                    if vocab_size is None or stop_action_id is None:
                        raise RuntimeError(
                            "Exact baseline summation requires configured vocabulary and STOP action IDs."
                        )
                    action_mask = self.env.compute_batch_action_mask(
                        inputs=x_batch["inputs"],
                        vocab_size=vocab_size,
                        stop_action_id=stop_action_id,
                        stop_mode=self.env._stop_mode,
                    )

                exact_baseline, q_all = compute_exact_baseline_summation(
                    model=self.model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    env=self.env,
                    n=self.rl_cfg.inner_unroll_n,
                    gamma=gamma,
                    checker_fn=self._checker_fn,
                    action_mask=action_mask,
                    policy_probs=old_probs,
                    successor_latent=(
                        policy_successor_latent if persistent_latent else None
                    ),
                )
                advantages_all = torch.where(
                    action_mask,
                    q_all - exact_baseline.unsqueeze(-1),
                    torch.zeros_like(q_all),
                )
                adv_clip = getattr(self.rl_cfg, "advantage_clip", None)
                if adv_clip is not None and adv_clip > 0:
                    advantages_all = _clip_and_recenter_advantages(
                        advantages_all,
                        old_probs,
                        action_mask,
                        adv_clip,
                    )
                batch_indices = torch.arange(actions.shape[0], device=actions.device)
                adv = advantages_all[batch_indices, actions]
            else:
                # Standard learned-baseline advantage (falls back to naive O(ε_A) bound)
                v_s, _ = self.model.used_value(
                    x_batch,
                    y_batch,
                    n=self.rl_cfg.inner_unroll_n,
                    z=z_batch,
                )
                v_next, _ = self.model.used_value(
                    x_next_batch,
                    y_next_batch,
                    n=self.rl_cfg.inner_unroll_n,
                    z=z_next_batch,
                )
                mask = (~dones).float()
                v_next_masked = v_next * mask

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

            if (
                not use_exact_baseline
                and getattr(self.rl_cfg, "batch_centered_advantage", False)
            ):
                # Batch-level centering is a variance-reduction heuristic, not
                # the exact per-state centering used by the CPI theorem.
                adv = adv - adv.mean()

            adv_clip = getattr(self.rl_cfg, "advantage_clip", None)
            if not use_exact_baseline and adv_clip is not None and adv_clip > 0:
                adv = adv.clamp(-adv_clip, adv_clip)

        # Get old policy distribution for KL computation and Importance Sampling
        kl_div = None
        enable_kl_trust_region = getattr(self.rl_cfg, "enable_kl_trust_region", False)
        use_importance_sampling = getattr(self.rl_cfg, "use_importance_sampling", True)
        log_prob_behavior: Optional[torch.Tensor] = None
        old_log_probs: Optional[torch.Tensor] = None

        with torch.no_grad():
            if use_importance_sampling:
                behavior_log_probs: List[torch.Tensor] = []
                for transition in transitions:
                    behavior_log_prob = transition.behavior_log_prob
                    if behavior_log_prob is None:
                        raise RuntimeError(
                            "Importance sampling requires collection-time behavior_log_prob "
                            "on every replay transition."
                        )
                    behavior_log_probs.append(behavior_log_prob.reshape(()))
                log_prob_behavior = torch.stack(behavior_log_probs).to(self.device)

            if enable_kl_trust_region:
                old_log_probs = old_probs.clamp(min=1e-8).log()

        dist, _ = self.policy_model_candidate.policy_dist(
            x_batch,
            y_batch,
            n=self.rl_cfg.inner_unroll_n,
            action_mask=action_mask,
            z=z_batch,
        )
        log_prob = dist.log_prob(actions)
        
        # Numerical stability: clamp log_prob to prevent -inf when action prob is 0
        # This can happen if action mask changed between collection and training
        log_prob = log_prob.clamp(min=LOG_PROB_MIN)
        
        # Importance Sampling Weight: rho = pi_cand(a|s) / pi_behavior(a|s)
        if use_importance_sampling:
            if log_prob_behavior is None:
                raise RuntimeError("Importance-sampling log probabilities were not initialized.")
            log_rho = log_prob - log_prob_behavior
            rho = log_rho.exp()
            rho = rho.detach()
        else:
            rho = torch.ones_like(log_prob)
        
        entropy = dist.entropy()
        # Handle NaN entropy (can happen with degenerate distributions)
        nan_mask = torch.isnan(entropy)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            logger.warning(
                "NaN entropy detected in %d/%d samples. Replacing with 0. "
                "This may indicate degenerate action distributions.",
                nan_count, entropy.numel()
            )
        entropy = torch.where(nan_mask, torch.zeros_like(entropy), entropy)
        entropy = entropy.mean()

        # Skip batch if advantages are all NaN (degenerate case)
        if torch.isnan(adv).all():
            return {
                "loss_policy": 0.0,
                "skipped_nan_adv": 1.0,
                "policy_optimizer_step": 0.0,
            }
        
        # Replace NaN advantages with 0 (neutral gradient)
        adv_clean = torch.where(torch.isnan(adv), torch.zeros_like(adv), adv)
        
        loss_policy = -(rho * log_prob * adv_clean.detach()).mean() - self.rl_cfg.entropy_coef * entropy

        # Add adaptive KL penalty if trust-region is enabled (PPO/TRPO-style)
        if enable_kl_trust_region:
            if old_log_probs is None:
                raise RuntimeError("Trust-region reference probabilities were not initialized.")
            new_log_probs = dist.logits.log_softmax(dim=-1)
            # KL(old || new) = sum(old_probs * (log_old - log_new))
            kl_div = _mean_categorical_kl(
                old_probs,
                old_log_probs,
                new_log_probs,
            )
            
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
                    "policy_optimizer_step": 0.0,
                }
                return result

        loss_policy.backward()
        policy_grad_clip = getattr(self.rl_cfg, "policy_grad_clip", None)
        if policy_grad_clip is not None and policy_grad_clip > 0 and self._policy_params:
            nn_utils.clip_grad_norm_(self._policy_params, policy_grad_clip)
        self.policy_opt.step()
        self._policy_optimizer_step_count += 1
        
        # Update adaptive KL coefficient after successful step
        if enable_kl_trust_region and kl_div is not None:
            self._update_kl_coef(kl_div.item())
        
        # === Policy update modes (Issue 4 - CPI guarantee) ===
        #
        # The CPI bound requires a POLICY-SPACE mixture π_new = (1-α)π_old + α·π_candidate.
        # The code supports three modes with different theory compatibility:
        #
        # 1. fixed_base_exact + theory_exact_mixture=True:
        #    - Keep policy_model_old and its recurrent map frozen
        #    - Collect replay from policy_model_old
        #    - Deploy/evaluate the explicit mixture from _mixed_policy_dist()
        #    - Optimize one candidate proposal without recursive promotion
        #
        # 2. distill_mixture_policy=True (HEURISTIC - Section 6.5):
        #    - Distill the mixture into policy_model_old via KL minimization
        #    - Introduces projection error NOT covered by theory
        #
        # 3. Default (HEURISTIC):
        #    - Parameter-space interpolation: param.lerp_(candidate_param, α)
        #    - NOT equivalent to policy-space mixture after softmax
        #    - The CPI improvement guarantee does NOT strictly apply
        
        theory_exact_mixture = getattr(self.rl_cfg, "theory_exact_mixture", False)
        distill_enabled = getattr(self.rl_cfg, "distill_mixture_policy", False)
        
        if theory_exact_mixture:
            # Both legacy exact-mixture mode and fixed-base mode retain the old
            # head. Only fixed_base_exact also freezes collection and recurrence.
            pass
        elif distill_enabled and self.old_policy_distill_opt is not None:
            # Mode 2: Distill mixture into policy_model_old (heuristic, not theory-exact)
            num_distill = min(len(self.replay), self.rl_cfg.batch_size)
            if num_distill > 0:
                transitions = self.replay.sample_batch(num_distill)
                x_d, y_d, _, _, _, _, _ = self._stack_batch(transitions)
                z_d = self._stack_latents(transitions, "latent")
                mask_d = self._compute_training_action_mask(x_d, y_d)

                with torch.no_grad():
                    mixed_dist, _ = self._mixed_policy_dist(
                        x_d,
                        y_d,
                        n=self.rl_cfg.inner_unroll_n,
                        action_mask=mask_d,
                        z=z_d,
                    )
                    target_probs = mixed_dist.probs.detach()

                self.policy_model_old.train()
                self.old_policy_distill_opt.zero_grad()
                old_dist, _ = self.policy_model_old.policy_dist(
                    x_d,
                    y_d,
                    n=self.rl_cfg.inner_unroll_n,
                    action_mask=mask_d,
                    z=z_d,
                )
                log_probs_old = old_dist.logits.log_softmax(dim=-1)
                kl = _mean_categorical_kl(
                    target_probs,
                    target_probs.clamp_min(1e-8).log(),
                    log_probs_old,
                )
                kl.backward()

                policy_grad_clip = getattr(self.rl_cfg, "policy_grad_clip", None)
                if policy_grad_clip is not None and policy_grad_clip > 0 and self._old_policy_params:
                    nn_utils.clip_grad_norm_(self._old_policy_params, policy_grad_clip)
                self.old_policy_distill_opt.step()
                self._distill_optimizer_step_count += 1
                self._sync_candidate_policy_from_old()
        else:
            # Mode 3: Parameter-space interpolation (heuristic, not theory-exact)
            # NOTE: This does NOT satisfy the CPI improvement guarantee because
            # interpolating logits is not equivalent to mixing probabilities.
            self._capture_current_preinterpolation_pair()
            self._sync_policy_old_towards_candidate()
            self._sync_candidate_policy_from_old()

        result = {
            "loss_policy": float(loss_policy.item()),
            "policy_optimizer_step": 1.0,
        }
        if kl_div is not None:
            result["policy_kl"] = float(kl_div.item())
            result["kl_coef"] = self._kl_coef
        return result

    def train_step(
        self,
        max_env_steps_to_collect: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run one guarded outer step so checkpoints have an explicit idle phase."""

        if self._train_step_active:
            raise RuntimeError("UPI train_step is already active.")
        self._train_step_active = True
        before, _, _ = aggregate_model_compute(self._model_roles())
        started = time.perf_counter()
        try:
            return self._train_step_impl(max_env_steps_to_collect)
        finally:
            self._training_wall_time_seconds += time.perf_counter() - started
            after, _, _ = aggregate_model_compute(self._model_roles())
            self._training_model_work = add_model_counters(
                self._training_model_work,
                subtract_model_counters(after, before),
            )
            self._train_step_active = False

    def _train_step_impl(
        self,
        max_env_steps_to_collect: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        One outer training step: collect data, then run value and policy updates.
        
        In fixed-base mode the candidate shares the immutable base recurrence,
        collection uses the base policy, and the value update changes only the
        policy-independent value head. Legacy mode retains its historical update
        order for reproducibility.
        """
        
        if max_env_steps_to_collect is not None and max_env_steps_to_collect < 0:
            raise ValueError(
                "max_env_steps_to_collect must be non-negative or None."
            )

        # Ensure the candidate uses the same recurrent map as the base before
        # collection and exact-mixture evaluation.
        self._sync_candidate_backbone_from_model()

        # Debug: track train_step phases periodically
        _should_log = self._train_step_count < 5 or self._train_step_count % 50 == 0
        if _should_log:
            print(f"[DEBUG] train_step {self._train_step_count}: starting episode collection", flush=True)

        remaining_env_steps = max_env_steps_to_collect
        episodes_collected = 0
        env_steps_before = self._env_step_count
        rollout_target = int(self.rl_cfg.rollout_episodes_per_step)
        if rollout_target < 1:
            raise RuntimeError("rollout_episodes_per_step must be at least one.")

        while self._completed_episodes_since_update < rollout_target:
            if remaining_env_steps is not None and remaining_env_steps <= 0:
                break
            episode_was_active = self._active_episode is not None
            episode_steps = self.collect_episode(max_env_steps=remaining_env_steps)
            if remaining_env_steps is not None:
                remaining_env_steps -= episode_steps
            if self._active_episode is None and (episode_was_active or episode_steps > 0):
                self._completed_episodes_since_update += 1
                episodes_collected += 1
            if episode_steps <= 0:
                break

        # A collector cap is not an optimization boundary.  Retain the active
        # episode and any completed-episode quota until the ordinary rollout
        # target is reached.  This makes logging and checkpoint intervals
        # observational rather than training hyperparameters.
        if self._completed_episodes_since_update < rollout_target:
            metrics = {
                "loss_value": 0.0,
                "loss_policy": 0.0,
                "term_stop": float(self.term_stats["stop"]),
                "term_solved": float(self.term_stats["solved"]),
                "term_budget": float(self.term_stats["budget"]),
                "env_steps_collected": float(
                    self._env_step_count - env_steps_before
                ),
                "env_steps_total": float(self._env_step_count),
                "episodes_collected": float(episodes_collected),
                "episodes_pending_update": float(
                    self._completed_episodes_since_update
                ),
                "active_episode": float(self._active_episode is not None),
                "optimization_performed": 0.0,
                "train_steps_total": float(self._train_step_count),
                "value_optimizer_steps_total": float(
                    self._value_optimizer_step_count
                ),
                "policy_optimizer_steps_total": float(
                    self._policy_optimizer_step_count
                ),
            }
            metrics.update(self.get_current_lr())
            return metrics

        if _should_log:
            print(f"[DEBUG] train_step {self._train_step_count}: episodes collected, starting value_update", flush=True)

        value_result = self.value_update(
            scheduled_train_step=self._train_step_count + 1,
        )

        if _should_log:
            print(f"[DEBUG] train_step {self._train_step_count}: value_update done, starting policy_update", flush=True)

        policy_result = self.policy_update()

        if _should_log:
            print(f"[DEBUG] train_step {self._train_step_count}: policy_update done", flush=True)
        
        # Extract loss values from dicts
        loss_val = value_result.get("loss_value", 0.0) if isinstance(value_result, dict) else value_result
        if isinstance(policy_result, dict):
            loss_policy = policy_result.get("loss_policy", 0.0)
        else:
            loss_policy = policy_result
            policy_result = {}
        value_optimizer_stepped = bool(
            isinstance(value_result, dict)
            and value_result.get("value_optimizer_step", 0.0) == 1.0
        )
        policy_optimizer_stepped = bool(
            policy_result.get("policy_optimizer_step", 0.0) == 1.0
        )

        debug_metrics: Dict[str, float] = {}
        
        # Include value target stats from value_update
        if isinstance(value_result, dict):
            for key in ["target_mean", "target_std", "target_min", "target_max", 
                        "value_mean", "value_std", "reward_mean", "reward_std", "value_clip"]:
                if key in value_result:
                    debug_metrics[key] = value_result[key]
        
        if self.rl_cfg.debug_checks and len(self.replay) >= self.rl_cfg.batch_size:
            with torch.no_grad():
                transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
                x_b, y_b, x_next_b, y_next_b, _, rewards_b, dones_b = self._stack_batch(transitions)
                z_b = self._stack_latents(transitions, "latent")
                z_next_b = self._stack_latents(transitions, "next_latent")
                v_s, _ = self.model.used_value(
                    x_b, y_b, n=self.rl_cfg.inner_unroll_n, z=z_b
                )
                v_next, _ = self.model.used_value(
                    x_next_b,
                    y_next_b,
                    n=self.rl_cfg.inner_unroll_n,
                    z=z_next_b,
                )
                mask = (~dones_b).float()
                td_target = rewards_b + self.rl_cfg.gamma * v_next * mask
                adv = td_target - v_s
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
            "env_steps_collected": float(self._env_step_count - env_steps_before),
            "env_steps_total": float(self._env_step_count),
            "episodes_collected": float(episodes_collected),
            "episodes_pending_update": 0.0,
            "active_episode": float(self._active_episode is not None),
            "optimization_performed": float(
                value_optimizer_stepped or policy_optimizer_stepped
            ),
            "value_optimizer_step": float(value_optimizer_stepped),
            "policy_optimizer_step": float(policy_optimizer_stepped),
        }
        metrics.update(debug_metrics)
        metrics.update(theory_metrics)
        
        # Include policy KL metrics if present (from KL trust region)
        for key in ("policy_kl", "kl_coef", "kl_early_stop"):
            if key in policy_result:
                metrics[key] = policy_result[key]
        
        # Step learning rate schedulers only if optimization occurred
        self._train_step_count += 1
        self._completed_episodes_since_update = 0
        self._step_lr_schedulers(
            value_optimizer_stepped=value_optimizer_stepped,
            policy_optimizer_stepped=policy_optimizer_stepped,
        )
        metrics["train_steps_total"] = float(self._train_step_count)
        metrics["value_optimizer_steps_total"] = float(
            self._value_optimizer_step_count
        )
        metrics["policy_optimizer_steps_total"] = float(
            self._policy_optimizer_step_count
        )
        metrics.update(self.get_current_lr())

        # Reset termination stats for the next logging window
        self.term_stats = {"stop": 0.0, "solved": 0.0, "budget": 0.0}
        
        return metrics

    def get_env_step_count(self) -> int:
        return int(self._env_step_count)

    def _model_roles(self) -> Dict[str, nn.Module]:
        roles = {
            "model": self.model,
            "policy_model_candidate": self.policy_model_candidate,
            "policy_model_old": self.policy_model_old,
            "target_model": self.target_model,
        }
        if self.preinterpolation_policy_base is not None:
            roles["preinterpolation_policy_base"] = (
                self.preinterpolation_policy_base
            )
        if self.preinterpolation_policy_candidate is not None:
            roles["preinterpolation_policy_candidate"] = (
                self.preinterpolation_policy_candidate
            )
        return roles

    def _record_memory_peaks(self) -> None:
        self._peak_process_rss_bytes = max(
            self._peak_process_rss_bytes,
            process_peak_rss_bytes(),
        )
        allocated, reserved = current_cuda_memory_peaks(self.device)
        if allocated is not None:
            self._peak_cuda_allocated_bytes = max(
                self._peak_cuda_allocated_bytes or 0,
                allocated,
            )
            self._peak_cuda_reserved_bytes = max(
                self._peak_cuda_reserved_bytes or 0,
                reserved or 0,
            )

    def compute_accounting_snapshot(self) -> Dict[str, object]:
        live_total, role_groups, uninstrumented = aggregate_model_compute(
            self._model_roles()
        )
        total = add_model_counters(
            self._training_model_work,
            self._evaluation_model_work,
        )
        if live_total != total:
            raise RuntimeError(
                "Model compute occurred outside UPI train/evaluation accounting."
            )
        self._record_memory_peaks()
        optimizer_steps = {field: 0 for field in OPTIMIZER_STEP_FIELDS}
        optimizer_steps.update(
            {
                "value": int(self._value_optimizer_step_count),
                "policy": int(self._policy_optimizer_step_count),
                "distillation": int(self._distill_optimizer_step_count),
                "puzzle_embedding": int(self._puzzle_optimizer_step_count),
            }
        )
        optimizer_total = sum(optimizer_steps.values())
        snapshot: Dict[str, object] = {
            "compute_schema_version": COMPUTE_SNAPSHOT_SCHEMA_VERSION,
            "model_work": {
                "total": total,
                "training": dict(self._training_model_work),
                "evaluation": dict(self._evaluation_model_work),
                "counter_definitions": dict(MODEL_COUNTER_DEFINITIONS),
                "role_groups": role_groups,
                "uninstrumented_roles": uninstrumented,
            },
            "progress": {
                "environment_interactions": int(self._env_step_count),
                "outer_updates": int(self._train_step_count),
                "optimizer_steps_total": optimizer_total,
                "optimizer_steps_by_kind": optimizer_steps,
            },
            "wall_time_seconds": {
                "training": float(self._training_wall_time_seconds),
                "evaluation": float(self._evaluation_wall_time_seconds),
            },
            "peak_memory_bytes": {
                "cuda_allocated": self._peak_cuda_allocated_bytes,
                "cuda_reserved": self._peak_cuda_reserved_bytes,
                "process_rss": int(self._peak_process_rss_bytes),
            },
        }
        return validate_compute_snapshot(snapshot)

    def compute_accounting_checkpoint_state(self) -> Dict[str, object]:
        """Return resumable compute state, including each unique model once."""

        live_total, _, uninstrumented = aggregate_model_compute(
            self._model_roles()
        )
        accounted = add_model_counters(
            self._training_model_work,
            self._evaluation_model_work,
        )
        if uninstrumented or live_total != accounted:
            raise RuntimeError(
                "UPI checkpoint compute accounting is incomplete."
            )
        self._record_memory_peaks()
        return {
            "schema_version": 1,
            "model_compute_state": capture_model_compute_state(
                self._model_roles()
            ),
            "training_model_work": dict(self._training_model_work),
            "evaluation_model_work": dict(self._evaluation_model_work),
            "training_wall_time_seconds": float(
                self._training_wall_time_seconds
            ),
            "evaluation_wall_time_seconds": float(
                self._evaluation_wall_time_seconds
            ),
            "peak_process_rss_bytes": int(self._peak_process_rss_bytes),
            "peak_cuda_allocated_bytes": self._peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self._peak_cuda_reserved_bytes,
        }

    def restore_compute_accounting_checkpoint_state(
        self, state: object, *, validate_only: bool = False
    ) -> None:
        expected = {
            "schema_version",
            "model_compute_state",
            "training_model_work",
            "evaluation_model_work",
            "training_wall_time_seconds",
            "evaluation_wall_time_seconds",
            "peak_process_rss_bytes",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("compute accounting checkpoint state has an invalid inventory")
        if state["schema_version"] != 1:
            raise ValueError("unsupported compute accounting checkpoint schema")
        training = validate_model_counters(
            state["training_model_work"], name="training_model_work"
        )
        evaluation = validate_model_counters(
            state["evaluation_model_work"], name="evaluation_model_work"
        )
        wall_values: list[float] = []
        for field in (
            "training_wall_time_seconds",
            "evaluation_wall_time_seconds",
        ):
            value = state[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field} must be a nonnegative finite number")
            wall_values.append(float(value))
        peak_rss = state["peak_process_rss_bytes"]
        if isinstance(peak_rss, bool) or not isinstance(peak_rss, int) or peak_rss < 0:
            raise ValueError("peak_process_rss_bytes must be a nonnegative integer")
        cuda_peaks: list[Optional[int]] = []
        for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            value = state[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field} must be null or a nonnegative integer")
            cuda_peaks.append(value)
        if self.device.type == "cpu" and cuda_peaks != [None, None]:
            raise ValueError("CPU compute accounting must store null CUDA peaks")
        if self.device.type == "cuda" and any(value is None for value in cuda_peaks):
            raise ValueError("CUDA compute accounting is missing CUDA peaks")

        accounted = add_model_counters(training, evaluation)
        raw_total = zero_model_counters()
        raw_model_state = validate_model_compute_state(
            self._model_roles(), state["model_compute_state"]
        )
        for index, item in enumerate(raw_model_state):
            raw_total = add_model_counters(
                raw_total,
                validate_model_counters(
                    item["counters"], name=f"model_compute_state[{index}].counters"
                ),
            )
        if accounted != raw_total:
            raise ValueError(
                "phase accounting does not equal the raw unique-model counters"
            )

        if validate_only:
            return

        restore_model_compute_state(self._model_roles(), raw_model_state)

        self._training_model_work = training
        self._evaluation_model_work = evaluation
        self._training_wall_time_seconds = wall_values[0]
        self._evaluation_wall_time_seconds = wall_values[1]
        self._peak_process_rss_bytes = max(peak_rss, process_peak_rss_bytes())
        current_allocated, current_reserved = current_cuda_memory_peaks(self.device)
        if self.device.type == "cuda":
            self._peak_cuda_allocated_bytes = max(
                cuda_peaks[0] or 0, current_allocated or 0
            )
            self._peak_cuda_reserved_bytes = max(
                cuda_peaks[1] or 0, current_reserved or 0
            )
        else:
            self._peak_cuda_allocated_bytes = None
            self._peak_cuda_reserved_bytes = None

    def get_debug_stats(self) -> Dict[str, float]:
        """
        Get debug statistics about RL behavior.
        Call this periodically to understand what the agent is doing.
        """
        stats = {}
        
        if self._debug_episode_lengths:
            stats["avg_episode_length"] = sum(self._debug_episode_lengths) / len(self._debug_episode_lengths)
            stats["min_episode_length"] = min(self._debug_episode_lengths)
            stats["max_episode_length"] = max(self._debug_episode_lengths)
        
        if self._debug_episode_returns:
            stats["avg_episode_return"] = sum(self._debug_episode_returns) / len(self._debug_episode_returns)
            stats["min_episode_return"] = min(self._debug_episode_returns)
            stats["max_episode_return"] = max(self._debug_episode_returns)
        
        if self._debug_stop_probs:
            stats["avg_stop_prob"] = sum(self._debug_stop_probs) / len(self._debug_stop_probs)
            stats["min_stop_prob"] = min(self._debug_stop_probs)
            stats["max_stop_prob"] = max(self._debug_stop_probs)
        
        if self._debug_score_changes:
            stats["avg_score_change"] = sum(self._debug_score_changes) / len(self._debug_score_changes)
            stats["min_score_change"] = min(self._debug_score_changes)
            stats["max_score_change"] = max(self._debug_score_changes)
        
        return stats
    
    def clear_debug_stats(self) -> None:
        """Clear accumulated debug statistics."""
        self._debug_episode_lengths.clear()
        self._debug_episode_returns.clear()
        self._debug_stop_probs.clear()
        self._debug_score_changes.clear()
    
    def clear_theory_stats(self) -> None:
        """Clear accumulated theory tracking statistics (drift, plan changes, value of memory)."""
        self._drift_values.clear()
        self._plan_changes.clear()
        self._value_of_memory.clear()
    
    def set_checker_fn(self, checker_fn) -> None:
        """
        Set the checker function for exact baseline computation (Theorem 5.9).
        
        When exact_baseline_summation=True, the trainer needs access to the checker
        function to compute Q̂(s,a) = r(s,a,s') + γV(s') for all actions.
        
        Args:
            checker_fn: Function (x, y) -> float returning checker score
        """
        self._checker_fn = checker_fn

    def set_puzzle_embedding_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Attach the sparse embedding optimizer at the value-update boundary."""

        if self._fixed_base_exact:
            raise ValueError(
                "fixed_base_exact forbids trainable puzzle embeddings because "
                "they change the recurrent actor map."
            )
        self.puzzle_emb_optimizer = optimizer
        self.puzzle_emb_optimizer.zero_grad()
    
    def get_theory_stats(self) -> Dict[str, float]:
        """
        Get current theory tracking statistics.
        
        Returns dict with:
        - drift_mean/max: C_drift(n) statistics (for persistent latents)
        - plan_change_mean/max: Δy_max statistics
        - value_of_memory statistics (if tracked)
        """
        stats: Dict[str, float] = {}
        
        if self._drift_values:
            stats["drift_mean"] = sum(self._drift_values) / len(self._drift_values)
            stats["drift_max"] = max(self._drift_values)
            stats["drift_count"] = len(self._drift_values)
        
        if self._plan_changes:
            stats["plan_change_mean"] = sum(self._plan_changes) / len(self._plan_changes)
            stats["plan_change_max"] = max(self._plan_changes)
            stats["plan_change_count"] = len(self._plan_changes)
        
        if self._value_of_memory:
            stats["value_of_memory_mean"] = sum(self._value_of_memory) / len(self._value_of_memory)
            stats["value_of_memory_max"] = max(self._value_of_memory)
        
        return stats
    
    def print_debug_stats(self) -> None:
        """Print a summary of debug statistics."""
        stats = self.get_debug_stats()
        if not stats:
            print("[DEBUG] No episode data collected yet")
            return
        
        print("\n" + "="*60)
        print("RL DEBUG STATISTICS")
        print("="*60)
        
        if "avg_episode_length" in stats:
            print(f"Episode Length: avg={stats['avg_episode_length']:.1f}, "
                  f"min={stats['min_episode_length']}, max={stats['max_episode_length']}")
        
        if "avg_episode_return" in stats:
            print(f"Episode Return: avg={stats['avg_episode_return']:.3f}, "
                  f"min={stats['min_episode_return']:.3f}, max={stats['max_episode_return']:.3f}")
        
        if "avg_stop_prob" in stats:
            print(f"STOP Probability (step 0): avg={stats['avg_stop_prob']:.3f}, "
                  f"min={stats['min_stop_prob']:.3f}, max={stats['max_stop_prob']:.3f}")
        
        if "avg_score_change" in stats:
            print(f"Score Change: avg={stats['avg_score_change']:.4f}, "
                  f"min={stats['min_score_change']:.4f}, max={stats['max_score_change']:.4f}")
        
        print(f"Termination: stop={self.term_stats['stop']}, "
              f"solved={self.term_stats['solved']}, budget={self.term_stats['budget']}")
        print("="*60 + "\n")

    def _compute_theory_metrics(self) -> Dict[str, float]:
        """
        Compute theory-related metrics from the paper:
        
        Section 4 (Contraction):
        - hat_Cz: Estimated C_z = max ||z^(1) - z^(0)|| (Eq. 9)
        - hat_Lz: Estimated local Lipschitz constant of inner map
        - hat_Lv: Estimated Lipschitz constant of value head w.r.t. z
        - unrolling_term: L_V * L_z^n * C_z / (1 - L_z) (Eq. 10)
        
        Section 4.2 (Two-timescale / Persistent latents):
        - drift_mean/max: Empirical C_drift(n) from collected episodes
        - plan_change_mean/max: Empirical Δy_max from collected episodes
        
        Section 5 (Bellman residual):
        - bellman_residual_*: Empirical Bellman residual statistics
        
        Section 5.4 (Value of memory):
        - value_of_memory_*: ||V_persistent - V_memoryless||
        """
        metrics: Dict[str, float] = {}
        
        try:
            with torch.no_grad():
                transitions = self.replay.sample_batch(min(self.rl_cfg.batch_size, len(self.replay)))
                x_b, y_b, x_next_b, y_next_b, _, rewards_b, dones_b = self._stack_batch(transitions)
                z_b = self._stack_latents(transitions, "latent")
                z_next_b = self._stack_latents(transitions, "next_latent")
                
                # === Section 4: Contraction metrics ===
                # Estimate C_z (Eq. 9)
                hat_Cz = estimate_Cz(self.model, x_b, y_b)
                metrics["hat_Cz"] = hat_Cz
                
                # Estimate local L_z
                z_n = self.model.eval_latent(x_b, y_b, n=self.rl_cfg.inner_unroll_n)
                batch = self.model._standardize_latent_batch(x_b, y_b)
                context = self.model._build_latent_context_with_plan(batch)
                hat_Lz_result = estimate_local_Lz(
                    self.model.inner, z_n, context, num_samples=4
                )
                hat_Lz = (
                    hat_Lz_result[0]
                    if isinstance(hat_Lz_result, tuple)
                    else hat_Lz_result
                )
                metrics["hat_Lz"] = hat_Lz
                
                # Estimate L_v if value head exists
                if self.model.value_head is not None:
                    # Match model.used_value() and test_theory_metrics.py:
                    # the value head is trained on flattened latent/context features,
                    # not pooled summaries.
                    z_vec = z_n.z_H.view(z_n.z_H.shape[0], -1)
                    input_embeddings = context["input_embeddings"]
                    plan_embeddings = context["plan_embeddings"]
                    x_embed = input_embeddings.view(input_embeddings.shape[0], -1)
                    y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)
                    combined_embed = torch.cat([x_embed, y_embed], dim=-1)
                    hat_Lv = estimate_Lv(self.model.value_head, z_vec, combined_embed, num_samples=4)
                    metrics["hat_Lv"] = hat_Lv
                    
                    # Compute unrolling term proxy (Eq. 10)
                    n = self.rl_cfg.inner_unroll_n
                    unrolling_term = compute_unrolling_term_proxy(hat_Lv, hat_Lz, hat_Cz, n)
                    metrics["unrolling_term"] = unrolling_term
                
                # === Section 5: Bellman residual ===
                v_s, _ = self.model.used_value(
                    x_b, y_b, n=self.rl_cfg.inner_unroll_n, z=z_b
                )
                v_next, _ = self.model.used_value(
                    x_next_b,
                    y_next_b,
                    n=self.rl_cfg.inner_unroll_n,
                    z=z_next_b,
                )
                residual_metrics = compute_empirical_bellman_residual(
                    v_s, rewards_b, v_next, dones_b, self.rl_cfg.gamma
                )
                metrics.update(residual_metrics)
                
                # === Section 4.2: Two-timescale / Drift metrics (Lemma 4.4) ===
                if self._drift_values:
                    metrics["drift_mean"] = sum(self._drift_values) / len(self._drift_values)
                    metrics["drift_max"] = max(self._drift_values)
                    
                # Plan change tracking (Assumption 4.3: Δy_max)
                if self._plan_changes:
                    metrics["plan_change_mean"] = sum(self._plan_changes) / len(self._plan_changes)
                    metrics["plan_change_max"] = max(self._plan_changes)
                
                # === Section 5.4: Value of memory (Corollary 5.4, Remark 5.5) ===
                if getattr(self.rl_cfg, "compute_value_of_memory", False):
                    # Compute value of memory residual
                    vom_metrics = compute_value_of_memory_residual(
                        model=self.model,
                        x_batch=x_b,
                        y_batch=y_b,
                        z_batch=z_b,
                        n=self.rl_cfg.inner_unroll_n,
                        rewards=rewards_b,
                        next_values=v_next,
                        dones=dones_b,
                        gamma=self.rl_cfg.gamma,
                    )
                    metrics.update(vom_metrics)
                
        except Exception as e:
            if self.debug_checks:
                print(f"[warning] Theory metrics computation failed: {e}")
        
        return metrics

    def evaluate_policy_metrics(
        self,
        env_cfg: PlanEditEnvConfig,
        dataset: Any,
        checker: Any,
    ) -> Dict[str, Any]:
        before, _, _ = aggregate_model_compute(self._model_roles())
        started = time.perf_counter()
        try:
            return self._evaluate_policy_metrics_impl(env_cfg, dataset, checker)
        finally:
            self._evaluation_wall_time_seconds += time.perf_counter() - started
            after, _, _ = aggregate_model_compute(self._model_roles())
            self._evaluation_model_work = add_model_counters(
                self._evaluation_model_work,
                subtract_model_counters(after, before),
            )

    def _evaluate_policy_metrics_impl(
        self,
        env_cfg: PlanEditEnvConfig,
        dataset: Any,
        checker: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate the deployed policy, returning both strict success rate and mean checker score.

        Uses the same episodic_latent setting as training. Ordinary policies
        use greedy evaluation. An exact probability-space mixture is sampled,
        because taking the argmax of the mixed distribution is a different
        deployment rule. Evaluation runs on a private deterministic RNG stream
        and restores the training RNG state before returning.

        Returns dict with:
            - mean_score: Average final checker score
            - success_rate: Fraction of episodes that reached max score (solved)
            - eval_policy_mode: "greedy" or "stochastic_exact_mixture"
            - eval_seed: Seed used by the isolated evaluation RNG stream
            - solved_count: Number of solved episodes
            - total_episodes: Total evaluation episodes
            - score_min: Minimum final score across episodes
            - score_max: Maximum final score across episodes
            - max_possible_score: Maximum possible score (from solution)
            - initial_score_mean: Mean score at episode start (before edits)
        """
        # Runtime import to avoid circular dependency
        from rl.evaluator import evaluate_plan_policy_with_scores

        episodic_latent = getattr(self.rl_cfg, "episodic_latent", True)
        exact_mixture = bool(getattr(self.rl_cfg, "theory_exact_mixture", False))
        preinterpolation_mixture = (
            self._evaluation_policy_mode == "preinterpolation_exact_mixture"
        )
        policy_dist_fn = None
        eval_policy_mode = "greedy"
        greedy_eval = True
        evaluation_model = self.policy_model_old
        additional_models: Tuple[nn.Module, ...] = ()
        if self._evaluation_policy_mode == "stochastic_deployed":
            eval_policy_mode = "stochastic_deployed_policy"
            greedy_eval = False
        elif preinterpolation_mixture:
            if (
                self.preinterpolation_policy_base is None
                or self.preinterpolation_policy_candidate is None
            ):
                raise RuntimeError("Pre-interpolation evaluation pair is unavailable.")
            policy_dist_fn = self._preinterpolation_mixed_policy_dist
            eval_policy_mode = "stochastic_preinterpolation_exact_mixture"
            greedy_eval = False
            evaluation_model = self.preinterpolation_policy_base
            additional_models = (self.preinterpolation_policy_candidate,)
        elif exact_mixture:
            policy_dist_fn = self._mixed_policy_dist
            eval_policy_mode = "stochastic_exact_mixture"
            greedy_eval = False
            additional_models = (self.policy_model_candidate,)

        eval_seed = int(self.rl_cfg.eval_seed)
        mean_score, success_rate, detailed_stats = evaluate_plan_policy_with_scores(
            model=evaluation_model,
            dataset=dataset,
            checker=checker,
            env_cfg=env_cfg,
            task_config=getattr(self.env, "task_config", None),
            num_episodes=self.rl_cfg.eval_num_episodes,
            inner_unroll_n=self.rl_cfg.inner_unroll_n,
            episodic_latent=episodic_latent,
            greedy=greedy_eval,
            policy_dist_fn=policy_dist_fn,
            evaluation_seed=eval_seed,
            collect_per_instance=True,
            additional_models=additional_models,
            record_local_seeding=True,
        )

        result = {
            "mean_score": mean_score,
            "success_rate": success_rate,
            "eval_policy_mode": eval_policy_mode,
            "eval_seed": eval_seed,
        }
        result.update(detailed_stats)
        return result

    def evaluate_policy_success_rate(self, env_cfg: PlanEditEnvConfig, dataset: Any, checker: Any) -> float:
        """
        Backwards-compatible alias returning only the strict success rate.
        """

        metrics = self.evaluate_policy_metrics(env_cfg=env_cfg, dataset=dataset, checker=checker)
        return metrics["success_rate"]

    # =========================================================================
    # IMITATION LEARNING (for bootstrapping RL from oracle demonstrations)
    # =========================================================================
    
    def imitation_update(
        self,
        oracle_transitions: List[Tuple[Any, Any, int]],  # List of (x, y, oracle_action)
        optimizer: Optional[torch.optim.Optimizer] = None,  # Optional separate optimizer
    ) -> Dict[str, float]:
        """
        Perform one imitation learning update from oracle demonstrations.
        
        Args:
            oracle_transitions: List of (x, y, oracle_action) tuples where
                oracle_action is the correct action to take in state (x, y).
            optimizer: Optional optimizer to use (default: self.policy_opt).
        
        Returns:
            Dict with 'imitation_loss' and 'imitation_accuracy'.
        """
        if len(oracle_transitions) == 0:
            return {"imitation_loss": 0.0, "imitation_accuracy": 0.0}
        
        opt = optimizer if optimizer is not None else self.policy_opt
        
        self.model.train()
        self.policy_model_candidate.train()
        opt.zero_grad()
        
        losses: List[torch.Tensor] = []
        total_correct = 0
        total_count = 0
        
        for x, y, oracle_action in oracle_transitions:
            batched = self._state_is_batched(x)
            batch_x = self._prepare_batch_x(x, batched=batched)
            batch_y = self._prepare_plan(y, batched=batched)
            
            # Get action mask
            action_mask = None
            if self.env.vocab_size is not None and self.env.stop_action_id is not None:
                action_mask = PlanEditEnv.compute_batch_action_mask(
                    batch_x["inputs"] if isinstance(batch_x, dict) else batch_x,
                    self.env.vocab_size,
                    self.env.stop_action_id,
                    stop_mode=self.env._stop_mode,
                )
            
            # Get policy distribution from candidate model
            dist, _ = self.policy_model_candidate.policy_dist(
                batch_x, batch_y, n=self.rl_cfg.inner_unroll_n, action_mask=action_mask
            )
            
            # Cross-entropy loss on oracle action
            target = torch.tensor([oracle_action], dtype=torch.long, device=self.device)
            logits = dist.logits if hasattr(dist, 'logits') else torch.log(dist.probs + 1e-10)
            loss = F.cross_entropy(logits, target)
            
            losses.append(loss)
            total_count += 1
            
            # Check accuracy
            pred = logits.argmax(1).item()
            if pred == oracle_action:
                total_correct += 1
        
        # Average loss and backprop
        avg_loss = torch.stack(losses).mean()
        avg_loss.backward()
        
        policy_grad_clip = self.rl_cfg.policy_grad_clip
        if policy_grad_clip is not None and policy_grad_clip > 0:
            nn_utils.clip_grad_norm_(self._policy_params, policy_grad_clip)
        
        opt.step()
        
        # Imitation may initialize the candidate proposal, but the fixed base
        # remains immutable after protocol selection.
        if not self._fixed_base_exact:
            self._sync_policy_old_towards_candidate()
        
        return {
            "imitation_loss": avg_loss.item(),
            "imitation_accuracy": total_correct / total_count if total_count > 0 else 0.0,
        }
    
    def collect_oracle_demonstrations(
        self,
        dataset: Any,
        checker: Any,
        num_episodes: int = 100,
    ) -> List[Tuple[Any, Any, int]]:
        """
        Collect oracle demonstrations by running the optimal policy.
        
        For Sudoku, the oracle action at state (x, y) for an empty cell at
        position `pos` is: action = pos * vocab_size + correct_token.
        
        Args:
            dataset: Dataset to sample puzzles from.
            checker: Checker function to evaluate puzzle states.
            num_episodes: Number of episodes to collect.
        
        Returns:
            List of (x, y, oracle_action) tuples.
        """
        demonstrations = []
        vocab_size = self.env.vocab_size
        if vocab_size is None:
            raise RuntimeError("Oracle demonstrations require a configured vocabulary size.")
        num_samples = min(num_episodes, len(dataset) if hasattr(dataset, '__len__') else num_episodes)
        
        for ep_idx in range(num_samples):
            # Get sample directly from dataset to access solution
            sample = dataset[ep_idx % len(dataset)]
            
            # Get inputs and solution from sample
            if isinstance(sample, dict):
                inputs = sample.get("inputs")
                # Solution might be stored as "solution", "labels", or "targets"
                solution = sample.get("solution", sample.get("labels", sample.get("targets")))
            else:
                continue  # Skip non-dict samples
            
            if inputs is None or solution is None:
                continue
            
            # Convert to numpy for easier manipulation
            if torch.is_tensor(inputs):
                inputs_np = inputs.cpu().numpy().flatten()
            else:
                inputs_np = inputs.flatten()
            
            if torch.is_tensor(solution):
                solution_np = solution.cpu().numpy().flatten()
            else:
                solution_np = solution.flatten()
            
            # Create initial state (copy inputs for y)
            y_current = inputs_np.copy()
            
            # Find empty cells and generate demonstrations for each
            empty_positions = [pos for pos in range(len(inputs_np)) if inputs_np[pos] == 1]
            
            for pos in empty_positions:
                correct_token = int(solution_np[pos])
                oracle_action = pos * vocab_size + correct_token
                
                # Create state for this step - include all required fields
                x_dict = {
                    "inputs": torch.tensor(inputs_np, dtype=torch.long),
                    "puzzle_identifiers": torch.tensor([ep_idx], dtype=torch.long),  # Use episode idx as puzzle id
                }
                y_tensor = torch.tensor(y_current.copy(), dtype=torch.long)
                
                demonstrations.append((x_dict, y_tensor, oracle_action))
                
                # Update y for next step (fill in correct answer)
                y_current[pos] = correct_token
        
        return demonstrations
    
    def imitation_pretrain(
        self,
        dataset: Any,
        checker: Any,
        num_epochs: int = 50,
        batch_size: int = 64,
        log_interval: int = 10,
        imitation_lr: float = 0.003,  # Separate LR for imitation (higher than RL)
    ) -> Dict[str, float]:
        """
        Pre-train the policy with imitation learning from oracle.
        
        This bootstraps the RL training by giving the policy a good starting point.
        Uses a separate (higher) learning rate than RL to enable fast imitation learning.
        
        Args:
            dataset: Dataset to sample puzzles from.
            checker: Checker function.
            num_epochs: Number of epochs of imitation learning.
            batch_size: Batch size for imitation updates.
            log_interval: How often to log progress.
            imitation_lr: Learning rate for imitation (default 0.003, higher than RL).
        
        Returns:
            Dict with final training stats.
        """
        import random
        
        # Create a separate optimizer for imitation learning with higher LR
        imitation_opt = torch.optim.Adam(self._policy_params, lr=imitation_lr)
        
        print("[Imitation] Collecting oracle demonstrations...")
        # Collect more demonstrations by repeating puzzles
        all_demos = []
        num_repeats = 10  # Repeat each puzzle multiple times for more training data
        for repeat in range(num_repeats):
            demos = self.collect_oracle_demonstrations(
                dataset=dataset,
                checker=checker,
                num_episodes=min(500, len(dataset) if hasattr(dataset, '__len__') else 500),
            )
            all_demos.extend(demos)
        print(f"[Imitation] Collected {len(all_demos)} demonstrations")
        
        if len(all_demos) == 0:
            print("[Imitation] Warning: No demonstrations collected!")
            return {"imitation_loss": 0.0, "imitation_accuracy": 0.0}
        
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            random.shuffle(all_demos)
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Process in batches
            for i in range(0, len(all_demos), batch_size):
                batch = all_demos[i:i + batch_size]
                stats = self.imitation_update(batch, optimizer=imitation_opt)
                
                epoch_loss += stats["imitation_loss"] * len(batch)
                epoch_correct += stats["imitation_accuracy"] * len(batch)
                epoch_total += len(batch)
            
            avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0.0
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            best_accuracy = max(best_accuracy, accuracy)
            
            if (epoch + 1) % log_interval == 0:
                print(f"[Imitation] Epoch {epoch+1}/{num_epochs}: "
                      f"loss={avg_loss:.4f}, accuracy={accuracy:.2%}")
            
            # Early stopping if perfect
            if accuracy >= 0.99:
                print(f"[Imitation] Early stopping at epoch {epoch+1} with {accuracy:.2%} accuracy")
                break
        
        print(f"[Imitation] Pre-training complete. Best accuracy: {best_accuracy:.2%}")
        return {"imitation_loss": avg_loss, "imitation_accuracy": best_accuracy}
