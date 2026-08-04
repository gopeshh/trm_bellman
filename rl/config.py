from typing import Literal, Optional
import warnings

from pydantic import BaseModel, Field


class RLConfig(BaseModel):
    """
    Configuration for UPI-TRM reinforcement learning.
    
    This dataclass holds all hyperparameters for:
    - Value/policy optimization
    - K-step bootstrapping
    - CPI mixture updates
    - Theory-exact features
    - STOP action handling
    
    FIXED-SNAPSHOT PROTOCOL NOTES:
    ==============================
    The primary finite-reference result does not require latent contraction.
    For the exact centered CPI specialization, use:

    1. **Exact baseline for the alpha-scaled advantage-error term:**
       Set `exact_baseline_summation=True` for discrete action spaces.
       This computes E_{a~π}[Q̂(s,a)] via exact summation, ensuring
       E_{a~π}[Â(s,a)] = 0 EXACTLY for each state s.

    2. **Fixed-base training:**
       Set `training_protocol="fixed_base_exact"` with exact K-step targets,
       exact mixture deployment, zero exploration mixture, and no scheduled
       recurrent-map mutation.

    3. **Distillation warning:**
       `distill_mixture_policy=True` is NOT covered by theory.
       The projection step introduces unanalyzed error.
    
    NOTE ON ADVANTAGE CENTERING:
    ----------------------------
    - `exact_baseline_summation=True`: Enables TRUE centering (E_{a~π}[Â(s,a)]=0 per state)
      This is REQUIRED for Theorem 5.9's O(α·ε_A) bound.
      
    - `batch_centered_advantage=True`: Only subtracts batch mean (a HEURISTIC).
      This does NOT enable the O(α·ε_A) bound - it's just for variance reduction.
    """
    
    # Discount / horizons
    gamma: float = 0.99
    K: int = 5  # K-step horizon
    inner_unroll_n: int = 4  # n for U_n(s)
    max_edits: int = 16
    
    # Task configuration
    task_name: str = "sudoku"  # "sudoku", "arc", "dummy"
    # Checker score threshold that indicates a solved puzzle (triggers episode termination)
    # For Sudoku checker: 10.0 = perfect match (100% cells correct scaled to [0, 10])
    # Set to None to disable early termination on solve
    solved_threshold: Optional[float] = 10.0
    # Use constraint-based checker for 4x4 Sudoku (provides denser intermediate signals)
    # When True, rewards are based on constraint violations (row/column/box duplicates)
    # rather than matching the known solution. This provides better learning signal.
    use_constraint_checker: bool = False
    # Use progress-based checker for 4x4 Sudoku (most informative scoring)
    # When True, score = filled_cells (if no violations) or filled_cells - penalty
    # This distinguishes between "partially filled" (e.g., 12) and "fully solved" (16)
    # Takes precedence over use_constraint_checker if both are True.
    use_progress_checker: bool = False

    # Use feasibility-aware checker for Sudoku (RECOMMENDED - most informative)
    # Score = filled - w_v * violations - w_z * zeroCand
    # Where zeroCand = number of empty cells with 0 legal candidates (dead-ends)
    # This provides:
    # - Dense signal for progress (filling cells)
    # - Strong penalty for violations (duplicate digits)
    # - Strong penalty for dead-ends (impossible states)
    # Takes precedence over all other checkers if True.
    use_feasibility_checker: bool = False
    feasibility_violation_weight: float = 2.0  # w_v: weight for constraint violations
    feasibility_zerocand_weight: float = 5.0   # w_z: weight for zero-candidate cells
    # If True, disable Sudoku digit-constraint masking and only block edits to given cells.
    # This restores the pre-constraint-aware action space used by historical 4x4 runs.
    disable_constraint_masking: bool = False
    
    # Latent z mode
    # True (default): z is reinitialized from (x, y) at every step (episodic)
    # False: z is initialized once per episode and carried/updated across steps (persistent)
    episodic_latent: bool = True

    # === STOP action behavior ===
    # Controls how the STOP action is handled during training
    # "terminal": STOP ends the episode (standard RL)
    # "noop": STOP is a no-op, episode continues (prevents STOP collapse)
    # "disabled": STOP action is masked out entirely
    stop_action_mode: str = "noop"  # "terminal", "noop", "disabled"
    # Penalty applied when STOP is chosen (only used with "noop" mode)
    stop_action_penalty: float = -0.1
    
    # === Reward shaping ===
    # True (default): Use potential-based reward shaping r = γ·Φ(s') - Φ(s)
    # False: Sparse terminal rewards only (receive checker score at episode end)
    reward_shaping: bool = True

    # === Absorbing state value (Paper Section 3.1, lines 653-656) ===
    # C_max is the maximum checker score (Φ(s_abs) in paper notation).
    # The paper defines V^π(s_abs) = -C_max for all policies.
    # This is used in K-step bootstrapping: when episode terminates, bootstrap
    # with -C_max instead of 0 (paper Eq. 12, line 677-678).
    # For Sudoku: C_max = 10.0 (checker returns 0-10 scale)
    C_max: float = 10.0

    # === Terminal rewards (Paper Remark 2.6: Rush-to-fail mitigation) ===
    # These are the r_0 terms in the shaped reward: r = r_0 + γ·Φ(s') - Φ(s)
    #
    # fail_terminal_reward: Applied when episode ends WITHOUT solving.
    #   Paper requires r_fail ≤ -γC_max to prevent "rush to fail".
    #   Default -10.0 = -C_max for Sudoku, satisfying this condition.
    #
    # solve_terminal_reward: Applied when episode ends WITH solving.
    #   Default 0.0 relies purely on shaping. Can add positive bonus.
    fail_terminal_reward: float = -10.0  # Theory-aligned default (= -C_max)
    solve_terminal_reward: float = 0.0
    
    # Target network EMA
    target_ema_tau: float = 0.995

    # CPI / TRPO-style "dials"
    mixture_alpha: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Convex probability-space mixture weight.",
    )
    trust_region_kl: float = 0.01  # KL divergence threshold for trust-region updates (0 = disabled)
    enable_kl_trust_region: bool = False  # If True, apply KL penalty/early stopping in policy update

    # Theory-facing toggles
    exact_k_step_targets: bool = False  # If True, use fixed-horizon γ^K bootstrap in K-step value update
    distill_mixture_policy: bool = False  # If True, distill the mixture policy into policy_model_old

    # Historical runs used the mutable ``legacy`` training path. The corrected
    # fixed-base protocol must be selected explicitly so old checkpoints and
    # configurations cannot be reinterpreted as theorem-facing runs.
    training_protocol: Literal["legacy", "fixed_base_exact"] = "legacy"
    
    # === Theory-exact mixture mode (Issue 4 - CPI guarantee) ===
    # 
    # The CPI bound (Theorem 5.9) is proven for a POLICY-SPACE mixture:
    #     π_new = (1 - α) * π_old + α * π_candidate
    # 
    # Default behavior (theory_exact_mixture=False):
    #     - Data collection uses policy-space mixture in _mixed_policy_dist()
    #     - After policy update, uses PARAMETER-SPACE interpolation via
    #       _sync_policy_old_towards_candidate(): param.lerp_(candidate_param, α)
    #     - This is NOT equivalent to policy-space mixture after softmax
    #     - The CPI improvement guarantee does NOT strictly apply
    # 
    # ``theory_exact_mixture`` selects exact probability-space deployment. It
    # does not by itself fix the training protocol. Pair it with
    # ``training_protocol="fixed_base_exact"`` to collect from one frozen base,
    # train only a policy-independent value head, and optimize one candidate
    # proposal without recursively promoting the mixture.
    # 
    # When distill_mixture_policy=True (Section 6.5):
    #     - The mixture is distilled into policy_model_old via KL minimization
    #     - This is a HEURISTIC approximation, NOT covered by theory
    #     - Introduces projection error not analyzed in Theorem 5.9
    theory_exact_mixture: bool = False

    # The deployment bridge keeps legacy training fixed while changing only the
    # frozen checkpoint evaluator. Both bridge cells capture the base/candidate
    # pair immediately before parameter interpolation; the treatment evaluates
    # its exact probability-space mixture instead of the interpolated actor.
    capture_preinterpolation_policy_pair: bool = False
    evaluation_policy_mode: Literal[
        "configured",
        "stochastic_deployed",
        "preinterpolation_exact_mixture",
    ] = "configured"
    
    # === Batch-level advantage centering (HEURISTIC, not theory-exact) ===
    # Subtracts batch mean from advantages: adv = adv - adv.mean()
    # This is a variance-reduction heuristic, NOT the theoretical centering from Theorem 5.9.
    # For theory-exact centering, use exact_baseline_summation=True instead.
    # RENAMED from 'centered_advantage' for clarity.
    batch_centered_advantage: bool = True
    
    # === Theorem 5.9: Exact baseline (KEY THEORETICAL CONTRIBUTION) ===
    # When True, compute E_{a ~ π}[Q̂(s,a)] via exact summation over ALL discrete actions.
    # This ensures E_{a~π}[Â(s,a)] = 0 EXACTLY for each state s (not just batch-level).
    # This is what enables the O(α·ε_A) bound instead of naive O(ε_A/(1-γ)).
    # Computationally expensive but tractable for discrete action spaces like Sudoku (~800 actions).
    exact_baseline_summation: bool = False
    
    # === Two-timescale / drift monitoring (Assumption 4.3, Lemma 4.4) ===
    # Track C_drift(n) = ||z^(n) - z_init(x,y)|| for persistent-latent analysis
    track_drift_metrics: bool = False
    # Track Δy_max = max plan change per step (for two-timescale bound)
    track_plan_change: bool = False
    
    # === Value of memory analysis (Section 5.4) ===
    # If True, compute V_memoryless and the "value of memory" residual
    compute_value_of_memory: bool = False
    
    # === Forward-invariant projection (Assumption 4.1, CRITICAL) ===
    # Projects latent z to ball of radius R after each update: z ← z · min(1, R/||z||)
    # This ensures z ∈ Z_inv = {z : ||z|| ≤ R} which is REQUIRED for contraction guarantees.
    # Without this, latents can escape the contractive region, breaking Assumption 4.2.
    # Paper Eq. 14. Recommended: 10.0 for typical hidden dimensions.
    latent_ball_radius: float = 10.0  # ENABLED by default for theory alignment

    # GAE (Generalized Advantage Estimation)
    # NOTE: GAE is a practical heuristic (Section 6.2) - not covered by formal theory bounds
    use_gae: bool = False  # If True, use GAE instead of 1-step TD advantages
    gae_lambda: float = 0.95  # λ parameter for GAE (higher = more bias towards Monte Carlo)

    # Optimization
    value_lr: float = 3e-4
    policy_lr: float = 3e-4
    backbone_lr: Optional[float] = None  # If None, PPO uses policy_lr for the shared trunk
    entropy_coef: float = 0.01
    value_grad_clip: Optional[float] = 5.0
    policy_grad_clip: Optional[float] = 1.0
    value_target_clip: Optional[float] = 20.0  # Default for progress checker (range 0-16)
    advantage_clip: Optional[float] = 10.0
    policy_epsilon: float = 0.0
    
    # Learning rate scheduling
    lr_schedule: str = "cosine"  # "constant", "cosine", "linear"
    lr_warmup_steps: int = 500  # warmup steps before decay starts
    lr_min_factor: float = 0.1  # minimum LR as fraction of initial (e.g., 0.1 = 10% of initial)

    # Lipschitz / contraction controls
    enable_contraction: bool = True
    target_Lz: float = 0.9
    target_Lv: float = 1.0
    opnorm_clamp_interval: int = 100  # Re-apply opnorm clamp every N steps (0 = disabled)
    opnorm_clamp_max_norm: float = 1.0  # Per-layer max operator norm for clamping
    opnorm_clamp_num_power_iters: int = 10  # Power iterations for spectral norm estimation
    opnorm_log_max_sigma: bool = False  # If True, log max σ(W) when clamp fires (adds overhead)

    # === Value head normalization toggle (2x2 ablation finding) ===
    # When enable_contraction=True, the value head normally gets spectral_norm + Lv scaling.
    # The 2x2 ablation showed this causes training collapse (targets saturate to ±20).
    # Set disable_value_head_norm=True to keep z→z contraction ON but skip value-head normalization.
    disable_value_head_norm: bool = False

    # Replay / data
    replay_capacity: int = 100_000
    batch_size: int = 256

    # Training loop
    num_train_steps: int = 10_000
    rollout_episodes_per_step: int = 4

    # Logging / evaluation
    log_interval: int = 10
    eval_interval: int = 50
    eval_num_episodes: int = 50
    # Evaluation uses a private RNG stream so changing eval cadence cannot
    # change subsequent training samples.
    eval_seed: int = 1729
    use_tqdm: bool = True
    debug_checks: bool = False
    track_theory_metrics: bool = False  # If True, compute and log C_z, L_z, L_v estimates

    # === DEPRECATED: Backward compatibility alias ===
    # Old name 'centered_advantage' maps to 'batch_centered_advantage'
    @property
    def centered_advantage(self) -> bool:
        """Deprecated alias for batch_centered_advantage. Use batch_centered_advantage instead."""
        return self.batch_centered_advantage

    def validate_theory_alignment(self, warn: bool = True) -> dict:
        """
        Check if configuration aligns with theoretical guarantees from the paper.
        
        Returns a dict with validation results and optionally emits warnings.
        
        Projection and clamping are optional geometric specializations. The
        fixed-snapshot finite-reference result does not require either one.
        Exact centering and exact probability-space mixture deployment remain
        required for the CPI specialization; distillation is not covered.
        """
        issues = []
        
        # Discount sanity check
        if not (0.0 < self.gamma < 1.0):
            issues.append(
                f"gamma={self.gamma} is outside (0,1). Theoretical guarantees "
                "assume a strictly discounted MDP. Set 0 < gamma < 1."
            )
        
        specialization_notes = []
        if self.latent_ball_radius <= 0:
            specialization_notes.append("forward-invariant projection disabled")
        if not self.enable_contraction:
            specialization_notes.append("clamping intervention disabled")
        elif self.target_Lz >= 1.0:
            specialization_notes.append("configured target_Lz is not below 1")
        
        # Check exact baseline (Theorem 5.9) - THE KEY THEORETICAL CONTRIBUTION
        if not self.exact_baseline_summation:
            issues.append(
                "exact_baseline_summation=False: Using approximate baseline. "
                "This gives O(ε_A/(1-γ)) bound instead of O(α·ε_A) from Theorem 5.9. "
                "For discrete action spaces (Sudoku), enable for tighter bounds."
            )
        if not self.exact_k_step_targets:
            issues.append("exact_k_step_targets=False: fixed-K target protocol is disabled.")
        if self.policy_epsilon != 0.0:
            issues.append(
                "policy_epsilon must be 0 for direct deployment of the stated CPI mixture."
            )
        if not 0.0 <= self.mixture_alpha <= 1.0:
            issues.append("mixture_alpha must lie in [0, 1] for a convex policy mixture.")

        if self.training_protocol == "fixed_base_exact":
            if not self.theory_exact_mixture:
                issues.append(
                    "fixed_base_exact requires theory_exact_mixture=True for "
                    "probability-space proposal deployment."
                )
            if self.enable_contraction and self.opnorm_clamp_interval > 0:
                issues.append(
                    "fixed_base_exact requires opnorm_clamp_interval=0 because "
                    "scheduled clamping would mutate the frozen recurrent map."
                )
        elif self.theory_exact_mixture:
            issues.append(
                "theory_exact_mixture=True with training_protocol='legacy' does "
                "not implement fixed-base training; use fixed_base_exact for a "
                "frozen one-step proposal."
            )
        
        # Check distillation (Section 6.5)
        if self.distill_mixture_policy:
            issues.append(
                "distill_mixture_policy=True: Distillation is NOT covered by theory "
                "(Section 6.5). The projected policy may not satisfy improvement guarantees."
            )
        
        # Check CPI mixture mode (Issue 4)
        if not self.theory_exact_mixture and not self.distill_mixture_policy:
            issues.append(
                "theory_exact_mixture=False: Using parameter-space interpolation for CPI, "
                "which is NOT equivalent to policy-space mixture. The CPI improvement "
                "guarantee does not strictly apply. Set theory_exact_mixture=True for "
                "theory-exact behavior (but slower due to evaluating two networks)."
            )
        
        # Rush-to-fail mitigation (Remark 2.6)
        # The theory requires fail_terminal_reward <= -gamma * C_max.
        # With C_max now explicit, we can check this properly.
        if self.reward_shaping:
            required_threshold = -self.gamma * self.C_max
            if self.fail_terminal_reward > required_threshold:
                issues.append(
                    f"fail_terminal_reward={self.fail_terminal_reward} > -γ·C_max={required_threshold:.2f} "
                    f"may allow 'rush to fail' (Remark 2.6). Set fail_terminal_reward <= {required_threshold:.2f}."
                )
        
        if warn and issues:
            for issue in issues:
                warnings.warn(f"[UPI-TRM Theory] {issue}", UserWarning)
        
        return {
            "theory_aligned": len(issues) == 0,
            "issues": issues,
            "forward_invariant": self.latent_ball_radius > 0,
            "clamping_enabled": self.enable_contraction,
            "global_contraction_certified": False,
            "exact_baseline": self.exact_baseline_summation,
            "distillation_used": self.distill_mixture_policy,
            "theory_exact_mixture": self.theory_exact_mixture,
            "training_protocol": self.training_protocol,
            "fixed_base_proposal_exact": self.is_fixed_base_proposal_exact(),
            "specialization_notes": specialization_notes,
        }

    def is_fixed_base_proposal_exact(self) -> bool:
        """
        Check whether configuration matches the frozen one-step proposal protocol.
        
        Returns True if configuration matches the exact CPI snapshot protocol.
        Projection and contraction are optional specializations of the finite-reference
        result, not prerequisites for exact centering or mixture deployment:
        - Exact K-step targets (Section 5.1)
        - Exact baseline summation (Theorem 5.9) - THE KEY REQUIREMENT
        - Theory-exact CPI mixture (Issue 4) - policy-space not parameter-space
        - No distillation (Section 6.5)
        
        NOTE: batch_centered_advantage is NOT checked because it's just a heuristic.
        The theory requires exact_baseline_summation for true centering.
        """
        return (
            0.0 < self.gamma < 1.0 and
            self.training_protocol == "fixed_base_exact" and
            self.exact_k_step_targets and
            self.exact_baseline_summation and  # THE KEY REQUIREMENT for O(α·ε_A)
            self.theory_exact_mixture and  # Policy-space mixture for CPI guarantee
            not self.distill_mixture_policy and
            self.policy_epsilon == 0.0 and
            0.0 <= self.mixture_alpha <= 1.0 and
            (not self.enable_contraction or self.opnorm_clamp_interval == 0)
        )

    def is_theory_exact(self) -> bool:
        """Compatibility alias for the exact fixed-base proposal check.

        This describes protocol mechanics only. It does not certify uniform
        theorem assumptions or learned-model performance.
        """

        return self.is_fixed_base_proposal_exact()
