from typing import Optional
import warnings

from pydantic import BaseModel


class RLConfig(BaseModel):
    """
    Configuration for UPI-TRM reinforcement learning.
    
    This dataclass holds all hyperparameters for:
    - Value/policy optimization
    - K-step bootstrapping
    - CPI mixture updates
    - Theory-exact features
    - STOP action handling
    
    THEORY ALIGNMENT NOTES (ICML 2026 Paper):
    =========================================
    For full theoretical guarantees (Theorem 5.9), you need:
    
    1. **Forward-invariant projection (Assumption 4.1):**
       Set `latent_ball_radius > 0` (default: 10.0) to ensure z ∈ Z_inv
       
    2. **Exact baseline for O(α·ε_A) bound (Theorem 5.9) [KEY CONTRIBUTION]:**
       Set `exact_baseline_summation=True` for discrete action spaces.
       This computes E_{a~π}[Q̂(s,a)] via exact summation, ensuring
       E_{a~π}[Â(s,a)] = 0 EXACTLY for each state s.
       Without this, you get the weaker O(ε_A/(1-γ)) bound.
       
    3. **Contraction requirement (Assumption 4.2):**
       Ensure `enable_contraction=True` and `target_Lz < 1.0`
       
    4. **Distillation warning (Section 6.5):**
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
    
    # === Terminal rewards (Paper Remark 2.6: Rush-to-fail mitigation) ===
    # These are the r_0 terms in the shaped reward: r = r_0 + γ·Φ(s') - Φ(s)
    #
    # fail_terminal_reward: Applied when episode ends WITHOUT solving.
    #   Set to -C_max (e.g., -10.0 for Sudoku) to prevent "rush to fail".
    #   The condition r_fail ≤ -γC_max ensures failing yields non-positive reward.
    #   Default 0.0 does NOT fully prevent rush-to-fail.
    #
    # solve_terminal_reward: Applied when episode ends WITH solving.
    #   Default 0.0 relies purely on shaping. Can add positive bonus.
    fail_terminal_reward: float = 0.0
    solve_terminal_reward: float = 0.0
    
    # Target network EMA
    target_ema_tau: float = 0.995

    # CPI / TRPO-style "dials"
    mixture_alpha: float = 0.1  # mixture weight between old and candidate policy
    trust_region_kl: float = 0.01  # KL divergence threshold for trust-region updates (0 = disabled)
    enable_kl_trust_region: bool = False  # If True, apply KL penalty/early stopping in policy update

    # Theory-exact toggles (Section 5 of paper)
    exact_k_step_targets: bool = False  # If True, use fixed-horizon γ^K bootstrap in K-step value update
    distill_mixture_policy: bool = False  # If True, distill the mixture policy into policy_model_old
    
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
    # Theory-exact mode (theory_exact_mixture=True):
    #     - Data collection uses policy-space mixture in _mixed_policy_dist()
    #     - Does NOT update policy_model_old after policy update
    #     - The deployed "policy" is always the explicit mixture π_new
    #     - This matches the CPI theory but requires evaluating TWO networks
    #       during data collection (slightly more expensive)
    # 
    # When distill_mixture_policy=True (Section 6.5):
    #     - The mixture is distilled into policy_model_old via KL minimization
    #     - This is a HEURISTIC approximation, NOT covered by theory
    #     - Introduces projection error not analyzed in Theorem 5.9
    theory_exact_mixture: bool = False
    
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
    entropy_coef: float = 0.01
    value_grad_clip: Optional[float] = 5.0
    policy_grad_clip: Optional[float] = 1.0
    value_target_clip: Optional[float] = 10.0
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
        
        Paper References:
        - Assumption 4.1: Forward-invariant region (latent_ball_radius)
        - Assumption 4.2: Contraction (enable_contraction, target_Lz)
        - Theorem 5.9: Exact baseline for O(α·ε_A) bound
        - Section 6.5: Distillation not covered by theory
        """
        issues = []
        
        # Discount sanity check
        if not (0.0 < self.gamma < 1.0):
            issues.append(
                f"gamma={self.gamma} is outside (0,1). Theoretical guarantees "
                "assume a strictly discounted MDP. Set 0 < gamma < 1."
            )
        
        # Check forward-invariant projection (Assumption 4.1)
        if self.latent_ball_radius <= 0:
            issues.append(
                "latent_ball_radius=0 disables forward-invariant projection (Assumption 4.1). "
                "Contraction guarantees may not hold. Set latent_ball_radius > 0."
            )
        
        # Check contraction (Assumption 4.2)
        if not self.enable_contraction:
            issues.append(
                "enable_contraction=False disables spectral normalization (Assumption 4.2). "
                "L_z < 1 contraction not enforced."
            )
        elif self.target_Lz >= 1.0:
            issues.append(
                f"target_Lz={self.target_Lz} >= 1.0 violates contraction requirement "
                "(Assumption 4.2). Set target_Lz < 1.0 for convergence guarantees."
            )
        
        # Check exact baseline (Theorem 5.9) - THE KEY THEORETICAL CONTRIBUTION
        if not self.exact_baseline_summation:
            issues.append(
                "exact_baseline_summation=False: Using approximate baseline. "
                "This gives O(ε_A/(1-γ)) bound instead of O(α·ε_A) from Theorem 5.9. "
                "For discrete action spaces (Sudoku), enable for tighter bounds."
            )
        elif not self.reward_shaping:
            issues.append(
                "exact_baseline_summation=True but reward_shaping=False. "
                "The exact baseline implementation assumes shaped rewards and will "
                "assert at runtime. Set reward_shaping=True."
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
        # For Sudoku-like tasks where the checker score is in [0, C_max],
        # the theory suggests setting fail_terminal_reward <= -gamma * C_max.
        # We don't know C_max here, but we can at least warn if fail_terminal_reward >= 0.
        if self.reward_shaping and self.fail_terminal_reward >= 0.0:
            issues.append(
                "fail_terminal_reward >= 0 with reward_shaping=True may allow 'rush to fail' "
                "behaviour, violating Remark 2.6. For theory-aligned configs, set "
                "fail_terminal_reward to a sufficiently negative value (e.g. -C_max)."
            )
        
        if warn and issues:
            for issue in issues:
                warnings.warn(f"[UPI-TRM Theory] {issue}", UserWarning)
        
        return {
            "theory_aligned": len(issues) == 0,
            "issues": issues,
            "forward_invariant": self.latent_ball_radius > 0,
            "contraction_enforced": self.enable_contraction and self.target_Lz < 1.0,
            "exact_baseline": self.exact_baseline_summation,
            "distillation_used": self.distill_mixture_policy,
            "theory_exact_mixture": self.theory_exact_mixture,
        }
    
    def is_theory_exact(self) -> bool:
        """
        Check if configuration enables the paper's theoretical guarantees.
        
        Returns True if configuration matches the paper's Theorem 5.9 requirements:
        - Forward-invariant projection (Assumption 4.1)
        - Contraction L_z < 1 (Assumption 4.2)
        - Exact K-step targets (Section 5.1)
        - Exact baseline summation (Theorem 5.9) - THE KEY REQUIREMENT
        - Theory-exact CPI mixture (Issue 4) - policy-space not parameter-space
        - No distillation (Section 6.5)
        
        NOTE: batch_centered_advantage is NOT checked because it's just a heuristic.
        The theory requires exact_baseline_summation for true centering.
        """
        return (
            self.latent_ball_radius > 0 and
            self.enable_contraction and
            self.target_Lz < 1.0 and
            self.exact_k_step_targets and
            self.exact_baseline_summation and  # THE KEY REQUIREMENT for O(α·ε_A)
            self.theory_exact_mixture and  # Policy-space mixture for CPI guarantee
            not self.distill_mixture_policy
        )
