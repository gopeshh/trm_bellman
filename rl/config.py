from typing import Optional

from pydantic import BaseModel


class RLConfig(BaseModel):
    # Discount / horizons
    gamma: float = 0.99
    K: int = 5  # K-step horizon
    inner_unroll_n: int = 4  # n for U_n(s)
    max_edits: int = 16
    
    # Latent z mode
    # True (default): z is reinitialized from (x, y) at every step (episodic)
    # False: z is initialized once per episode and carried/updated across steps (persistent)
    episodic_latent: bool = True

    # Target network EMA
    target_ema_tau: float = 0.995

    # CPI / TRPO-style "dials"
    mixture_alpha: float = 0.1  # mixture weight between old and candidate policy
    trust_region_kl: float = 0.01  # KL divergence threshold for trust-region updates (0 = disabled)
    enable_kl_trust_region: bool = False  # If True, apply KL penalty/early stopping in policy update

    # Theory-exact toggles
    exact_k_step_targets: bool = False  # If True, use fixed-horizon γ^K bootstrap in K-step value update
    centered_advantage: bool = True  # If True, use a centered advantage estimator in policy update
    distill_mixture_policy: bool = False  # If True, distill the mixture policy into policy_model_old

    # GAE (Generalized Advantage Estimation)
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

