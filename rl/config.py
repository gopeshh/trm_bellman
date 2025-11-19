from typing import Optional

from pydantic import BaseModel


class RLConfig(BaseModel):
    # Discount / horizons
    gamma: float = 0.99
    K: int = 5  # K-step horizon
    inner_unroll_n: int = 4  # n for U_n(s)
    max_edits: int = 16

    # Target network EMA
    target_ema_tau: float = 0.995

    # CPI / TRPO-style "dials"
    mixture_alpha: float = 0.1  # mixture weight between old and candidate policy
    trust_region_kl: float = 0.01  # reserved for future KL-trust-region variant

    # Theory-exact toggles
    exact_k_step_targets: bool = False  # If True, use fixed-horizon γ^K bootstrap in K-step value update
    centered_advantage: bool = False  # If True, use a centered advantage estimator in policy update
    distill_mixture_policy: bool = False  # If True, distill the mixture policy into policy_model_old

    # Optimization
    value_lr: float = 3e-4
    policy_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_grad_clip: Optional[float] = 5.0
    policy_grad_clip: Optional[float] = 1.0
    value_target_clip: Optional[float] = 10.0
    advantage_clip: Optional[float] = 10.0
    policy_epsilon: float = 0.0

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

