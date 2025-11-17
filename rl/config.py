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

    # Optimization
    value_lr: float = 3e-4
    policy_lr: float = 3e-4
    entropy_coef: float = 0.01

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

