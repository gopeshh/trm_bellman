# Makes the rl package importable (e.g., `from rl.config import RLConfig`).

from rl.config import RLConfig
from rl.replay import ReplayBuffer, Transition
from rl.value_targets import (
    compute_k_step_bootstrapped_target,
    compute_gae,
    compute_gae_trajectory,
    compute_empirical_bellman_residual,
    compute_td_advantage,
)
from rl.task_config import (
    TaskConfig,
    SudokuTaskConfig,
    DummyTaskConfig,
    ARCTaskConfig,
    get_task_config,
)

__all__ = [
    # Config
    "RLConfig",
    # Replay
    "ReplayBuffer",
    "Transition",
    # Value targets
    "compute_k_step_bootstrapped_target",
    "compute_gae",
    "compute_gae_trajectory",
    "compute_empirical_bellman_residual",
    "compute_td_advantage",
    # Task configs
    "TaskConfig",
    "SudokuTaskConfig",
    "DummyTaskConfig",
    "ARCTaskConfig",
    "get_task_config",
]
