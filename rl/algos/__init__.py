"""
Baseline RL algorithms for comparison with UPI-TRM.

This module provides PPO, A2C, and DQN trainers that can be used with either
the TRM backbone or simpler no-recursion encoders.
"""

from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.algos.a2c import A2CTrainer, A2CConfig
from rl.algos.dqn import DQNTrainer, DQNConfig

__all__ = [
    "PPOTrainer", "PPOConfig",
    "A2CTrainer", "A2CConfig",
    "DQNTrainer", "DQNConfig",
]
