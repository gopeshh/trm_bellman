"""
Baseline RL algorithms for comparison with UPI-TRM.

This module provides PPO and A2C trainers that can be used with either
the TRM backbone or simpler no-recursion encoders.
"""

from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.algos.a2c import A2CTrainer, A2CConfig

__all__ = ["PPOTrainer", "PPOConfig", "A2CTrainer", "A2CConfig"]
