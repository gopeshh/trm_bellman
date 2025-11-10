"""Reinforcement Learning package for meta-MDP training of recursive models.

This package implements RL infrastructure for training models to edit their own
internal states through policy optimization with value function guidance.
"""

__version__ = "0.1.0"

__all__ = [
    "meta_mdp",
    "policy_head",
    "value_head",
    "rollout",
    "advantages",
    "losses",
    "contraction",
]
