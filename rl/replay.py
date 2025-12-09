"""
Replay buffer and transition storage for UPI-TRM.

This module provides the replay buffer infrastructure for storing and sampling
transitions during RL training.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Union

import torch

# Type alias for environment state (dict with tensor fields)
StateDict = Dict[str, torch.Tensor]
# Type alias for plan (can be tensor or dict with "inputs" key)
PlanType = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass
class Transition:
    """
    A single transition (s, a, r, s', done) in the plan-space MDP.
    
    Attributes:
        x: Instance state (dict with "inputs", "puzzle_identifiers")
        y: Current plan tensor or dict
        action: Action taken (edit or STOP)
        reward: Reward received
        x_next: Next instance state
        y_next: Next plan tensor or dict
        done: Whether episode terminated
        episode_id: ID of the episode this transition belongs to
        timestep: Step within the episode
    """
    x: StateDict
    y: PlanType
    action: torch.Tensor
    reward: torch.Tensor
    x_next: StateDict
    y_next: PlanType
    done: torch.Tensor
    episode_id: int
    timestep: int


class ReplayBuffer:
    """
    Simple replay buffer with fixed capacity and uniform sampling.
    
    Uses a deque for O(1) append and automatic eviction of oldest transitions
    when capacity is exceeded.
    
    Args:
        capacity: Maximum number of transitions to store
        
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add(transition)
        >>> batch = buffer.sample_batch(32)
    """
    
    def __init__(self, capacity: int):
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.storage.append(transition)

    def __len__(self) -> int:
        """Return the current number of transitions."""
        return len(self.storage)

    def sample_batch(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
            
        Raises:
            AssertionError: If buffer has fewer than batch_size transitions
        """
        assert len(self.storage) >= batch_size, "Not enough transitions in replay buffer"
        indices = torch.randint(low=0, high=len(self.storage), size=(batch_size,)).tolist()
        return [self.storage[i] for i in indices]

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.storage.clear()
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions for sampling."""
        return len(self.storage) >= batch_size

