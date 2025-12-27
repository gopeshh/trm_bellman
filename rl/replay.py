"""
Replay buffer and transition storage for UPI-TRM.

This module provides the replay buffer infrastructure for storing and sampling
transitions during RL training.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

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

    def sample_sequences(
        self,
        batch_size: int,
        K: int,
        allow_overlap: bool = True,
    ) -> List[List[Transition]]:
        """
        Sample K-step contiguous sequences for multi-step returns.

        This method groups transitions by episode_id, then samples K consecutive
        transitions from the same episode. This is useful for computing K-step
        bootstrapped targets or n-step returns.

        Args:
            batch_size: Number of K-step sequences to sample
            K: Length of each sequence
            allow_overlap: If True, sampled sequences may overlap within episodes.
                          If False, sequences are sampled without replacement
                          (requires more data).

        Returns:
            List of K-step sequences, where each sequence is a list of K Transitions.
            If an episode has fewer than K remaining transitions, the sequence is
            truncated (will have fewer than K transitions).

        Raises:
            AssertionError: If buffer doesn't have enough valid sequences

        Example:
            >>> buffer = ReplayBuffer(10000)
            >>> # ... add transitions ...
            >>> sequences = buffer.sample_sequences(batch_size=32, K=5)
            >>> for seq in sequences:
            ...     # seq is a list of up to 5 consecutive transitions
            ...     rewards = [t.reward for t in seq]
        """
        # Group transitions by episode_id
        episode_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for buffer_idx, transition in enumerate(self.storage):
            episode_groups[transition.episode_id].append(
                (buffer_idx, transition.timestep)
            )

        # Sort each episode by timestep
        for ep_id in episode_groups:
            episode_groups[ep_id].sort(key=lambda x: x[1])

        # Build list of valid starting indices for K-step sequences
        # A valid starting index is one where we can get at least 1 transition
        # (we allow truncated sequences at episode boundaries)
        valid_starts: List[Tuple[int, int, int]] = []  # (buffer_idx, ep_id, ep_offset)

        for ep_id, indices in episode_groups.items():
            for ep_offset, (buffer_idx, timestep) in enumerate(indices):
                # Every position is a valid start (may be truncated)
                valid_starts.append((buffer_idx, ep_id, ep_offset))

        if len(valid_starts) < batch_size:
            raise AssertionError(
                f"Not enough valid sequence starts: {len(valid_starts)} < {batch_size}"
            )

        # Sample starting positions
        if allow_overlap:
            # Sample with replacement
            sample_indices = torch.randint(
                low=0, high=len(valid_starts), size=(batch_size,)
            ).tolist()
        else:
            # Sample without replacement
            perm = torch.randperm(len(valid_starts))[:batch_size].tolist()
            sample_indices = perm

        # Extract K-step sequences
        sequences = []
        for sample_idx in sample_indices:
            buffer_idx, ep_id, ep_offset = valid_starts[sample_idx]
            ep_indices = episode_groups[ep_id]

            # Get K consecutive transitions (or fewer if episode ends)
            seq = []
            for k in range(K):
                seq_offset = ep_offset + k
                if seq_offset >= len(ep_indices):
                    # Episode ended, truncate sequence
                    break

                buf_idx, _ = ep_indices[seq_offset]
                seq.append(self.storage[buf_idx])

                # Stop if this transition is terminal
                if self.storage[buf_idx].done:
                    break

            sequences.append(seq)

        return sequences

    def sample_full_episodes(
        self,
        num_episodes: int,
    ) -> List[List[Transition]]:
        """
        Sample complete episodes from the buffer.

        Useful for on-policy methods or episode-level analysis.

        Args:
            num_episodes: Number of episodes to sample

        Returns:
            List of episodes, where each episode is a list of Transitions
            ordered by timestep.

        Raises:
            AssertionError: If buffer doesn't have enough complete episodes
        """
        # Group by episode_id
        episode_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for buffer_idx, transition in enumerate(self.storage):
            episode_groups[transition.episode_id].append(
                (buffer_idx, transition.timestep)
            )

        # Find complete episodes (those with a terminal transition)
        complete_episodes: List[int] = []
        for ep_id, indices in episode_groups.items():
            # Sort by timestep
            indices.sort(key=lambda x: x[1])
            # Check if last transition is terminal
            last_buf_idx = indices[-1][0]
            if self.storage[last_buf_idx].done:
                complete_episodes.append(ep_id)

        if len(complete_episodes) < num_episodes:
            raise AssertionError(
                f"Not enough complete episodes: {len(complete_episodes)} < {num_episodes}"
            )

        # Sample episodes
        sample_indices = torch.randint(
            low=0, high=len(complete_episodes), size=(num_episodes,)
        ).tolist()

        episodes = []
        for idx in sample_indices:
            ep_id = complete_episodes[idx]
            indices = episode_groups[ep_id]
            indices.sort(key=lambda x: x[1])

            episode = [self.storage[buf_idx] for buf_idx, _ in indices]
            episodes.append(episode)

        return episodes

