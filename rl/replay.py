"""
Replay buffer and transition storage for UPI-TRM.

This module provides the replay buffer infrastructure for storing and sampling
transitions during RL training.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import torch

# Type alias for environment state (dict with tensor fields)
StateDict = Dict[str, torch.Tensor]
# Type alias for plan (can be tensor or dict with "inputs" key)
PlanType = Union[torch.Tensor, Dict[str, torch.Tensor]]
TERMINAL_REASONS = frozenset({"stop", "solved", "budget"})


@dataclass
class ReplayLatent:
    """Detached recurrent state retained with a persistent-latent transition."""

    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class Transition:
    """
    One transition in either the episodic or clock-complete augmented MDP.

    ``x`` retains the transition-relevant ``remaining_edits`` clock. In
    persistent mode, ``latent`` is the carry before the current recurrent
    unroll and ``next_latent`` is the post-unroll carry passed to the successor
    state. The edit action is applied after that recurrent unroll.
    
    Attributes:
        x: Instance state, including ``remaining_edits`` when available
        y: Current plan tensor or dict
        action: Action taken (edit or STOP)
        reward: Reward received
        x_next: Next instance state
        y_next: Next plan tensor or dict
        done: Whether episode terminated
        episode_id: ID of the episode this transition belongs to
        timestep: Step within the episode
        latent: Recurrent state before evaluating this state, in persistent mode
        next_latent: Recurrent state carried to the successor, in persistent mode
        behavior_log_prob: Log probability of the sampled action at collection time
        terminal_reason: Canonical terminal cause, or None for a nonterminal record
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
    latent: Optional[ReplayLatent] = None
    next_latent: Optional[ReplayLatent] = None
    behavior_log_prob: Optional[torch.Tensor] = None
    terminal_reason: Optional[str] = None


class ReplayIntegrityError(ValueError):
    """Raised when replay records do not form valid augmented transitions."""


def _scalar_bool(value: torch.Tensor, *, label: str) -> bool:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ReplayIntegrityError(f"`{label}` must be a scalar tensor.")
    if value.dtype != torch.bool:
        raise ReplayIntegrityError(f"`{label}` must have boolean dtype.")
    return bool(value.detach().cpu().reshape(()).item())


def _clock_value(state: StateDict, *, label: str) -> Optional[int]:
    if "remaining_edits" not in state:
        return None
    value = state["remaining_edits"]
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ReplayIntegrityError(
            f"`{label}.remaining_edits` must be a scalar tensor."
        )
    scalar = value.detach().cpu().reshape(()).item()
    integer = int(scalar)
    if float(scalar) != float(integer) or integer < 0:
        raise ReplayIntegrityError(
            f"`{label}.remaining_edits` must be a nonnegative integer, got {scalar!r}."
        )
    return integer


def _tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(left, right)
    )


def _value_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and _tensor_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if set(left) != set(right):
            return False
        return all(_value_equal(left[key], right[key]) for key in left)
    return bool(left == right)


def _latent_equal(left: ReplayLatent, right: ReplayLatent) -> bool:
    return _tensor_equal(left.z_H, right.z_H) and _tensor_equal(left.z_L, right.z_L)


def _validate_latent(latent: ReplayLatent, *, label: str) -> None:
    if not torch.is_tensor(latent.z_H) or not torch.is_tensor(latent.z_L):
        raise ReplayIntegrityError(f"`{label}` H/L components must be tensors.")
    if latent.z_H.shape != latent.z_L.shape:
        raise ReplayIntegrityError(f"`{label}` H/L components must have equal shapes.")
    if not bool(torch.isfinite(latent.z_H).all().item()) or not bool(
        torch.isfinite(latent.z_L).all().item()
    ):
        raise ReplayIntegrityError(f"`{label}` contains a non-finite recurrent state.")


def validate_transition(
    transition: Transition,
    *,
    require_clock: bool = False,
    require_terminal_reason: bool = False,
) -> None:
    """Validate one replay record without assuming an adjacent record exists."""

    if transition.episode_id < 0:
        raise ReplayIntegrityError("`episode_id` must be nonnegative.")
    if transition.timestep < 0:
        raise ReplayIntegrityError("`timestep` must be nonnegative.")
    done = _scalar_bool(transition.done, label="done")
    terminal_reason = getattr(transition, "terminal_reason", None)
    if terminal_reason is not None and terminal_reason not in TERMINAL_REASONS:
        raise ReplayIntegrityError(
            "`terminal_reason` must be one of "
            f"{sorted(TERMINAL_REASONS)}, got {terminal_reason!r}."
        )
    if not done and terminal_reason is not None:
        raise ReplayIntegrityError(
            "A nonterminal replay record cannot have a `terminal_reason`."
        )
    if done and require_terminal_reason and terminal_reason is None:
        raise ReplayIntegrityError(
            "Theorem-facing terminal replay requires a recognized `terminal_reason`."
        )

    has_latent = transition.latent is not None
    has_next_latent = transition.next_latent is not None
    if has_latent != has_next_latent:
        raise ReplayIntegrityError(
            "A transition must contain both `latent` and `next_latent`, or neither."
        )
    if transition.latent is not None and transition.next_latent is not None:
        _validate_latent(transition.latent, label="latent")
        _validate_latent(transition.next_latent, label="next_latent")

    clock = _clock_value(transition.x, label="x")
    next_clock = _clock_value(transition.x_next, label="x_next")
    if (clock is None) != (next_clock is None):
        raise ReplayIntegrityError(
            "A transition must retain `remaining_edits` in both x and x_next."
        )
    if require_clock and clock is None:
        raise ReplayIntegrityError(
            "Clock-complete replay requires `remaining_edits` in x and x_next."
        )
    if clock is not None and next_clock is not None:
        if clock <= 0:
            raise ReplayIntegrityError(
                "A replay transition cannot leave a state with zero remaining edits."
            )
        expected = max(clock - 1, 0)
        if next_clock != expected:
            raise ReplayIntegrityError(
                "Replay clock must decrement exactly once: "
                f"expected {expected}, got {next_clock}."
            )
        if next_clock == 0 and not done:
            raise ReplayIntegrityError(
                "A replay record reaching zero remaining edits must be terminal."
            )


def validate_transition_continuity(
    current: Transition,
    successor: Transition,
    *,
    require_clock: bool = False,
    require_terminal_reason: bool = False,
) -> None:
    """Validate that two records are adjacent states of one episode."""

    validate_transition(
        current,
        require_clock=require_clock,
        require_terminal_reason=require_terminal_reason,
    )
    validate_transition(
        successor,
        require_clock=require_clock,
        require_terminal_reason=require_terminal_reason,
    )
    if current.episode_id != successor.episode_id:
        raise ReplayIntegrityError("Adjacent records must have the same episode_id.")
    if _scalar_bool(current.done, label="done"):
        raise ReplayIntegrityError("No replay transition may follow a terminal record.")
    if successor.timestep != current.timestep + 1:
        raise ReplayIntegrityError(
            "Replay timesteps must be consecutive: "
            f"expected {current.timestep + 1}, got {successor.timestep}."
        )
    if not _value_equal(current.x_next, successor.x):
        raise ReplayIntegrityError("current.x_next does not equal successor.x.")
    if not _value_equal(current.y_next, successor.y):
        raise ReplayIntegrityError("current.y_next does not equal successor.y.")

    if current.next_latent is None:
        if successor.latent is not None:
            raise ReplayIntegrityError(
                "Replay sequence changes from episodic to persistent latent mode."
            )
    else:
        if successor.latent is None or not _latent_equal(
            current.next_latent, successor.latent
        ):
            raise ReplayIntegrityError(
                "current.next_latent does not equal successor.latent."
            )


def validate_transition_sequence(
    transitions: Sequence[Transition],
    *,
    require_clock: bool = False,
    require_terminal_reason: bool = False,
) -> None:
    """Validate every record and adjacency relation in one nonempty sequence."""

    if not transitions:
        raise ReplayIntegrityError("Cannot validate an empty replay sequence.")
    validate_transition(
        transitions[0],
        require_clock=require_clock,
        require_terminal_reason=require_terminal_reason,
    )
    for current, successor in zip(transitions, transitions[1:]):
        validate_transition_continuity(
            current,
            successor,
            require_clock=require_clock,
            require_terminal_reason=require_terminal_reason,
        )


class ReplayBuffer:
    """
    Simple replay buffer with fixed capacity and uniform sampling.
    
    Uses a deque for O(1) append and automatic eviction of oldest transitions
    when capacity is exceeded.
    
    Args:
        capacity: Maximum number of transitions to store
        theorem_facing: Validate clock-complete records and adjacency before append
        
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add(transition)
        >>> batch = buffer.sample_batch(32)
    """
    
    def __init__(self, capacity: int, *, theorem_facing: bool = False):
        self.storage: Deque[Transition] = deque(maxlen=capacity)
        self.theorem_facing = theorem_facing

    def add(
        self,
        transition: Transition,
        *,
        allow_legacy_missing_terminal_reason: bool = False,
    ) -> None:
        """Add a transition, failing before mutation in theorem-facing mode."""

        if self.theorem_facing:
            require_terminal_reason = not allow_legacy_missing_terminal_reason
            validate_transition(
                transition,
                require_clock=True,
                require_terminal_reason=require_terminal_reason,
            )
            if self.storage:
                previous = self.storage[-1]
                if previous.episode_id == transition.episode_id:
                    validate_transition_continuity(
                        previous,
                        transition,
                        require_clock=True,
                        require_terminal_reason=require_terminal_reason,
                    )
                else:
                    previous_done = _scalar_bool(previous.done, label="done")
                    if not previous_done:
                        raise ReplayIntegrityError(
                            "A new replay episode cannot follow a nonterminal record."
                        )
                    if transition.episode_id != previous.episode_id + 1:
                        raise ReplayIntegrityError(
                            "Theorem-facing replay episode IDs must be consecutive."
                        )
                    if transition.timestep != 0:
                        raise ReplayIntegrityError(
                            "The first record of a new replay episode must have timestep zero."
                        )
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

    def contiguous_segment(
        self,
        start_index: int,
        horizon: int,
        *,
        require_complete: bool,
        require_clock: bool = False,
    ) -> List[Transition]:
        """Return a validated segment ending at ``horizon`` or a terminal record.

        When ``require_complete`` is true, a nonterminal segment shorter than
        ``horizon`` raises ``ReplayIntegrityError``. This is the fail-closed
        primitive used by fixed-K target sampling.
        """

        if horizon < 1:
            raise ValueError("horizon must be at least one.")
        if start_index < 0 or start_index >= len(self.storage):
            raise IndexError(f"Replay start index {start_index} is out of range.")

        segment = [self.storage[start_index]]
        validate_transition(segment[0], require_clock=require_clock)
        if (
            start_index > 0
            and self.storage[start_index - 1].episode_id == segment[0].episode_id
        ):
            validate_transition_continuity(
                self.storage[start_index - 1],
                segment[0],
                require_clock=require_clock,
            )
        while len(segment) < horizon and not _scalar_bool(
            segment[-1].done, label="done"
        ):
            next_index = start_index + len(segment)
            if next_index >= len(self.storage):
                break
            successor = self.storage[next_index]
            if successor.episode_id != segment[0].episode_id:
                break
            validate_transition_continuity(
                segment[-1],
                successor,
                require_clock=require_clock,
            )
            segment.append(successor)

        if (
            require_complete
            and len(segment) < horizon
            and not _scalar_bool(segment[-1].done, label="done")
        ):
            raise ReplayIntegrityError(
                f"Replay start {start_index} has only {len(segment)} contiguous "
                f"nonterminal transitions; fixed horizon requires {horizon}."
            )
        return segment

    def has_complete_segment(
        self,
        start_index: int,
        horizon: int,
        *,
        require_clock: bool = False,
    ) -> bool:
        """Return whether a fixed-K or earlier-terminal segment is valid."""

        if horizon < 1:
            raise ValueError("horizon must be at least one.")
        try:
            self.contiguous_segment(
                start_index,
                horizon,
                require_complete=True,
                require_clock=require_clock,
            )
        except (IndexError, ReplayIntegrityError):
            return False
        return True

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
