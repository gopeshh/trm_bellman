"""Policy head for learning edit actions in the meta-MDP.

This module implements the edit policy πφ that learns to generate actions
(edits) that improve the model's internal reasoning process.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead:
    """Policy network that outputs edit actions given current state.

    The policy πφ(edit | x, state) learns to propose edits that will
    improve task performance through RL optimization.

    Attributes:
        state_dim: Dimension of state representations
        action_dim: Dimension of edit action space
        hidden_dim: Dimension of hidden layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """Initialize policy head.

        Args:
            state_dim: Dimension of state representations
            action_dim: Dimension of edit action space
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: Any, state: Any) -> Tuple[Any, Any]:
        """Forward pass to compute edit action distribution.

        Args:
            x: Input tensor (B, input_dim)
            state: Current internal state (B, state_dim)

        Returns:
            Tuple of (mean, log_std) for edit action distribution
        """
        pass

    def sample(self, x: Any, state: Any, deterministic: bool = False) -> Tuple[Any, Any]:
        """Sample edit action from policy.

        Args:
            x: Input tensor (B, input_dim)
            state: Current internal state (B, state_dim)
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        pass

    def log_prob(self, x: Any, state: Any, action: Any) -> Any:
        """Compute log probability of action under current policy.

        Args:
            x: Input tensor (B, input_dim)
            state: Current internal state (B, state_dim)
            action: Edit action (B, action_dim)

        Returns:
            Log probability tensor (B,)
        """
        pass

    def entropy(self, x: Any, state: Any) -> Any:
        """Compute entropy of policy distribution.

        Args:
            x: Input tensor (B, input_dim)
            state: Current internal state (B, state_dim)

        Returns:
            Entropy tensor (B,)
        """
        pass


class DiscretePolicyHead:
    """Discrete policy for categorical edit actions.

    Alternative to continuous PolicyHead for discrete edit spaces.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        """Initialize discrete policy head.

        Args:
            state_dim: Dimension of state representations
            num_actions: Number of discrete edit actions
            hidden_dim: Dimension of hidden layers
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

    def forward(self, x: Any, state: Any) -> Any:
        """Forward pass to compute action logits.

        Args:
            x: Input tensor (B, input_dim)
            state: Current internal state (B, state_dim)

        Returns:
            Action logits (B, num_actions)
        """
        pass


class EditPolicy(nn.Module):
    """Factorized edit policy over (position, value) actions.

    The policy factorizes the edit action a = (i, v) into:
    - i: position index (where to edit)
    - v: value index (what value to use)

    This allows learning which positions are important to edit and
    what values to use independently, with optional masking for
    valid positions and values.

    Attributes:
        z_dim: Dimension of internal state z
        y_dim: Dimension of current output y
        num_cells: Number of positions (cells) that can be edited
        num_vals: Number of possible values
        hidden: Hidden dimension
    """

    def __init__(self, z_dim: int, y_dim: int, num_cells: int, num_vals: int, hidden: int = 256):
        """Initialize factorized edit policy.

        Args:
            z_dim: Dimension of internal state representations
            y_dim: Dimension of current output
            num_cells: Number of positions (cells) that can be edited
            num_vals: Number of possible token values
            hidden: Hidden dimension (default: 256)
        """
        super().__init__()
        self.pos_head = nn.Linear(z_dim + y_dim, num_cells)
        self.val_head = nn.Linear(z_dim + y_dim, num_vals)

    def dist(
        self, y: torch.Tensor, z_n: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability distributions over positions and values.

        Args:
            y: Current output tensor (B, y_dim)
            z_n: Internal state tensor (B, z_dim)
            x: Input tensor (B, x_dim) - not used but kept for API consistency

        Returns:
            Tuple of (log_probs_pos, log_probs_val) where:
                - log_probs_pos: (B, num_cells)
                - log_probs_val: (B, num_vals)
        """
        h = torch.cat([z_n, y], dim=-1)
        return F.log_softmax(self.pos_head(h), -1), F.log_softmax(self.val_head(h), -1)

    def sample(
        self,
        y: torch.Tensor,
        z_n: torch.Tensor,
        x: torch.Tensor,
        mask_pos: Optional[torch.Tensor] = None,
        mask_val: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Sample edit action (position, value) from factorized policy.

        Args:
            y: Current output tensor (B, y_dim)
            z_n: Internal state tensor (B, z_dim)
            x: Input tensor (B, x_dim) - not used but kept for API consistency
            mask_pos: Optional position mask (B, num_cells) - binary mask
            mask_val: Optional value mask (B, num_vals) - binary mask

        Returns:
            Tuple of ((pos, val), log_prob) where:
                - pos: Position indices (B,)
                - val: Value indices (B,)
                - log_prob: Total log probability (B,)
        """
        lp_pos, lp_val = self.dist(y, z_n, x)
        if mask_pos is not None:
            lp_pos = lp_pos + mask_pos.log()
        pos = torch.distributions.Categorical(logits=lp_pos).sample()
        if mask_val is not None:
            lp_val = lp_val + mask_val[pos]
        val = torch.distributions.Categorical(logits=lp_val).sample()
        lp = lp_pos.gather(-1, pos.unsqueeze(-1)).squeeze(-1) + lp_val.gather(
            -1, val.unsqueeze(-1)
        ).squeeze(-1)
        return (pos, val), lp

    def log_prob(
        self,
        y: torch.Tensor,
        z_n: torch.Tensor,
        x: torch.Tensor,
        a: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probability of given action.

        Args:
            y: Current output tensor (B, y_dim)
            z_n: Internal state tensor (B, z_dim)
            x: Input tensor (B, x_dim) - not used but kept for API consistency
            a: Action tuple (pos, val) where:
                - pos: Position indices (B,)
                - val: Value indices (B,)

        Returns:
            Log probability (B,)
        """
        (pos, val) = a
        lp_pos, lp_val = self.dist(y, z_n, x)
        return lp_pos.gather(-1, pos.unsqueeze(-1)).squeeze(-1) + lp_val.gather(
            -1, val.unsqueeze(-1)
        ).squeeze(-1)
