from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


class EditPolicyHead(nn.Module):
    """
    Generic interface: given summarized (z, x, y), produce a distribution over edits + STOP.
    Inputs:
      - z_vec:   [B, latent_dim]   (summary of latent state, e.g. pooled z_H)
      - x_embed: [B, x_embed_dim]  (summary embedding of the input x; aligns with the plan-space meta-MDP context)
      - y_embed: [B, y_embed_dim]  (summary embedding of the plan y; matches the plan-space meta-MDP state)
    Output:
      - Categorical distribution over action_dim discrete actions.
    """

    def __init__(
        self,
        latent_dim: int,
        x_embed_dim: int,
        y_embed_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + x_embed_dim + y_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z_vec: torch.Tensor,
        x_embed: torch.Tensor,
        y_embed: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """
        Returns a Categorical over actions.
        - action_mask: optional boolean mask [B, action_dim] where False entries are invalid.
        """

        logits = self.mlp(torch.cat([z_vec, x_embed, y_embed], dim=-1))
        if action_mask is not None:
            # mask out invalid actions
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return Categorical(logits=logits)

