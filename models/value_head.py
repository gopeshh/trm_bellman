import torch
import torch.nn as nn


class LatentValueHead(nn.Module):
    """
    Simple MLP mapping (z_vec, x_embed_vec) -> scalar value per example.

    - z:   [B, z_dim]      (summary of latent state, e.g. from z_H)
    - x_embed: [B, x_dim]  (summary embedding of the input instance x)
    """

    def __init__(self, z_dim: int, x_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        # z: [B, z_dim], x_embed: [B, x_dim]
        inp = torch.cat([z, x_embed], dim=-1)
        return self.mlp(inp).squeeze(-1)  # [B]

