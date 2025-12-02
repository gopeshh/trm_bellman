import torch
import torch.nn as nn


class LatentValueHead(nn.Module):
    """
    Simple MLP mapping (z_vec, x_like_embed) -> scalar value per example.

    - z: [B, z_dim]          (summary of latent state, e.g. pooled z_H)
    - x_like_embed: [B, x_dim]  (summary embedding of the "context", which in our
                                 implementation is concat(x_embed, y_embed))
    
    Uses a bounded output activation (scaled tanh) to prevent value explosion
    and stabilize training with large reward scales.
    """

    def __init__(self, z_dim: int, x_dim: int, hidden_dim: int, v_max: float = 50.0):
        super().__init__()
        self.v_max = v_max
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        # z: [B, z_dim], x_embed: [B, x_dim]
        inp = torch.cat([z, x_embed], dim=-1)
        logits = self.mlp(inp).squeeze(-1)  # [B]
        # Bound output to [-v_max, v_max] using scaled tanh
        return self.v_max * torch.tanh(logits / self.v_max)

