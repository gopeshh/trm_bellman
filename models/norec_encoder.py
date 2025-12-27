"""
No-Recursion baseline encoder for comparison with TRM.

This module provides a simpler encoder architecture that does NOT use
the inner latent recursion of TRM. It encodes (x, y) directly into a
representation used for policy and value prediction.

This serves as an ablation to test whether TRM's recursive refinement
provides benefits over a direct feedforward approach.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.edit_policy import EditPolicyHead
from models.value_head import LatentValueHead


@dataclass
class NoRecEncoderConfig:
    """Configuration for No-Recursion Encoder."""

    # Input dimensions
    vocab_size: int = 11  # Token vocabulary size (9x9 Sudoku: 0-10)
    seq_len: int = 81  # Sequence length (9x9 Sudoku: 81)

    # Architecture
    hidden_dim: int = 128  # Hidden dimension
    num_layers: int = 2  # Number of encoder layers
    encoder_type: str = "mlp"  # "mlp" or "transformer"
    num_heads: int = 4  # Attention heads (for transformer)
    dropout: float = 0.1

    # RL heads
    rl_num_actions: int = 892  # Number of edit actions (seq_len * vocab_size + 1 for STOP)
    v_max: float = 50.0  # Value head output bound

    def model_dump(self) -> Dict[str, Any]:
        """For compatibility with TRM model."""
        return {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "encoder_type": self.encoder_type,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "rl_num_actions": self.rl_num_actions,
            "v_max": self.v_max,
        }

    def dict(self) -> Dict[str, Any]:
        """Alias for model_dump."""
        return self.model_dump()


class MLPEncoder(nn.Module):
    """Simple MLP encoder for (x, y) pairs."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Token embedding (shared for x and y)
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # MLP layers
        # Input: concatenated x and y embeddings (2 * seq_len * hidden_dim)
        input_dim = 2 * seq_len * hidden_dim
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode (x, y) pair.

        Args:
            x: [B, seq_len] input tokens
            y: [B, seq_len] plan tokens

        Returns:
            [B, hidden_dim] encoded representation
        """
        # Embed tokens
        x_emb = self.embed(x)  # [B, seq_len, hidden_dim]
        y_emb = self.embed(y)  # [B, seq_len, hidden_dim]

        # Flatten and concatenate
        x_flat = x_emb.view(x_emb.shape[0], -1)  # [B, seq_len * hidden_dim]
        y_flat = y_emb.view(y_emb.shape[0], -1)  # [B, seq_len * hidden_dim]
        combined = torch.cat([x_flat, y_flat], dim=-1)  # [B, 2 * seq_len * hidden_dim]

        # MLP encoding
        return self.mlp(combined)  # [B, hidden_dim]


class TransformerEncoder(nn.Module):
    """Transformer encoder for (x, y) pairs."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Type embedding (x vs y)
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(2 * seq_len * hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode (x, y) pair using transformer.

        Args:
            x: [B, seq_len] input tokens
            y: [B, seq_len] plan tokens

        Returns:
            [B, hidden_dim] encoded representation
        """
        B = x.shape[0]

        # Embed tokens
        x_emb = self.embed(x)  # [B, seq_len, hidden_dim]
        y_emb = self.embed(y)  # [B, seq_len, hidden_dim]

        # Add type embeddings
        x_type = self.type_embed(torch.zeros(B, self.seq_len, dtype=torch.long, device=x.device))
        y_type = self.type_embed(torch.ones(B, self.seq_len, dtype=torch.long, device=x.device))
        x_emb = x_emb + x_type
        y_emb = y_emb + y_type

        # Concatenate x and y
        combined = torch.cat([x_emb, y_emb], dim=1)  # [B, 2*seq_len, hidden_dim]

        # Add positional encoding
        combined = combined + self.pos_embed

        # Transformer encoding
        encoded = self.transformer(combined)  # [B, 2*seq_len, hidden_dim]

        # Flatten and project
        flat = encoded.view(B, -1)  # [B, 2*seq_len*hidden_dim]
        return self.output_proj(flat)  # [B, hidden_dim]


class NoRecursionEncoder(nn.Module):
    """
    No-Recursion baseline model for plan-space RL.

    This model provides the same interface as TinyRecursiveReasoningModel_ACTV1
    (used_value, policy_dist) but WITHOUT the inner latent recursion.

    Key differences from TRM:
    - No z^(0) -> z^(n) inner recursion
    - Direct feedforward encoding of (x, y)
    - Simpler architecture, fewer parameters
    - The `n` parameter is IGNORED (no unrolling)

    This serves as an ablation baseline to test whether TRM's recursive
    refinement provides benefits over a direct approach.

    Args:
        config: NoRecEncoderConfig with architecture settings
    """

    def __init__(self, config: NoRecEncoderConfig):
        super().__init__()

        self.config = config

        # Create encoder based on type
        if config.encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        else:  # Default: MLP
            self.encoder = MLPEncoder(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )

        # RL heads (same as TRM)
        # For value head: z_dim = hidden_dim, x_dim = hidden_dim (we use z as combined embed)
        self.value_head = LatentValueHead(
            z_dim=config.hidden_dim,
            x_dim=config.hidden_dim,  # Combined x+y embed
            hidden_dim=config.hidden_dim,
            v_max=config.v_max,
        )

        # For policy head: we'll use z as all three inputs (z, x_embed, y_embed)
        # This simplification is acceptable since the encoder already combines (x, y)
        self.edit_policy = EditPolicyHead(
            latent_dim=config.hidden_dim,
            x_embed_dim=config.hidden_dim,
            y_embed_dim=config.hidden_dim,
            action_dim=config.rl_num_actions,
            hidden_dim=config.hidden_dim * 2,
        )

    def _get_inputs(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract input tensor from state dict."""
        if isinstance(x, dict):
            return x["inputs"]
        return x

    def _get_plan(self, y: Any) -> torch.Tensor:
        """Extract plan tensor."""
        if isinstance(y, dict):
            return y.get("inputs", y)
        return y

    def encode(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode (x, y) pair.

        Args:
            x: State dict with "inputs" key
            y: Plan tensor

        Returns:
            [B, hidden_dim] encoded representation
        """
        x_inputs = self._get_inputs(x)
        y_inputs = self._get_plan(y)

        # Ensure inputs are long tensors for embedding
        x_inputs = x_inputs.long()
        y_inputs = y_inputs.long()

        return self.encoder(x_inputs, y_inputs)

    def used_value(
        self,
        x: Dict[str, torch.Tensor],
        y: Any,
        n: int = 0,  # IGNORED - no unrolling
        z: Any = None,  # IGNORED - no latent state
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute value V(x, y).

        This matches the TRM interface but ignores n and z since
        there is no latent recursion.

        Args:
            x: State dict with "inputs" key
            y: Plan tensor
            n: IGNORED (no unrolling)
            z: IGNORED (no latent state)

        Returns:
            Tuple of (value, None) - None for compatibility with TRM
        """
        # Encode (x, y)
        z_enc = self.encode(x, y)  # [B, hidden_dim]

        # Value head expects (z, x_embed) - we use z_enc for both
        value = self.value_head(z_enc, z_enc)

        return value, None

    def policy_dist(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        n: int = 0,  # IGNORED - no unrolling
        action_mask: Optional[torch.Tensor] = None,
        z: Any = None,  # IGNORED - no latent state
    ) -> Tuple[Categorical, None]:
        """
        Get policy distribution over actions.

        This matches the TRM interface but ignores n and z since
        there is no latent recursion.

        Args:
            x: State dict with "inputs" key
            y: Plan tensor
            n: IGNORED (no unrolling)
            action_mask: Optional boolean mask for valid actions
            z: IGNORED (no latent state)

        Returns:
            Tuple of (Categorical distribution, None) - None for compatibility
        """
        # Encode (x, y)
        z_enc = self.encode(x, y)  # [B, hidden_dim]

        # Policy head expects (z, x_embed, y_embed) - we use z_enc for all
        dist = self.edit_policy(z_enc, z_enc, z_enc, action_mask=action_mask)

        return dist, None

    def init_latent(self, x: Any, y: Any) -> None:
        """Dummy method for compatibility with TRM interface."""
        return None

    def unroll_latent(self, x: Any, y: Any, n: int) -> Tuple[None, None]:
        """Dummy method for compatibility with TRM interface."""
        return None, None

    def eval_latent(self, x: Any, y: Any, n: int) -> None:
        """Dummy method for compatibility with TRM interface."""
        return None

    def continue_latent(self, z: Any, x: Any, y: Any, n: int) -> Tuple[None, None]:
        """Dummy method for compatibility with TRM interface."""
        return None, None
