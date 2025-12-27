"""
Tests for baseline algorithms (PPO, A2C) and NoRecursionEncoder.

These tests verify the basic functionality of the baseline components
added for ICML 2026 paper comparison.
"""

import pytest
import torch
import torch.nn as nn

from models.norec_encoder import (
    NoRecEncoderConfig,
    NoRecursionEncoder,
    MLPEncoder,
    TransformerEncoder,
)


class TestNoRecEncoderConfig:
    """Test NoRecEncoderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NoRecEncoderConfig()
        assert config.vocab_size == 11
        assert config.seq_len == 81
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.encoder_type == "mlp"
        assert config.rl_num_actions == 892

    def test_custom_config(self):
        """Test custom configuration."""
        config = NoRecEncoderConfig(
            vocab_size=5,
            seq_len=16,
            hidden_dim=64,
            encoder_type="transformer",
            rl_num_actions=81,
        )
        assert config.vocab_size == 5
        assert config.seq_len == 16
        assert config.encoder_type == "transformer"

    def test_model_dump(self):
        """Test model_dump serialization."""
        config = NoRecEncoderConfig()
        dump = config.model_dump()
        assert isinstance(dump, dict)
        assert "vocab_size" in dump
        assert "hidden_dim" in dump


class TestMLPEncoder:
    """Test MLPEncoder module."""

    @pytest.fixture
    def encoder(self):
        return MLPEncoder(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=2,
        )

    def test_forward_shape(self, encoder):
        """Test output shape is correct."""
        batch_size = 4
        x = torch.randint(0, 5, (batch_size, 16))
        y = torch.randint(0, 5, (batch_size, 16))

        out = encoder(x, y)

        assert out.shape == (batch_size, 32)

    def test_forward_deterministic(self, encoder):
        """Test forward pass is deterministic in eval mode."""
        encoder.eval()
        x = torch.randint(0, 5, (2, 16))
        y = torch.randint(0, 5, (2, 16))

        out1 = encoder(x, y)
        out2 = encoder(x, y)

        torch.testing.assert_close(out1, out2)


class TestTransformerEncoder:
    """Test TransformerEncoder module."""

    @pytest.fixture
    def encoder(self):
        return TransformerEncoder(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )

    def test_forward_shape(self, encoder):
        """Test output shape is correct."""
        batch_size = 4
        x = torch.randint(0, 5, (batch_size, 16))
        y = torch.randint(0, 5, (batch_size, 16))

        out = encoder(x, y)

        assert out.shape == (batch_size, 32)


class TestNoRecursionEncoder:
    """Test NoRecursionEncoder module."""

    @pytest.fixture
    def config(self):
        return NoRecEncoderConfig(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=2,
            encoder_type="mlp",
            rl_num_actions=81,
        )

    @pytest.fixture
    def model(self, config):
        return NoRecursionEncoder(config)

    def test_encode(self, model):
        """Test encode method."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        z = model.encode(x, y)

        assert z.shape == (2, 32)

    def test_used_value(self, model):
        """Test used_value matches TRM interface."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        # n parameter should be ignored
        value, state = model.used_value(x, y, n=10)

        assert value.shape == (2, 1)
        assert state is None  # No latent state

    def test_policy_dist(self, model):
        """Test policy_dist matches TRM interface."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        # n parameter should be ignored
        dist, state = model.policy_dist(x, y, n=10)

        assert hasattr(dist, "sample")
        assert hasattr(dist, "log_prob")
        assert state is None  # No latent state

        # Sample should work
        action = dist.sample()
        assert action.shape == (2,)

    def test_policy_with_mask(self, model):
        """Test policy_dist with action mask."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        # Create mask that only allows first 10 actions
        mask = torch.zeros(2, 81, dtype=torch.bool)
        mask[:, :10] = True

        dist, _ = model.policy_dist(x, y, action_mask=mask)

        # Sample should be within valid range
        for _ in range(10):
            action = dist.sample()
            assert (action < 10).all()

    def test_dummy_methods(self, model):
        """Test compatibility methods return None."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        assert model.init_latent(x, y) is None
        assert model.unroll_latent(x, y, n=5) == (None, None)
        assert model.eval_latent(x, y, n=5) is None
        assert model.continue_latent(None, x, y, n=5) == (None, None)

    def test_transformer_encoder_type(self, config):
        """Test transformer encoder variant."""
        config.encoder_type = "transformer"
        model = NoRecursionEncoder(config)

        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        value, _ = model.used_value(x, y, n=0)
        assert value.shape == (2, 1)
