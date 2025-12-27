"""
Tests for baseline algorithms (PPO, A2C) and NoRecursionEncoder.

These tests verify the basic functionality of the baseline components
added for ICML 2026 paper comparison.

Converted to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch
import torch.nn as nn

from models.norec_encoder import (
    NoRecEncoderConfig,
    NoRecursionEncoder,
    MLPEncoder,
    TransformerEncoder,
)


class TestNoRecEncoderConfig(unittest.TestCase):
    """Test NoRecEncoderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NoRecEncoderConfig()
        self.assertEqual(config.vocab_size, 11)
        self.assertEqual(config.seq_len, 81)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.encoder_type, "mlp")
        self.assertEqual(config.rl_num_actions, 892)

    def test_custom_config(self):
        """Test custom configuration."""
        config = NoRecEncoderConfig(
            vocab_size=5,
            seq_len=16,
            hidden_dim=64,
            encoder_type="transformer",
            rl_num_actions=81,
        )
        self.assertEqual(config.vocab_size, 5)
        self.assertEqual(config.seq_len, 16)
        self.assertEqual(config.encoder_type, "transformer")

    def test_model_dump(self):
        """Test model_dump serialization."""
        config = NoRecEncoderConfig()
        dump = config.model_dump()
        self.assertIsInstance(dump, dict)
        self.assertIn("vocab_size", dump)
        self.assertIn("hidden_dim", dump)


class TestMLPEncoder(unittest.TestCase):
    """Test MLPEncoder module."""

    def setUp(self):
        self.encoder = MLPEncoder(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=2,
        )

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size = 4
        x = torch.randint(0, 5, (batch_size, 16))
        y = torch.randint(0, 5, (batch_size, 16))

        out = self.encoder(x, y)

        self.assertEqual(out.shape, (batch_size, 32))

    def test_forward_deterministic(self):
        """Test forward pass is deterministic in eval mode."""
        self.encoder.eval()
        x = torch.randint(0, 5, (2, 16))
        y = torch.randint(0, 5, (2, 16))

        out1 = self.encoder(x, y)
        out2 = self.encoder(x, y)

        torch.testing.assert_close(out1, out2)


class TestTransformerEncoder(unittest.TestCase):
    """Test TransformerEncoder module."""

    def setUp(self):
        self.encoder = TransformerEncoder(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size = 4
        x = torch.randint(0, 5, (batch_size, 16))
        y = torch.randint(0, 5, (batch_size, 16))

        out = self.encoder(x, y)

        self.assertEqual(out.shape, (batch_size, 32))


class TestNoRecursionEncoder(unittest.TestCase):
    """Test NoRecursionEncoder module."""

    def setUp(self):
        self.config = NoRecEncoderConfig(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=2,
            encoder_type="mlp",
            rl_num_actions=81,
        )
        self.model = NoRecursionEncoder(self.config)

    def test_encode(self):
        """Test encode method."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        z = self.model.encode(x, y)

        self.assertEqual(z.shape, (2, 32))

    def test_used_value(self):
        """Test used_value matches TRM interface."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        # n parameter should be ignored
        value, state = self.model.used_value(x, y, n=10)

        self.assertEqual(value.shape, (2,))  # LatentValueHead returns [B]
        self.assertIsNone(state)  # No latent state

    def test_policy_dist(self):
        """Test policy_dist matches TRM interface."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        # n parameter should be ignored
        dist, state = self.model.policy_dist(x, y, n=10)

        self.assertTrue(hasattr(dist, "sample"))
        self.assertTrue(hasattr(dist, "log_prob"))
        self.assertIsNone(state)  # No latent state

        # Sample should work
        action = dist.sample()
        self.assertEqual(action.shape, (2,))

    def test_policy_with_mask(self):
        """Test policy_dist with action mask."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        # Create mask that only allows first 10 actions
        mask = torch.zeros(2, 81, dtype=torch.bool)
        mask[:, :10] = True

        dist, _ = self.model.policy_dist(x, y, action_mask=mask)

        # Sample should be within valid range
        for _ in range(10):
            action = dist.sample()
            self.assertTrue((action < 10).all())

    def test_dummy_methods(self):
        """Test compatibility methods return None."""
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        self.assertIsNone(self.model.init_latent(x, y))
        self.assertEqual(self.model.unroll_latent(x, y, n=5), (None, None))
        self.assertIsNone(self.model.eval_latent(x, y, n=5))
        self.assertEqual(self.model.continue_latent(None, x, y, n=5), (None, None))

    def test_transformer_encoder_type(self):
        """Test transformer encoder variant."""
        config = NoRecEncoderConfig(
            vocab_size=5,
            seq_len=16,
            hidden_dim=32,
            num_layers=2,
            encoder_type="transformer",
            rl_num_actions=81,
        )
        model = NoRecursionEncoder(config)

        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))

        value, _ = model.used_value(x, y, n=0)
        self.assertEqual(value.shape, (2,))  # LatentValueHead returns [B]


if __name__ == "__main__":
    unittest.main()
