"""
Tests for the (x,y)-dependent latent initialization encoder - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, ZInitEncoder


def _tiny_trm_cfg_with_encoder(batch_size: int = 2, seq_len: int = 4, vocab_size: int = 16):
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=4,
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=False,
        rl_enable_z_init_encoder=True,  # Enable the encoder
    )


def _dummy_batch(batch_size: int, seq_len: int, vocab_size: int, num_identifiers: int = 4):
    return {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, num_identifiers, (batch_size,)),
    }


class TestZInitEncoder(unittest.TestCase):
    """Tests for ZInitEncoder."""

    def test_z_init_encoder_module_shapes(self):
        """Test ZInitEncoder produces correct output shapes."""
        batch_size = 4
        input_dim = 32
        hidden_dim = 64
        output_dim = 32
        seq_len = 8

        encoder = ZInitEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
        )

        x_embed = torch.randn(batch_size, input_dim)
        y_embed = torch.randn(batch_size, input_dim)

        z_init = encoder(x_embed, y_embed)

        self.assertEqual(z_init.shape, (batch_size, seq_len, output_dim))

    def test_z_init_encoder_different_inputs_produce_different_outputs(self):
        """Test that different (x, y) inputs produce different z^(0)."""
        torch.manual_seed(0)

        encoder = ZInitEncoder(
            input_dim=32,
            hidden_dim=64,
            output_dim=32,
            seq_len=8,
        )

        x_embed_1 = torch.randn(2, 32)
        y_embed_1 = torch.randn(2, 32)
        x_embed_2 = torch.randn(2, 32)
        y_embed_2 = torch.randn(2, 32)

        z_init_1 = encoder(x_embed_1, y_embed_1)
        z_init_2 = encoder(x_embed_2, y_embed_2)

        # Should not be identical (very unlikely with random inputs)
        self.assertFalse(torch.allclose(z_init_1, z_init_2))

    def test_trm_with_z_init_encoder_creates_encoder(self):
        """Test that TRM creates z_init_encoder when enabled."""
        cfg = _tiny_trm_cfg_with_encoder()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)

        self.assertIsNotNone(model.z_init_encoder)
        self.assertIsInstance(model.z_init_encoder, ZInitEncoder)

    def test_trm_without_z_init_encoder_has_none(self):
        """Test that TRM does not create z_init_encoder when disabled."""
        cfg = _tiny_trm_cfg_with_encoder()
        cfg["rl_enable_z_init_encoder"] = False
        model = TinyRecursiveReasoningModel_ACTV1(cfg)

        self.assertIsNone(model.z_init_encoder)

    def test_init_latent_with_encoder_produces_valid_carry(self):
        """Test that init_latent with encoder produces valid carry shapes."""
        torch.manual_seed(1)
        cfg = _tiny_trm_cfg_with_encoder()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        z0 = model.init_latent(x_batch, y_batch)

        expected_shape = (cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"])
        self.assertEqual(z0.z_H.shape, expected_shape)
        self.assertEqual(z0.z_L.shape, expected_shape)

    def test_init_latent_with_encoder_is_input_dependent(self):
        """Test that init_latent with encoder produces different outputs for different inputs."""
        torch.manual_seed(2)
        cfg = _tiny_trm_cfg_with_encoder()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch_1 = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch_1 = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        x_batch_2 = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch_2 = torch.ones(cfg["batch_size"], cfg["seq_len"], dtype=torch.long) * 5

        z0_1 = model.init_latent(x_batch_1, y_batch_1)
        z0_2 = model.init_latent(x_batch_2, y_batch_2)

        # With encoder, different inputs should produce different initial latents
        self.assertFalse(torch.allclose(z0_1.z_H, z0_2.z_H))

    def test_used_value_works_with_z_init_encoder(self):
        """Test that used_value works correctly when z_init_encoder is enabled."""
        torch.manual_seed(3)
        cfg = _tiny_trm_cfg_with_encoder()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        values, z_n = model.used_value(x_batch, y_batch, n=2)

        self.assertEqual(values.shape, (cfg["batch_size"],))
        self.assertFalse(torch.isnan(values).any())
        self.assertFalse(torch.isinf(values).any())
        self.assertIsNotNone(z_n)  # Check that z is returned

    def test_unroll_latent_works_with_z_init_encoder(self):
        """Test that unroll_latent works correctly when z_init_encoder is enabled."""
        torch.manual_seed(4)
        cfg = _tiny_trm_cfg_with_encoder()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        z_n, zs = model.unroll_latent(x_batch, y_batch, n=3)

        expected_shape = (cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"])
        self.assertEqual(z_n.z_H.shape, expected_shape)
        self.assertEqual(z_n.z_L.shape, expected_shape)
        self.assertEqual(len(zs), 4)  # z^(0) through z^(3)


if __name__ == "__main__":
    unittest.main()
