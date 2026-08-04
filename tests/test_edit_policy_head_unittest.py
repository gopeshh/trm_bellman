"""
Tests for EditPolicyHead - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch

from models.edit_policy import EditPolicyHead


class TestEditPolicyHead(unittest.TestCase):
    """Tests for EditPolicyHead shapes and masking."""

    def test_edit_policy_head_shapes_and_masking(self):
        """Test shapes and masking behavior."""
        batch_size = 4
        latent_dim = 8
        x_dim = 16
        y_dim = 16
        action_dim = 10

        head = EditPolicyHead(
            latent_dim=latent_dim,
            x_embed_dim=x_dim,
            y_embed_dim=y_dim,
            action_dim=action_dim,
            hidden_dim=32,
        )

        z = torch.randn(batch_size, latent_dim)
        x_embed = torch.randn(batch_size, x_dim)
        y_embed = torch.randn(batch_size, y_dim)

        # No mask
        dist = head(z, x_embed, y_embed)
        self.assertEqual(dist.logits.shape, (batch_size, action_dim))

        # With a 2D mask that disallows the last action
        mask_2d = torch.ones(batch_size, action_dim, dtype=torch.bool)
        mask_2d[:, -1] = False
        dist_masked = head(z, x_embed, y_embed, action_mask=mask_2d)
        self.assertTrue(torch.isinf(dist_masked.logits[:, -1]).all())

        # With a 1D mask that disallows the last action (should be broadcasted)
        mask_1d = torch.ones(action_dim, dtype=torch.bool)
        mask_1d[-1] = False
        dist_masked_1d = head(z, x_embed, y_embed, action_mask=mask_1d)
        self.assertTrue(torch.isinf(dist_masked_1d.logits[:, -1]).all(),
            "1D mask should be broadcasted and applied to all batch elements")
        self.assertEqual(dist_masked_1d.logits.shape, (batch_size, action_dim))

    def test_edit_policy_head_all_masked_edge_case(self):
        """An invalid environment mask must not become a uniform policy."""
        batch_size = 4
        latent_dim = 8
        x_dim = 16
        y_dim = 16
        action_dim = 10

        head = EditPolicyHead(
            latent_dim=latent_dim,
            x_embed_dim=x_dim,
            y_embed_dim=y_dim,
            action_dim=action_dim,
            hidden_dim=32,
        )

        z = torch.randn(batch_size, latent_dim)
        x_embed = torch.randn(batch_size, x_dim)
        y_embed = torch.randn(batch_size, y_dim)

        # Edge case: all actions masked for some batch elements (2D mask)
        mask = torch.ones(batch_size, action_dim, dtype=torch.bool)
        mask[0, :] = False  # First batch element has all actions masked
        mask[2, :] = False  # Third batch element has all actions masked

        with self.assertRaisesRegex(RuntimeError, "no valid action.*0, 2"):
            head(z, x_embed, y_embed, action_mask=mask)

    def test_edit_policy_head_1d_all_masked_edge_case(self):
        """A broadcast all-false mask must fail for every batch row."""
        batch_size = 4
        latent_dim = 8
        x_dim = 16
        y_dim = 16
        action_dim = 10

        head = EditPolicyHead(
            latent_dim=latent_dim,
            x_embed_dim=x_dim,
            y_embed_dim=y_dim,
            action_dim=action_dim,
            hidden_dim=32,
        )

        z = torch.randn(batch_size, latent_dim)
        x_embed = torch.randn(batch_size, x_dim)
        y_embed = torch.randn(batch_size, y_dim)

        # Edge case: 1D mask with ALL actions masked (shouldn't happen, but test defensive behavior)
        mask_1d = torch.zeros(action_dim, dtype=torch.bool)  # All False

        with self.assertRaisesRegex(RuntimeError, "no valid action.*0, 1, 2, 3"):
            head(z, x_embed, y_embed, action_mask=mask_1d)


if __name__ == "__main__":
    unittest.main()
