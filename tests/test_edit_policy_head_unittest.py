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
        """Test that all-masked rows don't produce NaN (defensive fallback)."""
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

        dist = head(z, x_embed, y_embed, action_mask=mask)

        # Should not have NaN in probabilities
        probs = dist.probs
        self.assertFalse(torch.isnan(probs).any(), "NaN found in probabilities when all actions are masked")

        # Fully masked rows should have uniform distribution (fallback)
        self.assertTrue(torch.allclose(probs[0], torch.ones(action_dim) / action_dim),
            "Fully masked row should have uniform distribution")
        self.assertTrue(torch.allclose(probs[2], torch.ones(action_dim) / action_dim),
            "Fully masked row should have uniform distribution")

        # Non-fully-masked rows should have -inf for masked actions
        self.assertEqual(torch.isinf(dist.logits[1]).sum(), 0, "Row 1 should have no -inf")
        self.assertEqual(torch.isinf(dist.logits[3]).sum(), 0, "Row 3 should have no -inf")

    def test_edit_policy_head_1d_all_masked_edge_case(self):
        """Test that 1D all-masked mask doesn't produce NaN (defensive fallback)."""
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

        dist = head(z, x_embed, y_embed, action_mask=mask_1d)

        # Should not have NaN in probabilities
        probs = dist.probs
        self.assertFalse(torch.isnan(probs).any(), "NaN found in probabilities when 1D all-masked")

        # All rows should have uniform distribution (fallback)
        for i in range(batch_size):
            self.assertTrue(torch.allclose(probs[i], torch.ones(action_dim) / action_dim),
                f"Row {i} should have uniform distribution when 1D mask is all-False")


if __name__ == "__main__":
    unittest.main()
