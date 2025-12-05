import torch

from models.edit_policy import EditPolicyHead


def test_edit_policy_head_shapes_and_masking():
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
    assert dist.logits.shape == (batch_size, action_dim)

    # With a mask that disallows the last action
    mask = torch.ones(batch_size, action_dim, dtype=torch.bool)
    mask[:, -1] = False
    dist_masked = head(z, x_embed, y_embed, action_mask=mask)
    assert torch.isinf(dist_masked.logits[:, -1]).all()

