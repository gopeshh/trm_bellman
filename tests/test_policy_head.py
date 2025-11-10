"""Unit tests for EditPolicy with factorized (position, value) actions."""

import pytest

# Try to import torch, skip tests if not available
torch = pytest.importorskip("torch", reason="torch not installed")

from rl.policy_head import EditPolicy  # noqa: E402


@pytest.mark.unit
def test_edit_policy_initialization():
    """Test EditPolicy initializes correctly with position and value heads."""
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    assert policy.pos_head is not None
    assert policy.val_head is not None
    assert policy.pos_head.out_features == num_cells
    assert policy.val_head.out_features == num_vals


@pytest.mark.unit
def test_edit_policy_dist_shapes():
    """Test dist() returns correct log probability shapes."""
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)  # x_dim can be different

    lp_pos, lp_val = policy.dist(y, z_n, x)

    assert lp_pos.shape == (
        batch_size,
        num_cells,
    ), f"Expected {(batch_size, num_cells)}, got {lp_pos.shape}"
    assert lp_val.shape == (
        batch_size,
        num_vals,
    ), f"Expected {(batch_size, num_vals)}, got {lp_val.shape}"


@pytest.mark.unit
def test_edit_policy_dist_valid_log_probs():
    """Test dist() returns valid log probabilities that sum to 1."""
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    lp_pos, lp_val = policy.dist(y, z_n, x)

    # Log probabilities should be <= 0
    assert (lp_pos <= 0).all(), "Log probs should be <= 0"
    assert (lp_val <= 0).all(), "Log probs should be <= 0"

    # Probabilities should sum to 1 (in linear space)
    assert torch.allclose(lp_pos.exp().sum(-1), torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(lp_val.exp().sum(-1), torch.ones(batch_size), atol=1e-5)


@pytest.mark.unit
def test_edit_policy_sample_shapes():
    """Test sample() returns correct action and log_prob shapes."""
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    (pos, val), lp = policy.sample(y, z_n, x)

    assert pos.shape == (batch_size,), f"Expected {(batch_size,)}, got {pos.shape}"
    assert val.shape == (batch_size,), f"Expected {(batch_size,)}, got {val.shape}"
    assert lp.shape == (batch_size,), f"Expected {(batch_size,)}, got {lp.shape}"


@pytest.mark.unit
def test_edit_policy_sample_valid_indices():
    """Test sample() returns valid position and value indices."""
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    (pos, val), lp = policy.sample(y, z_n, x)

    # Positions should be in [0, num_cells)
    assert (pos >= 0).all() and (pos < num_cells).all(), "Position indices out of range"

    # Values should be in [0, num_vals)
    assert (val >= 0).all() and (val < num_vals).all(), "Value indices out of range"


@pytest.mark.unit
def test_edit_policy_log_prob_matches_sample():
    """Test log_prob() matches the log probability returned by sample().

    This is the key acceptance test: the log probability computed
    independently should match what sample() returned.
    """
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    # Sample action and get log prob from sample
    (pos, val), lp_sample = policy.sample(y, z_n, x)

    # Compute log prob independently
    lp_computed = policy.log_prob(y, z_n, x, (pos, val))

    # They should match
    assert torch.allclose(
        lp_sample, lp_computed, atol=1e-5
    ), f"Log probs don't match: sample={lp_sample}, computed={lp_computed}"


@pytest.mark.unit
def test_edit_policy_sample_with_position_mask():
    """Test sample() respects position masking."""
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    # Create mask that only allows first 3 positions
    mask_pos = torch.zeros(batch_size, num_cells)
    mask_pos[:, :3] = 1.0

    # Sample multiple times
    for _ in range(20):
        (pos, val), lp = policy.sample(y, z_n, x, mask_pos=mask_pos)
        # All sampled positions should be < 3
        assert (pos < 3).all(), f"Position {pos} violated mask constraint"


@pytest.mark.unit
def test_edit_policy_deterministic_across_calls():
    """Test that given same inputs and seed, policy is deterministic."""
    batch_size = 2
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    # Sample with same seed
    torch.manual_seed(42)
    (pos1, val1), lp1 = policy.sample(y, z_n, x)

    torch.manual_seed(42)
    (pos2, val2), lp2 = policy.sample(y, z_n, x)

    assert torch.equal(pos1, pos2), "Positions should be deterministic with same seed"
    assert torch.equal(val1, val2), "Values should be deterministic with same seed"
    assert torch.allclose(lp1, lp2), "Log probs should be deterministic with same seed"


@pytest.mark.unit
def test_edit_policy_log_prob_for_arbitrary_action():
    """Test log_prob() works for arbitrary valid actions."""
    batch_size = 4
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    # Create arbitrary valid actions
    pos = torch.randint(0, num_cells, (batch_size,))
    val = torch.randint(0, num_vals, (batch_size,))

    # Compute log prob
    lp = policy.log_prob(y, z_n, x, (pos, val))

    assert lp.shape == (batch_size,)
    assert not torch.isnan(lp).any(), "Log probs contain NaN"
    assert not torch.isinf(lp).any(), "Log probs contain Inf"
    # Log probs should be negative (or zero in edge cases)
    assert (lp <= 0).all(), "Log probs should be <= 0"


@pytest.mark.unit
def test_edit_policy_different_actions_different_probs():
    """Test that different actions typically have different probabilities."""
    batch_size = 1
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim)
    z_n = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, 16)

    # Compute log probs for different actions
    pos1, val1 = torch.tensor([0]), torch.tensor([0])
    pos2, val2 = torch.tensor([5]), torch.tensor([3])

    lp1 = policy.log_prob(y, z_n, x, (pos1, val1))
    lp2 = policy.log_prob(y, z_n, x, (pos2, val2))

    # Different actions should (almost always) have different probs
    # This could fail in rare cases, but very unlikely
    assert not torch.allclose(lp1, lp2, atol=1e-4), "Different actions should have different probs"


@pytest.mark.unit
def test_edit_policy_gradients_flow():
    """Test that gradients flow through the policy."""
    batch_size = 2
    z_dim, y_dim = 64, 32
    num_cells, num_vals = 10, 5
    policy = EditPolicy(z_dim, y_dim, num_cells, num_vals)

    y = torch.randn(batch_size, y_dim, requires_grad=True)
    z_n = torch.randn(batch_size, z_dim, requires_grad=True)
    x = torch.randn(batch_size, 16)

    # Create arbitrary action
    pos = torch.randint(0, num_cells, (batch_size,))
    val = torch.randint(0, num_vals, (batch_size,))

    # Compute log prob
    lp = policy.log_prob(y, z_n, x, (pos, val))
    loss = -lp.mean()

    # Backpropagate
    loss.backward()

    # Check gradients exist
    assert y.grad is not None, "Gradients should flow to y"
    assert z_n.grad is not None, "Gradients should flow to z_n"
    assert policy.pos_head.weight.grad is not None, "Gradients should flow to pos_head"
    assert policy.val_head.weight.grad is not None, "Gradients should flow to val_head"
