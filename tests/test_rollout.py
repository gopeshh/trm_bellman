"""Unit tests for K-step rollout with bootstrapped targets."""

import pytest

# Try to import torch, skip tests if not available
torch = pytest.importorskip("torch", reason="torch not installed")

from rl.rollout import rollout_k  # noqa: E402


class MockPolicy:
    """Mock policy for testing that returns deterministic actions."""

    def __init__(self, actions):
        """Initialize with sequence of actions.

        Args:
            actions: List of (pos, val) tuples to return
        """
        self.actions = actions
        self.idx = 0

    def sample(self, y, z_n, x):
        """Return next action in sequence.

        Args:
            y: Current output (B, 9, 9)
            z_n: Internal state (B, z_dim)
            x: Input (B, x_dim)
        """
        if self.idx < len(self.actions):
            action = self.actions[self.idx]
            self.idx += 1
            return action, torch.tensor([0.0])  # Dummy log prob
        # If we run out, return first action
        return self.actions[0], torch.tensor([0.0])


class MockValueTarget:
    """Mock value target that returns fixed values."""

    def __init__(self, value):
        """Initialize with fixed value.

        Args:
            value: Value to return
        """
        self.value = value

    def __call__(self, z_n, x):
        """Return fixed value for any state.

        Args:
            z_n: Internal state (B, z_dim)
            x: Input (B, x_dim)
        """
        batch_size = z_n.shape[0] if hasattr(z_n, "shape") else 1
        return torch.full((batch_size,), self.value, dtype=torch.float)


class MockEnv:
    """Mock environment with controlled rewards and transitions."""

    def __init__(self, rewards, terminal_step=None):
        """Initialize with sequence of rewards.

        Args:
            rewards: List of reward values
            terminal_step: Optional step at which episode terminates
        """
        self.rewards = rewards
        self.terminal_step = terminal_step
        self.step_count = 0

    def step(self, s, a, x):
        """Return next state and reward."""
        x_cur, y = s

        # Get reward
        if self.step_count < len(self.rewards):
            r = torch.tensor([self.rewards[self.step_count]], dtype=torch.float)
        else:
            r = torch.tensor([0.0], dtype=torch.float)

        # Check if done
        done = torch.tensor([0.0])
        if self.terminal_step is not None and self.step_count >= self.terminal_step:
            done = torch.tensor([1.0])

        # Simple state transition: just modify y slightly (doesn't matter for test)
        y_prime = y.clone()

        self.step_count += 1

        info = {}
        return (x_cur, y_prime), r, done, info


@pytest.mark.unit
def test_rollout_k_basic_accumulation():
    """Test rollout_k accumulates rewards correctly."""
    batch_size = 1
    K = 3
    gamma = 0.9

    # Setup
    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    # Rewards: [1.0, 2.0, 3.0]
    rewards = [1.0, 2.0, 3.0]
    env = MockEnv(rewards)

    # Fixed actions (doesn't matter for this test)
    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    # Target value = 5.0
    value_target = MockValueTarget(5.0)

    # Expected: G = 1.0 + 0.9*2.0 + 0.9²*3.0 + 0.9³*5.0
    # = 1.0 + 1.8 + 2.43 + 3.645 = 8.875
    expected_G = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 5.0

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    assert torch.allclose(
        G, torch.tensor([expected_G]), atol=1e-5
    ), f"Expected G={expected_G}, got {G[0].item()}"


@pytest.mark.unit
def test_rollout_k_gamma_one_equals_sum():
    """Test that with gamma=1, return equals sum of rewards plus bootstrap."""
    batch_size = 1
    K = 4
    gamma = 1.0

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    rewards = [1.0, 2.0, 3.0, 4.0]
    env = MockEnv(rewards)

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    bootstrap_value = 10.0
    value_target = MockValueTarget(bootstrap_value)

    # Expected: G = 1 + 2 + 3 + 4 + 10 = 20
    expected_G = sum(rewards) + bootstrap_value

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    assert torch.allclose(
        G, torch.tensor([expected_G]), atol=1e-5
    ), f"Expected G={expected_G}, got {G[0].item()}"


@pytest.mark.unit
def test_rollout_k_monte_carlo_when_k_equals_episode():
    """Test rollout_k equals MC return when K=episode length and gamma=1.

    Key acceptance test: When K equals episode length and gamma=1,
    the return should equal the Monte Carlo return.
    """
    batch_size = 1
    episode_length = 5
    K = episode_length  # Same length
    gamma = 1.0

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    # Episode terminates after 5 steps
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    env = MockEnv(rewards, terminal_step=episode_length - 1)

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    # Bootstrap value shouldn't matter much since episode terminates
    value_target = MockValueTarget(100.0)  # Large value to test

    # Monte Carlo return = sum of all rewards (gamma=1)
    mc_return = sum(rewards)

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    # Should equal MC return (bootstrap has minimal effect due to early termination)
    # The implementation breaks on done.any(), so we get MC return
    assert torch.allclose(
        G, torch.tensor([mc_return]), atol=1e-2
    ), f"Expected MC return={mc_return}, got {G[0].item()}"


@pytest.mark.unit
def test_rollout_k_truncated_plus_bootstrap():
    """Test rollout_k equals truncated MC + bootstrap when K < episode length.

    Key acceptance test: When K < episode length, return should equal
    truncated MC return plus bootstrapped value.
    """
    batch_size = 1
    K = 3  # Truncate early
    gamma = 0.95

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    # Long episode (10 steps) but we only roll out K=3
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    env = MockEnv(rewards, terminal_step=None)  # No early termination

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    bootstrap_value = 7.5
    value_target = MockValueTarget(bootstrap_value)

    # Expected: G = 1.0 + 0.95*2.0 + 0.95²*3.0 + 0.95³*7.5
    # = 1.0 + 1.9 + 2.7075 + 6.43... = truncated MC + bootstrap
    expected_G = (
        rewards[0] + gamma * rewards[1] + gamma**2 * rewards[2] + gamma**3 * bootstrap_value
    )

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    assert torch.allclose(
        G, torch.tensor([expected_G]), atol=1e-4
    ), f"Expected G={expected_G}, got {G[0].item()}"


@pytest.mark.unit
def test_rollout_k_zero_gamma():
    """Test rollout_k with gamma=0 only considers immediate reward."""
    batch_size = 1
    K = 3
    gamma = 0.0

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    rewards = [5.0, 10.0, 15.0]
    env = MockEnv(rewards)

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    value_target = MockValueTarget(100.0)

    # With gamma=0: G = r_0 + 0*r_1 + 0*r_2 + 0*V = 5.0
    expected_G = rewards[0]

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    assert torch.allclose(
        G, torch.tensor([expected_G]), atol=1e-5
    ), f"Expected G={expected_G}, got {G[0].item()}"


@pytest.mark.unit
def test_rollout_k_returns_final_state():
    """Test rollout_k returns the final state after K steps."""
    batch_size = 1
    K = 3
    gamma = 0.99

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    rewards = [1.0, 2.0, 3.0]
    env = MockEnv(rewards)

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    value_target = MockValueTarget(5.0)

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    # Check that final state is returned
    x_final, y_final = s_final
    assert x_final.shape == x.shape
    assert y_final.shape == y.shape


@pytest.mark.unit
def test_rollout_k_early_termination():
    """Test rollout_k handles early termination correctly."""
    batch_size = 1
    K = 5  # Plan for 5 steps
    gamma = 0.9

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    # Episode ends after 2 steps
    rewards = [1.0, 2.0, 0.0, 0.0, 0.0]
    env = MockEnv(rewards, terminal_step=1)

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    value_target = MockValueTarget(10.0)

    # Should break early after step 2
    # G = 1.0 + 0.9*2.0 + 0.9²*10.0 (bootstrap after early termination)
    expected_G = 1.0 + 0.9 * 2.0 + 0.9**2 * 10.0

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    assert torch.allclose(
        G, torch.tensor([expected_G]), atol=1e-4
    ), f"Expected G={expected_G} with early termination, got {G[0].item()}"


@pytest.mark.unit
def test_rollout_k_no_grad():
    """Test rollout_k operates in no_grad mode."""
    batch_size = 1
    K = 2
    gamma = 0.99

    x = torch.randn(batch_size, 10, requires_grad=True)
    y = torch.zeros(batch_size, 9, 9, requires_grad=True)
    z_n = torch.randn(batch_size, 128, requires_grad=True)  # Mock internal state
    s0 = (x, y, z_n)

    rewards = [1.0, 2.0]
    env = MockEnv(rewards)

    actions = [(torch.tensor([0]), torch.tensor([1]))] * K
    policy = MockPolicy(actions)

    value_target = MockValueTarget(5.0)

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    # G should not require gradients (no_grad mode)
    assert not G.requires_grad, "rollout_k should operate in no_grad mode"


@pytest.mark.unit
def test_rollout_k_batch_processing():
    """Test rollout_k handles batched inputs correctly."""
    batch_size = 2
    K = 3
    gamma = 0.9

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    # Both batch elements get same rewards for simplicity
    rewards = [1.0, 2.0, 3.0]
    env = MockEnv(rewards)

    actions = [(torch.tensor([0, 1]), torch.tensor([1, 2]))] * K
    policy = MockPolicy(actions)

    value_target = MockValueTarget(5.0)

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    # Check batch dimension is preserved
    assert G.shape == (batch_size,), f"Expected shape (2,), got {G.shape}"

    # Both should have same return (same rewards)
    assert torch.allclose(G[0], G[1], atol=1e-5), "Batch elements should have same return"


@pytest.mark.unit
def test_rollout_k_single_step():
    """Test rollout_k with K=1 (single step)."""
    batch_size = 1
    K = 1
    gamma = 0.95

    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9)
    z_n = torch.randn(batch_size, 128)  # Mock internal state
    s0 = (x, y, z_n)

    rewards = [3.0]
    env = MockEnv(rewards)

    actions = [(torch.tensor([0]), torch.tensor([1]))]
    policy = MockPolicy(actions)

    bootstrap_value = 7.0
    value_target = MockValueTarget(bootstrap_value)

    # G = 3.0 + 0.95*7.0 = 3.0 + 6.65 = 9.65
    expected_G = 3.0 + 0.95 * 7.0

    G, s_final = rollout_k(policy, value_target, env, s0, K, gamma)

    assert torch.allclose(
        G, torch.tensor([expected_G]), atol=1e-5
    ), f"Expected G={expected_G}, got {G[0].item()}"
