"""
Tests for the refactored RL modules:
- rl/replay.py: ReplayBuffer, Transition
- rl/value_targets.py: K-step targets, GAE
- rl/task_config.py: TaskConfig hierarchy

These tests verify that the refactored code maintains the same behavior
as the original monolithic implementation.
"""

import pytest
import numpy as np
import torch

from puzzle_dataset import _sample_batch
from rl.replay import ReplayBuffer, Transition
from rl.value_targets import (
    compute_k_step_bootstrapped_target,
    compute_gae,
    compute_gae_trajectory,
    compute_empirical_bellman_residual,
    compute_td_advantage,
)
from rl.task_config import (
    TaskConfig,
    SudokuTaskConfig,
    DummyTaskConfig,
    ARCTaskConfig,
    get_task_config,
)
from rl.config import RLConfig


# =============================================================================
# Test: ReplayBuffer and Transition
# =============================================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer."""
    
    def test_add_and_len(self):
        """Test adding transitions and checking length."""
        buffer = ReplayBuffer(capacity=10)
        assert len(buffer) == 0
        
        transition = Transition(
            x={"inputs": torch.zeros(5)},
            y=torch.zeros(5),
            action=torch.tensor(0),
            reward=torch.tensor(1.0),
            x_next={"inputs": torch.zeros(5)},
            y_next=torch.zeros(5),
            done=torch.tensor(False),
            episode_id=0,
            timestep=0,
        )
        buffer.add(transition)
        assert len(buffer) == 1
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=5)
        
        for i in range(10):
            transition = Transition(
                x={"inputs": torch.tensor([i])},
                y=torch.tensor([i]),
                action=torch.tensor(i),
                reward=torch.tensor(float(i)),
                x_next={"inputs": torch.tensor([i])},
                y_next=torch.tensor([i]),
                done=torch.tensor(False),
                episode_id=i,
                timestep=0,
            )
            buffer.add(transition)
        
        assert len(buffer) == 5
        # Check that oldest entries were evicted
        assert buffer.storage[0].episode_id == 5
    
    def test_sample_batch(self):
        """Test sampling a batch of transitions."""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(20):
            transition = Transition(
                x={"inputs": torch.tensor([i])},
                y=torch.tensor([i]),
                action=torch.tensor(i),
                reward=torch.tensor(float(i)),
                x_next={"inputs": torch.tensor([i])},
                y_next=torch.tensor([i]),
                done=torch.tensor(False),
                episode_id=i,
                timestep=0,
            )
            buffer.add(transition)
        
        batch = buffer.sample_batch(10)
        assert len(batch) == 10
        assert all(isinstance(t, Transition) for t in batch)
    
    def test_sample_batch_insufficient(self):
        """Test that sampling fails when buffer is too small."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(5):
            transition = Transition(
                x={}, y=torch.zeros(1), action=torch.tensor(0),
                reward=torch.tensor(0.0), x_next={}, y_next=torch.zeros(1),
                done=torch.tensor(False), episode_id=i, timestep=0,
            )
            buffer.add(transition)
        
        with pytest.raises(AssertionError):
            buffer.sample_batch(10)


class TestPuzzleDatasetSampling:
    """Tests for deterministic sampling in puzzle_dataset helpers."""

    def test_sample_batch_uses_passed_rng(self):
        group_order = np.array([0, 1, 2], dtype=np.int64)
        puzzle_indices = np.array([0, 3, 7, 10], dtype=np.int64)
        group_indices = np.array([0, 1, 2, 3], dtype=np.int64)

        rng_a = np.random.Generator(np.random.Philox(seed=123))
        rng_b = np.random.Generator(np.random.Philox(seed=123))

        out_a = _sample_batch(
            rng=rng_a,
            group_order=group_order,
            puzzle_indices=puzzle_indices,
            group_indices=group_indices,
            start_index=0,
            global_batch_size=5,
        )
        out_b = _sample_batch(
            rng=rng_b,
            group_order=group_order,
            puzzle_indices=puzzle_indices,
            group_indices=group_indices,
            start_index=0,
            global_batch_size=5,
        )

        assert out_a[0] == out_b[0]
        assert np.array_equal(out_a[1], out_b[1])
        assert np.array_equal(out_a[2], out_b[2])
    
    def test_is_ready(self):
        """Test is_ready method."""
        buffer = ReplayBuffer(capacity=10)
        assert not buffer.is_ready(5)
        
        for i in range(5):
            transition = Transition(
                x={}, y=torch.zeros(1), action=torch.tensor(0),
                reward=torch.tensor(0.0), x_next={}, y_next=torch.zeros(1),
                done=torch.tensor(False), episode_id=i, timestep=0,
            )
            buffer.add(transition)
        
        assert buffer.is_ready(5)
        assert not buffer.is_ready(10)
    
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = ReplayBuffer(capacity=10)
        for i in range(5):
            buffer.add(Transition(
                x={}, y=torch.zeros(1), action=torch.tensor(0),
                reward=torch.tensor(0.0), x_next={}, y_next=torch.zeros(1),
                done=torch.tensor(False), episode_id=i, timestep=0,
            ))
        
        buffer.clear()
        assert len(buffer) == 0


# =============================================================================
# Test: Value Targets
# =============================================================================

class TestKStepTargets:
    """Tests for K-step bootstrapped targets."""
    
    def test_k_step_target_shape(self):
        """Test output shape of K-step targets."""
        batch_size, K = 4, 5
        rewards_K = torch.randn(batch_size, K)
        dones_K = torch.zeros(batch_size, K, dtype=torch.bool)
        steps_taken = torch.full((batch_size,), K, dtype=torch.long)
        v_K = torch.randn(batch_size)
        
        targets = compute_k_step_bootstrapped_target(
            rewards_K, dones_K, steps_taken, v_K, gamma=0.99, K=K
        )
        
        assert targets.shape == (batch_size,)
    
    def test_k_step_target_terminal(self):
        """Test that terminal states don't bootstrap."""
        batch_size, K = 2, 3
        rewards_K = torch.ones(batch_size, K)
        dones_K = torch.zeros(batch_size, K, dtype=torch.bool)
        dones_K[0, 1] = True  # First sample terminates at step 2
        steps_taken = torch.tensor([2, 3])
        v_K = torch.ones(batch_size) * 100  # Large bootstrap value
        
        targets = compute_k_step_bootstrapped_target(
            rewards_K, dones_K, steps_taken, v_K, gamma=0.99, K=K
        )
        
        # First sample should NOT bootstrap (done=True)
        # Second sample should bootstrap
        assert targets[0] < targets[1]
    
    def test_exact_k_step_rejects_incomplete_nonterminal_data(self):
        with pytest.raises(ValueError):
            compute_k_step_bootstrapped_target(
                torch.ones(1, 5),
                torch.zeros(1, 5, dtype=torch.bool),
                torch.tensor([3]),
                torch.ones(1),
                gamma=0.99,
                K=5,
                exact_k_step_targets=True,
            )


class TestGAE:
    """Tests for GAE computation."""
    
    def test_gae_shape(self):
        """Test output shape of GAE."""
        B = 8
        rewards = torch.randn(B)
        values = torch.randn(B)
        next_values = torch.randn(B)
        dones = torch.zeros(B, dtype=torch.bool)
        
        adv = compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95)
        assert adv.shape == (B,)
    
    def test_gae_lambda_zero_equals_td(self):
        """Test that λ=0 gives pure TD error."""
        B = 4
        rewards = torch.randn(B)
        values = torch.randn(B)
        next_values = torch.randn(B)
        dones = torch.zeros(B, dtype=torch.bool)
        
        gae_0 = compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.0)
        td = rewards + 0.99 * next_values - values
        
        assert torch.allclose(gae_0, td)
    
    def test_gae_trajectory(self):
        """Test full trajectory GAE."""
        T = 10
        rewards = torch.randn(T)
        values = torch.randn(T)
        dones = torch.zeros(T, dtype=torch.bool)
        dones[-1] = True
        
        adv = compute_gae_trajectory(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        assert adv.shape == (T,)


class TestTDAdvantage:
    """Tests for TD advantage computation."""
    
    def test_td_advantage_basic(self):
        """Test basic TD advantage computation."""
        B = 4
        rewards = torch.ones(B)
        values = torch.zeros(B)
        next_values = torch.ones(B)
        dones = torch.zeros(B, dtype=torch.bool)
        
        adv = compute_td_advantage(rewards, values, next_values, dones, gamma=0.99)
        expected = 1.0 + 0.99 * 1.0 - 0.0  # r + γV(s') - V(s)
        
        assert torch.allclose(adv, torch.full((B,), expected))
    
    def test_td_advantage_centered(self):
        """Test centered TD advantages have zero mean."""
        B = 100
        rewards = torch.randn(B)
        values = torch.randn(B)
        next_values = torch.randn(B)
        dones = torch.zeros(B, dtype=torch.bool)
        
        adv = compute_td_advantage(rewards, values, next_values, dones, gamma=0.99, centered=True)
        
        assert abs(adv.mean().item()) < 1e-6


class TestBellmanResidual:
    """Tests for Bellman residual computation."""
    
    def test_bellman_residual_keys(self):
        """Test that Bellman residual returns expected keys."""
        B = 4
        values = torch.randn(B)
        rewards = torch.randn(B)
        next_values = torch.randn(B)
        dones = torch.zeros(B, dtype=torch.bool)
        
        metrics = compute_empirical_bellman_residual(
            values, rewards, next_values, dones, gamma=0.99
        )
        
        assert "bellman_residual_mean" in metrics
        assert "bellman_residual_max" in metrics
        assert "bellman_residual_std" in metrics
    
    def test_bellman_residual_zero_for_optimal(self):
        """Test that Bellman residual is zero for optimal value function."""
        B = 4
        rewards = torch.ones(B)
        next_values = torch.zeros(B)  # Terminal
        dones = torch.ones(B, dtype=torch.bool)  # All terminal
        values = rewards.clone()  # V(s) = r for terminal states
        
        metrics = compute_empirical_bellman_residual(
            values, rewards, next_values, dones, gamma=0.99
        )
        
        assert abs(metrics["bellman_residual_mean"]) < 1e-6


# =============================================================================
# Test: Task Configs
# =============================================================================

class TestSudokuTaskConfig:
    """Tests for Sudoku task configuration."""
    
    def test_name(self):
        """Test task name."""
        config = SudokuTaskConfig()
        assert config.name == "sudoku"
    
    def test_is_given_cell(self):
        """Test given cell identification."""
        config = SudokuTaskConfig()
        assert not config.is_given_cell(1)  # Empty
        assert config.is_given_cell(2)      # Given clue
        assert config.is_given_cell(10)     # Given clue
    
    def test_checker_with_solution(self):
        """Test checker with solution available."""
        config = SudokuTaskConfig()
        x = {
            "inputs": torch.tensor([1, 2, 3, 4, 5]),
            "solution": torch.tensor([1, 2, 3, 4, 5]),
        }
        y = torch.tensor([1, 2, 3, 4, 5])  # Perfect match
        
        score = config.checker(x, y)
        assert score == 10.0  # Full score
    
    def test_checker_partial_match(self):
        """Test checker with partial match."""
        config = SudokuTaskConfig()
        x = {
            "inputs": torch.tensor([1, 2, 3, 4, 5]),
            "solution": torch.tensor([1, 2, 3, 4, 5]),
        }
        y = torch.tensor([1, 2, 9, 9, 9])  # 2/5 match
        
        score = config.checker(x, y)
        assert abs(score - 4.0) < 1e-6  # 2/5 * 10 = 4
    
    def test_is_solved(self):
        """Test solved detection."""
        config = SudokuTaskConfig()
        assert config.is_solved(10.0)
        assert config.is_solved(10.0 - 1e-7)
        assert not config.is_solved(9.9)


class TestDummyTaskConfig:
    """Tests for dummy task configuration."""
    
    def test_all_cells_editable(self):
        """Test that all cells are editable in dummy task."""
        config = DummyTaskConfig()
        assert not config.is_given_cell(0)
        assert not config.is_given_cell(1)
        assert not config.is_given_cell(100)
    
    def test_checker(self):
        """Test dummy checker."""
        config = DummyTaskConfig()
        x = {"inputs": torch.tensor([1.0, 2.0, 3.0])}
        y = torch.tensor([1.0, 2.0, 3.0])
        
        score = config.checker(x, y)
        assert score == 0.0  # Perfect match -> 0 L1 distance


class TestGetTaskConfig:
    """Tests for task config factory."""
    
    def test_get_sudoku(self):
        """Test getting Sudoku config."""
        config = get_task_config("sudoku")
        assert isinstance(config, SudokuTaskConfig)
    
    def test_get_dummy(self):
        """Test getting dummy config."""
        config = get_task_config("dummy")
        assert isinstance(config, DummyTaskConfig)
    
    def test_get_arc(self):
        """Test getting ARC config."""
        config = get_task_config("arc")
        assert isinstance(config, ARCTaskConfig)
    
    def test_case_insensitive(self):
        """Test that task names are case-insensitive."""
        assert isinstance(get_task_config("SUDOKU"), SudokuTaskConfig)
        assert isinstance(get_task_config("Dummy"), DummyTaskConfig)
    
    def test_unknown_task(self):
        """Test that unknown task raises error."""
        with pytest.raises(ValueError):
            get_task_config("unknown_task")


# =============================================================================
# Test: RLConfig New Options
# =============================================================================

class TestRLConfigNewOptions:
    """Tests for new RLConfig options."""
    
    def test_stop_action_mode_default(self):
        """Test default STOP action mode."""
        cfg = RLConfig()
        assert cfg.stop_action_mode == "noop"
    
    def test_stop_action_mode_options(self):
        """Test STOP action mode options."""
        cfg_terminal = RLConfig(stop_action_mode="terminal")
        assert cfg_terminal.stop_action_mode == "terminal"
        
        cfg_disabled = RLConfig(stop_action_mode="disabled")
        assert cfg_disabled.stop_action_mode == "disabled"
    
    def test_stop_action_penalty(self):
        """Test STOP action penalty default."""
        cfg = RLConfig()
        assert cfg.stop_action_penalty == -0.1
    
    def test_task_name_default(self):
        """Test default task name."""
        cfg = RLConfig()
        assert cfg.task_name == "sudoku"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
