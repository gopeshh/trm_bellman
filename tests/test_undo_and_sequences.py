"""
Tests for UNDO action in PlanEditEnv and sample_sequences in ReplayBuffer.

These tests verify the new features added for IMPLEMENTATION_GUIDELINE.md compliance.
"""

import pytest
import torch

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.replay import ReplayBuffer, Transition


# ====================
# UNDO Action Tests
# ====================

class DummyDataset:
    """Minimal dataset for testing."""

    def __init__(self, size: int = 10, seq_len: int = 16, vocab_size: int = 5):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create puzzle with some given cells (value > 1)
        inputs = torch.ones(self.seq_len, dtype=torch.long)
        # Set some cells as "given" (clues)
        inputs[0] = 2
        inputs[5] = 3
        inputs[10] = 4

        return {
            "inputs": inputs,
            "puzzle_identifiers": torch.tensor(idx),
        }


def dummy_checker(x, y):
    """Count correct cells (non-empty, matching some pattern)."""
    if isinstance(x, dict):
        x_inputs = x["inputs"]
    else:
        x_inputs = x

    if isinstance(y, dict):
        y_inputs = y.get("inputs", y)
    else:
        y_inputs = y

    # Simple checker: count cells with value > 1 (non-empty)
    return float((y_inputs > 1).sum().item())


class TestUndoAction:
    """Test UNDO action functionality in PlanEditEnv."""

    @pytest.fixture
    def config_with_undo(self):
        return PlanEditEnvConfig(
            max_edits=20,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=5,
            enable_undo=True,
            stop_action_mode="noop",
        )

    @pytest.fixture
    def config_without_undo(self):
        return PlanEditEnvConfig(
            max_edits=20,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=5,
            enable_undo=False,
            stop_action_mode="noop",
        )

    @pytest.fixture
    def env_with_undo(self, config_with_undo):
        dataset = DummyDataset()
        env = PlanEditEnv(dataset, dummy_checker, config_with_undo)
        # Action space: 16 * 5 + 2 = 82 (edits + STOP + UNDO)
        env.set_stop_action_id(80)  # STOP = 80, UNDO = 81
        return env

    @pytest.fixture
    def env_without_undo(self, config_without_undo):
        dataset = DummyDataset()
        env = PlanEditEnv(dataset, dummy_checker, config_without_undo)
        env.set_stop_action_id(80)
        return env

    def test_undo_action_id_set_automatically(self, env_with_undo):
        """Test that UNDO action ID is set when enable_undo=True."""
        assert env_with_undo.undo_action_id == 81  # STOP + 1

    def test_undo_action_id_not_set(self, env_without_undo):
        """Test that UNDO action ID is None when enable_undo=False."""
        assert env_without_undo.undo_action_id is None

    def test_history_initialized_on_reset(self, env_with_undo):
        """Test that edit history is initialized on reset."""
        x, y = env_with_undo.reset()

        assert len(env_with_undo._edit_history) == 1
        # First entry should match initial plan
        torch.testing.assert_close(env_with_undo._edit_history[0], y)

    def test_history_grows_on_edit(self, env_with_undo):
        """Test that history grows when edits are applied."""
        x, y = env_with_undo.reset()

        # Apply an edit (pos=1, tok=2 => action = 1*5 + 2 = 7)
        action = 7
        (x_next, y_next), reward, done, info = env_with_undo.step(action)

        assert len(env_with_undo._edit_history) == 2

    def test_undo_reverts_to_previous(self, env_with_undo):
        """Test that UNDO reverts to the previous plan state."""
        x, y_initial = env_with_undo.reset()
        y_initial_copy = y_initial.clone()

        # Apply an edit
        action = 7  # pos=1, tok=2
        (x1, y1), _, _, _ = env_with_undo.step(action)

        # Plan should be different
        assert not torch.equal(y1, y_initial_copy)

        # Apply UNDO
        undo_action = env_with_undo.undo_action_id
        (x2, y2), _, done, info = env_with_undo.step(undo_action)

        # Plan should be back to initial
        torch.testing.assert_close(y2, y_initial_copy)
        assert info["is_undo_action"] is True
        assert done is False  # UNDO doesn't terminate

    def test_undo_masked_at_initial(self, env_with_undo):
        """Test that UNDO is masked when there's nothing to undo."""
        x, y = env_with_undo.reset()

        mask = env_with_undo.get_action_mask()

        # UNDO should be masked (False) at initial state
        assert mask[env_with_undo.undo_action_id].item() is False

    def test_undo_valid_after_edit(self, env_with_undo):
        """Test that UNDO becomes valid after an edit."""
        x, y = env_with_undo.reset()

        # Initially masked
        mask = env_with_undo.get_action_mask()
        assert mask[env_with_undo.undo_action_id].item() is False

        # Apply an edit
        (x1, y1), _, _, _ = env_with_undo.step(7)

        # Now UNDO should be valid
        mask = env_with_undo.get_action_mask()
        assert mask[env_with_undo.undo_action_id].item() is True

    def test_multiple_undos(self, env_with_undo):
        """Test multiple consecutive UNDOs."""
        x, y0 = env_with_undo.reset()
        y0_copy = y0.clone()

        # Apply 3 edits
        (_, y1), _, _, _ = env_with_undo.step(7)   # Edit 1
        (_, y2), _, _, _ = env_with_undo.step(12)  # Edit 2
        (_, y3), _, _, _ = env_with_undo.step(17)  # Edit 3

        assert len(env_with_undo._edit_history) == 4

        # UNDO back to y2
        undo = env_with_undo.undo_action_id
        (_, y_after_undo1), _, _, _ = env_with_undo.step(undo)
        torch.testing.assert_close(y_after_undo1, y2)

        # UNDO back to y1
        (_, y_after_undo2), _, _, _ = env_with_undo.step(undo)
        torch.testing.assert_close(y_after_undo2, y1)

        # UNDO back to y0
        (_, y_after_undo3), _, _, _ = env_with_undo.step(undo)
        torch.testing.assert_close(y_after_undo3, y0_copy)

        # Now UNDO should be masked again
        mask = env_with_undo.get_action_mask()
        assert mask[env_with_undo.undo_action_id].item() is False


# ====================
# sample_sequences Tests
# ====================

class TestSampleSequences:
    """Test sample_sequences method in ReplayBuffer."""

    @pytest.fixture
    def buffer_with_episodes(self):
        """Create buffer with multiple complete episodes."""
        buffer = ReplayBuffer(capacity=1000)

        # Add 5 episodes of length 10 each
        for ep_id in range(5):
            for t in range(10):
                done = (t == 9)
                transition = Transition(
                    x={"inputs": torch.zeros(16)},
                    y=torch.zeros(16),
                    action=torch.tensor(t),
                    reward=torch.tensor(1.0),
                    x_next={"inputs": torch.zeros(16)},
                    y_next=torch.zeros(16),
                    done=torch.tensor(done),
                    episode_id=ep_id,
                    timestep=t,
                )
                buffer.add(transition)

        return buffer

    def test_sample_sequences_shape(self, buffer_with_episodes):
        """Test that sample_sequences returns correct number of sequences."""
        sequences = buffer_with_episodes.sample_sequences(batch_size=8, K=5)

        assert len(sequences) == 8

    def test_sequence_length(self, buffer_with_episodes):
        """Test that sequences have at most K transitions."""
        K = 5
        sequences = buffer_with_episodes.sample_sequences(batch_size=8, K=K)

        for seq in sequences:
            assert len(seq) <= K

    def test_sequence_contiguous(self, buffer_with_episodes):
        """Test that sequences are contiguous within episodes."""
        sequences = buffer_with_episodes.sample_sequences(batch_size=20, K=5)

        for seq in sequences:
            if len(seq) < 2:
                continue

            # All transitions should be from same episode
            ep_ids = [t.episode_id for t in seq]
            assert len(set(ep_ids)) == 1

            # Timesteps should be consecutive
            timesteps = [t.timestep for t in seq]
            for i in range(len(timesteps) - 1):
                assert timesteps[i + 1] == timesteps[i] + 1

    def test_sequence_truncated_at_done(self, buffer_with_episodes):
        """Test that sequences stop at terminal transitions."""
        sequences = buffer_with_episodes.sample_sequences(batch_size=50, K=20)

        for seq in sequences:
            # If sequence has a terminal transition, it should be the last one
            done_indices = [i for i, t in enumerate(seq) if t.done]
            if done_indices:
                assert done_indices[-1] == len(seq) - 1

    def test_sample_full_episodes(self, buffer_with_episodes):
        """Test sample_full_episodes method."""
        episodes = buffer_with_episodes.sample_full_episodes(num_episodes=3)

        assert len(episodes) == 3

        for ep in episodes:
            # Each episode should have 10 transitions
            assert len(ep) == 10

            # Last transition should be terminal
            assert ep[-1].done

            # Timesteps should be ordered
            timesteps = [t.timestep for t in ep]
            assert timesteps == list(range(10))

    def test_not_enough_sequences(self):
        """Test error when buffer doesn't have enough data."""
        buffer = ReplayBuffer(capacity=10)

        # Add only 3 transitions
        for t in range(3):
            transition = Transition(
                x={"inputs": torch.zeros(16)},
                y=torch.zeros(16),
                action=torch.tensor(t),
                reward=torch.tensor(1.0),
                x_next={"inputs": torch.zeros(16)},
                y_next=torch.zeros(16),
                done=torch.tensor(t == 2),
                episode_id=0,
                timestep=t,
            )
            buffer.add(transition)

        with pytest.raises(AssertionError, match="Not enough valid sequence starts"):
            buffer.sample_sequences(batch_size=10, K=5)
