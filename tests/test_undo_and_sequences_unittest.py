"""
Tests for sample_sequences in ReplayBuffer.

The UNDO action tests are skipped until the enable_undo feature is implemented
in PlanEditEnv.

Converted to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch

from rl.replay import ReplayBuffer, Transition


class TestSampleSequences(unittest.TestCase):
    """Test sample_sequences method in ReplayBuffer."""

    def setUp(self):
        """Create buffer with multiple complete episodes."""
        self.buffer = ReplayBuffer(capacity=1000)

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
                self.buffer.add(transition)

    def test_sample_sequences_shape(self):
        """Test that sample_sequences returns correct number of sequences."""
        sequences = self.buffer.sample_sequences(batch_size=8, K=5)
        self.assertEqual(len(sequences), 8)

    def test_sequence_length(self):
        """Test that sequences have at most K transitions."""
        K = 5
        sequences = self.buffer.sample_sequences(batch_size=8, K=K)

        for seq in sequences:
            self.assertLessEqual(len(seq), K)

    def test_sequence_contiguous(self):
        """Test that sequences are contiguous within episodes."""
        sequences = self.buffer.sample_sequences(batch_size=20, K=5)

        for seq in sequences:
            if len(seq) < 2:
                continue

            # All transitions should be from same episode
            ep_ids = [t.episode_id for t in seq]
            self.assertEqual(len(set(ep_ids)), 1)

            # Timesteps should be consecutive
            timesteps = [t.timestep for t in seq]
            for i in range(len(timesteps) - 1):
                self.assertEqual(timesteps[i + 1], timesteps[i] + 1)

    def test_sequence_truncated_at_done(self):
        """Test that sequences stop at terminal transitions."""
        sequences = self.buffer.sample_sequences(batch_size=50, K=20)

        for seq in sequences:
            # If sequence has a terminal transition, it should be the last one
            done_indices = [i for i, t in enumerate(seq) if t.done]
            if done_indices:
                self.assertEqual(done_indices[-1], len(seq) - 1)

    def test_sample_full_episodes(self):
        """Test sample_full_episodes method."""
        episodes = self.buffer.sample_full_episodes(num_episodes=3)

        self.assertEqual(len(episodes), 3)

        for ep in episodes:
            # Each episode should have 10 transitions
            self.assertEqual(len(ep), 10)

            # Last transition should be terminal
            self.assertTrue(ep[-1].done)

            # Timesteps should be ordered
            timesteps = [t.timestep for t in ep]
            self.assertEqual(timesteps, list(range(10)))

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

        with self.assertRaises(AssertionError):
            buffer.sample_sequences(batch_size=10, K=5)


# Note: UNDO action tests are skipped until enable_undo feature is implemented
# in PlanEditEnv. Once implemented, add TestUndoAction class here.


if __name__ == "__main__":
    unittest.main()
