"""
Tests for PlanEditEnv reward shaping - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig


class ScalarPuzzleDataset:
    def __init__(self):
        self.data = [
            {
                "inputs": torch.zeros(1, dtype=torch.float32),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def scalar_checker(x, y) -> float:
    if torch.is_tensor(y):
        val = y.view(-1)[0].item()
    else:
        val = float(y)
    return 1.0 if val == 3.0 else 0.0


class IncrementEnv(PlanEditEnv):
    def apply_edit(self, y, action: int, x):
        if torch.is_tensor(y):
            y_tensor = y.clone()
        else:
            y_tensor = torch.as_tensor(y)

        if action == self.stop_action_id:
            return y_tensor

        if action == 1:
            return (y_tensor + 1).view_as(y_tensor)
        return y_tensor


class TestPlanEditEnvRewardShaping(unittest.TestCase):
    """Tests for PlanEditEnv reward shaping."""

    def test_reward_shaping_matches_potential_form(self):
        """
        Test that reward shaping follows the canonical potential-based formula:
            r(s, a, s') = r_0 + γ·Φ(s') - Φ(s)
        where Φ(x, y) = c(x, y) is the checker score.
        """
        dataset = ScalarPuzzleDataset()
        gamma = 0.9
        cfg = PlanEditEnvConfig(max_edits=5, gamma=gamma, reward_shaping=True)
        env = IncrementEnv(dataset, scalar_checker, cfg)
        env.set_stop_action_id(stop_id=0)

        x, y = env.reset()
        old_y = y.clone()

        (_, y_next), reward, done, _ = env.step(action=1)

        phi_old = scalar_checker(x, old_y)
        phi_new = scalar_checker(x, y_next)
        r_0 = 0.0  # Base reward for intermediate steps

        # Canonical shaped reward: r = r_0 + gamma*Phi(s') - Phi(s)
        expected = r_0 + gamma * phi_new - phi_old

        self.assertFalse(done)
        self.assertLess(abs(reward - expected), 1e-6, f"Expected {expected}, got {reward}")

    def test_reward_shaping_with_different_gamma(self):
        """Test that γ factor is properly applied in reward shaping."""
        dataset = ScalarPuzzleDataset()

        # Test with different gamma values
        for gamma in [0.5, 0.9, 0.99]:
            cfg = PlanEditEnvConfig(max_edits=5, gamma=gamma, reward_shaping=True)
            env = IncrementEnv(dataset, scalar_checker, cfg)
            env.set_stop_action_id(stop_id=0)

            x, y = env.reset()
            old_y = y.clone()

            (_, y_next), reward, done, _ = env.step(action=1)

            phi_old = scalar_checker(x, old_y)
            phi_new = scalar_checker(x, y_next)
            expected = gamma * phi_new - phi_old

            self.assertLess(abs(reward - expected), 1e-6, f"gamma={gamma}: Expected {expected}, got {reward}")


if __name__ == "__main__":
    unittest.main()
