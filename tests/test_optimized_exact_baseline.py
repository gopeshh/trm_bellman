
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from utils.lipschitz import compute_exact_baseline_summation

class MockDist:
    def __init__(self, probs):
        self.probs = probs

class TestOptimizedExactBaseline(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 10
        self.vocab_size = 2
        self.stop_action_id = 9
        
        # Mock Environment
        self.env = MagicMock()
        self.env.vocab_size = self.vocab_size
        self.env.stop_action_id = self.stop_action_id
        self.env.is_stop_terminal.return_value = False
        self.env.config.solved_threshold = None
        self.env._stop_mode = "noop"
        
        # Mock Model
        self.model = MagicMock()
        self.probs = torch.ones(self.batch_size, self.num_actions) / self.num_actions
        self.model.policy_dist.return_value = (MockDist(self.probs), None)
        self.model.used_value.return_value = (torch.zeros(self.batch_size), None)
        
        # Inputs
        self.x_batch = {"inputs": torch.zeros(self.batch_size, 5)}
        self.y_batch = torch.zeros(self.batch_size, 5)
        self.n = 1
        self.gamma = 0.99
        self.checker_fn = lambda x, y: 0.0
        
        # Helper mocks
        self.patcher1 = patch('utils.lipschitz._apply_edit_batch')
        self.mock_apply_edit = self.patcher1.start()
        self.mock_apply_edit.return_value = self.y_batch # No-op edit
        
        self.patcher2 = patch('utils.lipschitz._compute_phi_batch')
        self.mock_compute_phi = self.patcher2.start()
        self.mock_compute_phi.return_value = (torch.zeros(self.batch_size), torch.zeros(self.batch_size), torch.zeros(self.batch_size, dtype=torch.bool))
        
        self.patcher3 = patch('utils.lipschitz._compute_batch_reward_via_env')
        self.mock_compute_reward = self.patcher3.start()
        self.mock_compute_reward.return_value = torch.ones(self.batch_size) # Reward = 1.0 everywhere

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()

    def test_optimization_skips_masked_actions(self):
        """
        Verify that compute_exact_baseline_summation skips loop iterations 
        for actions that are completely masked out.
        """
        # Create a mask where only actions 0 and 9 are valid
        # Actions 1-8 are invalid for ALL batch elements
        action_mask = torch.zeros(self.batch_size, self.num_actions, dtype=torch.bool)
        action_mask[:, 0] = True
        action_mask[:, 9] = True # STOP
        
        # Reset model call count
        self.model.used_value.reset_mock()
        
        # Run computation
        exact_baseline, q_all = compute_exact_baseline_summation(
            model=self.model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=self.n,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=action_mask
        )
        
        # Verify result shape
        self.assertEqual(exact_baseline.shape, (self.batch_size,))
        self.assertEqual(q_all.shape, (self.batch_size, self.num_actions))
        
        # Verify call count
        # Should be called for action 0.
        # Action 9 is STOP. If STOP is terminal, no used_value call. If noop, yes used_value call.
        # env.is_stop_terminal returns False (noop). So called for 0 and 9.
        # Total calls should be 2.
        # If optimization fails, it would be 10 calls.
        self.assertEqual(self.model.used_value.call_count, 2, 
                         f"Expected 2 calls (optimized), got {self.model.used_value.call_count}")
        
        # Verify masked values are -inf
        self.assertTrue(torch.isneginf(q_all[:, 1]).all())
        self.assertTrue(torch.isneginf(q_all[:, 8]).all())
        
        # Verify valid values are finite
        self.assertTrue(torch.isfinite(q_all[:, 0]).all())
        self.assertTrue(torch.isfinite(q_all[:, 9]).all())

    def test_no_mask_runs_all(self):
        """Verify fallback to full loop if mask is None."""
        self.model.used_value.reset_mock()
        
        compute_exact_baseline_summation(
            model=self.model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=self.n,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=None
        )
        
        # Should run for all 10 actions
        self.assertEqual(self.model.used_value.call_count, 10)

if __name__ == '__main__':
    unittest.main()

