
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, call, patch

from utils.lipschitz import compute_exact_baseline_summation
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import sudoku_solution_checker

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
        self.env._enable_undo = False
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
        self.model.record_action_value_evaluations.reset_mock()
        
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
        self.assertEqual(
            self.model.record_action_value_evaluations.call_args_list,
            [call(self.batch_size), call(self.batch_size)],
        )
        
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

    def test_fixed_policy_probabilities_are_reused_at_reference_depth(self):
        action_mask = torch.zeros(self.batch_size, self.num_actions, dtype=torch.bool)
        action_mask[:, :2] = True
        fixed_probs = torch.zeros(self.batch_size, self.num_actions)
        fixed_probs[:, 0] = 0.75
        fixed_probs[:, 1] = 0.25

        _, q_all = compute_exact_baseline_summation(
            model=self.model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=8,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=action_mask,
            policy_probs=fixed_probs,
        )
        baseline, _ = compute_exact_baseline_summation(
            model=self.model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=8,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=action_mask,
            policy_probs=fixed_probs,
        )

        expected = 0.75 * q_all[:, 0] + 0.25 * q_all[:, 1]
        torch.testing.assert_close(baseline, expected)
        self.model.policy_dist.assert_not_called()

    def test_budget_terminal_action_does_not_bootstrap(self):
        x_batch = dict(self.x_batch)
        x_batch["remaining_edits"] = torch.ones(self.batch_size, dtype=torch.long)
        action_mask = torch.zeros(self.batch_size, self.num_actions, dtype=torch.bool)
        action_mask[:, 0] = True
        self.model.used_value.return_value = (
            torch.full((self.batch_size,), 100.0),
            None,
        )

        _, q_all = compute_exact_baseline_summation(
            model=self.model,
            x_batch=x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=1,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=action_mask,
        )

        torch.testing.assert_close(q_all[:, 0], torch.ones(self.batch_size))

    def test_nonterminal_successor_value_receives_decremented_clock(self):
        x_batch = dict(self.x_batch)
        x_batch["remaining_edits"] = torch.full(
            (self.batch_size,), 2, dtype=torch.long
        )
        action_mask = torch.zeros(self.batch_size, self.num_actions, dtype=torch.bool)
        action_mask[:, 0] = True
        self.model.used_value.reset_mock()

        compute_exact_baseline_summation(
            model=self.model,
            x_batch=x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=1,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=action_mask,
        )

        successor_x = self.model.used_value.call_args.args[0]
        torch.testing.assert_close(
            successor_x["remaining_edits"],
            torch.ones(self.batch_size, dtype=torch.long),
        )

    def test_persistent_successor_latent_is_used_for_every_valid_action(self):
        action_mask = torch.zeros(
            self.batch_size, self.num_actions, dtype=torch.bool
        )
        action_mask[:, :2] = True
        fixed_probs = torch.zeros(self.batch_size, self.num_actions)
        fixed_probs[:, 0] = 0.4
        fixed_probs[:, 1] = 0.6
        successor_latent = object()
        self.model.used_value.reset_mock()

        compute_exact_baseline_summation(
            model=self.model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            env=self.env,
            n=1,
            gamma=self.gamma,
            checker_fn=self.checker_fn,
            action_mask=action_mask,
            policy_probs=fixed_probs,
            successor_latent=successor_latent,
        )

        self.assertEqual(self.model.used_value.call_count, 2)
        for call in self.model.used_value.call_args_list:
            self.assertIs(call.kwargs["z"], successor_latent)

    def test_persistent_successor_latent_requires_fixed_policy(self):
        with self.assertRaisesRegex(ValueError, "requires fixed policy_probs"):
            compute_exact_baseline_summation(
                model=self.model,
                x_batch=self.x_batch,
                y_batch=self.y_batch,
                env=self.env,
                n=1,
                gamma=self.gamma,
                checker_fn=self.checker_fn,
                successor_latent=object(),
            )


class TestExactBaselineEnvironmentParity(unittest.TestCase):
    def test_solution_reward_matches_environment_transition(self):
        class Dataset:
            seq_len = 2
            vocab_size = 4
            num_identifiers = 1

            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return {
                    "inputs": torch.tensor([2, 1]),
                    "puzzle_identifiers": torch.tensor(0),
                    "initial_plan": torch.tensor([2, 1]),
                    "solution": torch.tensor([2, 3]),
                }

        class ZeroValueModel:
            @staticmethod
            def policy_dist(x, y, n, action_mask=None):
                probs = action_mask.to(torch.float32)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                return torch.distributions.Categorical(probs=probs), None

            @staticmethod
            def used_value(x, y, n):
                return torch.zeros(y.shape[0]), None

        env = PlanEditEnv(
            Dataset(),
            sudoku_solution_checker,
            PlanEditEnvConfig(
                max_edits=2,
                gamma=0.99,
                reward_shaping=True,
                solved_threshold=10.0,
                vocab_size=4,
                stop_action_mode="disabled",
            ),
        )
        env.set_stop_action_id(8)
        x, y = env.reset(idx=0)
        x_batch = {
            key: value.reshape(1) if value.ndim == 0 else value.unsqueeze(0)
            for key, value in x.items()
            if torch.is_tensor(value)
        }
        action = 7
        action_mask = torch.zeros(1, 9, dtype=torch.bool)
        action_mask[:, action] = True

        _, q_all = compute_exact_baseline_summation(
            model=ZeroValueModel(),
            x_batch=x_batch,
            y_batch=y.unsqueeze(0),
            env=env,
            n=1,
            gamma=0.99,
            checker_fn=sudoku_solution_checker,
            action_mask=action_mask,
        )
        (_, _), reward, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertAlmostEqual(q_all[0, action].item(), reward, places=6)
        self.assertAlmostEqual(reward, -5.0, places=6)

if __name__ == '__main__':
    unittest.main()
