"""Regression tests for CleanRL rollout boundary and masking behavior."""

from __future__ import annotations

import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from rl.cleanrl.dqn_trm import _compute_td_target, _sample_random_action
from rl.cleanrl.ppo_trm import (
    _apply_truncation_bootstrap,
    _compute_gae,
    _episode_done_flags,
    _validate_exact_interaction_budget,
)
from rl.cleanrl.ppo_trm import _evaluate_sudoku


class TestCleanRLRegressions(unittest.TestCase):
    @staticmethod
    def _offline_dataset(start: int, count: int):
        from rl.training_setup import OfflinePuzzleDataset

        samples = []
        for index in range(count):
            token = start + index
            inputs = torch.tensor([token, token + 1], dtype=torch.long)
            samples.append(
                {
                    "inputs": inputs,
                    "initial_plan": inputs.clone(),
                    "solution": inputs.clone(),
                    "puzzle_identifiers": torch.tensor(index),
                }
            )
        return OfflinePuzzleDataset(
            samples=samples,
            seq_len=2,
            vocab_size=32,
            num_identifiers=count,
        )

    def test_external_environment_rejects_zero_edit_budget(self) -> None:
        from external_baselines.sudoku4x4_env import Sudoku4x4ExternalEnv

        with self.assertRaisesRegex(ValueError, "max_edits must be at least 1"):
            Sudoku4x4ExternalEnv(max_edits=0)

    def test_trm_config_propagates_explicit_disabled_projection(self) -> None:
        from rl.cleanrl.trm_adapter import _build_trm_cfg

        bundle = SimpleNamespace(
            seq_len=4,
            num_identifiers=2,
            vocab_size=5,
            num_actions=21,
        )
        trm_config = _build_trm_cfg(
            {
                "batch_size": 2,
                "max_edits": 3,
                "latent_projection_mode": "disabled",
            },
            bundle,
        )

        self.assertEqual(trm_config["rl_latent_projection_mode"], "disabled")
        self.assertIsNone(trm_config["rl_latent_ball_radius"])

    def test_cleanrl_bundle_uses_disjoint_held_out_pool(self) -> None:
        from rl.cleanrl.trm_adapter import build_sudoku_bundle

        train = self._offline_dataset(1, 3)
        evaluation = self._offline_dataset(10, 2)

        def load_dataset(*, split, **_kwargs):
            dataset = train if split == "train" else evaluation
            return dataset, dataset.seq_len, dataset.vocab_size, dataset.num_identifiers

        with patch("rl.training_setup.build_dataset_from_paths", side_effect=load_dataset):
            bundle = build_sudoku_bundle(
                {
                    "dataset_paths": ["fixture"],
                    "batch_size": 3,
                    "eval_episodes": 2,
                }
            )

        self.assertIs(bundle.dataset, train)
        self.assertIs(bundle.eval_dataset, evaluation)
        self.assertEqual(bundle.train_split, "train")
        self.assertEqual(bundle.eval_split, "test")
        self.assertIsNotNone(bundle.train_pool_sha256)
        self.assertIsNotNone(bundle.eval_pool_sha256)
        train_ids = {
            int(sample["puzzle_identifiers"].item()) for sample in train.samples
        }
        eval_ids = {
            int(sample["puzzle_identifiers"].item())
            for sample in evaluation.samples
        }
        self.assertFalse(train_ids.intersection(eval_ids))

    def test_cleanrl_bundle_rejects_overlapping_or_small_eval_pool(self) -> None:
        from rl.cleanrl.trm_adapter import build_sudoku_bundle

        train = self._offline_dataset(1, 3)
        overlap = self._offline_dataset(1, 2)

        def overlapping_loader(*, split, **_kwargs):
            dataset = train if split == "train" else overlap
            return dataset, dataset.seq_len, dataset.vocab_size, dataset.num_identifiers

        with patch(
            "rl.training_setup.build_dataset_from_paths",
            side_effect=overlapping_loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "pools overlap"):
                build_sudoku_bundle(
                    {
                        "dataset_paths": ["fixture"],
                        "batch_size": 3,
                        "eval_episodes": 2,
                    }
                )

        evaluation = self._offline_dataset(10, 1)

        def small_loader(*, split, **_kwargs):
            dataset = train if split == "train" else evaluation
            return dataset, dataset.seq_len, dataset.vocab_size, dataset.num_identifiers

        with patch("rl.training_setup.build_dataset_from_paths", side_effect=small_loader):
            with self.assertRaisesRegex(RuntimeError, "smaller than eval_episodes"):
                build_sudoku_bundle(
                    {
                        "dataset_paths": ["fixture"],
                        "batch_size": 3,
                        "eval_episodes": 2,
                    }
                )

    def test_ppo_truncation_bootstraps_once_and_stops_gae_trace(self) -> None:
        gamma = 0.9
        terminal_value = 5.0

        boundary = _episode_done_flags(
            np.asarray([False], dtype=np.bool_),
            np.asarray([True], dtype=np.bool_),
        )
        self.assertEqual(boundary.tolist(), [True])

        # The rollout adds gamma * V(final_observation) to this reward. The
        # reset state's large value and advantage must not cross truncation.
        rewards = torch.tensor([[1.0 + gamma * terminal_value], [100.0]])
        values = torch.tensor([[2.0], [99.0]])
        dones = torch.tensor([[0.0], [1.0]])
        advantages = _compute_gae(
            rewards,
            values,
            dones,
            next_done=torch.tensor([0.0]),
            next_value=torch.tensor([123.0]),
            gamma=gamma,
            gae_lambda=0.95,
        )

        expected = 1.0 + gamma * terminal_value - 2.0
        self.assertAlmostEqual(advantages[0, 0].item(), expected)

    def test_ppo_gae_terminal_cuts_nonfinite_next_episode(self) -> None:
        advantages = _compute_gae(
            rewards=torch.tensor([[1.0], [2.0], [3.0]]),
            values=torch.tensor([[0.5], [1.5], [float("nan")]]),
            # dones[t + 1] is the boundary after transition t.
            dones=torch.tensor([[0.0], [0.0], [1.0]]),
            next_done=torch.tensor([0.0]),
            next_value=torch.tensor([5.0]),
            gamma=0.9,
            gae_lambda=0.8,
        )

        torch.testing.assert_close(
            advantages[:2, 0],
            torch.tensor([2.21, 0.5]),
        )
        self.assertTrue(torch.isnan(advantages[2, 0]))

    def test_ppo_truncation_reward_correction_handles_all_boundaries(self) -> None:
        class ValueAgent:
            def __init__(self) -> None:
                self.calls = 0

            def get_value(self, _obs):
                self.calls += 1
                return torch.tensor([[5.0]])

        agent = ValueAgent()
        corrected = _apply_truncation_bootstrap(
            torch.tensor([1.0, 2.0, 3.0]),
            np.asarray([False, True, True], dtype=np.bool_),
            np.asarray([True, False, True], dtype=np.bool_),
            [
                {"final_observation": np.asarray([9.0], dtype=np.float32)},
                {"final_observation": np.asarray([8.0], dtype=np.float32)},
                {"final_observation": np.asarray([7.0], dtype=np.float32)},
            ],
            agent,
            torch.device("cpu"),
            gamma=0.9,
        )

        torch.testing.assert_close(corrected, torch.tensor([5.5, 2.0, 3.0]))
        self.assertEqual(agent.calls, 1)

    def test_ppo_truncation_requires_final_observation(self) -> None:
        class ValueAgent:
            def get_value(self, _obs):
                return torch.tensor([[5.0]])

        with self.assertRaisesRegex(RuntimeError, "missing final_observation"):
            _apply_truncation_bootstrap(
                torch.tensor([1.0]),
                np.asarray([False], dtype=np.bool_),
                np.asarray([True], dtype=np.bool_),
                [{}],
                ValueAgent(),
                torch.device("cpu"),
                gamma=0.9,
            )

    def test_ppo_interaction_budget_never_rounds_down(self) -> None:
        self.assertEqual(_validate_exact_interaction_budget(80_000, 1, 64), 64)
        self.assertEqual(_validate_exact_interaction_budget(80_000, 4, 20), 80)
        with self.assertRaisesRegex(ValueError, "Exact interaction accounting"):
            _validate_exact_interaction_budget(80_001, 4, 20)

    def test_dqn_epsilon_random_samples_only_valid_actions(self) -> None:
        random.seed(1729)
        mask = torch.tensor([False, False, True, False, False, True])
        sampled = {_sample_random_action(6, mask) for _ in range(200)}
        self.assertEqual(sampled, {2, 5})

    def test_dqn_epsilon_random_with_one_valid_action_is_deterministic(self) -> None:
        mask = torch.tensor([False, False, False, True])
        self.assertEqual(_sample_random_action(4, mask), 3)

    def test_dqn_epsilon_random_rejects_empty_mask(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "action mask is empty"):
            _sample_random_action(4, torch.zeros(4, dtype=torch.bool))

    def test_dqn_terminal_target_ignores_nonfinite_successor_q(self) -> None:
        targets = _compute_td_target(
            rewards=torch.tensor([1.0, 2.0]),
            discounts=torch.tensor([0.9, 0.9]),
            next_q=torch.tensor([3.0, float("nan")]),
            dones=torch.tensor([False, True]),
        )

        torch.testing.assert_close(targets, torch.tensor([3.7, 2.0]))

    def test_cleanrl_action_mask_rejects_empty_and_wrong_shape(self) -> None:
        from rl.cleanrl.trm_adapter import apply_action_mask

        logits = torch.zeros(2, 4)
        with self.assertRaisesRegex(RuntimeError, r"batch rows \[1\]"):
            apply_action_mask(
                logits,
                torch.tensor(
                    [
                        [True, False, False, False],
                        [False, False, False, False],
                    ]
                ),
            )
        with self.assertRaisesRegex(ValueError, "action mask"):
            apply_action_mask(logits, torch.ones(2, 3, dtype=torch.bool))

    def test_q_eval_adapter_preserves_remaining_edits(self) -> None:
        from rl.cleanrl.trm_adapter import _QNetworkEvalAdapter

        class CapturingQNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Linear(1, 1)
                self.observed = None

            def forward(self, obs, action_mask=None):
                self.observed = obs
                return torch.zeros(1, 3)

        q_network = CapturingQNetwork()
        adapter = _QNetworkEvalAdapter(q_network)
        adapter.policy_dist(
            {
                "inputs": torch.zeros(1, 4),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
                "remaining_edits": torch.tensor([6]),
            },
            torch.zeros(1, 4),
            action_mask=torch.ones(3, dtype=torch.bool),
        )

        self.assertIsNotNone(q_network.observed)
        self.assertEqual(q_network.observed["remaining_edits"].tolist(), [6.0])

    def test_cleanrl_ppo_evaluates_held_out_bundle_dataset(self) -> None:
        train_dataset = object()
        eval_dataset = object()
        bundle = SimpleNamespace(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            checker_fn=object(),
            env_cfg=object(),
            task_config=None,
            eval_split="test",
            eval_pool_sha256="pool-hash",
        )
        agent = SimpleNamespace(model=object())
        with patch(
            "rl.evaluator.evaluate_plan_policy_with_scores",
            return_value=(1.0, 0.5, {}),
        ) as evaluate:
            metrics = _evaluate_sudoku(
                agent,
                bundle,
                {"eval_episodes": 1},
                torch.device("cpu"),
            )

        self.assertIs(evaluate.call_args.kwargs["dataset"], eval_dataset)
        self.assertEqual(metrics["eval_split"], "test")
        self.assertEqual(metrics["eval_pool_sha256"], "pool-hash")


if __name__ == "__main__":
    unittest.main()
