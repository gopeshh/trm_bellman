"""Regression tests for CleanRL rollout boundary and masking behavior."""

from __future__ import annotations

import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl.cleanrl.dqn_trm import _sample_random_action
from rl.cleanrl.ppo_trm import _compute_gae, _episode_done_flags
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

    def test_dqn_epsilon_random_samples_only_valid_actions(self) -> None:
        random.seed(1729)
        mask = torch.tensor([False, False, True, False, False, True])
        sampled = {_sample_random_action(6, mask) for _ in range(200)}
        self.assertEqual(sampled, {2, 5})

    def test_dqn_epsilon_random_rejects_empty_mask(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "action mask is empty"):
            _sample_random_action(4, torch.zeros(4, dtype=torch.bool))

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
