"""
Tests for UPI-TRM logging smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import dummy_checker
from rl.training_setup import (
    DummyPuzzleDataset,
    build_dataset_from_paths,
    offset_puzzle_identifiers,
)
from rl.upi_trm_trainer import UPITrmTrainer
from upi_trm_train import resume_from_checkpoint, save_checkpoint


def _num_actions(seq_len: int, vocab_size: int) -> int:
    return seq_len * vocab_size + 1


def _tiny_trm_cfg(seq_len: int, vocab_size: int, num_identifiers: int, batch_size: int):
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=max(num_identifiers, batch_size),
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=False,
        rl_enable_policy_head=True,
        rl_num_actions=_num_actions(seq_len, vocab_size),
    )


class TestUPITrmLoggingSmoke(unittest.TestCase):
    """Smoke tests for UPI-TRM logging and evaluation hooks."""

    @staticmethod
    def _write_dataset_root(root: Path, num_puzzles: int) -> None:
        split_dir = root / "test"
        split_dir.mkdir(parents=True)
        inputs = np.arange(num_puzzles * 2, dtype=np.int32).reshape(num_puzzles, 2)
        np.save(split_dir / "all__inputs.npy", inputs)
        np.save(split_dir / "all__labels.npy", inputs + 1)
        np.save(
            split_dir / "all__puzzle_identifiers.npy",
            np.arange(num_puzzles, dtype=np.int32),
        )
        np.save(
            split_dir / "all__puzzle_indices.npy",
            np.arange(num_puzzles + 1, dtype=np.int32),
        )
        np.save(
            split_dir / "all__group_indices.npy",
            np.arange(num_puzzles + 1, dtype=np.int32),
        )
        (split_dir / "dataset.json").write_text(
            json.dumps(
                {
                    "seq_len": 2,
                    "vocab_size": 16,
                    "pad_id": 0,
                    "ignore_label_id": None,
                    "blank_identifier_id": 0,
                    "num_puzzle_identifiers": num_puzzles,
                    "total_groups": num_puzzles,
                    "mean_puzzle_examples": 1,
                    "total_puzzles": num_puzzles,
                    "sets": ["all"],
                }
            )
        )

    def test_logging_and_eval_hooks_run(self):
        """Test that logging and evaluation hooks run without errors."""
        dataset = DummyPuzzleDataset(num_instances=5, seq_len=8, vocab_size=16)
        env_cfg = PlanEditEnvConfig(max_edits=3, gamma=0.9, reward_shaping=True, vocab_size=dataset.vocab_size)
        env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
        env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

        rl_cfg = RLConfig(
            batch_size=2,
            num_train_steps=3,
            rollout_episodes_per_step=1,
            max_edits=3,
            log_interval=1,
            eval_interval=2,
            eval_num_episodes=5,
            use_tqdm=False,
            gamma=env_cfg.gamma,
        )

        model_cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=rl_cfg.batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))

        for _ in range(rl_cfg.num_train_steps):
            metrics = trainer.train_step()
            self.assertIn("loss_value", metrics)
            self.assertIn("loss_policy", metrics)

        success_rate = trainer.evaluate_policy_success_rate(env_cfg=env_cfg, dataset=dataset, checker=dummy_checker)
        self.assertIsInstance(success_rate, float)
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)

    def test_dataset_bootstrap_fallback_is_explicit(self):
        with patch("rl.training_setup.PuzzleDataset", side_effect=RuntimeError("boom")):
            with patch("builtins.print") as mock_print:
                dataset, *_ = build_dataset_from_paths(
                    ["missing-dataset"], pool_size=4, allow_dummy_fallback=True
                )

        self.assertIsInstance(dataset, DummyPuzzleDataset)
        emitted = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        )
        self.assertIn("falling back to dummy dataset", emitted)

    def test_requested_dataset_failure_is_fatal_by_default(self):
        with patch("rl.training_setup.PuzzleDataset", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "Failed to load the requested dataset"):
                build_dataset_from_paths(["missing-dataset"], pool_size=4)

    def test_eval_puzzle_identifiers_can_be_made_disjoint(self):
        train = DummyPuzzleDataset(num_instances=3, seq_len=4, vocab_size=4)
        evaluation = DummyPuzzleDataset(num_instances=2, seq_len=4, vocab_size=4)
        offset_puzzle_identifiers(evaluation, train.num_identifiers)

        train_ids = {
            int(sample["puzzle_identifiers"].item()) for sample in train.samples
        }
        eval_ids = {
            int(sample["puzzle_identifiers"].item()) for sample in evaluation.samples
        }
        self.assertFalse(train_ids.intersection(eval_ids))
        self.assertEqual(min(eval_ids), train.num_identifiers)

    def test_multiple_dataset_roots_use_disjoint_identifier_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            self._write_dataset_root(first, num_puzzles=2)
            self._write_dataset_root(second, num_puzzles=3)
            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=0,
                    dataset_paths=[str(first), str(second)],
                    global_batch_size=5,
                    test_set_mode=True,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="test",
            )

            identifiers = []
            for _set_name, batch, valid_count in dataset:
                identifiers.extend(
                    int(value)
                    for value in batch["puzzle_identifiers"][:valid_count]
                )

        self.assertEqual(identifiers, [0, 1, 2, 3, 4])
        self.assertEqual(dataset.metadata.num_puzzle_identifiers, 5)

    def test_checkpoint_roundtrip_restores_target_replay_and_counters(self):
        dataset = DummyPuzzleDataset(num_instances=4, seq_len=8, vocab_size=16)
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.9,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )

        def make_trainer():
            env = PlanEditEnv(dataset, dummy_checker, env_cfg)
            env.set_stop_action_id(_num_actions(dataset.seq_len, dataset.vocab_size) - 1)
            cfg = RLConfig(batch_size=2, max_edits=3, gamma=env_cfg.gamma)
            model = TinyRecursiveReasoningModel_ACTV1(
                _tiny_trm_cfg(
                    dataset.seq_len,
                    dataset.vocab_size,
                    dataset.num_identifiers,
                    cfg.batch_size,
                )
            )
            return model, UPITrmTrainer(model, env, cfg, torch.device("cpu")), cfg

        model, trainer, cfg = make_trainer()
        trainer.collect_episode()
        trainer._train_step_count = 7
        trainer._env_step_count = 11
        with torch.no_grad():
            next(iter(trainer.target_model.parameters())).fill_(0.123)

        with tempfile.TemporaryDirectory() as tmp:
            provenance = {
                "train_split": "train",
                "eval_split": "test",
                "train_pool_sha256": "train-hash",
                "eval_pool_sha256": "eval-hash",
                "train_count": 5,
                "eval_count": 5,
                "seq_len": dataset.seq_len,
                "vocab_size": dataset.vocab_size,
                "num_identifiers": dataset.num_identifiers,
                "eval_puzzle_id_offset": dataset.num_identifiers,
            }
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=7,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            expected_random = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(3)
            checkpoint_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint_payload["checkpoint_schema_version"], 2)
            self.assertEqual(checkpoint_payload["dataset_provenance"], provenance)
            self.assertIn("rng_state", checkpoint_payload)

            random.seed(9999)
            np.random.seed(9999)
            torch.manual_seed(9999)
            self.assertEqual(checkpoint_payload["model_config"]["hidden_size"], 32)
            self.assertEqual(checkpoint_payload["model_config"]["H_cycles"], 1)
            restored_model, restored, _ = make_trainer()
            with patch("upi_trm_train.torch.load", wraps=torch.load) as checkpoint_load:
                start_step = resume_from_checkpoint(
                    checkpoint_path,
                    restored_model,
                    restored,
                    "cpu",
                    expected_dataset_provenance=provenance,
                )

            self.assertEqual(checkpoint_load.call_args.kwargs["map_location"], "cpu")
            self.assertFalse(checkpoint_load.call_args.kwargs["weights_only"])
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.rand()), expected_numpy)
            torch.testing.assert_close(torch.rand(3), expected_torch)

        self.assertEqual(start_step, 7)
        self.assertEqual(restored._train_step_count, 7)
        self.assertEqual(restored._env_step_count, 11)
        self.assertEqual(len(restored.replay), len(trainer.replay))
        restored_transition = restored.replay.storage[0]
        for value in (
            *restored_transition.x.values(),
            restored_transition.y,
            *restored_transition.x_next.values(),
            restored_transition.y_next,
            restored_transition.action,
            restored_transition.reward,
            restored_transition.done,
        ):
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
        torch.testing.assert_close(
            next(iter(restored.target_model.parameters())),
            next(iter(trainer.target_model.parameters())),
        )

    def test_resume_rejects_dataset_mismatch_before_mutating_model(self):
        dataset = DummyPuzzleDataset(num_instances=3, seq_len=4, vocab_size=4)
        env_cfg = PlanEditEnvConfig(
            max_edits=2,
            gamma=0.9,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )

        def make_trainer():
            env = PlanEditEnv(dataset, dummy_checker, env_cfg)
            env.set_stop_action_id(
                _num_actions(dataset.seq_len, dataset.vocab_size) - 1
            )
            cfg = RLConfig(batch_size=2, max_edits=2, gamma=env_cfg.gamma)
            model = TinyRecursiveReasoningModel_ACTV1(
                _tiny_trm_cfg(
                    dataset.seq_len,
                    dataset.vocab_size,
                    dataset.num_identifiers,
                    cfg.batch_size,
                )
            )
            return model, UPITrmTrainer(model, env, cfg, torch.device("cpu")), cfg

        model, trainer, cfg = make_trainer()
        saved_provenance = {"train_pool_sha256": "saved", "eval_pool_sha256": "eval"}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=1,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=saved_provenance,
            )
            restored_model, restored, _ = make_trainer()
            before = {
                key: value.detach().clone()
                for key, value in restored_model.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                resume_from_checkpoint(
                    checkpoint_path,
                    restored_model,
                    restored,
                    "cpu",
                    expected_dataset_provenance={
                        "train_pool_sha256": "different",
                        "eval_pool_sha256": "eval",
                    },
                )
            for key, value in restored_model.state_dict().items():
                torch.testing.assert_close(value, before[key])


if __name__ == "__main__":
    unittest.main()
