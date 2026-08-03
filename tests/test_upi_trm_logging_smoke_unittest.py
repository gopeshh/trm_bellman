"""
Tests for UPI-TRM logging smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
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
from utils.dataset_provenance import (
    build_dataset_provenance,
    dataset_sample_sha256s,
)


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

    @staticmethod
    def _make_persistent_budget_trainer():
        class FixedDataset:
            seq_len = 2
            vocab_size = 4
            num_identifiers = 1

            def __init__(self):
                self.samples = [
                    {
                        "inputs": torch.ones(2, dtype=torch.long),
                        "puzzle_identifiers": torch.tensor(0, dtype=torch.long),
                        "initial_plan": torch.ones(2, dtype=torch.long),
                    }
                ]

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = FixedDataset()
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.9,
            reward_shaping=False,
            task_type="dummy",
            vocab_size=dataset.vocab_size,
            stop_action_mode="disabled",
            fail_terminal_reward=-1.0,
        )
        env = PlanEditEnv(
            dataset=dataset,
            checker=lambda _x, _y: 0.0,
            config=env_cfg,
        )
        env.set_stop_action_id(
            stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1
        )
        cfg = RLConfig(
            batch_size=8,
            replay_capacity=32,
            rollout_episodes_per_step=1,
            max_edits=3,
            gamma=env_cfg.gamma,
            K=1,
            inner_unroll_n=1,
            episodic_latent=False,
            task_name="dummy",
            solved_threshold=None,
            stop_action_mode="disabled",
            reward_shaping=False,
            fail_terminal_reward=-1.0,
            enable_contraction=False,
            latent_ball_radius=0.0,
            lr_schedule="constant",
            use_tqdm=False,
        )
        model = TinyRecursiveReasoningModel_ACTV1(
            _tiny_trm_cfg(
                dataset.seq_len,
                dataset.vocab_size,
                dataset.num_identifiers,
                cfg.batch_size,
            )
        )
        trainer = UPITrmTrainer(model, env, cfg, torch.device("cpu"))
        return model, trainer, cfg

    @staticmethod
    def _stub_updates(trainer):
        trainer.value_update = MagicMock(return_value={"loss_value": 0.0})
        trainer.policy_update = MagicMock(return_value={"loss_policy": 0.0})

    @staticmethod
    def _checkpoint_provenance(trainer, *, generation_seed=123):
        dataset = trainer.env.dataset
        stop_action_id = trainer.env.stop_action_id
        return build_dataset_provenance(
            builder_name="tests.FixedDataset",
            builder_version=1,
            generation_seed=generation_seed,
            train_record_sha256s=dataset_sample_sha256s(dataset),
            eval_record_sha256s=dataset_sample_sha256s(dataset),
            train_split="unit-train",
            eval_split="unit-eval",
            environment_config=dict(vars(trainer.env.config)),
            action_mask_config={
                "task_config_class": (
                    type(trainer.env.task_config).__name__
                    if trainer.env.task_config is not None
                    else None
                ),
                "disable_constraint_masking": (
                    trainer.env.config.disable_constraint_masking
                ),
                "stop_action_mode": trainer.env._stop_mode,
                "stop_action_id": stop_action_id,
                "enable_undo": trainer.env._enable_undo,
                "undo_action_id": trainer.env.undo_action_id,
                "vocab_size": trainer.env.vocab_size,
                "num_actions": (
                    trainer.env.undo_action_id + 1
                    if trainer.env.undo_action_id is not None
                    else stop_action_id + 1
                ),
                "masked_token_ids": [0, 1],
            },
        )

    def _assert_replay_equal(self, expected, actual):
        self.assertEqual(len(expected.replay), len(actual.replay))
        for left, right in zip(expected.replay.storage, actual.replay.storage):
            self.assertEqual(left.episode_id, right.episode_id)
            self.assertEqual(left.timestep, right.timestep)
            self.assertTrue(torch.equal(left.action, right.action))
            self.assertTrue(torch.equal(left.reward, right.reward))
            self.assertTrue(torch.equal(left.done, right.done))
            self.assertTrue(torch.equal(left.y, right.y))
            self.assertTrue(torch.equal(left.y_next, right.y_next))
            for key in left.x:
                self.assertTrue(torch.equal(left.x[key], right.x[key]))
                self.assertTrue(torch.equal(left.x_next[key], right.x_next[key]))
            self.assertIsNotNone(left.latent)
            self.assertIsNotNone(right.latent)
            self.assertIsNotNone(left.next_latent)
            self.assertIsNotNone(right.next_latent)
            torch.testing.assert_close(left.latent.z_H, right.latent.z_H)
            torch.testing.assert_close(left.latent.z_L, right.latent.z_L)
            torch.testing.assert_close(left.next_latent.z_H, right.next_latent.z_H)
            torch.testing.assert_close(left.next_latent.z_L, right.next_latent.z_L)

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

    def test_exact_cap_pauses_persistent_episode_without_optimizing(self):
        torch.manual_seed(101)
        _, trainer, _ = self._make_persistent_budget_trainer()
        self._stub_updates(trainer)

        first_metrics = trainer.train_step(max_env_steps_to_collect=1)
        self.assertEqual(trainer.get_env_step_count(), 1)
        self.assertEqual(trainer._train_step_count, 0)
        self.assertEqual(trainer._next_episode_id, 0)
        self.assertIsNotNone(trainer._active_episode)
        self.assertEqual(first_metrics["optimization_performed"], 0.0)
        trainer.value_update.assert_not_called()
        trainer.policy_update.assert_not_called()

        first = trainer.replay.storage[0]
        second_metrics = trainer.train_step(max_env_steps_to_collect=1)
        second = trainer.replay.storage[1]
        self.assertEqual(second.episode_id, first.episode_id)
        self.assertEqual(second.timestep, first.timestep + 1)
        self.assertTrue(torch.equal(first.y_next, second.y))
        for key in first.x_next:
            self.assertTrue(torch.equal(first.x_next[key], second.x[key]))
        torch.testing.assert_close(first.next_latent.z_H, second.latent.z_H)
        torch.testing.assert_close(first.next_latent.z_L, second.latent.z_L)
        self.assertEqual(second_metrics["optimization_performed"], 0.0)
        trainer.value_update.assert_not_called()
        trainer.policy_update.assert_not_called()

        final_metrics = trainer.train_step(max_env_steps_to_collect=1)
        self.assertEqual(final_metrics["optimization_performed"], 1.0)
        self.assertEqual(trainer.get_env_step_count(), 3)
        self.assertEqual(trainer._train_step_count, 1)
        self.assertEqual(trainer._next_episode_id, 1)
        self.assertIsNone(trainer._active_episode)
        trainer.value_update.assert_called_once()
        trainer.policy_update.assert_called_once()

    def test_collection_pause_schedule_matches_uninterrupted_episode(self):
        torch.manual_seed(202)
        _, uninterrupted, _ = self._make_persistent_budget_trainer()
        self._stub_updates(uninterrupted)
        torch.manual_seed(303)
        uninterrupted.train_step()
        uninterrupted_next_rng = torch.rand(4)

        torch.manual_seed(202)
        _, paused, _ = self._make_persistent_budget_trainer()
        self._stub_updates(paused)
        torch.manual_seed(303)
        for _ in range(3):
            paused.train_step(max_env_steps_to_collect=1)
        paused_next_rng = torch.rand(4)

        self._assert_replay_equal(uninterrupted, paused)
        torch.testing.assert_close(uninterrupted_next_rng, paused_next_rng)
        self.assertEqual(uninterrupted._train_step_count, paused._train_step_count)
        self.assertEqual(uninterrupted._next_episode_id, paused._next_episode_id)
        self.assertEqual(
            uninterrupted.value_update.call_count, paused.value_update.call_count
        )
        self.assertEqual(
            uninterrupted.policy_update.call_count, paused.policy_update.call_count
        )

    def test_schema_v3_resume_continues_persistent_episode_exactly(self):
        torch.manual_seed(404)
        model, original, cfg = self._make_persistent_budget_trainer()
        provenance = self._checkpoint_provenance(original)
        self._stub_updates(original)
        random.seed(505)
        np.random.seed(505)
        torch.manual_seed(505)
        original.train_step(max_env_steps_to_collect=1)
        saved_transition = original.replay.storage[0]

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                original,
                step=original.get_env_step_count(),
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            original.train_step(max_env_steps_to_collect=2)
            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(4)

            torch.manual_seed(999)
            restored_model, restored, _ = self._make_persistent_budget_trainer()
            self._stub_updates(restored)
            start_update = resume_from_checkpoint(
                checkpoint_path,
                restored_model,
                restored,
                "cpu",
                expected_dataset_provenance=provenance,
            )

            self.assertEqual(start_update, 0)
            self.assertEqual(restored.get_env_step_count(), 1)
            self.assertIsNotNone(restored._active_episode)
            self.assertEqual(restored._active_episode["timestep"], 1)
            torch.testing.assert_close(
                restored._active_episode["latent"].z_H,
                saved_transition.next_latent.z_H,
            )
            torch.testing.assert_close(
                restored._active_episode["latent"].z_L,
                saved_transition.next_latent.z_L,
            )

            restored.train_step(max_env_steps_to_collect=2)
            actual_python = random.random()
            actual_numpy = float(np.random.rand())
            actual_torch = torch.rand(4)

        self._assert_replay_equal(original, restored)
        self.assertEqual(expected_python, actual_python)
        self.assertEqual(expected_numpy, actual_numpy)
        torch.testing.assert_close(expected_torch, actual_torch)
        self.assertEqual(original._train_step_count, restored._train_step_count)
        self.assertEqual(original._next_episode_id, restored._next_episode_id)
        self.assertIsNone(restored._active_episode)

    def test_legacy_checkpoint_requires_explicit_weights_only_warm_start(self):
        torch.manual_seed(606)
        model, trainer, cfg = self._make_persistent_budget_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=self._checkpoint_provenance(trainer),
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            payload["checkpoint_schema_version"] = 2
            legacy_path = str(Path(tmp) / "legacy.pt")
            torch.save(payload, legacy_path)

            _, strict_trainer, _ = self._make_persistent_budget_trainer()
            with self.assertRaisesRegex(RuntimeError, "schema-v3"):
                resume_from_checkpoint(
                    legacy_path,
                    strict_trainer.model,
                    strict_trainer,
                    "cpu",
                )

            warm_model, warm_trainer, _ = self._make_persistent_budget_trainer()
            before = {
                key: value.detach().clone()
                for key, value in warm_model.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "resume is refused"):
                resume_from_checkpoint(
                    legacy_path,
                    warm_model,
                    warm_trainer,
                    "cpu",
                    allow_legacy_warm_start=True,
                )
            for key, value in warm_model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertEqual(warm_trainer.get_env_step_count(), 0)
            self.assertEqual(len(warm_trainer.replay), 0)
            self.assertIsNone(warm_trainer._active_episode)

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
            provenance = self._checkpoint_provenance(trainer)
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
            self.assertEqual(checkpoint_payload["checkpoint_schema_version"], 3)
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
        saved_provenance = self._checkpoint_provenance(trainer)
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
            python_rng_before = random.getstate()
            numpy_rng_before = np.random.get_state()
            torch_rng_before = torch.random.get_rng_state().clone()
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load, patch.object(
                restored.value_opt,
                "load_state_dict",
                wraps=restored.value_opt.load_state_dict,
            ) as value_optimizer_load, patch.object(
                restored.policy_opt,
                "load_state_dict",
                wraps=restored.policy_opt.load_state_dict,
            ) as policy_optimizer_load, patch.object(
                restored.replay,
                "clear",
                wraps=restored.replay.clear,
            ) as replay_clear, patch(
                "upi_trm_train._restore_rng_state"
            ) as restore_rng:
                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    resume_from_checkpoint(
                        checkpoint_path,
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=self._checkpoint_provenance(
                            restored,
                            generation_seed=999,
                        ),
                    )
            model_load.assert_not_called()
            value_optimizer_load.assert_not_called()
            policy_optimizer_load.assert_not_called()
            replay_clear.assert_not_called()
            restore_rng.assert_not_called()
            for key, value in restored_model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertEqual(random.getstate(), python_rng_before)
            numpy_rng_after = np.random.get_state()
            self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
            np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
            self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
            torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)


if __name__ == "__main__":
    unittest.main()
