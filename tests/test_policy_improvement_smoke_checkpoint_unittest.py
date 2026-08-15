#!/usr/bin/env fbpython
"""Focused exact-continuation tests for the Stage 0 PPO checkpoint."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from policy_improvement_smoke_checkpoint import (
    PolicyImprovementSmokeCheckpointError,
    build_ppo_smoke_checkpoint,
    load_stable_checkpoint,
    publish_checkpoint,
    validate_ppo_smoke_checkpoint,
)
from rl.algos.ppo import PPOConfig, PPOTrainer as ProductionPPOTrainer
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import dummy_checker
from utils.compute_accounting import (
    MODEL_COUNTER_FIELDS,
    process_peak_rss_bytes,
    zero_model_counters,
)


class _InstrumentedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self._counters = zero_model_counters()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self._counters["policy_api_calls"] += 1
        self._counters["policy_state_evaluations"] += len(value)
        self._counters["recurrent_latent_update_calls"] += 1
        self._counters["recurrent_latent_state_updates"] += len(value)
        self._counters["action_logits_evaluated"] += 2 * len(value)
        return self.linear(value)

    def compute_counter_snapshot(self) -> dict[str, int]:
        return dict(self._counters)

    def restore_compute_counters(self, value: object) -> None:
        assert isinstance(value, dict)
        assert set(value) == set(MODEL_COUNTER_FIELDS)
        self._counters = {name: int(item) for name, item in value.items()}


class _DummyPuzzleDataset:
    def __init__(self, *, num_instances: int, seq_len: int, vocab_size: int) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_instances
        self.samples = []
        for index in range(num_instances):
            inputs = torch.randint(
                low=0,
                high=vocab_size,
                size=(seq_len,),
                dtype=torch.long,
            )
            self.samples.append(
                {
                    "inputs": inputs,
                    "puzzle_identifiers": torch.tensor(index, dtype=torch.long),
                    "initial_plan": torch.zeros_like(inputs),
                    "solution": inputs.clone(),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class _Environment:
    def __init__(self) -> None:
        self.x = {"inputs": torch.tensor([1, 2]), "remaining_edits": torch.tensor(15)}
        self.y = torch.tensor([1, 2])
        self._state = {
            "schema_version": 1,
            "initialized": True,
            "x": copy.deepcopy(self.x),
            "y": self.y.clone(),
            "step_count": 1,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def load_checkpoint_state(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("invalid environment state")
        self._state = copy.deepcopy(value)
        self.x = copy.deepcopy(value["x"])
        self.y = value["y"].clone()


class _Rollout:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.x_list: list[Any] = []
        self.y_list: list[Any] = []
        self.actions: list[Any] = []
        self.log_probs: list[Any] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[Any] = []
        self.action_masks: list[Any] = []

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class _Config:
    num_steps: int = 16
    num_epochs: int = 1
    num_minibatches: int = 1


class PPOTrainer:
    def __init__(self) -> None:
        self.model = _InstrumentedModel()
        self.env = _Environment()
        self.config = _Config()
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss = self.model(torch.ones(1, 2)).sum()
        loss.backward()
        self.optimizer.step()
        self.rollout_buffer = _Rollout()
        for index in range(16):
            self.rollout_buffer.x_list.append(
                {"inputs": torch.tensor([index, index + 1])}
            )
            self.rollout_buffer.y_list.append(torch.tensor([index]))
            self.rollout_buffer.actions.append(torch.tensor(index % 2))
            self.rollout_buffer.log_probs.append(torch.tensor(-0.5))
            self.rollout_buffer.rewards.append(float(index))
            self.rollout_buffer.dones.append(index == 15)
            self.rollout_buffer.values.append(torch.tensor(float(index)))
            self.rollout_buffer.action_masks.append(torch.tensor([True, True]))
        self._train_step_count = 1
        self._env_step_count = 16
        self._optimizer_step_count = 1
        self._episode_count = 1
        self.term_stats = {"stop": 0, "solved": 0, "budget": 1}
        self._current_x = copy.deepcopy(self.env.x)
        self._current_y = self.env.y.clone()
        self._episode_rewards = [1.0]
        self._training_model_work = self.model.compute_counter_snapshot()
        self._evaluation_model_work = zero_model_counters()
        self._training_wall_time_seconds = 0.25
        self._evaluation_wall_time_seconds = 0.0
        self._peak_process_rss_bytes = process_peak_rss_bytes()
        self._peak_cuda_allocated_bytes = None
        self._peak_cuda_reserved_bytes = None

    def _model_roles(self) -> dict[str, torch.nn.Module]:
        return {"model": self.model}

    def compute_accounting_checkpoint_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_compute_state": [
                {
                    "roles": ["model"],
                    "counters": self.model.compute_counter_snapshot(),
                }
            ],
            "training_model_work": dict(self._training_model_work),
            "evaluation_model_work": dict(self._evaluation_model_work),
            "training_wall_time_seconds": self._training_wall_time_seconds,
            "evaluation_wall_time_seconds": self._evaluation_wall_time_seconds,
            "peak_process_rss_bytes": self._peak_process_rss_bytes,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
        }

    def advance_one_rollout(self) -> None:
        self.optimizer.zero_grad()
        batch = torch.randn(1, 2)
        loss = self.model(batch).square().sum()
        loss.backward()
        self.optimizer.step()
        self._train_step_count += 1
        self._env_step_count += 16
        self._optimizer_step_count += 1
        self._episode_count += 1
        self.term_stats["budget"] += 1
        self._training_model_work = self.model.compute_counter_snapshot()
        value = torch.randint(0, 1000, (1,)).item()
        self.env.x = {
            "inputs": torch.tensor([value, value + 1]),
            "remaining_edits": torch.tensor(15),
        }
        self.env.y = torch.tensor([value, value + 1])
        self.env._state["x"] = copy.deepcopy(self.env.x)
        self.env._state["y"] = self.env.y.clone()
        self._current_x = copy.deepcopy(self.env.x)
        self._current_y = self.env.y.clone()
        self._episode_rewards = [float(value)]
        self.rollout_buffer.clear()
        for index in range(16):
            self.rollout_buffer.x_list.append(
                {"inputs": torch.tensor([value + index, value + index + 1])}
            )
            self.rollout_buffer.y_list.append(torch.tensor([value + index]))
            self.rollout_buffer.actions.append(torch.tensor(index % 2))
            self.rollout_buffer.log_probs.append(torch.tensor(-0.5))
            self.rollout_buffer.rewards.append(float(value + index))
            self.rollout_buffer.dones.append(index == 15)
            self.rollout_buffer.values.append(torch.tensor(float(value + index)))
            self.rollout_buffer.action_masks.append(torch.tensor([True, True]))


def _identity() -> dict[str, object]:
    return {
        "protocol_sha256": "1" * 64,
        "registry_row_sha256": "2" * 64,
        "run_id": "s0-matched-ppo",
        "method_id": "matched_ppo",
        "seed": 7,
        "runtime_sha256": "3" * 64,
    }


def _tree_equal(left: object, right: object) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _production_trainer(seed: int) -> ProductionPPOTrainer:
    _seed_everything(seed)
    dataset = _DummyPuzzleDataset(num_instances=8, seq_len=4, vocab_size=5)
    environment = PlanEditEnv(
        dataset=dataset,
        checker=dummy_checker,
        config=PlanEditEnvConfig(
            max_edits=2,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
            task_type="dummy",
            stop_action_mode="terminal",
        ),
    )
    environment.set_stop_action_id(dataset.seq_len * dataset.vocab_size)
    model = TinyRecursiveReasoningModel_ACTV1(
        {
            "batch_size": 4,
            "seq_len": dataset.seq_len,
            "puzzle_emb_ndim": 0,
            "puzzle_emb_len": 0,
            "num_puzzle_identifiers": dataset.num_identifiers,
            "vocab_size": dataset.vocab_size,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 0,
            "L_layers": 1,
            "hidden_size": 16,
            "expansion": 2.0,
            "num_heads": 4,
            "pos_encodings": "rope",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.0,
            "forward_dtype": "float32",
            "mlp_t": False,
            "no_ACT_continue": True,
            "rl_enable_value_head": True,
            "rl_enable_contraction": False,
            "rl_enable_policy_head": True,
            "rl_num_actions": dataset.seq_len * dataset.vocab_size + 1,
            "rl_latent_projection_mode": "enabled",
            "rl_latent_ball_radius": 10.0,
        }
    )
    return ProductionPPOTrainer(
        model,
        environment,
        PPOConfig(
            num_steps=16,
            num_epochs=1,
            num_minibatches=1,
            num_train_steps=2,
            batch_size=16,
            inner_unroll_n=1,
        ),
        device=torch.device("cpu"),
    )


class PolicyImprovementSmokeCheckpointTest(unittest.TestCase):
    def test_production_ppo_uninterrupted_matches_two_process_resume(self) -> None:
        source = _production_trainer(1408)
        source.train_step(max_env_steps_to_collect=16)
        first = build_ppo_smoke_checkpoint(
            source,
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        source.train_step(max_env_steps_to_collect=16)
        uninterrupted = build_ppo_smoke_checkpoint(
            source,
            identity=_identity(),
            parent_checkpoint_sha256="9" * 64,
        )

        resumed = _production_trainer(1408)
        validate_ppo_smoke_checkpoint(
            first,
            resumed,
            expected_identity=_identity(),
            validate_only=True,
        )
        validate_ppo_smoke_checkpoint(
            first,
            resumed,
            expected_identity=_identity(),
            validate_only=False,
        )
        resumed.train_step(max_env_steps_to_collect=16)
        continued = build_ppo_smoke_checkpoint(
            resumed,
            identity=_identity(),
            parent_checkpoint_sha256="9" * 64,
        )

        for field in (
            "model_state_dict",
            "optimizer_state_dict",
            "progress",
            "termination_counts",
            "environment_state",
            "episode_state",
            "rollout_buffer",
            "rng_state",
        ):
            self.assertTrue(_tree_equal(uninterrupted[field], continued[field]), field)
        left_compute = uninterrupted["compute_accounting_state"]
        right_compute = continued["compute_accounting_state"]
        for field in (
            "model_compute_state",
            "training_model_work",
            "evaluation_model_work",
        ):
            self.assertTrue(_tree_equal(left_compute[field], right_compute[field]))

    def test_uninterrupted_matches_save_reload_continuation(self) -> None:
        torch.manual_seed(91)
        source = PPOTrainer()
        checkpoint = build_ppo_smoke_checkpoint(
            source,
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        source.advance_one_rollout()
        uninterrupted = build_ppo_smoke_checkpoint(
            source,
            identity=_identity(),
            parent_checkpoint_sha256="9" * 64,
        )

        probe = PPOTrainer()
        validate_ppo_smoke_checkpoint(
            checkpoint,
            probe,
            expected_identity=_identity(),
            validate_only=True,
        )
        resumed = PPOTrainer()
        validate_ppo_smoke_checkpoint(
            checkpoint,
            resumed,
            expected_identity=_identity(),
            validate_only=False,
        )
        resumed.advance_one_rollout()
        continued = build_ppo_smoke_checkpoint(
            resumed,
            identity=_identity(),
            parent_checkpoint_sha256="9" * 64,
        )
        self.assertEqual(uninterrupted["progress"], continued["progress"])
        self.assertEqual(
            uninterrupted["termination_counts"], continued["termination_counts"]
        )
        left_compute = uninterrupted["compute_accounting_state"]
        right_compute = continued["compute_accounting_state"]
        for field in (
            "schema_version",
            "model_compute_state",
            "training_model_work",
            "evaluation_model_work",
            "training_wall_time_seconds",
            "evaluation_wall_time_seconds",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        ):
            self.assertTrue(
                _tree_equal(left_compute[field], right_compute[field]), field
            )
        self.assertGreaterEqual(
            right_compute["peak_process_rss_bytes"],
            left_compute["peak_process_rss_bytes"],
        )
        self.assertTrue(
            _tree_equal(
                uninterrupted["model_state_dict"],
                continued["model_state_dict"],
            )
        )
        self.assertTrue(
            _tree_equal(
                uninterrupted["optimizer_state_dict"],
                continued["optimizer_state_dict"],
            )
        )
        self.assertTrue(
            _tree_equal(uninterrupted["rollout_buffer"], continued["rollout_buffer"])
        )
        self.assertTrue(_tree_equal(uninterrupted["rng_state"], continued["rng_state"]))

    def test_probe_then_restore_round_trip(self) -> None:
        source = PPOTrainer()
        payload = build_ppo_smoke_checkpoint(
            source,
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        probe = PPOTrainer()
        validate_ppo_smoke_checkpoint(
            payload,
            probe,
            expected_identity=_identity(),
            validate_only=True,
        )
        restored = PPOTrainer()
        validate_ppo_smoke_checkpoint(
            payload,
            restored,
            expected_identity=_identity(),
            validate_only=False,
        )
        self.assertEqual(restored._env_step_count, 16)
        self.assertEqual(restored._train_step_count, 1)
        self.assertEqual(restored.term_stats, source.term_stats)
        self.assertEqual(len(restored.rollout_buffer), 16)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    source.model.state_dict().values(),
                    restored.model.state_dict().values(),
                )
            )
        )

    def test_rejects_missing_model_key_before_live_restore(self) -> None:
        trainer = PPOTrainer()
        payload = build_ppo_smoke_checkpoint(
            trainer,
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        del payload["model_state_dict"]["linear.bias"]
        live_before = copy.deepcopy(trainer.model.state_dict())
        with self.assertRaises(PolicyImprovementSmokeCheckpointError):
            validate_ppo_smoke_checkpoint(
                payload,
                trainer,
                expected_identity=_identity(),
                validate_only=False,
            )
        for name, value in live_before.items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[name]))

    def test_rejects_wrong_optimizer_tensor_shape(self) -> None:
        trainer = PPOTrainer()
        payload = build_ppo_smoke_checkpoint(
            trainer,
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        first_state = next(iter(payload["optimizer_state_dict"]["state"].values()))
        first_state["exp_avg"] = torch.zeros(7)
        with self.assertRaises(PolicyImprovementSmokeCheckpointError):
            validate_ppo_smoke_checkpoint(
                payload,
                PPOTrainer(),
                expected_identity=_identity(),
                validate_only=True,
            )

    def test_rejects_identity_change(self) -> None:
        payload = build_ppo_smoke_checkpoint(
            PPOTrainer(),
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        changed = _identity()
        changed["seed"] = 8
        with self.assertRaises(PolicyImprovementSmokeCheckpointError):
            validate_ppo_smoke_checkpoint(
                payload,
                PPOTrainer(),
                expected_identity=changed,
                validate_only=True,
            )

    def test_durable_publication_and_digest_binding(self) -> None:
        payload = build_ppo_smoke_checkpoint(
            PPOTrainer(),
            identity=_identity(),
            parent_checkpoint_sha256=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "checkpoint.pt"
            digest = publish_checkpoint(payload, destination)
            loaded, observed = load_stable_checkpoint(
                destination, expected_sha256=digest
            )
            self.assertEqual(observed, digest)
            self.assertEqual(loaded["identity"], _identity())
            with self.assertRaises(PolicyImprovementSmokeCheckpointError):
                publish_checkpoint(payload, destination)


if __name__ == "__main__":
    unittest.main()
