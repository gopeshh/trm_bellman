import copy
import hashlib
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import torch
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.persistent_diagnostic_checkpoint import (
    _build_environment,
    _load_provenance_manifest,
    load_persistent_diagnostic_context,
    load_persistent_checkpoint,
    PersistentDiagnosticInputError,
    state_dict_sha256,
)
from rl.persistent_diagnostics import (
    _clip_and_recenter,
    _policy_gap_raw,
    collect_persistent_diagnostic_states,
    DeploymentSpec,
    hash_augmented_state,
    PersistentDiagnosticConfig,
    run_persistent_diagnostics,
    stack_full_tensor_state_fields,
)
from rl.replay import ReplayLatent, Transition
from rl.task_config import get_task_config
from rl.training_setup import build_dataset_from_paths, offset_puzzle_identifiers
from rl.upi_trm_trainer import UPITrmTrainer, _clip_and_recenter_advantages
from scripts.persistent_checkpoint_diagnostics import (
    PersistentDiagnosticArtifactError,
    load_diagnostic_spec,
    run_registered_diagnostics,
    write_artifact_bundle,
)
from torch import nn
from torch.distributions import Categorical
from utils.dataset_provenance import build_dataset_provenance
from utils.dataset_provenance import (
    dataset_input_sha256s,
    dataset_puzzle_identifier_sha256s,
    dataset_sample_sha256s,
    dataset_source_build_metadata,
    ordered_record_sha256,
)
from utils.run_identity import build_checkpoint_lineage, canonical_json_sha256


def _model_dump(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    return dict(config.dict())


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


class _OneRecordDataset:
    seq_len = 1
    vocab_size = 3
    num_identifiers = 1

    def __init__(self) -> None:
        self._sample = {
            "inputs": torch.tensor([1], dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0, dtype=torch.long),
            "initial_plan": torch.tensor([1], dtype=torch.long),
            "solution": torch.tensor([2], dtype=torch.long),
            "labels": torch.tensor([2], dtype=torch.long),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return {name: value.clone() for name, value in self._sample.items()}


class _AnalyticPersistentModel(nn.Module):
    """Deterministic recurrent policy with a zero value function."""

    def __init__(self, action_count: int) -> None:
        super().__init__()
        self.inner = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.inner.weight.fill_(0.25)
        self.action_count = action_count
        self.config = {
            "seq_len": 1,
            "puzzle_emb_ndim": 0,
            "num_puzzle_identifiers": 1,
            "vocab_size": 3,
            "L_cycles": 1,
            "L_layers": 1,
            "hidden_size": 1,
            "expansion": 1.0,
            "num_heads": 1,
            "pos_encodings": "none",
            "forward_dtype": "float32",
            "puzzle_emb_len": 0,
            "rl_enable_z_init_encoder": False,
            "rl_value_hidden_dim": 1,
            "rl_latent_ball_radius": 0.0,
        }

    def init_latent(
        self,
        x: Mapping[str, torch.Tensor],
        y: torch.Tensor,
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        del x
        zeros = torch.zeros((y.shape[0], 1, 1), device=y.device)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=zeros,
            z_L=zeros.clone(),
        )

    def update_latent(
        self,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        y: torch.Tensor,
        x: Mapping[str, torch.Tensor],
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        del x, y
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z.z_H + 1.0,
            z_L=z.z_L + 1.0,
        )

    def _advance(
        self,
        x: Mapping[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry | None,
    ) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        carry = self.init_latent(x, y) if z is None else z
        for _ in range(n):
            carry = self.update_latent(carry, y, x)
        return carry

    def used_value(
        self,
        x: Mapping[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry | None = None,
    ) -> tuple[torch.Tensor, TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        carry = self._advance(x, y, n, z)
        return torch.zeros(y.shape[0], device=y.device), carry

    def policy_dist(
        self,
        x: Mapping[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        action_mask: torch.Tensor | None = None,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry | None = None,
    ) -> tuple[Categorical, TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        carry = self._advance(x, y, n, z)
        if action_mask is None:
            probabilities = torch.full(
                (y.shape[0], self.action_count),
                1.0 / self.action_count,
                device=y.device,
            )
        else:
            valid = action_mask.to(device=y.device, dtype=torch.float32)
            probabilities = valid / valid.sum(dim=-1, keepdim=True)
        return Categorical(probs=probabilities), carry


class _CarryValueModel(_AnalyticPersistentModel):
    """Uses the first carried coordinate as a hand-computable value."""

    def used_value(
        self,
        x: Mapping[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry | None = None,
    ) -> tuple[torch.Tensor, TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        carry = self._advance(x, y, n, z)
        return carry.z_H.reshape(carry.z_H.shape[0], -1).mean(dim=-1), carry


class _BiasedPolicyModel(_AnalyticPersistentModel):
    def __init__(self, action_count: int, logits: list[float]) -> None:
        super().__init__(action_count)
        self.register_buffer("policy_logits", torch.tensor(logits))

    def policy_dist(
        self,
        x: Mapping[str, torch.Tensor],
        y: torch.Tensor,
        n: int,
        action_mask: torch.Tensor | None = None,
        z: TinyRecursiveReasoningModel_ACTV1InnerCarry | None = None,
    ) -> tuple[Categorical, TinyRecursiveReasoningModel_ACTV1InnerCarry]:
        carry = self._advance(x, y, n, z)
        logits = self.policy_logits.to(y.device).expand(y.shape[0], -1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return Categorical(logits=logits), carry


def _analytic_environment(
    stop_action_mode: str = "disabled",
    *,
    reward_shaping: bool = False,
) -> PlanEditEnv:
    dataset = _OneRecordDataset()
    config = PlanEditEnvConfig(
        max_edits=2,
        gamma=0.5,
        reward_shaping=reward_shaping,
        task_type="dummy",
        vocab_size=dataset.vocab_size,
        solved_threshold=None,
        stop_action_mode=stop_action_mode,
        fail_terminal_reward=0.0,
        solve_terminal_reward=0.0,
    )

    def checker(_x: Any, y: Any) -> float:
        plan = y if torch.is_tensor(y) else torch.as_tensor(y)
        return float(int(plan.reshape(-1)[0].item()) == 2)

    env = PlanEditEnv(dataset=dataset, checker=checker, config=config)
    env.set_stop_action_id(dataset.seq_len * dataset.vocab_size)
    return env


def _diagnostic_config() -> PersistentDiagnosticConfig:
    return PersistentDiagnosticConfig(
        inner_unroll_n=1,
        reference_depth_m=2,
        gamma=0.5,
        k_horizon=2,
        mc_repeats=4,
        mc_seed=17,
        collection_seed=23,
        mixture_alpha=0.25,
        max_retained_states=2,
        chunk_size=2,
    )


def _production_deployment_spec(
    current_policy: nn.Module,
    candidate_policy: nn.Module,
    config: PersistentDiagnosticConfig,
) -> DeploymentSpec:
    runtime = SimpleNamespace(
        rl_cfg=SimpleNamespace(
            mixture_alpha=config.mixture_alpha,
            policy_epsilon=0.0,
        ),
        policy_model_old=current_policy,
        policy_model_candidate=candidate_policy,
    )
    production_mixed_policy_dist = getattr(UPITrmTrainer, "_mixed_policy_dist")

    def policy_dist_fn(*args: Any, **kwargs: Any) -> Any:
        return production_mixed_policy_dist(runtime, *args, **kwargs)

    return DeploymentSpec(
        kind="policy_dist_callback",
        policy_dist_fn=policy_dist_fn,
        label="production_exact_probability_mixture_callback",
    )


def _registered_spec(record_count: int = 1) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_selection": {
            "kind": "ordered_prefix_without_replacement",
            "record_count": record_count,
        },
        "reference_depth_offsets": [1, 2, 4],
        "monte_carlo": {"repeats": 2, "seed": 17},
        "collection_seed": 23,
        "max_retained_states": 2,
        "chunk_size": 2,
        "probability_tolerance": 1e-6,
        "ratio_denominator_tolerance": 1e-12,
        "initial_latent_sensitivity": {
            "perturbation_l2_norm": 0.01,
            "seed": 29,
        },
    }


def _tiny_model_config(
    num_puzzle_identifiers: int = 1,
    vocab_size: int = 3,
) -> TinyRecursiveReasoningModel_ACTV1Config:
    return TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=1,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=num_puzzle_identifiers,
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=8,
        expansion=2.0,
        num_heads=1,
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
        rl_num_actions=vocab_size + 1,
        rl_value_hidden_dim=8,
        rl_latent_ball_radius=0.0,
    )


def _fixed_base_config() -> RLConfig:
    return RLConfig(
        gamma=0.5,
        K=2,
        inner_unroll_n=1,
        max_edits=2,
        task_name="dummy",
        solved_threshold=None,
        episodic_latent=False,
        stop_action_mode="disabled",
        reward_shaping=True,
        fail_terminal_reward=0.0,
        solve_terminal_reward=0.0,
        training_protocol="fixed_base_exact",
        exact_k_step_targets=True,
        exact_baseline_summation=True,
        theory_exact_mixture=True,
        distill_mixture_policy=False,
        policy_epsilon=0.0,
        enable_contraction=False,
        opnorm_clamp_interval=0,
        latent_ball_radius=0.0,
        batch_size=1,
        replay_capacity=4,
        use_tqdm=False,
    )


def _checkpoint_payload(
    *,
    model_config: TinyRecursiveReasoningModel_ACTV1Config | None = None,
    dataset_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = model_config or _tiny_model_config()
    rl_config = _fixed_base_config()
    model = TinyRecursiveReasoningModel_ACTV1(_model_dump(model_config))
    model_state = _clone_state_dict(model)
    environment_state = _analytic_environment(
        reward_shaping=True
    ).checkpoint_state()

    latent = ReplayLatent(
        z_H=torch.zeros(1, 1, model_config.hidden_size),
        z_L=torch.zeros(1, 1, model_config.hidden_size),
    )
    next_latent = ReplayLatent(
        z_H=torch.ones(1, 1, model_config.hidden_size),
        z_L=torch.ones(1, 1, model_config.hidden_size),
    )
    transition = Transition(
        x={
            "inputs": torch.tensor([1]),
            "puzzle_identifiers": torch.tensor([0]),
            "solution": torch.tensor([2]),
            "remaining_edits": torch.tensor(1),
        },
        y=torch.tensor([1]),
        action=torch.tensor(2),
        reward=torch.tensor(0.0),
        x_next={
            "inputs": torch.tensor([1]),
            "puzzle_identifiers": torch.tensor([0]),
            "solution": torch.tensor([2]),
            "remaining_edits": torch.tensor(0),
        },
        y_next=torch.tensor([2]),
        done=torch.tensor(True),
        episode_id=0,
        timestep=1,
        latent=latent,
        next_latent=next_latent,
        behavior_log_prob=torch.tensor(0.0),
    )
    provenance = dataset_provenance or build_dataset_provenance(
        builder_name="tests.OneRecordDataset",
        builder_version=1,
        generation_seed=7,
        train_record_sha256s=["a" * 64],
        eval_record_sha256s=["b" * 64],
        train_split="train",
        eval_split="heldout",
        environment_config={"max_edits": 2},
        action_mask_config={"stop_action_mode": "disabled"},
    )
    serialized_rl_config = _model_dump(rl_config)
    serialized_model_config = _model_dump(model_config)
    runtime_fingerprint = {"test_runtime": True}
    effective_config = {
        "effective_config_schema_version": 1,
        "algorithm": "upi_trm",
        "training_protocol": "fixed_base_exact",
        "backbone": "trm",
        "rl_config": copy.deepcopy(serialized_rl_config),
        "model_config": copy.deepcopy(serialized_model_config),
        "execution_device": "cpu",
        "runtime_fingerprint_sha256": canonical_json_sha256(
            runtime_fingerprint
        ),
        "dataset": {
            "train_split": provenance["splits"]["train"],
            "eval_split": provenance["splits"]["eval"],
            "train_record_count": provenance["ordered_records"]["train"][
                "count"
            ],
            "eval_record_count": provenance["ordered_records"]["eval"][
                "count"
            ],
        },
        "budget": {
            "outer_train_steps": serialized_rl_config["num_train_steps"],
            "environment_interactions": 1,
        },
        "schedule": {
            "log_outer_interval": serialized_rl_config["log_interval"],
            "eval_outer_interval": serialized_rl_config["eval_interval"],
            "save_outer_interval": 1,
            "log_environment_interval": 1,
            "eval_environment_interval": 1,
            "save_environment_interval": 1,
        },
        "evaluation": {
            "episode_count": serialized_rl_config["eval_num_episodes"],
            "seed": serialized_rl_config["eval_seed"],
            "pool_size": provenance["ordered_records"]["eval"]["count"],
        },
        "puzzle_embedding_optimizer": {
            "learning_rate": 0.01,
            "weight_decay": 0.1,
        },
        "imitation": {"enabled": False, "epochs": 0},
        "external_logging": "disabled",
        "debug_checks": serialized_rl_config["debug_checks"],
        "config_source_sha256s": [],
    }
    run_identity = {
        "run_identity_schema_version": 1,
        "run_id": "diagnostic.seed7",
        "training_seed": 7,
        "producer": {"git_commit": "a" * 40, "git_clean": True},
        "effective_config": effective_config,
        "effective_config_sha256": canonical_json_sha256(effective_config),
        "dataset_provenance_sha256": canonical_json_sha256(provenance),
        "initialization": {"kind": "random", "artifact_sha256": None},
    }
    parameter_names = [name for name, _ in model.named_parameters()]
    module_names = (
        "model",
        "policy_model_old",
        "policy_model_candidate",
        "target_model",
    )
    return {
        "checkpoint_schema_version": 5,
        "training_protocol": "fixed_base_exact",
        "execution_device": "cpu",
        "trainer_kind": "UPITrmTrainer",
        "step": 1,
        "progress": {
            "env_steps": 1,
            "outer_steps": 0,
            "value_optimizer_steps": 0,
            "policy_optimizer_steps": 0,
            "distill_optimizer_steps": 0,
            "puzzle_optimizer_steps": 0,
        },
        "model_state_dict": copy.deepcopy(model_state),
        "policy_model_old_state_dict": copy.deepcopy(model_state),
        "policy_model_candidate_state_dict": copy.deepcopy(model_state),
        "target_model_state_dict": copy.deepcopy(model_state),
        "value_optimizer_state_dict": {},
        "policy_optimizer_state_dict": {},
        "rng_state": {},
        "dataset_provenance": provenance,
        "model_config": serialized_model_config,
        "rl_config": serialized_rl_config,
        "run_identity": run_identity,
        "checkpoint_lineage": build_checkpoint_lineage(
            parent_checkpoint_sha256=None,
            parent_checkpoint_step=None,
            parent_environment_steps=None,
        ),
        "runtime_fingerprint": runtime_fingerprint,
        "module_training_modes": {name: False for name in module_names},
        "parameter_gradients": {
            module_name: {name: None for name in parameter_names}
            for module_name in module_names
        },
        "checkpoint_phase": "idle_between_training_calls",
        "trainer_state": {
            "env_step_count": 1,
            "train_step_count": 0,
            "value_optimizer_step_count": 0,
            "policy_optimizer_step_count": 0,
            "distill_optimizer_step_count": 0,
            "puzzle_optimizer_step_count": 0,
            "environment_state": environment_state,
            "collection_state": {
                "schema_version": 1,
                "completed_episodes_since_update": 0,
                "active_episode": None,
            },
        },
        "replay_buffer_size": 1,
        "replay_capacity": 4,
        "replay_transitions": [transition],
    }


def _refresh_run_identity(payload: dict[str, Any]) -> None:
    identity = payload["run_identity"]
    effective = identity["effective_config"]
    identity["effective_config"]["rl_config"] = copy.deepcopy(
        payload["rl_config"]
    )
    identity["effective_config"]["model_config"] = copy.deepcopy(
        payload["model_config"]
    )
    provenance = payload["dataset_provenance"]
    effective["dataset"] = {
        "train_split": provenance["splits"]["train"],
        "eval_split": provenance["splits"]["eval"],
        "train_record_count": provenance["ordered_records"]["train"]["count"],
        "eval_record_count": provenance["ordered_records"]["eval"]["count"],
    }
    effective["budget"]["outer_train_steps"] = payload["rl_config"][
        "num_train_steps"
    ]
    effective["schedule"]["log_outer_interval"] = payload["rl_config"][
        "log_interval"
    ]
    effective["schedule"]["eval_outer_interval"] = payload["rl_config"][
        "eval_interval"
    ]
    effective["evaluation"] = {
        "episode_count": payload["rl_config"]["eval_num_episodes"],
        "seed": payload["rl_config"]["eval_seed"],
        "pool_size": provenance["ordered_records"]["eval"]["count"],
    }
    effective["debug_checks"] = payload["rl_config"]["debug_checks"]
    identity["effective_config_sha256"] = canonical_json_sha256(
        identity["effective_config"]
    )
    identity["dataset_provenance_sha256"] = canonical_json_sha256(
        payload["dataset_provenance"]
    )


class TestPersistentDiagnostics(unittest.TestCase):
    def assertNestedEqual(self, left: Any, right: Any) -> None:
        if torch.is_tensor(left) or torch.is_tensor(right):
            self.assertTrue(torch.is_tensor(left) and torch.is_tensor(right))
            self.assertTrue(torch.equal(left, right))
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            self.assertTrue(isinstance(left, np.ndarray))
            self.assertTrue(isinstance(right, np.ndarray))
            self.assertTrue(np.array_equal(left, right))
            return
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            self.assertTrue(isinstance(left, Mapping))
            self.assertTrue(isinstance(right, Mapping))
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertNestedEqual(left[key], right[key])
            return
        if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
            self.assertIs(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assertNestedEqual(left_item, right_item)
            return
        self.assertEqual(left, right)

    def test_full_snapshot_residuals_fields_carry_clock_and_runtime(self) -> None:
        env = _analytic_environment()
        current = _AnalyticPersistentModel(action_count=4)
        evaluator = copy.deepcopy(current)
        candidate = copy.deepcopy(current)
        config = _diagnostic_config()

        env.reset(idx=0)
        env_before = copy.deepcopy(env.checkpoint_state())
        random.seed(101)
        np.random.seed(103)
        torch.manual_seed(107)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        current.train()

        collection = collect_persistent_diagnostic_states(
            current_policy=current,
            env=env,
            record_indices=[0],
            config=config,
        )

        self.assertEqual(len(collection.states), 2)
        self.assertNestedEqual(env.checkpoint_state(), env_before)
        self.assertNestedEqual(random.getstate(), python_before)
        self.assertNestedEqual(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertTrue(current.training)

        retained_x = collection.states[0].environment_state["x"]
        self.assertIn("solution", retained_x)
        self.assertIn("labels", retained_x)
        stacked = stack_full_tensor_state_fields(
            [state.environment_state["x"] for state in collection.states],
            torch.device("cpu"),
        )
        self.assertIn("solution", stacked)
        self.assertIn("labels", stacked)
        torch.testing.assert_close(stacked["solution"], torch.tensor([[2], [2]]))
        torch.testing.assert_close(stacked["labels"], torch.tensor([[2], [2]]))

        evaluator.train()
        candidate.eval()
        output = run_persistent_diagnostics(
            evaluator=evaluator,
            current_policy=current,
            candidate_policy=candidate,
            deployment=_production_deployment_spec(current, candidate, config),
            env=env,
            states=collection.states,
            config=config,
            endpoint_policy_pair_invariants_verified=True,
            provenance={"test_fixture": "analytic_two_step"},
        )

        self.assertNestedEqual(env.checkpoint_state(), env_before)
        self.assertNestedEqual(random.getstate(), python_before)
        self.assertNestedEqual(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertTrue(current.training)
        self.assertTrue(evaluator.training)
        self.assertFalse(candidate.training)

        rows_by_clock = {row["remaining_edits"]: row for row in output.state_rows}
        self.assertEqual(set(rows_by_clock), {1, 2})
        clock_two = rows_by_clock[2]
        self.assertAlmostEqual(clock_two["exact_one_step_backup"], 0.0)
        self.assertAlmostEqual(clock_two["exact_one_step_signed_residual"], 0.0)
        self.assertAlmostEqual(clock_two["mc_k_step_operator_mean"], 0.5)
        self.assertAlmostEqual(clock_two["mc_k_step_signed_residual"], -0.5)
        self.assertAlmostEqual(clock_two["mc_k_step_operator_standard_error"], 0.0)
        clock_one = rows_by_clock[1]
        self.assertAlmostEqual(clock_one["exact_one_step_backup"], 1.0)
        self.assertAlmostEqual(clock_one["exact_one_step_signed_residual"], -1.0)
        self.assertAlmostEqual(clock_one["mc_k_step_operator_mean"], 1.0)
        self.assertAlmostEqual(clock_one["mc_k_step_signed_residual"], -1.0)
        self.assertAlmostEqual(clock_one["mc_k_step_operator_standard_error"], 0.0)

        one_step = output.summary["exact_one_step_augmented_residual"]
        self.assertEqual(one_step["estimator"], "exact_action_sum_one_step")
        self.assertEqual(one_step["horizon"], 1)
        self.assertEqual(
            one_step["transition_oracle"],
            "PlanEditEnv.step from full snapshot",
        )
        mc_summary = output.summary["monte_carlo_k_step_augmented_residual"]
        self.assertEqual(mc_summary["estimator"], "monte_carlo_k_step")
        self.assertEqual(mc_summary["horizon"], 2)
        self.assertEqual(mc_summary["repeat_count_per_state"], 4)
        self.assertAlmostEqual(
            mc_summary["monte_carlo_standard_error"]["maximum"],
            0.0,
        )

        for row in output.mc_rows:
            if output.state_rows[row["state_index"]]["remaining_edits"] == 2:
                self.assertEqual(row["actions"], [2, 2])
                self.assertEqual(row["rewards"], [0.0, 1.0])
                self.assertAlmostEqual(row["k_step_return"], 0.5)
            else:
                self.assertEqual(row["actions"], [2])
                self.assertEqual(row["rewards"], [1.0])
                self.assertAlmostEqual(row["k_step_return"], 1.0)

        carry = output.summary["persistent_carry_and_clock"]
        self.assertEqual(carry["state_count"], 2)
        self.assertAlmostEqual(
            carry["observed_to_recomputed_post_unroll_error"]["maximum"],
            0.0,
        )
        continuity = carry["adjacent_continuity"]
        self.assertEqual(continuity["adjacent_pair_count"], 1)
        self.assertEqual(continuity["clock_decrement_violation_count"], 0)
        self.assertEqual(continuity["clock_link_violation_count"], 0)
        self.assertAlmostEqual(
            continuity["successor_to_next_input_carry_error"]["maximum"],
            0.0,
        )
        parity = output.summary["production_exact_baseline_oracle_parity"]
        self.assertAlmostEqual(
            parity["maximum_action_q_absolute_error_per_state"]["maximum"],
            0.0,
        )
        self.assertAlmostEqual(
            parity["baseline_absolute_error"]["maximum"],
            0.0,
        )
        deployment = output.summary["exact_mixture_deployment_gap"]
        self.assertEqual(
            deployment["measurement_kind"],
            "finite_batch_production_distribution_callback",
        )
        self.assertTrue(deployment["deployment_distribution_callback_tested"])
        self.assertFalse(deployment["full_episode_evaluation_loop_tested"])
        self.assertAlmostEqual(deployment["total_variation"]["maximum"], 0.0)
        self.assertEqual(output.summary["scope"], "finite_batch")
        self.assertEqual(
            output.summary["radial_projection_activation"]["status"],
            "not verifiable from supplied evidence",
        )
        self.assertIn("actual_to_perturbed_post_unroll_distance", output.summary[
            "initial_latent_sensitivity"
        ])

    def test_nonterminal_k_step_bootstrap_uses_gamma_k_and_carried_latent(self) -> None:
        env = _analytic_environment()
        current = _CarryValueModel(action_count=4)
        evaluator = copy.deepcopy(current)
        candidate = copy.deepcopy(current)
        config = replace(_diagnostic_config(), k_horizon=1)
        collection = collect_persistent_diagnostic_states(
            current_policy=current,
            env=env,
            record_indices=[0],
            config=config,
        )
        output = run_persistent_diagnostics(
            evaluator=evaluator,
            current_policy=current,
            candidate_policy=candidate,
            deployment=_production_deployment_spec(current, candidate, config),
            env=env,
            states=collection.states,
            config=config,
            endpoint_policy_pair_invariants_verified=True,
        )
        clock_two = next(
            row for row in output.state_rows if row["remaining_edits"] == 2
        )
        self.assertAlmostEqual(clock_two["value_n"], 1.0)
        self.assertAlmostEqual(clock_two["value_m"], 2.0)
        self.assertAlmostEqual(clock_two["mc_k_step_operator_mean"], 1.0)
        self.assertAlmostEqual(clock_two["mc_k_step_signed_residual"], 0.0)
        self.assertAlmostEqual(
            clock_two["mc_k_step_reference_operator_mean"], 1.5
        )
        self.assertAlmostEqual(
            clock_two["mc_k_step_reference_signed_residual"], 0.5
        )
        first_particle = next(
            row
            for row in output.mc_rows
            if row["state_index"] == clock_two["state_index"]
        )
        self.assertTrue(first_particle["bootstrapped"])
        self.assertEqual(first_particle["rewards"], [0.0])
        self.assertAlmostEqual(first_particle["k_step_return"], 1.0)
        self.assertAlmostEqual(first_particle["k_step_return_m"], 1.5)

    def test_distinct_candidate_and_concrete_deployment_gap(self) -> None:
        env = _analytic_environment(stop_action_mode="noop")
        current = _BiasedPolicyModel(4, [-10.0, -10.0, 0.0, 0.0])
        evaluator = copy.deepcopy(current)
        candidate = _BiasedPolicyModel(
            4,
            [-10.0, -10.0, math.log(3.0), 0.0],
        )
        deployed = _BiasedPolicyModel(
            4,
            [-10.0, -10.0, 0.0, math.log(3.0)],
        )
        config = _diagnostic_config()
        collection = collect_persistent_diagnostic_states(
            current_policy=current,
            env=env,
            record_indices=[0],
            config=config,
        )
        output = run_persistent_diagnostics(
            evaluator=evaluator,
            current_policy=current,
            candidate_policy=candidate,
            deployment=DeploymentSpec(
                kind="concrete_model",
                model=deployed,
                label="concrete_deployment",
            ),
            env=env,
            states=collection.states,
            config=config,
            endpoint_policy_pair_invariants_verified=True,
        )
        action_two = next(
            row
            for row in output.action_rows
            if row["state_index"] == 0 and row["action_index"] == 2
        )
        self.assertAlmostEqual(action_two["current_policy_probability"], 0.5)
        self.assertAlmostEqual(
            action_two["candidate_policy_probability"], 0.75, places=6
        )
        self.assertAlmostEqual(
            action_two["exact_mixture_probability"], 0.5625, places=6
        )
        self.assertAlmostEqual(
            action_two["deployed_policy_probability"], 0.25, places=6
        )
        gap = output.summary["exact_mixture_deployment_gap"]
        self.assertEqual(
            gap["measurement_kind"],
            "finite_batch_concrete_policy_comparison",
        )
        self.assertGreater(gap["total_variation"]["maximum"], 0.0)

    def test_projection_info_api_preserves_normal_latent_step(self) -> None:
        model = TinyRecursiveReasoningModel_ACTV1(_model_dump(_tiny_model_config()))
        x = {
            "inputs": torch.tensor([[1]]),
            "puzzle_identifiers": torch.tensor([0]),
            "solution": torch.tensor([[2]]),
        }
        y = torch.tensor([[1]])
        latent = model.init_latent(x, y)
        ordinary = model.update_latent(latent, y, x)
        diagnosed, pre_norm, active = model.update_latent_with_projection_info(
            latent,
            y,
            x,
        )
        torch.testing.assert_close(ordinary.z_H, diagnosed.z_H)
        torch.testing.assert_close(ordinary.z_L, diagnosed.z_L)
        self.assertEqual(tuple(pre_norm.shape), (1,))
        self.assertTrue(torch.isfinite(pre_norm).all())
        self.assertFalse(bool(active.item()))

    def test_configured_clipping_matches_training_helper(self) -> None:
        advantages = torch.tensor([[-20.0, 2.0, 5.0, 99.0]])
        probabilities = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        mask = torch.tensor([[True, True, True, False]])
        expected = _clip_and_recenter_advantages(
            advantages,
            probabilities,
            mask,
            3.0,
        )
        actual = _clip_and_recenter(
            advantages,
            probabilities,
            mask,
            3.0,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            (probabilities * actual).sum(dim=-1),
            torch.zeros(1),
            atol=2e-7,
            rtol=0.0,
        )

    def test_raw_policy_gap_ignores_tolerated_masked_mass(self) -> None:
        reference = torch.tensor([[0.6, 0.4, 1e-8]], dtype=torch.float64)
        comparison = torch.tensor([[0.6, 0.4, 0.0]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        gap = _policy_gap_raw(reference, comparison, mask)
        self.assertAlmostEqual(float(gap["total_variation"].item()), 0.0)
        self.assertAlmostEqual(
            float(gap["kl_reference_to_comparison"].item()), 0.0
        )
        self.assertFalse(bool(gap["support_mismatch"].item()))

    def test_augmented_hash_distinguishes_latent_and_clock(self) -> None:
        env = _analytic_environment()
        env.reset(idx=0)
        env.y = torch.tensor([2], dtype=torch.long)
        clock_two_snapshot = env.checkpoint_state()
        zero = ReplayLatent(
            z_H=torch.zeros(1, 1, 1),
            z_L=torch.zeros(1, 1, 1),
        )
        one = ReplayLatent(
            z_H=torch.ones(1, 1, 1),
            z_L=torch.ones(1, 1, 1),
        )
        zero_hash = hash_augmented_state(clock_two_snapshot, zero)
        self.assertNotEqual(
            zero_hash,
            hash_augmented_state(clock_two_snapshot, one),
        )

        (_, _), first_reward, first_done, _ = env.step(2)
        clock_one_snapshot = env.checkpoint_state()
        self.assertTrue(torch.equal(clock_two_snapshot["y"], clock_one_snapshot["y"]))
        self.assertTrue(
            torch.equal(
                clock_two_snapshot["x"]["inputs"],
                clock_one_snapshot["x"]["inputs"],
            )
        )
        self.assertFalse(first_done)
        self.assertAlmostEqual(first_reward, 0.0)
        self.assertNotEqual(
            zero_hash,
            hash_augmented_state(clock_one_snapshot, zero),
        )

        env.load_checkpoint_state(clock_two_snapshot)
        (_, _), from_two_reward, from_two_done, _ = env.step(2)
        env.load_checkpoint_state(clock_one_snapshot)
        (_, _), from_one_reward, from_one_done, _ = env.step(2)
        self.assertFalse(from_two_done)
        self.assertAlmostEqual(from_two_reward, 0.0)
        self.assertTrue(from_one_done)
        self.assertAlmostEqual(from_one_reward, 1.0)


class TestPersistentCheckpointLoader(unittest.TestCase):
    def _save(self, directory: str, payload: Mapping[str, Any], name: str) -> Path:
        path = Path(directory) / name
        torch.save(dict(payload), path)
        return path

    def test_valid_schema5_fixed_base_checkpoint_loads_without_rng_mutation(
        self,
    ) -> None:
        payload = _checkpoint_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, payload, "valid.pt")
            torch.manual_seed(211)
            rng_before = torch.random.get_rng_state().clone()
            loaded = load_persistent_checkpoint(path, device="cpu")

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(loaded.checkpoint_step, 1)
        self.assertEqual(loaded.environment_steps, 1)
        self.assertEqual(loaded.outer_steps, 0)
        self.assertEqual(loaded.value_optimizer_steps, 0)
        self.assertEqual(loaded.policy_optimizer_steps, 0)
        self.assertEqual(loaded.replay.transition_count, 1)
        self.assertEqual(loaded.replay.terminal_transition_count, 1)
        self.assertTrue(loaded.recurrent_map_shared)
        self.assertTrue(loaded.endpoint_policy_pair_invariants_verified)
        self.assertEqual(loaded.rl_config.training_protocol, "fixed_base_exact")
        self.assertFalse(loaded.rl_config.episodic_latent)
        self.assertFalse(loaded.evaluator.training)
        self.assertFalse(loaded.current_policy.training)
        self.assertFalse(loaded.candidate_policy.training)
        self.assertEqual(loaded.producer_code_commit, "a" * 40)
        self.assertEqual(loaded.training_seed, 7)
        self.assertEqual(loaded.run_id, "diagnostic.seed7")
        self.assertEqual(
            loaded.effective_config_sha256,
            payload["run_identity"]["effective_config_sha256"],
        )
        self.assertEqual(loaded.initialization_kind, "random")
        self.assertIsNone(loaded.initialization_artifact_sha256)
        self.assertIsNone(loaded.parent_checkpoint_sha256)
        self.assertIsNone(loaded.parent_checkpoint_step)

    def test_checkpoint_schema_versions_reject_alias_types_without_rng_mutation(
        self,
    ) -> None:
        base = _checkpoint_payload()
        mutations = (
            (
                "checkpoint",
                5,
                lambda payload, invalid: payload.__setitem__(
                    "checkpoint_schema_version", invalid
                ),
                "checkpoint schema 5",
            ),
            (
                "collector",
                1,
                lambda payload, invalid: payload["trainer_state"][
                    "collection_state"
                ].__setitem__("schema_version", invalid),
                "collector state has an unsupported schema",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for boundary, expected, mutate, message in mutations:
                for invalid in (True, float(expected), str(expected)):
                    with self.subTest(boundary=boundary, invalid=invalid):
                        payload = copy.deepcopy(base)
                        mutate(payload, invalid)
                        path = self._save(
                            directory,
                            payload,
                            f"{boundary}-{type(invalid).__name__}.pt",
                        )
                        torch.manual_seed(431)
                        rng_before = torch.random.get_rng_state().clone()
                        with self.assertRaisesRegex(
                            PersistentDiagnosticInputError,
                            message,
                        ):
                            load_persistent_checkpoint(path, device="cpu")
                        self.assertTrue(
                            torch.equal(torch.random.get_rng_state(), rng_before)
                        )

    def test_dataset_manifest_schema_rejects_alias_types_without_mutation(
        self,
    ) -> None:
        provenance = _checkpoint_payload()["dataset_provenance"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_manifest.json"
            for invalid in (True, 1.0, "1"):
                with self.subTest(invalid=invalid):
                    path.write_text(
                        json.dumps(
                            {
                                "manifest_schema_version": invalid,
                                "dataset_provenance": provenance,
                            },
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    before = path.read_bytes()
                    with self.assertRaisesRegex(
                        PersistentDiagnosticInputError,
                        "dataset-manifest schema version",
                    ):
                        _load_provenance_manifest(path)
                    self.assertEqual(path.read_bytes(), before)

    def test_rejects_checkpoint_mutated_during_single_open_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, _checkpoint_payload(), "mutable.pt")
            real_torch_load = torch.load

            def load_then_mutate(handle, *args, **kwargs):
                payload = real_torch_load(handle, *args, **kwargs)
                with path.open("ab") as writer:
                    writer.write(b"changed")
                return payload

            with patch(
                "rl.persistent_diagnostic_checkpoint.torch.load",
                side_effect=load_then_mutate,
            ):
                with self.assertRaisesRegex(
                    PersistentDiagnosticInputError,
                    "changed while it was being loaded",
                ):
                    load_persistent_checkpoint(path, device="cpu")

    def test_schema5_run_identity_is_strictly_bound(self) -> None:
        base = _checkpoint_payload()
        cases: list[tuple[str, dict[str, Any], str]] = []

        missing = copy.deepcopy(base)
        missing.pop("run_identity")
        cases.append(("missing", missing, "missing required fields"))

        bad_commit = copy.deepcopy(base)
        bad_commit["run_identity"]["producer"]["git_commit"] = "A" * 40
        cases.append(("bad_commit", bad_commit, "identity or lineage is invalid"))

        bad_config_hash = copy.deepcopy(base)
        bad_config_hash["run_identity"]["effective_config_sha256"] = "f" * 64
        cases.append(
            (
                "bad_config_hash",
                bad_config_hash,
                "identity or lineage is invalid",
            )
        )

        bad_lineage = copy.deepcopy(base)
        bad_lineage["checkpoint_lineage"]["parent_checkpoint_sha256"] = "a" * 64
        cases.append(
            ("bad_lineage", bad_lineage, "identity or lineage is invalid")
        )

        dataset_mismatch = copy.deepcopy(base)
        dataset_mismatch["run_identity"]["dataset_provenance_sha256"] = "f" * 64
        cases.append(
            (
                "dataset_mismatch",
                dataset_mismatch,
                "dataset provenance hash",
            )
        )

        config_mismatch = copy.deepcopy(base)
        config_mismatch["run_identity"]["effective_config"]["rl_config"][
            "K"
        ] += 1
        config_mismatch["run_identity"]["effective_config_sha256"] = (
            canonical_json_sha256(
                config_mismatch["run_identity"]["effective_config"]
            )
        )
        cases.append(
            ("config_mismatch", config_mismatch, "RL configuration")
        )

        schema4 = copy.deepcopy(base)
        schema4["checkpoint_schema_version"] = 4
        cases.append(("schema4", schema4, "checkpoint schema 5"))

        with tempfile.TemporaryDirectory() as directory:
            for name, payload, expected_message in cases:
                with self.subTest(name=name):
                    path = self._save(directory, payload, f"{name}.pt")
                    with self.assertRaisesRegex(
                        PersistentDiagnosticInputError,
                        expected_message,
                    ):
                        load_persistent_checkpoint(path, device="cpu")

    def test_state_hash_handles_scalar_bfloat_view_without_storage_overreach(
        self,
    ) -> None:
        backing = torch.arange(8, dtype=torch.bfloat16)
        scalar_view = backing[3]
        original = state_dict_sha256({"scalar": scalar_view})

        backing[0] = 100.0
        self.assertEqual(state_dict_sha256({"scalar": scalar_view}), original)
        backing[3] = 9.0
        self.assertNotEqual(state_dict_sha256({"scalar": scalar_view}), original)

    def test_accepts_candidate_edit_head_and_lagged_target_differences(self) -> None:
        payload = _checkpoint_payload()
        candidate = payload["policy_model_candidate_state_dict"]
        edit_name = next(
            name
            for name, value in candidate.items()
            if name.startswith("edit_policy.") and torch.is_floating_point(value)
        )
        candidate[edit_name] = candidate[edit_name].clone() + 0.01
        target = payload["target_model_state_dict"]
        value_name = next(
            name
            for name, value in target.items()
            if name.startswith("value_head.") and torch.is_floating_point(value)
        )
        target[value_name] = target[value_name].clone() + 0.02
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, payload, "candidate.pt")
            loaded = load_persistent_checkpoint(path, device="cpu")
        self.assertTrue(loaded.endpoint_policy_pair_invariants_verified)
        self.assertEqual(
            loaded.current_policy_identity.recurrent_map_sha256,
            loaded.candidate_policy_identity.recurrent_map_sha256,
        )
        self.assertNotEqual(
            loaded.current_policy_identity.full_state_sha256,
            loaded.candidate_policy_identity.full_state_sha256,
        )

    def test_rejects_undo_when_declared_state_omits_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, _checkpoint_payload(), "valid.pt")
            checkpoint = load_persistent_checkpoint(path, device="cpu")
        environment = _analytic_environment(reward_shaping=True)
        provenance = dict(checkpoint.dataset_provenance)
        undo_config = PlanEditEnvConfig(
            max_edits=2,
            gamma=0.5,
            reward_shaping=False,
            task_type="dummy",
            vocab_size=3,
            solved_threshold=None,
            stop_action_mode="disabled",
            fail_terminal_reward=0.0,
            solve_terminal_reward=0.0,
            enable_undo=True,
        )
        provenance["environment_config"] = vars(undo_config)
        with self.assertRaisesRegex(
            PersistentDiagnosticInputError,
            "reject UNDO",
        ):
            _build_environment(environment.dataset, checkpoint, provenance)

    def test_rejects_schema3_episodic_missing_pair_and_map_mismatch(self) -> None:
        base = _checkpoint_payload()
        cases: list[tuple[str, dict[str, Any], str]] = []

        schema3 = copy.deepcopy(base)
        schema3["checkpoint_schema_version"] = 3
        cases.append(("schema3", schema3, "checkpoint schema 5"))

        episodic = copy.deepcopy(base)
        episodic["rl_config"]["episodic_latent"] = True
        _refresh_run_identity(episodic)
        cases.append(("episodic", episodic, "episodic-latent"))

        missing_pair = copy.deepcopy(base)
        missing_pair.pop("policy_model_candidate_state_dict")
        cases.append(("missing_pair", missing_pair, "missing required fields"))

        map_mismatch = copy.deepcopy(base)
        candidate = map_mismatch["policy_model_candidate_state_dict"]
        recurrent_name = next(
            name
            for name, value in candidate.items()
            if name.startswith(("inner.", "z_init_encoder."))
            and torch.is_floating_point(value)
        )
        candidate[recurrent_name] = candidate[recurrent_name].clone() + 1.0
        cases.append(("map_mismatch", map_mismatch, "recurrent map"))

        clock_mismatch = copy.deepcopy(base)
        clock_transition = clock_mismatch["replay_transitions"][0]
        clock_transition.x["remaining_edits"] = torch.tensor(2)
        clock_transition.x_next["remaining_edits"] = torch.tensor(1)
        cases.append(("clock_mismatch", clock_mismatch, "clock disagrees"))

        zero_budget_nonterminal = copy.deepcopy(base)
        zero_budget_nonterminal["replay_transitions"][0].done = torch.tensor(False)
        cases.append(
            (
                "zero_budget_nonterminal",
                zero_budget_nonterminal,
                "reaching zero remaining edits must be terminal",
            )
        )

        positive_log_probability = copy.deepcopy(base)
        positive_log_probability["replay_transitions"][0].behavior_log_prob = (
            torch.tensor(0.1)
        )
        cases.append(
            (
                "positive_log_probability",
                positive_log_probability,
                "positive behavior log probability",
            )
        )

        skipped_episode_id = copy.deepcopy(base)
        skipped_transition = copy.deepcopy(
            skipped_episode_id["replay_transitions"][0]
        )
        skipped_transition.episode_id = 2
        skipped_transition.timestep = 0
        skipped_transition.x["remaining_edits"] = torch.tensor(2)
        skipped_transition.x_next["remaining_edits"] = torch.tensor(1)
        skipped_transition.done = torch.tensor(False)
        skipped_episode_id["replay_transitions"].append(skipped_transition)
        skipped_episode_id["replay_buffer_size"] = 2
        cases.append(
            (
                "skipped_episode_id",
                skipped_episode_id,
                "not consecutive",
            )
        )

        wrong_latent_shape = copy.deepcopy(base)
        transition = wrong_latent_shape["replay_transitions"][0]
        hidden_size = _tiny_model_config().hidden_size
        transition.latent = ReplayLatent(
            z_H=torch.zeros(1, 2, hidden_size),
            z_L=torch.zeros(1, 2, hidden_size),
        )
        cases.append(
            ("wrong_latent_shape", wrong_latent_shape, "wrong model shape")
        )

        capacity_mismatch = copy.deepcopy(base)
        capacity_mismatch["replay_capacity"] = 3
        cases.append(
            ("capacity_mismatch", capacity_mismatch, "replay capacity disagrees")
        )

        impossible_interaction_count = copy.deepcopy(base)
        impossible_interaction_count["progress"]["env_steps"] = 0
        impossible_interaction_count["trainer_state"]["env_step_count"] = 0
        cases.append(
            (
                "impossible_interaction_count",
                impossible_interaction_count,
                "fewer interactions",
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            for name, payload, expected_message in cases:
                with self.subTest(name=name):
                    path = self._save(directory, payload, f"{name}.pt")
                    with self.assertRaisesRegex(
                        PersistentDiagnosticInputError,
                        expected_message,
                    ):
                        load_persistent_checkpoint(path, device="cpu")


class TestPersistentDiagnosticArtifacts(unittest.TestCase):
    @staticmethod
    def _write_split(
        root: Path,
        split: str,
        inputs: list[int],
    ) -> None:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        input_array = np.asarray(inputs, dtype=np.int32).reshape(-1, 1)
        labels = np.full_like(input_array, 2)
        identifiers = np.arange(len(inputs), dtype=np.int32)
        boundaries = np.arange(len(inputs) + 1, dtype=np.int32)
        np.save(split_dir / "all__inputs.npy", input_array)
        np.save(split_dir / "all__labels.npy", labels)
        np.save(split_dir / "all__puzzle_identifiers.npy", identifiers)
        np.save(split_dir / "all__puzzle_indices.npy", boundaries)
        np.save(split_dir / "all__group_indices.npy", boundaries)
        (split_dir / "dataset.json").write_text(
            json.dumps(
                {
                    "seq_len": 1,
                    "vocab_size": 4,
                    "pad_id": 0,
                    "ignore_label_id": None,
                    "blank_identifier_id": 0,
                    "num_puzzle_identifiers": len(inputs),
                    "total_groups": len(inputs),
                    "mean_puzzle_examples": 1,
                    "total_puzzles": len(inputs),
                    "sets": ["all"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _context(self, directory: str) -> Any:
        checkpoint_path = Path(directory) / "checkpoint.pt"
        torch.save(_checkpoint_payload(), checkpoint_path)
        checkpoint = load_persistent_checkpoint(checkpoint_path, device="cpu")
        environment = _analytic_environment(reward_shaping=True)
        dataset = SimpleNamespace(
            eval_dataset=environment.dataset,
            environment=environment,
            manifest_sha256="1" * 64,
            train_ordered_sha256="2" * 64,
            eval_ordered_sha256="3" * 64,
            train_input_sha256s=("4" * 64,),
            eval_input_sha256s=("5" * 64,),
            checker_kind="dummy",
            checker=environment.checker,
        )
        return SimpleNamespace(checkpoint=checkpoint, dataset=dataset)

    def test_spec_is_strict_and_hashes_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            raw = (json.dumps(_registered_spec(), sort_keys=True) + "\n").encode(
                "ascii"
            )
            path.write_bytes(raw)
            parsed, digest = load_diagnostic_spec(path)
            self.assertEqual(parsed, _registered_spec())
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

            invalid = _registered_spec()
            invalid["unregistered_override"] = True
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(
                PersistentDiagnosticArtifactError,
                "unknown",
            ):
                load_diagnostic_spec(path)

            path.write_text(
                '{"schema_version":1,"schema_version":1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                PersistentDiagnosticArtifactError,
                "duplicate key",
            ):
                load_diagnostic_spec(path)

    def test_bundle_is_deterministic_anonymous_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(directory)
            spec = _registered_spec()
            output, collection = run_registered_diagnostics(
                context,
                spec,
                diagnostic_spec_sha256="6" * 64,
                diagnostic_code_commit="7" * 40,
            )
            repeated_output, repeated_collection = run_registered_diagnostics(
                context,
                spec,
                diagnostic_spec_sha256="6" * 64,
                diagnostic_code_commit="7" * 40,
            )
            self.assertEqual(output.summary, repeated_output.summary)
            self.assertEqual(output.state_rows, repeated_output.state_rows)
            self.assertEqual(output.action_rows, repeated_output.action_rows)
            self.assertEqual(output.depth_rows, repeated_output.depth_rows)
            self.assertEqual(output.mc_rows, repeated_output.mc_rows)
            self.assertEqual(collection.metadata, repeated_collection.metadata)
            self.assertEqual(
                set(output.summary["registered_reference_depths"]["offsets"]),
                {"1", "2", "4"},
            )
            self.assertEqual(
                max(row["depth"] for row in output.depth_rows),
                5,
            )
            self.assertEqual(
                set(output.action_rows[0]["registered_reference_depths"]),
                {"1", "2", "4"},
            )
            first = Path(directory) / "bundle-a"
            second = Path(directory) / "bundle-b"
            write_artifact_bundle(
                output_dir=first,
                context=context,
                spec=spec,
                diagnostic_spec_sha256="6" * 64,
                diagnostic_code_commit="7" * 40,
                output=output,
                collection=collection,
            )
            write_artifact_bundle(
                output_dir=second,
                context=context,
                spec=spec,
                diagnostic_spec_sha256="6" * 64,
                diagnostic_code_commit="7" * 40,
                output=output,
                collection=collection,
            )

            first_files = {path.name: path.read_bytes() for path in first.iterdir()}
            second_files = {
                path.name: path.read_bytes() for path in second.iterdir()
            }
            self.assertEqual(first_files, second_files)
            self.assertIn("per_action.jsonl", first_files)
            self.assertIn("augmented_states.jsonl", first_files)
            validation = json.loads(first_files["validation.json"])
            self.assertEqual(
                validation["diagnostic"]["code_commit"]["status"],
                "not verifiable from supplied evidence",
            )
            self.assertEqual(
                validation["checkpoint"]["producer_code_commit"],
                {"value": "a" * 40, "status": "verified from artifact"},
            )
            self.assertEqual(
                validation["checkpoint"]["training_seed"],
                {"value": 7, "status": "verified from artifact"},
            )
            self.assertEqual(
                validation["checkpoint"]["run_id"],
                {
                    "value": "diagnostic.seed7",
                    "status": "verified from artifact",
                },
            )
            self.assertEqual(
                validation["checkpoint"][
                    "optimizer_rng_and_live_trainer_restore"
                ]["status"],
                "not verifiable from supplied evidence",
            )
            self.assertEqual(
                validation["checkpoint"]["lineage"],
                {
                    "parent_checkpoint_sha256": None,
                    "parent_checkpoint_step": None,
                    "parent_environment_interactions": None,
                    "status": "verified from artifact",
                },
            )
            loaded_sources = validation["diagnostic"]["loaded_source_sha256s"]
            self.assertTrue(loaded_sources)
            self.assertTrue(
                {
                    "environment_transition",
                    "checker_implementation",
                    "heldout_dataset_implementation",
                    "production_deployment_distribution",
                    "production_episode_evaluator",
                }.issubset(
                    loaded_sources
                )
            )
            behavior_sources = validation["diagnostic"][
                "repository_behavior_source_manifest"
            ]["files"]
            self.assertTrue(
                {
                    "models/layers.py",
                    "models/edit_policy.py",
                    "models/value_head.py",
                    "rl/envs/plan_edit_env.py",
                    "rl/sudoku_utils.py",
                    "rl/task_config.py",
                }.issubset(behavior_sources)
            )
            combined = b"\n".join(first_files.values())
            self.assertNotIn(b"/" + b"home/", combined)
            self.assertNotIn(b"buiksat", combined)

            checksum_lines = first_files["SHA256SUMS"].decode("ascii").splitlines()
            expected = [
                f"{hashlib.sha256(payload).hexdigest()}  {name}"
                for name, payload in sorted(first_files.items())
                if name != "SHA256SUMS"
            ]
            self.assertEqual(checksum_lines, expected)
            with self.assertRaisesRegex(
                PersistentDiagnosticArtifactError,
                "already exists",
            ):
                write_artifact_bundle(
                    output_dir=first,
                    context=context,
                    spec=spec,
                    diagnostic_spec_sha256="6" * 64,
                    diagnostic_code_commit="7" * 40,
                    output=output,
                    collection=collection,
                )

    def test_context_materialization_binds_ordered_puzzle_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "anonymous_dataset"
            root.mkdir()
            (root / "build_config.json").write_text(
                json.dumps(
                    {
                        "builder": "tests.synthetic_dataset",
                        "build_schema_version": 1,
                        "seed": 41,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._write_split(root, "train", [0])
            self._write_split(root, "heldout", [1, 2, 3])

            train_dataset, seq_len, vocab_size, train_identifier_count = (
                build_dataset_from_paths(
                    [str(root)],
                    pool_size=1,
                    split="train",
                    allow_dummy_fallback=False,
                )
            )
            eval_dataset, _, _, eval_identifier_count = build_dataset_from_paths(
                [str(root)],
                pool_size=3,
                split="heldout",
                allow_dummy_fallback=False,
            )
            offset_puzzle_identifiers(eval_dataset, train_identifier_count)
            combined_identifier_count = (
                train_identifier_count + eval_identifier_count
            )
            env_config = PlanEditEnvConfig(
                max_edits=2,
                gamma=0.5,
                reward_shaping=True,
                task_type="dummy",
                vocab_size=vocab_size,
                solved_threshold=None,
                stop_action_mode="disabled",
                fail_terminal_reward=0.0,
                solve_terminal_reward=0.0,
            )
            task_config = get_task_config(
                "sudoku",
                disable_constraint_masking=False,
            )
            stop_action_id = seq_len * vocab_size
            source_metadata = dataset_source_build_metadata([str(root)])
            metadata = {
                "train_pool_sha256": ordered_record_sha256(
                    dataset_sample_sha256s(train_dataset)
                ),
                "eval_pool_sha256": ordered_record_sha256(
                    dataset_sample_sha256s(eval_dataset)
                ),
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "num_identifiers": combined_identifier_count,
                "eval_puzzle_id_offset": train_identifier_count,
                "dataset_source_names": [root.name],
                "source_build_metadata": source_metadata,
                "materialization_seed": 0,
                "train_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                    dataset_puzzle_identifier_sha256s(train_dataset)
                ),
                "eval_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                    dataset_puzzle_identifier_sha256s(eval_dataset)
                ),
            }
            provenance = build_dataset_provenance(
                builder_name="tests.synthetic_dataset",
                builder_version=1,
                generation_seed=41,
                train_record_sha256s=dataset_sample_sha256s(train_dataset),
                eval_record_sha256s=dataset_sample_sha256s(eval_dataset),
                train_split="train",
                eval_split="heldout",
                environment_config=vars(env_config),
                action_mask_config={
                    "task_config_class": type(task_config).__name__,
                    "task_config_name": task_config.name,
                    "disable_constraint_masking": False,
                    "stop_action_mode": "disabled",
                    "stop_action_id": stop_action_id,
                    "enable_undo": False,
                    "undo_action_id": None,
                    "vocab_size": vocab_size,
                    "num_actions": stop_action_id + 1,
                    "masked_token_ids": [0, 1],
                },
                metadata=metadata,
            )
            checkpoint_path = Path(directory) / "checkpoint.pt"
            checkpoint_payload = _checkpoint_payload(
                model_config=_tiny_model_config(
                    num_puzzle_identifiers=combined_identifier_count,
                    vocab_size=vocab_size,
                ),
                dataset_provenance=provenance,
            )
            checkpoint_payload["rl_config"]["eval_num_episodes"] = 2
            _refresh_run_identity(checkpoint_payload)
            torch.save(checkpoint_payload, checkpoint_path)
            manifest_path = Path(directory) / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_schema_version": 1,
                        "dataset_provenance": provenance,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            context = load_persistent_diagnostic_context(
                checkpoint_path,
                manifest_path,
                [root],
                device="cpu",
            )
            self.assertEqual(
                context.dataset.train_input_sha256s,
                tuple(dataset_input_sha256s(train_dataset)),
            )
            self.assertEqual(
                [
                    int(sample["puzzle_identifiers"].item())
                    for sample in context.dataset.eval_dataset.samples
                ],
                [1, 2, 3],
            )

            identifiers_path = (
                root / "heldout" / "all__puzzle_identifiers.npy"
            )
            np.save(identifiers_path, np.asarray([2, 1, 0], dtype=np.int32))
            with self.assertRaisesRegex(
                PersistentDiagnosticInputError,
                "Ordered puzzle identifiers",
            ):
                load_persistent_diagnostic_context(
                    checkpoint_path,
                    manifest_path,
                    [root],
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
