#!/usr/bin/env fbpython
"""Focused tests for the train-only Sudoku base-policy producer."""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import unittest
from unittest import mock
from contextlib import redirect_stdout
from pathlib import Path

from scripts.policy_improvement_base_policy import (
    authenticate_train_split,
    BASE_POLICY_SEED,
    HEALTH_CHECK_EPOCHS,
    BasePolicyProducerError,
    canonical_json_bytes,
    EARLY_STOP_ACCURACY,
    main,
    neutral_shared_config,
    REGISTERED_METHOD_CONFIGS,
    IMITATION_EPOCHS,
    RL_ENVIRONMENT_INTERACTIONS,
    SHARED_CONFIG_FIELDS,
    TRAIN_RECORD_COUNT,
    TRAINING_PROCEDURE,
    training_procedure_sha256,
)
from scripts.policy_improvement_v2_registry import load_v2_base_configs
from scripts.policy_improvement_v2_schema import load_strict_json, validate_v2_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/policy_improvement_v2/protocol.json"
DATASET = (
    ROOT / "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1"
)


class BasePolicyProcedureTest(unittest.TestCase):
    def test_procedure_is_budget_selected_and_train_only(self) -> None:
        self.assertIsNone(EARLY_STOP_ACCURACY)
        self.assertIsNone(TRAINING_PROCEDURE["early_stop_accuracy"])
        self.assertEqual(
            TRAINING_PROCEDURE["checkpoint_selection"], "final_budget_only"
        )
        self.assertEqual(RL_ENVIRONMENT_INTERACTIONS, 0)
        self.assertEqual(TRAINING_PROCEDURE["rl_environment_interactions"], 0)
        self.assertEqual(TRAINING_PROCEDURE["training_split"], "train")
        self.assertEqual(TRAINING_PROCEDURE["training_record_count"], 1024)
        self.assertEqual(
            TRAINING_PROCEDURE["demonstration_episodes_per_repeat"],
            TRAIN_RECORD_COUNT,
        )
        self.assertFalse(TRAINING_PROCEDURE["validation_data_used"])
        self.assertFalse(TRAINING_PROCEDURE["test_data_used"])
        self.assertTrue(TRAINING_PROCEDURE["shared_across_persistent_and_episodic"])
        # The disposable health budget must not reach the frozen procedure, or
        # the artifact would depend on a pre-flight flag.
        self.assertNotIn("health_check_epochs", TRAINING_PROCEDURE)
        self.assertNotIn(HEALTH_CHECK_EPOCHS, (IMITATION_EPOCHS,))

    def test_base_policy_seed_is_distinct_from_every_registered_seed(self) -> None:
        seeds = validate_v2_protocol(load_strict_json(PROTOCOL))["seeds"]
        registered = set(seeds["smoke"]) | set(seeds["pilot"]) | set(
            seeds["confirmatory"]
        )
        self.assertNotIn(BASE_POLICY_SEED, registered)

    def test_procedure_digest_is_canonical_and_stable(self) -> None:
        digest = training_procedure_sha256()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, training_procedure_sha256())
        # The digest must move if any frozen value moves.
        mutated = copy.deepcopy(TRAINING_PROCEDURE)
        mutated["imitation_epochs"] += 1
        self.assertNotEqual(
            digest,
            hashlib.sha256(canonical_json_bytes(mutated)).hexdigest(),
        )

    def test_demonstration_repeats_do_not_duplicate_the_corpus(self) -> None:
        """The oracle is deterministic, so repeats add cost, not information."""

        self.assertEqual(TRAINING_PROCEDURE["demonstration_repeats"], 1)

    def test_frozen_procedure_matches_what_the_producer_passes(self) -> None:
        """The procedure document must not attest a value the code ignores."""

        import ast

        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/policy_improvement_base_policy.py"
        ).read_text()
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "imitation_pretrain"
        ]
        self.assertTrue(calls)
        expected = {
            "demonstration_episodes": "TRAIN_RECORD_COUNT",
            "demonstration_repeats": "DEMONSTRATION_REPEATS",
            "early_stop_accuracy": "EARLY_STOP_ACCURACY",
            "imitation_lr": "IMITATION_LEARNING_RATE",
            "batch_size": "IMITATION_BATCH_SIZE",
        }
        for call in calls:
            supplied = {
                keyword.arg: keyword.value
                for keyword in call.keywords
                if keyword.arg is not None
            }
            for name, constant in expected.items():
                self.assertIn(name, supplied, name)
                self.assertIsInstance(supplied[name], ast.Name, name)
                self.assertEqual(supplied[name].id, constant, name)

        # Every frozen value the trainer consumes must be an accepted keyword.
        from rl.upi_trm_trainer import UPITrmTrainer

        parameters = inspect.signature(UPITrmTrainer.imitation_pretrain).parameters
        for name in expected:
            self.assertIn(name, parameters, name)

    def test_print_procedure_emits_one_canonical_document(self) -> None:
        # The producer writes canonical bytes, as the other v2 tools do.
        class _ByteStdout(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = io.BytesIO()

        buffer = _ByteStdout()
        with redirect_stdout(buffer):
            code = main(
                [
                    "--project-root",
                    str(ROOT),
                    "--expected-git-commit",
                    "0" * 40,
                    "--dataset-root",
                    str(DATASET),
                    "--output-root",
                    "/nonexistent",
                    "--print-procedure",
                ]
            )
        self.assertEqual(code, 0)
        lines = buffer.buffer.getvalue().decode("ascii").splitlines()
        self.assertEqual(len(lines), 1)
        document = json.loads(lines[0])
        self.assertEqual(
            document["training_procedure_sha256"], training_procedure_sha256()
        )


class NeutralSharedConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = validate_v2_protocol(load_strict_json(PROTOCOL))
        self.configs = load_v2_base_configs(self.protocol, ROOT)

    def test_every_consumed_field_is_shared_by_all_four_methods(self) -> None:
        shared = neutral_shared_config(self.configs)
        self.assertEqual(set(shared), set(SHARED_CONFIG_FIELDS))
        for field in SHARED_CONFIG_FIELDS:
            values = {
                canonical_json_bytes(self.configs[method][field])
                for method in REGISTERED_METHOD_CONFIGS
            }
            self.assertEqual(len(values), 1, field)

    def test_a_method_specific_value_fails_closed(self) -> None:
        configs = {name: dict(value) for name, value in self.configs.items()}
        configs["matched_ppo"]["gamma"] = 0.5
        with self.assertRaisesRegex(
            BasePolicyProducerError, "differs across registered methods"
        ):
            neutral_shared_config(configs)

    def test_missing_method_or_field_fails_closed(self) -> None:
        without_method = {
            name: value
            for name, value in self.configs.items()
            if name != "matched_ppo"
        }
        with self.assertRaisesRegex(BasePolicyProducerError, "missing"):
            neutral_shared_config(without_method)

        without_field = {name: dict(value) for name, value in self.configs.items()}
        del without_field["fixed_base_exact_persistent"]["gamma"]
        with self.assertRaisesRegex(BasePolicyProducerError, "does not define"):
            neutral_shared_config(without_field)

    def test_method_specific_fields_are_never_consumed(self) -> None:
        """The shared base must not adopt any policy-improvement setting."""

        forbidden = {
            "training_protocol",
            "theory_exact_mixture",
            "mixture_alpha",
            "episodic_latent",
            "algorithm",
            "K",
            "exact_baseline_summation",
            "exact_k_step_targets",
            "batch_centered_advantage",
            "value_target_clip",
            "ppo_clip_eps",
            "ppo_epochs",
        }
        self.assertEqual(forbidden & set(SHARED_CONFIG_FIELDS), set())


class TrainSplitAuthenticationTest(unittest.TestCase):
    def test_registered_train_split_authenticates(self) -> None:
        identity = authenticate_train_split(DATASET.resolve())
        self.assertEqual(identity["train_record_count"], 1024)
        self.assertEqual(
            identity["dataset_manifest_sha256"],
            "2572bb79faeec976dc83cb75b8520e59691a7c9dc3f8fe252554fc29bfe90ccd",
        )
        self.assertEqual(
            identity["train_manifest_sha256"],
            "05146037857b1adb42520e80a0c2ab250053a517196c8b8ac95e002aa40c6f74",
        )
        self.assertEqual(
            identity["train_ordered_record_sha256"],
            "73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446",
        )
        self.assertFalse(identity["validation_content_opened"])
        self.assertFalse(identity["test_content_opened"])

    def test_relative_dataset_root_fails_closed(self) -> None:
        with self.assertRaisesRegex(BasePolicyProducerError, "absolute canonical"):
            authenticate_train_split(Path("data"))

    def test_authentication_reads_no_held_out_manifest(self) -> None:
        """Only the corpus and train manifests may be opened."""

        opened: list[str] = []
        real_open = Path.open

        def recording_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            opened.append(str(self))
            return real_open(self, *args, **kwargs)

        with mock.patch.object(Path, "open", recording_open):
            authenticate_train_split(DATASET.resolve())
        self.assertTrue(opened)
        for path in opened:
            self.assertNotIn("validation", path)
            self.assertNotIn("/test", path)


if __name__ == "__main__":
    unittest.main()
