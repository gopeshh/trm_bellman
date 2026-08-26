#!/usr/bin/env fbpython
"""Negative coverage for the Stage 1 train-only base-policy restore path."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from phase4_runtime_launcher import (
    _normalize_child_args,
    ConfirmatoryRuntimeError,
    POLICY_IMPROVEMENT_FULL_PURPOSE,
    POLICY_SMOKE_PURPOSE,
)
from policy_improvement_smoke_runtime import (
    build_protocol_v2_model_config,
    STAGE0_INITIALIZATION_KIND,
)
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from scripts.policy_improvement_base_policy_restore import (
    apply_base_policy_state,
    authenticate_base_policy_artifact,
    BASE_POLICY_INITIALIZATION_KIND,
    BASE_POLICY_PRODUCER_SCHEMA_NAME,
    BasePolicyRestoreError,
    canonical_sha256,
    restore_base_policy_state,
)
from scripts.policy_improvement_v2_schema import (
    load_strict_json,
    sha256_json,
    validate_v2_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs/policy_improvement_v2"
HEX = "0" * 64
PROJECT_ROOT = "/home/buiksat/trm_bellman"


class _RLConfig:
    batch_size = 4
    enable_contraction = False
    target_Lz = 0.9
    target_Lv = 1.0
    disable_value_head_norm = False
    latent_projection_mode = "enabled"
    latent_ball_radius = 10.0


def _protocol() -> dict[str, Any]:
    return validate_v2_protocol(load_strict_json(V2 / "protocol.json"))


def _model_config() -> dict[str, Any]:
    protocol = _protocol()
    return build_protocol_v2_model_config(
        architecture=protocol["architecture"],
        rl_config=_RLConfig(),
        seq_len=16,
        vocab_size=5,
        num_identifiers=8,
        action_count=16 * 5 + 1,
    )


class BasePolicyRestoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.architecture = _protocol()["architecture"]
        self.model_config = _model_config()
        # The amendment binds the protocol architecture; the payload binds the
        # full model configuration built from it.
        self.architecture_sha256 = canonical_sha256(self.architecture)
        self.model_config_sha256 = canonical_sha256(self.model_config)
        torch.manual_seed(1904261137)
        model = TinyRecursiveReasoningModel_ACTV1(self.model_config)
        self.state = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        self.model_state_sha256 = state_dict_sha256(self.state)
        self.path = self._write_artifact("base_policy.pt", self._payload())
        self.amendment = self._amendment(self.path)

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        payload = {
            "schema_name": BASE_POLICY_PRODUCER_SCHEMA_NAME,
            "schema_version": 1,
            "initialization_kind": BASE_POLICY_INITIALIZATION_KIND,
            "model_config": self.model_config,
            "architecture_sha256": self.model_config_sha256,
            "model_state": self.state,
            "model_state_sha256": self.model_state_sha256,
            "training_procedure_sha256": "1" * 64,
            "producer_git_commit": "a" * 40,
        }
        payload.update(overrides)
        return payload

    def _write_artifact(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        torch.save(payload, path)
        return path

    def _amendment(self, path: Path, **artifact_overrides: Any) -> dict[str, Any]:
        protocol = validate_v2_protocol(load_strict_json(V2 / "protocol.json"))
        populations = load_strict_json(V2 / "populations.json")
        registry = load_strict_json(V2 / "registry.json")
        artifact = {
            "status": "available",
            "initialization_kind": BASE_POLICY_INITIALIZATION_KIND,
            "architecture_sha256": self.architecture_sha256,
            "model_state_sha256": self.model_state_sha256,
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": HEX,
            "training_dataset_manifest_sha256": HEX,
            "training_split": "train",
            "training_split_ordered_record_sha256": HEX,
            "training_procedure_sha256": "1" * 64,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_size_bytes": path.stat().st_size,
            "shared_across_persistent_and_episodic": True,
            "not_selected_by_validation_or_test": True,
        }
        artifact.update(artifact_overrides)
        return {
            "schema_name": "policy_improvement_base_policy_amendment_v2",
            "schema_version": 1,
            "amendment_id": "experiment-0-train-only-base-policy",
            "created_at_utc": "2026-08-26T01:48:30Z",
            "protocol_id": protocol["protocol_id"],
            "protocol_schema_name": protocol["schema_name"],
            "protocol_schema_version": protocol["schema_version"],
            "protocol_sha256": sha256_json(protocol),
            "population_registry_sha256": sha256_json(populations),
            "source_registry_schema_name": registry["schema_name"],
            "source_registry_schema_version": registry["registry_schema_version"],
            "source_registry_sha256": sha256_json(registry),
            "prior_amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
            "runtime_authorization_sha256": HEX,
            "validation_data_inspected": False,
            "test_data_opened": False,
            "base_policy_artifact": artifact,
        }

    # --- positive -------------------------------------------------------

    def test_registered_artifact_authenticates_and_restores(self) -> None:
        authenticated = authenticate_base_policy_artifact(
            self.path, amendment=self.amendment
        )
        self.assertEqual(authenticated.model_state_sha256, self.model_state_sha256)
        self.assertEqual(authenticated.size_bytes, self.path.stat().st_size)
        restored = restore_base_policy_state(
            authenticated,
            model_config=self.model_config,
            protocol_architecture=self.architecture,
        )
        self.assertEqual(state_dict_sha256(restored), self.model_state_sha256)

    def test_persistent_and_episodic_start_byte_identical(self) -> None:
        """Both latent modes must load the same frozen weights."""

        authenticated = authenticate_base_policy_artifact(
            self.path, amendment=self.amendment
        )
        digests = []
        for seed in (11, 22):
            torch.manual_seed(seed)
            model = TinyRecursiveReasoningModel_ACTV1(self.model_config)
            self.assertNotEqual(
                state_dict_sha256(model.state_dict()), self.model_state_sha256
            )
            digests.append(
                apply_base_policy_state(
                    model,
                    authenticated,
                    model_config=self.model_config,
                    protocol_architecture=self.architecture,
                )
            )
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(digests[0], self.model_state_sha256)

    # --- negative -------------------------------------------------------

    def test_authentication_never_deserializes_the_artifact(self) -> None:
        """The whole-file digest is proven while the file is still bytes."""

        with mock.patch.object(
            torch, "load", side_effect=AssertionError("deserialized too early")
        ):
            authenticate_base_policy_artifact(self.path, amendment=self.amendment)
            tampered = self._amendment(self.path, checkpoint_sha256="b" * 64)
            with self.assertRaisesRegex(BasePolicyRestoreError, "digest differs"):
                authenticate_base_policy_artifact(self.path, amendment=tampered)

    def test_wrong_checkpoint_digest_is_rejected(self) -> None:
        amendment = self._amendment(self.path, checkpoint_sha256="b" * 64)
        with self.assertRaisesRegex(BasePolicyRestoreError, "digest differs"):
            authenticate_base_policy_artifact(self.path, amendment=amendment)

    def test_wrong_size_is_rejected(self) -> None:
        amendment = self._amendment(self.path, checkpoint_size_bytes=123)
        with self.assertRaisesRegex(BasePolicyRestoreError, "size differs"):
            authenticate_base_policy_artifact(self.path, amendment=amendment)

    def test_arbitrary_caller_selected_checkpoint_is_rejected(self) -> None:
        """A different artifact cannot be substituted for the registered one."""

        other = self._write_artifact(
            "caller_selected.pt", self._payload(training_procedure_sha256="2" * 64)
        )
        self.assertNotEqual(other.read_bytes(), self.path.read_bytes())
        with self.assertRaises(BasePolicyRestoreError):
            authenticate_base_policy_artifact(other, amendment=self.amendment)

    def test_amendment_naming_another_protocol_architecture_is_rejected(
        self,
    ) -> None:
        authenticated = authenticate_base_policy_artifact(
            self.path, amendment=self.amendment
        )
        other = dict(self.architecture)
        other["hidden_size"] = int(other["hidden_size"]) * 2
        with self.assertRaisesRegex(
            BasePolicyRestoreError, "another protocol architecture"
        ):
            restore_base_policy_state(
                authenticated,
                model_config=self.model_config,
                protocol_architecture=other,
            )

    def test_payload_lying_about_its_model_configuration_is_rejected(self) -> None:
        path = self._write_artifact(
            "lying_architecture.pt", self._payload(architecture_sha256="d" * 64)
        )
        authenticated = authenticate_base_policy_artifact(
            path, amendment=self._amendment(path)
        )
        with self.assertRaisesRegex(
            BasePolicyRestoreError, "model configuration differs from its identity"
        ):
            restore_base_policy_state(
                authenticated,
                model_config=self.model_config,
                protocol_architecture=self.architecture,
            )

    def test_wrong_architecture_is_rejected(self) -> None:
        authenticated = authenticate_base_policy_artifact(
            self.path, amendment=self.amendment
        )
        other = dict(self.model_config)
        other["hidden_size"] = int(other["hidden_size"]) * 2
        with self.assertRaisesRegex(BasePolicyRestoreError, "architecture differs"):
            restore_base_policy_state(
                authenticated,
                model_config=other,
                protocol_architecture=self.architecture,
            )

    def test_wrong_model_state_is_rejected(self) -> None:
        torch.manual_seed(7)
        different = TinyRecursiveReasoningModel_ACTV1(self.model_config)
        payload = self._payload(
            model_state={
                name: tensor.detach().cpu()
                for name, tensor in different.state_dict().items()
            }
        )
        path = self._write_artifact("mismatched_state.pt", payload)
        # The payload's declared digest and the amendment still name the
        # original identity, so only hashing the restored tensors catches this.
        amendment = self._amendment(path, model_state_sha256=self.model_state_sha256)
        authenticated = authenticate_base_policy_artifact(path, amendment=amendment)
        with self.assertRaisesRegex(BasePolicyRestoreError, "model state differs"):
            restore_base_policy_state(
                authenticated,
                model_config=self.model_config,
                protocol_architecture=self.architecture,
            )

    def test_lying_declared_model_state_digest_is_rejected(self) -> None:
        path = self._write_artifact(
            "lying_digest.pt", self._payload(model_state_sha256="c" * 64)
        )
        amendment = self._amendment(path, model_state_sha256=self.model_state_sha256)
        authenticated = authenticate_base_policy_artifact(path, amendment=amendment)
        with self.assertRaisesRegex(BasePolicyRestoreError, "model_state_sha256"):
            restore_base_policy_state(
                authenticated,
                model_config=self.model_config,
                protocol_architecture=self.architecture,
            )

    def test_symlinked_and_relative_artifacts_are_rejected(self) -> None:
        link = self.root / "link.pt"
        link.symlink_to(self.path)
        with self.assertRaises(BasePolicyRestoreError):
            authenticate_base_policy_artifact(link, amendment=self.amendment)
        with self.assertRaisesRegex(BasePolicyRestoreError, "absolute"):
            authenticate_base_policy_artifact(
                Path("base_policy.pt"), amendment=self.amendment
            )

    def test_unavailable_or_stress_base_amendment_is_rejected(self) -> None:
        unavailable = copy.deepcopy(self.amendment)
        unavailable["base_policy_artifact"]["status"] = "unavailable"
        with self.assertRaises(BasePolicyRestoreError):
            authenticate_base_policy_artifact(self.path, amendment=unavailable)

        stress = copy.deepcopy(self.amendment)
        stress["base_policy_artifact"]["initialization_kind"] = "random_base_stress"
        with self.assertRaisesRegex(BasePolicyRestoreError, "train-only pretrained"):
            authenticate_base_policy_artifact(self.path, amendment=stress)

    def test_foreign_payload_schema_is_rejected(self) -> None:
        path = self._write_artifact(
            "foreign.pt", self._payload(schema_name="something_else")
        )
        authenticated = authenticate_base_policy_artifact(
            path, amendment=self._amendment(path)
        )
        with self.assertRaisesRegex(BasePolicyRestoreError, "schema differs"):
            restore_base_policy_state(
                authenticated,
                model_config=self.model_config,
                protocol_architecture=self.architecture,
            )

    def test_foreign_producer_commit_is_rejected(self) -> None:
        path = self._write_artifact(
            "foreign_commit.pt", self._payload(producer_git_commit="b" * 40)
        )
        amendment = self._amendment(path, producer_git_commit="a" * 40)
        authenticated = authenticate_base_policy_artifact(path, amendment=amendment)
        with self.assertRaisesRegex(BasePolicyRestoreError, "producer_git_commit"):
            restore_base_policy_state(
                authenticated,
                model_config=self.model_config,
                protocol_architecture=self.architecture,
            )


class Stage0IsolationTest(unittest.TestCase):
    def test_stage0_keeps_random_initialization(self) -> None:
        self.assertEqual(STAGE0_INITIALIZATION_KIND, "random")
        self.assertNotEqual(STAGE0_INITIALIZATION_KIND, BASE_POLICY_INITIALIZATION_KIND)

    def test_stage0_runtime_never_reaches_the_restore_path(self) -> None:
        source = (ROOT / "policy_improvement_smoke_runtime.py").read_text()
        self.assertNotIn("base_policy_restore", source)
        self.assertNotIn("apply_base_policy_state", source)

    def test_smoke_purpose_rejects_a_base_artifact(self) -> None:
        """Stage 0 cannot be handed the artifact through launcher argv."""

        smoke = [
            "--dataset-root",
            "/tmp/data",
            "--evidence-root",
            "/tmp/evidence",
            "--policy-improvement-protocol",
            "/tmp/protocol.json",
            "--policy-improvement-row-id",
            "s0-x",
            "--policy-improvement-smoke-segment",
            "prepare",
            "--train-manifest-sha256",
            HEX,
        ]
        _normalize_child_args(
            POLICY_SMOKE_PURPOSE, PROJECT_ROOT, smoke, policy_protocol_v2=True
        )
        with self.assertRaisesRegex(ConfirmatoryRuntimeError, "unsupported child"):
            _normalize_child_args(
                POLICY_SMOKE_PURPOSE,
                PROJECT_ROOT,
                ["--base-policy-artifact", "/tmp/base_policy.pt", *smoke],
                policy_protocol_v2=True,
            )


class FullLauncherArgumentTest(unittest.TestCase):
    ARGUMENTS = [
        "--evidence-root",
        "/tmp/evidence",
        "--dataset-root",
        "/tmp/data",
        "--row-id",
        "s1-fixed_base_exact_persistent-n2-k1-s784831257",
    ]

    def _normalize(self, arguments: list[str]) -> list[str]:
        return _normalize_child_args(
            POLICY_IMPROVEMENT_FULL_PURPOSE,
            PROJECT_ROOT,
            arguments,
            policy_project_root=PROJECT_ROOT,
            runtime_authorization_sha256=HEX,
            policy_protocol_v2=True,
        )

    def test_full_purpose_forwards_the_base_artifact(self) -> None:
        child = self._normalize(
            [*self.ARGUMENTS, "--base-policy-artifact", "/tmp/base_policy.pt"]
        )
        self.assertIn("--base-policy-artifact", child)
        self.assertEqual(
            child[child.index("--base-policy-artifact") + 1], "/tmp/base_policy.pt"
        )

    def test_full_purpose_requires_the_base_artifact(self) -> None:
        with self.assertRaises(ConfirmatoryRuntimeError):
            self._normalize(list(self.ARGUMENTS))

    def test_full_purpose_rejects_a_relative_base_artifact(self) -> None:
        with self.assertRaisesRegex(ConfirmatoryRuntimeError, "absolute"):
            self._normalize(
                [*self.ARGUMENTS, "--base-policy-artifact", "base_policy.pt"]
            )

    def test_full_purpose_rejects_a_repeated_base_artifact(self) -> None:
        with self.assertRaisesRegex(ConfirmatoryRuntimeError, "singleton"):
            self._normalize(
                [
                    *self.ARGUMENTS,
                    "--base-policy-artifact",
                    "/tmp/a.pt",
                    "--base-policy-artifact",
                    "/tmp/b.pt",
                ]
            )


if __name__ == "__main__":
    unittest.main()
