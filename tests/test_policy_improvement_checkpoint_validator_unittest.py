#!/usr/bin/env fbpython
"""Focused boundary tests for semantic Stage-0 checkpoint validation."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import policy_improvement_checkpoint_validator as validator
from policy_improvement_sealed_evidence import (
    authenticate_sealed_checkpoint_field,
    seal_generation_checkpoint,
    SealedCheckpoint,
    SealedCheckpointError,
)
import policy_improvement_smoke_checkpoint as smoke_checkpoint
from policy_improvement_smoke_checkpoint import load_stable_checkpoint
import policy_improvement_smoke_runtime as smoke
from scripts.policy_improvement_registry import generate_registry
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json,
    validate_protocol,
    validate_runtime_authorization,
)
from utils.run_identity import canonical_json_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs/policy_improvement_v1/protocol.json"


def _context(
    root: Path,
    *,
    method_id: str = "fixed_base_exact_persistent",
) -> smoke.SmokeContext:
    protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
    registry = generate_registry(protocol)
    row = next(
        item
        for item in registry["rows"]
        if item["phase"] == "stage0_smoke" and item["method_id"] == method_id
    )
    return smoke.SmokeContext(
        protocol=protocol,
        protocol_sha256=hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        registry=registry,
        registry_sha256=hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
        row=row,
        registry_row_sha256=hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
        source_root=root / "project",
        dataset_root=root / "dataset",
        dataset_manifest_sha256="1" * 64,
        dataset_producer_source={
            "git_commit": "a" * 40,
            "launcher_sha256": "2" * 64,
            "runtime_sha256": "3" * 64,
            "source_manifest_sha256": "4" * 64,
        },
        evidence_root=root / "evidence",
        run_root=root / "evidence" / "runs" / str(row["run_id"]),
        segment_budget=32,
        segment_name="resume",
        runtime_sha256="5" * 64,
        runtime_authorization_sha256="6" * 64,
        runtime_profile_sha256="7" * 64,
        selected_source_manifest_sha256="7" * 64,
        launcher_sha256="8" * 64,
        producer_commit="a" * 40,
        producer_manifest_sha256="7" * 64,
    )


def _authorization(protocol_sha256: str) -> dict[str, object]:
    return validate_runtime_authorization(
        {
            "schema_name": "policy_improvement_runtime_authorization_v1",
            "schema_version": 1,
            "authorization_id": "semantic-validator-test-v1",
            "created_at_utc": "2026-08-14T12:00:00Z",
            "protocol_sha256": protocol_sha256,
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": "1" * 64,
            "launcher_sha256": "2" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "a" * 40,
                    "runtime_sha256": "3" * 64,
                    "runtime_profile_sha256": "1" * 64,
                    "selected_source_manifest_sha256": "1" * 64,
                }
                for role in (
                    "policy-improvement-training",
                    "policy-improvement-evaluation",
                    "policy-improvement-audit",
                    "policy-improvement-analysis",
                )
            ],
        }
    )


_OPEN_SEALED_FIXTURES: list[SealedCheckpoint] = []


def tearDownModule() -> None:
    while _OPEN_SEALED_FIXTURES:
        _OPEN_SEALED_FIXTURES.pop().close()


def _sealed(
    checkpoint: Path,
    checkpoint_sha256: str | None = None,
    *,
    snapshot_kind: str | None = None,
) -> SealedCheckpoint:
    """Seal one fixture checkpoint the way authenticated evidence would."""

    digest = checkpoint_sha256 or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sealed = seal_generation_checkpoint(
        checkpoint.parent,
        checkpoint.name,
        expected_sha256=digest,
        expected_size_bytes=checkpoint.stat().st_size,
    )
    # Fixture temporary directories do not reproduce the immutable generation
    # layout the validator asserts on, so name the canonical location.
    prefix = "checkpoints" if snapshot_kind is None else f"checkpoints/{snapshot_kind}"
    sealed = replace(
        sealed,
        generation_relative_path=f"{prefix}/{checkpoint.name}",
    )
    _OPEN_SEALED_FIXTURES.append(sealed)
    return sealed


def _sealed_field(checkpoint: Path, checkpoint_sha256: str) -> dict[str, object]:
    return _sealed(checkpoint, checkpoint_sha256).as_request_field()


def _request(
    root: Path,
    checkpoint: Path,
    *,
    environment_interactions: int = 16,
) -> dict[str, object]:
    protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
    registry = generate_registry(protocol)
    row = next(
        item
        for item in registry["rows"]
        if item["phase"] == "stage0_smoke" and item["method_id"] == "matched_ppo"
    )
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "schema_name": "policy_improvement_checkpoint_validation_request_v2",
        "schema_version": 2,
        "checkpoint": _sealed_field(checkpoint, checkpoint_sha256),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "registry_row": row,
        "registry_row_sha256": hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
        "project_root": str(root / "project"),
        "dataset_root": str(root / "project" / str(protocol["dataset"]["root"])),
        "evidence_root": str(root / "evidence"),
        "runtime_authorization": _authorization(protocol_sha256),
        "run_id": row["run_id"],
        "method_id": row["method_id"],
        "seed": row["seed"],
        "snapshot_kind": "interaction_matched",
        "environment_interactions": environment_interactions,
        "parent_checkpoint_sha256": None,
        "initialization_sha256": "4" * 64,
    }


class _UPITrainer:
    def __init__(self) -> None:
        self.policy_model_old = torch.nn.Linear(2, 2)
        self.policy_model_candidate = torch.nn.Linear(2, 2)
        self.target_model = torch.nn.Linear(2, 2)
        self.preinterpolation_policy_base = None
        self.preinterpolation_policy_candidate = None

    def train_step(self) -> None:
        raise AssertionError("training must not run")


class PolicyImprovementCheckpointValidatorTest(unittest.TestCase):
    def test_stage0_context_verifies_only_train_and_validation_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _context(root)
            authorization = _authorization(fixture.protocol_sha256)
            dataset = fixture.protocol["dataset"]
            split_manifests = {
                split: dataset["splits"][split]["manifest_sha256"]["value"]
                for split in ("train", "validation", "test")
            }
            split_orders = {
                split: dataset["splits"][split]["ordered_record_sha256"]["value"]
                for split in ("train", "validation", "test")
            }
            verified = {
                "manifest_sha256": dataset["manifest_sha256"]["value"],
                "split_manifest_sha256": split_manifests,
                "split_ordered_record_sha256": split_orders,
            }
            with (
                mock.patch.object(
                    validator,
                    "discover_clean_git_source",
                    return_value={"git_commit": "a" * 40, "git_clean": True},
                ),
                mock.patch.object(
                    validator,
                    "load_registered_base_configs",
                    return_value={},
                ),
                mock.patch.object(
                    validator,
                    "generate_registry",
                    return_value=fixture.registry,
                ),
                mock.patch.object(
                    validator,
                    "verify_dataset",
                    return_value=verified,
                ) as verify_dataset,
            ):
                validator._build_context(
                    {"environment_interactions": 16},
                    fixture.protocol,
                    fixture.row,
                    authorization,
                    root,
                    root / "dataset",
                    root / "evidence",
                )
            verify_dataset.assert_called_once_with(
                root / "dataset",
                owner_root=root,
                expected_producer=mock.ANY,
                verify_test_content=False,
            )

    def test_request_rejects_non_smoke_interaction_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
            (root / "project" / str(protocol["dataset"]["root"])).mkdir(parents=True)
            generation = (
                root
                / "evidence"
                / "runs"
                / "placeholder"
                / "segments"
                / "env_000000016"
                / "checkpoints"
            )
            generation.mkdir(parents=True)
            checkpoint = generation / "checkpoint.pt"
            checkpoint.write_bytes(b"not used")
            request = _request(root, checkpoint)
            run_id = str(request["run_id"])
            correct_parent = (
                root
                / "evidence"
                / "runs"
                / run_id
                / "segments"
                / "env_000000016"
                / "checkpoints"
            )
            correct_parent.mkdir(parents=True)
            correct_checkpoint = correct_parent / "checkpoint.pt"
            checkpoint.replace(correct_checkpoint)
            correct_sha256 = hashlib.sha256(
                correct_checkpoint.read_bytes()
            ).hexdigest()
            request["checkpoint"] = _sealed_field(correct_checkpoint, correct_sha256)
            request["checkpoint_sha256"] = correct_sha256

            validator._validate_request(request)
            request["environment_interactions"] = 17
            with self.assertRaisesRegex(
                validator.PolicyImprovementCheckpointValidationError,
                "Stage-0 snapshot",
            ):
                validator._validate_request(request)

    def test_ppo_validation_reopens_bytes_and_derives_native_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = torch.nn.Linear(2, 2)
            payload = {
                "progress": {"environment_interactions": 32},
                "parent_checkpoint_sha256": "9" * 64,
                "model_state_dict": model.state_dict(),
            }
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            request = {
                "checkpoint_sha256": digest,
            }
            session = SimpleNamespace(
                trainer=object(),
                effective_config={"registered": True},
            )
            with mock.patch.object(
                validator,
                "validate_ppo_smoke_checkpoint",
            ) as strict_validate:
                result = validator._validate_ppo(request, _sealed(path), session)

            self.assertEqual(result[0], digest)
            self.assertEqual(result[1], "9" * 64)
            self.assertEqual(result[2], 32)
            self.assertEqual(result[3], canonical_json_sha256(result[4]))
            strict_validate.assert_called_once_with(
                mock.ANY,
                session.trainer,
                expected_identity=session.effective_config,
                validate_only=True,
            )

    def test_upi_validation_strictly_restores_every_model_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            parent = "a" * 64
            model = torch.nn.Linear(2, 2)
            trainer = _UPITrainer()
            session = SimpleNamespace(
                model=model,
                trainer=trainer,
                device=torch.device("cpu"),
                dataset_provenance={"dataset": "registered"},
                run_identity={"run": "registered"},
                evidence_identity={"evidence": "registered"},
            )
            raw = {
                "policy_improvement_smoke_identity": {
                    "schema_version": 1,
                    "run_id": context.row["run_id"],
                    "method_id": context.row["method_id"],
                    "seed": context.row["seed"],
                    "environment_interactions": 32,
                    "parent_checkpoint_sha256": parent,
                },
                "evidence_identity": session.evidence_identity,
                "training_invocation": {
                    "training_seed": context.row["seed"],
                    "run_id": context.row["run_id"],
                },
                "progress": {"env_steps": 32},
                "checkpoint_lineage": {"parent_checkpoint_sha256": parent},
                "model_state_dict": model.state_dict(),
                "policy_model_old_state_dict": trainer.policy_model_old.state_dict(),
                "policy_model_candidate_state_dict": (
                    trainer.policy_model_candidate.state_dict()
                ),
                "target_model_state_dict": trainer.target_model.state_dict(),
            }
            checkpoint = root / "checkpoint.pt"
            torch.save(raw, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            class TrainingModule:
                audit_calls = 0

                @staticmethod
                def _load_checkpoint_payload(
                    checkpoint_path: str | int, *, expected_sha256: str
                ) -> tuple[object, str]:
                    # The validator must hand loaders a sealed descriptor, never
                    # a pathname it could resolve a second time.
                    self.assertIsInstance(checkpoint_path, int)
                    self.assertNotIsInstance(checkpoint_path, bool)
                    with os.fdopen(os.dup(int(checkpoint_path)), "rb") as handle:
                        handle.seek(0)
                        payload_bytes = handle.read()
                        observed = hashlib.sha256(payload_bytes).hexdigest()
                        if observed != expected_sha256:
                            raise RuntimeError("digest mismatch")
                        handle.seek(0)
                        return (
                            torch.load(
                                handle,
                                map_location="cpu",
                                weights_only=False,
                            ),
                            observed,
                        )

                @classmethod
                def validate_checkpoint_state_for_audit(
                    cls,
                    checkpoint_path: str | int,
                    active_model: torch.nn.Module,
                    active_trainer: _UPITrainer,
                    device: str,
                    **kwargs: object,
                ) -> int:
                    assert isinstance(checkpoint_path, int) and not isinstance(
                        checkpoint_path, bool
                    ), "strict audit restore must consume the sealed descriptor"
                    del checkpoint_path, device
                    cls.audit_calls += 1
                    self.assertEqual(
                        kwargs["authorized_originating_runtime_sha256"],
                        context.runtime_sha256,
                    )
                    active_model.load_state_dict(raw["model_state_dict"], strict=True)
                    active_trainer.policy_model_old.load_state_dict(
                        raw["policy_model_old_state_dict"], strict=True
                    )
                    active_trainer.policy_model_candidate.load_state_dict(
                        raw["policy_model_candidate_state_dict"], strict=True
                    )
                    active_trainer.target_model.load_state_dict(
                        raw["target_model_state_dict"], strict=True
                    )
                    return 0

            request = {
                "checkpoint_sha256": digest,
                "environment_interactions": 32,
                "parent_checkpoint_sha256": parent,
                "seed": context.row["seed"],
                "run_id": context.row["run_id"],
            }
            sealed = _sealed(checkpoint, digest)
            result = validator._validate_upi(
                request,
                sealed,
                context,
                session,
                TrainingModule,
            )
            self.assertEqual(TrainingModule.audit_calls, 1)
            self.assertEqual(result[0], digest)
            self.assertEqual(result[1], parent)
            self.assertEqual(result[2], 32)
            self.assertEqual(result[3], canonical_json_sha256(result[4]))

            request["parent_checkpoint_sha256"] = "b" * 64
            with self.assertRaises(smoke.PolicyImprovementSmokeError):
                validator._validate_upi(
                    request,
                    sealed,
                    context,
                    session,
                    TrainingModule,
                )

    def test_top_level_returns_zero_execution_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"authenticated by patched semantic loader")
            context = _context(root, method_id="matched_ppo")
            row = context.row
            request = {field: None for field in validator._REQUEST_FIELDS}
            request.update(
                {
                    "run_id": row["run_id"],
                    "method_id": row["method_id"],
                    "seed": row["seed"],
                    "snapshot_kind": "interaction_matched",
                    "environment_interactions": 16,
                    "parent_checkpoint_sha256": None,
                    "checkpoint_sha256": "1" * 64,
                    "initialization_sha256": "2" * 64,
                }
            )
            trainer = SimpleNamespace(train_step=lambda: None)
            method_registration = next(
                method
                for method in context.protocol["methods"]
                if method["id"] == row["method_id"]
            )
            session = SimpleNamespace(
                trainer=trainer,
                initialization_sha256="2" * 64,
                method_config_sha256=method_registration["config_sha256"],
                effective_config_sha256="3" * 64,
                dataset_provenance={"registered": True},
                run_identity=None,
                evaluation_started=False,
            )
            semantic = (
                "1" * 64,
                None,
                16,
                canonical_json_sha256({"model": "4" * 64}),
                {"model": "4" * 64},
            )
            with (
                mock.patch.object(
                    validator,
                    "_validate_request",
                    return_value=(
                        context.protocol,
                        row,
                        {},
                        root,
                        root,
                        root,
                        _sealed(checkpoint),
                    ),
                ),
                mock.patch.object(validator, "_build_context", return_value=context),
                mock.patch.object(
                    validator,
                    "build_stage0_validation_session",
                    return_value=session,
                ),
                mock.patch.object(validator, "_validate_ppo", return_value=semantic),
                mock.patch.object(
                    validator.importlib,
                    "import_module",
                    return_value=object(),
                ),
            ):
                result = validator.validate_checkpoint(request)

            self.assertEqual(result["training_call_delta"], 0)
            self.assertEqual(result["evaluation_call_delta"], 0)
            self.assertEqual(result["optimizer_step_delta"], 0)
            self.assertEqual(result["initialization_sha256"], "2" * 64)

    def test_private_checkpoint_rewrite_binds_legacy_parent_in_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save({"checkpoint_schema_version": 4}, checkpoint)
            context = _context(root, method_id="legacy_parameter_interpolation")
            identity = smoke._policy_improvement_smoke_identity(
                context,
                environment_interactions=32,
                parent_checkpoint_sha256="f" * 64,
            )
            smoke._rewrite_checkpoint_with_smoke_identity(
                checkpoint,
                {"checkpoint_schema_version": 4},
                identity,
            )
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["policy_improvement_smoke_identity"], identity)
            smoke.validate_policy_improvement_smoke_identity(
                loaded["policy_improvement_smoke_identity"],
                context,
                environment_interactions=32,
                parent_checkpoint_sha256="f" * 64,
            )




def _write_marker(marker: str) -> str:
    Path(marker).write_bytes(b"the reducer executed")
    return marker


class _MarkerReducer:
    """A checkpoint payload whose reducer writes a marker file on load."""

    def __init__(self, marker: Path) -> None:
        self._marker = str(marker)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (_write_marker, (self._marker,))


class SealedCheckpointLoaderTest(unittest.TestCase):
    """Adversarial tests 6 and 7 against the real PPO/UPI loader entry."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.marker = self.root / "reducer-marker"
        self.checkpoint = self.root / "checkpoint.pt"
        torch.save({"payload": _MarkerReducer(self.marker)}, self.checkpoint)
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.size = self.checkpoint.stat().st_size

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_ppo_loader_refuses_a_hostile_reducer_it_is_told_to_load(self) -> None:
        """The end-to-end proof at the real PPO loader.

        Here the bytes are authenticated: the sealed descriptor carries exactly
        the digest the loader is told to expect, so every byte-level check
        passes and the loader proceeds.  It must still refuse to execute.
        """

        sealed = _sealed(self.checkpoint, self.digest)
        self.assertFalse(self.marker.exists())
        with self.assertRaises(
            smoke_checkpoint.PolicyImprovementSmokeCheckpointError
        ):
            load_stable_checkpoint(sealed.descriptor, expected_sha256=self.digest)
        self.assertFalse(self.marker.exists())

        # Positive control: the identical authenticated bytes DO execute under
        # the old unrestricted policy, so the assertion above is not vacuous.
        with open(self.checkpoint, "rb") as handle:
            torch.load(handle, map_location="cpu", weights_only=False)
        self.assertTrue(self.marker.exists())

    def test_unsealed_descriptor_never_reaches_the_loader(self) -> None:
        descriptor = os.memfd_create("unsealed", os.MFD_ALLOW_SEALING)
        try:
            os.write(descriptor, self.checkpoint.read_bytes())
            request = {
                "checkpoint": {
                    "schema_name": "policy_improvement_sealed_checkpoint_v1",
                    "descriptor": descriptor,
                    "sha256": self.digest,
                    "size_bytes": self.size,
                    "generation_relative_path": "checkpoints/checkpoint.pt",
                },
                "checkpoint_sha256": self.digest,
            }
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    request["checkpoint"], expected_sha256=self.digest
                )
        finally:
            os.close(descriptor)
        self.assertFalse(self.marker.exists())

    def test_pathname_in_place_of_a_descriptor_never_reaches_the_loader(self) -> None:
        for hostile in (
            str(self.checkpoint),
            {"path": str(self.checkpoint)},
            {
                "schema_name": "policy_improvement_sealed_checkpoint_v1",
                "descriptor": str(self.checkpoint),
                "sha256": self.digest,
                "size_bytes": self.size,
                "generation_relative_path": "checkpoints/checkpoint.pt",
            },
        ):
            with self.subTest(hostile=type(hostile).__name__):
                with self.assertRaises(SealedCheckpointError):
                    authenticate_sealed_checkpoint_field(
                        hostile, expected_sha256=self.digest
                    )
        self.assertFalse(self.marker.exists())

    def _generation_request(self, name: str) -> dict[str, object]:
        """Build a request whose roots and layout the validator accepts."""

        root = self.root / name
        protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
        (root / "project" / str(protocol["dataset"]["root"])).mkdir(parents=True)
        placeholder = root / "evidence" / "placeholder"
        placeholder.mkdir(parents=True)
        request = _request(root, self.checkpoint)
        generation = (
            root
            / "evidence"
            / "runs"
            / str(request["run_id"])
            / "segments"
            / "env_000000016"
            / "checkpoints"
        )
        generation.mkdir(parents=True)
        published = generation / "checkpoint.pt"
        published.write_bytes(self.checkpoint.read_bytes())
        request["checkpoint"] = _sealed_field(published, self.digest)
        request["checkpoint_sha256"] = self.digest
        # Sanity: the untampered request authenticates, so the negative cases
        # below fail for the reason under test rather than a fixture defect.
        validator._validate_request(request)
        self.assertFalse(self.marker.exists())
        return request

    def test_request_validation_rejects_hostile_bytes_before_any_load(self) -> None:
        request = self._generation_request("fixture")
        # A hostile package that declares a digest it does not have never
        # reaches deserialization: the sealed descriptor is authenticated first.
        request["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            validator.PolicyImprovementCheckpointValidationError,
            "authenticated sealed",
        ):
            validator._validate_request(request)
        self.assertFalse(self.marker.exists())

    def test_request_validation_rejects_an_out_of_generation_location(self) -> None:
        request = self._generation_request("fixture2")
        for hostile in (
            "attacker/checkpoint.pt",
            "checkpoint.pt",
            "checkpoints/interaction_matched/checkpoint.pt",
        ):
            with self.subTest(relative_path=hostile):
                sealed = dict(request["checkpoint"])
                sealed["generation_relative_path"] = hostile
                with self.assertRaisesRegex(
                    validator.PolicyImprovementCheckpointValidationError,
                    "immutable generation location",
                ):
                    validator._validate_request({**request, "checkpoint": sealed})
        self.assertFalse(self.marker.exists())

    def test_non_smoke_requests_bind_the_snapshot_kind_directory(self) -> None:
        """Cover the ``checkpoints/<snapshot_kind>/<name>`` branch."""

        root = self.root / "non-smoke"
        protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
        (root / "project" / str(protocol["dataset"]["root"])).mkdir(parents=True)
        registry = generate_registry(protocol)
        row = next(
            item
            for item in registry["rows"]
            if item["phase"] == "stage1_screen"
            and item["row_kind"] == "concrete"
            and item["tier"] != "smoke"
            and item["evaluation_split"] == "validation"
        )
        protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        authorization = _authorization(protocol_sha256)
        budget = protocol["budgets"][str(row["tier"])]
        final_interactions = int(budget["environment_interactions"])
        snapshot_kind = "compute_matched"
        generation = (
            root
            / "evidence"
            / "runs"
            / str(row["run_id"])
            / "segments"
            / f"env_{final_interactions:09d}"
        )
        published = generation / "checkpoints" / snapshot_kind / "rl_checkpoint.pt"
        published.parent.mkdir(parents=True)
        published.write_bytes(self.checkpoint.read_bytes())

        def build(relative_path: str) -> dict[str, object]:
            sealed = _sealed(published, self.digest)
            field = sealed.as_request_field()
            field["generation_relative_path"] = relative_path
            return {
                "schema_name": "policy_improvement_checkpoint_validation_request_v2",
                "schema_version": 2,
                "checkpoint": field,
                "checkpoint_sha256": self.digest,
                "protocol": protocol,
                "protocol_sha256": protocol_sha256,
                "registry_row": row,
                "registry_row_sha256": hashlib.sha256(
                    canonical_json_bytes(row)
                ).hexdigest(),
                "project_root": str(root / "project"),
                "dataset_root": str(
                    root / "project" / str(protocol["dataset"]["root"])
                ),
                "evidence_root": str(root / "evidence"),
                "runtime_authorization": authorization,
                "run_id": row["run_id"],
                "method_id": row["method_id"],
                "seed": row["seed"],
                "snapshot_kind": snapshot_kind,
                "environment_interactions": final_interactions,
                "parent_checkpoint_sha256": None,
                "initialization_sha256": "7" * 64,
                "registry_sha256": "8" * 64,
                "amendment_history_sha256": "9" * 64,
                "runtime_authorization_sha256": "a" * 64,
                "recurrent_map_applications": 1,
                "compute_target_recurrent_map_applications": 1,
                "test_open_sha256": None,
            }

        # The exact published layout is accepted, so the negatives below fail
        # for the reason under test.
        accepted = build(f"checkpoints/{snapshot_kind}/rl_checkpoint.pt")
        _, checked_row, _, _, _, _, sealed = validator._validate_request(accepted)
        self.assertEqual(checked_row["run_id"], row["run_id"])
        self.assertEqual(
            sealed.generation_relative_path,
            f"checkpoints/{snapshot_kind}/rl_checkpoint.pt",
        )

        for hostile in (
            "checkpoints/rl_checkpoint.pt",
            "checkpoints/interaction_matched/rl_checkpoint.pt",
            "checkpoints/scheduled/env_000000010/rl_checkpoint.pt",
            f"evaluations/{snapshot_kind}/rl_checkpoint.pt",
        ):
            with self.subTest(relative_path=hostile):
                with self.assertRaisesRegex(
                    validator.PolicyImprovementCheckpointValidationError,
                    "immutable generation location",
                ):
                    validator._validate_request(build(hostile))

    def test_importing_the_validator_writes_nothing_to_stdout(self) -> None:
        """The deferred Torch import must not break the one-JSON-document rule.

        The audit now imports Torch, the training module, and every checkpoint
        loader on the first sealed validation request, which happens while the
        audit report is being composed.  Anything those imports print would put
        a second document on audit stdout.
        """

        program = (
            "import io, sys\n"
            "from contextlib import redirect_stdout\n"
            "stream = io.StringIO()\n"
            "with redirect_stdout(stream):\n"
            "    import policy_improvement_checkpoint_validator\n"
            "captured = stream.getvalue()\n"
            "sys.stderr.write(repr(captured))\n"
            "assert captured == '', captured\n"
            "print('silent')\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        self.assertEqual(completed.stdout.strip(), "silent")


if __name__ == "__main__":
    unittest.main()
