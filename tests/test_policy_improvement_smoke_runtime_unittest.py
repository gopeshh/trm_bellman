#!/usr/bin/env fbpython
"""Focused transaction and measurement tests for the Stage 0 runtime."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import policy_improvement_smoke_runtime as smoke
import torch
from scripts.policy_improvement_registry import generate_registry
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json,
    RESULT_SCHEMA_VERSION,
    validate_protocol,
    validate_result,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs/policy_improvement_v1/protocol.json"


class PolicyImprovementSmokeRuntimeTest(unittest.TestCase):
    def _context_and_session(
        self,
        root: Path,
    ) -> tuple[smoke.SmokeContext, smoke.SmokeSession]:
        protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
        registry = generate_registry(protocol)
        row = next(item for item in registry["rows"] if item["phase"] == "stage0_smoke")
        method = next(
            item for item in protocol["methods"] if item["id"] == row["method_id"]
        )
        evidence_root = root / "evidence"
        evidence_root.mkdir(mode=0o700)
        context = smoke.SmokeContext(
            protocol=protocol,
            protocol_sha256=hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
            registry=registry,
            registry_sha256=hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
            row=row,
            registry_row_sha256=hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
            source_root=root / "source",
            dataset_root=root / "dataset",
            dataset_manifest_sha256="1" * 64,
            dataset_producer_source={
                "git_commit": "a" * 40,
                "launcher_sha256": "2" * 64,
                "runtime_sha256": "3" * 64,
                "source_manifest_sha256": "4" * 64,
            },
            evidence_root=evidence_root,
            run_root=(
                evidence_root / "policy-improvement-v1" / "runs" / str(row["run_id"])
            ),
            segment_budget=32,
            segment_name="resume",
            runtime_sha256="5" * 64,
            runtime_authorization_sha256="6" * 64,
            runtime_profile_sha256="7" * 64,
            selected_source_manifest_sha256="7" * 64,
            launcher_sha256="8" * 64,
            producer_commit="a" * 40,
            producer_manifest_sha256="9" * 64,
        )
        session = smoke.SmokeSession(
            model=SimpleNamespace(),
            trainer=SimpleNamespace(),
            rl_config=SimpleNamespace(),
            env_config=SimpleNamespace(),
            train_dataset=object(),
            evaluation_dataset=object(),
            checker=None,
            task_config=None,
            dataset_provenance={},
            effective_config={},
            effective_config_sha256="b" * 64,
            initialization_sha256="c" * 64,
            device=torch.device("cpu"),
            config_path=root / "config.yaml",
            method_config_sha256=str(method["config_sha256"]),
            run_identity=None,
            evidence_identity={},
        )
        return context, session

    def _dataset_patches(self):
        return mock.patch.multiple(
            smoke,
            dataset_sample_sha256s=mock.Mock(
                side_effect=lambda dataset, count=None: (
                    ["d" * 64] * (8 if count is not None else 16)
                )
            ),
            ordered_record_sha256=mock.Mock(return_value="e" * 64),
        )

    def test_context_and_revalidation_pass_strict_dataset_owner_root(self) -> None:
        """Exercise both real call boundaries against the keyword-only verifier."""

        protocol = copy.deepcopy(validate_protocol(load_strict_json(PROTOCOL_PATH)))
        registry = generate_registry(protocol)
        row = next(item for item in registry["rows"] if item["phase"] == "stage0_smoke")
        dataset_manifest = "1" * 64
        split_manifests = {
            "train": "2" * 64,
            "validation": "3" * 64,
            "test": "4" * 64,
        }
        split_orders = {
            "train": "5" * 64,
            "validation": "6" * 64,
            "test": "7" * 64,
        }
        producer = {
            "git_commit": "a" * 40,
            "launcher_sha256": "8" * 64,
            "runtime_sha256": "9" * 64,
            "source_manifest_sha256": "b" * 64,
        }
        protocol["dataset"]["manifest_sha256"] = {
            "status": "available",
            "value": dataset_manifest,
        }
        protocol["dataset"]["producer_source"] = {
            field: {"status": "available", "value": value}
            for field, value in producer.items()
        }
        for split in ("train", "validation", "test"):
            protocol["dataset"]["splits"][split]["manifest_sha256"] = {
                "status": "available",
                "value": split_manifests[split],
            }
            protocol["dataset"]["splits"][split]["ordered_record_sha256"] = {
                "status": "available",
                "value": split_orders[split],
            }
        protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        launcher_sha256 = "c" * 64
        runtime_sha256 = "d" * 64
        runtime_profile_sha256 = "e" * 64
        authorization = smoke.validate_runtime_authorization(
            {
                "schema_name": "policy_improvement_runtime_authorization_v1",
                "schema_version": 1,
                "authorization_id": "dataset-owner-boundary-v1",
                "created_at_utc": "2026-08-14T12:00:00Z",
                "protocol_sha256": protocol_sha256,
                "producer_git_commit": producer["git_commit"],
                "producer_source_manifest_sha256": runtime_profile_sha256,
                "launcher_sha256": launcher_sha256,
                "roles": [
                    {
                        "role": role,
                        "source_git_commit": producer["git_commit"],
                        "runtime_sha256": runtime_sha256,
                        "runtime_profile_sha256": runtime_profile_sha256,
                        "selected_source_manifest_sha256": (runtime_profile_sha256),
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
        authorization_sha256 = smoke.runtime_authorization_sha256(authorization)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            protocol_path = source_root / "configs/policy_improvement_v1/protocol.json"
            protocol_path.parent.mkdir(parents=True)
            protocol_path.write_text("{}\n", encoding="ascii")
            dataset_root = source_root / str(protocol["dataset"]["root"])
            dataset_root.mkdir(parents=True)
            evidence_root = root / "evidence"
            runtime_preflight = SimpleNamespace(
                phase4_role="policy-improvement-smoke",
                source_git_commit=producer["git_commit"],
                source_manifest_sha256=runtime_profile_sha256,
                runtime_sha256=runtime_sha256,
                runtime_descriptor=123,
                policy_runtime_authorization_json=canonical_json_bytes(
                    authorization
                ).decode("ascii"),
                policy_runtime_authorization_sha256=authorization_sha256,
                policy_runtime_profile_sha256=runtime_profile_sha256,
                policy_selected_source_manifest_sha256=(runtime_profile_sha256),
            )
            arguments = SimpleNamespace(
                source_project_root=str(source_root),
                policy_improvement_protocol=str(protocol_path),
                policy_improvement_row_id=row["run_id"],
                dataset_root=str(dataset_root),
                train_manifest_sha256=split_manifests["train"],
                validation_manifest_sha256=split_manifests["validation"],
                evidence_root=str(evidence_root),
                policy_improvement_smoke_segment="prepare",
            )
            verifier_calls: list[tuple[Path, Path]] = []

            def strict_verify_dataset(
                supplied_root: str | Path,
                *,
                owner_root: str | Path,
                expected_producer: object,
            ) -> dict[str, object]:
                self.assertEqual(Path(supplied_root), dataset_root)
                self.assertEqual(Path(owner_root), dataset_root.parent)
                self.assertEqual(expected_producer, producer)
                verifier_calls.append((Path(supplied_root), Path(owner_root)))
                return {
                    "manifest_sha256": dataset_manifest,
                    "producer_source": producer,
                    "split_manifest_sha256": split_manifests,
                    "split_ordered_record_sha256": split_orders,
                    "total_records": 1792,
                }

            with (
                mock.patch.dict(
                    smoke.os.environ,
                    {smoke._LAUNCHER_SHA256_ENV: launcher_sha256},
                    clear=False,
                ),
                mock.patch.object(
                    smoke,
                    "discover_clean_git_source",
                    return_value={
                        "git_commit": producer["git_commit"],
                        "git_clean": True,
                    },
                ),
                mock.patch.object(smoke, "load_strict_json", return_value=protocol),
                mock.patch.object(smoke, "validate_protocol", return_value=protocol),
                mock.patch.object(
                    smoke,
                    "load_registered_base_configs",
                    return_value={},
                ),
                mock.patch.object(smoke, "generate_registry", return_value=registry),
                mock.patch.object(
                    smoke,
                    "verify_dataset",
                    side_effect=strict_verify_dataset,
                ),
                mock.patch.object(
                    smoke,
                    "authorize_phase4_training_source",
                    return_value=SimpleNamespace(
                        source_manifest_sha256=runtime_profile_sha256
                    ),
                ),
                mock.patch.object(smoke, "_rehash_runtime"),
            ):
                context = smoke._load_context(
                    arguments,
                    runtime_preflight=runtime_preflight,
                )
                smoke._revalidate_external_inputs(
                    context,
                    runtime_preflight=runtime_preflight,
                )

            self.assertEqual(
                verifier_calls,
                [
                    (dataset_root, dataset_root.parent),
                    (dataset_root, dataset_root.parent),
                ],
            )

    def test_failed_result_uses_current_schema_and_is_provenance_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, session = self._context_and_session(Path(directory))
            with self._dataset_patches():
                result = smoke._failed_result(
                    context,
                    session,
                    phase="training",
                    error=RuntimeError("injected training failure"),
                )
        validate_result(result)
        self.assertEqual(result["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["applied_config_override"], context.row["config_override"]
        )
        self.assertEqual(
            result["identities"]["effective_config_sha256"],
            context.row["expected_effective_config_sha256"],
        )
        self.assertEqual(
            result["identities"]["runtime_authorization_sha256"],
            "6" * 64,
        )
        self.assertTrue(
            all(
                snapshot["status"] == "unavailable"
                for snapshot in result["evaluation_snapshots"]
            )
        )
        self.assertEqual(
            result["identities"]["evaluation_runtime_sha256"],
            {
                "status": "unavailable",
                "reason": "run_failed_before_evaluation",
            },
        )

    def test_failed_result_binds_evaluator_after_evaluation_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, session = self._context_and_session(Path(directory))
            session.evaluation_started = True
            with self._dataset_patches():
                result = smoke._failed_result(
                    context,
                    session,
                    phase="evaluation",
                    error=RuntimeError("injected evaluation failure"),
                )
        validate_result(result)
        self.assertEqual(
            result["identities"]["evaluation_runtime_sha256"],
            {"status": "available", "value": context.runtime_sha256},
        )
        self.assertEqual(
            result["identities"]["evaluation_source_git_commit"],
            {"status": "available", "value": context.producer_commit},
        )

    def test_failed_attempts_are_immutable_and_leave_no_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, session = self._context_and_session(Path(directory))
            with (
                self._dataset_patches(),
                mock.patch.object(
                    smoke,
                    "_revalidate_external_inputs",
                ),
            ):
                first = smoke._publish_failure_attempt(
                    context,
                    session,
                    runtime_preflight=object(),
                    phase="checkpoint",
                    error=RuntimeError("checkpoint failed"),
                )
                second = smoke._publish_failure_attempt(
                    context,
                    session,
                    runtime_preflight=object(),
                    phase="checkpoint",
                    error=RuntimeError("checkpoint failed"),
                )
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            attempts = first.parent
            self.assertEqual(
                sorted(path.name for path in attempts.iterdir()),
                sorted((first.name, second.name)),
            )
            self.assertEqual(
                list(attempts.glob(".failed-attempt-stage.*")),
                [],
            )

    def test_failed_attempt_write_and_rename_failures_do_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, session = self._context_and_session(Path(directory))
            attempts = context.run_root / "attempts" / "resume"
            with (
                self._dataset_patches(),
                mock.patch.object(
                    smoke,
                    "_write_json",
                    side_effect=OSError("injected write failure"),
                ),
                self.assertRaisesRegex(OSError, "injected write failure"),
            ):
                smoke._publish_failure_attempt(
                    context,
                    session,
                    runtime_preflight=object(),
                    phase="training",
                    error=RuntimeError("training failed"),
                )
            self.assertEqual(
                list(attempts.glob(".failed-attempt-stage.*")),
                [],
            )

            with (
                self._dataset_patches(),
                mock.patch.object(
                    smoke,
                    "_revalidate_external_inputs",
                ),
                mock.patch.object(
                    smoke,
                    "_rename_noreplace",
                    side_effect=OSError("injected rename failure"),
                ),
                self.assertRaisesRegex(OSError, "injected rename failure"),
            ):
                smoke._publish_failure_attempt(
                    context,
                    session,
                    runtime_preflight=object(),
                    phase="publication",
                    error=RuntimeError("publication failed"),
                )
            self.assertEqual(list(attempts.iterdir()), [])

    def test_gpu_sampler_uses_training_interval_and_cpu_claims_nothing(self) -> None:
        cpu = smoke._GpuTrainingSampler(torch.device("cpu"))
        cpu.start()
        self.assertEqual(
            cpu.stop(),
            smoke.GpuUtilizationSummary("cpu", None, ()),
        )

        cuda = smoke._GpuTrainingSampler(
            torch.device("cuda"),
            interval_seconds=10.0,
        )
        with mock.patch.object(torch.cuda, "utilization", return_value=37):
            cuda.start()
            summary = cuda.stop()
        self.assertEqual(summary.device_type, "cuda")
        self.assertEqual(summary.sampling_interval_seconds, 10.0)
        self.assertEqual(summary.samples, (0.37, 0.37))

    def test_unauthenticated_failure_publishes_nothing(self) -> None:
        with (
            mock.patch.object(
                smoke,
                "_parse_args",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                smoke,
                "_load_context",
                side_effect=smoke.PolicyImprovementSmokeError("authorization rejected"),
            ),
            mock.patch.object(smoke, "_publish_failure_attempt") as publish,
        ):
            with self.assertRaisesRegex(
                smoke.PolicyImprovementSmokeError,
                "authorization rejected",
            ):
                smoke.main(
                    [],
                    runtime_preflight=object(),
                    training_module=object(),
                )
        publish.assert_not_called()

    def test_authenticated_training_failure_returns_nonzero_and_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, session = self._context_and_session(Path(directory))
            destination = Path(directory) / "failed-attempt"
            with (
                mock.patch.object(
                    smoke,
                    "_parse_args",
                    return_value=SimpleNamespace(),
                ),
                mock.patch.object(
                    smoke,
                    "_load_context",
                    return_value=context,
                ),
                mock.patch.object(
                    smoke,
                    "_build_session",
                    return_value=session,
                ),
                mock.patch.object(
                    smoke,
                    "_restore_parent",
                    return_value=None,
                ),
                mock.patch.object(
                    smoke,
                    "_parent_gpu_utilization_summary",
                    return_value=smoke.GpuUtilizationSummary("cpu", None, ()),
                ),
                mock.patch.object(
                    smoke._GpuTrainingSampler,
                    "start",
                ),
                mock.patch.object(
                    smoke._GpuTrainingSampler,
                    "stop",
                    return_value=smoke.GpuUtilizationSummary("cpu", None, ()),
                ),
                mock.patch.object(
                    smoke,
                    "_train_to_budget",
                    side_effect=RuntimeError("injected training failure"),
                ),
                mock.patch.object(
                    smoke,
                    "_publish_failure_attempt",
                    return_value=destination,
                ) as publish,
            ):
                return_code = smoke.main(
                    [],
                    runtime_preflight=object(),
                    training_module=object(),
                )
        self.assertEqual(return_code, 1)
        self.assertEqual(publish.call_args.kwargs["phase"], "training")


if __name__ == "__main__":
    unittest.main()
