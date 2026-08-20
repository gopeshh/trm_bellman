"""Tests for the authenticated training-only throughput calibrator."""

from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import policy_improvement_throughput as throughput
from scripts.policy_improvement_throughput import (
    AUTOMATIC_CAPS,
    ENGINEERING_SEED,
    METHOD_IDS,
    OPTIONAL_CAP,
    execute_calibration,
    fit_engineering_model,
    load_throughput_registration,
    load_training_only_dataset,
    main,
    THROUGHPUT_SAMPLE_SCHEMA_NAME,
    THROUGHPUT_SAMPLE_SCHEMA_VERSION,
    ThroughputCalibrationError,
    ThroughputRegistration,
    ThroughputSampleRequest,
    validate_throughput_sample,
)
from utils.compute_accounting import (
    COMPUTE_SNAPSHOT_SCHEMA_VERSION,
    MODEL_COUNTER_DEFINITIONS,
    OPTIMIZER_STEP_FIELDS,
    build_training_compute_accounting,
    zero_model_counters,
)


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _registration() -> ThroughputRegistration:
    methods = [
        {"id": method_id, "config_sha256": _digest(f"config:{method_id}")}
        for method_id in METHOD_IDS
    ]
    protocol = {
        "methods": methods,
        "dataset": {"splits": {"train": {"count": 1024}}},
    }
    return ThroughputRegistration(
        project_root=Path("/project"),
        dataset_root=Path("/dataset"),
        protocol=protocol,
        registry={},
        method_configs={method_id: {} for method_id in METHOD_IDS},
        protocol_sha256=_digest("protocol"),
        registry_sha256=_digest("registry"),
        population_registry_sha256=_digest("populations"),
        runtime_authorization_sha256=_digest("authorization"),
        dataset_manifest_sha256=_digest("dataset"),
        train_manifest_sha256=_digest("train manifest"),
        train_ordered_record_sha256=_digest("train order"),
    )


def _compute(cap: int, elapsed: float) -> tuple[dict[str, object], dict[str, object]]:
    training = zero_model_counters()
    training.update(
        {
            "recurrent_latent_state_updates": cap * 2,
            "action_logits_evaluated": cap * 5,
            "action_values_evaluated": cap,
            "value_api_calls": cap // 8,
        }
    )
    zero = zero_model_counters()
    snapshot = {
        "compute_schema_version": COMPUTE_SNAPSHOT_SCHEMA_VERSION,
        "model_work": {
            "total": dict(training),
            "training": dict(training),
            "evaluation": dict(zero),
            "counter_definitions": dict(MODEL_COUNTER_DEFINITIONS),
            "role_groups": [["model"]],
            "uninstrumented_roles": [],
        },
        "progress": {
            "environment_interactions": cap,
            "outer_updates": 1,
            "optimizer_steps_total": 1,
            "optimizer_steps_by_kind": {
                field: int(field == "combined") for field in OPTIMIZER_STEP_FIELDS
            },
        },
        "wall_time_seconds": {"training": elapsed, "evaluation": 0.0},
        "peak_memory_bytes": {
            "cuda_allocated": None,
            "cuda_reserved": None,
            "process_rss": 4096,
        },
    }
    accounting = build_training_compute_accounting(
        snapshot,
        device="cpu",
        checkpoint_seconds=0.0,
        cuda_utilization={
            "device_type": "cpu",
            "sampling_interval_seconds": None,
            "samples": [],
        },
    )
    return snapshot, accounting


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def execute_throughput_sample(
        self, request: ThroughputSampleRequest
    ) -> dict[str, object]:
        self.calls.append(
            (
                request.method_id,
                request.environment_interactions,
                request.engineering_seed,
            )
        )
        method_index = METHOD_IDS.index(request.method_id)
        elapsed = 1.0 + method_index + request.environment_interactions * 0.01
        snapshot, accounting = _compute(request.environment_interactions, elapsed)
        registration = request.registration
        method = next(
            item
            for item in registration.protocol["methods"]
            if item["id"] == request.method_id
        )
        return {
            "schema_name": THROUGHPUT_SAMPLE_SCHEMA_NAME,
            "schema_version": THROUGHPUT_SAMPLE_SCHEMA_VERSION,
            "method_id": request.method_id,
            "environment_interaction_cap": request.environment_interactions,
            "engineering_seed": request.engineering_seed,
            "training_split": "train",
            "started_environment_interactions": 0,
            "completed_environment_interactions": request.environment_interactions,
            "session_setup_seconds": 0.25,
            "elapsed_seconds": elapsed + 0.25,
            "recurrent_map_applications": request.environment_interactions * 2,
            "method_config_sha256": method["config_sha256"],
            "effective_config_sha256": _digest(
                f"effective:{request.method_id}:{request.environment_interactions}"
            ),
            "dataset_manifest_sha256": registration.dataset_manifest_sha256,
            "train_manifest_sha256": registration.train_manifest_sha256,
            "train_ordered_record_sha256": (registration.train_ordered_record_sha256),
            "runtime_identity": {
                "role": "policy-improvement-full",
                "runtime_sha256": _digest("runtime"),
                "source_git_commit": "a" * 40,
                "source_manifest_sha256": _digest("profile"),
                "producer_source_manifest_sha256": _digest("producer"),
                "runtime_profile_sha256": _digest("profile"),
                "selected_source_manifest_sha256": _digest("profile"),
                "runtime_authorization_sha256": (
                    registration.runtime_authorization_sha256
                ),
                "launcher_sha256": _digest("launcher"),
            },
            "compute_snapshot": snapshot,
            "compute_accounting": accounting,
            "evaluation_rollouts": False,
            "validation_data_opened": False,
            "test_data_opened": False,
            "test_open_bound": False,
            "scientific_selection": False,
            "paper_evidence_eligible": False,
            "performance_metrics_collected": False,
        }


class ThroughputCalibrationTest(unittest.TestCase):
    def test_automatic_schedule_is_four_methods_by_two_caps(self) -> None:
        backend = _FakeBackend()
        result = execute_calibration(
            _registration(),
            backend=backend,
            authorize_4096_tier=False,
            maximum_total_predicted_4096_seconds=None,
        )
        self.assertEqual(
            backend.calls,
            [
                (method_id, cap, ENGINEERING_SEED)
                for cap in AUTOMATIC_CAPS
                for method_id in METHOD_IDS
            ],
        )
        self.assertEqual(len(result["samples"]), 8)
        self.assertFalse(result["schedule"]["optional_cap_executed"])
        self.assertFalse(result["scientific_selection"])
        self.assertFalse(result["performance_metrics_collected"])
        for model in result["final_models"].values():
            self.assertEqual(model["equation"], "t(N)=a+bN")
            self.assertEqual(model["fit_caps"], list(AUTOMATIC_CAPS))

    def test_optional_tier_requires_flag_and_prediction_ceiling_together(self) -> None:
        for authorized, ceiling in ((True, None), (False, 1000.0)):
            with (
                self.subTest(authorized=authorized, ceiling=ceiling),
                self.assertRaisesRegex(
                    ThroughputCalibrationError,
                    "both explicit authorization and a prediction ceiling",
                ),
            ):
                execute_calibration(
                    _registration(),
                    backend=_FakeBackend(),
                    authorize_4096_tier=authorized,
                    maximum_total_predicted_4096_seconds=ceiling,
                )

    def test_optional_tier_runs_only_after_every_1024_sample(self) -> None:
        backend = _FakeBackend()
        result = execute_calibration(
            _registration(),
            backend=backend,
            authorize_4096_tier=True,
            maximum_total_predicted_4096_seconds=1000.0,
        )
        self.assertEqual(
            backend.calls[:8],
            [
                (method_id, cap, ENGINEERING_SEED)
                for cap in AUTOMATIC_CAPS
                for method_id in METHOD_IDS
            ],
        )
        self.assertEqual(
            backend.calls[8:],
            [(method_id, OPTIONAL_CAP, ENGINEERING_SEED) for method_id in METHOD_IDS],
        )
        self.assertTrue(result["schedule"]["optional_cap_executed"])
        for model in result["final_models"].values():
            self.assertEqual(model["fit_caps"], [256, 1024, 4096])

    def test_prediction_ceiling_rejects_before_any_4096_sample(self) -> None:
        backend = _FakeBackend()
        with self.assertRaisesRegex(
            ThroughputCalibrationError,
            "exceeds the authorized resource ceiling",
        ):
            execute_calibration(
                _registration(),
                backend=backend,
                authorize_4096_tier=True,
                maximum_total_predicted_4096_seconds=1.0,
            )
        self.assertEqual(len(backend.calls), 8)
        self.assertNotIn(OPTIONAL_CAP, [cap for _, cap, _ in backend.calls])

    def test_dataset_loader_has_no_split_or_path_capability(self) -> None:
        registration = _registration()
        calls: list[tuple[Path, str, int]] = []

        def loader(root: Path, split: str, count: int) -> object:
            calls.append((root, split, count))
            return "train-only"

        self.assertEqual(
            load_training_only_dataset(registration, loader),
            "train-only",
        )
        self.assertEqual(calls, [(Path("/dataset"), "train", 1024)])

    def test_registration_rejects_unregistered_dataset_before_opening_it(self) -> None:
        dataset_manifest = b"dataset"
        train_manifest = b"train manifest"
        protocol = {
            "budgets": {
                "throughput_calibration": {
                    "automatic_caps": list(AUTOMATIC_CAPS),
                    "environment_interaction_caps": [
                        *AUTOMATIC_CAPS,
                        OPTIONAL_CAP,
                    ],
                    "evaluation_rollouts": False,
                    "split": "train",
                    "user_authorization_required_for": OPTIONAL_CAP,
                }
            },
            "dataset": {
                "root": "registered/dataset",
                "manifest_sha256": {
                    "status": "available",
                    "value": hashlib.sha256(dataset_manifest).hexdigest(),
                },
                "splits": {
                    "train": {
                        "count": 1024,
                        "manifest_sha256": {
                            "status": "available",
                            "value": hashlib.sha256(train_manifest).hexdigest(),
                        },
                        "ordered_record_sha256": {
                            "status": "available",
                            "value": _digest("train order"),
                        },
                    }
                },
            },
            "population_registry": {"sha256": _digest("populations")},
            "test_isolation": {
                "throughput_split": "train",
                "test_open_status": "not_created",
            },
        }
        method_configs = {method_id: {} for method_id in METHOD_IDS}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "configs/policy_improvement_v2"
            config.mkdir(parents=True)
            protocol_path = config / "protocol.json"
            registry_path = config / "registry.json"
            protocol_path.write_text("{}\n", encoding="ascii")
            registry_path.write_text("{}\n", encoding="ascii")
            attacker_dataset = root / "attacker-dataset"
            (attacker_dataset / "manifests").mkdir(parents=True)
            (attacker_dataset / "train").mkdir()
            (attacker_dataset / "MANIFEST.json").write_bytes(dataset_manifest)
            (attacker_dataset / "manifests/train.json").write_bytes(train_manifest)

            stable_reader = throughput._stable_regular_file_bytes
            with (
                mock.patch(
                    "scripts.policy_improvement_throughput.validate_v2_protocol",
                    return_value=protocol,
                ),
                mock.patch(
                    "scripts.policy_improvement_throughput.load_registered_populations",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_throughput.load_v2_base_configs",
                    return_value=method_configs,
                ),
                mock.patch(
                    "scripts.policy_improvement_throughput.validate_v2_registry_document",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_throughput._stable_regular_file_bytes",
                    wraps=stable_reader,
                ) as read_file,
            ):
                with mock.patch.object(
                    throughput,
                    "_canonical_registered_directory",
                    side_effect=lambda _project, supplied, _relative, name: (
                        throughput._regular_directory(supplied, name=name)
                    ),
                ):
                    accepted = load_throughput_registration(
                        project_root=root,
                        protocol_path=protocol_path,
                        registry_path=registry_path,
                        dataset_root=attacker_dataset,
                        runtime_authorization_sha256=_digest("authorization"),
                    )
                self.assertEqual(accepted.dataset_root, attacker_dataset)
                self.assertIn(
                    mock.call(
                        attacker_dataset / "MANIFEST.json",
                        name="dataset manifest",
                    ),
                    read_file.call_args_list,
                )

                read_file.reset_mock()
                with self.assertRaisesRegex(
                    ThroughputCalibrationError,
                    "dataset root is not the protocol-v2 dataset",
                ):
                    load_throughput_registration(
                        project_root=root,
                        protocol_path=protocol_path,
                        registry_path=registry_path,
                        dataset_root=attacker_dataset,
                        runtime_authorization_sha256=_digest("authorization"),
                    )
                self.assertNotIn(
                    mock.call(
                        attacker_dataset / "MANIFEST.json",
                        name="dataset manifest",
                    ),
                    read_file.call_args_list,
                )

                registered_dataset = root / "registered/dataset"
                (registered_dataset / "manifests").mkdir(parents=True)
                (registered_dataset / "train").mkdir()
                (registered_dataset / "MANIFEST.json").write_bytes(
                    dataset_manifest
                )
                (registered_dataset / "manifests/train.json").write_bytes(
                    train_manifest
                )
                accepted = load_throughput_registration(
                    project_root=root,
                    protocol_path=protocol_path,
                    registry_path=registry_path,
                    dataset_root=registered_dataset,
                    runtime_authorization_sha256=_digest("authorization"),
                )
                self.assertEqual(accepted.dataset_root, registered_dataset)

                def register_then_load(loader: mock.Mock) -> object:
                    registration = load_throughput_registration(
                        project_root=root,
                        protocol_path=protocol_path,
                        registry_path=registry_path,
                        dataset_root=registered_dataset,
                        runtime_authorization_sha256=_digest("authorization"),
                    )
                    return load_training_only_dataset(registration, loader)

                train_directory = registered_dataset / "train"
                train_directory.rmdir()
                train_directory.symlink_to(attacker_dataset, target_is_directory=True)
                split_loader = mock.Mock(return_value="must not load")
                read_file.reset_mock()
                with self.assertRaisesRegex(
                    ThroughputCalibrationError,
                    "train split directory must be a canonical non-symlink directory",
                ):
                    register_then_load(split_loader)
                split_loader.assert_not_called()
                self.assertNotIn(
                    mock.call(
                        registered_dataset / "MANIFEST.json",
                        name="dataset manifest",
                    ),
                    read_file.call_args_list,
                )

                train_directory.unlink()
                train_directory.mkdir()
                manifest_directory = registered_dataset / "manifests"
                (manifest_directory / "train.json").unlink()
                manifest_directory.rmdir()
                external_manifests = root / "external-manifests"
                external_manifests.mkdir()
                (external_manifests / "train.json").write_bytes(train_manifest)
                manifest_directory.symlink_to(
                    external_manifests,
                    target_is_directory=True,
                )
                split_loader.reset_mock()
                read_file.reset_mock()
                with self.assertRaisesRegex(
                    ThroughputCalibrationError,
                    "dataset manifest directory must be a canonical non-symlink",
                ):
                    register_then_load(split_loader)
                split_loader.assert_not_called()
                self.assertNotIn(
                    mock.call(
                        registered_dataset / "MANIFEST.json",
                        name="dataset manifest",
                    ),
                    read_file.call_args_list,
                )

    def test_sample_rejects_test_validation_or_evaluation_claims(self) -> None:
        registration = _registration()
        request = ThroughputSampleRequest(
            registration=registration,
            method_id=METHOD_IDS[0],
            environment_interactions=256,
        )
        original = _FakeBackend().execute_throughput_sample(request)
        mutations = (
            ("training_split", "test"),
            ("validation_data_opened", True),
            ("test_data_opened", True),
            ("test_open_bound", True),
            ("evaluation_rollouts", True),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = replacement
                with self.assertRaises(ThroughputCalibrationError):
                    validate_throughput_sample(changed, request=request)

    def test_sample_rejects_hidden_evaluation_compute(self) -> None:
        registration = _registration()
        request = ThroughputSampleRequest(
            registration=registration,
            method_id=METHOD_IDS[0],
            environment_interactions=256,
        )
        changed = _FakeBackend().execute_throughput_sample(request)
        changed = copy.deepcopy(changed)
        changed["compute_snapshot"]["wall_time_seconds"]["evaluation"] = 1.0
        with self.assertRaises(ValueError):
            validate_throughput_sample(changed, request=request)

    def test_fit_is_only_the_registered_linear_engineering_model(self) -> None:
        model = fit_engineering_model(
            [
                {"environment_interaction_cap": 256, "elapsed_seconds": 3.56},
                {"environment_interaction_cap": 1024, "elapsed_seconds": 11.24},
            ]
        )
        self.assertAlmostEqual(model["startup_seconds_a"], 1.0)
        self.assertAlmostEqual(model["steady_state_seconds_per_interaction_b"], 0.01)
        self.assertAlmostEqual(model["predicted_4096_seconds"], 41.96)
        self.assertEqual(
            set(model),
            {
                "schema_name",
                "schema_version",
                "equation",
                "fit_caps",
                "startup_seconds_a",
                "steady_state_seconds_per_interaction_b",
                "sum_squared_error",
                "predicted_4096_seconds",
            },
        )

    def test_source_tree_main_has_no_backend_fallback_or_dataset_open(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch(
                "scripts.policy_improvement_throughput.load_throughput_registration"
            ) as load,
            mock.patch("sys.stdout", stdout),
            mock.patch("sys.stderr", stderr),
        ):
            code = main(
                [
                    "--project-root",
                    "/missing",
                    "--protocol",
                    "/missing/protocol.json",
                    "--registry",
                    "/missing/registry.json",
                    "--dataset-root",
                    "/missing/dataset",
                    "--runtime-authorization-sha256",
                    "0" * 64,
                ],
                backend=None,
            )
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("authenticated packaged runtime", stderr.getvalue())
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
